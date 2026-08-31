"""Keypoint-servo action override for the surface-following task.

An :class:`gym.ActionWrapper` installed JUST OUTSIDE the control wrapper (inside the
efficient-reset / fragile / scorer wrappers). It takes over the leading POSE dims of the
action so the policy no longer emits the end-effector translation (and, with
``fix_orientation``, the rotation): those dims are computed each step from the surface
geometry and PREPENDED before the full vector is handed down to the control wrapper.

Frame recap (see :func:`wrappers.controllers.factory_control_utils.compute_ctrl_targets`):
the control wrapper reads the first 6 action dims as an EEF-frame pose delta, each mapping
``[-1, 1] -> +/- threshold`` per axis (``pos_threshold`` / ``rot_threshold``), then
``target_pos = eef_pos + R_eef * (action[0:3] * pos_threshold)`` and
``target_quat = eef_quat o Delta(action[3:6] * rot_threshold)``. So to command a world-frame
displacement / orientation we express it in the EEF frame and divide by the threshold.

Position (always, when enabled) — a capped servo toward the current setpoint keypoint:

    disp_world  = env.setpoint_pos - env.held_end_pos              # tip -> keypoint (env frame)
    offset_world = along*path_dir + off*d_lat + normal*surface_normal
    total_eef   = R_eef^T * (disp_world + offset_world)            # into the EEF frame
    action[0:3] = clip(total_eef / pos_threshold, -1, 1)

``held_end_pos`` and ``setpoint_pos`` are both env-relative, so their difference is free of
the env origin; the surface basis vectors (``path_dir``, ``d_lat``, ``surface_normal``) are
world-axis unit vectors. Clipping caps the per-step motion at ``pos_threshold`` (the
controller re-multiplies by it), giving a proportional servo — exact when close, saturated
when far. The pose action still passes through the control wrapper's EMA smoothing.

Orientation (optional ``fix_orientation``) — HOLD each env's initial (spawn) EEF orientation:

    q_target[e] = fingertip_midpoint_quat[e]  captured at reset (episode_length_buf == 0)
    dq_eef      = eef_quat^-1 o q_target                           # body-frame delta
    action[3:6] = clip(axis_angle(dq_eef) / rot_threshold, -1, 1)

The target is the FULL spawn orientation (all 3 rotations), snapshotted per-env the first control
step after a full OR per-env reset and then held for the rest of the episode. This keeps whatever
the reset set up — including the x-axis heading from ``spawn_align_eef_x_to_path`` and the grasp
tilt that keeps the peg on the surface — rather than a constant world rpy (which would level the
wrist and lift the peg off the plate). No offset is ever added to orientation. When
``fix_orientation`` is off the policy keeps the rotation dims.

Orientation->tip lever-arm decoupling (optional ``decouple_orientation_lever_arm``, needs
``fix_orientation``) — the position servo above drives the EEF, but the point we actually want on the
setpoint is the tip, offset from the EEF by ``r = held_end_pos - fingertip_midpoint_pos``. When the
orientation channel is simultaneously rotating the EEF by the applied delta ``Δq``, the tip is swept
by ``Δq·r - r`` (computed with ``quat_apply`` — EXACT for any angle, not a small-angle ``ω x r``, so
it holds if ``rot_threshold`` is raised). Enabling this subtracts that induced motion from the
position command so the tip still lands on the setpoint. ``r`` uses the loop-closure point
(``held_end_pos``, the tip center) — distinct from the stiffness adjoint's contact point
(``interaction_pos``), which differs by the tip sphere radius. Off by default (original
translation-only behavior).

Action-space surgery: the taken-over dims are a contiguous FRONT block — ``pos`` (0:3) always,
plus ``rot`` (3:6) when ``fix_orientation`` — so the wrapper is agnostic to whatever
force/gain dims the control wrapper appends after them. It shrinks the exposed action space by
3 (or 6) and overwrites ``unwrapped.action_space`` / ``unwrapped.cfg.action_space`` so skrl and
the runner both build the policy against the reduced space; the full-width vector is
reconstructed here before it reaches the control wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply

from .factory_control_utils import rotate_vec_to_eef

try:  # Isaac Sim >= 4.5
    import isaacsim.core.utils.torch as torch_utils
except ModuleNotFoundError:  # pragma: no cover - older Isaac layout
    import omni.isaac.core.utils.torch as torch_utils


class KeypointServoActionWrapper(gym.ActionWrapper):
    """Override the leading pose action dims with a keypoint servo (+ optional fixed orientation)."""

    # Env attributes required to compute the servo (validated on the first step).
    _REQUIRED_ATTRS = (
        "held_end_pos",
        "setpoint_pos",
        "path_dir",
        "d_lat",
        "surface_normal",
        "fingertip_midpoint_quat",
        "pos_threshold",
        "rot_threshold",
    )

    def __init__(self, env, cfg) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs

        self._along = float(cfg.along_track_offset)
        self._off = float(cfg.off_track_offset)
        self._normal = float(cfg.normal_offset)
        self._any_offset = any(abs(v) > 0.0 for v in (self._along, self._off, self._normal))

        self._fix_orientation = bool(cfg.fix_orientation)
        # Exact orientation->tip lever-arm decoupling (only meaningful when fixing orientation, since
        # otherwise there is no wrapper-commanded rotation to decouple against).
        self._decouple_lever = bool(getattr(cfg, "decouple_orientation_lever_arm", False)) and self._fix_orientation
        # Number of contiguous FRONT dims this wrapper takes over: pos (3), + rot (3) if fixing it.
        self._n_override = 6 if self._fix_orientation else 3

        # Per-env orientation hold target (E,4): each env's INITIAL (spawn) EEF orientation, latched at
        # reset. Initialized to identity; (re)captured for any env whose episode_length_buf is 0 (fresh
        # full or per-env reset) in _capture_reset_orientation. Holding the captured spawn orientation
        # keeps the peg on the surface (a constant world rpy would level the wrist and lift the peg off).
        if self._fix_orientation:
            self._q_target = torch.zeros((self.num_envs, 4), device=self.device)
            self._q_target[:, 0] = 1.0                                # identity until first capture
        else:
            self._q_target = None

        # Full (control-wrapper) action width, then the reduced policy-facing width.
        self._full_dim = int(env.action_space.shape[0])
        self._reduced_dim = self._full_dim - self._n_override
        if self._reduced_dim <= 0:
            raise ValueError(
                f"[keypoint-servo] taking over {self._n_override} leading dims leaves "
                f"{self._reduced_dim} action(s) for the policy (control action width "
                f"{self._full_dim}). The policy would have nothing to control — use a control "
                "config with gain/force dims (e.g. variable_diagonal / VICES / GAS), or set "
                "fix_orientation=false so the policy keeps the 3 rotation dims."
            )

        # Number of leading pose dims removed from the policy-facing action. The runner reads this
        # to remap any action-index-keyed model config (scale_down_action_dims, etc.) onto the
        # reduced layout, since those indices are authored against the FULL action vector.
        self.unwrapped._keypoint_servo_removed_dims = self._n_override

        # Expose the reduced space to the POLICY only, WITHOUT touching the env's internal action
        # tensors. skrl's IsaacLabWrapper reads ``unwrapped.single_action_space`` first (else
        # ``action_space``), and the runner reads ``env.action_space`` — so setting those three to
        # the reduced Box makes the actor/critic build against the reduced width.
        #
        # Deliberately NOT calling ``_configure_gym_env_spaces()`` / mutating ``cfg.action_space``:
        # that would reallocate ``self.actions`` (direct_rl_env.py samples it from
        # single_action_space) to the reduced width. But the control wrapper keeps ``self.actions``
        # (and ``prev_actions``) at the FULL width — it EMAs the full action into it
        # (hybrid_force_position_wrapper.py) — and the Factory/Forge base appends that full
        # ``prev_actions`` to the obs AND critic state. Leaving the env buffers full keeps obs/state
        # consistent with their (control-wrapper-grown) declared spaces; the env never clips the
        # incoming full action against single_action_space (DirectRLEnv.step passes it straight to
        # _pre_physics_step). The policy simply observes the full commanded action as prev_actions.
        reduced_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._reduced_dim,), dtype=np.float32
        )
        self.action_space = reduced_space
        self.unwrapped.action_space = reduced_space
        if hasattr(self.unwrapped, "single_action_space"):
            self.unwrapped.single_action_space = reduced_space

        self._validated = False

    # ------------------------------------------------------------------ setup
    def _validate(self) -> None:
        env = self.unwrapped
        missing = [a for a in self._REQUIRED_ATTRS if not hasattr(env, a)]
        if missing:
            raise RuntimeError(
                "KeypointServoActionWrapper requires a FlatSurfaceFollow env exposing "
                f"{list(self._REQUIRED_ATTRS)}; missing {missing}. Use an "
                "Isaac-FlatSurfaceFollow-* task."
            )
        self._validated = True

    # ------------------------------------------------------------------ servo
    def _pos_action(self, applied_aa: torch.Tensor | None = None) -> torch.Tensor:
        """EEF-frame, threshold-normalized position action (E,3) servoing the tip to the setpoint.

        When ``applied_aa`` (the clamped EEF-frame orientation delta this step is commanding) is
        supplied, the EXACT tip motion that rotation induces is removed from the command, so the
        servoed point (``held_end_pos``) still lands on the setpoint while the EEF is rotating (the
        ``decouple_orientation_lever_arm`` path). ``applied_aa=None`` reproduces the original
        translation-only servo.
        """
        env = self.unwrapped
        eef_quat = env.fingertip_midpoint_quat
        disp_world = env.setpoint_pos - env.held_end_pos                      # (E,3) tip -> keypoint
        if self._any_offset:
            disp_world = (
                disp_world
                + self._along * env.path_dir
                + self._off * env.d_lat
                + self._normal * env.surface_normal
            )
        total_eef = rotate_vec_to_eef(disp_world, eef_quat)                   # (E,3) into EEF frame
        if applied_aa is not None:
            # EXACT tip displacement from the applied rotation Δq: (Δq·r − r), in the EEF frame. r is
            # the arm to the LOOP-CLOSURE point (tip center ``held_end_pos``, where keypoints/success
            # live) — NOT the stiffness adjoint's contact point (``interaction_pos``), which differs by
            # the tip sphere radius; the decoupling arm must match the point being driven to the
            # setpoint. ``quat_apply`` is exact for ANY angle (no small-angle ω×r), so raising
            # ``rot_threshold`` stays correct. Δq is rebuilt from ``applied_aa`` exactly as
            # ``compute_ctrl_targets`` does, so it cancels the rotation that is actually applied.
            angle = applied_aa.norm(dim=1)                                   # (E,)
            axis = applied_aa / angle.clamp_min(1e-6).unsqueeze(-1)          # (E,3)
            dq_applied = torch_utils.quat_from_angle_axis(angle, axis)       # (E,4)
            identity = torch.zeros_like(dq_applied)
            identity[:, 0] = 1.0
            dq_applied = torch.where(angle.unsqueeze(-1) > 1e-6, dq_applied, identity)
            r_eef = rotate_vec_to_eef(env.held_end_pos - env.fingertip_midpoint_pos, eef_quat)
            tip_disp_rot_eef = quat_apply(dq_applied, r_eef) - r_eef          # (E,3) EEF frame, exact
            total_eef = total_eef - tip_disp_rot_eef
        return torch.clamp(total_eef / env.pos_threshold, -1.0, 1.0)

    def _capture_reset_orientation(self) -> None:
        """Latch each freshly-reset env's current (spawn) EEF orientation as the hold target.

        ``episode_length_buf == 0`` marks the first control step after a full OR per-env reset — the
        EEF is still at its reset pose (this runs pre-physics, before the policy's action is applied),
        so we snapshot ``fingertip_midpoint_quat`` there. It stays fixed for the rest of the episode
        (the buffer becomes >=1 after this step), so the target is exactly the spawn orientation.
        """
        env = self.unwrapped
        fresh = env.episode_length_buf == 0                                  # (E,) bool
        if fresh.any():
            self._q_target[fresh] = env.fingertip_midpoint_quat[fresh].detach().clone()

    def _applied_rot_axis_angle(self) -> torch.Tensor:
        """(E,3) EEF-frame axis-angle of the orientation delta the controller will ACTUALLY apply this
        step: the shortest rotation from the current EEF orientation to the held (spawn) target,
        clamped per-component to ``env.rot_threshold`` — the exact quantity ``compute_ctrl_targets``
        turns into the applied delta quaternion. Shared by the rotation action and the position
        lever-arm decoupling so both reference the identical rotation."""
        env = self.unwrapped
        eef_quat = env.fingertip_midpoint_quat                               # (E,4)
        q_target = self._q_target                                            # (E,4) per-env spawn orient.
        # Body-frame delta: target = eef o dq  =>  dq = eef^-1 o target.
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(eef_quat), q_target)
        # Canonicalize to the positive-w hemisphere so axis_angle gives the SHORTEST rotation.
        dq = torch.where(dq[:, 0:1] < 0.0, -dq, dq)
        aa = axis_angle_from_quat(dq)                                        # (E,3) EEF-frame axis-angle
        return torch.clamp(aa, -env.rot_threshold, env.rot_threshold)       # applied per-axis delta

    def _rot_action(self, applied_aa: torch.Tensor) -> torch.Tensor:
        """EEF-frame, threshold-normalized rotation action (E,3) driving the EEF to the held quat.
        ``applied_aa`` is already clamped to ``rot_threshold``, so this is just the normalization."""
        return torch.clamp(applied_aa / self.unwrapped.rot_threshold, -1.0, 1.0)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """Prepend the computed pose block onto the policy's (reduced) action -> full-width vector."""
        if not self._validated:
            self._validate()
        applied_aa = None
        if self._fix_orientation:
            self._capture_reset_orientation()                               # latch spawn orient. on reset
            applied_aa = self._applied_rot_axis_angle()                     # the rotation actually applied
        head = self._pos_action(applied_aa if self._decouple_lever else None)
        if self._fix_orientation:
            head = torch.cat((head, self._rot_action(applied_aa)), dim=1)   # (E,6)
        return torch.cat((head, action), dim=1)                             # (E, full_dim)
