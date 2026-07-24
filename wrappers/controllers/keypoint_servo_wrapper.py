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

Orientation (optional ``fix_orientation``) — drive the EEF to a constant world orientation:

    q_target    = quat_from_euler_xyz(fixed_rpy_deg)               # world Euler XYZ
    dq_eef      = eef_quat^-1 o q_target                           # body-frame delta
    action[3:6] = clip(axis_angle(dq_eef) / rot_threshold, -1, 1)

No offset is ever added to orientation. When ``fix_orientation`` is off the policy keeps the
rotation dims.

Action-space surgery: the taken-over dims are a contiguous FRONT block — ``pos`` (0:3) always,
plus ``rot`` (3:6) when ``fix_orientation`` — so the wrapper is agnostic to whatever
force/gain dims the control wrapper appends after them. It shrinks the exposed action space by
3 (or 6) and overwrites ``unwrapped.action_space`` / ``unwrapped.cfg.action_space`` so skrl and
the runner both build the policy against the reduced space; the full-width vector is
reconstructed here before it reaches the control wrapper.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import axis_angle_from_quat

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
        # Number of contiguous FRONT dims this wrapper takes over: pos (3), + rot (3) if fixing it.
        self._n_override = 6 if self._fix_orientation else 3

        # Constant world-frame target orientation (Euler XYZ, degrees -> rad), (1,4), expanded per step.
        if self._fix_orientation:
            rpy = [math.radians(float(v)) for v in cfg.fixed_rpy_deg]
            r, p, y = (torch.tensor([v], dtype=torch.float32, device=self.device) for v in rpy)
            self._q_target = torch_utils.quat_from_euler_xyz(r, p, y)  # (1,4)
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
    def _pos_action(self) -> torch.Tensor:
        """EEF-frame, threshold-normalized position action (E,3) servoing the tip to the setpoint."""
        env = self.unwrapped
        disp_world = env.setpoint_pos - env.held_end_pos                      # (E,3) tip -> keypoint
        if self._any_offset:
            disp_world = (
                disp_world
                + self._along * env.path_dir
                + self._off * env.d_lat
                + self._normal * env.surface_normal
            )
        total_eef = rotate_vec_to_eef(disp_world, env.fingertip_midpoint_quat)  # (E,3) into EEF frame
        return torch.clamp(total_eef / env.pos_threshold, -1.0, 1.0)

    def _rot_action(self) -> torch.Tensor:
        """EEF-frame, threshold-normalized rotation action (E,3) driving the EEF to the fixed quat."""
        env = self.unwrapped
        eef_quat = env.fingertip_midpoint_quat                               # (E,4)
        q_target = self._q_target.expand(self.num_envs, 4)                   # (E,4)
        # Body-frame delta: target = eef o dq  =>  dq = eef^-1 o target.
        dq = torch_utils.quat_mul(torch_utils.quat_conjugate(eef_quat), q_target)
        # Canonicalize to the positive-w hemisphere so axis_angle gives the SHORTEST rotation.
        dq = torch.where(dq[:, 0:1] < 0.0, -dq, dq)
        aa = axis_angle_from_quat(dq)                                        # (E,3) EEF-frame axis-angle
        return torch.clamp(aa / env.rot_threshold, -1.0, 1.0)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """Prepend the computed pose block onto the policy's (reduced) action -> full-width vector."""
        if not self._validated:
            self._validate()
        head = self._pos_action()
        if self._fix_orientation:
            head = torch.cat((head, self._rot_action()), dim=1)             # (E,6)
        return torch.cat((head, action), dim=1)                             # (E, full_dim)
