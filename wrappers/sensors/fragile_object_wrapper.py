"""Fragile peg: terminate an env when its contact force exceeds a break threshold.

When the contact force on the held peg exceeds the configured break threshold, the peg is
considered broken: that env is terminated immediately so it gets reset on the spot. Force is
read from the Forge force-torque sensor the same way the FORGE contact-penalty reward reads it
— ``force_sensor_smooth[:, :3]`` (the smoothed WORLD-frame force; the sensor's identity-rotation
``change_FT_frame`` re-references only the torque, so this force vector stays world) — so the
break threshold lives on the same scale as the env's per-env *threshold force*
(``contact_penalty_thresholds``). The runner caps that threshold-force range at the break force
(see :func:`learning.env_setup.build_env`), so the policy is never rewarded for tolerating a
force that would break the peg.

Two force-break modes:

* **magnitude** (default): break when ``‖force_sensor_smooth[:, :3]‖`` reaches the scalar
  ``break_force``.
* **directional** (``direction_break_force=True``): ``break_force`` is ``[shear, normal]``. The
  measured force is projected onto the held peg's long axis (its local +z rotated to world by
  ``held_quat``). The AXIAL component magnitude (force along the long axis) is compared to
  ``normal``; the residual PERPENDICULAR component magnitude (force with the axial part removed)
  is compared to ``shear``. Either exceedance breaks the peg. NOTE this is a true vector
  projection onto the (physics-realized, per-step) peg axis, not an index into the force vector.

Plus an optional **loss-of-contact** failure mode (``require_contact=True``, independent of the
force break — works even with an unbreakable peg): once an env has been in contact (per-axis
``env.in_contact`` reads True on any axis) at least once this episode, dropping out of contact on
ALL axes terminates it as if the peg broke. The check ARMS only after first contact (the peg
spawns above the surface and must descend first); the per-env "has contacted" latch resets each
episode. Requires the contact-sensor wrapper (it populates ``env.in_contact``).

This wrapper only adds termination (no extra reward term — breaking just ends the episode): it
monkeypatches the unwrapped env's ``_get_dones`` to OR a force-violation mask into the
``terminated`` flag. The actual per-env reset of the broken envs is then performed in the same
physics step by Isaac Lab's ``_reset_idx`` (made cheap + correct by the companion
:class:`~wrappers.sensors.efficient_reset_wrapper.EfficientResetWrapper`, which MUST also be
installed — broken envs reset out of sync with the rest).

Install AFTER the control wrapper and the efficient-reset wrapper, INSIDE the scorer (so the
scorer sees the final ``terminated``). Lazy-inits on the first ``step``/``reset`` once the robot
exists, mirroring the other wrappers.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply

# Stand-in "force" used for an unbreakable direction (threshold == -1). Large enough that the
# smoothed force never reaches it, small enough to stay well inside float32 range.
_UNBREAKABLE_FORCE = float(2**23)


def _as_break_tensor(value: float, num_envs: int, device) -> torch.Tensor:
    """Per-env break-threshold tensor; negative -> unbreakable (huge)."""
    v = _UNBREAKABLE_FORCE if float(value) < 0.0 else float(value)
    return torch.full((num_envs,), v, dtype=torch.float32, device=device)


class FragileObjectWrapper(gym.Wrapper):
    """Terminate envs whose contact force reaches the break threshold (fragile peg).

    ``break_force`` is a scalar force magnitude in the default (magnitude) mode, or a length-2
    ``[shear, normal]`` array when ``direction_break_force=True``.
    """

    def __init__(
        self,
        env,
        *,
        break_force,
        direction_break_force: bool = False,
        require_contact: bool = False,
        num_agents: int = 1,
    ) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        # num_agents is accepted for signature symmetry with the control wrappers; the break
        # threshold(s) are applied to every env (per the configured design).
        self.num_agents = int(num_agents)
        self.direction_break_force = bool(direction_break_force)
        self.require_contact = bool(require_contact)

        if self.direction_break_force:
            shear, normal = float(break_force[0]), float(break_force[1])
            self.shear_force = _as_break_tensor(shear, self.num_envs, self.device)
            self.normal_force = _as_break_tensor(normal, self.num_envs, self.device)
            self._fragile = bool(
                torch.any(self.shear_force < _UNBREAKABLE_FORCE).item()
                or torch.any(self.normal_force < _UNBREAKABLE_FORCE).item()
            )
        else:
            self.break_force = _as_break_tensor(float(break_force), self.num_envs, self.device)
            self._fragile = bool(torch.any(self.break_force < _UNBREAKABLE_FORCE).item())

        # Whether ANY failure mode is active (force break and/or loss-of-contact). When neither
        # is armed the wrapped _get_dones is a pure pass-through.
        self._active = self._fragile or self.require_contact

        # Held-peg local long axis (surface/Forge/AutoMate convention: the held body's +z).
        self._peg_axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Loss-of-contact latch: per-env "has been in contact this episode" (arms the check),
        # and the set of envs that will be reset at the end of the current step (so their latch
        # is cleared at the start of the next step, before it re-arms on new contact).
        self._has_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._original_get_dones = None
        self._wrapper_initialized = False
        self._last_violations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if hasattr(self.unwrapped, "extras") and "to_log" not in self.unwrapped.extras:
            self.unwrapped.extras["to_log"] = {}

    # ------------------------------------------------------------------ setup
    def _initialize_wrapper(self) -> None:
        if self._wrapper_initialized:
            return
        if not hasattr(self.unwrapped, "force_sensor_smooth"):
            raise RuntimeError(
                "FragileObjectWrapper requires the Forge force sensor "
                "(env.force_sensor_smooth). Use a Forge task (Isaac-Forge-*), an "
                "AutoMate-Assembly task (the adapter installs the force sensor), or the "
                "FlatSurfaceFollow task; stock Factory has no force sensing."
            )
        if self.direction_break_force and not hasattr(self.unwrapped, "held_quat"):
            raise RuntimeError(
                "FragileObjectWrapper(direction_break_force=True) requires env.held_quat to "
                "project the contact force onto the peg long axis. Use a task that exposes the "
                "held-object orientation (Forge / AutoMate-Assembly / FlatSurfaceFollow)."
            )
        if self.require_contact and not hasattr(self.unwrapped, "in_contact"):
            raise RuntimeError(
                "FragileObjectWrapper(require_contact=True) reads env.in_contact (per-axis "
                "contact bool), which is populated by the contact-sensor wrapper. Enable it "
                "(sensor_cfg.contact.enabled=True)."
            )
        if not hasattr(self.unwrapped, "_get_dones"):
            raise RuntimeError("[fragile] env has no _get_dones to wrap.")
        self._original_get_dones = self.unwrapped._get_dones
        self.unwrapped._get_dones = self._wrapped_get_dones
        self._wrapper_initialized = True

    # ----------------------------------------------------------------- dones
    def _compute_violations(self):
        """Per-env FORCE break mask (and per-cause log breakdown) for the smoothed contact force.

        The combined "Fragile / Peg Break" total (all causes) is emitted by the caller.
        """
        force = self.unwrapped.force_sensor_smooth[:, :3]  # (E,3) world-frame force
        to_log = {}
        if self.direction_break_force:
            # Peg long axis in WORLD frame: held body's local +z rotated by held_quat. Unit by
            # construction (held_quat is a unit quaternion); renormalize for numerical safety.
            axis = quat_apply(self.unwrapped.held_quat, self._peg_axis_local.expand(self.num_envs, 3))
            axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
            # Signed axial (normal) component = force . axis; residual is the shear vector.
            axial = (force * axis).sum(dim=1)                  # (E,) signed force along the axis
            axial_mag = axial.abs()
            shear_vec = force - axial.unsqueeze(-1) * axis     # (E,3) force with axial part removed
            shear_mag = torch.linalg.norm(shear_vec, dim=1)
            normal_violation = axial_mag >= self.normal_force
            shear_violation = shear_mag >= self.shear_force
            violations = torch.logical_or(normal_violation, shear_violation)
            to_log["Fragile / Break Normal"] = normal_violation.float()
            to_log["Fragile / Break Shear"] = shear_violation.float()
        else:
            force_mag = torch.linalg.norm(force, dim=1)
            violations = force_mag >= self.break_force
        return violations, to_log

    def _contact_loss_violations(self):
        """Per-env loss-of-contact break mask (and log series).

        Arms per env only after its first in-contact reading this episode; once armed, dropping
        out of contact on every axis is a break. ``env.in_contact`` is the contact-sensor
        wrapper's per-axis bool, refreshed at the END of the previous step (a one-step lag that,
        for a just-reset env, already reflects its post-reset out-of-contact spawn state).
        """
        in_contact_any = self.unwrapped.in_contact.any(dim=1)          # (E,) bool
        # Arm the latch on (and including) the first contact.
        self._has_contacted = torch.logical_or(self._has_contacted, in_contact_any)
        violations = torch.logical_and(self._has_contacted, torch.logical_not(in_contact_any))
        return violations, {"Fragile / Contact Loss": violations.float()}

    def _wrapped_get_dones(self):
        terminated, time_out = self._original_get_dones()
        if not self._active:
            return terminated, time_out

        # Clear the "has contacted" latch for envs that were reset at the end of the PREVIOUS
        # step (they start the new episode out of contact and must re-arm on fresh contact).
        self._has_contacted = torch.logical_and(
            self._has_contacted, torch.logical_not(self._reset_mask)
        )

        violations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        to_log = {}
        if self._fragile:
            force_violations, force_log = self._compute_violations()
            violations = torch.logical_or(violations, force_violations)
            to_log.update(force_log)
        if self.require_contact:
            contact_violations, contact_log = self._contact_loss_violations()
            violations = torch.logical_or(violations, contact_violations)
            to_log.update(contact_log)

        # Total break rate (all causes) -> the headline "Peg Break" series; per-cause tags above.
        to_log["Fragile / Peg Break"] = violations.float()
        self._last_violations = violations
        terminated = torch.logical_or(terminated, violations)
        # Envs reset at the end of THIS step (DirectRLEnv resets terminated|time_out in-step);
        # remember them so their contact latch clears next step.
        self._reset_mask = torch.logical_or(terminated, time_out)
        if hasattr(self.unwrapped, "extras"):
            # Per-env break flags -> mean-reduced by the scorer to per-step break rates.
            self.unwrapped.extras.setdefault("to_log", {}).update(to_log)
        return terminated, time_out

    # ------------------------------------------------------------------ gym
    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        # A full reset re-spawns every env out of contact — clear the loss-of-contact latch so
        # the check re-arms only on fresh contact (per-env done resets are handled in _get_dones).
        self._has_contacted.zero_()
        self._reset_mask.zero_()
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return out
