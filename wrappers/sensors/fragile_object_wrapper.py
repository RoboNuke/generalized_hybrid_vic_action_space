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
episode. Requires the contact-sensor wrapper (it populates ``env.in_contact``). A configurable
``require_contact_grace_steps`` (default 5) suppresses this break for the first N steps of each
episode so the reset-press rebound / contact vibration can settle before it can terminate — the
earliest loss-of-contact failure is step N+1 (the force break is unaffected by the grace).

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
        require_contact_grace_steps: int = 5,
        require_contact_debounce_steps: int = 3,
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
        # Loss-of-contact grace: suppress the loss-of-contact break for the first
        # `require_contact_grace_steps` steps of each episode so the reset-press rebound / contact
        # vibration can settle before the check can terminate. episode_length_buf is incremented
        # BEFORE _get_dones (direct_rl_env.step) and reset to 0 per-env on reset, so it reads 1 on
        # the first step; suppressing buf <= grace lets the break first fire at step grace+1
        # (grace=5 -> earliest loss-of-contact failure at step 6). 0 disables the grace. Applies
        # ONLY to the loss-of-contact mode; the force break is unaffected.
        self.require_contact_grace_steps = int(require_contact_grace_steps)
        # Loss-of-contact DEBOUNCE: the peg must read out-of-contact on ALL axes for this many
        # CONSECUTIVE steps before it counts as a contact-loss break, so a single-step sensor blip /
        # bounce doesn't end the episode (the peg gets a chance to re-seat). Default 3. A value of 1
        # is the old behaviour (any single off-contact step breaks). The per-env consecutive
        # out-of-contact streak resets to 0 the moment contact is re-made (or on episode reset).
        self.require_contact_debounce_steps = max(1, int(require_contact_debounce_steps))
        self._ooc_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        # Per-episode metric payload built in _wrapped_get_dones, merged into the env's
        # per_env_episode_stat channel in step() (block_agent averages it over FINISHED episodes).
        self._ep_stats: dict | None = None

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
        # Expose the break threshold(s) on the unwrapped env so the surface recorder can scale its
        # force bar so the TOP is the (normal-axis) break force. In magnitude mode there is one
        # scalar threshold on the total force; the normal component can't exceed it, so it doubles as
        # the normal-axis cap. In directional mode the normal axis has its own threshold.
        self.unwrapped.is_fragile = bool(self._fragile)
        if self._fragile:
            self.unwrapped.fragile_normal_break_force = (
                self.normal_force if self.direction_break_force else self.break_force
            )
            if self.direction_break_force:
                self.unwrapped.fragile_shear_break_force = self.shear_force
        self._wrapper_initialized = True

    # ----------------------------------------------------------------- dones
    def _compute_violations(self):
        """Per-env FORCE break masks for the smoothed contact force: (total, normal, shear).

        In magnitude mode ``normal``/``shear`` are None (no directional breakdown)."""
        force = self.unwrapped.force_sensor_smooth[:, :3]  # (E,3) world-frame force
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
            return violations, normal_violation, shear_violation
        force_mag = torch.linalg.norm(force, dim=1)
        violations = force_mag >= self.break_force
        return violations, None, None

    def _contact_loss_violations(self):
        """Per-env loss-of-contact break mask.

        Arms per env only after its first in-contact reading this episode; once armed, dropping
        out of contact on every axis is a break. ``env.in_contact`` is the contact-sensor
        wrapper's per-axis bool, refreshed at the END of the previous step (a one-step lag that,
        for a just-reset env, already reflects its post-reset out-of-contact spawn state).
        """
        in_contact_any = self.unwrapped.in_contact.any(dim=1)          # (E,) bool
        # Arm the latch on (and including) the first contact — always, even during the grace window,
        # so "has contacted this episode" stays correct; the grace only gates TERMINATION below.
        self._has_contacted = torch.logical_or(self._has_contacted, in_contact_any)
        armed_and_out = torch.logical_and(self._has_contacted, torch.logical_not(in_contact_any))
        # Consecutive out-of-contact streak: +1 while armed AND out of contact, reset to 0 the moment
        # contact is re-made (or not yet armed). A break counts only once the streak reaches the
        # debounce length, so a transient one-step blip is tolerated and can re-seat.
        self._ooc_streak = torch.where(
            armed_and_out, self._ooc_streak + 1, torch.zeros_like(self._ooc_streak)
        )
        violations = self._ooc_streak >= self.require_contact_debounce_steps
        if self.require_contact_grace_steps > 0:
            # No loss-of-contact break until buf > grace (earliest failure at step grace+1).
            past_grace = self.unwrapped.episode_length_buf > self.require_contact_grace_steps
            violations = torch.logical_and(violations, past_grace)
        return violations

    def _wrapped_get_dones(self):
        terminated, time_out = self._original_get_dones()
        if not self._active:
            self._ep_stats = None
            return terminated, time_out

        # Clear the "has contacted" latch AND the out-of-contact streak for envs that were reset at
        # the end of the PREVIOUS step (they start the new episode out of contact and must re-arm).
        self._has_contacted = torch.logical_and(
            self._has_contacted, torch.logical_not(self._reset_mask)
        )
        self._ooc_streak = torch.where(
            self._reset_mask, torch.zeros_like(self._ooc_streak), self._ooc_streak
        )

        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        violations = z
        normal_violation = shear_violation = None
        contact_violations = z
        if self._fragile:
            force_violations, normal_violation, shear_violation = self._compute_violations()
            violations = torch.logical_or(violations, force_violations)
        if self.require_contact:
            contact_violations = self._contact_loss_violations()
            violations = torch.logical_or(violations, contact_violations)
        self._last_violations = violations

        # PER-EPISODE metrics (NOT per-step): each entry is averaged by block_agent over the episodes
        # that FINISH this step (mask = env.reset_buf, NaN-skipped) via per_env_episode_stat, merged in
        # step(). A break fires on the episode's TERMINAL step, so the 0/1 flag over finishing episodes
        # is the per-episode break RATE, and `episode_length_buf` at the break is the step it happened
        # (NaN elsewhere -> the "... step" tags average the step ONLY over episodes of that cause).
        buf = self.unwrapped.episode_length_buf.float()
        nan = torch.full_like(buf, float("nan"))
        ep = {
            "Fragile / Peg Break": violations.float(),
            "Fragile / Peg Break step": torch.where(violations, buf, nan),
        }
        if self.require_contact:
            ep["Fragile / Contact Loss"] = contact_violations.float()
            ep["Fragile / Contact Loss step"] = torch.where(contact_violations, buf, nan)
        if self._fragile and self.direction_break_force:
            ep["Fragile / Break Normal"] = normal_violation.float()
            ep["Fragile / Break Shear"] = shear_violation.float()
            ep["Fragile / Break Normal step"] = torch.where(normal_violation, buf, nan)
            ep["Fragile / Break Shear step"] = torch.where(shear_violation, buf, nan)
        self._ep_stats = ep

        terminated = torch.logical_or(terminated, violations)
        # Envs reset at the end of THIS step (DirectRLEnv resets terminated|time_out in-step);
        # remember them so their contact latch clears next step.
        self._reset_mask = torch.logical_or(terminated, time_out)
        return terminated, time_out

    # ------------------------------------------------------------------ gym
    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        out = super().step(action)
        # Merge the per-episode fragile metrics (built in _wrapped_get_dones during the inner step)
        # into the env's per_env_episode_stat channel. block_agent averages these over the episodes
        # that finished this step (mask = reset_buf) -> per-episode break rates + average break step.
        ep = self._ep_stats
        if ep and isinstance(out[-1], dict):
            info = out[-1]
            info.setdefault("per_env_episode_stat", {}).update(ep)
            info.setdefault("per_env_episode_stat_mask", self.unwrapped.reset_buf.clone())
        self._ep_stats = None
        return out

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        # A full reset re-spawns every env out of contact — clear the loss-of-contact latch + streak
        # so the check re-arms only on fresh contact (per-env done resets are handled in _get_dones).
        self._has_contacted.zero_()
        self._ooc_streak.zero_()
        self._reset_mask.zero_()
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return out
