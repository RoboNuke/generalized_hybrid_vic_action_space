"""Fragile peg: terminate an env when its contact force exceeds a break threshold.

When the contact-force magnitude on the held peg reaches ``break_force`` (Newtons), the
peg is considered broken: that env is terminated immediately so it gets reset on the spot.
Force is read from the Forge force-torque sensor the same way the FORGE contact-penalty
reward reads it — ``‖force_sensor_smooth[:, :3]‖`` (the smoothed EE-frame force) — so the
break threshold lives on the same scale as the env's per-env *threshold force*
(``contact_penalty_thresholds``). The runner caps that threshold-force range at
``break_force`` (see :func:`learning.env_setup.build_env`), so the policy is never rewarded
for tolerating a force that would break the peg.

This wrapper only adds termination (no extra reward term — breaking just ends the episode):
it monkeypatches the unwrapped env's ``_get_dones`` to OR a force-violation mask into the
``terminated`` flag. The actual per-env reset of the broken envs is then performed in the
same physics step by Isaac Lab's ``_reset_idx`` (made cheap + correct by the companion
:class:`~wrappers.sensors.efficient_reset_wrapper.EfficientResetWrapper`, which MUST also be
installed — broken envs reset out of sync with the rest).

Install AFTER the control wrapper and the efficient-reset wrapper, INSIDE the scorer (so the
scorer sees the final ``terminated``). Lazy-inits on the first ``step``/``reset`` once the
robot exists, mirroring the other wrappers.
"""

from __future__ import annotations

import gymnasium as gym
import torch

# Stand-in "force" used for an unbreakable peg (break_force == -1). Large enough that the
# smoothed force never reaches it, small enough to stay well inside float32 range.
_UNBREAKABLE_FORCE = float(2**23)


class FragileObjectWrapper(gym.Wrapper):
    """Terminate envs whose smoothed contact force reaches ``break_force`` (fragile peg)."""

    def __init__(self, env, *, break_force: float, num_agents: int = 1) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        # num_agents is accepted for signature symmetry with the control wrappers; break_force
        # is a single scalar applied to every env (per the configured design).
        self.num_agents = int(num_agents)

        bf = _UNBREAKABLE_FORCE if float(break_force) < 0.0 else float(break_force)
        self.break_force = torch.full((self.num_envs,), bf, dtype=torch.float32, device=self.device)
        # If every env is unbreakable there is nothing to monitor.
        self._fragile = bool(torch.any(self.break_force < _UNBREAKABLE_FORCE).item())

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
                "(env.force_sensor_smooth). Use a Forge task (Isaac-Forge-*) or an "
                "AutoMate-Assembly task (the adapter installs the force sensor); stock "
                "Factory has no force sensing."
            )
        if not hasattr(self.unwrapped, "_get_dones"):
            raise RuntimeError("[fragile] env has no _get_dones to wrap.")
        self._original_get_dones = self.unwrapped._get_dones
        self.unwrapped._get_dones = self._wrapped_get_dones
        self._wrapper_initialized = True

    # ----------------------------------------------------------------- dones
    def _wrapped_get_dones(self):
        terminated, time_out = self._original_get_dones()
        if self._fragile:
            force_mag = torch.linalg.norm(self.unwrapped.force_sensor_smooth[:, :3], dim=1)
            violations = force_mag >= self.break_force
            self._last_violations = violations
            terminated = torch.logical_or(terminated, violations)
            if hasattr(self.unwrapped, "extras"):
                to_log = self.unwrapped.extras.setdefault("to_log", {})
                # Per-env break flag -> mean-reduced by the scorer to a per-step break rate.
                to_log["Fragile / Peg Break"] = violations.float()
        return terminated, time_out

    # ------------------------------------------------------------------ gym
    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return out
