"""Efficient per-env reset for Forge / Factory / AutoMate peg insertion.

Factory/Forge's ``_reset_idx`` is written assuming *all envs reset at the same time*
(``factory_env.py``: "We assume all envs will always be reset at the same time."). It moves
the assets/robot to default poses and then runs ``step_sim_no_action()`` settling loops and
``randomize_initial_state`` (DLS-IK loops that also step the whole sim) — operations that
advance *every* env's physics, so resetting only a subset corrupts the envs that are still
mid-episode (and is expensive besides).

When pegs are fragile (see :class:`~wrappers.sensors.fragile_object_wrapper.FragileObjectWrapper`)
individual envs terminate out of sync, triggering exactly those partial resets. This wrapper
fixes that:

* **Full reset** (``len(env_ids) == num_envs``): run the normal reset chain, then cache the
  whole scene's post-reset physics state (relative to env origins) plus the Factory/Forge
  per-env bookkeeping tensors.
* **Partial reset** (a subset): skip the expensive Factory path. Call the lightweight
  ``DirectRLEnv._reset_idx`` (scene/event/noise reset + zero ``episode_length_buf``), then
  *teleport* each broken env to a randomly-chosen donor env's cached post-reset state via
  ``scene.reset_to(..., is_relative=True)`` (which re-bases poses onto the broken env's
  origin), copy the donor's per-env bookkeeping tensors, **re-sample the Forge threshold
  force** for the broken envs (the most important Forge-specific fix — the donor-copied value
  would also be valid, but a fresh draw from the capped range matches native semantics and
  guarantees ``threshold <= break_force``), and zero the FT-smoothing state.

The per-step ``_reset_buffers`` (``ep_succeeded`` / Forge success-prediction buffers) is
already handled by Factory's ``_pre_physics_step`` keying off ``reset_buf`` — broken envs are
in ``reset_buf`` — so it is not repeated here.

Install order matters: this wrapper is installed AFTER (outside) the control wrapper but
INSIDE the fragile wrapper, so the runtime ``_reset_idx`` chain is
``control._wrapped -> efficient._wrapped -> (Forge full reset | DirectRL partial reset)``.
That means the control wrapper's per-env EMA/VIC reset bookkeeping runs for broken envs too.
"""

from __future__ import annotations

import gymnasium as gym
import torch

# Factory per-env bookkeeping tensors (env-relative or frame-independent, so a donor->broken
# copy needs no origin adjustment) that the lightweight DirectRL reset path does NOT restore.
_FACTORY_ATTRS = (
    "fixed_pos_obs_frame",
    "init_fixed_pos_obs_noise",
    "prev_joint_pos",
    "prev_fingertip_pos",
    "prev_fingertip_quat",
    "actions",
    "prev_actions",
    "ee_angvel_fd",
    "ee_linvel_fd",
)
# Forge-only per-env episode randomization that Forge._reset_idx re-samples for all envs each
# reset (and that the DirectRL path skips). Copied from the donor (a valid fresh sample);
# `contact_penalty_thresholds` is additionally re-sampled below.
_FORGE_ATTRS = (
    "ema_factor",
    "task_prop_gains",
    "task_deriv_gains",
    "pos_threshold",
    "rot_threshold",
    "dead_zone_thresholds",
    "flip_quats",
    "contact_penalty_thresholds",
)


class EfficientResetWrapper(gym.Wrapper):
    """Teleport-based partial reset that avoids Factory/Forge's all-envs settling path."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs

        self._wrapper_initialized = False
        self._full_reset_idx = None   # genuine (chained) full reset, captured at init
        self._direct_reset_idx = None  # bound DirectRLEnv._reset_idx (lightweight)
        self._cached_state = None      # scene.get_state(is_relative=True) post full reset
        self._cached_attrs = {}        # per-env bookkeeping tensors post full reset

    # ------------------------------------------------------------------ setup
    def _initialize_wrapper(self) -> None:
        if self._wrapper_initialized:
            return
        u = self.unwrapped
        if not hasattr(u, "_reset_idx"):
            raise RuntimeError("[efficient-reset] env has no _reset_idx to wrap.")
        if not hasattr(u, "scene") or not hasattr(u.scene, "get_state"):
            raise RuntimeError(
                "[efficient-reset] env scene has no get_state/reset_to; this IsaacLab "
                "version is unsupported."
            )
        # Resolve the lightweight DirectRLEnv._reset_idx (scene/event/noise reset only) and
        # bind it to the unwrapped instance, bypassing Factory/Forge's settling reset.
        direct_cls = next(
            (c for c in type(u).__mro__ if c.__name__ == "DirectRLEnv" and "_reset_idx" in c.__dict__),
            None,
        )
        if direct_cls is None:
            raise RuntimeError(
                "[efficient-reset] could not find DirectRLEnv._reset_idx in the env MRO; "
                "the partial-reset path needs the lightweight base reset."
            )
        self._direct_reset_idx = direct_cls._reset_idx.__get__(u, type(u))
        # Capture whatever is currently bound as the FULL reset (genuine Forge/Factory reset,
        # since this wrapper inits before the control wrapper re-patches _reset_idx).
        self._full_reset_idx = u._reset_idx
        u._reset_idx = self._wrapped_reset_idx
        self._wrapper_initialized = True

    # --------------------------------------------------------------- reset_idx
    def _wrapped_reset_idx(self, env_ids):
        env_ids = self._as_long(env_ids)
        # Full reset (or first reset before any cache exists): run the normal chain + cache.
        if env_ids.numel() >= self.num_envs or self._cached_state is None:
            self._full_reset_idx(env_ids)
            self._cache_states()
            return
        # Partial reset: lightweight base reset, then teleport from the cached donor states.
        self._direct_reset_idx(env_ids)
        self.unwrapped.episode_length_buf[env_ids] = 0
        self._perform_efficient_reset(env_ids)

    def _cache_states(self) -> None:
        u = self.unwrapped
        # Relative scene state so donor poses can be re-based onto a different env origin.
        self._cached_state = u.scene.get_state(is_relative=True)
        # Tasks may declare extra per-env tensors that their (non-Factory) reset sets and the
        # DirectRL partial path skips — e.g. the surface task's episode clocks / desired force.
        # Captured post-full-reset, so a donor copy restores correct fresh-episode values.
        extra_attrs = tuple(getattr(u, "_efficient_reset_extra_attrs", ()) or ())
        self._cached_attrs = {
            name: getattr(u, name).clone()
            for name in (*_FACTORY_ATTRS, *_FORGE_ATTRS, *extra_attrs)
            if hasattr(u, name)
        }

    def _perform_efficient_reset(self, env_ids) -> None:
        u = self.unwrapped
        n = int(env_ids.numel())
        source_idxs = torch.randint(0, self.num_envs, (n,), device=self.device)

        # 1) Physics state: write each broken env from a donor's cached relative state.
        #    is_relative=True re-bases the donor pose onto the broken env's origin.
        shuffled = self._shuffle_state(self._cached_state, source_idxs)
        u.scene.reset_to(shuffled, env_ids=env_ids, is_relative=True)

        # 2) Per-env bookkeeping tensors the DirectRL path didn't restore.
        for name, cached in self._cached_attrs.items():
            getattr(u, name)[env_ids] = cached[source_idxs]

        # 3) Re-sample the Forge threshold force fresh from the (break-force-capped) range, so
        #    broken envs get a brand-new episode threshold that can never exceed break_force.
        if hasattr(u, "contact_penalty_thresholds"):
            rng = self._threshold_range()
            if rng is not None:
                lo, hi = rng
                u.contact_penalty_thresholds[env_ids] = lo + torch.rand(n, device=self.device) * (hi - lo)

        # 4) Clear the FT-smoothing state so a stale pre-break force can't trigger an immediate
        #    re-break or feed a bogus contact penalty on the first post-reset step.
        for name in ("force_sensor_world_smooth", "force_sensor_smooth"):
            if hasattr(u, name):
                getattr(u, name)[env_ids] = 0.0

    # ------------------------------------------------------------------ utils
    def _shuffle_state(self, state, source_idxs):
        """Recursively rebuild the get_state() dict, indexing per-env tensors by donor."""
        out = {}
        for key, value in state.items():
            if isinstance(value, dict):
                out[key] = self._shuffle_state(value, source_idxs)
            elif torch.is_tensor(value) and value.shape[0] == self.num_envs:
                out[key] = value[source_idxs].clone()
            else:
                out[key] = value
        return out

    def _threshold_range(self):
        cfg = getattr(self.unwrapped, "cfg", None)
        task = getattr(cfg, "task", None) if cfg is not None else None
        rng = getattr(task, "contact_penalty_threshold_range", None) if task is not None else None
        if rng is None or len(rng) != 2:
            return None
        return float(rng[0]), float(rng[1])

    def _as_long(self, env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)

    # ------------------------------------------------------------------ gym
    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return super().reset(**kwargs)
