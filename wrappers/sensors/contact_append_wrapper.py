"""Append the per-axis contact state to the policy observation and/or critic state.

Sits OUTERMOST in the wrapper stack (around the skrl ``IsaacLabWrapper`` scorer) and
concatenates the contact bool (``env.unwrapped.in_contact``, shape ``(num_envs, 3)`` =
x/y/z, set each step by :class:`~wrappers.sensors.contact_sensor_wrapper.ContactSensorWrapper`)
onto the tensors the agent consumes:

* ``append_to_policy_obs``   — grows ``observation_space`` and appends to the obs returned
  by ``step()`` / ``reset()`` (the policy input).
* ``append_to_critic_state`` — grows ``state_space`` and appends to the tensor returned by
  ``state()`` (the critic input; asymmetric tasks only).

The two toggles are independent. Alignment is automatic: the wrapper appends the
just-updated contact to the obs/state it returns, so obs(t) carries contact(t). The
contact passes through the agent's observation preprocessor (i.e. it is normalized).

NOTE: composes via attribute delegation (like ``wrappers.recording.RecordingWrapper``)
rather than subclassing ``gymnasium.Wrapper`` — the wrapped scorer is a skrl
``IsaacLabWrapper``, not a ``gymnasium.Env``, so ``gym.Wrapper`` would reject it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _grow_box(space: gym.spaces.Box, k: int) -> gym.spaces.Box:
    """Return a copy of ``space`` (a 1-D Box) widened by ``k`` contact dims in [0, 1]."""
    low = np.concatenate([np.asarray(space.low).reshape(-1), np.zeros(k, dtype=np.float32)])
    high = np.concatenate([np.asarray(space.high).reshape(-1), np.ones(k, dtype=np.float32)])
    return gym.spaces.Box(low=low, high=high, dtype=space.dtype)


class ContactAppendWrapper:
    def __init__(
        self,
        env,
        *,
        append_to_policy_obs: bool = False,
        append_to_critic_state: bool = False,
        contact_dim: int = 3,
    ):
        self._env = env
        self._append_obs = bool(append_to_policy_obs)
        self._append_state = bool(append_to_critic_state)
        self._k = int(contact_dim)

        self._obs_space = env.observation_space
        if self._append_obs:
            self._obs_space = _grow_box(self._obs_space, self._k)

        base_state_space = getattr(env, "state_space", None)
        if self._append_state and base_state_space is not None:
            self._state_space = _grow_box(base_state_space, self._k)
        else:
            self._state_space = base_state_space

    def __getattr__(self, name: str) -> Any:
        # Called only when ``name`` is not found on self — forward to the inner env.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)

    # ---- spaces (overridden; everything else delegates) ----
    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state_space(self):
        return self._state_space

    # ---- contact helper ----
    def _contact(self, n_envs: int) -> torch.Tensor:
        """Current per-axis contact (n_envs, k) as float, or zeros if unavailable."""
        ic = getattr(self._env.unwrapped, "in_contact", None)
        if ic is None:
            return torch.zeros((n_envs, self._k), device=self._env.unwrapped.device)
        return ic.float()

    # ---- interaction ----
    def step(self, actions):
        obs, reward, terminated, truncated, info = self._env.step(actions)
        if self._append_obs:
            obs = torch.cat([obs, self._contact(obs.shape[0])], dim=-1)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        if self._append_obs:
            # No fresh sensor read at reset — start the episode with no-contact (zeros),
            # consistent with the SSL buffer's done-reset behavior.
            obs = torch.cat([obs, torch.zeros((obs.shape[0], self._k), device=obs.device)], dim=-1)
        return obs, info

    def state(self):
        state = self._env.state()
        if self._append_state and state is not None:
            state = torch.cat([state, self._contact(state.shape[0])], dim=-1)
        return state

    def close(self) -> None:
        self._env.close()
