"""Env-defined INTERACTION frame for Factory/Forge peg-insertion tasks.

The interaction frame is the frame at the point where the held object contacts the
environment. It exists ONLY while in contact (gated by the contact-sensor wrapper's
``env.in_contact``) and is consumed by the visualizer (and, optionally, observations) —
it does NOT feed control (the rotated-mode stiffness rotation is policy/config-defined).

For the surface-following task the env computes its own interaction frame (rim contact
point, z = surface normal). For the upstream peg tasks (FactoryEnv/ForgeEnv, which we
don't subclass) this thin wrapper supplies the analogous frame:

  * position = the held asset's geometric base (the insertion tip), via
    ``factory_utils.get_held_base_pose`` (env-relative, like ``held_pos``);
  * orientation = the fixed-asset (hole) frame — z is the hole/insertion axis, x the
    fixed-asset x — i.e. ``fixed_quat``.

All quantities are exposed on ``env.unwrapped`` as ``interaction_pos`` (env-relative),
``interaction_quat``, and ``interaction_exists`` (bool, per env), matching what the
surface env publishes so the visualizer reads them uniformly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch


class InteractionFrameWrapper(gym.Wrapper):
    """Publish the peg-insertion interaction frame on the env each step."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        u = self.unwrapped
        E, dev = u.num_envs, u.device
        u.interaction_pos = torch.zeros((E, 3), device=dev)
        u.interaction_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).unsqueeze(0).repeat(E, 1)
        u.interaction_exists = torch.zeros(E, dtype=torch.bool, device=dev)

    def _update(self) -> None:
        from isaaclab_tasks.direct.factory import factory_utils

        u = self.unwrapped
        held_base_pos, _ = factory_utils.get_held_base_pose(
            u.held_pos, u.held_quat, u.cfg_task.name, u.cfg_task.fixed_asset_cfg, u.num_envs, u.device
        )
        u.interaction_pos = held_base_pos                      # env-relative (viz adds env_origins)
        u.interaction_quat = u.fixed_quat                      # hole frame: z = insertion axis, x = fixed x
        in_contact = getattr(u, "in_contact", None)
        u.interaction_exists = (
            in_contact.any(dim=1)
            if in_contact is not None
            else torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        )

    def step(self, action):
        out = super().step(action)
        # Physics has advanced; held/fixed poses + in_contact are current.
        self._update()
        return out
