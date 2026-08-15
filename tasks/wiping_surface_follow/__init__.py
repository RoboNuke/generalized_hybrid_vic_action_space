"""Wiping surface task — a wiping policy on the per-env curved ridge, holding a rectangular SPONGE.
Reuses the curved surface + obs/action spaces; changes the held object (box) and the reward (base
reward of arXiv:2502.12599) with a sequence of on-surface waypoints visited sequentially.

Importing this module registers the gym id ``Isaac-FlatSurfaceFollow-Wiping-Direct-v0``. The
``Isaac-FlatSurfaceFollow-`` PREFIX is kept so every ``env_setup`` gate keyed off
``task.startswith("Isaac-FlatSurfaceFollow-")`` applies unchanged; ``-Wiping-`` distinguishes it.
"""

import gymnasium as gym

from .wiping_surface_follow_env import WipingSurfaceFollowEnv
from .wiping_surface_follow_env_cfg import WipingSurfaceFollowEnvCfg

gym.register(
    id="Isaac-FlatSurfaceFollow-Wiping-Direct-v0",
    # MUST be the "module:Class" string form so env_setup.py can resolve the concrete env class via
    # gym.spec(...).entry_point.split(":") (camera/contact patching).
    entry_point="tasks.wiping_surface_follow.wiping_surface_follow_env:WipingSurfaceFollowEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": WipingSurfaceFollowEnvCfg},
)
