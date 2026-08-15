"""Curved surface path-following task — a direct extension of the flat surface task with a single
cylindrical RIDGE across the path. Unlike the bumpy task, the curvature is REAL and fully reported
(the surface normal / contact point / observation setpoint are computed against the ridge). Difficulty
is a per-env curvature scalar alpha in [0, 1] (round-robin over a configurable grid of levels).

Importing this module registers the gym id ``Isaac-FlatSurfaceFollow-Curved-Direct-v0``. The
``Isaac-FlatSurfaceFollow-`` PREFIX is kept deliberately so every ``env_setup`` gate that keys off
``task.startswith("Isaac-FlatSurfaceFollow-")`` (control wrappers, grasp/weld, keypoint servo,
surface termination) applies to the curved task with no changes; ``-Curved-`` distinguishes it.
"""

import gymnasium as gym

from .curved_surface_follow_env import CurvedSurfaceFollowEnv
from .curved_surface_follow_env_cfg import CurvedSurfaceFollowEnvCfg

gym.register(
    id="Isaac-FlatSurfaceFollow-Curved-Direct-v0",
    # MUST be the "module:Class" string form so env_setup.py can resolve the concrete env class via
    # gym.spec(...).entry_point.split(":") (camera/contact patching).
    entry_point="tasks.curved_surface_follow.curved_surface_follow_env:CurvedSurfaceFollowEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CurvedSurfaceFollowEnvCfg},
)
