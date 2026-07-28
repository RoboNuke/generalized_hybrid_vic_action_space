"""Bumpy surface path-following task — a direct extension of the flat surface task with random
spherical-cap bumps on the plate (unobservable; a purely physical perturbation).

Importing this module registers the gym id ``Isaac-FlatSurfaceFollow-Bumpy-Direct-v0``. The
``Isaac-FlatSurfaceFollow-`` PREFIX is kept deliberately so every ``env_setup`` gate that keys off
``task.startswith("Isaac-FlatSurfaceFollow-")`` (control wrappers, grasp/weld, keypoint servo,
surface termination) applies to the bumpy task with no changes; ``-Bumpy-`` distinguishes it.
"""

import gymnasium as gym

from .bumpy_surface_follow_env import BumpySurfaceFollowEnv
from .bumpy_surface_follow_env_cfg import BumpySurfaceFollowEnvCfg

gym.register(
    id="Isaac-FlatSurfaceFollow-Bumpy-Direct-v0",
    # MUST be the "module:Class" string form so env_setup.py can resolve the concrete env class via
    # gym.spec(...).entry_point.split(":") (camera/contact patching).
    entry_point="tasks.bumpy_surface_follow.bumpy_surface_follow_env:BumpySurfaceFollowEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": BumpySurfaceFollowEnvCfg},
)
