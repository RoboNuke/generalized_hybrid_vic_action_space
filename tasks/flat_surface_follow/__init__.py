"""Surface path-following task (FORGE-style), defined in this repo.

A Franka holds a thin cylinder (the "held asset", like the Forge peg) and must
trace a straight line across a flat plate (the "fixed asset") from the near-edge
center to the far-edge center, at a target speed, while maintaining a commanded
normal force and a commanded angle between the surface normal and the cylinder
axis. The env subclasses ``ForgeEnv`` so it inherits force sensing and stays
compatible with the repo's hybrid force/position + variable-impedance control
wrappers and per-env logging.

Importing this module registers the gym id ``Isaac-FlatSurfaceFollow-Direct-v0``.
"""

import gymnasium as gym

from .flat_surface_follow_env import FlatSurfaceFollowEnv
from .flat_surface_follow_env_cfg import FlatSurfaceFollowEnvCfg

gym.register(
    id="Isaac-FlatSurfaceFollow-Direct-v0",
    # MUST be the "module:Class" string form so env_setup.py can resolve the
    # concrete env class via gym.spec(...).entry_point.split(":") (camera/contact
    # patching).
    entry_point="tasks.flat_surface_follow.flat_surface_follow_env:FlatSurfaceFollowEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": FlatSurfaceFollowEnvCfg},
)
