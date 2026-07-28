"""Env config for the BUMPY surface path-following task.

Subclasses ``FlatSurfaceFollowEnvCfg`` and swaps in the bump task. Everything else — action space,
episode length, obs/state layout (the bumps are unobservable, so NO new obs/state channels), events —
is inherited unchanged.
"""

from isaaclab.utils import configclass

from ..flat_surface_follow.flat_surface_follow_env_cfg import FlatSurfaceFollowEnvCfg
from .bumpy_surface_follow_tasks_cfg import BumpySurfaceFollowTask


@configclass
class BumpySurfaceFollowEnvCfg(FlatSurfaceFollowEnvCfg):
    task: BumpySurfaceFollowTask = BumpySurfaceFollowTask()
