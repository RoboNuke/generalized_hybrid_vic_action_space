"""Env config for the CURVED surface path-following task.

Subclasses ``FlatSurfaceFollowEnvCfg`` and swaps in the curved task. Everything else — action space,
episode length, obs/state layout (the local surface geometry is already reported to the critic via the
inherited ``surface_normal`` / ``path_dir`` channels, now computed against the ridge), events — is
inherited unchanged.
"""

from isaaclab.utils import configclass

from ..flat_surface_follow.flat_surface_follow_env_cfg import FlatSurfaceFollowEnvCfg
from .curved_surface_follow_tasks_cfg import CurvedSurfaceFollowTask


@configclass
class CurvedSurfaceFollowEnvCfg(FlatSurfaceFollowEnvCfg):
    task: CurvedSurfaceFollowTask = CurvedSurfaceFollowTask()
