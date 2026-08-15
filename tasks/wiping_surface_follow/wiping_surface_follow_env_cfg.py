"""Env config for the WIPING surface task. Swaps in the wiping task; obs/state layout, action space,
episode length, and events are inherited unchanged from the curved surface env cfg."""

from isaaclab.utils import configclass

from ..curved_surface_follow.curved_surface_follow_env_cfg import CurvedSurfaceFollowEnvCfg
from .wiping_surface_follow_tasks_cfg import WipingSurfaceFollowTask


@configclass
class WipingSurfaceFollowEnvCfg(CurvedSurfaceFollowEnvCfg):
    task: WipingSurfaceFollowTask = WipingSurfaceFollowTask()
