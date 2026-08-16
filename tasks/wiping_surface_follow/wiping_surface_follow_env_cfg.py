"""Env config for the WIPING surface task. Swaps in the wiping task; obs/state layout, action space,
episode length, and events are inherited unchanged from the curved surface env cfg."""

from isaaclab.utils import configclass

from ..curved_surface_follow.curved_surface_follow_env_cfg import CurvedSurfaceFollowEnvCfg
from .wiping_surface_follow_tasks_cfg import WipingSurfaceFollowTask


@configclass
class WipingSurfaceFollowEnvCfg(CurvedSurfaceFollowEnvCfg):
    task: WipingSurfaceFollowTask = WipingSurfaceFollowTask()

    # MDP matched to arXiv:2502.12599: 20 Hz control (physics 120 Hz / decimation 6) and a 200-step
    # horizon (H = 200 = episode_length_s / step_dt = 10 s / 0.05 s). The base surface task runs 15 Hz
    # (decimation 8, 150 steps); overriding decimation here gives the paper's 20 Hz + H = 200.
    decimation: int = 6
    episode_length_s = 10.0
