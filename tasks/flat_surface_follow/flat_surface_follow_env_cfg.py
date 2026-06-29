"""Env config for the surface path-following task.

Subclasses ``ForgeEnvCfg`` (force sensing + obs noise + control plumbing the repo
wrappers expect). Drops the Forge success-prediction action dim (``action_space``
6, the Factory width).

Observation convention (new EEF-frame control rework): the POLICY sees the goal
pose expressed RELATIVE TO THE GOAL FRAME and in the EEF frame, sign-aligned with
the action (a positive obs component => a positive action on that axis moves toward
the goal), plus EEF-frame velocities and the EEF-frame force (and, behind a config
toggle, the EEF-frame torque). The surface geometry (normal, path, progress, etc.)
is NOT observable — the policy infers contact from the force/torque. The CRITIC adds
the privileged geometry + full asset/robot state.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.direct.forge.forge_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ForgeEnvCfg

from .flat_surface_follow_tasks_cfg import FlatSurfaceFollowTask

# Register the new obs/state channels in the shared module-global dim dicts (same
# additive ``.update`` pattern Forge uses). Keys are unique to this task, so the
# Factory/Forge obs layouts are unaffected.
OBS_DIM_CFG.update(
    {
        "goal_pos_rel": 3,     # R_eefᵀ (goal_pos - eef_pos): goal position in the EEF frame
        "goal_rot_rel": 3,     # axis-angle(eef -> goal frame) in the EEF frame
        "ft_torque_eef": 3,    # EEF-frame torque (appended to obs_order only when observe_eef_torque)
        "target_speed": 1,
        "target_normal_force": 1,
    }
)
STATE_DIM_CFG.update(
    {
        "goal_pos_rel": 3,
        "goal_rot_rel": 3,
        "ft_torque_eef": 3,
        "target_speed": 1,
        "target_normal_force": 1,
        # Privileged surface geometry (critic only — the policy must infer it).
        "surface_normal": 3,
        "path_dir": 3,
        "progress": 1,
        "cross_track": 1,
        "orn_error": 1,
        "normal_force": 1,
    }
)


@configclass
class FlatSurfaceFollowEventCfg:
    """No domain-randomization events for the structural pass (dead-zone / gains /
    friction are set at reset). Kept as an empty configclass so the EventManager
    builds with zero terms."""

    pass


@configclass
class FlatSurfaceFollowEnvCfg(ForgeEnvCfg):
    action_space: int = 6  # Factory width; no Forge success-prediction dim.
    task: FlatSurfaceFollowTask = FlatSurfaceFollowTask()
    events: FlatSurfaceFollowEventCfg = FlatSurfaceFollowEventCfg()

    # Long enough to traverse plate_length (0.20 m) at target_speed (0.05 m/s) = 4 s,
    # with margin for approach + settling.
    episode_length_s = 15.0

    # POLICY obs (all EEF-frame): goal pose relative to the goal frame (sign-aligned),
    # EEF-frame velocities + force, command setpoints. The 3-D EEF torque is appended at
    # env construction when task.observe_eef_torque is set. No surface geometry — the
    # policy must infer contact from the force.
    obs_order: list = [
        "goal_pos_rel",
        "goal_rot_rel",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        "target_speed",
        "target_normal_force",
    ]
    # CRITIC state: same relative pose + EEF-frame vel/force, plus the full privileged
    # asset/robot state and the surface geometry.
    state_order: list = [
        "goal_pos_rel",
        "goal_rot_rel",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
        "fingertip_pos",
        "fingertip_quat",
        "joint_pos",
        "held_pos",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "pos_threshold",
        "rot_threshold",
        "surface_normal",
        "path_dir",
        "progress",
        "cross_track",
        "orn_error",
        "normal_force",
        "target_speed",
        "target_normal_force",
    ]
