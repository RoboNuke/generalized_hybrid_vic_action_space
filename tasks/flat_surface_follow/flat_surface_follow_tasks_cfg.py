"""Task + asset configuration for the surface path-following task.

Assets are spawned as PROCEDURAL PRIMITIVES (a kinematic cuboid plate + a
dynamic cylinder) via ``isaaclab.sim`` shape spawners — no USD files needed.
The plate is the "fixed asset" and the cylinder is the "held asset", mirroring
the Factory/Forge ``fixed_asset`` / ``held_asset`` ArticulationCfg fields but as
``RigidObjectCfg`` (a single primitive is a rigid body, not an articulation).

``FlatSurfaceFollowTask`` keeps FLAT config fields (``{thing}_scale`` /
``{thing}_target`` / ``{thing}_range_deg``) so task params are easy to override
via ``runner_cfg.env_cfg_overrides`` dotted paths (e.g. ``task.target_speed``).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgeTask

# Default geometry (meters). Defined as module constants so the same numbers feed
# both the FLAT task fields (read by the env logic) and the primitive spawn cfgs
# below (evaluated in the class body).
_PLATE_LENGTH = 0.20      # extent along the near->far traverse direction (plate local +x)
_PLATE_WIDTH = 0.20       # extent across the path (plate local +y)
_PLATE_THICKNESS = 0.02   # plate local +z
_PLATE_CENTER = (0.55, 0.0, 0.10)
_CYL_LENGTH = 0.10        # cylinder long axis (held local +z)
_CYL_DIAMETER = 0.012


@configclass
class PlateCfg(FixedAssetCfg):
    """Flat plate. ``height`` carries the plate thickness; friction/mass used by
    ``factory_utils.set_friction`` and the spawn below."""

    height = _PLATE_THICKNESS
    friction = 0.75
    mass = 1.0  # irrelevant (kinematic), but a sane value


@configclass
class CylinderHeldCfg(HeldAssetCfg):
    """Held cylinder. ``diameter`` sets the reset gripper width; ``height`` is the
    cylinder length used by the in-hand grasp offset and tip computation."""

    diameter = _CYL_DIAMETER
    height = _CYL_LENGTH
    friction = 0.75
    mass = 0.05


@configclass
class FlatSurfaceFollowTask(ForgeTask):
    name: str = "flat_surface_follow"

    fixed_asset_cfg: PlateCfg = PlateCfg()
    held_asset_cfg: CylinderHeldCfg = CylinderHeldCfg()

    # --- Plate geometry + spawn randomization ---
    plate_length: float = _PLATE_LENGTH
    plate_width: float = _PLATE_WIDTH
    plate_thickness: float = _PLATE_THICKNESS
    plate_center_pos: list = list(_PLATE_CENTER)
    plate_pos_noise: list = [0.02, 0.02, 0.02]   # world-frame per-axis +/- spawn jitter
    plate_yaw_range_deg: float = 360.0           # full in-plane orientation randomization
    plate_tilt_range_deg: float = 10.0           # small roll/pitch cone off world +z (see note)

    # --- Held cylinder in-hand randomization ---
    # Per-reset random tilt of the cylinder long axis in the gripper (roll/pitch/yaw,
    # degrees). All-zero => fixed tip-down grasp. Non-zero => "random angle in hand".
    inhand_tilt_range_deg: list = [0.0, 0.0, 0.0]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]  # in-hand position jitter (keep centered)

    # --- Start placement (NO force controller; spawn just above the surface) ---
    start_standoff: float = 0.001                 # >= 1 mm above surface along the normal
    # Hand-init randomization expressed in the SURFACE-LOCAL frame:
    #   x = along the path (near->far), y = across the path, z = along the surface normal.
    # Default: +/-1 cm in-plane, 0 in z (so the cylinder stays exactly start_standoff above).
    start_pos_noise: list = [0.01, 0.01, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 3.1416]  # yaw is free (cylinder is axisymmetric)

    # --- Task command setpoints ---
    target_speed: float = 0.05                    # m/s along the path
    target_normal_force: float = 5.0              # N into the surface
    commanded_axis_angle_deg: float = 0.0         # angle between surface normal and cylinder axis

    # --- Observation toggles ---
    observe_eef_torque: bool = False              # add the 3-D EEF-frame torque to the obs (policy + critic)

    # --- Success tolerances ---
    success_pos_tol: float = 0.01                 # cylinder tip within this of the far-edge center
    success_orn_tol_deg: float = 10.0             # orientation error within this of the command

    # --- Reward weights (FLAT; reward TERMS are a later pass — these wire the scorer) ---
    progress_scale: float = 1.0
    goal_kp_scale: float = 1.0
    cross_track_scale: float = 1.0
    speed_scale: float = 1.0
    normal_force_scale: float = 1.0
    orientation_scale: float = 1.0
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0

    # --- Procedural primitive spawns ---
    # Plate: kinematic so it never drifts under the cylinder's normal force.
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.CuboidCfg(
            size=(_PLATE_LENGTH, _PLATE_WIDTH, _PLATE_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PlateCfg().mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PlateCfg().friction, dynamic_friction=PlateCfg().friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.4, 0.6)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_PLATE_CENTER, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # Cylinder: dynamic, grasped in the gripper (gravity disabled like the Forge peg).
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.CylinderCfg(
            radius=_CYL_DIAMETER / 2.0,
            height=_CYL_LENGTH,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CylinderHeldCfg().mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=CylinderHeldCfg().friction, dynamic_friction=CylinderHeldCfg().friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.2)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )
