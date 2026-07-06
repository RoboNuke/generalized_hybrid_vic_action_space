"""Task + asset configuration for the surface path-following task.

Assets are spawned as PROCEDURAL PRIMITIVES (a kinematic cuboid plate + a
dynamic cylinder) via ``isaaclab.sim`` shape spawners — no USD files needed.
The plate is the "fixed asset" and the cylinder is the "held asset", mirroring
the Factory/Forge ``fixed_asset`` / ``held_asset`` ArticulationCfg fields but as
``RigidObjectCfg`` (a single primitive is a rigid body, not an articulation).

``FlatSurfaceFollowTask`` keeps FLAT config fields (``{thing}_scale`` /
``{thing}_target`` / ``{thing}_range_deg``) so task params are easy to override
via ``runner_cfg.env_cfg_overrides`` dotted paths (e.g. ``task.desired_speed_cm_s``).
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
_CYL_LENGTH = 0.15        # cylinder long axis (held local +z)
_CYL_DIAMETER = 0.008


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
    desired_speed_cm_s: float = 5.0               # v: desired along-track speed (cm/s) for the pace term

    # --- Observation toggles ---
    observe_eef_torque: bool = False              # add the 3-D EEF-frame torque to the obs (policy + critic)

    # --- Success tolerances ---
    success_pos_tol: float = 0.01                 # cylinder tip within this of the far-edge center
    success_orn_tol_deg: float = 10.0             # |orientation error| (deg) within this of the desired

    # --- Termination (per-env). Both default OFF. When EITHER is on, env_setup auto-attaches the
    # efficient-reset wrapper so partial resets teleport (no sim steps) instead of running Factory's
    # all-envs settling reset. terminated (failure/success) is NOT value-bootstrapped; time-out is. ---
    terminate_on_lag: bool = False                # end if the tool falls too far behind the setpoint
    terminate_on_success: bool = False            # end the moment success is reached (in contact)
    pace_lag_frac: float = 0.5                    # max along-track lag before termination, as a
                                                  # FRACTION of path length L (0.5 => half the path)

    # --- Reward terms ---
    # Convention: each term computes a RAW SIGNED value from REALIZED (measured/physics) state —
    # never control targets — and maps it through the keypoint squashing function
    #     r = weight * squashing_fn(value, a, b),   squashing_fn(x,a,b) = 1/(e^{a*x}+b+e^{-a*x})
    # which peaks at value=0 (perfect match). Each term carries its own weight + (a, b). With b=-1 the
    # peak is 1/(2+b)=1, so `weight` IS each term's max per-step contribution (terms balance directly).
    #
    # Force tracking: value = desired_force - (measured EEF force projected onto the surface normal).
    # The desired force acts along the surface-normal (z); it is sampled once per episode in
    # [force_desired_min, force_desired_max] (N), observed by the policy, and the measured EEF force
    # is projected onto the world surface normal for a fair (same-frame) difference.
    force_desired_min: float = 10.0               # N (fixed target for training: min == max => no per-run variation)
    force_desired_max: float = 10.0               # N
    force_weight: float = 1.0
    force_a: float = 0.25                         # squashing steepness over the force error (N); wide
                                                  # so there's a MONOTONIC gradient from light contact
                                                  # up to the target force (gated on contact below)
    force_b: float = -1.0                         # peak = 1
    # Orientation constraint: value = desired - (angle between the held object's z-axis and the
    # surface NORMAL), computed in RADIANS. angle-from-normal = arccos(|held_z . normal|): 0 = axis
    # parallel to the normal (tip-down/perpendicular), pi/2 = axis in the plane (flat). Free to rotate
    # about the normal (only this angle is constrained). Realized from the physics held orientation.
    # This field is DEGREES (readable) but the env converts to radians, so orientation_a acts on RAD.
    orientation_desired_angle_deg: float = 10.0   # desired tool-axis angle OFF the surface normal (deg)
    orientation_weight: float = 1.0
    orientation_a: float = 5.0                    # squashing steepness over the angle error (RADIANS)
    orientation_b: float = -1.0                   # peak = 1
    # Near-surface gate for the orientation reward: it pays ONLY when the tool tip is within this
    # height (m) above the contact point (tip_surface_dist < gate; contact/penetration count). Keeps
    # "hold the commanded angle on final approach" but prevents farming it while hovering high.
    orientation_gate_dist: float = 0.01           # 1 cm

    # Contact bonus: per-step +1 while the held object is in contact (0 otherwise), scaled by weight.
    # DISABLED (0.0): the CONTACT pull now comes from the contact-gated force reward (wide a=0.25),
    # which pays only in contact and is worth up to force_weight there — no separate bonus.
    contact_weight: float = 0.0
    # Straightness + pace (both built on the ideal straight path p0->p_g on the plate top surface):
    #   d   = unit start->goal direction; L = |p_g - p0| (path length, m); dp = tip - p0
    #   s     = dp . d            (along-track position, m)
    #   e_perp= dp . (n x d)      (cross-track, signed, m)            -> STRAIGHTNESS value
    #   s_ref = clamp(v . tau, 0, L), tau advances by dt only while in contact (a schedule clock)
    #   e_pace= s - s_ref         (pace error, signed, m)             -> PACE value
    # Tracking s_ref makes the tool reach the surface then trace the line at the desired speed.
    straightness_weight: float = 1.0
    straightness_a: float = 100.0                 # squashing steepness over the cross-track error (m)
    straightness_b: float = -1.0                  # peak = 1
    pace_weight: float = 1.0
    pace_a: float = 50.0                          # squashing steepness over the pace error (m)
    pace_b: float = -1.0                          # peak = 1

    # Time-to-success bonus: ONE-SHOT, paid on the first step success is reached. value = t* - t_succ
    # where t* = (first-contact time) + path_length / desired_speed is the ideal completion time and
    # t_succ is the actual time success is reached; equivalently L/v - (contact->success duration).
    # squashing_fn peaks (value=0) when success lands exactly at the ideal time, penalizing dawdling
    # AND cutting the path short. NOTE: this fires once per episode, so to make it count against the
    # per-step terms (which accumulate over ~hundreds of steps) scale success_time_weight up (10s-100s).
    success_time_weight: float = 600.0             # = max dense reward over a 150-step (10 s) episode
                                                   # (4 dense terms x peak 1 x 150), so an on-time success is
                                                   # always >= the full dense sum: it is always best to succeed.
    success_time_a: float = 1.0                    # squashing steepness over the time error (SECONDS)
    success_time_b: float = -1.0                   # peak = 1

    # Action penalties (linear, from FactoryEnv; NOT squashed). NOTE: ForgeTask (our parent) sets
    # action_grad_penalty_scale=0.1 (peg-tuned), which unfairly penalized the higher action-dim
    # control methods on the surface — so we EXPLICITLY zero BOTH here: no action punishment.
    # _get_rewards applies them with negative scales; raise to re-enable.
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
