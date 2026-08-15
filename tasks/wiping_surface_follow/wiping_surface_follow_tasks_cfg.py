"""Task config for the WIPING surface task (subclass of the CURVED surface task).

A wiping task built on the curved ridge: a Franka holds a rectangular SPONGE (a box) and must wipe
along a sequence of waypoints on the (per-env curved) surface while maintaining a target contact force.
It reuses the curved surface, the observation/action spaces, and the per-env curvature ladder (alpha)
UNCHANGED; only the held object and the reward change.

Reward mimics the BASE reward of "Learning a High-quality Robotic Wiping Policy ..." (arXiv:2502.12599,
Eq. 1), NOT their bounded reformulation:
    r = r_col                                if collision (over-force)     [terminating, optional]
      = r_con + r_force + r_way + r_ac       otherwise
  r_con   = w_con * I_contact
  r_force = w_force * exp(-(f_n - mu)^2 / (2 sigma^2)) * I_align   (I_align: EE motion aligned to the
            target waypoint, cos-sim > align_cos)
  r_way   = w_way * I_reached_a_waypoint      (+ final_bonus when the LAST waypoint is wiped)
  r_ac    = -w_ac * (|a_x| + |a_y| + |a_z|)   (EE acceleration -> smoothness)
  r_col   = -w_col                            (over-force "collision")

Held object: the sponge is gripped like the peg (on its WIDTH == the cylinder's ``diameter`` slot, the
gripper opening) and extends DOWN along held-local +z by ``height``; its wiping FACE (width x depth) is
the bottom face. So the inherited cyl_tip / cyl_axis geometry already yields the sponge's bottom-face
CENTRE and FACE NORMAL, and ``angle_from_normal`` = 0 means the sponge lies FLAT on the surface (desired
tool angle 0 deg). The name stays "flat_surface_follow" so env_setup gates match.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import HeldAssetCfg

from ..curved_surface_follow.curved_surface_follow_tasks_cfg import CurvedSurfaceFollowTask

# Sponge dimensions (m). Gripped along WIDTH (x, the gripper opening), extends DOWN along HEIGHT (z),
# wiping face = WIDTH x DEPTH. Extension is tall enough that the gripper clears the ridge peak.
_SPONGE_WIDTH = 0.02      # gripped dimension (held local x) == gripper opening slot ("diameter")
_SPONGE_DEPTH = 0.02      # face depth (held local y)
_SPONGE_HEIGHT = 0.06     # extension below the gripper (held local z); bottom face is the wiper


@configclass
class SpongeHeldCfg(HeldAssetCfg):
    """Held sponge (a box). ``diameter`` = the gripped WIDTH (drives the reset gripper opening, reusing
    the cylinder slot); ``height`` = the extension below the gripper (held local z); ``depth`` = the
    face depth (held local y). Bottom face (width x depth) is the wiper."""

    diameter = _SPONGE_WIDTH
    height = _SPONGE_HEIGHT
    depth = _SPONGE_DEPTH
    friction = 0.75
    mass = 0.05


@configclass
class WipingSurfaceFollowTask(CurvedSurfaceFollowTask):
    # NOTE: ``name`` stays "flat_surface_follow" (inherited) so the env_setup grasp/weld + keypoint-servo
    # gates match unchanged.

    held_asset_cfg: SpongeHeldCfg = SpongeHeldCfg()

    # Force source for measured_normal_force: use the WRIST F/T (not the plate-filtered "oracle"
    # contact sensor). The wiping surface is the separate ridge CAP prim, which the plate contact
    # sensor doesn't see, so the oracle would read ~0 on the ridge; the wrist reaction captures it and
    # needs no ContactSensorWrapper. (Override of the flat default "oracle".)
    force_source: str = "wrist_ft"

    # Sponge lies FLAT on the surface: spawn with no tilt and desire tool-axis angle 0 off the normal.
    spawn_orn_mean_deg: list = [0.0, 0.0, 0.0]
    orientation_desired_angle_deg: float = 0.0

    # Spawn the sponge tip ~1 mm above the surface; the inherited press-to-contact seats it. NOTE: this
    # peg-style point-tip reset only seats a SMALL face cleanly — a wide flat sponge face tips/bounces
    # during seating (2 cm face -> ~5 mm spawn clearance; 4 cm -> ~35 mm; 5 cm -> ~70 mm). A larger,
    # more sponge-like pad needs a flat-face-aware reset (lower the whole face + press evenly).
    spawn_tip_pos_mean: list = [0.0, 0.0, 0.001]

    # --- Wiping waypoints ---------------------------------------------------------------------
    # Number of waypoints along the wipe path (evenly spaced in path arc-length, lifted onto the
    # surface). The policy is shown ONE at a time (the current target); it advances SEQUENTIALLY when
    # the sponge reaches the current waypoint in contact. n=5 -> waypoints at 1/5..5/5 of the path.
    n_waypoints: int = 5
    # Distance (m) within which the sponge bottom-face centre counts as having REACHED the current
    # waypoint (must also be in contact). Advances the target and pays r_way.
    waypoint_reach_radius: float = 0.02

    # --- Wiping reward (paper Eq. 1 base terms) -----------------------------------------------
    wipe_contact_weight: float = 1.0      # w_con  : per-step reward while in contact
    wipe_force_weight: float = 1.0        # w_force: peak of the Gaussian force reward (in contact + aligned)
    wipe_force_target: float = 5.0        # mu (N) : target normal force (paper uses 60 N; ours is lighter)
    wipe_force_sigma: float = 5.0         # sigma (N): Gaussian width of the force reward
    wipe_align_cos: float = 0.8           # I_align: min cos-sim(EE motion dir, dir-to-waypoint) to pay force
    wipe_waypoint_weight: float = 10.0    # w_way  : sparse reward each time a waypoint is reached
    wipe_final_bonus: float = 50.0        # extra sparse reward when the FINAL waypoint is wiped (episode ends)
    wipe_accel_weight: float = 0.001      # w_ac   : EE-acceleration (|ax|+|ay|+|az|) smoothness penalty
    wipe_collision_weight: float = 10.0   # w_col  : penalty on an over-force "collision"
    wipe_collision_force: float = 30.0    # N: |normal force| above this counts as a collision
    # Terminate the episode on a collision (paper does). OFF by default: with per-env curvature this
    # task uses the FULL-reset path (no efficient reset), so frequent terminations = frequent settling
    # resets. Enable once the policy keeps force under control.
    terminate_on_collision: bool = False

    # --- Sponge box spawn (dynamic, gravity off like the peg) ---
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.CuboidCfg(
            size=(_SPONGE_WIDTH, _SPONGE_DEPTH, _SPONGE_HEIGHT),   # (x=width, y=depth, z=extension)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SpongeHeldCfg().mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=SpongeHeldCfg().friction, dynamic_friction=SpongeHeldCfg().friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.8, 0.35)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )
