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

    # Fixed grasp tilt [roll, pitch, yaw] (deg) of the cylinder RELATIVE TO THE GRIPPER, folded
    # into the peg-gripper weld. Set by env_setup.build_env from runner_cfg.rel_grasp_rot_init_deg
    # (grasp_rot_mode='fixed'); NOT a user override. The reset reads it to invert the tilt when
    # seating the cylinder, so a tilted grasp still spawns the cylinder at the desired orientation
    # (the eef/wrist absorbs the tilt). All-zero = aligned grip.
    grasp_weld_tilt_deg: list = [0.0, 0.0, 0.0]

    # --- Plate geometry + spawn randomization ---
    plate_length: float = _PLATE_LENGTH
    plate_width: float = _PLATE_WIDTH
    plate_thickness: float = _PLATE_THICKNESS
    plate_center_pos: list = list(_PLATE_CENTER)
    plate_pos_noise: list = [0.02, 0.02, 0.02]   # world-frame per-axis +/- spawn jitter
    plate_yaw_range_deg: float = 360.0           # full in-plane orientation randomization
    plate_tilt_range_deg: float = 10.0           # small roll/pitch cone off world +z (see note)

    # --- Held-cylinder spawn pose (NO force controller: the TIP is placed directly, then the arm
    #     is IK'd to match). The reset geometry is used only to POSITION the spawn — no privileged
    #     info reaches the policy. The cylinder TIP is placed at a configurable offset from the
    #     STARTING KEYPOINT k0 = the near-edge center (start), which is also the keypoint the policy
    #     is shown at step 0 (the target is held at k0 until first contact, so the peg descends
    #     straight onto k0, then the setpoint advances one keypoint at a time). An orientation offset
    #     is applied ABOUT THE TIP, so it never moves the tip.
    #
    #     Each of the 6 DOF has a MEAN and a STD. std == 0 => the mean is used EXACTLY (no sampling);
    #     std > 0 => the value is sampled from N(mean, std) per reset. ---
    # Tip position offset from the starting keypoint, in the SURFACE-LOCAL frame (x = along the path
    # near->far, y = across the path, z = along the surface normal, +z = above the plate). Meters.
    spawn_tip_pos_mean: list = [0.0, 0.0, 0.001]   # default: tip 1 mm above the starting keypoint
    spawn_tip_pos_std: list = [0.0, 0.0, 0.0]      # per-axis Gaussian std (m); 0 => fixed at the mean
    # Orientation offset of the cylinder about its tip, as WORLD-FRAME roll/pitch/yaw (DEGREES):
    # roll about world x, pitch about world y, yaw about world z. The zero-offset pose is the peg
    # straight up/down (axis along world +z, tip down) — INDEPENDENT of the surface orientation
    # (which the policy does not know a priori). roll/pitch tilt the axis off world-vertical; yaw
    # spins the (axisymmetric) cylinder about vertical.
    spawn_orn_mean_deg: list = [0.0, 10.0, 0.0]    # default: 10 deg pitch off world-vertical
    spawn_orn_std_deg: list = [0.0, 0.0, 0.0]      # per-axis Gaussian std (deg); 0 => fixed at the mean

    # --- Reset press-to-contact: after the peg is spawned (above the surface) and gripped, PRESS it
    #     toward the surface until each env reads in contact, then latch it in place, so episodes
    #     start already reading in contact instead of hovering. The gripper stays closed and a
    #     CUMULATIVE fingertip position target steps down along -surface-normal each settle step, so
    #     the tracking error (hence press force) BUILDS until contact is read — a constant offset can
    #     only ever build Kp*step and stalls a fraction of a mm above the plate. The target is clamped
    #     to lead the fingertip by at most reset_press_max_lead, capping the steady press force at
    #     ~Kp*max_lead (no spike at touchdown, no blow-up if the surface is absent). Each env is
    #     frozen the step it first reads in contact. "In contact" is the SAME per-axis contact sensor
    #     the policy sees at runtime, refreshed during reset via the ContactSensorWrapper hook (it is
    #     otherwise only updated in the wrapper's step()); with no live sensor it falls back to the
    #     measured normal force / a geometric tip-at-surface check. reset_press_max_dist caps the loop.
    #     This runs in the full-reset (all-envs) path, so the efficient-reset wrapper's cached donor
    #     state records the in-contact pose and per-env teleport resets reproduce it. ---
    reset_press_to_contact: bool = True            # master toggle (False => spawn hovering, as before)
    reset_press_step: float = 0.0005               # m: cumulative target descent per settle step
    reset_press_max_lead: float = 0.003            # m: max the target may lead the fingertip; the LATCHED
                                                   # lead => the sustained press force ~Kp*max_lead. Sized so
                                                   # that force stays a light ~1-2 N (comfortably above the
                                                   # contact-sensor threshold so it keeps reading, well below
                                                   # a hard press). Raise for a firmer start, lower for lighter.
    reset_press_max_dist: float = 0.02             # m: cap on total target travel (bounds the loop iterations)
    reset_press_contact_depth: float = 0.0         # m: no-sensor geometric backstop — stop once tip_surface_dist <= this
    reset_contact_force_threshold: float = 0.1     # N: no-sensor |measured normal force| contact fallback

    # --- Task command setpoints ---
    desired_speed_cm_s: float = 5.0               # v: desired along-track speed (cm/s) for the pace term

    # How the OBSERVATION setpoint (env.setpoint_pos — the keypoint target the obs reports and the
    # keypoint-servo wrapper tracks) advances along the ideal path:
    #  * False (default, ROBOT-DRIVEN): the target sits one keypoint ahead of the arm's REALIZED
    #    progress and waits for the arm to catch up before advancing.
    #  * True  (PACE-DRIVEN): the target advances on the time-based pace clock (v*pace_tau, same as the
    #    pace-reward setpoint s_ref) and never waits for the arm — it moves forward at desired_speed_cm_s
    #    regardless of whether the arm keeps up. Both modes hold at keypoint 0 until first contact.
    setpoint_pace_driven: bool = False

    # --- Observation toggles ---
    observe_eef_torque: bool = False              # add the 3-D EEF-frame torque to the obs (policy + critic)

    # --- Success tolerances ---
    success_pos_tol: float = 0.01                 # cylinder tip within this of the far-edge center
    success_orn_tol_deg: float = 10.0             # |orientation error| (deg) within this of the desired
    # Success = reached the goal pose (above tolerances) AND actually dragged enough of the path:
    # keypoints_achieved / keypoints_total >= success_keypoint_frac. This makes every keypoint MEAN
    # something (a fly-to-goal that skipped the surface can't succeed). Set to 0.0 to recover the old
    # pose-only success; set to 1.0 to require EVERY keypoint be achieved.
    success_keypoint_frac: float = 0.9

    # --- Keypoints (checkpoints): reward the ACTUAL drag, not a fly-to-goal shortcut ---
    # The checkpoints are evenly spaced arc-length points on the ideal path — spacing v*dt, so their
    # COUNT is L/(v*dt), NOT a free parameter (see keypoints_total in the env). A keypoint is ACHIEVED
    # the first time the progress frontier passes it during a step that is IN CONTACT and laterally
    # ON-TRACK (|cross_track| < keypoint_track_tol); the count is not capped to one boundary per step,
    # so legitimate pace variation is credited, but boundaries crossed out of contact / off-track are
    # forfeited (the frontier still advances, so a fly-ahead shortcut permanently loses them). The
    # success reward is weighted by achieved/total (partial credit) and success itself requires
    # achieved/total >= success_keypoint_frac.
    # Max lateral (cross-track) error for a keypoint crossing to count as achieved. Keeps "achieved"
    # meaningful — the tool must be near the path, not dragging far off to the side.
    keypoint_track_tol: float = 0.003

    # Fixed reward paid ONCE per keypoint, the first time it is cleanly ACHIEVED (crossed in contact,
    # one at a time, advancing the achieved frontier). A dense forward-progress signal that — unlike
    # the goal-gated success bonus — pays during the drag. Value = reward per keypoint (0 disables).
    keypoint_reward_weight: float = 1.0

    # --- Interaction (stiffness) frame: consumed by the rotated controllers' fixed_rotation_from_
    #     interaction variant and the viz marker (interaction_frame_world). ---
    # In BOTH modes x points from the contact point to the CURRENT goal keypoint (setpoint_pos),
    # projected ⊥ the mode's z-axis; y = z × x (cross-track).
    # "geometric": z = surface normal — pure surface frame; x = goal-keypoint dir projected ⊥ normal.
    # "dynamic":   z = direction of the measured contact REACTION (force_sensor_world_smooth, clean/
    #              EMA-smoothed, peg-gravity off), so the frame tilts off the normal by the friction
    #              angle; x = goal-keypoint dir with its component parallel to z (reaction) subtracted.
    # OFF-CONTACT (env.in_contact_any False — the single contact source of truth) BOTH modes collapse
    # to the control/EEF frame (identity stiffness rotation): no surface, so stiffness is control-frame.
    interaction_frame_mode: str = "geometric"

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
    force_desired_min: float = 5.0                # N (fixed target for training: min == max => no per-run variation)
    force_desired_max: float = 5.0                # N
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
    orientation_a: float = 2.0                    # squashing steepness over the angle error (RADIANS)
    orientation_b: float = -1.0                   # peak = 1
    # Near-surface gate for the orientation reward: it pays ONLY when the tool tip is within this
    # height (m) above the contact point (tip_surface_dist < gate; contact/penetration count). Keeps
    # "hold the commanded angle on final approach" but prevents farming it while hovering high.
    orientation_gate_dist: float = 0.1           # 10 cm - user changed to mostly disable this

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
    # OFF-CONTACT weight for the tracking terms: straightness/pace pay their full *_weight IN CONTACT
    # but only *_air_weight in the air. Downweighting the air contribution keeps a faint gradient for
    # lining up over the start, while making contact strictly more rewarding (so the policy can't just
    # hover-and-track). At 0.1 the whole air-track is at best a tenth of what contact unlocks.
    straightness_air_weight: float = 0.1
    pace_air_weight: float = 0.1

    # Alternative VELOCITY-based pace (task.vel_based_pace_enabled, default ON): SWAPS IN for the
    # position pace above. Instead of the along-track POSITION error (progress - s_ref, whose
    # time-based s_ref runs away once the tool lags — the gradient dies and it never catches up), it
    # rewards matching the DESIRED along-track SPEED: value = d(progress)/dt - v_des (m/s), squashed.
    # This stays a LIVE gradient at ANY lag, so it continuously pulls the tool forward at ~v_des and
    # breaks the "sit still in contact" local minimum. Reuses the "pace" reward-dict key, the
    # contact/air gating, and the per-term logging (so only one pace term is active at a time).
    vel_based_pace_enabled: bool = True
    vel_based_pace_weight: float = 2.0            # match the ~125 the constraint terms reach (pace was flat ~50 at 1.0)
    vel_based_pace_a: float = 30.0                # squashing steepness over the along-track SPEED error (m/s)
    vel_based_pace_b: float = -1.0                # peak = 1
    # Optional MIN-SPEED gate: when enabled, the velocity pace pays ZERO whenever the along-track
    # speed d(progress)/dt is below vel_based_pace_min_speed (m/s). A hard cliff — tiny jitter earns
    # nothing and the reward JUMPS on once the tool commits to real forward motion, so holding still
    # in contact gets no pace at all. Default OFF (smooth squash only).
    vel_based_pace_min_speed_enabled: bool = False
    vel_based_pace_min_speed: float = 0.001       # m/s (1 mm/s) — gate threshold when enabled

    # Time-to-success bonus: ONE-SHOT, paid on the first step success is reached. value = t* - t_succ
    # where t* = (first-contact time) + path_length / desired_speed is the ideal completion time and
    # t_succ is the actual time success is reached; equivalently L/v - (contact->success duration).
    # squashing_fn peaks (value=0) when success lands exactly at the ideal time, penalizing dawdling
    # AND cutting the path short. NOTE: this fires once per episode, so to make it count against the
    # per-step terms (which accumulate over ~hundreds of steps) scale success_time_weight up (10s-100s).
    success_time_weight: float = 600.0             # = max dense reward over a 150-step (10 s) episode
                                                   # (4 dense terms x peak 1 x 150), so an on-time success is
                                                   # always >= the full dense sum: it is always best to succeed.
    success_time_a: float = 0.25                    # squashing steepness over the time error (SECONDS)
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
