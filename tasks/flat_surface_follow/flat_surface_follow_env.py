"""Surface path-following environment (subclass of ``ForgeEnv``).

Reuses Forge/Factory machinery (robot control targets, force sensing, IK reset
placement, per-env logging) and overrides only what the new geometry needs:
scene assets (plate + cylinder), reset placement (the cylinder TIP is placed at a
configurable pose relative to the starting keypoint on a randomly-oriented plate,
then the arm is IK'd to match), observations, success, and a (currently stubbed)
reward. The control/logging wrappers attach unchanged.

NOTE: reward terms are intentionally NOT implemented yet — this pass establishes
the task STRUCTURE. ``_get_rewards`` returns zeros and ``_log_factory_metrics``
(inherited) latches successes so the scorer's success-rate logging still works.
"""

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_matrix

from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge import forge_utils
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from .flat_surface_follow_env_cfg import FlatSurfaceFollowEnvCfg


class FlatSurfaceFollowEnv(ForgeEnv):
    cfg: FlatSurfaceFollowEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Append the EEF-torque obs channels BEFORE super().__init__ (which sizes the
        # obs/state spaces from obs_order/state_order). Done here, not in the cfg, so a
        # runner env_cfg_override of task.observe_eef_torque (applied before gym.make) is
        # honored.
        if getattr(cfg.task, "observe_eef_torque", False):
            for order in (cfg.obs_order, cfg.state_order):
                if "ft_torque_eef" not in order:
                    order.append("ft_torque_eef")
        # Configurable PhysX position-solver iterations. Set BEFORE super().__init__() builds the
        # sim: the scene cap clamps every body, and the robot articulation's own count governs the
        # glued cylinder (a link on it) — the plate is kinematic, so these two set the realized
        # tip-plate contact iterations. (Honors a task.solver_position_iteration_count override,
        # applied before gym.make.)
        _spic = int(getattr(cfg.task, "solver_position_iteration_count", 192))
        cfg.sim.physx.max_position_iteration_count = _spic
        _robot_spawn = getattr(cfg.robot, "spawn", None)
        for _props in ("articulation_props", "rigid_props"):
            _p = getattr(_robot_spawn, _props, None)
            if _p is not None and hasattr(_p, "solver_position_iteration_count"):
                _p.solver_position_iteration_count = _spic
        super().__init__(cfg, render_mode, **kwargs)
        # Interaction-frame axes (recomputed every _compute_intermediate_values). Initialized to a
        # world-aligned frame (x=+x path, y=+y lateral, z=+z normal) so interaction_frame_world() is
        # valid before the first _compute (the controller's fixed-rotation variant may read it early).
        self.path_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3).clone()
        self.d_lat = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3).clone()
        self.surface_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3).clone()
        # Contact point under the tip (recomputed each _compute); init so interaction_frame_world()'s
        # goal-keypoint x-axis (setpoint_pos - contact_point) is safe before the first _compute.
        self.contact_point = torch.zeros((self.num_envs, 3), device=self.device)
        # Contact flag (set each _compute); init here so interaction_frame_world() is safe if the
        # controller queries the stiffness frame before the first _compute.
        self.in_contact_any = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Per-episode desired normal force (N), sampled in _reset_idx, observed by the policy.
        self.desired_force = torch.zeros((self.num_envs,), device=self.device)
        # Pace schedule clock (s): TIME since first contact — advances by the env step dt every step
        # once contact has been made (NOT gated on staying in contact), driving the time-based pace
        # setpoint s_ref = v*pace_tau used ONLY by the pace reward. Reset to 0 each episode.
        self.pace_tau = torch.zeros((self.num_envs,), device=self.device)
        # Moving setpoint (recomputed each _compute); init for safety before the first reset.
        self.s_ref = torch.zeros((self.num_envs,), device=self.device)
        self.setpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # The keypoint after the current setpoint (fallback goal for interaction_frame_world() when the
        # tip sits on the current keypoint); recomputed each _compute, init for safety before reset.
        self.next_setpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Time-to-success bonus state. t_contact: episode time (s) of FIRST contact (+inf until
        # contact); success_reward_given: whether the one-shot bonus has already been paid this
        # episode. Both reset each episode in _reset_idx.
        self.t_contact = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.success_reward_given = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Latched at the FIRST success step (0 until then): the step count reached and the scaled
        # success_time reward actually earned. Published (only for succeeding rollouts) as
        # Episode / Steps to success and Episode_Reward/success_time_on_success.
        self.success_step = torch.zeros((self.num_envs,), device=self.device)
        self.success_time_earned = torch.zeros((self.num_envs,), device=self.device)
        # This step's lag-termination flag (set in _get_dones), published as the lag-termination rate.
        self._term_lag = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Contact-quality accumulators (per episode; reset in _reset_idx). cq_contact: steps in
        # contact; cq_starts: no-contact->contact transitions (= number of contact runs); cq_breaks:
        # contact->no-contact transitions (bounces); cq_prev: previous step's contact (for transitions).
        # Published per episode as contact_quality/{contact_percentage, avg_contact_length, contact_breaks}.
        self.cq_contact = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_starts = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_breaks = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_prev = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Keypoint accounting (all progress-projection based; see _get_rewards):
        #  * keypoints_achieved: # keypoints whose arc-length the progress frontier passed during a
        #    GATED step (in contact AND on-track, |cross_track| < keypoint_track_tol). Multi-boundary
        #    steps count all of them; air/off-track crossings are forfeited. Sum -> the success reward
        #    is weighted by achieved/total, and success requires achieved/total >= success_keypoint_frac.
        #  * keypoints_passed: running-max keypoint index the projected progress has reached (crossed),
        #    contact or not, clean or not — a pure "how far along d did we get" frontier. passed >>
        #    achieved reveals progress-without-clean-contact vs. genuinely stalling.
        #  * setpoint_kp_idx: the current TARGET keypoint; drives the OBSERVATION setpoint and the
        #    keypoint-ball colouring. Starts at 0 (k0 = the near-edge spawn point) and is HELD there
        #    until first contact, so the peg descends straight down onto k0; after contact it advances
        #    to the next keypoint ahead of progress, one at a time (see _compute_intermediate_values).
        # prev_progress carries last step's progress for the per-step crossing test. Reset each episode.
        self.keypoints_achieved = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.keypoints_passed = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # Furthest keypoint index achieved (gated); gates the once-per-keypoint reward (see _get_rewards).
        self.kp_ach_frontier = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.setpoint_kp_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.prev_progress = torch.zeros((self.num_envs,), device=self.device)
        self.prev_cross_track = torch.zeros((self.num_envs,), device=self.device)
        # Derived each _compute from L/(v*dt); placeholder until then.
        self.keypoint_spacing = 1e-6
        self.keypoints_total = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        # Drag-performance accumulators over IN-CONTACT steps (running count/sum/sumsq -> per-rollout
        # mean+std for force / along-speed / perp-speed / theta(deg)). Published as drag_performance/*.
        self.drag_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._drag_metrics = ("force", "speed_d", "speed_perp", "theta")
        for _n in self._drag_metrics:
            setattr(self, f"drag_sum_{_n}", torch.zeros((self.num_envs,), device=self.device))
            setattr(self, f"drag_sumsq_{_n}", torch.zeros((self.num_envs,), device=self.device))
        # Per-episode tensors the efficient-reset wrapper must carry across per-env teleport resets.
        # Their fresh-episode values are captured in the post-full-reset cache (pace_tau=0,
        # t_contact=+inf, success_reward_given=False, desired_force=sampled), so a donor copy on a
        # partial reset restores correct fresh values without re-running randomize_initial_state.
        self._efficient_reset_extra_attrs = (
            "desired_force",
            "pace_tau",
            "t_contact",
            "success_reward_given",
            "success_step",
            "success_time_earned",
            "cq_contact",
            "cq_starts",
            "cq_breaks",
            "cq_prev",
            "keypoints_achieved",
            "keypoints_passed",
            "kp_ach_frontier",
            "setpoint_kp_idx",
            "prev_progress",
            "prev_cross_track",
            "drag_count",
        ) + tuple(f"drag_sum_{_n}" for _n in self._drag_metrics) \
          + tuple(f"drag_sumsq_{_n}" for _n in self._drag_metrics)

    # ------------------------------------------------------------------
    # Small geometry helpers
    # ------------------------------------------------------------------
    def _identity_quat(self, n=None):
        n = self.num_envs if n is None else n
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(n, 1)

    def _sample_spawn_dof(self, mean_list, std_list, n):
        """Sample ``n`` rows of a 3-DOF spawn quantity (position or orientation offset).

        Per component: ``std == 0`` => the mean is used EXACTLY (no sampling); ``std > 0`` =>
        the value is drawn from ``N(mean, std)``. Returns ``(n, 3)``.
        """
        mean = torch.tensor(mean_list, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        std = torch.tensor(std_list, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        sampled = torch.normal(mean, std)                 # std == 0 columns already return the mean
        return torch.where(std > 0.0, sampled, mean)      # explicit: no sampling where std == 0

    def _rotate_vec(self, quat, vec):
        """Rotate ``vec`` (E,3) or (3,) by ``quat`` (E,4): returns R(quat) @ vec.

        Implemented with ``tf_combine`` (zero translation) to avoid depending on a
        specific quat-apply export.
        """
        n = quat.shape[0]
        if vec.dim() == 1:
            vec = vec.unsqueeze(0).repeat(n, 1)
        zeros = torch.zeros((n, 3), device=self.device)
        _, out = torch_utils.tf_combine(quat, zeros, self._identity_quat(n), vec)
        return out

    def _surface_frame(self):
        """Compute the plate frame + path endpoints from the current fixed-asset pose.

        All quantities are env-relative (consistent with ``fingertip_midpoint_pos``
        and ``held_pos``, which subtract ``env_origins``). Returns:
            start (E,3)   near-edge center on the plate top surface
            goal  (E,3)   far-edge center on the plate top surface
            normal (E,3)  unit surface normal (plate local +z)
            path_dir (E,3) unit near->far direction (plate local +x)
            cross_dir (E,3) unit across-path direction (plate local +y)
        """
        half_l = 0.5 * self.cfg_task.plate_length
        half_t = 0.5 * self.cfg_task.plate_thickness
        ident = self._identity_quat()

        start_local = torch.zeros((self.num_envs, 3), device=self.device)
        start_local[:, 0] = -half_l
        start_local[:, 2] = half_t
        goal_local = torch.zeros((self.num_envs, 3), device=self.device)
        goal_local[:, 0] = half_l
        goal_local[:, 2] = half_t

        _, start = torch_utils.tf_combine(self.fixed_quat, self.fixed_pos, ident, start_local)
        _, goal = torch_utils.tf_combine(self.fixed_quat, self.fixed_pos, ident, goal_local)

        normal = self._rotate_vec(self.fixed_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        path_dir = goal - start
        path_dir = path_dir / torch.linalg.norm(path_dir, dim=-1, keepdim=True).clamp_min(1e-8)
        cross_dir = torch.cross(normal, path_dir, dim=-1)
        return start, goal, normal, path_dir, cross_dir

    def _surface_normal_and_dir_at(self, query_pos):
        """Surface normal + travel direction (path tangent) AT a surface query point.

        GENERAL hook for arbitrary (incl. NON-FLAT) surfaces. A curved-surface subclass overrides
        this to return the LOCAL surface normal (from the surface gradient / SDF / mesh) and the
        LOCAL travel tangent at ``query_pos`` — so the reward and the critic always see the geometry
        at the point of contact, which varies across a curved surface. For the flat plate the normal
        (plate +z) and the near->far travel direction are constant over the surface, so ``query_pos``
        is unused and the cached plate-frame values are returned.

        Args:  query_pos (E,3): the surface point to evaluate at (the contact point / nearest
            surface point to the tool tip). Returns (normal (E,3), path_dir (E,3)), unit vectors.
        """
        return self._plate_normal, self._plate_path_dir

    def interaction_frame_world(self):
        """(E,3,3) world<-interaction rotation — columns are the interaction-frame axes in world.
        Consumed by the controller's ``fixed_rotation_from_interaction`` stiffness variant (R =
        R_eefᵀ @ this = the true interaction->EEF rotation), the impedance metrics, and the viz marker.

        In BOTH modes the x-axis points from the contact point toward the CURRENT goal keypoint
        (``self.setpoint_pos``), projected clear of that mode's z-axis so x ⊥ z. (Previously x was the
        constant along-track ``path_dir``; the goal-keypoint direction also corrects lateral drift, and
        it is what the supervised-rotation loss target now supervises.) When the tip sits within 0.1 mm
        of the current keypoint — where that direction is ill-defined and x would be random — it falls
        back to the NEXT keypoint (``self.next_setpoint_pos``) so x stays meaningful and stable.

        ``interaction_frame_mode`` (task cfg):
          * "geometric" (default): z = surface normal — the pure surface frame; x = the goal-keypoint
            direction projected ⊥ normal (in-plane), y = z × x (cross-track). Recomputed each step at the
            contact point, so it's the LOCAL frame for curved surfaces too.
          * "dynamic": z = direction of the measured contact reaction (force_sensor_world_smooth, clean
            & EMA-smoothed; peg gravity is disabled so it's contact-only), which tilts off the normal
            by the friction angle. x = the goal-keypoint direction with its component parallel to the
            reaction (z) subtracted, y = z × x (cross-track).

        OFF-CONTACT, BOTH modes return R_eef (world<-eef), so the controller's R = R_eefᵀ·this =
        IDENTITY — with no surface to interact with, the stiffness is applied in the control (EEF)
        frame, not a surface/reaction frame. Contact is the env's single source of truth,
        ``self.in_contact_any`` (contact sensor / normal-force fallback)."""
        # x points from the contact point to the current goal keypoint. When the tip sits essentially
        # ON that keypoint (<0.1 mm), the direction is ill-defined (x goes random), so fall back to the
        # NEXT keypoint to keep x meaningful and stable.
        at_goal = (torch.linalg.norm(self.setpoint_pos - self.contact_point, dim=-1) < 1e-4)  # (E,) 0.1 mm
        goal_pos = torch.where(at_goal[:, None], self.next_setpoint_pos, self.setpoint_pos)  # (E,3)
        to_goal = goal_pos - self.contact_point                                      # (E,3) toward goal keypoint
        if getattr(self.cfg_task, "interaction_frame_mode", "geometric") == "dynamic":
            f = self.force_sensor_world_smooth[:, 0:3]                                # (E,3) world reaction
            z = f / torch.linalg.norm(f, dim=-1, keepdim=True).clamp_min(1e-6)        # z along the reaction
        else:
            z = self.surface_normal                                                  # z along the surface normal
        x = to_goal - (to_goal * z).sum(-1, keepdim=True) * z                         # goal dir ⊥ z
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
        frame = torch.stack([x, torch.cross(z, x, dim=-1), z], dim=-1)                # (E,3,3)
        # Off-contact -> R_eef so the stiffness rotation collapses to identity (stiffness in the EEF frame).
        R_eef = matrix_from_quat(self.fingertip_midpoint_quat)                        # (E,3,3) world<-eef
        return torch.where(self.in_contact_any[:, None, None], frame, R_eef)

    # ------------------------------------------------------------------
    # Scene: procedural plate (fixed) + cylinder (held)
    # ------------------------------------------------------------------
    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        # Drive the spawned plate mesh from the task fields so an env_cfg_override on
        # plate_length / plate_width / plate_thickness actually resizes it. The env's path
        # geometry already reads these fields (plate_length/thickness); the CuboidCfg.size was
        # the one place still baked from the module constant. _setup_scene runs during gym.make,
        # AFTER build_env applies env_cfg_overrides, so cfg_task reflects any override here.
        self.cfg_task.fixed_asset.spawn.size = (
            float(self.cfg_task.plate_length),
            float(self.cfg_task.plate_width),
            float(self.cfg_task.plate_thickness),
        )
        # Same story for friction: the spawn materials bake the module defaults and this env never
        # calls factory set_friction, so drive them from the task fields (config-overridable). PhysX
        # combines the plate's and cylinder's coefficients by their friction_combine_mode (default
        # "average"): realized mu = 0.5*(plate_friction + held_friction).
        self.cfg_task.fixed_asset.spawn.physics_material.static_friction = float(self.cfg_task.plate_friction)
        self.cfg_task.fixed_asset.spawn.physics_material.dynamic_friction = float(self.cfg_task.plate_friction)
        self.cfg_task.held_asset.spawn.physics_material.static_friction = float(self.cfg_task.held_friction)
        self.cfg_task.held_asset.spawn.physics_material.dynamic_friction = float(self.cfg_task.held_friction)
        self._fixed_asset = RigidObject(self.cfg_task.fixed_asset)
        self._held_asset = RigidObject(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # In-hand grasp pose (tip-down cylinder, optional in-hand tilt)
    # ------------------------------------------------------------------
    def get_handheld_asset_relative_pose(self):
        # In-grip pose of the cylinder. Position: the procedural CylinderCfg's prim origin is at the
        # cylinder's CENTER (it spans [-H/2, +H/2] along its local z), unlike the Factory peg USD
        # whose origin is at the base/tip. So grip at (H/2 - fingerpad) from the center => fingerpad
        # below the TOP end, with the cylinder hanging down and its lower end as the contact tip.
        # Orientation: a fixed grasp TILT (cylinder relative to the gripper), folded into the in-grip
        # rotation the SAME way the weld does (peg_weld_wrapper._compute_peg_in_fingertip):
        # rel_quat = quat_conjugate(perturb), perturb = quat_from_euler_xyz(grasp_weld_tilt_deg).
        # Zero tilt => identity (rigid ALIGNED grip), so the reset math below is unchanged there.
        rel_pos = torch.zeros((self.num_envs, 3), device=self.device)
        rel_pos[:, 2] = self.cfg_task.held_asset_cfg.height / 2.0
        rel_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        tilt = self.cfg_task.grasp_weld_tilt_deg
        if any(abs(float(v)) > 1e-9 for v in tilt):
            r, p, y = (float(np.deg2rad(float(v))) for v in tilt)
            perturb = torch_utils.quat_from_euler_xyz(
                torch.tensor([r], device=self.device),
                torch.tensor([p], device=self.device),
                torch.tensor([y], device=self.device),
            )
            rel_quat = torch_utils.quat_conjugate(perturb).repeat(self.num_envs, 1)
        else:
            rel_quat = self._identity_quat()
        return rel_pos, rel_quat

    # ------------------------------------------------------------------
    # Intermediate values: stash task geometry after the Forge base compute
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)  # ForgeEnv: noise + FT sensing

        # PATH parametrization from the fixed-asset (plate) pose: near/far edge centers on the top
        # surface. The plate's normal / near->far direction returned here are a BOOTSTRAP, used only
        # to locate the contact point; the normal + travel direction the reward/critic actually use
        # are the LOCAL ones queried at the contact point below (general for non-flat surfaces).
        start, goal, plate_normal, plate_path_dir, _ = self._surface_frame()
        self.start_world = start
        self.goal_world = goal
        self._plate_normal = plate_normal
        self._plate_path_dir = plate_path_dir

        # The procedural cylinder's origin is its CENTER (held_pos), so its two flat ends are
        # held_pos +/- (H/2)*cyl_axis. The CONTACT tip is the lower end (smaller projection onto the
        # plate normal) — sign-robust, independent of which way the grasp leaves held-frame +z.
        self.cyl_axis = self._rotate_vec(self.held_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        half = 0.5 * self.cfg_task.held_asset_cfg.height
        end_plus = self.held_pos + half * self.cyl_axis
        end_minus = self.held_pos - half * self.cyl_axis
        proj_plus = (end_plus * plate_normal).sum(-1, keepdim=True)
        proj_minus = (end_minus * plate_normal).sum(-1, keepdim=True)
        self.cyl_tip = torch.where(proj_plus < proj_minus, end_plus, end_minus)

        # CONTACT POINT ("point of contact if any"): the surface point at/under the tip — the tip
        # projected onto the surface (plate top plane through `start`). When touching, this IS the
        # contact point; otherwise the nearest surface point, so the local frame is always defined.
        signed_dist = ((self.cyl_tip - start) * plate_normal).sum(-1, keepdim=True)
        self.contact_point = self.cyl_tip - signed_dist * plate_normal
        # Signed height of the tool tip above the surface (m): >0 above, ~0 in contact, <0 penetrating.
        # Used by the orientation reward's near-surface gate.
        self.tip_surface_dist = signed_dist.squeeze(-1)

        # LOCAL surface frame AT THE CONTACT POINT — the normal + travel direction the reward and the
        # critic use. General hook: a non-flat surface overrides `_surface_normal_and_dir_at` to
        # return the LOCAL normal (surface gradient) + local tangent there. Flat plate => plate
        # constants, so this is numerically identical to before for the flat surface.
        normal, path_dir = self._surface_normal_and_dir_at(self.contact_point)
        self.surface_normal = normal
        self.path_dir = path_dir
        self.d_lat = torch.cross(normal, path_dir, dim=-1)                    # d_lat = n x d (in-plane lateral)
        self.cross_dir = self.d_lat

        # Ideal path p0 -> p_g. path_dir is the (local) travel direction at the contact point.
        rel = self.cyl_tip - start                                            # dp = tip - p0
        self.path_length = torch.linalg.norm(goal - start, dim=-1)            # L = |p_g - p0|
        self.progress = (rel * path_dir).sum(-1)                             # s = dp . d (along-track)
        self.cross_track = (rel * self.d_lat).sum(-1)                        # e_perp = dp . d_lat (signed)

        # TIME-BASED pace setpoint s_ref = clamp(v * pace_tau, 0, L), pace_tau = time since first
        # contact (advanced every step once contacted, in _get_rewards). Drives ONLY the pace reward
        # (progress - s_ref): the ideal keeps moving with time even off-contact, so drifting off just
        # costs pace reward and never rewards turning around. This is NOT the observation setpoint.
        v_des = float(self.cfg_task.desired_speed_cm_s) / 100.0              # cm/s -> m/s
        self.s_ref = (v_des * self.pace_tau).clamp_min(0.0).minimum(self.path_length)

        # Keypoints = evenly spaced arc-length checkpoints on the ideal path p0->p_g, spacing v*dt;
        # COUNT = L/(v*dt), NOT a free parameter. Achievement/passed accounting is in _get_rewards.
        step_dt = float(getattr(self, "step_dt", self.physics_dt * self.cfg.decimation))
        self.keypoint_spacing = max(v_des * step_dt, 1e-6)
        self.keypoints_total = torch.floor(self.path_length / self.keypoint_spacing).clamp_min(1.0).long()

        # OBSERVATION setpoint. BEFORE first contact the target is HELD at keypoint 0 (k0 = the
        # near-edge spawn point, directly under the tip): the peg is spawned over k0 and told to go to
        # k0, so it descends straight DOWN onto the surface. AFTER first contact (t_contact finite) the
        # target advances to the NEXT keypoint ahead of the SOURCE arc length (floor(source/spacing) + 1),
        # i.e. it drags along the line ONE keypoint at a time.
        #
        # Two source modes (task.setpoint_pace_driven):
        #  * ROBOT-DRIVEN (default): source = self.progress (the arm's realized along-track distance), so
        #    the target sits one keypoint ahead of where the arm actually is and WAITS for it to catch up.
        #    Advancing on the along-track projection means drifting off the surface along d still advances
        #    it (no turn-around incentive), and its lateral offset from the tip carries the "off the line" error.
        #  * PACE-DRIVEN: source = self.s_ref (the time-based pace target v*pace_tau), so the target marches
        #    forward on the pace CLOCK and never waits for the arm. Same keypoint discretization, ratchet,
        #    and pre-contact hold (s_ref is 0 until first contact, so both modes hold at k0 until then).
        setpoint_source = self.s_ref if bool(self.cfg_task.setpoint_pace_driven) else self.progress
        kp_passed_now = torch.floor(setpoint_source / self.keypoint_spacing).clamp_min(0).long()
        next_ahead = (kp_passed_now + 1).minimum(self.keypoints_total)
        made_contact = torch.isfinite(self.t_contact)                         # touched at any point this episode
        new_setpoint = torch.where(made_contact, next_ahead, torch.zeros_like(next_ahead))
        # RATCHET: the observation setpoint only ever advances. Moving backward (progress dropping)
        # does NOT pull the target back — it stays at the furthest keypoint reached this episode.
        # Reset to 0 in _reset_idx (and carried by the efficient-reset cache).
        self.setpoint_kp_idx = torch.maximum(self.setpoint_kp_idx, new_setpoint)
        setpoint_arclen = (self.setpoint_kp_idx.float() * self.keypoint_spacing).minimum(self.path_length)
        self.setpoint_pos = start + setpoint_arclen.unsqueeze(-1) * path_dir
        # The keypoint AFTER the current one (index+1, clamped to total/L). Used by interaction_frame_
        # world() as a fallback goal when the contact point sits essentially on the current keypoint,
        # where the direction to it (the frame's x-axis) is ill-defined.
        next_arclen = ((self.setpoint_kp_idx + 1).float() * self.keypoint_spacing).minimum(self.path_length)
        self.next_setpoint_pos = start + next_arclen.unsqueeze(-1) * path_dir

        # Measured normal force = projection of the FT force onto the true surface normal.
        # The raw smoothed force (force_sensor_world_smooth) is in the force_sensor child-joint
        # frame, NOT world — so rotate it to world (by the sensor body's world quaternion) BEFORE
        # projecting onto the world-frame normal. force_sensor_world_smooth is ALREADY world (Forge:
        # get_link_incoming_joint_force, EMA-smoothed; change_FT_frame re-references only the torque
        # via identity rotation, so the FORCE vector stays world). Project it DIRECTLY onto the world
        # normal — NO quat_apply. The old quat_apply(sensor_quat, ...) double-rotated a world vector,
        # making the sign depend on tool orientation (+ tip-down, - when tilted off the normal). Sign
        # verified by a contact smoke test: this reads POSITIVE when pressing into the surface (matches
        # desired_force > 0) and is orientation-INDEPENDENT (world force . world normal), so it no
        # longer flips as the tool tilts.
        self.measured_normal_force = (self.force_sensor_world_smooth[:, 0:3] * normal).sum(-1)

        # Orientation: angle between the held cylinder's long axis and the surface NORMAL, in RADIANS
        # (arccos(|axis . normal|); 0 = axis parallel to the normal = tip-down/perpendicular, pi/2 =
        # axis in the plane = flat). The cylinder is axisymmetric (|.| folds the axis to the acute
        # angle), so only this axis-vs-normal angle is constrained (free to spin about the normal).
        # REALIZED from the physics held orientation. orn_error = desired - actual (signed, RADIANS)
        # — the value the orientation reward squashes (and |orn_error| gates success). The config gives
        # the desired angle in DEGREES (human-readable); it is converted to radians here.
        self.angle_from_normal = torch.arccos(
            (self.cyl_axis * normal).sum(-1).abs().clamp(0.0, 1.0)
        )                                                                    # rad, 0 = tip-down
        self.orn_error = float(np.deg2rad(self.cfg_task.orientation_desired_angle_deg)) - self.angle_from_normal

        # --- Held-object END frame (the un-held / contact end) ---
        # Canonical "held object frame" = a frame at the un-held tip, oriented so z is the
        # cylinder long axis and, at nominal grasp, x aligns with the EEF x-axis. The held body
        # frame's x is -EEF_x at nominal (factory's flip_z grasp), so a 180° rotation of the held
        # body frame about its own z (== cylinder axis) lands held_end_x on EEF_x while keeping
        # z along the cylinder. Tracks any in-hand tilt (via held_quat). Position = the contact tip.
        # Used for alignment / insertion checks (NOT observations).
        rz180 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.held_end_pos = self.cyl_tip
        self.held_end_quat = torch_utils.quat_mul(self.held_quat, rz180)

        # --- Interaction frame (env-defined; viz/obs only, NOT control) ---
        # Position: the actual contact point of the (possibly tilted) lower cylinder face with the
        # plane — the lowest rim point. perp = component of -normal in the face plane (⊥ cyl axis);
        # contact = tip-center + radius * perp_dir. When the cylinder is perpendicular to the
        # surface, perp -> 0 and the contact collapses to the tip center.
        radius = 0.5 * self.cfg_task.held_asset_cfg.diameter
        neg_n = -normal
        perp = neg_n - (neg_n * self.cyl_axis).sum(-1, keepdim=True) * self.cyl_axis
        perp_norm = torch.linalg.norm(perp, dim=-1, keepdim=True)
        downhill = torch.where(perp_norm > 1e-6, perp / perp_norm.clamp_min(1e-6), torch.zeros_like(perp))
        self.interaction_pos = self.cyl_tip + radius * downhill

        # In-contact bool (drives the interaction frame + the pace schedule clock). Prefer the
        # contact-sensor wrapper's per-axis state; fall back to a small normal-force threshold when
        # the contact sensor is disabled, so the env stays self-contained. Computed BEFORE the
        # interaction frame below, which now consumes it (off-contact -> identity/EEF frame).
        cw = getattr(self, "in_contact", None)
        if torch.is_tensor(cw):
            self.in_contact_any = cw.any(dim=1)
        else:
            self.in_contact_any = self.measured_normal_force.abs() > 0.1
        self.interaction_exists = self.in_contact_any

        # Orientation = the SAME [path_dir, d_lat, surface_normal] frame the controller's fixed-rot
        # stiffness and the impedance metrics consume (interaction_frame_world()), so the viz marker
        # matches what control actually uses — one source of truth. (Previously x tracked the
        # instantaneous velocity direction, which diverged from the path frame: a viz-only mismatch.)
        R_int = self.interaction_frame_world()                                        # (E,3,3) world<-interaction (control frame)
        self.interaction_quat = quat_from_matrix(R_int)
        # Ground-truth rotation TARGET for the supervised-rotation loss: exactly the eef<-interaction
        # rotation a fixed-rot controller would use (R_eefᵀ·interaction_frame_world()); noise-free,
        # respects the mode (geometric/dynamic) and is identity off-contact. Published for the memory
        # buffer to pick up (flattened (E,9)); harmless when the loss is off.
        R_eef = matrix_from_quat(self.fingertip_midpoint_quat)                        # (E,3,3) world<-eef
        self.extras["gt_interaction_rot"] = (R_eef.transpose(1, 2) @ R_int).reshape(self.num_envs, 9)

        # Measured normal force, CONTACT-CONDITIONAL: NaN off-contact and tagged "(stat)" so
        # block_agent's _accum_dist_stat (which drops non-finite values) means ONLY over in-contact
        # steps. Including free-space ~0 readings pulls the average down by an ill-defined amount
        # (it scales with the hovering fraction of the episode). Guarded — extras may lack to_log yet.
        if hasattr(self, "extras"):
            masked_force = torch.where(
                self.in_contact_any,
                self.measured_normal_force,
                torch.full_like(self.measured_normal_force, float("nan")),
            )
            self.extras.setdefault("to_log", {})["Force / Normal Measured (stat)"] = masked_force.detach()

    # ------------------------------------------------------------------
    # Observations (all EEF-frame): goal-relative pose (sign-aligned) + EEF vel/force
    # ------------------------------------------------------------------
    def _fingertip_wrench(self):
        """Clean 6-D wrench (F, T) expressed in the fingertip-midpoint (EEF) frame.

        force_sensor_world_smooth is in the WORLD frame (Forge: get_link_incoming_joint_force,
        EMA-smoothed; its change_FT_frame keeps identity rotation, so the force vector stays world).
        Express it in the EEF frame the policy acts/controls in: (1) re-reference the torque from the
        sensor origin to the fingertip origin (add (p_sensor - p_fingertip) x F, world frame), then
        (2) rotate both F and T by the fingertip's world->local rotation. The result is BODY-FIXED —
        independent of the randomized world/plate orientation. (The prior version mislabeled the
        world force as sensor-frame in change_FT_frame, over-rotating it by R_sensor.)
        """
        raw = self.force_sensor_world_smooth
        F_w, T_w = raw[:, 0:3], raw[:, 3:6]
        bp = self._robot.data.body_pos_w
        fq = self._robot.data.body_quat_w[:, self.fingertip_body_idx]           # world<-fingertip
        r = bp[:, self.force_sensor_body_idx] - bp[:, self.fingertip_body_idx]  # sensor - fingertip (world)
        T_at_ft_w = T_w + torch.cross(r, F_w, dim=-1)                           # move reference to fingertip origin
        fq_conj = torch_utils.quat_conjugate(fq)                               # fingertip<-world
        return quat_apply(fq_conj, F_w), quat_apply(fq_conj, T_at_ft_w)

    def _get_observations(self):
        setpoint_pos = self.setpoint_pos

        def _setpoint_pos_rel(eef_pos, eef_quat):
            # MOVING-SETPOINT position relative to the tool, in the EEF frame, sign-aligned: a
            # positive component => a positive action on that EEF axis moves toward the setpoint.
            # No orientation setpoint — orientation is inferred elsewhere (force/torque + reward).
            return quat_apply(torch_utils.quat_conjugate(eef_quat), setpoint_pos - eef_pos)

        def _to_eef(vec, eef_quat):
            return quat_apply(torch_utils.quat_conjugate(eef_quat), vec)

        F_ft, T_ft = self._fingertip_wrench()  # clean EEF-frame wrench
        force_noise = torch.randn((self.num_envs, 3), device=self.device) * float(self.cfg.obs_rand.ft_force)

        tnf = self.desired_force[:, None]  # per-episode desired normal force (the force target)
        prev_actions = self.actions.clone()
        # EEF orientation (world frame): CLEAN quat for the critic; the policy gets a noisy estimate.
        # Forge's noisy_fingertip_quat zeroes the quaternion w,z (a peg-upright encoding) and is
        # INVALID for a full-orientation tool, so we build our own valid noisy quat: perturb the
        # clean quat by a random small body-frame rotation of magnitude ~N(0, fingertip_rot_deg)
        # (configurable via noise_cfg.fingertip_rot_deg -> obs_rand.fingertip_rot_deg).
        eef_quat = self.fingertip_midpoint_quat
        _axis = torch.randn((self.num_envs, 3), device=self.device)
        _axis = _axis / torch.linalg.norm(_axis, dim=1, keepdim=True).clamp_min(1e-8)
        _angle = torch.randn((self.num_envs,), device=self.device) * float(
            np.deg2rad(self.cfg.obs_rand.fingertip_rot_deg)
        )
        noisy_eef_quat = torch_utils.quat_mul(eef_quat, torch_utils.quat_from_angle_axis(_angle, _axis))

        # POLICY — noisy position + noisy orientation estimate.
        obs_dict = {
            "setpoint_pos_rel": _setpoint_pos_rel(self.noisy_fingertip_pos, noisy_eef_quat),
            "fingertip_quat": noisy_eef_quat,  # EEF orientation, world frame (noisy)
            "ee_linvel": _to_eef(self.fingertip_midpoint_linvel, noisy_eef_quat),
            "ee_angvel": _to_eef(self.fingertip_midpoint_angvel, noisy_eef_quat),
            "ft_force": F_ft + force_noise,
            "ft_torque_eef": T_ft + force_noise,
            "target_normal_force": tnf,
            "prev_actions": prev_actions,
        }
        # CRITIC — clean + privileged geometry. The contact-frame geometry (surface_normal, path_dir)
        # is ZEROED out of contact: with no contact point there is no defined surface frame, so the
        # critic gets a clean "no contact" signal. The INTERNAL self.surface_normal/self.path_dir stay
        # valid (the setpoint + reward need a direction through brief bounces) — only the observed copy
        # is gated. Raw plate pose (fixed_pos/fixed_quat) dropped: redundant with these + the setpoint.
        _contact = self.in_contact_any.float().unsqueeze(-1)
        state_dict = {
            "setpoint_pos_rel": _setpoint_pos_rel(self.fingertip_midpoint_pos, eef_quat),
            "ee_linvel": _to_eef(self.fingertip_midpoint_linvel, eef_quat),
            "ee_angvel": _to_eef(self.fingertip_midpoint_angvel, eef_quat),
            "ft_force": F_ft,
            "ft_torque_eef": T_ft,
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": eef_quat,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "task_prop_gains": self.task_prop_gains,
            "ema_factor": self.ema_factor,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "surface_normal": self.surface_normal * _contact,
            "path_dir": self.path_dir * _contact,
            "progress": self.progress[:, None],
            "cross_track": self.cross_track[:, None],
            "orn_error": self.orn_error[:, None],
            "normal_force": self.measured_normal_force[:, None],
            "target_normal_force": tnf,
            "prev_actions": prev_actions,
        }

        # Publish the 6-D rotation-matrix counterpart of every quaternion channel (inert unless
        # the corresponding *_rot6d key is in obs_order/state_order; selected before gym.make).
        from wrappers.sensors.orientation_obs import augment_obs_dict_with_rot6d

        augment_obs_dict_with_rot6d(obs_dict)
        augment_obs_dict_with_rot6d(state_dict)

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    # ------------------------------------------------------------------
    # Success + reward (reward is a STRUCTURAL STUB — terms come in a later pass)
    # ------------------------------------------------------------------
    def _get_curr_successes(self, success_threshold=None, check_rot=False):
        at_goal = torch.linalg.norm(self.cyl_tip - self.goal_world, dim=-1) < self.cfg_task.success_pos_tol
        oriented = self.orn_error.abs() < float(np.deg2rad(self.cfg_task.success_orn_tol_deg))  # both radians
        success = torch.logical_and(at_goal, oriented)
        # Success also requires the tool to have actually DRAGGED enough of the path, so a fly-to-goal
        # that skipped the surface can't read as success and every keypoint means something. The gate
        # is achieved/total >= success_keypoint_frac; set it to 0 for pure pose-only success, or 1.0
        # to demand EVERY keypoint achieved.
        kp_frac = float(self.cfg_task.success_keypoint_frac)
        if kp_frac > 0.0:
            coverage = self.keypoints_achieved.float() / self.keypoints_total.clamp_min(1).float()
            success = success & (coverage >= kp_frac)
        return success

    def viz_snapshot(self) -> dict:
        """Per-env quantities for the surface recorder overlays (keypoint balls, force/orientation
        gauges, top-down path inset). Call AFTER _compute_intermediate_values has run for the step
        (the recorder calls it right after env.step). All CPU tensors; viz-only, no training effect.

        force_squash / orn_squash are the SAME squashing_fn values the reward uses (in [0, 1], 1 =
        perfect), so the gauges read exactly what the force / orientation reward terms pay.
        """
        cfg = self.cfg_task
        force_squash = factory_utils.squashing_fn(
            self.desired_force - self.measured_normal_force, cfg.force_a, cfg.force_b
        )
        orn_squash = factory_utils.squashing_fn(self.orn_error, cfg.orientation_a, cfg.orientation_b)
        # Physical read-outs for the gauges: measured normal force (N), and the tool-axis angle
        # relative to the DESIRED angle-off-normal (deg, signed: + = more tilted than commanded).
        angle_dev_deg = torch.rad2deg(self.angle_from_normal) - float(cfg.orientation_desired_angle_deg)
        return {
            "start_w": self.start_world.detach().cpu(),          # (E,3) near-edge center (path p0)
            "goal_w": self.goal_world.detach().cpu(),            # (E,3) far-edge center (goal)
            "path_dir": self.path_dir.detach().cpu(),            # (E,3) along-track unit dir d
            "d_lat": self.d_lat.detach().cpu(),                  # (E,3) in-plane lateral unit dir
            "surface_normal": self.surface_normal.detach().cpu(),# (E,3) surface normal
            "tip_w": self.contact_point.detach().cpu(),          # (E,3) tip projected onto the surface
            "path_length": self.path_length.detach().cpu(),      # (E,) L
            "keypoints_total": self.keypoints_total.detach().cpu(),  # (E,) count
            "keypoint_spacing": float(self.keypoint_spacing),    # scalar (m)
            "progress": self.progress.detach().cpu(),            # (E,) along-track arc length
            "s_ref": self.s_ref.detach().cpu(),                  # (E,) time-based PACE setpoint arc length
            "in_contact": self.in_contact_any.detach().cpu().bool(),  # (E,)
            "force_squash": force_squash.detach().cpu(),         # (E,) in [0,1] (gauge fill/colour)
            "orn_squash": orn_squash.detach().cpu(),             # (E,) in [0,1] (gauge fill/colour)
            "force_N": self.measured_normal_force.detach().cpu(),        # (E,) measured normal force, N
            "desired_force_N": self.desired_force.detach().cpu(),        # (E,) target force, N
            "angle_dev_deg": angle_dev_deg.detach().cpu(),              # (E,) deg off the desired angle
            "tip_surface_dist": self.tip_surface_dist.detach().cpu(),  # (E,) signed height above surface
        }

    def _get_rewards(self):
        """Reward = bounded task terms + the Factory action penalties.

        Task terms (force, orientation, straightness, pace) are weight * squashing_fn(raw signed
        REALIZED value, a, b) — computed from measured/physics state only (never control targets),
        peaking at value=0. The two action penalties (action_penalty_ee, action_grad_penalty) are
        the FactoryEnv linear penalties applied with NEGATIVE scales, both default 0.0 (off). A
        one-shot success_time bonus squashes (ideal completion time - actual success time), plus a
        per-step 'contact' bonus (+contact_weight while in contact). Gating: straightness / pace pay in
        air AND contact for a continuous tracking signal, but the AIR contribution is downweighted to
        *_air_weight (contact pays the full *_weight); FORCE and success_time require contact; ORIENTATION
        requires being NEAR the surface (tip within orientation_gate_dist of the contact point); contact /
        action penalties are always evaluated. The scorer's _factory_scales must stay in sync with below.
        """
        curr_successes = self._get_curr_successes()
        cfg = self.cfg_task

        # --- Keypoint accounting (progress-projection based; drives the success weight + gate) ---
        # kp index at an arc-length = floor(progress / spacing). A keypoint is ACHIEVED the first time
        # the progress frontier passes it during a GATED step: in contact AND laterally on-track
        # (|cross_track| < keypoint_track_tol). Multiple boundaries crossed in one step all count (no
        # exactly-one rule) so legitimate pace variation isn't punished; but boundaries crossed while
        # NOT gated (in the air / off-track) are forfeited — kp_prev still advances through them, so a
        # fly-ahead shortcut permanently loses those keypoints. kp_ach_frontier is the furthest kp
        # index already credited (each keypoint counts once). PASSED: running-max kp index the
        # projection reached, gated or not (pure progress frontier). prev_progress is last step's value
        # (updated at the end of this method), so this must run before that update.
        Ktot = self.keypoints_total
        kp_prev = torch.floor(self.prev_progress / self.keypoint_spacing).clamp_min(0).long().minimum(Ktot)
        kp_curr = torch.floor(self.progress / self.keypoint_spacing).clamp_min(0).long().minimum(Ktot)
        on_track = self.cross_track.abs() < float(cfg.keypoint_track_tol)
        gate = self.in_contact_any & on_track                                # in contact AND on-track this step
        # Newly achieved = boundaries beyond BOTH last step's index and the achieved frontier, but only
        # when gated (so uncredited air/off-track crossings between kp_prev and the frontier are lost).
        newly_achieved = (kp_curr - torch.maximum(kp_prev, self.kp_ach_frontier)).clamp_min(0)
        newly_achieved = torch.where(gate, newly_achieved, torch.zeros_like(newly_achieved))  # (E,) # this step
        self.kp_ach_frontier = torch.where(gate, torch.maximum(self.kp_ach_frontier, kp_curr), self.kp_ach_frontier)
        self.keypoints_achieved = (self.keypoints_achieved + newly_achieved).minimum(Ktot)
        self.keypoints_passed = torch.maximum(self.keypoints_passed, kp_curr)
        # Success reward weight = fraction of keypoints achieved (partial credit for the drag).
        success_frac = self.keypoints_achieved.float() / self.keypoints_total.clamp_min(1).float()

        # Force tracking: desired (sampled, along the surface normal) - measured normal force
        # (world-rotated EEF force projected onto the surface normal, for a same-frame difference).
        force_value = self.desired_force - self.measured_normal_force          # (E,) N, signed
        # Orientation: desired - realized angle between the tool axis and the surface normal
        # (self.orn_error, RADIANS, signed; 0 = held exactly at the commanded angle).
        orn_value = self.orn_error                                            # (E,) rad, signed
        # Straightness: signed cross-track error e_perp = dp . d_lat (computed in _compute).
        straightness_value = self.cross_track                                # (E,) m, signed
        # Pace. Two modes (task.vel_based_pace_enabled, default ON — see task cfg):
        #  * VELOCITY-based (default): value = (measured along-track speed) - v_des, where the measured
        #    speed is d(progress)/dt and v_des = desired_speed_cm_s (m/s). LIVE gradient at any lag.
        #  * POSITION-based (legacy A/B): value = progress - s_ref (the time-based moving setpoint),
        #    which runs away once the tool falls behind so the gradient dies.
        # pace_a/pace_b/pace_wt select the active term's squashing params + weight (used below).
        pace_speed_gate = 1.0                                                # optional hard min-speed cliff (vel pace only); 1.0 = no gate
        if bool(cfg.vel_based_pace_enabled):
            _sdt = float(getattr(self, "step_dt", self.physics_dt * self.cfg.decimation))
            _vdes = float(cfg.desired_speed_cm_s) / 100.0                    # cm/s -> m/s
            v_along = (self.progress - self.prev_progress) / _sdt            # (E,) along-track speed (m/s)
            pace_value = v_along - _vdes                                     # (E,) speed error (m/s), signed
            pace_a, pace_b, pace_wt = cfg.vel_based_pace_a, cfg.vel_based_pace_b, cfg.vel_based_pace_weight
            # MIN-SPEED gate: below the threshold along-track speed, pay ZERO (hard cliff) so the tool
            # must commit to real forward motion; holding still earns no pace at all.
            if bool(cfg.vel_based_pace_min_speed_enabled):
                pace_speed_gate = (v_along >= float(cfg.vel_based_pace_min_speed)).float()   # (E,) 0/1
        else:
            pace_value = self.progress - self.s_ref                         # (E,) m, signed (position pace)
            pace_a, pace_b, pace_wt = cfg.pace_a, cfg.pace_b, cfg.pace_weight
        # Action penalties: EXACTLY as in FactoryEnv._get_factory_rew_dict (linear, NOT squashed).
        # action_penalty_ee penalizes raw action magnitude; action_grad_penalty penalizes the
        # step-to-step action change (chatter). Applied with NEGATIVE scales; both scales default
        # 0.0 (inherited from FactoryTask -> off). prev_actions holds the previous step's action
        # (refreshed at the end of this method).
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)

        # Time-to-success bonus (ONE-SHOT, paid on the first success step). t_now: episode wall-clock
        # time (s). Latch t_contact at the FIRST in-contact step; t* (ideal completion time) =
        # t_contact + L/v_des is the time if, from first contact, the tool traced the whole path L at
        # the desired speed. The bonus is squashing_fn(t* - t_success): peaks when success lands at
        # the ideal time, penalizing both dawdling and cutting the path short. Note
        # t* - t_success = L/v_des - (t_success - t_contact), so it scores the contact->success
        # DURATION against the ideal L/v_des and is invariant to the absolute clock offset.
        step_dt = float(getattr(self, "step_dt", self.physics_dt * self.cfg.decimation))
        t_now = self.episode_length_buf.float() * step_dt                     # (E,) s
        newly_contacted = self.in_contact_any & torch.isinf(self.t_contact)
        self.t_contact = torch.where(newly_contacted, t_now, self.t_contact)
        v_des = float(cfg.desired_speed_cm_s) / 100.0                         # cm/s -> m/s
        t_star = self.t_contact + self.path_length / max(v_des, 1e-6)         # ideal completion time (s)
        success_now = (
            curr_successes
            & self.in_contact_any
            & (~self.success_reward_given)
            & torch.isfinite(self.t_contact)
        )
        success_time_value = t_star - t_now                                  # (E,) ideal - actual, s
        # Success reward = weight * squashing(time error) * (achieved/total). BOTH the completion-time
        # squashing and the keypoint-achieved fraction scale the same weight, so a fast finish that
        # skipped keypoints is discounted in proportion to how much of the path it actually dragged.
        success_time_reward = torch.where(
            success_now,
            factory_utils.squashing_fn(success_time_value, cfg.success_time_a, cfg.success_time_b) * success_frac,
            torch.zeros_like(t_now),
        )
        # Latch (once, at first success): step reached + the SCALED success_time reward earned.
        self.success_step = torch.where(success_now, self.episode_length_buf.float(), self.success_step)
        self.success_time_earned = torch.where(
            success_now, success_time_reward * float(cfg.success_time_weight), self.success_time_earned
        )
        self.success_reward_given = self.success_reward_given | success_now

        # Gating. straightness / pace pay in air AND contact for a CONTINUOUS tracking signal across
        # touchdown (same path d=start->goal, setpoint frozen at start until contact), BUT the air
        # contribution is DOWNWEIGHTED to *_air_weight (folded as a per-env factor = air/full weight,
        # so the scalar scale and scorer stay = *_weight). This keeps a faint gradient for lining up
        # over the start while making contact strictly more rewarding (kills the hover-and-track farm).
        # ORIENTATION pays only NEAR the surface (tip within orientation_gate_dist of the contact
        # point) — holds the angle on final approach without the "hover high and hold 90deg" farm.
        # FORCE is gated on contact (a=0.25 off-contact would otherwise pay a farmable ~0.09/step in
        # air). contact / action penalties always evaluate; success_time is gated via success_now.
        contact = self.in_contact_any.float()
        near_surface = (self.tip_surface_dist < float(cfg.orientation_gate_dist)).float()
        straight_air = float(cfg.straightness_air_weight) / max(float(cfg.straightness_weight), 1e-9)
        pace_air = float(cfg.pace_air_weight) / max(float(pace_wt), 1e-9)      # ratio vs the ACTIVE pace weight
        straight_factor = torch.where(self.in_contact_any, torch.ones_like(contact), torch.full_like(contact, straight_air))
        pace_factor = torch.where(self.in_contact_any, torch.ones_like(contact), torch.full_like(contact, pace_air))
        rew_dict = {
            "force": factory_utils.squashing_fn(force_value, cfg.force_a, cfg.force_b) * contact,
            "orientation": factory_utils.squashing_fn(orn_value, cfg.orientation_a, cfg.orientation_b) * near_surface,
            "straightness": factory_utils.squashing_fn(straightness_value, cfg.straightness_a, cfg.straightness_b) * straight_factor,
            "pace": factory_utils.squashing_fn(pace_value, pace_a, pace_b) * pace_factor * pace_speed_gate,
            # Fixed bonus paid ONCE per keypoint, the first time it is achieved (count newly achieved
            # this step; usually 0/1 but >1 when a single fast step drags across multiple boundaries).
            "keypoint": newly_achieved.float(),
            "contact": contact,
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "success_time": success_time_reward,
        }
        rew_scales = {
            "force": float(cfg.force_weight),
            "orientation": float(cfg.orientation_weight),
            "straightness": float(cfg.straightness_weight),
            "pace": float(pace_wt),                                          # active pace weight (vel- or position-based)
            "keypoint": float(cfg.keypoint_reward_weight),
            "contact": float(cfg.contact_weight),
            "action_penalty_ee": -float(cfg.action_penalty_ee_scale),
            "action_grad_penalty": -float(cfg.action_grad_penalty_scale),
            "success_time": float(cfg.success_time_weight),
        }
        rew_buf = torch.zeros(self.num_envs, device=self.device)
        for name in rew_dict:
            rew_buf = rew_buf + rew_dict[name] * rew_scales[name]

        # Advance the pace clock by one env step once FIRST contact has happened (time-based, NOT gated
        # on staying in contact) so s_ref = v*(t - t_contact) keeps moving through a bounce.
        self.pace_tau = self.pace_tau + step_dt * torch.isfinite(self.t_contact).float()

        # --- Contact-quality accumulation + per-episode publish ---
        c = self.in_contact_any
        self.cq_contact += c.long()                                   # steps in contact this episode
        self.cq_starts += (c & ~self.cq_prev).long()                 # no-contact -> contact (run starts)
        self.cq_breaks += (~c & self.cq_prev).long()                 # contact -> no-contact (bounces)
        self.cq_prev = c.clone()

        # --- Drag-performance accumulation (IN CONTACT) + keypoint (checkpoint) crossing ---
        cf = c.float()
        theta_deg = torch.rad2deg(self.angle_from_normal)
        # Drag SPEED = the rate the CONTACT POINT moves along d / perp to d, i.e. d(progress)/dt and
        # d(cross_track)/dt — NOT the EEF finite-diff velocity. The old ee_linvel-based speed decoupled
        # from the keypoints: the tool advances the tip by PIVOTING (rotation) while the wrist barely
        # translates, so ee_linvel . d read ~0 even as progress/keypoints climbed. The progress rate is
        # exactly what the keypoints count, so this speed is consistent with them.
        v_along_prog = (self.progress - self.prev_progress) / step_dt
        v_perp_prog = (self.cross_track - self.prev_cross_track) / step_dt
        self.drag_count += c.long()
        for _n, _v in (("force", self.measured_normal_force), ("speed_d", v_along_prog),
                       ("speed_perp", v_perp_prog), ("theta", theta_deg)):
            getattr(self, f"drag_sum_{_n}").add_(_v * cf)
            getattr(self, f"drag_sumsq_{_n}").add_(_v * _v * cf)
        # (keypoint achievement/passed were computed earlier, before the success bonus, so it could be
        # weighted by achieved/total.) Roll the per-step history forward for the next step's crossing
        # test and drag-speed finite differences.
        self.prev_progress = self.progress.clone()
        self.prev_cross_track = self.cross_track.clone()
        # Publish per-episode metrics for the envs finishing this step (mask = reset_buf). Snapshots
        # (new tensors), so the subsequent _reset_idx zeroing the accumulators doesn't affect them.
        # block_agent means these over the finishing envs per agent -> contact_quality/<metric>.
        ep_len = self.episode_length_buf.clamp_min(1).float()
        contacted = self.cq_starts > 0                                # made contact at all this episode
        nan = torch.full_like(ep_len, float("nan"))
        first_contact_step = self.t_contact / step_dt                # +inf where never contacted
        # length of the post-first-contact phase (first-contact step F to end, inclusive = E - F + 1);
        # >=1 to avoid /0 on a last-step touch. post_contact_% = contact steps / this phase length.
        post_denom = (ep_len - first_contact_step + 1.0).clamp_min(1.0)
        self.extras["per_env_contact_quality"] = {
            # Over ALL rollouts (a no-contact rollout is a meaningful 0 here):
            "contact_percentage": self.cq_contact.float() / ep_len,                       # steps in contact / episode length
            "made_contact_rate": contacted.float(),                                      # 1 if it touched at all, else 0
            # Conditional on having touched (NaN otherwise -> block_agent averages ONLY over rollouts
            # that made contact, so a no-contact rollout doesn't drag these toward 0):
            "avg_contact_length": torch.where(contacted, self.cq_contact.float() / self.cq_starts.clamp_min(1).float(), nan),  # mean consecutive contact run
            "contact_breaks": torch.where(contacted, self.cq_breaks.float(), nan),        # # of contact->no-contact bounces
            "steps_to_first_contact": torch.where(contacted, first_contact_step, nan),    # approach speed
            "post_contact_percentage": torch.where(contacted, self.cq_contact.float() / post_denom, nan),  # contact held after touchdown
        }
        self.extras["per_env_contact_quality_mask"] = self.reset_buf.clone()

        # --- Drag-performance per-episode publish ---
        # Two-level stats: per rollout compute the mean AND std of each metric over its IN-CONTACT
        # steps; block_agent then reduces over rollouts, emitting {metric}_mean (avg-of-rollout-means)
        # and {metric}_std (spread-of-rollout-means), plus {metric}_intra_std_mean (avg within-rollout
        # std). NaN where a rollout never contacted, so the stat reducer skips it. keypoints_achieved /
        # keypoints_passed are plain per-rollout counts (0 valid); block_agent also emits each as a
        # "(max)" frontier over the interval.
        dcnt = self.drag_count.float().clamp_min(1.0)
        dhas = self.drag_count > 0
        drag = {
            "keypoints_achieved": self.keypoints_achieved.float(),
            "keypoints_passed": self.keypoints_passed.float(),
        }
        for _n in self._drag_metrics:
            _m = getattr(self, f"drag_sum_{_n}") / dcnt
            _var = (getattr(self, f"drag_sumsq_{_n}") / dcnt - _m * _m).clamp_min(0.0)
            drag[_n] = torch.where(dhas, _m, nan)                        # per-rollout mean (in contact)
            drag[f"{_n}_intra_std"] = torch.where(dhas, _var.sqrt(), nan)  # per-rollout within-run std
        self.extras["per_env_drag"] = drag
        self.extras["per_env_drag_mask"] = self.reset_buf.clone()

        # Success-conditional per-episode stats (NaN for rollouts that did NOT succeed, so block_agent
        # averages ONLY over successful trajectories). Full tag names -> logged verbatim.
        succeeded = self.success_reward_given
        self.extras["per_env_episode_stat"] = {
            "Episode / Steps to success": torch.where(succeeded, self.success_step, nan),
            "Episode_Reward/success_time_on_success": torch.where(succeeded, self.success_time_earned, nan),
            # Fraction of FINISHING rollouts ending because of lag (0/1, so it averages over ALL of
            # them, not just successes) -> lag-termination rate.
            "Episode / Lag termination rate": self._term_lag.float(),
        }
        self.extras["per_env_episode_stat_mask"] = self.reset_buf.clone()

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------
    def _get_dones(self):
        """Per-env (terminated, truncated).

        truncated = episode time-out (SAC bootstraps its value). terminated = optional per-env
        FAILURE/SUCCESS conditions (no bootstrap), each gated by a task toggle (both default off):
          * terminate_on_lag: the tool fell more than ``pace_lag_frac * L`` behind the moving
            setpoint (``s_ref - progress``) while in contact — the core "can't keep the commanded
            pace" failure. Only fires after first contact (``t_contact`` finite); off-contact the
            setpoint is frozen so lag can't grow, which lets brief bounces recover.
          * terminate_on_success: success reached while in contact — end immediately (the one-shot
            success_time bonus has already been paid this step).
        When EITHER toggle is on the run MUST carry the efficient-reset wrapper (env_setup attaches
        it automatically for this task) so the resulting partial resets teleport to a cached donor
        state instead of running Factory's all-envs settling reset. Overrides FactoryEnv._get_dones
        (which returns all-synced time-outs); still refreshes intermediate values first, as it does.
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        cfg = self.cfg_task
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        self._term_lag = torch.zeros_like(time_out)   # this step's lag-termination flag (for the metric)
        if bool(cfg.terminate_on_lag):
            lag = self.s_ref - self.progress                          # (E,) m, positive = behind
            lag_max = float(cfg.pace_lag_frac) * self.path_length     # (E,) m
            self._term_lag = (lag > lag_max) & torch.isfinite(self.t_contact)
            terminated = terminated | self._term_lag
        if bool(cfg.terminate_on_success):
            terminated = terminated | (self._get_curr_successes() & self.in_contact_any)
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset: grandparent (Factory) reset + our placement, then Forge dynamics rand
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        # FactoryEnv._reset_idx: default poses + _set_franka_to_default_pose +
        # our randomize_initial_state (dispatched via self). Skips ForgeEnv._reset_idx,
        # which writes the success-prediction action dim (6) we don't have.
        FactoryEnv._reset_idx(self, env_ids)

        # Forge per-reset dynamics randomization (sans success-pred action writes).
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        # Per-episode desired normal force (N), sampled in [min, max]; observed + used by the
        # force-tracking reward. Constant for the episode (a per-env setpoint).
        force_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        self.desired_force = self.cfg_task.force_desired_min + force_rand * (
            self.cfg_task.force_desired_max - self.cfg_task.force_desired_min
        )

        # Reset the pace schedule clock (the moving along-track setpoint restarts at p0).
        self.pace_tau = torch.zeros((self.num_envs,), device=self.device)

        # Reset the time-to-success bonus state: first-contact clock (+inf = no contact yet) and the
        # one-shot paid flag, plus the success step / earned-reward latches.
        self.t_contact = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.success_reward_given = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.success_step = torch.zeros((self.num_envs,), device=self.device)
        self.success_time_earned = torch.zeros((self.num_envs,), device=self.device)

        # Reset the contact-quality accumulators (new rollout).
        self.cq_contact = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_starts = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_breaks = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.cq_prev = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Reset the keypoint + drag-performance accumulators (new rollout).
        self.keypoints_achieved = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.keypoints_passed = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.kp_ach_frontier = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.setpoint_kp_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # k0 until contact
        self.prev_progress = torch.zeros((self.num_envs,), device=self.device)
        self.prev_cross_track = torch.zeros((self.num_envs,), device=self.device)
        self.drag_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        for _n in self._drag_metrics:
            setattr(self, f"drag_sum_{_n}", torch.zeros((self.num_envs,), device=self.device))
            setattr(self, f"drag_sumsq_{_n}", torch.zeros((self.num_envs,), device=self.device))

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def randomize_initial_state(self, env_ids):
        """Place the plate at a random orientation and the cylinder with its TIP at a configurable
        pose relative to the STARTING KEYPOINT (NO force-controlled contact — we spawn the tip a
        configurable height above the surface and let the policy establish contact)."""
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        n = len(env_ids)

        # (1) Plate pose: center + in-plane noise; full yaw + small roll/pitch tilt cone.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        rs = torch.rand((n, 3), dtype=torch.float32, device=self.device)
        pos_rand = (2.0 * (rs - 0.5)) @ torch.diag(
            torch.tensor(self.cfg_task.plate_pos_noise, dtype=torch.float32, device=self.device)
        )
        center = torch.tensor(self.cfg_task.plate_center_pos, dtype=torch.float32, device=self.device)
        fixed_state[:, 0:3] = center + pos_rand + self.scene.env_origins[env_ids]

        yaw = torch.deg2rad(torch.tensor(self.cfg_task.plate_yaw_range_deg, device=self.device)) * torch.rand(
            (n,), device=self.device
        )
        tilt_rng = torch.deg2rad(torch.tensor(self.cfg_task.plate_tilt_range_deg, device=self.device))
        roll_pitch = (2.0 * (torch.rand((n, 2), device=self.device) - 0.5)) * tilt_rng
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
        fixed_state[:, 3:7] = fixed_orn_quat
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # Noisy fixed-asset position observation offset (held for the episode).
        fixed_pos_noise = torch.randn((n, 3), dtype=torch.float32, device=self.device) @ torch.diag(
            torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        )
        self.init_fixed_pos_obs_noise[env_ids] = fixed_pos_noise

        self.step_sim_no_action()

        # (2) Surface frame + endpoints (env-relative). All envs reset together.
        start, goal, normal, path_dir, cross_dir = self._surface_frame()
        # The Forge action frame + FT reference = near-edge center (start).
        self.fixed_pos_obs_frame[:] = start

        # (3) HELD-CYLINDER SPAWN + arm IK. Place the cylinder TIP directly at a configurable pose
        # relative to the STARTING KEYPOINT, then invert the rigid grip to get the fingertip target
        # and IK the arm to it. The plate geometry is used only to POSITION the reset pose (no
        # privileged info reaches the policy). The tip position is set first; the orientation offset
        # is applied ABOUT THE TIP, so it never moves the tip.
        #
        # Per-DOF sampling (see _sample_spawn_dof): std == 0 => mean used exactly (no sampling);
        # std > 0 => N(mean, std). Position offset (m) and orientation offset (deg->rad) are both in
        # the SURFACE frame: x = along-path, y = cross-track, z = surface normal (rpy about those).
        pos_off = self._sample_spawn_dof(
            self.cfg_task.spawn_tip_pos_mean, self.cfg_task.spawn_tip_pos_std, self.num_envs
        )                                                                     # (E,3) m, surface-local
        orn_off = torch.deg2rad(
            self._sample_spawn_dof(self.cfg_task.spawn_orn_mean_deg, self.cfg_task.spawn_orn_std_deg, self.num_envs)
        )                                                                     # (E,3) rad, surface-local rpy

        # Starting keypoint = k0 = the near-edge center (start). The peg spawns directly over k0 and
        # the step-0 observation setpoint is ALSO k0 (held there until first contact, see
        # _compute_intermediate_values), so the peg descends straight down onto k0, then the setpoint
        # advances one keypoint at a time as it drags.
        start_kp = start                                                     # (E,3) k0 (near-edge center)

        # Surface-frame rotation (world<-surface): columns [along-path, cross-track, normal]. Used only
        # to map the POSITION offset, which is surface-local so that "z above the plate" = along the
        # normal (perpendicular distance) and x/y = along/across the path.
        R_surf = torch.stack([path_dir, cross_dir, normal], dim=-1)          # (E,3,3)
        q_surf = quat_from_matrix(R_surf)                                     # (E,4)

        # Desired TIP position: starting keypoint + surface-local offset rotated into world.
        p_tip = start_kp + quat_apply(q_surf, pos_off)                        # (E,3)
        # Desired cylinder orientation: WORLD-frame roll/pitch/yaw. The zero-offset pose is the peg
        # straight up/down (axis along world +z, tip down) — INDEPENDENT of the surface orientation,
        # which the policy does not know a priori. cyl_axis points from the tip toward the grip.
        held_quat_des = torch_utils.quat_from_euler_xyz(orn_off[:, 0], orn_off[:, 1], orn_off[:, 2])  # (E,4) world<-cyl
        z_hat = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        cyl_axis = quat_apply(held_quat_des, z_hat)                          # (E,3)

        # Fingertip target = the exact inverse of the seating in step (4):
        #   held = (fingertip ∘ flip_z) ∘ inverse(held_rel)   [step (4)]
        #   =>  fingertip = held ∘ held_rel ∘ flip_z
        # where held = the desired cylinder BODY pose (center at tip + (H/2)·cyl_axis, orientation
        # held_quat_des) and held_rel = get_handheld_asset_relative_pose() (which now carries the
        # fixed grasp TILT). Because held_rel absorbs the grasp tilt, the IK gives the eef/wrist a
        # DIFFERENT target while the cylinder still seats at held_quat_des (perpendicular to the
        # surface for the zero-offset pose) — the reset auto-compensates for the grasp tilt. For a
        # zero tilt (identity held_rel) this reduces EXACTLY to the old
        # (held_quat_des ∘ flip_z, p_tip + (H - fingerpad)·cyl_axis).
        H = self.cfg_task.held_asset_cfg.height
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        zeros3 = torch.zeros((self.num_envs, 3), device=self.device)
        held_rel_pos, held_rel_quat = self.get_handheld_asset_relative_pose()  # (pos, quat) — tilt-aware

        # Optionally set the grasp's free roll so the EEF x-axis is as parallel as possible to the
        # travel direction path_dir. CONTROL happens in the EEF frame, so we align the EEF x (not the
        # peg-tip x). The cylinder is axisymmetric, so its spin about its own axis is a free DOF: we
        # spin held_quat_des about the peg axis (cyl_axis), which leaves the peg pose (tip position +
        # tilt) unchanged and rotates the derived EEF orientation (held o held_rel o flip_z) by the
        # same world rotation. Picking the spin that lands the EEF x's component ⊥ the peg axis on
        # path_dir maximizes EEF_x · path_dir. With ZERO grasp tilt the EEF x is ⊥ the peg axis, so it
        # aligns exactly (and equals the peg-tip x); as the grasp PITCH grows the EEF x tilts off that
        # plane, so the best-aligned EEF x is only as parallel as the pitch allows (residual ≈ the
        # grasp pitch). cyl_axis is invariant under this spin, so held_center_pos below is unchanged.
        if self.cfg_task.spawn_align_eef_x_to_path:
            axis = cyl_axis / torch.linalg.norm(cyl_axis, dim=-1, keepdim=True).clamp_min(1e-8)
            x_hat = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            eef_quat0 = torch_utils.quat_mul(                                 # provisional EEF orientation
                torch_utils.quat_mul(held_quat_des, held_rel_quat), flip_z_quat)
            eef_x = quat_apply(eef_quat0, x_hat)                             # (E,3) EEF x-axis (world)
            u = eef_x - (eef_x * axis).sum(-1, keepdim=True) * axis          # EEF x   projected ⊥ axis
            w = path_dir - (path_dir * axis).sum(-1, keepdim=True) * axis    # path_dir projected ⊥ axis
            u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-8)
            w = w / torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
            cos = (u * w).sum(-1).clamp(-1.0, 1.0)                           # (E,)
            sin = (torch.cross(u, w, dim=-1) * axis).sum(-1)                 # (E,) signed about axis
            roll = torch.atan2(sin, cos)                                     # (E,) eef roll to apply
            held_quat_des = torch_utils.quat_mul(torch_utils.quat_from_angle_axis(roll, axis), held_quat_des)

        held_center_pos = p_tip + 0.5 * H * cyl_axis                          # (E,3) cylinder body origin
        _q1, _t1 = torch_utils.tf_combine(held_quat_des, held_center_pos, held_rel_quat, held_rel_pos)
        target_quat_all, target_pos_all = torch_utils.tf_combine(_q1, _t1, flip_z_quat, zeros3)

        # IK the arm to the FIXED fingertip target; reseed the arm for any env that didn't converge
        # and retry. Targets are NOT re-sampled on retry, so the spawn distribution stays unbiased.
        bad_envs = env_ids.clone()
        while True:
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=target_pos_all,
                ctrl_target_fingertip_midpoint_quat=target_quat_all,
                env_ids=bad_envs,
            )
            pos_bad = torch.linalg.norm(pos_error, dim=1) > 1e-3
            rot_bad = torch.norm(aa_error, dim=1) > 1e-3
            any_bad = torch.logical_or(pos_bad, rot_bad)
            bad_envs = bad_envs[any_bad.nonzero(as_tuple=False).squeeze(-1)]
            if bad_envs.shape[0] == 0:
                break
            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

        self.step_sim_no_action()

        # (4) Place the cylinder in the gripper (mirrors FactoryEnv.randomize_initial_state).
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )
        held_rel_pos, held_rel_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(held_rel_quat, held_rel_pos)
        held_quat, held_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )
        # No extra in-grip position jitter: the tip pose was placed explicitly in step (3) (any
        # desired jitter is expressed there via spawn_tip_pos_std), and the grip is rigid + aligned.

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = held_pos + self.scene.env_origins
        held_state[:, 3:7] = held_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # Close the gripper with quick-reset gains.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        # Operational gains for the press-to-contact step below (so the contact-force scale during
        # the press matches the runtime controller, not the stiff quick-reset grasp gains).
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        # (5) PRESS TO CONTACT. Gravity is still OFF (only the contact reaction acts), and this runs
        # in the full-reset (all-envs) path, so the efficient-reset wrapper's cached donor state
        # captures the in-contact pose and per-env teleport resets reproduce it. See the method.
        self._press_held_to_contact(normal)

        # Bookkeeping AFTER the press so the first step's finite-difference velocities start at ~0
        # from the FINAL (pressed) pose, not the pre-descent one.
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    # ------------------------------------------------------------------
    # Reset press-to-contact
    # ------------------------------------------------------------------
    def _press_held_to_contact(self, surface_normal):
        """Press the just-gripped held object toward the surface until each env reads in contact.

        Physical (controller-driven) press: the gripper stays closed and a CUMULATIVE fingertip
        position target steps down ``reset_press_step`` along ``-surface_normal`` each settle step,
        so the tracking error (hence press force) BUILDS until real contact registers — a constant
        offset can only ever build ``Kp * step`` and stalls a fraction of a mm above the plate. The
        target is clamped to lead the current fingertip by at most ``reset_press_max_lead``, capping
        the steady press force at ~``Kp * max_lead`` (no spike at touchdown; no blow-up if the
        surface is absent). The orientation target is held fixed (pure translation press).

        Each env is FROZEN the step it first reads in contact, latched at its DESCENDING press target
        (which leads the fingertip into the surface) rather than the current fingertip: the fingertip
        pose at first-contact is a dynamic overshoot whose static equilibrium sits just off the
        surface, so holding it would let the peg relax back out of contact. Still-moving envs keep
        descending; the loop ends when all are in contact or ``reset_press_max_dist`` is exhausted.

        Runs during reset with gravity OFF, on ALL envs (the full-reset path), so nothing falls and
        the whole in-contact configuration is caught by the efficient-reset donor cache (per-env
        teleport resets then reproduce it). Contact is read via :meth:`_reset_contact_mask`.
        """
        cfg = self.cfg_task
        if not bool(getattr(cfg, "reset_press_to_contact", True)):
            return
        step_len = float(cfg.reset_press_step)
        max_dist = float(cfg.reset_press_max_dist)
        max_lead = float(cfg.reset_press_max_lead)
        if step_len <= 0.0 or max_dist <= 0.0:
            return
        n_steps = int(max_dist / step_len) + 1

        # Unit descent direction (into the surface), per env.
        down = -surface_normal / torch.linalg.norm(surface_normal, dim=1, keepdim=True).clamp_min(1e-8)

        # CUMULATIVE position target, seeded at the current fingertip. It steps DOWN each iteration so
        # the tracking error (hence press force) can grow past a single step — a constant offset can
        # only ever build ``Kp * step_len`` and stalls before real contact. The orientation target is
        # held fixed (pure translation press). Latched targets freeze a settled env in place.
        press_pos = self.fingertip_midpoint_pos.clone()
        press_quat = self.fingertip_midpoint_quat.clone()
        latched_pos = self.fingertip_midpoint_pos.clone()
        latched_quat = self.fingertip_midpoint_quat.clone()
        settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for _ in range(n_steps):
            moving = ~settled
            if not bool(moving.any()):
                break
            # Advance the cumulative target down for moving envs, then CLAMP it so it never leads the
            # current fingertip by more than ``max_lead`` along ``down``: this caps the steady press
            # force at ~``Kp * max_lead`` (no spike when the tip finally touches; no blow-up if the
            # surface is somehow absent) while still building enough force to read as in contact.
            press_pos[moving] = press_pos[moving] + step_len * down[moving]
            lead = ((press_pos - self.fingertip_midpoint_pos) * down).sum(-1, keepdim=True)  # +down = below tip
            press_pos = press_pos - (lead - max_lead).clamp_min(0.0) * down
            m = moving.unsqueeze(-1)
            tgt_pos = torch.where(m, press_pos, latched_pos)
            tgt_quat = torch.where(m, press_quat, latched_quat)
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=tgt_pos,
                ctrl_target_fingertip_midpoint_quat=tgt_quat,
                ctrl_target_gripper_dof_pos=0.0,  # keep the gripper closed
            )
            self.step_sim_no_action()

            newly = self._reset_contact_mask() & ~settled
            if bool(newly.any()):
                # Latch the hold target at the DESCENDING press target (which leads the fingertip
                # into the surface), NOT the current fingertip: the fingertip pose at first-contact
                # is a dynamic overshoot whose static equilibrium sits just OFF the surface, so
                # holding it would let the peg relax out of contact. Holding press_pos maintains the
                # steady ~contact-threshold press through the hold and into the cached state.
                latched_pos[newly] = press_pos[newly]
                latched_quat[newly] = press_quat[newly]
                settled |= newly

        # Safety: any env that never registered contact within the max-descent cap (e.g. a spawn
        # configured higher than reset_press_max_dist) latches at its current press target, so the
        # hold phase below does not yank it back up to the pre-press seed pose.
        if not bool(settled.all()):
            stuck = ~settled
            latched_pos[stuck] = press_pos[stuck]
            latched_quat[stuck] = press_quat[stuck]

        # Hold everything at its latched pose for a few steps so the last-settled envs damp to ~0
        # velocity before the state is cached / gravity is restored.
        for _ in range(3):
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=latched_pos,
                ctrl_target_fingertip_midpoint_quat=latched_quat,
                ctrl_target_gripper_dof_pos=0.0,
            )
            self.step_sim_no_action()

    def _reset_contact_mask(self):
        """Per-env in-contact bool that stops the reset press (freshly refreshed).

        When the runtime per-axis contact sensor is live this returns exactly its reading
        (``env.in_contact.any(dim=1)``) — the SAME "in contact in any direction" the policy sees at
        runtime — refreshed here via the ContactSensorWrapper's ``_refresh_in_contact`` hook (the
        wrapper's own ``step()`` never runs during a reset). The press then builds force until the
        sensor genuinely registers, so episodes start reading in contact on that sensor rather than
        merely touching. A geometric "tip reached the surface" check is deliberately NOT OR-ed in
        here: on a rigid plate the tip cannot penetrate, so it would fire at a feather-touch and
        latch the press before the sensor reads contact.

        When no live sensor exists (contact wrapper disabled, or not yet initialized on the very
        first reset), it falls back to the measured normal force OR a geometric tip-at-surface
        backstop — there is no sensor threshold to build toward, so a light arrival is the signal.
        """
        cfg = self.cfg_task
        refresh = getattr(self, "_refresh_in_contact", None)
        sensor_live = bool(refresh()) if callable(refresh) else False
        cw = getattr(self, "in_contact", None)
        if sensor_live and torch.is_tensor(cw):
            return cw.any(dim=1)
        force = self.measured_normal_force.abs() > float(cfg.reset_contact_force_threshold)
        geom = self.tip_surface_dist <= float(cfg.reset_press_contact_depth)
        return force | geom
