"""Surface path-following environment (subclass of ``ForgeEnv``).

Reuses Forge/Factory machinery (robot control targets, force sensing, IK reset
placement, per-env logging) and overrides only what the new geometry needs:
scene assets (plate + cylinder), reset placement (cylinder spawned just above the
near-edge center of a randomly-oriented plate), observations, success, and a
(currently stubbed) reward. The control/logging wrappers attach unchanged.

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
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_from_matrix

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
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Small geometry helpers
    # ------------------------------------------------------------------
    def _identity_quat(self, n=None):
        n = self.num_envs if n is None else n
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(n, 1)

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
        rel_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # IMPORTANT: the procedural CylinderCfg's prim origin is at the cylinder's
        # CENTER (it spans [-H/2, +H/2] along its local z), unlike the Factory peg
        # USD whose origin is at the base/tip. So grip at (H/2 - fingerpad) from the
        # center => fingerpad below the TOP end, with the cylinder hanging down and its
        # lower end as the contact tip. (Using the full H here mis-grips by H/2.)
        rel_pos[:, 2] = self.cfg_task.held_asset_cfg.height / 2.0
        rel_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length

        rel_quat = self._identity_quat()
        tilt = self.cfg_task.inhand_tilt_range_deg
        if any(abs(float(t)) > 0.0 for t in tilt):
            rand = 2.0 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)  # [-1, 1]
            tilt_rad = torch.deg2rad(torch.tensor(tilt, dtype=torch.float32, device=self.device))
            d = rand @ torch.diag(tilt_rad)
            perturb = torch_utils.quat_from_euler_xyz(d[:, 0], d[:, 1], d[:, 2])
            rel_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(perturb), rel_quat)
        return rel_pos, rel_quat

    # ------------------------------------------------------------------
    # Intermediate values: stash task geometry after the Forge base compute
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)  # ForgeEnv: noise + FT sensing

        start, goal, normal, path_dir, cross_dir = self._surface_frame()
        self.start_world = start
        self.goal_world = goal
        self.surface_normal = normal
        self.path_dir = path_dir
        self.cross_dir = cross_dir

        # The procedural cylinder's origin is its CENTER (held_pos), so its two flat
        # ends are held_pos +/- (H/2)*cyl_axis. The CONTACT tip is the lower end
        # (smaller projection onto the surface normal) — computed sign-robustly so we
        # don't depend on which way the grasp leaves the held-frame +z pointing.
        self.cyl_axis = self._rotate_vec(self.held_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        half = 0.5 * self.cfg_task.held_asset_cfg.height
        end_plus = self.held_pos + half * self.cyl_axis
        end_minus = self.held_pos - half * self.cyl_axis
        proj_plus = (end_plus * normal).sum(-1, keepdim=True)
        proj_minus = (end_minus * normal).sum(-1, keepdim=True)
        self.cyl_tip = torch.where(proj_plus < proj_minus, end_plus, end_minus)

        rel = self.cyl_tip - start
        self.progress = (rel * path_dir).sum(-1)
        along = self.progress.unsqueeze(-1) * path_dir
        normal_comp = (rel * normal).sum(-1, keepdim=True) * normal
        self.cross_track = torch.linalg.norm(rel - along - normal_comp, dim=-1)
        self.v_along = (self.ee_linvel_fd * path_dir).sum(-1)

        # Measured normal force = projection of the FT force onto the true surface normal.
        # The raw smoothed force (force_sensor_world_smooth) is in the force_sensor child-joint
        # frame, NOT world — so rotate it to world (by the sensor body's world quaternion) BEFORE
        # projecting onto the world-frame normal. Dotting the sensor-frame force against the world
        # normal (the old code) is only correct when the tool is axis-aligned with world; it breaks
        # as soon as the plate/tool tilts. Positive => pressing into the surface (flip if needed).
        sensor_quat = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        force_world = quat_apply(sensor_quat, self.force_sensor_world_smooth[:, 0:3])
        self.measured_normal_force = (force_world * normal).sum(-1)

        # Orientation error. The cylinder is axisymmetric (no preferred +/- along its
        # axis), so use the LINE angle between the cylinder axis and the surface normal:
        # |cyl_axis . normal| = cos(tilt-from-normal) (1 => perpendicular/tip-down,
        # 0 => lying flat). Compare against the commanded angle. Sign-robust.
        axis_from_normal = torch.arccos((self.cyl_axis * normal).sum(-1).abs().clamp(0.0, 1.0))
        commanded = float(np.deg2rad(self.cfg_task.commanded_axis_angle_deg))
        self.orn_error = (axis_from_normal - commanded).abs()

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

        # Orientation: z = surface normal; x = tangential motion direction (else toward goal,
        # else the path direction); y = z × x. Orthonormalized against the normal.
        vel_t = self.ee_linvel_fd - (self.ee_linvel_fd * normal).sum(-1, keepdim=True) * normal
        goal_dir = self.goal_world - self.interaction_pos
        goal_t = goal_dir - (goal_dir * normal).sum(-1, keepdim=True) * normal
        x_dir = torch.where(torch.linalg.norm(vel_t, dim=-1, keepdim=True) > 1e-4, vel_t, goal_t)
        x_dir = x_dir - (x_dir * normal).sum(-1, keepdim=True) * normal
        x_norm = torch.linalg.norm(x_dir, dim=-1, keepdim=True)
        x_axis = torch.where(x_norm > 1e-6, x_dir / x_norm.clamp_min(1e-6), path_dir)
        y_axis = torch.cross(normal, x_axis, dim=-1)
        self.interaction_quat = quat_from_matrix(torch.stack([x_axis, y_axis, normal], dim=2))
        # Exists only while in contact (per the contact-sensor wrapper's in_contact state).
        in_contact = getattr(self, "in_contact", None)
        self.interaction_exists = (
            in_contact.any(dim=1)
            if in_contact is not None
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

        # Debug series (guarded — extras may not yet have a to_log dict).
        if hasattr(self, "extras"):
            self.extras.setdefault("to_log", {})["Force / Normal Measured"] = self.measured_normal_force.detach()

    # ------------------------------------------------------------------
    # Observations (all EEF-frame): goal-relative pose (sign-aligned) + EEF vel/force
    # ------------------------------------------------------------------
    def _fingertip_wrench(self):
        """Clean 6-D wrench (F, T) re-expressed in the fingertip-midpoint frame.

        Same sensor->fingertip reframe the control wrapper uses (``change_FT_frame`` with
        the live sensor + fingertip body poses): rotate F and T into the fingertip frame and
        re-reference the torque to the fingertip origin. So the observed force/torque is in
        the same frame the policy acts/controls in.
        """
        raw = self.force_sensor_world_smooth
        bq = self._robot.data.body_quat_w
        bp = self._robot.data.body_pos_w
        sframe = (bq[:, self.force_sensor_body_idx], bp[:, self.force_sensor_body_idx])
        fframe = (bq[:, self.fingertip_body_idx], bp[:, self.fingertip_body_idx])
        return forge_utils.change_FT_frame(raw[:, 0:3], raw[:, 3:6], sframe, fframe)

    def _goal_frame_quat(self):
        """Goal-frame orientation = surface frame at the far edge (z = normal, x = path)."""
        z = self.surface_normal
        x = self.path_dir
        y = torch.cross(z, x, dim=-1)
        return quat_from_matrix(torch.stack([x, y, z], dim=2))

    def _get_observations(self):
        goal_quat = self._goal_frame_quat()
        goal_pos = self.goal_world

        def _pose_rel(eef_pos, eef_quat):
            # Goal pose relative to the goal frame, expressed in the EEF frame, sign-aligned:
            # a positive component => a positive action on that EEF axis moves toward the goal.
            conj = torch_utils.quat_conjugate(eef_quat)
            pos_rel = quat_apply(conj, goal_pos - eef_pos)                          # R_eefᵀ (goal - eef)
            rot_rel = axis_angle_from_quat(torch_utils.quat_mul(conj, goal_quat))   # eef -> goal in eef
            return pos_rel, rot_rel

        def _to_eef(vec, eef_quat):
            return quat_apply(torch_utils.quat_conjugate(eef_quat), vec)

        F_ft, T_ft = self._fingertip_wrench()  # clean fingertip-frame wrench
        force_noise = torch.randn((self.num_envs, 3), device=self.device) * float(self.cfg.obs_rand.ft_force)

        ts = torch.full((self.num_envs, 1), float(self.cfg_task.target_speed), device=self.device)
        tnf = torch.full((self.num_envs, 1), float(self.cfg_task.target_normal_force), device=self.device)
        prev_actions = self.actions.clone()

        # POLICY — noisy fingertip estimate (positions/orientation carry the Forge obs noise).
        p_pos_rel, p_rot_rel = _pose_rel(self.noisy_fingertip_pos, self.noisy_fingertip_quat)
        obs_dict = {
            "goal_pos_rel": p_pos_rel,
            "goal_rot_rel": p_rot_rel,
            "ee_linvel": _to_eef(self.fingertip_midpoint_linvel, self.noisy_fingertip_quat),
            "ee_angvel": _to_eef(self.fingertip_midpoint_angvel, self.noisy_fingertip_quat),
            "ft_force": F_ft + force_noise,
            "ft_torque_eef": T_ft + force_noise,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "target_speed": ts,
            "target_normal_force": tnf,
            "prev_actions": prev_actions,
        }
        # CRITIC — clean + privileged geometry.
        c_pos_rel, c_rot_rel = _pose_rel(self.fingertip_midpoint_pos, self.fingertip_midpoint_quat)
        state_dict = {
            "goal_pos_rel": c_pos_rel,
            "goal_rot_rel": c_rot_rel,
            "ee_linvel": _to_eef(self.fingertip_midpoint_linvel, self.fingertip_midpoint_quat),
            "ee_angvel": _to_eef(self.fingertip_midpoint_angvel, self.fingertip_midpoint_quat),
            "ft_force": F_ft,
            "ft_torque_eef": T_ft,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "ema_factor": self.ema_factor,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "surface_normal": self.surface_normal,
            "path_dir": self.path_dir,
            "progress": self.progress[:, None],
            "cross_track": self.cross_track[:, None],
            "orn_error": self.orn_error[:, None],
            "normal_force": self.measured_normal_force[:, None],
            "target_speed": ts,
            "target_normal_force": tnf,
            "prev_actions": prev_actions,
        }

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    # ------------------------------------------------------------------
    # Success + reward (reward is a STRUCTURAL STUB — terms come in a later pass)
    # ------------------------------------------------------------------
    def _get_curr_successes(self, success_threshold=None, check_rot=False):
        at_goal = torch.linalg.norm(self.cyl_tip - self.goal_world, dim=-1) < self.cfg_task.success_pos_tol
        oriented = self.orn_error < float(np.deg2rad(self.cfg_task.success_orn_tol_deg))
        return torch.logical_and(at_goal, oriented)

    def _get_rewards(self):
        curr_successes = self._get_curr_successes()
        # TODO(reward pass): populate rew_dict with progress / goal_kp / cross_track /
        # speed / normal_force / orientation / action penalties (all (E,) tensors,
        # bounded via factory_utils.squashing_fn). The scorer reads the matching
        # scales from FlatSurfaceFollowWrapper._factory_scales().
        rew_dict: dict[str, torch.Tensor] = {}
        rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

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

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

    def randomize_initial_state(self, env_ids):
        """Place the plate at a random orientation and the cylinder just above the
        near-edge center (NO force-controlled contact — we spawn >= start_standoff
        above the surface and let the policy establish contact)."""
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

        # (3) IK the hand to (near-edge center + standoff along the normal), pointing
        # into the plate, with surface-local hand-init randomization. Retry per-env.
        # Fingertip-above-surface offset: with the center-origin grasp (origin placed
        # (H/2 - fingerpad) below the fingertip), the lower contact tip sits a further
        # H/2 below the origin, i.e. (H - fingerpad) below the fingertip. So putting the
        # fingertip (H - fingerpad + standoff) above the surface lands the tip exactly
        # start_standoff above it. (Do NOT change this to H/2 — the grasp offset already
        # carries the center-origin correction.)
        offset = (
            self.cfg_task.held_asset_cfg.height
            - self.cfg_task.robot_cfg.franka_fingerpad_length
            + self.cfg_task.start_standoff
        )
        target_pos_all = start + normal * offset
        target_quat_all = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        bad_envs = env_ids.clone()
        while True:
            n_bad = bad_envs.shape[0]

            # Surface-local hand-position noise: x along path, y across, z along normal.
            rs = 2.0 * (torch.rand((n_bad, 3), device=self.device) - 0.5)
            off_local = rs @ torch.diag(
                torch.tensor(self.cfg_task.start_pos_noise, dtype=torch.float32, device=self.device)
            )
            pos_noise_world = (
                off_local[:, 0:1] * path_dir[bad_envs]
                + off_local[:, 1:2] * cross_dir[bad_envs]
                + off_local[:, 2:3] * normal[bad_envs]
            )
            target_pos = target_pos_all.clone()
            target_pos[bad_envs] = target_pos[bad_envs] + pos_noise_world

            # Orientation: hand-down in the plate frame (roll=pi -> gripper points along
            # -normal), plus the commanded axis tilt (pitch) and free yaw noise.
            local_euler = torch.zeros((n_bad, 3), device=self.device)
            local_euler[:, 0] = np.pi
            local_euler[:, 1] = float(np.deg2rad(self.cfg_task.commanded_axis_angle_deg))
            yaw_noise = (2.0 * (torch.rand((n_bad,), device=self.device) - 0.5)) * float(
                self.cfg_task.hand_init_orn_noise[2]
            )
            local_euler[:, 2] = yaw_noise
            local_quat = torch_utils.quat_from_euler_xyz(local_euler[:, 0], local_euler[:, 1], local_euler[:, 2])
            target_quat_all[bad_envs] = torch_utils.quat_mul(self.fixed_quat[bad_envs], local_quat)

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=target_pos,
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

        rs = 2.0 * (torch.rand((self.num_envs, 3), device=self.device) - 0.5)
        held_pos_noise = rs @ torch.diag(
            torch.tensor(self.cfg_task.held_asset_pos_noise, dtype=torch.float32, device=self.device)
        )
        held_quat, held_pos = torch_utils.tf_combine(
            q1=held_quat, t1=held_pos, q2=self._identity_quat(), t2=held_pos_noise
        )

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

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
