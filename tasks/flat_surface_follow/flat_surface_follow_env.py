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
from isaaclab.utils.math import quat_apply, quat_from_matrix

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
        # Per-episode desired normal force (N), sampled in _reset_idx, observed by the policy.
        self.desired_force = torch.zeros((self.num_envs,), device=self.device)
        # Pace schedule clock (s): advances by the env step dt only while in contact; drives the
        # moving along-track setpoint s_ref. Reset to 0 each episode.
        self.pace_tau = torch.zeros((self.num_envs,), device=self.device)
        # Moving setpoint (recomputed each _compute); init for safety before the first reset.
        self.s_ref = torch.zeros((self.num_envs,), device=self.device)
        self.setpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Time-to-success bonus state. t_contact: episode time (s) of FIRST contact (+inf until
        # contact); success_reward_given: whether the one-shot bonus has already been paid this
        # episode. Both reset each episode in _reset_idx.
        self.t_contact = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.success_reward_given = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Per-episode tensors the efficient-reset wrapper must carry across per-env teleport resets.
        # Their fresh-episode values are captured in the post-full-reset cache (pace_tau=0,
        # t_contact=+inf, success_reward_given=False, desired_force=sampled), so a donor copy on a
        # partial reset restores correct fresh values without re-running randomize_initial_state.
        self._efficient_reset_extra_attrs = (
            "desired_force",
            "pace_tau",
            "t_contact",
            "success_reward_given",
        )

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

        # Ideal straight path p0 -> p_g on the plate top surface. path_dir is d (unit start->goal).
        rel = self.cyl_tip - start                                            # dp = tip - p0
        self.path_length = torch.linalg.norm(goal - start, dim=-1)            # L = |p_g - p0|
        self.d_lat = torch.cross(normal, path_dir, dim=-1)                    # d_lat = n x d (in-plane lateral)
        self.progress = (rel * path_dir).sum(-1)                             # s = dp . d (along-track)
        self.cross_track = (rel * self.d_lat).sum(-1)                        # e_perp = dp . d_lat (signed)
        self.v_along = (self.ee_linvel_fd * path_dir).sum(-1)

        # Moving along-track setpoint s_ref = clamp(v*tau, 0, L) (tau = pace clock, advanced in
        # _get_rewards). The setpoint POINT is at arc-length s_ref along the path on the surface;
        # the policy/critic track this instead of the final goal, so the same interface generalizes
        # to non-straight paths.
        v_des = float(self.cfg_task.desired_speed_cm_s) / 100.0              # cm/s -> m/s
        self.s_ref = (v_des * self.pace_tau).clamp_min(0.0).minimum(self.path_length)
        self.setpoint_pos = start + self.s_ref.unsqueeze(-1) * path_dir

        # Measured normal force = projection of the FT force onto the true surface normal.
        # The raw smoothed force (force_sensor_world_smooth) is in the force_sensor child-joint
        # frame, NOT world — so rotate it to world (by the sensor body's world quaternion) BEFORE
        # projecting onto the world-frame normal. Dotting the sensor-frame force against the world
        # normal (the old code) is only correct when the tool is axis-aligned with world; it breaks
        # as soon as the plate/tool tilts. Positive => pressing into the surface (flip if needed).
        sensor_quat = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        force_world = quat_apply(sensor_quat, self.force_sensor_world_smooth[:, 0:3])
        self.measured_normal_force = (force_world * normal).sum(-1)

        # Orientation: angle of the held object's z-axis to the surface PLANE
        # (asin(|axis . normal|); 90deg = perpendicular/tip-down, 0deg = lying in the plane). The
        # cylinder is axisymmetric, so only this plane angle is constrained (free about the normal).
        # REALIZED from the physics held orientation. orn_error = desired - actual (signed, degrees)
        # — the value the orientation reward squashes (and |orn_error| gates success).
        self.held_angle_to_plane = torch.rad2deg(
            torch.arcsin((self.cyl_axis * normal).sum(-1).abs().clamp(0.0, 1.0))
        )
        self.orn_error = float(self.cfg_task.orientation_desired_angle_deg) - self.held_angle_to_plane

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
        # In-contact bool (drives the interaction frame + the pace schedule clock). Prefer the
        # contact-sensor wrapper's per-axis state; fall back to a small normal-force threshold when
        # the contact sensor is disabled, so the env stays self-contained.
        cw = getattr(self, "in_contact", None)
        if torch.is_tensor(cw):
            self.in_contact_any = cw.any(dim=1)
        else:
            self.in_contact_any = self.measured_normal_force.abs() > 0.1
        self.interaction_exists = self.in_contact_any

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

    def _get_observations(self):
        setpoint_pos = self.setpoint_pos

        def _setpoint_pos_rel(eef_pos, eef_quat):
            # MOVING-SETPOINT position relative to the tool, in the EEF frame, sign-aligned: a
            # positive component => a positive action on that EEF axis moves toward the setpoint.
            # No orientation setpoint — orientation is inferred elsewhere (force/torque + reward).
            return quat_apply(torch_utils.quat_conjugate(eef_quat), setpoint_pos - eef_pos)

        def _to_eef(vec, eef_quat):
            return quat_apply(torch_utils.quat_conjugate(eef_quat), vec)

        F_ft, T_ft = self._fingertip_wrench()  # clean fingertip-frame wrench
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
        # CRITIC — clean + privileged geometry.
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
        oriented = self.orn_error.abs() < float(self.cfg_task.success_orn_tol_deg)  # orn_error in degrees
        return torch.logical_and(at_goal, oriented)

    def _get_rewards(self):
        """Reward = bounded task terms + the Factory action penalties.

        Task terms (force, orientation, straightness, pace) are weight * squashing_fn(raw signed
        REALIZED value, a, b) — computed from measured/physics state only (never control targets),
        peaking at value=0. The two action penalties (action_penalty_ee, action_grad_penalty) are
        the FactoryEnv linear penalties applied with NEGATIVE scales, both default 0.0 (off). A
        one-shot success_time bonus squashes (ideal completion time - actual success time). The
        surface-relative terms (orientation, straightness, pace, success_time) are GATED on contact
        with the held object — they pay 0 with no contact; force and the action penalties are not
        gated. The scorer's _factory_scales must stay in sync with the weights/scales below.
        """
        curr_successes = self._get_curr_successes()
        cfg = self.cfg_task

        # Force tracking: desired (sampled, along the surface normal) - measured normal force
        # (world-rotated EEF force projected onto the surface normal, for a same-frame difference).
        force_value = self.desired_force - self.measured_normal_force          # (E,) N, signed
        # Orientation: desired plane angle - realized plane angle (self.orn_error, deg, signed).
        orn_value = self.orn_error                                            # (E,) deg, signed
        # Straightness: signed cross-track error e_perp = dp . d_lat (computed in _compute).
        straightness_value = self.cross_track                                # (E,) m, signed
        # Pace: along-track error vs the moving setpoint s_ref (computed in _compute from the pace
        # clock). The clock advances by one env step only while in contact (updated below).
        pace_value = self.progress - self.s_ref                             # (E,) m, signed
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
        success_time_reward = torch.where(
            success_now,
            factory_utils.squashing_fn(success_time_value, cfg.success_time_a, cfg.success_time_b),
            torch.zeros_like(t_now),
        )
        self.success_reward_given = self.success_reward_given | success_now

        # Contact gate: with NO contact on the held object the surface-relative signals are
        # meaningless — and would otherwise be farmable by hovering at the start (s_ref frozen at 0,
        # progress/cross_track ~0) — so orientation / straightness / pace pay 0 off-contact. Force
        # stays active off-contact (it drives the tool INTO the surface) and the action penalties
        # always apply. success_time is already gated via success_now (requires in_contact_any).
        contact = self.in_contact_any.float()
        rew_dict = {
            "force": factory_utils.squashing_fn(force_value, cfg.force_a, cfg.force_b),
            "orientation": factory_utils.squashing_fn(orn_value, cfg.orientation_a, cfg.orientation_b) * contact,
            "straightness": factory_utils.squashing_fn(straightness_value, cfg.straightness_a, cfg.straightness_b) * contact,
            "pace": factory_utils.squashing_fn(pace_value, cfg.pace_a, cfg.pace_b) * contact,
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "success_time": success_time_reward,
        }
        rew_scales = {
            "force": float(cfg.force_weight),
            "orientation": float(cfg.orientation_weight),
            "straightness": float(cfg.straightness_weight),
            "pace": float(cfg.pace_weight),
            "action_penalty_ee": -float(cfg.action_penalty_ee_scale),
            "action_grad_penalty": -float(cfg.action_grad_penalty_scale),
            "success_time": float(cfg.success_time_weight),
        }
        rew_buf = torch.zeros(self.num_envs, device=self.device)
        for name in rew_dict:
            rew_buf = rew_buf + rew_dict[name] * rew_scales[name]

        # Advance the pace clock by one env step, ONLY where in contact (tau_{t+1} = tau_t + dt*in_contact).
        self.pace_tau = self.pace_tau + step_dt * self.in_contact_any.float()

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
        if bool(cfg.terminate_on_lag):
            lag = self.s_ref - self.progress                          # (E,) m, positive = behind
            lag_max = float(cfg.pace_lag_frac) * self.path_length     # (E,) m
            terminated = terminated | ((lag > lag_max) & torch.isfinite(self.t_contact))
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
        # one-shot paid flag.
        self.t_contact = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.success_reward_given = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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
            # Hand pitch from straight-down = the tool axis's angle to the surface NORMAL =
            # 90 - (desired angle to the surface PLANE), so the cylinder spawns at the desired tilt.
            local_euler[:, 1] = float(np.deg2rad(90.0 - self.cfg_task.orientation_desired_angle_deg))
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
