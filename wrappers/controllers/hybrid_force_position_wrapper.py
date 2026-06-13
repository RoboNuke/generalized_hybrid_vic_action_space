"""
Hybrid Force-Position Control Wrapper

Hybrid force/position control for Isaac Lab Forge/Factory envs. A per-axis selection
matrix decides, each control step, whether an axis is force-controlled or
position-controlled. Which axes are *eligible* for force control is fixed by a 6-D
binary ``force_axes`` config vector (1 = eligible, 0 = pose-only) — this replaces the
old ``ctrl_mode`` enum.

Control law (per task axis):  tau = J^T( S * k_f * (f_d - f) + (1 - S) * k_p * (x_d - x) )

Action layout (policy output), with ``N = sum(force_axes)`` eligible axes::

    [ base pose action (base_n) | N force selections | N force targets ]

``base_n`` is the env's native action width: 6 for Factory ([pos(3), rot(3)]) or 7 for
Forge (pos(3), rot(3), success-prediction at index 6). The base slice is passed through
to the env (so Forge's reward still reads ``actions[:, 6]``); only the trailing ``2N``
selection/target dims drive the controller.

Force sensing is sourced from the Forge env's own ``force_sensor_smooth`` (a 6-D,
EEF-frame, EMA-smoothed wrench refreshed each physics sub-step inside
``_compute_intermediate_values``) — no separate force-torque sensor wrapper is needed.

Frame note: ``force_sensor_smooth`` forces are world-orientation (free vectors), which
matches the geometric-Jacobian control wrench, so translational force axes (x, y, z) are
frame-consistent. Its torques are referenced about the fixed-asset (bolt) origin while
the pose-control torque is about the fingertip, so *rotational* force axes (Rx, Ry, Rz)
would be mis-referenced; the wrapper warns if any are enabled.
"""

import torch
import gymnasium as gym
import numpy as np

from .factory_control_utils import (
    compute_ctrl_targets,
    compute_pose_motion_wrench,
    get_pose_error,
    compute_dof_torque_from_wrench,
)

try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    try:
        import omni.isaac.core.utils.torch as torch_utils
    except ImportError:
        torch_utils = None

AXIS_NAMES = ["X", "Y", "Z", "RX", "RY", "RZ"]


class HybridForcePositionWrapper(gym.Wrapper):
    """Hybrid force/position control via a fixed ``force_axes`` eligibility mask."""

    # Subclass hooks (HybridVICWrapper overrides these).
    _ALLOW_ZERO_N = False         # whether force_axes may select 0 axes (VIC mode).

    def _extra_action_dims(self):
        """Trailing action dims beyond [pose | N sel | N force] (0 for plain hybrid)."""
        return 0

    def __init__(self, env, controller_cfg, num_agents: int = 1):
        """
        Args:
            env: base Isaac Lab Forge/Factory env (gym).
            controller_cfg: ``ControlCfg``; force-control fields and the EMA factor are read
                directly off it (shared, flat config across all control wrappers).
            num_agents: block-parallel agent count (must divide num_envs).
        """
        if torch_utils is None:
            raise ImportError(
                "HybridForcePositionWrapper requires Isaac Sim/Lab torch utilities (isaacsim/omni)."
            )

        # Native action width before expansion (6 Factory / 7 Forge).
        self._base_n = int(getattr(env.unwrapped.cfg, "action_space", 6))

        super().__init__(env)
        self.torch_utils = torch_utils
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.num_agents = int(num_agents)
        if self.num_envs % self.num_agents != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_agents ({self.num_agents})"
            )
        self.envs_per_agent = self.num_envs // self.num_agents

        if controller_cfg is None:
            raise ValueError(f"{type(self).__name__} requires a ControlCfg, got None.")
        cfg = controller_cfg
        self.cfg_h = cfg
        self.ema_factor = float(controller_cfg.ema_factor)

        # ---- force_axes -> eligible axes ----
        force_axes = list(cfg.force_axes)
        if len(force_axes) != 6 or not set(force_axes) <= {0, 1}:
            raise ValueError(f"force_axes must be a length-6 binary vector, got {force_axes!r}")
        self.force_axes = torch.tensor(force_axes, dtype=torch.float32, device=self.device)
        self.eligible_idx = self.force_axes.nonzero(as_tuple=False).view(-1)  # (N,)
        self.N = int(self.eligible_idx.numel())
        if self.N == 0 and not self._ALLOW_ZERO_N:
            raise ValueError("force_axes selects no axes; use control_type='pose' instead of hybrid.")
        if any(force_axes[i] for i in (3, 4, 5)):
            print(
                "[hybrid] WARNING: force_axes enables a rotational axis (Rx/Ry/Rz). The measured "
                "torque is referenced about the fixed-asset origin while pose torque is about the "
                "fingertip — rotational force control is NOT frame-consistent yet. Prefer "
                "translational-only force_axes (e.g. [0,0,1,0,0,0])."
            )

        # EMA / mode flags
        self.no_sel_ema = bool(cfg.no_sel_ema)
        self.apply_ema_force = bool(cfg.apply_ema_force)
        self.ema_mode = cfg.ema_mode  # "action" | "wrench"
        self.use_delta_force = bool(cfg.use_delta_force)
        self.async_z_bounds = bool(cfg.async_z_force_bounds)

        E = self.num_envs

        def _rep(vals):
            return torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(E, 1)

        # Per-axis bounds / thresholds, packed as 6-D [force(3), torque(3)].
        self.force_bounds = _rep(cfg.force_action_bounds)       # (E,3)
        self.torque_bounds = _rep(cfg.torque_action_bounds)     # (E,3)
        self.force_threshold = _rep(cfg.force_action_threshold)
        self.torque_threshold = _rep(cfg.torque_action_threshold)
        self.pos_bounds = None  # set below from env ctrl cfg
        self.bounds6 = torch.cat([self.force_bounds, self.torque_bounds], dim=1)       # (E,6)
        self.thresh6 = torch.cat([self.force_threshold, self.torque_threshold], dim=1)  # (E,6)
        # Eligible-axis slices (E, N) used to scatter the N policy force dims.
        self.elig_bounds = self.bounds6[:, self.eligible_idx]
        self.elig_thresh = self.thresh6[:, self.eligible_idx]

        # Position bounds come from the env ctrl cfg (set there by the runner from ControlCfg).
        pos_bounds_val = getattr(self.unwrapped.cfg.ctrl, "pos_action_bounds", None)
        if pos_bounds_val is None:
            raise ValueError("env cfg.ctrl.pos_action_bounds is required for hybrid control.")
        self.pos_bounds = _rep(pos_bounds_val)

        # ---- force-control stiffness (pure proportional), masked by force_axes ----
        # Diagonal here; HybridVICWrapper overrides _force_wrench with a full 6x6 K_f matrix.
        mask = self.force_axes.unsqueeze(0)  # (1,6)
        self.kp = _rep(cfg.default_task_force_gains) * mask

        # ---- control state ----
        self.sel_matrix = torch.zeros((E, 6), device=self.device)
        self.target_force_for_control = torch.zeros((E, 6), device=self.device)
        self.task_wrench = torch.zeros((E, 6), device=self.device)
        self.ema_task_wrench = torch.zeros((E, 6), device=self.device)
        self._should_log_wrenches = False

        # ---- action space expansion: base_n + 2N (+ subclass extras, e.g. gain matrices) ----
        self.action_space_size = self._base_n + 2 * self.N + self._extra_action_dims()
        self.ema_actions = torch.zeros((E, self.action_space_size), device=self.device)
        self.control_actions = self.ema_actions

        self._original_pre_physics_step = None
        self._original_apply_action = None
        self._original_reset_idx = None

        self._update_dimensions()

        self._wrapper_initialized = False
        if hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        if hasattr(self.unwrapped, "extras") and "to_log" not in self.unwrapped.extras:
            self.unwrapped.extras["to_log"] = {}

    @property
    def policy_selection_layout(self) -> tuple[list[int], list[int], list[int]]:
        """Index lists for the hybrid actor's selection/position/force gated pairs.

        Returns ``(selection_dims, pos_component_dims, force_component_dims)`` over the
        policy action vector ``[ pose(base_n) | N sel | N force | extras ]``:

          * ``selection_dims[k]       = base_n + k``       — Bernoulli gate for eligible axis k
          * ``force_component_dims[k] = base_n + N + k``   — continuous force target
          * ``pos_component_dims[k]   = eligible_idx[k]``  — the pose-block dim that force-axis gates

        Consumed by the runner to build ``HybridControlBlockSimBaActor``. The trailing
        gain dims (hybrid-vic / ctrl-action-interface) are not gated — they stay plain
        continuous and are not referenced here.
        """
        bn, N = self._base_n, self.N
        eligible = [int(j) for j in self.eligible_idx.tolist()]
        selection_dims = [bn + k for k in range(N)]
        force_component_dims = [bn + N + k for k in range(N)]
        pos_component_dims = eligible
        return selection_dims, pos_component_dims, force_component_dims

    # ----------------------------------------------------- logging gates
    # These decide which diagnostic series are worth writing to ``extras['to_log']`` for the
    # current configuration, so we don't emit flat/constant/never-active channels. Overridden
    # by formulations that add config switches (e.g. ctrl-action-interface's constant gains).
    @property
    def _force_log_axes(self) -> list[int]:
        """Force-eligible axis indices; force-control series are logged only for these
        (e.g. z-only force => just Z, never x/y/torques)."""
        return [int(i) for i in self.eligible_idx.tolist()]

    def _force_control_enabled(self) -> bool:
        """Whether force control is active, so force-related series carry information."""
        return self.N > 0

    # ------------------------------------------------------------------ setup
    def _update_dimensions(self):
        """Grow action/obs/state spaces by the added control dims and rebuild gym spaces."""
        diff = self.action_space_size - self._base_n
        if hasattr(self.unwrapped.cfg, "observation_space"):
            self.unwrapped.cfg.observation_space += diff
        if hasattr(self.unwrapped.cfg, "state_space"):
            self.unwrapped.cfg.state_space += diff
        if hasattr(self.unwrapped.cfg, "action_space"):
            self.unwrapped.cfg.action_space = self.action_space_size
        if hasattr(self.unwrapped, "_configure_gym_env_spaces"):
            self.unwrapped._configure_gym_env_spaces()
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_space_size,), dtype=np.float32
        )
        self.unwrapped.action_space = self.action_space

    def _initialize_wrapper(self):
        """Monkeypatch the env control hooks (once, after the robot exists)."""
        if self._wrapper_initialized:
            return
        if not hasattr(self.unwrapped, "force_sensor_smooth"):
            raise RuntimeError(
                "Hybrid control requires the Forge force sensor (env.force_sensor_smooth). "
                "Use a Forge task (Isaac-Forge-*); stock Factory has no force sensing."
            )
        if hasattr(self.unwrapped, "_pre_physics_step"):
            self._original_pre_physics_step = self.unwrapped._pre_physics_step
            self.unwrapped._pre_physics_step = self._wrapped_pre_physics_step
        else:
            raise RuntimeError("[hybrid] env has no _pre_physics_step to wrap.")
        if hasattr(self.unwrapped, "_apply_action"):
            self._original_apply_action = self.unwrapped._apply_action
            self.unwrapped._apply_action = self._wrapped_apply_action
        else:
            raise RuntimeError("[hybrid] env has no _apply_action to wrap.")
        if hasattr(self.unwrapped, "_reset_idx"):
            self._original_reset_idx = self.unwrapped._reset_idx
            self.unwrapped._reset_idx = self._wrapped_reset_idx
        # NOTE: _get_rewards is intentionally NOT overridden — the controller adds no reward terms.
        self._wrapper_initialized = True

    @property
    def robot_force_torque(self):
        """Latest physics-step EEF-frame wrench from the Forge force sensor (6-D [F, T])."""
        return self.unwrapped.force_sensor_smooth

    @property
    def fixed_pos_action_frame(self):
        """Bolt-top action frame (Forge computes this as a local in _apply_action)."""
        return self.unwrapped.fixed_pos_obs_frame + self.unwrapped.init_fixed_pos_obs_noise

    # ------------------------------------------------------------- pre-physics
    def _wrapped_pre_physics_step(self, action):
        if action.shape[1] != self.action_space_size:
            raise ValueError(
                f"[hybrid] action dim {action.shape[1]} != expected {self.action_space_size} "
                f"(base_n={self._base_n}, N={self.N})"
            )

        env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_ema_actions(env_ids)

        # Let the base env run its reset_buffers + EMA into self.actions (the buffer is now
        # action_space_size wide, so the full action is accepted). We overwrite self.actions
        # below with our own EMA so the base/Forge reward still reads our base-pose slice
        # (incl. the Forge success dim at index 6 for base_n==7).
        if self._original_pre_physics_step:
            self._original_pre_physics_step(action.clone())

        self._apply_ema_to_actions(action)
        self.unwrapped.actions = self.ema_actions.clone()

        self._compute_control_targets()

        if hasattr(self.unwrapped, "_compute_intermediate_values"):
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)
        self._should_log_wrenches = True

    def _reset_ema_actions(self, env_ids):
        self.ema_actions[env_ids] = 0.0
        self.ema_task_wrench[env_ids] = 0.0

    def _apply_ema_to_actions(self, action):
        """EMA per segment: base-pose via the env's own EMA (bit-exact, per-env ema_factor);
        selection unless no_sel_ema; force if apply_ema_force; trailing dims via a subclass hook."""
        f = self.ema_factor
        bn, N = self._base_n, self.N
        e = self.ema_actions
        # base pose (+ optional Forge success dim): the base _pre_physics_step (called just
        # before us) already wrote its own EMA — with Forge's per-env randomized ema_factor —
        # into self.unwrapped.actions[:, :bn]. Reuse it verbatim so pose control is bit-exact.
        e[:, :bn] = self.unwrapped.actions[:, :bn]
        # force selection
        sel = action[:, bn:bn + N]
        if self.no_sel_ema:
            e[:, bn:bn + N] = sel
        else:
            e[:, bn:bn + N] = f * sel + (1 - f) * e[:, bn:bn + N]
        # force targets
        force = action[:, bn + N:bn + 2 * N]
        if self.apply_ema_force or self.ema_mode == "wrench":
            e[:, bn + N:bn + 2 * N] = f * force + (1 - f) * e[:, bn + N:bn + 2 * N]
        else:
            e[:, bn + N:bn + 2 * N] = force
        # Trailing action dims (none for plain hybrid; gain matrices for hybrid_vic).
        self._ema_extra_actions(action)
        # In wrench mode, control from raw actions (the wrench itself is EMA'd later).
        self.control_actions = action.clone() if self.ema_mode == "wrench" else self.ema_actions

    def _ema_extra_actions(self, action):
        """Hook for trailing action dims beyond [pose | sel | force]. No-op for plain hybrid."""
        pass

    # ----------------------------------------------------------- control targets
    def _compute_control_targets(self):
        bn, N = self._base_n, self.N
        ca = self.control_actions

        # 1) Selection matrix: scatter N selection bits into the eligible axes (others = 0 = pose).
        sel_bits = (ca[:, bn:bn + N] > 0.5).float()                 # (E, N)
        self.sel_matrix = torch.zeros((self.num_envs, 6), device=self.device)
        self.sel_matrix[:, self.eligible_idx] = sel_bits
        # Selection is only meaningful on force-eligible axes — pose-only axes are always
        # position-controlled, so skip their flat-zero series (and emit nothing when force
        # control is off entirely).
        if self._force_control_enabled() and hasattr(self.unwrapped, "extras"):
            log = self.unwrapped.extras["to_log"]
            for i in self._force_log_axes:
                log[f"Control Mode / Force Control {AXIS_NAMES[i]}"] = self.sel_matrix[:, i]

        # 2+3) Pose (position + orientation) targets — bit-exact with the base ForgeEnv
        # controller, including delta_pos / delta_yaw which the env reward (action penalty)
        # reads. Uses the EMA'd pose actions ca[:, 0:3] / ca[:, 3:6] (same slices the base reads).
        ctrl_pos, ctrl_quat, delta_pos, delta_yaw = compute_ctrl_targets(self.unwrapped, ca)
        self.unwrapped.ctrl_target_fingertip_midpoint_pos = ctrl_pos
        self.unwrapped.ctrl_target_fingertip_midpoint_quat = ctrl_quat
        self.unwrapped.delta_pos = delta_pos
        self.unwrapped.delta_yaw = delta_yaw

        # 4) Force target: scatter the N force actions into a 6-D wrench at the eligible axes.
        force_acts = ca[:, bn + N:bn + 2 * N]                       # (E, N)
        measured = self.robot_force_torque
        self.target_force_for_control = torch.zeros((self.num_envs, 6), device=self.device)
        if self.use_delta_force:
            delta = force_acts * self.elig_thresh
            tgt = torch.clip(
                delta + measured[:, self.eligible_idx], -self.elig_bounds, self.elig_bounds
            )
        else:
            tgt = force_acts * self.elig_bounds
        self.target_force_for_control[:, self.eligible_idx] = tgt
        # Async-z: z force commands cannot be positive (only if z is force-eligible).
        if self.async_z_bounds and self.force_axes[2] > 0:
            self.target_force_for_control[:, 2] = (
                self.target_force_for_control[:, 2] - self.force_bounds[:, 2]
            ) / 2.0

        # Log applied force-goal magnitude — only when force control is active, and per-axis
        # only for force-eligible translational axes (others have no force goal).
        if self._force_control_enabled() and hasattr(self.unwrapped, "extras"):
            log = self.unwrapped.extras["to_log"]
            fg = self.target_force_for_control[:, :3]
            log["Control Target / Force Goal Norm"] = torch.norm(fg, p=2, dim=-1)
            for i in self._force_log_axes:
                if i < 3:
                    log[f"Control Target / Force {AXIS_NAMES[i]} Goal"] = torch.abs(fg[:, i])

    def _pose_motion_wrench(self):
        """Pose PD motion wrench using the env's diagonal task gains + dead zone.

        Bit-exact with the base controller. Overridden by HybridVICWrapper to use full
        6x6 stiffness/damping matrices from the policy instead of the env's diagonal gains.
        """
        pos_error, aa_error = get_pose_error(
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=self.unwrapped.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.unwrapped.ctrl_target_fingertip_midpoint_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, aa_error), dim=1)
        return compute_pose_motion_wrench(
            delta_pose,
            self.unwrapped.fingertip_midpoint_linvel,
            self.unwrapped.fingertip_midpoint_angvel,
            task_prop_gains=self.unwrapped.task_prop_gains,
            task_deriv_gains=self.unwrapped.task_deriv_gains,
            dead_zone_thresholds=getattr(self.unwrapped, "dead_zone_thresholds", None),
            matrix=False,
        )

    def _force_wrench(self, measured):
        """Force-control wrench: pure proportional stiffness on force error.

        Diagonal force stiffness (``self.kp``, masked by force_axes). Overridden by
        HybridVICWrapper to use a full 6x6 stiffness matrix K_f from the policy.
        """
        return self.kp * (self.target_force_for_control - measured)

    # ------------------------------------------------------------- apply action
    def _wrapped_apply_action(self):
        # Curr-yaw bookkeeping (Forge tracks this for success/termination checks).
        _, _, curr_yaw = self.torch_utils.get_euler_xyz(self.unwrapped.fingertip_midpoint_quat)
        self.unwrapped.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        if self.unwrapped.last_update_timestamp < self.unwrapped._robot._data._sim_timestamp:
            self.unwrapped._compute_intermediate_values(dt=self.unwrapped.physics_dt)

        self.unwrapped.ctrl_target_gripper_dof_pos = 0.0
        measured = self.robot_force_torque

        # Pose PD motion wrench (bit-exact with the base controller incl. dead zone).
        # Subclasses (hybrid_vic) override _pose_motion_wrench to use full K/D matrices.
        pose_wrench = self._pose_motion_wrench()

        # Force-control wrench: pure stiffness on force error (no damping/PID).
        # Subclasses (hybrid_vic) override _force_wrench to use a full 6x6 K_f matrix.
        force_wrench = self._force_wrench(measured)

        # Log force + position error magnitude once per env.step (first apply_action call).
        if self._should_log_wrenches and hasattr(self.unwrapped, "extras"):
            log = self.unwrapped.extras["to_log"]
            # Force error only on force-eligible translational axes; position error always
            # (pose control is active on every axis).
            if self._force_control_enabled():
                ferr = self.target_force_for_control[:, :3] - measured[:, :3]
                for i in self._force_log_axes:
                    if i < 3:
                        log[f"Controller Output / Force Error {AXIS_NAMES[i]}"] = ferr[:, i]
            perr = self.unwrapped.ctrl_target_fingertip_midpoint_pos - self.unwrapped.fingertip_midpoint_pos
            for i in range(3):
                log[f"Controller Output / Position Error {AXIS_NAMES[i]}"] = perr[:, i]
            self._should_log_wrenches = False

        # Blend via the selection matrix (force-axes already baked into sel_matrix).
        # On the pose axes (sel=0) this is exactly the base controller's motion wrench, so
        # pose-only control stays bit-exact (no extra wrench bound-zeroing — base has none).
        task_wrench = (1 - self.sel_matrix) * pose_wrench + self.sel_matrix * force_wrench
        if self.ema_mode == "wrench":
            task_wrench = self.ema_factor * task_wrench + (1 - self.ema_factor) * self.ema_task_wrench
            self.ema_task_wrench = task_wrench.clone()

        # Task wrench -> joint torques (Jᵀ + null-space posture), then actuate.
        self.unwrapped.joint_torque, task_wrench = compute_dof_torque_from_wrench(
            cfg=self.unwrapped.cfg,
            dof_pos=self.unwrapped.joint_pos,
            dof_vel=self.unwrapped.joint_vel,
            task_wrench=task_wrench,
            jacobian=self.unwrapped.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.unwrapped.arm_mass_matrix,
            device=self.unwrapped.device,
        )
        self.task_wrench = task_wrench.clone()

        self.unwrapped.ctrl_target_joint_pos[:, 7:9] = self.unwrapped.ctrl_target_gripper_dof_pos
        self.unwrapped.joint_torque[:, 7:9] = 0.0
        self.unwrapped._robot.set_joint_position_target(self.unwrapped.ctrl_target_joint_pos)
        self.unwrapped._robot.set_joint_effort_target(self.unwrapped.joint_torque)

    # ------------------------------------------------------------------- resets
    def _wrapped_reset_idx(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_ema_actions(env_ids)
        if self._original_reset_idx is not None:
            self._original_reset_idx(env_ids)

    def step(self, action):
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return super().step(action)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if not self._wrapper_initialized and hasattr(self.unwrapped, "_robot"):
            self._initialize_wrapper()
        return obs, info
