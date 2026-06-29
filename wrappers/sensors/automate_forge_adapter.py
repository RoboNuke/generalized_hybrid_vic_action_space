"""Make an AutoMate ``AssemblyEnv`` present itself to the rest of the pipeline as a FORGE env.

IsaacLab's AutoMate ``AssemblyEnv`` (``Isaac-AutoMate-Assembly-Direct-v0``) is a *sibling*
of ``FactoryEnv`` (``AssemblyEnv(DirectRLEnv)``), not a ``FactoryEnv`` subclass. It shares
Factory's control plumbing (``actions``/``prev_actions``/``_pre_physics_step``,
``fixed_pos_obs_frame``/``init_fixed_pos_obs_noise``, ``pos_threshold``/``rot_threshold``,
6-D current-EE-relative ``_apply_action``) but has its own reward (SDF + SoftDTW imitation +
SBC curriculum), no force sensor, and goal-relative observations. This adapter bolts the
FORGE feature set onto an ``AssemblyEnv`` *instance* so the existing controllers, contact
sensor and the ``forge`` scorer work unchanged via duck-typing:

  * adds a force-torque sensor exactly like FORGE (read + EMA-smooth + reframe + noise),
    exposing ``force_sensor_smooth``/``noisy_force`` for both observations and hybrid control;
  * applies FORGE-style observation noise to the fingertip pose + finite-diff EE velocities;
  * appends a noisy ``ft_force`` (3) channel to the policy obs and a clean one to the critic
    state, plus the per-env contact-penalty ``force_threshold`` (1) to both — exactly like FORGE
    (forge_env_cfg.py obs_order/state_order) — so the policy can observe the contact-force budget
    it is penalised against;
  * echoes the full action vector (FORGE's ``prev_actions``) onto BOTH the policy obs and the
    critic state — AutoMate's ``AssemblyEnv`` omits this, but the control wrapper assumes the
    FORGE convention (it grows the obs/state spaces by the gain-action dims), so without the
    echo the declared and actual obs sizes diverge;
  * keeps AutoMate's NATIVE reward (SDF + imitation + success bonus) — the adapter does
    NOT substitute the Factory/Forge keypoint reward. The imitation weight is set to 0
    (``imitation_rwd_scale``), so the dense reward is SDF + success; imitation is still
    computed so the component stays visible in TensorBoard. Each per-env reward component
    is routed through ``_log_factory_metrics`` for per-agent ``logs_rew/<term>`` logging;
  * keeps AutoMate's action handling unchanged (no goal-relative actions, no dead-zone,
    no success-prediction action; ``action_space`` stays 6).

Two pieces, mirroring :mod:`wrappers.sensors.contact_sensor_wrapper`:

  * :func:`install_automate_forge_adapter` — pre-``gym.make`` ``env_cfg`` mutation (robot USD
    swap to the force-sensor-bearing Factory franka, ``activate_contact_sensors=True``,
    Forge-like ``obs_rand``, FORGE reward-scale fields, ``ft_smoothing_factor``).
  * :class:`AutoMateForgeAdapter` — post-``gym.make`` ``gym.Wrapper`` applied INNERMOST (before
    the control wrapper, which requires ``env.unwrapped.force_sensor_smooth`` at init); it
    installs the FT sensor / obs-noise / FORGE reward onto ``env.unwrapped`` and grows the
    obs/state spaces by the appended ``ft_force`` dims.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch

import gymnasium as gym

# FORGE defaults (isaaclab forge_env_cfg.py:33-35,100 / forge_tasks_cfg.py:13-18).
_FT_SMOOTHING_FACTOR = 0.25
_OBS_RAND_FINGERTIP_POS = 0.00025
_OBS_RAND_FINGERTIP_ROT_DEG = 0.1
_OBS_RAND_FT_FORCE = 1.0
_ACTION_PENALTY_EE_SCALE = 0.0
_ACTION_PENALTY_ASSET_SCALE = 0.001
_ACTION_GRAD_PENALTY_SCALE = 0.1
_CONTACT_PENALTY_SCALE = 0.05
_CONTACT_PENALTY_THRESHOLD_RANGE = [5.0, 10.0]
# AutoMate native-reward weights. SDF stays at AutoMate's default (1.0); imitation is
# disabled (0.0) so the dense reward is SDF + success. Imitation is still computed so its
# component is visible in TensorBoard.
_IMITATION_RWD_SCALE = 0.0

# Appended FORGE channels: 3-D force vector + 1-D contact-penalty threshold.
FT_DIM = 3
FORCE_THRESHOLD_DIM = 1


def _forge_like_obs_rand_cfg():
    """Build an obs-rand cfg carrying FORGE's noise fields (so the runner's field-copy loop +
    debug print find ``fingertip_pos``/``fingertip_rot_deg``/``ft_force``) plus AutoMate's
    ``fixed_asset_pos`` (read in ``randomize_fixed_initial_state``)."""
    from isaaclab.utils import configclass

    @configclass
    class ForgeLikeObsRandCfg:
        fixed_asset_pos = [0.001, 0.001, 0.001]
        fingertip_pos = _OBS_RAND_FINGERTIP_POS
        fingertip_rot_deg = _OBS_RAND_FINGERTIP_ROT_DEG
        ft_force = _OBS_RAND_FT_FORCE

    return ForgeLikeObsRandCfg()


def install_automate_forge_adapter(env_cfg, runner_cfg: Any = None) -> None:
    """Mutate an AutoMate ``AssemblyEnvCfg`` in place so the env spawns FORGE-compatibly.

    Must be called BEFORE ``gym.make`` and BEFORE ``_apply_env_cfg_overrides`` — the adapter
    only WRITES its Forge defaults here (it reads no user-override value), so running it first
    lets ``env_cfg_overrides`` win for the fields it sets (``ft_smoothing_factor``, the task
    reward-penalty scales, etc.). Also runs before the runner's ctrl/obs_rand copy loops +
    obs-noise print so those see the Forge-like ``obs_rand``. Does NOT touch ``if_sbc`` — that
    stays a config field the user can flip via ``env_cfg_overrides`` (e.g. ``tasks.insertion.if_sbc``).
    """
    # 1. Robot USD swap: AutoMate's franka_mimic.usd lives in a different Nucleus dir and has
    #    no "force_sensor" body. Use Factory/Forge's franka_mimic.usd (which has it) and turn
    #    on contact sensors so the FT reading + held/fixed ContactSensor work.
    from isaaclab_tasks.direct.factory.factory_tasks_cfg import ASSET_DIR as FACTORY_ASSET_DIR

    env_cfg.robot.spawn.usd_path = f"{FACTORY_ASSET_DIR}/franka_mimic.usd"
    env_cfg.robot.spawn.activate_contact_sensors = True

    # 2. Forge-like obs noise config (replaces AutoMate's bare ObsRandCfg; keeps fixed_asset_pos).
    env_cfg.obs_rand = _forge_like_obs_rand_cfg()

    task = env_cfg.tasks[env_cfg.task_name]

    # 2b. Assembly (plug/socket pair) selection. ``env_cfg.tasks`` is a dict, so the runner's
    #     dotted env_cfg_overrides can't reach it — use runner_cfg.automate_assembly_id and
    #     recompute the derived asset paths (assembly_dir / disassembly_path_json) from the id.
    assembly_id = getattr(runner_cfg, "automate_assembly_id", None) if runner_cfg is not None else None
    if assembly_id:
        from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR as AUTOMATE_ASSET_DIR

        assembly_id = str(assembly_id)
        task.assembly_id = assembly_id
        task.assembly_dir = f"{AUTOMATE_ASSET_DIR}/{assembly_id}/"
        task.disassembly_path_json = f"{task.assembly_dir}/disassemble_traj.json"
        task.eval_filename = f"evaluation_{assembly_id}.h5"
        # The nested fixed/held spawn USD paths are baked at class-definition time from the
        # DEFAULT assembly_id (assembly_tasks_cfg.py: usd_path=f"{assembly_dir}{...usd_path}"),
        # so updating assembly_dir alone leaves the env spawning the default plug/socket USDs
        # while the SDF .obj meshes + grasp/disassembly JSON (keyed off assembly_dir/assembly_id)
        # come from the new assembly -> geometry mismatch. AutoMate's own run_w_id.py sidesteps
        # this by textually rewriting the cfg source before import; we mutate at runtime, so
        # recompute the two spawn paths here to match the new assembly_dir.
        task.fixed_asset.spawn.usd_path = f"{task.assembly_dir}{task.fixed_asset_cfg.usd_path}"
        task.held_asset.spawn.usd_path = f"{task.assembly_dir}{task.held_asset_cfg.usd_path}"
        print(
            f"[automate-forge] assembly_id={assembly_id} (assembly_dir={task.assembly_dir}); "
            f"fixed_asset.usd={task.fixed_asset.spawn.usd_path}, "
            f"held_asset.usd={task.held_asset.spawn.usd_path}",
            flush=True,
        )

    # 3. Reward setup. The adapter uses AutoMate's NATIVE reward (SDF + imitation +
    #    success), so zero the imitation weight -> dense reward is SDF + success.
    task.imitation_rwd_scale = _IMITATION_RWD_SCALE  # 0.0 (imitation still computed/logged)
    #    The FORGE reward-scale fields below are now vestigial for the reward itself, but the
    #    ForgeWrapper scorer's _factory_scales()/_forge_scales() still read them to build its
    #    scale tables, so they must exist. contact_penalty_threshold_range also feeds the
    #    force-threshold observable. (The native reward ignores the penalty scales.)
    task.action_penalty_ee_scale = _ACTION_PENALTY_EE_SCALE
    task.action_penalty_asset_scale = _ACTION_PENALTY_ASSET_SCALE
    task.action_grad_penalty_scale = _ACTION_GRAD_PENALTY_SCALE  # AutoMate default is 0.0
    task.contact_penalty_scale = _CONTACT_PENALTY_SCALE
    task.contact_penalty_threshold_range = list(_CONTACT_PENALTY_THRESHOLD_RANGE)

    # 4. Force smoothing factor (FORGE reads cfg.ft_smoothing_factor).
    env_cfg.ft_smoothing_factor = _FT_SMOOTHING_FACTOR

    print(
        "[automate-forge] env_cfg patched: robot->Factory/franka_mimic.usd "
        "(activate_contact_sensors=True), Forge-like obs_rand, AutoMate native reward "
        f"(SDF + success; imitation_rwd_scale={_IMITATION_RWD_SCALE}), "
        f"ft_smoothing_factor={_FT_SMOOTHING_FACTOR}.",
        flush=True,
    )


class AutoMateForgeAdapter(gym.Wrapper):
    """Bolt FORGE force-sensing / obs-noise / reward onto an AutoMate ``AssemblyEnv`` instance.

    Applied innermost (right after ``gym.make``). All install work happens synchronously in
    ``__init__`` so ``env.unwrapped.force_sensor_smooth`` exists before the control wrapper's
    ``__init__`` asserts it.
    """

    def __init__(self, env, runner_cfg=None) -> None:
        super().__init__(env)
        u = env.unwrapped
        self._u = u
        dev = u.device
        n = u.num_envs

        # Optional peg IN-HAND OFFSET (grasp uncertainty). AutoMate always grasps the peg at
        # the NOMINAL plug_grasps.json grasp — the gripper's world pose is fixed and we do NOT
        # touch plug_grasp_pos_local/quat_local. ``runner_cfg.automate_grasp_pose`` instead shifts
        # the PEG relative to that fixed grasp by [x, y, z, roll, pitch, yaw] (meters / DEGREES,
        # XYZ-Euler) in the PLUG's local frame. Default / all-zero / None => no shift (nominal).
        #
        # Mechanism: AutoMate's reset (randomize_initial_state) places the peg at the socket
        # frame (pre_grasp=True), IK's the gripper to grasp it at the nominal grasp, then RE-places
        # the peg (pre_grasp=False) and closes the gripper. We wrap randomize_held_initial_state
        # so that on the pre_grasp=False placement — the one the gripper actually closes on — the
        # peg is nudged by the offset (applied in the peg's body frame). The gripper, already IK'd
        # to the nominal grasp, then closes on the off-nominal peg (physical/friction grasp, so
        # only modest offsets stay seated). pre_grasp=True is left untouched so the gripper IK
        # still targets the nominal grasp -> gripper world pose stays fixed.
        grasp = getattr(runner_cfg, "automate_grasp_pose", None) if runner_cfg is not None else None
        if grasp is not None and any(float(v) != 0.0 for v in grasp):
            import isaacsim.core.utils.torch as torch_utils

            from isaaclab.utils.math import quat_from_euler_xyz

            g = torch.as_tensor(grasp, dtype=torch.float32, device=dev)
            assert g.numel() == 6, (
                f"automate_grasp_pose must be 6 floats [x,y,z, roll,pitch,yaw(deg)], got {g.numel()}"
            )
            _off_pos = g[:3]
            _off_rpy = torch.deg2rad(g[3:])
            _off_quat = quat_from_euler_xyz(_off_rpy[0], _off_rpy[1], _off_rpy[2])  # (4,) wxyz

            _orig_rhis = u.randomize_held_initial_state

            def _rhis_with_offset(env_ids, pre_grasp):
                _orig_rhis(env_ids, pre_grasp)
                if pre_grasp:
                    return  # leave the gripper-IK target placement at the nominal pose
                ids = env_ids
                m = ids.shape[0]
                pos_w = u._held_asset.data.root_pos_w[ids]
                quat_w = u._held_asset.data.root_quat_w[ids]
                # peg_world ∘ offset  (offset expressed in the peg's body frame)
                new_quat, new_pos = torch_utils.tf_combine(
                    quat_w, pos_w, _off_quat.unsqueeze(0).repeat(m, 1), _off_pos.unsqueeze(0).repeat(m, 1)
                )
                u._held_asset.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1), env_ids=ids)
                u._held_asset.write_root_velocity_to_sim(torch.zeros((m, 6), device=dev), env_ids=ids)
                u._held_asset.reset()
                u.step_sim_no_action()

            u.randomize_held_initial_state = _rhis_with_offset
            print(f"[automate-forge] peg in-hand offset (plug-local, x,y,z,roll,pitch,yaw deg): {grasp}", flush=True)

        # ---- FORGE force-sensor state (forge_env.py:35-38) ----
        u.force_sensor_body_idx = u._robot.body_names.index("force_sensor")
        u.force_sensor_world_smooth = torch.zeros((n, 6), device=dev)
        u.force_sensor_smooth = torch.zeros((n, 6), device=dev)
        u.noisy_force = torch.zeros((n, 3), device=dev)
        u.flip_quats = torch.ones((n,), dtype=torch.float32, device=dev)

        # Noisy obs tensors seeded from current (clean) values so _get_observations is safe
        # even if it runs before the first wrapped _compute_intermediate_values.
        u.noisy_fingertip_pos = u.fingertip_midpoint_pos.clone()
        u.noisy_fingertip_quat = u.fingertip_midpoint_quat.clone()

        # Contact-penalty thresholds (forge_env.py:318-320) — sampled now since the first
        # reward step may precede a reset. Read from cfg_task (the same task cfg install_*
        # seeds with _CONTACT_PENALTY_THRESHOLD_RANGE) so env_cfg_overrides on
        # tasks.<name>.contact_penalty_threshold_range actually changes the sampled thresholds,
        # matching what the forge scorer/reward read off cfg_task.
        self._contact_lo, self._contact_hi = u.cfg_task.contact_penalty_threshold_range
        u.contact_penalty_thresholds = self._sample_contact_thresholds()

        # ---- scorer-compat attrs (wrappers/scorers/forge.py reads these) ----
        u.success_pred_scale = 0.0  # success prediction excluded -> term scale stays 0
        u.first_pred_success_tx = {
            t: torch.zeros(n, device=dev, dtype=torch.long) for t in (0.5, 0.6, 0.7, 0.8, 0.9)
        }

        # ---- FORGE-style logging methods, reused verbatim from upstream, bound to this env ----
        from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
        from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

        u._log_factory_metrics = types.MethodType(FactoryEnv._log_factory_metrics, u)
        u._log_forge_metrics = types.MethodType(ForgeEnv._log_forge_metrics, u)

        # ---- monkeypatch behaviour onto the env instance ----
        self._orig_civ = u._compute_intermediate_values
        self._orig_get_obs = u._get_observations
        self._orig_reset_idx = u._reset_idx
        u._compute_intermediate_values = self._make_civ()
        u._get_observations = self._make_get_obs()
        u._get_rewards = self._make_get_rewards()
        u._reset_idx = self._make_reset_idx()

        # ---- grow obs/state spaces by appended ft_force (+3) AND the prev_actions echo ----
        # FORGE's policy obs and critic state both carry the full action vector ("prev_actions"):
        # factory_env.py does `observation_space += action_space` / `state_space += action_space`
        # at init and appends prev_actions in _get_observations. The control wrapper (applied next)
        # relies on that convention — it grows the spaces by only the EXTRA action dims it adds,
        # assuming the env already echoes the BASE action vector. AutoMate's AssemblyEnv does NOT
        # echo actions (assembly_env.py builds obs/state from obs_order/state_order alone), so we
        # add the base action dims here (+ _get_observations appends prev_actions below). Combined:
        # adapter adds base_act, control wrapper adds (action_space_size - base_act) => the full
        # action vector is accounted for, matching the prev_actions actually appended.
        self._base_act_dim = int(u.cfg.action_space)  # base pose dims; control wrapper adds the rest
        grow = FT_DIM + FORCE_THRESHOLD_DIM + self._base_act_dim
        if hasattr(u.cfg, "observation_space"):
            u.cfg.observation_space += grow
        if hasattr(u.cfg, "state_space"):
            u.cfg.state_space += grow
        if hasattr(u, "_configure_gym_env_spaces"):
            u._configure_gym_env_spaces()

        print(
            "[automate-forge] adapter installed: FT sensor (force_sensor_smooth/noisy_force), "
            f"FORGE reward (no success-pred), FORGE obs noise, prev_actions echo; "
            f"obs/state grown by +{grow} (ft={FT_DIM}, force_threshold={FORCE_THRESHOLD_DIM}, "
            f"base_actions={self._base_act_dim}).",
            flush=True,
        )

    # ------------------------------------------------------------------ helpers
    def _sample_contact_thresholds(self) -> torch.Tensor:
        u = self._u
        return self._contact_lo + torch.rand((u.num_envs,), device=u.device) * (
            self._contact_hi - self._contact_lo
        )

    # ------------------------------------------------------------------ patches
    def _make_civ(self):
        """Wrap ``_compute_intermediate_values``: FORGE obs noise + force sensing (forge_env.py:57-114)."""
        u = self._u
        orig = self._orig_civ
        dev = u.device
        n = u.num_envs

        import isaacsim.core.utils.torch as torch_utils

        from isaaclab.utils.math import axis_angle_from_quat
        from isaaclab_tasks.direct.forge import forge_utils

        def civ(dt):
            orig(dt)
            cfg = u.cfg

            # Fingertip position noise.
            pos_n = cfg.obs_rand.fingertip_pos
            u.noisy_fingertip_pos = u.fingertip_midpoint_pos + torch.randn((n, 3), device=dev) * pos_n

            # Fingertip rotation noise (random axis, small angle). FULL 6-DOF orientation: no
            # quat w,z zeroing and no sign flip (those belong to the upright-only reduced-quat
            # scheme and are invalid once the gripper can rotate freely, which it now always can).
            axis = torch.randn((n, 3), device=dev)
            axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True)
            angle = torch.randn((n,), device=dev) * np.deg2rad(cfg.obs_rand.fingertip_rot_deg)
            u.noisy_fingertip_quat = torch_utils.quat_mul(
                u.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(angle, axis)
            )

            # Finite-diff EE velocities from the noisy fingertip values (overwrites clean FD).
            u.ee_linvel_fd = (u.noisy_fingertip_pos - u.prev_fingertip_pos) / dt
            u.prev_fingertip_pos = u.noisy_fingertip_pos.clone()
            rot_diff = torch_utils.quat_mul(
                u.noisy_fingertip_quat, torch_utils.quat_conjugate(u.prev_fingertip_quat)
            )
            rot_diff = rot_diff * torch.sign(rot_diff[:, 0]).unsqueeze(-1)
            # FULL angular velocity (no roll/pitch zeroing).
            u.ee_angvel_fd = axis_angle_from_quat(rot_diff) / dt
            u.prev_fingertip_quat = u.noisy_fingertip_quat.clone()

            # Force sensor: read incoming joint force, EMA-smooth, reframe to the bolt frame, noise.
            force_world = u._robot.root_physx_view.get_link_incoming_joint_force()[:, u.force_sensor_body_idx]
            alpha = cfg.ft_smoothing_factor
            u.force_sensor_world_smooth = alpha * force_world + (1 - alpha) * u.force_sensor_world_smooth
            u.force_sensor_smooth = torch.zeros_like(u.force_sensor_world_smooth)
            idq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).unsqueeze(0).repeat(n, 1)
            u.force_sensor_smooth[:, :3], u.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
                u.force_sensor_world_smooth[:, 0:3],
                u.force_sensor_world_smooth[:, 3:6],
                (idq, torch.zeros((n, 3), device=dev)),
                (idq, u.fixed_pos_obs_frame + u.init_fixed_pos_obs_noise),
            )
            u.noisy_force = u.force_sensor_smooth[:, 0:3] + torch.randn((n, 3), device=dev) * cfg.obs_rand.ft_force

        return civ

    def _make_get_obs(self):
        """Wrap ``_get_observations``: emit noisy fingertip/vel terms + append ft_force."""
        u = self._u
        orig = self._orig_get_obs

        def get_obs():
            # AutoMate builds its obs/state dicts directly from these instance tensors; swap in
            # the noisy versions around the original call so the shared fingertip/velocity terms
            # carry FORGE-style noise, then restore the clean tensors (control uses them).
            saved = (
                u.fingertip_midpoint_pos,
                u.fingertip_midpoint_quat,
                u.fingertip_midpoint_linvel,
                u.fingertip_midpoint_angvel,
            )
            u.fingertip_midpoint_pos = u.noisy_fingertip_pos
            u.fingertip_midpoint_quat = u.noisy_fingertip_quat
            u.fingertip_midpoint_linvel = u.ee_linvel_fd
            u.fingertip_midpoint_angvel = u.ee_angvel_fd
            try:
                out = orig()
            finally:
                (
                    u.fingertip_midpoint_pos,
                    u.fingertip_midpoint_quat,
                    u.fingertip_midpoint_linvel,
                    u.fingertip_midpoint_angvel,
                ) = saved
            # Append force channels, the per-env contact-penalty threshold, then the full action
            # vector (FORGE's prev_actions echo, forge_env.py:124-137): noisy force to the policy
            # obs, clean force to the critic state, the threshold (n,1) and the same action vector
            # to both. Order matches FORGE's obs_order/state_order (ft_force, force_threshold,
            # prev_actions) so the threshold is a real observable, letting the policy see the
            # contact-force budget it is penalised against. The control wrapper has resized
            # u.actions to the full pose+gain width (via _configure_gym_env_spaces -> sample_space),
            # so this echoes every commanded dim; it is zeros until the first step.
            prev_actions = u.actions
            force_threshold = u.contact_penalty_thresholds[:, None]
            out["policy"] = torch.cat(
                [out["policy"], u.noisy_force, force_threshold, prev_actions], dim=-1
            )
            out["critic"] = torch.cat(
                [out["critic"], u.force_sensor_smooth[:, 0:3], force_threshold, prev_actions], dim=-1
            )
            return out

        return get_obs

    def _make_get_rewards(self):
        """Replace ``_get_rewards`` with AutoMate's NATIVE reward (SDF + imitation + success).

        Mirrors ``AssemblyEnv._update_rew_buf`` / ``_get_rewards`` rather than substituting
        the Factory/Forge keypoint reward. The only reason this overrides ``_get_rewards`` at
        all (instead of letting the native method run) is logging: native AutoMate writes only
        scalar means to top-level ``extras``, which the per-agent logging path doesn't read.
        Routing the per-env ``rew_dict`` through ``_log_factory_metrics`` (hooked by the
        ForgeWrapper scorer) gets each component published per-agent as ``logs_rew/<term>`` and
        keeps ``per_env_curr_successes`` / ``ep_success_times`` flowing. ``imitation_rwd_scale``
        is 0 (set in install), so the dense reward is SDF + success; imitation is still computed
        so its component stays visible.
        """
        u = self._u

        from isaaclab_tasks.direct.automate import automate_algo_utils as automate_algo
        from isaaclab_tasks.direct.automate import industreal_algo_utils as industreal_algo

        def get_rewards():
            cfg_task = u.cfg_task

            # Success from AutoMate's own asset-correct check (also drives ep_success_times via
            # _log_factory_metrics below and the SBC advance).
            curr_successes = automate_algo.check_plug_inserted_in_socket(
                u.held_pos,
                u.fixed_pos,
                u.disassembly_dists,
                u.keypoints_held,
                u.keypoints_fixed,
                cfg_task.close_error_thresh,
                u.episode_length_buf,
            )

            # --- AutoMate native dense reward (assembly_env._update_rew_buf) ---
            # SDF reward: distance of the plug's surface sample points (current pose) to the
            # plug volume placed at the assembled (socket) pose — geometry-aware, all 6 DOF.
            sdf = industreal_algo.get_sdf_reward(
                u.plug_mesh,
                u.plug_sample_points,
                u.held_pos,
                u.held_quat,
                u.fixed_pos,
                u.fixed_quat,
                u.wp_device,
                u.device,
            )
            rew_dict = {
                "sdf": sdf,
                "curr_success": curr_successes.float(),
            }
            rew_buf = cfg_task.sdf_rwd_scale * sdf + curr_successes.float()

            # Imitation reward (Soft-DTW of the EEF path vs the recorded disassembly trajectory)
            # — computed ONLY when weighted in. The CUDA Soft-DTW path JIT-compiles numba.cuda
            # kernels (requires libnvvm); skipping it when the weight is 0 keeps a non-imitation
            # run from depending on numba/NVVM (and from paying the DTW cost). When enabled, it is
            # added to rew_buf and logged as logs_rew/imitation.
            if cfg_task.imitation_rwd_scale != 0.0:
                curr_eef_pos = (u.fingertip_midpoint_pos - u.gripper_goal_pos).reshape(-1, 3)
                imitation = automate_algo.get_imitation_reward_from_dtw(
                    u.eef_pos_traj, curr_eef_pos, u.prev_fingertip_midpoint_pos, u.soft_dtw_criterion, u.device
                )
                u.prev_fingertip_midpoint_pos = torch.cat(
                    (u.prev_fingertip_midpoint_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
                )
                rew_dict["imitation"] = imitation
                rew_buf = rew_buf + cfg_task.imitation_rwd_scale * imitation

            # Route per-env components through the scorer-hooked logger so SAC logs each one
            # per-agent (logs_rew/<term>) and captures per_env_curr_successes / ep_success_times.
            # Only _log_factory_metrics is needed (no Forge-specific terms).
            u._log_factory_metrics(rew_dict, curr_successes)

            # SBC curriculum advance (reward itself is NOT scaled by SBC). Kept togglable via
            # cfg_task.if_sbc so the spawn-height curriculum still progresses when enabled.
            if torch.any(u.reset_buf) and getattr(cfg_task, "if_sbc", False):
                u.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(u.ep_succeeded) / u.num_envs,
                    cfg_task=cfg_task,
                    curriculum_height_bound=u.curriculum_height_bound,
                    curriculum_height_step=u.curriculum_height_step,
                    curr_max_disp=u.curr_max_disp,
                )
                u.extras["curr_max_disp"] = u.curr_max_disp

            return rew_buf

        return get_rewards

    def _make_reset_idx(self):
        """Wrap ``_reset_idx``: FORGE per-reset randomization (forge_env.py:318-330), no dead-zone."""
        u = self._u
        orig = self._orig_reset_idx
        dev = u.device
        n = u.num_envs

        def reset_idx(env_ids):
            orig(env_ids)
            # All envs reset together (AutoMate assumption); resample full-width like FORGE.
            u.contact_penalty_thresholds = self._sample_contact_thresholds()
            u.force_sensor_world_smooth[:, :] = 0.0
            u.flip_quats = torch.ones((n,), dtype=torch.float32, device=dev)
            u.flip_quats[torch.rand(n) > 0.5] = -1.0

        return reset_idx
