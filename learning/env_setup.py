"""Shared Isaac Lab environment construction for the trainer and tooling.

The full env-build pipeline (parse env_cfg -> AutoMate Forge adapter -> user
overrides -> ctrl/obs_rand push -> optional recorder camera -> optional contact
sensor -> ``gym.make`` -> adapter/control/contact/scorer wrappers) lives here so
``learning/runner.py`` (training) and ``init_calib.py`` (calibration GIFs) build
*exactly* the same environment with no drift.

``build_env`` must be imported only AFTER ``AppLauncher`` has booted Omniverse:
its heavy ``isaaclab`` / ``gymnasium`` / wrapper imports are deferred into the
function body, mirroring how ``runner.main()`` defers them.
"""

from __future__ import annotations

import dataclasses


def _apply_env_cfg_overrides(env_cfg, overrides: dict) -> None:
    """Apply ``runner_cfg.env_cfg_overrides`` to an Isaac Lab env_cfg in place.

    Each key is a dotted path resolved segment-by-segment against ``env_cfg``.
    A segment is resolved by ``getattr`` on a normal object, or by key lookup
    when the current node is a ``dict`` (AutoMate stores its per-task config in
    ``env_cfg.tasks``, a plain dict — e.g. ``tasks.insertion.if_sbc`` walks the
    ``"insertion"`` key, then sets the ``if_sbc`` attribute on that task cfg).
    The final segment is set with ``setattr`` (or item-assignment for a dict).
    Strict — a missing intermediate or leaf segment raises with the full path so
    typos fail loudly instead of being silently absorbed by the dataclass/dict.
    """
    if not overrides:
        return

    def _has(node, key):
        return key in node if isinstance(node, dict) else hasattr(node, key)

    def _get(node, key):
        return node[key] if isinstance(node, dict) else getattr(node, key)

    def _set(node, key, value):
        if isinstance(node, dict):
            node[key] = value
        else:
            setattr(node, key, value)

    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ValueError(
                f"env_cfg_overrides keys must be non-empty dotted strings, got {dotted_path!r}"
            )
        parts = dotted_path.split(".")
        target = env_cfg
        for segment in parts[:-1]:
            if not _has(target, segment):
                raise AttributeError(
                    f"env_cfg_overrides: '{dotted_path}' — '{segment}' not found on "
                    f"{type(target).__name__}"
                )
            target = _get(target, segment)
        leaf = parts[-1]
        if not _has(target, leaf):
            raise AttributeError(
                f"env_cfg_overrides: '{dotted_path}' — '{leaf}' not found on "
                f"{type(target).__name__}"
            )
        _set(target, leaf, value)
        print(f"[runner] env_cfg override: {dotted_path} = {value}")


def build_env(
    args,
    runner_cfg,
    sac_cfg,
    ppo_cfg,
    controller_cfg,
    noise_cfg,
    sensor_cfg,
    agent_type,
    *,
    force_camera: bool = False,
):
    """Construct the full wrapped Isaac Lab env exactly as training does.

    Returns ``(env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs)``.

    ``force_camera=True`` injects the recorder ``TiledCamera`` regardless of the
    config's ``recorder.enabled`` (used by ``init_calib.py`` so a camera is always
    present); camera parameters are read from ``sac_cfg.recorder`` (the shared
    ``RecorderCfg``). Must be called AFTER ``AppLauncher`` has booted.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg
    from wrappers import (
        default_wrapper_for_task,
        fallback_wrapper_name,
        make_wrapper,
    )

    # ---- env ----
    total_envs = runner_cfg.num_envs * runner_cfg.num_agents
    env_cfg = parse_env_cfg(runner_cfg.task, device=args.device, num_envs=total_envs)

    # AutoMate Assembly drop-in: mutate env_cfg so the env spawns FORGE-compatibly (robot USD
    # with a force_sensor body, Forge-like obs_rand, FORGE reward scales). Must run before the
    # ctrl/obs_rand copy loops + obs-noise print below and before gym.make. The matching
    # AutoMateForgeAdapter wrapper is attached right after gym.make (innermost).
    is_automate_assembly = runner_cfg.task.startswith("Isaac-AutoMate-Assembly-")
    if is_automate_assembly:
        from wrappers.sensors.automate_forge_adapter import install_automate_forge_adapter
        install_automate_forge_adapter(env_cfg, runner_cfg)

    # Apply user env_cfg_overrides AFTER the Forge adapter writes its defaults so the user wins
    # for the fields the adapter sets (ft_smoothing_factor, the task reward-penalty scales, etc.).
    # Placed BEFORE the ctrl/obs_rand copy loops below so controller_cfg/noise_cfg stay the
    # authoritative source for env_cfg.ctrl / env_cfg.obs_rand (their dedicated config knobs),
    # while env_cfg_overrides owns everything else.
    _apply_env_cfg_overrides(env_cfg, runner_cfg.env_cfg_overrides)

    # Fragile peg / efficient reset (Forge/Factory/AutoMate peg insertion only). The
    # threshold-force cap runs AFTER user overrides so break_force is the hard ceiling on the
    # FORGE contact-penalty threshold range, regardless of any user-set range.
    if runner_cfg.fragile_peg_enabled or runner_cfg.efficient_reset_enabled:
        _is_forge = runner_cfg.task.startswith("Isaac-Forge-")
        _is_factory = runner_cfg.task.startswith("Isaac-Factory-")
        if not (_is_forge or _is_factory or is_automate_assembly):
            raise ValueError(
                "runner_cfg.fragile_peg_enabled / efficient_reset_enabled require a "
                f"Forge/Factory/AutoMate-Assembly task, but task is {runner_cfg.task!r}."
            )
        if runner_cfg.fragile_peg_enabled and not (_is_forge or is_automate_assembly):
            raise ValueError(
                "runner_cfg.fragile_peg_enabled requires a force-sensor-bearing task "
                "(Isaac-Forge-* or AutoMate-Assembly); stock Factory has no force sensing, "
                f"but task is {runner_cfg.task!r}."
            )
        # Cap the FORGE per-env threshold force at break_force: both the obs force_threshold
        # and the contact-penalty reward read contact_penalty_thresholds, which must never
        # exceed the force that breaks the peg. No-op for tasks without the field.
        if (
            runner_cfg.fragile_peg_enabled
            and runner_cfg.break_force > 0.0
            and hasattr(env_cfg, "task")
            and hasattr(env_cfg.task, "contact_penalty_threshold_range")
        ):
            _rng = list(env_cfg.task.contact_penalty_threshold_range)
            _rng[1] = min(_rng[1], runner_cfg.break_force)
            _rng[0] = min(_rng[0], _rng[1])
            env_cfg.task.contact_penalty_threshold_range = _rng
            print(
                f"[runner] fragile peg: capped contact_penalty_threshold_range to {_rng} "
                f"(break_force={runner_cfg.break_force} N)."
            )

    # Push the controller config's ctrl fields onto env_cfg.ctrl so the base controller
    # (and factory_control_utils) see the YAML-overridable gains. ControlCfg subclasses the
    # env's ForgeCtrlCfg, so it carries every ctrl field by inheritance — copy exactly the
    # fields the env's ctrl cfg declares (avoids any hardcoded field list).
    if hasattr(env_cfg, "ctrl"):
        for f in dataclasses.fields(type(env_cfg.ctrl)):
            if hasattr(controller_cfg, f.name):
                setattr(env_cfg.ctrl, f.name, getattr(controller_cfg, f.name))
        # full_orientation_control is a ControlCfg-only field (not declared on the base
        # ForgeCtrlCfg), so the field-copy loop above skips it. Push it explicitly so both
        # factory_control_utils.compute_ctrl_targets and the AutoMate obs adapter read it
        # off env_cfg.ctrl.
        env_cfg.ctrl.full_orientation_control = controller_cfg.full_orientation_control

    # Push the noise config onto env_cfg.obs_rand. NoiseCfg subclasses the env's
    # ForgeObsRandCfg, so it carries every obs-noise field by inheritance — copy exactly the
    # fields the env's obs_rand cfg declares (avoids any hardcoded field list). Setting any
    # level to 0 / [0, 0, 0] disables that noise term.
    if hasattr(env_cfg, "obs_rand"):
        for f in dataclasses.fields(type(env_cfg.obs_rand)):
            if hasattr(noise_cfg, f.name):
                setattr(env_cfg.obs_rand, f.name, getattr(noise_cfg, f.name))
        print(
            f"[runner] obs noise: fingertip_pos={env_cfg.obs_rand.fingertip_pos} "
            f"fingertip_rot_deg={env_cfg.obs_rand.fingertip_rot_deg} "
            f"ft_force={env_cfg.obs_rand.ft_force} "
            f"fixed_asset_pos={env_cfg.obs_rand.fixed_asset_pos}"
        )

    # Resolve the concrete direct-API env class ONCE (from the gym registry). Both the
    # recorder-camera injection and the contact-sensor install patch this class's
    # ``_setup_scene``, and they MUST target the SAME class or the two patches won't compose.
    # (Forge inherits ``FactoryEnv._setup_scene``; patching ``ForgeEnv`` for the camera while
    # patching ``FactoryEnv`` for the sensor lets the subclass shim shadow + bypass the base
    # shim, so the contact sensor never registers.) Different direct envs (FactoryEnv, ForgeEnv,
    # AutoMate's AssemblyEnv) are distinct classes, so resolve the concrete one from the registry.
    camera_on = (agent_type == "sac" and sac_cfg.recorder.enabled) or force_camera
    contact_enabled = sensor_cfg.contact.enabled
    _env_cls = None
    if camera_on or contact_enabled:
        import importlib

        _spec = gym.spec(runner_cfg.task)
        _entry = _spec.entry_point
        if not isinstance(_entry, str) or ":" not in _entry:
            raise RuntimeError(
                f"cannot resolve the env class to patch for task {runner_cfg.task!r}; "
                f"gym entry_point={_entry!r} is not a 'module:Class' string."
            )
        _mod_name, _cls_name = _entry.split(":")
        _env_cls = getattr(importlib.import_module(_mod_name), _cls_name)
        if not hasattr(_env_cls, "_setup_scene"):
            raise RuntimeError(
                f"env class {_env_cls.__name__} (task {runner_cfg.task!r}) has no "
                "``_setup_scene`` to patch."
            )

    # Inject the recorder camera. Direct envs like Factory/Forge construct their scenes
    # manually inside ``_setup_scene`` and never read ``env_cfg.scene.<sensor>`` cfg attrs —
    # so the manager-based auto-discovery path (e.g. cartpole_camera) doesn't fire for them.
    # The working pattern is: spawn the TiledCamera prim under ``/World/envs/env_.*/Camera``
    # BEFORE ``self.scene.clone_environments(...)`` (so the clone replicates env_0 -> all envs
    # including the camera), then register with ``scene.sensors``. We bolt that on by wrapping
    # ``_setup_scene`` and intercepting the FIRST ``clone_environments`` call via a one-shot
    # instance-level shim: (1) spawn TiledCamera on env_0, (2) call the original clone, (3)
    # register the sensor. ``force_camera`` (init_calib) injects regardless of recorder.enabled.
    if camera_on:
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.sim.spawners.sensors import PinholeCameraCfg
        from wrappers.recording import CAMERA_KEY as _RECORDER_CAMERA_KEY

        rec = sac_cfg.recorder
        cam_cfg = TiledCameraCfg(
            prim_path=f"/World/envs/env_.*/{_RECORDER_CAMERA_KEY}",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(rec.camera_pos),
                rot=tuple(rec.camera_quat),
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=float(rec.focal_length),
                focus_distance=float(rec.focus_distance),
                horizontal_aperture=float(rec.horizontal_aperture),
                clipping_range=tuple(rec.clipping_range),
            ),
            width=int(rec.width),
            height=int(rec.height),
            update_period=0.0,
        )

        _original_setup_scene = _env_cls._setup_scene

        def _patched_setup_scene(self):
            # Install a one-shot instance-level shim on this scene's
            # ``clone_environments`` so the camera spawn happens between the
            # env's manual robot/asset spawns and its clone call (which
            # replicates env_0 -> env_1..N, including the camera).
            _orig_clone = self.scene.clone_environments

            def _shim_clone(*args, **kwargs):
                # Restore first to make this fire exactly once.
                self.scene.clone_environments = _orig_clone
                print(
                    f"[recorder] spawning TiledCamera at {cam_cfg.prim_path} "
                    "before clone_environments…",
                    flush=True,
                )
                cam = TiledCamera(cam_cfg)
                ret = _orig_clone(*args, **kwargs)
                self.scene._sensors[_RECORDER_CAMERA_KEY] = cam
                print(
                    f"[recorder] env clone complete; camera registered as "
                    f"scene.sensors[{_RECORDER_CAMERA_KEY!r}].",
                    flush=True,
                )
                return ret

            self.scene.clone_environments = _shim_clone
            return _original_setup_scene(self)

        _env_cls._setup_scene = _patched_setup_scene
        print(
            f"[runner] recorder enabled: TiledCamera '{_RECORDER_CAMERA_KEY}' "
            f"({rec.width}x{rec.height}) will spawn inside {_env_cls.__name__}._setup_scene "
            f"pre-clone; record_every_k_resets={rec.record_every_k_resets}"
        )

    # Contact sensor: must be registered with the scene DURING env construction (before
    # InteractiveScene.clone_environments), so patch the env class's _setup_scene here, before
    # gym.make. Forge/Factory/AutoMate only — the sensor mounts on the held/fixed peg-insertion
    # assets. The runtime ContactSensorWrapper (added below) reads it.
    if contact_enabled:
        if not (runner_cfg.task.startswith("Isaac-Forge-")
                or runner_cfg.task.startswith("Isaac-Factory-")
                or is_automate_assembly):
            raise ValueError(
                f"sensor_cfg.contact.enabled=True requires a Forge/Factory/AutoMate-Assembly "
                f"task (held/fixed peg-insertion assets), but task is {runner_cfg.task!r}."
            )
        # ContactSensor reads per-env bodies via the USD stage (create_rigid_body_view),
        # but Factory/Forge default to clone_in_fabric=True, which keeps the cloned
        # env_1..N bodies in Fabric only — so the sensor finds just env_0 and fails its
        # body-count check. Fabric cloning is incompatible with contact sensors, so turn
        # it off here (a no-op for the camera recorder above).
        if getattr(env_cfg.scene, "clone_in_fabric", False):
            env_cfg.scene.clone_in_fabric = False
            print("[runner] contact sensor enabled: forcing scene.clone_in_fabric=False "
                  "(fabric clones are invisible to the contact sensor's body view).")
        # Patch the SAME concrete env class the camera injection used (resolved above) so the
        # two _setup_scene shims compose. Patching a base class (e.g. FactoryEnv) while the
        # camera patched the concrete subclass (ForgeEnv) would let the subclass shim bypass
        # this one, and the contact sensor would never register.
        from wrappers.sensors.contact_sensor_wrapper import install_contact_sensor
        install_contact_sensor(sensor_cfg.contact, env_class=_env_cls)

    # Optional in-gripper grasp-rotation offset (Forge/Factory peg insertion only): patch
    # get_handheld_asset_relative_pose so the peg is grasped at a roll/pitch/yaw tilt relative to
    # the gripper — 'random' = per-reset random, 'fixed' = constant signed offset. Must run before
    # gym.make (it patches the env class). 'none' => not installed => upstream behavior.
    if runner_cfg.grasp_rot_mode != "none":
        if not (runner_cfg.task.startswith("Isaac-Forge-")
                or runner_cfg.task.startswith("Isaac-Factory-")):
            raise ValueError(
                f"runner_cfg.grasp_rot_mode={runner_cfg.grasp_rot_mode!r} requires a Forge/Factory "
                "peg-insertion task (it patches FactoryEnv.get_handheld_asset_relative_pose), but "
                f"task is {runner_cfg.task!r}."
            )
        from wrappers.sensors.grasp_tilt_wrapper import install_grasp_rot_randomization
        install_grasp_rot_randomization(runner_cfg.rel_grasp_rot_init_deg, runner_cfg.grasp_rot_mode)

    env = gym.make(runner_cfg.task, cfg=env_cfg, render_mode=None)

    # AutoMate-as-Forge adapter: must wrap the raw env BEFORE the control wrapper, since the
    # hybrid controller asserts env.unwrapped.force_sensor_smooth at construction. Installs the
    # FT sensor / obs noise / FORGE reward onto env.unwrapped and grows the obs/state spaces.
    if is_automate_assembly:
        from wrappers.sensors.automate_forge_adapter import AutoMateForgeAdapter
        env = AutoMateForgeAdapter(env, runner_cfg)
        print("[runner] AutoMate-as-Forge adapter attached (innermost, before control wrapper).")

    # Control wrapper (must wrap the raw gym env BEFORE the scorer/IsaacLabWrapper so the
    # expanded action/obs/state spaces are visible when the models are built below). The
    # control wrapper monkeypatches env.unwrapped's control hooks and grows the action
    # space; "pose" uses the base controller and adds nothing.
    control_type = controller_cfg.control_type
    # For hybrid control types, keep a reference to the control wrapper so the runner
    # can read its selection/position/force action layout for the hybrid actor.
    ctrl_wrapper = None
    if control_type == "pose-VICES":
        from wrappers.controllers.vic_pose_wrapper import VICPoseWrapper
        env = VICPoseWrapper(env, controller_cfg)
        print(f"[runner] control wrapper: VICPoseWrapper (control_type={control_type})")
    elif control_type == "hybrid":
        from wrappers.controllers.hybrid_force_position_wrapper import HybridForcePositionWrapper
        env = HybridForcePositionWrapper(env, controller_cfg, num_agents=runner_cfg.num_agents)
        ctrl_wrapper = env
        print(f"[runner] control wrapper: HybridForcePositionWrapper (control_type={control_type})")
    elif control_type == "hybrid-vic":
        from wrappers.controllers.hybrid_vic_wrapper import HybridVICWrapper
        env = HybridVICWrapper(env, controller_cfg, num_agents=runner_cfg.num_agents)
        ctrl_wrapper = env
        print(f"[runner] control wrapper: HybridVICWrapper (control_type={control_type})")
    elif control_type == "ctrl-action-interface":
        from wrappers.controllers.ctrl_action_interface import CtrlActionInterfaceWrapper
        env = CtrlActionInterfaceWrapper(env, controller_cfg, num_agents=runner_cfg.num_agents)
        ctrl_wrapper = env
        print(
            f"[runner] control wrapper: CtrlActionInterfaceWrapper "
            f"(control_type={control_type}, gain_mapping={controller_cfg.gain_mapping})"
        )
    elif control_type != "pose":
        raise ValueError(f"[runner] unknown controller_cfg.control_type: {control_type!r}")

    # Efficient reset (installed BEFORE fragile, just OUTSIDE the control wrapper): patches
    # _reset_idx so the runtime chain is control._wrapped -> efficient._wrapped -> reset, i.e.
    # the control wrapper's per-env EMA/VIC reset still runs for broken envs. On a partial
    # reset it teleports broken envs to a cached donor state instead of running Factory's
    # all-envs settling reset. Required whenever fragile pegs are on.
    if runner_cfg.efficient_reset_enabled:
        from wrappers.sensors.efficient_reset_wrapper import EfficientResetWrapper
        env = EfficientResetWrapper(env)
        print("[runner] efficient-reset wrapper attached (per-env teleport reset).")

    # Fragile peg (OUTSIDE efficient reset, INSIDE the scorer): patches _get_dones to break
    # (terminate) any env whose smoothed contact force reaches break_force. The break itself
    # triggers a per-env reset handled by the efficient-reset wrapper above.
    if runner_cfg.fragile_peg_enabled:
        from wrappers.sensors.fragile_object_wrapper import FragileObjectWrapper
        env = FragileObjectWrapper(
            env, break_force=runner_cfg.break_force, num_agents=runner_cfg.num_agents
        )
        print(
            f"[runner] fragile-peg wrapper attached (break_force={runner_cfg.break_force} N)."
        )

    # Contact-sensor wrapper (logging-only): sits at the same layer as the control
    # wrappers (inside the scorer/IsaacLabWrapper) so the per-axis in-contact flags it
    # writes into extras['to_log'] each step are forwarded to TensorBoard by the
    # scorer's _forward_to_log. Does not change the action/obs/state spaces.
    if contact_enabled:
        from wrappers.sensors.contact_sensor_wrapper import ContactSensorWrapper
        env = ContactSensorWrapper(env, sensor_cfg.contact)
        print(
            f"[runner] contact-sensor wrapper attached "
            f"(threshold={sensor_cfg.contact.contact_force_threshold} N, "
            f"log_contact_state={sensor_cfg.contact.log_contact_state})"
        )

    # Energy-metrics wrapper (logging-only): same layer as the contact sensor — inside
    # the scorer (so its energy_metrics/* entries in extras['to_log'] are forwarded) and
    # outside the control wrapper (so joint_torque reflects the applied control). Does
    # not change the action/obs/state spaces.
    if sensor_cfg.energy.enabled:
        from wrappers.sensors.energy_metrics_wrapper import EnergyMetricsWrapper
        env = EnergyMetricsWrapper(env, sensor_cfg.energy)
        print("[runner] energy-metrics wrapper attached (energy_metrics/* tab)")

    # Always wrap with a subclass of skrl's IsaacLabWrapper. The wrapper is chosen
    # by task prefix (e.g. "forge"/"factory"/"lift"); when no task-specific wrapper
    # matches, fall back to RewardDecompositionWrapper so every manager-based task
    # gets per-env per-term reward logging in per-episode units (matches the units
    # of `Reward / Total reward (mean)`). Direct-API envs without a reward_manager
    # fall through gracefully — the hook is a no-op.
    selected_wrapper = default_wrapper_for_task(runner_cfg.task) or fallback_wrapper_name()
    env = make_wrapper(selected_wrapper, env)

    # Contact-state input append (optional): grow obs/state spaces with the per-axis
    # contact bool so the policy/critic can condition on it. MUST wrap before the spaces
    # are read below so the model builder + preprocessors see the grown dims.
    _cc = sensor_cfg.contact
    if _cc.append_to_policy_obs or _cc.append_to_critic_state:
        if not _cc.enabled:
            raise ValueError(
                "sensor_cfg.contact.append_to_policy_obs / append_to_critic_state require "
                "sensor_cfg.contact.enabled=True (the contact sensor provides in_contact)."
            )
        from wrappers.sensors.contact_append_wrapper import ContactAppendWrapper
        from wrappers.sensors.contact_sensor_wrapper import CONTACT_DIM
        env = ContactAppendWrapper(
            env,
            append_to_policy_obs=_cc.append_to_policy_obs,
            append_to_critic_state=_cc.append_to_critic_state,
            contact_dim=CONTACT_DIM,
        )
        print(
            f"[runner] contact append: policy_obs={_cc.append_to_policy_obs} "
            f"critic_state={_cc.append_to_critic_state} (+{CONTACT_DIM} dims each)"
        )

    return env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs
