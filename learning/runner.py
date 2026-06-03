"""Train/eval entry point for BlockSimba-SAC on Isaac Lab tasks.

Mostly skrl boilerplate for SAC; expected to be modified as the project grows.
The Isaac Lab `AppLauncher` must boot before any `isaaclab.envs` / `isaaclab_tasks`
imports — that's why those imports live inside `main()` after `app_launcher.app`.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Project root (parent of learning/) — used to anchor the default --config path so
# the runner works regardless of the user's CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BlockSimba-SAC trainer/eval")
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML config file. Provides runner_cfg/sac_cfg/ppo_cfg/model_cfg. "
             f"Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Overrides sac_cfg.experiment.experiment_name from --config.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Log root directory. The final per-run output dir is "
             "<logdir>/<sac_cfg.experiment.directory>/<experiment_name>. "
             "If sac_cfg.experiment.directory equals the basename of --logdir "
             "(legacy: both set to 'runs'), the family-subdir level is "
             "collapsed to keep pre-existing experiment trees intact. "
             "Relative paths are resolved against the project root. Defaults "
             "to 'runs/' when omitted.",
    )
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Folder path. Multi-agent: parent containing 0/, 1/, ... subdirs. "
             "Single-agent: a folder with checkpoints/ckpt_<step>.pt directly. "
             "Mode is auto-detected.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific step to load. If omitted, the latest ckpt found is used.",
    )
    # All of the following used to be required CLI flags. They now default to
    # None and fall back to runner_cfg in YAML when omitted. Still overridable
    # from CLI for one-off experiments.
    parser.add_argument("--task", type=str, default=None,
                        help="Overrides runner_cfg.task. e.g. Isaac-Lift-Cube-Franka-v0.")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Overrides runner_cfg.num_envs (envs PER agent).")
    parser.add_argument("--num_agents", type=int, default=None,
                        help="Overrides runner_cfg.num_agents (block-parallel agents).")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Overrides runner_cfg.total_timesteps in train mode "
                             "(transitions PER AGENT).")
    parser.add_argument("--eval_timesteps", type=int, default=None,
                        help="Overrides runner_cfg.eval_timesteps in eval mode.")
    parser.add_argument("--memory_size", type=int, default=None,
                        help="Overrides runner_cfg.memory_size (replay buffer per agent).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Overrides runner_cfg.seed. -1 means non-deterministic.")
    AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device
    return parser


def _apply_env_cfg_overrides(env_cfg, overrides: dict) -> None:
    """Apply ``runner_cfg.env_cfg_overrides`` to an Isaac Lab env_cfg in place.

    Each key is a dotted path resolved by ``getattr`` against ``env_cfg``; the
    final segment is set with ``setattr``. Strict — if any intermediate or leaf
    attribute is missing, raises ``AttributeError`` with the full path so typos
    fail loudly instead of being silently absorbed by the dataclass.
    """
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ValueError(
                f"env_cfg_overrides keys must be non-empty dotted strings, got {dotted_path!r}"
            )
        parts = dotted_path.split(".")
        target = env_cfg
        for segment in parts[:-1]:
            if not hasattr(target, segment):
                raise AttributeError(
                    f"env_cfg_overrides: '{dotted_path}' — '{segment}' not found on "
                    f"{type(target).__name__}"
                )
            target = getattr(target, segment)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise AttributeError(
                f"env_cfg_overrides: '{dotted_path}' — '{leaf}' not found on "
                f"{type(target).__name__}"
            )
        setattr(target, leaf, value)
        print(f"[runner] env_cfg override: {dotted_path} = {value}")


def main() -> None:
    args = build_parser().parse_args()

    # If the chosen YAML has ``sac_cfg.recorder.enabled: true``, IsaacLab will
    # refuse to spawn the recorder TiledCamera without ``--enable_cameras``.
    # Peek the YAML BEFORE booting AppLauncher and force the flag on so the
    # user doesn't have to remember to pass it.
    try:
        import yaml as _yaml
        with open(args.config) as _f:
            _peek = _yaml.safe_load(_f) or {}
        _agent_type_peek = str(_peek.get("runner_cfg", {}).get("agent_type", "sac")).lower()
        _cfg_key = "ppo_cfg" if _agent_type_peek == "ppo" else "sac_cfg"
        if (
            _peek.get(_cfg_key, {})
                 .get("recorder", {})
                 .get("enabled", False)
            and not getattr(args, "enable_cameras", False)
        ):
            args.enable_cameras = True
            print(f"[runner] {_cfg_key}.recorder.enabled=true — forcing --enable_cameras on.")
    except Exception as _e:
        # If the YAML is malformed we'll surface the error during ConfigManager
        # load below; don't block AppLauncher here.
        print(f"[runner] could not peek YAML for recorder.enabled: {_e!r}")

    # Boot Omniverse before any isaaclab.envs imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Silence IsaacLab's per-call "quat_rotate{,_inverse} will be deprecated" spam.
    # These are emitted via omni.log on the "isaaclab.utils.math" channel (not Python
    # warnings), so raise that channel's threshold to ERROR. omni is only importable
    # after AppLauncher boots.
    try:
        import omni.log

        omni.log.get_log().set_channel_level(
            "isaaclab.utils.math", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
        )
    except Exception as e:  # pragma: no cover - best-effort log tidy-up
        print(f"[runner] could not raise isaaclab.utils.math log level: {e!r}", flush=True)

    # Project root on sys.path so `models.block_simba` and `learning.sac` resolve
    # regardless of CWD.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    import dataclasses
    from memory.multi_random import MultiRandomMemory
    from models.block_simba import (
        BlockSimBaActor,
        BlockSimBaQCritic,
        BlockSimBaValueCritic,
        HybridControlBlockSimBaActor,
    )
    from learning.sac import SAC
    from learning.ppo import PPO
    from learning.losses import AuxLossManager
    from configs.manager import ConfigManager
    from wrappers import (
        default_wrapper_for_task,
        fallback_wrapper_name,
        make_wrapper,
    )

    # Load all registered configs from a single YAML file (defaults to configs/default.yaml).
    loaded = ConfigManager.load(args.config)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    ppo_cfg = loaded["ppo_cfg"]
    model_cfg = loaded["model_cfg"]
    controller_cfg = loaded["controller_cfg"]
    sensor_cfg = loaded["sensor_cfg"]
    # Auxiliary-loss switches (which extra losses are on and their per-target
    # weights). Absent loss_cfg section -> all-off default -> vanilla SAC.
    loss_cfg = loaded["loss_cfg"]

    # Which learning algorithm to run. ``active_cfg`` supplies the fields the runner
    # plumbs that exist on both SAC_CFG and PPO_CFG (experiment dir, observation
    # preprocessor, rewards_shaper, recorder, mixed_precision).
    agent_type = str(runner_cfg.agent_type).lower()
    if agent_type not in ("sac", "ppo"):
        raise ValueError(
            f"runner_cfg.agent_type must be 'sac' or 'ppo', got {runner_cfg.agent_type!r}"
        )
    active_cfg = sac_cfg if agent_type == "sac" else ppo_cfg

    # CLI > YAML for runner-level fields. Apply overrides to the loaded RunnerCfg so
    # the dump-config-to-disk step at the end records the values actually used.
    if args.task is not None:            runner_cfg.task = args.task
    if args.num_envs is not None:        runner_cfg.num_envs = args.num_envs
    if args.num_agents is not None:      runner_cfg.num_agents = args.num_agents
    if args.total_timesteps is not None: runner_cfg.total_timesteps = args.total_timesteps
    if args.eval_timesteps is not None:  runner_cfg.eval_timesteps = args.eval_timesteps
    if args.memory_size is not None:     runner_cfg.memory_size = args.memory_size
    if args.seed is not None:            runner_cfg.seed = args.seed

    # In eval mode the trainer runs for runner_cfg.eval_timesteps instead of total_timesteps.
    effective_total_timesteps = (
        runner_cfg.eval_timesteps if args.mode == "eval" else runner_cfg.total_timesteps
    )

    # YAML-friendly reward shaping: rewards_shaper_scale (float) -> multiplicative lambda.
    # Mirrors skrl runner's convention. Direct assignment of rewards_shaper still wins
    # if a programmatic caller set it before this point.
    if active_cfg.rewards_shaper is None and active_cfg.rewards_shaper_scale is not None:
        scale = float(active_cfg.rewards_shaper_scale)
        active_cfg.rewards_shaper = lambda rewards, *args, **kwargs: rewards * scale
    elif active_cfg.rewards_shaper is not None and active_cfg.rewards_shaper_scale is not None:
        raise ValueError(
            f"Both {agent_type}_cfg.rewards_shaper and {agent_type}_cfg.rewards_shaper_scale "
            "are set; use exactly one."
        )

    set_seed(runner_cfg.seed if runner_cfg.seed >= 0 else None)

    # ---- env ----
    total_envs = runner_cfg.num_envs * runner_cfg.num_agents
    env_cfg = parse_env_cfg(runner_cfg.task, device=args.device, num_envs=total_envs)
    _apply_env_cfg_overrides(env_cfg, runner_cfg.env_cfg_overrides)

    # Push the controller config's ctrl fields onto env_cfg.ctrl so the base controller
    # (and factory_control_utils) see the YAML-overridable gains. ControlCfg subclasses the
    # env's ForgeCtrlCfg, so it carries every ctrl field by inheritance — copy exactly the
    # fields the env's ctrl cfg declares (avoids any hardcoded field list).
    if hasattr(env_cfg, "ctrl"):
        for f in dataclasses.fields(type(env_cfg.ctrl)):
            if hasattr(controller_cfg, f.name):
                setattr(env_cfg.ctrl, f.name, getattr(controller_cfg, f.name))

    # Inject the recorder camera. Direct envs like Factory/Forge construct
    # their scenes manually inside ``FactoryEnv._setup_scene`` and never read
    # ``env_cfg.scene.<sensor>`` cfg attrs — so the auto-discovery path used
    # by manager-based envs (e.g. cartpole_camera) doesn't fire for them.
    # Cartpole's working pattern is: spawn the TiledCamera USD prim under
    # ``/World/envs/env_.*/Camera`` BEFORE ``self.scene.clone_environments(...)``
    # runs (so the clone replicates env_0 -> env_1..N including the camera),
    # then register with ``scene.sensors``. We bolt that pattern onto Factory
    # by wrapping ``FactoryEnv._setup_scene``: run the original (which spawns
    # the table/robot/peg AND calls clone_environments at its tail), but
    # intercept the FIRST ``self.scene.clone_environments`` call from inside
    # the original via a one-shot instance-level shim that does:
    #   1. spawn TiledCamera on env_0
    #   2. call original clone_environments (replicates env_0 -> all envs)
    #   3. register the sensor with scene._sensors
    if agent_type == "sac" and sac_cfg.recorder.enabled:
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.sim.spawners.sensors import PinholeCameraCfg
        from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
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

        _original_factory_setup_scene = FactoryEnv._setup_scene

        def _patched_factory_setup_scene(self):
            # Install a one-shot instance-level shim on this scene's
            # ``clone_environments`` so the camera spawn happens between
            # Factory's manual robot/peg/table spawns and its clone call.
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
            return _original_factory_setup_scene(self)

        FactoryEnv._setup_scene = _patched_factory_setup_scene
        print(
            f"[runner] recorder enabled: TiledCamera '{_RECORDER_CAMERA_KEY}' "
            f"({rec.width}x{rec.height}) will spawn inside FactoryEnv._setup_scene "
            f"pre-clone; record_every_k_resets={rec.record_every_k_resets}"
        )

    # Contact sensor: must be registered with the scene DURING env construction
    # (before InteractiveScene.clone_environments), so patch FactoryEnv._setup_scene
    # here, before gym.make. Forge/Factory only — the sensor mounts on the held/fixed
    # peg-insertion assets. The runtime ContactSensorWrapper (added below) reads it.
    contact_enabled = sensor_cfg.contact.enabled
    if contact_enabled:
        if not (runner_cfg.task.startswith("Isaac-Forge-")
                or runner_cfg.task.startswith("Isaac-Factory-")):
            raise ValueError(
                f"sensor_cfg.contact.enabled=True requires a Forge/Factory task (held/fixed "
                f"peg-insertion assets), but task is {runner_cfg.task!r}."
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
        from wrappers.sensors.contact_sensor_wrapper import install_contact_sensor
        install_contact_sensor(sensor_cfg.contact)

    env = gym.make(runner_cfg.task, cfg=env_cfg, render_mode=None)

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

    # Always wrap with a subclass of skrl's IsaacLabWrapper. The wrapper is chosen
    # by task prefix (e.g. "forge"/"factory"/"lift"); when no task-specific wrapper
    # matches, fall back to RewardDecompositionWrapper so every manager-based task
    # gets per-env per-term reward logging in per-episode units (matches the units
    # of `Reward / Total reward (mean)`). Direct-API envs without a reward_manager
    # fall through gracefully — the hook is a no-op.
    selected_wrapper = default_wrapper_for_task(runner_cfg.task) or fallback_wrapper_name()
    env = make_wrapper(selected_wrapper, env)

    device = torch.device(args.device)
    n_agents = runner_cfg.num_agents
    obs_space = env.observation_space
    act_space = env.action_space

    # Asymmetric actor-critic detection: skrl's IsaacLabWrapper exposes `state_space`
    # (and `state()` returning a tensor) when the underlying env returns a
    # {"policy": ..., "critic": ...} dict obs. When present, the critic is built
    # with the (typically larger) state vector and SAC stores both obs/states in
    # memory. Strict: detection is automatic; presence of state_space ALSO
    # requires env.state() to be non-None at runtime, which SAC enforces on the
    # first record_transition call.
    state_space = getattr(env, "state_space", None)
    if state_space is not None:
        print(
            f"[runner] asymmetric actor-critic detected: obs_dim={obs_space.shape[0]}, "
            f"state_dim={state_space.shape[0]} (critic uses state)"
        )

    # Observation preprocessor: YAML carries the class as a string name; the agent
    # resolves it via the registry. We inject the runtime kwargs (size, device) here
    # since YAML can't carry a Box space or torch.device object. Skip if user
    # explicitly set kwargs already.
    if active_cfg.observation_preprocessor is not None:
        if not isinstance(active_cfg.observation_preprocessor_kwargs, dict):
            active_cfg.observation_preprocessor_kwargs = {}
        active_cfg.observation_preprocessor_kwargs.setdefault("size", obs_space)
        active_cfg.observation_preprocessor_kwargs.setdefault("device", device)

    # PPO value preprocessor (single shared RunningStandardScaler over scalar values).
    if agent_type == "ppo" and ppo_cfg.value_preprocessor is not None:
        if not isinstance(ppo_cfg.value_preprocessor_kwargs, dict):
            ppo_cfg.value_preprocessor_kwargs = {}
        ppo_cfg.value_preprocessor_kwargs.setdefault("size", 1)
        ppo_cfg.value_preprocessor_kwargs.setdefault("device", device)

    # ---- models ----
    actor_kwargs = dataclasses.asdict(model_cfg.actor)
    critic_kwargs = dataclasses.asdict(model_cfg.critic)

    # selection_distribution / selection_init_bias apply only to the hybrid actor;
    # pop them so they never reach the plain BlockSimBaActor (which doesn't accept them).
    selection_distribution = actor_kwargs.pop("selection_distribution", "product")
    selection_init_bias = actor_kwargs.pop("selection_init_bias", 0.0)

    # Critic input space: state_space when asymmetric, else obs_space.
    critic_input_space = state_space if state_space is not None else obs_space

    # Actor: hybrid control types get the selection-gated HybridControlBlockSimBaActor
    # (product / match), otherwise the plain squashed-Gaussian actor.
    if ctrl_wrapper is not None:
        sel_dims, pos_dims, force_dims = ctrl_wrapper.policy_selection_layout
        policy = HybridControlBlockSimBaActor(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            num_agents=n_agents,
            selection_dims=sel_dims,
            pos_component_dims=pos_dims,
            force_component_dims=force_dims,
            selection_distribution=selection_distribution,
            selection_init_bias=selection_init_bias,
            **actor_kwargs,
        )
        print(
            f"[runner] hybrid actor: style={selection_distribution!r} "
            f"selection_dims={sel_dims} pos={pos_dims} force={force_dims} "
            f"selection_init_bias={selection_init_bias} (init p≈{1/(1+2.718281828**(-selection_init_bias)):.3f})"
        )
    else:
        policy = BlockSimBaActor(
            observation_space=obs_space,
            action_space=act_space,
            device=device,
            num_agents=n_agents,
            **actor_kwargs,
        )

    if agent_type == "sac":
        def make_q():
            return BlockSimBaQCritic(
                observation_space=critic_input_space,
                action_space=act_space,
                device=device,
                num_agents=n_agents,
                **critic_kwargs,
            )

        critic_1, critic_2 = make_q(), make_q()
        target_critic_1, target_critic_2 = make_q(), make_q()

        models = {
            "policy": policy,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": target_critic_1,
            "target_critic_2": target_critic_2,
        }
    else:  # ppo — single state-value critic V(s)
        value = BlockSimBaValueCritic(
            observation_space=critic_input_space,
            action_space=act_space,
            device=device,
            num_agents=n_agents,
            **critic_kwargs,
        )
        models = {"policy": policy, "value": value}

    # ---- memory (per-agent partitioned sampling) ----
    if agent_type == "sac":
        # `--memory_size` is transitions PER AGENT (each agent owns env partition
        # [i*epa, (i+1)*epa)). Per-env depth yielding memory_size per-agent transitions
        # is memory_size // num_envs. skrl allocates one (per_env_depth, total_envs, *)
        # tensor, so physical storage is per_env_depth * total_envs.
        per_env_memory = max(1, runner_cfg.memory_size // runner_cfg.num_envs)
        realized_per_agent = per_env_memory * runner_cfg.num_envs
        print(
            f"[runner] replay memory: requested_per_agent={runner_cfg.memory_size:,}  "
            f"per_env={per_env_memory:,}  realized_per_agent={realized_per_agent:,}  "
            f"physical_storage={per_env_memory * total_envs:,} "
            f"(num_envs/agent={runner_cfg.num_envs}, n_agents={n_agents})"
        )
    else:  # ppo — on-policy rollout buffer of depth `rollouts` per env
        per_env_memory = max(1, int(ppo_cfg.rollouts))
        rollout_rows = per_env_memory * total_envs
        if rollout_rows % ppo_cfg.mini_batches != 0:
            raise ValueError(
                f"PPO requires (rollouts * total_envs) % mini_batches == 0 so sample_all "
                f"partitions evenly; got rollouts={per_env_memory} * total_envs={total_envs} "
                f"= {rollout_rows}, not divisible by mini_batches={ppo_cfg.mini_batches}."
            )
        print(
            f"[runner] rollout memory: rollouts={per_env_memory} per env  "
            f"total_rows={rollout_rows:,}  mini_batches={ppo_cfg.mini_batches}  "
            f"(rows/minibatch={rollout_rows // ppo_cfg.mini_batches:,})"
        )
    memory = MultiRandomMemory(
        memory_size=per_env_memory,
        num_envs=env.num_envs,
        num_agents=n_agents,
        device=device,
        replacement=True,
    )

    # ---- agent config (loaded above; `--memory_size` only affects SAC's replay) ----
    cfg = active_cfg
    if agent_type == "sac":
        # batch_size is PER AGENT — each agent samples cfg.batch_size from its slice.
        assert realized_per_agent >= cfg.batch_size, (
            f"per-agent replay buffer ({realized_per_agent}) < batch_size ({cfg.batch_size}); "
            f"increase memory_size (need at least {cfg.batch_size} per agent) or reduce batch_size"
        )
        print(
            f"[runner] batch_size={cfg.batch_size} per agent  "
            f"(memory.sample returns {cfg.batch_size * n_agents:,} total rows / grad step)"
        )
    # CLI > YAML > auto-generated for the run name.
    exp_name = args.experiment_name or cfg.experiment.experiment_name or f"{runner_cfg.task}_{agent_type}_N{n_agents}"
    cfg.experiment.experiment_name = exp_name

    # Final per-run output dir = <log_root>/<family>/<experiment_name>
    #   * log_root: --logdir CLI (e.g. "runs/", or absolute), default "runs/".
    #   * family:   sac_cfg.experiment.directory from YAML (e.g. "pick_block",
    #               "forge_pih"). Treated as a subdirectory under log_root so
    #               experiments of the same kind stay grouped together.
    #
    # Legacy collapse: configs that pre-date this layout set
    # sac_cfg.experiment.directory == "runs" (same as the default log root),
    # which would otherwise produce a "runs/runs/<exp>" tree. When the family
    # basename matches the log_root basename, we drop the nested level so
    # those configs keep landing at "runs/<exp>" without YAML edits.
    log_root = args.logdir if args.logdir is not None else "runs"
    family = (cfg.experiment.directory or "").strip()
    log_root_basename = os.path.basename(os.path.normpath(log_root)) if log_root else ""
    family_basename = os.path.basename(os.path.normpath(family)) if family else ""
    if not family or family_basename == log_root_basename:
        final_directory = log_root
    else:
        final_directory = os.path.join(log_root, family)
    # Anchor a relative path to the project root so runs always land at
    # <project_root>/<final_directory>/<experiment_name>, not wherever the
    # user happened to invoke from.
    if not os.path.isabs(final_directory):
        final_directory = os.path.join(_PROJECT_ROOT, final_directory)
    cfg.experiment.directory = final_directory
    print(f"[runner] experiment dir: {os.path.join(final_directory, exp_name)}")

    # ---- agent ----
    agent_cls = SAC if agent_type == "sac" else PPO
    agent = agent_cls(
        models=models,
        memory=memory,
        observation_space=obs_space,
        action_space=act_space,
        state_space=state_space,
        device=device,
        cfg=cfg,
        num_agents=n_agents,
        # Build the auxiliary-loss manager from loss_cfg and hand it to the agent.
        # from_cfg() reads the per-loss flat fields and validates targets; it
        # returns an empty (no-op) manager when nothing is enabled. Both SAC and
        # PPO accept and apply this.
        aux_losses=AuxLossManager.from_cfg(loss_cfg),
    )

    # ---- trainer ----
    # `total_timesteps` is interpreted as raw env_steps (env.step() calls). One env_step
    # advances every parallel env by one tick, so the per-agent transition count is
    # `env_steps * num_envs` and total physical transitions written is
    # `env_steps * num_envs * num_agents`. Floor at 1 so degenerate configs always run.
    env_steps = max(1, effective_total_timesteps)
    transitions_per_agent = env_steps * runner_cfg.num_envs
    realized_total_transitions = env_steps * total_envs
    print(
        f"[runner] timesteps ({args.mode}): env_steps={env_steps:,}  "
        f"transitions_per_agent={transitions_per_agent:,}  "
        f"realized_total_transitions={realized_total_transitions:,} "
        f"(num_envs/agent={runner_cfg.num_envs}, n_agents={n_agents})"
    )
    trainer = SequentialTrainer(
        cfg={"timesteps": env_steps, "headless": args.headless},
        env=env,
        agents=agent,
    )

    # Init the agent before trainer.train() so per-agent dirs exist for the config
    # dump below. SAC.init() is idempotent — trainer.train() will call it again
    # but the second call returns immediately.
    agent.init(trainer_cfg=trainer.cfg)

    # Recorder wrapper: opens a 3x4 grid GIF + TB video every K-th *global*
    # reset (steps where every env reports done). Constructed after the SAC
    # agent so its critics + state preprocessor can be passed in. The wrapper
    # composes around the existing wrapper stack via attribute delegation, so
    # the trainer's view of the env is unchanged.
    if agent_type == "ppo" and ppo_cfg.recorder.enabled:
        raise NotImplementedError(
            "ppo_cfg.recorder.enabled=True is not supported: the RecordingWrapper overlays "
            "Q-values from SAC's twin critics, which PPO (state-value critic) lacks. Disable "
            "the recorder for PPO until it is generalized to a value critic."
        )
    if agent_type == "sac" and sac_cfg.recorder.enabled:
        from wrappers.recording import RecordingWrapper

        rec_max_ep_len = int(getattr(env.unwrapped, "max_episode_length", 0))
        if rec_max_ep_len <= 0:
            raise RuntimeError(
                "sac_cfg.recorder.enabled=True but env.unwrapped.max_episode_length is "
                "not set. The recorder pre-allocates a per-env frame buffer of fixed "
                "max-episode-length."
            )
        rec_output_dir = os.path.join(agent.experiment_dir, sac_cfg.recorder.output_subdir)
        # SAC writes images via a per-agent torch SummaryWriter; agent 0 is the
        # canonical destination since the grid is global, not per-agent.
        rec_image_writer = agent.per_agent_image_writers[0] if agent.per_agent_image_writers else None
        env = RecordingWrapper(
            env=env,
            recorder_cfg=sac_cfg.recorder,
            critic_1=agent.critic_1,
            critic_2=agent.critic_2,
            state_preprocessor=getattr(agent, "_state_preprocessor", None),
            max_episode_length=rec_max_ep_len,
            output_dir=rec_output_dir,
            image_writer=rec_image_writer,
        )
        # The trainer was constructed pointing at the un-wrapped env; rebind so
        # it talks to the recorder wrapper.
        trainer.env = env
        print(f"[runner] recorder wrapper attached; outputs -> {rec_output_dir}")

    # Snapshot the merged-and-CLI-applied configs into each agent's results dir so
    # any run can be reconstructed later without consulting the original YAML.
    for i in range(n_agents):
        cfg_path = os.path.join(agent.experiment_dir, str(i), "config.yaml")
        ConfigManager.dump(loaded, cfg_path)

    # Optional checkpoint load — works for both train (resume) and eval.
    if args.checkpoint is not None:
        agent.load(args.checkpoint, step=args.checkpoint_step)
    elif args.mode == "eval":
        raise ValueError("--checkpoint is required for --mode eval")

    # Wrap trainer.train()/eval() so any exception is flushed to stdout BEFORE
    # simulation_app.close() runs — Isaac's shutdown can swallow late stderr and
    # mask exceptions, leaving rc=0 with an empty checkpoint dir as the only
    # symptom. Re-raise so the launcher sees a non-zero exit.
    import traceback
    print(f"[runner] entering trainer.{args.mode}() with timesteps={trainer.cfg.timesteps}",
          flush=True)
    train_exc: BaseException | None = None
    try:
        if args.mode == "train":
            trainer.train()
        else:
            trainer.eval()
        print(f"[runner] trainer.{args.mode}() returned normally", flush=True)
    except BaseException as e:
        train_exc = e
        print(f"[runner] trainer.{args.mode}() raised {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[runner] env.close() raised: {e!r}", flush=True)
        # If training raised, exit non-zero NOW — Isaac's simulation_app.close()
        # internally calls os._exit(0) on shutdown, which would otherwise mask
        # our exception and report rc=0 to the launcher.
        if train_exc is not None:
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[runner] simulation_app.close() raised: {e!r}", flush=True)


if __name__ == "__main__":
    main()
