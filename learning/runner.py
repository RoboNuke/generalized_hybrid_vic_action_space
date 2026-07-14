"""Train/eval entry point for BlockSimba-SAC on Isaac Lab tasks.

Mostly skrl boilerplate for SAC; expected to be modified as the project grows.
The Isaac Lab `AppLauncher` must boot before any `isaaclab.envs` / `isaaclab_tasks`
imports — that's why those imports live inside `main()` after `app_launcher.app`.
"""

import argparse
import os
import sys

# Pre-initialize torch._dynamo single-threaded, BEFORE isaaclab/Omniverse boot any worker
# threads. torch 2.8 wraps Optimizer.add_param_group with a dynamo-disable decorator that does a
# LAZY `import torch._dynamo` on the first optimizer construction (our entropy_optimizer in
# SAC.__init__). If that lazy import races a concurrent torch touch from an Omniverse thread, it
# returns a half-initialized module and crashes with "partially initialized module 'torch._dynamo'
# has no attribute 'external_utils' (circular import)". Importing it here completes its __init__
# cleanly up front so the later lazy import just returns the finished module. We never use
# torch.compile, so this is purely defensive. (Belt-and-suspenders: sac_block_e2e.sh also exports
# TORCHDYNAMO_DISABLE=1.)
import torch  # noqa: F401
import torch._dynamo  # noqa: F401

from isaaclab.app import AppLauncher

# Project root (parent of learning/) — used to anchor the default --config path so
# the runner works regardless of the user's CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "default.yaml")


def _ckpt_step(value: str):
    """argparse type for --checkpoint_step: the literal ``"best"`` (loads each agent's
    ``ckpt_best.pt`` — the highest-success-rate checkpoint) or an integer step."""
    return value if value == "best" else int(value)


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
        "--overlay",
        type=str,
        action="append",
        default=None,
        help="Path to an overlay YAML deep-merged over --config before validation. "
             "Only needs the keys it changes (e.g. a sac_cfg.recorder block). Repeatable; "
             "later overlays win. Nested mappings merge key-by-key; lists/scalars replace.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Overrides sac_cfg.experiment.experiment_name from --config.",
    )
    parser.add_argument(
        "--experiment_directory",
        type=str,
        default=None,
        help="Overrides sac_cfg/ppo_cfg.experiment.directory from --config (the "
             "'family' subdir under --logdir). Lets you save runs to different "
             "places without editing the YAML.",
    )
    parser.add_argument(
        "--wandb_tag",
        type=str,
        action="append",
        default=None,
        help="Append a tag to every wandb run created this launch (repeatable: "
             "--wandb_tag a --wandb_tag b). Merged with any experiment.wandb_kwargs.tags "
             "from the config. Makes runs easy to filter in the wandb UI.",
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
        "--record_agent_dir",
        type=str,
        default=None,
        help="Path to a SINGLE trained agent's folder (e.g. runs/log_dir/1_fixed/0, "
             "containing checkpoints/ckpt_*.pt). Switches the runner into per-agent "
             "recording mode: forces num_agents=1, loads ONLY this agent's weights "
             "(its own policy + twin critics), collects --num_trajectories complete "
             "episodes, and writes an agent-specific best/median/worst grid GIF. "
             "Defaults --config to <record_agent_dir>/config.yaml when --config is unset.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Record mode only: collect at least this many complete trajectories "
             "before composing the grid. Overrides sac_cfg.recorder.num_trajectories.",
    )
    parser.add_argument(
        "--record_output_dir",
        type=str,
        default=None,
        help="Record mode only: where to write the GIF. Defaults to "
             "<record_agent_dir>/<recorder.output_subdir>.",
    )
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
        type=_ckpt_step,
        default=None,
        help="Specific step to load, or 'best' to load each agent's ckpt_best.pt "
             "(highest-success-rate checkpoint). If omitted, the latest ckpt found is used.",
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Per-agent recording mode: default the base config to the one snapshotted
    # next to the checkpoint (runner dumps it to <agent_dir>/config.yaml at train
    # time) when the caller didn't pass --config explicitly.
    if args.record_agent_dir is not None and args.config == _DEFAULT_CONFIG_PATH:
        args.config = os.path.join(args.record_agent_dir, "config.yaml")

    # If the chosen YAML has ``sac_cfg.recorder.enabled: true``, IsaacLab will
    # refuse to spawn the recorder TiledCamera without ``--enable_cameras``.
    # Peek the YAML BEFORE booting AppLauncher and force the flag on so the
    # user doesn't have to remember to pass it.
    try:
        import yaml as _yaml

        def _deep_merge_peek(_base, _ov):
            _out = dict(_base)
            for _k, _v in _ov.items():
                if isinstance(_out.get(_k), dict) and isinstance(_v, dict):
                    _out[_k] = _deep_merge_peek(_out[_k], _v)
                else:
                    _out[_k] = _v
            return _out

        with open(args.config) as _f:
            _peek = _yaml.safe_load(_f) or {}
        # Apply the same overlays here so an overlay that flips recorder.enabled on
        # (e.g. a record overlay) still triggers the --enable_cameras auto-force.
        for _ov_path in (args.overlay or []):
            with open(_ov_path) as _ovf:
                _ov_peek = _yaml.safe_load(_ovf) or {}
            if isinstance(_ov_peek, dict):
                _peek = _deep_merge_peek(_peek, _ov_peek)
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

    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    import tasks.flat_surface_follow  # noqa: F401  registers the repo-local Isaac-FlatSurfaceFollow- id
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

    # Load all registered configs from a single YAML file (defaults to configs/default.yaml).
    loaded = ConfigManager.load(args.config, overlay_paths=args.overlay)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    ppo_cfg = loaded["ppo_cfg"]
    model_cfg = loaded["model_cfg"]
    controller_cfg = loaded["controller_cfg"]
    noise_cfg = loaded["noise_cfg"]
    sensor_cfg = loaded["sensor_cfg"]
    # Auxiliary-loss switches (which extra losses are on and their per-target
    # weights). Absent loss_cfg section -> all-off default -> vanilla SAC.
    loss_cfg = loaded["loss_cfg"]
    # Sampling-Based reset Curriculum (absent -> default-constructed, disabled).
    reset_curriculum_cfg = loaded["reset_curriculum_cfg"]

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
    # Per-agent recording: one agent's slice is loaded into a num_agents=1 run, so
    # the env, policy, and twin critics all belong to that single agent.
    if args.record_agent_dir is not None: runner_cfg.num_agents = 1
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
    #
    # A config reloaded from a dumped config.yaml (e.g. record mode loading
    # <agent_dir>/config.yaml) carries rewards_shaper back as a lossy repr string
    # (e.g. "__main__.main.<locals>.<lambda>") rather than a real callable, alongside
    # the original rewards_shaper_scale. Treat that string as unset so it gets rebuilt
    # from the scale below instead of tripping the mutual-exclusion check.
    if isinstance(active_cfg.rewards_shaper, str):
        active_cfg.rewards_shaper = None
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
    from learning.env_setup import build_env

    env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs = build_env(
        args, runner_cfg, sac_cfg, ppo_cfg, controller_cfg, noise_cfg, sensor_cfg, agent_type,
        reset_curriculum_cfg=reset_curriculum_cfg,
    )

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
    # since YAML can't carry a Box space or torch.device object. These two keys are
    # runtime-only and are ALWAYS overwritten (never setdefault): a config reloaded
    # from a dumped config.yaml carries them back as lossy repr strings
    # (e.g. "Box(-inf, inf, (30,), float32)") that would otherwise reach the
    # preprocessor verbatim and crash. Other user kwargs (epsilon, ...) are preserved.
    if active_cfg.observation_preprocessor is not None:
        if not isinstance(active_cfg.observation_preprocessor_kwargs, dict):
            active_cfg.observation_preprocessor_kwargs = {}
        active_cfg.observation_preprocessor_kwargs["size"] = obs_space
        active_cfg.observation_preprocessor_kwargs["device"] = device

    # PPO value preprocessor (single shared RunningStandardScaler over scalar values).
    if agent_type == "ppo" and ppo_cfg.value_preprocessor is not None:
        if not isinstance(ppo_cfg.value_preprocessor_kwargs, dict):
            ppo_cfg.value_preprocessor_kwargs = {}
        ppo_cfg.value_preprocessor_kwargs["size"] = 1
        ppo_cfg.value_preprocessor_kwargs["device"] = device

    # ---- models ----
    actor_kwargs = dataclasses.asdict(model_cfg.actor)
    critic_kwargs = dataclasses.asdict(model_cfg.critic)

    # disable_success_pred force-zeros the Forge success-prediction action (dim 6) so the actor
    # allocates no parameters to the inert head. build_env() has already validated the flag is
    # Forge-only (7-dim base action, index 6 = success prediction; survives the control-wrapper
    # action-space expansion since the base slice [:7] is passed through unchanged). Merge with
    # any user-specified force-zero list.
    if runner_cfg.disable_success_pred:
        _fz = list(actor_kwargs.get("force_zero_action_dims") or [])
        if 6 not in _fz:
            _fz.append(6)
        actor_kwargs["force_zero_action_dims"] = sorted(set(_fz))
        print(
            f"[runner] disable_success_pred: force_zero_action_dims -> "
            f"{actor_kwargs['force_zero_action_dims']} (index 6 = success prediction)"
        )

    # selection_distribution / selection_init_bias apply only to the hybrid actor;
    # pop them so they never reach the plain BlockSimBaActor (which doesn't accept them).
    selection_distribution = actor_kwargs.pop("selection_distribution", "product")
    selection_init_bias = actor_kwargs.pop("selection_init_bias", 0.0)

    # Critic input space: state_space when asymmetric, else obs_space.
    critic_input_space = state_space if state_space is not None else obs_space

    # Actor: hybrid control types get the selection-gated HybridControlBlockSimBaActor
    # (product / match), otherwise the plain squashed-Gaussian actor. Wrapped in a factory so
    # SimBa periodic resets (sac_cfg.periodic_reset_*) can rebuild fresh, identically-constructed
    # models mid-training (see learning/sac.py::_periodic_reset).
    def make_models():
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
            return {
                "policy": policy,
                "critic_1": make_q(),
                "critic_2": make_q(),
                "target_critic_1": make_q(),
                "target_critic_2": make_q(),
            }
        else:  # ppo — single state-value critic V(s)
            value = BlockSimBaValueCritic(
                observation_space=critic_input_space,
                action_space=act_space,
                device=device,
                num_agents=n_agents,
                **critic_kwargs,
            )
            return {"policy": policy, "value": value}

    models = make_models()

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

    # CLI > YAML for the experiment "family" directory. Applied before the
    # final-directory computation below so --experiment_directory redirects runs
    # without touching the YAML.
    if args.experiment_directory is not None:
        cfg.experiment.directory = args.experiment_directory

    # --wandb_tag: APPEND to experiment.wandb_kwargs["tags"] (deduped, order preserved) so
    # make_wandb_run passes them straight to wandb.init(tags=...). Merges with any config tags.
    if getattr(args, "wandb_tag", None):
        wk = dict(getattr(cfg.experiment, "wandb_kwargs", {}) or {})
        tags = list(wk.get("tags") or [])
        for t in args.wandb_tag:
            if t and t not in tags:
                tags.append(t)
        wk["tags"] = tags
        cfg.experiment.wandb_kwargs = wk

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
    if args.record_agent_dir is None:
        print(f"[runner] experiment dir: {os.path.join(final_directory, exp_name)}")

    # Record mode writes no tensorboard output — only the GIF. Suppress per-agent
    # writers (write_interval=0) and point the (otherwise unused) experiment dir at a
    # throwaway under the agent folder so the runs tree stays clean. The record branch
    # rmtree's it before exit.
    if args.record_agent_dir is not None:
        cfg.experiment.write_interval = 0
        cfg.experiment.directory = os.path.abspath(args.record_agent_dir)
        cfg.experiment.experiment_name = ".record_run"
        print(f"[runner] record mode: scratch experiment dir "
              f"{os.path.join(cfg.experiment.directory, cfg.experiment.experiment_name)} "
              "(removed on exit; GIF goes to the agent's videos/ dir)")

    # Supervised-selection loss (SSL): when enabled, the agent buffers the per-axis
    # contact ground truth (sliced to the force-eligible axes) as the BCE target. The
    # eligible axes come from controller_cfg.force_axes (ascending = x,y,z order), so the
    # contact width tracks sum(force_axes), aligned 1:1 with the hybrid selection head.
    contact_axes = None
    if loss_cfg.supervised_selection_enabled:
        eligible = [i for i, v in enumerate(controller_cfg.force_axes) if v]
        if not sensor_cfg.contact.enabled:
            raise ValueError(
                "loss_cfg.supervised_selection_enabled=True requires "
                "sensor_cfg.contact.enabled=True (the contact sensor provides the target)."
            )
        if ctrl_wrapper is None:
            raise ValueError(
                "loss_cfg.supervised_selection_enabled=True requires a hybrid control_type "
                "(a selection head); got control_type="
                f"{controller_cfg.control_type!r}."
            )
        if not eligible or any(a not in (0, 1, 2) for a in eligible):
            raise ValueError(
                "supervised_selection requires translational force-eligible axes (force_axes "
                f"nonzero ⊆ {{0,1,2}} = x,y,z); the 3-axis ContactSensor has no rotational "
                f"contact. Got force_axes={list(controller_cfg.force_axes)}."
            )
        contact_axes = eligible
        print(f"[runner] supervised selection loss: contact_axes={contact_axes} "
              f"(force-eligible x/y/z; contact_dim={len(contact_axes)})")

    # supervised-rotation loss: supervise the GAS policy's LEARNED rotation frame toward the
    # ground-truth interaction frame (what a fixed-rot controller would use). Needs a learned
    # rotation; the control wrapper exposes env.unwrapped._rot6d_action_slice (None otherwise).
    rot6d_slice = None
    if loss_cfg.supervised_rotation_enabled:
        rot6d_slice = getattr(env.unwrapped, "_rot6d_action_slice", None)
        if rot6d_slice is None:
            raise ValueError(
                "loss_cfg.supervised_rotation_enabled=True requires a LEARNED rotation frame "
                "(controller_cfg.gain_mapping='rotated' WITHOUT fixed_rotation_from_interaction/rpy); "
                "the controller exposed no rot6d action slice (env._rot6d_action_slice is None)."
            )
        print(f"[runner] supervised rotation loss: rot6d_slice={rot6d_slice} "
              "(GAS rotation dims supervised toward the ground-truth interaction frame)")

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
        contact_axes=contact_axes,
        rot6d_slice=rot6d_slice,
    )

    # SimBa periodic reset: give the agent a factory that rebuilds fresh, identically-constructed
    # networks (see sac_cfg.periodic_reset_*). Attached unconditionally; the agent only calls it
    # when periodic_reset_enabled. SAC only (PPO has no reset path).
    if agent_type == "sac":
        agent._model_factory = make_models

    # ---- per-agent recording mode ----
    # Loads ONE agent's slice (its own policy + twin critics) into this num_agents=1
    # run, collects complete trajectories, and writes an agent-specific grid GIF.
    # Self-contained: it never builds a trainer and exits via os._exit() so the
    # training path below is untouched.
    if args.record_agent_dir is not None:
        if agent_type != "sac":
            raise NotImplementedError(
                "--record_agent_dir is SAC-only (the V-Est overlay uses the twin critics)."
            )
        if not sac_cfg.recorder.enabled:
            raise ValueError(
                "--record_agent_dir requires sac_cfg.recorder.enabled=true so the recorder "
                "TiledCamera is injected into the scene (set it in your --overlay record config)."
            )
        from learning.recording_eval import collect_and_record, collect_annotated_ranked, collect_stills_grid
        from wrappers.recording import CAMERA_KEY

        rc = 0
        try:
            # init() wires up preprocessors/checkpoint modules; write_interval was
            # forced to 0 (no TB writers / tfevents). Then load ONLY this agent's
            # weights via the slot loader — it skips optimizer state, so a slice of an
            # N>1 checkpoint loads cleanly into this 1-agent run (agent.load() refuses
            # that).
            # trainer_cfg=None: skrl's base.init does dataclasses.asdict(trainer_cfg)
            # and only special-cases None (-> {}); a plain dict would raise. We don't
            # train, so the trainer cfg is irrelevant here.
            agent.init(trainer_cfg=None)
            agent._load_one_into_slot(args.record_agent_dir, target_slot=0, step=args.checkpoint_step)
            agent.training = False
            agent.enable_models_training_mode(False)

            scene = env.unwrapped.scene
            if not hasattr(scene, "sensors") or CAMERA_KEY not in scene.sensors:
                raise RuntimeError(
                    f"recorder TiledCamera ({CAMERA_KEY!r}) not found in scene; build_env "
                    "should have injected it when recorder.enabled=true."
                )
            camera = scene.sensors[CAMERA_KEY]
            max_ep_len = int(getattr(env.unwrapped, "max_episode_length", 0))
            n_traj = (
                args.num_trajectories
                if args.num_trajectories is not None
                else sac_cfg.recorder.num_trajectories
            )
            out_dir = args.record_output_dir or os.path.join(
                args.record_agent_dir, sac_cfg.recorder.output_subdir
            )
            print(f"[record] agent_dir={args.record_agent_dir}  num_trajectories={n_traj}  "
                  f"num_envs={env.num_envs}  max_ep_len={max_ep_len}  out_dir={out_dir}", flush=True)

            if bool(getattr(sac_cfg.recorder, "stills_grid", False)):
                gif_path = collect_stills_grid(
                    env=env,
                    agent=agent,
                    recorder_cfg=sac_cfg.recorder,
                    camera=camera,
                    max_episode_length=max_ep_len,
                    output_dir=out_dir,
                )
            elif bool(getattr(sac_cfg.recorder, "annotated_ranked", False)):
                gif_path = collect_annotated_ranked(
                    env=env,
                    agent=agent,
                    recorder_cfg=sac_cfg.recorder,
                    camera=camera,
                    max_episode_length=max_ep_len,
                    num_trajectories=int(n_traj),
                    output_dir=out_dir,
                )
            else:
                gif_path = collect_and_record(
                    env=env,
                    agent=agent,
                    recorder_cfg=sac_cfg.recorder,
                    camera=camera,
                    max_episode_length=max_ep_len,
                    num_trajectories=int(n_traj),
                    output_dir=out_dir,
                )
            # Validate the soft contract: a non-raising return must have actually
            # written a non-empty video file. collect_and_record can return a path
            # without a usable file on disk (silent write failure, degenerate
            # collection). Raise here so the except below sets rc=1 -> os._exit(1)
            # and the batch launcher marks this agent FAIL instead of OK.
            if not gif_path or not os.path.isfile(gif_path) or os.path.getsize(gif_path) == 0:
                raise RuntimeError(
                    f"recording produced no valid output file (got {gif_path!r}); "
                    "treating as failure"
                )
            print(f"[record] DONE -> {gif_path}", flush=True)
        except BaseException as e:  # noqa: BLE001 - flush before Isaac teardown
            import traceback
            print(f"[record] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            rc = 1

        # Teardown ordering mirrors the trainer path: on FAILURE we os._exit(1)
        # BEFORE simulation_app.close(), because Isaac's shutdown frequently forces
        # its own exit code and would otherwise mask our failure (making the batch
        # launcher mark a crashed run as OK).
        try:
            env.close()
        except Exception as e:
            print(f"[record] env.close() raised: {e!r}", flush=True)
        # Remove the throwaway experiment dir created by agent.init().
        try:
            import shutil
            _tmp = os.path.join(os.path.abspath(args.record_agent_dir), ".record_run")
            if os.path.isdir(_tmp):
                shutil.rmtree(_tmp, ignore_errors=True)
        except Exception:
            pass
        sys.stdout.flush(); sys.stderr.flush()
        if rc != 0:
            os._exit(1)
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[record] simulation_app.close() raised: {e!r}", flush=True)
        os._exit(0)

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
            # Annotate the tqdm bar with progress toward the next TB/wandb flush:
            # the agent logs when timestep % write_interval == 0, so "log x/W" shows
            # how many steps into the current logging interval we are. We swap the
            # `tqdm` reference inside the SequentialTrainer module ONLY (a shim whose
            # .tqdm is a postfix-setting subclass), so no other progress bar is
            # affected and skrl's train loop stays untouched. Train mode only.
            _wi = int(getattr(agent, "write_interval", 0) or 0)
            if _wi > 0 and not getattr(trainer.cfg, "disable_progressbar", False):
                import skrl.trainers.torch.sequential as _seq

                class _IntervalBar(_seq.tqdm.tqdm):
                    def update(self, n=1):
                        # Set the postfix to the POST-increment count BEFORE super()
                        # so the redraw inside update() shows the current value (not
                        # the previous cycle's). refresh=False piggybacks on tqdm's
                        # own redraw cadence rather than forcing a draw every step.
                        self.set_postfix_str(f"log {(self.n + n) % _wi}/{_wi}", refresh=False)
                        return super().update(n)

                class _TqdmShim:
                    tqdm = _IntervalBar

                _seq.tqdm = _TqdmShim
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

        # Flush + close every per-agent writer synchronously. BOTH exit paths below
        # use os._exit(), which skips Python/atexit cleanup — so the writers'
        # background flush threads never run. Flush here or lose the final interval's
        # scalars and any buffered image events. (write_tracking_data already flushes
        # scalars on each interval; this also covers the image writers, which are
        # otherwise never flushed, and the trailing partial interval.)
        for _w in (getattr(agent, "per_agent_writers", None) or []):
            try:
                _w.flush(); _w.close()
            except Exception as e:
                print(f"[runner] per-agent writer flush/close raised: {e!r}", flush=True)
        for _w in (getattr(agent, "per_agent_image_writers", None) or []):
            try:
                _w.flush(); _w.close()
            except Exception as e:
                print(f"[runner] per-agent image writer flush/close raised: {e!r}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()

        # If training raised, exit non-zero NOW so the launcher sees the failure.
        if train_exc is not None:
            os._exit(1)

        # SUCCESS: tear down Isaac, then FORCE a clean, deterministic exit. We do not
        # rely on simulation_app.close()'s own exit behavior or on normal interpreter
        # teardown to set the exit code — Isaac Sim's C++/CUDA/PhysX shutdown
        # frequently segfaults/aborts AFTER all real work is done, which makes a fully
        # successful run report a nonzero exit code. The batch launcher
        # (exp_file_launcher.bash) captures $? and would then mark the run FAILED even
        # though everything succeeded. Mirroring the os._exit(1) path above, an
        # explicit os._exit(0) guarantees rc=0 once we've gotten this far.
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[runner] simulation_app.close() raised: {e!r}", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
