"""Interactive env visualizer: boot a config in the GUI and watch it live.

Boots IsaacSim/IsaacLab **with the GUI** (headless is intentionally disallowed)
and builds the environment EXACTLY like training does
(``learning.env_setup.build_env`` — the same pipeline ``learning/runner.py`` and
``launchers/sac_block_e2e.sh`` use), so what you see is what the trainer sees.

Actions:
  * ``--checkpoint <dir>`` given  -> roll out that trained policy (greedy/sampled
    from the actor), loaded via the SAME ``agent.load`` path the runner/eval uses.
  * no checkpoint                 -> uniform random actions on [-1, 1] (the policy's
    tanh-squashed support; matches SAC/PPO's ``random_timesteps`` sampling).

Keyboard shortcuts (focus the Isaac Sim window):
  * ``r`` — reset every env via the env's own ``_reset_idx`` (NOT ``env.reset()``),
            which is how partial/auto resets happen on the real system.
  * ``p`` — toggle pause: physics stops (nothing moves) but rendering keeps running,
            so you can orbit/pan the viewport with no stepping lag.
  * ``q`` — quit and tear down the sim cleanly.

Like ``runner.py``, the IsaacLab ``AppLauncher`` must boot before any
``isaaclab.envs`` / ``isaaclab_tasks`` imports — those live inside ``main()``.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# This script lives in <project_root>/utilities/, so the project root is one level up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive GUI visualizer for Isaac Lab envs (always windowed).",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML config file (same format runner.py uses). "
             f"Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Folder path to a trained agent checkpoint. Multi-agent: parent "
             "containing 0/, 1/, ... subdirs. Single-agent: a folder with "
             "checkpoints/ckpt_<step>.pt directly. If omitted, random actions are used.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific step to load. If omitted, the latest ckpt found is used.",
    )
    # Runner-level overrides (mirror runner.py; CLI > YAML). num_envs/num_agents are
    # intentionally NOT exposed — this tool always spawns exactly one env (see below).
    parser.add_argument("--task", type=str, default=None,
                        help="Overrides runner_cfg.task.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Overrides runner_cfg.seed. -1 means non-deterministic.")
    AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device, --enable_cameras
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ---- hard-disallow headless ----
    # The entire point of this tool is to eyeball the env, so a windowed app is
    # mandatory. Refuse rather than silently overriding so a stray --headless from
    # muscle memory is surfaced, not ignored.
    if getattr(args, "headless", False):
        print("[visualize] --headless is not allowed: this tool exists to render the "
              "env in a window. Drop the flag.", file=sys.stderr)
        sys.exit(2)
    args.headless = False

    # Resolve config to absolute (allow a project-root-relative path).
    if not os.path.isabs(args.config):
        cand = os.path.join(_PROJECT_ROOT, args.config)
        args.config = cand if os.path.exists(cand) else os.path.abspath(args.config)
    if not os.path.isfile(args.config):
        print(f"[visualize] config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Boot Omniverse (windowed) before any isaaclab.envs imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Silence IsaacLab's per-call quat-deprecation spam on the math channel
    # (mirrors runner.py / init_calib.py).
    try:
        import omni.log

        omni.log.get_log().set_channel_level(
            "isaaclab.utils.math", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
        )
    except Exception as e:  # pragma: no cover - best-effort log tidy-up
        print(f"[visualize] could not raise isaaclab.utils.math log level: {e!r}", flush=True)

    # Project root on sys.path so `learning.*` / `models.*` / `wrappers.*` resolve
    # regardless of CWD.
    sys.path.insert(0, _PROJECT_ROOT)

    import dataclasses

    import carb
    import carb.input
    import omni.appwindow
    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from skrl.utils import set_seed

    from configs.manager import ConfigManager
    from learning.env_setup import build_env
    from learning.losses import AuxLossManager
    from learning.sac import SAC
    from learning.ppo import PPO
    from models.block_simba import (
        BlockSimBaActor,
        BlockSimBaQCritic,
        BlockSimBaValueCritic,
        HybridControlBlockSimBaActor,
    )
    from memory.multi_random import MultiRandomMemory

    # ---- config (same loader as training) ----
    loaded = ConfigManager.load(args.config)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    ppo_cfg = loaded["ppo_cfg"]
    model_cfg = loaded["model_cfg"]
    controller_cfg = loaded["controller_cfg"]
    noise_cfg = loaded["noise_cfg"]
    sensor_cfg = loaded["sensor_cfg"]
    loss_cfg = loaded["loss_cfg"]

    agent_type = str(runner_cfg.agent_type).lower()
    if agent_type not in ("sac", "ppo"):
        raise ValueError(
            f"runner_cfg.agent_type must be 'sac' or 'ppo', got {runner_cfg.agent_type!r}"
        )
    active_cfg = sac_cfg if agent_type == "sac" else ppo_cfg

    # CLI > YAML for the runner-level fields we expose.
    if args.task is not None: runner_cfg.task = args.task
    if args.seed is not None: runner_cfg.seed = args.seed

    # Always visualize exactly ONE env, regardless of what the config requests
    # (cheaper boot, single scene to watch). A single-agent run loads a single-agent
    # checkpoint dir; for a multi-agent training run, point --checkpoint at one
    # agent's subdir (e.g. runs/<exp>/0), which loads into this 1-agent run.
    runner_cfg.num_envs = 1
    runner_cfg.num_agents = 1
    print("[visualize] forcing num_envs=1, num_agents=1 (single env).", flush=True)

    # The recorder TiledCamera is a GIF-export concern, irrelevant to live GUI
    # viewing — and injecting it would demand --enable_cameras. Force it off so the
    # env build matches training in every way that affects dynamics, minus the
    # passive recorder camera. (build_env reads sac_cfg.recorder.enabled for the
    # camera; the RecordingWrapper itself is added by runner.py, not here.)
    if sac_cfg.recorder.enabled:
        sac_cfg.recorder.enabled = False
        print("[visualize] sac_cfg.recorder.enabled forced False (live GUI view; "
              "no recorder camera/GIF).", flush=True)

    # Resolve seed to a concrete value (so a default/-1 run is still reproducible).
    seed = runner_cfg.seed
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)
    runner_cfg.seed = seed
    set_seed(seed)
    print(f"[visualize] seed={seed} (pass --seed {seed} to reproduce).", flush=True)

    # ---- env (identical to training) ----
    env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs = build_env(
        args, runner_cfg, sac_cfg, ppo_cfg, controller_cfg, noise_cfg, sensor_cfg, agent_type
    )

    device = torch.device(args.device)
    n_agents = runner_cfg.num_agents
    obs_space = env.observation_space
    act_space = env.action_space
    state_space = getattr(env, "state_space", None)

    # ---- optional policy (mirrors runner.py's model + agent build) ----
    # Only built when a checkpoint is supplied; otherwise we drive random actions.
    agent = None
    scratch_dir = None
    if args.checkpoint is not None:
        # Observation preprocessor needs the runtime Box/device kwargs injected (YAML
        # can't carry them); always overwrite these two keys, exactly as runner.py does.
        if active_cfg.observation_preprocessor is not None:
            if not isinstance(active_cfg.observation_preprocessor_kwargs, dict):
                active_cfg.observation_preprocessor_kwargs = {}
            active_cfg.observation_preprocessor_kwargs["size"] = obs_space
            active_cfg.observation_preprocessor_kwargs["device"] = device
        if agent_type == "ppo" and ppo_cfg.value_preprocessor is not None:
            if not isinstance(ppo_cfg.value_preprocessor_kwargs, dict):
                ppo_cfg.value_preprocessor_kwargs = {}
            ppo_cfg.value_preprocessor_kwargs["size"] = 1
            ppo_cfg.value_preprocessor_kwargs["device"] = device

        actor_kwargs = dataclasses.asdict(model_cfg.actor)
        critic_kwargs = dataclasses.asdict(model_cfg.critic)
        selection_distribution = actor_kwargs.pop("selection_distribution", "product")
        selection_init_bias = actor_kwargs.pop("selection_init_bias", 0.0)
        critic_input_space = state_space if state_space is not None else obs_space

        if ctrl_wrapper is not None:
            sel_dims, pos_dims, force_dims = ctrl_wrapper.policy_selection_layout
            policy = HybridControlBlockSimBaActor(
                observation_space=obs_space, action_space=act_space, device=device,
                num_agents=n_agents, selection_dims=sel_dims, pos_component_dims=pos_dims,
                force_component_dims=force_dims, selection_distribution=selection_distribution,
                selection_init_bias=selection_init_bias, **actor_kwargs,
            )
        else:
            policy = BlockSimBaActor(
                observation_space=obs_space, action_space=act_space, device=device,
                num_agents=n_agents, **actor_kwargs,
            )

        if agent_type == "sac":
            def make_q():
                return BlockSimBaQCritic(
                    observation_space=critic_input_space, action_space=act_space,
                    device=device, num_agents=n_agents, **critic_kwargs,
                )
            models = {
                "policy": policy,
                "critic_1": make_q(), "critic_2": make_q(),
                "target_critic_1": make_q(), "target_critic_2": make_q(),
            }
        else:
            value = BlockSimBaValueCritic(
                observation_space=critic_input_space, action_space=act_space,
                device=device, num_agents=n_agents, **critic_kwargs,
            )
            models = {"policy": policy, "value": value}

        # Inference-only memory: depth 1 per env (we never sample it). Keeps the
        # allocation tiny vs. the training replay buffer.
        memory = MultiRandomMemory(
            memory_size=1, num_envs=env.num_envs, num_agents=n_agents,
            device=device, replacement=True,
        )

        # No TB output for a viz run: silence writers and point the (unused)
        # experiment dir at a scratch folder removed on exit.
        cfg = active_cfg
        cfg.experiment.write_interval = 0
        scratch_dir = os.path.join(_PROJECT_ROOT, "runs", ".viz_scratch")
        cfg.experiment.directory = scratch_dir
        cfg.experiment.experiment_name = ".viz_run"

        # Supervised-selection contact axes (only consulted during training updates,
        # which we never run — but compute it the same way so the agent ctor matches).
        contact_axes = None
        if loss_cfg.supervised_selection_enabled and sensor_cfg.contact.enabled and ctrl_wrapper is not None:
            contact_axes = [i for i, v in enumerate(controller_cfg.force_axes) if v] or None

        agent_cls = SAC if agent_type == "sac" else PPO
        agent = agent_cls(
            models=models, memory=memory, observation_space=obs_space,
            action_space=act_space, state_space=state_space, device=device,
            cfg=cfg, num_agents=n_agents,
            aux_losses=AuxLossManager.from_cfg(loss_cfg), contact_axes=contact_axes,
        )
        agent.init(trainer_cfg=None)
        agent.load(args.checkpoint, step=args.checkpoint_step)
        agent.training = False
        agent.enable_models_training_mode(False)
        print(f"[visualize] loaded policy from {args.checkpoint} "
              f"(step={'latest' if args.checkpoint_step is None else args.checkpoint_step}).",
              flush=True)
    else:
        print("[visualize] no --checkpoint given: driving uniform random actions on [-1, 1].",
              flush=True)

    action_dim = act_space.shape[0]

    # ---- keyboard state ----
    # The callback runs on the app's input thread; it only flips flags so the main
    # loop stays the single place that touches the sim.
    state = {"paused": False, "quit": False, "reset": False}

    def _on_keyboard(event, *_):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input
        if key == carb.input.KeyboardInput.R:
            state["reset"] = True
        elif key == carb.input.KeyboardInput.P:
            state["paused"] = not state["paused"]
            print(f"[visualize] {'PAUSED (rendering only)' if state['paused'] else 'RESUMED'}",
                  flush=True)
        elif key == carb.input.KeyboardInput.Q:
            state["quit"] = True
            print("[visualize] quit requested.", flush=True)
        return True

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard)

    print("[visualize] controls: [r] reset (_reset_idx)  [p] pause/resume  [q] quit",
          flush=True)

    exc = None
    try:
        u = env.unwrapped
        reset_ids = torch.arange(u.num_envs, device=u.device)

        # One wrapped reset up front: initializes the wrapper chain AND clears
        # gymnasium's OrderEnforcing guard (step() before any reset() raises).
        obs, _ = env.reset()

        def _resolve_state(o):
            try:
                st = env.state()
            except Exception:
                st = None
            return st if st is not None else o

        cur_state = _resolve_state(obs)

        # no_grad over the whole loop: the controller wrappers' EMA buffers splice the
        # action tensor in-place each step; a live autograd graph would grow unbounded
        # and OOM the GPU. (agent.act already guards internally; this also covers the
        # random-action path and any wrapper bookkeeping.)
        with torch.no_grad():
            while simulation_app.is_running() and not state["quit"]:
                # Reset is honored even while paused, then rendered so the new pose
                # is visible without resuming. Mirrors how partial resets happen on
                # the real system: the env's OWN _reset_idx, not env.reset().
                if state["reset"]:
                    state["reset"] = False
                    u._reset_idx(reset_ids)
                    u.sim.render()
                    print("[visualize] reset via _reset_idx().", flush=True)

                if state["paused"]:
                    # Render ONLY — no physics step. sim.render() pumps the rendering
                    # app (so the viewport stays interactive for orbit/pan and keyboard
                    # events keep flowing) but does NOT advance physics, so the robot
                    # holds its exact current pose. (simulation_app.update() would step
                    # physics while the timeline plays; with no control torque recomputed
                    # the effort-controlled arm drifts toward its default joint pose.)
                    u.sim.render()
                    continue

                if agent is not None:
                    actions, _ = agent.act(
                        obs, cur_state, timestep=10**9, timesteps=10**9
                    )
                else:
                    actions = torch.rand(
                        (total_envs, action_dim), dtype=torch.float32, device=device
                    ) * 2.0 - 1.0

                obs, _reward, _terminated, _truncated, _info = env.step(actions)
                cur_state = _resolve_state(obs)
    except BaseException as e:  # noqa: BLE001 - flush before Isaac teardown
        exc = e
        import traceback
        print(f"[visualize] raised {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        try:
            input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass
        try:
            env.close()
        except Exception as e:
            print(f"[visualize] env.close() raised: {e!r}", flush=True)
        # Drop the throwaway experiment dir created by agent.init().
        if scratch_dir is not None:
            try:
                import shutil
                if os.path.isdir(scratch_dir):
                    shutil.rmtree(scratch_dir, ignore_errors=True)
            except Exception:
                pass
        sys.stdout.flush(); sys.stderr.flush()

        if exc is not None:
            os._exit(1)
        # Mirror runner.py: tear down Isaac then force a clean exit (Isaac's
        # C++/CUDA shutdown frequently segfaults after all real work is done).
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[visualize] simulation_app.close() raised: {e!r}", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
