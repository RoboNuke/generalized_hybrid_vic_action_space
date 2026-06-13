"""Initialization calibration tool: film repeated env resets under zero action.

Boots IsaacSim/IsaacLab and builds the environment EXACTLY like training
(``learning.env_setup.build_env`` — the same pipeline ``runner.py`` uses), then:

  * resets the env ``--num_resets`` times (default 100);
  * after each reset, steps a ZERO action for ``--settle_seconds`` (default 1.0 s)
    of sim time, capturing one camera frame per step;
  * concatenates every reset's clip into a single GIF (``--out``).

The recorder ``TiledCamera`` is always injected (``build_env(force_camera=True)``)
and ``--enable_cameras`` is forced on, so no config changes are needed. Useful for
eyeballing spawn poses / grasp poses / randomization without launching training.

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
        description="Reset/zero-action calibration GIF for Isaac Lab envs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to a YAML config file (same format runner.py uses). "
             f"Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    # Runner-level overrides (mirror runner.py; CLI > YAML). num_envs/num_agents
    # are intentionally NOT exposed — calibration always runs a single env.
    parser.add_argument("--task", type=str, default=None,
                        help="Overrides runner_cfg.task.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Overrides runner_cfg.seed. -1 means non-deterministic.")
    # Calibration-specific knobs.
    parser.add_argument("--num_resets", type=int, default=100,
                        help="How many times to reset + film (default 100).")
    parser.add_argument("--settle_seconds", type=float, default=1.0,
                        help="Seconds of sim time to film after each reset (default 1.0). "
                             "Captured one frame per control step.")
    parser.add_argument("--action_mode", choices=["hold", "zero"], default="hold",
                        help="hold (default): robot holds its post-reset pose — physics "
                             "is stepped with NO action applied (step_sim_no_action), so the "
                             "arm stays put; best for inspecting the initialization. "
                             "zero: step a zero policy action each step — the arm servos to "
                             "the controller's zero-action target (for ctrl-action-interface "
                             "that moves it toward the socket frame).")
    parser.add_argument("--out", type=str, default="init_calib.mp4",
                        help="Output path. Extension picks the format: .mp4 (default, "
                             "streamed to disk frame-by-frame via ffmpeg — constant memory, "
                             "recommended) or .gif (streamed to a temp .mp4 then converted "
                             "with ffmpeg; GIF can't be appended incrementally by PIL).")
    parser.add_argument("--fps", type=int, default=0,
                        help="GIF playback fps. 0 (default) = real-time from sim dt.")
    # Camera resolution. The recorder defaults to a tiny 240x180 so 12 tiles fit the
    # training 4x3 grid; here we film a single env, so default to a full size (4:3,
    # matching the camera's aperture/framing).
    parser.add_argument("--width", type=int, default=960,
                        help="Camera/GIF width in pixels (default 960).")
    parser.add_argument("--height", type=int, default=720,
                        help="Camera/GIF height in pixels (default 720).")
    # Camera placement (offset relative to each env's origin; envs aren't rotated, so this is
    # effectively world-frame). Default matches sac_cfg.recorder (RoboNuke eval setup): pos
    # (1.0, 0.0, 0.35). Raise the 3rd value to move the camera UP in world z.
    parser.add_argument("--camera_pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Camera position offset [x y z] (meters) from the env origin. "
                             "Increase Z to move the camera up. Default: recorder's (1.0 0.0 0.35).")
    parser.add_argument("--camera_quat", type=float, nargs=4, default=None,
                        metavar=("W", "X", "Y", "Z"),
                        help="Camera orientation quaternion [w x y z] (ROS convention). "
                             "Default: recorder's downward-tilted view.")
    AppLauncher.add_app_launcher_args(parser)  # adds --headless, --device
    return parser


def _read_rgb(camera):
    """Pull the latest TiledCamera RGB as CPU uint8 ``(N, H, W, 3)``.

    Mirrors ``wrappers.recording.RecordingWrapper._read_camera_rgb``.
    """
    import torch  # local: only valid after AppLauncher boot

    rgb = camera.data.output["rgb"]  # device tensor; (N, H, W, C)
    if rgb.dim() != 4:
        raise RuntimeError(
            f"recorder camera returned unexpected shape {tuple(rgb.shape)}; expected (N, H, W, C)"
        )
    if rgb.shape[-1] == 4:  # drop alpha if present
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb.detach().cpu()


def _make_progress(total: int):
    """Return an iterable over ``range(total)`` with a reset progress bar.

    Uses ``tqdm`` when available; otherwise falls back to a minimal
    carriage-return bar so the tool has no hard dependency on tqdm.
    """
    try:
        from tqdm import tqdm
        return tqdm(range(total), desc="resets", unit="reset")
    except Exception:
        def _fallback():
            for i in range(total):
                yield i
                done = i + 1
                bar_w = 30
                filled = int(bar_w * done / total)
                bar = "#" * filled + "-" * (bar_w - filled)
                end = "\n" if done == total else ""
                print(f"\r[init_calib] resets [{bar}] {done}/{total}", end=end, flush=True)
        return _fallback()


def _open_frame_writer(out_path: str, fps: int):
    """Open a streaming frame writer that flushes each frame to DISK immediately.

    Avoids holding every frame in RAM (the old ``np.stack`` + PIL-GIF path buffered the
    whole animation, which OOMs/bricks on long runs). MP4 is encoded incrementally by
    ffmpeg (constant memory). GIF cannot be appended frame-by-frame by PIL/imageio, so a
    ``.gif`` target streams to a temp ``.mp4`` and is converted with ffmpeg at the end
    (also low-memory). Uses imageio's BUNDLED ffmpeg, so no system ffmpeg is required.

    Returns ``(append, finalize)``: ``append(frame_hwc_uint8)`` writes one frame;
    ``finalize()`` closes/encodes and returns the final output path.
    """
    import imageio

    ext = os.path.splitext(out_path)[1].lower()
    is_gif = ext == ".gif"
    video_path = (out_path + ".tmp.mp4") if is_gif else out_path
    # macro_block_size=2 keeps dimensions even (libx264 requirement) with negligible
    # resize; quality ~8/10 is a good size/clarity tradeoff for these renders.
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=2)

    def append(frame):
        writer.append_data(frame)

    def finalize():
        writer.close()
        if not is_gif:
            return out_path
        import subprocess
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        palette = out_path + ".palette.png"
        # Two-pass palette (palettegen -> paletteuse) for decent GIF color; ffmpeg
        # streams from the temp mp4 on disk, so memory stays low.
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", video_path,
                        "-vf", "palettegen", palette], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", video_path,
                        "-i", palette, "-lavfi", "paletteuse", out_path], check=True)
        for p in (video_path, palette):
            try:
                os.remove(p)
            except OSError:
                pass
        return out_path

    return append, finalize


def main() -> None:
    args = build_parser().parse_args()

    # The recorder camera needs IsaacLab's rendering pipeline — always force it on.
    args.enable_cameras = True

    # Boot Omniverse before any isaaclab.envs imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Silence IsaacLab's per-call quat-deprecation spam on the math channel
    # (mirrors runner.py).
    try:
        import omni.log

        omni.log.get_log().set_channel_level(
            "isaaclab.utils.math", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE
        )
    except Exception as e:  # pragma: no cover - best-effort log tidy-up
        print(f"[init_calib] could not raise isaaclab.utils.math log level: {e!r}", flush=True)

    # Project root on sys.path so `learning.*` / `wrappers.*` resolve regardless of CWD.
    sys.path.insert(0, _PROJECT_ROOT)

    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from skrl.utils import set_seed

    from configs.manager import ConfigManager
    from learning.env_setup import build_env
    from wrappers.recording import CAMERA_KEY

    # ---- config (same loader as training) ----
    loaded = ConfigManager.load(args.config)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    ppo_cfg = loaded["ppo_cfg"]
    controller_cfg = loaded["controller_cfg"]
    noise_cfg = loaded["noise_cfg"]
    sensor_cfg = loaded["sensor_cfg"]

    agent_type = str(runner_cfg.agent_type).lower()
    if agent_type not in ("sac", "ppo"):
        raise ValueError(
            f"runner_cfg.agent_type must be 'sac' or 'ppo', got {runner_cfg.agent_type!r}"
        )

    # CLI > YAML for the runner-level fields we expose.
    if args.task is not None: runner_cfg.task = args.task
    if args.seed is not None: runner_cfg.seed = args.seed

    # Calibration always films exactly one env: force single-env, single-agent
    # regardless of what the config requests (cheaper boot, less GPU).
    runner_cfg.num_envs = 1
    runner_cfg.num_agents = 1
    print("[init_calib] forcing num_envs=1, num_agents=1 (single env).", flush=True)

    # Full-size camera. build_env reads the resolution AND placement off sac_cfg.recorder
    # (the tiny 240x180 grid-tile default); override here so the single-env video is full size
    # and optionally repositioned (e.g. raise the camera in world z).
    sac_cfg.recorder.width = args.width
    sac_cfg.recorder.height = args.height
    if args.camera_pos is not None:
        sac_cfg.recorder.camera_pos = tuple(args.camera_pos)
    if args.camera_quat is not None:
        sac_cfg.recorder.camera_quat = tuple(args.camera_quat)
    print(
        f"[init_calib] camera resolution: {args.width}x{args.height}, "
        f"pos={tuple(sac_cfg.recorder.camera_pos)}, quat={tuple(sac_cfg.recorder.camera_quat)}.",
        flush=True,
    )

    # Resolve the seed to a CONCRETE value (so a default/-1 "random" run is still
    # reproducible) and print it. Pass the printed value via --seed to replay the exact
    # same sequence of resets.
    seed = runner_cfg.seed
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)
    runner_cfg.seed = seed
    set_seed(seed)
    print(f"[init_calib] seed={seed} (pass --seed {seed} to reproduce this run).", flush=True)

    # ---- env (identical to training, camera forced on) ----
    env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs = build_env(
        args, runner_cfg, sac_cfg, ppo_cfg, controller_cfg, noise_cfg, sensor_cfg,
        agent_type, force_camera=True,
    )

    train_exc = None
    try:
        device = torch.device(args.device)
        assert total_envs == 1, f"expected a single env, got total_envs={total_envs}"

        # Resolve the injected camera and the per-step sim duration.
        scene = env.unwrapped.scene
        if not hasattr(scene, "sensors") or CAMERA_KEY not in scene.sensors:
            raise RuntimeError(
                f"calibration: expected a TiledCamera at scene.sensors[{CAMERA_KEY!r}] "
                "but none was found (build_env(force_camera=True) should have injected it)."
            )
        camera = scene.sensors[CAMERA_KEY]

        sim_dt = float(env.unwrapped.cfg.sim.dt)
        decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))
        dt_per_step = sim_dt * decimation
        steps_per_clip = max(1, round(args.settle_seconds / dt_per_step))
        fps = args.fps if args.fps > 0 else max(1, round(1.0 / dt_per_step))

        action_dim = env.action_space.shape[0]

        hold_mode = args.action_mode == "hold"
        if hold_mode and not hasattr(env.unwrapped, "step_sim_no_action"):
            raise RuntimeError(
                "--action_mode hold requires env.unwrapped.step_sim_no_action() (steps "
                f"physics + renders without applying an action); {type(env.unwrapped).__name__} "
                "has none. Use --action_mode zero."
            )
        zero_action = (
            None if hold_mode
            else torch.zeros((total_envs, action_dim), dtype=torch.float32, device=device)
        )

        print(
            f"[init_calib] num_resets={args.num_resets}, settle_seconds={args.settle_seconds} "
            f"-> {steps_per_clip} frames/clip (dt_per_step={dt_per_step:.5f}s), "
            f"action_mode={args.action_mode}, action_dim={action_dim}, gif_fps={fps}",
            flush=True,
        )

        # Open the streaming writer BEFORE the loop so each frame goes straight to disk.
        out_path = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        append_frame, finalize_writer = _open_frame_writer(out_path, fps)

        # ---- capture loop: (reset -> settle) x num_resets ----
        # no_grad: controller EMA buffers otherwise leak an unbounded autograd graph.
        # Frames are streamed to disk one at a time (constant memory) — never buffered.
        u = env.unwrapped
        # One wrapped reset up front: it initializes the wrapper chain AND clears gymnasium's
        # OrderEnforcing guard (step() before any reset() raises ResetNeeded).
        env.reset()
        # Per iteration, call the env's own _reset_idx() to force a real, full re-randomization
        # (+ re-grasp). skrl's IsaacLabWrapper.reset() only resets the sim the FIRST time
        # (it flips `_reset_once`, then returns cached obs — in training the env auto-resets
        # inside step()), so repeated env.reset() would be a no-op and freeze the scene after
        # reset #1. _reset_idx is self-contained (assumes all envs reset together) and the
        # adapter's per-reset randomization is patched onto it.
        reset_ids = torch.arange(u.num_envs, device=u.device)
        # Active hold: these factory/automate envs use operational-space (effort) control, which
        # is open-loop unstable if the torque isn't recomputed each substep — step_sim_no_action
        # alone reuses the last reset torque and the arm diverges/flies off (taking the plug).
        # move_gripper_in_place() re-targets the CURRENT pose, keeps the gripper closed, and
        # re-runs generate_ctrl_signals each substep, so the arm holds stably.
        _active_hold = hasattr(u, "move_gripper_in_place")
        progress = _make_progress(args.num_resets)
        n_frames = 0
        with torch.no_grad():
            for _r in progress:
                u._reset_idx(reset_ids)
                for _ in range(steps_per_clip):
                    if hold_mode:
                        # Hold the post-reset pose. One control step = `decimation` physics
                        # substeps (matches env.step cadence); recompute the holding torque
                        # each substep so the controller actively stays in place.
                        for _ in range(decimation):
                            if _active_hold:
                                u.move_gripper_in_place(0.0)
                            u.step_sim_no_action()
                    else:
                        env.step(zero_action)
                    # Refresh the TiledCamera. step_sim_no_action()'s sim.step(render=True) does
                    # NOT drive the RTX-sensor render that DirectRLEnv.step() triggers via an
                    # explicit sim.render() (factory_env render_interval path) — so without this
                    # the camera buffer froze on its first frame. Render, then force the sensor
                    # to recompute its output before reading.
                    u.sim.render()
                    camera.update(dt=0.0, force_recompute=True)
                    append_frame(_read_rgb(camera)[0].numpy())  # (H, W, 3) uint8 -> disk
                    n_frames += 1

        final_path = finalize_writer()
        print(f"[init_calib] wrote {n_frames} frames -> {final_path}", flush=True)

    except BaseException as e:
        train_exc = e
        import traceback
        print(f"[init_calib] raised {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[init_calib] env.close() raised: {e!r}", flush=True)
        sys.stdout.flush(); sys.stderr.flush()

        if train_exc is not None:
            os._exit(1)

        # Mirror runner.py: tear down Isaac then force a deterministic clean exit
        # (Isaac's C++/CUDA shutdown frequently segfaults after all work is done).
        try:
            simulation_app.close()
        except Exception as e:
            print(f"[init_calib] simulation_app.close() raised: {e!r}", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
