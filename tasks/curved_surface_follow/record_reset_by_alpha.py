#!/usr/bin/env python
"""Reset-snapshot video for the CURVED surface task — ONE env per curvature level (alpha).

100% STANDALONE and independent of ``learning/recording_eval.py``'s ``collect_reset_snapshots``
(which tiles a single-env grid and cannot label per-env curvature). The curved task assigns a FIXED
per-env curvature ``alpha`` round-robin over ``n_curvature_levels`` levels, so if we build exactly
``n_curvature_levels`` envs, env ``i`` gets ``alpha = i/(n-1)`` — one env per alpha, already in order.

This script:
  * boots Isaac and builds the curved env with ``num_envs == n_curvature_levels`` (one env per alpha),
    the recorder ``TiledCamera`` forced on (``build_env(force_camera=True)``);
  * repeats a fresh full re-randomization (``_reset_idx`` on ALL envs) ``--n_resets`` times;
  * after each reset, HOLDS every arm at its spawn fingertip pose (no policy) for ``--hold_s`` seconds
    of sim time, capturing one frame per step so spawn dynamics (a settle/pop) are visible;
  * tiles the envs in increasing-alpha order, LABELS each tile with its alpha value (and the spawn
    tip-height above the curved surface), and writes one mp4.

The result shows ``--n_resets`` resets from each of the ``n_curvature_levels`` alphas side by side.

Usage (run from the ``general`` conda env, which has isaacsim):
    conda run -n general python tasks/curved_surface_follow/record_reset_by_alpha.py \
        --n_resets 10 --headless --out runs/curved_reset_by_alpha.mp4

Like ``runner.py`` / ``init_calib.py``, the IsaacLab ``AppLauncher`` must boot before any
``isaaclab.*`` imports, so those live inside ``main()``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# tasks/curved_surface_follow/ -> project root is three levels up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "tests", "curved_surface_test.yaml")
_CURVED_TASK = "Isaac-FlatSurfaceFollow-Curved-Direct-v0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curved-surface reset snapshots, one env per alpha level.")
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG,
                   help=f"Base YAML (runner.py format). Default {_DEFAULT_CONFIG}.")
    p.add_argument("--task", type=str, default=_CURVED_TASK,
                   help=f"Task id to reset-film (any curved-surface subclass with per-env alpha, e.g. the "
                        f"wiping task). Default {_CURVED_TASK}.")
    p.add_argument("--n_levels", type=int, default=11,
                   help="Number of curvature levels = number of envs (one per alpha). Default 11 "
                        "-> alpha in {0, 0.1, ..., 1.0}.")
    p.add_argument("--n_resets", type=int, default=10,
                   help="Fresh full re-randomizations to film per alpha (default 10).")
    p.add_argument("--hold_s", type=float, default=0.6,
                   help="Seconds of sim time to hold at the spawn pose after each reset (default 0.6). "
                        "One captured frame per control step.")
    p.add_argument("--out", type=str, default=os.path.join(_PROJECT_ROOT, "runs", "curved_reset_by_alpha.mp4"),
                   help="Output mp4 path.")
    p.add_argument("--fps", type=int, default=15, help="Video fps (default 15 = env step rate).")
    p.add_argument("--tile_w", type=int, default=384, help="Per-tile camera width (default 384).")
    p.add_argument("--tile_h", type=int, default=288, help="Per-tile camera height (default 288).")
    p.add_argument("--camera_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Camera position offset [x y z] (m) from each env origin. Default: the "
                        "recorder's known-good 3/4 view. Lower Z / add -Y for a more side-on ridge view.")
    p.add_argument("--camera_quat", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"),
                   help="Camera orientation quaternion [w x y z]. Default: recorder's downward view.")
    p.add_argument("--seed", type=int, default=None, help="Override runner_cfg.seed (-1 = random).")
    AppLauncher.add_app_launcher_args(parser=p)  # adds --headless, --device, --enable_cameras
    return p


def _annotate(frame, alpha: float, tip_mm: float):
    """Draw a prominent ``alpha = X.XX`` label (top) + spawn tip-height read-out (bottom) on a tile."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8).copy())
    d = ImageDraw.Draw(img)
    W, H = img.size
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, W // 16))
        small = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, W // 24))
    except Exception:
        big = small = ImageFont.load_default()

    def _box(text, font, xy, fill, txt_rgb=(255, 255, 255)):
        try:
            tb = d.textbbox((0, 0), text, font=font); tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = d.textsize(text, font=font)
        x, y = xy
        d.rectangle([x - 4, y - 3, x + tw + 4, y + th + 5], fill=fill)
        d.text((x, y), text, font=font, fill=txt_rgb)
        return tw, th

    # alpha label, top-centre, colour ramps grey(flat)->red(most curved) so difficulty reads at a glance.
    r = int(60 + 175 * float(alpha)); g = int(120 - 70 * float(alpha)); b = int(70 - 40 * float(alpha))
    txt = f"alpha = {float(alpha):.2f}"
    try:
        tb = d.textbbox((0, 0), txt, font=big); tw = tb[2] - tb[0]
    except Exception:
        tw = d.textsize(txt, font=big)[0]
    _box(txt, big, ((W - tw) // 2, 6), (r, max(0, g), max(0, b)))
    # spawn tip height above the curved surface (mm), bottom-left.
    _box(f"tip {tip_mm:+.1f} mm", small, (6, H - max(14, W // 22) - 8), (25, 25, 28))
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    args = build_parser().parse_args()
    args.enable_cameras = True  # the recorder camera needs IsaacLab's render pipeline

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    sys.path.insert(0, _PROJECT_ROOT)

    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401  registers Isaac-* gym ids
    from skrl.utils import set_seed

    from configs.manager import ConfigManager
    from learning.env_setup import build_env
    from learning.recording_eval import read_camera_rgb, set_camera_active
    from learning import surface_viz as sv
    from wrappers.recording import CAMERA_KEY
    from wrappers.recording_grid import write_video

    try:
        import omni.log
        omni.log.get_log().set_channel_level(
            "isaaclab.utils.math", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
    except Exception:
        pass

    # ---- config (same loader as training) ----
    loaded = ConfigManager.load(args.config)
    runner_cfg = loaded["runner_cfg"]
    sac_cfg = loaded["sac_cfg"]
    ppo_cfg = loaded["ppo_cfg"]
    controller_cfg = loaded["controller_cfg"]
    noise_cfg = loaded["noise_cfg"]
    sensor_cfg = loaded["sensor_cfg"]
    keypoint_servo_cfg = loaded["keypoint_servo_cfg"]
    agent_type = str(runner_cfg.agent_type).lower()

    n = int(args.n_levels)
    runner_cfg.task = args.task
    runner_cfg.num_envs = n
    runner_cfg.num_agents = 1
    # Keep the task's level count in lockstep with the env count so env i -> alpha i/(n-1) exactly.
    if not isinstance(runner_cfg.env_cfg_overrides, dict):
        runner_cfg.env_cfg_overrides = {}
    runner_cfg.env_cfg_overrides["task.n_curvature_levels"] = n
    if args.seed is not None:
        runner_cfg.seed = args.seed

    # Full-size-ish per-tile camera + placement (side-on so the ridge profile shows).
    sac_cfg.recorder.width = int(args.tile_w)
    sac_cfg.recorder.height = int(args.tile_h)
    if args.camera_pos is not None:
        sac_cfg.recorder.camera_pos = tuple(args.camera_pos)
    if args.camera_quat is not None:
        sac_cfg.recorder.camera_quat = tuple(args.camera_quat)

    seed = runner_cfg.seed
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)
    runner_cfg.seed = seed
    set_seed(seed)
    print(f"[reset_by_alpha] seed={seed}, n_levels={n}, n_resets={args.n_resets}", flush=True)

    env, ctrl_wrapper, is_automate_assembly, env_cfg, total_envs = build_env(
        args, runner_cfg, sac_cfg, ppo_cfg, controller_cfg, noise_cfg, sensor_cfg,
        agent_type, keypoint_servo_cfg=keypoint_servo_cfg, force_camera=True,
    )

    try:
        uenv = env.unwrapped
        scene = uenv.scene
        if not hasattr(scene, "sensors") or CAMERA_KEY not in scene.sensors:
            raise RuntimeError(f"recorder camera {CAMERA_KEY!r} not injected into the scene.")
        camera = scene.sensors[CAMERA_KEY]
        num_envs = int(uenv.num_envs)
        assert num_envs == n, f"expected {n} envs (one per alpha), got {num_envs}"

        alpha = uenv.alpha.detach().cpu().numpy()               # (n,) per-env curvature, already in order
        order = list(np.argsort(alpha))                        # increasing-alpha tile order (defensive)
        cols = min(4, num_envs)
        rows = int(math.ceil(num_envs / cols))
        fps = int(args.fps)
        hold_frames = max(1, int(round(float(args.hold_s) * fps)))
        dt = float(uenv.physics_dt)
        print(f"[reset_by_alpha] {num_envs} envs ({rows}x{cols}), {args.n_resets} resets x "
              f"{hold_frames} held frames; alphas={[round(float(a), 2) for a in alpha]}", flush=True)

        all_ids = torch.arange(num_envs, device=uenv.device)
        out_frames: list[np.ndarray] = []

        set_camera_active(camera, True)
        with torch.no_grad():
            env.reset()  # one-time init reset
            for r in range(int(args.n_resets)):
                uenv._reset_idx(all_ids)                         # force a fresh full re-randomization
                uenv._compute_intermediate_values(dt)
                # Hold every arm at its spawn fingertip pose so anything that moves is spawn dynamics.
                hold_pos = uenv.fingertip_midpoint_pos.clone()
                hold_quat = uenv.fingertip_midpoint_quat.clone()
                for _s in range(hold_frames):
                    uenv.generate_ctrl_signals(
                        ctrl_target_fingertip_midpoint_pos=hold_pos,
                        ctrl_target_fingertip_midpoint_quat=hold_quat,
                        ctrl_target_gripper_dof_pos=0.0,
                    )
                    uenv.step_sim_no_action()
                    for _ in range(2):                           # render (1-frame camera latency)
                        uenv.sim.render()
                    try:
                        camera.update(0.0)
                    except TypeError:
                        camera.update()
                    rgb = read_camera_rgb(camera)                # (n, H, W, 3) cpu uint8
                    snap = uenv.viz_snapshot()
                    tip_mm = snap["tip_surface_dist"].numpy() * 1000.0
                    tiles = [
                        _annotate(rgb[e].numpy(), float(alpha[e]), float(tip_mm[e]))
                        for e in order
                    ]
                    out_frames.append(sv.montage(tiles, rows, cols))
                print(f"[reset_by_alpha] reset {r + 1}/{args.n_resets} done "
                      f"(mean spawn tip height {float(snap['tip_surface_dist'].numpy().mean()) * 1000:.2f} mm)",
                      flush=True)
        set_camera_active(camera, False)

        out_path = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        video = np.stack(out_frames)                             # (T, H, W, 3)
        base = os.path.splitext(out_path)[0]
        written = write_video(video, base, fps=fps, fmt="mp4")
        print(f"[reset_by_alpha] wrote {written} ({video.shape[0]} frames, {rows}x{cols} alphas)", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
