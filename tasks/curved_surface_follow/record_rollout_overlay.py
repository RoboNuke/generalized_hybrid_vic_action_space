#!/usr/bin/env python
"""Short surface-overlay rollout for the CURVED surface task (no trained checkpoint required).

Drives the peg with the repo's SCRIPTED keypoint-servo action override (servos the tip along the
per-env setpoint that follows the ridge) instead of a trained policy, so we get a meaningful
traversal of each curved surface, and renders it through the existing ``surface_tracking`` overlay
(in-scene keypoint balls + force/orientation gauges + top-down path inset) via
``learning.recording_eval.collect_recording``.

Builds ``n_levels`` envs (one per curvature alpha, round-robin) so the single grid shows the overlay
across the whole difficulty ladder at once. The policy action is ZERO on whatever dims the servo
leaves (with fix_orientation=false, the 3 EEF-rotation dims — zero holds orientation while the servo
drives translation).

Usage (run from the ``general`` conda env):
    conda run -n general python tasks/curved_surface_follow/record_rollout_overlay.py \
        --headless --out_dir runs/curved_rollout

Note: the overlay's in-scene keypoint balls are placed on the straight start->goal chord (the flat
overlay's assumption); the PEG follows the true curved ridge, so balls and peg can diverge in height.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "tests", "curved_surface_test.yaml")
_CURVED_TASK = "Isaac-FlatSurfaceFollow-Curved-Direct-v0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curved-surface scripted rollout with the surface overlay.")
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG, help=f"Base YAML. Default {_DEFAULT_CONFIG}.")
    p.add_argument("--task", type=str, default=_CURVED_TASK,
                   help=f"Task id (any curved-surface subclass, e.g. the wiping task). Default {_CURVED_TASK}. "
                        "For the wiping task also pass its config, e.g. --config "
                        "configs/exp_cfgs/tests/wiping_surface_test.yaml.")
    p.add_argument("--n_levels", type=int, default=6,
                   help="Curvature levels = number of envs (one per alpha, tiled together). Default 6 "
                        "-> alphas {0, 0.2, 0.4, 0.6, 0.8, 1.0}. Kept modest so the HIGH-RES tiles below "
                        "render in a reasonable time.")
    p.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "runs", "curved_rollout"),
                   help="Directory for the output mp4.")
    p.add_argument("--tile_w", type=int, default=800, help="Per-tile RGB width (default 800; high-res so "
                   "the overlay gauges/text no longer dominate the frame).")
    p.add_argument("--tile_h", type=int, default=600, help="Per-tile RGB height (default 600; 4:3).")
    p.add_argument("--press", type=float, default=0.0015,
                   help="How far (m) the scripted controller drives the target BELOW the surface to "
                        "maintain contact force (normal_offset = -press). Default 1.5 mm (~5-10 N).")
    p.add_argument("--lead", type=float, default=0.003,
                   help="Along-track lead (m) ahead of the setpoint — a small forward bias. Default 3 mm.")
    p.add_argument("--fps", type=int, default=15, help="Video fps (default 15 = env step rate).")
    p.add_argument("--seed", type=int, default=None, help="Override runner_cfg.seed (-1 = random).")
    AppLauncher.add_app_launcher_args(parser=p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    sys.path.insert(0, _PROJECT_ROOT)

    import torch

    import isaaclab_tasks  # noqa: F401
    from skrl.utils import set_seed

    from configs.manager import ConfigManager
    from learning.env_setup import build_env
    from learning.recording_eval import collect_recording
    from wrappers.recording import CAMERA_KEY

    try:
        import omni.log
        omni.log.get_log().set_channel_level(
            "isaaclab.utils.math", omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
    except Exception:
        pass

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
    # The surface tasks now hard-require the contact sensor for in-contact detection; enable it here so
    # the rollout works regardless of the base config (observe_in_contact stays off -> obs unchanged).
    sensor_cfg.contact.enabled = True
    runner_cfg.num_envs = n
    runner_cfg.num_agents = 1
    if not isinstance(runner_cfg.env_cfg_overrides, dict):
        runner_cfg.env_cfg_overrides = {}
    runner_cfg.env_cfg_overrides["task.n_curvature_levels"] = n
    # ROBOT-DRIVEN setpoint: the target sits ~one keypoint ahead of realized progress, so the servo
    # makes small reachable steps and holds a clean, stable contact pose (pace-driving marches the
    # target to the far edge and the servo slips the peg off-track chasing it).
    runner_cfg.env_cfg_overrides["task.setpoint_pace_driven"] = False
    # STIFF ROTATION gains (Forge default is only 28) so contact torque can't flop the peg to
    # horizontal — it establishes and HOLDS a clean tip-down, on-track contact on every curvature.
    # NOTE: this basic pose controller makes/holds contact well but does NOT traverse the ridge — a
    # 15 cm tip-down peg can't be dragged over the curve by an open-loop position servo (it stalls at
    # the base or slips off-track). Clean path-following on the curved contact task needs a LEARNED
    # variable-impedance policy — exactly the research point of this env.
    runner_cfg.env_cfg_overrides["ctrl.default_task_prop_gains"] = [800.0, 800.0, 800.0, 200.0, 200.0, 200.0]
    if args.seed is not None:
        runner_cfg.seed = args.seed

    # Fixed-gain POSE controller: no gain dims, so the servo drives translation with the baked
    # default stiffness and a ZERO policy action simply holds orientation (a variable-gain controller
    # would read zero action as zero stiffness -> no motion). control_type="pose" => 6 base dims.
    controller_cfg.control_type = "pose"

    # Scripted keypoint-servo = the "basic controller": each step it servos the peg tip toward the
    # current ridge setpoint, led ``lead`` m ahead along the path and driven ``press`` m BELOW the
    # surface (normal_offset < 0) so it maintains a real contact force as it climbs/descends the
    # curve — the whole point of the demo (watch the force readout stay near target through the hump).
    # fix_orientation=false leaves the 3 EEF-rotation dims to the (zero) policy, holding orientation
    # while the servo translates.
    keypoint_servo_cfg.enabled = True
    keypoint_servo_cfg.fix_orientation = False
    keypoint_servo_cfg.along_track_offset = float(args.lead)
    keypoint_servo_cfg.off_track_offset = 0.0
    keypoint_servo_cfg.normal_offset = -float(args.press)   # negative = press INTO the surface

    # Recorder: surface overlay, tile ALL envs from one episode (grid_select=all — ranking is
    # meaningless for a scripted rollout), full episodes for viewing.
    rec = sac_cfg.recorder
    rec.enabled = True
    rec.mode = "trajectories"
    # Wiping task -> the wiping overlay (waypoints coloured by cleaned state); else the surface-tracking
    # keypoint overlay.
    rec.overlay = "wiping" if "Wiping" in args.task else "surface_tracking"
    rec.grid_select = "all"
    rec.full_episode = True
    rec.ball_diameter_frac = 1.0
    rec.video_format = "mp4"
    rec.width = int(args.tile_w)
    rec.height = int(args.tile_h)
    rec.fps = int(args.fps)
    rec.num_trajectories = n  # one episode of n fresh spawns
    rec.inset_frac = 0.30     # shrink the top-down minimap so it stops covering the scene

    seed = runner_cfg.seed
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)
    runner_cfg.seed = seed
    set_seed(seed)
    print(f"[rollout_overlay] seed={seed}, n_levels={n}", flush=True)

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
        act_dim = int(env.action_space.shape[0])
        num_envs = int(uenv.num_envs)
        device = torch.device(args.device)
        print(f"[rollout_overlay] {num_envs} envs, policy action dim after keypoint-servo = {act_dim} "
              "(zero -> hold orientation; servo drives translation)", flush=True)

        class ZeroAgent:
            """Minimal agent: the scripted keypoint-servo does the driving, so the policy action is
            zero on the dims the servo leaves. Matches the (actions, extra) contract collect_recording
            expects; no critics are needed by the surface overlay."""

            def act(self, obs, state, timestep=0, timesteps=0):
                return torch.zeros((num_envs, act_dim), dtype=torch.float32, device=device), None

        out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        max_len = int(uenv.max_episode_length)

        with torch.no_grad():   # controller EMA buffers leak an autograd graph without this
            path = collect_recording(
                env=env,
                agent=ZeroAgent(),
                recorder_cfg=rec,
                camera=camera,
                max_episode_length=max_len,
                num_trajectories=n,
                output_dir=out_dir,
                gif_name=("wiping_rollout_overlay.mp4" if "Wiping" in args.task
                          else "curved_rollout_overlay.mp4"),
            )
        print(f"[rollout_overlay] wrote {path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
