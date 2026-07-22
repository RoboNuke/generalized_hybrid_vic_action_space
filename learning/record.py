#!/usr/bin/env python
"""Per-agent video recorder — record ONE trained agent, or a whole wandb tag.

Thin, friendly front-end over ``learning/runner.py``'s record mode.

SINGLE-AGENT mode: give it a single agent folder (the ``0/``, ``1/``, ... dir that
holds ``checkpoints/`` and the snapshotted ``config.yaml``) plus a record overlay
config; it loads that one agent's weights, rolls out, collects complete
trajectories, and writes an agent-specific best/median/worst grid video next to
the checkpoint (``<agent_dir>/videos/recording.mp4``).

WANDB-TAG mode: give it ``--wandb_entity/--wandb_project/--wandb_tag`` and it finds
every run with that tag, downloads each run's ``ckpt_best.pt`` from wandb, records
it, and writes each video to::

    runs/{project}/{tag}/{group_name}/{run_name}.mp4

where ``group_name`` is the run's wandb group (the config name, e.g. ``7_GAS_dyn``)
and ``run_name`` is the agent run (e.g. ``7_GAS_dyn_agent0``). The training config
isn't stored on wandb, so the per-group config is taken from
``--wandb_config_dir/{group_name}.yaml`` (defaults to the glued_surface configs).

Examples
--------
    # one agent
    python learning/record.py \
        --agent_dir runs/log_dir/1_fixed/0 \
        --record_config configs/exp_cfgs/glued_surface/_record_video.yaml \
        --checkpoint_step best --headless --enable_cameras

    # every run of a wandb tag, downloaded straight from wandb
    python learning/record.py \
        --wandb_entity hur --wandb_project surface_baselines --wandb_tag high-ent_high-gain \
        --record_config configs/exp_cfgs/glued_surface/_record_video.yaml \
        --headless --enable_cameras

The single-agent base config defaults to ``<agent_dir>/config.yaml`` (override with
``--config``). Everything in the record overlay is deep-merged over it.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys


def _project_root_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record a trained agent (or a whole wandb tag) to a grid video.")
    # --- single-agent mode ---
    p.add_argument(
        "--agent_dir", type=str, default=None,
        help="Single agent folder, e.g. runs/log_dir/1_fixed/0 "
             "(contains checkpoints/ckpt_*.pt and config.yaml). Omit when using --wandb_tag.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Base config. Defaults to <agent_dir>/config.yaml.",
    )
    # --- wandb-tag (batch) mode ---
    p.add_argument("--wandb_tag", type=str, default=None,
                   help="Record EVERY run carrying this wandb tag: download each run's ckpt_best.pt "
                        "and write runs/{project}/{tag}/{group}/{run_name}.mp4. Mutually exclusive "
                        "with --agent_dir.")
    p.add_argument("--wandb_entity", type=str, default="hur", help="wandb entity (wandb-tag mode).")
    p.add_argument("--wandb_project", type=str, default="surface_baselines",
                   help="wandb project (wandb-tag mode).")
    p.add_argument("--wandb_config_dir", type=str, default="configs/exp_cfgs/glued_surface",
                   help="Directory holding per-group training configs {group}.yaml (wandb-tag mode; "
                        "the config.yaml isn't stored on wandb, so we use the repo config by group).")
    p.add_argument("--wandb_run_filter", type=str, default=None,
                   help="Optional substring: in wandb-tag mode, only record runs whose name contains it "
                        "(e.g. '7_GAS' for one group, or '7_GAS_dyn_agent0' for a single agent).")
    p.add_argument("--wandb_keep_ckpts", action="store_true",
                   help="Keep the downloaded ckpt_best.pt files after recording (default: delete them "
                        "once the video is written, since they are ~0.85 GB each).")
    # --- shared record knobs ---
    p.add_argument(
        "--record_config", type=str, action="append", default=None,
        help="Record overlay YAML (the recorder/eval settings). Repeatable; later wins. "
             "e.g. configs/exp_cfgs/glued_surface/_record_video.yaml",
    )
    p.add_argument(
        "--num_trajectories", type=int, default=None,
        help="Collect at least this many trajectories before composing the grid "
             "(overrides recorder.num_trajectories).",
    )
    p.add_argument("--checkpoint_step", type=lambda v: v if v == "best" else int(v), default=None,
                   help="Specific ckpt step to load, or 'best' for the agent's ckpt_best.pt "
                        "(highest-success-rate checkpoint); default = latest. wandb-tag mode forces 'best'.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write the video (single-agent mode). "
                        "Default <agent_dir>/<recorder.output_subdir>.")
    p.add_argument("--device", type=str, default=None, help="Torch/sim device, e.g. cuda:0.")
    p.add_argument("--headless", action="store_true", help="Run Isaac headless (still records).")
    p.add_argument("--enable_cameras", action="store_true",
                   help="Usually auto-forced by recorder.enabled; pass to be explicit.")
    return p


def _record_single(args) -> None:
    """Record one agent via runner.py record mode (runner.main os._exit's on completion)."""
    agent_dir = os.path.abspath(args.agent_dir)
    if not os.path.isdir(agent_dir):
        raise SystemExit(f"[record] agent_dir not found: {agent_dir}")
    if not args.record_config:
        raise SystemExit("[record] --record_config is required (the recorder overlay YAML).")

    base_config = args.config or os.path.join(agent_dir, "config.yaml")
    if not os.path.isfile(base_config):
        raise SystemExit(
            f"[record] base config not found: {base_config} "
            "(expected the config.yaml snapshotted next to the checkpoint; pass --config to override)."
        )

    # Translate into runner.py record-mode flags.
    argv: list[str] = [
        "--record_agent_dir", agent_dir,
        "--config", base_config,
    ]
    for ov in args.record_config:
        argv += ["--overlay", ov]
    if args.num_trajectories is not None:
        argv += ["--num_trajectories", str(args.num_trajectories)]
    if args.checkpoint_step is not None:
        argv += ["--checkpoint_step", str(args.checkpoint_step)]
    if args.output_dir is not None:
        argv += ["--record_output_dir", args.output_dir]
    if args.device is not None:
        argv += ["--device", args.device]
    if args.headless:
        argv += ["--headless"]
    if args.enable_cameras:
        argv += ["--enable_cameras"]

    from learning import runner
    try:
        runner.main(argv)
    except BaseException as e:  # noqa: BLE001
        # In record mode runner.main() exits the process itself (os._exit) on BOTH
        # success and in-recording failure, so reaching here means the failure happened
        # BEFORE the recording guard's try block (config reload, env build, the SAC-only
        # / recorder-enabled checks). os._exit(1) now — before any atexit/Isaac teardown —
        # so a batch launcher correctly sees a nonzero exit and prints FAIL.
        import traceback
        print(f"[record] FAILED before recording started: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _record_from_wandb(args) -> None:
    """Find every run with --wandb_tag, download each ckpt_best.pt, record it, and write
    runs/{project}/{tag}/{group}/{run_name}.mp4. Each record runs in its own subprocess
    (runner.main os._exit's, so we can't loop in-process)."""
    if not args.record_config:
        raise SystemExit("[record] --record_config is required (the recorder overlay YAML).")
    import wandb

    entity, project, tag = args.wandb_entity, args.wandb_project, args.wandb_tag
    api = wandb.Api()
    runs = [r for r in api.runs(f"{entity}/{project}") if tag in r.tags]
    if args.wandb_run_filter:
        runs = [r for r in runs if args.wandb_run_filter in r.name]
    if not runs:
        raise SystemExit(f"[record] no runs carry tag {tag!r} in {entity}/{project}"
                         + (f" matching {args.wandb_run_filter!r}" if args.wandb_run_filter else ""))
    print(f"[record] {len(runs)} run(s) tagged {tag!r} in {entity}/{project}"
          + (f" matching {args.wandb_run_filter!r}" if args.wandb_run_filter else ""), flush=True)

    ok: list[str] = []
    failed: list[str] = []
    for r in sorted(runs, key=lambda r: r.name):
        group = r.group or r.name.rsplit("_agent", 1)[0]
        config = os.path.join(args.wandb_config_dir, f"{group}.yaml")
        if not os.path.isfile(config):
            print(f"[record] SKIP {r.name}: group config not found: {config}", flush=True)
            failed.append(r.name)
            continue

        base = os.path.join("runs", project, tag, group)
        agent_dir = os.path.join(base, "_ckpts", r.name)          # staging: ckpt + intermediate video
        ck_dir = os.path.join(agent_dir, "checkpoints")
        vid_dir = os.path.join(agent_dir, "vid")
        os.makedirs(ck_dir, exist_ok=True)
        dst = os.path.join(base, f"{r.name}.mp4")

        # 1) download ckpt_best.pt (skip if already present)
        ckpt = os.path.join(ck_dir, "ckpt_best.pt")
        if not os.path.isfile(ckpt):
            try:
                print(f"[record] downloading ckpt_best.pt for {r.name} ...", flush=True)
                r.file("ckpt_best.pt").download(root=ck_dir, replace=True)
            except Exception as e:  # noqa: BLE001
                print(f"[record] SKIP {r.name}: ckpt_best.pt download failed: {e}", flush=True)
                failed.append(r.name)
                continue

        # 2) record this agent in a fresh subprocess (single-agent mode)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--agent_dir", agent_dir, "--config", config,
               "--checkpoint_step", "best", "--output_dir", vid_dir]
        for ov in args.record_config:
            cmd += ["--record_config", ov]
        if args.num_trajectories is not None:
            cmd += ["--num_trajectories", str(args.num_trajectories)]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.headless:
            cmd += ["--headless"]
        if args.enable_cameras:
            cmd += ["--enable_cameras"]
        print(f"[record] recording {r.name} (group {group}) ...", flush=True)
        rc = subprocess.run(cmd).returncode

        # 3) place the produced video at runs/{project}/{tag}/{group}/{run_name}.mp4
        vids = sorted(glob.glob(os.path.join(vid_dir, "*.mp4")))
        if rc == 0 and vids:
            shutil.move(vids[0], dst)
            print(f"[record] wrote {dst}", flush=True)
            ok.append(r.name)
            if not args.wandb_keep_ckpts:
                shutil.rmtree(agent_dir, ignore_errors=True)
        else:
            print(f"[record] FAILED {r.name} (subprocess rc={rc}, mp4s found={vids})", flush=True)
            failed.append(r.name)

    print(f"\n[record] done: {len(ok)} ok, {len(failed)} failed.", flush=True)
    if failed:
        print(f"[record] failed: {failed}", flush=True)
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    _project_root_on_path()
    if args.wandb_tag and args.agent_dir:
        raise SystemExit("[record] pass EITHER --agent_dir (single) OR --wandb_tag (batch), not both.")
    if args.wandb_tag:
        _record_from_wandb(args)
        return
    if not args.agent_dir:
        raise SystemExit("[record] --agent_dir is required (single-agent mode), or use --wandb_tag.")
    _record_single(args)


if __name__ == "__main__":
    main()
