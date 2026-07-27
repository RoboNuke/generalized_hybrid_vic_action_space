#!/usr/bin/env python
"""Per-agent video recorder — record ONE trained agent, or a whole wandb tag.

Thin, friendly front-end over ``learning/runner.py``'s record mode.

SINGLE-AGENT mode: give it a single agent folder (the ``0/``, ``1/``, ... dir that
holds ``checkpoints/`` and the snapshotted ``config.yaml``) plus a record overlay
config; it loads that one agent's weights, rolls out, collects complete
trajectories, and writes an agent-specific best/median/worst grid video next to
the checkpoint (``<agent_dir>/videos/recording.mp4``).

WANDB-TAG mode: give it ``--wandb_tag`` (aka ``--tag``) with ``--project``/``--entity`` and it
finds every run with that tag, downloads each run's ``ckpt_best.pt`` + its ``runtime_config.yaml``
(the EXACT training env) into ``runs/wandb/{project}_{tag}/{method}/{agent}/``, and records each
agent in place (``<agent>/videos/recording.mp4``) -- one subprocess per agent. ``--download_only``
fetches without rendering, ``--methods`` filters by wandb group, ``--force`` re-downloads.

This is the ONE recording entry point (single agent + wandb batch); the recorder itself is just
runner.py with a camera, and WHAT/HOW it draws is set in the ``_*.yaml`` record config
(``recorder.mode`` = trajectories | reset_snapshots, ``recorder.overlay`` = surface_tracking | none).

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


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    p.add_argument("--wandb_tag", "--tag", dest="wandb_tag", type=str, default=None,
                   help="Record EVERY run carrying this wandb tag: download each run's ckpt_best.pt + its "
                        "runtime_config.yaml into runs/wandb/{project}_{tag}/{method}/{agent}/ and record "
                        "each agent in place. Mutually exclusive with --agent_dir.")
    p.add_argument("--wandb_entity", "--entity", dest="wandb_entity", type=str, default="hur",
                   help="wandb entity (wandb-tag mode).")
    p.add_argument("--wandb_project", "--project", dest="wandb_project", type=str, default="surface_baselines",
                   help="wandb project (wandb-tag mode). Accepts 'entity/project' too.")
    p.add_argument("--wandb_run_filter", type=str, default=None,
                   help="Optional substring: only record runs whose name contains it (wandb-tag mode).")
    p.add_argument("--methods", nargs="*", default=None,
                   help="wandb-tag mode: only record these wandb groups/methods (by group name).")
    p.add_argument("--download_only", action="store_true",
                   help="wandb-tag mode: fetch the ckpt/config tree but do NOT render.")
    p.add_argument("--force", action="store_true",
                   help="wandb-tag mode: re-download ckpt_best.pt even if already present.")
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
    """Download every run carrying --wandb_tag (its ckpt_best.pt + runtime_config.yaml) into
    runs/wandb/{project}_{tag}/{method}/{agent}/, then record each agent in single-agent mode -- one
    subprocess per agent, because runner.main os._exit's and would otherwise stop the batch. Uses the
    run's OWN runtime_config.yaml (the exact training env) as the base config. --download_only fetches
    the tree without rendering; --methods filters by wandb group; --force re-downloads."""
    if not args.download_only and not args.record_config:
        raise SystemExit("[record] --record_config is required for --wandb_tag (unless --download_only).")
    import wandb

    project = args.wandb_project
    path = project if "/" in project else (f"{args.wandb_entity}/{project}" if args.wandb_entity else project)
    out = os.path.abspath(args.output_dir or os.path.join(
        _PROJECT_ROOT, "runs", "wandb", f"{project.replace('/', '_')}_{args.wandb_tag}"))

    api = wandb.Api(timeout=60)
    runs = list(api.runs(path, filters={"tags": args.wandb_tag}))
    if args.wandb_run_filter:
        runs = [r for r in runs if args.wandb_run_filter in (r.name or "")]
    if not runs:
        raise SystemExit(f"[record] no runs in {path} tagged {args.wandb_tag!r}"
                         + (f" matching {args.wandb_run_filter!r}" if args.wandb_run_filter else ""))
    print(f"[record] {len(runs)} run(s) in {path} tagged {args.wandb_tag!r} -> {out}", flush=True)

    def _agent_index(r):
        name = r.name or ""
        if "_agent" in name:
            try:
                return int(name.rsplit("_agent", 1)[1])
            except ValueError:
                pass
        return 0

    agent_dirs: list[str] = []
    skipped: list[str] = []
    for r in sorted(runs, key=lambda r: r.name or ""):
        method = r.group or (r.name or "unknown").rsplit("_agent", 1)[0]
        if args.methods and method not in args.methods:
            continue
        ai = _agent_index(r)
        agent_dir = os.path.join(out, method, str(ai))
        ck_dir = os.path.join(agent_dir, "checkpoints")
        best = os.path.join(ck_dir, "ckpt_best.pt")
        label = f"{method}/agent{ai} ({r.name})"
        if os.path.isfile(best) and not args.force:
            print(f"[record] have {label} (skip download; --force to refresh)", flush=True)
            agent_dirs.append(agent_dir); continue
        files = {fl.name for fl in r.files()}
        if "ckpt_best.pt" not in files:
            print(f"[record] SKIP {label}: no ckpt_best.pt on wandb "
                  "(run predates the ckpt-best backup, or hasn't hit a best yet)", flush=True)
            skipped.append(label); continue
        os.makedirs(ck_dir, exist_ok=True)
        print(f"[record] downloading {label} ...", flush=True)
        r.file("ckpt_best.pt").download(root=ck_dir, replace=True)
        if "runtime_config.yaml" in files:
            r.file("runtime_config.yaml").download(root=agent_dir, replace=True)
            os.replace(os.path.join(agent_dir, "runtime_config.yaml"), os.path.join(agent_dir, "config.yaml"))
        else:
            print(f"[record]   WARNING: {label} has no runtime_config.yaml; pass --config for it.", flush=True)
        agent_dirs.append(agent_dir)

    if not agent_dirs:
        raise SystemExit("[record] no agents with ckpt_best.pt were fetched -- nothing to record.")
    if args.download_only:
        print(f"[record] tree ready at {out} (--download_only; skipped {len(skipped)}).", flush=True)
        return

    ok: list[str] = []
    failed: list[str] = []
    for ad in agent_dirs:
        cfg = os.path.join(ad, "config.yaml")
        cmd = [sys.executable, os.path.abspath(__file__), "--agent_dir", ad, "--checkpoint_step", "best"]
        if os.path.isfile(cfg):
            cmd += ["--config", cfg]
        for ov in (args.record_config or []):
            cmd += ["--record_config", ov]
        if args.num_trajectories is not None:
            cmd += ["--num_trajectories", str(args.num_trajectories)]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.headless:
            cmd += ["--headless"]
        if args.enable_cameras:
            cmd += ["--enable_cameras"]
        print(f"[record] recording {ad} ...", flush=True)
        rc = subprocess.run(cmd).returncode
        (ok if rc == 0 else failed).append(ad)

    print(f"\n[record] done: {len(ok)} ok, {len(failed)} failed, {len(skipped)} skipped.", flush=True)
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
