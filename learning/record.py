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
from datetime import datetime


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
    p.add_argument("--wandb_group", nargs="*", default=None,
                   help="wandb-tag mode: only process runs in these wandb group(s) (the {method}_{angle} "
                        "group name). The primary selector alongside --wandb_tag. Alias of --methods.")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Deprecated alias of --wandb_group.")
    p.add_argument("--download_only", action="store_true",
                   help="wandb-tag mode: fetch the ckpt/config tree but do NOT eval/record.")
    p.add_argument("--force", action="store_true",
                   help="wandb-tag mode: re-download ckpt_best.pt even if already present.")
    # --- from-wandb eval/record pipeline (tag+group -> download -> run -> upload trace) ---
    p.add_argument("--mode", choices=["eval", "record"], default="record",
                   help="wandb-tag mode: 'eval' runs trainer.eval() and uploads a per-step trace to "
                        "each ORIGINAL run (no eval run created); 'record' additionally renders the "
                        "video locally and uploads a rec_-prefixed trace. Default: record.")
    p.add_argument("--no_upload", action="store_true",
                   help="wandb-tag mode: run eval/record but do NOT upload the trace back to wandb "
                        "(leaves it in the work dir for inspection).")
    p.add_argument("--keep_local", action="store_true",
                   help="wandb-tag mode: keep the per-run work dir (checkpoint + trace) after upload "
                        "instead of deleting it. Videos are always kept.")
    p.add_argument("--work_dir", type=str, default=None,
                   help="wandb-tag mode: scratch root for downloads/traces. "
                        "Default runs/_eval_tmp/{project}_{tag}. Cleaned per-run unless --keep_local.")
    p.add_argument("--video_dir", type=str, default=None,
                   help="record mode: where to persist videos. Default runs/eval_videos/{project}_{tag}.")
    p.add_argument("--num_envs", type=int, default=None, help="Override envs per run (passed to runner).")
    p.add_argument("--eval_timesteps", type=int, default=None,
                   help="eval mode: override eval_timesteps (env steps to roll out).")
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
    p.add_argument("--step_trace_out", type=str, default=None,
                   help="Single-agent mode: write a per-step / per-env trace parquet to this path "
                        "(passed through to runner). In wandb-tag mode the pipeline sets this itself.")
    p.add_argument("--overlay", type=str, action="append", default=None,
                   help="Deep-merge overlay YAML forwarded to runner.py --overlay (repeatable). In "
                        "eval mode this lets a sweep pin one parameter per invocation. For eval, "
                        "overlays carrying runner_cfg.env_cfg_overrides ARE allowed (unlike record).")
    p.add_argument("--trace_label", type=str, default=None,
                   help="wandb-tag mode: label folded into the uploaded trace filename -> "
                        "eval_<label>_<ts>.parquet (record: rec_eval_<label>_<ts>). Used by the sweep "
                        "wrapper to encode <param-label>_<value> so each swept value's file is distinct.")
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
    if getattr(args, "num_envs", None) is not None:
        argv += ["--num_envs", str(args.num_envs)]
    if args.checkpoint_step is not None:
        argv += ["--checkpoint_step", str(args.checkpoint_step)]
    if args.output_dir is not None:
        argv += ["--record_output_dir", args.output_dir]
    if getattr(args, "step_trace_out", None):
        argv += ["--step_trace_out", args.step_trace_out]
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


def _agent_index(r) -> int:
    name = r.name or ""
    if "_agent" in name:
        try:
            return int(name.rsplit("_agent", 1)[1])
        except ValueError:
            pass
    return 0


def _eval_cmd(agent_dir: str, trace_path: str, args) -> list[str]:
    """runner.py --mode eval on a single downloaded agent, writing the step trace to
    trace_path and creating NO wandb run. TB output (if any) is redirected under the
    work dir so it is cleaned with it."""
    cfg = os.path.join(agent_dir, "config.yaml")
    # Redirect the experiment dir to a LOCAL temp under the work dir. The downloaded
    # runtime_config's experiment.directory is the original (HPC) absolute path, which
    # skrl's SummaryWriter would try to create -> PermissionDenied on /nfs. Both --logdir
    # and --experiment_directory are pointed here so the final dir is local and cleaned.
    _evalrun = os.path.join(agent_dir, "_evalrun")
    cmd = [sys.executable, os.path.join(_PROJECT_ROOT, "learning", "runner.py"),
           "--mode", "eval",
           "--checkpoint", agent_dir,
           "--checkpoint_step", "best",
           "--num_agents", "1",
           "--no_wandb",
           "--eval_weights_only",
           "--step_trace_out", trace_path,
           "--logdir", _evalrun,
           "--experiment_directory", _evalrun]
    if os.path.isfile(cfg):
        cmd += ["--config", cfg]
    for ov in (args.record_config or []):  # allow eval overlays too (optional)
        cmd += ["--overlay", ov]
    for ov in (args.overlay or []):         # sweep / general overlays (env_cfg_overrides ok in eval)
        cmd += ["--overlay", ov]
    if args.num_envs is not None:
        cmd += ["--num_envs", str(args.num_envs)]
    if args.eval_timesteps is not None:
        cmd += ["--eval_timesteps", str(args.eval_timesteps)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.headless:
        cmd += ["--headless"]
    return cmd


def _record_cmd(agent_dir: str, trace_path: str, video_dir: str, args) -> list[str]:
    """record.py single-agent mode on a downloaded agent: renders the video to video_dir
    (persisted) and writes the step trace to trace_path (uploaded, then temp cleaned)."""
    cfg = os.path.join(agent_dir, "config.yaml")
    cmd = [sys.executable, os.path.abspath(__file__), "--agent_dir", agent_dir,
           "--checkpoint_step", "best", "--step_trace_out", trace_path,
           "--output_dir", video_dir]
    if os.path.isfile(cfg):
        cmd += ["--config", cfg]
    for ov in (args.record_config or []):
        cmd += ["--record_config", ov]
    for ov in (args.overlay or []):
        cmd += ["--record_config", ov]   # _record_single forwards --record_config as runner --overlay
    if args.num_trajectories is not None:
        cmd += ["--num_trajectories", str(args.num_trajectories)]
    if args.num_envs is not None:
        cmd += ["--num_envs", str(args.num_envs)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.headless:
        cmd += ["--headless"]
    if args.enable_cameras:
        cmd += ["--enable_cameras"]
    return cmd


def _run_from_wandb(args) -> None:
    """Unified from-wandb eval/record pipeline (NO local checkpoints kept, NO eval runs created).

    Selects runs by ``--wandb_tag`` (+ optional ``--wandb_group``), and for EACH run:
      1. download ``ckpt_best.pt`` + ``runtime_config.yaml`` into a scratch work dir,
      2. subprocess eval (``--mode eval``) or record (``--mode record``) that writes a per-step
         trace parquet + (record only) a video,
      3. upload the trace to that ORIGINAL run's Files as ``eval_<ts>`` / ``rec_eval_<ts>``
         (the parent holds the run object, so no run-id ever touches disk),
      4. delete the work dir (videos are persisted separately).
    ``--download_only`` stops after step 1; ``--no_upload`` skips step 3; ``--keep_local`` skips step 4.
    """
    mode = args.mode
    if mode == "record" and not args.download_only and not args.record_config:
        raise SystemExit("[record] --record_config is required for --mode record (unless --download_only).")
    import wandb

    project = args.wandb_project
    path = project if "/" in project else (f"{args.wandb_entity}/{project}" if args.wandb_entity else project)
    work_root = os.path.abspath(args.work_dir or os.path.join(
        _PROJECT_ROOT, "runs", "_eval_tmp", f"{project.replace('/', '_')}_{args.wandb_tag}"))
    video_root = os.path.abspath(args.video_dir or os.path.join(
        _PROJECT_ROOT, "runs", "eval_videos", f"{project.replace('/', '_')}_{args.wandb_tag}"))

    api = wandb.Api(timeout=60)
    runs = list(api.runs(path, filters={"tags": args.wandb_tag}))
    if args.wandb_run_filter:
        runs = [r for r in runs if args.wandb_run_filter in (r.name or "")]
    groups = args.wandb_group or args.methods
    if groups:
        runs = [r for r in runs if (r.group or "") in set(groups)]
    if not runs:
        raise SystemExit(f"[{mode}] no runs in {path} tagged {args.wandb_tag!r}"
                         + (f" in groups {groups}" if groups else "")
                         + (f" matching {args.wandb_run_filter!r}" if args.wandb_run_filter else ""))
    print(f"[{mode}] {len(runs)} run(s) in {path} tagged {args.wandb_tag!r}", flush=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # Optional label (e.g. "<param-label>_<value>" from the sweep wrapper) -> filename-safe.
    _lbl = ""
    if args.trace_label:
        import re as _re
        _lbl = _re.sub(r"[^A-Za-z0-9._-]+", "_", str(args.trace_label)).strip("_")
        _lbl = f"{_lbl}_" if _lbl else ""
    trace_name = f"{'rec_' if mode == 'record' else ''}eval_{_lbl}{ts}.parquet"

    ok: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    for r in sorted(runs, key=lambda r: r.name or ""):
        method = r.group or (r.name or "unknown").rsplit("_agent", 1)[0]
        ai = _agent_index(r)
        label = f"{method}/agent{ai} ({r.name})"
        agent_dir = os.path.join(work_root, method, str(ai))
        ck_dir = os.path.join(agent_dir, "checkpoints")
        best = os.path.join(ck_dir, "ckpt_best.pt")

        files = {fl.name for fl in r.files()}
        if "ckpt_best.pt" not in files:
            print(f"[{mode}] SKIP {label}: no ckpt_best.pt on wandb "
                  "(run predates the ckpt-best backup, or hasn't hit a best yet)", flush=True)
            skipped.append(label); continue
        if not (os.path.isfile(best) and not args.force):
            os.makedirs(ck_dir, exist_ok=True)
            print(f"[{mode}] downloading {label} ...", flush=True)
            r.file("ckpt_best.pt").download(root=ck_dir, replace=True)
        if "runtime_config.yaml" in files:
            r.file("runtime_config.yaml").download(root=agent_dir, replace=True)
            os.replace(os.path.join(agent_dir, "runtime_config.yaml"), os.path.join(agent_dir, "config.yaml"))
        else:
            print(f"[{mode}]   WARNING: {label} has no runtime_config.yaml; the run's exact "
                  "env can't be reconstructed.", flush=True)

        if args.download_only:
            ok.append(label); continue

        trace_path = os.path.join(agent_dir, trace_name)
        if mode == "eval":
            cmd = _eval_cmd(agent_dir, trace_path, args)
        else:
            cmd = _record_cmd(agent_dir, trace_path, os.path.join(video_root, method, str(ai)), args)

        print(f"[{mode}] running {label} ...", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[{mode}] FAILED {label} (rc={rc})", flush=True)
            failed.append(label)
        elif not os.path.isfile(trace_path):
            print(f"[{mode}] FAILED {label}: no trace written to {trace_path}", flush=True)
            failed.append(label)
        else:
            uploaded = True
            if not args.no_upload:
                try:
                    # r holds the ORIGINAL training run; upload_file adds the parquet to its
                    # Files (root=agent_dir -> stored under its basename). No run-id on disk,
                    # no eval run created.
                    r.upload_file(trace_path, root=agent_dir)
                    print(f"[{mode}] uploaded {trace_name} -> {r.entity}/{r.project}/{r.id}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{mode}] UPLOAD FAILED for {label}: {e!r}", flush=True)
                    uploaded = False
            (ok if uploaded else failed).append(label)

        if not args.keep_local:
            shutil.rmtree(agent_dir, ignore_errors=True)

    print(f"\n[{mode}] done: {len(ok)} ok, {len(failed)} failed, {len(skipped)} skipped.", flush=True)
    if failed:
        print(f"[{mode}] failed: {failed}", flush=True)
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    _project_root_on_path()
    if args.wandb_tag and args.agent_dir:
        raise SystemExit("[record] pass EITHER --agent_dir (single) OR --wandb_tag (batch), not both.")
    if args.wandb_tag:
        _run_from_wandb(args)
        return
    if not args.agent_dir:
        raise SystemExit("[record] --agent_dir is required (single-agent mode), or use --wandb_tag.")
    _record_single(args)


if __name__ == "__main__":
    main()
