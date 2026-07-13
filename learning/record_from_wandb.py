#!/usr/bin/env python
"""Pull best checkpoints from a wandb project+tag and record videos locally — no rsync.

Finds every wandb run in ``<entity>/<project>`` carrying ``--tag``, buckets them by their wandb
**group** (which is the method, e.g. ``1_fixed`` / ``5_GAS`` — the seeds of a method share a group),
downloads each run's backed-up **``ckpt_best.pt``** and **``runtime_config.yaml``** into a local
per-agent tree, then runs the existing recorder over that tree:

    <output_dir>/<method>/<agent_index>/checkpoints/ckpt_best.pt
    <output_dir>/<method>/<agent_index>/config.yaml

Requires runs trained with the ckpt-best wandb backup (MetricWriter.save_file_live) so ``ckpt_best.pt``
is in the run's Files. Camera rendering must run from the ``general`` env (isaacsim 5.1), so invoke
this whole script with that interpreter, e.g.:

    /home/hunter/miniconda3/envs/general/bin/python learning/record_from_wandb.py \
        --project surface_baselines --entity hur --tag keypoint-rework \
        --record_config configs/exp_cfgs/glued_surface/_record_stills.yaml

Add ``--download_only`` to just fetch the tree (skip rendering). Anything after ``--`` is forwarded
verbatim to ``learning/record.py`` (via record_group.bash), e.g. ``-- --num_trajectories 24``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True,
                   help="wandb project, or 'entity/project'. e.g. surface_baselines")
    p.add_argument("--entity", default=None, help="wandb entity (e.g. hur); omit if given in --project.")
    p.add_argument("--tag", required=True, help="Only runs carrying this wandb tag. e.g. keypoint-rework")
    p.add_argument("--record_config", default=None,
                   help="Record overlay YAML passed to record_group.bash (required unless --download_only). "
                        "e.g. configs/exp_cfgs/glued_surface/_record_stills.yaml")
    p.add_argument("--output_dir", default=None,
                   help="Where to build the <method>/<agent>/ tree. Default runs/wandb/<project>_<tag>.")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Optional: only these groups/methods (by wandb group name).")
    p.add_argument("--checkpoint_step", default="best",
                   help="Forwarded to record.py (default 'best' -> ckpt_best.pt).")
    p.add_argument("--download_only", action="store_true", help="Fetch the tree but do not render.")
    p.add_argument("--force", action="store_true", help="Re-download even if ckpt_best.pt already exists.")
    p.add_argument("record_args", nargs=argparse.REMAINDER,
                   help="Everything after '--' is forwarded to record.py.")
    return p


def _agent_index(run) -> int:
    ai = run.config.get("agent_index")
    if ai is not None:
        return int(ai)
    # Fallback: parse the trailing _agent<N> off the run name.
    name = run.name or ""
    if "_agent" in name:
        try:
            return int(name.rsplit("_agent", 1)[1])
        except ValueError:
            pass
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if not args.download_only and not args.record_config:
        raise SystemExit("--record_config is required unless --download_only is set.")

    import wandb

    path = args.project if "/" in args.project else (
        f"{args.entity}/{args.project}" if args.entity else args.project)
    out = os.path.abspath(args.output_dir or os.path.join(
        _PROJECT_ROOT, "runs", "wandb", f"{args.project.replace('/', '_')}_{args.tag}"))

    api = wandb.Api(timeout=60)
    runs = list(api.runs(path, filters={"tags": args.tag}))
    if not runs:
        raise SystemExit(f"no runs in {path} tagged '{args.tag}'.")
    print(f"[wandb-rec] {len(runs)} run(s) in {path} tagged '{args.tag}' -> {out}", flush=True)

    got, skipped = [], []
    for r in runs:
        method = r.group or (r.name or "unknown").rsplit("_agent", 1)[0]
        if args.methods and method not in args.methods:
            continue
        ai = _agent_index(r)
        agent_dir = os.path.join(out, method, str(ai))
        ckpt_dir = os.path.join(agent_dir, "checkpoints")
        best_path = os.path.join(ckpt_dir, "ckpt_best.pt")
        label = f"{method}/agent{ai} ({r.name})"

        if os.path.isfile(best_path) and not args.force:
            print(f"[wandb-rec] have {label} (skip download; --force to refresh)", flush=True)
            got.append(agent_dir)
            continue

        files = {f.name for f in r.files()}
        if "ckpt_best.pt" not in files:
            print(f"[wandb-rec] SKIP {label}: no ckpt_best.pt in wandb Files "
                  "(run predates the ckpt-best backup, or hasn't hit a best yet)", flush=True)
            skipped.append(label)
            continue

        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"[wandb-rec] downloading {label} ...", flush=True)
        r.file("ckpt_best.pt").download(root=ckpt_dir, replace=True)
        # Config: runtime_config.yaml is attached on run finish; save it as the config.yaml record.py expects.
        if "runtime_config.yaml" in files:
            r.file("runtime_config.yaml").download(root=agent_dir, replace=True)
            os.replace(os.path.join(agent_dir, "runtime_config.yaml"), os.path.join(agent_dir, "config.yaml"))
        else:
            print(f"[wandb-rec]   WARNING: {label} has no runtime_config.yaml yet (run not finished?); "
                  "record.py will need --config for this agent.", flush=True)
        got.append(agent_dir)

    print(f"\n[wandb-rec] downloaded/kept {len(got)} agent(s); skipped {len(skipped)}.", flush=True)
    for s in skipped:
        print(f"[wandb-rec]   skipped: {s}", flush=True)
    if not got:
        raise SystemExit("no agents with ckpt_best.pt were fetched — nothing to record.")

    if args.download_only or not args.record_config:
        print(f"\n[wandb-rec] tree ready at {out}. To render:\n"
              f"  PYTHON={sys.executable} bash launchers/record_group.bash "
              f"{args.record_config or '<record_config>'} {out} --checkpoint_step {args.checkpoint_step}",
              flush=True)
        return

    # Render everything via the existing group launcher (globs <out>/<method>/<agent>/checkpoints/*.pt).
    extra = [a for a in args.record_args if a != "--"]
    cmd = ["bash", os.path.join(_PROJECT_ROOT, "launchers", "record_group.bash"),
           os.path.abspath(args.record_config), out, "--checkpoint_step", args.checkpoint_step, *extra]
    env = {**os.environ, "PYTHON": sys.executable}  # record.py runs in THIS interpreter (the general env)
    print(f"\n[wandb-rec] recording: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=_PROJECT_ROOT, env=env).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
