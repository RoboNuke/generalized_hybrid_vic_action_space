#!/usr/bin/env python
"""Per-agent video recorder — record ONE trained agent's policy.

Thin, friendly front-end over ``learning/runner.py``'s record mode. Give it a
single agent folder (the ``0/``, ``1/``, ... dir that holds ``checkpoints/`` and
the snapshotted ``config.yaml``) plus a record overlay config; it loads that one
agent's weights (its own policy + twin critics), rolls out, collects complete
trajectories, and writes an agent-specific best/median/worst grid GIF next to the
checkpoint (``<agent_dir>/videos/recording.gif``).

Examples
--------
    python learning/record.py \
        --agent_dir runs/log_dir/1_fixed/0 \
        --record_config configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml \
        --headless

    python learning/record.py \
        --agent_dir runs/log_dir/1_fixed/0 \
        --record_config configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml \
        --num_trajectories 48 --headless

The config is read from ``<agent_dir>/config.yaml`` automatically (override with
``--config``). Everything in the record overlay is deep-merged over it.
"""

from __future__ import annotations

import argparse
import os
import sys


def _project_root_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record a single trained agent to a grid GIF.")
    p.add_argument(
        "--agent_dir", type=str, required=True,
        help="Single agent folder, e.g. runs/log_dir/1_fixed/0 "
             "(contains checkpoints/ckpt_*.pt and config.yaml).",
    )
    p.add_argument(
        "--record_config", type=str, action="append", default=None,
        help="Record overlay YAML (the recorder/eval settings). Repeatable; later wins. "
             "e.g. configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Base config. Defaults to <agent_dir>/config.yaml.",
    )
    p.add_argument(
        "--num_trajectories", type=int, default=None,
        help="Collect at least this many trajectories before composing the grid "
             "(overrides recorder.num_trajectories).",
    )
    p.add_argument("--checkpoint_step", type=lambda v: v if v == "best" else int(v), default=None,
                   help="Specific ckpt step to load, or 'best' for the agent's ckpt_best.pt "
                        "(highest-success-rate checkpoint); default = latest.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write the GIF. Default <agent_dir>/<recorder.output_subdir>.")
    p.add_argument("--device", type=str, default=None, help="Torch/sim device, e.g. cuda:0.")
    p.add_argument("--headless", action="store_true", help="Run Isaac headless (still records).")
    p.add_argument("--enable_cameras", action="store_true",
                   help="Usually auto-forced by recorder.enabled; pass to be explicit.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    _project_root_on_path()

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
    runner.main(argv)


if __name__ == "__main__":
    main()
