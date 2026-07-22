#!/usr/bin/env python
"""Split a best/median/worst GRID recording into one video per tile.

The recorder (``learning/recording_eval.py`` -> ``surface_viz.montage``) lays the
selected trajectories out as a ``rows x cols`` grid: each ``H x W`` tile placed at
``y = gap + r*(H+gap)``, ``x = gap + c*(W+gap)`` on a ``rows*H+(rows+1)*gap`` by
``cols*W+(cols+1)*gap`` canvas. For the annotated-ranked video that is a 3x4 grid
(row 0 = best, row 1 = avg/median, row 2 = worst), gap 6 px.

This tool re-reads that grid video and writes each tile out as its own file to::

    {video_dir}/{video_stem}_splits/{row_label}_{col}.{gif|mp4}

Tile size is derived from the actual video dimensions + (rows, cols, gap), so it
stays exact regardless of the per-tile resolution. Everything is a CLI flag.

Examples
--------
    python utilities/split_grid_video.py \
        runs/surface_baselines/high-ent_high-gain/7_GAS_dyn/7_GAS_dyn_agent0.mp4 \
        --record_config configs/exp_cfgs/glued_surface/_record_video.yaml

    python utilities/split_grid_video.py grid.mp4 --format gif --rows 3 --cols 4
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio


def _load_recorder_block(path: str) -> dict:
    """Return the ``recorder:`` mapping from a record-overlay YAML (searched recursively),
    or {} if not found / unreadable. Used only for defaults (fps, grid rows/cols)."""
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[split] warning: could not read record_config {path!r}: {e}", flush=True)
        return {}

    def _find(node):
        if isinstance(node, dict):
            if "recorder" in node and isinstance(node["recorder"], dict):
                return node["recorder"]
            for v in node.values():
                hit = _find(v)
                if hit is not None:
                    return hit
        return None

    return _find(data) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split a best/median/worst grid video into one video per tile.")
    p.add_argument("video", help="Input grid video (the annotated best/median/worst mp4).")
    p.add_argument("--record_config", default=None,
                   help="Record overlay YAML that made the video; used for default fps and (for "
                        "non-ranked grids) grid_rows/grid_cols. Optional.")
    p.add_argument("--rows", type=int, default=None,
                   help="Grid rows. Default: 3 (best/avg/worst ranked video), or the config's "
                        "grid_rows for a non-ranked grid.")
    p.add_argument("--cols", type=int, default=None,
                   help="Grid cols. Default: 4 (ranked video), or the config's grid_cols.")
    p.add_argument("--gap", type=int, default=6,
                   help="Pixel gap around/between tiles used by the montage (default 6).")
    p.add_argument("--row_labels", default="best,avg,worst",
                   help="Comma-separated row labels top->bottom (default 'best,avg,worst'); "
                        "must have exactly --rows entries.")
    p.add_argument("--format", choices=("mp4", "gif"), default="mp4", help="Output format (default mp4).")
    p.add_argument("--fps", type=int, default=None,
                   help="Output fps. Default: record_config recorder.fps, else the source video's fps, else 30.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the tiles. Default {video_dir}/{video_stem}_splits.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        raise SystemExit(f"[split] video not found: {video}")

    rec = _load_recorder_block(args.record_config) if args.record_config else {}
    ranked = bool(rec.get("annotated_ranked", False))
    # rows/cols: CLI > (config grid for non-ranked) > ranked default 3x4.
    rows = args.rows if args.rows is not None else (
        3 if (ranked or not rec) else int(rec.get("grid_rows", 3)))
    cols = args.cols if args.cols is not None else (
        4 if (ranked or not rec) else int(rec.get("grid_cols", 4)))
    gap = args.gap
    labels = [s.strip() for s in args.row_labels.split(",") if s.strip()]
    if len(labels) != rows:
        raise SystemExit(f"[split] --row_labels has {len(labels)} entries but --rows is {rows}: {labels}")

    reader = imageio.get_reader(video)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 30) or 30)
    fps = args.fps or int(rec.get("fps", 0)) or int(round(src_fps)) or 30

    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = args.output_dir or os.path.join(os.path.dirname(video), f"{stem}_splits")
    os.makedirs(out_dir, exist_ok=True)
    ext = args.format

    writers: dict[tuple[int, int], object] = {}
    n_frames = 0
    H = W = None
    try:
        for i, frame in enumerate(reader):
            if i == 0:
                Hgrid, Wgrid = frame.shape[:2]
                H = (Hgrid - (rows + 1) * gap) // rows
                W = (Wgrid - (cols + 1) * gap) // cols
                if H <= 0 or W <= 0:
                    raise SystemExit(
                        f"[split] bad geometry: video {Wgrid}x{Hgrid}, rows={rows} cols={cols} gap={gap} "
                        f"-> tile {W}x{H}. Check --rows/--cols/--gap.")
                print(f"[split] grid {Wgrid}x{Hgrid} -> {rows}x{cols} tiles of {W}x{H} (gap {gap}), "
                      f"fps={fps}, format={ext}", flush=True)
                for r in range(rows):
                    for c in range(cols):
                        path = os.path.join(out_dir, f"{labels[r]}_{c}.{ext}")
                        if ext == "mp4":
                            writers[(r, c)] = imageio.get_writer(
                                path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
                        else:  # gif
                            writers[(r, c)] = imageio.get_writer(path, mode="I", fps=fps, loop=0)
            for (r, c), w in writers.items():
                y = gap + r * (H + gap)
                x = gap + c * (W + gap)
                w.append_data(frame[y:y + H, x:x + W])
            n_frames += 1
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        reader.close()

    print(f"[split] wrote {len(writers)} tiles x {n_frames} frames -> {out_dir}", flush=True)
    for r in range(rows):
        for c in range(cols):
            print(f"          {labels[r]}_{c}.{ext}")


if __name__ == "__main__":
    main()
