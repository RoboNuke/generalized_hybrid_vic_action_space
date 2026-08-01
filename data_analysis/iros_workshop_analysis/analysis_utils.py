"""Shared analysis utilities for the IROS-workshop pitch-sweep notebook.

Self-contained single source of truth for BOTH reading the pitch-sweep data and
drawing its figures -- this module deliberately imports nothing from the parent
``data_analysis`` package (``data_loader`` / ``plot_tools``) so the IROS-workshop
analysis can be copied/shipped as a standalone folder.

Experiment layout (n methods x k grasp angles)
----------------------------------------------
The sweep varies two factors:

* ``method``      -- controller family: ``fixed_geo``, ``GAS_geo``, ``GAS``, ``VICES``.
* ``grasp_angle`` -- in-gripper grasp tilt (deg): ``0``, ``15``, ``30``, ``45``.

Each (method, angle) cell has several seeds. On wandb they are logged as runs
named ``{method}_{grasp_angle}_agent{agent_idx}`` grouped by ``{method}_{grasp_angle}``.
:func:`download_wandb_data` mirrors them into ``runs/{project}_{tag}/{group}/{agent_idx}/``
as TensorBoard event files, and :func:`load_data` reads that tree into the data
model::

    DATA = { "{method}_{angle}": [ {tag: (steps, values)}, ... one dict per seed ] }

The plotting layer never sees raw folders: :func:`build_collections` parses those
group keys into a :class:`Collections` view that pools seeds three ways -- by
method (over all angles), by angle (over all methods), and per (method, angle)
cell -- which are exactly the three aggregations the notebook figures need.

Plotting follows the "single-plot function + aggregator" pattern: :func:`plot_metric_lines`
draws ONE axes (mean line + shaded 95% CI band per series) and every ``figure_*``
helper assembles a figure/subplot grid by calling it. Colors come from the fixed
:data:`METHOD_COLORS` / :data:`ANGLE_COLORS` palettes; figures always save as SVG.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# All user-tunable styling AND figure/function default values live in plot_config.py. It is
# reloaded here so that editing plot_config.py and re-running `importlib.reload(analysis_utils)`
# in a notebook picks up the change (reloading THIS module alone would keep the cached
# plot_config values). Edit plot_config.py -> re-run a plot cell -> it updates everywhere.
import importlib as _importlib
import plot_config as _plot_config
_importlib.reload(_plot_config)
from plot_config import *   # noqa: F401,F403  fonts, palettes, sizes, label/legend/geometry defaults

# Fallback for optional knobs so removing one from plot_config doesn't crash the plots. Only fills
# in a default when the name is ABSENT -- if plot_config defines it, that value is kept untouched.
globals().setdefault("LEGEND_HANDLE_LENGTH", 2.0)   # matplotlib's default legend handle length


# =========================================================================== #
# Paths
# =========================================================================== #
def find_project_root(marker: str = "runs") -> str:
    """Walk up from the cwd until a directory containing ``marker`` is found.

    The notebook lives in ``data_analysis/iros_workshop_analysis/`` but ``runs/``
    lives at the repo root, so paths are anchored to whichever ancestor holds
    ``runs/``. Falls back to the cwd if no ancestor qualifies.
    """
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(os.getcwd())
        d = parent


def runs_root(marker: str = "runs") -> str:
    """Absolute path to the ``runs/`` folder under the project root."""
    return os.path.join(find_project_root(marker), marker)


def step_ceiling_from_xlim(xlim) -> float:
    """Largest step used for run selection: the XLIM upper bound (``inf`` if unset)."""
    if xlim is not None and xlim[1] is not None:
        return float(xlim[1])
    return np.inf


# =========================================================================== #
# Loading TensorBoard event files
# =========================================================================== #
def load_run(run_dir: str) -> dict:
    """Load all scalar tags from every event file in one numbered seed dir.

    Several ``events.out.tfevents.*`` files (some empty) are merged; if a tag
    appears in more than one file the copy with more points wins.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tags: dict = {}
    for ef in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        ea = EventAccumulator(ef, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events], dtype=float)
            values = np.array([e.value for e in events], dtype=float)
            if tag not in tags or len(steps) > len(tags[tag][0]):
                tags[tag] = (steps, values)
    return tags


def load_data(folder_name: str, root: str | None = None, verbose: bool = True) -> dict:
    """Load every experiment group under ``runs/{folder_name}`` into the data model.

    ``root`` defaults to :func:`runs_root`. Returns ``{group: [ {tag: (steps, values)}, ... ]}``;
    groups (and non-numeric sub-dirs such as ``plots_*``) with no readable seed are skipped.
    """
    base = os.path.join(root or runs_root(), folder_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No such folder: {base}")

    data: dict = {}
    for group in sorted(os.listdir(base)):
        group_dir = os.path.join(base, group)
        if not os.path.isdir(group_dir):
            continue
        runs = []
        for run_name in sorted((d for d in os.listdir(group_dir) if d.isdigit()), key=int):
            run_tags = load_run(os.path.join(group_dir, run_name))
            if run_tags:
                runs.append(run_tags)
        if runs:
            data[group] = runs
            if verbose:
                print(f"{group}: {len(runs)} seed(s) loaded")
    return data


# =========================================================================== #
# wandb -> local TensorBoard cache (download once, reused on every re-run)
# =========================================================================== #
_AGENT_RE = re.compile(r"agent[_-]?(\d+)", re.IGNORECASE)


def _agent_index(run, fallback: int) -> int:
    """Seed index for a wandb run: ``config['agent_index']``, else parsed from the
    run name (``..._agent{idx}``), else the enumeration ``fallback``."""
    idx = run.config.get("agent_index")
    if idx is not None:
        return int(idx)
    m = _AGENT_RE.search(run.name or "")
    return int(m.group(1)) if m else int(fallback)


def _group_complete(group_dir: str, expected: int) -> bool:
    """True if ``group_dir`` already holds ``expected`` numbered seeds, each with an event file."""
    if not os.path.isdir(group_dir):
        return False
    run_dirs = [d for d in os.listdir(group_dir) if d.isdigit()]
    if len(run_dirs) < expected:
        return False
    return all(glob.glob(os.path.join(group_dir, d, "events.out.tfevents.*")) for d in run_dirs)


def _write_tfevents(run_dir: str, hist, step_key: str = "_step") -> int:
    """Write one wandb history DataFrame to a TensorBoard event file in ``run_dir``.

    Every numeric scalar column becomes an ``add_scalar`` series keyed on ``step_key``
    (wandb ``_step`` == training timestep). Internal (``_*`` / ``gradient*``) columns and
    non-finite points are skipped. Returns the number of scalar series written.
    """
    import math

    from torch.utils.tensorboard import SummaryWriter

    if step_key not in hist.columns:
        raise KeyError(f"wandb history missing {step_key!r}; columns={list(hist.columns)[:8]}...")

    writer = SummaryWriter(log_dir=run_dir)
    steps = hist[step_key].to_numpy()
    n_series = 0
    for col in hist.columns:
        if col == step_key or col.startswith("_") or col.startswith("gradient"):
            continue
        vals = hist[col].to_numpy()
        wrote = False
        for s, v in zip(steps, vals):
            if v is None or s is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                break  # a non-scalar column (media / string) -- skip it entirely
            if not math.isfinite(fv) or not math.isfinite(float(s)):
                continue
            writer.add_scalar(col, fv, int(s))
            wrote = True
        n_series += int(wrote)
    writer.flush()
    writer.close()
    return n_series


def _download_subset(runs_subset: list, base: str, samples: int, step_key: str,
                     force: bool, verbose: bool, kind: str) -> None:
    """Write one subset of wandb runs (all training, or all eval) into ``base`` as
    ``{group}/{agent_idx}/events.*``. Bucketed by wandb ``group`` (``{method}_{angle}``)
    and numbered by :func:`_agent_index`. Complete groups are skipped unless ``force``.
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in runs_subset:
        groups[r.group or r.name].append(r)
    for group in sorted(groups):
        grp_runs = groups[group]
        group_dir = os.path.join(base, group)
        if not force and _group_complete(group_dir, len(grp_runs)):
            if verbose:
                print(f"[wandb-dl]   [{kind}] {group}: cached ({len(grp_runs)} seeds), skip")
            continue
        if force and verbose:
            print(f"[wandb-dl]   [{kind}] {group}: force refresh -- overwriting local cache")
        for i, r in enumerate(sorted(grp_runs, key=lambda rr: _agent_index(rr, 0))):
            idx = _agent_index(r, i)
            run_dir = os.path.join(group_dir, str(idx))
            os.makedirs(run_dir, exist_ok=True)
            for old in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
                os.remove(old)  # clear any partial prior download for a clean rewrite
            hist = r.history(samples=samples, pandas=True)
            n = _write_tfevents(run_dir, hist, step_key)
            if verbose:
                print(f"[wandb-dl]   [{kind}] {group}/{idx}: {n} series ({len(hist)} steps) from {r.name}")


def download_wandb_data(project: str, tag: str, entity: str | None = "hur",
                        root: str | None = None, samples: int = 100_000,
                        step_key: str = "_step", force: bool = False,
                        verbose: bool = True, eval_tag: str = "eval",
                        include_eval: bool = False) -> str:
    """Mirror the TRAINING runs in ``{entity}/{project}`` carrying ``tag`` into
    ``runs/{project}_{tag}/{group}/{agent_idx}/`` as TensorBoard event files -- the exact
    layout :func:`load_data` reads -- and return the folder name ``"{project}_{tag}"``.

    This analysis is purely about training performance over env steps, so **eval runs are
    excluded by default**. That exclusion is also load-bearing for correctness: a training run
    and its eval run share the SAME wandb ``group`` (``{method}_{angle}``) AND the same per-agent
    index, and each eval run logs only a single summary row -- so if both landed in one tree the
    single-point eval run would overwrite the training curve for that seed (the curve then
    collapses to one step and every plot looks blank). Eval runs are identified by ``eval_tag``
    (``"eval"``) and skipped. Pass ``include_eval=True`` to ALSO mirror them, kept safely apart
    in ``runs/{project}_{tag}_{eval_tag}/``.

    **Download-once cache:** a group whose local dir already holds a complete set of seeds is
    skipped, so re-running reuses the data. Pass ``force=True`` to re-fetch and overwrite.

    Requires ``wandb`` and ``torch`` (imported lazily so the local-only workflow needs neither).
    """
    import wandb

    root = root or runs_root()
    api = wandb.Api(timeout=60)
    path = project if "/" in project else (f"{entity}/{project}" if entity else project)
    runs = list(api.runs(path, filters={"tags": tag}))
    if not runs:
        raise RuntimeError(f"no wandb runs found in {path!r} tagged {tag!r}")

    eval_runs = [r for r in runs if eval_tag in (r.tags or [])]
    train_runs = [r for r in runs if eval_tag not in (r.tags or [])]
    train_folder = f"{project}_{tag}"
    eval_folder = f"{project}_{tag}_{eval_tag}"

    if verbose:
        note = f", {len(eval_runs)} eval -> {eval_folder}/" if include_eval else \
               f" ({len(eval_runs)} eval run(s) skipped)"
        print(f"[wandb-dl] {len(runs)} run(s) in {path} tagged '{tag}': "
              f"{len(train_runs)} train -> {train_folder}/{note}")
    _download_subset(train_runs, os.path.join(root, train_folder),
                     samples, step_key, force, verbose, "train")
    if include_eval and eval_runs:
        _download_subset(eval_runs, os.path.join(root, eval_folder),
                         samples, step_key, force, verbose, "eval")
    return train_folder


# =========================================================================== #
# Smoothing
# =========================================================================== #
def moving_average(vals: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over ``window`` points (edges shrink the window, no zero-pad dip)."""
    if window is None or window <= 1 or vals.size == 0:
        return vals
    w = int(min(window, vals.size))
    kernel = np.ones(w)
    sums = np.convolve(vals, kernel, mode="same")
    counts = np.convolve(np.ones_like(vals), kernel, mode="same")
    return sums / counts


def smooth_runs(runs: list, metric: str, window: int) -> list:
    """Copy of ``runs`` with ``metric``'s values replaced by their moving average.

    Steps are untouched; every other tag is shared by reference. Runs without
    ``metric`` pass through unchanged.
    """
    if window is None or window <= 1:
        return runs
    out = []
    for run in runs:
        if metric in run:
            steps, vals = run[metric]
            run = {**run, metric: (steps, moving_average(vals, window))}
        out.append(run)
    return out


# =========================================================================== #
# Aggregation (mean + SEM across a set of seeds) and the CI band
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Per-metric display scaling (e.g. success rate 0-1 -> 0-100 %)
# --------------------------------------------------------------------------- #
# METRIC_SCALE / METRIC_UNIT now live in plot_config.py (imported via `from plot_config import *`).


def metric_scale(tag: str) -> float:
    return METRIC_SCALE.get(tag, 1.0)


def metric_unit(tag: str) -> str:
    return METRIC_UNIT.get(tag, "")


def _series(run: dict, tag: str):
    """``(steps, values)`` for ``tag`` with the display scale applied to VALUES only (not steps)."""
    steps, vals = run[tag]
    s = metric_scale(tag)
    return steps, (vals * s if s != 1.0 else vals)


def _scaled_clip(clip, tag: str):
    """Scale a ``(min, max)`` clip / axis-limit into the metric's display units (None-safe)."""
    if clip is None:
        return None
    s = metric_scale(tag)
    return tuple(None if c is None else c * s for c in clip)


def _with_unit(label, tag: str):
    """Append the metric's unit to an axis label: ``'Success rate' -> 'Success rate (%)'``."""
    u = metric_unit(tag)
    return f"{label} ({u})" if (label and u) else label


def aggregate_runs(runs: list, metric: str):
    """Return ``(grid, mean, sem, n)`` for ``metric`` across a set of seeds -- a **ragged**
    mean over the UNION of their step grids.

    Each seed is interpolated onto the union grid but only WITHIN its own step range (no
    extrapolation past where it stopped); outside that range it contributes nothing. At every
    grid step the ``mean``/``sem`` are taken over just the seeds that actually reach it, so
    pooling curves of unequal length does NOT collapse to the shortest -- the mean simply
    averages fewer seeds (and the CI widens) past the point where the short runs end. ``sem``
    is ``std / sqrt(count)`` there (0 where a step has a single seed); ``n`` is the max seed
    count over the grid. Returns ``None`` if no seed carries ``metric``.

    When every seed spans the same steps (e.g. within one (method, angle) cell) this reduces
    to the ordinary mean +/- SEM. Pooling several cells is just concatenating their seed lists
    before calling this -- how the by-method / by-angle aggregations are formed, where run
    lengths genuinely differ (some sweep cells stop earlier than others).
    """
    series = [_series(r, metric) for r in runs if metric in r]
    if not series:
        return None

    grid = np.unique(np.concatenate([s[0] for s in series]))
    if grid.size == 0:
        return None

    stacked = np.full((len(series), grid.size), np.nan)
    for i, (steps, vals) in enumerate(series):
        interp = np.interp(grid, steps, vals)  # np.interp clamps outside [min,max]...
        inside = (grid >= steps.min()) & (grid <= steps.max())
        stacked[i, inside] = interp[inside]     # ...so mask to each seed's own range

    counts = np.sum(~np.isnan(stacked), axis=0)
    keep = counts > 0
    grid, stacked, counts = grid[keep], stacked[:, keep], counts[keep]

    mean = np.nanmean(stacked, axis=0)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(stacked, axis=0, ddof=1)
    sem = np.where(counts > 1, std / np.sqrt(counts), 0.0)
    return grid, mean, sem, int(counts.max())


def _clip(arr: np.ndarray, bounds):
    """Clip ``arr`` to ``bounds`` = ``(lo, hi)`` (either end ``None`` = unbounded)."""
    if bounds is None:
        return arr
    lo, hi = bounds
    if lo is not None:
        arr = np.maximum(arr, lo)
    if hi is not None:
        arr = np.minimum(arr, hi)
    return arr


def ci_band(runs: list, metric: str, ci_z: float = 1.96, ci_clip=None):
    """``(grid, mean, lower, upper, n)`` -- mean and the ``+/- ci_z * SEM`` band for ``metric``.

    ``ci_clip`` = ``(min, max)`` clips ONLY the band edges (the shaded region), so e.g. a
    success-rate band never renders below 0 or above 1 while the mean line is untouched.
    Either end may be ``None`` to leave that side unbounded. Returns ``None`` if ``metric``
    is absent from every seed.
    """
    ci_clip = _scaled_clip(ci_clip, metric)          # clip is given in raw units; scale to display
    agg = aggregate_runs(runs, metric)
    if agg is None:
        return None
    grid, mean, sem, n = agg
    half = ci_z * sem
    lower = _clip(mean - half, ci_clip)
    upper = _clip(mean + half, ci_clip)
    return grid, mean, lower, upper, n


# =========================================================================== #
# Per-agent reduction to a single representative value (peak-perf analysis)
# =========================================================================== #
# Two ways to boil ONE agent's run down to a single number per metric -- the two
# sides of the peak-perf notebook's USE_EVAL switch:
#   * mode="best" -- from the agent's TRAINING run, find the step where its own success
#     rate peaks and read every metric AT that step (its best checkpoint by success).
#   * mode="eval" -- the agent's eval run holds a single summary row (a post-training
#     evaluation); take that value directly. Read from the EVAL data tree.
# These feed :func:`summarize_agents` (mean +/- CI across an agent set), which the bar
# figures reduce per (method, angle) cell or per pooled method / angle.
def best_index(run: dict, selection_metric: str, step_ceiling: float = np.inf):
    """Index where ``selection_metric`` peaks within the step window (None if absent/empty)."""
    if selection_metric not in run:
        return None
    steps, vals = run[selection_metric]
    keep = np.flatnonzero(steps <= step_ceiling)
    if keep.size == 0:
        return None
    return int(keep[np.argmax(vals[keep])])


def value_at_step(run: dict, metric: str, target_step: float, step_ceiling: float = np.inf):
    """Value of ``metric`` at the in-window step nearest ``target_step`` (None if absent)."""
    if metric not in run:
        return None
    steps, vals = _series(run, metric)
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    steps, vals = steps[keep], vals[keep]
    return float(vals[int(np.argmin(np.abs(steps - target_step)))])


def value_last(run: dict, metric: str, step_ceiling: float = np.inf):
    """Value of ``metric`` at the last in-window step (None if absent). For an eval run (a
    single logged row) this is simply that row's value.
    """
    if metric not in run:
        return None
    steps, vals = _series(run, metric)
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    return float(vals[keep][-1])


def value_mean_tail(run: dict, metric: str, tail_n: int = 5, step_ceiling: float = np.inf):
    """Mean of ``metric`` over the last ``tail_n`` in-window steps (None if absent).

    Denoises a converged quantity (e.g. end-of-training stiffness) versus the single last point.
    """
    if metric not in run:
        return None
    steps, vals = _series(run, metric)
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    v = vals[keep]
    return float(v[-min(int(tail_n), v.size):].mean())


def per_agent_value(run: dict, metric: str, mode: str, selection_metric: str = None,
                    step_ceiling: float = np.inf, tail_n: int = 5):
    """One representative scalar of ``metric`` for a single agent's ``run``.

    ``mode``:
      * ``"best"``      -- ``metric`` at the step where ``selection_metric`` peaks (best checkpoint).
      * ``"eval"`` / ``"last"`` -- the run's last in-window value (an eval run's single row, or the
        converged end-of-training value on a training run).
      * ``"mean_tail"`` -- mean over the last ``tail_n`` in-window steps (denoised convergence).

    Returns ``None`` if the needed tag is missing.
    """
    if mode in ("eval", "last"):
        return value_last(run, metric, step_ceiling)
    if mode == "mean_tail":
        return value_mean_tail(run, metric, tail_n, step_ceiling)
    if mode == "best":
        bi = best_index(run, selection_metric, step_ceiling)
        if bi is None:
            return None
        best_step = run[selection_metric][0][bi]
        return value_at_step(run, metric, best_step, step_ceiling)
    raise ValueError(f"unknown mode: {mode!r} (expected 'best'/'eval'/'last'/'mean_tail')")


def summarize_agents(runs: list, metric: str, mode: str, selection_metric: str = None,
                     ci_z: float = 1.96, step_ceiling: float = np.inf):
    """``(mean, ci)`` of ``metric`` across an agent set, each agent reduced by
    :func:`per_agent_value`. ``ci`` is ``ci_z * SEM`` (0 for a single agent). ``None`` if no
    agent yields a value.
    """
    vals = [per_agent_value(r, metric, mode, selection_metric, step_ceiling) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    ci = ci_z * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(ci)


def tag_scalar(runs: list, tag: str, reduce: str = "last", step_ceiling: float = np.inf):
    """Single scalar for ``tag`` over a set of runs: the mean across runs of each run's
    per-run reduction (:func:`per_agent_value` with ``mode=reduce``). ``None`` if no run has it.

    Used by the stiffness-ellipse cartoon, which needs one representative value per pooled key
    (method or angle) for each stiffness tag.
    """
    vals = [per_agent_value(r, tag, reduce, None, step_ceiling) for r in runs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# =========================================================================== #
# Method / grasp-angle grouping (parse the "{method}_{angle}" group keys)
# =========================================================================== #
# METHOD_ORDER (canonical method display order) now lives in plot_config.py.


def parse_group(group: str):
    """Split a ``"{method}_{angle}"`` group key into ``(method, angle_str)``.

    The angle is the trailing numeric token, so a method that itself contains an
    underscore (``GAS_geo``) parses correctly: ``"GAS_geo_15" -> ("GAS_geo", "15")``.
    Returns ``(group, None)`` if there is no trailing numeric token.
    """
    method, sep, angle = group.rpartition("_")
    if sep and angle.isdigit():
        return method, angle
    return group, None


def _ordered_methods(methods) -> list:
    present = list(methods)
    ranked = [m for m in METHOD_ORDER if m in present]
    ranked += [m for m in present if m not in METHOD_ORDER]
    return ranked


def _ordered_angles(angles) -> list:
    return sorted(angles, key=lambda a: float(a))


@dataclass
class Collections:
    """Three pooled views of the sweep, built from the raw ``DATA`` dict by
    :func:`build_collections` -- one per aggregation the figures need.

    * ``by_method[method]``            -- every seed of a method, pooled over ALL grasp angles.
    * ``by_angle[angle]``              -- every seed at a grasp angle, pooled over ALL methods.
    * ``by_cell[(method, angle)]``     -- the seeds of a single (method, angle) cell.

    ``methods`` / ``angles`` are the display-ordered keys actually present.
    """
    by_method: dict
    by_angle: dict
    by_cell: dict
    methods: list
    angles: list


def build_collections(data: dict, verbose: bool = True) -> Collections:
    """Parse ``data``'s ``"{method}_{angle}"`` groups into a :class:`Collections` view.

    Groups whose key does not parse to ``(method, numeric angle)`` are skipped with a
    warning (so stray folders don't silently pollute a pooled series).
    """
    by_method: dict = {}
    by_angle: dict = {}
    by_cell: dict = {}
    methods: set = set()
    angles: set = set()
    for group, runs in data.items():
        method, angle = parse_group(group)
        if angle is None:
            if verbose:
                print(f"[collections] skip unparseable group: {group!r}")
            continue
        methods.add(method)
        angles.add(angle)
        by_method.setdefault(method, []).extend(runs)
        by_angle.setdefault(angle, []).extend(runs)
        by_cell[(method, angle)] = runs

    methods_o = _ordered_methods(methods)
    angles_o = _ordered_angles(angles)
    if verbose:
        print(f"[collections] {len(methods_o)} method(s) x {len(angles_o)} angle(s): "
              f"methods={methods_o}, angles={angles_o}")
    return Collections(by_method, by_angle, by_cell, methods_o, angles_o)


# =========================================================================== #
# Styling: fonts, palettes, display names, sizes, bar-label & legend spacing, bar geometry
# -- ALL now in plot_config.py (imported via `from plot_config import *`). Edit that file and
# re-run a plot cell to restyle everything. Only the internal fallback cycles stay here.
# =========================================================================== #
_FALLBACK_CYCLE = plt.get_cmap("tab10").colors
_MARKER_CYCLE = ["o", "s", "^", "h", "D", "v", "P", "X"]


def method_color(method: str, idx: int = 0) -> str:
    return METHOD_COLORS.get(method, _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])


def method_name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def angle_color(angle: str, idx: int = 0) -> str:
    return ANGLE_COLORS.get(str(angle), _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])


def angle_name(angle: str) -> str:
    return f"{angle}°"  # e.g. "15°"


def angle_marker(angle: str, idx: int = 0) -> str:
    return ANGLE_MARKERS.get(str(angle), _MARKER_CYCLE[idx % len(_MARKER_CYCLE)])


@dataclass
class Style:
    """Context threaded through every figure helper: the CI z, shared x-axis, output
    folder, and the color/name lookups. Built once in the notebook's params cell.
    """
    ci_z: float = 1.96
    xlabel: str = "Env Steps"
    xlim: tuple | None = None
    plots_dir: str | None = None
    # Palette overrides: leave None (the default) to read the LIVE plot_config palettes at draw
    # time, so editing METHOD_COLORS/etc. + reloading au updates every figure WITHOUT rebuilding
    # this Style. Pass a dict only to override the palette for this one Style.
    method_colors: dict | None = None
    method_names: dict | None = None
    angle_colors: dict | None = None
    angle_markers: dict | None = None

    def mcolor(self, method: str, idx: int = 0) -> str:
        colors = self.method_colors if self.method_colors is not None else METHOD_COLORS
        return colors.get(method, _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])

    def mname(self, method: str) -> str:
        names = self.method_names if self.method_names is not None else METHOD_NAMES
        return names.get(method, method)

    def acolor(self, angle: str, idx: int = 0) -> str:
        colors = self.angle_colors if self.angle_colors is not None else ANGLE_COLORS
        return colors.get(str(angle), _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])

    def aname(self, angle: str) -> str:
        return angle_name(angle)

    def amarker(self, angle: str, idx: int = 0) -> str:
        markers = self.angle_markers if self.angle_markers is not None else ANGLE_MARKERS
        return markers.get(str(angle), _MARKER_CYCLE[idx % len(_MARKER_CYCLE)])

    @property
    def step_ceiling(self) -> float:
        return step_ceiling_from_xlim(self.xlim)

    def save(self, fig, name: str) -> str:
        """Save ``fig`` into ``plots_dir`` as ``<name>.svg`` (spaces/slashes -> ``_``)."""
        if self.plots_dir is None:
            raise ValueError("Style.plots_dir is not set")
        os.makedirs(self.plots_dir, exist_ok=True)
        fname = re.sub(r"[ /]+", "_", name.strip())
        path = os.path.join(self.plots_dir, f"{fname}.svg")
        fig.savefig(path, format="svg", bbox_inches="tight")
        print(f"saved {path}")
        return path


def save_figure(fig, name: str, style: Style) -> str:
    """Module-level alias for :meth:`Style.save` (notebook brevity)."""
    return style.save(fig, name)


def _grid_y(ax):
    """Apply the horizontal (y-axis) gridlines per the SHOW_Y_GRID config. Kwargs are passed ONLY
    when enabling -- matplotlib keeps the grid ON if you pass style kwargs alongside visible=False.
    """
    if SHOW_Y_GRID:
        ax.grid(True, axis="y", alpha=GRID_ALPHA)
    else:
        ax.grid(False, axis="y")


# =========================================================================== #
# Single-plot builders (draw ONE axes; the figure_* aggregators call these)
# =========================================================================== #
# Axes-fraction anchor + stacking direction for each flat corner placement.
# (loc, x-side L/R, y-side T/B). The anchor is computed from LEGEND_CORNER_INSET at draw time.
_LEGEND_CORNERS = {
    "upper left":  ("upper left",  "L", "T"),
    "upper right": ("upper right", "R", "T"),
    "lower left":  ("lower left",  "L", "B"),
    "lower right": ("lower right", "R", "B"),
    "top left":    ("upper left",  "L", "T"),
    "top right":   ("upper right", "R", "T"),
}


def _place_legend(ax, handles=None, *, pos: str = "box", slot: int = 0, loc: str = "best"):
    """Draw a legend with **no title, ever**. ``pos`` selects placement/layout:

    * ``"box"``     -- an in-axes VERTICAL box at ``loc`` (the classic legend).
    * ``"below"``   -- a single frameless horizontal row UNDER the axes.
    * a corner (``"upper left"``/``"top left"``/``"upper right"``/``"lower left"``/``"lower right"``)
      or any other matplotlib loc string -- a single frameless horizontal row (flat) at that spot.

    ``slot`` stacks a SECOND flat legend clear of the first (0 = first row, 1 = next) so the two
    Pareto legends don't overlap. ``handles`` defaults to the axes' labeled artists. Returns the
    Legend (or ``None`` if there is nothing to show).
    """
    if handles is None:
        handles = ax.get_legend_handles_labels()[0]
    if not handles:
        return None
    n = len(handles)
    if pos == "box":
        return ax.legend(handles=handles, loc=loc, fontsize=FONT_LEGEND,
                         handlelength=LEGEND_HANDLE_LENGTH)
    # Flat (horizontal) layouts share the global entry spacing + internal border pad.
    flat = dict(ncol=n, frameon=False, fontsize=FONT_LEGEND,
                columnspacing=LEGEND_COLUMN_SPACING, handletextpad=LEGEND_HANDLETEXTPAD,
                borderpad=LEGEND_BORDER_PAD, handlelength=LEGEND_HANDLE_LENGTH)
    if pos == "below":
        # Baseline auto-clears the tick labels AND the x-axis title (when set) so the legend never
        # overlaps them, at ANY figure size / font: estimate their height in inches from the font
        # sizes and convert to a fraction of THIS axes' height. LEGEND_BELOW_OFFSET then nudges it
        # further down (or up if negative); slot stacks a 2nd row clear of the first.
        fig = ax.figure
        drop_in = (FONT_TICK / 72.0) * 1.8 + 0.05              # below the x tick labels
        if ax.get_xlabel():
            drop_in += (FONT_AXIS_LABEL / 72.0) * 2.0 + 0.03   # ... and below the x-axis title
        ax_h_in = max(ax.get_position().height * fig.get_figheight(), 0.3)
        base = drop_in / ax_h_in
        y = -(base + LEGEND_BELOW_OFFSET) - 0.10 * slot
        return ax.legend(handles=handles, loc="upper center",
                         bbox_to_anchor=(0.5, y), **flat)
    corner = _LEGEND_CORNERS.get(pos)
    if corner is None:  # any other matplotlib loc string: flat, frameless, no stacking anchor
        return ax.legend(handles=handles, loc=pos, **flat)
    cloc, xside, yside = corner
    inset = LEGEND_CORNER_INSET
    ax_x = inset if xside == "L" else 1.0 - inset      # flush to the corner when inset = 0
    ax_y = (1.0 - inset) if yside == "T" else inset
    ydir = -1 if yside == "T" else +1                  # stack a 2nd legend away from that edge
    return ax.legend(handles=handles, loc=cloc,
                     bbox_to_anchor=(ax_x, ax_y + ydir * 0.09 * slot), **flat)


def plot_metric_lines(ax, runs_by_key: dict, metric: str, key_order: list, *,
                      color_fn, name_fn, ci_z: float = 1.96, ci_clip=None,
                      smooth_window: int = 1, xlabel: str | None = None,
                      ylabel: str | None = None, title: str | None = None,
                      xlim=None, ylim=None, legend: bool = True,
                      legend_loc: str = "best", show_n: bool = False):
    """Draw a mean line + shaded 95% CI band per series onto ``ax`` (the ONE-axes primitive).

    ``runs_by_key`` maps a series key (a method or an angle) to its pooled seed list;
    ``key_order`` fixes draw/legend order. ``color_fn(key, idx)`` and ``name_fn(key)``
    supply the fixed palette color and legend label -- pass the method or angle variants
    from :class:`Style`. The band is ``mean +/- ci_z * SEM``, its edges clipped to
    ``ci_clip = (min, max)`` (the mean line is never clipped). ``smooth_window > 1``
    centered-moving-averages each series first. Empty / metric-less series are skipped.
    Returns ``ax``.
    """
    for idx, key in enumerate(key_order):
        runs = runs_by_key.get(key) or []
        if smooth_window and smooth_window > 1:
            runs = smooth_runs(runs, metric, smooth_window)
        band = ci_band(runs, metric, ci_z=ci_z, ci_clip=ci_clip)
        if band is None:
            continue
        grid, mean, lower, upper, n = band
        color = color_fn(key, idx)
        label = name_fn(key) + (f" (n={n})" if show_n else "")
        gid = re.sub(r"\s+", "_", name_fn(key))
        line, = ax.plot(grid, mean, color=color, linewidth=LINE_WIDTH, label=label)
        line.set_gid(gid)
        band_artist = ax.fill_between(grid, lower, upper, color=color,
                                      alpha=BAND_ALPHA, linewidth=0)
        band_artist.set_gid(f"{gid}_band")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    _grid_y(ax)
    if legend:
        ax.legend(loc=legend_loc, fontsize=FONT_LEGEND, handlelength=LEGEND_HANDLE_LENGTH)
    return ax


def _label_bars(ax, bars, means, errors_lower, errors_upper, y_lim, *, show_labels: bool = True,
                label_fontsize: int = None, label_decimal: int = 1,
                bold_labels: bool = None, show_label_ci: bool = None):
    """Print each bar's mean near its LOWER error cap: inside the bar just below the lower whisker
    when the bar is tall enough, else above the upper whisker. Labels are rotated 90 deg.

    ``y_lim = (min, max)`` sets the y-range that the placement fractions (``BAR_LABEL_GAP_FRAC``,
    ``BAR_LABEL_MIN_INSIDE_FRAC``) are measured against, so the label sits the same visual distance
    from the cap at any scale. Bars with a non-positive / non-finite mean are skipped.
    """
    if not show_labels:
        return
    label_fontsize = FONT_BAR_LABEL if label_fontsize is None else label_fontsize
    bold_labels = DEFAULT_BOLD_LABELS if bold_labels is None else bold_labels
    show_label_ci = DEFAULT_SHOW_LABEL_CI if show_label_ci is None else show_label_ci
    y0, y1 = y_lim
    y_range = (y1 - y0) or 1.0
    gap = BAR_LABEL_GAP_FRAC * y_range
    min_inside = BAR_LABEL_MIN_INSIDE_FRAC * y_range
    for bar, mean, err_lo, err_hi in zip(bars, means, errors_lower, errors_upper):
        if mean is None or not np.isfinite(mean) or mean <= 0:
            continue
        err_display = (err_lo + err_hi) / 2.0
        lower_cap = bar.get_height() - err_lo
        if (lower_cap - y0) >= min_inside:              # room to sit inside, below the lower cap
            label_y = lower_cap - gap
            va = "top"
        else:                                          # too short -> ride above the upper cap
            label_y = bar.get_height() + err_hi + gap
            va = "bottom"
        text = (f"{mean:.{label_decimal}f}±{err_display:.{label_decimal}f}" if show_label_ci
                else f"{mean:.{label_decimal}f}")
        # rotation_mode="anchor" aligns the ROTATED text on the anchor, so ha="center" actually
        # centers the sideways label on the bar; BAR_LABEL_X_OFFSET nudges it in bar-widths.
        label_x = bar.get_x() + bar.get_width() * (0.5 + BAR_LABEL_X_OFFSET)
        ax.text(label_x, label_y, text, ha="center", va=va, rotation=90, rotation_mode="anchor",
                fontsize=label_fontsize, color="black",
                fontweight="bold" if bold_labels else "normal")


def plot_metric_bars(ax, heights_by_key: dict, key_order: list, *, color_fn, name_fn,
                     ci_clip=None, xlabel: str | None = None, ylabel: str | None = None,
                     title: str | None = None, ylim=None, rotate_labels: int = 0,
                     show_labels: bool = True, label_decimal: int = 1,
                     bold_labels: bool = None, show_label_ci: bool = None):
    """Draw a bar per series with a 95% CI error bar onto ``ax`` (bar analogue of
    :func:`plot_metric_lines`).

    ``heights_by_key`` maps each series key to ``(value, ci)`` (a mean and its half-width,
    e.g. ``ci_z * SEM`` from :func:`aggregate_runs`). ``ci_clip = (min, max)`` clips the
    error-bar WHISKER extent (``value +/- ci``) to those limits, matching the line-plot band's
    clip, while the bar height is left as-is. Returns ``ax``. Used by the peak-perf marginal
    bar figures (:func:`figure_bars_by_method` / :func:`figure_bars_by_angle`).
    """
    xs = list(range(len(key_order)))
    vals, lower_err, upper_err, colors, labels = [], [], [], [], []
    for idx, key in enumerate(key_order):
        if key not in heights_by_key:
            vals.append(np.nan); lower_err.append(0.0); upper_err.append(0.0)
        else:
            v, ci = heights_by_key[key]
            lo = _clip(np.array([v - ci]), ci_clip)[0]
            hi = _clip(np.array([v + ci]), ci_clip)[0]
            vals.append(v)
            lower_err.append(max(v - lo, 0.0))
            upper_err.append(max(hi - v, 0.0))
        colors.append(color_fn(key, idx))
        labels.append(name_fn(key))
    bars = ax.bar(xs, vals, yerr=[lower_err, upper_err], capsize=ERROR_CAPSIZE,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 1.0})
    for b, key in zip(bars, key_order):
        b.set_gid(re.sub(r"\s+", "_", name_fn(key)))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rotate_labels,
                       ha="right" if rotate_labels else "center", fontsize=FONT_TICK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    _grid_y(ax)
    ax.margins(x=0.02)
    _label_bars(ax, bars, vals, lower_err, upper_err, ylim if ylim is not None else ax.get_ylim(),
                show_labels=show_labels, label_decimal=label_decimal, bold_labels=bold_labels,
                show_label_ci=show_label_ci)
    return ax


def plot_grouped_bars(ax, x_keys: list, group_keys: list, stat_fn, *, color_fn, group_name_fn,
                      x_name_fn, ci_clip=None, xlabel: str | None = None,
                      ylabel: str | None = None, title: str | None = None,
                      ylim=None, legend: bool = True, legend_pos: str = "box",
                      group_width: float = BAR_GROUP_WIDTH, bar_width: float | None = BAR_WIDTH,
                      bar_gap: float = BAR_GAP, capsize: float = BAR_CAPSIZE,
                      edge_color=BAR_EDGE_COLOR, edge_width: float = BAR_EDGE_WIDTH,
                      error_color=BAR_ERROR_COLOR, error_linewidth: float = BAR_ERROR_LINEWIDTH,
                      show_labels: bool = True, label_decimal: int = 1, bold_labels: bool = None,
                      show_label_ci: bool = None):
    """Clustered bars: each x-axis tick is an ``x_key`` holding one bar per ``group_key``.

    ``stat_fn(x_key, group_key)`` returns ``(value, ci)`` (mean + 95%-CI half-width) or ``None``
    to omit that bar. Inner bars are colored by ``color_fn(group_key, idx)`` and the group legend
    is labeled by ``group_name_fn``; x ticks by ``x_name_fn``. ``ci_clip = (min, max)`` clips the
    error-bar whiskers (``value +/- ci``) to those limits, leaving bar heights untouched.

    Geometry (RoboNuke-style, all settable / globally in plot_config): the ``group_width`` fraction
    of one x-unit is split into ``len(group_keys)`` equal slots; each bar is drawn at ``bar_width``
    if given, else the slot minus ``bar_gap`` (a fraction of the slot). Styling knobs: ``capsize``,
    ``edge_color`` / ``edge_width`` (bar outline), ``error_color`` (ecolor) and ``error_linewidth``.
    Returns ``ax``. The two orientations (x=method/bars=angle and x=angle/bars=method) are the
    same call with the roles swapped.
    """
    nG = max(len(group_keys), 1)
    slot = group_width / nG                             # center-to-center spacing of bars in a group
    drawn = bar_width if bar_width is not None else slot * (1.0 - bar_gap)
    xs = np.arange(len(x_keys), dtype=float)
    records = []                                        # (bars, means, err_lo, err_hi) for labeling
    for gi, gk in enumerate(group_keys):
        offset = (gi - (nG - 1) / 2.0) * slot
        vals, lower_err, upper_err = [], [], []
        for xk in x_keys:
            stat = stat_fn(xk, gk)
            if stat is None:
                vals.append(np.nan); lower_err.append(0.0); upper_err.append(0.0)
            else:
                v, ci = stat
                lo = _clip(np.array([v - ci]), ci_clip)[0]
                hi = _clip(np.array([v + ci]), ci_clip)[0]
                vals.append(v)
                lower_err.append(max(v - lo, 0.0))
                upper_err.append(max(hi - v, 0.0))
        bars = ax.bar(xs + offset, vals, width=drawn,
                      yerr=[lower_err, upper_err], capsize=capsize, ecolor=error_color,
                      color=color_fn(gk, gi), edgecolor=edge_color, linewidth=edge_width,
                      error_kw={"elinewidth": error_linewidth}, label=group_name_fn(gk))
        for b in bars:
            b.set_gid(re.sub(r"\s+", "_", group_name_fn(gk)))
        records.append((bars, vals, lower_err, upper_err))
    ax.set_xticks(xs)
    ax.set_xticklabels([x_name_fn(xk) for xk in x_keys], fontsize=FONT_TICK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    _grid_y(ax)
    ax.margins(x=0.02)
    eff_ylim = ylim if ylim is not None else ax.get_ylim()
    for rec_bars, rec_vals, rec_lo, rec_hi in records:
        _label_bars(ax, rec_bars, rec_vals, rec_lo, rec_hi, eff_ylim, show_labels=show_labels,
                    label_decimal=label_decimal, bold_labels=bold_labels, show_label_ci=show_label_ci)
    if legend:
        _place_legend(ax, pos=legend_pos)
    return ax


# =========================================================================== #
# Figure aggregators (assemble a figure by calling the single-plot builders)
# =========================================================================== #
def figure_by_method(collections: Collections, metric: str, style: Style, *,
                     ylabel: str, title: str | None = None, ylim=None, ci_clip=None,
                     smooth_window: int = 1, legend_loc: str = "best",
                     figsize=None):
    """One axes: ``metric`` vs env steps, one mean+CI line per METHOD (seeds pooled over
    all grasp angles), colored by the method palette. Returns the Figure.
    """
    ylabel = _with_unit(ylabel, metric)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    plot_metric_lines(ax, collections.by_method, metric, collections.methods,
                      color_fn=style.mcolor, name_fn=style.mname, ci_z=style.ci_z,
                      ci_clip=ci_clip, smooth_window=smooth_window, xlabel=style.xlabel,
                      ylabel=ylabel, title=title, xlim=style.xlim, ylim=ylim,
                      legend_loc=legend_loc)
    fig.tight_layout()
    return fig


def figure_by_angle(collections: Collections, metric: str, style: Style, *,
                    ylabel: str, title: str | None = None, ylim=None, ci_clip=None,
                    smooth_window: int = 1, legend_loc: str = "best",
                    figsize=None):
    """One axes: ``metric`` vs env steps, one mean+CI line per GRASP ANGLE (seeds pooled
    over all methods), colored by the ordinal angle palette. Returns the Figure.
    """
    ylabel = _with_unit(ylabel, metric)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    plot_metric_lines(ax, collections.by_angle, metric, collections.angles,
                      color_fn=style.acolor, name_fn=style.aname, ci_z=style.ci_z,
                      ci_clip=ci_clip, smooth_window=smooth_window, xlabel=style.xlabel,
                      ylabel=ylabel, title=title, xlim=style.xlim, ylim=ylim,
                      legend_loc=legend_loc)
    fig.tight_layout()
    return fig


def figure_per_angle_panels(collections: Collections, metric: str, style: Style, *,
                            ylabel: str, suptitle: str | None = None, ylim=None,
                            ci_clip=None, smooth_window: int = 1, ncols: int = 2,
                            figsize_per=None, legend_loc: str = "best"):
    """Small-multiples grid: one panel per GRASP ANGLE, each overlaying the METHOD mean+CI
    lines for that single angle (seeds within each (method, angle) cell pooled).

    Every panel shares the method palette; the legend is drawn on the first panel only
    (the colors are identical across panels). Returns the Figure.
    """
    ylabel = _with_unit(ylabel, metric)
    angles = collections.angles
    n = len(angles)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    if figsize_per is None:
        figsize_per = (DEFAULT_FIG_WIDTH / ncols, DEFAULT_FIG_HEIGHT / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    for i, angle in enumerate(angles):
        ax = axes[i // ncols][i % ncols]
        runs_by_method = {m: collections.by_cell.get((m, angle), []) for m in collections.methods}
        # x/y labels only on the outer edges to keep the grid uncluttered.
        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1) or (i + ncols >= n)
        plot_metric_lines(ax, runs_by_method, metric, collections.methods,
                          color_fn=style.mcolor, name_fn=style.mname, ci_z=style.ci_z,
                          ci_clip=ci_clip, smooth_window=smooth_window,
                          xlabel=style.xlabel if is_bottom else None,
                          ylabel=ylabel if is_left else None,
                          title=f"grasp {style.aname(angle)}", xlim=style.xlim, ylim=ylim,
                          legend=(i == 0), legend_loc=legend_loc)
    for j in range(n, nrows * ncols):  # hide unused axes
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    return fig


# =========================================================================== #
# Peak-performance bar figures (one representative value per agent, mode-selected)
# =========================================================================== #
# Each agent is reduced to a single number per metric (best training checkpoint, or the
# eval value) by :func:`summarize_agents`; bars show mean +/- 95% CI over the agents in a
# (method, angle) cell -- or over a pooled method / angle for the "aggregated over all runs"
# marginals. All four helpers below share ``mode`` ("best" or "eval") and ``selection_metric``.
def _cell_stat(collections: Collections, metric: str, style: Style, mode: str,
               selection_metric: str):
    """A ``stat(method, angle) -> (mean, ci) | None`` closure over one (method, angle) cell."""
    def stat(method, angle):
        runs = collections.by_cell.get((method, angle))
        if not runs:
            return None
        return summarize_agents(runs, metric, mode, selection_metric,
                                ci_z=style.ci_z, step_ceiling=style.step_ceiling)
    return stat


def figure_bars_method_x_angle(collections: Collections, metric: str, style: Style, *,
                               mode: str, selection_metric: str, ylabel: str,
                               title: str | None = None, xlabel: str | None = None,
                               ylim=None, ci_clip=None, legend_pos: str = "box",
                               group_width: float = BAR_GROUP_WIDTH, bar_width: float | None = BAR_WIDTH,
                               bar_gap: float = BAR_GAP,
                               show_labels: bool = True, label_decimal: int = 1,
                               bold_labels: bool = None, show_label_ci: bool = None,
                               figsize=None):
    """Clustered bars: x-axis = METHOD, one bar per GRASP ANGLE within each method
    (colored by the angle palette). Returns the Figure.
    """
    ci_clip, ylim, ylabel = _scaled_clip(ci_clip, metric), _scaled_clip(ylim, metric), _with_unit(ylabel, metric)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_bars(ax, collections.methods, collections.angles,
                      lambda m, a: stat(m, a), color_fn=style.acolor,
                      group_name_fn=style.aname, x_name_fn=style.mname, ci_clip=ci_clip,
                      xlabel=xlabel, ylabel=ylabel, title=title, ylim=ylim,
                      legend_pos=legend_pos, group_width=group_width, bar_width=bar_width,
                      bar_gap=bar_gap, show_labels=show_labels, label_decimal=label_decimal,
                      bold_labels=bold_labels, show_label_ci=show_label_ci)
    fig.tight_layout()
    return fig


def figure_bars_angle_x_method(collections: Collections, metric: str, style: Style, *,
                               mode: str, selection_metric: str, ylabel: str,
                               title: str | None = None, xlabel: str | None = None,
                               ylim=None, ci_clip=None, legend_pos: str = "box",
                               group_width: float = BAR_GROUP_WIDTH, bar_width: float | None = BAR_WIDTH,
                               bar_gap: float = BAR_GAP,
                               show_labels: bool = True, label_decimal: int = 1,
                               bold_labels: bool = None, show_label_ci: bool = None,
                               figsize=None):
    """Clustered bars: x-axis = GRASP ANGLE, one bar per METHOD within each angle
    (colored by the method palette). Returns the Figure.
    """
    ci_clip, ylim, ylabel = _scaled_clip(ci_clip, metric), _scaled_clip(ylim, metric), _with_unit(ylabel, metric)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_bars(ax, collections.angles, collections.methods,
                      lambda a, m: stat(m, a), color_fn=style.mcolor,
                      group_name_fn=style.mname, x_name_fn=style.aname, ci_clip=ci_clip,
                      xlabel=xlabel, ylabel=ylabel, title=title, ylim=ylim,
                      legend_pos=legend_pos, group_width=group_width, bar_width=bar_width,
                      bar_gap=bar_gap, show_labels=show_labels, label_decimal=label_decimal,
                      bold_labels=bold_labels, show_label_ci=show_label_ci)
    fig.tight_layout()
    return fig


def figure_bars_by_method(collections: Collections, metric: str, style: Style, *,
                          mode: str, selection_metric: str, ylabel: str,
                          title: str | None = None, xlabel: str | None = None,
                          ylim=None, ci_clip=None,
                          show_labels: bool = True, label_decimal: int = 1,
                          bold_labels: bool = None, show_label_ci: bool = None,
                          figsize=None):
    """Single bar per METHOD, aggregated over ALL grasp angles (every seed of the method,
    pooled). Colored by the method palette. Returns the Figure.
    """
    ci_clip, ylim, ylabel = _scaled_clip(ci_clip, metric), _scaled_clip(ylim, metric), _with_unit(ylabel, metric)
    heights = {m: summarize_agents(collections.by_method[m], metric, mode, selection_metric,
                                   ci_z=style.ci_z, step_ceiling=style.step_ceiling)
               for m in collections.methods}
    heights = {m: v for m, v in heights.items() if v is not None}
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    plot_metric_bars(ax, heights, collections.methods, color_fn=style.mcolor,
                     name_fn=style.mname, ci_clip=ci_clip, xlabel=xlabel, ylabel=ylabel,
                     title=title, ylim=ylim, show_labels=show_labels, label_decimal=label_decimal,
                      bold_labels=bold_labels, show_label_ci=show_label_ci)
    fig.tight_layout()
    return fig


def figure_bars_by_angle(collections: Collections, metric: str, style: Style, *,
                         mode: str, selection_metric: str, ylabel: str,
                         title: str | None = None, xlabel: str | None = None,
                         ylim=None, ci_clip=None,
                          show_labels: bool = True, label_decimal: int = 1,
                          bold_labels: bool = None, show_label_ci: bool = None,
                          figsize=None):
    """Single bar per GRASP ANGLE, aggregated over ALL methods (every seed at the angle,
    pooled). Colored by the ordinal angle palette. Returns the Figure.
    """
    ci_clip, ylim, ylabel = _scaled_clip(ci_clip, metric), _scaled_clip(ylim, metric), _with_unit(ylabel, metric)
    heights = {a: summarize_agents(collections.by_angle[a], metric, mode, selection_metric,
                                   ci_z=style.ci_z, step_ceiling=style.step_ceiling)
               for a in collections.angles}
    heights = {a: v for a, v in heights.items() if v is not None}
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    plot_metric_bars(ax, heights, collections.angles, color_fn=style.acolor,
                     name_fn=style.aname, ci_clip=ci_clip, xlabel=xlabel, ylabel=ylabel,
                     title=title, ylim=ylim, show_labels=show_labels, label_decimal=label_decimal,
                      bold_labels=bold_labels, show_label_ci=show_label_ci)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Horizontal grouped bars (value runs along x; categories stack down the y-axis)
# --------------------------------------------------------------------------- #
def _label_barh(ax, bars, means, errors_lower, errors_upper, x_lim, *, show_labels: bool = True,
                label_fontsize: int = None, label_decimal: int = 1, bold_labels: bool = None,
                show_label_ci: bool = None, rotation: float = 0):
    """Horizontal-bar analogue of :func:`_label_bars`: print each bar's mean near its LEFT (lower)
    error cap, inside the bar when there is room else just past the right cap. ``x_lim`` is the
    VALUE-axis range the placement fractions are measured against; ``rotation`` rotates the label.
    """
    if not show_labels:
        return
    label_fontsize = FONT_BAR_LABEL if label_fontsize is None else label_fontsize
    bold_labels = DEFAULT_BOLD_LABELS if bold_labels is None else bold_labels
    show_label_ci = DEFAULT_SHOW_LABEL_CI if show_label_ci is None else show_label_ci
    x0, x1 = x_lim
    x_range = (x1 - x0) or 1.0
    gap = BAR_LABEL_GAP_FRAC * x_range
    min_inside = BAR_LABEL_MIN_INSIDE_FRAC * x_range
    for bar, mean, err_lo, err_hi in zip(bars, means, errors_lower, errors_upper):
        if mean is None or not np.isfinite(mean) or mean <= 0:
            continue
        err_display = (err_lo + err_hi) / 2.0
        width = bar.get_width()
        lower_cap = width - err_lo                       # left error cap (value - err)
        # BAR_LABEL_X_OFFSET shifts the label PERPENDICULAR to the bar -- vertically here, in bar
        # thicknesses -- e.g. to nudge it off the error whisker (same global as the vertical bars).
        yc = bar.get_y() + bar.get_height() * (0.5 + BAR_LABEL_X_OFFSET)
        if (lower_cap - x0) >= min_inside:               # room to sit inside, left of the left cap
            label_x, ha = lower_cap - gap, "right"
        else:                                            # too short -> ride past the right cap
            label_x, ha = width + err_hi + gap, "left"
        text = (f"{mean:.{label_decimal}f}±{err_display:.{label_decimal}f}" if show_label_ci
                else f"{mean:.{label_decimal}f}")
        ax.text(label_x, yc, text, ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
                fontsize=label_fontsize, color="black",
                fontweight="bold" if bold_labels else "normal")


def plot_grouped_barh(ax, cat_keys: list, group_keys: list, stat_fn, *, color_fn, group_name_fn,
                      cat_name_fn, bar_width: float = BARH_BAR_WIDTH, group_gap: float = BARH_GROUP_GAP,
                      bar_gap: float = BARH_BAR_GAP, capsize: float = BAR_CAPSIZE,
                      edge_color=BAR_EDGE_COLOR, edge_width: float = BAR_EDGE_WIDTH,
                      error_color=BAR_ERROR_COLOR, error_linewidth: float = BAR_ERROR_LINEWIDTH,
                      ci_clip=None,
                      value_label: str | None = None, cat_label: str | None = None,
                      title: str | None = None, xlim=None, legend: bool = True,
                      legend_pos: str = "box", show_labels: bool = True, label_decimal: int = 1,
                      bold_labels: bool = None, show_label_ci: bool = None, label_rotation: float = 0,
                      reverse_categories: bool = False):
    """HORIZONTAL clustered bars: each y tick is a ``cat_key`` holding one bar per ``group_key``;
    the value (``stat_fn``) runs along x. ``bar_width`` is each bar's drawn thickness, ``bar_gap`` the
    gap BETWEEN bars within a group, and ``group_gap`` the gap between category groups (all in y data
    units). ``reverse_categories`` flips the top-to-bottom order (default: ``cat_keys[0]`` on top).
    Value labels are drawn by :func:`_label_barh` (rotated by ``label_rotation``). Returns ``ax``.
    """
    nG = max(len(group_keys), 1)
    slot = bar_width + bar_gap                           # one bar + its trailing gap
    group_extent = nG * bar_width + (nG - 1) * bar_gap   # total thickness of a group's bars
    step = group_extent + group_gap                      # center-to-center spacing between groups
    centers = np.arange(len(cat_keys)) * step
    records = []
    for gi, gk in enumerate(group_keys):
        offset = (gi - (nG - 1) / 2.0) * slot
        vals, lower_err, upper_err, ys = [], [], [], []
        for ci, ck in enumerate(cat_keys):
            ys.append(centers[ci] + offset)
            stat = stat_fn(ck, gk)
            if stat is None:
                vals.append(np.nan); lower_err.append(0.0); upper_err.append(0.0)
            else:
                v, cci = stat
                lo = _clip(np.array([v - cci]), ci_clip)[0]
                hi = _clip(np.array([v + cci]), ci_clip)[0]
                vals.append(v)
                lower_err.append(max(v - lo, 0.0))
                upper_err.append(max(hi - v, 0.0))
        bars = ax.barh(ys, vals, height=bar_width, xerr=[lower_err, upper_err],
                       capsize=capsize, ecolor=error_color, color=color_fn(gk, gi),
                       edgecolor=edge_color, linewidth=edge_width,
                       error_kw={"elinewidth": error_linewidth}, label=group_name_fn(gk))
        for b in bars:
            b.set_gid(re.sub(r"\s+", "_", group_name_fn(gk)))
        records.append((bars, vals, lower_err, upper_err))
    ax.set_yticks(centers)
    ax.set_yticklabels([cat_name_fn(ck) for ck in cat_keys], fontsize=FONT_TICK)
    if not reverse_categories:
        ax.invert_yaxis()                                # default: cat_keys[0] (e.g. 0deg) on top
    if value_label is not None:
        ax.set_xlabel(value_label, fontsize=FONT_AXIS_LABEL)
    if cat_label is not None:
        ax.set_ylabel(cat_label, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.margins(y=0.02)
    eff_xlim = xlim if xlim is not None else ax.get_xlim()
    for rb, rv, rlo, rhi in records:
        _label_barh(ax, rb, rv, rlo, rhi, eff_xlim, show_labels=show_labels,
                    label_decimal=label_decimal, bold_labels=bold_labels,
                    show_label_ci=show_label_ci, rotation=label_rotation)
    if legend:
        _place_legend(ax, pos=legend_pos)
    return ax


def figure_barh_angle_x_method(collections: Collections, metric: str, style: Style, *,
                               mode: str, selection_metric: str, ylabel: str,
                               title: str | None = None, xlabel: str | None = None, xlim=None,
                               ci_clip=None, legend_pos: str = "box", bar_width: float = BARH_BAR_WIDTH,
                               group_gap: float = BARH_GROUP_GAP, bar_gap: float = BARH_BAR_GAP,
                               label_rotation: float = 0, reverse_categories: bool = False,
                               show_labels: bool = True, label_decimal: int = 1,
                               bold_labels: bool = None, show_label_ci: bool = None,
                               figsize=None):
    """HORIZONTAL clustered bars: y-axis = GRASP ANGLE, one bar per METHOD within each angle, the
    value (``metric``) running along x. Same data/orientation-flip of :func:`figure_bars_angle_x_method`.
    ``bar_width`` sets each bar's thickness, ``bar_gap`` the gap between bars within a group, ``group_gap``
    the space between angle groups, and ``label_rotation`` the rate-label angle. ``ylabel`` labels the
    value (x) axis; ``xlabel`` the grasp-angle (y) axis. Returns the Figure.
    """
    ci_clip = _scaled_clip(ci_clip, metric)
    xlim = _scaled_clip(xlim, metric)
    ylabel = _with_unit(ylabel, metric)                  # rate/value label -> x axis (with unit)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_barh(ax, collections.angles, collections.methods, lambda a, m: stat(m, a),
                      color_fn=style.mcolor, group_name_fn=style.mname, cat_name_fn=style.aname,
                      bar_width=bar_width, group_gap=group_gap, bar_gap=bar_gap, ci_clip=ci_clip,
                      value_label=ylabel, cat_label=xlabel, title=title, xlim=xlim,
                      legend_pos=legend_pos, label_rotation=label_rotation,
                      reverse_categories=reverse_categories, show_labels=show_labels,
                      label_decimal=label_decimal, bold_labels=bold_labels, show_label_ci=show_label_ci)
    fig.tight_layout()
    return fig


# =========================================================================== #
# Pareto-style scatter (one point per run: y-metric vs x-metric)
# =========================================================================== #
# Each run becomes a single (x, y) point via the SAME per-agent reduction as the bars
# (best training checkpoint, or the eval value), so you see the run-to-run spread and the
# trade-off frontier -- e.g. success (up) vs force (right, lower is better): the upper-left
# runs dominate. Every point carries TWO encodings: COLOR = method, SHAPE = grasp angle
# (circle/square/triangle/hexagon), so a point's method and angle are both readable.
def pareto_front(points, minimize_x: bool = True, maximize_y: bool = True):
    """Non-dominated frontier of ``points`` (list of ``(x, y)``), returned as ``(xs, ys)``
    sorted by x.

    Defaults treat LOW x and HIGH y as good (force down, success/reward up), so a point is
    dominated iff another has ``x' <= x`` and ``y' >= y``. Sweeping x ascending and keeping a
    point only when its y beats the running best yields the upper-left skyline -- the best y
    achievable within each x budget. Flip ``minimize_x`` / ``maximize_y`` for other objectives.
    """
    sx = 1.0 if minimize_x else -1.0
    sy = 1.0 if maximize_y else -1.0
    # Sort by adjusted-x ascending, adjusted-y descending on ties, so at equal x the better
    # y is seen first and the worse one is correctly dropped as dominated.
    pts = sorted(((sx * x, sy * y, x, y) for x, y in points), key=lambda p: (p[0], -p[1]))
    fx, fy = [], []
    best = -np.inf
    for _, syy, x, y in pts:
        if syy > best:
            fx.append(x); fy.append(y); best = syy
    return fx, fy


def plot_pareto_scatter(ax, cells: dict, x_metric: str, y_metric: str, methods: list,
                        angles: list, style: Style, *, mode: str, selection_metric: str,
                        xlabel: str | None = None, ylabel: str | None = None,
                        title: str | None = None, xlim=None, ylim=None,
                        method_legend: bool = True, angle_legend: bool = True,
                        method_legend_loc: str = "upper right",
                        angle_legend_loc: str = "lower right", legend_pos: str = "box",
                        pareto: bool = False,
                        pareto_color: str = "0.35", s: float = 46, alpha: float = 0.82):
    """Scatter one point per run onto ``ax``: ``x = x_metric``, ``y = y_metric``, each reduced
    per run by :func:`per_agent_value` (``mode`` = ``"best"``/``"eval"``).

    ``cells`` maps ``(method, angle) -> runs`` (use ``collections.by_cell``); the ``methods`` and
    ``angles`` lists select and order which cells are drawn. Each point is COLORED by method
    (``style.mcolor``) and SHAPED by angle (``style.amarker``). Two legends are built -- a color
    legend for methods and a shape legend for angles (the latter only when more than one angle
    is drawn, so single-angle panels skip it). ``pareto=True`` overlays the non-dominated
    frontier (:func:`pareto_front`, low-force/high-y) as a dashed line. Returns ``ax``.
    """
    import matplotlib.lines as mlines

    pts = []
    for mi, method in enumerate(methods):
        for ai, angle in enumerate(angles):
            xs, ys = [], []
            for run in (cells.get((method, angle)) or []):
                x = per_agent_value(run, x_metric, mode, selection_metric, style.step_ceiling)
                y = per_agent_value(run, y_metric, mode, selection_metric, style.step_ceiling)
                if x is not None and y is not None:
                    xs.append(x); ys.append(y); pts.append((x, y))
            if xs:
                ax.scatter(xs, ys, color=style.mcolor(method, mi), marker=style.amarker(angle, ai),
                           s=s, alpha=alpha, edgecolor="white", linewidth=0.5, zorder=3)

    if pareto and len(pts) >= 2:
        fx, fy = pareto_front(pts)
        ax.plot(fx, fy, "--", color=pareto_color, linewidth=1.6, zorder=2,
                marker="D", markersize=4, markerfacecolor="none", label="Pareto front")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, axis="x", alpha=GRID_ALPHA)
    _grid_y(ax)

    # Two independent legends (NO titles): color = method, shape = grasp angle. Placement
    # follows legend_pos ("box"/"below"/corner); the second legend stacks clear of the first.
    placed = []
    if method_legend:
        mh = [mlines.Line2D([], [], color=style.mcolor(m, i), marker="o", linestyle="",
                            markersize=7, label=style.mname(m)) for i, m in enumerate(methods)]
        if pareto:
            mh.append(mlines.Line2D([], [], color=pareto_color, marker="D", linestyle="--",
                                    markerfacecolor="none", markersize=5, label="Pareto front"))
        placed.append(_place_legend(ax, mh, pos=legend_pos, slot=0, loc=method_legend_loc))
    if angle_legend and len(angles) > 1:
        if placed and placed[0] is not None:
            ax.add_artist(placed[0])  # keep the method legend when adding the second
        ah = [mlines.Line2D([], [], color="0.35", marker=style.amarker(a, i), linestyle="",
                            markersize=7, label=style.aname(a)) for i, a in enumerate(angles)]
        _place_legend(ax, ah, pos=legend_pos, slot=1, loc=angle_legend_loc)
    return ax


def figure_pareto(collections: Collections, x_metric: str, y_metric: str, style: Style, *,
                  mode: str, selection_metric: str, xlabel: str, ylabel: str,
                  title: str | None = None, xlim=None, ylim=None, pareto: bool = False,
                  legend_pos: str = "box", method_legend_loc: str = "upper right",
                  angle_legend_loc: str = "lower right", figsize=None):
    """Single axes: every run (all methods, all grasp angles) as one point -- ``y_metric`` vs
    ``x_metric``, COLOR = method, SHAPE = grasp angle. ``pareto=True`` overlays the frontier.
    ``legend_pos`` ("box"/"below"/a corner) places the two legends. Returns the Figure.
    """
    xlim, ylim = _scaled_clip(xlim, x_metric), _scaled_clip(ylim, y_metric)
    xlabel, ylabel = _with_unit(xlabel, x_metric), _with_unit(ylabel, y_metric)
    fig, ax = plt.subplots(figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    plot_pareto_scatter(ax, collections.by_cell, x_metric, y_metric, collections.methods,
                        collections.angles, style, mode=mode, selection_metric=selection_metric,
                        xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim,
                        pareto=pareto, legend_pos=legend_pos,
                        method_legend_loc=method_legend_loc, angle_legend_loc=angle_legend_loc)
    fig.tight_layout()
    return fig


def figure_pareto_panels(collections: Collections, x_metric: str, y_metric: str, style: Style, *,
                         mode: str, selection_metric: str, xlabel: str, ylabel: str,
                         suptitle: str | None = None, ncols: int = 2, xlim=None, ylim=None,
                         pareto: bool = False, legend_pos: str = "box", figsize_per=None,
                         method_legend_loc: str = "best"):
    """Small-multiples grid: one panel per GRASP ANGLE, each a per-run scatter of ``y_metric``
    vs ``x_metric`` for that angle. Points are colored by method and use that angle's marker
    shape; ``pareto=True`` overlays a per-angle frontier. Panels share x/y scales so the angles
    are directly comparable; the method legend is on the first panel only (the shape is constant
    within a panel and named in its title). Returns the Figure.
    """
    xlim, ylim = _scaled_clip(xlim, x_metric), _scaled_clip(ylim, y_metric)
    xlabel, ylabel = _with_unit(xlabel, x_metric), _with_unit(ylabel, y_metric)
    angles = collections.angles
    n = len(angles)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    if figsize_per is None:
        figsize_per = (DEFAULT_FIG_WIDTH / ncols, DEFAULT_FIG_HEIGHT / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, sharex=True, sharey=True)
    for i, angle in enumerate(angles):
        ax = axes[i // ncols][i % ncols]
        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1) or (i + ncols >= n)
        plot_pareto_scatter(ax, collections.by_cell, x_metric, y_metric, collections.methods,
                            [angle], style, mode=mode, selection_metric=selection_metric,
                            xlabel=xlabel if is_bottom else None,
                            ylabel=ylabel if is_left else None,
                            title=f"grasp {style.aname(angle)}", xlim=xlim, ylim=ylim,
                            pareto=pareto, method_legend=(i == 0), angle_legend=False,
                            legend_pos=legend_pos, method_legend_loc=method_legend_loc)
    for j in range(n, nrows * ncols):  # hide unused axes
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    return fig


# =========================================================================== #
# Surface-frame stiffness (directional curves, K_n vs K_par scatter, ovals)
# =========================================================================== #
# SURFACE_STIFFNESS_TAGS / PRINCIPAL_STIFFNESS_TAGS (which logged series the stiffness figures
# read) now live in plot_config.py.


def figure_lines_panels(collections: Collections, specs: list, style: Style, *,
                        series: str = "method", ncols: int = 2, ci_clip=None,
                        smooth_window: int = 1, sharey: bool = True, suptitle: str | None = None,
                        figsize_per=None, legend_loc: str = "best",
                        pooled_label: str = "all runs", pooled_color: str = "#333333"):
    """Small-multiples of a metric-vs-env-steps line plot, one panel per ``specs`` entry.

    Each ``specs`` entry is ``{"tag", "ylabel", "title"}``. ``series`` selects what the lines in
    every panel represent:
      * ``"method"`` -- one mean+CI line per method (pooled over grasp angles),
      * ``"angle"``  -- one line per grasp angle (pooled over methods),
      * ``"pooled"`` -- a single line pooling ALL runs (aggregated over methods AND angles).

    Panels share the y-axis by default (``sharey``) so directions are comparable; the legend is
    on the first panel only (``"pooled"`` needs none). Returns the Figure.
    """
    n = len(specs)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    if figsize_per is None:
        figsize_per = (DEFAULT_FIG_WIDTH / ncols, DEFAULT_FIG_HEIGHT / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, sharey=sharey)
    if series == "method":
        rbk, order, cfn, nfn = collections.by_method, collections.methods, style.mcolor, style.mname
    elif series == "angle":
        rbk, order, cfn, nfn = collections.by_angle, collections.angles, style.acolor, style.aname
    elif series == "pooled":
        all_runs = [r for cell in collections.by_cell.values() for r in cell]
        rbk, order = {"all": all_runs}, ["all"]
        cfn, nfn = (lambda k, i=0: pooled_color), (lambda k: pooled_label)
    else:
        raise ValueError(f"unknown series: {series!r} (expected 'method'/'angle'/'pooled')")

    for i, spec in enumerate(specs):
        ax = axes[i // ncols][i % ncols]
        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1) or (i + ncols >= n)
        plot_metric_lines(ax, rbk, spec["tag"], order, color_fn=cfn, name_fn=nfn,
                          ci_z=style.ci_z, ci_clip=ci_clip, smooth_window=smooth_window,
                          xlabel=style.xlabel if is_bottom else None,
                          ylabel=spec.get("ylabel", "stiffness") if is_left else None,
                          title=spec.get("title"), xlim=style.xlim,
                          legend=(i == 0 and series != "pooled"), legend_loc=legend_loc,
                          show_n=(series != "pooled"))
    for j in range(n, nrows * ncols):  # hide unused axes
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    return fig


def _stiffness_oval(runs, style, reduce, tilt_mode, surf_tags, principal_tags):
    """Reconstruct one surface-stiffness oval ``(semi_v, semi_h, angle_deg, has_principals)``
    in TRUE units for a set of runs. ``semi_v`` is the semi-axis on the (angle-tilted) vertical
    (normal) axis, ``semi_h`` across it; ``angle_deg`` is measured off the surface normal.

    ``tilt_mode`` (matching the surface-stiffness cartoon):
      * ``"zaxis"``  -- ONLY the polar tilt of the authored normal axis (principal_z) off the
        true surface normal: ``cos^2(theta) = (k_normal - lam_t)/(principal_z - lam_t)`` with
        ``lam_t = mean(principal_x, principal_y)`` (exact under tangential isotropy).
      * ``"inplane"`` -- the full (along, normal) eigen-tilt: the eigenvalue nearest k_cross is
        the shared axis; the other two give ``cos(2 psi) = (k_normal - k_along)/(lam+ - lam-)``.
      * ``"none"``   -- no reconstruction; upright oval of the raw diagonal (k_normal, k_along).
    """
    sc = style.step_ceiling
    k_n = tag_scalar(runs, surf_tags["normal"], reduce, sc)
    k_a = tag_scalar(runs, surf_tags["along_track"], reduce, sc)
    if k_n is None or k_a is None:
        return None
    k_c = tag_scalar(runs, surf_tags["cross_track"], reduce, sc)
    px = tag_scalar(runs, principal_tags["x"], reduce, sc)
    py = tag_scalar(runs, principal_tags["y"], reduce, sc)
    pz = tag_scalar(runs, principal_tags["z"], reduce, sc)
    have_p = None not in (k_c, px, py, pz)

    if tilt_mode == "none" or not have_p:
        return max(k_n, 0.0), max(k_a, 0.0), 0.0, False
    if tilt_mode == "zaxis":
        lam_n = pz
        lam_t = 0.5 * (px + py)
        denom = lam_n - lam_t
        if abs(denom) > 0.02 * max(lam_n, lam_t, 1.0):
            cos2 = np.clip((k_n - lam_t) / denom, 0.0, 1.0)
            angle = float(np.rad2deg(np.arccos(np.sqrt(cos2))))
        else:
            angle = 0.0
        return max(lam_n, 0.0), max(lam_t, 0.0), angle, True
    # inplane
    evals = sorted((px, py, pz))
    j = int(np.argmin([abs(e - k_c) for e in evals]))
    block = [evals[i] for i in range(3) if i != j]
    lam_m, lam_p = block[0], block[1]
    denom = lam_p - lam_m
    cos2 = np.clip((k_n - k_a) / denom, -1.0, 1.0) if denom > 1e-9 else 1.0
    angle = float(np.rad2deg(0.5 * np.arccos(cos2)))
    return max(lam_p, 0.0), max(lam_m, 0.0), angle, True


def figure_stiffness_ellipses(runs_by_key: dict, key_order: list, style: Style, *,
                              color_fn, name_fn, reduce: str = "last", tilt_mode: str = "zaxis",
                              scale_mode: str = "range", ncols: int | None = None,
                              figsize_per=None, surf_tags=None, principal_tags=None,
                              suptitle: str | None = None):
    """Cartoon row of the surface-frame stiffness "oval", one panel per key (method or angle).

    Vertical = surface normal, horizontal = along-track; each oval's semi-axes come from the
    reduced stiffness of that key's runs (:func:`_stiffness_oval`, ``tilt_mode``) and its tilt is
    drawn off the vertical normal. ``scale_mode="range"`` affine-maps the global [min, max] over
    all semi-axes onto [0.25, 1.0] so small differences pop (read shape/tilt, not absolute size);
    ``"max"`` keeps true relative geometry. Ovals are outlined/filled by ``color_fn(key, idx)``
    and titled by ``name_fn``; the tilt angle (deg off normal) is annotated unless near-circular.
    Returns the Figure.
    """
    surf_tags = surf_tags or SURFACE_STIFFNESS_TAGS
    principal_tags = principal_tags or PRINCIPAL_STIFFNESS_TAGS

    ell = {}
    for key in key_order:
        runs = runs_by_key.get(key)
        if not runs:
            continue
        oval = _stiffness_oval(runs, style, reduce, tilt_mode, surf_tags, principal_tags)
        if oval is not None:
            ell[key] = oval

    keys = [k for k in key_order if k in ell]
    if not keys:
        raise ValueError("no stiffness data to draw (surface-stiffness tags missing?)")

    vals = [v for k in keys for v in ell[k][:2]]
    range_note = None
    if scale_mode == "range" and vals:
        gmin, gmax = min(vals), max(vals)
        span = (gmax - gmin) or 1.0
        to_disp = lambda v: 0.25 + 0.75 * (v - gmin) / span
        range_note = (f"Range-stretched scale: semi-axis 0.25 = {gmin:.0f}, 1.0 = {gmax:.0f} "
                      f"(stiffness units, global min/max)")
    else:
        scale = max(vals) if vals else 1.0
        to_disp = lambda v: v / scale

    t = np.linspace(0, 2 * np.pi, 100)

    def _xy(semi_v, semi_h, ang):
        ex, ey = semi_h * np.cos(t), semi_v * np.sin(t)
        return (ex * np.cos(ang) - ey * np.sin(ang), ex * np.sin(ang) + ey * np.cos(ang))

    ncols = ncols if ncols is not None else max(1, len(keys))
    nrows = max(1, (len(keys) + ncols - 1) // ncols)
    if figsize_per is None:
        figsize_per = (DEFAULT_FIG_WIDTH / ncols, DEFAULT_FIG_HEIGHT / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for k, ax in enumerate(flat):
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= len(keys):
            ax.axis("off")
            continue
        key = keys[k]
        outline = color_fn(key, k)
        ax.plot([0, 0], [-1.1, 1.1], color="0.6", linestyle=":", linewidth=1)          # normal
        ax.plot([-1.1, 1.1], [0, 0], color="0.85", linestyle="-", linewidth=1, zorder=0)  # surface
        semi_v, semi_h, angle_deg, has_p = ell[key]
        big, small = max(semi_v, semi_h), min(semi_v, semi_h)
        near_circular = not (big > 1e-9 and (big - small) / big >= 0.02)
        draw_ang = 0.0 if near_circular else np.deg2rad(angle_deg)
        xr, yr = _xy(to_disp(semi_v), to_disp(semi_h), draw_ang)
        ax.fill(xr, yr, color=outline, alpha=0.35, zorder=2)
        ax.plot(xr, yr, color=outline, linewidth=1.3, zorder=3)
        if has_p and not near_circular:
            ax.text(1.12, 1.12, f"{angle_deg:.0f}°", ha="right", va="top",
                    fontsize=8, color=outline, zorder=5)
        ax.set_title(name_fn(key), fontsize=FONT_TITLE)
    _desc = {"none": "no tilt (raw diagonal)", "zaxis": "normal-axis polar tilt",
             "inplane": "eigenframe rotation"}.get(tilt_mode, tilt_mode)
    fig.suptitle(suptitle or f"Surface-frame stiffness oval (vertical = normal, "
                 f"horizontal = along-track; tilt = {_desc})", fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    if range_note:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.12))
        fig.text(0.5, 0.01, range_note, ha="center", va="bottom", fontsize=8, color="0.3")
    return fig


# =========================================================================== #
# Combined "success" box: by-method curves + angle x method bars, with base icons
# =========================================================================== #
def _resolve_icon_path(p: str) -> str:
    """Resolve an icon image path. Absolute paths and paths that exist as-typed (relative to the
    kernel's CWD) are used directly; otherwise the path is resolved relative to THIS module's
    folder -- i.e. the notebook's folder (``data_analysis/iros_workshop_analysis``), so paths
    written "relative to the notebook" work regardless of where the kernel was launched.
    """
    if os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p)
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, p)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    raise FileNotFoundError(
        f"icon image not found: {p!r}. Tried it relative to the current working dir "
        f"({os.path.abspath(p)!r}) and to the notebook folder ({cand!r}).")


def add_bottom_icons(ax, x_positions, *, images=None, colors=None, height_frac=0.2,
                     y_base: float = 0.0, zorder: int = 5):
    """Place a SQUARE icon flush to the bottom of ``ax``, centered on each x in ``x_positions``
    (data x-coords). Side length = ``height_frac`` of the axes HEIGHT, kept square in display.

    ``images[i]`` is an ``HxWx(3|4)`` array or an image path; ``None`` -> a solid ``colors[i]``
    placeholder square. Call AFTER layout (it forces a draw to size the axes). Returns the list of
    the created ``AnnotationBbox`` artists.
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.colors import to_rgba

    fig = ax.figure
    fig.canvas.draw()                                    # ensure the axes has a pixel size
    side_px = height_frac * ax.get_window_extent().height
    boxes = []
    for i, x in enumerate(x_positions):
        img = images[i] if images else None
        if img is None:
            rgba = to_rgba(colors[i] if colors else "0.6")
            img = np.ones((16, 16, 4)) * np.array(rgba)  # solid-color placeholder
        elif isinstance(img, str):
            img = plt.imread(_resolve_icon_path(img))     # raster path (e.g. a high-res PNG)
        zoom = side_px / img.shape[0]                    # scale so the icon is side_px tall
        # antialiased downsamples a high-res icon smoothly (vs 'nearest', which would alias).
        oi = OffsetImage(img, zoom=zoom, interpolation="antialiased")
        ab = AnnotationBbox(oi, (x, y_base), xycoords=("data", "axes fraction"),
                            box_alignment=(0.5, 0.0), frameon=False, pad=0.0, zorder=zorder)
        ax.add_artist(ab)
        boxes.append(ab)
    return boxes


def figure_success_combo(collections: Collections, style: Style, *, mode: str,
                         selection_metric: str, curve_collections: Collections | None = None,
                         metric: str = "Episode / Success rate", ylabel: str = "Success rate",
                         ci_clip=(0.0, 1.0), left_title: str | None = None,
                         right_title: str | None = None, legend_pos: str = "below",
                         group_width: float = BAR_GROUP_WIDTH, bar_width: float | None = BAR_WIDTH,
                         bar_gap: float = BAR_GAP,
                         icon_images=None, icon_colors=None, icon_height_frac: float = 0.2,
                         figsize=None):
    """Two-column figure: LEFT = ``metric`` vs env steps, one mean+CI line per METHOD (as
    ``success_by_method`` in performance_analysis); RIGHT = ``metric`` by GRASP ANGLE x METHOD
    grouped bars (as ``success_angle_x_method`` here).

    The left curves always come from ``curve_collections`` (default ``collections``) -- pass the
    TRAINING collection there when the bars use eval. If ``icon_images`` (paths/arrays, one per
    grasp angle) is given, each is drawn as a SQUARE icon flush to the bottom of the RIGHT panel,
    centered on its grasp angle (side = ``icon_height_frac`` of the panel height); with
    ``icon_images=None`` (the default) no icons are drawn. Returns the Figure.
    """
    curve_collections = curve_collections or collections
    yl = _with_unit(ylabel, metric)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize or (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT / 2))

    # LEFT: success-by-method training curves. plot_metric_lines -> ci_band scales ci_clip, so pass raw.
    plot_metric_lines(axL, curve_collections.by_method, metric, curve_collections.methods,
                      color_fn=style.mcolor, name_fn=style.mname, ci_z=style.ci_z, ci_clip=ci_clip,
                      xlabel=style.xlabel, ylabel=yl, title=left_title, xlim=style.xlim,
                      legend_loc="best")

    # RIGHT: angle x method grouped bars. plot_grouped_bars clips directly, so pass the SCALED clip.
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_bars(axR, collections.angles, collections.methods, lambda a, m: stat(m, a),
                      color_fn=style.mcolor, group_name_fn=style.mname, x_name_fn=style.aname,
                      ci_clip=_scaled_clip(ci_clip, metric), xlabel="Grasp Angle", ylabel=yl,
                      title=right_title, legend_pos=legend_pos, group_width=group_width,
                      bar_width=bar_width, bar_gap=bar_gap)

    fig.tight_layout()
    # Square icons flush to the bottom of the RIGHT panel, centered on each grasp-angle tick.
    # Only drawn when icon_images is provided -- icon_images=None draws nothing.
    if icon_images is not None:
        if icon_colors is None:
            icon_colors = [style.acolor(a, i) for i, a in enumerate(collections.angles)]
        add_bottom_icons(axR, list(range(len(collections.angles))), images=icon_images,
                         colors=icon_colors, height_frac=icon_height_frac)
    return fig


def figure_success_combo_h(collections: Collections, style: Style, *, mode: str,
                           selection_metric: str, curve_collections: Collections | None = None,
                           metric: str = "Episode / Success rate", ylabel: str = "Success rate",
                           cat_label: str | None = "Grasp Angle", ci_clip=(0.0, 1.0),
                           left_title: str | None = None, right_title: str | None = None,
                           legend_pos: str = "below", bar_width: float = 0.25, group_gap: float = 0.35,
                           bar_gap: float = 0.02, label_rotation: float = 0,
                           reverse_categories: bool = False, curve_width: float = 6.0,
                           bars_width: float = 4.8, panel_height: float = 4.8, wspace: float = 1.0,
                           margins=(0.9, 0.25, 1.15, 0.35), show_labels: bool = True,
                           label_decimal: int = 1, bold_labels: bool = None, show_label_ci: bool = None):
    """Like :func:`figure_success_combo` but the RIGHT panel is the HORIZONTAL angle x method bars
    (:func:`plot_grouped_barh`), sharing the same ``bar_width`` / ``group_gap`` / ``bar_gap`` /
    ``label_rotation`` / ``reverse_categories`` knobs.

    Both panels are drawn at FIXED sizes (inches) so neither stretches: the LEFT (env-steps) axes is
    ``curve_width`` x ``panel_height`` and the RIGHT (bars) axes ``bars_width`` x ``panel_height``.
    ``wspace`` is the whitespace between them (added on the bars' side); ``margins`` = (left, right,
    bottom, top) inches reserved for labels/legend. The figure size is derived from these, so
    changing the total never resizes the sub-plots. Returns the Figure.
    """
    from mpl_toolkits.axes_grid1 import Divider, Size

    curve_collections = curve_collections or collections
    yl = _with_unit(ylabel, metric)
    ml, mr, mb, mt = margins
    fig_w = ml + curve_width + wspace + bars_width + mr
    fig_h = mb + panel_height + mt
    fig = plt.figure(figsize=(fig_w, fig_h))
    hsizes = [Size.Fixed(ml), Size.Fixed(curve_width), Size.Fixed(wspace),
              Size.Fixed(bars_width), Size.Fixed(mr)]
    vsizes = [Size.Fixed(mb), Size.Fixed(panel_height), Size.Fixed(mt)]
    div = Divider(fig, (0.0, 0.0, 1.0, 1.0), hsizes, vsizes, aspect=False)
    axL = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=1, ny=1))  # env-steps
    axR = fig.add_axes(div.get_position(), axes_locator=div.new_locator(nx=3, ny=1))  # horizontal bars

    # LEFT: success-by-method training curves. plot_metric_lines -> ci_band scales ci_clip, so pass raw.
    plot_metric_lines(axL, curve_collections.by_method, metric, curve_collections.methods,
                      color_fn=style.mcolor, name_fn=style.mname, ci_z=style.ci_z, ci_clip=ci_clip,
                      xlabel=style.xlabel, ylabel=yl, title=left_title, xlim=style.xlim,
                      legend_loc="best")

    # RIGHT: HORIZONTAL angle x method bars. plot_grouped_barh clips directly, so pass the SCALED clip.
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_barh(axR, collections.angles, collections.methods, lambda a, m: stat(m, a),
                      color_fn=style.mcolor, group_name_fn=style.mname, cat_name_fn=style.aname,
                      bar_width=bar_width, group_gap=group_gap, bar_gap=bar_gap,
                      ci_clip=_scaled_clip(ci_clip, metric), value_label=yl, cat_label=cat_label,
                      title=right_title, legend_pos=legend_pos, label_rotation=label_rotation,
                      reverse_categories=reverse_categories, show_labels=show_labels,
                      label_decimal=label_decimal, bold_labels=bold_labels, show_label_ci=show_label_ci)
    return fig
