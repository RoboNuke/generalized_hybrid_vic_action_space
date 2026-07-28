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
    series = [r[metric] for r in runs if metric in r]
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
    steps, vals = run[metric]
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
    steps, vals = run[metric]
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    return float(vals[keep][-1])


def per_agent_value(run: dict, metric: str, mode: str, selection_metric: str,
                    step_ceiling: float = np.inf):
    """One representative scalar of ``metric`` for a single agent's ``run``.

    ``mode="best"`` returns ``metric`` at the step where ``selection_metric`` peaks in that
    run (its best checkpoint); ``mode="eval"`` returns the run's single (last in-window)
    value. Returns ``None`` if the needed tag is missing.
    """
    if mode == "eval":
        return value_last(run, metric, step_ceiling)
    if mode == "best":
        bi = best_index(run, selection_metric, step_ceiling)
        if bi is None:
            return None
        best_step = run[selection_metric][0][bi]
        return value_at_step(run, metric, best_step, step_ceiling)
    raise ValueError(f"unknown mode: {mode!r} (expected 'best' or 'eval')")


def summarize_agents(runs: list, metric: str, mode: str, selection_metric: str,
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


# =========================================================================== #
# Method / grasp-angle grouping (parse the "{method}_{angle}" group keys)
# =========================================================================== #
# Canonical display order; anything present but unlisted is appended in sorted order.
METHOD_ORDER = ["fixed_geo", "GAS_geo", "GAS", "VICES"]


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
# Styling: fonts, method & grasp-angle palettes, display names
# =========================================================================== #
FONT_SUPTITLE = 15
FONT_TITLE = 12
FONT_AXIS_LABEL = 11
FONT_TICK = 9
FONT_LEGEND = 9

LINE_WIDTH = 2.0        # mean line
BAND_ALPHA = 0.20       # CI shaded region
GRID_ALPHA = 0.30
ERROR_CAPSIZE = 3       # bar-chart error-bar caps

# Method palette (categorical). Validated colorblind-safe as a set
# (worst adjacent CVD dE 25.0 on a light surface): blue / orange / teal / violet.
METHOD_COLORS = {
    "fixed_geo": "#F28E4C",    
    "GAS_geo":   "#4A90C9",   
    "GAS":       "#58c558",   
    "VICES":     "#e06d80", 
    #"fixed_geo": "#2a78d6",   # blue
    #"GAS_geo":   "#eb6834",   # orange
    #"GAS":       "#1baf7a",   # teal
    #"VICES":     "#4a3aa7",   # violet
}
METHOD_NAMES = {
    "fixed_geo": "Fixed-Geo",
    "GAS_geo":   "GAS-Geo",
    "GAS":       "GAS",
    "VICES":     "VICES",
}

# Grasp-angle palette (ordinal: one blue hue, light -> dark with the angle).
# Validated as an ordinal ramp (monotone lightness, light end clears the surface).
ANGLE_COLORS = {
    "0":  "#4c75e6ff",
    "15": "#52cf72",
    "30": "#cc8b4e",
    "45": "#CE4747",
}

# Grasp-angle marker shapes (scatter plots encode angle by SHAPE, method by COLOR, so a
# point carries both). circle / square / triangle / hexagon for 0 / 15 / 30 / 45.
ANGLE_MARKERS = {
    "0":  "o",   # circle
    "15": "s",   # square
    "30": "^",   # triangle
    "45": "h",   # hexagon
}

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
    method_colors: dict = field(default_factory=lambda: dict(METHOD_COLORS))
    method_names: dict = field(default_factory=lambda: dict(METHOD_NAMES))
    angle_colors: dict = field(default_factory=lambda: dict(ANGLE_COLORS))
    angle_markers: dict = field(default_factory=lambda: dict(ANGLE_MARKERS))

    def mcolor(self, method: str, idx: int = 0) -> str:
        return self.method_colors.get(method, _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])

    def mname(self, method: str) -> str:
        return self.method_names.get(method, method)

    def acolor(self, angle: str, idx: int = 0) -> str:
        return self.angle_colors.get(str(angle), _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])

    def aname(self, angle: str) -> str:
        return angle_name(angle)

    def amarker(self, angle: str, idx: int = 0) -> str:
        return self.angle_markers.get(str(angle), _MARKER_CYCLE[idx % len(_MARKER_CYCLE)])

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


# =========================================================================== #
# Single-plot builders (draw ONE axes; the figure_* aggregators call these)
# =========================================================================== #
def plot_metric_lines(ax, runs_by_key: dict, metric: str, key_order: list, *,
                      color_fn, name_fn, ci_z: float = 1.96, ci_clip=None,
                      smooth_window: int = 1, xlabel: str | None = None,
                      ylabel: str | None = None, title: str | None = None,
                      xlim=None, ylim=None, legend: bool = True,
                      legend_loc: str = "best", show_n: bool = True):
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
    ax.grid(True, alpha=GRID_ALPHA)
    if legend:
        ax.legend(loc=legend_loc, fontsize=FONT_LEGEND)
    return ax


def plot_metric_bars(ax, heights_by_key: dict, key_order: list, *, color_fn, name_fn,
                     ci_clip=None, ylabel: str | None = None, title: str | None = None,
                     ylim=None, rotate_labels: int = 0):
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
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.margins(x=0.02)
    return ax


def plot_grouped_bars(ax, x_keys: list, group_keys: list, stat_fn, *, color_fn, group_name_fn,
                      x_name_fn, ci_clip=None, ylabel: str | None = None, title: str | None = None,
                      ylim=None, legend: bool = True, legend_title: str | None = None,
                      group_width: float = 0.82):
    """Clustered bars: each x-axis tick is an ``x_key`` holding one bar per ``group_key``.

    ``stat_fn(x_key, group_key)`` returns ``(value, ci)`` (mean + 95%-CI half-width) or ``None``
    to omit that bar. Inner bars are colored by ``color_fn(group_key, idx)`` and the group legend
    is labeled by ``group_name_fn``; x ticks by ``x_name_fn``. ``ci_clip = (min, max)`` clips the
    error-bar whiskers (``value +/- ci``) to those limits, leaving bar heights untouched.
    Returns ``ax``. The two orientations (x=method/bars=angle and x=angle/bars=method) are the
    same call with the roles swapped.
    """
    nG = max(len(group_keys), 1)
    bw = group_width / nG
    xs = np.arange(len(x_keys), dtype=float)
    for gi, gk in enumerate(group_keys):
        offset = (gi - (nG - 1) / 2.0) * bw
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
        bars = ax.bar(xs + offset, vals, width=bw * 0.9,
                      yerr=[lower_err, upper_err], capsize=ERROR_CAPSIZE,
                      color=color_fn(gk, gi), edgecolor="black", linewidth=0.5,
                      error_kw={"elinewidth": 0.9}, label=group_name_fn(gk))
        for b in bars:
            b.set_gid(re.sub(r"\s+", "_", group_name_fn(gk)))
    ax.set_xticks(xs)
    ax.set_xticklabels([x_name_fn(xk) for xk in x_keys], fontsize=FONT_TICK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontsize=FONT_TITLE)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.margins(x=0.02)
    if legend:
        ax.legend(title=legend_title, fontsize=FONT_LEGEND)
    return ax


# =========================================================================== #
# Figure aggregators (assemble a figure by calling the single-plot builders)
# =========================================================================== #
def figure_by_method(collections: Collections, metric: str, style: Style, *,
                     ylabel: str, title: str | None = None, ylim=None, ci_clip=None,
                     smooth_window: int = 1, legend_loc: str = "best",
                     figsize=(7.5, 4.8)):
    """One axes: ``metric`` vs env steps, one mean+CI line per METHOD (seeds pooled over
    all grasp angles), colored by the method palette. Returns the Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
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
                    figsize=(7.5, 4.8)):
    """One axes: ``metric`` vs env steps, one mean+CI line per GRASP ANGLE (seeds pooled
    over all methods), colored by the ordinal angle palette. Returns the Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
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
                            figsize_per=(4.6, 3.5), legend_loc: str = "best"):
    """Small-multiples grid: one panel per GRASP ANGLE, each overlaying the METHOD mean+CI
    lines for that single angle (seeds within each (method, angle) cell pooled).

    Every panel shares the method palette; the legend is drawn on the first panel only
    (the colors are identical across panels). Returns the Figure.
    """
    angles = collections.angles
    n = len(angles)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
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
                               title: str | None = None, ylim=None, ci_clip=None,
                               figsize=(8.0, 4.8)):
    """Clustered bars: x-axis = METHOD, one bar per GRASP ANGLE within each method
    (colored by the angle palette). Returns the Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_bars(ax, collections.methods, collections.angles,
                      lambda m, a: stat(m, a), color_fn=style.acolor,
                      group_name_fn=style.aname, x_name_fn=style.mname, ci_clip=ci_clip,
                      ylabel=ylabel, title=title, ylim=ylim, legend_title="grasp angle")
    fig.tight_layout()
    return fig


def figure_bars_angle_x_method(collections: Collections, metric: str, style: Style, *,
                               mode: str, selection_metric: str, ylabel: str,
                               title: str | None = None, ylim=None, ci_clip=None,
                               figsize=(8.0, 4.8)):
    """Clustered bars: x-axis = GRASP ANGLE, one bar per METHOD within each angle
    (colored by the method palette). Returns the Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stat = _cell_stat(collections, metric, style, mode, selection_metric)
    plot_grouped_bars(ax, collections.angles, collections.methods,
                      lambda a, m: stat(m, a), color_fn=style.mcolor,
                      group_name_fn=style.mname, x_name_fn=style.aname, ci_clip=ci_clip,
                      ylabel=ylabel, title=title, ylim=ylim, legend_title="method")
    fig.tight_layout()
    return fig


def figure_bars_by_method(collections: Collections, metric: str, style: Style, *,
                          mode: str, selection_metric: str, ylabel: str,
                          title: str | None = None, ylim=None, ci_clip=None,
                          figsize=(6.5, 4.5)):
    """Single bar per METHOD, aggregated over ALL grasp angles (every seed of the method,
    pooled). Colored by the method palette. Returns the Figure.
    """
    heights = {m: summarize_agents(collections.by_method[m], metric, mode, selection_metric,
                                   ci_z=style.ci_z, step_ceiling=style.step_ceiling)
               for m in collections.methods}
    heights = {m: v for m, v in heights.items() if v is not None}
    fig, ax = plt.subplots(figsize=figsize)
    plot_metric_bars(ax, heights, collections.methods, color_fn=style.mcolor,
                     name_fn=style.mname, ci_clip=ci_clip, ylabel=ylabel, title=title, ylim=ylim)
    fig.tight_layout()
    return fig


def figure_bars_by_angle(collections: Collections, metric: str, style: Style, *,
                         mode: str, selection_metric: str, ylabel: str,
                         title: str | None = None, ylim=None, ci_clip=None,
                         figsize=(6.5, 4.5)):
    """Single bar per GRASP ANGLE, aggregated over ALL methods (every seed at the angle,
    pooled). Colored by the ordinal angle palette. Returns the Figure.
    """
    heights = {a: summarize_agents(collections.by_angle[a], metric, mode, selection_metric,
                                   ci_z=style.ci_z, step_ceiling=style.step_ceiling)
               for a in collections.angles}
    heights = {a: v for a, v in heights.items() if v is not None}
    fig, ax = plt.subplots(figsize=figsize)
    plot_metric_bars(ax, heights, collections.angles, color_fn=style.acolor,
                     name_fn=style.aname, ci_clip=ci_clip, ylabel=ylabel, title=title, ylim=ylim)
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
                        angle_legend_loc: str = "lower right", pareto: bool = False,
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
    ax.grid(True, alpha=GRID_ALPHA)

    # Two independent legends: color = method, shape = grasp angle.
    placed = []
    if method_legend:
        mh = [mlines.Line2D([], [], color=style.mcolor(m, i), marker="o", linestyle="",
                            markersize=7, label=style.mname(m)) for i, m in enumerate(methods)]
        if pareto:
            mh.append(mlines.Line2D([], [], color=pareto_color, marker="D", linestyle="--",
                                    markerfacecolor="none", markersize=5, label="Pareto front"))
        placed.append(ax.legend(handles=mh, loc=method_legend_loc, fontsize=FONT_LEGEND,
                                title="method"))
    if angle_legend and len(angles) > 1:
        if placed:
            ax.add_artist(placed[0])  # keep the method legend when adding the second
        ah = [mlines.Line2D([], [], color="0.35", marker=style.amarker(a, i), linestyle="",
                            markersize=7, label=style.aname(a)) for i, a in enumerate(angles)]
        ax.legend(handles=ah, loc=angle_legend_loc, fontsize=FONT_LEGEND, title="grasp angle")
    return ax


def figure_pareto(collections: Collections, x_metric: str, y_metric: str, style: Style, *,
                  mode: str, selection_metric: str, xlabel: str, ylabel: str,
                  title: str | None = None, xlim=None, ylim=None, pareto: bool = False,
                  method_legend_loc: str = "upper right", angle_legend_loc: str = "lower right",
                  figsize=(7.2, 5.4)):
    """Single axes: every run (all methods, all grasp angles) as one point -- ``y_metric`` vs
    ``x_metric``, COLOR = method, SHAPE = grasp angle. ``pareto=True`` overlays the frontier.
    Returns the Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_pareto_scatter(ax, collections.by_cell, x_metric, y_metric, collections.methods,
                        collections.angles, style, mode=mode, selection_metric=selection_metric,
                        xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim,
                        pareto=pareto, method_legend_loc=method_legend_loc,
                        angle_legend_loc=angle_legend_loc)
    fig.tight_layout()
    return fig


def figure_pareto_panels(collections: Collections, x_metric: str, y_metric: str, style: Style, *,
                         mode: str, selection_metric: str, xlabel: str, ylabel: str,
                         suptitle: str | None = None, ncols: int = 2, xlim=None, ylim=None,
                         pareto: bool = False, figsize_per=(4.4, 3.6),
                         method_legend_loc: str = "best"):
    """Small-multiples grid: one panel per GRASP ANGLE, each a per-run scatter of ``y_metric``
    vs ``x_metric`` for that angle. Points are colored by method and use that angle's marker
    shape; ``pareto=True`` overlays a per-angle frontier. Panels share x/y scales so the angles
    are directly comparable; the method legend is on the first panel only (the shape is constant
    within a panel and named in its title). Returns the Figure.
    """
    angles = collections.angles
    n = len(angles)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
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
                            method_legend_loc=method_legend_loc)
    for j in range(n, nrows * ncols):  # hide unused axes
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE)
    fig.tight_layout()
    return fig
