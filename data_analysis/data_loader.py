"""Shared data-loading and reduction utilities for the analysis notebooks.

Single source of truth for *reading* TensorBoard scalar data and reducing it to
the arrays/statistics the plotting layer consumes. The notebooks
(``analysis.ipynb``, ``stiff_rot_analysis.ipynb``) only set parameters and call
into here -- no loading logic lives in a notebook.

Data model produced by :func:`load_data`::

    DATA = { exp_group: [ {tag: (steps, values)}, ... one dict per numbered run ] }

* ``exp_group``  -- a sub-folder of ``runs/{folder}/`` (e.g. ``5_GAS``).
* each numbered run (``0``, ``1``, ...) becomes one ``{tag: (steps, values)}`` dict.
* ``steps``/``values`` are float ``np.ndarray`` of equal length.

Everything downstream (CI bands, best-point tables, per-phase summaries) is built
on top of that dict-of-lists.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def find_project_root(marker: str = "runs") -> str:
    """Walk up from the cwd until a folder containing ``marker`` is found.

    The notebooks live in ``data_analysis/`` but ``runs/`` and ``plots/`` live at
    the repo root, so paths are anchored to whichever ancestor directory holds
    ``runs/``. Works whether the kernel's cwd is ``data_analysis/`` or the repo
    root; falls back to the cwd if no ancestor qualifies.
    """
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root, give up
            return os.path.abspath(os.getcwd())
        d = parent


def runs_root(marker: str = "runs") -> str:
    """Absolute path to the ``runs/`` folder under the project root."""
    return os.path.join(find_project_root(marker), marker)


def step_ceiling_from_xlim(xlim) -> float:
    """Largest step used for run selection / stats: the XLIM upper bound.

    Returns ``inf`` if ``xlim`` (or its upper bound) is unset. Both the TOP_N run
    ranking and the best-point table clip to this, so longer runs are truncated
    and every method is compared at equal data efficiency.
    """
    if xlim is not None and xlim[1] is not None:
        return float(xlim[1])
    return np.inf


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_run(run_dir: str) -> dict:
    """Load all scalar tags from every event file in a single numbered run dir.

    A numbered run may contain several ``events.out.tfevents.*`` files (some
    empty); all are merged. If a tag appears in more than one file, the copy with
    more points wins.
    """
    tags: dict = {}
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    for ef in event_files:
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
    """Load every experiment group under ``runs/{folder_name}``.

    ``root`` defaults to :func:`runs_root`. Returns the dict-of-lists described in
    the module docstring; groups with no readable run are skipped.
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
        run_names = [d for d in os.listdir(group_dir) if d.isdigit()]
        for run_name in sorted(run_names, key=int):
            run_tags = load_run(os.path.join(group_dir, run_name))
            if run_tags:
                runs.append(run_tags)
        if runs:
            data[group] = runs
            if verbose:
                print(f"{group}: {len(runs)} runs loaded")
    return data


# --------------------------------------------------------------------------- #
# wandb -> local TensorBoard cache (download once, both notebooks reuse)
# --------------------------------------------------------------------------- #
def _group_complete(group_dir: str, expected: int) -> bool:
    """True if ``group_dir`` already holds a full set of downloaded runs.

    Requires at least ``expected`` numbered run sub-dirs, each carrying an
    ``events.out.tfevents.*`` file. Used as the download-once cache check so a
    complete group is never re-fetched (and a partial one IS re-fetched).
    """
    if not os.path.isdir(group_dir):
        return False
    run_dirs = [d for d in os.listdir(group_dir) if d.isdigit()]
    if len(run_dirs) < expected:
        return False
    return all(
        glob.glob(os.path.join(group_dir, d, "events.out.tfevents.*")) for d in run_dirs
    )


def _write_tfevents(run_dir: str, hist, step_key: str = "_step") -> int:
    """Write one wandb run's history DataFrame to a TensorBoard event file in ``run_dir``.

    Every numeric scalar column becomes an ``add_scalar`` series keyed on ``step_key``
    (wandb ``_step`` == the training timestep, so the step axis matches the native TB
    files). Non-numeric / internal (``_*``, ``gradient*``) columns and non-finite points
    are skipped. Returns the number of scalar series written.
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
                break  # a non-scalar column (media / string) — skip it entirely
            if not math.isfinite(fv) or not math.isfinite(float(s)):
                continue
            writer.add_scalar(col, fv, int(s))
            wrote = True
        n_series += int(wrote)
    writer.flush()
    writer.close()
    return n_series


def download_wandb_data(project: str, tag: str, entity: str | None = "hur",
                        root: str | None = None, samples: int = 100_000,
                        step_key: str = "_step", verbose: bool = True) -> str:
    """Download every wandb run in ``{entity}/{project}`` carrying ``tag`` into the local
    ``runs/{project}_{tag}/{group}/{agent_index}/`` tree as TensorBoard event files — the
    exact layout :func:`load_data` reads — then return the folder name ``"{project}_{tag}"``.

    Runs are bucketed by their wandb ``group`` (the method, e.g. ``5_GAS``) and numbered by
    ``config['agent_index']``. **Download-once cache:** a group whose local dir already holds
    a complete set of runs is skipped (see :func:`_group_complete`), so re-running this — or
    the *other* analysis notebook — reuses the data without re-downloading, even if plotting
    settings change. Delete a group's folder to force a refresh.

    Requires ``wandb`` (API access) and ``torch`` (the tfevents writer); both are imported
    lazily so the local-only workflow needs neither.
    """
    from collections import defaultdict

    import wandb

    folder = f"{project}_{tag}"
    base = os.path.join(root or runs_root(), folder)

    api = wandb.Api(timeout=60)
    path = project if "/" in project else (f"{entity}/{project}" if entity else project)
    runs = list(api.runs(path, filters={"tags": tag}))
    if not runs:
        raise RuntimeError(f"no wandb runs found in {path!r} tagged {tag!r}")

    groups: dict = defaultdict(list)
    for r in runs:
        groups[r.group or r.name].append(r)

    if verbose:
        print(f"[wandb-dl] {len(runs)} run(s) in {path} tagged '{tag}' "
              f"-> {len(groups)} group(s) -> {base}")

    for group in sorted(groups):
        grp_runs = groups[group]
        group_dir = os.path.join(base, group)
        if _group_complete(group_dir, len(grp_runs)):
            if verbose:
                print(f"[wandb-dl]   {group}: cached ({len(grp_runs)} runs), skip")
            continue
        for i, r in enumerate(sorted(grp_runs, key=lambda rr: rr.config.get("agent_index", 0))):
            idx = int(r.config.get("agent_index", i))
            run_dir = os.path.join(group_dir, str(idx))
            os.makedirs(run_dir, exist_ok=True)
            for old in glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
                os.remove(old)  # clear any partial prior download for a clean rewrite
            hist = r.history(samples=samples, pandas=True)
            n = _write_tfevents(run_dir, hist, step_key)
            if verbose:
                print(f"[wandb-dl]   {group}/{idx}: {n} series ({len(hist)} steps) from {r.name}")
    return folder


def filter_top_n(data: dict, n: int, metric: str,
                 step_ceiling: float = np.inf, verbose: bool = True) -> dict:
    """Keep only the top-``n`` runs per group, ranked by each run's peak ``metric``.

    The peak is taken over steps at or below ``step_ceiling``, so ranking matches
    how the best-point table evaluates runs. Runs missing ``metric`` (or with no
    data in the window) rank last. ``n == -1`` (or ``None``) keeps all runs.
    """
    if n is None or n < 0:
        return data

    def peak(run):
        if metric not in run:
            return -np.inf
        steps, vals = run[metric]
        keep = steps <= step_ceiling
        return float(np.max(vals[keep])) if keep.any() else -np.inf

    out: dict = {}
    for group, runs in data.items():
        ranked = sorted(runs, key=peak, reverse=True)
        out[group] = ranked[:n]
        if verbose:
            print(f"{group}: kept top {len(out[group])}/{len(runs)} runs (by {metric})")
    return out


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #
def moving_average(vals: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over ``window`` points.

    Edges shrink the window to the points available (no zero-padding dip), so the
    smoothed curve starts/ends on the real data rather than decaying to 0.
    """
    if window is None or window <= 1 or vals.size == 0:
        return vals
    w = int(min(window, vals.size))
    kernel = np.ones(w)
    sums = np.convolve(vals, kernel, mode="same")
    counts = np.convolve(np.ones_like(vals), kernel, mode="same")
    return sums / counts


def smooth_data(data: dict, metric: str, window: int) -> dict:
    """Copy of ``data`` with ``metric``'s values replaced by their moving average.

    Steps are untouched; only ``metric``'s values are smoothed (every other tag
    is shared by reference). Runs without ``metric`` pass through unchanged.
    """
    out: dict = {}
    for group, runs in data.items():
        new_runs = []
        for run in runs:
            if metric in run:
                steps, vals = run[metric]
                run = {**run, metric: (steps, moving_average(vals, window))}
            new_runs.append(run)
        out[group] = new_runs
    return out


# --------------------------------------------------------------------------- #
# Time-series aggregation (mean + SEM across a group's runs)
# --------------------------------------------------------------------------- #
def aggregate_runs(runs: list, metric: str):
    """Return ``(grid, mean, sem)`` for ``metric`` across a group's runs.

    Runs are interpolated onto the shared step grid (the union of their steps,
    clipped to the overlap window) so they can differ in length. ``sem`` is the
    standard error of the mean across runs (0 for a single run). Returns ``None``
    if no run carries ``metric``.
    """
    series = [r[metric] for r in runs if metric in r]
    if not series:
        return None

    lo = max(s[0].min() for s in series)
    hi = min(s[0].max() for s in series)
    grid = np.unique(np.concatenate([s[0] for s in series]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    if grid.size == 0:
        return None

    stacked = np.vstack([np.interp(grid, steps, vals) for steps, vals in series])
    mean = stacked.mean(axis=0)
    n = stacked.shape[0]
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return grid, mean, sem


# --------------------------------------------------------------------------- #
# Best-point reduction (value of each metric at the selection metric's peak)
# --------------------------------------------------------------------------- #
def best_index(run: dict, selection_metric: str, step_ceiling: float = np.inf):
    """Index into ``selection_metric`` where it peaks (None if absent/all clipped).

    Only steps at or below ``step_ceiling`` are considered; the returned index
    still refers to the original (unclipped) arrays.
    """
    if selection_metric not in run:
        return None
    steps, vals = run[selection_metric]
    keep = np.flatnonzero(steps <= step_ceiling)
    if keep.size == 0:
        return None
    return int(keep[np.argmax(vals[keep])])


def value_at_best(run: dict, metric: str, best_step: float, step_ceiling: float = np.inf):
    """Value of ``metric`` at the step nearest ``best_step`` (None if absent).

    Steps above ``step_ceiling`` are dropped before matching so a clipped run
    can't read a value past the cutoff.
    """
    if metric not in run:
        return None
    steps, vals = run[metric]
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    steps, vals = steps[keep], vals[keep]
    return float(vals[int(np.argmin(np.abs(steps - best_step)))])


def best_point_stats(data: dict, selection_metric: str, metric: str,
                     ci_z: float = 1.96, step_ceiling: float = np.inf,
                     offset: float = 0.0, rms: bool = False) -> dict:
    """``{group: (value, ci)}`` of ``metric`` taken at each run's best selection point.

    Each run is reduced to ``metric`` at the step where ``selection_metric`` peaks, then
    ``offset`` is subtracted -- so a metric with a target value can be re-centered on it
    (``value - offset == 0`` means "on target", e.g. ``offset`` = desired force). Across a
    group's runs:

    * ``rms=False`` (default): ``value`` is the mean signed error, ``ci`` its ``ci_z * SEM``.
    * ``rms=True``: ``value`` is the root-mean-square error ``sqrt(mean(e_i**2))``. Unlike the
      signed mean, RMS does NOT let per-run over/undershoots cancel to a misleadingly small
      number; ``ci`` remains ``ci_z * SEM`` of the per-run errors (a run-to-run spread cue).

    ``offset`` is in the metric's RAW units -- it is applied here, BEFORE any display
    ``scale`` the table multiplies in later. With ``offset=0.0, rms=False`` this reproduces
    the original plain-mean behavior.
    """
    out: dict = {}
    for group, runs in data.items():
        vals = []
        for run in runs:
            bi = best_index(run, selection_metric, step_ceiling)
            if bi is None:
                continue
            best_step = run[selection_metric][0][bi]
            v = value_at_best(run, metric, best_step, step_ceiling)
            if v is not None:
                vals.append(v)
        if vals:
            err = np.array(vals) - offset
            n = len(err)
            sem = err.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            value = float(np.sqrt(np.mean(err ** 2))) if rms else float(err.mean())
            out[group] = (value, ci_z * sem)
    return out


# --------------------------------------------------------------------------- #
# Scalar reduction for end-of-training / per-phase summaries
# --------------------------------------------------------------------------- #
def reduce_run_value(run: dict, tag: str, reduce: str = "last",
                     step_ceiling: float = np.inf, selection_metric: str | None = None,
                     tail_n: int = 5):
    """Reduce one run's ``tag`` time series to a single scalar.

    ``reduce``:
      * ``"last"``      -- value at the last in-window step (default).
      * ``"mean_tail"`` -- mean of the last ``tail_n`` in-window points (denoises a
                           converged metric).
      * ``"max"`` / ``"min"`` -- extreme over the window.
      * ``"at_best"``   -- value at the step where ``selection_metric`` peaks
                           (requires ``selection_metric``).

    Returns ``None`` if ``tag`` is absent or no point falls in the window.
    """
    if tag not in run:
        return None
    steps, vals = run[tag]
    keep = steps <= step_ceiling
    if not keep.any():
        return None
    steps, vals = steps[keep], vals[keep]

    if reduce == "last":
        return float(vals[-1])
    if reduce == "mean_tail":
        return float(vals[-min(int(tail_n), vals.size):].mean())
    if reduce == "max":
        return float(vals.max())
    if reduce == "min":
        return float(vals.min())
    if reduce == "at_best":
        if selection_metric is None:
            raise ValueError("reduce='at_best' requires selection_metric")
        bi = best_index(run, selection_metric, step_ceiling)
        if bi is None:
            return None
        best_step = run[selection_metric][0][bi]
        return value_at_best(run, tag, best_step, step_ceiling)
    raise ValueError(f"unknown reduce: {reduce!r}")


def summarize_tag(data: dict, tag: str, reduce: str = "last", ci_z: float = 1.96,
                  step_ceiling: float = np.inf, selection_metric: str | None = None,
                  tail_n: int = 5) -> dict:
    """``{group: (mean, ci, vals)}`` of ``tag`` reduced per run then aggregated.

    ``vals`` is the per-run array (one scalar per run, via :func:`reduce_run_value`);
    ``mean``/``ci`` summarize it across runs (``ci`` = ``ci_z * SEM``, 0 for a
    single run). Groups with no usable run are omitted.
    """
    out: dict = {}
    for group, runs in data.items():
        vals = [reduce_run_value(run, tag, reduce, step_ceiling, selection_metric, tail_n)
                for run in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        ci = ci_z * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        out[group] = (float(arr.mean()), float(ci), arr)
    return out


def phase_summary(data: dict, tag_template: str, phases, **kwargs) -> dict:
    """Per-phase reduction of one metric family: ``{group: {phase: (mean, ci, vals)}}``.

    ``tag_template`` contains a ``{phase}`` placeholder, e.g.
    ``"RotationFrame_{phase}/k_axial_mean"`` or
    ``"Impedance_Stiffness_{phase}/pos_Z"``. Extra kwargs (``reduce``, ``ci_z``,
    ``step_ceiling``, ``selection_metric``, ``tail_n``) pass through to
    :func:`summarize_tag`. A (group, phase) pair with no data is simply absent.
    """
    out: dict = {group: {} for group in data}
    for phase in phases:
        tag = tag_template.format(phase=phase)
        per_group = summarize_tag(data, tag, **kwargs)
        for group, triple in per_group.items():
            out[group][phase] = triple
    return {g: d for g, d in out.items() if d}
