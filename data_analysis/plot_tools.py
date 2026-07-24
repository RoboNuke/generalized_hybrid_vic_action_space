"""Shared plotting utilities for the analysis notebooks.

Single source of truth for *drawing* figures from the data structures produced by
``data_loader``. The notebooks (``analysis.ipynb``, ``stiff_rot_analysis.ipynb``)
set parameters (which metric, labels, limits, output folder) and call these
functions; no matplotlib lives in a notebook beyond ``plt.show()`` /
``tight_layout()``.

Styling, the confidence-band z, the shared x-axis, and the output folder are
bundled in :class:`PlotStyle` so a notebook builds one ``STYLE`` object in its
globals cell and threads it through every call instead of passing a dozen kwargs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import data_loader as dl


# --------------------------------------------------------------------------- #
# Default styling for the glued_rot_rew controller ablation
# --------------------------------------------------------------------------- #
# Raw experiment-group folder -> pretty legend name.
GLUED_ROT_DISPLAY_NAMES = {
    "1_fixed": "Fixed",
    "2_VICES": "VICES",
    "3_choleskey": "Cholesky",
    "4_GAS_fixed_rot": "Ours (GT Geo-Rot)",
    "5_GAS": "Ours (No-Losses)", 
    "6_dynm_fixed_rot": "Ours (GT Dyn-Rot)",
    "7_GAS_dyn": "Ours (Dyn-Loss)", 
    "8_GAS_geo": "Ours (Geo-Loss)"
}

# Fixed color per group (stable across every plot).
GLUED_ROT_GROUP_COLORS = {
    "1_fixed": "#ff9500",          # orange  -- isotropic baseline
    "2_VICES": "#ff0000",          # red     -- diagonal, axis-aligned
    "3_choleskey": "#1f77b4",      # blue    -- full SPD, fixed frame
    "4_GAS_fixed_rot": "#FF00EE",  # purple  -- diagonal in a fixed rotated frame
    "5_GAS": "#1fb426",            # green   -- diagonal + learned rotation
    "6_dynm_fixed_rot":  "#7107b8",
    "7_GAS_dyn":  "#0004ff", 
    "8_GAS_geo":  "#00ffee"
}


@dataclass
class PlotStyle:
    """Bundle of display/styling parameters threaded through every plot call.

    ``display_names`` / ``group_colors`` map raw group folders to legend names and
    colors. ``ci_z`` scales the SEM into a confidence band (1.96 -> ~95%).
    ``xlabel`` / ``xlim`` set the shared x-axis for time-series plots. ``plots_dir``
    is where :meth:`save` writes SVGs.
    """
    display_names: dict = field(default_factory=lambda: dict(GLUED_ROT_DISPLAY_NAMES))
    group_colors: dict = field(default_factory=lambda: dict(GLUED_ROT_GROUP_COLORS))
    ci_z: float = 1.96
    xlabel: str = "Env Steps"
    xlim: tuple | None = None
    plots_dir: str | None = None
    fallback_cmap: str = "tab10"

    def name(self, group: str) -> str:
        return self.display_names.get(group, group)

    def color(self, group: str, idx: int = 0):
        fallback = plt.get_cmap(self.fallback_cmap).colors
        return self.group_colors.get(group, fallback[idx % len(fallback)])

    @property
    def step_ceiling(self) -> float:
        return dl.step_ceiling_from_xlim(self.xlim)

    def save(self, fig, name: str):
        """Save ``fig`` into ``plots_dir`` as ``<name>.svg`` (``/`` -> ``_``)."""
        if self.plots_dir is None:
            raise ValueError("PlotStyle.plots_dir is not set")
        os.makedirs(self.plots_dir, exist_ok=True)
        fname = name.replace(" / ", "_").replace("/", "_").strip()
        path = os.path.join(self.plots_dir, f"{fname}.svg")
        fig.savefig(path, format="svg", bbox_inches="tight")
        print(f"saved {path}")
        return path


def save_plot(fig, name: str, style: PlotStyle):
    """Module-level alias for :meth:`PlotStyle.save` (kept for notebook brevity)."""
    return style.save(fig, name)


# --------------------------------------------------------------------------- #
# Title helper
# --------------------------------------------------------------------------- #
def title_with_n(title: str, n: int) -> str:
    """Replace a trailing ``(...)`` in ``title`` with ``(n={n})`` (else append it).

    Keeps the base name and swaps whatever was in the final parenthetical (e.g.
    ``(Best)``, ``(n=3)``) for the actual trajectory count behind the curves.
    """
    base = title.rstrip()
    if base.endswith(")"):
        i = base.rfind("(")
        if i != -1:
            base = base[:i].rstrip()
    return f"{base} (n={n})"


# --------------------------------------------------------------------------- #
# Time-series plot (mean + CI band) -- shared by both notebooks
# --------------------------------------------------------------------------- #
def plot_metric(data: dict, metric: str, ylabel: str, title: str, style: PlotStyle,
                groups=None, ax=None, legend_loc="best", ylim=None,
                xlabel=None, xlim=None):
    """Plot mean + CI band of ``metric`` vs. step for each experiment group.

    Runs in a group are interpolated onto a shared step grid (see
    :func:`data_loader.aggregate_runs`); the band is ``mean +/- style.ci_z * SEM``.
    ``xlabel``/``xlim`` default to ``style.xlabel``/``style.xlim`` so every plot
    shares one x-scale; pass them to override a single plot. ``ylim`` is
    ``(min, max)`` with ``None`` autoscaling that bound.

    The trailing ``(...)`` of ``title`` is replaced with ``(n=x)`` for the number
    of runs actually aggregated. Each mean line / band gets a gid (the display
    name, spaces -> ``_``) so curves are selectable by name in Inkscape.
    """
    xlabel = style.xlabel if xlabel is None else xlabel
    xlim = style.xlim if xlim is None else xlim
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    groups = groups if groups is not None else list(data.keys())
    run_counts = []
    for idx, group in enumerate(groups):
        if group not in data:
            print(f"[skip] group not loaded: {group}")
            continue
        agg = dl.aggregate_runs(data[group], metric)
        if agg is None:
            print(f"[skip] metric '{metric}' not found in: {group}")
            continue
        run_counts.append(sum(1 for r in data[group] if metric in r))
        grid, mean, sem = agg
        color = style.color(group, idx)
        name = style.name(group)
        gid = name.replace(" ", "_")
        line, = ax.plot(grid, mean, color=color, label=name)
        line.set_gid(gid)
        band = ax.fill_between(grid, mean - style.ci_z * sem, mean + style.ci_z * sem,
                               color=color, alpha=0.2, linewidth=0)
        band.set_gid(f"{gid}_band")

    n = max(run_counts) if run_counts else 0
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_n(title, n))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3)
    return ax


# --------------------------------------------------------------------------- #
# Best-point LaTeX table
# --------------------------------------------------------------------------- #
def build_latex_table(data: dict, selection_metric: str, table_metrics: list,
                      table_groups: list, style: PlotStyle,
                      default_decimals: int = 3) -> str:
    """Build a booktabs LaTeX table: rows = methods, columns = ``table_metrics``.

    For each run, freeze the step where ``selection_metric`` peaks (within
    ``style.step_ceiling``) and read every column metric there; report
    ``mean +/- style.ci_z * SEM`` across a method's runs. The best entry per column
    is bolded (max if ``higher_is_better`` else min). Each ``table_metrics`` dict:
    ``tag``, ``header``, ``higher_is_better``, ``unit`` (LaTeX, escape ``%``),
    ``scale``, ``decimals`` (defaults to ``default_decimals``).

    Two optional per-column keys handle target-relative metrics:

    * ``offset`` (default ``0``): subtracted from each run's value (RAW units, before
      ``scale``), so a metric with a goal reads as an error centered on ``0``.
    * ``rms`` (default ``False``): report the root-mean-square error across a group's runs
      instead of the signed mean, so per-run over/undershoots don't cancel. Pair with
      ``higher_is_better=False`` (smaller error is better).
    """
    sc = style.step_ceiling
    # Key by column identity, not tag, so the SAME tag may appear in two columns with
    # different offset/rms options (e.g. raw force AND target-relative RMS force error).
    stats = {id(s): dl.best_point_stats(data, selection_metric, s["tag"], style.ci_z, sc,
                                        offset=s.get("offset", 0.0), rms=s.get("rms", False))
             for s in table_metrics}

    best_group = {}
    for s in table_metrics:
        col = stats[id(s)]
        if col:
            best_group[id(s)] = (max if s["higher_is_better"] else min)(
                col, key=lambda g: col[g][0])

    def header(s):
        arrow = "\\uparrow" if s["higher_is_better"] else "\\downarrow"
        unit = s.get("unit", "")
        unit = f" ({unit})" if unit else ""
        return f"{s['header']}{unit} ${arrow}$"

    def fmt(mean, ci, s, bold):
        scale = s.get("scale", 1)
        d = s.get("decimals", default_decimals)
        body = f"{mean * scale:.{d}f} \\pm {ci * scale:.{d}f}"
        body = f"\\mathbf{{{body}}}" if bold else body
        return f"${body}$"

    lines = [
        "% Requires \\usepackage{booktabs} in your preamble.",
        "\\begin{tabular}{l" + "c" * len(table_metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(header(s) for s in table_metrics) + " \\\\",
        "\\midrule",
    ]
    for group in table_groups:
        cells = []
        for s in table_metrics:
            col = stats[id(s)]
            if group not in col:
                cells.append("--")
                continue
            mean, ci = col[group]
            cells.append(fmt(mean, ci, s, best_group.get(id(s)) == group))
        lines.append(style.name(group) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _plot_unit(u: str) -> str:
    """Convert a LaTeX ``unit`` (as written for the LaTeX table) to matplotlib-friendly text.

    Any degree spelling (``\\circ``, ``\\degree``, ``\\textdegree``, ``\\deg`` -- with or
    without ``^{...}`` / ``$...$`` scaffolding) becomes the Unicode degree glyph ``°``,
    which renders in matplotlib's default font with no mathtext. ``\\%`` becomes ``%``. Any
    other string (including genuine mathtext like ``m/s$^2$``) passes through unchanged.
    """
    if not u:
        return ""
    if any(tok in u for tok in ("\\circ", "\\degree", "\\textdegree", "\\deg")):
        return "°"
    return u.replace("\\%", "%")


def plot_metric_bars(data: dict, metric_specs: list, groups: list, style: PlotStyle,
                     selection_metric: str, ncols: int = 2, figsize_per=(4.2, 3.4),
                     suptitle: str | None = None):
    """Grid of bar charts (one subplot per metric spec): x = methods, y = the best-point value.

    Each bar's height is the value :func:`data_loader.best_point_stats` reduces for that group
    (the RMS error vs ``offset`` when the spec sets ``rms=True``, else the mean signed value),
    scaled by ``scale``; the error bar is ``style.ci_z * SEM`` (a 95% CI at ``ci_z = 1.96``).

    ``metric_specs`` entries are the SAME table-metrics-style dicts
    :func:`build_latex_table` reads (``tag``, ``header``, ``unit``, ``scale``, ``offset``,
    ``rms``), so a bar chart and the summary table stay consistent from one source of truth.
    ``groups`` is the method order (e.g. from :func:`resolve_row_order`); bars are colored by
    ``style.color`` and tagged with the display name (selectable in Inkscape). Returns the Figure.
    """
    n = len(metric_specs)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    labels = [style.name(g) for g in groups]
    colors = [style.color(g, i) for i, g in enumerate(groups)]
    xs = list(range(len(groups)))
    for k, s in enumerate(metric_specs):
        ax = axes[k // ncols][k % ncols]
        col = dl.best_point_stats(data, selection_metric, s["tag"], style.ci_z,
                                  style.step_ceiling,
                                  offset=s.get("offset", 0.0), rms=s.get("rms", False))
        scale = s.get("scale", 1)
        heights = [col[g][0] * scale if g in col else np.nan for g in groups]
        errs = [col[g][1] * scale if g in col else 0.0 for g in groups]
        bars = ax.bar(xs, heights, yerr=errs, capsize=4, color=colors,
                      edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.0})
        for b, g in zip(bars, groups):
            b.set_gid(style.name(g).replace(" ", "_"))
        unit = _plot_unit(s.get("unit", ""))
        ax.set_ylabel(s["header"] + (f" ({unit})" if unit else ""))
        # Arrow shows which direction is GOOD (same convention as the LaTeX table header).
        arrow = {True: " ↑", False: " ↓"}.get(s.get("higher_is_better"), "")
        ax.set_title(s["header"] + arrow)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.02)
    for k in range(n, nrows * ncols):  # hide any unused axes in the grid
        axes[k // ncols][k % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def resolve_row_order(data: dict, row_order: list, style: PlotStyle) -> list:
    """Resolve ``row_order`` (display names OR raw folders) to loaded group keys.

    Keeps only loaded methods, preserving the requested order; warns on entries
    that don't resolve. Used for the table's row order.
    """
    name_to_group = {style.name(g): g for g in data}
    out = []
    for entry in row_order:
        if entry in data:
            out.append(entry)
        elif entry in name_to_group:
            out.append(name_to_group[entry])
        else:
            print(f"[row-order] not a loaded method, skipping: {entry!r}")
    return out


# --------------------------------------------------------------------------- #
# Per-phase plots (stiffness / rotation-frame analysis)
# --------------------------------------------------------------------------- #
def _phase_xy(summary_for_group: dict, phases: list):
    """(x indices, means, cis) for a group's per-phase summary, in ``phases`` order."""
    xs, means, cis = [], [], []
    for i, ph in enumerate(phases):
        if ph in summary_for_group:
            m, ci, _ = summary_for_group[ph]
            xs.append(i)
            means.append(m)
            cis.append(ci)
    return np.array(xs), np.array(means), np.array(cis)


def plot_phase_trajectories(data: dict, specs: list, phases: list, style: PlotStyle,
                            groups=None, reduce="mean_tail", selection_metric=None,
                            figsize_per=(3.2, 2.6), suptitle=None):
    """Small-multiples grid of per-phase metric trajectories (one row per metric).

    ``specs`` is a list of ``{tag_template, label}`` dicts; ``tag_template`` carries
    a ``{phase}`` placeholder (e.g. ``"RotationFrame_{phase}/anisotropy_ratio_mean"``).
    Each subplot shows the metric across ``phases`` (free_space -> search ->
    insertion) with one mean+CI line per controller, so you read off *how each
    controller adapts its stiffness as it engages*. Returns the Figure.
    """
    groups = groups if groups is not None else list(data.keys())
    nrows = len(specs)
    fig, axes = plt.subplots(nrows, 1, figsize=(figsize_per[0] * 2.2,
                                                figsize_per[1] * nrows), squeeze=False)
    for r, spec in enumerate(specs):
        ax = axes[r][0]
        summ = dl.phase_summary(data, spec["tag_template"], phases, reduce=reduce,
                                ci_z=style.ci_z, step_ceiling=style.step_ceiling,
                                selection_metric=selection_metric)
        for idx, group in enumerate(groups):
            if group not in summ:
                continue
            xs, means, cis = _phase_xy(summ[group], phases)
            if xs.size == 0:
                continue
            color = style.color(group, idx)
            ax.plot(xs, means, "-o", color=color, label=style.name(group), markersize=4)
            ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.18, linewidth=0)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases)
        ax.set_ylabel(spec["label"])
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1][0].set_xlabel("Task phase")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_axial_lateral_scatter(data: dict, phases: list, style: PlotStyle,
                               axial_template="RotationFrame_{phase}/k_axial_mean",
                               lateral_template="RotationFrame_{phase}/k_lateral_mean",
                               groups=None, reduce="mean_tail", selection_metric=None,
                               ax=None, annotate=True):
    """Scatter k_axial vs k_lateral, one marker per (group, phase), arrows by phase.

    Each controller traces a path through (k_lateral, k_axial) space as it goes
    free_space -> search -> insertion. The ``y = x`` line is isotropy: points below
    it are compliant-along-the-peg / stiff-laterally (the usual insertion
    strategy). Returns the Axes.
    """
    groups = groups if groups is not None else list(data.keys())
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax_summ = dl.phase_summary(data, axial_template, phases, reduce=reduce,
                               ci_z=style.ci_z, step_ceiling=style.step_ceiling,
                               selection_metric=selection_metric)
    lat_summ = dl.phase_summary(data, lateral_template, phases, reduce=reduce,
                                ci_z=style.ci_z, step_ceiling=style.step_ceiling,
                                selection_metric=selection_metric)

    allv = []
    for idx, group in enumerate(groups):
        if group not in ax_summ or group not in lat_summ:
            continue
        color = style.color(group, idx)
        xs, ys = [], []
        for ph in phases:
            if ph in ax_summ[group] and ph in lat_summ[group]:
                xs.append(lat_summ[group][ph][0])
                ys.append(ax_summ[group][ph][0])
        if not xs:
            continue
        allv += xs + ys
        ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1)
        ax.scatter(xs, ys, color=color, s=[30 + 50 * i for i in range(len(xs))],
                   label=style.name(group), zorder=3)
        if annotate:
            for i, ph in enumerate(phases[:len(xs)]):
                ax.annotate(ph[0].upper(), (xs[i], ys[i]), fontsize=7,
                            ha="center", va="center", color="white", zorder=4)

    if allv:
        lo, hi = min(allv), max(allv)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="isotropic (k$_\\parallel$=k$_\\perp$)")
    ax.set_xlabel("k$_\\perp$  (lateral / in-plane stiffness)")
    ax.set_ylabel("k$_\\parallel$  (axial / along-insertion stiffness)")
    ax.set_title("Stiffness anisotropy by phase\n(marker size grows free→search→insertion)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_stiffness_ellipses(data: dict, phases: list, style: PlotStyle,
                            groups=None, reduce="mean_tail", selection_metric=None,
                            axial_template="RotationFrame_{phase}/k_axial_mean",
                            lateral_template="RotationFrame_{phase}/k_lateral_mean",
                            zangle_template="RotationFrame_{phase}/z_angle",
                            figsize_per=(2.3, 2.3),
                            color_by_axial=False, ghost=False, ghost_axial=None,
                            scale_mode="max", cmap="viridis"):
    """Cartoon of the translational stiffness ellipse vs. the peg axis, per phase.

    Grid: rows = controller, cols = phase. The peg axis is drawn vertical; each
    ellipse uses k_axial (semi-axis along peg), k_lateral (perpendicular), and is
    tilted by ``z_angle`` (deg between the policy's stiffness-frame z and the peg
    z; 0 for modes that don't emit it). Turns the abstract numbers into a shape:
    round = isotropic, squished-along-peg = compliant insertion, tilted = the
    off-axis expressiveness GAS adds. Returns the Figure.

    Three cues (composable) make axial-stiffness variation easier to read:

    * ``color_by_axial``: fill each ellipse by its *true* k_axial via ``cmap``
      with a shared colorbar -- separates panels even when shapes look alike.
    * ``ghost``: draw a reference ellipse (dashed grey) in every panel as a fixed
      anchor, so each ellipse's deviation from the reference is salient. By
      default the reference is the grand-mean ellipse; pass ``ghost_axial`` (a
      stiffness value in true units) to instead anchor on a fixed circle of that
      radius -- mapped through the same display scale and identical on every panel.
    * ``scale_mode``: ``"max"`` normalizes both semi-axes by the global largest
      value (true geometry, but small ellipses cluster near zero). ``"range"``
      affine-maps the global [min, max] over ALL semi-axis values (axial AND
      lateral pooled) onto [0.25, 1.0] -- the same transform on both axes, so it
      stays *relatively* geometric while stretching the dynamic range so axial
      differences between panels pop.
    """
    groups = groups if groups is not None else list(data.keys())
    a = dl.phase_summary(data, axial_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    l = dl.phase_summary(data, lateral_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    z = dl.phase_summary(data, zangle_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)

    # Pool all semi-axis magnitudes (axial + lateral) to fix a shared display scale.
    axial_vals = [a[g][p][0] for g in a for p in a[g]]
    vals = axial_vals + [l[g][p][0] for g in l for p in l[g]]
    range_note = None
    if scale_mode == "range" and vals:
        # Affine remap of global [min, max] -> [0.25, 1.0], applied to both axes.
        gmin, gmax = min(vals), max(vals)
        span = (gmax - gmin) or 1.0
        to_disp = lambda v: 0.25 + 0.75 * (v - gmin) / span
        range_note = (f"Range-stretched scale: semi-axis 0.25 = {gmin:.0f}, "
                      f"1.0 = {gmax:.0f} (stiffness units, global min/max over all axes)")
    else:
        scale = max(vals) if vals else 1.0
        to_disp = lambda v: v / scale

    # Optional colormap encoding of the true (absolute) axial stiffness.
    sm = None
    if color_by_axial and axial_vals:
        norm = mpl.colors.Normalize(vmin=min(axial_vals), vmax=max(axial_vals))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    # Reference ("ghost") ellipse parameters (axial, lateral, angle) in true units.
    # ghost_axial pins a fixed circle of that stiffness; else use the grand mean.
    ghost_params = None
    if ghost and ghost_axial is not None:
        ghost_params = (float(ghost_axial), float(ghost_axial), 0.0)
    elif ghost and axial_vals:
        z_vals = [z[g][p][0] for g in z for p in z[g]]
        ghost_params = (float(np.mean(axial_vals)),
                        float(np.mean([l[g][p][0] for g in l for p in l[g]])),
                        float(np.mean(z_vals)) if z_vals else 0.0)

    t = np.linspace(0, 2 * np.pi, 100)

    def _ellipse_xy(ka, kl, ang):
        # Semi-axis ka along the (tilted) peg direction, kl across it.
        ex, ey = kl * np.cos(t), ka * np.sin(t)
        return (ex * np.cos(ang) - ey * np.sin(ang),
                ex * np.sin(ang) + ey * np.cos(ang))

    nrows, ncols = len(groups), len(phases)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, constrained_layout=color_by_axial)
    for r, group in enumerate(groups):
        outline = style.color(group, r)
        for c, ph in enumerate(phases):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            # Peg axis reference (vertical).
            ax.plot([0, 0], [-1.1, 1.1], color="0.6", linestyle=":", linewidth=1)
            if ghost_params is not None:
                gx, gy = _ellipse_xy(to_disp(ghost_params[0]), to_disp(ghost_params[1]),
                                     np.deg2rad(ghost_params[2]))
                ax.plot(gx, gy, color="0.7", linestyle="--", linewidth=1.0, zorder=1)
            if group in a and ph in a[group] and group in l and ph in l[group]:
                ka = to_disp(a[group][ph][0])
                kl = to_disp(l[group][ph][0])
                ang = np.deg2rad(z[group][ph][0]) if (group in z and ph in z[group]) else 0.0
                xr, yr = _ellipse_xy(ka, kl, ang)
                fill = sm.to_rgba(a[group][ph][0]) if sm is not None else outline
                ax.fill(xr, yr, color=fill, alpha=0.6 if sm is not None else 0.35, zorder=2)
                ax.plot(xr, yr, color=outline, linewidth=1.2, zorder=3)
            if r == 0:
                ax.set_title(ph, fontsize=9)
            if c == 0:
                ax.set_ylabel(style.name(group), fontsize=9)
    if sm is not None:
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                     label="k_axial (true units)")
    fig.suptitle("Translational stiffness ellipse vs. peg axis (dotted = insertion axis)")
    if not color_by_axial:
        fig.tight_layout()
    if range_note:
        # Footnote spelling out the display scale; reserve a strip at the bottom.
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.06))
        fig.text(0.5, 0.01, range_note, ha="center", va="bottom", fontsize=8,
                 color="0.3")
    return fig


def plot_stiffness_ellipses_coupling(data: dict, phases: list, style: PlotStyle,
                                     groups=None, reduce="mean_tail", selection_metric=None,
                                     axial_template="RotationFrame_{phase}/k_axial_mean",
                                     lateral_template="RotationFrame_{phase}/k_lateral_mean",
                                     cross_template="RotationFrame_{phase}/cross_coupling_mean",
                                     socket_template="RotationFrame_{phase}/peg_socket_angle_mean",
                                     figsize_per=(2.3, 2.3),
                                     color_by_axial=False, ghost=False, ghost_axial=None,
                                     scale_mode="max", cmap="viridis", socket_color=None):
    """Stiffness-ellipse cartoon whose TILT is recovered from ``cross_coupling``.

    Identical layout, scaling, colour, and ghost cues to ``plot_stiffness_ellipses``,
    but it does NOT read ``z_angle`` (which is logged for the rotated modes only, so
    that version forces every baseline upright). Instead it reconstructs, per phase,
    the 2x2 block of the translational stiffness K in the plane spanned by the peg
    axis z_hat and the cross-coupling direction. In the orthonormal basis
    ``(x = cross dir, y = peg axis)`` that block is exactly the logged scalars::

        M = [[ d,  b ],     a = k_axial       = z_hatᵀ K z_hat                 (along peg)
             [ b,  a ]]     b = cross_coupling = ‖(I − z_hat z_hatᵀ) K z_hat‖  (EXACT off-diagonal)
                            d ≈ k_lateral     = (tr K − k_axial)/2            (mean lateral stiffness)

    because ``cross_dirᵀ K z_hat = ‖(I − z_hat z_hatᵀ) K z_hat‖`` identically. The
    principal axis of M is tilted off the peg axis by ``psi = ½·atan2(2b, a − d)`` with
    eigenvalues ``(a+d)/2 ± hypot((a−d)/2, b)`` used as the ellipse semi-axes. So
    ``b = 0`` (peg is a principal axis: GAS-fixed-rot, or VICES once the peg is
    vertical at insertion) reproduces the upright ellipse, while ``b > 0`` (VICES /
    Cholesky off the peg in free/search) now tilts as it should -- for EVERY mode.

    Each panel also draws the principal axis nearest the peg as a solid line in the ellipse
    colour and writes the tilt ``gamma`` (deg) in the wedge between that axis and the
    vertical peg axis, with a small arc. ``gamma = ½·atan2(2·cc, |k_axial − k_lateral|)``
    in ``[0, 45]`` is the rotation of the stiffness eigenframe off the peg/lateral axes:
    it is driven by the coupling ``cc`` (0 when uncoupled), NOT by which diagonal is larger.
    The annotation is SKIPPED for a near-circular ellipse (relative eccentricity < 2%),
    where the principal direction is undefined -- this is what stops the isotropic Fixed
    gains (a == d with cc = float noise ~2e-5, which would otherwise snap the raw angle to
    45 deg) from being drawn as tilted. The ``cross_coupling`` value used is printed in each
    panel's bottom-right corner.

    A second dotted line marks the PHYSICAL mean peg<->socket angle (``socket_template``),
    drawn off the peg axis on the same side as the stiffness tilt so the two offsets are read
    side by side: the solid line is how far the STIFFNESS frame is rotated off the peg, the
    dotted one how far the peg itself sits off the socket. It is drawn in the method's own
    colour (dotted style distinguishes it from the solid principal axis); pass ``socket_color``
    to force a single fixed colour instead. Its angle is labelled in the same colour. Both
    peg<->socket and the stiffness tilt are magnitudes (direction is cosmetic), so they are
    co-drawn to one side purely for comparison.

    Two approximations are baked in and worth reading the cartoon with: ``d`` is taken
    as the logged ``k_lateral`` (the mean of both perpendicular stiffnesses, exact only
    under transverse isotropy), and ``cross_coupling`` is a magnitude (env-averaged), so
    only ``|psi|`` is recovered -- the left/right tilt direction is cosmetic. Returns
    the Figure. See ``plot_stiffness_ellipses`` for the ``color_by_axial`` / ``ghost`` /
    ``scale_mode`` cue semantics, which are unchanged.
    """
    groups = groups if groups is not None else list(data.keys())
    a = dl.phase_summary(data, axial_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    l = dl.phase_summary(data, lateral_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    cc = dl.phase_summary(data, cross_template, phases, reduce=reduce, ci_z=style.ci_z,
                          step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    ps = dl.phase_summary(data, socket_template, phases, reduce=reduce, ci_z=style.ci_z,
                          step_ceiling=style.step_ceiling, selection_metric=selection_metric)

    # Eigen-reconstruct the 2x2 block M=[[d,b],[b,a]] per (group, phase) in TRUE units:
    # the eigenvalue paired with the psi-tilted eigenvector is the larger one (lam_plus),
    # so it is the semi-axis placed along the tilted axis. b=0 collapses to (a along peg,
    # d across) -- i.e. the upright ellipse -- including the a<d swap (psi -> 90deg).
    ell = {}   # group -> phase -> (lam_plus, lam_minus, psi_deg)
    for g in a:
        for p in a[g]:
            if not (g in l and p in l[g]):
                continue
            a_t = a[g][p][0]                                          # k_axial
            d_t = l[g][p][0]                                          # k_lateral (~ d)
            b_t = cc[g][p][0] if (g in cc and p in cc[g]) else 0.0    # cross_coupling
            psi = 0.5 * np.arctan2(2.0 * b_t, a_t - d_t)              # tilt off peg axis
            spread = float(np.hypot((a_t - d_t) / 2.0, b_t))
            mean = (a_t + d_t) / 2.0
            ell.setdefault(g, {})[p] = (max(mean + spread, 0.0),
                                        max(mean - spread, 0.0),
                                        float(np.rad2deg(psi)),
                                        float(b_t))

    # Colour still encodes TRUE k_axial; display scale pools the eigen-semi-axes drawn.
    axial_vals = [a[g][p][0] for g in a for p in a[g]]
    vals = [v for g in ell for p in ell[g] for v in ell[g][p][:2]]
    range_note = None
    if scale_mode == "range" and vals:
        gmin, gmax = min(vals), max(vals)
        span = (gmax - gmin) or 1.0
        to_disp = lambda v: 0.25 + 0.75 * (v - gmin) / span
        range_note = (f"Range-stretched scale: semi-axis 0.25 = {gmin:.0f}, "
                      f"1.0 = {gmax:.0f} (stiffness units, global min/max over eigen-axes)")
    else:
        scale = max(vals) if vals else 1.0
        to_disp = lambda v: v / scale

    sm = None
    if color_by_axial and axial_vals:
        norm = mpl.colors.Normalize(vmin=min(axial_vals), vmax=max(axial_vals))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    # Ghost reference ellipse (major, minor, angle_deg) in true units: fixed circle if
    # ghost_axial is given, else the grand-mean eigen-ellipse.
    ghost_params = None
    if ghost and ghost_axial is not None:
        ghost_params = (float(ghost_axial), float(ghost_axial), 0.0)
    elif ghost and vals:
        majors = [ell[g][p][0] for g in ell for p in ell[g]]
        minors = [ell[g][p][1] for g in ell for p in ell[g]]
        angs = [ell[g][p][2] for g in ell for p in ell[g]]
        ghost_params = (float(np.mean(majors)), float(np.mean(minors)), float(np.mean(angs)))

    t = np.linspace(0, 2 * np.pi, 100)

    def _ellipse_xy(ka, kl, ang):
        # Semi-axis ka along the (psi-tilted) major direction, kl across it.
        ex, ey = kl * np.cos(t), ka * np.sin(t)
        return (ex * np.cos(ang) - ey * np.sin(ang),
                ex * np.sin(ang) + ey * np.cos(ang))

    nrows, ncols = len(groups), len(phases)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, constrained_layout=color_by_axial)
    for r, group in enumerate(groups):
        outline = style.color(group, r)
        for c, ph in enumerate(phases):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            # Peg axis reference (vertical).
            ax.plot([0, 0], [-1.1, 1.1], color="0.6", linestyle=":", linewidth=1)
            if ghost_params is not None:
                gx, gy = _ellipse_xy(to_disp(ghost_params[0]), to_disp(ghost_params[1]),
                                     np.deg2rad(ghost_params[2]))
                ax.plot(gx, gy, color="0.7", linestyle="--", linewidth=1.0, zorder=1)
            if group in ell and ph in ell[group]:
                lam_p, lam_m, psi_deg, b_t = ell[group][ph]
                ka = to_disp(lam_p)                      # major semi-axis (display)
                kl = to_disp(lam_m)                      # minor semi-axis (display)
                psi = np.deg2rad(psi_deg)                # major-axis orientation (signed)
                xr, yr = _ellipse_xy(ka, kl, psi)
                fill = sm.to_rgba(a[group][ph][0]) if sm is not None else outline
                ax.fill(xr, yr, color=fill, alpha=0.6 if sm is not None else 0.35, zorder=2)
                ax.plot(xr, yr, color=outline, linewidth=1.2, zorder=3)
                # The cross-coupling actually used for this panel (true units), bottom-right.
                ax.text(1.12, -1.12, f"cc={b_t:.1f}", ha="right", va="bottom",
                        fontsize=7, color="0.4", zorder=5)
                # Side the stiffness tilt is drawn on (major-near => left, minor-near =>
                # right); the socket line shares it so the two offsets compare directly.
                side = 1.0 if psi_deg <= 45.0 else -1.0
                # Physical mean peg<->socket angle: a dotted reference axis. Match the
                # method's colour by default (dotted vs the solid principal axis keeps them
                # distinct); socket_color overrides with a single fixed colour.
                sc = outline if socket_color is None else socket_color
                if group in ps and ph in ps[group]:
                    sock_deg = ps[group][ph][0]
                    sphi = side * np.deg2rad(sock_deg)
                    sux, suy = -np.sin(sphi), np.cos(sphi)
                    ax.plot([-1.08 * sux, 1.08 * sux], [-1.08 * suy, 1.08 * suy],
                            color=sc, linestyle=":", linewidth=1.3, zorder=4)
                    ax.text(0.82 * sux, 0.82 * suy, f"{sock_deg:.1f}°", ha="center",
                            va="center", fontsize=7.5, color=sc, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                # Principal-axis tilt = rotation of the stiffness eigenframe off the peg,
                # gamma = ½·atan2(2·cc, |k_axial-k_lateral|) in [0,45], driven by cc (0 when
                # uncoupled). Skip it for a near-circular ellipse, where the principal
                # DIRECTION is undefined and the raw angle is meaningless (the isotropic
                # Fixed gains: cc is float noise ~2e-5 but a==d, so it would snap to 45deg).
                ecc = (lam_p - lam_m) / (lam_p + lam_m) if (lam_p + lam_m) > 0 else 0.0
                if ecc >= 0.02:
                    # Near-peg eigenvector (closest to vertical) and its length; phi is its
                    # SIGNED angle off the peg axis so the line lies on the drawn ellipse.
                    if psi_deg <= 45.0:
                        phi, near_len = psi, lam_p        # major axis is the near-peg one
                    else:
                        phi, near_len = psi - np.pi / 2.0, lam_m  # minor axis is near-peg
                    gamma_deg = abs(np.rad2deg(phi))
                    kn = to_disp(near_len)
                    ux, uy = -np.sin(phi), np.cos(phi)
                    ax.plot([-kn * ux, kn * ux], [-kn * uy, kn * uy],
                            color=outline, linewidth=1.7, zorder=4)
                    # Thin arc from the peg axis to that principal axis + the value on the
                    # wedge bisector (faint box so it stays legible over the fill).
                    arc_r, txt_r = 0.34, 0.52
                    aa = np.linspace(0.0, phi, 24)
                    ax.plot(-np.sin(aa) * arc_r, np.cos(aa) * arc_r,
                            color=outline, linewidth=1.0, zorder=4)
                    ax.text(-np.sin(phi / 2.0) * txt_r, np.cos(phi / 2.0) * txt_r,
                            f"{gamma_deg:.1f}°", ha="center", va="center", fontsize=8,
                            color="0.15", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
            if r == 0:
                ax.set_title(ph, fontsize=9)
            if c == 0:
                ax.set_ylabel(style.name(group), fontsize=9)
    if sm is not None:
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                     label="k_axial (true units)")
    fig.suptitle("Translational stiffness ellipse vs. peg axis (tilt from cross-coupling)")
    if not color_by_axial:
        fig.tight_layout()
    if range_note:
        # Footnote spelling out the display scale; reserve a strip at the bottom.
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.06))
        fig.text(0.5, 0.01, range_note, ha="center", va="bottom", fontsize=8,
                 color="0.3")
    return fig


def plot_phase_heatmap(data: dict, tag_template: str, phases: list, style: PlotStyle,
                       groups=None, reduce="mean_tail", selection_metric=None,
                       cmap="viridis", title=None, cbar_label=None, ax=None,
                       fmt="{:.2f}"):
    """Heatmap of one per-phase metric: rows = controller, cols = phase.

    Cell color/text = mean over runs of ``tag_template`` (with ``{phase}``) at the
    chosen ``reduce``. The compact at-a-glance summary -- who is compliant where.
    Returns the Axes.
    """
    groups = groups if groups is not None else list(data.keys())
    summ = dl.phase_summary(data, tag_template, phases, reduce=reduce, ci_z=style.ci_z,
                            step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    groups = [g for g in groups if g in summ]
    mat = np.full((len(groups), len(phases)), np.nan)
    for r, g in enumerate(groups):
        for c, ph in enumerate(phases):
            if ph in summ[g]:
                mat[r, c] = summ[g][ph][0]

    if ax is None:
        _, ax = plt.subplots(figsize=(1.4 * len(phases) + 2, 0.7 * len(groups) + 1.5))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([style.name(g) for g in groups])
    for r in range(len(groups)):
        for c in range(len(phases)):
            if not np.isnan(mat[r, c]):
                ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center",
                        color="white", fontsize=8)
    cbar = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    return ax


def plot_metric_vs_success(data: dict, metric_tag: str, success_tag: str,
                           style: PlotStyle, groups=None, metric_reduce="mean_tail",
                           success_reduce="max", ax=None, xlabel=None, ylabel=None,
                           title=None):
    """Scatter a per-run stiffness/rotation metric against per-run success.

    One point per run (colored by controller); directly tests whether more
    off-diagonal / anisotropic expressiveness correlates with insertion success.
    A Pearson r over all runs is annotated. Returns the Axes.
    """
    groups = groups if groups is not None else list(data.keys())
    sc = style.step_ceiling
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    xs_all, ys_all = [], []
    for idx, group in enumerate(groups):
        if group not in data:
            continue
        color = style.color(group, idx)
        xs, ys = [], []
        for run in data[group]:
            x = dl.reduce_run_value(run, metric_tag, metric_reduce, sc)
            y = dl.reduce_run_value(run, success_tag, success_reduce, sc)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            ax.scatter(xs, ys, color=color, label=style.name(group), s=45, zorder=3)
            xs_all += xs
            ys_all += ys

    if len(xs_all) > 2:
        r = np.corrcoef(xs_all, ys_all)[0, 1]
        ax.annotate(f"Pearson r = {r:.2f}  (n={len(xs_all)})", xy=(0.03, 0.96),
                    xycoords="axes fraction", va="top", fontsize=9)
    ax.set_xlabel(xlabel or metric_tag)
    ax.set_ylabel(ylabel or success_tag)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# --------------------------------------------------------------------------- #
# Surface-frame stiffness plots (flat-surface-follow analysis, NO task phases)
# --------------------------------------------------------------------------- #
# The flat-surface task has no free->search->insertion phases, and the natural
# reference frame is the SURFACE frame [along-track, cross-track, normal] rather
# than the peg axis. The controller logs the applied stiffness resolved onto those
# axes as ``Impedance_Stiffness/k_along_track_mean`` / ``k_cross_track_mean`` /
# ``k_normal_mean`` (diag(R_surfᵀ K R_surf); see ctrl_action_interface
# ._log_stiffness_frame_metrics). These helpers are the phaseless surface-frame
# analogues of plot_axial_lateral_scatter / plot_stiffness_ellipses above.
SURFACE_STIFFNESS_TAGS = {
    "along_track": "Impedance_Stiffness/k_along_track_mean",
    "cross_track": "Impedance_Stiffness/k_cross_track_mean",
    "normal":      "Impedance_Stiffness/k_normal_mean",
}

# Principal stiffnesses (eigenvalues of the applied position-K block, frame-free), logged
# as _k_native. With the surface-frame DIAGONAL above they recover the stiffness-oval TILT
# without any new logging: M = Q diag(principal) Qᵀ with diag(M) = (k_along, k_cross,
# k_normal) known, so the (along, normal) block's rotation off the surface axes is fixed by
# the two block eigenvalues and the two diagonal entries (see plot_surface_stiffness_ellipses).
PRINCIPAL_STIFFNESS_TAGS = {
    "x": "Impedance_Stiffness/principal_x_mean",
    "y": "Impedance_Stiffness/principal_y_mean",
    "z": "Impedance_Stiffness/principal_z_mean",
}


def plot_surface_stiffness_scatter(data: dict, style: PlotStyle,
                                   x_tag=SURFACE_STIFFNESS_TAGS["along_track"],
                                   y_tag=SURFACE_STIFFNESS_TAGS["normal"],
                                   xlabel="k$_\\parallel$  (along-track stiffness)",
                                   ylabel="k$_n$  (normal / into-surface stiffness)",
                                   groups=None, reduce="mean_tail", selection_metric=None,
                                   ax=None):
    """Scatter of two surface-frame stiffnesses, one marker (mean ± CI) per controller.

    Defaults to k_normal (y) vs k_along_track (x) -- the phaseless, surface-frame
    analogue of :func:`plot_axial_lateral_scatter`. Each controller is reduced to a
    single end-of-training point (``reduce``) with CI whiskers across its runs. The
    ``y = x`` line is isotropy: points above it press harder INTO the surface than
    they resist ALONG it (the usual surface-following strategy), below it the
    reverse. Returns the Axes.
    """
    groups = groups if groups is not None else list(data.keys())
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    xs = dl.summarize_tag(data, x_tag, reduce=reduce, ci_z=style.ci_z,
                          step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    ys = dl.summarize_tag(data, y_tag, reduce=reduce, ci_z=style.ci_z,
                          step_ceiling=style.step_ceiling, selection_metric=selection_metric)

    allv = []
    for idx, group in enumerate(groups):
        if group not in xs or group not in ys:
            continue
        xm, xci, _ = xs[group]
        ym, yci, _ = ys[group]
        color = style.color(group, idx)
        ax.errorbar(xm, ym, xerr=xci, yerr=yci, fmt="o", color=color, markersize=7,
                    capsize=3, elinewidth=1, label=style.name(group), zorder=3)
        allv += [xm, ym]

    if allv:
        lo, hi = min(allv), max(allv)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1,
                label="isotropic (k$_n$=k$_\\parallel$)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Surface-frame stiffness by controller\n(marker = mean, whiskers = 95% CI across runs)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_surface_stiffness_ellipses(data: dict, style: PlotStyle,
                                    normal_tag=SURFACE_STIFFNESS_TAGS["normal"],
                                    along_tag=SURFACE_STIFFNESS_TAGS["along_track"],
                                    cross_tag=SURFACE_STIFFNESS_TAGS["cross_track"],
                                    principal_tags=PRINCIPAL_STIFFNESS_TAGS,
                                    tilt_mode="inplane",
                                    groups=None, reduce="mean_tail", selection_metric=None,
                                    ncols=None, figsize_per=(2.3, 2.3),
                                    color_by_normal=False, ghost=False, ghost_normal=None,
                                    scale_mode="max", cmap="viridis"):
    """Cartoon of the stiffness "oval" in the surface plane, one panel per controller.

    Each oval shows two of the three surface directions -- the SURFACE NORMAL on the
    vertical axis and the ALONG-TRACK direction horizontally (cross-track is dropped) --
    so the shape reads as a clean 2-D profile of "how hard it presses down vs. how hard
    it resists along the path". Phaseless, surface-frame analogue of
    :func:`plot_stiffness_ellipses_coupling`. Returns the Figure.

    TILT (reconstructed, no extra logging). No method authors K in the surface frame
    (baselines are diagonal in the EEF frame, rotated modes in their own interaction
    frame), so the applied stiffness resolved onto [along, cross, normal] is generally NOT
    diagonal -- its eigenframe is rotated off the surface axes. That rotation is recovered
    from quantities ALREADY logged: the eigenvalues ``principal_{x,y,z}`` of K and the
    surface-frame diagonal ``(k_along, k_cross, k_normal) = diag(Rₛᵤᵣfᵀ K Rₛᵤᵣf)``. Writing
    ``M = Q·diag(principal)·Qᵀ`` with ``diag(M)`` known, the eigenvalue nearest ``k_cross``
    is taken as the cross-track (shared) axis and the other two ``λ₊ ≥ λ₋`` are the
    (along, normal) block's eigenvalues, so the block is fixed up to a sign by ::

        cos(2·psi) = (k_normal − k_along) / (λ₊ − λ₋)          psi ∈ [0°, 90°]
        semi-axes  = λ₊ (major, along the psi-tilted axis), λ₋ (minor)

    ``psi`` is the tilt of the stiffer principal axis off the surface NORMAL. This is exact
    when the eigenframe rotation lies in the along-normal plane (a pitch about cross-track,
    the physically dominant case: ``k_cross`` then equals one eigenvalue). Only ``|psi|`` is
    recovered (the left/right sign is cosmetic) -- the same trade the peg-axis coupling
    version makes. ``λ₊ = λ₋`` (isotropic) collapses to a circle; ``psi = 0`` reproduces the
    upright oval (``k_normal`` vertical, ``k_along`` horizontal), so surface-aligned
    baselines stay upright and only genuine off-surface rotation tilts.

    ``tilt_mode`` selects which rotation the tilt shows (see the reconstruction comment for
    the exact formulas): ``"inplane"`` (default) is the full (along, normal) eigen-tilt
    described above; ``"zaxis"`` shows ONLY the polar tilt of the authored normal axis
    (principal_z) off the true surface normal, ignoring any tangential-plane spin -- a
    single-DOF Rayleigh inversion that is exact under tangential isotropy and needs only
    k_normal + the eigenvalues.

    Cues (composable), matching the peg version:

    * ``color_by_normal``: fill each oval by its *true* k_normal via ``cmap`` + shared
      colorbar, so panels separate by colour even when the shapes look alike.
    * ``ghost``: draw a reference oval (dashed grey) in every panel as a fixed anchor.
      Defaults to the grand-mean (upright) oval; pass ``ghost_normal`` (true stiffness
      units) to anchor on a fixed circle of that radius instead.
    * ``scale_mode``: ``"max"`` normalizes the semi-axes by the global largest value
      (true geometry); ``"range"`` affine-maps the global [min, max] over ALL semi-axis
      values onto [0.25, 1.0] so small differences between panels pop.
    """
    groups = groups if groups is not None else list(data.keys())
    _sm = lambda tag: dl.summarize_tag(data, tag, reduce=reduce, ci_z=style.ci_z,
                                       step_ceiling=style.step_ceiling,
                                       selection_metric=selection_metric)
    n = _sm(normal_tag)
    al = _sm(along_tag)
    cr = _sm(cross_tag)
    px, py, pz = _sm(principal_tags["x"]), _sm(principal_tags["y"]), _sm(principal_tags["z"])

    # Reconstruct each oval as (semi_v, semi_h, angle_deg) in TRUE units, per tilt_mode. semi_v
    # is the semi-axis drawn on the (angle-tilted) near-vertical axis, semi_h across it; angle is
    # measured off the surface normal (vertical). angle=0 => semi_v vertical, semi_h horizontal.
    # Groups missing the principal tags fall back to the upright oval. tilted_any flags whether any
    # panel could actually tilt (principals present) so the caption can flag the fallback.
    #
    # tilt_mode="inplane": the FULL (along, normal) eigen-tilt. The eigenvalue closest to k_cross
    #   is the shared cross-track axis; the other two are this block's eigenvalues lam+ >= lam-,
    #   and the major-axis tilt is fixed by the diagonal split
    #       cos(2*psi) = (k_normal - k_along)/(lam+ - lam-).
    #   Assumes the eigenframe rotation lies in the along-normal plane (a pitch about cross-track).
    #
    # tilt_mode="zaxis": ONLY the polar tilt of the authored normal axis (principal_z) off the true
    #   surface normal, ignoring any tangential-plane spin. k_normal = ez^T K ez is a Rayleigh
    #   quotient, so with the tangential eigenvalues taken equal (lam_t = mean(principal_x,
    #   principal_y)) it inverts EXACTLY to
    #       cos^2(theta) = (k_normal - lam_t)/(principal_z - lam_t).
    #   semi_v = principal_z (the normal-axis stiffness, leaning theta off vertical), semi_h = lam_t.
    #   Only needs k_normal + the eigenvalues (not k_along), and drops the along-normal-plane
    #   assumption. Ill-conditioned when principal_z ~ lam_t (normal barely distinguishable from
    #   tangential) -> theta undefined, drawn upright.
    ell = {}          # group -> (semi_v, semi_h, angle_deg, has_principals)
    tilted_any = False
    for g in n:
        if g not in al:
            continue
        k_n, k_a = n[g][0], al[g][0]
        have_p = all(g in s for s in (cr, px, py, pz))
        if have_p and tilt_mode == "zaxis":
            tilted_any = True
            lam_n = pz[g][0]                                          # authored normal-axis stiffness
            lam_t = 0.5 * (px[g][0] + py[g][0])                       # tangential mean (transverse iso.)
            denom = lam_n - lam_t
            if abs(denom) > 0.02 * max(lam_n, lam_t, 1.0):
                cos2 = np.clip((k_n - lam_t) / denom, 0.0, 1.0)
                angle = float(np.rad2deg(np.arccos(np.sqrt(cos2))))  # [0, 90], normal axis off vertical
            else:
                angle = 0.0                                          # ill-conditioned -> upright
            semi_v, semi_h = lam_n, lam_t                            # normal-axis vertical (tilted), tangent across
        elif have_p:                                                 # tilt_mode == "inplane"
            tilted_any = True
            evals = sorted((px[g][0], py[g][0], pz[g][0]))
            j = int(np.argmin([abs(e - cr[g][0]) for e in evals]))   # cross-track (shared) axis
            block = [evals[i] for i in range(3) if i != j]           # (along, normal) eigenvalues
            lam_m, lam_p = block[0], block[1]                        # minor <= major
            denom = lam_p - lam_m
            cos2 = np.clip((k_n - k_a) / denom, -1.0, 1.0) if denom > 1e-9 else 1.0
            angle = float(np.rad2deg(0.5 * np.arccos(cos2)))         # [0, 90], major off normal
            semi_v, semi_h = lam_p, lam_m                            # major vertical (tilted), minor across
        else:
            semi_v, semi_h = k_n, k_a                                # upright fallback (no principals)
            angle = 0.0
        ell[g] = (max(semi_v, 0.0), max(semi_h, 0.0), angle, have_p)

    vals = [v for g in ell for v in ell[g][:2]]                 # pool semi-axes for scaling
    range_note = None
    if scale_mode == "range" and vals:
        gmin, gmax = min(vals), max(vals)
        span = (gmax - gmin) or 1.0
        to_disp = lambda v: 0.25 + 0.75 * (v - gmin) / span
        range_note = (f"Range-stretched scale: semi-axis 0.25 = {gmin:.0f}, "
                      f"1.0 = {gmax:.0f} (stiffness units, global min/max over eigen-axes)")
    else:
        scale = max(vals) if vals else 1.0
        to_disp = lambda v: v / scale

    normal_vals = [n[g][0] for g in n]
    sm = None
    if color_by_normal and normal_vals:
        norm = mpl.colors.Normalize(vmin=min(normal_vals), vmax=max(normal_vals))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))

    # Ghost reference oval (semi-vertical, semi-horizontal, angle_deg): upright by construction.
    ghost_params = None
    if ghost and ghost_normal is not None:
        ghost_params = (float(ghost_normal), float(ghost_normal), 0.0)
    elif ghost and normal_vals:
        ghost_params = (float(np.mean(normal_vals)),
                        float(np.mean([al[g][0] for g in al])), 0.0)

    t = np.linspace(0, 2 * np.pi, 100)

    def _ellipse_xy(semi_v, semi_h, ang):
        # semi_v along the (ang-tilted) vertical major axis, semi_h across it; ang measured
        # from the surface normal (vertical). ang=0 -> semi_v vertical, semi_h horizontal.
        ex, ey = semi_h * np.cos(t), semi_v * np.sin(t)
        return (ex * np.cos(ang) - ey * np.sin(ang),
                ex * np.sin(ang) + ey * np.cos(ang))

    plotted = [g for g in groups if g in ell]
    ncols = ncols if ncols is not None else max(1, len(plotted))
    nrows = max(1, (len(plotted) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False, constrained_layout=color_by_normal)
    flat_axes = axes.ravel()
    for k, ax in enumerate(flat_axes):
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= len(plotted):
            ax.axis("off")
            continue
        group = plotted[k]
        outline = style.color(group, k)
        # Surface normal reference (vertical) + surface plane (horizontal).
        ax.plot([0, 0], [-1.1, 1.1], color="0.6", linestyle=":", linewidth=1)
        ax.plot([-1.1, 1.1], [0, 0], color="0.85", linestyle="-", linewidth=1, zorder=0)
        if ghost_params is not None:
            gx, gy = _ellipse_xy(to_disp(ghost_params[0]), to_disp(ghost_params[1]),
                                 np.deg2rad(ghost_params[2]))
            ax.plot(gx, gy, color="0.7", linestyle="--", linewidth=1.0, zorder=1)
        semi_v, semi_h, angle_deg, has_p = ell[group]
        # The tilt sign is cosmetic (only the magnitude is recovered). A near-circular oval has no
        # defined principal direction, so draw it upright and skip the label.
        big, small = max(semi_v, semi_h), min(semi_v, semi_h)
        near_circular = not (big > 1e-9 and (big - small) / big >= 0.02)
        draw_ang = 0.0 if near_circular else np.deg2rad(angle_deg)
        xr, yr = _ellipse_xy(to_disp(semi_v), to_disp(semi_h), draw_ang)
        fill = sm.to_rgba(n[group][0]) if sm is not None else outline
        ax.fill(xr, yr, color=fill, alpha=0.6 if sm is not None else 0.35, zorder=2)
        ax.plot(xr, yr, color=outline, linewidth=1.2, zorder=3)
        # Tilt annotation (deg off the surface normal): the eigenframe rotation ("inplane") or the
        # normal-axis polar tilt ("zaxis"). Skipped for a near-circular oval (relative eccentricity
        # < 2%), where the direction is undefined -- keeps an isotropic baseline from being labelled
        # with a spurious angle. Fallback (no principals) is drawn upright.
        if has_p and not near_circular:
            ax.text(1.12, 1.12, f"{angle_deg:.0f}°", ha="right", va="top",
                    fontsize=7, color=outline, zorder=5)
        ax.set_title(style.name(group), fontsize=9)
    if sm is not None:
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                     label="k_normal (true units)")
    _tilt_desc = ("normal-axis polar tilt off the surface normal" if tilt_mode == "zaxis"
                  else "eigenframe rotation off the surface axes")
    fig.suptitle("Surface-frame stiffness oval  (vertical = normal, horizontal = along-track; "
                 f"tilt = {_tilt_desc})")
    if not color_by_normal:
        fig.tight_layout()
    notes = [nt for nt in (range_note,
             (None if tilted_any else
              "No principal_{x,y,z} tags in these runs -> ovals drawn upright; the eigenframe "
              "tilt cannot be reconstructed from the surface diagonal alone.")) if nt]
    if notes:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.04 + 0.03 * len(notes)))
        for i, nt in enumerate(notes):
            fig.text(0.5, 0.01 + 0.03 * (len(notes) - 1 - i), nt,
                     ha="center", va="bottom", fontsize=8, color="0.3")
    return fig
