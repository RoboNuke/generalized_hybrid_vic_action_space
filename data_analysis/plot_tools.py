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
    "4_GAS_fixed_rot": "GAS (fixed rot)",
    "5_GAS": "GAS",
}

# Fixed color per group (stable across every plot).
GLUED_ROT_GROUP_COLORS = {
    "1_fixed": "#ff9500",          # orange  -- isotropic baseline
    "2_VICES": "#ff0000",          # red     -- diagonal, axis-aligned
    "3_choleskey": "#1f77b4",      # blue    -- full SPD, fixed frame
    "4_GAS_fixed_rot": "#7107b8",  # purple  -- diagonal in a fixed rotated frame
    "5_GAS": "#1fb426",            # green   -- diagonal + learned rotation
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
    """
    sc = style.step_ceiling
    stats = {s["tag"]: dl.best_point_stats(data, selection_metric, s["tag"], style.ci_z, sc)
             for s in table_metrics}

    best_group = {}
    for s in table_metrics:
        col = stats[s["tag"]]
        if col:
            best_group[s["tag"]] = (max if s["higher_is_better"] else min)(
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
            col = stats[s["tag"]]
            if group not in col:
                cells.append("--")
                continue
            mean, ci = col[group]
            cells.append(fmt(mean, ci, s, best_group.get(s["tag"]) == group))
        lines.append(style.name(group) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


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
                            figsize_per=(2.3, 2.3)):
    """Cartoon of the translational stiffness ellipse vs. the peg axis, per phase.

    Grid: rows = controller, cols = phase. The peg axis is drawn vertical; each
    ellipse uses k_axial (semi-axis along peg), k_lateral (perpendicular), and is
    tilted by ``z_angle`` (deg between the policy's stiffness-frame z and the peg
    z; 0 for modes that don't emit it). Turns the abstract numbers into a shape:
    round = isotropic, squished-along-peg = compliant insertion, tilted = the
    off-axis expressiveness GAS adds. Returns the Figure.
    """
    groups = groups if groups is not None else list(data.keys())
    a = dl.phase_summary(data, axial_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    l = dl.phase_summary(data, lateral_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)
    z = dl.phase_summary(data, zangle_template, phases, reduce=reduce, ci_z=style.ci_z,
                         step_ceiling=style.step_ceiling, selection_metric=selection_metric)

    # Global scale so every ellipse is comparable: normalize by the largest semi-axis.
    scale = 1.0
    vals = [a[g][p][0] for g in a for p in a[g]] + [l[g][p][0] for g in l for p in l[g]]
    if vals:
        scale = max(vals)

    nrows, ncols = len(groups), len(phases)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    t = np.linspace(0, 2 * np.pi, 100)
    for r, group in enumerate(groups):
        color = style.color(group, r)
        for c, ph in enumerate(phases):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            # Peg axis reference (vertical).
            ax.plot([0, 0], [-1.1, 1.1], color="0.6", linestyle=":", linewidth=1)
            if group in a and ph in a[group] and group in l and ph in l[group]:
                ka = a[group][ph][0] / scale
                kl = l[group][ph][0] / scale
                ang = np.deg2rad(z[group][ph][0]) if (group in z and ph in z[group]) else 0.0
                # Ellipse: semi-axis ka along the (tilted) peg direction, kl across it.
                ex = kl * np.cos(t)
                ey = ka * np.sin(t)
                xr = ex * np.cos(ang) - ey * np.sin(ang)
                yr = ex * np.sin(ang) + ey * np.cos(ang)
                ax.fill(xr, yr, color=color, alpha=0.35)
                ax.plot(xr, yr, color=color, linewidth=1.2)
            if r == 0:
                ax.set_title(ph, fontsize=9)
            if c == 0:
                ax.set_ylabel(style.name(group), fontsize=9)
    fig.suptitle("Translational stiffness ellipse vs. peg axis (dotted = insertion axis)")
    fig.tight_layout()
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
