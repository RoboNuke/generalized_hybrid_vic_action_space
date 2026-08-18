"""Central plotting config for the IROS-workshop pitch-sweep notebooks.

Every user-tunable styling constant (fonts, palettes, sizes, label/legend spacing) AND the
recurring figure/function default values live HERE, in one place. ``analysis_utils`` imports
everything from this file (``from plot_config import *``) and reloads it on every
``importlib.reload(analysis_utils)``, so:

    edit plot_config.py  ->  re-run any plot cell  ->  the change shows up everywhere.

Nothing here imports ``analysis_utils`` (or anything heavy), so it stays a clean settings file.
"""

# =========================================================================== #
# Method / grasp-angle order
# =========================================================================== #
# Canonical method display order; anything present but unlisted is appended in sorted order.
METHOD_ORDER = ["fixed_geo", "GAS_geo", "GAS", "VICES"]

# =========================================================================== #
# Palettes & display names
# =========================================================================== #
# Method palette (categorical). Validated colorblind-safe as a set
# (worst adjacent CVD dE 25.0 on a light surface): blue / orange / teal / violet.
METHOD_COLORS = {
    "fixed_geo": "#F1965A",
    "GAS":   "#5193C9",
    "GAS_geo":       "#3cc53c",
    "VICES":     "#df7486",
    "cholesky":     "#b436c5",
    #"fixed_geo": "#2a78d6",   # blue
    #"GAS_geo":   "#eb6834",   # orange
    #"GAS":       "#1baf7a",   # teal
    #"VICES":     "#4a3aa7",   # violet
}
METHOD_NAMES = {
    "fixed_geo": "RA-TFF",
    "GAS_geo":   "RA-GAS",
    "GAS":       "RA-Free",
    "VICES":     "VICES",
    "cholesky":  "Chol"
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

# =========================================================================== #
# Fonts, line/band, grid
# =========================================================================== #
FONT_SUPTITLE = 15
FONT_TITLE = 13
FONT_AXIS_LABEL = 7
FONT_TICK = 7
FONT_LEGEND = 6
FONT_BAR_LABEL = 7

LINE_WIDTH = 2.0        # mean line
BAND_ALPHA = 0.20       # CI shaded region
GRID_ALPHA = 0.30
SHOW_Y_GRID = False     # horizontal (y-axis) gridlines on every plot (False = off)
ERROR_CAPSIZE = 5       # bar-chart error-bar caps

# =========================================================================== #
# Default figure size (inches)
# =========================================================================== #
# Used by every single-axes figure (read at draw time). Multi-panel / combo layouts scale each
# panel to (DEFAULT_FIG_WIDTH / ncols, DEFAULT_FIG_HEIGHT / ncols), so the whole figure stays
# DEFAULT_FIG_WIDTH wide. Figures that pass an explicit figsize keep their own size.
DEFAULT_FIG_WIDTH = 3.5 #8 #3.5
DEFAULT_FIG_HEIGHT = 2.2 #4.8 #2.2     # 3.5 * 4.8/8 -- keeps the previous 8x4.8 aspect ratio

# =========================================================================== #
# Bar value labels (the mean printed on each bar, near its lower error cap)
# =========================================================================== #
# Placement is SCALE-RELATIVE (fractions of the y-axis range) so it looks the same at any y-scale
# -- 0-1, 0-100 %, or 0-1000 reward.
BAR_LABEL_GAP_FRAC = 0.1             # gap between the label and the error cap it hugs (frac of y-range)
BAR_LABEL_MIN_INSIDE_FRAC = 0.15      # min room below the lower cap to keep the label INSIDE the bar;
                                      # a shorter bar puts the label ABOVE it instead (frac of y-range)
DEFAULT_BOLD_LABELS = False           # bold the bar value labels
DEFAULT_SHOW_LABEL_CI = False         # append "±ci" to the label (False = just the mean)
BAR_LABEL_X_OFFSET = -0.2             # horizontal nudge of the label, in BAR WIDTHS (0 = centered
                                      # on the bar; +/- shifts it right/left, e.g. off the error bar)

# =========================================================================== #
# VERTICAL grouped-bar geometry & styling (plot_grouped_bars) -- RoboNuke values
# =========================================================================== #
# Geometry: within each x-group the bars span BAR_GROUP_WIDTH of one x-unit; each bar's slot is
# BAR_GROUP_WIDTH / n_bars and it is drawn at BAR_WIDTH if given, else the slot minus BAR_GAP.
BAR_GROUP_WIDTH = 0.85       # fraction of one x-unit spanned by a group (RoboNuke GROUP_WIDTH)
BAR_WIDTH = None             # explicit per-bar thickness; None = auto (slot * (1 - BAR_GAP)), the
                             # RoboNuke default bar_width = GROUP_WIDTH / n_series
BAR_GAP = 0.0                # gap between bars within a group, as a fraction of each bar's slot
                             # (RoboNuke draws bars edge-to-edge: 0.0)
BAR_CAPSIZE = 3              # error-bar cap size (RoboNuke ERROR_CAPSIZE)
BAR_ERROR_COLOR = "black"    # error-bar color (RoboNuke ERROR_COLOR / ecolor)
BAR_ERROR_LINEWIDTH = 0.9    # error-bar line width (elinewidth)
BAR_EDGE_COLOR = "black"     # bar outline color (set None for RoboNuke's edgeless bars)
BAR_EDGE_WIDTH = 0.5         # bar outline width

# HORIZONTAL grouped-bar geometry (plot_grouped_barh) -- explicit thickness + gaps, in y data units.
BARH_BAR_WIDTH = 0.25    # each bar's drawn thickness
BARH_GROUP_GAP = 0.35    # gap between adjacent category groups
BARH_BAR_GAP = 0.02      # gap between bars within a group

# =========================================================================== #
# Legend spacing (the "below"/corner flat layouts). matplotlib defaults are 2.0 / 0.8 / 0.4.
# =========================================================================== #
#   LEGEND_COLUMN_SPACING -- columnspacing BETWEEN entries.
#   LEGEND_HANDLETEXTPAD  -- gap between a handle (color box) and its label.
#   LEGEND_BORDER_PAD     -- borderpad: whitespace inside the legend, BEFORE the first color box
#                            and after the last. Lower it to tighten that leading gap.
#   LEGEND_CORNER_INSET   -- axes-fraction gap between the plot edge and a flat CORNER legend
#                            ("upper left" etc.); 0.0 = flush to the corner.
#   LEGEND_BELOW_OFFSET   -- extra downward gap (axes-height fractions) for the "below" legend.
#                            0.0 already clears the x-axis TITLE; raise it to push the legend
#                            further down, lower it (negative) to tuck it closer to the plot.
LEGEND_COLUMN_SPACING = 0.9
LEGEND_HANDLETEXTPAD = 0.5
LEGEND_BORDER_PAD = -0.0 #1.6
LEGEND_CORNER_INSET = 0.0 #0.055
LEGEND_BELOW_OFFSET = -1.36

# =========================================================================== #
# Per-metric display scaling (e.g. success rate 0-1 -> 0-100 %)
# =========================================================================== #
# Every reduction multiplies that tag's values by METRIC_SCALE, the axis label gets METRIC_UNIT
# appended, and any (min,max) clip/ylim is scaled to match. Add a tag to change it everywhere.
METRIC_SCALE = {"Episode / Success rate": 100.0}   # display multiplier per tag (default 1.0)
METRIC_UNIT = {"Episode / Success rate": "%"}      # axis-label unit per tag (default "")

# =========================================================================== #
# Surface-frame stiffness tags (which logged series the stiffness figures read)
# =========================================================================== #
SURFACE_STIFFNESS_TAGS = {
    "along_track": "Impedance_Stiffness/k_along_track_mean",
    "cross_track": "Impedance_Stiffness/k_cross_track_mean",
    "normal":      "Impedance_Stiffness/k_normal_mean",
}
PRINCIPAL_STIFFNESS_TAGS = {
    "x": "Impedance_Stiffness/principal_x_mean",
    "y": "Impedance_Stiffness/principal_y_mean",
    "z": "Impedance_Stiffness/principal_z_mean",
}
