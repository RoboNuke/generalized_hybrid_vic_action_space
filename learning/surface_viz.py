"""Surface-follow recorder overlays: keypoint balls, force/orientation gauges, top-down path inset.

Split into two layers:

* **In-scene** (:class:`KeypointBallMarkers`): a per-keypoint sphere drawn into the 3D scene (a real
  USD ``PointInstancer``, so the offscreen recorder camera captures it), one sphere prototype per
  status code (see :data:`STATUS_RGB`). Diameter is half the keypoint spacing. Needs Isaac, so it is
  imported lazily.

* **2D compositing** (everything else): pure numpy / PIL / matplotlib, so it is unit-testable without
  Isaac. :func:`draw_gauge` paints a red->green bar from a squashing value; :func:`topdown_inset`
  renders the matplotlib top-down path (with keypoint status circles); :func:`compose_tile` stacks a
  frame + gauges + inset; :func:`montage` tiles the per-env stills into one grid image.

Per-keypoint achieve/pass STATUS is NOT computed here — it is owned by the env
(``FlatSurfaceFollowEnv.keypoint_status``, same gate as the reward) and read from
``viz_snapshot()``. The recorder only colours balls/circles from those codes.
"""

from __future__ import annotations

import numpy as np

# Keypoint status colours (RGB, 0-255). Index == the status code == the marker prototype index fed
# to marker_indices. Five mutually high-contrast hues so the WHY of a miss is legible at a glance:
#   0 white   — not yet passed
#   1 green    — ACHIEVED (crossed while in contact AND on-track)
#   2 orange   — passed but NOT in contact (on-track otherwise)
#   3 cyan      — passed but OFF-TRACK (in contact otherwise, cross-track too large)
#   4 red        — passed but BOTH off contact AND off-track
STATUS_RGB = np.array(
    [[235, 235, 235], [30, 220, 70], [255, 145, 0], [0, 210, 235], [235, 30, 30]], dtype=np.uint8
)
N_STATUS = STATUS_RGB.shape[0]
# The moving CURRENT-goal / pace marker is a separate sphere in its own (magenta) hue, distinct from
# all five status colours so it never reads as a keypoint state.
GOAL_RGB = np.array([200, 40, 235], dtype=np.uint8)
# Back-compat aliases (older callers referenced BALL_RGB / GOAL_IDX).
BALL_RGB = STATUS_RGB
GOAL_IDX = 0
_RED = np.array([225, 45, 40], dtype=np.float32)
_GREEN = np.array([40, 210, 80], dtype=np.float32)
_YELLOW = (250, 225, 40)          # target / reference tick on the gauges (thick, high-contrast)


# ----------------------------------------------------------------------------- status tracking
# NOTE: per-keypoint achieved/passed status is NOT tracked here. It is computed once in the env
# (``FlatSurfaceFollowEnv._get_rewards`` -> ``self.keypoint_status``) with the exact gate the reward
# uses, and read straight out of ``viz_snapshot()["keypoint_status"]`` (codes 0..4, see STATUS_RGB).
# The recorder used to re-derive it from the (progress, in-contact) trace, which silently drifted
# from the reward; keep the single source of truth in the env.


# ----------------------------------------------------------------------------- geometry
def keypoint_world_positions(start_w, path_dir, spacing: float, k: int) -> np.ndarray:
    """(E, k, 3) world positions of keypoints 1..k = start + j*spacing*path_dir."""
    start_w = np.asarray(start_w, dtype=np.float64)
    path_dir = np.asarray(path_dir, dtype=np.float64)
    js = np.arange(1, k + 1, dtype=np.float64)[None, :, None]     # (1,k,1)
    return start_w[:, None, :] + js * spacing * path_dir[:, None, :]


def project_uv(points_w, center_w, u_dir, v_dir):
    """Project world points onto the plate (u=along path, v=lateral) about the plate center."""
    points_w = np.asarray(points_w, dtype=np.float64)
    rel = points_w - np.asarray(center_w, dtype=np.float64)
    u = (rel * np.asarray(u_dir, dtype=np.float64)).sum(-1)
    v = (rel * np.asarray(v_dir, dtype=np.float64)).sum(-1)
    return u, v


# ----------------------------------------------------------------------------- 2D drawing
def _blend_red_green(s: float) -> tuple[int, int, int]:
    s = float(np.clip(s, 0.0, 1.0))
    c = (1.0 - s) * _RED + s * _GREEN
    return tuple(int(x) for x in c)


def closeness_color(s: float) -> tuple[int, int, int]:
    """Public red->green blend for the raw-value read-outs: ``s`` in [0,1], 1 = ideal (green), 0 =
    far from ideal (red)."""
    return _blend_red_green(s)


def draw_gauge(height: int, color_value: float, label: str, width: int = 54, text: str | None = None,
               fill: float | None = None, mode: str = "bar", target_frac: float | None = None):
    """A vertical gauge (H, width, 3) uint8. Colour is red->green from ``color_value`` in [0,1] (the
    reward closeness). The bar geometry is set by ``fill`` and ``mode``:

      * mode="bar":    fill in [0,1] fills from the BOTTOM up (0 or negative = empty).
      * mode="center": fill in [-1,1] fills from the CENTRE — up for positive, down for negative.

    ``target_frac`` (0..1, bar mode) draws a faint target tick. ``text`` is the printed read-out."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (28, 28, 32))
    d = ImageDraw.Draw(img)
    pad = 5
    bx0, by0, bx1, by1 = pad, pad + 14, width - pad, height - pad     # bar track (leave a label strip)
    d.rectangle([bx0, by0, bx1, by1], fill=(50, 50, 56), outline=(90, 90, 96))
    col = _blend_red_green(color_value)
    if fill is None:
        fill = color_value
    if mode == "center":
        cy = (by0 + by1) // 2
        f = float(np.clip(fill, -1.0, 1.0))
        h = int(round((by1 - by0) / 2 * abs(f)))
        if h > 0:
            y0, y1 = (cy - h, cy) if f > 0 else (cy, cy + h)          # up = positive, down = negative
            d.rectangle([bx0 + 1, y0, bx1 - 1, y1], fill=col)
        # Yellow reference line at the desired angle (centre = zero deviation), 3x thicker (was 1px).
        d.line([bx0, cy, bx1, cy], fill=_YELLOW, width=3)
    else:  # bar (bottom-up)
        f = float(np.clip(fill, 0.0, 1.0))
        fill_h = int(round((by1 - by0) * f))
        if fill_h > 0:
            d.rectangle([bx0 + 1, by1 - fill_h, bx1 - 1, by1 - 1], fill=col)
        if target_frac is not None:
            ty = int(round(by1 - (by1 - by0) * float(np.clip(target_frac, 0.0, 1.0))))
            d.line([bx0, ty, bx1, ty], fill=_YELLOW, width=3)        # target tick, 3x thicker (was 1px)
    d.text((pad, 2), label, fill=(220, 220, 225))
    d.text((pad - 2, by1 - 12), text if text is not None else f"{color_value:.2f}", fill=(245, 245, 245))
    return np.asarray(img, dtype=np.uint8)


def topdown_inset(trace_u, trace_v, contact, over, start_uv, goal_uv, half_u, half_v, px: int = 300,
                  keypoint_uv=None, keypoint_status=None):
    """Matplotlib top-down of the plate + tip path -> (px, px, 3) uint8.

    trace_u/trace_v/contact/over are per-step arrays for ONE env. Segments are drawn only between
    consecutive steps that are BOTH over the surface; bright blue while in contact, light-grey in
    air. HIGH-CONTRAST palette on a near-black plate, path drawn 3x thicker than before for
    legibility. Goal = green circle, start = red x, ideal path = yellow dotted line along d.

    ``keypoint_uv`` (k, 2) + ``keypoint_status`` (k,) draw the per-keypoint circles in their STATUS
    colours (see :data:`STATUS_RGB`), mirroring the in-scene balls onto the minimap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(px / 100.0, px / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])          # fill the figure — no wasted black margin
    m = 1.04                                         # just enough room so the square outline isn't clipped
    ax.set_xlim(-half_u * m, half_u * m)
    ax.set_ylim(-half_v * m, half_v * m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((0.04, 0.04, 0.05))             # near-black plate for maximum path contrast

    # Tabletop square (a bright, thick white border).
    ax.add_patch(plt.Rectangle((-half_u, -half_v), 2 * half_u, 2 * half_v,
                               fill=False, edgecolor=(1.0, 1.0, 1.0), lw=3.5))
    # Ideal path along d (start -> goal), yellow dotted (thicker for contrast).
    ax.plot([start_uv[0], goal_uv[0]], [start_uv[1], goal_uv[1]],
            linestyle=(0, (2, 2)), color=(1.0, 0.9, 0.1), lw=2.4, zorder=2)

    # Keypoint circles in their status colours (drawn under the tip path so the live trace stays on
    # top, but over the ideal line). Black edge so light-status balls still pop on the dark plate.
    if keypoint_uv is not None and keypoint_status is not None:
        kp = np.asarray(keypoint_uv, dtype=np.float64).reshape(-1, 2)
        ks = np.asarray(keypoint_status).reshape(-1).astype(int)
        if kp.shape[0]:
            cols = STATUS_RGB[np.clip(ks, 0, N_STATUS - 1)] / 255.0
            ax.scatter(kp[:, 0], kp[:, 1], s=110, c=cols, edgecolors=(0, 0, 0), lw=1.4, zorder=4)

    tu = np.asarray(trace_u); tv = np.asarray(trace_v)
    over = np.asarray(over, dtype=bool); contact = np.asarray(contact, dtype=bool)
    air = (0.80, 0.82, 0.86); touch = (0.15, 0.6, 1.0)   # high-contrast: bright blue in contact, light grey in air
    for i in range(1, len(tu)):
        if not (over[i] and over[i - 1]):
            continue
        col = touch if (contact[i] and contact[i - 1]) else air
        ax.plot(tu[i - 1 : i + 1], tv[i - 1 : i + 1], color=col, lw=5.4, zorder=5,
                solid_capstyle="round")
    # Start (red x) and goal (green circle).
    ax.scatter([start_uv[0]], [start_uv[1]], marker="x", s=80, c=[(1.0, 0.2, 0.2)], lw=3.0, zorder=6)
    ax.scatter([goal_uv[0]], [goal_uv[1]], marker="o", s=95, facecolors="none",
               edgecolors=[(0.2, 0.95, 0.35)], lw=3.0, zorder=6)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    arr = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return arr


def _paste(dst, src, x, y):
    h, w = src.shape[:2]
    dst[y : y + h, x : x + w] = src


def compose_tile(frame, force_squash, orn_squash, inset, border_rgb=None, pad=6,
                 force_text=None, orn_text=None, force_fill=None, orn_fill=None,
                 status_label=None, status_color=None, force_target_frac=0.5, readouts=None,
                 kp_counts=None):
    """One annotated tile: [force gauge | orientation gauge | frame], both gauges on the LEFT, with
    the top-down inset pasted into the frame's bottom-left corner. Gauge COLOUR = squash closeness;
    the FORCE gauge fills from the bottom (force_fill in [0,1], empty at <=0, yellow target tick at
    ``force_target_frac``) and the ANGLE gauge fills from the centre (orn_fill in [-1,1], up/down by
    sign). force_text / orn_text are the physical read-outs. Optional coloured border. ``status_label``
    (with ``status_color`` RGB) draws a status pill in the frame's BOTTOM-RIGHT corner. ``readouts`` is
    an optional list of ``(text, rgb)`` lines stacked ABOVE that pill (same size), each colour-coded by
    how close the value is to ideal. ``kp_counts`` is an optional list of ``(text, rgb)`` lines stacked
    DOWN from the frame's TOP-RIGHT corner (the per-keypoint-outcome tally)."""
    from PIL import Image, ImageDraw, ImageFont

    frame = np.asarray(frame, dtype=np.uint8).copy()
    H, W = frame.shape[:2]
    if inset is not None:
        iw = int(W // 3 * 1.6)                       # top-down inset ~2x larger (scales with frame width)
        ins = np.asarray(Image.fromarray(inset).resize((iw, iw)), dtype=np.uint8)
        _paste(frame, ins, pad, H - iw - pad)
    if border_rgb is not None:                       # green success border around the IMAGE frame only
        b = 6
        frame[:b, :] = border_rgb; frame[-b:, :] = border_rgb
        frame[:, :b] = border_rgb; frame[:, -b:] = border_rgb
    if status_label or readouts or kp_counts:        # status pill + read-outs + keypoint tally
        img = Image.fromarray(frame); d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        def _tsize(s):
            try:
                tb = d.textbbox((0, 0), s, font=font); return tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                return d.textsize(s, font=font)

        mx, my = 8, 6
        bx1, by1 = W - pad, H - pad
        if kp_counts:                                # per-keypoint-outcome tally, TOP-RIGHT, stacked down
            gap = 4
            ty = pad
            for text, rgb in kp_counts:
                tw_, th_ = _tsize(text)
                ry0, ry1 = ty, ty + th_ + 2 * my
                rx0 = bx1 - tw_ - 2 * mx
                d.rectangle([rx0, ry0, bx1, ry1], fill=(18, 18, 20))   # dark backing keeps coloured text legible
                d.text((rx0 + mx, ry0 + my - 2), text, fill=tuple(int(c) for c in rgb), font=font)
                ty = ry1 + gap
        top = by1 + 4                                # running top edge; readouts stack upward from the pill
        if status_label:
            tw_, th_ = _tsize(status_label)
            bx0, by0 = bx1 - tw_ - 2 * mx, by1 - th_ - 2 * my
            d.rectangle([bx0, by0, bx1, by1], fill=tuple(status_color) if status_color else (180, 180, 185))
            d.text((bx0 + mx, by0 + my - 2), status_label, fill=(20, 20, 22), font=font)
            top = by0
        if readouts:
            gap = 4
            for text, rgb in reversed(list(readouts)):   # draw bottom-up so the list reads top -> down
                tw_, th_ = _tsize(text)
                ry1 = top - gap
                ry0 = ry1 - th_ - 2 * my
                rx0 = bx1 - tw_ - 2 * mx
                d.rectangle([rx0, ry0, bx1, ry1], fill=(18, 18, 20))   # dark backing keeps coloured text legible
                d.text((rx0 + mx, ry0 + my - 2), text, fill=tuple(int(c) for c in rgb), font=font)
                top = ry0
        frame = np.asarray(img, dtype=np.uint8)
    fg = draw_gauge(H, force_squash, "F", text=force_text, fill=force_fill, mode="bar",
                    target_frac=force_target_frac)
    og = draw_gauge(H, orn_squash, "A", text=orn_text, fill=orn_fill, mode="center")
    tile = np.concatenate([fg, og, frame], axis=1)
    return tile


def montage(tiles, rows: int, cols: int, gap: int = 6, bg=(15, 15, 18)):
    """Tile a list of equal-size (H,W,3) images into a rows x cols grid image."""
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles to montage")
    H, W = tiles[0].shape[:2]
    out = np.zeros((rows * H + (rows + 1) * gap, cols * W + (cols + 1) * gap, 3), dtype=np.uint8)
    out[:] = np.array(bg, dtype=np.uint8)
    for idx, t in enumerate(tiles[: rows * cols]):
        r, c = divmod(idx, cols)
        y = gap + r * (H + gap); x = gap + c * (W + gap)
        out[y : y + H, x : x + W] = t[:H, :W]
    return out


# ----------------------------------------------------------------------------- in-scene markers
class KeypointBallMarkers:
    """Per-keypoint spheres drawn into the 3D scene (captured by the recorder camera).

    Five sphere prototypes, one per status code (see :data:`STATUS_RGB`); each keypoint instance
    selects one via marker_indices. Positions are fixed for the episode (set in
    :meth:`set_positions`); only colours change per step. Imported lazily because Isaac Lab markers
    require the app to have booted.
    """

    _NAMES = ("unreached", "achieved", "off_contact", "off_track", "off_both")

    def __init__(self, prim_path: str, radius: float):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

        def _sphere(rgb):
            return sim_utils.SphereCfg(
                radius=float(radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(c / 255.0 for c in rgb)),
            )

        # Ordered dict: prototype index i (== status code) picks STATUS_RGB[i].
        markers = {self._NAMES[i]: _sphere(STATUS_RGB[i]) for i in range(N_STATUS)}
        cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        self._markers = VisualizationMarkers(cfg)
        self._translations = None

    def set_positions(self, translations_w) -> None:
        """translations_w: (N, 3) torch/np world positions for all keypoints of all envs (env-major)."""
        import torch

        self._translations = torch.as_tensor(np.asarray(translations_w), dtype=torch.float32)

    def update(self, marker_indices) -> None:
        import torch

        idx = torch.as_tensor(np.asarray(marker_indices), dtype=torch.long)
        self._markers.visualize(translations=self._translations, marker_indices=idx)


class GoalMarker:
    """A single big purple sphere per env that tracks a moving target. Used for the goal keypoint
    (opaque) and, at reduced opacity, the time-based PACE setpoint. Exaggerated (default 4x the
    keypoint-ball radius) so the moving target is easy to follow in the video."""

    def __init__(self, prim_path: str, radius: float, color=tuple(GOAL_RGB), opacity: float = 1.0):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={"goal": sim_utils.SphereCfg(
                radius=float(radius),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(c / 255.0 for c in color), opacity=float(opacity)),
            )},
        )
        self._markers = VisualizationMarkers(cfg)

    def update(self, translations_w) -> None:
        import torch

        t = torch.as_tensor(np.asarray(translations_w), dtype=torch.float32)
        idx = torch.zeros(t.shape[0], dtype=torch.long)
        self._markers.visualize(translations=t, marker_indices=idx)
