"""Surface-follow recorder overlays: keypoint balls, force/orientation gauges, top-down path inset.

Split into two layers:

* **In-scene** (:class:`KeypointBallMarkers`): a per-keypoint sphere drawn into the 3D scene (a real
  USD ``PointInstancer``, so the offscreen recorder camera captures it). Blue = not yet reached,
  green = ACHIEVED (crossed in contact, one-at-a-time), red = passed but not achieved. Diameter is
  half the keypoint spacing. Needs Isaac, so it is imported lazily.

* **2D compositing** (everything else): pure numpy / PIL / matplotlib, so it is unit-testable without
  Isaac. :class:`KeypointStatusTracker` mirrors the env's achieve/pass logic from the per-step
  (progress, in-contact) trace; :func:`draw_gauge` paints a red->green bar from a squashing value;
  :func:`topdown_inset` renders the matplotlib top-down path; :func:`compose_tile` stacks a frame +
  gauges + inset; :func:`montage` tiles the per-env stills into one grid image.
"""

from __future__ import annotations

import numpy as np

# Ball colours (RGB, 0-255). Index == the marker prototype index fed to marker_indices:
# 0 blue (unreached), 1 green (achieved), 2 red (passed only), 3 dark purple (the CURRENT goal
# keypoint — overrides status each frame and reverts to the status colour once passed).
# 0 blue (unreached), 1 green (achieved), 2 red (passed only), 3 purple (goal/pace marker).
BALL_RGB = np.array([[15, 35, 205], [40, 200, 90], [255, 25, 25], [120, 30, 175]], dtype=np.uint8)
GOAL_IDX = 3
_RED = np.array([220, 60, 50], dtype=np.float32)
_GREEN = np.array([45, 200, 95], dtype=np.float32)


# ----------------------------------------------------------------------------- status tracking
class KeypointStatusTracker:
    """Per-env, per-keypoint status from the (progress, in-contact) trace — mirrors the env.

    Status codes (also the ball colour index): 0 = not yet reached, 1 = ACHIEVED (the step that
    crossed it was in contact and crossed EXACTLY ONE keypoint), 2 = passed but not achieved
    (crossed as part of a multi-keypoint jump or off contact). Achieved never downgrades.
    """

    def __init__(self, num_envs: int, k_per_env: int, spacing: float):
        self.n = int(num_envs)
        self.k = int(k_per_env)                       # keypoints tracked per env (1..k at arc j*spacing)
        self.spacing = float(spacing)
        # status[:, j] is keypoint j (1-based); column 0 is unused so index == keypoint number.
        self.status = np.zeros((self.n, self.k + 1), dtype=np.uint8)
        # Seed prev_progress at 0 (NOT at the first frame's progress) so the first update counts the
        # crossing 0 -> progress and colours everything the peg is already past on frame 1 — matching
        # the env's keypoints_passed (which resets prev_progress to 0). Otherwise those initial
        # keypoints, already behind the peg when the tracker starts watching, stay stuck blue.
        self.prev_progress = np.zeros(self.n, dtype=np.float64)
        self.setpoint_idx = np.ones(self.n, dtype=int)   # current goal keypoint (1..k), coloured purple

    def update(self, progress: np.ndarray, in_contact: np.ndarray) -> None:
        progress = np.asarray(progress, dtype=np.float64).reshape(self.n)
        in_contact = np.asarray(in_contact).reshape(self.n).astype(bool)
        # Current goal keypoint = the next one ahead of the projected progress (mirrors the env's
        # setpoint_kp_idx), RATCHETED so it only ever advances (never pulled back if the arm reverses).
        new_setpoint = np.clip(np.floor(progress / self.spacing).astype(int) + 1, 1, self.k)
        self.setpoint_idx = np.maximum(self.setpoint_idx, new_setpoint)
        kp_prev = np.clip(np.floor(self.prev_progress / self.spacing).astype(int), 0, self.k)
        kp_curr = np.clip(np.floor(progress / self.spacing).astype(int), 0, self.k)
        for e in range(self.n):
            a, b = kp_prev[e], kp_curr[e]
            if b <= a:                                 # no forward crossing this step
                continue
            crossed = b - a
            for j in range(a + 1, b + 1):              # keypoints newly crossed this step
                if self.status[e, j] == 1:             # already achieved -> keep
                    continue
                if crossed == 1 and in_contact[e]:
                    self.status[e, j] = 1              # clean single-keypoint drag in contact
                else:
                    self.status[e, j] = 2              # passed as part of a jump / off contact
        self.prev_progress = progress.copy()

    def marker_indices(self) -> np.ndarray:
        """Flat (n*k,) int array of ball STATUS colour indices (0/1/2) for keypoints 1..k of every
        env, env-major. The current goal is drawn as a SEPARATE moving marker, not by recolouring a
        ball, so this returns pure status."""
        return self.status[:, 1 : self.k + 1].reshape(-1).astype(np.int64)


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
        d.line([bx0, cy, bx1, cy], fill=(150, 150, 155))              # zero line at centre
        f = float(np.clip(fill, -1.0, 1.0))
        h = int(round((by1 - by0) / 2 * abs(f)))
        if h > 0:
            y0, y1 = (cy - h, cy) if f > 0 else (cy, cy + h)          # up = positive, down = negative
            d.rectangle([bx0 + 1, y0, bx1 - 1, y1], fill=col)
    else:  # bar (bottom-up)
        f = float(np.clip(fill, 0.0, 1.0))
        fill_h = int(round((by1 - by0) * f))
        if fill_h > 0:
            d.rectangle([bx0 + 1, by1 - fill_h, bx1 - 1, by1 - 1], fill=col)
        if target_frac is not None:
            ty = int(round(by1 - (by1 - by0) * float(np.clip(target_frac, 0.0, 1.0))))
            d.line([bx0, ty, bx1, ty], fill=(210, 210, 120))         # target tick
    d.text((pad, 2), label, fill=(220, 220, 225))
    d.text((pad - 2, by1 - 12), text if text is not None else f"{color_value:.2f}", fill=(245, 245, 245))
    return np.asarray(img, dtype=np.uint8)


def topdown_inset(trace_u, trace_v, contact, over, start_uv, goal_uv, half_u, half_v, px: int = 300):
    """Matplotlib top-down of the plate + tip path -> (px, px, 3) uint8.

    trace_u/trace_v/contact/over are per-step arrays for ONE env. Segments are drawn only between
    consecutive steps that are BOTH over the surface; dark blue while in contact, light blue in air.
    Goal = green circle, start = red x, ideal path = yellow dotted line along d.
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
    ax.set_facecolor((0.12, 0.12, 0.14))

    # Tabletop square (the thick white border).
    ax.add_patch(plt.Rectangle((-half_u, -half_v), 2 * half_u, 2 * half_v,
                               fill=False, edgecolor=(0.9, 0.9, 0.92), lw=3.0))
    # Ideal path along d (start -> goal), yellow dotted.
    ax.plot([start_uv[0], goal_uv[0]], [start_uv[1], goal_uv[1]],
            linestyle=(0, (2, 2)), color=(0.95, 0.85, 0.15), lw=1.4, zorder=2)

    tu = np.asarray(trace_u); tv = np.asarray(trace_v)
    over = np.asarray(over, dtype=bool); contact = np.asarray(contact, dtype=bool)
    light = (0.55, 0.75, 1.0); dark = (0.05, 0.20, 0.75)
    for i in range(1, len(tu)):
        if not (over[i] and over[i - 1]):
            continue
        col = dark if (contact[i] and contact[i - 1]) else light
        ax.plot(tu[i - 1 : i + 1], tv[i - 1 : i + 1], color=col, lw=1.8, zorder=3,
                solid_capstyle="round")
    # Start (red x) and goal (green circle).
    ax.scatter([start_uv[0]], [start_uv[1]], marker="x", s=70, c=[(0.9, 0.15, 0.15)], lw=2.5, zorder=5)
    ax.scatter([goal_uv[0]], [goal_uv[1]], marker="o", s=80, facecolors="none",
               edgecolors=[(0.15, 0.85, 0.25)], lw=2.5, zorder=5)

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
                 force_text=None, orn_text=None, force_fill=None, orn_fill=None):
    """One annotated tile: [force gauge | orientation gauge | frame], both gauges on the LEFT, with
    the top-down inset pasted into the frame's bottom-left corner. Gauge COLOUR = squash closeness;
    the FORCE gauge fills from the bottom (force_fill in [0,1], empty at <=0, target tick at desired)
    and the ANGLE gauge fills from the centre (orn_fill in [-1,1], up/down by sign). force_text /
    orn_text are the physical read-outs. Optional coloured border."""
    from PIL import Image

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
    fg = draw_gauge(H, force_squash, "F", text=force_text, fill=force_fill, mode="bar", target_frac=0.5)
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

    Three sphere prototypes (blue/green/red); each keypoint instance selects one via marker_indices.
    Positions are fixed for the episode (set in :meth:`set_positions`); only colours change per step.
    Imported lazily because Isaac Lab markers require the app to have booted.
    """

    def __init__(self, prim_path: str, radius: float):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

        def _sphere(rgb):
            return sim_utils.SphereCfg(
                radius=float(radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(c / 255.0 for c in rgb)),
            )

        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={"blue": _sphere(BALL_RGB[0]), "green": _sphere(BALL_RGB[1]), "red": _sphere(BALL_RGB[2])},
        )
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

    def __init__(self, prim_path: str, radius: float, color=tuple(BALL_RGB[GOAL_IDX]), opacity: float = 1.0):
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
