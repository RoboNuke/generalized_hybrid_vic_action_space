"""Video keyframe-montage utilities for ``keyframe_analysis.ipynb``.

Reads an MP4 rollout video and composites keyframes spaced through it into a SINGLE image -- a
stroboscopic overlay. With a fixed camera and a robot/peg that moves across the scene, overlaying
the keyframes with transparency shows the arm at successive positions in one frame, so you read
the task unfolding (here, left -> right) at a glance.

Self-contained (numpy + a video reader + matplotlib only). The notebook loads the video once and
then re-runs :func:`keyframe_montage` with different spacing / crop / alpha to tune the look.
"""

from __future__ import annotations

import os

import numpy as np


# --------------------------------------------------------------------------- #
# Video loading
# --------------------------------------------------------------------------- #
def load_video(path: str) -> np.ndarray:
    """Load a video into an ``(N, H, W, 3)`` uint8 RGB array.

    Tries ``imageio`` (ffmpeg backend) first, then falls back to OpenCV (``cv2``). Raises if
    neither can read a frame.
    """
    try:
        import imageio.v3 as iio
        frames = np.stack([f for f in iio.imiter(path)])
        if frames.ndim == 4 and frames.shape[-1] >= 3:
            return frames[..., :3]
        return frames
    except Exception as imageio_err:
        try:
            import cv2
        except Exception:
            raise RuntimeError(
                f"could not read {path!r} with imageio ({imageio_err}) and cv2 is unavailable")
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"could not read any frames from {path!r}")
        return np.stack(frames)


def load_frame(path: str, index: int = 0) -> np.ndarray:
    """Read a single frame (default the first) from a video as an ``(H, W, 3)`` uint8 RGB array.

    Only decodes up to the requested frame (cheap for a first-frame preview). Tries ``imageio``,
    then OpenCV.
    """
    frame = None
    try:
        import imageio.v3 as iio
        for i, f in enumerate(iio.imiter(path)):
            if i == index:
                frame = np.asarray(f)
                break
    except Exception:
        frame = None
    if frame is None:
        try:
            import cv2
        except Exception:
            raise RuntimeError(f"could not read frame {index} from {path!r} (no working video reader)")
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, fr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read frame {index} from {path!r}")
        frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return frame[..., :3]


# --------------------------------------------------------------------------- #
# Cropping + keyframe selection + montage
# --------------------------------------------------------------------------- #
def crop_frame(frame: np.ndarray, left: int = 0, right: int = 0,
               top: int = 0, bottom: int = 0) -> np.ndarray:
    """Return ``frame`` with the given number of pixels cropped off each side."""
    h, w = frame.shape[:2]
    return frame[top:(h - bottom) if bottom else h, left:(w - right) if right else w]


def keyframe_indices(n: int, num_keyframes: int, start: int = 0, end: int | None = None) -> list:
    """``num_keyframes`` frame indices evenly spaced from ``start`` to ``end`` (inclusive).

    ``start`` skips leading frames and ``end`` (default the last frame, ``n - 1``) trims trailing
    ones, so you can montage a MIDDLE chunk of the video. Indices are rounded to ints and
    de-duplicated, so a short ``[start, end]`` range may yield fewer than ``num_keyframes``.
    """
    end = (n - 1) if end is None else max(0, min(int(end), n - 1))
    start = max(0, min(int(start), end))
    num = max(1, int(num_keyframes))
    if num == 1:
        return [start]
    out, seen = [], set()
    for v in np.linspace(start, end, num).round().astype(int):
        i = int(v)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def keyframe_montage(frames: np.ndarray, *, num_keyframes: int = 8, start: int = 0,
                     end: int | None = None, crop=(0, 0, 0, 0), alpha: float = 0.6,
                     alpha_start: float | None = None, alpha_end: float | None = None,
                     mode: str = "foreground", fg_gain: float = 8.0,
                     background: np.ndarray | None = None, reverse: bool = False):
    """Composite keyframes spaced through ``frames`` into ONE stroboscopic image.

    * ``num_keyframes`` -- how many keyframes to overlay, evenly spaced from ``start`` to ``end``.
    * ``start`` / ``end`` -- first / last frame to use (``end`` defaults to the last frame); set both
      to montage a MIDDLE chunk of the video.
    * ``alpha_start`` / ``alpha_end`` -- if given, each keyframe's opacity RAMPS linearly from
      ``alpha_start`` (first pose) to ``alpha_end`` (last), instead of the uniform ``alpha``. In
      ``"foreground"`` mode this reads as a gradient where earlier poses stay visible at the
      ``alpha_start`` floor while later ones are stronger -- a much gentler progression than the
      uniform ``over`` fade. Leave both ``None`` to use the flat ``alpha``.
    * ``crop``         -- ``(left, right, top, bottom)`` pixels cropped off every frame.
    * ``alpha``        -- opacity each keyframe is blended in at (0-1).
    * ``mode``:
        - ``"foreground"`` (default) -- overlay only each keyframe's MOVING parts (where it differs
          from a static background = the per-pixel median over the video) at ``alpha`` on that clean
          background. Every pose shows about equally, so you see the arm at each position -- best for
          "the robot moving through the task". ``fg_gain`` sharpens the moving-vs-static mask.
        - ``"over"`` -- plain alpha-over compositing in order: the LAST pose is solid and earlier
          ones trail off (``reverse=True`` makes the FIRST pose the solid one). A motion-trail look.
    * ``background``   -- optional precomputed background (full-frame, uncropped); defaults to the
      per-pixel median of ``frames``. Pass it to avoid recomputing while tuning other knobs.

    Returns ``(image_uint8, indices)`` -- the composite ``(H, W, 3)`` image and the keyframe indices used.
    """
    n = len(frames)
    idx = keyframe_indices(n, num_keyframes, start, end)
    # Per-keyframe opacity: a linear ramp alpha_start -> alpha_end if either is given, else flat alpha.
    if alpha_start is None and alpha_end is None:
        alphas = [alpha] * len(idx)
    else:
        a0 = alpha if alpha_start is None else alpha_start
        a1 = alpha if alpha_end is None else alpha_end
        alphas = list(np.linspace(a0, a1, len(idx))) if len(idx) > 1 else [a1]
    if reverse:                                          # keep each frame's opacity tied to its time
        idx, alphas = idx[::-1], alphas[::-1]
    kfs = [crop_frame(frames[i], *crop).astype(np.float64) for i in idx]

    if mode == "over":
        comp = kfs[0].copy()
        for kf, a in zip(kfs[1:], alphas[1:]):
            comp = comp * (1.0 - a) + kf * a
    elif mode == "foreground":
        if background is None:
            background = np.median(frames.astype(np.float64), axis=0)
        bg = crop_frame(background, *crop).astype(np.float64)
        comp = bg.copy()
        for kf, a in zip(kfs, alphas):
            diff = np.abs(kf - bg).mean(axis=2, keepdims=True) / 255.0   # 0..1 moving-ness
            m = np.clip(diff * fg_gain, 0.0, 1.0) * a                    # foreground opacity (ramped)
            comp = comp * (1.0 - m) + kf * m
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'foreground' or 'over')")

    return np.clip(comp, 0, 255).astype(np.uint8), idx


# --------------------------------------------------------------------------- #
# Saving / showing images (raw pixels -> no white border)
# --------------------------------------------------------------------------- #
def name_from_path(video_path: str) -> str:
    """A readable name from a ``.../{group}/{agent}/videos/{split}/{file}`` path (fallback: file stem)."""
    try:
        p = os.path.normpath(video_path).split(os.sep)
        return f"{p[-5]}_agent{p[-4]}_{os.path.splitext(p[-1])[0]}"
    except Exception:
        return os.path.splitext(os.path.basename(video_path))[0]


def save_image(img: np.ndarray, out_dir: str, name: str) -> str:
    """Write ``img`` (a raw ``H x W x 3`` array) to ``out_dir/name.svg`` with NO axes/figure white
    border. The image is a raster embedded in the SVG at its native resolution. Creates ``out_dir``
    and returns the path.
    """
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.svg")
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])   # axes fills the figure -> no surrounding border
    ax.imshow(np.asarray(img))
    ax.axis("off")
    fig.savefig(path, format="svg", pad_inches=0)
    plt.close(fig)
    print("saved", path)
    return path


def show_image(img: np.ndarray):
    """Display ``img`` inline with the axes filling the whole figure -- no white border/margin."""
    import matplotlib.pyplot as plt
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100.0, h / 100.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])   # axes fills the figure: no surrounding border
    ax.imshow(img)
    ax.axis("off")
    plt.show()
    return fig
