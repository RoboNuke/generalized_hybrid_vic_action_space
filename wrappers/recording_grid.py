"""Compose per-env frame buffers into a 3x4 grid video, draw borders and overlays,
and write the result as both a GIF on disk and a TB ``add_video`` event.

Layout (rows top -> bottom):
    row 0: 4 best envs by final episode return
    row 1: 4 envs whose final return is closest to the median
    row 2: 4 worst envs by final episode return

Border policy:
    * No border drawn while the env is still alive in a given frame.
    * From the timestep at which ``terminated[i] | truncated[i]`` first fires
      (and forever after, since the frame itself freezes), a 3-pixel border is
      drawn — green if ``is_success[i]`` was True at termination, red otherwise.

Per-tile overlays:
    * top-left: ``TotRew: <accum reward, frozen at termination>``
    * bottom-left: ``V-Est: <min(Q1, Q2)(s, a_executed) at that step>``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RecordingWrapper requires Pillow. Install with `pip install Pillow`."
    ) from e


GRID_ROWS = 3
GRID_COLS = 4
TILES = GRID_ROWS * GRID_COLS  # 12


def select_grid_indices(returns: torch.Tensor) -> torch.Tensor:
    """Return 12 env indices ordered (best 4, middle 4, worst 4) by ``returns``.

    ``returns`` has shape ``(num_envs,)``. If ``num_envs < 12``, fill with the
    available envs (no padding tile — caller will see a shorter grid). The
    "middle 4" are the four envs whose returns are closest to the median. Within
    every row the four indices are sorted by DESCENDING return, so each row reads
    high -> low return left to right.
    """
    n = int(returns.numel())
    if n == 0:
        raise ValueError("select_grid_indices: empty returns tensor")
    sorted_idx = torch.argsort(returns)  # ascending
    if n <= TILES:
        # Not enough envs to differentiate; just return them best-first.
        return torch.flip(sorted_idx, dims=(0,))

    # Each row is sorted by DESCENDING return so every row reads high -> low left to right.
    best = torch.flip(sorted_idx[-4:], dims=(0,))   # highest returns, descending
    worst = torch.flip(sorted_idx[:4], dims=(0,))   # lowest returns, still descending within the row

    # 4 envs nearest the median return (selected by distance), then re-ordered by descending
    # return so the middle row is sorted the same way as the others.
    median_val = returns[sorted_idx[n // 2]]
    dist = (returns - median_val).abs()
    # Exclude indices already used in best/worst.
    used = torch.zeros(n, dtype=torch.bool, device=returns.device)
    used[best] = True
    used[worst] = True
    dist_masked = dist.clone()
    dist_masked[used] = float("inf")
    middle = torch.topk(dist_masked, k=4, largest=False).indices
    middle = middle[torch.argsort(returns[middle], descending=True)]

    return torch.cat([best, middle, worst], dim=0)


def _freeze_frames(
    frames: torch.Tensor,
    term_step: torch.Tensor,
) -> torch.Tensor:
    """Freeze each env's frames after termination by repeating the last live frame.

    :param frames: (num_envs, T, H, W, 3) uint8 tensor.
    :param term_step: (num_envs,) int64 — the timestep at which env first
        terminated, or ``T`` if it survived. Frames at indices ``> term_step``
        are overwritten with ``frames[i, term_step[i]]``.
    """
    num_envs, T, _, _, _ = frames.shape
    out = frames.clone()
    for i in range(num_envs):
        ts = int(term_step[i].item())
        if ts < T - 1:
            out[i, ts + 1 :] = out[i, ts]
    return out


def _draw_tile(
    canvas: Image.Image,
    tile: np.ndarray,
    x: int,
    y: int,
    *,
    show_border: bool,
    border_color: str,
    tot_rew: float,
    v_est: float | None,
    font: ImageFont.ImageFont,
) -> None:
    """Paste a single ``tile`` (HxWx3 uint8 numpy) onto ``canvas`` at ``(x, y)``
    and draw the border + overlays."""
    h, w = tile.shape[:2]
    canvas.paste(Image.fromarray(tile), (x, y))
    draw = ImageDraw.Draw(canvas)
    if show_border:
        draw.rectangle([(x, y), (x + w - 1, y + h - 1)], outline=border_color, width=3)
    draw.text((x + 3, y + 3), f"TotRew: {tot_rew:.2f}", fill=(0, 255, 0), font=font)
    if v_est is not None:
        draw.text(
            (x + 3, y + h - 16), f"V-Est: {v_est:.2f}", fill=(0, 255, 0), font=font
        )


def build_grid_video(
    frames: torch.Tensor,
    returns: torch.Tensor,
    term_step: torch.Tensor,
    is_success: torch.Tensor,
    values: torch.Tensor | None,
    engaged: torch.Tensor | None = None,
    success_seq: torch.Tensor | None = None,
) -> np.ndarray:
    """Compose the 3x4 grid video.

    :param frames: ``(num_envs, T, H, W, 3)`` uint8 tensor on CPU. Frames at
        ``t > term_step[i]`` may be stale; this function freezes them.
    :param returns: ``(num_envs,)`` float — final episode return per env (frozen
        at first termination during the session).
    :param term_step: ``(num_envs,)`` int64 — first-termination step, or ``T``
        if the env survived the whole session.
    :param is_success: ``(num_envs,)`` bool — success flag at first termination.
        Used only for the legacy terminal border (when ``success_seq`` is None).
    :param values: ``(num_envs, T)`` float (or None) — per-step ``min(Q1, Q2)``.
        Stale entries past ``term_step`` are ignored (overlay shows the value at
        ``term_step`` once frozen).
    :param success_seq: ``(num_envs, T)`` bool (or None) — PER-STEP "in success
        position" flag. When provided, the border is drawn per frame: **green** on
        steps in the success position, **orange** on steps merely engaged, none
        otherwise (the terminal red/green behavior is disabled).
    :param engaged: ``(num_envs, T)`` bool (or None) — PER-STEP engagement flag,
        used for the orange border. Ignored unless ``success_seq`` is provided.
    :returns: ``(T, gridH, gridW, 3)`` uint8 numpy array.
    """
    selected = select_grid_indices(returns)
    n_sel = int(selected.numel())
    n_show = min(n_sel, TILES)
    selected = selected[:n_show]

    # Subset to the selected tiles FIRST, then freeze only those. Freezing the full
    # M-trajectory stack would clone every collected buffer (M x T x H x W x 3); at high
    # recorder resolutions that extra copy is gigabytes and can OOM the host. Advanced
    # indexing already copies, so only the 12 kept trajectories are ever cloned.
    sel_frames = _freeze_frames(frames[selected], term_step[selected]).cpu().numpy()  # (n_show, T, H, W, 3)
    sel_returns = returns[selected].cpu().numpy()
    sel_term = term_step[selected].cpu().numpy()
    sel_succ = is_success[selected].cpu().numpy().astype(bool)
    if values is not None:
        sel_values = values[selected].cpu().numpy()  # (n_show, T)
    else:
        sel_values = None
    # Per-step engagement / success borders (preferred when available).
    sel_success_seq = (
        success_seq[selected].cpu().numpy().astype(bool) if success_seq is not None else None
    )
    sel_engaged = (
        engaged[selected].cpu().numpy().astype(bool) if engaged is not None else None
    )

    T = sel_frames.shape[1]
    H = sel_frames.shape[2]
    W = sel_frames.shape[3]
    grid_h = GRID_ROWS * H
    grid_w = GRID_COLS * W

    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    out = np.zeros((T, grid_h, grid_w, 3), dtype=np.uint8)
    for t in range(T):
        canvas = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
        for tile_idx in range(n_show):
            row = tile_idx // GRID_COLS
            col = tile_idx % GRID_COLS
            x = col * W
            y = row * H
            ts = int(sel_term[tile_idx])
            terminated_now = t >= ts
            if sel_success_seq is not None:
                # Per-step border, frozen at termination: green in the success
                # position, orange when (only) engaged, none otherwise.
                tt = min(t, ts, T - 1)
                if sel_success_seq[tile_idx, tt]:
                    show_border, border_color = True, "green"
                elif sel_engaged is not None and sel_engaged[tile_idx, tt]:
                    show_border, border_color = True, "orange"
                else:
                    show_border, border_color = False, None
            else:
                # Legacy: a single terminal border, green on success else red.
                show_border = terminated_now
                border_color = "green" if sel_succ[tile_idx] else "red"
            tot_rew = float(sel_returns[tile_idx])
            if sel_values is not None:
                # Freeze value at termination.
                v_t = min(t, max(0, ts - 1)) if ts > 0 else min(t, T - 1)
                v_est: float | None = float(sel_values[tile_idx, v_t])
            else:
                v_est = None
            _draw_tile(
                canvas,
                sel_frames[tile_idx, t],
                x,
                y,
                show_border=show_border,
                border_color=border_color,
                tot_rew=tot_rew,
                v_est=v_est,
                font=font,
            )
        out[t] = np.asarray(canvas)
    return out


def write_gif(grid: np.ndarray, path: str, fps: int) -> None:
    """Save ``grid`` (``(T, H, W, 3)`` uint8 numpy) as an animated GIF."""
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames: list[Image.Image] = [Image.fromarray(grid[t]) for t in range(grid.shape[0])]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def write_mp4(grid: np.ndarray, path: str, fps: int) -> None:
    """Save ``grid`` (``(T, H, W, 3)`` uint8 numpy) as an H.264 MP4.

    Far smaller and smoother to play back than a GIF at video resolutions (a 1080p GIF
    is hundreds of MB and stutters because viewers can't decode frames fast enough). Uses
    the ffmpeg binary bundled with ``imageio-ffmpeg``; ``yuv420p`` keeps it playable in
    browsers / QuickTime, which require even frame dimensions, so a stray odd row/column
    is trimmed.
    """
    import imageio

    h, w = grid.shape[1], grid.shape[2]
    h2, w2 = h - (h % 2), w - (w % 2)
    g = grid[:, :h2, :w2, :] if (h2, w2) != (h, w) else grid
    writer = imageio.get_writer(
        path, format="FFMPEG", fps=int(max(1, fps)), codec="libx264",
        quality=8, macro_block_size=1, pixelformat="yuv420p",
    )
    try:
        for t in range(g.shape[0]):
            writer.append_data(np.ascontiguousarray(g[t]))
    finally:
        writer.close()


def write_video(grid: np.ndarray, path_base: str, fps: int, fmt: str = "mp4") -> str:
    """Write ``grid`` to ``<path_base>.<fmt>`` and return the path written.

    ``fmt`` is ``"mp4"`` (H.264; recommended — compact and smooth) or ``"gif"`` (large and
    laggy at video resolutions, kept for compatibility).
    """
    fmt = str(fmt).lower()
    if fmt == "mp4":
        path = f"{path_base}.mp4"
        write_mp4(grid, path, fps)
    elif fmt == "gif":
        path = f"{path_base}.gif"
        write_gif(grid, path, fps)
    else:
        raise ValueError(f"recorder video_format must be 'mp4' or 'gif', got {fmt!r}")
    return path


def write_tb_video(image_writer, tag: str, grid: np.ndarray, fps: int, global_step: int) -> None:
    """Write ``grid`` as a TB video event. ``add_video`` expects ``(N, T, C, H, W)``
    uint8."""
    video = torch.from_numpy(grid).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    image_writer.add_video(tag, video, global_step=global_step, fps=fps)
