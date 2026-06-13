"""Per-agent trajectory collection + 3x4-grid GIF for a SINGLE trained agent.

Unlike the in-training ``wrappers/recording.py`` (which captures one parallel
episode across ALL scene envs during training and overlays Q-values from agent 0),
this module is built for *post-hoc, per-agent* evaluation:

* It is driven by a ``num_agents == 1`` run, so the env's envs, the policy, and
  the twin critics all belong to ONE agent. The grid and the V-Est overlay are
  therefore agent-specific.
* It collects **complete episode trajectories** — running as many full-batch
  episodes as needed — until at least ``num_trajectories`` are gathered, then
  selects best-4 / median-4 / worst-4 by return across the whole collected set
  (via :func:`recording_grid.build_grid_video`).

The heavy lifting of grid composition / GIF writing is reused from
``wrappers/recording_grid.py``; the per-step frame/return/value capture mirrors
``RecordingWrapper`` so the two stay visually consistent.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import torch

from wrappers.recording import _coerce_done, _unpack_act
from wrappers.recording_grid import build_grid_video, write_video, write_tb_video


def set_camera_active(camera: Any, active: bool) -> None:
    """Toggle TiledCamera rasterization by mutating ``update_period`` in place
    (0.0 = every step, large = effectively off). Mirrors RecordingWrapper."""
    target = 0.0 if active else 1.0e9
    for owner in (camera, getattr(camera, "cfg", None)):
        if owner is None:
            continue
        if hasattr(owner, "update_period"):
            try:
                owner.update_period = target
            except Exception:
                pass


def read_camera_rgb(camera: Any) -> torch.Tensor:
    """Pull the latest RGB tensor and return CPU uint8 ``(N, H, W, 3)``."""
    rgb = camera.data.output["rgb"]
    if rgb.dim() != 4:
        raise RuntimeError(
            f"recorder camera returned unexpected shape {tuple(rgb.shape)}; expected (N, H, W, C)"
        )
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb.detach().cpu()


def _compute_min_q(
    critic_1: Any,
    critic_2: Any,
    state_preprocessor: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """``min(Q1, Q2)(state, action)`` for every env as a CPU ``(N,)`` float tensor."""
    with torch.no_grad():
        processed = state_preprocessor(state)
        inputs = {"observations": processed, "taken_actions": action}
        q1, _, _ = _unpack_act(critic_1.act(inputs, role="critic_1"))
        q2, _, _ = _unpack_act(critic_2.act(inputs, role="critic_2"))
        return torch.minimum(q1, q2).view(-1).float().cpu()


def _resolve_state(env: Any, obs: torch.Tensor) -> torch.Tensor:
    """Critic state for asymmetric AC; falls back to obs when the env is symmetric."""
    try:
        st = env.state()
    except Exception:
        st = None
    return st if st is not None else obs


def collect_and_record(
    *,
    env: Any,
    agent: Any,
    recorder_cfg: Any,
    camera: Any,
    max_episode_length: int,
    num_trajectories: int,
    output_dir: str,
    gif_name: str = "recording.gif",
    image_writer: Any = None,
    global_step: int = 0,
) -> str:
    """Roll out the (single-agent) policy, collect >= ``num_trajectories`` complete
    episodes, and write a best/median/worst 3x4-grid GIF to ``output_dir``.

    The agent's OWN ``critic_1``/``critic_2`` supply the per-step V-Est overlay.
    Returns the written GIF path.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_envs = int(env.num_envs)
    H, W = int(recorder_cfg.height), int(recorder_cfg.width)
    T = int(max_episode_length)
    if T <= 0:
        raise RuntimeError(
            "max_episode_length must be > 0 to size the per-episode frame buffer; "
            f"got {max_episode_length!r}."
        )

    state_pre = getattr(agent, "_state_preprocessor", None) or (lambda s: s)
    critic_1, critic_2 = agent.critic_1, agent.critic_2

    # Enough full-batch episodes to reach the target (deterministic, no infinite loop).
    num_episodes = max(1, math.ceil(int(num_trajectories) / max(1, num_envs)))
    print(
        f"[record] collecting >= {num_trajectories} trajectories: "
        f"{num_episodes} episode(s) x {num_envs} envs = {num_episodes * num_envs} trajectories",
        flush=True,
    )

    coll_frames: list[torch.Tensor] = []
    coll_returns: list[torch.Tensor] = []
    coll_term: list[torch.Tensor] = []
    coll_succ: list[torch.Tensor] = []
    coll_values: list[torch.Tensor] = []
    coll_succ_seq: list[torch.Tensor] = []   # per-step "in success position"
    coll_eng_seq: list[torch.Tensor] = []    # per-step "engaged"

    set_camera_active(camera, True)
    try:
        for ep in range(num_episodes):
            frames = torch.zeros((num_envs, T, H, W, 3), dtype=torch.uint8)
            returns = torch.zeros(num_envs, dtype=torch.float32)
            values = torch.zeros((num_envs, T), dtype=torch.float32)
            term_step = torch.full((num_envs,), T, dtype=torch.int64)
            success = torch.zeros(num_envs, dtype=torch.bool)
            succ_seq = torch.zeros((num_envs, T), dtype=torch.bool)   # per-step success position
            eng_seq = torch.zeros((num_envs, T), dtype=torch.bool)    # per-step engaged
            env_done = torch.zeros(num_envs, dtype=torch.bool)

            obs, _ = env.reset()
            state = _resolve_state(env, obs)

            for t in range(T):
                actions, _ = agent.act(obs, state, timestep=10**9, timesteps=10**9)
                pre_state = state if state is not None else obs
                v = _compute_min_q(critic_1, critic_2, state_pre, pre_state, actions)

                obs, reward, terminated, truncated, info = env.step(actions)
                rgb = read_camera_rgb(camera)  # (num_envs, H, W, 3) cpu uint8

                alive = ~env_done
                alive_idx = alive.nonzero(as_tuple=False).view(-1)
                if alive_idx.numel() > 0:
                    frames[alive_idx, t] = rgb[alive_idx]
                    returns[alive_idx] += reward.detach().view(-1).float().cpu()[alive_idx]
                    values[alive_idx, t] = v[alive_idx]

                    # Per-step border signals: "in success position" (green) and
                    # "engaged" (orange). Both ride in info as per-env (num_envs,)
                    # tensors; engaged is a 0/1 indicator so threshold at 0.5.
                    succ_now = info.get("is_success", None)
                    if isinstance(succ_now, torch.Tensor):
                        succ_seq[alive_idx, t] = succ_now.view(-1).bool().cpu()[alive_idx]
                    eng_now = info.get("per_env_curr_engaged", None)
                    if isinstance(eng_now, torch.Tensor):
                        eng_b = (eng_now.view(-1).float().cpu() > 0.5)
                        eng_seq[alive_idx, t] = eng_b[alive_idx]

                term_now = _coerce_done(terminated).cpu()
                trunc_now = _coerce_done(truncated).cpu()
                new_done = (term_now | trunc_now) & alive
                if bool(new_done.any()):
                    idx = new_done.nonzero(as_tuple=False).view(-1)
                    term_step[idx] = t
                    succ = info.get("is_success", None)
                    if isinstance(succ, torch.Tensor):
                        success[idx] = succ.view(-1).bool().cpu()[idx]
                    env_done[idx] = True

                state = _resolve_state(env, obs)
                if bool(env_done.all()):
                    break

            # Harvest every env's completed trajectory from this episode.
            for e in range(num_envs):
                coll_frames.append(frames[e])
                coll_returns.append(returns[e])
                coll_term.append(term_step[e].clamp(max=T))
                coll_succ.append(success[e])
                coll_values.append(values[e])
                coll_succ_seq.append(succ_seq[e])
                coll_eng_seq.append(eng_seq[e])
            print(
                f"[record] episode {ep + 1}/{num_episodes} done — "
                f"{len(coll_frames)} trajectories collected",
                flush=True,
            )
    finally:
        set_camera_active(camera, False)

    F = torch.stack(coll_frames)      # (M, T, H, W, 3) — a fresh contiguous copy
    # coll_frames held views into each episode's (num_envs, T, H, W, 3) buffer; now that
    # F owns its own copy, drop those refs (and the last episode's live buffer) so the
    # per-episode frame tensors are freed before grid composition. At high recorder
    # resolution these are gigabytes and otherwise linger through the whole grid/GIF step.
    coll_frames.clear()
    del frames
    R = torch.stack(coll_returns)     # (M,)
    TS = torch.stack(coll_term)       # (M,)
    SU = torch.stack(coll_succ)       # (M,)
    VL = torch.stack(coll_values)     # (M, T)
    SU_SEQ = torch.stack(coll_succ_seq)  # (M, T) per-step success position
    EN_SEQ = torch.stack(coll_eng_seq)   # (M, T) per-step engaged

    # build_grid_video selects best-4 / median-4 / worst-4 by return across all M,
    # and draws the per-step green(success)/orange(engaged) border.
    grid = build_grid_video(
        frames=F, returns=R, term_step=TS, is_success=SU, values=VL,
        engaged=EN_SEQ, success_seq=SU_SEQ,
    )
    # Write mp4 (default) or gif per recorder_cfg.video_format; the extension follows the
    # chosen format regardless of gif_name's suffix.
    fmt = getattr(recorder_cfg, "video_format", "mp4")
    path_base = os.path.join(output_dir, os.path.splitext(gif_name)[0])
    out_path = write_video(grid, path_base, fps=int(recorder_cfg.fps), fmt=fmt)
    print(
        f"[record] wrote {out_path} ({grid.shape[0]} frames, selected from {F.shape[0]} trajectories)",
        flush=True,
    )
    if image_writer is not None:
        write_tb_video(
            image_writer, tag="Video / grid_3x4", grid=grid,
            fps=int(recorder_cfg.fps), global_step=int(global_step),
        )
    return out_path
