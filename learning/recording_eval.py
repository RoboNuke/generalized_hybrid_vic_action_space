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

import numpy as np
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


def collect_stills_grid(
    *,
    env: Any,
    agent: Any,
    recorder_cfg: Any,
    camera: Any,
    max_episode_length: int,
    output_dir: str,
    out_name: str = "surface_stills.png",
) -> str:
    """Surface-task recorder: run ONE rollout and write a rows x cols PNG montage of annotated
    still frames (one env per tile) with keypoint balls (in-scene), force + orientation gauges, and
    a top-down path inset. Requires the env to expose ``viz_snapshot()`` (FlatSurfaceFollowEnv)."""
    from learning import surface_viz as sv
    from PIL import Image

    uenv = env.unwrapped
    if not hasattr(uenv, "viz_snapshot"):
        raise RuntimeError("stills_grid requires FlatSurfaceFollowEnv (env.viz_snapshot missing).")

    rows, cols = int(recorder_cfg.grid_rows), int(recorder_cfg.grid_cols)
    overlays = bool(getattr(recorder_cfg, "surface_overlays", True))
    n_tiles = rows * cols
    num_envs = int(env.num_envs)
    if num_envs < n_tiles:
        print(f"[record] WARNING: num_envs={num_envs} < grid {rows}x{cols}={n_tiles}; "
              f"tiling only {num_envs} envs.", flush=True)
        n_tiles = num_envs
    H, W = int(recorder_cfg.height), int(recorder_cfg.width)
    T = int(max_episode_length)
    os.makedirs(output_dir, exist_ok=True)

    # Per-step captures (kept for every env; we display each env's last-alive frame).
    frames = torch.zeros((num_envs, T, H, W, 3), dtype=torch.uint8)
    force_sq = np.zeros((num_envs, T), dtype=np.float32)      # gauge fill/colour (reward closeness)
    orn_sq = np.zeros((num_envs, T), dtype=np.float32)
    force_N = np.zeros((num_envs, T), dtype=np.float32)       # gauge read-out: measured force (N)
    angle_dev = np.zeros((num_envs, T), dtype=np.float32)     # gauge read-out: deg off desired angle
    tr_u = np.zeros((num_envs, T), dtype=np.float32)
    tr_v = np.zeros((num_envs, T), dtype=np.float32)
    tr_c = np.zeros((num_envs, T), dtype=bool)
    tr_o = np.zeros((num_envs, T), dtype=bool)
    term_step = np.full(num_envs, T - 1, dtype=np.int64)
    success = np.zeros(num_envs, dtype=bool)
    env_done = np.zeros(num_envs, dtype=bool)

    markers = None
    tracker = None
    const = {}  # per-env plate constants for the inset (filled at t=0)
    cur_s_ref = np.zeros(num_envs, dtype=np.float32)  # latest time-based pace arc length (for pace marker)

    set_camera_active(camera, True)
    obs, _ = env.reset()
    state = _resolve_state(env, obs)
    try:
        for t in range(T):
            if markers is not None:                                  # colour balls from status so far
                markers.update(tracker.marker_indices())
                gidx = np.clip(tracker.setpoint_idx - 1, 0, k - 1)   # current goal keypoint per env
                goal_marker.update(base[np.arange(num_envs), gidx] + goal_lift)
                pace_marker.update(start_w_env + cur_s_ref[:, None] * path_dir_np + goal_lift)
            actions, _ = agent.act(obs, state, timestep=10**9, timesteps=10**9)
            obs, reward, terminated, truncated, info = env.step(actions)
            rgb = read_camera_rgb(camera)                            # (num_envs,H,W,3)
            snap = uenv.viz_snapshot()

            if t == 0:
                spacing = float(snap["keypoint_spacing"])
                k = int(snap["keypoints_total"].min().item())
                # Ball diameter is a fraction of the keypoint spacing (spec 0.5; enlarged for
                # legibility). start_world/path_dir are ENV-LOCAL (like env.goal_world in the control
                # wrapper, which adds env_origins) so shift into world frame; lift onto the surface.
                ball_frac = float(getattr(recorder_cfg, "ball_diameter_frac", 1.6))
                radius = spacing * ball_frac / 2.0
                normal = snap["surface_normal"].numpy()                       # (E,3)
                env_origins = uenv.scene.env_origins.detach().cpu().numpy()   # (E,3)
                base = sv.keypoint_world_positions(snap["start_w"], snap["path_dir"], spacing, k)
                base = base + env_origins[:, None, :]                         # (E,k,3) env-world, on surface
                markers = sv.KeypointBallMarkers("/World/Visuals/surface_keypoints", radius=radius)
                markers.set_positions((base + normal[:, None, :] * radius).reshape(-1, 3))
                # Separate, exaggerated (4x) moving marker for the CURRENT goal keypoint.
                goal_radius = radius * 4.0
                goal_lift = normal * goal_radius                             # (E,3)
                goal_marker = sv.GoalMarker("/World/Visuals/surface_goal", radius=goal_radius)
                # Same big purple sphere, slightly transparent, for the TIME-BASED pace setpoint.
                pace_marker = sv.GoalMarker("/World/Visuals/surface_pace", radius=goal_radius, opacity=0.4)
                start_w_env = snap["start_w"].numpy() + env_origins          # (E,3) path start, on surface
                path_dir_np = snap["path_dir"].numpy()                       # (E,3)
                tracker = sv.KeypointStatusTracker(num_envs, k, spacing)
                des_force = np.maximum(snap["desired_force_N"].numpy(), 1e-6)  # (E,) force-gauge scale
                # Plate frame per env for the top-down inset.
                start_w = snap["start_w"].numpy(); goal_w = snap["goal_w"].numpy()
                u_dir = snap["path_dir"].numpy(); v_dir = snap["d_lat"].numpy()
                center = 0.5 * (start_w + goal_w)
                su, svv = sv.project_uv(start_w, center, u_dir, v_dir)
                gu, gvv = sv.project_uv(goal_w, center, u_dir, v_dir)
                half = 0.5 * snap["path_length"].numpy()             # square plate: half_u == half_v
                const = dict(center=center, u=u_dir, v=v_dir, start_uv=np.stack([su, svv], 1),
                             goal_uv=np.stack([gu, gvv], 1), half=half)

            # tip projection for the inset trace
            tu, tv = sv.project_uv(snap["tip_w"].numpy(), const["center"], const["u"], const["v"])
            over = (np.abs(tu) <= const["half"]) & (np.abs(tv) <= const["half"])
            alive = ~env_done
            force_sq[alive, t] = snap["force_squash"].numpy()[alive]
            orn_sq[alive, t] = snap["orn_squash"].numpy()[alive]
            force_N[alive, t] = snap["force_N"].numpy()[alive]
            angle_dev[alive, t] = snap["angle_dev_deg"].numpy()[alive]
            tr_u[alive, t] = tu[alive]; tr_v[alive, t] = tv[alive]
            tr_c[alive, t] = snap["in_contact"].numpy()[alive]
            tr_o[alive, t] = over[alive]
            alive_idx = np.nonzero(alive)[0]
            if alive_idx.size:
                frames[torch.from_numpy(alive_idx), t] = rgb[torch.from_numpy(alive_idx)]

            cur_s_ref = snap["s_ref"].numpy()                        # for next step's pace-marker update
            tracker.update(snap["progress"].numpy(), snap["in_contact"].numpy())

            term_now = _coerce_done(terminated).cpu().numpy()
            trunc_now = _coerce_done(truncated).cpu().numpy()
            new_done = (term_now | trunc_now) & alive
            if new_done.any():
                idx = np.nonzero(new_done)[0]
                term_step[idx] = t
                succ = info.get("is_success", None)
                if isinstance(succ, torch.Tensor):
                    success[idx] = succ.view(-1).bool().cpu().numpy()[idx]
                env_done[idx] = True
            state = _resolve_state(env, obs)
            if env_done.all():
                break
    finally:
        set_camera_active(camera, False)

    # Compose one annotated tile per env (its last-alive frame), then montage.
    tiles = []
    for e in range(n_tiles):
        di = int(term_step[e])
        frame = frames[e, di].numpy()
        inset = None
        if overlays:
            inset = sv.topdown_inset(
                tr_u[e, : di + 1], tr_v[e, : di + 1], tr_c[e, : di + 1], tr_o[e, : di + 1],
                const["start_uv"][e], const["goal_uv"][e], float(const["half"][e]), float(const["half"][e]),
            )
        border = (45, 200, 95) if success[e] else None
        if overlays:
            tiles.append(sv.compose_tile(
                frame, float(force_sq[e, di]), float(orn_sq[e, di]), inset, border,
                force_text=f"{force_N[e, di]:.1f}N", orn_text=f"{angle_dev[e, di]:+.0f}°",
                force_fill=float(force_N[e, di] / (2.0 * des_force[e])),
                orn_fill=float(angle_dev[e, di] / 30.0)))
        else:
            tiles.append(frame if border is None else sv.compose_tile(frame, 0, 0, None, border))

    grid = sv.montage(tiles, rows, cols)
    out_path = os.path.join(output_dir, out_name)
    Image.fromarray(grid).save(out_path)
    print(f"[record] wrote stills grid {out_path} "
          f"({rows}x{cols} tiles, {int(success.sum())}/{n_tiles} succeeded)", flush=True)

    # Optional full mp4 of the SAME rollout: per-frame gauges + a top-down path that grows over time
    # (the in-scene keypoint balls animate for free in the captured frames). The matplotlib inset is
    # cached at a stride so we render ~T/K of them per env instead of one per frame.
    if bool(getattr(recorder_cfg, "stills_grid_video", False)):
        from wrappers.recording_grid import write_video

        K = 3
        Tmax = int(term_step[:n_tiles].max()) + 1
        inset_cache = []  # per env: inset image at times 0, K, 2K, ... (path up to that time)
        for e in range(n_tiles):
            di = int(term_step[e])
            cache = []
            if overlays:
                for tt in range(0, di + 1, K):
                    cache.append(sv.topdown_inset(
                        tr_u[e, : tt + 1], tr_v[e, : tt + 1], tr_c[e, : tt + 1], tr_o[e, : tt + 1],
                        const["start_uv"][e], const["goal_uv"][e],
                        float(const["half"][e]), float(const["half"][e])))
            inset_cache.append(cache or [None])

        video = None
        for t in range(Tmax):
            tiles_t = []
            for e in range(n_tiles):
                di = int(term_step[e])
                q = min(t, di)                                  # freeze finished envs on their last frame
                frame = frames[e, q].numpy()
                border = (45, 200, 95) if (success[e] and t >= di) else None
                if overlays:
                    ins = inset_cache[e][min(q // K, len(inset_cache[e]) - 1)]
                    tiles_t.append(sv.compose_tile(
                        frame, float(force_sq[e, q]), float(orn_sq[e, q]), ins, border,
                        force_text=f"{force_N[e, q]:.1f}N", orn_text=f"{angle_dev[e, q]:+.0f}°",
                        force_fill=float(force_N[e, q] / (2.0 * des_force[e])),
                        orn_fill=float(angle_dev[e, q] / 30.0)))
                else:
                    tiles_t.append(frame if border is None else sv.compose_tile(frame, 0, 0, None, border))
            gframe = sv.montage(tiles_t, rows, cols)
            if video is None:
                video = np.zeros((Tmax,) + gframe.shape, dtype=np.uint8)
            video[t] = gframe
        vid_base = os.path.join(output_dir, os.path.splitext(out_name)[0])
        vid_path = write_video(video, vid_base, fps=int(recorder_cfg.fps), fmt="mp4")
        print(f"[record] wrote stills video {vid_path} ({Tmax} frames)", flush=True)

    return out_path
