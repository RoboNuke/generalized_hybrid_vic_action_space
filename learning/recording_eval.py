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
    # stills_grid_png=False writes ONLY the mp4 below (no PNG montage) — for a pure video request.
    write_png = bool(getattr(recorder_cfg, "stills_grid_png", True))
    if write_png:
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
        if not write_png:
            out_path = vid_path  # nothing else written; return the mp4 path

    return out_path


def collect_annotated_ranked(
    *,
    env: Any,
    agent: Any,
    recorder_cfg: Any,
    camera: Any,
    max_episode_length: int,
    num_trajectories: int,
    output_dir: str,
    gif_name: str = "recording.mp4",
) -> str:
    """Collect >= ``num_trajectories`` full episodes, select best-4 / median-4 / worst-4 by return,
    and write an ANNOTATED 3x4 grid video of the selected 12 — the surface overlays (in-scene keypoint
    balls + force/orientation gauges + top-down path inset) drawn on the RANKED selection.

    This is the annotated viz layered ON TOP of the best/median/worst grid (unlike
    :func:`collect_stills_grid`, which tiles one unranked rollout). Surface-task only (requires
    ``env.viz_snapshot``). Returns the written mp4 path.
    """
    from learning import surface_viz as sv
    from wrappers.recording_grid import select_grid_indices, write_video

    uenv = env.unwrapped
    if not hasattr(uenv, "viz_snapshot"):
        raise RuntimeError("annotated_ranked requires FlatSurfaceFollowEnv (env.viz_snapshot missing).")

    num_envs = int(env.num_envs)
    H, W = int(recorder_cfg.height), int(recorder_cfg.width)
    T = int(max_episode_length)
    if T <= 0:
        raise RuntimeError(f"max_episode_length must be > 0; got {max_episode_length!r}.")
    rows, cols = 3, 4                       # best-4 / median-4 / worst-4 (matches the plain grid video)
    n_sel = rows * cols
    overlays = bool(getattr(recorder_cfg, "surface_overlays", True))
    ball_frac = float(getattr(recorder_cfg, "ball_diameter_frac", 1.6))
    os.makedirs(output_dir, exist_ok=True)
    num_episodes = max(1, math.ceil(int(num_trajectories) / max(1, num_envs)))
    print(f"[record] annotated-ranked: collecting >= {num_trajectories} trajectories "
          f"({num_episodes} ep x {num_envs} envs), then best/median/worst {rows}x{cols}", flush=True)

    # Per-tile STATUS indicator (drawn bottom-right of each tile): in-progress until the trajectory
    # ends, then its terminal cause. Determined at termination from is_success + the fragile wrapper's
    # per-env break flags in env.extras["to_log"].
    _ST_INPROG, _ST_DONE, _ST_BROKE, _ST_LOST = 0, 1, 2, 3
    _STATUS_LABEL = {_ST_INPROG: "in-progress", _ST_DONE: "completed",
                     _ST_BROKE: "broke peg", _ST_LOST: "lost-contact"}
    _STATUS_COLOR = {_ST_INPROG: (180, 180, 185), _ST_DONE: (60, 200, 90),
                     _ST_BROKE: (225, 70, 70), _ST_LOST: (240, 160, 50)}

    def _read_flag(info, key):
        # The fragile wrapper stashes per-env break flags in extras["to_log"], but the reward-
        # decomposition scorer MOVES them into info["per_env_to_log"] and CLEARS to_log every step —
        # so read them from the per-step info dict here (reading uenv.extras would see them cleared).
        petl = info.get("per_env_to_log") if isinstance(info, dict) else None
        v = petl.get(key) if isinstance(petl, dict) else None
        return v.detach().cpu().numpy() if v is not None else np.zeros(num_envs)

    # Per-trajectory stores, indexed GLOBALLY across episodes (like collect_and_record).
    coll_frames: list[torch.Tensor] = []
    coll_returns: list[float] = []
    coll_term: list[int] = []
    coll_succ: list[bool] = []
    coll_status: list[int] = []                       # terminal status per trajectory (see _ST_* below)
    coll_fsq: list[np.ndarray] = []; coll_osq: list[np.ndarray] = []
    coll_fN: list[np.ndarray] = [];  coll_ang: list[np.ndarray] = []
    coll_tru: list[np.ndarray] = []; coll_trv: list[np.ndarray] = []
    coll_trc: list[np.ndarray] = []; coll_tro: list[np.ndarray] = []
    coll_start_uv: list[np.ndarray] = []; coll_goal_uv: list[np.ndarray] = []
    coll_half: list[float] = [];     coll_desforce: list[float] = []

    # In-scene markers persist across episodes; positions/colours are re-set each reset.
    markers = goal_marker = pace_marker = None
    k = 0

    set_camera_active(camera, True)
    try:
        for ep in range(num_episodes):
            frames = torch.zeros((num_envs, T, H, W, 3), dtype=torch.uint8)
            returns = torch.zeros(num_envs, dtype=torch.float32)
            term_step = np.full(num_envs, T - 1, dtype=np.int64)
            success = np.zeros(num_envs, dtype=bool)
            status = np.full(num_envs, _ST_INPROG, dtype=np.int64)      # terminal status per env
            fsq = np.zeros((num_envs, T), np.float32); osq = np.zeros((num_envs, T), np.float32)
            fN = np.zeros((num_envs, T), np.float32);  ang = np.zeros((num_envs, T), np.float32)
            tru = np.zeros((num_envs, T), np.float32); trv = np.zeros((num_envs, T), np.float32)
            trc = np.zeros((num_envs, T), bool);       tro = np.zeros((num_envs, T), bool)
            env_done = np.zeros(num_envs, dtype=bool)
            tracker = None; const = {}; des_force = None
            base = start_w_env = path_dir_np = goal_lift = None
            cur_s_ref = np.zeros(num_envs, np.float32)

            obs, _ = env.reset()
            state = _resolve_state(env, obs)

            # Per-episode marker/tracker setup BEFORE the capture loop, from a POST-RESET snapshot,
            # so the very FIRST captured frame already shows the fresh overlay (setpoint at k0, no
            # keypoints coloured, pace at the start). Previously this ran inside the t==0 branch
            # AFTER the first step, so frame 0 rendered the PREVIOUS episode's stale markers — the
            # peg looked reset but the keypoints/setpoint showed the prior run's END state, which
            # read as "the task starts midway / keypoints already marked before the video starts".
            snap = uenv.viz_snapshot()
            spacing = float(snap["keypoint_spacing"])
            k = int(snap["keypoints_total"].min().item())
            radius = spacing * ball_frac / 2.0
            normal = snap["surface_normal"].numpy()
            env_origins = uenv.scene.env_origins.detach().cpu().numpy()
            base = sv.keypoint_world_positions(snap["start_w"], snap["path_dir"], spacing, k)
            base = base + env_origins[:, None, :]
            goal_radius = radius * 4.0
            goal_lift = normal * goal_radius
            if markers is None:                                          # create USD prims once
                markers = sv.KeypointBallMarkers("/World/Visuals/surface_keypoints", radius=radius)
                goal_marker = sv.GoalMarker("/World/Visuals/surface_goal", radius=goal_radius)
                pace_marker = sv.GoalMarker("/World/Visuals/surface_pace", radius=goal_radius, opacity=0.4)
            markers.set_positions((base + normal[:, None, :] * radius).reshape(-1, 3))
            start_w_env = snap["start_w"].numpy() + env_origins
            path_dir_np = snap["path_dir"].numpy()
            tracker = sv.KeypointStatusTracker(num_envs, k, spacing)
            des_force = np.maximum(snap["desired_force_N"].numpy(), 1e-6)
            start_w = snap["start_w"].numpy(); goal_w = snap["goal_w"].numpy()
            u_dir = snap["path_dir"].numpy(); v_dir = snap["d_lat"].numpy()
            center = 0.5 * (start_w + goal_w)
            su, svv = sv.project_uv(start_w, center, u_dir, v_dir)
            gu, gvv = sv.project_uv(goal_w, center, u_dir, v_dir)
            half = 0.5 * snap["path_length"].numpy()
            const = dict(center=center, u=u_dir, v=v_dir, start_uv=np.stack([su, svv], 1),
                         goal_uv=np.stack([gu, gvv], 1), half=half)

            for t in range(T):
                markers.update(tracker.marker_indices())                 # fresh at t=0 (set BEFORE the step renders)
                gidx = np.clip(tracker.setpoint_idx - 1, 0, k - 1)
                goal_marker.update(base[np.arange(num_envs), gidx] + goal_lift)
                pace_marker.update(start_w_env + cur_s_ref[:, None] * path_dir_np + goal_lift)
                actions, _ = agent.act(obs, state, timestep=10**9, timesteps=10**9)
                obs, reward, terminated, truncated, info = env.step(actions)
                rgb = read_camera_rgb(camera)
                snap = uenv.viz_snapshot()

                tu, tv = sv.project_uv(snap["tip_w"].numpy(), const["center"], const["u"], const["v"])
                over = (np.abs(tu) <= const["half"]) & (np.abs(tv) <= const["half"])
                alive = ~env_done
                fsq[alive, t] = snap["force_squash"].numpy()[alive]
                osq[alive, t] = snap["orn_squash"].numpy()[alive]
                fN[alive, t] = snap["force_N"].numpy()[alive]
                ang[alive, t] = snap["angle_dev_deg"].numpy()[alive]
                tru[alive, t] = tu[alive]; trv[alive, t] = tv[alive]
                trc[alive, t] = snap["in_contact"].numpy()[alive]
                tro[alive, t] = over[alive]
                alive_idx = np.nonzero(alive)[0]
                if alive_idx.size:
                    ii = torch.from_numpy(alive_idx)
                    frames[ii, t] = rgb[ii]
                    returns[ii] += reward.detach().view(-1).float().cpu()[ii]
                cur_s_ref = snap["s_ref"].numpy()
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
                    # Terminal status: success -> completed; else the fragile break cause (loss-of-contact
                    # takes precedence over the force break); a plain time-out leaves it in-progress.
                    _cl = _read_flag(info, "Fragile / Contact Loss") > 0.5
                    _pb = _read_flag(info, "Fragile / Peg Break") > 0.5
                    for e in idx:
                        if success[e]:
                            status[e] = _ST_DONE
                        elif _cl[e]:
                            status[e] = _ST_LOST
                        elif _pb[e]:
                            status[e] = _ST_BROKE
                    env_done[idx] = True
                state = _resolve_state(env, obs)
                if env_done.all():
                    break

            for e in range(num_envs):                                    # harvest this episode's trajectories
                coll_frames.append(frames[e].clone())
                coll_returns.append(float(returns[e])); coll_term.append(int(term_step[e]))
                coll_succ.append(bool(success[e])); coll_status.append(int(status[e]))
                coll_fsq.append(fsq[e].copy()); coll_osq.append(osq[e].copy())
                coll_fN.append(fN[e].copy());   coll_ang.append(ang[e].copy())
                coll_tru.append(tru[e].copy()); coll_trv.append(trv[e].copy())
                coll_trc.append(trc[e].copy()); coll_tro.append(tro[e].copy())
                coll_start_uv.append(const["start_uv"][e].copy()); coll_goal_uv.append(const["goal_uv"][e].copy())
                coll_half.append(float(const["half"][e])); coll_desforce.append(float(des_force[e]))
            del frames
            print(f"[record] episode {ep + 1}/{num_episodes} done — {len(coll_frames)} trajectories", flush=True)
    finally:
        set_camera_active(camera, False)

    R = torch.tensor(coll_returns)
    sel = select_grid_indices(R).tolist()                                # 12 indices: best-4/median-4/worst-4
    print(f"[record] selected {len(sel)}/{len(coll_frames)} by return "
          f"(best={[round(coll_returns[j], 1) for j in sel[:4]]} "
          f"worst={[round(coll_returns[j], 1) for j in sel[-4:]]})", flush=True)

    # Animate the selected trajectories; insets cached at stride K (matplotlib is the bottleneck).
    K = 3
    # Render the FULL episode length (not just up to the longest selected trajectory) so every video
    # is the same task-length clip; trajectories that ended early freeze at their last frame (q below).
    Tmax = T
    # Freeze/trim index: a trajectory that ends at step `coll_term[j]` (fragile break, loss-of-contact,
    # or time-out) has that FINAL captured frame/snapshot contaminated — Isaac Lab resets the env
    # in-step on terminated|truncated, so the camera + viz_snapshot at `coll_term[j]` already show the
    # POST-reset state (fresh random plate yaw, tip back at the start). Left in, it renders as a tile
    # that "jumps mid-plate" / freezes on a rotated frame. Freeze one frame earlier so the tile holds
    # its last clean pre-reset state instead. (Costs at most one frame off a trajectory that never
    # reset, which is imperceptible.)
    disp_term = {j: max(0, int(coll_term[j]) - 1) for j in sel}
    inset_cache: dict[int, list] = {}
    if overlays:
        for j in sel:
            di = disp_term[j]; cache = []
            for tt in range(0, di + 1, K):
                cache.append(sv.topdown_inset(
                    coll_tru[j][: tt + 1], coll_trv[j][: tt + 1], coll_trc[j][: tt + 1], coll_tro[j][: tt + 1],
                    coll_start_uv[j], coll_goal_uv[j], coll_half[j], coll_half[j]))
            inset_cache[j] = cache or [None]

    fmt = getattr(recorder_cfg, "video_format", "mp4")
    video = None
    for t in range(Tmax):
        tiles = []
        for j in sel:
            di = disp_term[j]; q = min(t, di)                            # freeze finished trajectories (pre-reset)
            frame = coll_frames[j][q].numpy()
            border = (45, 200, 95) if (coll_succ[j] and t >= di) else None
            # Status indicator (bottom-right): in-progress until the tile freezes, then its terminal cause.
            _st = coll_status[j] if t >= di else _ST_INPROG
            if overlays:
                ins = inset_cache[j][min(q // K, len(inset_cache[j]) - 1)]
                tiles.append(sv.compose_tile(
                    frame, float(coll_fsq[j][q]), float(coll_osq[j][q]), ins, border,
                    force_text=f"{coll_fN[j][q]:.1f}N", orn_text=f"{coll_ang[j][q]:+.0f}°",
                    force_fill=float(coll_fN[j][q] / (2.0 * coll_desforce[j])),
                    orn_fill=float(coll_ang[j][q] / 30.0),
                    status_label=_STATUS_LABEL[_st], status_color=_STATUS_COLOR[_st]))
            else:
                tiles.append(sv.compose_tile(frame, 0, 0, None, border,
                                             status_label=_STATUS_LABEL[_st], status_color=_STATUS_COLOR[_st]))
        gframe = sv.montage(tiles, rows, cols)
        if video is None:
            video = np.zeros((Tmax,) + gframe.shape, dtype=np.uint8)
        video[t] = gframe

    vid_base = os.path.join(output_dir, os.path.splitext(gif_name)[0])
    vid_path = write_video(video, vid_base, fps=int(recorder_cfg.fps), fmt=fmt)
    print(f"[record] wrote annotated-ranked video {vid_path} "
          f"({Tmax} frames, {rows}x{cols} of {len(coll_frames)} trajectories)", flush=True)
    return vid_path


def collect_reset_snapshots(
    *,
    env: Any,
    agent: Any,                                 # accepted for call-site symmetry; UNUSED (no policy)
    recorder_cfg: Any,
    camera: Any,
    max_episode_length: int,                    # accepted for symmetry; unused
    num_trajectories: int,                      # accepted for symmetry; unused
    output_dir: str,
    gif_name: str = "initial_conditions.mp4",
) -> str:
    """Reset the env ``reset_snapshots_count`` times; for EACH reset HOLD the robot at its spawn
    fingertip pose (no policy) and step the sim ``reset_snapshots_hold_s`` * fps physics steps,
    capturing every step — so any INITIAL spawn DYNAMICS (a first-step pop, contact jitter, or slow
    settle) are visible even though the robot is commanded to stay put. Designed for ONE env per view
    (set num_envs=1); with >1 env it tiles them.

    The top-down inset ACCUMULATES the peg-tip trace over the held steps: a stable spawn stays a dot on
    the red-x START, drift/bounce draws a short path. red-x = START, green-o = GOAL. The gauge read-outs
    show the (near-zero) spawn contact force and the tip height above the surface (mm). The in-scene USD
    markers (keypoint balls + goal + setpoint) are drawn too. Surface-task only. Returns the mp4 path.
    """
    import math
    from learning import surface_viz as sv
    from wrappers.recording_grid import write_video

    uenv = env.unwrapped
    if not hasattr(uenv, "viz_snapshot"):
        raise RuntimeError("reset_snapshots requires FlatSurfaceFollowEnv (env.viz_snapshot missing).")

    num_envs = int(env.num_envs)
    overlays = bool(getattr(recorder_cfg, "surface_overlays", True))
    ball_frac = float(getattr(recorder_cfg, "ball_diameter_frac", 1.0))
    n_resets = int(getattr(recorder_cfg, "reset_snapshots_count", 8))
    hold_s = float(getattr(recorder_cfg, "reset_snapshots_hold_s", 1.0))
    fps = int(recorder_cfg.fps)
    hold_frames = max(1, int(round(hold_s * fps)))
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    dt = float(uenv.physics_dt)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[record] reset-dynamics: {n_resets} resets x {hold_frames} held steps, "
          f"{num_envs} env(s) ({rows}x{cols}), robot held at spawn pose", flush=True)

    # env.reset() only re-randomizes ONCE on this env stack; repeated calls are a no-op (the varied
    # rollouts elsewhere come from mid-episode auto-resets, which route through _reset_idx). So drive a
    # true fresh spawn each iteration by calling the (monkeypatched) _reset_idx on ALL envs directly.
    import torch as _torch
    all_ids = _torch.arange(uenv.num_envs, device=uenv.device)

    markers = goal_marker = pace_marker = None
    set_camera_active(camera, True)
    out_frames: list[np.ndarray] = []
    try:
        env.reset()                                                    # one-time init reset
        for r in range(n_resets):
            uenv._reset_idx(all_ids)                                    # force a fresh full re-randomization
            uenv._compute_intermediate_values(dt)
            snap = uenv.viz_snapshot()

            spacing = float(snap["keypoint_spacing"])
            k = int(snap["keypoints_total"].min().item())
            radius = spacing * ball_frac / 2.0
            normal = snap["surface_normal"].numpy()
            env_origins = uenv.scene.env_origins.detach().cpu().numpy()
            base = sv.keypoint_world_positions(snap["start_w"], snap["path_dir"], spacing, k)
            base = base + env_origins[:, None, :]
            goal_radius = radius * 4.0
            goal_lift = normal * goal_radius
            if markers is None:                                          # create USD prims once
                markers = sv.KeypointBallMarkers("/World/Visuals/surface_keypoints", radius=radius)
                goal_marker = sv.GoalMarker("/World/Visuals/surface_goal", radius=goal_radius)
                pace_marker = sv.GoalMarker("/World/Visuals/surface_pace", radius=goal_radius, opacity=0.4)
            markers.set_positions((base + normal[:, None, :] * radius).reshape(-1, 3))
            tracker = sv.KeypointStatusTracker(num_envs, k, spacing)     # all keypoints fresh at reset
            markers.update(tracker.marker_indices())
            gidx = np.full(num_envs, k - 1)
            goal_marker.update(base[np.arange(num_envs), gidx] + goal_lift)
            start_w_env = snap["start_w"].numpy() + env_origins
            pace_marker.update(start_w_env + snap["s_ref"].numpy()[:, None] * snap["path_dir"].numpy() + goal_lift)

            # Per-env inset frame (fixed for this reset) + spawn tip, for the accumulating trace.
            start_w = snap["start_w"].numpy(); goal_w = snap["goal_w"].numpy()
            u_dir = snap["path_dir"].numpy(); v_dir = snap["d_lat"].numpy()
            center = 0.5 * (start_w + goal_w)
            su, svv = sv.project_uv(start_w, center, u_dir, v_dir)
            gu, gvv = sv.project_uv(goal_w, center, u_dir, v_dir)
            half = 0.5 * snap["path_length"].numpy()
            tu0, tv0 = sv.project_uv(snap["tip_w"].numpy(), center, u_dir, v_dir)

            # HOLD target = the spawn fingertip pose. Commanding it every step keeps the robot put, so
            # anything that moves is spawn dynamics (physics), not control.
            hold_pos = uenv.fingertip_midpoint_pos.clone()
            hold_quat = uenv.fingertip_midpoint_quat.clone()
            tr_u = [[float(tu0[e])] for e in range(num_envs)]
            tr_v = [[float(tv0[e])] for e in range(num_envs)]
            tr_c = [[bool(snap["in_contact"].numpy()[e])] for e in range(num_envs)]
            tr_o = [[bool(abs(tu0[e]) <= half[e] and abs(tv0[e]) <= half[e])] for e in range(num_envs)]
            max_drift = np.zeros(num_envs)

            for s in range(hold_frames):
                uenv.generate_ctrl_signals(
                    ctrl_target_fingertip_midpoint_pos=hold_pos,
                    ctrl_target_fingertip_midpoint_quat=hold_quat,
                    ctrl_target_gripper_dof_pos=0.0,
                )
                uenv.step_sim_no_action()                               # steps physics (+ refreshes state); no render
                for _ in range(2):                                      # render for the camera (1-frame latency)
                    uenv.sim.render()
                try:
                    camera.update(0.0)
                except TypeError:
                    camera.update()
                rgb = read_camera_rgb(camera)                           # (E,H,W,3) uint8
                snap = uenv.viz_snapshot()
                tu, tv = sv.project_uv(snap["tip_w"].numpy(), center, u_dir, v_dir)
                fsq = snap["force_squash"].numpy(); osq = snap["orn_squash"].numpy()
                fN = snap["force_N"].numpy(); tipd = snap["tip_surface_dist"].numpy()
                incontact = snap["in_contact"].numpy()

                tiles = []
                for e in range(num_envs):
                    tr_u[e].append(float(tu[e])); tr_v[e].append(float(tv[e]))
                    tr_c[e].append(bool(incontact[e]))
                    tr_o[e].append(bool(abs(tu[e]) <= half[e] and abs(tv[e]) <= half[e]))
                    max_drift[e] = max(max_drift[e], float(np.hypot(tu[e] - tu0[e], tv[e] - tv0[e])))
                    frame = rgb[e].numpy()
                    if overlays:
                        ins = sv.topdown_inset(
                            np.array(tr_u[e]), np.array(tr_v[e]),
                            np.array(tr_c[e]), np.array(tr_o[e]),
                            np.array([su[e], svv[e]]), np.array([gu[e], gvv[e]]),
                            float(half[e]), float(half[e]))
                        tiles.append(sv.compose_tile(
                            frame, float(fsq[e]), float(osq[e]), ins, None,
                            force_text=f"{fN[e]:.1f}N", orn_text=f"{tipd[e]*1000:.1f}mm",
                            force_fill=float(max(fN[e], 0.0) / 10.0), orn_fill=0.0))
                    else:
                        tiles.append(frame)
                out_frames.append(sv.montage(tiles, rows, cols))
            print(f"[record] reset {r + 1}/{n_resets}: spawn tip height (mm) "
                  f"mean={snap['tip_surface_dist'].numpy().mean()*1000:.1f}  "
                  f"max in-plane tip drift over hold (mm) = {max_drift.max()*1000:.2f}", flush=True)
    finally:
        set_camera_active(camera, False)

    video = np.stack(out_frames)
    vid_base = os.path.join(output_dir, os.path.splitext(gif_name)[0])
    vid_path = write_video(video, vid_base, fps=fps, fmt=getattr(recorder_cfg, "video_format", "mp4"))
    print(f"[record] wrote reset-snapshots video {vid_path} "
          f"({n_resets} resets x {hold_frames} frames, {rows}x{cols} envs)", flush=True)
    return vid_path
