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


def _collect_plain(
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


def _build_readouts(force_sq, fN, shr, desF, pac, vdes, xtr, tol):
    """Four raw-value read-out lines for the bottom-right of a tile, each coloured green->red by how
    close it is to ideal: current normal force, current shear force, along-track pace, cross-track
    error. Returns a list of ``(text, rgb)`` for ``surface_viz.compose_tile(readouts=...)``.

    Closeness: normal force reuses the reward's force squash; shear is ideal at 0 (scaled by the
    desired normal force); pace is ideal at the desired speed; cross-track is ideal at 0 (scaled by
    the keypoint on-track tolerance)."""
    from learning import surface_viz as sv

    ref_shear = max(float(desF), 1.0)
    shear_close = float(np.clip(1.0 - abs(float(shr)) / ref_shear, 0.0, 1.0))
    vdes_ = max(float(vdes), 1e-6)
    pace_close = float(np.clip(1.0 - abs(float(pac) - float(vdes)) / vdes_, 0.0, 1.0))
    tol_ = max(float(tol), 1e-6)
    xtr_close = float(np.clip(1.0 - abs(float(xtr)) / tol_, 0.0, 1.0))
    return [
        (f"N: {float(fN):.1f}N",            sv.closeness_color(float(force_sq))),
        (f"Shear: {float(shr):.1f}N",       sv.closeness_color(shear_close)),
        (f"Pace: {float(pac) * 100:.1f}cm/s", sv.closeness_color(pace_close)),
        (f"XTrk: {float(xtr) * 1000:+.0f}mm", sv.closeness_color(xtr_close)),
    ]


def _build_kp_counts(status_row):
    """Per-keypoint-outcome tally for the TOP-RIGHT of a tile — one coloured line per outcome plus a
    remaining (unpassed) line. ``status_row`` is this tile's (k,) status codes (0..4). Returns a list
    of ``(text, rgb)`` for ``surface_viz.compose_tile(kp_counts=...)``; colours match STATUS_RGB so
    each line reads as the same colour as its keypoint balls/circles."""
    from learning import surface_viz as sv

    s = np.asarray(status_row).reshape(-1)
    rgb = sv.STATUS_RGB
    return [
        (f"Achieved: {int((s == 1).sum())}",   tuple(int(c) for c in rgb[1])),
        (f"No contact: {int((s == 2).sum())}", tuple(int(c) for c in rgb[2])),
        (f"Off track: {int((s == 3).sum())}",  tuple(int(c) for c in rgb[3])),
        (f"Both off: {int((s == 4).sum())}",   tuple(int(c) for c in rgb[4])),
        (f"Remaining: {int((s == 0).sum())}",  tuple(int(c) for c in rgb[0])),
    ]


def _force_bar(fN, desF, break_force):
    """Force-bar fill fraction + yellow-tick fraction. Fragile peg (break_force given): the bar TOP
    is the break force and the yellow tick marks the desired force (off-centre). Non-fragile: legacy
    scaling with the desired force at mid-bar. Returns ``(fill, target_frac)`` both in [0, 1]-ish."""
    desF = max(float(desF), 1e-6)
    if break_force is not None and float(break_force) > 1e-6:
        brk = float(break_force)
        return float(fN) / brk, desF / brk
    return float(fN) / (2.0 * desF), 0.5


def _keypoint_uv(start_uv, goal_uv, half, spacing, k):
    """Per-env keypoint positions in plate (u, v) coordinates: keypoint j (1..k) sits at arc length
    j*spacing along the start->goal segment. Returns ``(E, k, 2)``."""
    start_uv = np.asarray(start_uv, dtype=np.float64)          # (E,2)
    goal_uv = np.asarray(goal_uv, dtype=np.float64)            # (E,2)
    L = np.maximum(2.0 * np.asarray(half, dtype=np.float64), 1e-6)   # (E,) path length
    js = np.arange(1, k + 1, dtype=np.float64)                 # (k,)
    frac = np.clip((js[None, :] * float(spacing)) / L[:, None], 0.0, 1.0)   # (E,k)
    return start_uv[:, None, :] + frac[:, :, None] * (goal_uv - start_uv)[:, None, :]


def _collect_surface(
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
    grid_select = str(getattr(recorder_cfg, "grid_select", "ranked"))
    if grid_select == "all":
        # Tile ALL envs from one episode (square-ish) — for random/untrained rollouts where the
        # best/median/worst ranking is meaningless. num_trajectories should == num_envs.
        cols = int(math.ceil(math.sqrt(num_envs)))
        rows = int(math.ceil(num_envs / cols))
    else:
        rows, cols = 3, 4                   # best-4 / median-4 / worst-4 (matches the plain grid video)
    n_sel = rows * cols
    overlays = True
    ball_frac = float(getattr(recorder_cfg, "ball_diameter_frac", 1.6))
    os.makedirs(output_dir, exist_ok=True)
    num_episodes = max(1, math.ceil(int(num_trajectories) / max(1, num_envs)))
    print(f"[record] annotated-ranked: collecting >= {num_trajectories} trajectories "
          f"({num_episodes} ep x {num_envs} envs), then best/median/worst {rows}x{cols}", flush=True)

    # Per-tile STATUS indicator (drawn bottom-right of each tile): in-progress until the trajectory
    # ends, then its terminal cause. Determined at termination from is_success + the fragile wrapper's
    # per-env break flags in env.extras["to_log"].
    _ST_INPROG, _ST_DONE, _ST_BROKE, _ST_LOST, _ST_TRAVERSED, _ST_TIMEOUT = 0, 1, 2, 3, 4, 5
    _STATUS_LABEL = {_ST_INPROG: "in-progress", _ST_DONE: "success",
                     _ST_BROKE: "broke peg", _ST_LOST: "lost-contact",
                     _ST_TRAVERSED: "traversed <90%", _ST_TIMEOUT: "timeout"}
    _STATUS_COLOR = {_ST_INPROG: (180, 180, 185), _ST_DONE: (60, 200, 90),
                     _ST_BROKE: (225, 70, 70), _ST_LOST: (240, 160, 50),
                     _ST_TRAVERSED: (235, 205, 60), _ST_TIMEOUT: (120, 130, 200)}

    def _read_flag(info, key):
        # The fragile wrapper publishes its per-env break flags in info["per_env_episode_stat"] (the
        # per-episode channel). At an env's terminal step that env's flag is 1 iff it ended in this
        # break cause, so reading it here for the just-finished env gives the terminal status.
        pes = info.get("per_env_episode_stat") if isinstance(info, dict) else None
        v = pes.get(key) if isinstance(pes, dict) else None
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
    coll_shr: list[np.ndarray] = []; coll_xtr: list[np.ndarray] = []      # shear force (N), cross-track (m)
    coll_pac: list[np.ndarray] = []; coll_kpst: list[np.ndarray] = []     # along-track speed (m/s), per-frame kp status
    coll_kp_uv: list[np.ndarray] = []                                     # (k,2) keypoint plate positions
    coll_break: list = []                                                 # per-traj normal break force (N) or None
    coll_tol: list[float] = [];      coll_vdes: list[float] = []          # on-track tol (m), desired speed (m/s)

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
            shr = np.zeros((num_envs, T), np.float32); xtr = np.zeros((num_envs, T), np.float32)
            pac = np.zeros((num_envs, T), np.float32)
            env_done = np.zeros(num_envs, dtype=bool)
            cur_status = cur_setpoint = None; const = {}; des_force = None
            base = start_w_env = path_dir_np = goal_lift = None
            cur_s_ref = np.zeros(num_envs, np.float32)

            obs, _ = env.reset()
            state = _resolve_state(env, obs)

            # Per-episode marker/status setup BEFORE the capture loop, from a POST-RESET snapshot,
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
            # Keypoint status/goal come straight from the env (authoritative) — no re-derivation here.
            cur_status = snap["keypoint_status"].numpy()                # (E,Kmax) from the post-reset snapshot
            cur_setpoint = snap["setpoint_kp_idx"].numpy()             # (E,) current goal keypoint index
            kpst = np.zeros((num_envs, T, k), dtype=np.uint8)            # per-frame keypoint status (for the minimap)
            des_force = np.maximum(snap["desired_force_N"].numpy(), 1e-6)
            track_tol = float(snap["keypoint_track_tol"])
            v_des = float(snap["desired_speed"])
            bf_t = snap.get("break_force_N")                            # present only when the peg is fragile
            break_force = bf_t.numpy() if bf_t is not None else None
            start_w = snap["start_w"].numpy(); goal_w = snap["goal_w"].numpy()
            u_dir = snap["path_dir"].numpy(); v_dir = snap["d_lat"].numpy()
            center = 0.5 * (start_w + goal_w)
            su, svv = sv.project_uv(start_w, center, u_dir, v_dir)
            gu, gvv = sv.project_uv(goal_w, center, u_dir, v_dir)
            half = 0.5 * snap["path_length"].numpy()
            const = dict(center=center, u=u_dir, v=v_dir, start_uv=np.stack([su, svv], 1),
                         goal_uv=np.stack([gu, gvv], 1), half=half)
            kp_uv = _keypoint_uv(const["start_uv"], const["goal_uv"], half, spacing, k)   # (E,k,2)

            for t in range(T):
                markers.update(cur_status[:, :k].reshape(-1).astype(np.int64))   # env status, set BEFORE the step renders
                gidx = np.clip(cur_setpoint - 1, 0, k - 1)
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
                shr[alive, t] = snap["shear_force_N"].numpy()[alive]
                xtr[alive, t] = snap["cross_track"].numpy()[alive]
                pac[alive, t] = snap["along_track_speed"].numpy()[alive]
                tru[alive, t] = tu[alive]; trv[alive, t] = tv[alive]
                trc[alive, t] = snap["in_contact"].numpy()[alive]
                tro[alive, t] = over[alive]
                alive_idx = np.nonzero(alive)[0]
                if alive_idx.size:
                    ii = torch.from_numpy(alive_idx)
                    frames[ii, t] = rgb[ii]
                    returns[ii] += reward.detach().view(-1).float().cpu()[ii]
                cur_s_ref = snap["s_ref"].numpy()
                cur_status = snap["keypoint_status"].numpy()            # authoritative env status after this step
                cur_setpoint = snap["setpoint_kp_idx"].numpy()
                kpst[:, t, :] = cur_status[:, :k]                        # freeze-safe: rendered up to disp_term only

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
                    _tr = _read_flag(info, "Episode / Traversed under-achieved") > 0.5
                    _to = _read_flag(info, "Episode / Timeout incomplete") > 0.5
                    for e in idx:
                        # Priority: success, then the terminal-BREAK causes, then the two non-break
                        # non-success outcomes (went the full distance but under 90% coverage; or plain
                        # ran-out-of-time before finishing). A finished tile is never left "in-progress".
                        if success[e]:
                            status[e] = _ST_DONE
                        elif _cl[e]:
                            status[e] = _ST_LOST
                        elif _pb[e]:
                            status[e] = _ST_BROKE
                        elif _tr[e]:
                            status[e] = _ST_TRAVERSED
                        elif _to[e]:
                            status[e] = _ST_TIMEOUT
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
                coll_shr.append(shr[e].copy()); coll_xtr.append(xtr[e].copy()); coll_pac.append(pac[e].copy())
                coll_kpst.append(kpst[e].copy()); coll_kp_uv.append(kp_uv[e].copy())
                coll_break.append(float(break_force[e]) if break_force is not None else None)
                coll_tol.append(track_tol); coll_vdes.append(v_des)
            del frames
            print(f"[record] episode {ep + 1}/{num_episodes} done — {len(coll_frames)} trajectories", flush=True)
    finally:
        set_camera_active(camera, False)

    R = torch.tensor(coll_returns)
    if grid_select == "all":
        # All trajectories in collection order (one episode of num_envs fresh spawns), capped to the grid.
        sel = list(range(min(n_sel, len(coll_frames))))
        print(f"[record] tiling all {len(sel)} trajectories ({rows}x{cols}, grid_select=all)", flush=True)
    else:
        sel = select_grid_indices(R).tolist()                            # 12 indices: best-4/median-4/worst-4
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
                    coll_start_uv[j], coll_goal_uv[j], coll_half[j], coll_half[j],
                    keypoint_uv=coll_kp_uv[j], keypoint_status=coll_kpst[j][tt]))
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
                desF = coll_desforce[j]; fN_q = float(coll_fN[j][q])
                force_fill, force_tf = _force_bar(fN_q, desF, coll_break[j])
                readouts = _build_readouts(coll_fsq[j][q], fN_q, coll_shr[j][q], desF,
                                           coll_pac[j][q], coll_vdes[j], coll_xtr[j][q], coll_tol[j])
                kp_counts = _build_kp_counts(coll_kpst[j][q])
                tiles.append(sv.compose_tile(
                    frame, float(coll_fsq[j][q]), float(coll_osq[j][q]), ins, border,
                    force_text=f"{fN_q:.1f}N", orn_text=f"{coll_ang[j][q]:+.0f}°",
                    force_fill=float(force_fill), orn_fill=float(coll_ang[j][q] / 30.0),
                    force_target_frac=float(force_tf),
                    status_label=_STATUS_LABEL[_st], status_color=_STATUS_COLOR[_st],
                    readouts=readouts, kp_counts=kp_counts))
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


# ---------------------------------------------------------------------------------------------------
# Overlay registry + THE single standalone trajectory recorder (mode='trajectories').
# ---------------------------------------------------------------------------------------------------
# recorder.overlay -> renderer that collects + composes the ranked best/median/worst grid. Add a new
# task overlay by writing its renderer (same kw signature) and registering it here — nothing else changes.
OVERLAY_RENDERERS = {
    "surface_tracking": _collect_surface,   # keypoint-status balls + force/orientation gauges + path minimap
    "none": _collect_plain,                 # plain frames, no overlay -- env-agnostic (forge / peg / ...)
}


def collect_recording(
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
    """The one standalone trajectory recorder (mode='trajectories'): collect >= num_trajectories
    episodes, rank best-4 / median-4 / worst-4 by return, write a 3x4 grid mp4 with the per-tile
    overlay selected by ``recorder_cfg.overlay`` (see OVERLAY_RENDERERS). 'none' is env-agnostic;
    'surface_tracking' requires env.viz_snapshot. Returns the written mp4 path."""
    # Full-episode VIEWING toggle (recorder-only, runtime, display-not-dynamics): disable success/lag
    # early-termination so every tile plays the whole episode. Kept OUT of env_cfg_overrides on purpose
    # (that path is guarded) so it can never masquerade as the training env.
    if bool(getattr(recorder_cfg, "full_episode", False)):
        _ct = getattr(env.unwrapped, "cfg_task", None)
        if _ct is not None:
            for _flag in ("terminate_on_success", "terminate_on_lag"):
                if hasattr(_ct, _flag):
                    setattr(_ct, _flag, False)
            print("[record] recorder.full_episode=true — early-termination disabled for the recording "
                  "(display only; env dynamics unchanged).", flush=True)
    overlay = str(getattr(recorder_cfg, "overlay", "surface_tracking"))
    renderer = OVERLAY_RENDERERS.get(overlay)
    if renderer is None:
        raise ValueError(
            f"unknown recorder.overlay {overlay!r}; registered overlays: {sorted(OVERLAY_RENDERERS)}"
        )
    return renderer(env=env, agent=agent, recorder_cfg=recorder_cfg, camera=camera,
                    max_episode_length=max_episode_length, num_trajectories=num_trajectories,
                    output_dir=output_dir, gif_name=gif_name)


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
    overlays = True
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
            markers.update(snap["keypoint_status"].numpy()[:, :k].reshape(-1).astype(np.int64))  # all unreached at reset
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
