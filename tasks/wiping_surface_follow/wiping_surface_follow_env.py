"""Wiping surface environment (subclass of ``CurvedSurfaceFollowEnv``).

A wiping task on the per-env curved ridge. Reuses the curved surface, the observation/action spaces,
and the curvature ladder (alpha) UNCHANGED. Two things change:

  * HELD OBJECT: a rectangular SPONGE (a box) instead of the cylinder. The sponge is gripped on its
    WIDTH (the cylinder's ``diameter`` slot) and extends DOWN along held-local +z by ``height``, so the
    inherited ``cyl_tip`` / ``cyl_axis`` geometry already gives the sponge's bottom-face CENTRE and FACE
    NORMAL; ``angle_from_normal`` = 0 then means the sponge lies FLAT on the surface. No geometry
    override is needed beyond swapping the asset (done in the task cfg) and desiring a 0-deg tool angle.

  * REWARD + WAYPOINTS: the reward mimics the BASE reward of arXiv:2502.12599 (Eq. 1). A sequence of
    ``n_waypoints`` points spread along the wipe path is placed ON the surface; the policy is shown ONE
    at a time (``setpoint_pos``, so the obs is unchanged) and it advances SEQUENTIALLY when the sponge
    reaches the current waypoint in contact (Option A: sequential, advance-on-reach).

Reward (per step): r_col if a collision (over-force), else r_con + r_force + r_way + r_ac. See the task
cfg for the term definitions + weights.
"""

import numpy as np
import torch

from ..curved_surface_follow.curved_surface_follow_env import CurvedSurfaceFollowEnv
from .wiping_surface_follow_env_cfg import WipingSurfaceFollowEnvCfg


class WipingSurfaceFollowEnv(CurvedSurfaceFollowEnv):
    cfg: WipingSurfaceFollowEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.n_waypoints = max(1, int(self.cfg_task.n_waypoints))
        # Index of the CURRENT target waypoint per env (0..n_waypoints). == n_waypoints means all wiped.
        self.wp_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # Per-step flags (set in _compute_intermediate_values, read in _get_rewards).
        self._wp_reached_now = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._wp_final_now = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._collided = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Previous-step EEF linear velocity, for the acceleration penalty (finite difference).
        self._prev_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        # Current waypoint set (E, n, 3), refreshed each _compute; init for safety.
        self.wipe_waypoints = torch.zeros((self.num_envs, self.n_waypoints, 3), device=self.device)
        # Carry the wiping per-episode state across efficient-reset teleports (not used by default, but
        # kept consistent with the parent's mechanism). alpha/caps come from the curved parent.
        self._efficient_reset_extra_attrs = tuple(self._efficient_reset_extra_attrs) + ("wp_idx",)

    # ------------------------------------------------------------------
    # Waypoints on the surface
    # ------------------------------------------------------------------
    def _wipe_waypoints(self, n):
        """(E, n, 3) env-relative waypoints along the wipe path, ON the ridge surface. Waypoint j
        (1..n) sits at along-path arc-length (j/n)*path_length, lifted onto the surface — so the set
        spans the path and the last waypoint is the far edge."""
        L = self.path_length                                              # (E,)
        cols = []
        for j in range(1, n + 1):
            arclen = (float(j) / n) * L                                   # (E,) arc-length of waypoint j
            cols.append(self.path_point_on_surface(arclen))              # (E,3) on the surface
        return torch.stack(cols, dim=1)                                   # (E, n, 3)

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)
        n = self.n_waypoints
        wps = self._wipe_waypoints(n)                                     # (E, n, 3)
        self.wipe_waypoints = wps
        ar = torch.arange(self.num_envs, device=self.device)
        idx = self.wp_idx.clamp(max=n - 1)
        cur = wps[ar, idx]                                                # (E,3) current target waypoint
        # Reached = sponge bottom-face centre within reach radius AND in contact, and not already done.
        dist = torch.linalg.norm(self.held_end_pos - cur, dim=-1)
        reached = (self.wp_idx < n) & (dist < float(self.cfg_task.waypoint_reach_radius)) & self.in_contact_any
        self._wp_reached_now = reached
        self.wp_idx = torch.where(reached, self.wp_idx + 1, self.wp_idx)  # advance sequentially (ratchet)
        self._wp_final_now = reached & (self.wp_idx >= n)                 # the LAST waypoint was just wiped

        # The SINGLE target shown to the policy = the (new) current waypoint. Overrides the curved
        # env's path setpoint; obs (setpoint_pos_rel) and the keypoint-servo both read this unchanged.
        idx2 = self.wp_idx.clamp(max=n - 1)
        self.setpoint_pos = wps[ar, idx2]
        self.next_setpoint_pos = wps[ar, (self.wp_idx + 1).clamp(max=n - 1)]
        self.setpoint_kp_idx = idx2                                       # keep the viz/servo index in sync

    # ------------------------------------------------------------------
    # Viz: expose the wiping waypoints + how many are cleaned (for the 'wiping' recorder overlay)
    # ------------------------------------------------------------------
    def viz_snapshot(self) -> dict:
        out = super().viz_snapshot()
        # Recompute the waypoints from the CURRENT plate pose (robust even if _compute has not run yet
        # right after a reset). Env-relative, like start_w -> the recorder adds env_origins.
        start, goal, _, _, _ = self._surface_frame()
        L = torch.linalg.norm(goal - start, dim=-1)
        n = self.n_waypoints
        wps = torch.stack([self.path_point_on_surface((float(j) / n) * L) for j in range(1, n + 1)], dim=1)
        # Per-waypoint LOCAL surface normal (unit, world frame) AT EACH waypoint, so the recorder can
        # lift each marker PERPENDICULAR to the surface there — a single contact-point normal would tilt
        # the balls off the curve (some appear higher, some lower).
        nrms = []
        for j in range(n):
            nj, _ = self._surface_normal_and_dir_at(wps[:, j, :])
            nrms.append(nj / torch.linalg.norm(nj, dim=-1, keepdim=True).clamp_min(1e-8))
        out["wipe_waypoints_w"] = wps.detach().cpu()                        # (E, n, 3) env-relative, on surface
        out["wipe_waypoint_normals_w"] = torch.stack(nrms, dim=1).detach().cpu()  # (E, n, 3) unit normals
        out["wp_idx"] = self.wp_idx.detach().cpu()                          # (E,) waypoints cleaned so far
        out["n_waypoints"] = int(n)
        return out

    # ------------------------------------------------------------------
    # Success = all waypoints wiped (consumed by the scorer -> is_success, and _log_factory_metrics)
    # ------------------------------------------------------------------
    def _get_curr_successes(self, success_threshold=None, check_rot=False):
        return self.wp_idx >= self.n_waypoints

    # ------------------------------------------------------------------
    # Reward: base terms of arXiv:2502.12599 Eq. 1
    # ------------------------------------------------------------------
    def _get_rewards(self):
        cfg = self.cfg_task
        step_dt = float(getattr(self, "step_dt", self.physics_dt * self.cfg.decimation))
        contact = self.in_contact_any.float()

        # r_con: contact flag.
        r_con = float(cfg.wipe_contact_weight) * contact

        # r_force: Gaussian around the target normal force, gated by motion aligned to the target
        # waypoint (I_align) and by contact (off-contact f~0 would otherwise farm the Gaussian tail).
        f = self.measured_normal_force
        mu = float(cfg.wipe_force_target)
        sig = max(float(cfg.wipe_force_sigma), 1e-6)
        gauss = torch.exp(-((f - mu) ** 2) / (2.0 * sig * sig))
        vel = self.fingertip_midpoint_linvel
        vdir = vel / torch.linalg.norm(vel, dim=-1, keepdim=True).clamp_min(1e-6)
        to_wp = self.setpoint_pos - self.held_end_pos
        wpdir = to_wp / torch.linalg.norm(to_wp, dim=-1, keepdim=True).clamp_min(1e-6)
        i_align = ((vdir * wpdir).sum(-1) > float(cfg.wipe_align_cos)).float()
        r_force = float(cfg.wipe_force_weight) * gauss * i_align * contact

        # r_way: sparse reward each time a waypoint is reached, + a larger bonus on the final waypoint.
        r_way = (
            float(cfg.wipe_waypoint_weight) * self._wp_reached_now.float()
            + float(cfg.wipe_final_bonus) * self._wp_final_now.float()
        )

        # r_ac: EE-acceleration smoothness penalty (finite-diff of the EEF linear velocity).
        accel = (self.fingertip_midpoint_linvel - self._prev_linvel) / step_dt
        r_ac = -float(cfg.wipe_accel_weight) * accel.abs().sum(-1)
        self._prev_linvel = self.fingertip_midpoint_linvel.clone()

        # Collision (over-force): r_col REPLACES the other terms this step (paper Eq. 1).
        collided = self.measured_normal_force.abs() > float(cfg.wipe_collision_force)
        r_col = torch.full_like(contact, -float(cfg.wipe_collision_weight))
        rew_buf = torch.where(collided, r_col, r_con + r_force + r_way + r_ac)

        # --- logging ---
        curr_successes = self._get_curr_successes()
        rew_dict = {"contact": r_con, "force": r_force, "waypoint": r_way, "accel": r_ac}
        if hasattr(self, "extras"):
            tl = self.extras.setdefault("to_log", {})
            tl["Wipe / normal force (N)"] = f.detach()
            tl["Wipe / waypoints reached"] = self.wp_idx.float().detach()
            tl["Wipe / contact rate"] = contact.detach()
            tl["Wipe / aligned rate"] = i_align.detach()
            # Per-episode wipe-complete success, bucketed by curvature quartile (perf vs difficulty).
            succeeded = curr_successes.float()
            nan = torch.full_like(succeeded, float("nan"))
            bucket = (self.alpha * 4.0).floor().clamp(0, 3).long()
            stat = {"Episode / Wipe complete": succeeded}
            for i, (lo, hi) in enumerate(((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))):
                stat[f"Wipe complete / alpha [{lo:.2f}-{hi:.2f}]"] = torch.where(bucket == i, succeeded, nan)
            self.extras["per_env_episode_stat"] = stat
            self.extras["per_env_episode_stat_mask"] = self.reset_buf.clone()

        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    # ------------------------------------------------------------------
    # Termination: over-force collision (optional); truncate on all-waypoints-done / time-out
    # ------------------------------------------------------------------
    def _get_dones(self):
        self._compute_intermediate_values(dt=self.physics_dt)
        n = self.n_waypoints
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        all_done = self.wp_idx >= n
        collided = self.measured_normal_force.abs() > float(self.cfg_task.wipe_collision_force)
        self._collided = collided
        truncated = time_out | all_done
        terminated = collided if bool(self.cfg_task.terminate_on_collision) else torch.zeros_like(time_out)
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset: fresh waypoint sequence + acceleration history
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.wp_idx[env_ids] = 0
        self._wp_reached_now[env_ids] = False
        self._wp_final_now[env_ids] = False
        self._collided[env_ids] = False
        # Seed the acceleration finite-diff from the (near-zero) post-reset velocity so the first step
        # doesn't read a huge spurious acceleration.
        self._prev_linvel[env_ids] = self.fingertip_midpoint_linvel[env_ids].clone()
