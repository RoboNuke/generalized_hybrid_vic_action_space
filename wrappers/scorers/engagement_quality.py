"""Engagement-quality diagnostics for Forge/Factory peg insertion.

The per-step ``curr_engaged`` bonus fires only when ALL of the engagement criteria hold at once
(see ``FactoryEnv._get_curr_successes`` + the optional gates in
:mod:`wrappers.scorers.insertion_reward`):

  * **depth**       — the held-asset base is at/below the engagement height
                      (``z_disp < fixed_asset_cfg.height * engage_threshold``);
  * **centering**   — lateral offset to the socket axis ``< 0.0025`` m;
  * **orientation** — peg-axis-vs-socket-axis angle ``< z_align_max_deg`` (the ``check_z_aligned`` gate);
  * **yaw**         — (optional) EE yaw ``< ee_success_yaw`` (the ``check_yaw`` gate).

When engagement is rare it is hard to tell WHICH criterion is the bottleneck. This module
publishes, every step, a small ``engagement_quality`` metric family so that question is directly
answerable on TensorBoard:

  * ``engagement_quality/depth_met``         — fraction of steps meeting the depth criterion
                                               (``_mean``) plus its spread (``_std``);
  * ``engagement_quality/centering_g_depth`` — AMONG depth-met steps, fraction also centered;
  * ``engagement_quality/orientation_g_depth`` — among depth-met, fraction also z-aligned;
  * ``engagement_quality/yaw_g_depth``       — among depth-met, fraction also yaw-aligned (only when
                                               ``check_yaw`` is requested);
  * ``engagement_quality/all_g_depth``       — among depth-met, fraction meeting EVERY other criterion
                                               (i.e. the would-be engagement rate, restricted to the
                                               depth-met population). A cross-check against
                                               ``Episode / Engagement rate``.
  * ``engagement_quality/xy_dist_uncentered`` — lateral offset (m, ``_mean`` + ``_std``) of the pegs
                                               that reach depth but are NOT centered — "how far off is
                                               the peg when it's stuck", to place the fine-keypoint reach.
  * ``engagement_quality/angle_unengaged``   — geodesic peg-vs-socket axis angle (deg, ``_mean`` +
                                               ``_std``) of the pegs that reach depth but do NOT engage —
                                               whether orientation is still blocking insertion.

The "_g_depth" (given-depth) conditionals are computed by publishing ``NaN`` for every env that does
NOT meet depth on that step; ``BlockAgent._accum_dist_stat`` drops non-finite samples, so the running
mean/std are taken over exactly the depth-met env-steps. Depth is the natural condition because it is
the easy criterion to hit (press the peg down) — the interesting question is, once down, what stops
it engaging.

Implementation mirrors :mod:`wrappers.scorers.kp_z_align_reward` / :mod:`wrappers.scorers.insertion_reward`:
a single monkeypatch on ``FactoryEnv._log_factory_metrics`` (called once per step inside
``_get_rewards``), so it composes with the reward patches regardless of install order (they replace
``_get_rewards`` / ``_get_factory_rew_dict`` / ``_get_curr_successes``, never ``_log_factory_metrics``).
The geometry is recomputed here from the true peg/socket poses with the ENGAGE thresholds, so the
metrics describe the engagement gate specifically (not the tighter success gate). Reward-side /
privileged — nothing here enters an observation. Call BEFORE ``gym.make``.
"""

from __future__ import annotations


def _quat_z_axis(quat):
    """World-frame z-axis (third rotation-matrix column) of a wxyz quaternion ``(N, 4)``."""
    import torch

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1
    )


def install_engagement_quality(
    z_align_max_deg: float = 15.0,
    check_yaw: bool = False,
    centering_threshold: float = 0.0025,
) -> None:
    """Patch ``FactoryEnv._log_factory_metrics`` to publish the ``engagement_quality`` metric family.

    :param z_align_max_deg: max peg-vs-socket axis angle (deg) for the orientation criterion. Match
        ``runner_cfg.z_align_max_deg`` so the metric agrees with the ``check_z_aligned`` engage gate.
    :param check_yaw: also report the EE-yaw criterion (uses ``cfg_task.ee_success_yaw``). Set this to
        match the ``engage_check_yaw`` gate; off by default (the yaw gate is rarely used for peg).
    :param centering_threshold: lateral-offset threshold (m) for the centering criterion. Default
        ``0.0025`` mirrors stock ``FactoryEnv._get_curr_successes``.
    """
    import torch

    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
    import isaacsim.core.utils.torch as torch_utils

    z_max = float(z_align_max_deg)
    xy_max = float(centering_threshold)
    want_yaw = bool(check_yaw)

    _orig = FactoryEnv._log_factory_metrics

    def _patched(self, rew_dict, curr_successes):
        out = _orig(self, rew_dict, curr_successes)

        # --- recompute the engagement criteria from the true poses, ENGAGE thresholds ---
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos, self.fixed_quat, self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
        )
        xy_dist = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * self.cfg_task.engage_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * self.cfg_task.engage_threshold
        else:  # not an insertion task — nothing meaningful to report
            return out

        depth = z_disp < height_threshold
        centered = xy_dist < xy_max

        peg_z = _quat_z_axis(self.held_quat)
        socket_z = _quat_z_axis(self.fixed_quat)
        cos_angle = (peg_z * socket_z).sum(dim=-1).clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.arccos(cos_angle))
        oriented = angle_deg < z_max

        others = centered & oriented
        if want_yaw:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            yaw_ok = curr_yaw < self.cfg_task.ee_success_yaw
            others = others & yaw_ok

        # --- publish (per-env (stat) tensors; BlockAgent emits _mean + _std) ---
        # Conditionals are masked to the depth-met population with NaN (dropped by the accumulator).
        nan = float("nan")

        def given_depth(crit):
            return torch.where(depth, crit.float(), torch.full_like(xy_dist, nan))

        to_log = self.extras.setdefault("to_log", {})
        to_log["engagement_quality/depth_met (stat)"] = depth.float()
        to_log["engagement_quality/centering_g_depth (stat)"] = given_depth(centered)
        to_log["engagement_quality/orientation_g_depth (stat)"] = given_depth(oriented)
        if want_yaw:
            to_log["engagement_quality/yaw_g_depth (stat)"] = given_depth(yaw_ok)
        to_log["engagement_quality/all_g_depth (stat)"] = given_depth(others)

        # --- reward-shaping diagnostics: for pegs that reach depth but FAIL, quantify HOW they fail
        # (mean + std via the (stat) suffix). NaN outside each subpopulation so the accumulator
        # averages over exactly that subset. ---
        # (1) lateral offset (m) of the depth-met-but-NOT-centered pegs = "how far off-center when
        #     stuck" — tells us where to place the fine-keypoint reach/weight.
        uncentered_at_depth = depth & (~centered)
        to_log["engagement_quality/xy_dist_uncentered (stat)"] = torch.where(
            uncentered_at_depth, xy_dist, torch.full_like(xy_dist, nan)
        )
        # (2) geodesic peg-vs-socket axis angle (deg) of the depth-met-but-NOT-engaged pegs
        #     ("reach depth but not insertion") — tells us whether orientation is still a blocker.
        unengaged_at_depth = depth & (~others)
        to_log["engagement_quality/angle_unengaged (stat)"] = torch.where(
            unengaged_at_depth, angle_deg, torch.full_like(angle_deg, nan)
        )
        return out

    FactoryEnv._log_factory_metrics = _patched
    print(
        f"[engagement-quality] patched FactoryEnv._log_factory_metrics: publishing "
        f"engagement_quality/* (depth_met + centering/orientation"
        f"{'/yaw' if want_yaw else ''}/all given depth); "
        f"z_align_max_deg={z_max}, centering_threshold={xy_max}, check_yaw={want_yaw}.",
        flush=True,
    )
