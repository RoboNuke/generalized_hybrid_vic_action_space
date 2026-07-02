"""Configurable insertion-reward gating/rebalancing for Forge/Factory peg insertion.

The stock Factory reward (``FactoryEnv._get_factory_rew_dict``) sums a per-step ``curr_engaged``
bonus and a per-step ``curr_success`` bonus, both at scale ``1.0``. Both masks come from
``_get_curr_successes`` (position centering + insertion depth), optionally AND-ed with a rotation
gate. Two things this module makes configurable, all via runner_cfg, before ``gym.make``:

1. **Bonus scales** — ``curr_engaged_scale`` / ``curr_success_scale`` (stock 1.0 each).

2. **Alignment gates on the success/engaged masks** — generalising the stock rotation check. The
   stock ``check_rot`` is renamed to **``check_yaw``** (it only requires ``curr_yaw <
   cfg_task.ee_success_yaw`` — END-EFFECTOR YAW, nothing else). A new **``check_z_aligned``** gate
   requires the angle between the held asset's z-axis and the fixed asset's z-axis to be below
   ``z_align_max_deg`` degrees — i.e. the peg's insertion axis actually points down the socket.
   ``check_z_aligned`` AND-s into the mask exactly the way ``check_yaw`` does, and can gate the
   engaged mask, the success mask, or both. Useful as task geometries get more complex (tilted
   grasps, off-axis sockets) where "centered + deep" no longer implies "correctly oriented".

Success is NOT a terminal condition (``_get_dones`` returns time-out only), so gating success only
affects the reward term and the logged success rate, never episode length.

Monkeypatch (mirrors :mod:`wrappers.scorers.kp_z_align_reward`):
  * ``_get_curr_successes`` is REPLACED with a version taking ``check_yaw`` / ``check_z_aligned``
    (``check_rot`` kept as a back-compat alias for ``check_yaw``);
  * ``_get_factory_rew_dict`` is WRAPPED (composes with kp_z_align) to recompute ``curr_engaged``
    with the engaged gates and to set the two scales;
  * ``_get_rewards`` is REPLACED to compute ``curr_successes`` with the success gates, so the reward
    term AND the logged success rate stay consistent (nut_thread's stock yaw check is preserved).
Call BEFORE ``gym.make``.
"""

from __future__ import annotations


def _quat_z_axis(quat):
    """World-frame z-axis (third rotation-matrix column) of a wxyz quaternion ``(N, 4)``."""
    import torch

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1
    )


def install_insertion_reward(
    curr_engaged_scale: float = 1.0,
    curr_success_scale: float = 1.0,
    kp_baseline_scale: float = 1.0,
    kp_coarse_scale: float = 1.0,
    kp_fine_scale: float = 1.0,
    engage_check_yaw: bool = False,
    engage_check_z_aligned: bool = False,
    success_check_yaw: bool = False,
    success_check_z_aligned: bool = False,
    z_align_max_deg: float = 15.0,
    z_cliff_cutoff: bool = False,
) -> None:
    """Patch FactoryEnv to rescale and/or alignment-gate the engaged & success bonuses.

    :param curr_engaged_scale: reward scale for the per-step ``curr_engaged`` bonus (stock 1.0).
    :param curr_success_scale: reward scale for the per-step ``curr_success`` bonus (stock 1.0).
    :param engage_check_yaw: AND the engaged mask with the EE-yaw gate.
    :param engage_check_z_aligned: AND the engaged mask with the peg-vs-socket z-alignment gate.
    :param success_check_yaw: AND the success mask with the EE-yaw gate (nut_thread is always gated).
    :param success_check_z_aligned: AND the success mask with the z-alignment gate.
    :param z_align_max_deg: max peg-vs-socket axis angle (deg) for ``check_z_aligned`` to pass.
    """
    import torch

    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
    import isaacsim.core.utils.torch as torch_utils

    ce = float(curr_engaged_scale)
    cs = float(curr_success_scale)
    kb = float(kp_baseline_scale)
    kc = float(kp_coarse_scale)
    kf = float(kp_fine_scale)
    z_max = float(z_align_max_deg)

    # ---- 1) replacement _get_curr_successes with check_yaw / check_z_aligned ----
    def _get_curr_successes(self, success_threshold, check_yaw=False, check_z_aligned=False, check_rot=None):
        """Success mask: centered + at-depth, optionally AND-ed with yaw and/or z-alignment gates."""
        if check_rot is not None:  # back-compat: old callers passed check_rot for the yaw gate
            check_yaw = bool(check_yaw) or bool(check_rot)

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
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
        is_centered = xy_dist < 0.0025

        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = z_disp < height_threshold
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_yaw:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            curr_successes = torch.logical_and(curr_successes, curr_yaw < self.cfg_task.ee_success_yaw)

        if check_z_aligned:
            peg_z = _quat_z_axis(self.held_quat)
            socket_z = _quat_z_axis(self.fixed_quat)
            cos_angle = (peg_z * socket_z).sum(dim=-1).clamp(-1.0, 1.0)
            angle_deg = torch.rad2deg(torch.arccos(cos_angle))
            curr_successes = torch.logical_and(curr_successes, angle_deg < z_max)

        return curr_successes

    FactoryEnv._get_curr_successes = _get_curr_successes

    # ---- 2) wrap _get_factory_rew_dict: re-gate curr_engaged + set scales (composes w/ kp_z_align)
    _orig_rew_dict = FactoryEnv._get_factory_rew_dict

    def _patched_rew_dict(self, curr_successes):
        rew_dict, rew_scales = _orig_rew_dict(self, curr_successes)
        if engage_check_yaw or engage_check_z_aligned:
            rew_dict["curr_engaged"] = self._get_curr_successes(
                success_threshold=self.cfg_task.engage_threshold,
                check_yaw=engage_check_yaw,
                check_z_aligned=engage_check_z_aligned,
            ).float()
        rew_scales["curr_engaged"] = ce
        rew_scales["curr_success"] = cs
        # Keypoint-term weights (stock 1.0 each). Multiply so this composes with any upstream
        # change to the base scales. Lets a config rebalance baseline vs coarse vs fine after a
        # keypoint_scale (spacing) change that shifted their relative magnitudes.
        rew_scales["kp_baseline"] = rew_scales.get("kp_baseline", 1.0) * kb
        rew_scales["kp_coarse"] = rew_scales.get("kp_coarse", 1.0) * kc
        rew_scales["kp_fine"] = rew_scales.get("kp_fine", 1.0) * kf
        # Stash the EFFECTIVE per-term scales (after the kp_*/curr_* knobs) so the reward-
        # decomposition scorer publishes Episode_Reward/<term> at the SAME scale the training
        # reward uses — and omits zero-weight terms — instead of mirroring stock 1.0 scales.
        self._reward_term_scales = rew_scales
        return rew_dict, rew_scales

    FactoryEnv._get_factory_rew_dict = _patched_rew_dict

    # ---- 3) replace _get_rewards: success mask uses the success gates (reward + logged metric) ----
    def _get_rewards(self):
        # nut_thread keeps its stock yaw requirement; peg/gear add gates only when configured.
        check_yaw = success_check_yaw or (self.cfg_task.name == "nut_thread")
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold,
            check_yaw=check_yaw,
            check_z_aligned=success_check_z_aligned,
        )
        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        # ---- z-cliff cutoff: zero the KEYPOINT terms (baseline/coarse/fine) — NOT alignment — for
        # any env whose peg tip is below the hole mouth but not centered. Removes the "descend
        # off-center onto the block face" local minimum: no depth credit unless over the bore, so the
        # only way to earn keypoint reward below the mouth is through the opening (a hard funnel). ----
        if z_cliff_cutoff:
            held_base_pos, _ = factory_utils.get_held_base_pose(
                self.held_pos, self.held_quat, self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
            )
            target_base_pos, _ = factory_utils.get_target_held_base_pose(
                self.fixed_pos, self.fixed_quat, self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
            )
            xy_dist = torch.linalg.vector_norm(
                target_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
            )
            z_disp = held_base_pos[:, 2] - target_base_pos[:, 2]
            is_centered = xy_dist < 0.0025
            below_mouth = z_disp < float(self.cfg_task.fixed_asset_cfg.height)  # tip below the hole top
            allow = (is_centered | ~below_mouth).float()                        # 1 = reward, 0 = dead zone
            for _k in ("kp_baseline", "kp_coarse", "kp_fine"):
                if _k in rew_dict:
                    rew_dict[_k] = rew_dict[_k] * allow

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name in rew_dict:
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]
        self.prev_actions = self.actions.clone()
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    FactoryEnv._get_rewards = _get_rewards

    print(
        f"[insertion-reward] patched FactoryEnv: curr_engaged_scale={ce}, curr_success_scale={cs}, "
        f"kp_scales(baseline={kb}, coarse={kc}, fine={kf}), "
        f"engage(check_yaw={engage_check_yaw}, check_z_aligned={engage_check_z_aligned}), "
        f"success(check_yaw={success_check_yaw}, check_z_aligned={success_check_z_aligned}), "
        f"z_align_max_deg={z_max}, z_cliff_cutoff={z_cliff_cutoff}.",
        flush=True,
    )
