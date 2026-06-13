"""Optional in-gripper grasp-rotation offset for Forge/Factory peg insertion.

When enabled (``runner_cfg.grasp_rot_mode`` is ``"random"`` or ``"fixed"``), the held asset
(peg) is grasped at a roll/pitch/yaw tilt RELATIVE TO THE GRIPPER, applied per env every reset.
In ``"random"`` mode the tilt is re-sampled ``U(-val, +val)`` per env per reset (simulates a
suboptimal / uncertain grasp); in ``"fixed"`` mode it is a constant signed offset, identical
every env every reset. The peg pose (``held_quat``) is in the critic's
``state_order`` but NOT the actor's ``obs_order`` (``forge_env_cfg.py``), so the tilt is
unobservable to the policy and must be handled via force feedback.

Single-function monkeypatch, mirroring :mod:`wrappers.sensors.contact_sensor_wrapper`: we wrap
``FactoryEnv.get_handheld_asset_relative_pose`` (``factory_env.py:553``), the one method that
defines the peg-in-gripper transform. It returns ``(rel_pos, rel_quat)`` of shape
``(num_envs, …)`` and is re-invoked every reset, so sampling inside it gives correct per-env,
per-reset randomization (the same way upstream samples ``held_asset_pos_noise`` against
``self.num_envs``). Forge inherits the method from ``FactoryEnv``, so patching the base covers
the Forge tasks. Call BEFORE ``gym.make``.

INTENDED FOR PEG INSERT. The perturbation is composed onto whatever base transform the original
returns; for ``peg_insert`` that base is identity, so the realized peg-vs-gripper tilt equals the
sampled rotation. (``nut_thread``'s base is a fixed -90° yaw, so enabling it there would add the
random tilt on top of that base — don't.)
"""

from __future__ import annotations

from typing import Sequence


def install_grasp_rot_randomization(
    rel_grasp_rot_init_deg: Sequence[float], mode: str = "random"
) -> None:
    """Patch ``FactoryEnv.get_handheld_asset_relative_pose`` to add a grasp tilt.

    :param rel_grasp_rot_init_deg: ``[roll, pitch, yaw]`` in degrees. In ``"random"`` mode each
        axis is an independent symmetric range, sampled ``U(-val, +val)`` per env per reset; in
        ``"fixed"`` mode it is a constant signed offset applied identically every env every reset.
        An all-zero vector makes this a no-op (identity perturbation).
    :param mode: ``"random"`` or ``"fixed"`` (``"none"`` never reaches here — the caller skips
        installation entirely).
    """
    import numpy as np
    import torch

    import isaacsim.core.utils.torch as torch_utils  # wxyz layout, same as factory_env.py:10
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

    if mode not in ("random", "fixed"):
        raise ValueError(f"install_grasp_rot_randomization: mode must be 'random' or 'fixed', got {mode!r}")

    # Per-axis angles in radians, captured once (deg→rad). Signed (negatives preserved) so the
    # values double as the symmetric ± range ('random') or the fixed signed offset ('fixed').
    ranges_rad = torch.tensor(
        [np.deg2rad(float(v)) for v in rel_grasp_rot_init_deg], dtype=torch.float32
    )

    _original = FactoryEnv.get_handheld_asset_relative_pose

    def _patched(self):
        rel_pos, rel_quat = _original(self)  # (num_envs, 3), (num_envs, 4)
        ranges = ranges_rad.to(self.device)
        if mode == "random":
            # Per-env, per-axis uniform tilt in [-range, +range] for roll/pitch/yaw.
            d = (2.0 * torch.rand((self.num_envs, 3), device=self.device) - 1.0) * ranges
        else:  # "fixed": constant signed offset, identical across all envs.
            d = ranges.unsqueeze(0).expand(self.num_envs, 3)
        perturb = torch_utils.quat_from_euler_xyz(d[:, 0], d[:, 1], d[:, 2])  # (num_envs, 4)
        # Compose in the gripper's local frame (conjugate-left) so the tilt axes stay
        # gripper-relative for any base transform. For peg_insert (identity base) the order is
        # distributionally irrelevant.
        rel_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(perturb), rel_quat)
        return rel_pos, rel_quat

    FactoryEnv.get_handheld_asset_relative_pose = _patched
    _desc = (
        f"random tilt ±[roll, pitch, yaw]={list(rel_grasp_rot_init_deg)} deg"
        if mode == "random"
        else f"fixed offset [roll, pitch, yaw]={list(rel_grasp_rot_init_deg)} deg"
    )
    print(
        "[grasp-tilt] FactoryEnv.get_handheld_asset_relative_pose patched: peg-in-gripper "
        f"{_desc} (per env, per reset).",
        flush=True,
    )
