"""Remove the upright-gripper (EEF/world-alignment) assumption from the base Forge/Factory env.

The stock Forge env was built around a gripper LOCKED to (roll=pi, pitch=0, yaw=phi). Several
places exploit that lock and silently break once the gripper is allowed full 6-DOF rotation
(which the project's new EEF-frame control convention does):

  1. ``ForgeEnv._compute_intermediate_values`` builds the noisy fingertip orientation obs by
     ZEROING the quaternion's w,z components (``noisy_fingertip_quat[:, [0, 3]] = 0``) and
     applying a sign flip. That is exact ONLY for the locked family (where the quat is
     ``(0, x, y, 0)``); for a tilted tool it discards real orientation, so the policy observes a
     corrupted, 2-DOF orientation. It also zeroes the roll/pitch finite-diff angular velocity
     (``ee_angvel_fd[:, 0:2] = 0``).
  2. ``FactoryEnv.close_gripper_in_place`` (used during reset's grasp-close) servos the gripper
     to roll=pi, pitch=0 — yanking any TILTED initial orientation back to upright.

This installer patches both at the CLASS level (so every Forge/Factory subclass, incl. the
surface task, is fixed) with full-rotation versions: a corrected ``_compute_intermediate_values``
(noise applied to the full quaternion, full angular-velocity FD — no zeroing/flip) and a
``close_gripper_in_place`` that holds the CURRENT pose. Call BEFORE ``gym.make``. Idempotent.

Repo-local (not an edit of the IsaacLab tree) so it travels with the project (e.g. the HPC
container, which rebuilds IsaacLab fresh). The corrected ``_compute`` mirrors
``ForgeEnv._compute_intermediate_values`` minus the three upright-only lines; if IsaacLab's
Forge env changes materially, re-sync this copy.
"""

from __future__ import annotations


def install_forge_full_rotation() -> None:
    """Patch ForgeEnv / FactoryEnv to support full 6-DOF gripper rotation (idempotent)."""
    import numpy as np
    import torch

    import isaacsim.core.utils.torch as torch_utils
    from isaaclab.utils.math import axis_angle_from_quat

    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
    from isaaclab_tasks.direct.forge import forge_utils
    from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

    if getattr(ForgeEnv, "_full_rotation_patched", False):
        return

    def _compute_intermediate_values(self, dt):
        """ForgeEnv obs noise WITHOUT the upright-only zeroing/flip (full 6-DOF orientation)."""
        # Grandparent (Factory) gives the clean pose / velocities / jacobians / FD.
        FactoryEnv._compute_intermediate_values(self, dt)

        pos_noise_level = self.cfg.obs_rand.fingertip_pos
        rot_noise_level_deg = self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level] * 3, dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis = rot_noise_axis / torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True).clamp_min(1e-8)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        # FULL noisy orientation: perturb the clean quat by a small random rotation. (No
        # ``[:, [0, 3]] = 0`` and no ``* flip_quats`` — those are valid only for an upright tool.)
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )

        # Finite-difference velocities from the noisy estimates.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat = rot_diff_quat * torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        # FULL angular velocity (no ``[:, 0:2] = 0`` roll/pitch zeroing).
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # Force sensing (unchanged from Forge).
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]
        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth
        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise = force_noise * self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def close_gripper_in_place(self):
        """Hold the CURRENT fingertip pose while the gripper closes (no upright forcing)."""
        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    ForgeEnv._compute_intermediate_values = _compute_intermediate_values
    FactoryEnv.close_gripper_in_place = close_gripper_in_place
    ForgeEnv._full_rotation_patched = True
    print(
        "[full-rotation] patched ForgeEnv._compute_intermediate_values (no quat w,z / angvel "
        "zeroing) and FactoryEnv.close_gripper_in_place (hold current pose) for full 6-DOF.",
        flush=True,
    )
