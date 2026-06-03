"""
Factory Control Utilities

Control functions extracted from envs/factory/factory_control.py for use in control wrappers.
These functions implement operational space control and hybrid force-position control.

Extracted functions:
- compute_pose_task_wrench (factory_control.py:70-106)
- compute_force_task_wrench (factory_control.py:108-116)
- compute_dof_torque_from_wrench (factory_control.py:120-159)
- get_pose_error (factory_control.py:161-202)
- _apply_task_space_gains (factory_control.py:245-267)
"""

import math
import torch

try:
    import isaacsim.core.utils.torch as torch_utils
except ImportError:
    try:
        import omni.isaac.core.utils.torch as torch_utils
    except ImportError:
        torch_utils = None

try:
    from isaaclab.utils.math import axis_angle_from_quat
except ImportError:
    try:
        from omni.isaac.lab.utils.math import axis_angle_from_quat
    except ImportError:
        try:
            from omni.isaac.lab.utils.math import axis_angle_from_quat
        except ImportError:
            axis_angle_from_quat = None


def compute_pose_task_wrench(
    cfg,
    dof_pos,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device
):
    """Compute task-space wrench for pose control."""
    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Set tau = k_p * task_pos_error - k_d * task_vel_error
    task_wrench_motion = _apply_task_space_gains(
        delta_fingertip_pose=delta_fingertip_pose,
        fingertip_midpoint_linvel=fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=fingertip_midpoint_angvel,
        task_prop_gains=task_prop_gains,
        task_deriv_gains=task_deriv_gains,
    )
    return task_wrench_motion


def compute_force_task_wrench(
    cfg,
    dof_pos,
    eef_force,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    ctrl_target_force,
    task_gains,
    task_deriv_gains,
    device,
    # PID control parameters (optional)
    task_integ_gains=None,
    force_integral_error=None,
    prev_force_error=None,
    physics_dt=None,
    enable_derivative=False,
    enable_integral=False,
):
    """Compute task-space wrench for force control with optional PID.

    Args:
        cfg: Environment configuration
        dof_pos: Joint positions
        eef_force: End-effector force/torque measurements
        fingertip_midpoint_linvel: Fingertip linear velocity (unused, kept for API compatibility)
        fingertip_midpoint_angvel: Fingertip angular velocity (unused, kept for API compatibility)
        ctrl_target_force: Target force/torque
        task_gains: Proportional gains (Kp)
        task_deriv_gains: Derivative gains (Kd) - auto-calculated as 2*sqrt(Kp) for critical damping
        device: Torch device
        task_integ_gains: Integral gains (Ki) - optional, required if enable_integral=True
        force_integral_error: Accumulated integral error - optional, required if enable_integral=True
        prev_force_error: Previous force error for derivative calculation
        physics_dt: Physics timestep for derivative calculation
        enable_derivative: Enable D term (true derivative of force error)
        enable_integral: Enable I term
    """
    # Proportional term (always active)
    force_error = ctrl_target_force - eef_force
    force_wrench_p = task_gains * force_error

    # Derivative term - uses error delta (not divided by dt to avoid noise amplification)
    force_wrench_d = None
    if enable_derivative and task_deriv_gains is not None and prev_force_error is not None:
        force_error_delta = force_error - prev_force_error
        force_wrench_d = task_deriv_gains * force_error_delta

        # DEBUG: Print derivative control values (env 0, Z axis only to reduce spam)
        # print(f"[DEBUG DERIV] target_force_z={ctrl_target_force[0, 2].item():.2f}, "
        #       f"measured_force_z={eef_force[0, 2].item():.2f}, "
        #       f"force_error_z={force_error[0, 2].item():.2f}")
        # print(f"[DEBUG DERIV] prev_error_z={prev_force_error[0, 2].item():.2f}, "
        #       f"error_delta_z={force_error_delta[0, 2].item():.2f}")
        # print(f"[DEBUG DERIV] wrench_P_z={force_wrench_p[0, 2].item():.2f}, "
        #       f"wrench_D_z={force_wrench_d[0, 2].item():.2f}, "
        #       f"wrench_total_z={(force_wrench_p[0, 2] + force_wrench_d[0, 2]).item():.2f}")

    force_wrench = force_wrench_p
    if force_wrench_d is not None:
        force_wrench = force_wrench + force_wrench_d

    # Integral term - optional
    if enable_integral and task_integ_gains is not None and force_integral_error is not None:
        force_wrench += task_integ_gains * force_integral_error

    return force_wrench


def compute_dof_torque_from_wrench(
    cfg,
    dof_pos,
    dof_vel,
    task_wrench,
    jacobian,
    arm_mass_matrix,
    device,
):
    """Compute joint torques for given task wrench with null space compensation."""
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)

    # Set tau = J^T * tau, i.e., map tau into joint space as desired
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Null space computation for natural arm posture
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    arm_mass_matrix_task = torch.inverse(
        jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
    )
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))

    # Nullspace computation
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (
        2 * math.pi
    ) - math.pi  # normalize to [-pi, pi]

    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    # Clamp torques to safe limits
    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)

    return dof_torque, task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    """Compute task-space error between target and current fingertip pose."""
    if torch_utils is None:
        raise ImportError("torch_utils not available. Please ensure Isaac Sim is properly installed.")

    if axis_angle_from_quat is None:
        raise ImportError("axis_angle_from_quat not available. Please ensure Isaac Lab is properly installed.")

    # Compute pos error
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos

    # Compute rot error
    if jacobian_type == "geometric":
        # Check for shortest path using quaternion dot product
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[:, 0]  # scalar component

        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)

        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def _apply_task_space_gains(
    delta_fingertip_pose,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    task_prop_gains,
    task_deriv_gains
):
    """Apply PD gains to task-space error."""
    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to linear error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rotational error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )

    return task_wrench


# ---------------------------------------------------------------------------
# Bit-exact base-controller helpers (used by the hybrid / hybrid_vic wrappers).
#
# These reproduce ForgeEnv._apply_action's target generation and
# factory_control.compute_dof_torque's motion-wrench + dead-zone path EXACTLY, so a
# pose-only control wrapper matches the stock env bit-for-bit.
# ---------------------------------------------------------------------------


def wrap_yaw(angle):
    """Match ``factory_utils.wrap_yaw``: keep yaw on a continuous span past the joint limit."""
    return torch.where(angle > math.radians(235.0), angle - 2 * math.pi, angle)


def compute_ctrl_targets(env, actions):
    """Bit-exact replica of ``ForgeEnv._apply_action`` target generation.

    ``actions`` is the (EMA'd) action tensor; its ``[:, 0:3]`` / ``[:, 3:6]`` are the
    pose (position / rotation) actions — same slices the base env reads. Returns
    ``(ctrl_target_pos, ctrl_target_quat, delta_pos, delta_yaw)`` where the two deltas
    match what the base sets on the env for its action-penalty reward.
    """
    device = env.device
    num_envs = env.num_envs

    # Step 0: scale actions to the allowed range (diag(pos/rot_action_bounds)).
    pos_actions = actions[:, 0:3] @ torch.diag(
        torch.tensor(env.cfg.ctrl.pos_action_bounds, device=device)
    )
    rot_actions = actions[:, 3:6] @ torch.diag(
        torch.tensor(env.cfg.ctrl.rot_action_bounds, device=device)
    )

    # Step 1: desired pose targets in EE frame.
    fixed_pos_action_frame = env.fixed_pos_obs_frame + env.init_fixed_pos_obs_noise
    ctrl_target_preclipped_pos = fixed_pos_action_frame + pos_actions

    rot_actions = rot_actions.clone()
    rot_actions[:, 0:2] = 0.0
    # Joint-limit yaw map (assumes limit in (+x,-y)-quadrant of world frame).
    rot_actions[:, 2] = math.radians(-180.0) + math.radians(270.0) * (rot_actions[:, 2] + 1.0) / 2.0
    bolt_frame_quat = torch_utils.quat_from_euler_xyz(
        roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
    )
    rot_180_euler = torch.tensor([math.pi, 0.0, 0.0], device=device).repeat(num_envs, 1)
    quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
        roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
    )
    ctrl_target_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

    # Step 2a: clip position targets toward current pose, within pos_threshold.
    delta_pos = ctrl_target_preclipped_pos - env.fingertip_midpoint_pos
    pos_error_clipped = torch.clip(delta_pos, -env.pos_threshold, env.pos_threshold)
    ctrl_target_pos = env.fingertip_midpoint_pos + pos_error_clipped

    # Step 2b: clip orientation targets in Euler space (yaw uses the joint-limit wrap).
    curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(env.fingertip_midpoint_quat)
    desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_preclipped_quat)
    desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

    curr_yaw = wrap_yaw(curr_yaw)
    desired_yaw = wrap_yaw(desired_yaw)
    delta_yaw = desired_yaw - curr_yaw
    clipped_yaw = torch.clip(delta_yaw, -env.rot_threshold[:, 2], env.rot_threshold[:, 2])
    desired_xyz[:, 2] = curr_yaw + clipped_yaw

    desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * math.pi, desired_roll)
    desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * math.pi, desired_pitch)

    delta_roll = desired_roll - curr_roll
    clipped_roll = torch.clip(delta_roll, -env.rot_threshold[:, 0], env.rot_threshold[:, 0])
    desired_xyz[:, 0] = curr_roll + clipped_roll

    curr_pitch = torch.where(curr_pitch > math.pi, curr_pitch - 2 * math.pi, curr_pitch)
    desired_pitch = torch.where(desired_pitch > math.pi, desired_pitch - 2 * math.pi, desired_pitch)

    delta_pitch = desired_pitch - curr_pitch
    clipped_pitch = torch.clip(delta_pitch, -env.rot_threshold[:, 1], env.rot_threshold[:, 1])
    desired_xyz[:, 1] = curr_pitch + clipped_pitch

    ctrl_target_quat = torch_utils.quat_from_euler_xyz(
        roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
    )
    return ctrl_target_pos, ctrl_target_quat, delta_pos, delta_yaw


def compute_pose_motion_wrench(
    delta_pose,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    task_prop_gains,
    task_deriv_gains,
    dead_zone_thresholds=None,
    matrix=False,
):
    """Task-space pose PD motion wrench, with the base env's dead zone.

    ``delta_pose`` is (E,6) = [pos_error(3), axis_angle_error(3)]; vels are (E,3).
    * ``matrix=False``: ``task_prop_gains`` / ``task_deriv_gains`` are (E,6) diagonal gains
      applied elementwise — bit-exact with the base ``_apply_task_space_gains``.
    * ``matrix=True``:  they are (E,6,6) and applied as ``K @ delta_pose - D @ vel``.
    The dead zone (``where(|w|<dz, 0, sign(w)*(|w|-dz))``) reproduces the base controller's
    low-force unreliability model when ``dead_zone_thresholds`` is provided.
    """
    if matrix:
        vel = torch.cat([fingertip_midpoint_linvel, fingertip_midpoint_angvel], dim=1)  # (E,6)
        task_wrench = (
            torch.bmm(task_prop_gains, delta_pose.unsqueeze(-1)).squeeze(-1)
            - torch.bmm(task_deriv_gains, vel.unsqueeze(-1)).squeeze(-1)
        )
    else:
        task_wrench = torch.zeros_like(delta_pose)
        task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * delta_pose[:, 0:3] + task_deriv_gains[:, 0:3] * (
            0.0 - fingertip_midpoint_linvel
        )
        task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * delta_pose[:, 3:6] + task_deriv_gains[:, 3:6] * (
            0.0 - fingertip_midpoint_angvel
        )

    if dead_zone_thresholds is not None:
        task_wrench = torch.where(
            task_wrench.abs() < dead_zone_thresholds,
            torch.zeros_like(task_wrench),
            task_wrench.sign() * (task_wrench.abs() - dead_zone_thresholds),
        )
    return task_wrench


# ----------------------------------------------------------------------------------------
# Action-space gain-mapping math (used by CtrlActionInterfaceWrapper). Pure torch, no Isaac
# dependency, so these are unit-testable on CPU.
# ----------------------------------------------------------------------------------------

def geom_scale(a, lo, hi, eps=1e-6):
    """Geometric (log-uniform) map of actions ``a`` in [-1, 1] to ``[lo, hi]``.

    ``k = lo * (hi/lo)^((a+1)/2)`` — i.e. ``a=-1`` -> ``lo``, ``a=+1`` -> ``hi``, with the
    interpolation uniform in log-space. ``a`` is clamped to [-1, 1] first (sampled actions
    can exceed the tanh range). ``lo``/``hi`` broadcast against ``a`` (scalars or per-channel
    tensors). Where ``lo < eps`` the geometric map is undefined (needs ``lo > 0``); those
    channels return 0 so a degenerate ``lo=0`` bound disables stiffness instead of producing
    NaN/Inf.

    Args:
        a: action tensor, any shape.
        lo, hi: lower/upper bounds, broadcastable to ``a``.
        eps: threshold below which ``lo`` is treated as zero.

    Returns:
        Tensor shaped like ``a`` (after broadcasting with lo/hi).
    """
    t = (a.clamp(-1.0, 1.0) + 1.0) * 0.5
    lo = torch.as_tensor(lo, dtype=a.dtype, device=a.device)
    hi = torch.as_tensor(hi, dtype=a.dtype, device=a.device)
    safe_lo = torch.clamp(lo, min=eps)
    scaled = safe_lo * (hi / safe_lo).pow(t)
    return torch.where(lo < eps, torch.zeros_like(scaled), scaled)


def rotation_6d_to_matrix(v6, eps=1e-8):
    """Map a 6-D rotation representation to a (E,3,3) rotation matrix via Gram-Schmidt.

    Continuity-friendly 6D representation (Zhou et al., 2019): the first 3 components form
    the first column direction, the next 3 are orthogonalized against it; the third column
    is their cross product. The result is orthonormal with det = +1 by construction.

    Args:
        v6: (E, 6) tensor.
        eps: small value guarding the normalizations against zero-length inputs.

    Returns:
        (E, 3, 3) rotation matrices (columns = [b1, b2, b3]).
    """
    a1 = v6[:, 0:3]
    a2 = v6[:, 3:6]
    b1 = a1 / a1.norm(dim=1, keepdim=True).clamp_min(eps)
    a2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=1, keepdim=True).clamp_min(eps)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=2)  # columns


def build_lower_triangular_3x3(diag_vals, offdiag_vals):
    """Assemble a batch of 3x3 lower-triangular matrices from diagonal + off-diagonal values.

    Off-diagonal entries fill the strictly-lower triangle in the order returned by
    ``torch.tril_indices(3, 3, offset=-1)``: ``(1,0), (2,0), (2,1)``.

    Args:
        diag_vals: (E, 3) diagonal entries (L[0,0], L[1,1], L[2,2]).
        offdiag_vals: (E, 3) strictly-lower entries (L[1,0], L[2,0], L[2,1]).

    Returns:
        (E, 3, 3) lower-triangular matrices.
    """
    E = diag_vals.shape[0]
    L = torch.zeros((E, 3, 3), dtype=diag_vals.dtype, device=diag_vals.device)
    diag_idx = torch.arange(3, device=diag_vals.device)
    L[:, diag_idx, diag_idx] = diag_vals
    off = torch.tril_indices(3, 3, offset=-1, device=diag_vals.device)
    L[:, off[0], off[1]] = offdiag_vals
    return L


def block_diag_2(A, B):
    """Stack two (E,3,3) blocks into a (E,6,6) block-diagonal matrix (A top-left, B bottom-right)."""
    E = A.shape[0]
    M = torch.zeros((E, 6, 6), dtype=A.dtype, device=A.device)
    M[:, 0:3, 0:3] = A
    M[:, 3:6, 3:6] = B
    return M