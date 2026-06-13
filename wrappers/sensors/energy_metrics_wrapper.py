"""Per-step energy / effort metrics for the Forge (and Factory) peg-insertion envs.

Logging-only wrapper, modelled on :class:`~wrappers.sensors.contact_sensor_wrapper.ContactSensorWrapper`.
Each ``step()`` it reads the live robot state off ``env.unwrapped`` and publishes a
handful of per-env ``(num_envs,)`` tensors under ``extras['to_log']``, which
:class:`~wrappers.scorers.reward_decomposition.RewardDecompositionWrapper` forwards
RAW to ``info['per_env_to_log']`` and SAC's block agent reduces per agent. The
observation / action / state spaces are NOT changed.

All six metrics are *instantaneous per-step* quantities; their summary statistic is
chosen by the block agent's reduction convention (a ``(max)`` tag suffix → peak,
else mean), so this wrapper only computes the per-step value:

  * ``energy_metrics/max_force (max)`` / ``energy_metrics/avg_force`` — force
    magnitude ``‖force‖₂`` from the Forge FT sensor (EE-frame). The ``(max)`` suffix
    makes the block agent take the true peak any env saw, over the log interval.
  * ``energy_metrics/sq_joint_vel`` — ``Σ_j q̇_j²`` over the 7 arm DOFs.
  * ``energy_metrics/sq_ee_vel`` — ``Σ v²`` over the 3 EE linear-velocity components.
  * ``energy_metrics/work_power`` — ``Σ_j τ_j·q̇_j`` over the arm DOFs, i.e. mean
    mechanical power (W); NOT integrated energy (J).
  * ``energy_metrics/battery_power`` — ``battery_scale · Σ_j τ_j²``: a copper-loss
    proxy for electrical draw. The sim exposes no motor model (no torque constant
    ``Kt`` / winding resistance ``R`` / bus voltage ``V``), so a true electrical
    figure is not derivable; since ``I = τ/Kt`` ⇒ ``P_copper = I²R ∝ τ²``, the sum
    of squared joint torques (scaled) is the standard available proxy.

Tag names keep exactly one ``/`` (the trailing ``(max)`` adds no slash), as required
by TensorBoard. Only metrics whose flat ``log_*`` toggle is on are emitted.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

# First 7 DOFs are the Franka arm joints; DOFs 7:9 are the gripper (effort-zeroed by
# the base env's generate_ctrl_signals, so they contribute nothing to work/battery).
_ARM = slice(0, 7)


class EnergyMetricsWrapper(gym.Wrapper):
    """Per-step energy/effort diagnostics for Forge/Factory (logging-only)."""

    def __init__(self, env, energy_cfg: Any) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs

        self._log_max_force = bool(energy_cfg.log_max_force)
        self._log_avg_force = bool(energy_cfg.log_avg_force)
        self._log_sq_joint_vel = bool(energy_cfg.log_sq_joint_vel)
        self._log_sq_ee_vel = bool(energy_cfg.log_sq_ee_vel)
        self._log_work_power = bool(energy_cfg.log_work_power)
        self._log_battery_power = bool(energy_cfg.log_battery_power)
        self._battery_scale = float(energy_cfg.battery_scale)

        if hasattr(self.unwrapped, "extras") and "to_log" not in self.unwrapped.extras:
            self.unwrapped.extras["to_log"] = {}

    # ------------------------------------------------------------------ metrics
    def _log_metrics(self) -> None:
        u = self.unwrapped
        if not hasattr(u, "extras"):
            return
        to_log = u.extras.setdefault("to_log", {})

        # Force magnitude from the Forge FT sensor (EE-frame wrench, force = [:, :3]).
        # Absent on non-Forge envs (e.g. plain Factory) — skip the force metrics there.
        if self._log_max_force or self._log_avg_force:
            wrench = getattr(u, "force_sensor_smooth", None)
            if wrench is not None:
                force_mag = wrench[:, :3].detach().norm(dim=1)  # (num_envs,)
                if self._log_max_force:
                    # (max) suffix: block agent takes the peak over the agent's envs
                    # AND over the log interval -> true max force seen.
                    to_log["energy_metrics/max_force (max)"] = force_mag
                if self._log_avg_force:
                    to_log["energy_metrics/avg_force"] = force_mag

        joint_vel = getattr(u, "joint_vel", None)
        joint_torque = getattr(u, "joint_torque", None)

        if self._log_sq_joint_vel and joint_vel is not None:
            qd = joint_vel[:, _ARM].detach()
            to_log["energy_metrics/sq_joint_vel"] = (qd * qd).sum(dim=1)

        if self._log_sq_ee_vel:
            ee_linvel = getattr(u, "fingertip_midpoint_linvel", None)
            if ee_linvel is not None:
                v = ee_linvel.detach()
                to_log["energy_metrics/sq_ee_vel"] = (v * v).sum(dim=1)

        if self._log_work_power and joint_vel is not None and joint_torque is not None:
            qd = joint_vel[:, _ARM].detach()
            tau = joint_torque[:, _ARM].detach()
            # Mechanical power Σ τ·q̇ [W] (can be negative); mean-reduced downstream.
            to_log["energy_metrics/work_power"] = (tau * qd).sum(dim=1)

        if self._log_battery_power and joint_torque is not None:
            tau = joint_torque[:, _ARM].detach()
            # Copper-loss proxy ∝ Σ τ² (I = τ/Kt ⇒ P_copper = I²R ∝ τ²), scaled.
            to_log["energy_metrics/battery_power"] = self._battery_scale * (tau * tau).sum(dim=1)

    # ------------------------------------------------------------------ gym
    def step(self, action):
        out = super().step(action)
        # After super().step(), joint_vel / joint_torque / force_sensor_smooth /
        # fingertip_midpoint_linvel all reflect this step's applied control.
        self._log_metrics()
        return out
