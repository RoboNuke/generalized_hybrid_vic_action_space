"""
Hybrid Force/Position + Variable Impedance Control Wrapper (unification).

Extends the hybrid force/position wrapper with full-matrix variable impedance. On top of
the hybrid `[pose | N sel | N force]` action, the policy additionally outputs THREE full
6x6 matrices (separate inputs):

  * **K** — pose stiffness     (impedance: K @ pose_error)
  * **D** — pose damping       (impedance: -D @ vel)
  * **K_f** — force stiffness  (force control: K_f @ (f_d - f))

Force control is pure stiffness — it has its OWN stiffness matrix (separate from the pose
impedance K) and NO damping matrix. (No force PID derivative/integral.)

Action layout:  `[ pose(base_n) | N selections | N force targets | 36 K | 36 D | 36 K_f ]`
  ``action_space = base_n + 2N + 108``.

With ``force_axes`` all-zero (N=0) this reduces to pure variable-impedance pose control
(a full-matrix VIC). Pose targets and the wrench→torque path stay bit-exact with the base
env; only the gains differ (policy matrices vs the env's diagonal gains). Matrix entries are
mapped per-entry linearly from [-1, 1] into per-axis (row-indexed) bounds: K -> gain_min/max,
D -> damping_min/max, K_f -> force_gain_min/max (length-6 lists on ``ControlCfg``).
"""

import torch

from .hybrid_force_position_wrapper import HybridForcePositionWrapper, AXIS_NAMES
from .factory_control_utils import get_pose_error, compute_pose_motion_wrench


class HybridVICWrapper(HybridForcePositionWrapper):
    """Hybrid force/position control + full 6x6 matrix variable impedance + matrix force stiffness."""

    _ALLOW_ZERO_N = True       # force_axes all-zero => pure VIC mode

    GAIN_DIMS = 108            # 36 (pose K) + 36 (pose D) + 36 (force K_f)

    def _extra_action_dims(self):
        return self.GAIN_DIMS

    def __init__(self, env, controller_cfg, num_agents: int = 1):
        super().__init__(env, controller_cfg, num_agents=num_agents)
        cfg = self.cfg_h  # ControlCfg
        # Per-axis length-6 bounds, row-indexed (1,6,1) so row i of each 6x6 matrix maps into
        # that axis's [lo, hi]. K -> gain_*, D -> damping_*, K_f -> force_gain_*.
        def _col(vals):
            return torch.tensor(vals, dtype=torch.float32, device=self.device).view(1, 6, 1)
        self._gain_lo, self._gain_hi = _col(cfg.gain_min), _col(cfg.gain_max)
        self._damp_lo, self._damp_hi = _col(cfg.damping_min), _col(cfg.damping_max)
        self._fgain_lo, self._fgain_hi = _col(cfg.force_gain_min), _col(cfg.force_gain_max)
        # Parsed matrices for the current env-step (set in _compute_control_targets).
        E = self.num_envs
        eye = torch.eye(6, device=self.device).unsqueeze(0).repeat(E, 1, 1)
        self._K_pose = eye.clone()
        self._D_pose = eye.clone()
        self._K_force = eye.clone()

    def _ema_extra_actions(self, action):
        # Gain dims sit after [pose | N sel | N force]; carried raw (gains are not smoothed).
        start = self._base_n + 2 * self.N
        self.ema_actions[:, start:] = action[:, start:]

    def _parse_gain_matrices(self):
        """Slice + reshape the trailing 108 dims into (E,6,6) K, D, K_f, mapped to [min,max]."""
        start = self._base_n + 2 * self.N
        g = self.control_actions[:, start:start + self.GAIN_DIMS]
        k_raw = g[:, 0:36].reshape(-1, 6, 6).clamp(-1.0, 1.0)
        d_raw = g[:, 36:72].reshape(-1, 6, 6).clamp(-1.0, 1.0)
        kf_raw = g[:, 72:108].reshape(-1, 6, 6).clamp(-1.0, 1.0)
        K = self._gain_lo + (k_raw + 1.0) * 0.5 * (self._gain_hi - self._gain_lo)
        D = self._damp_lo + (d_raw + 1.0) * 0.5 * (self._damp_hi - self._damp_lo)
        K_f = self._fgain_lo + (kf_raw + 1.0) * 0.5 * (self._fgain_hi - self._fgain_lo)
        return K, D, K_f

    def _compute_control_targets(self):
        super()._compute_control_targets()
        # Parse + cache the gain matrices once per env-step (control_actions is fixed across
        # the decimation sub-steps where _pose_motion_wrench / _force_wrench run).
        self._K_pose, self._D_pose, self._K_force = self._parse_gain_matrices()

        # Compact impedance log: per-axis diagonal of each matrix. _compute_control_targets
        # already runs once per env.step (it's invoked from _pre_physics_step), so this is
        # logged unconditionally — like the base's force-goal log — not gated on the
        # apply-action _should_log_wrenches flag (which isn't set yet at this point).
        if hasattr(self.unwrapped, "extras"):
            log = self.unwrapped.extras["to_log"]
            for name, M in (("K", self._K_pose), ("D", self._D_pose), ("Kf", self._K_force)):
                diag = torch.diagonal(M, dim1=1, dim2=2)  # (E,6)
                for i in range(6):
                    log[f"Impedance / {name} {AXIS_NAMES[i]}"] = diag[:, i]

    def _pose_motion_wrench(self):
        """Pose motion wrench using the policy's full 6x6 K/D (matrix path) + dead zone."""
        pos_error, aa_error = get_pose_error(
            fingertip_midpoint_pos=self.unwrapped.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.unwrapped.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=self.unwrapped.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.unwrapped.ctrl_target_fingertip_midpoint_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, aa_error), dim=1)
        return compute_pose_motion_wrench(
            delta_pose,
            self.unwrapped.fingertip_midpoint_linvel,
            self.unwrapped.fingertip_midpoint_angvel,
            task_prop_gains=self._K_pose,
            task_deriv_gains=self._D_pose,
            dead_zone_thresholds=getattr(self.unwrapped, "dead_zone_thresholds", None),
            matrix=True,
        )

    def _force_wrench(self, measured):
        """Force-control wrench using the policy's full 6x6 force-stiffness matrix K_f."""
        force_error = (self.target_force_for_control - measured).unsqueeze(-1)  # (E,6,1)
        return torch.bmm(self._K_force, force_error).squeeze(-1)
