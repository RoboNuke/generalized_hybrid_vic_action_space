"""
Control Action Interface Wrapper — switchable K / D / K_f action-space formulations.

Subclass of :class:`~wrappers.controllers.hybrid_vic_wrapper.HybridVICWrapper`. It reuses
*all* of the hybrid-VIC machinery (action-space expansion, target generation, pose/force
wrench blending, Jᵀ→torque) and overrides only how the policy's trailing "gain" actions
become the three control matrices:

  * **K**   — pose stiffness   (impedance: ``K @ pose_error``)
  * **D**   — pose damping     (impedance: ``-D @ vel``), always DERIVED from K (never sampled)
  * **K_f** — force stiffness  (force control: ``K_f @ (f_d - f)``), no damping

A single config string ``controller_cfg.ctrl_action_interface.gain_mapping`` selects the
formulation. All formulations assume policy actions in [-1, 1] and scale them internally.

Gain-action block layout (appended after the base ``[pose | N sel | N force]``)::

    [ pose-K params (pdim) | K_f params (pdim, only if use_hybrid_force) ]

where ``pdim`` depends on ``gain_mapping``:

  * ``constant``          (pdim=0)  — diagonal K from ``default_task_prop_gains`` (K_f from
    ``default_task_force_gains``); nothing is added to the action space.
  * ``variable_diagonal`` (pdim=6)  — per-axis diagonal K, geometric scaling.
  * ``cholesky``          (pdim=12) — block-diagonal SPD K from TWO independent 3x3 Cholesky
    factors (position block, rotation block): ``K = blkdiag(L_pos L_posᵀ, L_rot L_rotᵀ)``.
    Per-block layout of the 6 dims: ``[3 L-diagonal, 3 L-off-diagonal]``; the off-diagonal
    order matches ``torch.tril_indices(3,3,offset=-1)`` -> ``(1,0),(2,0),(2,1)``. D = zeros
    for now (the Λ-based double-diagonalization damping is deferred).
  * ``rotated``           (pdim=12) — diagonal K (6, geometric) + a 6-D rotation vector (6)
    -> 3x3 rotation R (Gram-Schmidt) applied block-diagonally: ``K = blkdiag(R,R) diag(k) blkdiag(R,R)ᵀ``.

Scaling is geometric ``k = lo * (hi/lo)^((a+1)/2)`` with per-channel bounds (position /
rotation for K; force / torque for K_f); a lower bound < eps yields 0 on that channel.
``use_hybrid_force=False`` => no selection/force-target dims (force_axes all-zero) and K_f=0.
"""

import torch

from .hybrid_vic_wrapper import HybridVICWrapper
from .factory_control_utils import (
    geom_scale,
    rotation_6d_to_matrix,
    build_lower_triangular_3x3,
    block_diag_2,
)


class CtrlActionInterfaceWrapper(HybridVICWrapper):
    """Hybrid-VIC control with a config-selectable K / D / K_f action-space formulation."""

    _ALLOW_ZERO_N = True                  # force_axes all-zero => no force control

    # Pose-K action dims consumed by each formulation (K_f mirrors this when hybrid is on).
    _PDIM = {"constant": 0, "variable_diagonal": 6, "cholesky": 12, "rotated": 12}

    def __init__(self, env, controller_cfg, num_agents: int = 1):
        # Must be set BEFORE super().__init__: the base __init__ calls _update_dimensions()
        # -> _extra_action_dims() during construction.
        cfg = controller_cfg
        self._mode = cfg.gain_mapping
        self._use_hybrid = cfg.use_hybrid_force
        self._pdim = self._PDIM[self._mode]

        super().__init__(env, controller_cfg, num_agents=num_agents)

        dev = self.device
        self._chol_rho = float(cfg.chol_offdiag_rho)

        # Per-axis geometric bounds as (1, 6) tensors [x, y, z, Rx, Ry, Rz]: pose stiffness K
        # from gain_min/max, force stiffness K_f from force_gain_min/max.
        def _row(vals):
            return torch.tensor(vals, dtype=torch.float32, device=dev).unsqueeze(0)
        self._k_lo, self._k_hi = _row(cfg.gain_min), _row(cfg.gain_max)
        self._kf_lo, self._kf_hi = _row(cfg.force_gain_min), _row(cfg.force_gain_max)

        # Constant-mode diagonals (used by the "constant" formulation).
        self._k_const = torch.tensor(
            cfg.default_task_prop_gains, dtype=torch.float32, device=dev
        )                                                        # (6,)
        self._kf_const = torch.tensor(
            cfg.default_task_force_gains, dtype=torch.float32, device=dev
        )                                                        # (6,)

    def _extra_action_dims(self):
        return self._pdim + (self._pdim if self._use_hybrid else 0)

    # ------------------------------------------------------------------ gain mapping
    def _parse_gain_matrices(self):
        """Build (E,6,6) pose K, pose D, force K_f from the trailing gain actions per mode."""
        start = self._base_n + 2 * self.N
        ca = self.control_actions
        pose_block = ca[:, start:start + self._pdim]
        K, D = self._build_pose_KD(pose_block)

        if self._use_hybrid:
            f_start = start + self._pdim
            K_f = self._build_force_K(ca[:, f_start:f_start + self._pdim])
        else:
            K_f = torch.zeros((self.num_envs, 6, 6), device=self.device)
        return K, D, K_f

    def _build_pose_KD(self, a):
        """Pose stiffness K and (derived) damping D = 2*sqrt(K), per ``self._mode``."""
        E = self.num_envs
        if self._mode == "constant":
            kdiag = self._k_const.unsqueeze(0).expand(E, 6)
            return torch.diag_embed(kdiag), torch.diag_embed(self._crit_damp(kdiag))

        if self._mode == "variable_diagonal":
            kdiag = geom_scale(a, self._k_lo, self._k_hi)
            return torch.diag_embed(kdiag), torch.diag_embed(self._crit_damp(kdiag))

        if self._mode == "rotated":
            kdiag = geom_scale(a[:, 0:6], self._k_lo, self._k_hi)
            Rf = block_diag_2(*self._rot_blocks(a[:, 6:12]))
            RfT = Rf.transpose(1, 2)
            K = Rf @ torch.diag_embed(kdiag) @ RfT
            D = Rf @ torch.diag_embed(self._crit_damp(kdiag)) @ RfT
            return K, D

        # cholesky: SPD K, damping deferred (zeros).
        K = self._build_cholesky_K(a, self._k_lo, self._k_hi)
        return K, torch.zeros((E, 6, 6), device=self.device)

    def _build_force_K(self, a):
        """Force stiffness K_f (no damping), per ``self._mode`` with force/torque bounds."""
        E = self.num_envs
        if self._mode == "constant":
            return torch.diag_embed(self._kf_const.unsqueeze(0).expand(E, 6))
        if self._mode == "variable_diagonal":
            return torch.diag_embed(geom_scale(a, self._kf_lo, self._kf_hi))
        if self._mode == "rotated":
            kdiag = geom_scale(a[:, 0:6], self._kf_lo, self._kf_hi)
            Rf = block_diag_2(*self._rot_blocks(a[:, 6:12]))
            return Rf @ torch.diag_embed(kdiag) @ Rf.transpose(1, 2)
        return self._build_cholesky_K(a, self._kf_lo, self._kf_hi)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _crit_damp(kdiag):
        """Critical damping for a diagonal stiffness vector: D = 2*sqrt(K)."""
        return 2.0 * torch.sqrt(kdiag.clamp_min(0.0))

    @staticmethod
    def _rot_blocks(v6):
        """3x3 rotation R from a 6-D vector, returned twice for block-diagonal blkdiag(R,R)."""
        R = rotation_6d_to_matrix(v6)
        return R, R

    def _build_cholesky_K(self, a, lo, hi):
        """Block-diagonal SPD K from two independent 3x3 Cholesky factors.

        ``a`` is (E,12) = [pos_diag(3), pos_off(3), rot_diag(3), rot_off(3)]. The L diagonal is
        geometrically scaled with the *sqrt* of the per-channel K bounds (since K=LLᵀ squares
        it); the L off-diagonal is ``chol_offdiag_rho * sqrt(lo*hi) * clamp(a,-1,1)`` using the
        block's scalar bound product.
        """
        sqrt_lo = torch.sqrt(lo.clamp_min(0.0))
        sqrt_hi = torch.sqrt(hi.clamp_min(0.0))

        lp_diag = geom_scale(a[:, 0:3], sqrt_lo[:, 0:3], sqrt_hi[:, 0:3])
        lr_diag = geom_scale(a[:, 6:9], sqrt_lo[:, 3:6], sqrt_hi[:, 3:6])

        ref_pos = torch.sqrt((lo[0, 0] * hi[0, 0]).clamp_min(0.0))
        ref_rot = torch.sqrt((lo[0, 3] * hi[0, 3]).clamp_min(0.0))
        lp_off = self._chol_rho * ref_pos * a[:, 3:6].clamp(-1.0, 1.0)
        lr_off = self._chol_rho * ref_rot * a[:, 9:12].clamp(-1.0, 1.0)

        Lp = build_lower_triangular_3x3(lp_diag, lp_off)
        Lr = build_lower_triangular_3x3(lr_diag, lr_off)
        Kp = Lp @ Lp.transpose(1, 2)
        Kr = Lr @ Lr.transpose(1, 2)
        return block_diag_2(Kp, Kr)
