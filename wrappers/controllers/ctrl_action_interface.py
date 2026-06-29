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
  * ``cholesky``          (pdim=6 | 21)  — variable SPD K. By default only the POSITION 3x3
    block is policy-set (rotation uses constant gains): ``K = blkdiag(K_pos, diag(k_rot_const))``,
    6 dims = ``[3 diag, 3 off-diagonal]``. With ``full_gain_matrix`` the policy sets the whole
    6x6 SPD K directly: 21 dims = ``[6 diag, 15 off-diagonal]``. Off-diagonal order matches
    ``torch.tril_indices(n,n,-1)``. K uses the variance/correlation form
    ``K = diag(√k)·Corr·diag(√k)``: the diagonal stiffness ``k = geom_scale(diag, lo, hi)`` lands
    EXACTLY in ``[lo, hi]`` while ``Corr = C Cᵀ`` (row-normalized lower-triangular C with
    strictly-lower entries ``chol_offdiag_rho·clamp(off,-1,1)``) is a true correlation matrix, so
    off-diagonals stay bounded (``|Corr[i,j]| ≤ ρ/√(1+ρ²) ≤ 1``) and K stays SPD. D is critically
    damped (matrix critical damping ``D = 2·K^{1/2}``).
  * ``rotated``           (pdim=9 | 12) — diagonal K + a 6-D rotation vector -> 3x3 rotation R
    (Gram-Schmidt) applied block-diagonally with R SHARED across blocks:
    ``K = blkdiag(R,R) diag(k) blkdiag(R,R)ᵀ``. R is anchored to the held asset's NOMINAL in-hand
    frame ``F_flip`` (the live EEF flipped 180° about y, matching factory's grasp convention) and
    composed to world (``R_world = R_eef · R_flip · R_local``) before building K, so the stiffness
    axes ride with the gripper and a given rpy means the same orientation as ``rel_grasp_rot_init_deg``
    on every axis. R rotates BOTH the position and orientation blocks in all cases. By default only
    the position diagonal gains are policy-set, the orientation gains held at the constants (3 diag +
    6 rot6d = 9); with ``full_gain_matrix`` all 6 diagonals are policy-set (6 diag + 6 rot6d = 12).
    The shared R keeps rot6d at 6 dims either way. Setting ``fixed_rotation_rpy = [roll, pitch, yaw]``
    (degrees) instead FIXES the local frame to that constant: the rot6d dims drop out (pdim = 3 or 6,
    diagonal gains only) while R still rotates both blocks of K / D / K_f exactly as the learned
    variant does.

Scaling is geometric ``k = lo * (hi/lo)^((a+1)/2)`` with per-channel bounds (position /
rotation for K; force / torque for K_f); a lower bound < eps yields 0 on that channel.
``use_hybrid_force=False`` => no selection/force-target dims (force_axes all-zero) and K_f=0.
"""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_from_matrix, matrix_from_quat

from .hybrid_vic_wrapper import HybridVICWrapper
from .factory_control_utils import (
    geom_scale,
    rotation_6d_to_matrix,
    euler_xyz_to_matrix,
    build_lower_triangular,
    block_diag_2,
)


class CtrlActionInterfaceWrapper(HybridVICWrapper):
    """Hybrid-VIC control with a config-selectable K / D / K_f action-space formulation."""

    _ALLOW_ZERO_N = True                  # force_axes all-zero => no force control

    @staticmethod
    def _pose_pdim(mode, full, fixed_rot=False):
        """Pose-K action dims consumed by ``mode`` (K_f mirrors this when hybrid is on).

        ``constant`` => 0, ``variable_diagonal`` => 6 (per-axis diagonal, always full 6-DOF).
        ``cholesky``/``rotated`` cover ``n`` policy-set diagonal DOFs: n=6 with
        ``full_gain_matrix`` (position + orientation), else n=3 (position gains only;
        orientation gains held at the constants — but ``rotated`` still rotates that block):

          * cholesky = [diag(n), off(n*(n-1)/2)]  -> 6  (n=3) or 21 (n=6)
          * rotated  = [diag(n), rot6d(6)]        -> 9  (n=3) or 12 (n=6); the 3x3 rotation R
            is SHARED across the position and orientation blocks, so rot6d stays 6 either way.
            With ``fixed_rot`` (``fixed_rotation_rpy`` set) R is constant, so the rot6d block
            is dropped entirely: rotated = [diag(n)] -> 3 (n=3) or 6 (n=6).
        """
        if mode == "constant":
            return 0
        if mode == "variable_diagonal":
            return 6
        n = 6 if full else 3
        if mode == "rotated":
            return n if fixed_rot else n + 6
        return n + n * (n - 1) // 2  # cholesky

    def __init__(self, env, controller_cfg, num_agents: int = 1):
        # Must be set BEFORE super().__init__: the base __init__ calls _update_dimensions()
        # -> _extra_action_dims() during construction.
        cfg = controller_cfg
        self._mode = cfg.gain_mapping
        self._use_hybrid = cfg.use_hybrid_force
        self._full = bool(cfg.full_gain_matrix)
        # Fixed-rotation variant of ``rotated``: when fixed_rotation_rpy is supplied the R
        # frame is constant (config-defined) rather than policy-emitted, so the rot6d action
        # dims drop out. Only meaningful for gain_mapping="rotated" (validated in the cfg).
        self._fixed_rpy = cfg.fixed_rotation_rpy if self._mode == "rotated" else None
        self._fixed_rot = self._fixed_rpy is not None
        self._pdim = self._pose_pdim(self._mode, self._full, self._fixed_rot)

        super().__init__(env, controller_cfg, num_agents=num_agents)

        dev = self.device
        self._chol_rho = float(cfg.chol_offdiag_rho)

        # Constant local rotation frame R_local (1,3,3) shared by K/D/K_f when fixed_rotation_rpy
        # is set; layered on the held-asset nominal frame F_flip in _rotation_frame. Config supplies
        # roll/pitch/yaw in DEGREES (easier to author); convert to radians here.
        self._R_fixed = (
            euler_xyz_to_matrix(*[math.radians(v) for v in self._fixed_rpy], device=dev)
            if self._fixed_rot else None
        )

        # Held asset's NOMINAL in-hand frame is the EEF flipped 180° about y. This mirrors
        # factory's grasp placement (flip_z_quat = (0,0,1,0); factory_env.py:782-783): the peg's
        # body frame sits flipped relative to the gripper, and rel_grasp_rot_init_deg is applied in
        # THAT frame. The rotated stiffness R is anchored to the same frame (F_flip = R_eef @ R_flip)
        # so a given rpy means the same orientation for the stiffness ellipsoid as for the grasp
        # tilt, on every axis (not just y). Built from the identical quaternion the env uses so the
        # two conventions cannot drift apart.
        self._R_flip = matrix_from_quat(
            torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=dev)
        )  # (1,3,3) = Ry(180)

        # Per-axis geometric bounds as (1, 6) tensors [x, y, z, Rx, Ry, Rz]: pose stiffness K
        # from gain_min/max, force stiffness K_f from force_gain_min/max.
        def _row(vals):
            return torch.tensor(vals, dtype=torch.float32, device=dev).unsqueeze(0)
        self._k_lo, self._k_hi = _row(cfg.gain_min), _row(cfg.gain_max)
        self._kf_lo, self._kf_hi = _row(cfg.force_gain_min), _row(cfg.force_gain_max)
        # rot_action_bounds (rad) — scales the [-1,1] orientation action into the commanded
        # axis-angle rotation vector; used only by _log_commanded_orientation.
        self._rot_action_bounds_row = _row(cfg.rot_action_bounds)        # (1,3) rad

        # Constant-mode diagonals (used by the "constant" formulation).
        self._k_const = torch.tensor(
            cfg.default_task_prop_gains, dtype=torch.float32, device=dev
        )                                                        # (6,)
        self._kf_const = torch.tensor(
            cfg.default_task_force_gains, dtype=torch.float32, device=dev
        )                                                        # (6,)

        # Eval-recording axis-frame markers (all default off; enabled per-marker from the
        # record overlay). Lazy-created on first _update_frame_viz call; _R_frame is cached
        # each step in _build_pose_KD for the rotated stiffness frame.
        self._viz_rotation = bool(cfg.visualize_rotation_frame)
        self._viz_peg_tip = bool(cfg.visualize_peg_tip_frame)
        self._viz_eef = bool(cfg.visualize_eef_frame)
        self._viz_hole = bool(cfg.visualize_hole_frame)
        self._rotation_frame_scale = float(cfg.rotation_frame_axis_scale)
        self._peg_tip_frame_scale = float(cfg.peg_tip_frame_axis_scale)
        self._eef_frame_scale = float(cfg.eef_frame_axis_scale)
        self._hole_frame_scale = float(cfg.hole_frame_axis_scale)
        self._viz_interaction = bool(cfg.visualize_interaction_frame)
        self._interaction_frame_scale = float(cfg.interaction_frame_axis_scale)
        self._marker_rotation = None
        self._marker_peg_tip = None
        self._marker_eef = None
        self._marker_hole = None
        self._marker_interaction = None
        self._R_frame = None

        # Translational stiffness ellipsoid. Eigenvalues of the position 3x3 block of K are linearly
        # mapped from the scalar position-gain range [lo, hi] to a semi-axis length in [min, max] m.
        self._viz_ellipsoid = bool(cfg.visualize_stiffness_ellipsoid)
        self._ellipsoid_compliance = bool(cfg.stiffness_ellipsoid_compliance)
        self._ellip_min = float(cfg.stiffness_ellipsoid_min_scale)
        self._ellip_max = float(cfg.stiffness_ellipsoid_max_scale)
        self._ellip_opacity = float(cfg.stiffness_ellipsoid_opacity)
        self._ellip_lo = float(self._k_lo[0, :3].min())
        self._ellip_den = max(float(self._k_hi[0, :3].max()) - self._ellip_lo, 1e-6)
        self._marker_ellipsoid = None

    def _extra_action_dims(self):
        return self._pdim + (self._pdim if self._use_hybrid else 0)

    # ---- logging gates (driven by this formulation's config switches) ----
    def _force_control_enabled(self) -> bool:
        # Force control needs both eligible axes and the hybrid-force toggle; without the
        # toggle K_f is zeros and there are no selection/force-target dims.
        return self._use_hybrid and self.N > 0

    def _gains_variable(self) -> bool:
        # "constant" mode uses fixed K/D/K_f (no policy gain dims) — nothing to plot.
        return self._mode != "constant"

    def _k_coupling_logged(self) -> bool:
        # Only the non-diagonal K formulations have off-diagonal coupling to report.
        return self._mode in ("rotated", "cholesky")

    def _k_coupling_dim(self) -> int:
        # ``rotated`` rotates BOTH the position and orientation blocks (cross blocks are
        # structurally zero), so the whole 6x6 carries coupling regardless of full mode.
        # ``cholesky`` only couples the policy-set block: all 6 DOFs in full mode, else the
        # position 3x3 (its non-full orientation block is a constant diagonal).
        if self._mode == "rotated":
            return 6
        return 6 if self._full else 3

    def _damping_logged(self) -> bool:
        # Damping D is a full matrix only for rotated/cholesky; the diagonal modes derive
        # D = 2*sqrt(K), so there's nothing new to publish there.
        return self._mode in ("rotated", "cholesky")

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

    def _compute_control_targets(self):
        # Parse gains (caches self._R_frame for rotated mode) via the base, then refresh the
        # eval-recording frame markers. This runs in _pre_physics_step, so marker transforms
        # are set before the recorder camera renders the frame (no one-step lag).
        super()._compute_control_targets()
        self._log_rotation_frame_angle()
        self._log_stiffness_frame_metrics()
        self._log_commanded_orientation()
        self._update_frame_viz()

    def _log_commanded_orientation(self):
        """Log the policy's commanded orientation (deg) about roll/pitch/yaw, BOTH pre-EMA (raw
        network output) and post-EMA (what feeds the controller this step).

        The orientation action occupies pose dims 3:6 in [-1, 1]; scaling by rot_action_bounds
        gives the commanded axis-angle rotation vector (rad) about the EEF x/y/z axes, reported
        here in degrees. The pre-EMA slice comes from the raw action stashed by the base wrapper
        (``_raw_action``); the post-EMA slice from ``control_actions`` (the EMA'd action actually
        parsed into control targets). Six "(stat)" series (mean+std, no histogram). Most
        meaningful under full_orientation_control, the only mode that consumes all three rot
        components — it shows whether the policy is actually requesting the pitch needed to fight
        a tilted grasp, and how much the EMA damps that request."""
        if not hasattr(self.unwrapped, "extras"):
            return
        raw = getattr(self, "_raw_action", None)
        if raw is None:
            return
        bounds = self._rot_action_bounds_row                            # (1,3) rad
        raw_deg = torch.rad2deg(raw[:, 3:6] * bounds)                   # (E,3) pre-EMA command
        ema_deg = torch.rad2deg(self.control_actions[:, 3:6] * bounds)  # (E,3) post-EMA command
        to_log = self.unwrapped.extras["to_log"]
        for k, axis in enumerate(("roll", "pitch", "yaw")):
            to_log[f"RotationFrame/cmd_{axis}_raw (stat)"] = raw_deg[:, k]
            to_log[f"RotationFrame/cmd_{axis}_ema (stat)"] = ema_deg[:, k]

    def _log_rotation_frame_angle(self):
        """Log the angle (deg) between the policy's rotation-frame z-axis and the peg-tip z-axis.

        Only meaningful for gain_mapping="rotated" (the only mode that emits R); silently skipped
        otherwise. ``_R_frame`` is the stiffness rotation already composed to WORLD coordinates
        (same convention the rotated-stiffness-frame marker draws it with), so its z-axis is
        R[:, :, 2]; the peg-tip frame z-axis is the held asset's local +z mapped to world via
        held_quat. Tag carries exactly one '/'.
        """
        if self._mode != "rotated" or self._R_frame is None:
            return
        if not hasattr(self.unwrapped, "extras"):
            return
        env = self.unwrapped
        # _R_frame is now EEF-frame (R_(eef<-interaction)); compose with R_eef to get world axes.
        R_world = matrix_from_quat(env.fingertip_midpoint_quat) @ self._R_frame
        z_net = R_world[:, :, 2]                                         # (E,3) world z of R
        local_z = torch.zeros((self.num_envs, 3), device=self.device)
        local_z[:, 2] = 1.0
        z_peg = quat_apply(env.held_quat, local_z)                       # (E,3) world z of peg tip
        cos = (z_net * z_peg).sum(dim=1).clamp(-1.0, 1.0)                # both already unit-norm
        angle_deg = torch.rad2deg(torch.acos(cos))                      # (E,)
        self.unwrapped.extras["to_log"]["RotationFrame/z_angle"] = angle_deg

    def _log_stiffness_frame_metrics(self):
        """Log gauge-invariant stiffness-vs-peg-axis metrics for the translational K_eff block.

        Let ẑ be the unit peg axis (held asset local +z mapped to world via held_quat) and
        K = K_pose[:, :3, :3] the ACTUAL applied translational stiffness — the same 3x3 block
        the ellipsoid marker eigendecomposes. ẑ and K live in the SAME (world) frame, so every
        quantity below is invariant to how that shared frame is oriented:

          * k_axial          k∥ = ẑᵀKẑ                  stiffness presented along insertion
          * k_lateral        k⊥ = (tr K − k∥)/2         mean of the two in-plane stiffnesses.
                                                        Basis-free: tr K = Σ eᵢᵀKeᵢ over ANY
                                                        orthonormal basis, so the lateral pair
                                                        sums to tr K − k∥ regardless of how u,v
                                                        are oriented in the plane ⊥ ẑ.
          * anisotropy_ratio ρ = k∥/k⊥                  <1 ⇒ compliant-along / stiff-laterally
                                                        (usual insertion compliance); >1 reverse.
          * cross_coupling   ‖(I−ẑẑᵀ)Kẑ‖               sideways restoring force from a unit
                                                        axial push; exactly 0 iff ẑ is a
                                                        principal axis of K, and grows as the
                                                        stiffness ellipsoid tilts off the peg —
                                                        the off-diagonal expressiveness the
                                                        peg-aligned/diagonal baselines lack.
          * condition_number λmax/λmin                  conditioning of K (eigvalsh).

        Each is published per-env under a ``(dist)`` tag, so the agent emits its mean +
        std scalars and a histogram over each write interval (see BlockAgent). Logged for
        ALL gain_mapping modes (baselines included) so the comparison is apples-to-apples —
        baselines should show cross_coupling≈0 and ρ≈1. Tags carry exactly one '/'.
        """
        if not hasattr(self.unwrapped, "extras"):
            return
        env = self.unwrapped
        # _K_pose is now applied in the EEF frame; rotate the translational block to world so the
        # comparison against the (world) peg axis stays gauge-consistent.
        R_eef = matrix_from_quat(env.fingertip_midpoint_quat)           # (E,3,3) world<-eef
        K = R_eef @ self._K_pose[:, :3, :3] @ R_eef.transpose(1, 2)      # (E,3,3) world translational stiffness
        local_z = torch.zeros((self.num_envs, 3), device=self.device)
        local_z[:, 2] = 1.0
        z = quat_apply(env.held_quat, local_z)                          # (E,3) unit peg axis in world

        Kz = (K @ z.unsqueeze(-1)).squeeze(-1)                          # (E,3) K ẑ
        k_axial = (z * Kz).sum(dim=1)                                   # (E,) ẑᵀKẑ
        trace = K.diagonal(dim1=-2, dim2=-1).sum(dim=1)                 # (E,) tr K
        k_lateral = (trace - k_axial) * 0.5                             # (E,)
        eps = 1e-8
        anisotropy_ratio = k_axial / k_lateral.clamp_min(eps)          # (E,) ρ
        cross_coupling = (Kz - k_axial.unsqueeze(-1) * z).norm(dim=1)   # (E,) ‖(I−ẑẑᵀ)Kẑ‖
        evals = torch.linalg.eigvalsh(K)                               # (E,3) ascending; K symmetric PSD
        condition_number = evals[:, -1] / evals[:, 0].clamp_min(eps)   # (E,) λmax/λmin

        to_log = self.unwrapped.extras["to_log"]
        to_log["RotationFrame/k_axial (dist)"] = k_axial
        to_log["RotationFrame/k_lateral (dist)"] = k_lateral
        to_log["RotationFrame/anisotropy_ratio (dist)"] = anisotropy_ratio
        to_log["RotationFrame/cross_coupling (dist)"] = cross_coupling
        to_log["RotationFrame/condition_number (dist)"] = condition_number

        # Angle (deg) between the peg insertion axis (held asset local +z, == ẑ above) and the
        # socket axis (fixed asset local +z) — exactly the quantity the optional kp_z_align
        # reward squashes toward 0, surfaced here in degrees for EVERY gain_mapping (baselines
        # included) so orientation-alignment progress is visible even when the reward is off.
        # The "(stat)" suffix emits mean + std scalars only (no histogram), unlike "(dist)".
        socket_z = quat_apply(env.fixed_quat, local_z)                  # (E,3) world z of socket
        cos_ps = (z * socket_z).sum(dim=1).clamp(-1.0, 1.0)            # ẑ, socket_z already unit-norm
        peg_socket_angle = torch.rad2deg(torch.acos(cos_ps))           # (E,)
        to_log["RotationFrame/peg_socket_angle (stat)"] = peg_socket_angle

    def _update_frame_viz(self):
        """Draw the enabled eval-recording axis frames at the peg tip / EEF.

        Each marker is gated by its config toggle and lazy-created on first use; the rotated
        stiffness frame additionally requires gain_mapping="rotated" (else there's no R). All
        markers default off, so training pays nothing here.
        """
        if not (self._viz_rotation or self._viz_peg_tip or self._viz_eef
                or self._viz_hole or self._viz_ellipsoid or self._viz_interaction):
            return

        from .frame_viz import AxisFrameMarker, EllipsoidMarker

        env = self.unwrapped
        env_origins = env.scene.env_origins                                  # (E,3) world
        E = self.num_envs

        show_rotation = self._viz_rotation and self._mode == "rotated" and self._R_frame is not None

        # Control happens at the EEF now, so the rotated stiffness frame + stiffness ellipsoid are
        # drawn there (world<-eef rotation R_eef), not at the peg tip.
        R_eef = matrix_from_quat(env.fingertip_midpoint_quat)               # (E,3,3) world<-eef
        eef_pos_w = env.fingertip_midpoint_pos + env_origins

        # Peg-tip pose (only the peg-tip marker still needs it). held_base_pos is env-relative.
        if self._viz_peg_tip:
            from isaaclab_tasks.direct.factory import factory_utils
            held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
                env.held_pos, env.held_quat, env.cfg_task.name,
                env.cfg_task.fixed_asset_cfg, E, self.device,
            )
            tip_pos_w = held_base_pos + env_origins

        if show_rotation:
            if self._marker_rotation is None:
                self._marker_rotation = AxisFrameMarker(
                    "/World/Visuals/RotatedStiffnessFrame", self._rotation_frame_scale)
            # _R_frame is EEF-frame (R_(eef<-interaction)); compose to world, draw at the EEF.
            self._marker_rotation.update(eef_pos_w, quat_from_matrix(R_eef @ self._R_frame))

        if self._viz_peg_tip:
            if self._marker_peg_tip is None:
                self._marker_peg_tip = AxisFrameMarker(
                    "/World/Visuals/PegTipFrame", self._peg_tip_frame_scale)
            # The held asset's ACTUAL orientation (carries the grasp tilt). The stiffness R is
            # anchored to the peg-nominal frame F_flip and the real peg sits at F_flip @ grasp_tilt,
            # so this frame coincides with the rotated-stiffness frame exactly when
            # fixed_rotation_rpy == rel_grasp_rot_init_deg — that overlap is the alignment check.
            self._marker_peg_tip.update(tip_pos_w, held_base_quat)

        if self._viz_eef:
            if self._marker_eef is None:
                self._marker_eef = AxisFrameMarker(
                    "/World/Visuals/EEFFrame", self._eef_frame_scale)
            # The actual fingertip (EEF) orientation. The rotated stiffness R is anchored to the
            # peg-nominal frame F_flip = R_eef @ R_flip — this EEF frame flipped 180° about y — and
            # the stiffness frame sits fixed_rotation_rpy from THAT (equivalently, it coincides with
            # the actual peg-tip frame when fixed_rotation_rpy == rel_grasp_rot_init_deg).
            self._marker_eef.update(
                env.fingertip_midpoint_pos + env_origins, env.fingertip_midpoint_quat)

        if self._viz_hole:
            if self._marker_hole is None:
                self._marker_hole = AxisFrameMarker(
                    "/World/Visuals/HoleFrame", self._hole_frame_scale)
            # The socket insertion-target frame: where the held asset's geometric base must arrive for
            # success, in the fixed asset's orientation (factory_utils.get_target_held_base_pose bakes
            # in the task/FORGE fixed-asset offsets). fixed_pos is env-relative (like held_pos), so add
            # env_origins; the peg-tip frame coincides with this exactly on a successful insertion.
            from isaaclab_tasks.direct.factory import factory_utils
            hole_pos, hole_quat = factory_utils.get_target_held_base_pose(
                env.fixed_pos, env.fixed_quat, env.cfg_task.name,
                env.cfg_task.fixed_asset_cfg, E, self.device,
            )
            self._marker_hole.update(hole_pos + env_origins, hole_quat)

        if self._viz_interaction and hasattr(env, "interaction_pos"):
            if self._marker_interaction is None:
                self._marker_interaction = AxisFrameMarker(
                    "/World/Visuals/InteractionFrame", self._interaction_frame_scale)
            # The env-defined contact frame, drawn at the contact point — ONLY while in contact.
            # VisualizationMarkers draws every instance (no per-instance hide), so non-contact envs
            # are parked far below the ground to keep them off-camera.
            pos_w = env.interaction_pos + env_origins
            exists = getattr(env, "interaction_exists", None)
            if exists is not None:
                hidden = pos_w.clone()
                hidden[:, 2] = -1000.0
                pos_w = torch.where(exists.unsqueeze(-1), pos_w, hidden)
            self._marker_interaction.update(pos_w, env.interaction_quat)

        if self._viz_ellipsoid:
            if self._marker_ellipsoid is None:
                self._marker_ellipsoid = EllipsoidMarker(
                    "/World/Visuals/StiffnessEllipsoid", opacity=self._ellip_opacity)
            # Principal axes/magnitudes of the ACTUAL applied translational stiffness (position 3x3
            # block of K_pose) via eigendecomposition — correct for every gain_mapping, including
            # cholesky coupling. eigh returns ascending eigenvalues with orthonormal eigenvector
            # columns; column i is the principal axis for eigenvalue i.
            evals, evecs = torch.linalg.eigh(self._K_pose[:, :3, :3])     # (E,3), (E,3,3)
            # Linear eigenvalue -> semi-axis map over the position gain range; compliance reverses it
            # (stiff axis short). Clamp so out-of-range eigenvalues land on the endpoints.
            t = ((evals - self._ellip_lo) / self._ellip_den).clamp(0.0, 1.0)
            if self._ellipsoid_compliance:
                t = 1.0 - t
            scales = self._ellip_min + t * (self._ellip_max - self._ellip_min)   # (E,3)
            # eigh's eigenvector matrix may be improper (det -1); flip one column so it is a valid
            # rotation for quat_from_matrix (axis sign is irrelevant for a symmetric ellipsoid).
            flip = (torch.linalg.det(evecs) < 0).unsqueeze(-1)                   # (E,1)
            col0 = torch.where(flip, -evecs[:, :, 0], evecs[:, :, 0])
            evecs = torch.stack((col0, evecs[:, :, 1], evecs[:, :, 2]), dim=2)
            # eigenvectors are in the EEF frame; rotate to world and draw at the EEF.
            self._marker_ellipsoid.update(eef_pos_w, quat_from_matrix(R_eef @ evecs), scales)

    def _build_pose_KD(self, a):
        """Pose stiffness K and critically-damped D, per ``self._mode``.

        Diagonal modes use D = 2*sqrt(K) per axis; ``rotated`` rotates that critically-
        damped diagonal (== matrix critical damping since K = R diag(k) Rᵀ); ``cholesky``
        uses the full matrix critical damping 2*V*sqrt(Λ)*Vᵀ (see ``_crit_damp_matrix``).

        For ``rotated`` / ``cholesky`` the policy controls the position 3x3 block's gains by
        default; with ``full_gain_matrix`` it controls all 6 DOFs (position + orientation).
        ``rotated`` always applies the shared R to BOTH the position and orientation blocks —
        ``full_gain_matrix`` only switches the orientation diagonal gains between policy-set
        and the constants, never whether they are rotated. ``cholesky``'s non-full orientation
        block is the constant diagonal (``_const_rot_block``), since it has no rotation frame.
        """
        E = self.num_envs
        if self._mode == "constant":
            kdiag = self._k_const.unsqueeze(0).expand(E, 6)
            return torch.diag_embed(kdiag), torch.diag_embed(self._crit_damp(kdiag))

        if self._mode == "variable_diagonal":
            kdiag = geom_scale(a, self._k_lo, self._k_hi)
            return torch.diag_embed(kdiag), torch.diag_embed(self._crit_damp(kdiag))

        if self._mode == "rotated":
            # Rotate a critically-damped diagonal by R. R is shared across blocks and rotates
            # BOTH the position and orientation blocks in every case; ``full_gain_matrix`` only
            # decides whether the orientation diagonal gains are policy-set or held at the
            # constants — never whether they're rotated. R is anchored to the peg-nominal frame
            # F_flip (see _rotation_frame): with ``fixed_rotation_rpy`` the local frame is the
            # config constant, otherwise it's decoded from the trailing 6 action dims.
            R = self._rotation_frame(a)                                            # (E,3,3)
            self._R_frame = R  # cached for the eval-recording rotated-stiffness-frame marker
            if self._full:
                kdiag = geom_scale(a[:, 0:6], self._k_lo, self._k_hi)              # (E,6)
            else:
                # Position gains policy-set; orientation gains held at the constants.
                kpos = geom_scale(a[:, 0:3], self._k_lo[:, 0:3], self._k_hi[:, 0:3])  # (E,3)
                krot = self._k_const[3:6].unsqueeze(0).expand(E, 3)                   # (E,3)
                kdiag = torch.cat((kpos, krot), dim=1)                                # (E,6)
            K = self._rotate_blockdiag(R, kdiag)
            D = self._rotate_blockdiag(R, self._crit_damp(kdiag))
            return K, D

        # cholesky: full SPD K with matrix critical damping. Full mode builds the whole 6x6;
        # otherwise the policy sets the position 3x3 and the rotation block is constant.
        if self._full:
            K = self._build_cholesky_block(a, self._k_lo, self._k_hi)
            return K, self._crit_damp_matrix(K)
        Kpos = self._build_cholesky_block(a, self._k_lo[:, 0:3], self._k_hi[:, 0:3])
        Krot, Drot = self._const_rot_block(self._k_const, E)
        return block_diag_2(Kpos, Krot), block_diag_2(self._crit_damp_matrix(Kpos), Drot)

    def _build_force_K(self, a):
        """Force stiffness K_f (no damping), mirroring the pose-K formulation exactly.

        For rotated/cholesky the policy controls only the force (translation) 3x3 block's
        gains by default; with ``full_gain_matrix`` it controls all 6 DOFs. In ``rotated`` the
        shared R rotates both the force and torque blocks regardless — full mode only switches
        the torque gains between policy-set and the constants. In ``cholesky`` the non-full
        torque block is the constant diagonal.
        """
        E = self.num_envs
        if self._mode == "constant":
            return torch.diag_embed(self._kf_const.unsqueeze(0).expand(E, 6))
        if self._mode == "variable_diagonal":
            return torch.diag_embed(geom_scale(a, self._kf_lo, self._kf_hi))
        if self._mode == "rotated":
            R = self._rotation_frame(a)
            if self._full:
                kdiag = geom_scale(a[:, 0:6], self._kf_lo, self._kf_hi)
            else:
                # Force (translation) gains policy-set; torque gains held at the constants;
                # the shared R rotates both blocks either way (mirrors pose K).
                kpos = geom_scale(a[:, 0:3], self._kf_lo[:, 0:3], self._kf_hi[:, 0:3])
                ktor = self._kf_const[3:6].unsqueeze(0).expand(E, 3)
                kdiag = torch.cat((kpos, ktor), dim=1)
            return self._rotate_blockdiag(R, kdiag)
        # cholesky
        if self._full:
            return self._build_cholesky_block(a, self._kf_lo, self._kf_hi)
        Kpos = self._build_cholesky_block(a, self._kf_lo[:, 0:3], self._kf_hi[:, 0:3])
        Ktor, _ = self._const_rot_block(self._kf_const, E)
        return block_diag_2(Kpos, Ktor)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _crit_damp(kdiag):
        """Critical damping for a diagonal stiffness vector: D = 2*sqrt(K)."""
        return 2.0 * torch.sqrt(kdiag.clamp_min(0.0))

    def _rotation_frame(self, a):
        """(E,3,3) stiffness rotation ``R = R_(eef←interaction)`` for ``rotated`` mode.

        New convention: gains are authored in the INTERACTION frame and rotated into the EEF
        frame (where control now happens) — ``K_eef = R · diag(k) · Rᵀ``. ``R`` IS that
        interaction→EEF rotation, supplied directly by the policy (Gram-Schmidt of the trailing
        6 action dims) or the config (``fixed_rotation_rpy``). No world composition and no
        ``F_flip`` peg-nominal anchor anymore: the pose error the gains multiply is itself
        expressed in the EEF frame, so ``R`` alone places the stiffness axes.

        NOTE (viz/metrics): the cached ``self._R_frame`` is now EEF-frame, not world. The
        rotated-stiffness-frame marker, ``_log_rotation_frame_angle`` and
        ``_log_stiffness_frame_metrics`` still assume world axes — they're corrected in the
        visualization stage (compose with ``R_eef`` for drawing / metrics).
        """
        if self._fixed_rot:
            return self._R_fixed.expand(a.shape[0], 3, 3)
        return rotation_6d_to_matrix(a[:, -6:])

    @staticmethod
    def _rotate_blockdiag(R, kdiag):
        """Block-diagonal congruence ``blkdiag(R,R) diag(k) blkdiag(R,R)ᵀ`` for (E,6) ``k``.

        The shared 3x3 rotation ``R`` is applied to both the position (k[:,0:3]) and
        orientation (k[:,3:6]) sub-blocks, yielding a (E,6,6) block-diagonal SPD matrix
        whose translation/orientation blocks stay decoupled.
        """
        RT = R.transpose(1, 2)
        top = R @ torch.diag_embed(kdiag[:, 0:3]) @ RT
        bot = R @ torch.diag_embed(kdiag[:, 3:6]) @ RT
        return block_diag_2(top, bot)

    def _const_rot_block(self, k_const, E):
        """Constant rotation/torque (K, D) 3x3 blocks from ``k_const[3:6]``.

        Matches how ``constant`` mode sets gains, so the rotation (pose) / torque (force)
        axes keep fixed gains in the rotated/cholesky formulations instead of being
        policy-driven. Pass the pose constants for K/D; force callers pass the force
        constants and ignore the returned D.
        """
        krot = k_const[3:6].unsqueeze(0).expand(E, 3)   # (E,3)
        return torch.diag_embed(krot), torch.diag_embed(self._crit_damp(krot))

    @staticmethod
    def _crit_damp_matrix(K, zeta=1.0):
        """Matrix critical damping for a symmetric PSD stiffness K (E,n,n).

        Eigendecompose K = V diag(λ) Vᵀ (orthonormal V) and return
        ``D = 2 V diag(ζ √λ) Vᵀ`` with ζ=1 for critical damping. Batched and fully on
        the input device/dtype via ``torch.linalg.eigh`` (no host transfers), so it
        parallelizes across envs. Reduces to ``D = 2√K`` when K is diagonal.
        """
        # Symmetrize for numerical safety (eigh reads the lower triangle anyway) and clamp
        # eigenvalues at 0 to guard against tiny negative round-off in the PSD K.
        Ksym = 0.5 * (K + K.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(Ksym)                 # (E,n), (E,n,n)
        d = 2.0 * zeta * torch.sqrt(evals.clamp_min(0.0))      # (E,n)
        return evecs @ torch.diag_embed(d) @ evecs.transpose(-1, -2)

    def _build_cholesky_block(self, a, lo, hi):
        """Single n×n SPD block in variance/correlation form from a leading action slice.

        ``n`` is inferred from ``lo`` (3 for the position-only block, 6 for the full matrix);
        ``a`` is consumed as [diag(n), off(n*(n-1)/2)]. The block is ``K = diag(√k)·Corr·diag(√k)``:

          * ``k = geom_scale(diag, lo, hi)`` sets the diagonal stiffness EXACTLY in ``[lo, hi]``;
          * ``Corr = C Cᵀ`` is a correlation matrix, where C is the lower-triangular factor with
            unit diagonal and strictly-lower entries ``chol_offdiag_rho * clamp(off, -1, 1)``,
            then row-normalized to unit length.

        Because the rows of C are unit-norm, ``Corr`` has unit diagonal and (Cauchy–Schwarz)
        ``|Corr[i,j]| = |⟨row_i, row_j⟩| ≤ 1``, so the congruence gives ``diag(K) = k`` exactly
        and ``|K[i,j]| ≤ √(k_i k_j)`` — K stays SPD and the diagonal can never exceed ``hi`` no
        matter the coupling. ``rho`` is the soft knob: ``rho=0`` -> pure diagonal (no coupling);
        a single off-diagonal entry is capped at ``|Corr| ≤ rho/√(1+rho²)``, rising toward the
        ±1 ceiling as rho grows. (Replaces the old raw ``K=LLᵀ`` form whose off-diagonal energy
        added into — and blew up — the diagonal, compounded by an off-diagonal scaled in
        stiffness units (~√(lo·hi)) instead of √-stiffness units.)
        """
        n = lo.shape[-1]
        n_off = n * (n - 1) // 2
        k = geom_scale(a[:, 0:n], lo, hi)                  # (E,n) diagonal stiffness in [lo, hi]
        s = torch.sqrt(k.clamp_min(0.0))                   # √k for the diag(√k)·Corr·diag(√k) congruence
        off = self._chol_rho * a[:, n:n + n_off].clamp(-1.0, 1.0)  # strictly-lower entries of C (tril order)
        C = build_lower_triangular(torch.ones_like(k), off)
        C = C / C.norm(dim=2, keepdim=True)                # normalize each row to unit length
        corr = C @ C.transpose(1, 2)                       # correlation matrix (unit diagonal)
        S = torch.diag_embed(s)
        return S @ corr @ S                                # SPD, diag(K) = k exactly
