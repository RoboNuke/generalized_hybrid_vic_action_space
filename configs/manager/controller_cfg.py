"""Control configuration — a single dataclass shared by every control wrapper.

``ControlCfg`` subclasses Isaac Lab's :class:`ForgeCtrlCfg` (which subclasses the Factory
:class:`CtrlCfg`), so every operational-space control field the Forge/Factory env already
defines (``ema_factor``, ``pos_action_bounds``, ``default_task_prop_gains``, ``kp_null``,
``default_dead_zone``, …) is inherited here rather than re-declared. The runner copies these
inherited fields straight onto ``env_cfg.ctrl`` (see ``learning/runner.py``), so the control
gains can be tuned from the experiment YAML in ONE place and flow through to the Forge env.

On top of those inherited fields it adds the wrapper-specific knobs (force control, variable
impedance, the action-space formulation), using ONE shared naming scheme across all wrappers:

* stiffness / damping bounds are per-axis length-6 lists ``[x, y, z, Rx, Ry, Rz]``:
  ``gain_min``/``gain_max`` (pose stiffness K), ``damping_min``/``damping_max`` (pose damping
  D, used by the raw ``hybrid-vic`` wrapper), ``force_gain_min``/``force_gain_max`` (force
  stiffness K_f).

``control_type`` selects the wrapper the runner attaches:
  * ``"pose"``                  — base Forge/Factory pose controller (no wrapper).
  * ``"pose-VICES"``            — :class:`VICPoseWrapper` (commands translational stiffness).
  * ``"hybrid"``                — :class:`HybridForcePositionWrapper`.
  * ``"hybrid-vic"``            — :class:`HybridVICWrapper` (raw 6x6 K, D, K_f matrices).
  * ``"ctrl-action-interface"`` — :class:`CtrlActionInterfaceWrapper` (config-selectable
    ``gain_mapping``: constant / variable_diagonal / cholesky / rotated).

NOTE: importing this module imports Isaac Lab (ForgeCtrlCfg), so it must be imported only
after the Omniverse ``AppLauncher`` has booted (the runner does this).
"""

from isaaclab.utils import configclass
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg

CONTROL_TYPES = ("pose", "pose-VICES", "hybrid", "hybrid-vic", "ctrl-action-interface")

# Allowed K / D / K_f action-space formulations for CtrlActionInterfaceWrapper.
GAIN_MAPPINGS = ("constant", "variable_diagonal", "cholesky", "rotated")


@configclass
class ControlCfg(ForgeCtrlCfg):
    """Unified control config (registered YAML header ``controller_cfg``).

    Inherits all Forge/Factory ``ctrl`` fields; adds the wrapper knobs below.
    """

    control_type: str = "pose"

    # ---- variable-impedance bounds (per-axis length-6 [x, y, z, Rx, Ry, Rz]) ----
    # Geometric scaling (ctrl-action-interface) needs a lower bound > ~1e-6; a smaller lower
    # bound disables that channel (k=0) instead of producing NaN.
    gain_min: list = [100.0, 100.0, 100.0, 5.0, 5.0, 5.0]
    gain_max: list = [2000.0, 2000.0, 2000.0, 100.0, 100.0, 100.0]
    damping_min: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    damping_max: list = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    force_gain_min: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    force_gain_max: list = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]

    # ---- VICPoseWrapper ----
    apply_ema_to_gains: bool = False

    # ---- hybrid force/position control ----
    # force_axes: length-6 binary [x, y, z, Rx, Ry, Rz]; 1 => axis eligible for force control.
    force_axes: list = [0, 0, 0, 0, 0, 0]
    force_action_bounds: list = [50.0, 50.0, 50.0]
    torque_action_bounds: list = [5.0, 5.0, 5.0]
    force_action_threshold: list = [5.0, 5.0, 5.0]
    torque_action_threshold: list = [1.0, 1.0, 1.0]
    # Diagonal force stiffness used by HybridForcePositionWrapper and as the constant K_f
    # (ctrl-action-interface "constant" mode).
    default_task_force_gains: list = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    async_z_force_bounds: bool = False
    apply_ema_force: bool = False
    no_sel_ema: bool = False
    target_init_mode: str = "zero"
    ema_mode: str = "action"  # "action" or "wrench"
    use_delta_force: bool = False

    # ---- ctrl-action-interface formulation ----
    gain_mapping: str = "variable_diagonal"
    use_hybrid_force: bool = False
    chol_offdiag_rho: float = 0.0
    # cholesky/rotated only: learn the full 6-DOF gain matrix (position + orientation)
    # instead of the translational 3x3 block alone (rotation block fixed to constant gains).
    full_gain_matrix: bool = False

    # rotated only: when set to [roll, pitch, yaw] (DEGREES) the K/D/K_f rotation frame R
    # is FIXED to this orientation instead of being emitted by the policy. Drops the 6 rot6d
    # action dims (and their force-block mirror when use_hybrid_force); the policy still sets
    # the diagonal gains. None => R is learned from the action space (default rotated mode).
    # NOTE: in either case R is anchored to the held asset's NOMINAL in-hand frame F_flip (the EEF
    # flipped 180° about y, matching factory's grasp convention), not world or the raw EEF — it is
    # composed to world before building K, so the stiffness axes track the gripper AND this rpy
    # means the same orientation as rel_grasp_rot_init_deg on every axis. So [0,30,0] here lines the
    # stiffness ellipsoid up with the same-valued grasp tilt. This rpy is the orientation in F_flip.
    fixed_rotation_rpy: list | None = None

    # When True, command full 3-DOF orientation (roll/pitch/yaw) via a delta-axis-angle
    # rotation scaled by rot_action_bounds, instead of the default yaw-only map (roll/pitch
    # forced to 0). Also un-zeros the orientation observation channels (AutoMate adapter).
    # Remember to set the actor's force_zero_action_dims=null so Rx/Ry are learnable.
    full_orientation_control: bool = False

    # Orientation OBSERVATION representation (policy + critic). "quat" keeps the raw (w,x,y,z)
    # quaternion; "6d_rot_mat" swaps every quaternion-valued orientation channel (fingertip /
    # held / fixed) for the 6-D rotation-matrix rep (first two columns of R; Zhou et al. 2019) --
    # continuous, sign-unambiguous, smoother to optimize (no double-cover / unit-norm pathology).
    # Wired in learning.env_setup before gym.make (it resizes the obs); physics/control unchanged.
    orientation_obs_mode: str = "quat"

    # ---- eval-recording frame visualization (ctrl-action-interface only) ----
    # Each marker is an independent RGB coordinate-axis frame drawn into the sim (captured by
    # the recorder camera). All default OFF so training is unaffected; enable per-marker in the
    # eval/record overlay. Each has its own axis length (meters); co-located peg-tip frames are
    # disambiguated by giving them different scales.
    #
    # rotated stiffness frame (the rotation R), drawn at the peg tip; only meaningful when
    # gain_mapping="rotated" (silently skipped in other modes).
    visualize_rotation_frame: bool = False
    rotation_frame_axis_scale: float = 0.05
    # peg-tip frame (the held asset's own orientation), drawn at the peg tip.
    visualize_peg_tip_frame: bool = False
    peg_tip_frame_axis_scale: float = 0.04
    # EEF frame (the fingertip_midpoint orientation), drawn at the robot end-effector. This is
    # the frame the rotated stiffness R is defined relative to, so the stiffness frame should be
    # exactly fixed_rotation_rpy away from it.
    visualize_eef_frame: bool = False
    eef_frame_axis_scale: float = 0.06
    # hole frame (the socket insertion-target frame), drawn at the fixed asset. This is the pose
    # the peg-tip frame must reach for success (factory_utils.get_target_held_base_pose, which bakes
    # in the task/FORGE fixed-asset offsets), so the peg-tip and hole frames coincide on insertion.
    visualize_hole_frame: bool = False
    hole_frame_axis_scale: float = 0.05
    # interaction frame (env-defined contact frame: z = surface normal at the contact point,
    # x = motion/goal direction), drawn at the contact point ONLY while in contact. Supplied by
    # the env (env.interaction_pos/quat/exists) — surface task computes its own; peg tasks get
    # it from InteractionFrameWrapper. Silently skipped when the env exposes no interaction frame.
    visualize_interaction_frame: bool = False
    interaction_frame_axis_scale: float = 0.04
    # Translational stiffness ellipsoid (the position 3x3 block of K), drawn at the peg tip. Its
    # principal axes/magnitudes come from an eigendecomposition of the ACTUAL applied K_pos, so it
    # is correct for every gain_mapping (including cholesky coupling, which no single frame shows).
    # Each eigenvalue is LINEARLY mapped from the position gain range [min(gain_min[:3]),
    # max(gain_max[:3])] to a semi-axis length in [min_scale, max_scale] meters (out-of-range
    # eigenvalues clamp to the endpoints). compliance=False => STIFF axis is LONG (length grows with
    # gain); compliance=True reverses the map so a stiff axis is SHORT (displacement-under-unit-force
    # intuition). Defaults: 0.2 cm at the min gain, 2 cm at the max gain.
    visualize_stiffness_ellipsoid: bool = False
    stiffness_ellipsoid_min_scale: float = 0.002   # semi-axis length (m) at the min gain
    stiffness_ellipsoid_max_scale: float = 0.02    # semi-axis length (m) at the max gain
    stiffness_ellipsoid_compliance: bool = False   # False=stiffness (stiff=long), True=compliance (stiff=short)
    stiffness_ellipsoid_opacity: float = 0.35      # surface opacity [0..1]; lower = more see-through
    # NOTE: the peg-tip position is read directly from the env's geometric base frame
    # (factory_utils.get_held_base_pose, which bakes in the task/FORGE asset offsets) — there is
    # deliberately no peg-tip offset knob here to keep in sync.

    def __post_init__(self):
        if self.control_type not in CONTROL_TYPES:
            raise ValueError(
                f"ControlCfg.control_type must be one of {CONTROL_TYPES}, got {self.control_type!r}"
            )
        if self.gain_mapping not in GAIN_MAPPINGS:
            raise ValueError(
                f"ControlCfg.gain_mapping must be one of {GAIN_MAPPINGS}, got {self.gain_mapping!r}"
            )
        if self.ema_mode not in ("action", "wrench"):
            raise ValueError(f"ControlCfg.ema_mode must be 'action' or 'wrench', got {self.ema_mode!r}")
        if len(self.force_axes) != 6 or not set(self.force_axes) <= {0, 1}:
            raise ValueError(
                f"ControlCfg.force_axes must be a length-6 binary vector, got {self.force_axes!r}"
            )
        # Length-6 per-axis bound lists.
        for name in ("gain_min", "gain_max", "damping_min", "damping_max",
                     "force_gain_min", "force_gain_max", "default_task_force_gains"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"ControlCfg.{name} must be length 6, got {getattr(self, name)!r}")
        # Length-3 force/torque vectors.
        for name in ("force_action_bounds", "torque_action_bounds",
                     "force_action_threshold", "torque_action_threshold"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"ControlCfg.{name} must be length 3, got {getattr(self, name)!r}")
        # Bound ordering (lo <= hi element-wise).
        for lo_name, hi_name in (("gain_min", "gain_max"),
                                 ("damping_min", "damping_max"),
                                 ("force_gain_min", "force_gain_max")):
            lo, hi = getattr(self, lo_name), getattr(self, hi_name)
            if not all(a <= b for a, b in zip(lo, hi)):
                raise ValueError(
                    f"ControlCfg requires element-wise {lo_name} <= {hi_name}; got {lo} !<= {hi}"
                )
        if self.chol_offdiag_rho < 0.0:
            raise ValueError(f"ControlCfg.chol_offdiag_rho must be >= 0, got {self.chol_offdiag_rho!r}")
        if self.fixed_rotation_rpy is not None:
            if self.gain_mapping != "rotated":
                raise ValueError(
                    "ControlCfg.fixed_rotation_rpy is only valid with gain_mapping='rotated'; "
                    f"got gain_mapping={self.gain_mapping!r}."
                )
            if len(self.fixed_rotation_rpy) != 3:
                raise ValueError(
                    "ControlCfg.fixed_rotation_rpy must be length 3 [roll, pitch, yaw] (degrees), "
                    f"got {self.fixed_rotation_rpy!r}"
                )

        # control_type-specific force_axes consistency (each wrapper also re-checks).
        n_force = sum(self.force_axes)
        if self.control_type == "hybrid" and n_force < 1:
            raise ValueError(
                f"control_type='hybrid' requires >=1 force-eligible axis; got force_axes={self.force_axes!r}."
            )
        if self.control_type == "ctrl-action-interface":
            if self.use_hybrid_force and n_force < 1:
                raise ValueError(
                    "use_hybrid_force=True requires >=1 force-eligible axis; "
                    f"got force_axes={self.force_axes!r}."
                )
            if not self.use_hybrid_force and n_force != 0:
                raise ValueError(
                    "use_hybrid_force=False requires force_axes all-zero; "
                    f"got force_axes={self.force_axes!r}."
                )
