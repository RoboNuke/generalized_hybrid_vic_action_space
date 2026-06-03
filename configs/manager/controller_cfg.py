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
