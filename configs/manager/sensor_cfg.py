"""Sensor configuration loaded from YAML (registered section ``sensor_cfg``).

Currently carries the contact-sensor settings used by
:mod:`wrappers.sensors.contact_sensor_wrapper` to add an *in-contact* boolean
(3-D, one flag per task-space translation axis x/y/z) to Forge/Factory
peg-insertion tasks. The boolean is derived from an IsaacLab
:class:`~isaaclab.sensors.ContactSensor` mounted on the held asset (peg) and
filtered against the fixed asset (hole) — it does NOT use the robot's joint
force-torque reading (the "contact-sensor version" of the upstream
Continuous_Force_RL implementation).

This is logging-only: the wrapper publishes per-axis contact fractions through
the existing ``extras['to_log']`` framework (forwarded to TensorBoard by
:class:`~wrappers.scorers.reward_decomposition.RewardDecompositionWrapper`). It
does not grow the observation / state spaces.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class ContactCfg:
    """Contact-sensor settings.

    The sensor reads ``ContactSensor.data.force_matrix_w`` (the per-pair contact
    force between the held and fixed assets, in world frame), rotates it into the
    end-effector frame, and flags per-axis contact where ``|force_component| >
    contact_force_threshold``. A threshold of 1.0 N reproduces the upstream
    behaviour (a hardcoded ``atol=1.0`` in ``torch.isclose``).
    """

    enabled: bool = False
    """Master switch. When False the runner installs neither the sensor nor the wrapper."""

    contact_force_threshold: float = 1.0
    """Per-axis EE-frame contact-force threshold (N). |f_x|,|f_y|,|f_z| above this => in contact."""

    log_contact_state: bool = True
    """Publish per-axis ``Contact / In-Contact {X,Y,Z,Any}`` fractions via ``to_log``."""

    append_to_policy_obs: bool = False
    """Append the per-axis contact bool (x/y/z) to the POLICY observation (grows
    observation_space by 3). Lets the actor condition on contact. Requires ``enabled``."""

    append_to_critic_state: bool = False
    """Append the per-axis contact bool (x/y/z) to the CRITIC state (grows state_space by
    3; asymmetric tasks only). Requires ``enabled``."""

    # Prim-path expressions for the held (sensor) and fixed (filter) asset *roots*.
    # Forge/Factory spawn both with activate_contact_sensors=True; the contact-reporting
    # rigid body is an asset-specific CHILD prim of these roots (peg one level down, hole
    # two), which the wrapper auto-discovers from the cloned stage — so these stay at the
    # articulation roots and need no per-task tuning.
    held_prim_expr: str = "/World/envs/env_.*/HeldAsset"
    fixed_prim_expr: str = "/World/envs/env_.*/FixedAsset"


@dataclasses.dataclass(kw_only=True)
class EnergyCfg:
    """Per-step energy / effort metrics for Forge/Factory peg insertion.

    Logging-only: :class:`~wrappers.sensors.energy_metrics_wrapper.EnergyMetricsWrapper`
    reads the live robot state each step and publishes per-env tensors under the
    ``energy_metrics/`` TensorBoard tab via the existing ``extras['to_log']`` path.
    All values are instantaneous per-step quantities; ``max_force`` is reduced to the
    peak any env saw (via a ``(max)`` tag suffix), the rest to interval means.
    """

    enabled: bool = False
    """Master switch. When False the runner installs neither config nor wrapper."""

    log_max_force: bool = True
    """``energy_metrics/max_force (max)`` — peak FT-sensor force magnitude ‖f‖₂."""

    log_avg_force: bool = True
    """``energy_metrics/avg_force`` — mean FT-sensor force magnitude ‖f‖₂."""

    log_sq_joint_vel: bool = True
    """``energy_metrics/sq_joint_vel`` — Σ q̇² over the 7 arm DOFs."""

    log_sq_ee_vel: bool = True
    """``energy_metrics/sq_ee_vel`` — Σ v² over the 3 EE linear-velocity components."""

    log_work_power: bool = True
    """``energy_metrics/work_power`` — Σ τ·q̇ over arm DOFs, i.e. mechanical power (W)."""

    log_battery_power: bool = True
    """``energy_metrics/battery_power`` — ``battery_scale·Σ τ²`` copper-loss proxy for
    electrical draw (no motor model in sim, so a proxy is the best available)."""

    battery_scale: float = 1.0
    """Scale on the Σ τ² copper-loss proxy (unitless; I=τ/Kt ⇒ P_copper=I²R ∝ τ²)."""


@dataclasses.dataclass(kw_only=True)
class SensorCfg:
    """Top-level sensor config (registered YAML section ``sensor_cfg``)."""

    contact: ContactCfg = dataclasses.field(default_factory=ContactCfg)
    energy: EnergyCfg = dataclasses.field(default_factory=EnergyCfg)
