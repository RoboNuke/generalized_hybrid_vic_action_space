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

    # Prim-path expressions for the held (sensor) and fixed (filter) asset *roots*.
    # Forge/Factory spawn both with activate_contact_sensors=True; the contact-reporting
    # rigid body is an asset-specific CHILD prim of these roots (peg one level down, hole
    # two), which the wrapper auto-discovers from the cloned stage — so these stay at the
    # articulation roots and need no per-task tuning.
    held_prim_expr: str = "/World/envs/env_.*/HeldAsset"
    fixed_prim_expr: str = "/World/envs/env_.*/FixedAsset"


@dataclasses.dataclass(kw_only=True)
class SensorCfg:
    """Top-level sensor config (registered YAML section ``sensor_cfg``)."""

    contact: ContactCfg = dataclasses.field(default_factory=ContactCfg)
