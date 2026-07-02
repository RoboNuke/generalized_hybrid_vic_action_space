from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class ResetCurriculumCfg:
    """Sampling-Based Curriculum (SBC) on the initial peg pose for Forge/Factory peg insertion.

    Own config section (``reset_curriculum_cfg`` header) — kept out of ``runner_cfg`` because it is
    a self-contained subsystem consumed only by
    :func:`~wrappers.sensors.reset_curriculum_wrapper.install_reset_curriculum`.

    Each reset samples, per env and per axis independently, from ``[floor, max]`` where ``max`` is
    the FULL task (constant) and ``floor = min + c*(max-min)`` rises with the per-agent curriculum
    level ``c`` (IndustReal/AutoMate SBC). At ``c=0`` the whole task-space is sampled; as ``c->1``
    only the hardest spawns remain. Per-axis ``max``: x,y from ``hand_init_pos_noise[:2]``, z from
    ``hand_init_pos[2]``, tilt from ``runner_cfg.rel_grasp_rot_init_deg`` (the fixed grasp).
    ``num_agents`` and the grasp tilt are read from ``runner_cfg`` at install time.

    Requires ``runner_cfg.grasp_rot_mode == 'fixed'`` (constant grasp / weld angle).
    """

    enabled: bool = False
    """Master switch. When True, install the SBC reset curriculum (peg-insertion tasks only)."""

    increase_rate: float = 0.05
    """Per-update step added to a per-agent ``c`` (clipped to [0,1]) when its success EMA exceeds
    :attr:`increase_threshold`."""

    decrease_rate: float = 0.05
    """Per-update step subtracted from ``c`` when its success EMA is below :attr:`decrease_threshold`."""

    increase_threshold: float = 0.5
    """Per-agent per-reset-batch success rate above which ``c`` is raised (curriculum gets harder).
    The batch rate is a mean over that agent's ~num_envs envs (precise at 512 envs/agent), used raw
    (no EMA) — it will jitter batch-to-batch and that's fine. Note: at low ``c`` the SBC samples the
    FULL range, so this gates on full-range success — set it relative to what the policy realistically
    achieves across the whole task, not the easy end."""

    decrease_threshold: float = 0.2
    """Success-EMA below which an agent's ``c`` is lowered (curriculum backs off). Keep < increase."""

    min_pos: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.005])
    """Easy-end (c=0 floor) spawn offsets ``[x, y, z]`` in METERS. ``x``/``y`` are lateral-offset
    magnitudes (applied ±). ``z`` is the PEG TIP (held_base) height relative to the TOP of the hole
    (peg frame, NOT the gripper): 0 = tip at the hole mouth, negative = tip inside the bore, so the
    hole bottom is ``-fixed_asset_cfg.height``. The wrapper auto-converts to the gripper target using
    the measured fingertip->peg-tip offset, and still clamps to the never-in-success floor
    (:attr:`success_margin_z`). E.g. ``0.005 - hole_height`` = tip 5 mm above the hole bottom."""

    min_orn: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Easy-end (c=0 floor) peg tilt ``[roll, pitch, yaw]`` in DEGREES (same convention as
    ``rel_grasp_rot_init_deg``). ``[0,0,0]`` => start fully aligned. Per-axis max is the grasp tilt."""

    align_below_z: float = 0.0
    """Depth-safety guard (METERS). If a sampled fingertip height ``z`` is below this, that env is
    forced ALIGNED (tilt=0) AND CENTERED (x=y=0), so a peg spawning at/below the hole top goes
    straight down the bore instead of interpenetrating the hole wall at an angle. ``<= 0`` =>
    auto-derive from the grasp's peg-below-fingertip offset at the first reset (recommended)."""

    success_margin_z: float = 0.005
    """Keep the peg base at least this far (METERS) ABOVE the success depth, so no env ever starts
    already-successful. The wrapper measures the fingertip->peg-base map at the first reset and
    raises the z-sampling floor so ``peg_base_z_disp >= success_threshold*hole_height +
    success_margin_z`` always holds. Default 5 mm. Set 0 to disable."""
