"""Runner-level configuration loaded from YAML.

Holds the values the runner used to take exclusively from CLI flags (task,
num_envs, num_agents, total_timesteps, memory_size, eval_timesteps, seed).
Moving them into YAML lets a launcher invoke the runner with just a config
path + experiment name, while CLI flags remain available as one-off overrides
when provided.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(kw_only=True)
class RunnerCfg:
    """Per-task runner-level hyperparameters."""

    task: str
    """Isaac Lab gym id, e.g. ``"Isaac-Lift-Cube-Franka-v0"``."""

    agent_type: str = "sac"
    """Which learning algorithm to run: ``"sac"`` (default; uses ``sac_cfg``) or
    ``"ppo"`` (uses ``ppo_cfg``). SAC stays the default so existing configs are
    unaffected."""

    env_cfg_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Free-form overrides applied to the parsed Isaac Lab ``env_cfg`` *before*
    ``gym.make``. Keys are dotted attribute paths resolved against the env_cfg
    dataclass tree (e.g. ``task.hand_init_pos_noise``). Values must be a primitive
    or a list of primitives — whatever the leaf field's type expects. Unknown
    leaves and non-dotted keys are hard errors. Empty dict = no overrides.

    Dotted segments are resolved by attribute access, or by key lookup when the
    current node is a ``dict`` — so AutoMate's per-task config (``env_cfg.tasks``,
    a plain dict) IS reachable, e.g. ``tasks.insertion.if_sbc: false`` to disable
    the SBC spawn curriculum. (``automate_assembly_id`` still selects the assembly.)"""

    automate_assembly_id: str | None = None
    """AutoMate-only: which plug/socket assembly to run (e.g. ``"00015"``, ``"00652"``).
    Selects the ``Isaac-AutoMate-Assembly-Direct-v0`` env's asset pair; the adapter
    recomputes the derived asset paths (``assembly_dir`` / ``disassembly_path_json``) from
    it. ``None`` keeps the upstream default (``00015``). Ignored for non-AutoMate tasks."""

    automate_grasp_pose: list[float] | None = None
    """AutoMate-only: peg IN-HAND offset (grasp uncertainty). The gripper always grasps at the
    nominal ``plug_grasps.json`` grasp (its world pose is fixed); these 6 floats
    ``[x, y, z, roll, pitch, yaw]`` shift the PEG relative to that grasp — position in METERS,
    orientation in DEGREES (XYZ-Euler) — in the plug's local frame. ``None`` / all-zero = no
    shift (nominal grasp). The peg is held by friction, so only modest offsets stay seated.
    Ignored for non-AutoMate tasks. See ``wrappers/sensors/automate_forge_adapter.py``."""

    grasp_rot_mode: str = "none"
    """How the held peg is grasped relative to the gripper (Forge/Factory peg-insertion only).
    One of:

    - ``"none"``   — upstream behavior: peg perfectly aligned with the gripper (no patch).
    - ``"random"`` — the peg is grasped at a RANDOM roll/pitch/yaw tilt, re-sampled per env every
      reset (simulates a suboptimal/uncertain grasp; the tilt is unobservable to the actor, so it
      must rely on force feedback). The per-axis ± ranges come from :attr:`rel_grasp_rot_init_deg`.
    - ``"fixed"``  — the peg is grasped at a CONSTANT signed tilt, the same every env every reset.
      The offset is :attr:`rel_grasp_rot_init_deg` interpreted as a literal ``[roll, pitch, yaw]``
      (signed, so negatives are allowed).

    For ``random``/``fixed`` the env's grasp transform (``get_handheld_asset_relative_pose``) is
    patched before ``gym.make`` (see ``wrappers/sensors/grasp_tilt_wrapper.py``)."""

    rel_grasp_rot_init_deg: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    """``[roll, pitch, yaw]`` in degrees, interpreted per :attr:`grasp_rot_mode`:
    when ``"random"`` it is the symmetric ± range, each axis sampled independently ``U(-val, +val)``
    per env per reset (e.g. ``[0, 3, 0]`` = pitch-only ±3°; components must be >= 0);
    when ``"fixed"`` it is a constant signed offset applied every reset (e.g. ``[0, -45, 0]`` =
    a steady -45° pitch). All-zero = no tilt. Ignored when ``grasp_rot_mode`` is ``"none"``."""

    num_envs: int
    """Envs PER agent. Total Isaac envs = ``num_envs * num_agents``."""

    num_agents: int = 1
    """Block-parallel agents trained simultaneously."""

    total_timesteps: int = 10_000
    """Training duration in **env_steps** (i.e. ``env.step()`` calls). Each env_step
    advances every parallel env by one tick, so total transitions written to the
    replay buffer = ``total_timesteps * num_envs * num_agents``. This is the value
    passed straight through to the trainer's ``timesteps`` field."""

    eval_timesteps: int = 250
    """Eval duration in env_steps. Used in place of ``total_timesteps`` when
    ``--mode eval`` is selected."""

    memory_size: int = 1_000_000
    """Replay buffer capacity PER AGENT. Per-env depth = ``memory_size // num_envs``."""

    seed: int = -1
    """Global seed. ``-1`` lets skrl pick a non-deterministic one."""

    fragile_peg_enabled: bool = False
    """Make the peg fragile (Forge / AutoMate-Assembly only). When True, an env whose
    contact-force magnitude ``‖force_sensor_smooth[:, :3]‖`` reaches :attr:`break_force`
    "breaks": that env is terminated immediately and reset on the spot (per-env reset, no
    full ``env.reset()``). Installs :class:`~wrappers.sensors.fragile_object_wrapper.FragileObjectWrapper`.
    Requires :attr:`efficient_reset_enabled` (broken envs reset out of sync, which the native
    Factory/Forge ``_reset_idx`` — written assuming all envs reset together — cannot handle)."""

    break_force: float = -1.0
    """Contact-force magnitude (N) on ``force_sensor_smooth[:, :3]`` at/above which a fragile
    peg breaks. ``-1.0`` = unbreakable (mapped to a huge value internally). When positive it
    ALSO caps the FORGE per-env *threshold force*: ``contact_penalty_threshold_range[1]`` is
    clamped to ``break_force`` before ``gym.make`` so the sampled threshold force (used in the
    obs ``force_threshold`` and the contact-penalty reward) can never exceed the break force.
    Only used when :attr:`fragile_peg_enabled`."""

    efficient_reset_enabled: bool = False
    """Use the efficient per-env reset (Forge / Factory / AutoMate-Assembly only). Installs
    :class:`~wrappers.sensors.efficient_reset_wrapper.EfficientResetWrapper`, which on a PARTIAL
    reset (a subset of envs, e.g. after a fragile-peg break) teleports a random donor env's
    cached post-reset state into the broken envs instead of running Factory's expensive
    all-envs settling/IK reset path. Required by :attr:`fragile_peg_enabled`; may also be used
    alone for any task that produces out-of-sync per-env resets."""

    def __post_init__(self):
        if self.grasp_rot_mode not in ("none", "random", "fixed"):
            raise ValueError(
                "RunnerCfg.grasp_rot_mode must be one of 'none', 'random', 'fixed', "
                f"got {self.grasp_rot_mode!r}"
            )
        if len(self.rel_grasp_rot_init_deg) != 3:
            raise ValueError(
                "RunnerCfg.rel_grasp_rot_init_deg must be length 3 [roll, pitch, yaw], "
                f"got {self.rel_grasp_rot_init_deg!r}"
            )
        # Only 'random' uses ± range semantics; 'fixed' is a signed offset (negatives allowed).
        if self.grasp_rot_mode == "random" and any(
            v < 0.0 for v in self.rel_grasp_rot_init_deg
        ):
            raise ValueError(
                "RunnerCfg.rel_grasp_rot_init_deg components must be >= 0 for "
                "grasp_rot_mode='random' (symmetric ± range), "
                f"got {self.rel_grasp_rot_init_deg!r}"
            )
        # Fragile pegs reset individual envs out of sync; the native Factory/Forge reset path
        # assumes all envs reset together, so the efficient per-env reset is mandatory.
        if self.fragile_peg_enabled and not self.efficient_reset_enabled:
            raise ValueError(
                "RunnerCfg.fragile_peg_enabled=True requires efficient_reset_enabled=True: "
                "broken pegs reset individual envs mid-episode, which the native "
                "Factory/Forge _reset_idx (written assuming all envs reset together) corrupts."
            )
