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

    glue_peg_to_gripper: bool = False
    """Rigidly weld the held peg to the gripper for the whole rollout (Forge/Factory
    ``peg_insert`` only). When True, a per-env ``UsdPhysics.FixedJoint`` is authored between the
    fingertip body and the peg body before play, so the peg is mounted to the gripper instead of
    held by friction (which lets it creep). Forces are preserved (the welded peg stays dynamic;
    its peg-vs-socket reaction transmits through the joint). Installs
    :func:`~wrappers.sensors.peg_weld_wrapper.install_peg_weld`.

    Requires :attr:`grasp_rot_mode` ``== "fixed"``: a GPU-pipeline joint's local frame is parsed
    at play time and cannot change between steps, so the welded peg-in-gripper transform must be a
    constant — only well-defined for a deterministic (fixed) grasp. When enabled, the runner also
    forces the env's ``held_asset_pos_noise`` to zero (a nonzero per-reset in-grip position jitter
    would disagree with the constant weld frame and make PhysX fight the reset teleport)."""

    kp_z_align_enabled: bool = False
    """Add the ``kp_z_align`` orientation keypoint reward (Forge/Factory peg-insertion only).
    When True, a reward term pulls the held peg's z-axis onto the socket's z-axis — rewarding the
    orientation alignment that a tilted grasp (:attr:`grasp_rot_mode`) fights and that the base
    keypoint reward only weakly constrains. The term is ``squashing_fn(angle, a, b)`` over the
    peg-axis-vs-socket-axis angle (radians), summed into the reward with scale 1.0 (like the other
    ``kp_*`` terms) and published as ``logs_rew/kp_z_align``. The angle is computed from the true
    peg/socket poses (reward-side, privileged) and never enters any observation. Installs
    :func:`~wrappers.scorers.kp_z_align_reward.install_kp_z_align_reward`."""

    kp_z_align_a: float = 20.0
    """Steepness ``a`` of the ``kp_z_align`` squashing function ``1/(exp(a·x)+b+exp(-a·x))``.
    Larger => a sharper reward basin (only near-perfect alignment scores). Ignored when
    :attr:`kp_z_align_enabled` is False."""

    kp_z_align_b: float = 1.33
    """Offset ``b`` of the ``kp_z_align`` squashing function; sets the peak height ``1/(2+b)`` at
    perfect alignment (angle = 0). Ignored when :attr:`kp_z_align_enabled` is False."""

    curr_engaged_scale: float = 1.0
    """Reward scale on the per-step ``curr_engaged`` bonus (Forge/Factory peg insertion). Stock is
    1.0. This bonus fires every step the peg is merely NEAR the socket (``engage_threshold``, no
    centering/rotation requirement), so for a tilted/glued grasp it can dominate the return as a
    "hover at the mouth, jammed" local optimum. Lower it (e.g. 0.1–0.3) to stop that hovering from
    paying. Applied by patching ``FactoryEnv`` when this or any other insertion-reward knob
    (scales / ``*_check_yaw`` / ``*_check_z_aligned`` / ``ee_success_yaw_deg``) is off-default."""

    curr_success_scale: float = 1.0
    """Reward scale on the per-step ``curr_success`` bonus (full insertion: tight centering + near
    full depth). Stock is 1.0, which leaves it ~200× smaller than the accumulated engaged bonus, so
    the return is effectively blind to actual insertion. Raise it (e.g. 10–50) so inserting clearly
    beats hovering engaged."""

    kp_baseline_scale: float = 1.0
    """Multiplicative weight on the ``kp_baseline`` keypoint reward (stock 1.0). ``kp_baseline`` uses
    a broad squashing (``keypoint_coef_baseline`` ``a=5``), so it is nearly insensitive to widening
    ``keypoint_scale`` — when the keypoint spacing was increased, baseline kept its magnitude while
    coarse/fine shrank, so baseline now dominates the return ("near the hole" pays almost as well as
    inserting). Lower this (e.g. 0.2–0.5) to restore the baseline:coarse:fine balance. Applied by the
    insertion-reward patch (off-default value triggers it)."""

    kp_coarse_scale: float = 1.0
    """Multiplicative weight on the ``kp_coarse`` keypoint reward (stock 1.0; ``a=50``)."""

    kp_fine_scale: float = 1.0
    """Multiplicative weight on the ``kp_fine`` keypoint reward (stock 1.0; ``a=100``). The
    last-inch/insertion term — most suppressed by a wide ``keypoint_scale``; raise to re-emphasize
    final insertion."""

    engage_check_yaw: bool = False
    """Gate the ``curr_engaged`` bonus on the EE-yaw check (``check_yaw``, formerly ``check_rot``).
    NOTE: this only constrains END-EFFECTOR YAW (``curr_yaw < ee_success_yaw``) — NOT the peg's
    pitch/roll tilt — and the socket is randomly yawed, so it is rarely the right lever for a
    pitch-tilted grasp. For axis alignment use :attr:`engage_check_z_aligned`."""

    engage_check_z_aligned: bool = False
    """Gate the ``curr_engaged`` bonus on z-axis alignment: the angle between the held asset's
    z-axis and the fixed asset's z-axis must be < :attr:`z_align_max_deg`. Makes the dominant
    per-step engaged bonus require the peg to actually point down the socket (not just be near it),
    so a tilted/jammed hover stops paying."""

    success_check_yaw: bool = False
    """Gate the ``curr_success`` mask (reward AND logged success rate) on the EE-yaw check. The
    nut_thread task is always yaw-gated regardless; this adds it for peg/gear tasks."""

    success_check_z_aligned: bool = False
    """Gate the ``curr_success`` mask on z-axis alignment (< :attr:`z_align_max_deg`). Requires the
    peg to be correctly oriented to count as a success — useful as geometries get more complex."""

    z_align_max_deg: float = 15.0
    """Max angle (DEGREES) between the held and fixed asset z-axes for the ``check_z_aligned`` gate
    to pass. Shared by the engaged and success z-alignment gates. Only used when one is enabled."""

    ee_success_yaw_deg: float | None = None
    """Yaw threshold (DEGREES) for the ``check_yaw`` gate, written to ``env_cfg.task.ee_success_yaw``
    (radians) before ``gym.make``. ``None`` keeps the task default (0.0). Only meaningful when a
    ``check_yaw`` gate is active (e.g. :attr:`engage_check_yaw`, or the nut_thread success check)."""

    disable_success_pred: bool = False
    """Turn off the base Forge env's success-prediction head end to end (Isaac-Forge- tasks only).
    Forge's 7th action (``actions[:, 6]``) is a learned success predictor: it feeds the
    ``success_pred_error`` reward term and the ``early_term_*`` prediction-quality metrics, but never
    drives the controller. When ``True`` this single switch disables the whole mechanism:

    - **Reward off** — ``env_cfg.task.delay_until_ratio`` is forced to ``1.1`` (build_env), which
      ``true_successes.mean()`` can never reach, so the env's ``success_pred_scale`` stays ``0`` and
      ``success_pred_error`` contributes nothing to the return.
    - **Action force-zeroed** — index 6 is appended to the actor's ``force_zero_action_dims`` (runner),
      so the policy allocates no parameters to the inert head (output pinned to ``0`` ->
      ``policy_success_pred = 0.5``). NOTE: this changes the continuous action count, so re-tune
      ``target_entropy`` (it is a placeholder anyway).
    - **Metrics withheld** — :class:`~wrappers.scorers.forge.ForgeWrapper` stops publishing
      ``logs_rew/success_pred_error``, ``Episode_Reward/success_pred_error`` and the per-threshold
      ``early_term_*`` series.

    ``False`` (default) keeps the upstream Forge behavior. Requires an ``Isaac-Forge-`` task (the head
    is Forge-specific); a hard error otherwise."""

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
        # A constant gripper weld is only well-defined for a DETERMINISTIC grasp: the GPU pipeline
        # parses a joint's local frame at play time and cannot change it per step. 'random' grasp
        # tilt is therefore incompatible. 'fixed' (peg tasks) and 'none' (surface task, whose grasp
        # tilt comes from task.inhand_tilt_range_deg) are both deterministic and allowed; the
        # per-task consistency (which tilt source feeds the weld) is enforced in env_setup.build_env.
        if self.glue_peg_to_gripper and self.grasp_rot_mode == "random":
            raise ValueError(
                "RunnerCfg.glue_peg_to_gripper=True is incompatible with grasp_rot_mode='random' "
                "(the welded in-gripper transform is authored once before play and cannot change "
                "per step). Use grasp_rot_mode='fixed' (peg) or 'none' (surface)."
            )
        # Fragile pegs reset individual envs out of sync; the native Factory/Forge reset path
        # assumes all envs reset together, so the efficient per-env reset is mandatory.
        if self.fragile_peg_enabled and not self.efficient_reset_enabled:
            raise ValueError(
                "RunnerCfg.fragile_peg_enabled=True requires efficient_reset_enabled=True: "
                "broken pegs reset individual envs mid-episode, which the native "
                "Factory/Forge _reset_idx (written assuming all envs reset together) corrupts."
            )
