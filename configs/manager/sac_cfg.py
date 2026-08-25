from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class RecorderCfg:
    """Configures the optional ``RecordingWrapper`` that produces 3x4-grid GIFs and
    TB videos of vectorized rollouts during training. See ``wrappers/recording.py``.

    The recorder injects a single ``TiledCameraCfg`` into the env scene before
    ``gym.make()`` (so the camera prim is created per-env). When ``enabled`` is
    False, the wrapper is not constructed and no camera is added — zero overhead.
    """

    enabled: bool = False
    """Master switch. When False, no camera is added and the wrapper is not built."""

    record_every_k_resets: int = 20
    """Open a recording session every K-th observed *global* reset (a step where
    ``(terminated | truncated).all()`` is True). Per-env async resets in between
    do not count. The first session opens at the first global reset (unless
    ``record_on_start`` is set — then a session also opens on the very first step)."""

    record_on_start: bool = False
    """When True, open a recording session immediately on the first ``step()``
    instead of waiting for the first global reset. Lets the captured episode line
    up with the env's first rollout, so ``eval_timesteps`` can be a single episode
    (``max_episode_length``) rather than needing a spare warm-up episode. Default
    False preserves the train-time cadence (first session at the K-th global reset)."""

    width: int = 240
    """Per-tile camera width in pixels."""

    height: int = 180
    """Per-tile camera height in pixels."""

    camera_pos: tuple[float, float, float] = (1.0, 0.0, 0.35)
    """Camera position offset relative to each env's local origin. Default
    matches the RoboNuke/Continuous_Force_RL eval setup."""

    camera_quat: tuple[float, float, float, float] = (-0.3535534, 0.6123724, 0.6123724, -0.3535534)
    """Camera orientation as a (w, x, y, z) quaternion under the ROS
    convention. Default matches RoboNuke/Continuous_Force_RL."""

    focal_length: float = 24.0
    """PinholeCamera focal length (mm-equivalent). RoboNuke default."""

    focus_distance: float = 0.05
    """PinholeCamera focus distance. RoboNuke default."""

    horizontal_aperture: float = 20.955
    """PinholeCamera horizontal aperture (mm). RoboNuke default."""

    clipping_range: tuple[float, float] = (0.1, 20.0)
    """PinholeCamera near/far clip planes (m). RoboNuke default."""

    fps: int = 30
    """Playback rate for the saved video and TB video."""

    video_format: str = "mp4"
    """Output container for the recorded grid: ``"mp4"`` (H.264 — compact and smooth, the
    default and strongly recommended at video resolutions) or ``"gif"`` (huge and stuttery
    for large frames; kept only for compatibility)."""

    num_trajectories: int = 24
    """Used only by the standalone per-agent recorder (``learning/record.py`` /
    ``learning/recording_eval.py``): collect at least this many complete episode
    trajectories (running as many full-batch episodes as needed), then select the
    best-4 / median-4 / worst-4 by return for the 3x4 grid. Has no effect on the
    in-training ``RecordingWrapper``."""

    output_subdir: str = "videos"
    """Subdirectory under the SAC experiment dir where GIFs are written."""

    full_episode: bool = False
    """Recorder-only VIEWING toggle. When True, the collect loop disables success/lag
    early-termination AT RUNTIME (``terminate_on_success``/``terminate_on_lag`` -> False on the
    live env) so every tile plays the full episode instead of freezing the instant it succeeds.
    This is display-only: it changes episode LENGTH, never the dynamics/control/policy, and it is
    kept OUT of ``env_cfg_overrides`` on purpose (that path is guarded/forbidden for record
    overlays) so it can never masquerade as the training env. Use the eval path, not the recorder,
    for any quantitative metric."""

    mode: str = "trajectories"
    """WHAT to record — the single standalone-recorder dispatch (one collect path, no bespoke variants):
      * "trajectories"    — collect >= ``num_trajectories`` episodes, rank best-4 / median-4 / worst-4
                            by return, and write an annotated 3x4 grid mp4. The per-tile overlay is
                            chosen by ``overlay`` below. This is the one-and-only trajectory recording.
      * "reset_snapshots" — diagnostic: reset the env ``reset_snapshots_count`` times and render ONLY
                            each post-reset INITIAL state (no policy, no physics stepping), holding
                            ``reset_snapshots_hold_s`` s each. Tiles all ``num_envs``; for inspecting
                            spawn / initial conditions."""

    overlay: str = "none"
    """WHICH task-specific overlay to draw on each grid tile. Registry-backed (see
    ``learning/recording_eval.py::OVERLAY_RENDERERS``) so future tasks register their own by name:
      * "surface_tracking" — surface-follow overlay: in-scene keypoint-status balls, force + orientation
                             gauges, and the top-down path minimap. Requires ``env.viz_snapshot``.
      * "none"             — plain frames, no overlay; works for ANY env (forge / peg / humanoid / ...)."""

    reset_snapshots_count: int = 8
    """Number of resets to render for ``mode='reset_snapshots'`` (one held frame block per reset)."""

    reset_snapshots_hold_s: float = 1.0
    """Seconds to hold each reset's initial-state frame on screen (``mode='reset_snapshots'``)."""

    ball_diameter_frac: float = 1.6
    """Keypoint-ball diameter as a fraction of the keypoint spacing. The spec value 0.5 renders ~1.6
    mm balls that are near-invisible from the recorder camera; the default here is enlarged for
    legibility. Lower it toward 0.5 for a physically faithful size."""

    grid_select: str = "ranked"
    """How ``mode='trajectories'`` chooses which collected trajectories to tile:
      * "ranked" (default) — best-4 / median-4 / worst-4 by return in a 3x4 grid (the training-agent
                             view: meaningful only when the policy varies in quality across envs).
      * "all"             — tile ALL num_envs trajectories from a single episode in a square-ish grid
                             (ceil(sqrt(num_envs)) cols). Use for random/untrained rollouts where the
                             best/median/worst ranking is meaningless. Set num_trajectories == num_envs
                             so exactly one fresh-spawn episode fills the grid (e.g. 16 -> 4x4)."""

    grid_by_curvature: bool = False
    """CURVED surface only (surface_tracking overlay): lay the grid out as ``3 x n_curvature_levels``
    where each COLUMN is one curvature level (0 = flat … K-1 = most curved) and the rows are that
    level's BEST / MEDIAN / WORST trajectory by return. Requires the env to expose ``_env_level_idx``
    (the curved task does). Ignored on non-curved surfaces. For a well-filled grid, set ``num_envs``
    to a multiple of ``n_curvature_levels`` (e.g. 3-4 envs per level) so each column has enough
    trajectories to rank. Overrides ``grid_select`` when on."""

    # ---- DEPRECATED (backward-compat only; IGNORED by the recorder) ----------------------------------
    # Replaced by ``mode`` + ``overlay`` above. Kept ONLY so that already-snapshotted training
    # config.yaml files (which serialized the full old RecorderCfg) still load under the strict
    # ConfigManager — the standalone recorder never reads them. Do NOT set these in new configs; use
    # ``mode`` (trajectories | reset_snapshots) and ``overlay`` (surface_tracking | none) instead.
    annotated_ranked: bool = False
    stills_grid: bool = False
    stills_grid_video: bool = False
    stills_grid_png: bool = True
    grid_rows: int = 4
    grid_cols: int = 4
    reset_snapshots: bool = False
    surface_overlays: bool = True


@dataclasses.dataclass(kw_only=True)
class SAC_CFG(AgentCfg):
    """Configuration for the SAC agent."""

    tensorboard: bool = True
    """Write per-agent TensorBoard scalar events (under ``<experiment_dir>/<i>/``). Set
    ``false`` to log ONLY to wandb: the per-agent ``MetricWriter`` then skips the TB
    ``SummaryWriter`` and mirrors scalars to wandb alone. Requires ``experiment.wandb: true``
    — if both this and wandb are off, TensorBoard is kept on so logging isn't silently lost.
    (Sibling of ``experiment`` because skrl's ``ExperimentCfg`` has no such field.)"""

    gradient_steps: int = 1
    """Number of gradient steps to perform for each update."""

    batch_size: int = 64
    """Batch size **per agent** for sampling transitions from memory during training.
    With ``num_agents`` block-parallel agents, the total memory sample size is
    ``batch_size * num_agents`` (each agent draws ``batch_size`` transitions from
    its own env partition)."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    polyak: float = 0.005
    """Parameter to control the update of the target networks by polyak averaging.

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.torch.base.Model.update_parameters` for more details.
    """

    actor_lr: float = 1e-3
    """Learning rate for the actor (policy) optimizer."""

    critic_lr: float = 1e-3
    """Learning rate for the critic (Q-network) optimizer."""

    entropy_lr: float = 1e-3
    """Learning rate for the entropy coefficient (log_alpha) optimizer."""

    weight_decay: float = 0.0
    """Decoupled weight decay applied by AdamW to the actor and critic optimizers.
    The entropy optimizer is intentionally excluded — pulling log_alpha toward 0
    has no principled meaning. Set to 0.0 to disable (equivalent to Adam)."""

    lr_schedule: str = "constant"
    """Built-in actor+critic learning-rate schedule. ``"constant"`` (default) keeps the
    LRs fixed (today's behavior). ``"cosine"`` decays actor_lr/critic_lr from their initial
    value to :attr:`lr_end` on a cosine schedule over training (FlashSAC uses 3e-4->1.5e-4).
    T_max is derived automatically from the run length. Ignored if an explicit
    :attr:`learning_rate_scheduler` is provided (that takes precedence)."""

    lr_end: float = 1.5e-4
    """Final learning rate for ``lr_schedule="cosine"`` (CosineAnnealingLR ``eta_min``).
    Applies to the actor and critic optimizers; the entropy LR is left constant."""

    learning_rate_scheduler: type | tuple[type | None, type | None, type | None] | None = None
    """Learning rate scheduler class for the actor and critic networks, and entropy coefficient.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    observation_preprocessor: type | None = None
    """Preprocessor class to process the environment's observations.

    See :ref:`preprocessors` for more details.
    """

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    grad_norm_clip: float = 0
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    learn_entropy: bool = True
    """Whether to learn the entropy coefficient."""

    initial_entropy_value: float = 0.2
    """Initial value for the entropy coefficient."""

    target_entropy: float | None = None
    """Target value for computing the entropy loss."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function. YAML can't carry callables — set
    ``rewards_shaper_scale`` instead and the runner builds a multiplicative
    lambda. Direct assignment is still supported for programmatic configs."""

    rewards_shaper_scale: float | None = None
    """Scalar reward multiplier. When set, the runner installs
    ``rewards_shaper = lambda r, *a, **k: r * scale`` before constructing SAC.
    Mirrors skrl's runner convention; null disables shaping."""

    gripper_action_idx: int | None = None
    """Index of the binary gripper action dim, if any. When set, SAC logs
    ``Gripper / open rate`` (fraction of `actions[..., idx] >= 0`) plus mean and
    std of the raw value, so a stuck/locked gripper is visible in tensorboard.
    For Lift Franka, set to 7 (act_dim=8: arm 0-6, gripper 7). Diagnostic only;
    works regardless of whether the gripper dim is sampled from a Gaussian or
    a Bernoulli (see ``model_cfg.actor.bernoulli_action_dims``)."""

    phase_split_families: list[str] = dataclasses.field(default_factory=list)
    """Per-env metric families to split by insertion phase (free_space / search /
    insertion) in the per-agent TensorBoard logs. Each entry is a ``{family}`` — the
    text before the single ``/`` in a per-env ``to_log`` tag (e.g. ``"energy_metrics"``
    for ``energy_metrics/avg_force``). For every listed family, each of its tags is
    re-emitted as ``{family}_{phase}/{metric_name}`` (e.g. ``energy_metrics_search/
    avg_force``), reduced over only the envs in that phase each step (same
    max/min/mean/dist convention as the un-split tag), and the un-split tag is NOT
    logged. Phase per env: free_space = no contact and not engaged, search = in contact
    but not engaged, insertion = engaged (contact irrelevant). Requires the
    contact-sensor and Forge/Factory scorer wrappers (they publish the contact +
    engagement signals); listing a family without them raises. Empty list disables it."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    # ------------------------------------------------------------------
    # FlashSAC knobs (only honored when model_cfg.architecture == "flashsac";
    # all default to the vanilla-SAC behavior so existing configs are unaffected).
    # ------------------------------------------------------------------
    reward_scaling_enabled: bool = False
    """FlashSAC adaptive reward scaling (Eq. 6): divide sampled rewards by
    ``max(sqrt(running_return_var + eps), G_r_max / g_max)``. Off = no scaling."""
    reward_scaling_g_max: float = 5.0
    """``G_max`` in the Eq. 6 denominator. Must equal ``model_cfg.critic.v_max`` so the
    normalized returns land inside the categorical support (validated in the runner)."""
    reward_scaling_eps: float = 1e-8
    """Numerical epsilon under the variance sqrt in Eq. 6."""

    weight_norm_enabled: bool = False
    """FlashSAC weight normalization: after each optimizer step, project every linear's
    output rows onto the unit sphere and every norm param vector to sqrt(d)."""

    noise_repeat_enabled: bool = False
    """FlashSAC noise repetition: hold the policy's sampling noise for k consecutive
    rollout steps, k ~ truncated Zeta. Off = fresh reparameterized sample each step."""
    noise_repeat_zeta_mu: float = 2.0
    """Zeta exponent for the repeat-length distribution (pmf ~ k^-mu)."""
    noise_repeat_max: int = 16
    """Maximum repeat length (Zeta truncation)."""

    actor_update_period: int = 1
    """Update the actor (and entropy coefficient) once every N gradient steps
    (FlashSAC delays it, e.g. 2). 1 = every step (vanilla SAC)."""

    target_entropy_mode: str = "neg_action_dim"
    """How the automatic-entropy target is set (when ``learn_entropy`` and
    ``target_entropy`` is null). ``"neg_action_dim"`` (default) = ``-|A|`` (today's
    behavior). ``"unified"`` = FlashSAC's ``0.5*|A|*log(2*pi*e*sigma^2)`` using
    ``entropy_sigma_target``."""
    entropy_sigma_target: float = 0.15
    """Per-dimension action std defining the unified entropy target (FlashSAC)."""

    periodic_reset_enabled: bool = False
    """SimBa-style periodic parameter reset (primacy-bias / plasticity-loss mitigation). When True,
    the actor + critic (+ target critic) networks, their optimizers, and the entropy coefficient are
    reinitialized from scratch every :attr:`periodic_reset_frequency` env steps, up to
    :attr:`periodic_reset_max` times. The replay buffer and observation-normalization stats are KEPT
    (only the learners are reset), so learning re-fits fresh networks to the accumulated data —
    SimBa (arXiv:2410.09754, §7.3) resets "the entire network and optimizer" periodically."""

    periodic_reset_frequency: int = 0
    """Env steps between periodic resets (see :attr:`periodic_reset_enabled`). ``<= 0`` disables.
    SimBa uses 500k *gradient* steps on long runs; for a short run pick a value giving a few resets
    with room to recover after the last (e.g. 5000 over a 20k-step run => resets at 5k/10k/15k)."""

    periodic_reset_max: int = 0
    """Maximum number of periodic resets over a run. ``<= 0`` => unbounded (reset on every
    frequency boundary). Cap this so the last reset leaves enough steps to recover."""

    recorder: RecorderCfg = dataclasses.field(default_factory=RecorderCfg)
    """Configures the optional ``RecordingWrapper`` (3x4-grid GIFs + TB videos).
    Default is disabled. See :class:`RecorderCfg` and ``wrappers/recording.py``."""

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
