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

    stills_grid: bool = False
    """Surface-task only. When True, ``learning/record.py`` runs ONE rollout and writes a single
    ``grid_rows x grid_cols`` PNG montage of annotated still frames (one env per tile) instead of the
    best/median/worst grid video. Each tile carries the surface overlays (keypoint balls in-scene,
    a force gauge, an orientation gauge, and a matplotlib top-down path inset). num_envs should be
    >= grid_rows*grid_cols so every tile is a distinct env."""

    grid_rows: int = 4
    grid_cols: int = 4
    """Montage layout for ``stills_grid`` (rows x cols tiles)."""

    stills_grid_video: bool = False
    """When ``stills_grid`` is on, ALSO write a full mp4 of the same rollout (per-frame gauges +
    a growing top-down path; the in-scene keypoint balls animate in the captured frames)."""

    surface_overlays: bool = True
    """When ``stills_grid`` is on, draw the surface-follow overlays (force/orientation gauges +
    top-down path inset) on each tile. The in-scene keypoint balls are added separately by the
    recorder regardless. Turn off to get plain frames."""

    ball_diameter_frac: float = 1.6
    """Keypoint-ball diameter as a fraction of the keypoint spacing. The spec value 0.5 renders ~1.6
    mm balls that are near-invisible from the recorder camera; the default here is enlarged for
    legibility. Lower it toward 0.5 for a physically faithful size."""


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
