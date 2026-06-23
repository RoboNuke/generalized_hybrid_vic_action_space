from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg

from configs.manager.sac_cfg import RecorderCfg


@dataclasses.dataclass(kw_only=True)
class PPO_CFG(AgentCfg):
    """Configuration for the (block-parallel, multi-agent) PPO agent.

    Follows the flat-field style of :class:`~configs.manager.sac_cfg.SAC_CFG`: the
    policy and value networks each get their own learning rate (dual AdamW, decoupled
    weight decay), rather than skrl's single ``learning_rate`` tuple.
    """

    # ---- Core PPO loop ----
    rollouts: int = 16
    """Collection steps between updates. Drives the on-policy memory depth: the
    rollout buffer holds ``rollouts`` transitions per env (``rollouts * total_envs``
    total)."""

    learning_epochs: int = 8
    """Number of passes over the collected rollout per update."""

    mini_batches: int = 2
    """Number of mini-batches the rollout is split into each epoch. The runner
    asserts ``(rollouts * total_envs) % mini_batches == 0`` so ``sample_all``
    partitions evenly (per-agent ordering preserved)."""

    discount_factor: float = 0.99
    """Reward discount γ. Range ``[0.0, 1.0]``."""

    gae_lambda: float = 0.95
    """TD(λ) coefficient for Generalized Advantage Estimation."""

    # ---- Optimizers (dual AdamW; one combined backward, two steps) ----
    policy_lr: float = 3e-4
    """Learning rate for the policy (actor) AdamW optimizer."""

    value_lr: float = 3e-4
    """Learning rate for the value (critic) AdamW optimizer."""

    weight_decay: float = 0.0
    """Decoupled weight decay applied by AdamW to both optimizers (0.0 == Adam)."""

    learning_rate_scheduler: type | tuple[type | None, type | None] | None = None
    """LR scheduler class for the policy and value optimizers.

    * A single class is applied to both.
    * A 2-tuple ``(policy, value)`` assigns one each.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the LR scheduler constructor(s). ``optimizer`` is
    injected automatically and must not be provided. A 2-tuple assigns per-network."""

    # ---- Preprocessors ----
    observation_preprocessor: type | None = None
    """Preprocessor class for the environment's observations (e.g. RunningStandardScaler)."""

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor."""

    value_preprocessor: type | None = None
    """Preprocessor class for the value network's output (typically
    RunningStandardScaler with size 1). Applied as a SINGLE shared instance (not
    per-agent) to avoid the env-axis/time-axis split on ``(T, total_envs, 1)``
    return/value tensors."""

    value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the value preprocessor's constructor."""

    # ---- Exploration / warmup ----
    random_timesteps: int = 0
    """Random-action steps before sampling from the policy. Usually 0 for PPO."""

    learning_starts: int = 0
    """Number of env steps before the first update is performed."""

    # ---- PPO objective ----
    grad_norm_clip: float = 0.5
    """Global-norm gradient clip. ``<= 0`` disables."""

    ratio_clip: float = 0.2
    """Clipping coefficient for the clipped surrogate objective."""

    value_clip: float = 0.2
    """Clipping coefficient for the predicted value in the value loss. ``<= 0`` disables."""

    entropy_loss_scale: float = 0.0
    """Entropy-bonus scaling factor (uses the actor's ``get_entropy``)."""

    value_loss_scale: float = 1.0
    """Value-loss scaling factor."""

    kl_threshold: float = 0.0
    """Approx-KL early-stop threshold; the worst agent's KL triggers the break.
    ``0.0`` disables early stopping."""

    time_limit_bootstrap: bool = False
    """Bootstrap the value at truncation (episode timeout) so time-limit cutoffs
    aren't treated as terminal."""

    # ---- Diagnostics ----
    gripper_action_idx: int | None = None
    """Index of a binary gripper action dim, if any. Parity with SAC's Gripper/ tab."""

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

    # ---- Reward shaping (YAML can't carry callables) ----
    rewards_shaper: Callable | None = None
    """Reward shaping function. Set ``rewards_shaper_scale`` instead from YAML; the
    runner installs a multiplicative lambda."""

    rewards_shaper_scale: float | None = None
    """Scalar reward multiplier; when set the runner builds
    ``rewards_shaper = lambda r, *a, **k: r * scale``. ``None`` disables shaping."""

    # ---- Performance / recording ----
    mixed_precision: bool = False
    """Enable automatic mixed precision."""

    recorder: RecorderCfg = dataclasses.field(default_factory=RecorderCfg)
    """Optional rollout recorder (shared with SAC). NOTE: the RecordingWrapper
    overlays Q-values from SAC's twin critics, which PPO lacks; the runner raises
    if ``recorder.enabled`` under PPO until the recorder is generalized."""

    def expand(self) -> None:
        """Expand scalar scheduler/kwargs into per-network 2-tuples (policy, value)."""
        super().expand()
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
