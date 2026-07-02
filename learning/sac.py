from __future__ import annotations

import itertools
from typing import Any

import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer

from configs.manager.preprocessor_registry import resolve_preprocessor
from configs.manager.sac_cfg import SAC_CFG
from learning.block_agent import BlockAgent
from learning.losses import AuxLossManager, LossContext
from models.block_simba import (
    merge_optimizer_states,
    slice_optimizer_state,
)
from models.preprocessor_wrapper import PerAgentPreprocessorWrapper


class SAC(BlockAgent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: SAC_CFG | dict = {},
        num_agents: int = 1,
        aux_losses: "AuxLossManager | None" = None,
        contact_axes: list[int] | None = None,
    ) -> None:
        """Soft Actor-Critic (SAC) with per-agent block-parallel independence.

        Each agent owns a fixed env partition (envs ``[i*epa, (i+1)*epa)``); each has
        its own learnable entropy coefficient and its own tensorboard writer. No
        metrics are aggregated across agents.

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
            For ``num_agents > 1`` this should be a ``MultiRandomMemory`` so that
            sampled mini-batches preserve the per-agent env partitioning.
        :param observation_space: Observation space.
        :param action_space: Action space.
        :param device: Data allocation and computation device.
        :param cfg: Agent's configuration.
        :param num_agents: Number of block-parallel agents.
        :param aux_losses: Optional manager of additional losses to add to the
            policy and/or critic loss each gradient step. ``None`` disables them
            (vanilla SAC). See ``learning/losses.py``.
        """
        self.cfg: SAC_CFG
        # Optional manager of additional losses (built by the runner from loss_cfg
        # and passed in). Kept as an opaque handle — SAC only calls has_target()/
        # compute() on it during update(); None means vanilla SAC. See
        # learning/losses.py.
        self._aux_losses = aux_losses
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=SAC_CFG(**cfg) if isinstance(cfg, dict) else cfg,
            num_agents=num_agents,
            contact_axes=contact_axes,
        )

        # Asymmetric actor-critic: when ``state_space`` is provided, the critic
        # consumes the (typically larger) state vector while the actor still uses
        # the policy observation. Memory then stores both ``observations`` and
        # ``states``. When ``state_space is None``, the critic uses observations
        # (symmetric — backward-compatible).
        self.state_space = state_space
        self._asymmetric: bool = state_space is not None

        # models — all five required, no silent fallback to None
        required = ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
        missing = [k for k in required if k not in self.models or self.models[k] is None]
        if missing:
            raise ValueError(f"SAC requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]

        # SimBa periodic reset (sac_cfg.periodic_reset_*): the runner attaches a model factory
        # (agent._model_factory) that rebuilds fresh networks; _periodic_reset() swaps them in.
        self._model_factory = None
        self._n_periodic_resets = 0

        # checkpointing is handled per-agent by write_checkpoint()/load() — we don't
        # populate self.checkpoint_modules so the base Agent's bundled save path stays out.

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # entropy — per-agent (N, 1) coefficient. Adam state is element-wise so a single
        # optimizer over the (N, 1) parameter is fully independent across agents.
        self._entropy_coefficient = torch.full(
            (num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            # target_entropy is action-space dependent; same scalar across agents.
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(self._entropy_coefficient.clone()).requires_grad_(True)
            # Entropy gets AdamW with weight_decay=0 — pulling log_alpha toward 0 has no
            # principled meaning, so the user-configured weight_decay is intentionally
            # NOT applied here.
            self.entropy_optimizer = torch.optim.AdamW(
                [self.log_entropy_coefficient],
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

        # set up optimizers and learning rate schedulers (AdamW with decoupled weight decay)
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.cfg.actor_lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.critic_optimizer = torch.optim.AdamW(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self.cfg.critic_lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic_scheduler is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        # set up target networks
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # set up observation preprocessor.
        # `cfg.observation_preprocessor` may be a class, a registered string name (from
        # YAML), or None. Resolve to a class first; then build N independent instances
        # and wrap them so per-agent batch slices route to per-agent preprocessors.
        preproc_cls = resolve_preprocessor(self.cfg.observation_preprocessor)
        if preproc_cls is not None:
            preproc_list = [
                preproc_cls(**self.cfg.observation_preprocessor_kwargs)
                for _ in range(num_agents)
            ]
            self._observation_preprocessor = PerAgentPreprocessorWrapper(num_agents, preproc_list)
        else:
            self._observation_preprocessor = self._empty_preprocessor

        # State preprocessor (asymmetric setup only). Same class as obs preprocessor,
        # sized to state_space. The runner injects size+device into preprocessor_kwargs;
        # for state we re-use the same kwargs but override `size` to state_space.
        if self._asymmetric and preproc_cls is not None:
            state_kwargs = dict(self.cfg.observation_preprocessor_kwargs)
            state_kwargs["size"] = state_space
            state_preproc_list = [preproc_cls(**state_kwargs) for _ in range(num_agents)]
            self._state_preprocessor = PerAgentPreprocessorWrapper(num_agents, state_preproc_list)
        else:
            self._state_preprocessor = self._empty_preprocessor

    # --------------------------------------------------------------
    # Memory layout (BlockAgent.init hook)
    # --------------------------------------------------------------
    def _create_memory_tensors(self) -> None:
        """Replay-buffer tensors. Symmetric: obs only. Asymmetric: obs (actor) + states (critic)."""
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
            ]

            if self._asymmetric:
                self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
                self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
                self._tensors_names.extend(["states", "next_states"])

            # Optional ground-truth contact tensor for the supervised-selection loss.
            self._maybe_create_contact_tensor()

    # --------------------------------------------------------------
    # Interaction
    # --------------------------------------------------------------
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions from the policy. ``states`` is accepted for trainer compatibility but ignored."""
        inputs = {"observations": self._observation_preprocessor(observations)}
        if self.training and timestep < self.cfg.random_timesteps:
            # Uniform on [-1, 1] — matches the tanh-squashed policy's support.
            # Skips skrl's default random_act which calls Box.sample() on the env's
            # action_space; Isaac Lab advertises Box(-inf, +inf), so that fallback
            # samples N(0, 1) and pumps the replay buffer with actions outside the
            # policy's reachable range, producing a misleading reward cliff at the
            # random→policy hand-off.
            n = observations.shape[0]
            actions = torch.rand(n, *self.action_space.shape, device=self.device) * 2.0 - 1.0
            return actions, {}
        # no_grad: these actions are for environment interaction only — the update
        # re-runs the policy forward on replay-buffer minibatches for the gradient.
        # Without this, the returned actions carry a live autograd graph that the
        # controller wrappers' EMA buffers (e.g. hybrid_force_position_wrapper's
        # self.ema_actions / the base env's self.actions) splice in-place across
        # steps, growing an unbounded graph that OOMs the GPU after a few thousand steps.
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Per-agent reward/episode bookkeeping + memory write.

        Skips ``super().record_transition`` because the base implementation accumulates
        a single global reward stream; we publish per-agent rewards instead.
        """
        # Per-agent env-metric ingestion (reward decomposition, distance/velocity,
        # success rate/times, Forge early-term prediction, infos["log"] mirror).
        # Uses RAW (pre-shaping) rewards — matches the ordering relative to the
        # reward_shaper applied below for the memory write.
        self._ingest_step_metrics(
            rewards=rewards, terminated=terminated, truncated=truncated, infos=infos
        )

        if self.training:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            extra_kwargs: dict[str, torch.Tensor] = {}
            if self._asymmetric:
                # Strict: in asymmetric mode the trainer MUST pass per-step states.
                if states is None or next_states is None:
                    raise RuntimeError(
                        "asymmetric SAC requires states and next_states from the "
                        "trainer (env.state() must return non-None). Got "
                        f"states={states is not None}, next_states={next_states is not None}."
                    )
                extra_kwargs["states"] = states
                extra_kwargs["next_states"] = next_states
            # One-step-aligned contact ground truth for the SSL (no-op unless contact_axes).
            self._buffer_contact_for_write(
                terminated=terminated, truncated=truncated, infos=infos, add_kwargs=extra_kwargs
            )
            self.memory.add_samples(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=terminated,
                **extra_kwargs,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            if timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    # algorithm wall-clock duplicated to every per-agent log
                    self.track_per_agent(
                        "Stats / Algorithm update time (ms)",
                        [timer.elapsed_time_ms] * self.num_agents,
                    )
            self._maybe_periodic_reset(timestep)

        # base.post_interaction handles checkpointing + calls write_tracking_data on interval
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    # --------------------------------------------------------------
    # SimBa periodic reset (plasticity-loss / primacy-bias mitigation)
    # --------------------------------------------------------------
    def _maybe_periodic_reset(self, timestep: int) -> None:
        """Trigger a hard network reset on the ``periodic_reset_frequency`` env-step boundary,
        up to ``periodic_reset_max`` times. See :class:`SAC_CFG` ``periodic_reset_*``."""
        freq = self.cfg.periodic_reset_frequency
        if not self.cfg.periodic_reset_enabled or freq <= 0 or timestep <= 0:
            return
        mx = self.cfg.periodic_reset_max
        if mx > 0 and self._n_periodic_resets >= mx:
            return
        if timestep % freq == 0:
            self._periodic_reset()
            self._n_periodic_resets += 1
            print(f"[sac] periodic reset #{self._n_periodic_resets} at timestep {timestep}", flush=True)
        # log the running reset count to every per-agent stream (0 until the first reset)
        self.track_per_agent(
            "Stats / Periodic resets", [float(self._n_periodic_resets)] * self.num_agents
        )

    def _periodic_reset(self) -> None:
        """SimBa-style hard reset: rebuild actor + twin critics (+ targets), their optimizers, and
        the entropy coefficient from scratch — KEEPING the replay buffer and obs-normalization
        stats. Mirrors SimBa (arXiv:2410.09754 §7.3): reinitialize the entire network + optimizer."""
        if self._model_factory is None:
            raise RuntimeError(
                "periodic_reset_enabled but no _model_factory attached to the agent "
                "(runner must set agent._model_factory)."
            )
        fresh = self._model_factory()
        for k in ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            self.models[k] = fresh[k]
        self.policy = self.models["policy"]
        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]

        # targets start equal to the fresh critics (hard copy), frozen — mirrors __init__.
        self.target_critic_1.freeze_parameters(True)
        self.target_critic_2.freeze_parameters(True)
        self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # rebuild optimizers (fresh Adam moment state) + schedulers, exactly as in __init__.
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.actor_lr, weight_decay=self.cfg.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
        )
        if self.cfg.learning_rate_scheduler[0] is not None:
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
            )
        if self.cfg.learning_rate_scheduler[1] is not None:
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
            )

        # reset entropy coefficient + its optimizer (part of "the entire network and optimizer").
        self._entropy_coefficient = torch.full(
            (self.num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            self.log_entropy_coefficient = torch.log(
                self._entropy_coefficient.clone()
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.AdamW(
                [self.log_entropy_coefficient], lr=self.cfg.entropy_lr, weight_decay=0.0
            )

    # --------------------------------------------------------------
    # Update
    # --------------------------------------------------------------
    def update(self, *, timestep: int, timesteps: int) -> None:
        N = self.num_agents
        # cfg.batch_size is PER AGENT. The memory's sample() interprets the
        # batch_size argument as per-agent and internally returns N * batch_size
        # rows partitioned [agent0 | agent1 | ...].
        B = self.cfg.batch_size

        for gradient_step in range(self.cfg.gradient_steps):
            sampled_list = self.memory.sample(
                names=self._tensors_names, batch_size=B
            )[0]
            sampled = dict(zip(self._tensors_names, sampled_list))
            sampled_observations = sampled["observations"]
            sampled_actions = sampled["actions"]
            sampled_rewards = sampled["rewards"]
            sampled_next_observations = sampled["next_observations"]
            sampled_terminated = sampled["terminated"]

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                }
                # In asymmetric mode, the critic consumes states (not obs). Build
                # separate input dicts for the critic networks; the actor still
                # uses the policy obs above.
                if self._asymmetric:
                    critic_inputs = {
                        "observations": self._state_preprocessor(sampled["states"], train=True),
                    }
                    critic_next_inputs = {
                        "observations": self._state_preprocessor(sampled["next_states"], train=True),
                    }
                else:
                    critic_inputs = inputs
                    critic_next_inputs = next_inputs

                with torch.no_grad():
                    next_actions, outputs = self.policy.act(next_inputs, role="policy")
                    next_log_prob = outputs["log_prob"]

                    target_q1_values, _ = self.target_critic_1.act(
                        {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _ = self.target_critic_2.act(
                        {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # (N*B, 1)
                    target_q_values = torch.min(target_q1_values, target_q2_values) - ent_flat * next_log_prob
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor * sampled_terminated.logical_not() * target_q_values
                    )

                critic_1_values, _ = self.critic_1.act({**critic_inputs, "taken_actions": sampled_actions}, role="critic_1")
                critic_2_values, _ = self.critic_2.act({**critic_inputs, "taken_actions": sampled_actions}, role="critic_2")

                critic_loss = (
                    F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
                ) / 2

                # ---- additional (auxiliary) critic-target losses ----
                # Optional extra losses configured via loss_cfg and built into the
                # AuxLossManager passed at construction. We fold them into
                # critic_loss *here*, inside the autocast block and before the
                # critic backward/step below, so their gradients reach the critic
                # params on the same step. `aux_critic_raw` maps loss name -> the
                # detached per-agent (N,) raw value (for tensorboard); it stays {}
                # when no aux losses or none target the critic, so the logging loop
                # later is a no-op. The has_target guard skips building a
                # LossContext entirely in the common (no-aux) case.
                aux_critic_raw: dict[str, torch.Tensor] = {}
                if self._aux_losses is not None and self._aux_losses.has_target("critic"):
                    # Hand the loss the critic-block tensors it may need. aux_total
                    # is the weighted scalar sum (weight * raw.mean()) to add.
                    aux_total, aux_critic_raw = self._aux_losses.compute(
                        LossContext(
                            agent=self,
                            target="critic",
                            sampled=sampled,
                            critic_1_values=critic_1_values,
                            critic_2_values=critic_2_values,
                            target_values=target_values,
                            critic_inputs=critic_inputs,
                        ),
                        "critic",
                    )
                    critic_loss = critic_loss + aux_total

            # critic step
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                # Critic Q for the policy gradient: actor uses obs, critic uses state.
                critic_1_pi, _ = self.critic_1.act({**critic_inputs, "taken_actions": actions}, role="critic_1")
                critic_2_pi, _ = self.critic_2.act({**critic_inputs, "taken_actions": actions}, role="critic_2")

                ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # detached, no grad
                policy_loss = (ent_flat * log_prob - torch.min(critic_1_pi, critic_2_pi)).mean()

                # ---- additional (auxiliary) policy-target losses ----
                # Mirror of the critic-side hook above, on the policy side: fold
                # any policy-target aux losses into policy_loss inside the autocast
                # block, before the policy backward/step below, so their gradients
                # reach the actor on the same step. The freshly re-sampled `actions`
                # / `log_prob` (grad-carrying) are passed in the context — e.g. the
                # built-in ActionL2Loss differentiates through `actions`.
                # `aux_policy_raw` is {} unless a loss targets the policy.
                aux_policy_raw: dict[str, torch.Tensor] = {}
                if self._aux_losses is not None and self._aux_losses.has_target("policy"):
                    aux_total, aux_policy_raw = self._aux_losses.compute(
                        LossContext(
                            agent=self,
                            target="policy",
                            sampled=sampled,
                            actions=actions,
                            log_prob=log_prob,
                            policy_outputs=outputs,
                            inputs=inputs,
                        ),
                        "policy",
                    )
                    policy_loss = policy_loss + aux_total

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            # Unscale unconditionally so the per-agent grad-norm slices we
            # capture below reflect true (un-amped) magnitudes. If mixed
            # precision is off the scaler is a no-op; if on, scaler.step()
            # detects the prior unscale and skips a redundant pass.
            self.scaler.unscale_(self.policy_optimizer)
            if self.write_interval > 0:
                self._last_action_head_grad_norm = self._compute_actor_head_grad_norm()
            if self.cfg.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

            # per-agent entropy step
            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    log_prob_per_agent = log_prob.view(N, B, 1).mean(dim=1)  # (N, 1)
                    entropy_loss_per_agent = -(
                        self.log_entropy_coefficient
                        * (log_prob_per_agent + self._target_entropy).detach()
                    )  # (N, 1)
                    entropy_loss = entropy_loss_per_agent.sum()

                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())  # (N, 1)

            self.scaler.update()

            # target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # per-agent metric tracking
            if self.write_interval > 0:
                def split(t):  # (N*B, *) -> (N, B, -1)
                    return t.view(N, B, -1)

                policy_terms = (ent_flat * log_prob - torch.min(critic_1_pi, critic_2_pi))
                self.track_per_agent("Loss / Policy loss",
                                     split(policy_terms).mean(dim=(1, 2)))
                critic_loss_per_agent = 0.5 * (
                    F.mse_loss(split(critic_1_values), split(target_values), reduction="none").mean(dim=(1, 2))
                    + F.mse_loss(split(critic_2_values), split(target_values), reduction="none").mean(dim=(1, 2))
                )
                self.track_per_agent("Loss / Critic loss", critic_loss_per_agent)

                # Auxiliary losses — log each enabled loss's per-agent, unweighted
                # (raw) value, captured above when the loss was folded into the
                # policy/critic loss. `value` is the (N,) tensor track_per_agent
                # expects (element i -> agent i's writer). The "_actor"/"_critic"
                # suffix records which optimizer the loss fed. Tag names use a
                # SINGLE "/" only: tensorboard treats "/" as a section separator,
                # so a second slash would nest the metric (see project memory).
                # Both dicts are {} when no aux loss targets that side, so these
                # loops are no-ops in the vanilla case.
                for term_name, value in aux_policy_raw.items():
                    self.track_per_agent(f"Loss/ {term_name}_raw_actor", value)
                for term_name, value in aux_critic_raw.items():
                    self.track_per_agent(f"Loss/ {term_name}_raw_critic", value)

                self.track_per_agent("Q-network / Q1 (max)",  split(critic_1_values).amax(dim=(1, 2)))
                self.track_per_agent("Q-network / Q1 (min)",  split(critic_1_values).amin(dim=(1, 2)))
                self.track_per_agent("Q-network / Q1 (mean)", split(critic_1_values).mean(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (max)",  split(critic_2_values).amax(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (min)",  split(critic_2_values).amin(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (mean)", split(critic_2_values).mean(dim=(1, 2)))

                self.track_per_agent("Target / Target (max)",  split(target_values).amax(dim=(1, 2)))
                self.track_per_agent("Target / Target (min)",  split(target_values).amin(dim=(1, 2)))
                self.track_per_agent("Target / Target (mean)", split(target_values).mean(dim=(1, 2)))

                # Action diagnostics — surface tanh saturation and log_prob collapse.
                with torch.no_grad():
                    abs_a = actions.abs()
                    saturation = (abs_a > 0.99).float()
                self.track_per_agent("Action / |a| max",       split(abs_a).amax(dim=(1, 2)))
                self.track_per_agent("Action / |a| mean",      split(abs_a).mean(dim=(1, 2)))
                self.track_per_agent("Action / saturation rate", split(saturation).mean(dim=(1, 2)))
                self.track_per_agent("Action / log_prob (mean)", split(log_prob).mean(dim=(1, 2)))

                # Pose-action magnitude — the first 6 action dims are the raw,
                # pre-scaled pose command (x, y, z, rx, ry, rz) the policy emits
                # before the env rescales to physical units. Tracks the average
                # magnitude and its spread per axis so we can see how hard the
                # policy drives each pose channel independent of gain/gripper dims.
                with torch.no_grad():
                    pose_abs_pa = split(abs_a[..., :6])        # (N, B, 6)
                for i, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
                    self.track_per_agent(f"Action / pose |{axis}| mean", pose_abs_pa[..., i].mean(dim=1))
                    self.track_per_agent(f"Action / pose |{axis}| std",  pose_abs_pa[..., i].std(dim=1))

                # Continuous-action L2 norm — surfaces "do nothing" collapse.
                # If the policy parks all continuous dims near 0 (e.g. when the
                # entropy term dominates and the critic gradient is tiny), L2
                # norm trends toward 0. Excluding Bernoulli dims keeps {-1,+1}
                # gripper outputs from inflating the norm artificially.
                cont_idx = getattr(self.policy, "_cont_action_idx", None)
                if cont_idx is not None and cont_idx.numel() > 0:
                    with torch.no_grad():
                        cont_actions = actions.index_select(-1, cont_idx)   # (N*B, num_cont)
                        cont_l2 = cont_actions.norm(dim=-1)                 # (N*B,)
                        cont_l2_per_agent = cont_l2.view(N, -1)             # (N, B)
                    self.track_per_agent("Action / continuous L2 (max)",  cont_l2_per_agent.amax(dim=1))
                    self.track_per_agent("Action / continuous L2 (min)",  cont_l2_per_agent.amin(dim=1))
                    self.track_per_agent("Action / continuous L2 (mean)", cont_l2_per_agent.mean(dim=1))
                    self.track_per_agent("Action / continuous L2 (std)",  cont_l2_per_agent.std(dim=1))

                # Gripper diagnostic — open rate is the headline metric. If it's
                # stuck near 0 or 1 the gripper is locked and the agent can't grasp.
                gidx = self.cfg.gripper_action_idx
                if gidx is not None:
                    with torch.no_grad():
                        g = actions[..., gidx].unsqueeze(-1)         # (N*B, 1)
                        g_open = (g >= 0).float()
                    self.track_per_agent("Gripper / open rate",   split(g_open).mean(dim=(1, 2)))
                    self.track_per_agent("Gripper / action mean", split(g).mean(dim=(1, 2)))
                    self.track_per_agent("Gripper / action std",  split(g).flatten(1).std(dim=1))

                if self.cfg.learn_entropy:
                    self.track_per_agent("Loss / Entropy loss", entropy_loss_per_agent.squeeze(-1))
                    self.track_per_agent("Coefficient / Entropy coefficient",
                                         self._entropy_coefficient.squeeze(-1))

                # Action-head grad norm: an actor health metric under the
                # actor diagnostics tab (Action / *). Tracked whenever the
                # grad-norm slice was captured this update.
                if self._last_action_head_grad_norm is not None:
                    self.track_per_agent(
                        "Action / action head grad norm",
                        self._last_action_head_grad_norm,
                    )

                if self.policy_scheduler:
                    lr = self.policy_scheduler.get_last_lr()[0]
                    self.track_per_agent("Learning / Policy learning rate", [lr] * N)
                if self.critic_scheduler:
                    lr = self.critic_scheduler.get_last_lr()[0]
                    self.track_per_agent("Learning / Critic learning rate", [lr] * N)

    # --------------------------------------------------------------
    # Per-agent checkpoint hooks (BlockAgent does the slicing/stitching)
    # --------------------------------------------------------------
    def _checkpoint_model_keys(self) -> list[str]:
        return ["policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]

    def _checkpoint_optimizer_keys(self) -> list[str]:
        return ["policy_optimizer", "critic_optimizer"]

    def _required_checkpoint_keys(self) -> set[str]:
        return super()._required_checkpoint_keys() | {
            "entropy_coefficient",
            "log_entropy_coefficient",
            "entropy_optimizer",
        }

    def _build_extra_checkpoint(self, i: int) -> dict:
        """Per-agent entropy coefficient + (conditional) entropy optimizer."""
        return {
            "entropy_coefficient": self._entropy_coefficient[i].detach().clone().cpu(),
            "log_entropy_coefficient": (
                self.log_entropy_coefficient.detach()[i].clone().cpu()
                if self.cfg.learn_entropy else None
            ),
            "entropy_optimizer": (
                slice_optimizer_state(self.entropy_optimizer.state_dict(), i, self.num_agents)
                if self.cfg.learn_entropy else None
            ),
        }

    def _load_extra_into_slot(self, target_slot: int, ckpt: dict, path: str) -> None:
        """Load the per-slot entropy coefficient; validate learn_entropy agreement."""
        with torch.no_grad():
            self._entropy_coefficient[target_slot].copy_(
                ckpt["entropy_coefficient"].to(self.device)
            )
            # cfg.learn_entropy and saved log_entropy_coefficient must agree.
            saved_log_ent = ckpt["log_entropy_coefficient"]
            if self.cfg.learn_entropy and saved_log_ent is None:
                raise ValueError(
                    f"cfg.learn_entropy=True but checkpoint at {path} has "
                    f"log_entropy_coefficient=None (saved with learn_entropy=False)."
                )
            if not self.cfg.learn_entropy and saved_log_ent is not None:
                raise ValueError(
                    f"cfg.learn_entropy=False but checkpoint at {path} contains a "
                    f"log_entropy_coefficient (saved with learn_entropy=True)."
                )
            if self.cfg.learn_entropy:
                self.log_entropy_coefficient.data[target_slot].copy_(
                    saved_log_ent.to(self.device)
                )

    def _load_extra_optimizers(self, per_agent_ckpts: list[dict], path: str) -> None:
        """Stitch the (conditional) entropy optimizer; validate learn_entropy agreement.

        Works for both single-agent (``len == 1``) and multi-agent loads.
        """
        ent_opt_present = [c["entropy_optimizer"] is not None for c in per_agent_ckpts]
        if any(ent_opt_present) != all(ent_opt_present):
            raise ValueError(
                f"Inconsistent entropy_optimizer presence across per-agent checkpoints: "
                f"{ent_opt_present}. All agents must have been saved with the same "
                f"learn_entropy setting."
            )
        all_have = all(ent_opt_present)
        if self.cfg.learn_entropy and not all_have:
            raise ValueError(
                f"cfg.learn_entropy=True but checkpoint(s) under {path} have "
                f"entropy_optimizer=None (saved with learn_entropy=False)."
            )
        if not self.cfg.learn_entropy and all_have:
            raise ValueError(
                f"cfg.learn_entropy=False but checkpoint(s) under {path} contain "
                f"entropy_optimizer state (saved with learn_entropy=True)."
            )
        if self.cfg.learn_entropy:
            self.entropy_optimizer.load_state_dict(
                merge_optimizer_states(
                    [c["entropy_optimizer"] for c in per_agent_ckpts], len(per_agent_ckpts)
                )
            )
