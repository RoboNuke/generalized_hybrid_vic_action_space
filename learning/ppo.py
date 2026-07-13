"""Block-parallel multi-agent PPO, sharing all logging/checkpoint machinery with SAC.

Ported from skrl's PPO and the block-parallel BlockPPO in
RoboNuke/Continuous_Force_RL, but built on :class:`~learning.block_agent.BlockAgent`
so it reproduces SAC's per-agent tensorboard output (only the algorithm-specific
metrics differ). The hybrid action-distribution styles (``product`` / ``match``)
live in the actor model, so this update loop is distribution-agnostic — it just
consumes ``policy.act(...)["log_prob"]`` and ``policy.get_entropy()``.

Optimizers: dual AdamW (policy + value), one combined backward
(``total_loss.backward()`` populates both networks' grads in a single graph
traversal), then both optimizers step over disjoint parameter sets.
"""

from __future__ import annotations

import collections
from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent  # noqa: F401  (kept for type parity / subclass docs)
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from configs.manager.preprocessor_registry import resolve_preprocessor
from configs.manager.ppo_cfg import PPO_CFG
from learning.block_agent import BlockAgent
from learning.losses import AuxLossManager, LossContext
from models.preprocessor_wrapper import PerAgentPreprocessorWrapper


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
    num_agents: int = 1,
    envs_per_agent: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with PER-AGENT advantage normalization.

    Tensors are shaped ``(rollout, total_envs, 1)`` with the env axis carrying the
    block layout ``[agent0 envs, agent1 envs, ...]``. GAE is computed per-env over
    the time axis; advantages are then normalized INDEPENDENTLY per agent block
    (a deliberate divergence from stock PPO's global normalization, matching the
    block-parallel independence the rest of the codebase assumes).
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i]
            + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage

    returns = advantages + values

    # Per-agent normalization: reshape the env axis into (num_agents, envs_per_agent)
    # and standardize over time + within-agent envs, keeping each agent independent.
    T = advantages.shape[0]
    adv = advantages.view(T, num_agents, envs_per_agent, -1)
    mean = adv.mean(dim=(0, 2), keepdim=True)
    std = adv.std(dim=(0, 2), keepdim=True)
    advantages = ((adv - mean) / (std + 1e-8)).view(T, num_agents * envs_per_agent, -1)

    return returns, advantages


class PPO(BlockAgent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: PPO_CFG | dict = {},
        num_agents: int = 1,
        aux_losses: "AuxLossManager | None" = None,
        contact_axes: list[int] | None = None,
        rot6d_slice: tuple[int, int] | None = None,
    ) -> None:
        """Proximal Policy Optimization (PPO) with per-agent block-parallel independence.

        :param models: ``{"policy": <actor>, "value": <BlockSimBaValueCritic>}``.
        :param memory: On-policy rollout memory (``MultiRandomMemory``); ``memory_size``
            is the per-env rollout length (``cfg.rollouts``).
        :param state_space: When provided, the value critic consumes the state vector
            (asymmetric); memory then also stores ``states``.
        :param cfg: PPO configuration.
        :param num_agents: Number of block-parallel agents.
        :param aux_losses: Optional auxiliary-loss manager (same interface as SAC).
        """
        self.cfg: PPO_CFG
        self._aux_losses = aux_losses
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=PPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
            num_agents=num_agents,
            contact_axes=contact_axes,
            rot6d_slice=rot6d_slice,
        )

        self.state_space = state_space
        self._asymmetric: bool = state_space is not None

        # models — both required, no silent fallback
        required = ("policy", "value")
        missing = [k for k in required if k not in self.models or self.models[k] is None]
        if missing:
            raise ValueError(f"PPO requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self.value = self.models["value"]

        # checkpointing is handled per-agent by BlockAgent — we don't populate
        # self.checkpoint_modules so the base Agent's bundled save path stays out.

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            self.policy.broadcast_parameters()
            if self.value is not self.policy:
                self.value.broadcast_parameters()

        # automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # dual AdamW optimizers (one combined backward, two disjoint steps)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.policy_lr, weight_decay=self.cfg.weight_decay
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=self.cfg.value_lr, weight_decay=self.cfg.weight_decay
        )
        self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
        self.value_scheduler = self.cfg.learning_rate_scheduler[1]
        if self.policy_scheduler is not None:
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
            )
        if self.value_scheduler is not None:
            self.value_scheduler = self.cfg.learning_rate_scheduler[1](
                self.value_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
            )

        # observation preprocessor — N independent instances wrapped per-agent
        # (same pattern as SAC, so the per-agent checkpoint slicing applies).
        preproc_cls = resolve_preprocessor(self.cfg.observation_preprocessor)
        if preproc_cls is not None:
            preproc_list = [
                preproc_cls(**self.cfg.observation_preprocessor_kwargs) for _ in range(num_agents)
            ]
            self._observation_preprocessor = PerAgentPreprocessorWrapper(num_agents, preproc_list)
        else:
            self._observation_preprocessor = self._empty_preprocessor

        # state preprocessor (asymmetric only) — critic input normalizer, per-agent
        if self._asymmetric and preproc_cls is not None:
            state_kwargs = dict(self.cfg.observation_preprocessor_kwargs)
            state_kwargs["size"] = state_space
            state_preproc_list = [preproc_cls(**state_kwargs) for _ in range(num_agents)]
            self._state_preprocessor = PerAgentPreprocessorWrapper(num_agents, state_preproc_list)
        else:
            self._state_preprocessor = self._empty_preprocessor

        # value preprocessor — a SINGLE shared instance (size 1). Per-agent wrapping
        # would split the wrong axis on the (T, total_envs, 1) value/return tensors.
        value_preproc_cls = resolve_preprocessor(self.cfg.value_preprocessor)
        if value_preproc_cls is not None:
            self._value_preprocessor = value_preproc_cls(**self.cfg.value_preprocessor_kwargs)
        else:
            self._value_preprocessor = self._empty_preprocessor

        # rollout temporaries
        self._current_log_prob = None
        self._current_values = None
        self._current_next_observations = None
        self._current_next_states = None
        self._rollout = 0

    # --------------------------------------------------------------
    # Memory layout (BlockAgent.init hook)
    # --------------------------------------------------------------
    def _create_memory_tensors(self) -> None:
        """On-policy rollout tensors. ``states`` present only in asymmetric mode."""
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # Names SAMPLED in mini-batches (rewards/terminated/truncated are read in
            # bulk for GAE via get_tensor_by_name, not sampled).
            if self._asymmetric:
                self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
                self._tensors_names = [
                    "observations", "states", "actions", "log_prob", "values", "returns", "advantages",
                ]
            else:
                self._tensors_names = [
                    "observations", "actions", "log_prob", "values", "returns", "advantages",
                ]

            # Optional ground-truth contact tensor for the supervised-selection loss.
            self._maybe_create_contact_tensor()
            self._maybe_create_rot_target_tensor()

    # --------------------------------------------------------------
    # Checkpoint hooks (BlockAgent does the slicing/stitching)
    # --------------------------------------------------------------
    def _checkpoint_model_keys(self) -> list[str]:
        return ["policy", "value"]

    def _checkpoint_optimizer_keys(self) -> list[str]:
        return ["policy_optimizer", "value_optimizer"]

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _value_inputs(self, observations: torch.Tensor, states: torch.Tensor, *, train: bool = False) -> dict:
        """Critic input dict: preprocessed states (asymmetric) or observations (symmetric)."""
        if self._asymmetric:
            return {"observations": self._state_preprocessor(states, train=train)}
        return {"observations": self._observation_preprocessor(observations, train=train)}

    # --------------------------------------------------------------
    # Interaction
    # --------------------------------------------------------------
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions; stash log_prob + value for the on-policy buffer."""
        inputs = {"observations": self._observation_preprocessor(observations)}

        if self.training and timestep < self.cfg.random_timesteps:
            # Uniform on [-1, 1]; still evaluate the policy/value on the emitted
            # action so the rollout's log_prob / value are well-defined.
            n = observations.shape[0]
            actions = torch.rand(n, *self.action_space.shape, device=self.device) * 2.0 - 1.0
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                _, outputs = self.policy.act({**inputs, "taken_actions": actions}, role="policy")
                self._current_log_prob = outputs["log_prob"]
                values, _ = self.value.act(self._value_inputs(observations, states), role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)
            return actions, outputs

        # no_grad: rollout sampling only. The PPO update recomputes log_prob/value
        # from the stored rollout for the gradient, so the action graph here is never
        # backpropagated — but if left attached it gets spliced into the controller
        # wrappers' EMA buffers in-place every step, growing an unbounded graph that
        # leaks GPU memory until OOM. (Stored log_prob/values are detached on buffer copy.)
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]
            if self.training:
                values, _ = self.value.act(self._value_inputs(observations, states), role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)
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
        """Per-agent metric ingestion (shared) + on-policy rollout write."""
        # Shared per-agent env-metric ingestion uses RAW (pre-shaping) rewards.
        self._ingest_step_metrics(
            rewards=rewards, terminated=terminated, truncated=truncated, infos=infos
        )

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            shaped = rewards
            if self.cfg.rewards_shaper is not None:
                shaped = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap and truncated.any():
                with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    next_values, _ = self.value.act(
                        self._value_inputs(next_observations, next_states), role="value"
                    )
                    next_values = self._value_preprocessor(next_values, inverse=True)
                shaped = shaped + self.cfg.discount_factor * next_values * truncated

            add_kwargs: dict[str, torch.Tensor] = dict(
                observations=observations,
                actions=actions,
                rewards=shaped,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
            )
            if self._asymmetric:
                if states is None or next_states is None:
                    raise RuntimeError(
                        "asymmetric PPO requires states and next_states from the trainer "
                        "(env.state() must return non-None)."
                    )
                add_kwargs["states"] = states
            # One-step-aligned contact ground truth for the SSL (no-op unless contact_axes).
            self._buffer_contact_for_write(
                terminated=terminated, truncated=truncated, infos=infos, add_kwargs=add_kwargs
            )
            # One-step-aligned rotation-frame target for the supervised-rotation loss (no-op unless rot6d_slice).
            self._buffer_rot_target_for_write(
                terminated=terminated, truncated=truncated, infos=infos, add_kwargs=add_kwargs
            )
            self.memory.add_samples(**add_kwargs)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            self._rollout += 1
            if not (self._rollout % self.cfg.rollouts) and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_per_agent(
                        "Stats / Algorithm update time (ms)",
                        [timer.elapsed_time_ms] * self.num_agents,
                    )
        # base.post_interaction handles checkpointing + write_tracking_data on interval
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    # --------------------------------------------------------------
    # Update
    # --------------------------------------------------------------
    def update(self, *, timestep: int, timesteps: int) -> None:
        N = self.num_agents
        epa = self.memory.num_envs // N

        # ---- compute returns + per-agent-normalized advantages (GAE) ----
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            last_values, _ = self.value.act(
                self._value_inputs(self._current_next_observations, self._current_next_states),
                role="value",
            )
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
            num_agents=N,
            envs_per_agent=epa,
        )
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # per-agent metric accumulators (lists of (N,) tensors, averaged at the end)
        acc: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
        # Per-agent (N,) raw aux-loss values, keyed by loss name. AuxLossManager.compute
        # returns the unweighted per-agent tensor; we keep the last minibatch's value.
        aux_policy_acc: dict[str, torch.Tensor] = {}
        aux_value_acc: dict[str, torch.Tensor] = {}

        for epoch in range(self.cfg.learning_epochs):
            kl_epoch: list[torch.Tensor] = []

            for batch in self.memory.sample_all(
                names=self._tensors_names, mini_batches=self.cfg.mini_batches, shuffle=True
            ):
                sampled = dict(zip(self._tensors_names, batch))
                rows = sampled["observations"].shape[0] // N

                def split(t):  # (N*rows, *) -> per-agent mean (N,)
                    return t.view(N, rows, -1).mean(dim=(1, 2))

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(
                            sampled["observations"], train=not epoch
                        )
                    }
                    if self._asymmetric:
                        value_inputs = {
                            "observations": self._state_preprocessor(sampled["states"], train=not epoch)
                        }
                    else:
                        value_inputs = inputs

                    # policy log_prob on the taken actions (PPO ratio)
                    _, outputs = self.policy.act(
                        {**inputs, "taken_actions": sampled["actions"]}, role="policy"
                    )
                    next_log_prob = outputs["log_prob"]

                    ratio_log = next_log_prob - sampled["log_prob"]
                    with torch.no_grad():
                        kl_per_agent = (((ratio_log).exp() - 1.0) - ratio_log).view(N, rows, -1).mean(dim=(1, 2))
                    kl_epoch.append(kl_per_agent)
                    acc["KL divergence"].append(kl_per_agent)

                    # KL early-stop: triggered by the WORST agent (conservative).
                    if self.cfg.kl_threshold and kl_per_agent.max() > self.cfg.kl_threshold:
                        break

                    # entropy (per-agent + scalar bonus)
                    entropy = self.policy.get_entropy(role="policy")
                    entropy_per_agent = entropy.view(N, rows, -1).mean(dim=(1, 2))
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * entropy.mean()
                    else:
                        entropy_loss = torch.zeros((), device=self.device)

                    # clipped surrogate policy loss
                    ratio = torch.exp(ratio_log)
                    surrogate = sampled["advantages"] * ratio
                    surrogate_clipped = sampled["advantages"] * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_terms = -torch.min(surrogate, surrogate_clipped)   # (N*rows, 1)
                    policy_loss = policy_terms.mean()

                    # clipped value loss
                    predicted_values, _ = self.value.act(value_inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled["values"] + torch.clip(
                            predicted_values - sampled["values"],
                            min=-self.cfg.value_clip, max=self.cfg.value_clip,
                        )
                    value_sq = (sampled["returns"] - predicted_values) ** 2   # (N*rows, 1)
                    value_loss = self.cfg.value_loss_scale * value_sq.mean()

                    # auxiliary losses (added to the single combined backward)
                    aux_total = torch.zeros((), device=self.device)
                    aux_policy_terms: dict[str, torch.Tensor] = {}
                    aux_value_terms: dict[str, torch.Tensor] = {}
                    if self._aux_losses is not None and self._aux_losses.has_target("policy"):
                        # Fresh policy sample so action-regularizers (e.g. action_l2)
                        # differentiate through a reparameterized sample, like SAC.
                        fresh_actions, fresh_out = self.policy.act(inputs, role="policy")
                        p_total, aux_policy_terms = self._aux_losses.compute(
                            LossContext(
                                agent=self, target="policy", sampled=sampled,
                                actions=fresh_actions, log_prob=fresh_out["log_prob"],
                                policy_outputs=fresh_out, inputs=inputs,
                            ),
                            "policy",
                        )
                        aux_total = aux_total + p_total
                    if self._aux_losses is not None and self._aux_losses.has_target("critic"):
                        v_total, aux_value_terms = self._aux_losses.compute(
                            LossContext(
                                agent=self, target="critic", sampled=sampled,
                                critic_1_values=predicted_values, critic_2_values=None,
                                target_values=sampled["returns"], critic_inputs=value_inputs,
                            ),
                            "critic",
                        )
                        aux_total = aux_total + v_total

                    total_loss = policy_loss + entropy_loss + value_loss + aux_total

                # ---- single combined backward, two disjoint optimizer steps ----
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.value is not self.policy:
                        self.value.reduce_parameters()

                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.value_optimizer)
                if self.write_interval > 0:
                    self._last_action_head_grad_norm = self._compute_actor_head_grad_norm()
                if self.cfg.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_norm_clip)
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.value_optimizer)
                self.scaler.update()

                # ---- per-agent metric accumulation ----
                if self.write_interval > 0:
                    with torch.no_grad():
                        acc["Loss / Policy loss"].append(split(policy_terms))
                        acc["Loss / Value loss"].append(split(value_sq) * self.cfg.value_loss_scale)
                        acc["Policy / Standard deviation"].append(
                            self.policy._g_distribution.stddev.view(N, rows, -1).mean(dim=(1, 2))
                        )
                        acc["Policy / Clip fraction"].append(
                            split(((ratio - 1.0).abs() > self.cfg.ratio_clip).float())
                        )
                        if self.cfg.entropy_loss_scale:
                            acc["Loss / Entropy loss"].append(entropy_per_agent)
                        # action diagnostics from the taken actions (parity with SAC)
                        abs_a = sampled["actions"].abs()
                        acc["Action / |a| max"].append(abs_a.view(N, rows, -1).amax(dim=(1, 2)))
                        acc["Action / |a| mean"].append(abs_a.view(N, rows, -1).mean(dim=(1, 2)))
                        acc["Action / saturation rate"].append(
                            (abs_a > 0.99).float().view(N, rows, -1).mean(dim=(1, 2))
                        )
                        gidx = self.cfg.gripper_action_idx
                        if gidx is not None:
                            g = sampled["actions"][..., gidx].unsqueeze(-1)
                            acc["Gripper / open rate"].append((g >= 0).float().view(N, rows, -1).mean(dim=(1, 2)))
                            acc["Gripper / action mean"].append(g.view(N, rows, -1).mean(dim=(1, 2)))
                    for name, val in aux_policy_terms.items():
                        aux_policy_acc[name] = val
                    for name, val in aux_value_terms.items():
                        aux_value_acc[name] = val

            # ---- LR schedulers (per epoch) ----
            mean_kl = torch.stack(kl_epoch).mean() if kl_epoch else torch.zeros((), device=self.device)
            for sched in (self.policy_scheduler, self.value_scheduler):
                if sched:
                    if isinstance(sched, KLAdaptiveLR):
                        kl = mean_kl.detach()
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                            kl /= config.torch.world_size
                        sched.step(kl.item())
                    else:
                        sched.step()

        # ---- flush per-agent metrics ----
        if self.write_interval > 0:
            for tag, vals in acc.items():
                if vals:
                    self.track_per_agent(tag, torch.stack(vals).mean(dim=0))
            # Aux losses: per-agent raw values under single-"/" tags, matching SAC
            # (`Loss/ {name}_raw_actor` / `_raw_critic`). `val` is already (N,).
            for name, val in aux_policy_acc.items():
                self.track_per_agent(f"Loss/ {name}_raw_actor", val)
            for name, val in aux_value_acc.items():
                self.track_per_agent(f"Loss/ {name}_raw_critic", val)
            if self._last_action_head_grad_norm is not None:
                self.track_per_agent("Action / action head grad norm", self._last_action_head_grad_norm)
            if self.policy_scheduler:
                self.track_per_agent("Learning / Policy learning rate",
                                     [self.policy_scheduler.get_last_lr()[0]] * N)
            else:
                self.track_per_agent("Learning / Policy learning rate", [self.cfg.policy_lr] * N)
            if self.value_scheduler:
                self.track_per_agent("Learning / Value learning rate",
                                     [self.value_scheduler.get_last_lr()[0]] * N)
            else:
                self.track_per_agent("Learning / Value learning rate", [self.cfg.value_lr] * N)
