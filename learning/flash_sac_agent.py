"""FlashSAC agent: ``FlashSAC(SAC)`` overriding only the update/rollout hooks.

Reuses the entire SAC machinery in ``learning/sac.py`` (per-agent block-parallel
optimizers, AMP, memory, entropy coefficient, logging, checkpointing) and overrides
just the pieces FlashSAC changes:

* :meth:`_compute_critic_loss` — cross-batch categorical (distributional) Bellman loss
  with an EMA target critic.
* :meth:`_compute_actor_loss` — cross-batch actor forward + expected-value Q.
* :meth:`_sample_rollout_action` — temporally-correlated noise repetition.
* :meth:`_should_update_actor` — actor-update delay.
* :meth:`_post_optimizer_step` — weight normalization.
* :meth:`_update_return_stats` — adaptive reward scaling statistics.
* :meth:`_compute_actor_head_grad_norm` — FlashSAC actor head (mean_w) grad norm.

All FlashSAC-specific tensor math lives in ``models/flash_sac.py``.
"""

from __future__ import annotations

import math

import gymnasium
import numpy as np
import torch

from learning.sac import SAC
from models.flash_sac import (
    FlashReturnScaler,
    block_cross_cat,
    block_cross_split,
    build_zeta_cdf,
    categorical_ce_loss,
    categorical_td_target,
    min_q_log_probs,
    normalize_flash_parameters,
    sample_zeta_lengths,
    squash_log_prob_correction,
)
from torch.distributions import Normal


class FlashSAC(SAC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # n-step return exponent for the categorical target (paper uses 1).
        self._n_step = 1

        # ---- unified entropy target (FlashSAC) ----
        # 0.5 * |A| * log(2*pi*e*sigma^2). Only when learning entropy with no explicit
        # target and the "unified" mode is selected; otherwise SAC's -|A| default stands.
        if self.cfg.learn_entropy and self.cfg.target_entropy is None \
                and getattr(self.cfg, "target_entropy_mode", "neg_action_dim") == "unified":
            if isinstance(self.action_space, gymnasium.spaces.Box):
                action_dim = float(np.prod(self.action_space.shape))
            else:
                action_dim = 1.0
            sigma = float(self.cfg.entropy_sigma_target)
            self._target_entropy = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * sigma * sigma)

        # ---- actor-update delay counter ----
        self._flash_grad_steps = 0

        # ---- noise-repetition state (lazily sized to the rollout batch) ----
        self._zeta_cdf = build_zeta_cdf(
            self.cfg.noise_repeat_zeta_mu, self.cfg.noise_repeat_max, self.device
        )
        self._nr_noise: torch.Tensor | None = None       # (rows, act)
        self._nr_count: torch.Tensor | None = None       # (rows, 1) int
        self._nr_len: torch.Tensor | None = None         # (rows, 1) int

        # ---- adaptive reward scaling (lazily sized once rewards are seen) ----
        self._return_scaler: FlashReturnScaler | None = None

        # Validate support/scale coupling early (fail loud).
        if self.cfg.reward_scaling_enabled:
            v_max = None
            for m in (self.critic_1, self.critic_2):
                v_max = getattr(m, "v_max", None)
                if v_max is not None:
                    break
            if v_max is not None and abs(float(v_max) - float(self.cfg.reward_scaling_g_max)) > 1e-9:
                raise ValueError(
                    f"reward_scaling_g_max ({self.cfg.reward_scaling_g_max}) must equal the "
                    f"categorical critic v_max ({v_max}) so normalized returns land inside the support."
                )

    # ------------------------------------------------------------------
    # Rollout: noise repetition
    # ------------------------------------------------------------------
    def _sample_rollout_action(self, inputs: dict, *, timestep: int):
        if not self.cfg.noise_repeat_enabled or not self.training:
            return self.policy.act(inputs, role="policy")

        obs = inputs["observations"]
        mean, log_std = self.policy.get_mean_and_std(obs, training=self.policy.training)
        std = log_std.exp()
        rows, act = mean.shape

        if self._nr_noise is None or self._nr_noise.shape != mean.shape:
            self._nr_noise = torch.randn_like(mean)
            self._nr_count = torch.zeros(rows, 1, dtype=torch.long, device=self.device)
            self._nr_len = sample_zeta_lengths(self._zeta_cdf, (rows,)).view(rows, 1)

        # Reinit noise where the hold has elapsed (count == 0 or count >= len).
        reinit = (self._nr_count == 0) | (self._nr_count >= self._nr_len)
        fresh_noise = torch.randn_like(mean)
        fresh_len = sample_zeta_lengths(self._zeta_cdf, (rows,)).view(rows, 1)
        self._nr_noise = torch.where(reinit, fresh_noise, self._nr_noise)
        self._nr_len = torch.where(reinit, fresh_len, self._nr_len)
        self._nr_count = torch.where(reinit, torch.zeros_like(self._nr_count), self._nr_count)

        u = mean + std * self._nr_noise
        a = torch.tanh(u)
        self._nr_count = self._nr_count + 1

        log_prob = (
            Normal(mean, std).log_prob(u).sum(dim=-1, keepdim=True)
            - squash_log_prob_correction(u).unsqueeze(-1)
        )
        return a, {"log_prob": log_prob, "mean_actions": torch.tanh(mean), "log_std": log_std}

    # ------------------------------------------------------------------
    # Update gating + hooks
    # ------------------------------------------------------------------
    def _should_update_actor(self, gradient_step: int) -> bool:
        period = max(int(self.cfg.actor_update_period), 1)
        ran = (self._flash_grad_steps % period) == 0
        self._flash_grad_steps += 1
        return ran

    def _post_optimizer_step(self) -> None:
        if self.cfg.weight_norm_enabled:
            normalize_flash_parameters(self.policy, self.critic_1, self.critic_2)

    def _update_return_stats(self, *, rewards, terminated, truncated) -> None:
        if not self.training or not self.cfg.reward_scaling_enabled:
            return
        if self._return_scaler is None:
            epa = rewards.shape[0] // self.num_agents
            self._return_scaler = FlashReturnScaler(
                num_agents=self.num_agents,
                envs_per_agent=epa,
                gamma=self.cfg.discount_factor,
                g_max=self.cfg.reward_scaling_g_max,
                device=self.device,
                eps=self.cfg.reward_scaling_eps,
            )
        self._return_scaler.update(rewards, terminated, truncated)

    # ------------------------------------------------------------------
    # Critic loss — cross-batch categorical (C51) with EMA target
    # ------------------------------------------------------------------
    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, B):
        N = self.num_agents
        sampled_actions = sampled["actions"]
        sampled_rewards = sampled["rewards"]
        sampled_terminated = sampled["terminated"]

        if self.cfg.reward_scaling_enabled and self._return_scaler is not None:
            sampled_rewards = self._return_scaler.scale(sampled_rewards, B)

        gamma_n = self.cfg.discount_factor ** self._n_step
        n_atoms = self.critic_1.n_atoms
        v_min, v_max = self.critic_1.v_min, self.critic_1.v_max

        cur_state = critic_inputs["observations"]
        nxt_state = critic_next_inputs["observations"]

        with torch.no_grad():
            # Next actions from the actor (running stats — no BN update in the critic step).
            next_actions, next_out = self.policy.act(
                {"observations": next_inputs["observations"], "training": False}, role="policy"
            )
            next_log_prob = next_out["log_prob"]                              # (N*B, 1)
            ent_flat = self._expand_per_agent(self._entropy_coefficient, B)   # (N*B, 1)
            actor_entropy = ent_flat * next_log_prob                          # (N*B, 1)

            # Cross-batch: [cur(s,a) ; next(s',a')] per agent, one training=True forward.
            obs_all = block_cross_cat(cur_state, nxt_state, N)
            act_all = block_cross_cat(sampled_actions, next_actions, N)

            tq1_all, tlp1_all = self.target_critic_1.forward_dist(obs_all, act_all, training=True)
            tq2_all, tlp2_all = self.target_critic_2.forward_dist(obs_all, act_all, training=True)
            _, tq1_next = block_cross_split(tq1_all, N)
            _, tq2_next = block_cross_split(tq2_all, N)
            _, tlp1_next = block_cross_split(tlp1_all, N)
            _, tlp2_next = block_cross_split(tlp2_all, N)

            next_log_probs = min_q_log_probs(tq1_next, tq2_next, tlp1_next, tlp2_next)
            target_probs = categorical_td_target(
                next_log_probs=next_log_probs,
                reward=sampled_rewards,
                done=sampled_terminated.float(),
                actor_entropy=actor_entropy,
                gamma=gamma_n,
                n_atoms=n_atoms,
                v_min=v_min,
                v_max=v_max,
            )

        # Online critics predict on the SAME cross-batch (shared BN stats); use current half.
        pq1_all, plp1_all = self.critic_1.forward_dist(obs_all, act_all, training=True)
        pq2_all, plp2_all = self.critic_2.forward_dist(obs_all, act_all, training=True)
        plp1_cur, _ = block_cross_split(plp1_all, N)
        plp2_cur, _ = block_cross_split(plp2_all, N)

        critic_loss = 0.5 * (
            categorical_ce_loss(target_probs, plp1_cur) + categorical_ce_loss(target_probs, plp2_cur)
        )

        # Expected values for logging (current halves) + expected target.
        q1_cur, _ = block_cross_split(pq1_all, N)
        q2_cur, _ = block_cross_split(pq2_all, N)
        target_values = (target_probs * self.critic_1.atoms).sum(dim=-1, keepdim=True)
        return critic_loss, q1_cur, q2_cur, target_values

    # ------------------------------------------------------------------
    # Actor loss — cross-batch actor forward + expected-value Q
    # ------------------------------------------------------------------
    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, B):
        N = self.num_agents

        # Cross-batch actor forward (current + next policy obs), BN batch stats; use
        # current half for the loss (matches the reference's actor update).
        cur_actor_obs = inputs["observations"]
        nxt_actor_obs = self._observation_preprocessor(sampled["next_observations"], train=True)
        obs_all = block_cross_cat(cur_actor_obs, nxt_actor_obs, N)
        actions_all, out_all = self.policy.act({"observations": obs_all, "training": True}, role="policy")
        actions, _ = block_cross_split(actions_all, N)
        log_prob, _ = block_cross_split(out_all["log_prob"], N)

        # Q for the policy gradient: critic on current (state, actions), running stats
        # (training=False, no BN update). Gradient flows through `actions` into the policy.
        q1, _ = self.critic_1.forward_dist(critic_inputs["observations"], actions, training=False)
        q2, _ = self.critic_2.forward_dist(critic_inputs["observations"], actions, training=False)
        q = torch.min(q1, q2)

        ent_flat = self._expand_per_agent(self._entropy_coefficient, B)
        policy_loss = (ent_flat * log_prob - q).mean()

        outputs = {"log_prob": log_prob, "mean_actions": None, "log_std": out_all.get("log_std")}
        return policy_loss, actions, log_prob, q1, q2, outputs, ent_flat

    # ------------------------------------------------------------------
    # FlashSAC actor head grad norm (mean_w) for the logging diagnostic
    # ------------------------------------------------------------------
    def _compute_actor_head_grad_norm(self):
        w = self.policy.mean_w.weight.grad
        if w is None:
            return None
        N = self.num_agents
        parts = [w.reshape(N, -1)]
        if self.policy.mean_w.bias is not None and self.policy.mean_w.bias.grad is not None:
            parts.append(self.policy.mean_w.bias.grad.reshape(N, -1))
        return torch.cat(parts, dim=1).norm(dim=1)
