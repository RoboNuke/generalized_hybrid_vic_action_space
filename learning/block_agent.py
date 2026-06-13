"""Shared base for block-parallel multi-agent agents (SAC, PPO).

``BlockAgent`` factors out everything that is INDEPENDENT of the learning
algorithm: per-agent tensorboard writers, the per-agent env-metric ingestion
(reward decomposition, distance/velocity, success rate/times, Forge early-term
prediction, ``infos["log"]`` mirroring), the interval flush, and per-agent
checkpoint slicing. Concrete agents (``SAC``, ``PPO``) subclass it and implement
only the algorithm-specific pieces (models/optimizers, ``act``, ``update``, and
the memory layout) plus a few checkpoint hooks.

The split is deliberately pure code-motion: SAC's tensorboard output and
checkpoints are unchanged by the extraction.
"""

from __future__ import annotations

import collections
import glob
import os
import re
from typing import Any

import gymnasium

import numpy as np
import torch

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils.tensorboard import SummaryWriter

from models.block_simba import (
    assign_block_slice,
    merge_optimizer_states,
    slice_block_state_dict,
    slice_optimizer_state,
)
from models.preprocessor_wrapper import PerAgentPreprocessorWrapper


class BlockAgent(Agent):
    """Base class holding the algorithm-agnostic block-parallel machinery.

    Each of ``num_agents`` agents owns a fixed env partition (envs
    ``[i*epa, (i+1)*epa)``) and gets its own tensorboard writer; no metric is
    aggregated across agents.
    """

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg=None,
        num_agents: int = 1,
        contact_axes: list[int] | None = None,
    ) -> None:
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.num_agents = num_agents

        # Optional ground-truth contact buffering for the supervised-selection loss.
        # ``contact_axes`` are the force-eligible sensor-contact columns (nonzero
        # ``controller_cfg.force_axes``, ascending x/y/z order); None disables it.
        # The stored contact width is ``len(contact_axes) == sum(force_axes)`` — it tracks
        # the force axes, not a fixed 3.
        self._contact_axes = (
            torch.as_tensor(list(contact_axes), dtype=torch.long, device=self.device)
            if contact_axes else None
        )
        self._contact_dim = len(contact_axes) if contact_axes else 0
        # One-step buffer: the contact the policy SAW at obs(t), written with transition t.
        self._pending_contact: torch.Tensor | None = None

        # per-agent tracking buffers (writers created in init() once experiment_dir is set)
        self.per_agent_writers: list[SummaryWriter] = []
        # Separate torch.utils.tensorboard.SummaryWriter per agent dedicated to
        # image events. skrl's SummaryWriter (used for scalars) doesn't
        # implement add_image; both writers point at the same log dir so TB
        # picks up both event streams together.
        self.per_agent_image_writers: list = []
        self.per_agent_tracking: list[collections.defaultdict] = []
        # Episode totals / lengths accumulated since the last write_interval flush.
        # write_tracking_data computes max/min/mean over the interval and then clears,
        # so the published values reflect ONLY the episodes that finished in the
        # current interval — no rolling-window lag.
        self._per_agent_track_rewards: list[list[float]] = []
        self._per_agent_track_timesteps: list[list[int]] = []
        # Finished-trajectory success labels accumulated since the last write_interval
        # flush. Cleared in write_tracking_data after the per-agent emit, so the published
        # rate always reflects only this-interval episodes (no rolling-window lag).
        self._per_agent_episodes_this_interval: list[list[int]] = []
        # Finished-trajectory engagement labels accumulated since the last write_interval
        # flush. Same semantics/lifecycle as the success buffer above (cleared in
        # write_tracking_data after emit) — drives the Episode / Engagement rate metric.
        self._per_agent_episodes_this_interval_engaged: list[list[int]] = []
        # Per-agent per-trajectory forward-distance accumulator (filled when a
        # task-specific wrapper publishes `info["per_env_episode_distance"]`),
        # cleared in write_tracking_data after emit. Currently the AntSuccess
        # wrapper publishes this; absent for other tasks (consumer skips).
        self._per_agent_distances_this_interval: list[list[float]] = []
        # Same pattern for per-trajectory average forward velocity (m/s).
        self._per_agent_velocities_this_interval: list[list[float]] = []
        # Last update's action-head gradient norm, captured pre-optimizer-step and
        # surfaced in write_tracking_data so the TB scalar doesn't have to be
        # re-derived after the step has already been applied.
        self._last_action_head_grad_norm: torch.Tensor | None = None
        # Per-agent best this-interval success rate seen so far. Initialized in
        # init() (one entry per agent). Each time write_tracking_data publishes a
        # finished-episode success rate that strictly beats an agent's best, that
        # agent's `ckpt_best.pt` is (re)written. -1.0 sentinel guarantees the first
        # interval with any finished episode establishes the initial best.
        self._best_success_rate: list[float] = []

    # --------------------------------------------------------------
    # Per-agent helpers
    # --------------------------------------------------------------
    def _expand_per_agent(self, x_n1: torch.Tensor, batch_per_agent: int) -> torch.Tensor:
        """``(N, 1) -> (N*B, 1)`` to broadcast against flat batch tensors."""
        return x_n1.repeat_interleave(batch_per_agent, dim=0)

    def track_per_agent(self, tag: str, values_per_agent) -> None:
        """Buffer a scalar per agent under ``tag``; ``values_per_agent`` is iterable of length N."""
        if not self.per_agent_tracking:
            return
        for i in range(self.num_agents):
            v = values_per_agent[i]
            self.per_agent_tracking[i][tag].append(v.item() if torch.is_tensor(v) else float(v))

    def _compute_actor_head_grad_norm(self):
        """Return the per-agent action-head gradient norm tensor (or None).

        The actor's ``fc_out`` BlockLinear holds the only action-head-private
        parameters: ``weight`` is (N, total_out, hidden) and ``bias`` is
        (N, total_out), with the action slice at the first ``_policy_out_dim``
        rows. Backbone params (fc_in, resblocks, ln_out, and the std rows of
        fc_out if state-dependent std is on) are shared, so we report only the
        head-private slice norm.

        Returns ``None`` if grads haven't been populated yet (e.g. the very
        first call before any backward) or the action slice is empty.
        """
        actor_mean = self.policy.actor_mean
        fc_out = actor_mean.fc_out
        N = self.num_agents
        out_dim = fc_out.weight.shape[1]
        policy_out = getattr(self.policy, "_policy_out_dim", out_dim - actor_mean.std_out_dim)

        # Grads may be None if backward didn't reach this param (e.g. on the
        # very first call before optimizer has stepped at all). Guard.
        wg = fc_out.weight.grad
        bg = fc_out.bias.grad
        if wg is None or bg is None:
            return None

        if policy_out <= 0:
            return None
        wa = wg[:, :policy_out, :].reshape(N, -1)   # (N, policy_out * hidden)
        ba = bg[:, :policy_out].reshape(N, -1)      # (N, policy_out)
        return torch.cat([wa, ba], dim=1).norm(dim=1)  # (N,)

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------
    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize per-agent writers and (via a hook) memory tensors.

        Drops the inherited single ``self.writer`` — every metric is published per-agent.

        Idempotent: ``trainer.train()`` calls ``init()`` internally, so the runner is
        free to call it manually first (e.g. to materialize per-agent folders for a
        config-dump). Subsequent calls are no-ops to avoid duplicating writers.
        """
        if getattr(self, "_init_done", False):
            return
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # tear down the shared writer the base class created (we publish per-agent only).
        # Base only sets self.writer when write_interval > 0; otherwise the attribute
        # may not exist at all.
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

        # Skrl's base init() also writes an empty tfevents file to <experiment_dir>/
        # (from the now-closed shared writer) and creates an empty <experiment_dir>/
        # checkpoints/ directory. Both belong to the per-agent subfolders only —
        # remove the top-level orphans so the experiment dir is clean. Use os.rmdir
        # (not rmtree) on the checkpoints folder so any unexpected file blocks the
        # deletion loudly rather than getting silently nuked.
        for events_file in glob.glob(os.path.join(self.experiment_dir, "events.out.tfevents.*")):
            try:
                os.remove(events_file)
            except OSError:
                pass
        ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            try:
                os.rmdir(ckpt_dir)
            except OSError:
                pass

        # per-agent writers + per-agent reward/episode deques.
        # Layout: <experiment_dir>/<i>/ holds tensorboard events AND checkpoints for agent i,
        # so each agent's folder is fully self-contained.
        if self.write_interval > 0:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            for i in range(self.num_agents):
                agent_log_dir = os.path.join(self.experiment_dir, str(i))
                self.per_agent_writers.append(
                    SummaryWriter(log_dir=agent_log_dir)
                )
                # Image writer (torch SummaryWriter) — same log dir so TB
                # aggregates scalar + image events under the same agent run.
                self.per_agent_image_writers.append(
                    TorchSummaryWriter(log_dir=agent_log_dir)
                )
                self.per_agent_tracking.append(collections.defaultdict(list))
                self._per_agent_track_rewards.append([])
                self._per_agent_track_timesteps.append([])
                self._per_agent_episodes_this_interval.append([])
                self._per_agent_episodes_this_interval_engaged.append([])
                self._per_agent_distances_this_interval.append([])
                self._per_agent_velocities_this_interval.append([])
                self._best_success_rate.append(-1.0)

        # Algorithm-specific memory tensors (obs/actions/... for SAC, on-policy
        # rollout tensors for PPO).
        self._create_memory_tensors()

        self._init_done = True

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Flush per-agent tracking buckets to per-agent writers."""
        for i, writer in enumerate(self.per_agent_writers):
            for tag, values in self.per_agent_tracking[i].items():
                if not values:
                    continue
                if tag.endswith("(min)"):
                    writer.add_scalar(tag=tag, value=float(np.min(values)), timestep=timestep)
                elif tag.endswith("(max)"):
                    writer.add_scalar(tag=tag, value=float(np.max(values)), timestep=timestep)
                else:
                    writer.add_scalar(tag=tag, value=float(np.mean(values)), timestep=timestep)

            # Episode totals + lengths over episodes that finished since the last
            # flush. Cleared here so the next interval starts fresh — no rolling-
            # window lag against `Episode_Termination/<term>` (which Isaac Lab logs
            # as the steady-state distribution over termination terms).
            rewards_list = self._per_agent_track_rewards[i]
            if rewards_list:
                arr = np.array(rewards_list, dtype=np.float64)
                writer.add_scalar(tag="Reward / Total reward (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Reward / Total reward (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Reward / Total reward (mean)", value=float(arr.mean()), timestep=timestep)
                rewards_list.clear()

            timesteps_list = self._per_agent_track_timesteps[i]
            if timesteps_list:
                arr = np.array(timesteps_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Total timesteps (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Total timesteps (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Total timesteps (mean)", value=float(arr.mean()), timestep=timestep)
                timesteps_list.clear()

            # Success rate over trajectories that finished since the last flush.
            ep = self._per_agent_episodes_this_interval[i]
            if ep:
                interval_success_rate = float(np.mean(ep))
                writer.add_scalar(
                    tag="Episode / Success rate",
                    value=interval_success_rate,
                    timestep=timestep,
                )
                ep.clear()

                # Keep a per-agent "best" checkpoint: whenever this interval's
                # success rate beats the agent's previous best, (re)write
                # ckpt_best.pt for that agent. The first interval with any finished
                # episode always wins (sentinel -1.0). `Episode / Best success
                # rate` is published every interval that has episodes so the TB
                # line is a flat-until-improved trace of the running best.
                if interval_success_rate > self._best_success_rate[i]:
                    self._best_success_rate[i] = interval_success_rate
                    self._write_best_checkpoint(i, timestep, interval_success_rate)
                writer.add_scalar(
                    tag="Episode / Best success rate",
                    value=self._best_success_rate[i],
                    timestep=timestep,
                )

            # Engagement rate over trajectories that finished since the last flush.
            # Same source/lifecycle as success rate, but for the curr_engaged flag
            # (peg close to socket). Cleared after emit so the rate reflects only
            # this-interval episodes (no rolling-window lag).
            eng = self._per_agent_episodes_this_interval_engaged[i]
            if eng:
                interval_engagement_rate = float(np.mean(eng))
                writer.add_scalar(
                    tag="Episode / Engagement rate",
                    value=interval_engagement_rate,
                    timestep=timestep,
                )
                eng.clear()

            # Per-trajectory forward distance (max/min/mean) — populated only when
            # a task-specific wrapper publishes per_env_episode_distance.
            dist_list = self._per_agent_distances_this_interval[i]
            if dist_list:
                arr = np.array(dist_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Distance traveled (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Distance traveled (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Distance traveled (mean)", value=float(arr.mean()), timestep=timestep)
                dist_list.clear()

            # Per-trajectory average velocity (max/min/mean) — same conditional
            # population as distance.
            vel_list = self._per_agent_velocities_this_interval[i]
            if vel_list:
                arr = np.array(vel_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Velocity (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Velocity (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Velocity (mean)", value=float(arr.mean()), timestep=timestep)
                vel_list.clear()

            self.per_agent_tracking[i].clear()
            # Persist synchronously: skrl's SummaryWriter flushes on a background
            # thread (flush_secs=120), but the runner's simulation_app.close() does
            # os._exit(0) on shutdown, which kills that thread before it flushes —
            # so fast runs would otherwise lose their events. flush() blocks until
            # the queued events hit disk.
            writer.flush()

    # --------------------------------------------------------------
    # Per-agent env-metric ingestion (shared by SAC and PPO record_transition)
    # --------------------------------------------------------------
    def _ingest_step_metrics(
        self,
        *,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
    ) -> None:
        """Per-agent reward/episode bookkeeping + env-published metric partitioning.

        Algorithm-agnostic: both SAC and PPO call this at the top of
        ``record_transition`` with the RAW (pre-shaping) rewards. Publishes nothing
        unless ``write_interval > 0``.
        """
        if self.write_interval > 0:
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(rewards)
            self._cumulative_timesteps.add_(1)

            total_envs = rewards.shape[0]
            epa = total_envs // self.num_agents
            rewards_per_agent = rewards.view(self.num_agents, epa, 1)

            self.track_per_agent("Reward / Instantaneous reward (max)",
                                 rewards_per_agent.amax(dim=(1, 2)))
            self.track_per_agent("Reward / Instantaneous reward (min)",
                                 rewards_per_agent.amin(dim=(1, 2)))
            self.track_per_agent("Reward / Instantaneous reward (mean)",
                                 rewards_per_agent.mean(dim=(1, 2)))

            # Per-env episode finishes; partition by agent index. Stats over the
            # accumulated episodes (max/min/mean) are emitted in write_tracking_data
            # at write_interval boundaries — see that method for the flush + clear.
            done = (terminated + truncated).bool().view(-1)
            cum_r_flat = self._cumulative_rewards.view(-1)
            cum_t_flat = self._cumulative_timesteps.view(-1)
            for i in range(self.num_agents):
                env_lo, env_hi = i * epa, (i + 1) * epa
                done_slice = done[env_lo:env_hi]
                if done_slice.any():
                    finished_envs = done_slice.nonzero(as_tuple=False).view(-1) + env_lo
                    self._per_agent_track_rewards[i].extend(cum_r_flat[finished_envs].tolist())
                    self._per_agent_track_timesteps[i].extend(cum_t_flat[finished_envs].tolist())
                    self._cumulative_rewards.view(-1)[finished_envs] = 0
                    self._cumulative_timesteps.view(-1)[finished_envs] = 0

            # Per-agent reward decomposition (preferred): a wrapper publishes
            # already-normalized per-env per-term values in `infos["per_env_rew"]`
            # plus a bool mask in `infos["per_env_rew_mask"]`. We partition by
            # agent (env i belongs to agent i // epa) and mean over the resetting
            # envs in that agent's slice. Tag matches Isaac Lab's convention so
            # old + new tensorboards plot continuously.
            per_env_rew_terms: set[str] = set()
            if (
                isinstance(infos, dict)
                and "per_env_rew" in infos
                and "per_env_rew_mask" in infos
            ):
                per_env = infos["per_env_rew"]
                mask = infos["per_env_rew_mask"]
                if torch.is_tensor(mask) and mask.any():
                    for term, per_env_vals in per_env.items():
                        per_env_rew_terms.add(f"Episode_Reward/{term}")
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_mask = mask[env_lo:env_hi]
                            if not agent_mask.any():
                                continue
                            vals = per_env_vals[env_lo:env_hi][agent_mask]
                            self.per_agent_tracking[i][f"Episode_Reward/{term}"].append(
                                float(vals.mean().item())
                            )

            # Per-trajectory forward-distance ingestion (task-specific wrapper).
            # Wrappers like AntSuccessWrapper publish `info["per_env_episode_distance"]`
            # (the final per-env displacement for the just-ended episode) plus a
            # mask. We accumulate per-agent into a per-interval list; max/min/mean
            # are emitted in write_tracking_data, then cleared.
            if (
                isinstance(infos, dict)
                and "per_env_episode_distance" in infos
                and "per_env_episode_distance_mask" in infos
                and self._per_agent_distances_this_interval
            ):
                distances = infos["per_env_episode_distance"]
                dist_mask = infos["per_env_episode_distance_mask"]
                if torch.is_tensor(dist_mask) and dist_mask.any():
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_mask = dist_mask[env_lo:env_hi]
                        if not agent_mask.any():
                            continue
                        vals = distances[env_lo:env_hi][agent_mask]
                        self._per_agent_distances_this_interval[i].extend(
                            vals.tolist()
                        )

            # Per-trajectory average-velocity ingestion (task-specific wrapper).
            # Same pattern as distance — accumulate per-agent into a per-interval
            # list, flush + clear in write_tracking_data.
            if (
                isinstance(infos, dict)
                and "per_env_episode_velocity" in infos
                and "per_env_episode_velocity_mask" in infos
                and self._per_agent_velocities_this_interval
            ):
                velocities = infos["per_env_episode_velocity"]
                vel_mask = infos["per_env_episode_velocity_mask"]
                if torch.is_tensor(vel_mask) and vel_mask.any():
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_mask = vel_mask[env_lo:env_hi]
                        if not agent_mask.any():
                            continue
                        vals = velocities[env_lo:env_hi][agent_mask]
                        self._per_agent_velocities_this_interval[i].extend(
                            vals.tolist()
                        )

            # Per-step per-agent partition for Factory/Forge diagnostic tensors.
            # The wrappers publish raw per-env tensors under `per_env_*` keys;
            # we slice each agent's [env_lo:env_hi] portion, compute the right
            # aggregate (env-mean, mean-over-non-zero, etc.), and push to that
            # agent's tracking bucket. Without this, the env-aggregated scalars
            # in info["log"]["logs_rew_*"] / "successes" / "success_times"
            # would be the same value across all agents.
            per_env_logs_terms_handled: set[str] = set()
            if isinstance(infos, dict):
                # logs_rew/<term>: env-mean of unscaled per-step reward components.
                per_env_logs = infos.get("per_env_logs_rew")
                if isinstance(per_env_logs, dict):
                    for term, vals in per_env_logs.items():
                        # Wrappers MUST publish per-env tensors here (filtered
                        # at source). Anything else is a wrapper-side bug — fail
                        # loud so we don't silently drop a metric.
                        if not torch.is_tensor(vals):
                            raise TypeError(
                                f"per_env_logs_rew[{term!r}] is {type(vals).__name__}, "
                                f"expected torch.Tensor of shape ({total_envs},)"
                            )
                        if vals.dim() == 0 or vals.shape[0] != total_envs:
                            raise ValueError(
                                f"per_env_logs_rew[{term!r}] has shape {tuple(vals.shape)}, "
                                f"expected first dim == total_envs ({total_envs}). "
                                f"Filter scalar/aggregated terms in the wrapper before publishing."
                            )
                        per_env_logs_terms_handled.add(f"logs_rew_{term}")
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_vals = vals[env_lo:env_hi].float()
                            self.per_agent_tracking[i][f"logs_rew/{term}"].append(
                                float(agent_vals.mean().item())
                            )

                # Per-step control-wrapper diagnostics (e.g. "Contact / In-Contact *").
                # Wrappers stash per-env tensors in extras["to_log"]; the reward-
                # decomposition wrapper forwards them RAW under per_env_to_log so each
                # agent gets the env-mean over its OWN env partition. Mirroring the
                # global env-mean from info["log"] instead would make the value
                # identical across agents (the bug this fixes). The tag is logged
                # verbatim, so it never collides with info["log"] (those entries are
                # disjoint by construction in _forward_to_log).
                per_env_to_log = infos.get("per_env_to_log")
                if isinstance(per_env_to_log, dict):
                    for tag, vals in per_env_to_log.items():
                        if not torch.is_tensor(vals):
                            raise TypeError(
                                f"per_env_to_log[{tag!r}] is {type(vals).__name__}, "
                                f"expected torch.Tensor of shape ({total_envs},)"
                            )
                        if vals.dim() == 0 or vals.shape[0] != total_envs:
                            raise ValueError(
                                f"per_env_to_log[{tag!r}] has shape {tuple(vals.shape)}, "
                                f"expected first dim == total_envs ({total_envs})."
                            )
                        # Reduce each agent's env partition by the SAME convention
                        # write_tracking_data uses for the interval: a "(max)"/"(min)"
                        # tag suffix takes the per-env peak/trough (so e.g. a max-force
                        # tag reports the true maximum any env saw, not the peak of the
                        # env-mean); everything else env-means.
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            sl = vals[env_lo:env_hi].float()
                            if tag.endswith("(max)"):
                                red = sl.max()
                            elif tag.endswith("(min)"):
                                red = sl.min()
                            else:
                                red = sl.mean()
                            self.per_agent_tracking[i][tag].append(float(red.item()))

                # Per-agent `successes` and `success_times` mirror upstream
                # Factory's `_log_factory_metrics` semantics, just sliced to each
                # agent's env partition. Factory only writes these to extras on
                # specific steps; we replicate that timing per-agent so the TB
                # values are directly comparable to a single-agent run.
                #
                #   `successes`     : at any step where the agent has a resetting
                #                     env, count_nonzero(curr_successes[slice]) /
                #                     epa  (fraction of agent's envs at goal at
                #                     reset moment).
                #   `success_times` : at any step where any env in the slice has
                #                     a nonzero ep_success_time, mean step-of-
                #                     first-success across those envs.
                curr_succ = infos.get("per_env_curr_successes")
                if curr_succ is not None:
                    if not (torch.is_tensor(curr_succ) and curr_succ.dim() > 0
                            and curr_succ.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_curr_successes has shape "
                            f"{tuple(curr_succ.shape) if torch.is_tensor(curr_succ) else type(curr_succ).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    done_flat = (terminated + truncated).bool().view(-1)
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        done_slice = done_flat[env_lo:env_hi]
                        if done_slice.any():
                            agent_succ = curr_succ[env_lo:env_hi]
                            self.per_agent_tracking[i]["successes"].append(
                                float(agent_succ.float().mean().item())
                            )
                            # Episode / Success rate (env-driven): record each
                            # finished trajectory's success flag so the interval
                            # mean emitted in write_tracking_data is a true
                            # per-trajectory success rate, sourced from the env's
                            # success signal rather than any predictor head.
                            done_idx = done_slice.nonzero(as_tuple=False).view(-1)
                            self._per_agent_episodes_this_interval[i].extend(
                                agent_succ[done_idx].long().tolist()
                            )

                # Engagement rate (env-driven): identical partitioning to success,
                # sourced from the per-step curr_engaged flag (peg close to socket)
                # snapshotted at the reset moment. Records each finished trajectory's
                # engagement flag so write_tracking_data's interval mean is a true
                # per-trajectory engagement rate.
                curr_eng = infos.get("per_env_curr_engaged")
                if curr_eng is not None:
                    if not (torch.is_tensor(curr_eng) and curr_eng.dim() > 0
                            and curr_eng.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_curr_engaged has shape "
                            f"{tuple(curr_eng.shape) if torch.is_tensor(curr_eng) else type(curr_eng).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    done_flat = (terminated + truncated).bool().view(-1)
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        done_slice = done_flat[env_lo:env_hi]
                        if done_slice.any():
                            agent_eng = curr_eng[env_lo:env_hi]
                            done_idx = done_slice.nonzero(as_tuple=False).view(-1)
                            self._per_agent_episodes_this_interval_engaged[i].extend(
                                agent_eng[done_idx].long().tolist()
                            )

                ep_succ_times = infos.get("per_env_ep_success_times")
                if ep_succ_times is not None:
                    if not (torch.is_tensor(ep_succ_times) and ep_succ_times.dim() > 0
                            and ep_succ_times.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_ep_success_times has shape "
                            f"{tuple(ep_succ_times.shape) if torch.is_tensor(ep_succ_times) else type(ep_succ_times).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_times = ep_succ_times[env_lo:env_hi]
                        nonzero = agent_times > 0
                        if nonzero.any():
                            self.per_agent_tracking[i]["success_times"].append(
                                float(agent_times[nonzero].float().mean().item())
                            )

                # Forge early_term_*: per-agent versions of Forge's success-
                # prediction quality metrics, fed by the env's published
                # per-env first_pred_success_tx (per threshold) + ep_success_times.
                # Tag layout: `<thresh>/early_term_*` so each TB tab is one τ.
                first_pred = infos.get("per_env_first_pred_success_tx")
                if isinstance(first_pred, dict) and torch.is_tensor(ep_succ_times):
                    for thresh, fst in first_pred.items():
                        if not (torch.is_tensor(fst) and fst.dim() > 0
                                and fst.shape[0] == total_envs):
                            raise ValueError(
                                f"per_env_first_pred_success_tx[{thresh!r}] has shape "
                                f"{tuple(fst.shape) if torch.is_tensor(fst) else type(fst).__name__}, "
                                f"expected ({total_envs},)"
                            )
                        tag_root = f"{float(thresh):.1f}"
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_fst = fst[env_lo:env_hi]
                            agent_est = ep_succ_times[env_lo:env_hi]

                            delay_mask = (agent_est != 0) & (agent_fst != 0)
                            if delay_mask.any():
                                delay = (agent_fst[delay_mask] - agent_est[delay_mask]).float().mean()
                                self.per_agent_tracking[i][f"{tag_root}/early_term_delay_all"].append(
                                    float(delay.item())
                                )
                                correct_mask = delay_mask & (agent_fst > agent_est)
                                if correct_mask.any():
                                    cd = (agent_fst[correct_mask] - agent_est[correct_mask]).float().mean()
                                    self.per_agent_tracking[i][f"{tag_root}/early_term_delay_correct"].append(
                                        float(cd.item())
                                    )
                            pred_mask = agent_fst != 0
                            if pred_mask.any():
                                tps = (agent_est[pred_mask] > 0) & (agent_est[pred_mask] < agent_fst[pred_mask])
                                self.per_agent_tracking[i][f"{tag_root}/early_term_precision"].append(
                                    float(tps.float().sum().item() / pred_mask.float().sum().item())
                                )
                            true_mask = agent_est > 0
                            if true_mask.any() and pred_mask.any():
                                tps = (agent_est[pred_mask] > 0) & (agent_est[pred_mask] < agent_fst[pred_mask])
                                self.per_agent_tracking[i][f"{tag_root}/early_term_recall"].append(
                                    float(tps.float().sum().item() / true_mask.float().sum().item())
                                )

            # Isaac Lab managers publish mean-across-resetting-envs scalars under
            # infos["log"] on reset steps (e.g. "Episode_Reward/lifting_object",
            # "Episode_Termination/<name>"). The env is shared across agents so we
            # mirror each scalar into every per-agent writer — except for
            # `Episode_Reward/<term>` keys already handled per-agent above.
            if isinstance(infos, dict):
                env_log = infos.get("log")
                if isinstance(env_log, dict) and env_log:
                    for k, v in env_log.items():
                        if k in per_env_rew_terms:
                            continue  # superseded by the per-agent decomposition
                        # Skip env-aggregated logs_rew_/successes/success_times
                        # if a wrapper is publishing per-env tensors for them
                        # (they get logged per-agent above instead).
                        if k in per_env_logs_terms_handled:
                            continue
                        if "per_env_curr_successes" in infos and k == "successes":
                            continue
                        if "per_env_ep_success_times" in infos and k == "success_times":
                            continue
                        if "per_env_first_pred_success_tx" in infos and (
                            k.startswith("early_term_delay_all/")
                            or k.startswith("early_term_delay_correct/")
                            or k.startswith("early_term_precision/")
                            or k.startswith("early_term_recall/")
                        ):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
                            scalar = float(v.item())
                        elif isinstance(v, (int, float, np.integer, np.floating)):
                            scalar = float(v)
                        else:
                            continue
                        # TB tag rewrite: turn `logs_rew_<term>` into
                        # `logs_rew/<term>` so the dashboard groups all
                        # reward-component scalars under a single section.
                        tag = (
                            "logs_rew/" + k[len("logs_rew_"):]
                            if k.startswith("logs_rew_")
                            else k
                        )
                        for i in range(self.num_agents):
                            self.per_agent_tracking[i][tag].append(scalar)

    # --------------------------------------------------------------
    # Memory-tensor hook (subclass-specific)
    # --------------------------------------------------------------
    def _create_memory_tensors(self) -> None:
        """Create the algorithm-specific memory tensors and set ``self._tensors_names``."""
        raise NotImplementedError

    # --------------------------------------------------------------
    # Ground-truth contact buffering (supervised-selection loss) — shared by SAC/PPO
    # --------------------------------------------------------------
    def _maybe_create_contact_tensor(self) -> None:
        """When ``contact_axes`` is set, create the ``in_contact`` memory tensor (width =
        ``len(contact_axes) == sum(force_axes)``) and add it to ``self._tensors_names`` so
        it is returned in the sampled minibatch dict. Call at the end of each subclass
        ``_create_memory_tensors``."""
        if self._contact_axes is not None and self.memory is not None:
            self.memory.create_tensor(name="in_contact", size=self._contact_dim, dtype=torch.float32)
            self._tensors_names.append("in_contact")

    def _buffer_contact_for_write(
        self, *, terminated: torch.Tensor, truncated: torch.Tensor, infos: Any, add_kwargs: dict
    ) -> None:
        """One-step-aligned contact for the SSL target. Injects into ``add_kwargs`` the
        contact the policy SAW at this transition's obs(t) (= the previous step's
        post-step contact, sliced to the force-eligible axes), then refreshes the pending
        buffer from this step's ``infos["in_contact"]`` (zeroing envs that just reset, so a
        new episode starts with no contact). No-op when ``contact_axes`` is None."""
        if self._contact_axes is None:
            return
        n_envs = terminated.shape[0]
        if self._pending_contact is None:
            self._pending_contact = torch.zeros((n_envs, self._contact_dim), device=self.device)
        add_kwargs["in_contact"] = self._pending_contact
        # refresh: this step's post-step contact (the contact obs(t+1) will carry)
        raw = infos.get("in_contact") if isinstance(infos, dict) else None
        if raw is not None:
            nxt = raw.index_select(-1, self._contact_axes).float().clone()
        else:
            nxt = torch.zeros((n_envs, self._contact_dim), device=self.device)
        done = (terminated + truncated).bool().view(-1)
        nxt[done] = 0.0
        self._pending_contact = nxt

    # --------------------------------------------------------------
    # Per-agent checkpoint save/load (generic; specialized via hooks)
    # --------------------------------------------------------------
    def _checkpoint_model_keys(self) -> list[str]:
        """Attribute names of block-parallel ``Model``s to slice/assign on save/load."""
        raise NotImplementedError

    def _checkpoint_optimizer_keys(self) -> list[str]:
        """Attribute names of optimizers whose state is sliced/merged across agents."""
        raise NotImplementedError

    def _required_checkpoint_keys(self) -> set[str]:
        """Top-level keys every per-agent checkpoint must contain."""
        keys = {"step", "num_agents", "agent_idx", "observation_preprocessor"}
        keys.update(self._checkpoint_model_keys())
        keys.update(self._checkpoint_optimizer_keys())
        return keys

    def _build_extra_checkpoint(self, i: int) -> dict:
        """Extra per-agent checkpoint entries beyond models/optimizers/preprocessor."""
        return {}

    def _load_extra_into_slot(self, target_slot: int, ckpt: dict, path: str) -> None:
        """Load extra per-agent state (non-optimizer) for slot ``target_slot``."""
        pass

    def _load_extra_optimizers(self, per_agent_ckpts: list[dict], path: str) -> None:
        """Load extra (conditional) optimizers — e.g. SAC's entropy optimizer."""
        pass

    def _build_per_agent_checkpoint(self, i: int, step: int) -> dict:
        """Build the per-agent checkpoint dict for slot ``i`` at training ``step``."""
        ckpt = {
            "step": int(step),
            "num_agents": int(self.num_agents),
            "agent_idx": int(i),
            "observation_preprocessor": self._build_preprocessor_state_for(i),
        }
        for key in self._checkpoint_model_keys():
            ckpt[key] = slice_block_state_dict(getattr(self, key), i, self.num_agents)
        for key in self._checkpoint_optimizer_keys():
            ckpt[key] = slice_optimizer_state(getattr(self, key).state_dict(), i, self.num_agents)
        ckpt.update(self._build_extra_checkpoint(i))
        return ckpt

    def _build_preprocessor_state_for(self, i: int):
        """Return the preprocessor state dict for agent ``i``, or None if no preprocessor.

        If a ``PerAgentPreprocessorWrapper`` is configured, the per-agent state for slot
        ``i`` MUST be present — anything else is a configuration error.
        """
        if not isinstance(self._observation_preprocessor, PerAgentPreprocessorWrapper):
            return None
        full = self._observation_preprocessor.state_dict()
        key = f"agent_{i}"
        if key not in full:
            raise KeyError(
                f"PerAgentPreprocessorWrapper has no state for {key}; "
                f"got keys {sorted(full.keys())}"
            )
        return full[key]

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Save one ``ckpt_{timestep}.pt`` file per agent, each in its own folder.

        Replaces the base Agent's bundled checkpoint write; we save sliced state
        per-agent so each agent's folder is fully independent.
        """
        tag = str(timestep)
        for i in range(self.num_agents):
            ckpt_dir = os.path.join(self.experiment_dir, str(i), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")
            torch.save(self._build_per_agent_checkpoint(i, timestep), path)

    def _write_best_checkpoint(self, i: int, timestep: int, success_rate: float) -> None:
        """(Re)write agent ``i``'s ``ckpt_best.pt`` — the highest this-interval
        success-rate checkpoint seen so far.

        Overwrites the single fixed-name file each time the agent improves, so it
        always holds the current best. The non-numeric ``best`` tag keeps it out of
        the "latest" search in ``_resolve_ckpt_file`` (its glob ranks by integer
        step, and ``ckpt_best.pt`` parses to -1, so it never wins). ``best_success_rate``
        is stamped into the dict for provenance; load ignores unknown extra keys.
        """
        ckpt_dir = os.path.join(self.experiment_dir, str(i), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = self._build_per_agent_checkpoint(i, timestep)
        ckpt["best_success_rate"] = float(success_rate)
        torch.save(ckpt, os.path.join(ckpt_dir, "ckpt_best.pt"))

    # --- load helpers ---
    @staticmethod
    def _is_single_agent_dir(path: str) -> bool:
        """A folder is a 'single-agent' folder if it contains checkpoints/ckpt_*.pt directly."""
        return bool(glob.glob(os.path.join(path, "checkpoints", "ckpt_*.pt")))

    @staticmethod
    def _resolve_ckpt_file(agent_dir: str, step: int | None) -> str:
        """Return the path to ckpt_{step}.pt, or the latest if step is None."""
        ckpt_dir = os.path.join(agent_dir, "checkpoints")
        if step is not None:
            path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return path
        candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No ckpt_*.pt files in {ckpt_dir}")

        def _step_of(p: str) -> int:
            m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        return max(candidates, key=_step_of)

    def _load_preprocessor_into_slot(self, target_slot: int, ckpt: dict, path: str) -> None:
        """Load the per-agent observation-preprocessor state into slot ``target_slot``."""
        wrapper_configured = isinstance(self._observation_preprocessor, PerAgentPreprocessorWrapper)
        saved_preproc = ckpt["observation_preprocessor"]
        if wrapper_configured and saved_preproc is None:
            raise ValueError(
                f"PerAgentPreprocessorWrapper is configured but checkpoint at {path} "
                f"has no observation_preprocessor state."
            )
        if not wrapper_configured and saved_preproc is not None:
            raise ValueError(
                f"Checkpoint at {path} contains observation_preprocessor state but the "
                f"current run has no PerAgentPreprocessorWrapper configured."
            )
        if wrapper_configured:
            preproc = self._observation_preprocessor.preprocessor_list[target_slot]
            if preproc is None:
                raise ValueError(
                    f"PerAgentPreprocessorWrapper slot {target_slot} is None; cannot "
                    f"load preprocessor state from {path}."
                )
            preproc.load_state_dict(saved_preproc)

    def _load_one_into_slot(self, agent_dir: str, target_slot: int, step: int | None) -> dict:
        """Load a single per-agent ckpt file from ``agent_dir`` into block slot ``target_slot``.

        Loads weights, per-slot preprocessor state, and (via hook) any extra per-slot
        state. Optimizer state is NOT loaded here (caller stitches optimizer states in
        bulk). Returns the raw checkpoint dict for follow-up handling.
        """
        path = self._resolve_ckpt_file(agent_dir, step)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Validate required top-level keys are present (no silent fallback).
        missing = self._required_checkpoint_keys() - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint at {path} is missing required keys: {sorted(missing)}")

        for key in self._checkpoint_model_keys():
            assign_block_slice(getattr(self, key), target_slot, self.num_agents, ckpt[key])

        self._load_preprocessor_into_slot(target_slot, ckpt, path)
        self._load_extra_into_slot(target_slot, ckpt, path)

        return ckpt

    def load(self, path: str, *, step: int | None = None) -> None:
        """Load weights/state from a checkpoint folder.

        Two modes auto-detected from ``path``:

        * **Single-agent**: ``path/checkpoints/ckpt_*.pt`` exists directly. Requires
          the current run to have ``num_agents == 1``; loads into slot 0.
        * **Multi-agent**: ``path`` contains subfolders ``0/``, ``1/``, ..., each with
          its own ``checkpoints/ckpt_*.pt``. Strict ``num_agents`` match required.

        :param path: Folder path (per the modes above).
        :param step: Optional specific training step to load. If ``None`` (default),
            the latest ``ckpt_<step>.pt`` found in each folder is used.
        """
        if self._is_single_agent_dir(path):
            if self.num_agents != 1:
                raise ValueError(
                    f"Single-agent checkpoint at {path} requires num_agents=1, "
                    f"but this run has num_agents={self.num_agents}"
                )
            ckpt = self._load_one_into_slot(path, target_slot=0, step=step)
            if "num_agents" not in ckpt:
                raise KeyError(f"Checkpoint at {path} is missing required key 'num_agents'")
            src_n = int(ckpt["num_agents"])
            # A slice taken from an N>1 file carries optimizer state with param_groups that
            # don't fit a 1-agent optimizer. We refuse to load — silently dropping it would
            # give the user a 1-agent run with fresh Adam moments under the same name.
            if src_n != 1:
                raise ValueError(
                    f"Refusing to load single-agent checkpoint at {path}: it was sliced "
                    f"from a num_agents={src_n} run, so its optimizer state cannot be "
                    f"restored into a 1-agent optimizer. Use multi-agent load mode "
                    f"(point --checkpoint at the parent run dir) or train fresh."
                )
            for key in self._checkpoint_optimizer_keys():
                getattr(self, key).load_state_dict(merge_optimizer_states([ckpt[key]], 1))
            self._load_extra_optimizers([ckpt], path)
            return

        # Multi-agent: expect path/0, path/1, ..., path/(N-1) all present.
        per_agent_ckpts: list[dict] = []
        for i in range(self.num_agents):
            agent_dir = os.path.join(path, str(i))
            if not os.path.isdir(agent_dir):
                raise FileNotFoundError(
                    f"Expected per-agent dir {agent_dir} for num_agents={self.num_agents}"
                )
            ckpt = self._load_one_into_slot(agent_dir, target_slot=i, step=step)
            if "num_agents" not in ckpt:
                raise KeyError(f"Checkpoint at {agent_dir} is missing required key 'num_agents'")
            if ckpt["num_agents"] != self.num_agents:
                raise ValueError(
                    f"Checkpoint at {agent_dir} has num_agents={ckpt['num_agents']} but "
                    f"current run has num_agents={self.num_agents}"
                )
            per_agent_ckpts.append(ckpt)

        # Stitch optimizer state across agents.
        for key in self._checkpoint_optimizer_keys():
            getattr(self, key).load_state_dict(
                merge_optimizer_states([c[key] for c in per_agent_ckpts], self.num_agents)
            )
        self._load_extra_optimizers(per_agent_ckpts, path)
