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
from learning.metric_writer import MetricWriter, make_wandb_run


# Insertion-task phases for per-phase metric splitting (cfg.phase_split_families).
# Ordered so the phase id is the tuple index: 0 free_space, 1 search, 2 insertion.
# Classification priority is insertion > search > free_space (see _compute_phase_ids).
_PHASE_NAMES = ("free_space", "search", "insertion")


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
        # Per-agent scalar sink: a raw skrl SummaryWriter, or (when experiment.wandb
        # is set) a MetricWriter wrapping it that also mirrors to a per-agent wandb run.
        self.per_agent_writers: list = []
        # Separate torch.utils.tensorboard.SummaryWriter per agent dedicated to
        # image events. skrl's SummaryWriter (used for scalars) doesn't
        # implement add_image; both writers point at the same log dir so TB
        # picks up both event streams together.
        # Kept as an (always-empty) list: the record path defensively reads
        # ``per_agent_image_writers[0] if ... else None``. We no longer create image
        # SummaryWriters (histograms were removed), so the recorder gets None and writes
        # its GIF to disk only — exactly the record-mode behavior (write_interval=0).
        self.per_agent_image_writers: list = []
        self.per_agent_tracking: list[collections.defaultdict] = []
        # Per-agent GPU running sufficient statistics for "(dist)"/"(stat)" metrics: one
        # {count, sum, sumsq} accumulator (float64 device scalars) per tag, created once and
        # reused (zeroed in write_tracking_data). The per-env samples are reduced on-device each
        # step — NO per-step GPU->CPU copy and NO Python-object churn (the old per-step
        # ``.tolist()`` into Python lists fragmented host RAM). Flush emits mean + std.
        self.per_agent_dist_stats: list[dict] = []
        # Per-agent GPU running (count, sum, min, max) for the per-step / per-grad-step
        # SCALAR metrics that used to be appended to ``per_agent_tracking`` as Python floats
        # via ``float(x.item())`` (track_per_agent, logs_rew, per_env_to_log scalar reductions,
        # phase-split). Each ``.item()`` was a GPU->CPU sync barrier fired dozens of times per
        # step; folding the already-reduced scalar on-device removes ALL per-step syncs and the
        # Python-list churn. write_tracking_data emits by tag suffix ("(max)"->max, "(min)"->min,
        # else mean) — identical to the old np.max/np.min/np.mean over the interval's list.
        self.per_agent_scalar_stats: list[dict] = []
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
        # Per-env (total_envs) "engaged on ANY step this episode" latch, OR-accumulated from
        # the single per_env_curr_engaged info signal each step and cleared per-env on episode
        # end. Lets Episode / Engagement rate count a trajectory as engaged if it engaged at
        # ANY point in the rollout (not just the terminal step). Lazily sized on first step.
        self._ever_engaged_latch: torch.Tensor | None = None
        # Finished-trajectory "ever succeeded this episode" labels (the pre-reset
        # ep_succeeded latch), same lifecycle as the buffers above — drives the
        # Episode / Ever success rate metric. Unlike Episode / Success rate (peg
        # inserted on the TERMINAL step only), this counts a trajectory that
        # reached success on ANY step, so the gap between the two surfaces
        # transient insertions that were not held to episode end.
        self._per_agent_episodes_this_interval_ever: list[list[int]] = []
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
        """Fold one already-reduced scalar per agent under ``tag`` into the on-device running
        (count, sum, min, max). ``values_per_agent`` is an iterable of length N (one scalar per
        agent — tensor or float). No GPU->CPU sync per call: the prior ``float(v.item())`` here
        was the single biggest per-step sync source (called ~dozens of times per gradient step
        from sac.py/ppo.py). write_tracking_data reduces by tag suffix at the interval boundary."""
        if not self.per_agent_scalar_stats:
            return
        for i in range(self.num_agents):
            self._accum_scalar(i, tag, values_per_agent[i])

    def _accum_scalar(self, i: int, tag: str, value) -> None:
        """Fold one already-reduced scalar ``value`` (0-dim tensor or python number) into agent
        ``i``'s on-device running (n, sum, min, max) for ``tag``. Allocation-free after the first
        call (accumulators zeroed/reset in write_tracking_data, never reallocated) and sync-free
        (no ``.item()``). Mirrors the old per-step ``per_agent_tracking[i][tag].append(...)``:
        folding the per-step reduced scalar with n+=1 makes the interval mean (sum/n) equal the
        old np.mean over the per-step list, and running min/max equal np.min/np.max over it."""
        if not self.per_agent_scalar_stats:
            return
        if torch.is_tensor(value):
            v = value.detach().to(torch.float64).reshape(())
        else:
            v = torch.as_tensor(float(value), dtype=torch.float64, device=self.device)
        acc = self.per_agent_scalar_stats[i].get(tag)
        if acc is None:
            acc = {
                "n":   torch.zeros((), dtype=torch.float64, device=self.device),
                "sum": torch.zeros((), dtype=torch.float64, device=self.device),
                "min": torch.full((), float("inf"), dtype=torch.float64, device=self.device),
                "max": torch.full((), float("-inf"), dtype=torch.float64, device=self.device),
            }
            self.per_agent_scalar_stats[i][tag] = acc
        acc["n"] += 1.0
        acc["sum"] += v
        torch.minimum(acc["min"], v, out=acc["min"])
        torch.maximum(acc["max"], v, out=acc["max"])

    def _accum_dist_stat(self, i: int, tag: str, vals: torch.Tensor) -> None:
        """Fold per-env samples ``vals`` into agent ``i``'s running (count, sum, sumsq) for
        ``tag`` — entirely on-device. The accumulator (three float64 device scalars) is created
        once per tag and reused (zeroed at flush), so there is NO per-step host allocation and NO
        GPU->CPU copy (the cause of the prior host-RAM leak). ``write_tracking_data`` turns these
        into the metric's mean + std."""
        if not self.per_agent_dist_stats:
            return
        v = vals.reshape(-1).to(torch.float64)
        # Drop non-finite samples: lets a publisher use NaN to mark "not applicable this step"
        # (e.g. engagement_quality's depth-conditioned fractions, NaN where depth isn't met) so the
        # running mean/std cover only the valid population; also guards existing (dist)/(stat) tags
        # against an inf sample (e.g. a singular-matrix condition number) poisoning the sum.
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return
        acc = self.per_agent_dist_stats[i].get(tag)
        if acc is None:
            acc = {
                "n":  torch.zeros((), dtype=torch.float64, device=self.device),
                "s":  torch.zeros((), dtype=torch.float64, device=self.device),
                "s2": torch.zeros((), dtype=torch.float64, device=self.device),
            }
            self.per_agent_dist_stats[i][tag] = acc
        acc["n"] += v.numel()
        acc["s"] += v.sum()
        acc["s2"] += torch.dot(v, v)

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
        # The base Agent.init() fires its OWN single shared wandb.init() (with
        # sync_tensorboard) for the shared writer we discard below. We publish a
        # per-agent wandb run ourselves, so hide experiment.wandb across the base
        # init and restore it afterward — the base never creates that stray run.
        _exp = getattr(self.cfg, "experiment", None)
        _wandb_flag = bool(getattr(_exp, "wandb", False)) if _exp is not None else False
        if _wandb_flag:
            _exp.wandb = False
        try:
            super().init(trainer_cfg=trainer_cfg)
        finally:
            if _wandb_flag:
                _exp.wandb = True
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
        # Backend is the single experiment.wandb bool (skrl's ExperimentCfg): when set,
        # each agent's SummaryWriter is wrapped in a MetricWriter that mirrors the same
        # scalars to that agent's own wandb run (write_tracking_data is unchanged — it
        # only calls add_scalar/flush, which the wrapper forwards to both backends).
        wandb_enabled = bool(getattr(getattr(self.cfg, "experiment", None), "wandb", False))
        # SAC_CFG.tensorboard (sibling of experiment): False => log only to wandb. MetricWriter
        # already no-ops every TB call when its tb_writer is None, so we just pass None.
        tb_enabled = bool(getattr(self.cfg, "tensorboard", True))
        if not tb_enabled and not wandb_enabled:
            print("[block_agent] tensorboard=false but experiment.wandb is off — keeping "
                  "TensorBoard so scalars aren't silently lost.", flush=True)
            tb_enabled = True
        if self.write_interval > 0:
            for i in range(self.num_agents):
                agent_log_dir = os.path.join(self.experiment_dir, str(i))
                os.makedirs(agent_log_dir, exist_ok=True)  # SummaryWriter normally makes this; ensure it exists when TB is off
                tb_writer = SummaryWriter(log_dir=agent_log_dir) if tb_enabled else None
                if wandb_enabled:
                    wandb_run = make_wandb_run(
                        agent_index=i,
                        num_agents=self.num_agents,
                        experiment_dir=self.experiment_dir,
                        log_dir=agent_log_dir,
                        cfg=self.cfg,
                    )
                    # The runner dumps the verbatim runtime config to this path
                    # AFTER init(); MetricWriter.close() attaches it to the run.
                    cfg_path = os.path.join(agent_log_dir, "config.yaml")
                    self.per_agent_writers.append(
                        MetricWriter(tb_writer, wandb_run, config_path=cfg_path)
                    )
                else:
                    self.per_agent_writers.append(tb_writer)
                self.per_agent_tracking.append(collections.defaultdict(list))
                self.per_agent_dist_stats.append({})
                self.per_agent_scalar_stats.append({})
                self._per_agent_track_rewards.append([])
                self._per_agent_track_timesteps.append([])
                self._per_agent_episodes_this_interval.append([])
                self._per_agent_episodes_this_interval_engaged.append([])
                self._per_agent_episodes_this_interval_ever.append([])
                self._per_agent_distances_this_interval.append([])
                self._per_agent_velocities_this_interval.append([])
                self._best_success_rate.append(-1.0)

        # Algorithm-specific memory tensors (obs/actions/... for SAC, on-policy
        # rollout tensors for PPO).
        self._create_memory_tensors()

        self._init_done = True

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Flush per-agent tracking buckets to per-agent writers."""
        # Per-interval memory snapshot (process-global; mirrored to every agent's writer under the
        # Stats family, next to "Stats / Algorithm update time"). Linear host_rss growth ⇒ a host
        # leak; growing gpu_alloc/reserved ⇒ a GPU leak. Cheap to compute once per flush.
        try:
            import psutil
            _host_rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        except Exception:
            _host_rss_gb = None
        if torch.cuda.is_available():
            _gpu_alloc_gb = torch.cuda.memory_allocated() / 1e9
            _gpu_reserved_gb = torch.cuda.memory_reserved() / 1e9
        else:
            _gpu_alloc_gb = _gpu_reserved_gb = None

        for i, writer in enumerate(self.per_agent_writers):
            if _host_rss_gb is not None:
                writer.add_scalar(tag="Stats / Host RSS (GB)", value=_host_rss_gb, timestep=timestep)
            if _gpu_alloc_gb is not None:
                writer.add_scalar(tag="Stats / GPU allocated (GB)", value=_gpu_alloc_gb, timestep=timestep)
                writer.add_scalar(tag="Stats / GPU reserved (GB)", value=_gpu_reserved_gb, timestep=timestep)
            for tag, values in self.per_agent_tracking[i].items():
                if not values:
                    continue
                if tag.endswith("(min)"):
                    writer.add_scalar(tag=tag, value=float(np.min(values)), timestep=timestep)
                elif tag.endswith("(max)"):
                    writer.add_scalar(tag=tag, value=float(np.max(values)), timestep=timestep)
                else:
                    writer.add_scalar(tag=tag, value=float(np.mean(values)), timestep=timestep)

            # On-device per-step/per-grad-step scalar accumulators (track_per_agent,
            # logs_rew, per_env_to_log scalars, phase-split). Reduce by tag suffix to match
            # the old per-interval np.max/np.min/np.mean over the Python list, with a single
            # GPU->CPU sync per tag HERE (at the flush) instead of one per step. Accumulators
            # are zeroed/reset in place for reuse — no reallocation.
            for tag, acc in self.per_agent_scalar_stats[i].items():
                if float(acc["n"].item()) == 0.0:
                    continue
                if tag.endswith("(max)"):
                    val = acc["max"]
                elif tag.endswith("(min)"):
                    val = acc["min"]
                else:
                    val = acc["sum"] / acc["n"]
                writer.add_scalar(tag=tag, value=float(val.item()), timestep=timestep)
                acc["n"].zero_(); acc["sum"].zero_()
                acc["min"].fill_(float("inf")); acc["max"].fill_(float("-inf"))

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

            # Engagement rate over trajectories that finished since the last flush:
            # fraction of episodes that were engaged (curr_engaged) on ANY step of the
            # rollout (see the per-env ever-engaged latch above), not just the terminal
            # step. Cleared after emit so the rate reflects only this-interval episodes.
            eng = self._per_agent_episodes_this_interval_engaged[i]
            if eng:
                interval_engagement_rate = float(np.mean(eng))
                writer.add_scalar(
                    tag="Episode / Engagement rate",
                    value=interval_engagement_rate,
                    timestep=timestep,
                )
                eng.clear()

            # Ever-success rate over trajectories that finished since the last
            # flush: fraction that reached success on at least one step (vs. the
            # terminal-step-only Success rate). Same source/lifecycle as above;
            # cleared after emit so it reflects only this-interval episodes.
            ever = self._per_agent_episodes_this_interval_ever[i]
            if ever:
                writer.add_scalar(
                    tag="Episode / Ever success rate",
                    value=float(np.mean(ever)),
                    timestep=timestep,
                )
                ever.clear()

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

            # "(dist)"/"(stat)" metrics: emit mean + std from the on-device running sufficient
            # statistics (count, sum, sumsq) accumulated this interval — no per-env samples are
            # retained, so there is no host-RAM churn. std = sqrt(max(sumsq/n - mean^2, 0)).
            # Both suffixes behave identically (histograms were removed); the suffix is only
            # stripped for the base tag, and the two scalars keep exactly one '/'. Accumulators
            # are zeroed IN PLACE (reused next interval) rather than reallocated.
            for tag, acc in self.per_agent_dist_stats[i].items():
                n = acc["n"]
                if n.item() == 0.0:
                    continue
                if tag.endswith(" (dist)"):
                    base = tag[: -len(" (dist)")]
                elif tag.endswith(" (stat)"):
                    base = tag[: -len(" (stat)")]
                else:
                    base = tag
                mean = acc["s"] / n
                var = (acc["s2"] / n) - mean * mean
                writer.add_scalar(tag=f"{base}_mean", value=float(mean.item()), timestep=timestep)
                writer.add_scalar(tag=f"{base}_std",
                                  value=float(var.clamp_min(0.0).sqrt().item()), timestep=timestep)
                acc["n"].zero_(); acc["s"].zero_(); acc["s2"].zero_()

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
                            self._accum_scalar(i, f"Episode_Reward/{term}", vals.mean())

            # Per-agent contact-quality metrics (surface task): a wrapper publishes per-env
            # per-EPISODE values in infos["per_env_contact_quality"] + a done mask; we mean over the
            # finishing envs in each agent's slice -> contact_quality/<metric> (one value per
            # completed rollout, so it reads as "average over rollouts" like the success rate).
            if (
                isinstance(infos, dict)
                and "per_env_contact_quality" in infos
                and "per_env_contact_quality_mask" in infos
            ):
                cq = infos["per_env_contact_quality"]
                cqmask = infos["per_env_contact_quality_mask"]
                if torch.is_tensor(cqmask) and cqmask.any():
                    for metric, per_env_vals in cq.items():
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_mask = cqmask[env_lo:env_hi]
                            if not agent_mask.any():
                                continue
                            vals = per_env_vals[env_lo:env_hi][agent_mask]
                            # Drop NaN: metrics that are only defined for rollouts that touched
                            # (steps_to_first_contact, post_contact_percentage) are NaN otherwise,
                            # so this yields a conditional "over rollouts that made contact" average.
                            vals = vals[torch.isfinite(vals)]
                            if vals.numel() == 0:
                                continue
                            self._accum_scalar(i, f"contact_quality/{metric}", vals.mean())

            # Per-agent drag-performance (surface task): per-env per-EPISODE rollout stats. For each
            # key emit BOTH the mean and the std over the finishing envs -> drag_performance/{key}_mean
            # and {key}_std. So a rollout-mean key X gives X_mean (avg over rollouts, the headline) and
            # X_std (spread of rollout means); a rollout-std key X_intra_std gives X_intra_std_mean (the
            # AVERAGE within-rollout std). NaN (no-contact rollouts) skipped.
            if (
                isinstance(infos, dict)
                and "per_env_drag" in infos
                and "per_env_drag_mask" in infos
            ):
                dmask = infos["per_env_drag_mask"]
                if torch.is_tensor(dmask) and dmask.any():
                    for metric, per_env_vals in infos["per_env_drag"].items():
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_mask = dmask[env_lo:env_hi]
                            if not agent_mask.any():
                                continue
                            vals = per_env_vals[env_lo:env_hi][agent_mask]
                            vals = vals[torch.isfinite(vals)]
                            if vals.numel() == 0:
                                continue
                            self._accum_scalar(i, f"drag_performance/{metric}_mean", vals.mean())
                            if vals.numel() > 1:
                                self._accum_scalar(i, f"drag_performance/{metric}_std", vals.std())
                            # Frontier: furthest any single rollout dragged (peak over the interval's
                            # finishing envs). The " (max)" suffix makes write_tracking_data reduce by
                            # max across the interval, so this is the true best rollout, not an average.
                            if metric == "keypoints_met":
                                self._accum_scalar(i, "drag_performance/keypoints_met (max)", vals.max())

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
                            self._accum_scalar(i, f"logs_rew/{term}", agent_vals.mean())

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
                    # Optional per-phase metric split (free_space / search / insertion).
                    # Families listed in cfg.phase_split_families have each of their
                    # `{family}/{name}` tags re-emitted as `{family}_{phase}/{name}`,
                    # reduced over only this step's in-phase envs (same max/min/mean/dist
                    # convention as the un-split path). `phase_ids` is computed lazily on
                    # the first matching tag — only then are the contact + engagement
                    # inputs required. See _compute_phase_ids / _track_phase_split.
                    phase_split = set(getattr(self.cfg, "phase_split_families", None) or ())
                    phase_ids: torch.Tensor | None = None
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
                        # Phase split takes precedence over the un-split reduction: a
                        # matched family's tag is replaced by its three per-phase tags
                        # (never emitted un-split). Family = text before the single '/'.
                        if phase_split and tag.partition("/")[0].strip() in phase_split:
                            if phase_ids is None:
                                phase_ids = self._compute_phase_ids(infos, total_envs)
                            self._track_phase_split(tag, vals, phase_ids, epa)
                            continue
                        # A "(dist)"/"(stat)" tag wants mean + std over the interval's full per-env
                        # distribution: fold each agent's per-env slice into its on-device running
                        # stats (no per-step copy / no Python lists). Handled here, before the
                        # scalar-reduction path below, so it never lands in per_agent_tracking.
                        if tag.endswith("(dist)") or tag.endswith("(stat)"):
                            for i in range(self.num_agents):
                                env_lo, env_hi = i * epa, (i + 1) * epa
                                self._accum_dist_stat(i, tag, vals[env_lo:env_hi])
                            continue
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
                            self._accum_scalar(i, tag, red)

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

                # Engagement rate (env-driven): a trajectory counts as engaged if the peg was
                # engaged on ANY step of the rollout, not just the terminal step. We OR the single
                # per_env_curr_engaged info signal (the one source of truth for "engaged") into a
                # per-env latch each step, record the latch for finished trajectories, then clear
                # those envs' latch for the next episode. Same per-agent partitioning as success;
                # mirrors the Ever success rate semantics so transient engagement isn't missed.
                curr_eng = infos.get("per_env_curr_engaged")
                if curr_eng is not None:
                    if not (torch.is_tensor(curr_eng) and curr_eng.dim() > 0
                            and curr_eng.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_curr_engaged has shape "
                            f"{tuple(curr_eng.shape) if torch.is_tensor(curr_eng) else type(curr_eng).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    eng_flat = curr_eng.bool().view(-1).to(self.device)
                    if (self._ever_engaged_latch is None
                            or self._ever_engaged_latch.shape[0] != total_envs):
                        self._ever_engaged_latch = torch.zeros(
                            total_envs, dtype=torch.bool, device=self.device
                        )
                    # OR in THIS step's engagement (incl. the terminal step) before recording.
                    self._ever_engaged_latch |= eng_flat
                    done_flat = (terminated + truncated).bool().view(-1).to(self.device)
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        done_slice = done_flat[env_lo:env_hi]
                        if done_slice.any():
                            agent_eng = self._ever_engaged_latch[env_lo:env_hi]
                            done_idx = done_slice.nonzero(as_tuple=False).view(-1)
                            self._per_agent_episodes_this_interval_engaged[i].extend(
                                agent_eng[done_idx].long().tolist()
                            )
                    # Clear the latch for finished trajectories so the next episode starts fresh.
                    self._ever_engaged_latch[done_flat] = False

                # Ever-success rate (env-driven): identical done-gated, per-agent
                # partitioning as success/engagement, sourced from the pre-reset
                # ep_succeeded latch (peg reached the success state on at least one
                # step of the episode). Records each finished trajectory's latch so
                # write_tracking_data's interval mean is a true per-trajectory
                # ever-success rate; its gap to Episode / Success rate is unstable
                # (transient) insertions not held to the terminal step.
                curr_ever = infos.get("per_env_ever_success")
                if curr_ever is not None:
                    if not (torch.is_tensor(curr_ever) and curr_ever.dim() > 0
                            and curr_ever.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_ever_success has shape "
                            f"{tuple(curr_ever.shape) if torch.is_tensor(curr_ever) else type(curr_ever).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    done_flat = (terminated + truncated).bool().view(-1)
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        done_slice = done_flat[env_lo:env_hi]
                        if done_slice.any():
                            agent_ever = curr_ever[env_lo:env_hi]
                            done_idx = done_slice.nonzero(as_tuple=False).view(-1)
                            self._per_agent_episodes_this_interval_ever[i].extend(
                                agent_ever[done_idx].long().tolist()
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
                            scalar = v.reshape(())  # keep on-device; no per-step .item()
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
                        # On-device fold (same destination as the per-agent decomposition
                        # so a tag like Episode_Reward/<term> never lands in BOTH the list
                        # and the scalar accumulator -> no double-emit). Mean-of-per-step,
                        # matching the old np.mean over the appended list.
                        for i in range(self.num_agents):
                            self._accum_scalar(i, tag, scalar)

    # --------------------------------------------------------------
    # Per-phase metric splitting (cfg.phase_split_families)
    # --------------------------------------------------------------
    def _compute_phase_ids(self, infos: Any, total_envs: int) -> torch.Tensor:
        """Per-env insertion-task phase id (``(total_envs,)`` long) for metric splitting.

        * ``free_space`` (0): not engaged and no contact on any axis.
        * ``search``     (1): in contact on some axis but not engaged.
        * ``insertion``  (2): engaged (contact state irrelevant).

        Sourced from the same per-step signals the episode-rate metrics use:
        ``infos["per_env_curr_engaged"]`` (peg close to socket, thresholded at 0.5)
        and ``infos["in_contact"]`` (per-axis contact bool from the contact sensor).
        Raised loud if either is missing/misshaped — a family was configured for phase
        splitting without the wrappers that publish these, which would otherwise
        silently mislabel every step as free_space.
        """
        curr_eng = infos.get("per_env_curr_engaged") if isinstance(infos, dict) else None
        in_contact = infos.get("in_contact") if isinstance(infos, dict) else None
        if not (torch.is_tensor(curr_eng) and curr_eng.dim() > 0 and curr_eng.shape[0] == total_envs):
            raise ValueError(
                "phase_split_families is set but infos['per_env_curr_engaged'] is "
                f"{tuple(curr_eng.shape) if torch.is_tensor(curr_eng) else type(curr_eng).__name__}, "
                f"expected a ({total_envs},) tensor (is the Forge/Factory scorer wrapper active?)."
            )
        if not (torch.is_tensor(in_contact) and in_contact.dim() > 0 and in_contact.shape[0] == total_envs):
            raise ValueError(
                "phase_split_families is set but infos['in_contact'] is "
                f"{tuple(in_contact.shape) if torch.is_tensor(in_contact) else type(in_contact).__name__}, "
                f"expected a ({total_envs}, ...) tensor (is the contact-sensor wrapper active?)."
            )
        engaged = curr_eng.reshape(total_envs) > 0.5
        contact_any = (in_contact.reshape(total_envs, -1) != 0).any(dim=1)
        phase = torch.zeros(total_envs, dtype=torch.long, device=engaged.device)
        phase[contact_any] = 1          # search (overwritten by insertion below)
        phase[engaged] = 2              # insertion wins regardless of contact
        return phase

    def _track_phase_split(
        self, tag: str, vals: torch.Tensor, phase_ids: torch.Tensor, epa: int
    ) -> None:
        """Emit a ``{family}/{name}`` per-env metric as three ``{family}_{phase}/{name}``
        tags, each reduced over only this step's in-phase envs in the agent's partition.

        Reduction matches the un-split convention picked from the tag suffix: ``(dist)``/``(stat)``
        fold the per-phase per-env samples into on-device running stats (mean+std at flush),
        ``(max)``/``(min)`` take the per-phase peak/trough, everything else env-means. A phase with
        no envs this step contributes nothing, so its interval value reflects only the steps in
        which it was active. New tags keep exactly one '/' (family carries none, and the
        single-slash tag convention guarantees ``name`` carries none either). The ``(dist)``/
        ``(stat)`` suffix is preserved on ``new_tag`` so write_tracking_data strips it for the
        base name."""
        family, _, name = tag.partition("/")
        family = family.strip()
        name = name.strip()
        is_dist = tag.endswith("(dist)")
        is_stat = tag.endswith("(stat)")
        is_max = tag.endswith("(max)")
        is_min = tag.endswith("(min)")
        for i in range(self.num_agents):
            env_lo, env_hi = i * epa, (i + 1) * epa
            sl = vals[env_lo:env_hi].float()
            ph = phase_ids[env_lo:env_hi]
            for pid, pname in enumerate(_PHASE_NAMES):
                mask = ph == pid
                if not mask.any():
                    continue
                phase_vals = sl[mask]
                new_tag = f"{family}_{pname}/{name}"
                if is_dist or is_stat:
                    self._accum_dist_stat(i, new_tag, phase_vals)
                elif is_max:
                    self._accum_scalar(i, new_tag, phase_vals.max())
                elif is_min:
                    self._accum_scalar(i, new_tag, phase_vals.min())
                else:
                    self._accum_scalar(i, new_tag, phase_vals.mean())

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
