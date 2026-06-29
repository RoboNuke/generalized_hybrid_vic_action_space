"""Per-agent metric sink: TensorBoard always, Weights & Biases optionally.

``BlockAgent`` publishes every scalar through a per-agent
``skrl.utils.tensorboard.SummaryWriter`` (``write_tracking_data``). To mirror
those same scalars to Weights & Biases WITHOUT touching the ~30 ``add_scalar``
call sites, we wrap each per-agent writer in :class:`MetricWriter`, a drop-in
that exposes the exact ``add_scalar(tag=, value=, timestep=)`` / ``flush`` /
``close`` surface ``write_tracking_data`` (and the runner's shutdown loop) use.

Backend selection is the single ``experiment.wandb`` bool already present on
skrl's ``ExperimentCfg`` — no new config field. When it is True, each of the
``num_agents`` block-parallel agents gets its OWN wandb run (grouped under the
experiment name), faithfully mirroring today's N overlaid TensorBoard runs.

wandb semantics handled here:
  * One concurrent run per agent via ``wandb.init(reinit="create_new")`` — each
    returns an independent ``Run`` we ``.log()`` to explicitly (no global
    ``wandb.run`` collision between agents in the same process).
  * Scalars added within one ``write_tracking_data`` interval share a timestep;
    we BUFFER them and emit a single ``run.log(dict, step=timestep)`` on
    ``flush()`` — one network round-trip per agent per interval, and wandb's
    monotonic-step rule is satisfied because timesteps increase across intervals.
  * ``close()`` calls ``run.finish()``, which blocks until the run is synced.
    The runner's shutdown loop already calls ``flush(); close()`` on every
    per-agent writer BEFORE ``os._exit(0)`` (which would otherwise kill wandb's
    background sync), so no runner change is needed — same hazard/fix as the
    synchronous TensorBoard ``flush()``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


class MetricWriter:
    """Drop-in for skrl's per-agent ``SummaryWriter`` that also mirrors to wandb.

    Mirrors only the methods ``BlockAgent``/runner use on a per-agent writer:
    ``add_scalar``, ``flush``, ``close``. TensorBoard writes pass through
    immediately; wandb scalars are buffered per interval and committed on
    ``flush()`` (see module docstring).
    """

    def __init__(self, tb_writer, wandb_run=None) -> None:
        self._tb = tb_writer
        self._wandb_run = wandb_run
        self._pending: dict[str, float] = {}
        self._pending_step: int | None = None

    def add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        if self._tb is not None:
            self._tb.add_scalar(tag=tag, value=value, timestep=timestep)
        if self._wandb_run is not None:
            # wandb merges multiple log() calls at the same step; we instead batch
            # the whole interval into one dict to keep it to a single round-trip.
            self._pending[tag] = value
            self._pending_step = timestep

    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()
        if self._wandb_run is not None and self._pending:
            self._wandb_run.log(self._pending, step=self._pending_step)
            self._pending = {}
            self._pending_step = None

    def close(self) -> None:
        # Commit any scalars buffered since the last flush before finishing.
        self.flush()
        if self._tb is not None:
            self._tb.close()
        if self._wandb_run is not None:
            # Blocks until the run's data is synced (online) — must happen before
            # the runner's os._exit(0) or the tail of the run is lost.
            self._wandb_run.finish()
            self._wandb_run = None


def make_wandb_run(
    *,
    agent_index: int,
    num_agents: int,
    experiment_dir: str,
    log_dir: str,
    cfg: Any,
):
    """Create one wandb run for agent ``agent_index``.

    Everything is derived from the existing ``experiment`` config so enabling
    wandb needs only ``experiment.wandb: true`` in YAML:

      * project = basename of the experiment "family" dir (the parent of
        ``experiment_dir``, e.g. ``forge_pih``)
      * group   = experiment_name (basename of ``experiment_dir``) — so the N
        agents of one run cluster together
      * name    = ``<experiment_name>_agent<i>``

    Any of these (plus ``entity``, ``tags``, ``mode``, ``id``, ...) can be
    overridden via ``experiment.wandb_kwargs``. Returns a wandb ``Run``.
    """
    try:
        import wandb
    except ImportError as e:  # explicit opt-in => fail loudly, not silently
        raise RuntimeError(
            "experiment.wandb=True but the 'wandb' package is not installed. "
            "Install it (pip install wandb) or set experiment.wandb=False."
        ) from e

    experiment_name = os.path.basename(experiment_dir.rstrip("/")) or "experiment"
    family = os.path.basename(os.path.dirname(experiment_dir.rstrip("/"))) or "skrl"

    experiment_cfg = getattr(cfg, "experiment", None)
    user_kwargs = {}
    if experiment_cfg is not None:
        user_kwargs = dict(getattr(experiment_cfg, "wandb_kwargs", {}) or {})

    # Best-effort config dump for the run's overview/sweep panes. asdict never
    # raises on Callable/class fields (deepcopy returns them as-is); wandb str()s
    # anything non-JSON. Fall back to just the experiment block if asdict trips.
    try:
        run_config = dataclasses.asdict(cfg)
    except Exception:
        run_config = {}
        if experiment_cfg is not None:
            try:
                run_config = dataclasses.asdict(experiment_cfg)
            except Exception:
                run_config = {}
    run_config.update(agent_index=agent_index, num_agents=num_agents)

    kwargs: dict[str, Any] = dict(user_kwargs)
    kwargs.setdefault("project", family)
    kwargs.setdefault("group", experiment_name)
    kwargs.setdefault("name", f"{experiment_name}_agent{agent_index}")
    kwargs.setdefault("dir", log_dir)
    # Independent concurrent run per agent in one process (vs. the default, which
    # would finish the previous agent's run on each init()).
    kwargs.setdefault("reinit", "create_new")
    config = dict(kwargs.pop("config", {}) or {})
    config.update(run_config)
    kwargs["config"] = config

    return wandb.init(**kwargs)
