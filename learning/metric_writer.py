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

    ``config_path`` is the per-agent ``config.yaml`` the runner dumps (the exact
    merged + CLI-applied runtime config used to reconstruct a run). It does not
    exist yet at construction time — the runner dumps it after ``agent.init()`` —
    so we attach it to the run at ``close()`` (via ``run.save(policy="now")``),
    by which point the file is guaranteed on disk. This uploads the verbatim
    file to the run's Files tab as ``runtime_config.yaml`` (NOT as an artifact,
    and NOT named ``config.yaml`` — wandb reserves that for its own config),
    complementing the structured ``wandb.config`` dict set at init.
    """

    def __init__(self, tb_writer, wandb_run=None, config_path: str | None = None) -> None:
        self._tb = tb_writer
        self._wandb_run = wandb_run
        self._config_path = config_path
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
            # Attach the verbatim runtime config (now dumped on disk) to the run's
            # Files before finishing. Saved as "runtime_config.yaml" (NOT
            # "config.yaml": wandb reserves that name for its own serialized
            # wandb.config, so reusing it would collide). Copy into wandb's own
            # files dir and save with policy="now" so it uploads immediately and
            # leaves no duplicate in the experiment tree.
            if self._config_path and os.path.isfile(self._config_path):
                try:
                    import shutil
                    files_dir = self._wandb_run.settings.files_dir
                    dst = os.path.join(files_dir, "runtime_config.yaml")
                    shutil.copyfile(self._config_path, dst)
                    self._wandb_run.save(dst, base_path=files_dir, policy="now")
                except Exception as e:  # never let a logging extra abort shutdown
                    print(f"[metric_writer] wandb runtime_config.yaml save failed: {e!r}",
                          flush=True)
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
      * group   = experiment_name with any ``.yaml``/``.yml`` extension stripped
        (so e.g. ``1_fixed.yaml`` -> ``1_fixed``) — the N agents cluster together
      * name    = ``<stripped_experiment_name>_agent<i>``

    Note: only the wandb-facing names drop the extension; ``experiment_dir`` (the
    on-disk tfevents/checkpoint layout) keeps the full ``experiment_name``.

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
    # Strip a YAML extension from the wandb-facing names only (configs are commonly
    # named e.g. "1_fixed.yaml"); the on-disk experiment_dir keeps the full name.
    wandb_name = experiment_name
    for _ext in (".yaml", ".yml"):
        if wandb_name.lower().endswith(_ext):
            wandb_name = wandb_name[: -len(_ext)]
            break

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
    kwargs.setdefault("group", wandb_name)
    kwargs.setdefault("name", f"{wandb_name}_agent{agent_index}")
    kwargs.setdefault("dir", log_dir)
    # Independent concurrent run per agent in one process (vs. the default, which
    # would finish the previous agent's run on each init()).
    kwargs.setdefault("reinit", "create_new")
    config = dict(kwargs.pop("config", {}) or {})
    config.update(run_config)
    kwargs["config"] = config

    return wandb.init(**kwargs)
