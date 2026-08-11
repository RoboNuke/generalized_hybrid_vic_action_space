#!/usr/bin/env python
"""Per-step evaluation trace recorder.

Captures a full per-step / per-env record of an eval (or record) rollout and
dumps it to a single parquet file. The point is to stop creating throwaway eval
*runs* in wandb and instead attach the raw per-step data as a file to the
ORIGINAL training run — from which any aggregated metric (mean / max / std /
conditional rate) can be re-derived offline.

Design: it taps the SAME data the metric system consumes, one step upstream of
the reduction, so new metrics are captured automatically. Every per-env signal
the agent logs arrives as a ``(num_envs,)`` tensor in a handful of ``infos``
channels (chiefly ``infos["per_env_to_log"]``, forwarded verbatim by the
reward-decomposition wrapper). :meth:`StepTraceRecorder.capture` iterates those
channels by KEY rather than naming metrics, so adding a metric to ``to_log``
needs no change here.

What is stored, per (env, step):
  * ``reward``, ``terminated``, ``truncated``  (raw, pre-shaping reward)
  * ``obs_0..obs_k``  (always) and ``act_0..act_m``  (expanded columns)
  * one column per per-env metric signal found in ``infos`` (verbatim tag names),
    plus the masks/flags that gate the conditional metrics, so the exact
    reductions can be reproduced downstream.

Per-episode channels (reward decomposition, contact/drag quality) only publish
on reset steps; they are NaN on the steps where they are absent. Columns that
first appear late are back-filled with NaN so the frame stays rectangular.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # torch is always present in the runtime; import lazily-safe for tooling
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# infos dict-of-tensors channels: {channel_key: column_prefix}. Each child value
# is a per-env tensor; trailing dims are mean-collapsed to (num_envs,) to match
# how the metric system forwards them.
_DICT_CHANNELS = {
    "per_env_to_log": "",              # keys already carry the full tag verbatim
    "per_env_rew": "Episode_Reward/",
    "per_env_contact_quality": "contact_quality/",
    "per_env_drag": "per_env_drag/",
}

# flat per-env tensors in infos worth capturing by name (masks + success/engage
# flags that gate the conditional reductions).
_FLAT_KEYS = (
    "per_env_rew_mask",
    "per_env_contact_quality_mask",
    "per_env_drag_mask",
    "per_env_curr_successes",
    "per_env_curr_engaged",
    "per_env_ever_success",
    "is_success",
)

# never sweep these generically (they are the bulk transition tensors or are
# already captured above).
_SKIP_GENERIC = {
    "observations", "actions", "states", "next_observations", "next_states",
    "rewards", "terminated", "truncated", "per_env_trace",
    *_DICT_CHANNELS.keys(), *_FLAT_KEYS,
}


def _to_np(x: Any) -> np.ndarray | None:
    """Detach a tensor/array to a CPU float32 numpy array; None passes through."""
    if x is None:
        return None
    if torch is not None and torch.is_tensor(x):
        return x.detach().to("cpu", torch.float32).numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return None


def _per_env_1d(x: Any, num_envs: int) -> np.ndarray | None:
    """Coerce to a (num_envs,) float array, mean-collapsing trailing dims. Returns
    None if the value isn't a per-env tensor (first dim != num_envs)."""
    a = _to_np(x)
    if a is None or a.ndim == 0 or a.shape[0] != num_envs:
        return None
    if a.ndim > 1:
        a = a.reshape(a.shape[0], -1).mean(axis=1)
    return a.reshape(-1)


class StepTraceRecorder:
    """Accumulates per-step / per-env signals and flushes them to one parquet.

    ``num_envs`` is the TOTAL env count this process drives (all agents). In the
    from-wandb eval/record pipeline each run is a single agent, so this is just
    that agent's env slice.
    """

    def __init__(self, out_path: str, num_envs: int, *, include_obs: bool = True) -> None:
        self.out_path = os.path.abspath(out_path)
        self.num_envs = int(num_envs)
        self.include_obs = bool(include_obs)
        self._obs: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._scalar_steps: list[dict[str, np.ndarray]] = []
        self._cols: set[str] = set()
        self.n_steps = 0

    # ------------------------------------------------------------------
    def capture(
        self,
        *,
        observations: Any,
        actions: Any,
        rewards: Any,
        terminated: Any,
        truncated: Any,
        infos: Any,
        states: Any = None,
    ) -> None:
        """Record one env-step. Cheap: everything is copied to CPU numpy here so no
        GPU tensors (and no autograd graph) are retained across steps."""
        n = self.num_envs
        rec: dict[str, np.ndarray] = {}

        r = _per_env_1d(rewards, n)
        if r is not None:
            rec["reward"] = r
        t = _per_env_1d(terminated, n)
        if t is not None:
            rec["terminated"] = t
        tr = _per_env_1d(truncated, n)
        if tr is not None:
            rec["truncated"] = tr

        if isinstance(infos, dict):
            # Named multi-dim pose/vector channel: expand each (num_envs, K) into
            # `<name>_0..K-1` columns (NOT mean-collapsed, unlike the metric channels).
            pt = infos.get("per_env_trace")
            if isinstance(pt, dict):
                for name, vals in pt.items():
                    a = _to_np(vals)
                    if a is None or a.ndim == 0 or a.shape[0] != n:
                        continue
                    a = a.reshape(n, -1)
                    if a.shape[1] == 1:
                        rec[name] = a[:, 0]           # scalar per-env (e.g. keypoint counts)
                    else:
                        for j in range(a.shape[1]):   # vector per-env (pose / velocity)
                            rec[f"{name}_{j}"] = a[:, j]
            for chan, prefix in _DICT_CHANNELS.items():
                sub = infos.get(chan)
                if isinstance(sub, dict):
                    for tag, vals in sub.items():
                        col = _per_env_1d(vals, n)
                        if col is not None:
                            rec[f"{prefix}{tag}"] = col
            for key in _FLAT_KEYS:
                col = _per_env_1d(infos.get(key), n)
                if col is not None:
                    rec[key] = col
            # generic sweep: any other per-env (num_envs,) tensor -> info/<key>.
            # keeps future signals captured without editing this file.
            for key, val in infos.items():
                if key in _SKIP_GENERIC or not (torch is not None and torch.is_tensor(val)):
                    continue
                if val.dim() == 1 and val.shape[0] == n:
                    col = _per_env_1d(val, n)
                    if col is not None:
                        rec[f"info/{key}"] = col

        self._cols.update(rec.keys())
        self._scalar_steps.append(rec)
        if self.include_obs:
            o = _to_np(observations)
            self._obs.append(o.reshape(n, -1) if o is not None else np.empty((n, 0), np.float32))
            # Critic input (asymmetric AC): the full privileged state vector. None on a
            # symmetric env -> empty (no state_* columns emitted).
            s = _to_np(states)
            self._state.append(s.reshape(n, -1) if s is not None else np.empty((n, 0), np.float32))
        a = _to_np(actions)
        self._act.append(a.reshape(n, -1) if a is not None else np.empty((n, 0), np.float32))
        self.n_steps += 1

    # ------------------------------------------------------------------
    def flush(self) -> str | None:
        """Write the accumulated trace to ``out_path`` as parquet. Returns the path,
        or None if nothing was captured."""
        if self.n_steps == 0:
            print("[step_trace] nothing captured; no file written.", flush=True)
            return None
        import pandas as pd

        n, S = self.num_envs, self.n_steps
        rows = n * S
        # index columns: env fastest-varying within a step (row = step*n + env).
        step_col = np.repeat(np.arange(S, dtype=np.int32), n)
        env_col = np.tile(np.arange(n, dtype=np.int32), S)
        data: dict[str, np.ndarray] = {"step": step_col, "env": env_col}

        # metric columns, rectangular with NaN back-fill for absent steps.
        for col in sorted(self._cols):
            buf = np.full((S, n), np.nan, dtype=np.float32)
            for si, rec in enumerate(self._scalar_steps):
                v = rec.get(col)
                if v is not None:
                    buf[si] = v
            data[col] = buf.reshape(rows)
        # terminated/truncated read cleaner as small ints (still NaN-free here).
        for flag in ("terminated", "truncated"):
            if flag in data:
                data[flag] = np.nan_to_num(data[flag]).astype(np.uint8)

        def _stack_wide(chunks: list[np.ndarray], prefix: str) -> None:
            if not chunks:
                return
            width = max(c.shape[1] for c in chunks)
            if width == 0:
                return
            arr = np.full((S, n, width), np.nan, dtype=np.float32)
            for si, c in enumerate(chunks):
                if c.shape[1]:
                    arr[si, :, : c.shape[1]] = c
            arr = arr.reshape(rows, width)
            for j in range(width):
                data[f"{prefix}{j}"] = arr[:, j]

        if self.include_obs:
            _stack_wide(self._obs, "obs_")
            _stack_wide(self._state, "state_")
        _stack_wide(self._act, "act_")

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        try:
            df.to_parquet(self.out_path, index=False)
        except Exception as e:
            raise RuntimeError(
                f"[step_trace] failed to write parquet to {self.out_path} ({e!r}); "
                "is pyarrow installed?"
            ) from e
        print(
            f"[step_trace] wrote {rows} rows x {len(df.columns)} cols "
            f"({S} steps x {n} envs) -> {self.out_path}",
            flush=True,
        )
        return self.out_path
