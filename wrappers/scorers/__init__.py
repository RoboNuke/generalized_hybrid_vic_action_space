"""Per-environment "scorer" wrappers.

Each scorer wraps an unwrapped Isaac Lab env for one task family (Forge, Factory,
Lift, Ant, ...) and is responsible for that env's logging + success metrics:
injecting ``info["is_success"]`` and publishing the per-env reward / success
tensors (``per_env_rew``, ``per_env_curr_successes``, ``per_env_ep_success_times``,
...) that SAC partitions per agent and logs.

:class:`~wrappers.scorers.reward_decomposition.RewardDecompositionWrapper` is the
shared base; the task-specific scorers subclass it. They are registered by short
name in :mod:`wrappers` (see ``wrappers/__init__.py``).
"""
