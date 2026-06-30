"""Per-env reward decomposition + success flag for the surface path-following task.

Identical machinery to :class:`FactoryWrapper` (hooks ``_log_factory_metrics`` on
the unwrapped env, publishes ``info["is_success"]`` / ``per_env_rew`` /
``per_env_to_log``), but with the surface-follow reward-term SCALES read from the
task cfg instead of the Factory keypoint scales. ``FlatSurfaceFollowEnv`` reuses the
inherited ``_log_factory_metrics``, so the parent hook attaches unchanged.

The reward terms themselves are not implemented yet (structural pass) — the
scales below name the terms a later reward pass will emit so per-agent
``Episode_Reward/<term>`` logging lights up automatically once they exist.
"""

from __future__ import annotations

from wrappers.scorers.factory import FactoryWrapper


class FlatSurfaceFollowWrapper(FactoryWrapper):
    """Factory-style scorer with surface-follow reward scales."""

    def _factory_scales(self) -> dict[str, float]:
        # Weights for the per-term reward decomposition logging — must match the rew_scales in
        # FlatSurfaceFollowEnv._get_rewards. Extend as new reward terms are added.
        cfg = self._unwrapped.cfg_task
        return {
            "force": float(cfg.force_weight),
            "orientation": float(cfg.orientation_weight),
            "straightness": float(cfg.straightness_weight),
            "pace": float(cfg.pace_weight),
            "action_penalty_ee": -float(cfg.action_penalty_ee_scale),
            "action_grad_penalty": -float(cfg.action_grad_penalty_scale),
        }
