"""Auxiliary (additional) losses for the SAC trainer.

This module is intentionally self-contained and knows nothing about the SAC
internals beyond a small read-only ``LossContext``. SAC receives a single opaque
:class:`AuxLossManager` at construction and, at each gradient step, asks it for
the weighted total to add to the policy or critic loss before that optimizer's
backward. SAC never imports loss names, the registry, or the config dataclass.

Adding a new loss:

1. Subclass :class:`AuxLoss`, set ``name`` and ``supported_targets``, implement
   ``compute``, and decorate the class with :func:`register_loss`.
2. Add the three matching fields to ``configs/manager/loss_cfg.py``:
   ``<name>_enabled``, ``<name>_policy_weight``, ``<name>_critic_weight``.

The field-name prefix must equal the loss's ``name`` — :meth:`AuxLossManager.from_cfg`
reads the config by that convention.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid a circular import (agents import nothing from here)
    from learning.block_agent import BlockAgent
    from configs.manager.loss_cfg import LossCfg


TARGETS = ("policy", "critic")
"""The optimizers an auxiliary loss may feed."""


@dataclasses.dataclass
class LossContext:
    """Read-only bundle of everything a loss may need, populated at the call site.

    SAC builds this twice per gradient step — once in the critic block
    (``target="critic"``) and once in the policy block (``target="policy"``) — so
    only the fields relevant to the current target are guaranteed to be set. A
    loss should only touch fields valid for the target(s) it declares support for.
    """

    agent: "BlockAgent"
    target: str

    # Raw sampled minibatch tensors (observations, actions, rewards,
    # next_observations, terminated, and states/next_states when asymmetric).
    sampled: dict = dataclasses.field(default_factory=dict)

    # ---- policy-block fields (set when target == "policy") ----
    actions: torch.Tensor | None = None
    log_prob: torch.Tensor | None = None
    policy_outputs: dict | None = None
    inputs: dict | None = None

    # ---- critic-block fields (set when target == "critic") ----
    critic_1_values: torch.Tensor | None = None
    critic_2_values: torch.Tensor | None = None
    target_values: torch.Tensor | None = None
    critic_inputs: dict | None = None


class AuxLoss:
    """Base class for an auxiliary loss.

    Subclasses set ``name`` (matching the ``LossCfg`` field prefix) and
    ``supported_targets`` (the subset of :data:`TARGETS` the loss can validly
    feed), and implement :meth:`compute`. Losses are constructed with no
    arguments.
    """

    name: str = ""
    supported_targets: tuple[str, ...] = ()

    def compute(self, ctx: LossContext) -> torch.Tensor:
        """Return the **per-agent** raw (unweighted) loss as an ``(N,)`` tensor.

        Element ``i`` is agent ``i``'s loss (``N == ctx.agent.num_agents``). The
        manager reduces it with ``.mean()`` (× the configured weight) for the
        gradient and logs the per-agent values unweighted, so SAC's per-agent
        tensorboard tracking stays meaningful. Reshape flat ``(N*B, ...)`` batch
        tensors with ``.view(ctx.agent.num_agents, -1)`` before reducing.
        """
        raise NotImplementedError


LOSS_REGISTRY: dict[str, type[AuxLoss]] = {}
"""Registry of available auxiliary losses, keyed by ``AuxLoss.name``."""


def register_loss(cls: type[AuxLoss]) -> type[AuxLoss]:
    """Class decorator: register an :class:`AuxLoss` subclass by its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    if cls.name in LOSS_REGISTRY:
        raise ValueError(
            f"Duplicate loss name {cls.name!r} "
            f"({cls.__name__} vs {LOSS_REGISTRY[cls.name].__name__})"
        )
    unknown = set(cls.supported_targets) - set(TARGETS)
    if unknown or not cls.supported_targets:
        raise ValueError(
            f"{cls.__name__}.supported_targets must be a non-empty subset of "
            f"{TARGETS}, got {cls.supported_targets!r}"
        )
    LOSS_REGISTRY[cls.name] = cls
    return cls


@register_loss
class ActionL2Loss(AuxLoss):
    """Mean squared action magnitude — a simple policy-side action regularizer.

    Penalizes large actions, nudging the policy toward smaller control outputs.
    Policy-only: it differentiates through the freshly sampled ``ctx.actions``,
    which is a policy-block quantity.
    """

    name = "action_l2"
    supported_targets = ("policy",)

    def compute(self, ctx: LossContext) -> torch.Tensor:
        # actions: (N*B, A) -> per-agent mean squared magnitude (N,).
        return ctx.actions.pow(2).view(ctx.agent.num_agents, -1).mean(dim=1)


@dataclasses.dataclass
class _LossEntry:
    """One enabled loss plus its resolved per-target weights (internal to the
    manager). Built once in :meth:`AuxLossManager.from_cfg`; ``loss`` is the
    instantiated :class:`AuxLoss`."""

    name: str
    loss: AuxLoss
    policy_weight: float
    critic_weight: float

    def weight_for(self, target: str) -> float:
        """Weight to apply for ``target`` (``"policy"`` -> policy_weight, else
        critic_weight). ``0.0`` means this loss does not contribute to ``target``."""
        return self.policy_weight if target == "policy" else self.critic_weight


class AuxLossManager:
    """Holds the enabled auxiliary losses and sums their weighted contributions.

    Built from a :class:`~configs.manager.loss_cfg.LossCfg` by :meth:`from_cfg`.
    SAC depends only on :meth:`has_target` and :meth:`compute` — a small duck-typed
    interface — so it stays decoupled from the config and registry.
    """

    def __init__(self, entries: list[_LossEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_cfg(cls, cfg: "LossCfg") -> "AuxLossManager":
        """Build a manager from ``cfg`` by reading the per-loss flat fields.

        For each registered loss ``name`` it reads ``<name>_enabled``,
        ``<name>_policy_weight`` and ``<name>_critic_weight``. A registered loss
        whose fields are absent from ``cfg`` raises (config and registry must stay
        in sync); a nonzero weight on a target the loss does not support also
        raises.
        """
        entries: list[_LossEntry] = []
        # Iterate the registry (not the cfg) so every registered loss is checked
        # for its three matching cfg fields — keeps config and code in sync.
        for name, loss_cls in LOSS_REGISTRY.items():
            try:
                enabled = getattr(cfg, f"{name}_enabled")
                policy_weight = float(getattr(cfg, f"{name}_policy_weight"))
                critic_weight = float(getattr(cfg, f"{name}_critic_weight"))
            except AttributeError as exc:
                raise AttributeError(
                    f"LossCfg is missing field(s) for registered loss {name!r}; "
                    f"expected {name}_enabled, {name}_policy_weight and "
                    f"{name}_critic_weight"
                ) from exc

            if not enabled:
                continue

            for target, weight in (("policy", policy_weight), ("critic", critic_weight)):
                if weight != 0.0 and target not in loss_cls.supported_targets:
                    raise ValueError(
                        f"Loss {name!r} has {name}_{target}_weight={weight} but does "
                        f"not support the {target!r} target (supported: "
                        f"{loss_cls.supported_targets})"
                    )

            if policy_weight == 0.0 and critic_weight == 0.0:
                continue  # enabled but contributes nothing — skip cleanly

            entries.append(
                _LossEntry(
                    name=name,
                    loss=loss_cls(),
                    policy_weight=policy_weight,
                    critic_weight=critic_weight,
                )
            )
        return cls(entries)

    def has_target(self, target: str) -> bool:
        """True iff some enabled loss has a nonzero weight for ``target``."""
        return any(e.weight_for(target) != 0.0 for e in self._entries)

    def compute(
        self, ctx: LossContext, target: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(weighted_total, {name: per_agent_raw})`` for ``target``.

        ``weighted_total`` is the scalar added to the loss before backward —
        ``sum`` over applicable losses of ``weight * raw.mean()`` — and is a real
        ``0.0`` tensor on the agent's device when no loss applies, so callers can
        add it unconditionally. The dict maps each loss name to its detached,
        **unweighted**, per-agent ``(N,)`` value for logging.
        """
        total = torch.zeros((), device=ctx.agent.device)
        raw_per_agent: dict[str, torch.Tensor] = {}
        for entry in self._entries:
            weight = entry.weight_for(target)
            if weight == 0.0:
                continue
            raw = entry.loss.compute(ctx)  # (N,) per-agent, unweighted
            total = total + weight * raw.mean()
            raw_per_agent[entry.name] = raw.detach()
        return total, raw_per_agent
