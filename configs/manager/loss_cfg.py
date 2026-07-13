from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class LossCfg:
    """Switches for the optional auxiliary losses added on top of SAC's built-in
    critic / policy / entropy losses (see ``learning/losses.py``).

    Convention — **three flat fields per loss**, so it is trivial to read off
    which optimizer each loss feeds:

    * ``<name>_enabled``        — master switch for the loss.
    * ``<name>_policy_weight``  — weight on the policy (actor) optimizer.
    * ``<name>_critic_weight``  — weight on the critic optimizer.

    A loss may feed both targets at once; a weight of ``0.0`` means it is not
    applied to that target. Adding a new loss = add these three fields here and
    register the matching ``AuxLoss`` subclass (same ``name``) in
    ``learning/losses.py``. The ``AuxLossManager`` reads these fields by name, so
    the prefix must match the loss's ``name`` exactly.

    All defaults are disabled / zero, so an absent ``loss_cfg`` section (or this
    default-constructed config) reproduces vanilla SAC behavior.
    """

    # ---- action_l2: example action-magnitude (L2) penalty (policy-only) ----
    action_l2_enabled: bool = False
    """Enable the example action-magnitude (L2) penalty."""

    action_l2_policy_weight: float = 0.0
    """Weight applied to the action-L2 loss on the policy (actor) optimizer."""

    action_l2_critic_weight: float = 0.0
    """Weight applied to the action-L2 loss on the critic optimizer. This loss is
    policy-only, so a nonzero value here is rejected at build time — keep it 0.0."""

    # ---- supervised_selection: BCE(selection prob, ground-truth contact) (policy-only) ----
    supervised_selection_enabled: bool = False
    """Enable the supervised selection loss (SSL): BCE between the hybrid policy's per-axis
    selection probability and the ground-truth contact state of the force-eligible axes.
    Requires a hybrid control_type AND sensor_cfg.contact.enabled (the runner validates)."""

    supervised_selection_policy_weight: float = 0.0
    """Weight applied to the SSL on the policy (actor) optimizer."""

    supervised_selection_critic_weight: float = 0.0
    """Weight applied to the SSL on the critic optimizer. The SSL is policy-only, so a
    nonzero value here is rejected at build time — keep it 0.0."""

    # ---- supervised_rotation: chordal(GAS learned rotation, ground-truth interaction frame) (policy-only) ----
    supervised_rotation_enabled: bool = False
    """Enable the supervised rotation loss: chordal distance between the GAS policy's LEARNED
    rotation frame and the ground-truth interaction frame (what a fixed-rot controller would use).
    Requires gain_mapping='rotated' with a learned rotation (the runner validates + buffers the
    target). No effect for fixed-rot / non-rotated control."""

    supervised_rotation_policy_weight: float = 0.0
    """Weight applied to the supervised-rotation loss on the policy (actor) optimizer."""

    supervised_rotation_critic_weight: float = 0.0
    """Weight applied to the supervised-rotation loss on the critic optimizer. Policy-only, so a
    nonzero value here is rejected at build time — keep it 0.0."""
