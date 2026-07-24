from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class KeypointServoCfg:
    """Keypoint-servo action override for the surface-following task (``keypoint_servo_cfg`` header).

    Own top-level config section — NOT nested under ``runner_cfg`` — because it is a self-contained
    action-space subsystem consumed only by
    :class:`~wrappers.controllers.keypoint_servo_wrapper.KeypointServoActionWrapper` (installed by
    :func:`~learning.env_setup.build_env`). Surface task only (``Isaac-FlatSurfaceFollow-*``).

    When :attr:`enabled`, the wrapper takes over the leading pose dims of the action so the policy no
    longer emits the end-effector translation (and, with :attr:`fix_orientation`, the rotation). Each
    step it servos the held-cylinder tip toward the current setpoint keypoint (``env.setpoint_pos``)
    plus a constant directional offset, capped at the per-step ``env.pos_threshold`` (the controller
    re-multiplies by that threshold, so this is a proportional servo — exact when close, saturated
    when far). The displacement is computed in the world/env frame and rotated into the EEF frame the
    controller expects.

    Offsets are authored in the SURFACE frame (added in world before projection):
      * :attr:`along_track_offset` — along ``env.path_dir`` (the start->goal travel direction ``d``).
      * :attr:`off_track_offset`   — along ``env.d_lat`` (in-plane lateral ``n x d``).
      * :attr:`normal_offset`      — along ``env.surface_normal``.

    The removed action dims are a contiguous FRONT block (pos = 0:3, and rot = 3:6 when
    :attr:`fix_orientation`), so the wrapper is agnostic to whatever force/gain dims a given control
    wrapper appends after them, and shrinks the exposed action space by 3 (or 6) accordingly.
    """

    enabled: bool = False
    """Master switch. When True, install the keypoint-servo action override (surface task only)."""

    along_track_offset: float = 0.0
    """Constant offset (m) added to the target along ``env.path_dir`` (the along-track direction d)."""

    off_track_offset: float = 0.0
    """Constant offset (m) added to the target along ``env.d_lat`` (the off-track lateral n x d)."""

    normal_offset: float = 0.0
    """Constant offset (m) added to the target along ``env.surface_normal`` (surface normal)."""

    fix_orientation: bool = False
    """When True, the wrapper ALSO takes over the rotation dims (3:6) and HOLDS each env's INITIAL
    (spawn) EEF orientation — captured at reset and maintained for the whole episode (no offset is
    ever applied to orientation). This keeps whatever the reset set up (the grasp tilt that keeps the
    peg on the surface, plus the x-axis heading from ``task.spawn_align_eef_x_to_path``); a constant
    world orientation would instead level the wrist and lift the peg off the plate. When False, the
    policy keeps the orientation dims."""
