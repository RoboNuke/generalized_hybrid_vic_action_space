"""Task config for the CURVED surface path-following task.

A direct extension of the flat surface task: identical in every way except the plate top carries a
single CYLINDRICAL RIDGE running ACROSS the path (its axis along the cross-path direction), so as the
tool traverses near->far it climbs over a hump and back down, and the surface normal rotates along the
way. Unlike the bumpy task (whose bumps are hidden), the curvature here is REAL and fully reported:
the surface normal, contact point, and observation setpoint are all computed against the curved
surface (see :class:`CurvedSurfaceFollowEnv`).

Difficulty is a per-env scalar ``alpha in [0, 1]`` (0 = flat, 1 = most curved). alpha is FIXED per env
(NOT re-sampled each reset) because a per-reset collider scale change is not honored by PhysX. The
levels are an evenly spaced grid of ``n_curvature_levels`` values in [0, 1], assigned to envs in
REPEATING (round-robin) order (env i -> level[i % n]); see the env docstring for why round-robin.

Only the knobs below are added — every reward term, spawn field, press-to-contact setting, and
obs/state layout is inherited unchanged from ``FlatSurfaceFollowTask``. The ``name`` stays
"flat_surface_follow" (inherited) so the ``env_setup`` grasp/weld + keypoint-servo gates match.
"""

from isaaclab.utils import configclass

from ..flat_surface_follow.flat_surface_follow_tasks_cfg import FlatSurfaceFollowTask


@configclass
class CurvedSurfaceFollowTask(FlatSurfaceFollowTask):
    # NOTE: ``name`` stays "flat_surface_follow" (inherited) so the glue-peg weld + keypoint-servo
    # gates in env_setup (which check task.name == "flat_surface_follow") match unchanged.

    # --- Curvature difficulty ladder ----------------------------------------------------------
    # Number of curvature difficulty levels: an evenly spaced alpha grid in [0, 1]. env i is assigned
    # level[i % n] in REPEATING (round-robin) order so block_agent's CONTIGUOUS-index split still gives
    # every agent the full alpha spectrum (a blocked assignment would let an agent see only some
    # alphas). n = 11 -> alpha in {0, 0.1, 0.2, ..., 1.0}. n = 1 -> all flat (reduces to the flat task).
    n_curvature_levels: int = 11

    # Fixed base radius (m) of the ridge cross-section circle, BEFORE the per-level alpha z-squash. MUST
    # exceed plate_length / 2 so the arc spans the plate with non-vertical edges. The ridge peak height
    # at alpha = 1 is R - sqrt(R^2 - (plate_length/2)^2); the top curvature at level alpha is alpha / R.
    # (Fixed for all levels so every cap stays modest-sized; alpha squashes the cross-section along the
    # surface normal, keeping the footprint fixed and the peak height linear in alpha.)
    cap_curvature_radius: float = 0.20

    # |FT force| (N) above which peg<->ridge contact counts as in-contact. The plate-filtered contact
    # sensor does not see the separate ridge prim, so the curved env OR-s in this force-based signal
    # (the FT joint-force reaction IS source-agnostic). Used by both the runtime in_contact_any and the
    # reset press-to-contact latch. Mirrors the bumpy task's bump_contact_force_threshold.
    cap_contact_force_threshold: float = 0.1
