"""Task config for the BUMPY surface path-following task.

A direct extension of the flat surface task: identical in every way except the plate top carries
random spherical-cap BUMPS of configurable density, curvature radius, and height. The bumps are a
PURELY PHYSICAL perturbation — they are NOT observable to the policy or critic. All reported surface
geometry (surface_normal, path_dir, contact_point, progress, cross_track, keypoints, the force
projected onto the normal, ...) is computed against the FLAT plate exactly as in the parent task, as
if the bumps were not there. The robot only experiences the bumps through physics: the spheres shove
the peg around, which shows up as position deviations and a changed force profile it must reject to
keep hitting the flat-plate force/orientation/path targets.

Because the geometry is reported flat, this subclass adds ONLY the bump knobs below — every reward
term, spawn field, press-to-contact setting, and obs/state layout is inherited unchanged from
``FlatSurfaceFollowTask``. The name is kept as ``flat_surface_follow`` so ``env_setup`` grasp/weld
and keypoint-servo gating (which keys off the id prefix + task name) applies with no changes.
"""

from isaaclab.utils import configclass

from ..flat_surface_follow.flat_surface_follow_tasks_cfg import FlatSurfaceFollowTask


@configclass
class BumpySurfaceFollowTask(FlatSurfaceFollowTask):
    # NOTE: ``name`` stays "flat_surface_follow" (inherited) so the glue-peg weld + keypoint-servo
    # gates in env_setup (which check task.name == "flat_surface_follow") match unchanged.

    # --- Bump field ---------------------------------------------------------------------------
    # Random spherical-cap bumps sprinkled over the plate top. Each bump is an analytic kinematic
    # SPHERE (no mesh cooking), so the whole field can be re-randomized every reset by cheap
    # vectorized pose writes (like the plate itself). curvature radius = the sphere radius; the
    # protrusion above the plate top is controlled purely by the sphere's z (a cap of height h shows
    # when the center sits at plate_top + R - h). density sets how many bumps populate the plate.

    # Bumps per m^2 of plate top. The build-time bump COUNT = round(bump_density * plate_length *
    # plate_width). 0.0 => NO bumps (the task reduces EXACTLY to the flat surface task — the
    # regression guard). Read in _setup_scene AFTER env_cfg_overrides, so an override resizes the pool.
    bump_density: float = 250.0                  # ~10 bumps over the default 0.20 x 0.20 m plate

    # Sphere radius R (m) == the bump's curvature radius. FIXED globally (all bumps share it) for a
    # first cut: reset is then a pure vectorized pose write (no per-bump radius authoring).
    bump_curvature_radius: float = 0.02

    # Per-bump protrusion above the plate top (m), sampled U[bump_min_height, bump_max_height] each
    # reset. Must be < bump_curvature_radius (a spherical cap can't be taller than its sphere radius).
    bump_min_height: float = 0.001               # avoid flush/invisible bumps
    bump_max_height: float = 0.004

    # Extra inset (m) of bump centers from the plate edges, BEYOND the automatic cap-footprint inset
    # (so no cap ever hangs off the plate rim). Raise to keep bumps clear of the near/far path ends.
    bump_edge_margin: float = 0.0

    # |FT force| (N) above which the bumpy env counts the peg as in contact. The plate-filtered
