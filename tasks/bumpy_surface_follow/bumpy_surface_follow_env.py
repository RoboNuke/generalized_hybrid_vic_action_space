"""Bumpy surface path-following environment (subclass of ``FlatSurfaceFollowEnv``).

Identical to the flat surface task except the plate top carries random spherical-cap BUMPS. The bumps
are a purely PHYSICAL perturbation and are NOT observable to the policy or critic — all reported
surface geometry is the FLAT-plate geometry inherited from the parent (see the task cfg docstring).

This subclass therefore overrides only:
  * ``_setup_extra_assets`` / ``_register_extra_assets`` — spawn + register the bump sphere pool.
  * ``_randomize_plate_pose``                            — after the plate is (re)placed, (re)place the
                                                           bumps relative to its new pose. This is
                                                           called both on the initial reset and on
                                                           every IK-retry board re-sample, so the bumps
                                                           always follow the plate.
  * ``_compute_in_contact_any`` / ``_reset_contact_mask`` — OR-in a force-based contact signal so
                                                           peg<->bump contact (which the plate-filtered
                                                           ContactSensor doesn't see) still registers.

The whole surface-frame / reward / obs / keypoint machinery is inherited untouched (it reads the flat
plate), so the policy sees a flat surface and feels the bumps only through the physics.

Physical bumps are analytic KINEMATIC spheres (a ``RigidObjectCollection``): collision is computed
from the radius (no PhysX mesh cooking), so the field is re-randomized each reset by cheap vectorized
pose writes. The dynamic peg collides with the kinematic caps and is pushed; the reaction propagates
through the arm to the FT joint-force sensor exactly like plate contact (the FT sensor measures the
articulation joint reaction, which is agnostic to what the peg touches).

NOTE on efficient reset: bump poses are written only in the full-reset (``randomize_initial_state``)
path. The efficient-reset wrapper (auto-attached only when terminate_on_lag/success is set) teleports
per-env state from a single cached donor WITHOUT restoring per-env bump layouts, so a teleported env
would keep its own (stale) bumps under a donor's peg pose. Until bump poses are added to that cache,
do not enable terminate_on_* with bumps (leave the default full-reset path, which re-places bumps
every reset).
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.utils.math import matrix_from_quat

from ..flat_surface_follow.flat_surface_follow_env import FlatSurfaceFollowEnv
from .bumpy_surface_follow_env_cfg import BumpySurfaceFollowEnvCfg


class BumpySurfaceFollowEnv(FlatSurfaceFollowEnv):
    cfg: BumpySurfaceFollowEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Efficient-reset (per-env teleport) faithfulness: a RigidObjectCollection is NOT part of
        # scene.get_state()/reset_to(), so a teleported env would otherwise keep STALE bumps under a
        # donor's peg pose. Carry the per-env bump layout (_bump_local, plate-local) across teleports
        # by adding it to the wrapper's donor-copied extra attrs; _efficient_reset_finalize then writes
        # the physical sphere poses relative to the just-teleported plate. (No-op when there are no
        # bumps, i.e. bump_density=0.)
        if getattr(self, "_bumps", None) is not None:
            self._efficient_reset_extra_attrs = tuple(self._efficient_reset_extra_attrs) + ("_bump_local",)

    def _efficient_reset_finalize(self, env_ids):
        """Called by EfficientResetWrapper at the end of a partial (teleport) reset. ``_bump_local``
        was already donor-copied (extra attr); write the physical sphere poses relative to the
        teleported plate so the reset env's bumps match the donor's — a faithful reconstruction."""
        if getattr(self, "_bumps", None) is not None:
            self._place_bumps(env_ids)

    # ------------------------------------------------------------------
    # Build-time: create the bump sphere pool (before clone) + register it (after clone)
    # ------------------------------------------------------------------
    def _bump_count(self) -> int:
        """Number of bumps = round(density * plate area). Read AFTER env_cfg_overrides (this runs in
        _setup_scene, during gym.make, after build_env applies overrides), so an override on
        bump_density / plate size resizes the pool."""
        area = float(self.cfg_task.plate_length) * float(self.cfg_task.plate_width)
        return max(0, int(round(float(self.cfg_task.bump_density) * area)))

    def _setup_extra_assets(self):
        n = self._bump_count()
        self._num_bumps = n
        # Per-env bump layout in the PLATE-LOCAL frame (x, y, z), re-sampled each reset. Allocated
        # even when n == 0 so downstream code can branch cleanly on _bumps.
        self._bump_local = torch.zeros((self.num_envs, max(n, 1), 3), device=self.device)
        if n == 0:
            # No bumps -> reduces exactly to the flat surface task (regression guard).
            self._bumps = None
            return

        R = float(self.cfg_task.bump_curvature_radius)
        rigid_objects = {}
        for i in range(n):
            rigid_objects[f"bump_{i}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Bump_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=R,
                    # Kinematic (never moves under the peg's push), gravity off — mirrors the plate.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        max_depenetration_velocity=5.0,
                        solver_position_iteration_count=int(self.cfg_task.solver_position_iteration_count),
                        solver_velocity_iteration_count=1,
                        max_contact_impulse=1e32,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=float(self.cfg_task.plate_friction),
                        dynamic_friction=float(self.cfg_task.plate_friction),
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.55)),
                ),
                # Spawned out of the way; the real per-env poses are written every reset in _place_bumps.
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        self._bumps = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=rigid_objects))

    def _register_extra_assets(self):
        if getattr(self, "_bumps", None) is not None:
            self.scene.rigid_object_collections["bumps"] = self._bumps

    # ------------------------------------------------------------------
    # Reset: place the bumps relative to the (re)placed plate
    # ------------------------------------------------------------------
    def _randomize_plate_pose(self, env_ids):
        # Parent samples + writes the plate pose (and IK retries re-sample it for stragglers). Bumps
        # live in the plate frame, so re-place them right after — this covers BOTH the initial reset
        # and every retry board re-sample, keeping bumps glued to the plate.
        super()._randomize_plate_pose(env_ids)
        if getattr(self, "_bumps", None) is not None:
            self._randomize_bumps(env_ids)

    def _randomize_bumps(self, env_ids):
        """Sample a fresh bump layout (local x/y on the plate top + protrusion height) for ``env_ids``
        and write the resulting world poses. Curvature radius (sphere radius) is fixed globally."""
        n = self._num_bumps
        k = int(env_ids.shape[0])
        R = float(self.cfg_task.bump_curvature_radius)
        half_t = 0.5 * float(self.cfg_task.plate_thickness)
        h_min = float(self.cfg_task.bump_min_height)
        h_max = float(self.cfg_task.bump_max_height)

        # Cap footprint radius at the TALLEST allowed bump; inset centers by it (+ user margin) so no
        # cap hangs off the plate rim. foot = sqrt(R^2 - (R - h_max)^2).
        foot = float((R * R - (R - h_max) ** 2) ** 0.5) if R > h_max else R
        margin = foot + float(self.cfg_task.bump_edge_margin)
        half_l = max(0.5 * float(self.cfg_task.plate_length) - margin, 0.0)
        half_w = max(0.5 * float(self.cfg_task.plate_width) - margin, 0.0)

        u = torch.rand((k, n, 2), device=self.device)
        xloc = (2.0 * u[..., 0] - 1.0) * half_l
        yloc = (2.0 * u[..., 1] - 1.0) * half_w
        h = h_min + torch.rand((k, n), device=self.device) * max(h_max - h_min, 0.0)
        # Sphere center so a cap of height h shows above the plate top (plate top local z = +half_t).
        # The sphere is mostly BURIED in/under the plate: center sits (R - h) BELOW the plate top, so
        # the sphere top (center + R) lands exactly h above it. center_local_z = half_t - (R - h).
        zloc = half_t - (R - h)
        self._bump_local[env_ids, :n] = torch.stack([xloc, yloc, zloc], dim=-1)
        self._place_bumps(env_ids)

    def _place_bumps(self, env_ids):
        """Compose the plate WORLD pose with the stored local bump layout and write the sphere world
        poses. Reads the plate transform straight from the PhysX view (reflects the just-written pose
        immediately, unlike the timestamp-gated ``.data.root_pose_w``). Spheres are orientation-
        invariant, so only the plate translation + rotation are needed and the written quat is identity."""
        n = self._num_bumps
        tf = self._fixed_asset.root_physx_view.get_transforms()  # (E, 7): world [pos(3), quat_xyzw(4)]
        pos_w = tf[env_ids, 0:3]                                  # (k, 3)
        quat_wxyz = tf[env_ids][:, [6, 3, 4, 5]]                  # xyzw -> wxyz for matrix_from_quat
        R_plate = matrix_from_quat(quat_wxyz)                     # (k, 3, 3) world<-plate
        local = self._bump_local[env_ids, :n]                    # (k, n, 3)
        world = pos_w[:, None, :] + torch.einsum("kij,knj->kni", R_plate, local)  # (k, n, 3)
        ident = torch.zeros((int(env_ids.shape[0]), n, 4), device=self.device)
        ident[..., 0] = 1.0
        pose = torch.cat([world, ident], dim=-1)                 # (k, n, 7)
        self._bumps.write_object_pose_to_sim(pose, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Contact: the plate-filtered ContactSensor misses peg<->bump contact, so OR-in the (source-
    # agnostic) FT joint-force reaction. Used at runtime and by the reset press-to-contact latch.
    # ------------------------------------------------------------------
    def _force_in_contact(self):
        thr = float(self.cfg_task.bump_contact_force_threshold)
        return torch.linalg.norm(self.force_sensor_world_smooth[:, 0:3], dim=-1) > thr

    def _compute_in_contact_any(self):
        return super()._compute_in_contact_any() | self._force_in_contact()

    def _reset_contact_mask(self):
        return super()._reset_contact_mask() | self._force_in_contact()
