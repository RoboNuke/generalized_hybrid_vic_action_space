"""Curved surface path-following environment (subclass of ``FlatSurfaceFollowEnv``).

Identical to the flat surface task except the plate top carries a single CYLINDRICAL RIDGE running
ACROSS the path (axis along the cross-path direction), so as the tool traverses near->far it climbs
over a hump and back down, and the surface normal rotates along the way. Unlike the bumpy task (whose
bumps are a hidden physical perturbation), the curvature here is REAL and fully reported: the surface
normal, the contact point, the tip-surface distance, and the observation setpoint are all computed
against the curved surface, so the policy/critic see the true LOCAL geometry at the contact point.

Difficulty is a per-env scalar ``alpha in [0, 1]`` (0 = flat, 1 = most curved). The ridge cross-section
is a circle of fixed base radius ``cap_curvature_radius`` squashed along the surface normal by
``alpha`` (the paper's z-scale trick): the footprint is fixed and the peak height scales linearly with
alpha, so alpha is a clean curvature/difficulty knob (top curvature = alpha / R). ``alpha = 0`` reduces
EXACTLY to the flat task (the ridge collapses to the plate top and the geometry hooks return the flat
plate values).

alpha is FIXED per env (NOT re-sampled each reset): a per-reset collider scale change is not honored by
PhysX (Isaac's ``randomize_rigid_body_scale`` warns runtime scaling is "unpredictable"), so the squash
is baked into the collision geometry at spawn. The levels are an evenly spaced grid of
``n_curvature_levels`` values in [0, 1], assigned to envs in REPEATING (round-robin) order:
``env i -> level[i % n]``. Round-robin (NOT contiguous blocks) is required so ``block_agent``'s
contiguous-index split gives every agent the full alpha spectrum — a blocked assignment would let an
agent see only some alpha values.

Physical realization (mirrors the bumpy task's kinematic sphere pool): a pool of ``K - 1`` kinematic
cylinder MESHES, one per NON-zero level, each z-squashed (baked onto the collider's ``xformOp:scale``
BEFORE cloning, so it is cooked into the collision geometry — the SUPPORTED path) by its level's alpha.
Every env carries the whole pool; per env we place the ONE cap matching its level on the plate and
PARK the rest far below (cheap vectorized pose writes on reset). Level 0 (alpha = 0) has no cap -> the
flat plate, exactly.

Metrics: on top of the inherited success rate, per-episode success is also published bucketed into the
four alpha quartiles ([0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0]) so the success-vs-difficulty curve
is directly readable.

NOTE on efficient reset: SUPPORTED. The held object is spawned at the NEAR EDGE, which sits at plate-top
height (ridge h = 0) for EVERY curvature, so a donor's held-object pose teleports validly onto any
alpha env. ``scene.reset_to`` teleports the robot + plate + held object but NOT the RigidObjectCollection
caps, so ``_efficient_reset_finalize`` re-places each env's own (fixed-alpha) cap relative to the
teleported plate (mirroring the bumpy task's bump re-placement). This lets ``terminate_on_lag`` /
``terminate_on_success`` AND the fragile-object / over-force collision wrapper drive cheap per-env
teleport resets on the curved + wiping tasks.
"""

import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

from ..flat_surface_follow.flat_surface_follow_env import FlatSurfaceFollowEnv
from .curved_surface_follow_env_cfg import CurvedSurfaceFollowEnvCfg


class CurvedSurfaceFollowEnv(FlatSurfaceFollowEnv):
    cfg: CurvedSurfaceFollowEnvCfg

    # ------------------------------------------------------------------
    # Curvature-ladder constants (config-derived; read AFTER env_cfg_overrides)
    # ------------------------------------------------------------------
    def _ridge_consts(self):
        """Return ``(K, levels, w, R, edge, top)`` derived from the task cfg (read here, in
        _setup_extra_assets during gym.make, AFTER build_env applies env_cfg_overrides).

          K      number of curvature levels (>= 1)
          levels evenly spaced alpha grid in [0, 1] (python list of K floats; [0.0] when K == 1)
          w      path half-length = ridge half-footprint (m) = plate_length / 2
          R      fixed base radius of the ridge cross-section circle (m); must exceed w
          edge   sqrt(R^2 - w^2) (m): the per-alpha height subtracted so the ridge pins to h = 0 at |x| = w
          top    plate top in plate-ROOT-local z (m) = plate_thickness / 2
        """
        K = max(1, int(self.cfg_task.n_curvature_levels))
        levels = [0.0] if K == 1 else [j / (K - 1) for j in range(K)]
        w = 0.5 * float(self.cfg_task.plate_length)
        R = float(self.cfg_task.cap_curvature_radius)
        if R <= w:
            raise ValueError(
                f"cap_curvature_radius ({R}) must exceed plate_length/2 ({w}) so the ridge arc spans "
                "the plate with non-vertical edges."
            )
        edge = float((R * R - w * w) ** 0.5)
        top = 0.5 * float(self.cfg_task.plate_thickness)
        return K, levels, w, R, edge, top

    # ------------------------------------------------------------------
    # Build-time: create the z-squashed cap pool (before clone) + register it (after clone)
    # ------------------------------------------------------------------
    def _setup_extra_assets(self):
        K, levels, w, R, edge, top = self._ridge_consts()
        # Stash the geometry constants + per-env curvature assignment. num_envs / device are already set
        # by DirectRLEnv before _setup_scene, so this is safe here (and available before the first reset,
        # which calls _place_caps and the geometry hooks).
        self._ridge_w, self._ridge_R, self._ridge_edge, self._plate_top_local = w, R, edge, top
        idx = torch.arange(self.num_envs, device=self.device) % K            # (E,) round-robin level index
        self._env_level_idx = idx
        self.alpha = torch.tensor(levels, device=self.device, dtype=torch.float32)[idx]  # (E,) curvature in [0,1]

        # One physical cap per NON-zero level (level 0 = flat = no cap). Every env carries the whole pool.
        self._cap_alpha = [float(levels[j]) for j in range(1, K)]            # alphas for caps (len K - 1)
        self._num_caps = len(self._cap_alpha)
        if self._num_caps == 0:
            self._caps = None                                                # K == 1: all flat, no caps
            return

        height = 1.5 * float(self.cfg_task.plate_width)                      # cap length along its axis (cross-path)
        rigid_objects = {}
        for m in range(self._num_caps):
            rigid_objects[f"cap_{m}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Cap_{m}",
                spawn=sim_utils.MeshCylinderCfg(
                    radius=R,
                    height=height,
                    axis="Y",                                                # long axis across the path
                    activate_contact_sensors=True,                           # oracle contact sensor sees peg<->ridge contact (filtered in)
                    # Kinematic (never moves under the peg), gravity off — mirrors the plate.
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
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.5, 0.6)),
                ),
                # Spawned parked far below; the real per-env poses are written every reset in _place_caps.
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        self._caps = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=rigid_objects))
        # Bake the per-cap alpha z-squash into the collision geometry BEFORE clone_environments (a
        # spawn-time scale is cooked into the collider — the SUPPORTED path; runtime scaling is not).
        self._bake_cap_zscale()

    def _bake_cap_zscale(self):
        """Set ``xformOp:scale = (1, 1, alpha_m)`` on each cap's collision-geometry prim, on the env_0
        SOURCE prim before cloning so every clone inherits it and it is cooked into the collider. This
        squashes the circular cross-section along the cap's local z (the surface normal) by alpha,
        turning it into the elliptical ridge the analytic geometry below assumes."""
        from pxr import Gf, UsdGeom

        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        for m, a in enumerate(self._cap_alpha):
            # The geometry (and its collider) live at .../geometry/mesh under the spawned cap prim.
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Cap_{m}/geometry/mesh")
            if not prim.IsValid():
                # Fall back to the cap root if the geometry child path differs in this Isaac version
                # (scaling the root propagates to the child geometry too).
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/Cap_{m}")
            xf = UsdGeom.Xformable(prim)
            scale_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            if scale_op is None:
                scale_op = xf.AddScaleOp()
            scale_op.Set(Gf.Vec3f(1.0, 1.0, float(a)))

    def _register_extra_assets(self):
        if getattr(self, "_caps", None) is not None:
            self.scene.rigid_object_collections["caps"] = self._caps
            # The peg rides the ridge CAP (a separate rigid body), not the plate — so add the caps to
            # the oracle contact sensor's filter. The ContactSensorWrapper reads this and includes it in
            # its rigid-contact-view filter_patterns, then sums peg<->{plate,cap} force. Keeps force +
            # in_contact on the SAME oracle source as the flat task (no force_source switch, no FT hacks).
            # ONE glob per cap index (each matches exactly num_envs bodies — PhysX's contact view
            # requires per-env filters, so a single "Cap_*" wildcard (num_envs*num_caps) is rejected).
            # Only one cap is active per env; the parked ones read ~0 and drop out of the summed force.
            self._extra_contact_filter_globs = [
                f"/World/envs/env_*/Cap_{m}" for m in range(self._num_caps)
            ]

    def _efficient_reset_finalize(self, env_ids):
        """Called by EfficientResetWrapper at the end of a partial (teleport) reset. ``scene.reset_to``
        teleports the robot + plate + held object but NOT the RigidObjectCollection caps (they aren't in
        the scene-state dict), so re-place each env's ACTIVE cap relative to the just-teleported plate.
        This is valid because the held object is spawned at the NEAR EDGE, which sits at plate-top height
        (ridge h = 0) for EVERY curvature — so a donor's held-object pose is correct under any env's
        alpha, and only the cap (the env keeps its own fixed alpha) needs re-placing. Enables efficient
        reset (and thus the fragile-object / collision-termination wrapper) for the curved + wiping tasks."""
        if getattr(self, "_caps", None) is not None:
            self._place_caps(env_ids)

    # ------------------------------------------------------------------
    # Reset: place each env's active cap on its (re)placed plate; park the rest
    # ------------------------------------------------------------------
    def _randomize_plate_pose(self, env_ids):
        # Parent samples + writes the plate pose (and IK retries re-sample it for stragglers). The ridge
        # lives in the plate frame, so re-place the caps right after — this covers BOTH the initial reset
        # and every retry board re-sample, keeping the active cap glued to the plate.
        super()._randomize_plate_pose(env_ids)
        if getattr(self, "_caps", None) is not None:
            self._place_caps(env_ids)

    def _place_caps(self, env_ids):
        """Place each env's ACTIVE cap (the one whose level == env's level) on its plate and PARK the
        rest far below. alpha = 0 envs have no active cap (the flat plate). Reads the plate transform
        straight from the PhysX view (reflects the just-written pose immediately)."""
        tf = self._fixed_asset.root_physx_view.get_transforms()             # (E, 7) world [pos(3), quat_xyzw(4)]
        k = int(env_ids.shape[0])
        pos_w = tf[env_ids, 0:3]                                            # (k, 3)
        quat_wxyz = tf[env_ids][:, [6, 3, 4, 5]]                            # xyzw -> wxyz
        R_plate = matrix_from_quat(quat_wxyz)                               # (k, 3, 3) world<-plate
        env_level = self._env_level_idx[env_ids]                           # (k,) level index per env

        pose = torch.zeros((k, self._num_caps, 7), device=self.device)
        parked = pos_w.clone()
        parked[:, 2] -= 5.0                                                 # far below the plate (never touched)
        for m in range(self._num_caps):
            L = m + 1                                                       # this cap's level index (1..K-1)
            a = self._cap_alpha[m]
            # Plate-local z of the cap CENTER so the (z-squashed) top pins to the plate top at |x| = w:
            # top surface = z_center + alpha*sqrt(R^2 - x^2); at x = w that must equal plate top, so
            # z_center = plate_top - alpha*sqrt(R^2 - w^2). The cap axis (Y) and squash-z align with the
            # plate frame via the plate quat.
            center_local = torch.tensor([0.0, 0.0, self._plate_top_local - a * self._ridge_edge], device=self.device)
            world = pos_w + torch.einsum("kij,j->ki", R_plate, center_local)  # (k, 3) cap center world
            active = (env_level == L).unsqueeze(-1)                          # (k, 1)
            pose[:, m, 0:3] = torch.where(active, world, parked)
            pose[:, m, 3:7] = quat_wxyz
        self._caps.write_object_pose_to_sim(pose, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Curved-surface geometry (plate-local <-> world helpers + the ridge profile)
    # ------------------------------------------------------------------
    def _to_plate_local(self, p_world):
        """World point (E,3, env-relative) -> plate-local (plate ROOT frame)."""
        return quat_apply(torch_utils.quat_conjugate(self.fixed_quat), p_world - self.fixed_pos)

    def _to_world(self, p_local):
        """Plate-local point (E,3) -> world (env-relative)."""
        return self.fixed_pos + quat_apply(self.fixed_quat, p_local)

    def _dir_to_world(self, v_local):
        """Plate-local direction (E,3) -> world (rotation only)."""
        return quat_apply(self.fixed_quat, v_local)

    def _ridge_geom_local(self, x):
        """Ridge surface at the plate-local along-path coordinate ``x`` (E,), per env (uses ``alpha``).

        Returns ``(h, n_local, t_local)``:
          h       (E,)   surface height above the plate top (0 at |x| >= w, and 0 where alpha == 0)
          n_local (E,3)  plate-local surface normal   (unnormalized; the caller renormalizes)
          t_local (E,3)  plate-local along-path tangent (unnormalized)

        Surface z(x) = plate_top + alpha*(sqrt(R^2 - x^2) - sqrt(R^2 - w^2)) for |x| <= w. The slope is
        dz/dx = -alpha*x/sqrt(R^2 - x^2); the (unit-after-normalization) normal is (-dz/dx, 0, 1) and the
        tangent is (1, 0, dz/dx). alpha == 0 rows give h = 0, normal = +z, tangent = +x (the flat plate).
        """
        w, R = self._ridge_w, self._ridge_R
        a = self.alpha                                                     # (E,)
        xc = x.clamp(-w, w)                                                # ridge defined on [-w, w]; flat beyond
        s = torch.sqrt((R * R - xc * xc).clamp_min(1e-12))                # sqrt(R^2 - x^2)
        h = a * (s - self._ridge_edge)                                    # height above plate top
        slope = -a * xc / s                                               # dz/dx
        zeros = torch.zeros_like(slope)
        ones = torch.ones_like(slope)
        n_local = torch.stack([-slope, zeros, ones], dim=-1)             # (E,3)
        t_local = torch.stack([ones, zeros, slope], dim=-1)             # (E,3)
        return h, n_local, t_local

    # ------------------------------------------------------------------
    # Geometry hooks overridden from the flat env (contact projection, local frame, setpoint path)
    # ------------------------------------------------------------------
    def _project_tip_to_surface(self, cyl_tip, start, plate_normal):
        """Project the tool tip onto the RIDGE (vertical drop in the plate frame) -> (contact_point,
        tip_surface_dist). tip_surface_dist is the signed clearance of the tip SURFACE above the ridge
        (plate-local z; >0 above, ~0 in contact, <0 penetrating) — the rounded-tip ``_tip_sphere_r`` is
        subtracted from the tip-center height, matching the flat env's convention."""
        p_loc = self._to_plate_local(cyl_tip)                            # (E,3) tip in plate frame
        x = p_loc[:, 0]
        h, _, _ = self._ridge_geom_local(x)
        surf_z = self._plate_top_local + h                                # (E,) ridge surface height under the tip
        surf_loc = torch.stack([p_loc[:, 0], p_loc[:, 1], surf_z], dim=-1)
        return self._to_world(surf_loc), p_loc[:, 2] - surf_z - self._tip_sphere_r

    def _surface_normal_and_dir_at(self, query_pos):
        """LOCAL surface normal + along-path tangent at the ridge point ``query_pos`` (E,3, world). The
        base _compute renormalizes both, so returning unnormalized directions is fine."""
        x = self._to_plate_local(query_pos)[:, 0]
        _, n_local, t_local = self._ridge_geom_local(x)
        return self._dir_to_world(n_local), self._dir_to_world(t_local)

    def path_point_on_surface(self, arclen):
        """(E,3) env-relative point ON THE RIDGE at along-path arc-length ``arclen`` (E,). The along-path
        coordinate maps to plate-local x = -w + arclen (arclen in [0, L] -> x in [-w, w]); the point is
        lifted onto the ridge so it lies on the curved surface, not the straight start->goal chord.
        Consumed by _point_on_path (the observation setpoint) AND by the recorder to draw the keypoint
        path / goal / pace markers on the surface (via keypoint_world_positions_on_surface below)."""
        x = (-self._ridge_w + arclen).clamp(-self._ridge_w, self._ridge_w)  # (E,)
        h, _, _ = self._ridge_geom_local(x)
        pt_loc = torch.stack([x, torch.zeros_like(x), self._plate_top_local + h], dim=-1)
        return self._to_world(pt_loc)

    def _point_on_path(self, start, path_dir, arclen):
        """Observation setpoint on the RIDGE at arc-length ``arclen`` (overrides the flat straight-line
        placement so the tracked setpoint follows the curved surface)."""
        return self.path_point_on_surface(arclen)

    def keypoint_world_positions_on_surface(self, k):
        """(E, k, 3) env-relative world positions of keypoints 1..k ON THE RIDGE (not the straight
        chord). Keypoint j sits at along-path arc-length j*keypoint_spacing, lifted onto the surface.
        The recorder prefers this over the flat ``surface_viz.keypoint_world_positions`` when present,
        so the drawn path hugs the curve instead of floating in the air above it."""
        spacing = float(self.keypoint_spacing)
        cols = []
        for j in range(1, int(k) + 1):
            arclen = torch.full((self.num_envs,), j * spacing, device=self.device).minimum(self.path_length)
            cols.append(self.path_point_on_surface(arclen))
        return torch.stack(cols, dim=1)  # (E, k, 3)

    # Contact detection needs NO override here: the ridge caps are in the oracle contact sensor's
    # filter (see _register_extra_assets), so env.in_contact and the oracle force already include
    # peg<->ridge contact — same machinery and same source as the flat plate.

    # ------------------------------------------------------------------
    # Metrics: success bucketed by curvature difficulty (alpha quartiles)
    # ------------------------------------------------------------------
    def _get_rewards(self):
        rew_buf = super()._get_rewards()
        # Publish per-episode success into the four alpha-quartile buckets (NaN outside the env's bucket,
        # so block_agent — which drops non-finite values — averages each series ONLY over its bucket's
        # finishing envs, giving a per-difficulty success rate). Uses the same per-episode success flag
        # (success_reward_given) and reset mask the parent already set for per_env_episode_stat.
        stat = self.extras.get("per_env_episode_stat", None)
        if stat is not None:
            succeeded = self.success_reward_given.float()
            nan = torch.full_like(succeeded, float("nan"))
            bucket = (self.alpha * 4.0).floor().clamp(0, 3).long()          # 0..3: [0,.25) [.25,.5) [.5,.75) [.75,1]
            for i, (lo, hi) in enumerate(((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))):
                stat[f"Success by curvature / alpha [{lo:.2f}-{hi:.2f}]"] = torch.where(
                    bucket == i, succeeded, nan
                )
        return rew_buf
