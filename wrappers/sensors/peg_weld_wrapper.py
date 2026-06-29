"""Rigidly mount the held peg to the gripper by folding it into the Franka articulation.

Motivation: by default the peg is held in the closed gripper by *friction only*
(``factory_env.randomize_initial_state`` teleports it into the fingers with gravity off,
then closes the gripper — there is no joint). Under a contact-rich insertion policy the
peg can creep / rotate in the grip during execution. This installer rigidly fixes the peg
to the fingertip for the whole rollout.

HOW (and why this specific way): the obvious approach — a ``FixedJoint`` between the fingertip
and the peg with ``excludeFromArticulation=True`` — is a maximal-coordinate **loop joint**
between two separate articulations (Franka + the peg's single-body articulation). PhysX leaks
native host memory every step maintaining that loop constraint (confirmed: ~0.5 MB/physics-step;
host RAM climbs until OOM). Instead we fold the peg INTO the Franka articulation as a
reduced-coordinate **link**:

  1. On the env-0 prototype, BEFORE ``clone_environments``: strip the peg's ArticulationRootAPI
     (so it is a plain rigid body, not its own articulation) and author a fixed joint from the
     fingertip to the peg with ``excludeFromArticulation=False``. PhysX then folds the peg in as
     a Franka link. ``replicate_physics`` propagates it to every env — Fabric-compatible
     (``clone_in_fabric=True``), no per-env USD prims, no loop joint, **no leak** (verified flat
     over 6000 steps). The peg's contact reaction still propagates through the articulation to
     the wrist, so the force feedback the controller relies on is preserved.
  2. The separate ``_held_asset`` Articulation the env created can no longer initialize (it has
     no ArticulationRootAPI). We cancel its play-init callback, drop it from the scene, and
     replace it with :class:`_HeldLinkShim`, which serves the peg pose from the Franka
     articulation (so ``held_pos``/``held_quat`` keep working unchanged) and no-ops the reset
     teleport — the joint holds the peg, so there is nothing to teleport.

The peg PRIM stays at ``/World/envs/env_*/HeldAsset/...``, so the contact-sensor / reward paths
that reference it are unchanged; only the env's articulation-level access is rerouted.

HARD CONSTRAINTS (enforced by the caller, ``learning/env_setup.py``):

  * GPU pipeline: the joint's local frame is USD-authored and parsed by PhysX at play time — it
    CANNOT change between steps. So the peg-in-gripper transform must be a CONSTANT, fixed before
    ``sim.reset()``; only well-defined when the grasp is deterministic, hence ONLY when
    ``grasp_rot_mode == "fixed"``.
  * ``held_asset_pos_noise`` zeroed (the caller does this) so the env's intended grasp matches
    the fixed link frame.
  * The held-asset mass/material randomization events must be disabled (the caller does this):
    they resolve the ``held_asset`` scene entity, which no longer exists as a separate asset.
  * ``peg_insert`` task only — the analytic transform mirrors the ``peg_insert`` branch of
    ``get_handheld_asset_relative_pose``.

The transform is computed analytically as ``flip_z ∘ asset_in_hand`` — exactly the
peg-relative-to-fingertip transform ``randomize_initial_state`` builds (``factory_env.py``),
including the fixed grasp tilt. Call BEFORE ``gym.make``; Forge inherits ``_setup_scene`` from
Factory, so patching the concrete env class the runner resolves covers the Forge tasks.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Matches the per-env index in a cloned prim path (e.g. ".../env_3/...").
_ENV_IDX_RE = re.compile(r"/env_(\d+)/")
# Robot body the peg is welded to: its world pose is ``fingertip_midpoint_*`` in
# ``factory_env.py`` (``fingertip_body_idx = body_names.index("panda_fingertip_centered")``),
# i.e. the exact frame ``randomize_initial_state`` seats the peg against.
_FINGERTIP_BODY_NAME = "panda_fingertip_centered"
# Joint prim name authored under each env's root.
_WELD_PRIM_NAME = "PegGripperWeld"


def _env_index(path: str) -> int | None:
    m = _ENV_IDX_RE.search(path)
    return int(m.group(1)) if m else None


def _find_descendant_by_name(root_prim, name: str):
    """BFS for the first descendant prim whose name matches ``name`` (root included)."""
    queue = [root_prim]
    while queue:
        prim = queue.pop(0)
        if prim.GetName() == name:
            return prim
        queue.extend(prim.GetChildren())
    return None


def _find_rigid_body(root_prim):
    """BFS for the first descendant carrying ``UsdPhysics.RigidBodyAPI`` (root included).

    The held asset spawns as a single-link articulation; the rigid body that the weld must
    target (and whose root pose the reset writes) is the link carrying the RigidBodyAPI,
    which may be the root or a child depending on the asset USD.
    """
    from pxr import UsdPhysics

    queue = [root_prim]
    while queue:
        prim = queue.pop(0)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        queue.extend(prim.GetChildren())
    return None


def _compute_peg_in_fingertip(env, tilt_deg: Sequence[float]):
    """Analytic peg-body pose in the (raw) fingertip frame, as ``(pos_xyz, quat_wxyz)``.

    Reproduces ``randomize_initial_state``'s seating (``factory_env.py:744-760``) for
    ``peg_insert``: the peg world pose is ``(fingertip ∘ flip_z) ∘ asset_in_hand``, so the
    peg pose RELATIVE TO THE RAW FINGERTIP body is ``flip_z ∘ asset_in_hand`` — which is
    exactly the joint's ``localPose0`` (frame on body0=fingertip that must coincide with the
    peg's body origin). The fixed grasp tilt is folded into ``held_asset_relative_quat`` the
    same way :mod:`wrappers.sensors.grasp_tilt_wrapper` does.
    """
    import numpy as np
    import torch

    import isaacsim.core.utils.torch as torch_utils  # wxyz layout, matches factory_env.py

    device = env.device
    height = env.cfg_task.held_asset_cfg.height
    fingerpad = env.cfg_task.robot_cfg.franka_fingerpad_length

    # Base held-asset relative pose for peg_insert: pure +z offset, identity orientation.
    # NOTE: keep every tensor float32 — the env's torch_utils (TorchScript) refuse mixed
    # dtypes, and np.deg2rad below yields float64, so cast the Euler angles explicitly.
    rel_pos = torch.zeros((1, 3), device=device, dtype=torch.float32)
    # Held-asset origin offset along the grasp axis. The peg USD origin is at the base
    # (height - fingerpad); the procedural surface CYLINDER origin is at its CENTER
    # (height/2 - fingerpad) — matching FlatSurfaceFollowEnv.get_handheld_asset_relative_pose.
    if env.cfg_task.name == "flat_surface_follow":
        rel_pos[0, 2] = height / 2.0 - fingerpad
    else:
        rel_pos[0, 2] = height - fingerpad
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)

    # Fixed grasp tilt, composed onto the (identity) base exactly as grasp_tilt_wrapper:
    # rel_quat = quat_mul(quat_conjugate(perturb), base).
    r, p, y = (float(np.deg2rad(float(v))) for v in tilt_deg)
    perturb = torch_utils.quat_from_euler_xyz(
        torch.tensor([r], device=device, dtype=torch.float32),
        torch.tensor([p], device=device, dtype=torch.float32),
        torch.tensor([y], device=device, dtype=torch.float32),
    )
    rel_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(perturb), identity)

    # asset_in_hand = inverse(held_asset_relative); peg_in_fingertip = flip_z ∘ asset_in_hand.
    ah_quat, ah_pos = torch_utils.tf_inverse(rel_quat, rel_pos)
    flip_z = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    zeros3 = torch.zeros((1, 3), device=device, dtype=torch.float32)
    p0_quat, p0_pos = torch_utils.tf_combine(flip_z, zeros3, ah_quat, ah_pos)
    return p0_pos[0].tolist(), p0_quat[0].tolist()  # ([x,y,z], [w,x,y,z])


def _strip_articulation_root(root_prim) -> list:
    """Remove ArticulationRootAPI / PhysxArticulationAPI from the subtree so the peg becomes a
    plain rigid body — a prerequisite for PhysX folding it into the Franka articulation (a body
    that is already its own articulation cannot be absorbed into another)."""
    from pxr import Usd, UsdPhysics, PhysxSchema

    removed = []
    for prim in Usd.PrimRange(root_prim):
        changed = False
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            changed = True
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            changed = True
        if changed:
            removed.append(prim.GetPath().pathString)
    return removed


class _HeldLinkData:
    """Serves the peg's ``root_*`` state by reading the Franka articulation link the peg was
    folded into — so ``factory_env``'s ``held_pos``/``held_quat`` (read from
    ``_held_asset.data.root_pos_w``/``root_quat_w``) keep working unchanged."""

    def __init__(self, shim: "_HeldLinkShim"):
        self._shim = shim

    @property
    def root_pos_w(self):
        s = self._shim
        return s._robot.data.body_pos_w[:, s._idx()]

    @property
    def root_quat_w(self):
        s = self._shim
        return s._robot.data.body_quat_w[:, s._idx()]

    @property
    def root_lin_vel_w(self):
        s = self._shim
        return s._robot.data.body_lin_vel_w[:, s._idx()]

    @property
    def root_ang_vel_w(self):
        s = self._shim
        return s._robot.data.body_ang_vel_w[:, s._idx()]

    @property
    def root_state_w(self):
        import torch

        s = self._shim
        return torch.cat(
            (self.root_pos_w, self.root_quat_w, self.root_lin_vel_w, self.root_ang_vel_w), dim=-1
        )

    @property
    def default_root_state(self):
        import torch

        s = self._shim
        # Only used to build the (now no-op) reset teleport target; values are irrelevant.
        return torch.zeros((s._num_envs, 13), device=s._device)


class _HeldLinkShim:
    """Drop-in replacement for the removed held-asset Articulation. Exposes the peg pose via the
    Franka articulation (the peg is a link now) and no-ops the reset teleport — the fixed joint
    holds the peg in the grasp, so there is nothing to write. ``root_physx_view`` is ``None`` so
    the patched ``set_friction`` skips it (the peg keeps its spawn-default friction)."""

    def __init__(self, robot, peg_body_name: str, num_envs: int, device):
        self._robot = robot
        self._peg_body_name = peg_body_name
        self._num_envs = num_envs
        self._device = device
        self._peg_idx = None
        self.root_physx_view = None
        self.data = _HeldLinkData(self)

    def _idx(self) -> int:
        if self._peg_idx is None:
            self._peg_idx = list(self._robot.body_names).index(self._peg_body_name)
        return self._peg_idx

    def write_root_pose_to_sim(self, *args, **kwargs):
        pass

    def write_root_velocity_to_sim(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


def install_peg_weld(rel_grasp_rot_init_deg: Sequence[float], env_class: Any = None) -> None:
    """Patch ``<env_class>._setup_scene`` to fold the peg into the Franka articulation as a fixed
    LINK (reduced-coordinate), giving a rigid grasp with no maximal-coordinate loop joint.

    :param rel_grasp_rot_init_deg: the ``"fixed"`` ``[roll, pitch, yaw]`` (deg) grasp tilt,
        folded into the link transform so the peg sits at the tilted grasp the env seats.
    :param env_class: the concrete direct-API env class whose ``_setup_scene`` runs at
        ``gym.make`` (defaults to ``FactoryEnv``; Forge inherits it).

    See the module docstring for the mechanism. Call BEFORE ``gym.make``. Composes with the
    other ``_setup_scene`` shims (each wraps ``clone_environments``, restore-then-call).
    """
    if env_class is None:
        from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

        env_class = FactoryEnv

    # The de-articulated peg / link shim have no root_physx_view; make set_friction skip them
    # (the peg keeps its spawn-default friction; it is rigidly fixed, not friction-held).
    import isaaclab_tasks.direct.factory.factory_utils as factory_utils

    if not getattr(factory_utils.set_friction, "_peg_weld_safe", False):
        _orig_set_friction = factory_utils.set_friction

        def _safe_set_friction(asset, value, num_envs):
            if getattr(asset, "root_physx_view", None) is None:
                return
            return _orig_set_friction(asset, value, num_envs)

        _safe_set_friction._peg_weld_safe = True
        factory_utils.set_friction = _safe_set_friction

    _original_setup_scene = env_class._setup_scene

    def _patched_setup_scene(self):
        if self.cfg_task.name not in ("peg_insert", "flat_surface_follow"):
            raise ValueError(
                "install_peg_weld supports the 'peg_insert' / 'flat_surface_follow' tasks (the weld "
                f"transform mirrors their grasp), but cfg_task.name is {self.cfg_task.name!r}."
            )

        state = {}

        def _author_link_on_prototype():
            """On env_0 before clone: de-articulate the peg and author a NON-excluded fixed joint
            so PhysX folds the peg into the Franka articulation as a reduced-coordinate link."""
            import isaaclab.sim as sim_utils
            from pxr import Gf, Sdf, UsdPhysics

            pos0, quat0 = _compute_peg_in_fingertip(self, rel_grasp_rot_init_deg)  # ([xyz],[wxyz])
            robot_expr = self.cfg.robot.prim_path  # "/World/envs/env_.*/Robot"
            held_expr = self.cfg_task.held_asset.prim_path  # "/World/envs/env_.*/HeldAsset"
            robot_roots = {_env_index(p.GetPath().pathString): p for p in sim_utils.find_matching_prims(robot_expr)}
            held_roots = {_env_index(p.GetPath().pathString): p for p in sim_utils.find_matching_prims(held_expr)}
            if not robot_roots or set(robot_roots) != set(held_roots):
                raise RuntimeError(
                    "install_peg_weld: could not pair Robot/HeldAsset prototype prims "
                    f"(robot envs {sorted(robot_roots)}, held envs {sorted(held_roots)})."
                )
            stage = robot_roots[next(iter(robot_roots))].GetStage()
            n_made = 0
            for env_idx, robot_root in robot_roots.items():
                fingertip = _find_descendant_by_name(robot_root, _FINGERTIP_BODY_NAME)
                peg_body = _find_rigid_body(held_roots[env_idx])
                if fingertip is None or peg_body is None:
                    raise RuntimeError(
                        f"install_peg_weld: env_{env_idx} missing weld bodies "
                        f"(fingertip={fingertip is not None}, peg_body={peg_body is not None})."
                    )
                # De-articulate the peg so PhysX can fold it into the Franka articulation.
                _strip_articulation_root(held_roots[env_idx])
                state["peg_body_name"] = peg_body.GetName()
                env_root = held_roots[env_idx].GetParent().GetPath().pathString
                joint_path = f"{env_root}/{_WELD_PRIM_NAME}"
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets([Sdf.Path(fingertip.GetPath().pathString)])
                joint.CreateBody1Rel().SetTargets([Sdf.Path(peg_body.GetPath().pathString)])
                # excludeFromArticulation=FALSE: fold the peg in as a reduced-coordinate Franka
                # link (NOT a maximal-coordinate loop joint — that variant leaks in PhysX).
                joint.CreateExcludeFromArticulationAttr().Set(False)
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in pos0)))
                joint.CreateLocalRot0Attr().Set(
                    Gf.Quatf(float(quat0[0]), Gf.Vec3f(float(quat0[1]), float(quat0[2]), float(quat0[3])))
                )
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                n_made += 1
            print(
                f"[peg-weld] folded peg ({state['peg_body_name']!r}) into the Franka articulation as a "
                f"link on the env_{sorted(robot_roots)} prototype ({n_made} joint(s)); replicate_physics "
                f"propagates to all envs (Fabric-compatible, reduced-coordinate, no loop joint); "
                f"peg_in_fingertip pos={[round(v, 6) for v in pos0]}, quat_wxyz={[round(v, 6) for v in quat0]} "
                f"(fixed tilt {list(rel_grasp_rot_init_deg)} deg).",
                flush=True,
            )

        # Fold the peg in on the env-0 prototype right BEFORE the clone (only the prototype exists
        # then, and PhysX will still re-parse the articulation). Restore-then-call so this fires
        # once and composes with other _setup_scene shims.
        _orig_clone = self.scene.clone_environments

        def _clone_with_link(*args, **kwargs):
            self.scene.clone_environments = _orig_clone
            _author_link_on_prototype()
            return _orig_clone(*args, **kwargs)

        self.scene.clone_environments = _clone_with_link
        _original_setup_scene(self)  # spawns assets, clones (with the fold), re-registers held_asset

        # The peg is a Franka link now, so the separate held_asset articulation must NOT initialize
        # (it has no ArticulationRootAPI). Cancel its play-init callbacks, drop it from the scene,
        # and replace it with a shim that serves the peg pose from the Franka articulation.
        orig_held = self._held_asset
        for _h in ("_initialize_handle", "_invalidate_initialize_handle"):
            handle = getattr(orig_held, _h, None)
            if handle is not None:
                try:
                    handle.unsubscribe()
                except Exception:
                    pass
                setattr(orig_held, _h, None)
        self.scene._articulations.pop("held_asset", None)
        # The surface task registers the held cylinder as a RigidObject (procedural primitive), not
        # an Articulation, so it lives in scene._rigid_objects — drop it from there too.
        if hasattr(self.scene, "_rigid_objects"):
            self.scene._rigid_objects.pop("held_asset", None)
        self._held_asset = _HeldLinkShim(self._robot, state["peg_body_name"], self.num_envs, self.device)
        print(
            f"[peg-weld] replaced held_asset articulation with a Franka-link shim "
            f"(peg body {state['peg_body_name']!r}); reset teleport is now a no-op (the link holds it).",
            flush=True,
        )

    env_class._setup_scene = _patched_setup_scene
    print(
        f"[peg-weld] {env_class.__name__}._setup_scene patched: held asset folded into the Franka "
        "articulation as a rigid link before clone (Fabric-compatible; reduced-coordinate; no "
        "loop-joint leak).",
        flush=True,
    )
