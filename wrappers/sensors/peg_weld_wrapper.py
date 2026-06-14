"""Rigidly weld the held peg to the gripper (Forge/Factory peg insertion).

Motivation: by default the peg is held in the closed gripper by *friction only*
(``factory_env.randomize_initial_state`` teleports it into the fingers with gravity off,
then closes the gripper — there is no joint). Under a contact-rich insertion policy the
peg can creep / rotate in the grip during execution. This installer adds a true
``UsdPhysics.FixedJoint`` per env between the fingertip body and the peg body so the peg
is rigidly mounted to the gripper for the whole rollout.

Why a fixed joint and not per-step pose-pinning: a welded peg stays a fully dynamic body,
so its peg-vs-socket reaction propagates *through the joint* into the arm — the wrist /
contact-force signals the controller relies on are preserved (re-teleporting the peg each
step would zero that reaction and corrupt the force feedback). See the discussion in the
project notes; this is "option C".

HARD CONSTRAINTS (enforced by the caller, ``learning/env_setup.py``):

  * GPU pipeline limitation: a joint's kinematic local frame is USD-authored and parsed by
    PhysX at play time — it CANNOT be changed between physics steps. So the welded
    peg-in-gripper transform must be a CONSTANT, fixed before ``sim.reset()``. That is only
    well-defined when the grasp is deterministic, hence this is allowed ONLY when
    ``grasp_rot_mode == "fixed"`` (a constant signed tilt, identical every env every reset).
  * ``held_asset_pos_noise`` must be zeroed (the caller does this): the native reset still
    teleports the peg into the grip every reset, and a nonzero per-reset position jitter
    would disagree with the constant weld frame and make PhysX fight the teleport (force
    spikes). With it zeroed, the reset teleport target equals the weld frame, so the two
    agree and the reset is conflict-free.
  * ``peg_insert`` task only — the analytic weld transform below mirrors the ``peg_insert``
    branch of ``get_handheld_asset_relative_pose``; ``gear_mesh`` / ``nut_thread`` have
    different base transforms.

The weld transform is NOT guessed from live poses (unavailable pre-play): it is computed
analytically as ``flip_z ∘ asset_in_hand`` — exactly the peg-relative-to-fingertip transform
that ``randomize_initial_state`` builds (``factory_env.py:744-760``), including the fixed
grasp tilt (composed the same way as ``grasp_tilt_wrapper``). Because it reuses the env's own
seating math, the weld frame and the reset teleport coincide by construction.

Single-installer, mirroring :mod:`wrappers.sensors.contact_sensor_wrapper`: patch the env
class's ``_setup_scene`` so the joints are authored after ``clone_environments`` (real per-env
prims exist) but before play (the only window PhysX will parse new joints on the GPU
pipeline). Call BEFORE ``gym.make``. Forge inherits ``_setup_scene`` from Factory, so patching
the concrete env class the runner resolves covers the Forge tasks.
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
# Joint prim name authored under each env's HeldAsset subtree.
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
    rel_pos = torch.zeros((1, 3), device=device)
    rel_pos[0, 2] = height - fingerpad
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

    # Fixed grasp tilt, composed onto the (identity) base exactly as grasp_tilt_wrapper:
    # rel_quat = quat_mul(quat_conjugate(perturb), base).
    r, p, y = (np.deg2rad(float(v)) for v in tilt_deg)
    perturb = torch_utils.quat_from_euler_xyz(
        torch.tensor([r], device=device),
        torch.tensor([p], device=device),
        torch.tensor([y], device=device),
    )
    rel_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(perturb), identity)

    # asset_in_hand = inverse(held_asset_relative); peg_in_fingertip = flip_z ∘ asset_in_hand.
    ah_quat, ah_pos = torch_utils.tf_inverse(rel_quat, rel_pos)
    flip_z = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    zeros3 = torch.zeros((1, 3), device=device)
    p0_quat, p0_pos = torch_utils.tf_combine(flip_z, zeros3, ah_quat, ah_pos)
    return p0_pos[0].tolist(), p0_quat[0].tolist()  # ([x,y,z], [w,x,y,z])


def install_peg_weld(rel_grasp_rot_init_deg: Sequence[float], env_class: Any = None) -> None:
    """Patch ``<env_class>._setup_scene`` to weld the peg to the gripper, one joint per env.

    :param rel_grasp_rot_init_deg: the ``"fixed"`` ``[roll, pitch, yaw]`` (deg) grasp tilt,
        folded into the weld transform so the weld matches the tilted grasp the env seats.
    :param env_class: the concrete direct-API env class whose ``_setup_scene`` runs at
        ``gym.make`` (defaults to ``FactoryEnv``; Forge inherits it).

    Authors the joints AFTER the original ``_setup_scene`` (so ``clone_environments`` has
    created real per-env prims) and BEFORE play (still inside env construction), which is the
    only window PhysX parses new joints on the GPU pipeline. Composes with the contact-sensor
    ``_setup_scene`` patch (each wraps the previous).
    """
    if env_class is None:
        from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

        env_class = FactoryEnv

    _original_setup_scene = env_class._setup_scene

    def _patched_setup_scene(self):
        _original_setup_scene(self)  # spawn assets, clone_environments, lights — all pre-play

        if self.cfg_task.name != "peg_insert":
            raise ValueError(
                "install_peg_weld supports the 'peg_insert' task only (the weld transform "
                f"mirrors the peg_insert grasp), but cfg_task.name is {self.cfg_task.name!r}."
            )

        import isaaclab.sim as sim_utils
        from pxr import Gf, Sdf, UsdPhysics

        pos0, quat0 = _compute_peg_in_fingertip(self, rel_grasp_rot_init_deg)  # ([xyz],[wxyz])

        robot_expr = self.cfg.robot.prim_path  # "/World/envs/env_.*/Robot"
        held_expr = self.cfg_task.held_asset.prim_path  # "/World/envs/env_.*/HeldAsset"
        robot_roots = {_env_index(p.GetPath().pathString): p for p in sim_utils.find_matching_prims(robot_expr)}
        held_roots = {_env_index(p.GetPath().pathString): p for p in sim_utils.find_matching_prims(held_expr)}
        if not robot_roots or set(robot_roots) != set(held_roots):
            raise RuntimeError(
                "install_peg_weld: could not pair per-env Robot/HeldAsset prims "
                f"(robot envs {sorted(robot_roots)}, held envs {sorted(held_roots)}). "
                "Is clone_in_fabric disabled? (fabric clones are USD-invisible)."
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
            # Author the joint under the env ROOT (not under either articulation subtree) so PhysX
            # treats it as a maximal loop joint between the two articulations via its Body0/Body1
            # rels, rather than trying to absorb it into the held-asset articulation.
            env_root = held_roots[env_idx].GetParent().GetPath().pathString
            joint_path = f"{env_root}/{_WELD_PRIM_NAME}"
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(fingertip.GetPath().pathString)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(peg_body.GetPath().pathString)])
            # localPose0 = peg_in_fingertip (frame on the fingertip that the joint pins to the
            # peg's body origin); localPose1 = identity (peg body origin == its root pose).
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in pos0)))
            joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(quat0[0]), Gf.Vec3f(float(quat0[1]), float(quat0[2]), float(quat0[3])))
            )
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            n_made += 1

        print(
            f"[peg-weld] authored {n_made} FixedJoint(s) welding {_FINGERTIP_BODY_NAME} -> peg "
            f"body; peg_in_fingertip pos={ [round(v, 6) for v in pos0] }, "
            f"quat_wxyz={ [round(v, 6) for v in quat0] } (fixed tilt {list(rel_grasp_rot_init_deg)} deg).",
            flush=True,
        )

    env_class._setup_scene = _patched_setup_scene
    print(
        "[peg-weld] FactoryEnv._setup_scene patched: peg will be rigidly welded to the gripper "
        "(constant grasp, GPU-safe pre-play authoring).",
        flush=True,
    )
