"""In-contact boolean from an IsaacLab ContactSensor (Forge/Factory peg insertion).

This is the "contact-sensor version" of the upstream Continuous_Force_RL contact
detection: it reads an IsaacLab :class:`~isaaclab.sensors.ContactSensor` mounted on
the held asset (peg) and filtered against the fixed asset (hole), instead of the
robot's joint force-torque reading. The sensor reports a per-pair contact-force
vector (``data.force_matrix_w``); there is no native per-direction boolean, so a 3-D
*in-contact* flag (one per task-space translation axis x/y/z) is produced by rotating
that force from world into the end-effector frame and thresholding each component.

Logging-only: the per-axis booleans are stashed on ``env.unwrapped.in_contact``
(shape ``(num_envs, 3)``) and published as per-env float tensors under
``extras['to_log']`` (``Contact / In-Contact {X,Y,Z,Any}``), which
:class:`~wrappers.scorers.reward_decomposition.RewardDecompositionWrapper`
mean-reduces and forwards to TensorBoard as fraction-of-time-in-contact metrics.
The observation / state spaces are NOT changed.

Two pieces, because an IsaacLab sensor must be registered with the scene *during*
env construction (before ``InteractiveScene.clone_environments``):

  * :func:`install_contact_sensor` — patches ``FactoryEnv._setup_scene`` to create +
    register the ContactSensor (mirrors the runner's recorder-camera install). Call
    BEFORE ``gym.make``. Forge inherits ``_setup_scene`` from Factory, so patching
    Factory covers both.
  * :class:`ContactSensorWrapper` — a ``gym.Wrapper`` (same layer as the control
    wrappers) that reads the live sensor each ``step()`` and logs.
"""

from __future__ import annotations

import re
from typing import Any

import gymnasium as gym
import torch

# Scene key the sensor is registered under (shared by installer + wrapper).
CONTACT_SENSOR_KEY = "held_fixed_contact_sensor"
_AXIS_NAMES = ("X", "Y", "Z")
# Per-axis contact width produced by the ContactSensor (one flag per translation axis).
CONTACT_DIM = len(_AXIS_NAMES)
# Matches the per-env index in a cloned prim path (e.g. ".../env_0/...") so a concrete
# env-0 prim path can be turned back into the cross-env ".../env_.*/..." regex the
# ContactSensor expects.
_ENV_IDX_RE = re.compile(r"/env_\d+/")


def make_contact_sensor_cfg(held_prim_expr: str, fixed_prim_expr: str):
    """Build the IsaacLab ``ContactSensorCfg`` for held-vs-fixed contact reporting.

    ``update_period=0.0`` -> refresh every physics step; ``history_length=0`` -> only
    the latest reading; ``filter_prim_paths_expr`` is required so ``force_matrix_w``
    (per-pair contact force) is populated.
    """
    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(
        prim_path=held_prim_expr,
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        filter_prim_paths_expr=[fixed_prim_expr],
        track_air_time=False,
    )


def _resolve_contact_body_expr(root_expr: str) -> str:
    """Resolve an asset-root prim expr to the rigid body that carries the contact API.

    ``activate_contact_sensors=True`` applies the PhysX contact-report API to the *rigid
    body* of the spawned asset, which for the Factory/Forge held (peg) and fixed (hole)
    assets is a CHILD prim of the ``HeldAsset``/``FixedAsset`` articulation root, not the
    root itself (the peg sits one level down, the hole two). A ContactSensor whose
    ``prim_path`` points at the root therefore finds no contact-reporting body and raises
    at init. The exact child name is asset-specific (``factory_peg_8mm``,
    ``factory_gear_medium``, asset-variant pegs, ...), so instead of hardcoding it we walk
    the already-cloned stage under ``root_expr`` (env-0) and return the path of the first
    descendant carrying ``PhysxContactReportAPI``, rewritten back to the cross-env
    ``.../env_.*/...`` regex form. Falls back to ``root_expr`` unchanged if none is found
    (the ContactSensor then raises its own descriptive error).
    """
    from pxr import PhysxSchema

    import isaaclab.sim as sim_utils

    matches = sim_utils.find_matching_prims(root_expr)
    if not matches:
        return root_expr
    # BFS over the env-0 asset subtree (root included) for the contact-reporting body.
    queue = list(matches[0:1])
    while queue:
        prim = queue.pop(0)
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            return _ENV_IDX_RE.sub("/env_.*/", prim.GetPath().pathString)
        queue.extend(prim.GetChildren())
    return root_expr


def install_contact_sensor(contact_cfg: Any, env_class: Any = None) -> None:
    """Patch ``<env_class>._setup_scene`` to register the ContactSensor before clone.

    Mirrors the recorder-camera install in the runner: wrap the first
    ``clone_environments`` call so the sensor is created and registered into
    ``scene.sensors`` right after env cloning. The held/fixed prims already exist by
    then (Factory spawns them earlier in ``_setup_scene``) and were spawned with
    ``activate_contact_sensors=True``, so the ContactSensor simply attaches to them.

    ``env_class`` is the direct-API env class whose ``_setup_scene`` actually runs at
    ``gym.make`` for the task. Defaults to ``FactoryEnv`` (covers Factory + Forge, which
    inherits ``_setup_scene``). AutoMate's ``AssemblyEnv`` does NOT subclass ``FactoryEnv``
    and defines its own ``_setup_scene`` (it likewise calls ``clone_environments`` last and
    uses ``HeldAsset``/``FixedAsset`` prims with ``activate_contact_sensors=True``), so the
    runner passes ``env_class=AssemblyEnv`` for ``Isaac-AutoMate-*`` tasks.

    The configured ``held_prim_expr``/``fixed_prim_expr`` name the asset *roots*; the
    contact-reporting rigid body is a child prim whose name is asset-specific, so the
    actual sensor/filter paths are resolved via :func:`_resolve_contact_body_expr` from
    the (env-0) prims that the env spawns earlier in ``_setup_scene``.

    The sensor is created and registered into ``scene.sensors`` BEFORE the wrapped
    ``clone_environments`` call: the per-env rigid-contact views are set up during
    physics replication, so a sensor registered only after cloning would see just env-0's
    body (``body_view.count != num_envs``) and fail its init check. Resolution still
    happens here (not up front) because the assets must already be spawned to walk for
    the child body — they are, since cloning is the last step of ``_setup_scene``.

    Composes with the recorder's own ``_setup_scene`` patch: each ``clone_environments``
    shim restores the previously-captured clone and calls it, so both shims fire.
    """
    from isaaclab.sensors import ContactSensor

    if env_class is None:
        from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

        env_class = FactoryEnv

    _original_setup_scene = env_class._setup_scene

    def _patched_setup_scene(self):
        _orig_clone = self.scene.clone_environments

        def _shim_clone(*args, **kwargs):
            # Restore first so this fires exactly once.
            self.scene.clone_environments = _orig_clone
            # Resolve + register the sensor BEFORE cloning so its rigid-contact views are
            # replicated to every env (assets are already spawned at this point).
            held_expr = _resolve_contact_body_expr(contact_cfg.held_prim_expr)
            fixed_expr = _resolve_contact_body_expr(contact_cfg.fixed_prim_expr)
            sensor_cfg = make_contact_sensor_cfg(held_expr, fixed_expr)
            self.scene.sensors[CONTACT_SENSOR_KEY] = ContactSensor(sensor_cfg)
            print(
                f"[contact-sensor] registered ContactSensor '{CONTACT_SENSOR_KEY}' on "
                f"{sensor_cfg.prim_path} (filter {sensor_cfg.filter_prim_paths_expr})",
                flush=True,
            )
            return _orig_clone(*args, **kwargs)

        self.scene.clone_environments = _shim_clone
        return _original_setup_scene(self)

    env_class._setup_scene = _patched_setup_scene
    print(
        f"[contact-sensor] {env_class.__name__}._setup_scene patched to install "
        f"'{CONTACT_SENSOR_KEY}' (threshold={contact_cfg.contact_force_threshold} N)."
    )


class ContactSensorWrapper(gym.Wrapper):
    """Per-step in-contact boolean from the held-vs-fixed ContactSensor (logging-only)."""

    def __init__(self, env, contact_cfg: Any) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self._threshold = float(contact_cfg.contact_force_threshold)
        self._log_contact = bool(contact_cfg.log_contact_state)

        self._sensor = None
        self._initialized = False

        # Expose the raw flag on the env so a future change can feed it to obs/state.
        self.unwrapped.in_contact = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        if hasattr(self.unwrapped, "extras") and "to_log" not in self.unwrapped.extras:
            self.unwrapped.extras["to_log"] = {}

    # ------------------------------------------------------------------ setup
    def _initialize(self) -> None:
        if self._initialized:
            return
        sensors = getattr(self.unwrapped.scene, "sensors", None)
        if not sensors or CONTACT_SENSOR_KEY not in sensors:
            raise RuntimeError(
                f"ContactSensorWrapper: scene has no sensor '{CONTACT_SENSOR_KEY}'. "
                "Ensure the runner calls install_contact_sensor(...) BEFORE gym.make() "
                "(the sensor must be registered during env construction)."
            )
        self._sensor = sensors[CONTACT_SENSOR_KEY]
        self._initialized = True

    # --------------------------------------------------------------- contact
    def _update_contact(self) -> None:
        from isaaclab.utils.math import quat_rotate_inverse

        fm = self._sensor.data.force_matrix_w  # (num_envs, B, M, 3) or None before first update
        if fm is None:
            return
        # Body index 0 = the (single) held-asset body; sum over filtered bodies (M) ->
        # world-frame net contact force (num_envs, 3).
        force_w = fm[:, 0, :, :].sum(dim=1)
        # Rotate into the EE / force-torque frame (matches the hybrid-control frame) so
        # the per-axis flags align with the task-space control axes.
        force_ee = quat_rotate_inverse(self.unwrapped.fingertip_midpoint_quat, force_w)
        in_contact = force_ee.abs() > self._threshold  # (num_envs, 3) bool
        self.unwrapped.in_contact = in_contact

        if self._log_contact and hasattr(self.unwrapped, "extras"):
            to_log = self.unwrapped.extras.setdefault("to_log", {})
            for i, name in enumerate(_AXIS_NAMES):
                to_log[f"Contact / In-Contact {name}"] = in_contact[:, i].float()
            to_log["Contact / In-Contact Any"] = in_contact.any(dim=1).float()

    # ------------------------------------------------------------------ gym
    def step(self, action):
        if not self._initialized and hasattr(self.unwrapped, "scene"):
            self._initialize()
        out = super().step(action)
        # After super().step(), the scene (and thus the contact sensor) has been
        # updated for this step, and fingertip_midpoint_quat reflects the current pose.
        if self._initialized:
            self._update_contact()
        # Expose the RAW per-axis contact bool (num_envs, 3) on the info dict so agents
        # can buffer it as a supervised-selection target / append it as an input feature.
        # Separate key from the to_log / per_env_to_log logging path; the outer scorer
        # wrapper passes info through unchanged. Absent until the sensor's first update.
        in_contact = getattr(self.unwrapped, "in_contact", None)
        if in_contact is not None and isinstance(out[4], dict):
            out[4]["in_contact"] = in_contact.float()
        return out

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        if not self._initialized and hasattr(self.unwrapped, "scene"):
            self._initialize()
        return out
