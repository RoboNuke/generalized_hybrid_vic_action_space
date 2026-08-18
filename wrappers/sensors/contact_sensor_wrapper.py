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
    """Per-step held-vs-fixed in-contact boolean, read from a Fabric-compatible PhysX rigid
    contact view that this wrapper creates itself.

    Why not IsaacLab's ``ContactSensor``: its ``_initialize_impl`` builds the PhysX view at scene
    setup, where under ``clone_in_fabric=True`` the replicated per-env bodies aren't yet resolvable,
    so it sees only env-0 and fails its body-count check (this is what forced ``clone_in_fabric=False``
    and caused the host-RAM leak via per-step USD ops on real prims). Instead we create the
    ``rigid_contact_view`` LAZILY on the first step — once the sim is live the peg/hole bodies are
    resolvable by a clean ``/World/envs/env_*/.../body`` glob — and read ``get_contact_force_matrix``,
    the same per-pair force the stock sensor exposed as ``force_matrix_w``.
    """

    def __init__(self, env, contact_cfg: Any) -> None:
        super().__init__(env)
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self._threshold = float(contact_cfg.contact_force_threshold)
        self._log_contact = bool(contact_cfg.log_contact_state)
        # EMA-smooth the (unnoised) contact force before thresholding into in_contact, so the contact
        # signal matches the smoothing on the Forge wrist force the break check reads (both then use
        # the unnoised + EMA'd force). Same alpha as the Forge force sensor (ft_smoothing_factor,
        # default 0.25); updated once per env-step, and reset per-episode so a stale reading from the
        # previous episode never carries into a fresh spawn.
        # EMA alpha for the contact (oracle) force — SAME as the Forge wrist-FT EMA
        # (ft_smoothing_factor). The env drives _refresh_in_contact once per PHYSICS SUBSTEP (see
        # flat_surface_follow_env._compute_intermediate_values), so this EMA averages over the
        # substeps exactly like the wrist FT (matched filtering: same alpha, same update rate, so it
        # scales with the physics rate the same way). A sim-timestamp gate (_last_refresh_ts) makes
        # the refresh idempotent, so a redundant post-step call from step() is a no-op.
        self._ema_alpha = float(getattr(getattr(self.unwrapped, "cfg", None), "ft_smoothing_factor", 0.25))
        self._last_refresh_ts = None
        self._force_ema = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # DIAGNOSTIC (force-source noise probe): world-frame EMA of the contact force + last raw
        # reading, published each step into extras["per_env_trace"] alongside the wrist FT so an
        # eval trace can compare the two sources' per-axis mean/std. Cheap; safe to leave in.
        self._force_ema_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._last_contact_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # Oracle contact force (world frame, EMA'd) exposed to the env for the "oracle" force_source
        # (reward + fragile break read this instead of the wrist FT). Zeros until the view is live.
        self.unwrapped.contact_force_world_ema = self._force_ema_w

        # Resolve the held (sensor) + fixed (filter) contact-reporting body paths from the env-0
        # prototype (a real USD prim at this point) and put them in PhysX GLOB form ('env_*', no
        # regex groups), which resolves ALL envs under clone_in_fabric=True.
        self._peg_glob = _resolve_contact_body_expr(contact_cfg.held_prim_expr).replace(".*", "*")
        self._hole_glob = _resolve_contact_body_expr(contact_cfg.fixed_prim_expr).replace(".*", "*")
        self._contact_view = None
        self._dt = None

        # Exposed so the obs/append + phase-split paths can read it; zeros until the view is live.
        self.unwrapped.in_contact = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        # Hook the env can call to refresh `in_contact` OUTSIDE this wrapper's step() — e.g. the
        # surface task's reset-time press-to-contact, which settles via step_sim_no_action (the
        # wrapper's step() never runs during a reset, so `in_contact` would otherwise stay stale).
        # Returns True when the live view produced a fresh reading; no logging side effects.
        self.unwrapped._refresh_in_contact = self._refresh_in_contact
        if hasattr(self.unwrapped, "extras") and "to_log" not in self.unwrapped.extras:
            self.unwrapped.extras["to_log"] = {}

    # ------------------------------------------------------------------ setup
    def _ensure_view(self) -> bool:
        """Create the rigid contact view once the sim is live (Fabric bodies then resolvable)."""
        if self._contact_view is not None:
            return True
        from isaacsim.core.simulation_manager import SimulationManager

        psv = SimulationManager.get_physics_sim_view()
        if psv is None:
            return False
        # Extra filter bodies the peg can contact besides the plate (e.g. the curved-ridge CAPS or the
        # bumpy BUMPS — separate rigid bodies with activate_contact_sensors=True). The task env exposes
        # them as PhysX globs via ``_extra_contact_filter_globs``; adding them to filter_patterns makes
        # the SAME oracle contact force include peg<->{cap,bump} contact, since _refresh_in_contact sums
        # get_contact_force_matrix over ALL filters. No force_source switch, no FT-force detection hacks.
        extra = list(getattr(self.unwrapped, "_extra_contact_filter_globs", None) or [])
        try:
            cv = psv.create_rigid_contact_view(
                self._peg_glob,
                filter_patterns=[self._hole_glob, *extra],
                # Cap on simultaneous peg<->hole contact POINTS PhysX aggregates into the force
                # matrix each step. The prior 64*num_envs was an untuned over-guess from the
                # leak-fix rewrite; the native contact-data collection scales with this, and host
                # RSS leaks ~proportional to actual contacts during insertion. We only threshold
                # the summed force into a per-axis in_contact BOOLEAN (the FT feedback the policy
                # uses is Forge's separate wrist force_sensor, not this), so 4/env — IsaacLab's
                # stock default — is plenty and cannot affect force feedback.
                max_contact_data_count=4 * self.num_envs,
            )
        except Exception:
            return False  # bodies not resolvable yet — retry next step
        if cv is None or cv.sensor_count != self.num_envs:
            return False
        self._contact_view = cv
        self._dt = float(self.unwrapped.sim.get_physics_dt())
        print(
            f"[contact-sensor] Fabric contact view ready: sensor_count={cv.sensor_count} "
            f"peg={self._peg_glob} filters={[self._hole_glob, *extra]}",
            flush=True,
        )
        return True

    # --------------------------------------------------------------- contact
    def _refresh_in_contact(self) -> bool:
        """Recompute ``env.in_contact`` from the live contact view. No logging side effects, so
        it is safe to call mid-reset (e.g. the surface task's press-to-contact). Returns True if
        the view was ready and ``in_contact`` was updated, False otherwise (view not yet live)."""
        from isaaclab.utils.math import quat_rotate_inverse

        if not self._ensure_view():
            return False
        # Idempotency gate: the env calls this once per PHYSICS SUBSTEP; the wrapper's post-step
        # step() also calls it. Only recompute when the sim has actually advanced since the last
        # refresh, so a substep is EMA'd exactly once (mirrors Factory's per-substep FT-update gate).
        try:
            _ts = float(self.unwrapped._robot._data._sim_timestamp)
            if self._last_refresh_ts is not None and _ts <= self._last_refresh_ts:
                return True
            self._last_refresh_ts = _ts
        except Exception:
            pass  # no robot timestamp available -> fall back to unconditional refresh
        # (num_envs, M_filters, 3) per-pair contact force; sum over filtered bodies -> world net.
        fmat = self._contact_view.get_contact_force_matrix(self._dt)
        force_w = fmat.sum(dim=1)
        # Rotate into the EE / force-torque frame so per-axis flags align with the control axes.
        force_ee = quat_rotate_inverse(self.unwrapped.fingertip_midpoint_quat, force_w)
        # DIAGNOSTIC: keep the raw world contact force + a world-frame EMA (same alpha / seeding as
        # the EE thresholding EMA below) so the trace can report the contact-source noise directly.
        self._last_contact_w = force_w
        _buf = getattr(self.unwrapped, "episode_length_buf", None)
        _fresh = (_buf <= 1).unsqueeze(-1) if torch.is_tensor(_buf) else True
        _blended_w = self._ema_alpha * force_w + (1.0 - self._ema_alpha) * self._force_ema_w
        self._force_ema_w = torch.where(_fresh, force_w, _blended_w) if torch.is_tensor(_buf) else force_w
        self.unwrapped.contact_force_world_ema = self._force_ema_w  # expose the fresh oracle to the env
        # EMA-smooth (unnoised) before thresholding. Envs at the very start of an episode
        # (episode_length_buf <= 1) seed the EMA with the raw reading so no history bleeds across the
        # per-episode reset; everyone else blends alpha*raw + (1-alpha)*prev.
        buf = getattr(self.unwrapped, "episode_length_buf", None)
        fresh = (buf <= 1).unsqueeze(-1) if torch.is_tensor(buf) else True
        blended = self._ema_alpha * force_ee + (1.0 - self._ema_alpha) * self._force_ema
        self._force_ema = torch.where(fresh, force_ee, blended) if torch.is_tensor(buf) else force_ee
        self.unwrapped.in_contact = self._force_ema.abs() > self._threshold  # (num_envs, 3) bool
        return True

    def _update_contact(self) -> None:
        if not self._refresh_in_contact():
            return
        in_contact = self.unwrapped.in_contact
        # DIAGNOSTIC force-source probe: publish both sources (raw + EMA, world frame) + the surface
        # normal + per-axis in_contact into the per-env trace the StepTraceRecorder captures. The env
        # rebuilt per_env_trace this step (in _get_observations); we merge, so nothing is clobbered.
        ex = getattr(self.unwrapped, "extras", None)
        pt = ex.get("per_env_trace") if isinstance(ex, dict) else None
        if isinstance(pt, dict):
            u = self.unwrapped
            if hasattr(u, "force_sensor_world"):
                pt["dbg_wrist_ft_w_raw"] = u.force_sensor_world[:, 0:3].detach()
            if hasattr(u, "force_sensor_world_smooth"):
                pt["dbg_wrist_ft_w_ema"] = u.force_sensor_world_smooth[:, 0:3].detach()
            pt["dbg_contact_w_raw"] = self._last_contact_w.detach()
            pt["dbg_contact_w_ema"] = self._force_ema_w.detach()
            if hasattr(u, "surface_normal") and torch.is_tensor(u.surface_normal):
                pt["dbg_surface_normal"] = u.surface_normal.detach()
            pt["dbg_in_contact_axes"] = in_contact.float()
        if self._log_contact and hasattr(self.unwrapped, "extras"):
            to_log = self.unwrapped.extras.setdefault("to_log", {})
            for i, name in enumerate(_AXIS_NAMES):
                to_log[f"Contact / In-Contact {name}"] = in_contact[:, i].float()
            to_log["Contact / In-Contact Any"] = in_contact.any(dim=1).float()

    # ------------------------------------------------------------------ gym
    def step(self, action):
        out = super().step(action)
        # After super().step() the physics has advanced and fingertip_midpoint_quat is current.
        self._update_contact()
        # Expose the RAW per-axis contact bool on the info dict (SSL target / obs feature).
        in_contact = getattr(self.unwrapped, "in_contact", None)
        if in_contact is not None and isinstance(out[4], dict):
            out[4]["in_contact"] = in_contact.float()
        return out

    def reset(self, **kwargs):
        # Full reset re-spawns every env out of contact — clear the smoothed-force history so the
        # first-step reading of the new episode starts fresh (the per-env buf<=1 seed handles
        # mid-episode resets).
        self._force_ema.zero_()
        self._force_ema_w.zero_()
        self._last_refresh_ts = None
        return super().reset(**kwargs)
