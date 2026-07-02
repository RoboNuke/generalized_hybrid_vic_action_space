"""Sampling-Based Curriculum (SBC) for Forge/Factory peg-insertion resets.

Implements the IndustReal/AutoMate SBC (see AutoMate, arXiv:2407.08028): rather than fixing every
episode at a single difficulty (which biases the early policy to an over-simplified task), each
reset samples the initial peg pose from ``[floor, max]`` where ``max`` is the FULL task (constant)
and only the ``floor`` rises with the curriculum level ``c``:

    floor_axis = min_axis + c * (max_axis - min_axis)
    v_axis     ~ U(floor_axis, max_axis)     # sampled INDEPENDENTLY per env, per axis

So at ``c=0`` the floor is the configured ``min`` and the agent sees the ENTIRE range (easy ->
hardest); as ``c -> 1`` the floor rises to ``max`` and only the hardest spawns remain. ``max`` never
changes, so the true task is always represented. Each of the 6 axes draws independently, so one env
can be hard in one axis and easy in another.

Axes (each sampled independently each reset), with per-axis task ``max``:
  * pos x, y — lateral offset (applied ±)   max = cfg_task.hand_init_pos_noise[:2]
  * pos z    — peg height above hole tip     max = cfg_task.hand_init_pos[2]
  * orn roll/pitch/yaw — peg tilt (signed)   max = grasp_tilt_deg[roll,pitch,yaw] (the fixed grasp)
``min_pos`` / ``min_orn_deg`` are 3-vectors (same order/convention as ``hand_init_pos_noise`` and
``rel_grasp_rot_init_deg``). Only rotation axes with a nonzero grasp tilt are curriculumed (an axis
with grasp max 0 stays 0).

PER-AGENT ``c`` (NOT global, NOT per-env): the block-parallel run trains ``num_agents`` policies on
contiguous env partitions (agent ``a`` owns envs ``[a*epa, (a+1)*epa)``, ``epa = num_envs //
num_agents`` — matching ``learning/block_agent.py``). Each agent keeps its OWN ``c`` and success
EMA, updated from ITS OWN envs, and every env in that agent's partition uses that agent's ``c``.
Per-agent (vs global) so a fast seed can't raise a slow seed's difficulty and freeze it. Published
per-agent to ``extras['to_log']['ResetCurriculum/c']`` / ``.../success_ema`` (per-env tensors, so
block_agent's per-agent partitioning logs each agent's own value).

``c`` update (per reset batch, per agent, from the just-ended ``ep_succeeded`` latch):
    ema_a = (1-beta)*ema_a + beta*mean(success over agent a's resetting envs)
    if ema_a > increase_threshold: c_a = min(1, c_a + increase_rate)
    elif ema_a < decrease_threshold: c_a = max(0, c_a - decrease_rate)

Tilt realisation (GLUED mode): the peg-in-gripper weld is a fixed rotation ``R_grasp`` and cannot
change per reset. To land the peg at a sampled tilt ``R_samp`` (relative to vertical), pre-rotate the
reset gripper so ``gripper ∘ R_grasp == hand_down ∘ R_samp`` -> gripper = ``hand_down ∘ R_samp ∘
R_grasp^-1`` (R_samp = R_grasp => gripper straight down => full tilt; R_samp = I => gripper cocked
back => peg vertical).

Hook: wraps ``FactoryEnv.randomize_initial_state`` (call BEFORE gym.make); lets the stock reset run,
then repositions the gripper (glued: welded peg rides with it) to the sampled pose via the env's own
``set_pos_inverse_kinematics``. Composes with the weld + efficient-reset (both untouched).

!!! SIM-VALIDATION REQUIRED (untestable outside Isaac Sim) — see the smoke test !!!
  1. IK reachability at c=0 (gripper cocked back near the table): confirm reset IK-retry doesn't spike.
  2. tilt frame/sign of ``hand_down ∘ R_samp ∘ R_grasp^-1``: confirm the spawned peg-vs-socket angle
     tracks the sampled tilt.
  3. lateral offset is on the FINGERTIP; the welded peg tip is offset by the grasp lever (~1-2cm).
"""

from __future__ import annotations

from typing import Sequence


def install_reset_curriculum(
    *,
    num_agents: int,
    min_pos: Sequence[float],
    min_orn_deg: Sequence[float],
    increase_rate: float,
    decrease_rate: float,
    increase_threshold: float,
    decrease_threshold: float,
    grasp_tilt_deg: Sequence[float],
    align_below_z: float = 0.0,
    success_margin_z: float = 0.005,
    success_ema_beta: float = 0.05,
    ik_iters: int = 3,
) -> None:
    """Patch ``FactoryEnv.randomize_initial_state`` to apply per-agent SBC to the initial peg pose.

    :param num_agents: number of block-parallel agents (env partitions); ``c`` is per agent.
    :param min_pos: easy-end (c=0 floor) ``[x, y, z]`` offsets in METERS (x,y lateral magnitudes; z height).
    :param min_orn_deg: easy-end ``[roll, pitch, yaw]`` peg tilt in DEGREES (``[0,0,0]`` => aligned).
    :param grasp_tilt_deg: fixed grasp tilt ``[roll, pitch, yaw]`` (deg) = the per-axis tilt ``max``.
    """
    import torch
    import isaacsim.core.utils.torch as torch_utils

    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

    n_ag = int(num_agents)
    min_pos_t = [float(v) for v in min_pos]                       # [x, y, z] meters
    grasp_rad = [float(torch.deg2rad(torch.tensor(float(v)))) for v in grasp_tilt_deg]   # [r,p,y]
    min_orn_rad = [float(torch.deg2rad(torch.tensor(float(v)))) for v in min_orn_deg]

    _orig = FactoryEnv.randomize_initial_state
    state = {"c": None, "ema": None, "epa": None, "z_off": None, "z_floor_succ": None}

    def _patched(self, env_ids):
        device = self.device
        if state["c"] is None:
            state["epa"] = max(1, self.num_envs // n_ag)
            state["c"] = [0.0] * n_ag
            state["ema"] = [0.0] * n_ag
        epa = state["epa"]

        # --- (a) per-agent c update from the just-ended episodes (BEFORE stock reset clears them) ---
        if hasattr(self, "ep_succeeded") and len(env_ids) > 0:
            ag = torch.div(env_ids, epa, rounding_mode="floor").clamp(max=n_ag - 1)
            succ = self.ep_succeeded[env_ids].float()
            for a in range(n_ag):
                m = ag == a
                if m.any():
                    e = (1.0 - success_ema_beta) * state["ema"][a] + success_ema_beta * succ[m].mean().item()
                    state["ema"][a] = e
                    if e > increase_threshold:
                        state["c"][a] = min(1.0, state["c"][a] + increase_rate)
                    elif e < decrease_threshold:
                        state["c"][a] = max(0.0, state["c"][a] - decrease_rate)

        # --- (b) run the stock reset (fixed asset randomize, IK to nominal, seat/close/weld) ---
        out = _orig(self, env_ids)

        n = len(env_ids)
        if n > 0:
            ag = torch.div(env_ids, epa, rounding_mode="floor").clamp(max=n_ag - 1)
            c_vec = torch.tensor(state["c"], device=device, dtype=torch.float32)[ag]     # (n,)

            def sbc(minv, maxv):
                floor = minv + c_vec * (maxv - minv)                 # (n,), per-env floor
                return floor + torch.rand(n, device=device) * (maxv - floor)

            max_x = float(self.cfg_task.hand_init_pos_noise[0])
            max_y = float(self.cfg_task.hand_init_pos_noise[1])
            max_z = float(self.cfg_task.hand_init_pos[2])

            # NEVER-START-IN-SUCCESS floor: raise the z lower bound so the peg base is always
            # >= success_margin_z above the success depth. Measure the fingertip-height -> peg-base
            # z_disp map once from the (nominal) stock reset: offset_c = base_zdisp - fingertip_h, so
            # z_disp = fingertip_h + offset_c. Require z_disp >= success_threshold*height + margin
            # => fingertip_h >= success_threshold*height + margin - offset_c =: z_floor_succ.
            if state["z_floor_succ"] is None:
                from isaaclab_tasks.direct.factory import factory_utils
                hb, _ = factory_utils.get_held_base_pose(
                    self.held_pos, self.held_quat, self.cfg_task.name,
                    self.cfg_task.fixed_asset_cfg, self.num_envs, self.device)
                tb, _ = factory_utils.get_target_held_base_pose(
                    self.fixed_pos, self.fixed_quat, self.cfg_task.name,
                    self.cfg_task.fixed_asset_cfg, self.num_envs, self.device)
                base_zdisp = hb[env_ids, 2] - tb[env_ids, 2]
                fh = self.fingertip_midpoint_pos[env_ids, 2] - self.fixed_pos_obs_frame[env_ids, 2]
                offset_c = float((base_zdisp - fh).mean().item())
                height = float(self.cfg_task.fixed_asset_cfg.height)
                st = float(self.cfg_task.success_threshold)
                state["z_floor_succ"] = st * height + success_margin_z - offset_c
            z_lo = max(min_pos_t[2], state["z_floor_succ"])

            xmag = sbc(min_pos_t[0], max_x) * torch.sign(torch.rand(n, device=device) - 0.5)
            ymag = sbc(min_pos_t[1], max_y) * torch.sign(torch.rand(n, device=device) - 0.5)
            z = sbc(z_lo, max_z)
            # per-axis signed tilt in [floor, grasp]  (axes with grasp==0 stay 0)
            tr = sbc(min_orn_rad[0], grasp_rad[0])
            tp = sbc(min_orn_rad[1], grasp_rad[1])
            ty = sbc(min_orn_rad[2], grasp_rad[2])

            # (c') DEPTH-SAFETY: if the peg would spawn at/below the hole top, force it ALIGNED +
            # CENTERED so it descends the bore instead of interpenetrating the wall at an angle.
            # Threshold = align_below_z if set, else the peg-below-fingertip z-offset auto-measured
            # from the (nominal) stock reset just run: z_off = hand_init_pos[2] - (peg_root_z - tip_z).
            if state["z_off"] is None:
                tipz0 = self.fixed_pos_obs_frame[env_ids, 2]
                pegz0 = self.held_pos[env_ids, 2]
                state["z_off"] = float((float(self.cfg_task.hand_init_pos[2]) - (pegz0 - tipz0)).mean().item())
            thr = align_below_z if align_below_z > 0.0 else state["z_off"]
            below = z < thr                                          # (n,) bool
            if below.any():
                zeros = torch.zeros_like(tr)
                tr = torch.where(below, zeros, tr)
                tp = torch.where(below, zeros, tp)
                ty = torch.where(below, zeros, ty)
                xmag = torch.where(below, torch.zeros_like(xmag), xmag)
                ymag = torch.where(below, torch.zeros_like(ymag), ymag)

            # (c) reposition the gripper (welded peg rides along) to the sampled pose.
            target_pos = self.fixed_pos_obs_frame[env_ids].clone()
            target_pos[:, 0] += xmag
            target_pos[:, 1] += ymag
            target_pos[:, 2] += z

            hand = torch.tensor(self.cfg_task.hand_init_orn, device=device).unsqueeze(0).repeat(n, 1)
            hand_quat = torch_utils.quat_from_euler_xyz(hand[:, 0], hand[:, 1], hand[:, 2])
            samp_quat = torch_utils.quat_from_euler_xyz(tr, tp, ty)                        # R_samp
            g = torch.tensor(grasp_rad, device=device).unsqueeze(0).repeat(n, 1)
            grasp_quat = torch_utils.quat_from_euler_xyz(g[:, 0], g[:, 1], g[:, 2])        # R_grasp
            # gripper = hand_down ∘ R_samp ∘ R_grasp^-1
            target_quat = torch_utils.quat_mul(
                torch_utils.quat_mul(hand_quat, samp_quat), torch_utils.quat_conjugate(grasp_quat)
            )

            for _ in range(ik_iters):
                self.set_pos_inverse_kinematics(
                    ctrl_target_fingertip_midpoint_pos=target_pos,
                    ctrl_target_fingertip_midpoint_quat=target_quat,
                    env_ids=env_ids,
                )
                self.step_sim_no_action()

            # --- smoke-test diagnostic: SAMPLED vs ACHIEVED peg tilt/height (RESET_CURRICULUM_DEBUG=1) ---
            import os
            if os.environ.get("RESET_CURRICULUM_DEBUG"):
                def _zax(q):
                    w, x, y, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                    return torch.stack((2 * (x * zz + w * y), 2 * (y * zz - w * x), 1 - 2 * (x * x + y * y)), dim=-1)
                pz = _zax(self.held_quat[env_ids])
                sz = _zax(self.fixed_quat[env_ids])
                ach = torch.rad2deg(torch.arccos((pz * sz).sum(-1).clamp(-1.0, 1.0)))   # peg-vs-socket angle
                samp = torch.rad2deg(torch.sqrt(tr * tr + tp * tp + ty * ty))           # ~|pitch| for pitch-only grasp
                hz = self.held_pos[env_ids, 2] - self.fixed_pos_obs_frame[env_ids, 2]
                # never-in-success check: peg-base z_disp vs the success depth (both from the env)
                from isaaclab_tasks.direct.factory import factory_utils as _fu
                _hb, _ = _fu.get_held_base_pose(self.held_pos, self.held_quat, self.cfg_task.name,
                                                self.cfg_task.fixed_asset_cfg, self.num_envs, self.device)
                _tb, _ = _fu.get_target_held_base_pose(self.fixed_pos, self.fixed_quat, self.cfg_task.name,
                                                       self.cfg_task.fixed_asset_cfg, self.num_envs, self.device)
                _zd = (_hb[env_ids, 2] - _tb[env_ids, 2])
                _sd = float(self.cfg_task.success_threshold) * float(self.cfg_task.fixed_asset_cfg.height)
                print(
                    f"[curric-dbg] c={[round(v,2) for v in state['c']]} n={n} | "
                    f"tilt SAMP {samp.min():.1f}/{samp.mean():.1f}/{samp.max():.1f} "
                    f"ACH {ach.min():.1f}/{ach.mean():.1f}/{ach.max():.1f} |err|={(ach-samp).abs().mean():.1f}deg | "
                    f"z SAMP {z.min():.3f}/{z.mean():.3f}/{z.max():.3f} peg_h {hz.min():.3f}/{hz.mean():.3f}/{hz.max():.3f} | "
                    f"base_zdisp min={_zd.min():.4f} (succ<{_sd:.4f}; margin above={_zd.min()-_sd:+.4f})",
                    flush=True,
                )

        # --- (d) publish per-agent c / ema as per-env tensors (block_agent means over each slice) ---
        all_ag = torch.div(torch.arange(self.num_envs, device=device), epa,
                           rounding_mode="floor").clamp(max=n_ag - 1)
        to_log = self.extras.setdefault("to_log", {})
        to_log["ResetCurriculum/c"] = torch.tensor(state["c"], device=device, dtype=torch.float32)[all_ag]
        to_log["ResetCurriculum/success_ema"] = torch.tensor(state["ema"], device=device, dtype=torch.float32)[all_ag]
        return out

    FactoryEnv.randomize_initial_state = _patched
    print(
        f"[reset-curriculum] per-agent SBC installed (num_agents={n_ag}): "
        f"min_pos={list(min_pos)} m, min_orn={list(min_orn_deg)} deg -> max(task pos, grasp={list(grasp_tilt_deg)} deg); "
        f"inc(rate={increase_rate}, thr={increase_threshold}) dec(rate={decrease_rate}, thr={decrease_threshold}).",
        flush=True,
    )
