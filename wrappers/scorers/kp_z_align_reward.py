"""Optional peg-axis orientation keypoint reward for Forge/Factory peg insertion.

When enabled (``runner_cfg.kp_z_align_enabled``), a ``kp_z_align`` term is added to the
Factory/Forge reward that pulls the held peg's z-axis onto the socket's z-axis — i.e. it
rewards ORIENTATION alignment, the degree of freedom the base keypoint reward only weakly
constrains and that a tilted grasp (see ``wrappers/sensors/grasp_tilt_wrapper.py``) deliberately
fights. The signal is the angle (radians) between the peg z-axis and the socket z-axis, fed
through the same bounded keypoint squashing function the other ``kp_*`` terms use:

    kp_z_align = 1 / (exp(a * angle) + b + exp(-a * angle))

so it is maximal (``1 / (2 + b)``) when the axes coincide (angle = 0) and decays toward 0 as the
peg tilts away. ``a`` sharpens the basin (larger => only near-perfect alignment scores);
``b`` sets the peak height. Defaults ``a=20``, ``b=1.33`` mirror the fine keypoint coefficients.

The angle is computed from the TRUE peg/socket poses (``held_quat`` / ``fixed_quat``). This is a
reward term, evaluated sim-side — it is privileged and never enters any observation. Using the
true peg axis (rather than reconstructing it from the noisy EEF orientation + the known grasp
offset) makes the term yaw-invariant: a rotation about the insertion axis leaves the peg z-axis
on the socket z-axis, so it is correctly unpenalised.

Single-function monkeypatch, mirroring :mod:`wrappers.sensors.grasp_tilt_wrapper`: we wrap
``FactoryEnv._get_factory_rew_dict`` (``factory_env.py:424``), the one method that builds the
per-term reward dict. It returns ``(rew_dict, rew_scales)``; we append ``kp_z_align`` to both, so
``FactoryEnv._get_rewards`` sums it into ``rew_buf`` and ``_log_factory_metrics`` publishes it
(``logs_rew_kp_z_align`` and the per-agent ``logs_rew/kp_z_align``). Forge inherits the method
(its ``_get_rewards`` calls ``super()._get_rewards()``), so patching the base covers Forge too.
Call BEFORE ``gym.make``. The reward scorers' ``_factory_scales`` carry a matching
``kp_z_align: 1.0`` entry so the per-episode reward decomposition scales it correctly.

INTENDED FOR PEG INSERT (and structurally similar fixed-asset insertion tasks): the socket
z-axis (``fixed_quat``'s local +z) is the insertion axis the peg must align to.
"""

from __future__ import annotations


def install_kp_z_align_reward(a: float = 20.0, b: float = 1.33) -> None:
    """Patch ``FactoryEnv._get_factory_rew_dict`` to add the ``kp_z_align`` orientation term.

    :param a: squashing-function steepness (larger => sharper reward basin around alignment).
    :param b: squashing-function offset (sets the peak height ``1 / (2 + b)`` at angle = 0).
    """
    import torch

    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

    a = float(a)
    b = float(b)

    def _quat_z_axis(quat):
        """World-frame z-axis (third rotation-matrix column) of a wxyz quaternion ``(N, 4)``."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.stack(
            (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1
        )

    _original = FactoryEnv._get_factory_rew_dict

    def _patched(self, curr_successes):
        rew_dict, rew_scales = _original(self, curr_successes)
        # Angle (rad) between the peg z-axis and the socket z-axis; 0 == aligned. clamp guards
        # arccos against |dot| > 1 from float error. Per-env shape (num_envs,), like the other
        # keypoint terms, so the scorers can publish it per agent.
        peg_z = _quat_z_axis(self.held_quat)
        socket_z = _quat_z_axis(self.fixed_quat)
        cos_angle = (peg_z * socket_z).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.arccos(cos_angle)
        rew_dict["kp_z_align"] = factory_utils.squashing_fn(angle, a, b)
        rew_scales["kp_z_align"] = 1.0
        return rew_dict, rew_scales

    FactoryEnv._get_factory_rew_dict = _patched
    print(
        f"[kp-z-align] FactoryEnv._get_factory_rew_dict patched: added kp_z_align orientation "
        f"reward (peg-axis vs socket-axis angle, squashing a={a}, b={b}).",
        flush=True,
    )
