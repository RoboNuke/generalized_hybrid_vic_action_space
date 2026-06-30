"""6-D rotation-matrix representation for orientation observations.

A raw quaternion is a poor neural-net input for a policy that must control orientation: the
double cover (``q`` and ``-q`` are the SAME rotation) gives sign discontinuities, and the
unit-norm manifold makes the landscape non-smooth -- worst near the "equator" (``w ~= 0``),
which is exactly where a hand-down gripper sits. The 6-D representation of Zhou et al. (2019),
"On the Continuity of Rotation Representations in Neural Networks" -- the first two columns of
the rotation matrix -- is continuous, sign-unambiguous, and has no norm constraint, giving a
smoother optimization landscape.

This module lets a config switch any quaternion-valued orientation obs channel
(``fingertip_quat`` / ``held_quat`` / ``fixed_quat``) to its 6-D form (``*_rot6d``, dim 6)
WITHOUT touching the physics or control. The env always publishes BOTH reps in its obs/state
dict (``augment_obs_dict_with_rot6d``); ``cfg.obs_order`` / ``cfg.state_order`` selects which
one the policy/critic actually consumes (``apply_orientation_obs_mode``). Because the obs is
sized from those orders (``observation_space = sum(OBS_DIM_CFG[k] for k in obs_order)``), the
``*_rot6d`` dims must be registered first (``register_rot6d_dims``). All wiring runs in
``learning.env_setup`` BEFORE ``gym.make``.
"""

from __future__ import annotations

import torch

ORIENTATION_OBS_MODES = ("quat", "6d_rot_mat")

# Quaternion obs key -> its 6-D rotation-matrix counterpart. Covers every world-frame
# orientation channel the Factory/Forge/surface obs+state dicts publish.
QUAT_TO_ROT6D_KEYS = {
    "fingertip_quat": "fingertip_rot6d",
    "held_quat": "held_rot6d",
    "fixed_quat": "fixed_rot6d",
}
ROT6D_DIM = 6


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) quaternion -> 6-D rep: the first two COLUMNS of its rotation matrix.

    ``quat``: (..., 4). Returns (..., 6) = ``concat(R[:, 0], R[:, 1])``. Computed directly from
    the quaternion (no external import); ``R`` is quadratic in the components, so ``R(-q) ==
    R(q)`` and the double-cover sign flip vanishes. Inputs are renormalized defensively.
    """
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # First two columns of R(q) (the images of the body x- and y-axes in world).
    col0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)
    col1 = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)
    return torch.cat((col0, col1), dim=-1)


def register_rot6d_dims(obs_dim_cfg: dict, state_dim_cfg: dict) -> None:
    """Register every ``*_rot6d`` channel (dim 6) in the shared Factory dim dicts so
    ``observation_space``/``state_space`` size correctly when an order uses the 6-D key."""
    for rot6d_key in QUAT_TO_ROT6D_KEYS.values():
        obs_dim_cfg[rot6d_key] = ROT6D_DIM
        state_dim_cfg[rot6d_key] = ROT6D_DIM


def augment_obs_dict_with_rot6d(obs_dict: dict) -> dict:
    """For each quaternion key present in ``obs_dict``, add its ``*_rot6d`` counterpart (in place).

    Always safe to call regardless of the selected mode: it only ADDS keys.
    ``collapse_obs_dict`` reads only the keys named in the obs/state order, so the unused rep
    costs one small tensor op and is otherwise inert.
    """
    for quat_key, rot6d_key in QUAT_TO_ROT6D_KEYS.items():
        if quat_key in obs_dict:
            obs_dict[rot6d_key] = quat_to_rot6d(obs_dict[quat_key])
    return obs_dict


def apply_orientation_obs_mode(env_cfg, mode: str) -> None:
    """Select the orientation representation by rewriting ``obs_order``/``state_order`` in place.

    ``quat`` (default): leave the orders untouched. ``6d_rot_mat``: replace every quaternion key
    with its ``*_rot6d`` counterpart. Must run BEFORE ``gym.make`` (the env sizes its obs from
    these orders in ``__init__``). Raises on an unknown mode.
    """
    if mode not in ORIENTATION_OBS_MODES:
        raise ValueError(
            f"controller_cfg.orientation_obs_mode must be one of {ORIENTATION_OBS_MODES}, got {mode!r}."
        )
    if mode == "quat":
        return

    def _swap(order):
        return [QUAT_TO_ROT6D_KEYS.get(k, k) for k in order]

    swapped = []
    if hasattr(env_cfg, "obs_order"):
        env_cfg.obs_order = _swap(env_cfg.obs_order)
        swapped.append("obs_order")
    if hasattr(env_cfg, "state_order"):
        env_cfg.state_order = _swap(env_cfg.state_order)
        swapped.append("state_order")
    print(
        f"[orientation-obs] mode=6d_rot_mat: swapped quaternion -> 6-D rotation-matrix rep in "
        f"{', '.join(swapped) or '(no orders found)'}.",
        flush=True,
    )
