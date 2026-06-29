"""Verify the 'uniform SO(3)' baseline and show what it looks like in RPY vs our sampled rotations.

Skepticism-driven check. It:
  1. Generates uniformly-random SO(3) rotations TWO independent ways (normalized-Gaussian
     quaternions, and scipy's Rotation.random if available) and proves they are uniform by
     comparing the geodesic-angle-from-identity histogram to the exact analytic density
     (1 - cos θ)/180 (per degree), whose mean is pi/2 + 2/pi = 126.47 deg.
  2. Converts those uniform rotations to roll/pitch/yaw (the SAME XYZ extraction the main script
     uses) and overlays the analytic Tait-Bryan marginals: roll and yaw are UNIFORM on [-180,180],
     the middle angle (pitch) has density (1/2)cos(pitch) -- NOT uniform. This is the only
     'RPY artifact': it is real, and it affects pitch only.
  3. Plots OUR sampled rotations' RPY marginals beside the uniform ones (loaded from
     runs/_rotation_sampling/rotation_sampling.npz) so the comparison is direct.

Writes runs/_rotation_sampling/uniform_rpy_check.png and prints numeric verification.
Isaac-free (numpy/torch/matplotlib, optional scipy).
"""

import os
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of utilities/
_OUT = os.path.join(_PROJECT_ROOT, "runs", "_rotation_sampling")


def euler_xyz_from_matrix(R):
    """roll/pitch/yaw (deg) from R (N,3,3), inverse of R = Rz(yaw)Ry(pitch)Rx(roll) -- identical
    to the convention in inspect_rotation_sampling.py / factory_control_utils."""
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def geodesic_from_identity_deg(R):
    tr = np.trace(R, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def uniform_rotations_quaternion(n, seed=0):
    """Uniform SO(3) via normalized 4D Gaussian quaternions (q ~ N(0,I)/|.| is uniform on S^3,
    which double-covers SO(3) uniformly -> Haar measure)."""
    g = np.random.RandomState(seed).randn(n, 4)
    q = g / np.linalg.norm(g, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], axis=1).reshape(n, 3, 3)


def main():
    N = 200_000
    Ru = uniform_rotations_quaternion(N, seed=0)

    # ---- proof of uniformity #1: geodesic density matches analytic, mean = pi/2 + 2/pi ----
    g_emp = geodesic_from_identity_deg(Ru)
    analytic_mean = np.degrees(np.pi / 2 + 2 / np.pi)
    print(f"[verify] UNIFORM via Gaussian-quaternion (N={N}):")
    print(f"  geodesic-from-identity: empirical mean={g_emp.mean():.2f} deg, "
          f"analytic mean={analytic_mean:.2f} deg (pi/2 + 2/pi)")
    rl_u, pt_u, yw_u = euler_xyz_from_matrix(Ru)
    print(f"  roll  : mean={rl_u.mean():7.2f} std={rl_u.std():6.2f}  (uniform[-180,180] -> std 103.92)")
    print(f"  yaw   : mean={yw_u.mean():7.2f} std={yw_u.std():6.2f}  (uniform[-180,180] -> std 103.92)")
    print(f"  pitch : mean={pt_u.mean():7.2f} std={pt_u.std():6.2f}  (density (1/2)cos -> std 39.23, NOT uniform)")

    # ---- proof of uniformity #2: independent generator (scipy), if available ----
    try:
        from scipy.spatial.transform import Rotation
        Rs = Rotation.random(N, random_state=1).as_matrix()
        g2 = geodesic_from_identity_deg(Rs)
        rl2, pt2, yw2 = euler_xyz_from_matrix(Rs)
        print(f"[verify] UNIFORM via scipy Rotation.random (independent): geodesic mean={g2.mean():.2f}, "
              f"roll std={rl2.std():.2f}, pitch std={pt2.std():.2f}, yaw std={yw2.std():.2f}")
    except Exception as e:
        print(f"[verify] scipy cross-check unavailable ({e!r}); relying on analytic comparison only.")

    # ---- our sampled rotations (from the last inspect run) ----
    ours = None
    npz_path = os.path.join(_OUT, "rotation_sampling.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        ours = dict(roll=d["roll_deg"], pitch=d["pitch_deg"], yaw=d["yaw_deg"],
                    geo=d["angle_from_identity_deg"], cfg=str(d["config"]))
        print(f"[verify] loaded our samples from {npz_path} (N={len(ours['roll'])}, {os.path.basename(ours['cfg'])})")
    else:
        print(f"[verify] {npz_path} not found; plotting uniform only. Run inspect_rotation_sampling.py first.")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    deg = np.linspace(-180, 180, 361)
    pdeg = np.linspace(-90, 90, 181)
    gdeg = np.linspace(0, 180, 361)

    def panel(ax, data, rng, bins, title, analytic=None, alabel=None):
        ax.hist(data, bins=bins, range=rng, density=True, color="#4477aa", alpha=0.8)
        if analytic is not None:
            ax.plot(analytic[0], analytic[1], "r-", lw=2, label=alabel)
            ax.legend(fontsize=9)
        ax.set_title(title); ax.set_xlabel("deg"); ax.set_xlim(*rng)

    # Row 0: UNIFORM SO(3) with analytic overlays
    panel(axes[0, 0], rl_u, (-180, 180), 72, "UNIFORM SO(3): roll",
          (deg, np.full_like(deg, 1 / 360.0)), "analytic: uniform 1/360")
    panel(axes[0, 1], pt_u, (-90, 90), 36, "UNIFORM SO(3): pitch (middle angle)",
          (pdeg, 0.5 * np.cos(np.radians(pdeg)) * np.pi / 180), "analytic: (1/2)cos(pitch)")
    panel(axes[0, 2], yw_u, (-180, 180), 72, "UNIFORM SO(3): yaw",
          (deg, np.full_like(deg, 1 / 360.0)), "analytic: uniform 1/360")
    panel(axes[0, 3], g_emp, (0, 180), 60, "UNIFORM SO(3): geodesic from identity",
          (gdeg, (1 - np.cos(np.radians(gdeg))) / 180.0), "analytic: (1-cos)/180")
    axes[0, 3].axvline(analytic_mean, color="k", ls=":", lw=1)

    # Row 1: OUR samples (same axes), with the uniform analytic curves drawn faintly for reference
    if ours is not None:
        cfgname = os.path.basename(ours["cfg"])
        panel(axes[1, 0], ours["roll"], (-180, 180), 72, f"OUR samples: roll  ({cfgname})")
        axes[1, 0].plot(deg, np.full_like(deg, 1 / 360.0), "r--", lw=1, alpha=0.6, label="uniform ref")
        axes[1, 0].legend(fontsize=8)
        panel(axes[1, 1], ours["pitch"], (-90, 90), 36, "OUR samples: pitch")
        axes[1, 1].plot(pdeg, 0.5 * np.cos(np.radians(pdeg)) * np.pi / 180, "r--", lw=1, alpha=0.6)
        panel(axes[1, 2], ours["yaw"], (-180, 180), 72, "OUR samples: yaw")
        axes[1, 2].plot(deg, np.full_like(deg, 1 / 360.0), "r--", lw=1, alpha=0.6)
        panel(axes[1, 3], ours["geo"], (0, 180), 60, "OUR samples: geodesic from identity")
        axes[1, 3].plot(gdeg, (1 - np.cos(np.radians(gdeg))) / 180.0, "r--", lw=1, alpha=0.6, label="uniform ref")
        axes[1, 3].axvline(analytic_mean, color="k", ls=":", lw=1); axes[1, 3].legend(fontsize=8)
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "run inspect_rotation_sampling.py first", ha="center", transform=ax.transAxes)

    fig.suptitle("Truly-uniform SO(3) rotations in RPY (top, with analytic curves) vs our sampled "
                 "rotations (bottom).  Roll & yaw ARE uniform; pitch (middle angle) is (1/2)cos -- "
                 "the only RPY artifact.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, "uniform_rpy_check.png")
    os.makedirs(_OUT, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[verify] wrote figure -> {out}")


if __name__ == "__main__":
    main()
