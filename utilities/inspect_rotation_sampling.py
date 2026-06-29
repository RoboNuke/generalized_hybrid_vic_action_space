"""Inspect the range of rotation matrices the policy samples *at network initialization*.

Standalone diagnostic (Isaac-free). The action sampling is done entirely by the model, and the
env normalizes observations so every input lands in ~[-1, 1]; we therefore feed random uniform
[-1, 1] observations of the deployed width to the freshly-initialized actor, draw action samples,
take the trailing 6-D rotation action, map it through the same Gram-Schmidt ``rotation_6d_to_matrix``
the controller uses, and characterize the resulting SO(3) distribution against what a *uniformly
random* rotation would look like.

FIXED deployment geometry (FORGE peg-in-hole + GAS rotated full-gain controller) — these never
change; they are exactly how the rotation is produced/consumed downstream:

  * FORGE peg-in-hole base action_space = 7 (Factory 6-DoF pose delta + 1 success-pred dim, idx 6).
  * The GAS rotated controller (gain_mapping='rotated', full_gain_matrix=True, no hybrid force)
    appends a pose-gain block of pdim = 12 (6 diagonal gains + 6-D rotation), so the policy action
    is 7 + 12 = 19 with the rotation 6-D at indices [13:19].
  * disable_success_pred force-zeros action dim 6; there are no hybrid-force selection dims.
  * The network-facing observation width is 36 (verified from the env). The sampled rotation
    distribution is provably invariant to this value (kaiming init + input normalization), so only
    the rotation indices / action width matter.

Two modes
---------
* single (default): sample from the config's actor settings and write rotation_sampling.png — the
  four marginal distributions roll / pitch / yaw / geodesic-from-identity, each a density histogram
  with the analytic *uniform-rotation* reference drawn as a red dotted line (so you can read how far
  from "broadly exploring all orientations" the policy is). Also writes rotation_sampling.npz.

* sweep (--sweep): a 2-D grid over act_init_std (y-axis) x initial output-weight scale (x-axis,
  ``last_layer_scale`` applied globally so it scales the rotation outputs). For each grid cell it
  measures, for each of the four marginals, the total deviation from the uniform reference
  (sum over bins of |observed - expected| density * bin-width, a total-variation distance in
  [0, 2]; 0 = perfectly uniform, larger = more concentrated). Writes four heatmaps
  (roll / pitch / yaw / geodesic) to rotation_sampling_sweep.png + .npz.

Usage:
    python utilities/inspect_rotation_sampling.py --config configs/exp_cfgs/glued_peg_FORGE/5_GAS.yaml
    python utilities/inspect_rotation_sampling.py --config <cfg> --sweep
"""

import argparse
import os
import sys

import numpy as np
import torch
import gymnasium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from models.block_simba import HybridControlBlockSimBaActor  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "exp_cfgs", "default.yaml")

# ---- FIXED deployment geometry (see module docstring) — never changes ----
OBS_DIM = 36
ACTION_DIM = 19
ROT_INDICES = [13, 14, 15, 16, 17, 18]   # the 6-D rotation action
FORCE_ZERO_DIMS = [6]                     # success-pred head disabled
SELECTION_DIMS: list = []                 # no hybrid-force axes (N=0)

# Actor hyperparameter defaults mirroring configs/manager/model_cfg.ActorCfg (the YAML may omit
# fields; we can't import the real config class — it pulls in Isaac Lab).
_ACTOR_DEFAULTS = dict(
    act_init_std=0.60653066, actor_n=2, actor_latent=512, last_layer_scale=1.0,
    clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum",
    use_state_dependent_std=False, bernoulli_action_dims=None, force_zero_action_dims=None,
    scale_down_action_dims=None, selection_init_bias=0.0, selection_distribution="product",
)


def rotation_6d_to_matrix(v6, eps=1e-8):
    """Exact copy of wrappers.controllers.factory_control_utils.rotation_6d_to_matrix
    (Gram-Schmidt 6-D -> (E,3,3)). Vendored to avoid importing the Isaac-touching wrappers pkg."""
    a1, a2 = v6[:, 0:3], v6[:, 3:6]
    b1 = a1 / a1.norm(dim=1, keepdim=True).clamp_min(eps)
    a2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=1, keepdim=True).clamp_min(eps)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=2)


def euler_xyz_from_matrix(R):
    """roll/pitch/yaw (deg), inverse of R = Rz(yaw)Ry(pitch)Rx(roll) (project XYZ convention)."""
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def geodesic_from_identity_deg(R):
    tr = np.trace(R, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# ---- analytic uniform-SO(3) marginal densities (per degree) — the red dotted references ----
def dens_uniform_angle(x_deg):           # roll & yaw: uniform on [-180, 180]
    return np.full_like(np.asarray(x_deg, float), 1.0 / 360.0)


def dens_pitch(x_deg):                    # middle Tait-Bryan angle: (1/2)cos(pitch)
    return 0.5 * np.cos(np.radians(x_deg)) * (np.pi / 180.0)


def dens_geodesic(x_deg):                 # angle from identity: (1 - cos)/180
    return (1.0 - np.cos(np.radians(x_deg))) / 180.0


# (key, axis-label, range, bins, analytic-density-fn)
QUANTS = [
    ("roll",     "roll (deg)",                  (-180, 180), 72, dens_uniform_angle),
    ("pitch",    "pitch (deg)",                 (-90, 90),   36, dens_pitch),
    ("yaw",      "yaw (deg)",                   (-180, 180), 72, dens_uniform_angle),
    ("geodesic", "geodesic from identity (deg)", (0, 180),   60, dens_geodesic),
]


def deviation_from_uniform(data, rng, bins, densfn):
    """Total deviation of a marginal from the uniform-rotation reference: sum over bins of
    |observed_density - expected_density| * bin_width (a total-variation distance in [0, 2];
    0 = matches uniform exactly, larger = more concentrated / structured)."""
    obs, edges = np.histogram(data, bins=bins, range=rng, density=True)
    width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(np.sum(np.abs(obs - densfn(centers))) * width)


# ----------------------------------------------------------------------------------------
def load_yaml(config_path, overlay_paths):
    import yaml

    def _merge(b, o):
        out = dict(b)
        for k, v in o.items():
            out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
        return out
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for ov in (overlay_paths or []):
        with open(ov) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG_PATH,
                   help="YAML config (provides the actor hyperparameters). The action/obs geometry "
                        "is FIXED to the FORGE peg-in-hole + GAS rotated deployment.")
    p.add_argument("--overlay", type=str, action="append", default=None)
    p.add_argument("--samples", type=int, default=40000, help="Rotation samples per init seed.")
    p.add_argument("--chunk", type=int, default=8192, help="Forward-pass batch size.")
    p.add_argument("--num_seeds", type=int, default=8,
                   help="Pool samples over this many weight-init seeds (seed, seed+1, ...). The "
                        "init is seed-dependent, so >1 gives a robust, reproducible distribution.")
    p.add_argument("--seed", type=int, default=0, help="Base weight-init seed.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "runs", "_rotation_sampling"))
    # sweep
    p.add_argument("--sweep", action="store_true", help="2-D act_init_std x init-weight heatmap sweep.")
    p.add_argument("--act_init_std_values", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0], help="y-axis values (act_init_std).")
    p.add_argument("--last_layer_scale_values", type=float, nargs="+",
                   default=[0.01, 0.03, 0.1, 0.3, 1.0], help="x-axis values (initial output-weight scale).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    rot_idx = torch.tensor(ROT_INDICES, dtype=torch.long, device=device)

    cfg = load_yaml(args.config, args.overlay)
    actor = {**_ACTOR_DEFAULTS, **((cfg.get("model_cfg", {}) or {}).get("actor", {}) or {})}

    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
    act_space = gymnasium.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

    base_actor_kwargs = dict(
        act_init_std=actor["act_init_std"], actor_n=actor["actor_n"], actor_latent=actor["actor_latent"],
        last_layer_scale=actor["last_layer_scale"], clip_log_std=actor["clip_log_std"],
        min_log_std=actor["min_log_std"], max_log_std=actor["max_log_std"], reduction=actor["reduction"],
        use_state_dependent_std=actor["use_state_dependent_std"],
        bernoulli_action_dims=actor["bernoulli_action_dims"], force_zero_action_dims=FORCE_ZERO_DIMS,
        scale_down_action_dims=actor["scale_down_action_dims"],
    )
    sel_dist = actor["selection_distribution"]
    sel_bias = actor["selection_init_bias"]

    def seed_all(s):
        import random
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    def build_policy(actor_kwargs, seed):
        seed_all(seed)
        pol = HybridControlBlockSimBaActor(
            observation_space=obs_space, action_space=act_space, device=device, num_agents=1,
            selection_dims=SELECTION_DIMS, pos_component_dims=[], force_component_dims=[],
            selection_distribution=sel_dist, selection_init_bias=sel_bias, **actor_kwargs)
        pol.eval()
        return pol

    def sample_R(policy):
        """Draw args.samples rotations; obs ~ uniform[-1,1] (normalized-input range), fresh per sample."""
        out, drawn = [], 0
        with torch.no_grad():
            while drawn < args.samples:
                n = min(args.chunk, args.samples - drawn)
                obs = torch.empty(n, OBS_DIM, device=device).uniform_(-1.0, 1.0)
                a, _ = policy.act({"observations": obs}, role="policy")
                out.append(a.index_select(1, rot_idx).detach().to("cpu"))
                drawn += n
        return rotation_6d_to_matrix(torch.cat(out, 0))

    def sample_marginals(actor_kwargs):
        """Pool over seeds; return dict roll/pitch/yaw/geodesic (numpy deg)."""
        seeds = [args.seed + i for i in range(max(1, args.num_seeds))]
        Rs = [sample_R(build_policy(actor_kwargs, sd)).numpy() for sd in seeds]
        R = np.concatenate(Rs, 0)
        roll, pitch, yaw = euler_xyz_from_matrix(R)
        return dict(roll=roll, pitch=pitch, yaw=yaw, geodesic=geodesic_from_identity_deg(R)), len(seeds), R.shape[0]

    # =====================================================================================
    #  SWEEP MODE — 4 heatmaps of deviation-from-uniform over (act_init_std x init weight)
    # =====================================================================================
    if args.sweep:
        ys = args.act_init_std_values            # rows / y-axis
        xs = args.last_layer_scale_values        # cols / x-axis
        print(f"[inspect] SWEEP act_init_std(y)={ys}  x  last_layer_scale(x)={xs}  "
              f"({len(ys)*len(xs)} cells, {max(1,args.num_seeds)} seed(s) each)")
        # dev[q] is a (len(ys), len(xs)) grid of total-variation deviations from uniform.
        dev = {q[0]: np.zeros((len(ys), len(xs))) for q in QUANTS}
        for i, a in enumerate(ys):
            for j, w in enumerate(xs):
                ak = dict(base_actor_kwargs)
                ak["act_init_std"] = a
                ak["last_layer_scale"] = w
                ak["scale_down_action_dims"] = None   # global: the init weight scales rotation too
                marg, _, _ = sample_marginals(ak)
                for key, _lab, rng, bins, dfn in QUANTS:
                    dev[key][i, j] = deviation_from_uniform(marg[key], rng, bins, dfn)
            print(f"  act_init_std={a:<5g} done")

        png = os.path.join(args.out_dir, "rotation_sampling_sweep.png")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            vmax = max(dev[k].max() for k in dev)
            fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
            for ax, (key, lab, _rng, _b, _d) in zip(axes, QUANTS):
                im = ax.imshow(dev[key], origin="lower", aspect="auto", cmap="viridis",
                               vmin=0, vmax=vmax)
                ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:g}" for v in xs])
                ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{v:g}" for v in ys])
                ax.set_xlabel("initial output-weight scale (last_layer_scale)")
                ax.set_ylabel("act_init_std")
                ax.set_title(f"{key}: deviation from uniform")
                for i in range(len(ys)):
                    for j in range(len(xs)):
                        ax.text(j, i, f"{dev[key][i, j]:.2f}", ha="center", va="center",
                                color="w" if dev[key][i, j] < 0.6 * vmax else "k", fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle("Rotation-sampling sweep — total deviation from uniform-SO(3) marginal "
                         "(0 = uniform/broad, larger = concentrated)  |  "
                         f"{os.path.basename(args.config)}", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(png, dpi=120)
            print(f"[inspect] wrote sweep figure -> {png}")
        except Exception as e:
            print(f"[inspect] could not render sweep figure ({e!r}); skipping PNG.")

        npz = os.path.join(args.out_dir, "rotation_sampling_sweep.npz")
        np.savez_compressed(npz, act_init_std_values=np.array(ys),
                            last_layer_scale_values=np.array(xs),
                            **{f"dev_{k}": dev[k] for k in dev}, config=os.path.abspath(args.config))
        print(f"[inspect] wrote sweep data -> {npz}")
        return

    # =====================================================================================
    #  SINGLE MODE — four marginal histograms vs the uniform reference
    # =====================================================================================
    print(f"[inspect] FORGE peg-in-hole + GAS rotated:  obs_dim={OBS_DIM}  action_dim={ACTION_DIM}  "
          f"rot_indices={ROT_INDICES}  force_zero={FORCE_ZERO_DIMS}")
    print(f"[inspect] act_init_std={base_actor_kwargs['act_init_std']}  "
          f"last_layer_scale={base_actor_kwargs['last_layer_scale']}  "
          f"scale_down_action_dims={base_actor_kwargs['scale_down_action_dims']}")
    marg, nseeds, nsamp = sample_marginals(dict(base_actor_kwargs))
    print(f"[inspect] {nsamp} samples over {nseeds} seed(s); deviation-from-uniform "
          "(0 = uniform/broad, larger = concentrated):")
    devs = {}
    for key, _lab, rng, bins, dfn in QUANTS:
        devs[key] = deviation_from_uniform(marg[key], rng, bins, dfn)
        print(f"  {key:9s} dev={devs[key]:.3f}")

    png_path = os.path.join(args.out_dir, "rotation_sampling.png")
    npz_path = os.path.join(args.out_dir, "rotation_sampling.npz")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
        for ax, (key, lab, rng, bins, dfn) in zip(axes, QUANTS):
            ax.hist(marg[key], bins=bins, range=rng, density=True, color="#4477aa", alpha=0.85)
            xref = np.linspace(rng[0], rng[1], 400)
            ax.plot(xref, dfn(xref), "r:", lw=2, label="uniform-rotation reference")
            ax.set_title(f"{key}   (dev from uniform = {devs[key]:.2f})")
            ax.set_xlabel(lab); ax.set_ylabel("density"); ax.set_xlim(*rng); ax.legend(fontsize=8)
        fig.suptitle("Sampled rotation marginals @ init vs uniform-SO(3) reference (red dotted)  |  "
                     f"{os.path.basename(args.config)}  |  N={nsamp} over {nseeds} seed(s), "
                     f"act_init_std={base_actor_kwargs['act_init_std']}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(png_path, dpi=120)
        print(f"[inspect] wrote figure -> {png_path}")
    except Exception as e:
        print(f"[inspect] could not render figure ({e!r}); skipping PNG.")

    np.savez_compressed(
        npz_path, roll_deg=marg["roll"], pitch_deg=marg["pitch"], yaw_deg=marg["yaw"],
        angle_from_identity_deg=marg["geodesic"], deviations=np.array([devs[k] for k in devs]),
        deviation_keys=np.array(list(devs.keys())), config=os.path.abspath(args.config))
    print(f"[inspect] wrote raw samples -> {npz_path}")


if __name__ == "__main__":
    main()
