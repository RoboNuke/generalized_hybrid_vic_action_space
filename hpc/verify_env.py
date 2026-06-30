#!/usr/bin/env python3
"""Verify the built Isaac Lab Apptainer image — run SEPARATELY from the build so a
failed check never forces an image rebuild.

Run inside the container, on a GPU node:

    srun -A virl-grp -p dgxh,dgx2,tiamat,gpu,eecs2 --gres=gpu:1 --time=0:20:00 --pty bash
    ~/hpc-share/isaac/run.sh python hpc/verify_env.py

Stages (fail fast, each prints a [n/N] line):
  1. torch imports, reports version + CUDA + GPU visibility
  2. skrl imports
  3. isaacsim / isaaclab / isaaclab_tasks / isaaclab_assets are installed (find_spec,
     no execution — these pull omni.* which only resolves after the app boots)
  4. Isaac Sim boots headless via AppLauncher, THEN isaaclab + isaaclab_tasks import
     (omni.* now resolves), and the repo's task registrations load
"""
import importlib.util


def step(n, total, msg):
    print(f"[{n}/{total}] {msg}", flush=True)


def main() -> int:
    total = 4

    step(1, total, "import torch")
    import torch
    print(f"      torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"cuda_available={torch.cuda.is_available()} | "
          f"devices={torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        print("      WARNING: CUDA not visible — are you on a GPU node and using --nv?", flush=True)

    step(2, total, "import skrl")
    import skrl
    print(f"      skrl {skrl.__version__}", flush=True)

    step(3, total, "isaacsim / isaaclab packages installed (find_spec)")
    pkgs = ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_assets")
    missing = [m for m in pkgs if importlib.util.find_spec(m) is None]
    assert not missing, f"NOT installed: {missing}"
    print(f"      installed: {' '.join(pkgs)}", flush=True)

    step(4, total, "boot Isaac Sim headless, then import isaaclab_tasks (omni resolves)")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    try:
        import isaaclab            # noqa: F401  (omni.* now resolves)
        import isaaclab_tasks      # noqa: F401  (triggers task registration)
        print("      Isaac Sim booted; isaaclab + isaaclab_tasks import OK", flush=True)
    finally:
        app.close()

    print("VERIFY OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
