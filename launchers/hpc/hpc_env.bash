#!/usr/bin/env bash
# launchers/hpc/hpc_env.bash — single source of truth for HPC/SLURM/container settings.
#
# This file is meant to be SOURCED, not executed. Both hpc_batch.bash (the SLURM job,
# on the compute node) and sbatch_launcher.bash (the submitter, on the login node)
# source it so neither has to hard-code paths or worry about an unset variable.
#
# >>> EDIT THE VALUES BELOW — this is the one place you configure HPC runs. <<<
#
# Every variable uses the `: "${VAR:=default}"` form, which means: if VAR is already
# set in your shell environment, that wins; otherwise the default here is used. So you
# can also override any single value for a one-off submit, e.g.:
#   SIF_IMAGE=~/images/other.sif bash launchers/hpc/sbatch_launcher.bash <folder>
# but for the normal workflow just set them here once.

# ===== Project root (auto-derived; normally leave as-is) =====
# Resolve from THIS file's location so it works at any clone path (HPC home != local home).
_HPC_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd -- "$_HPC_ENV_DIR/../.." && pwd)}"
export PROJECT_ROOT

# ===== Container (Apptainer / Singularity) =====
# Absolute path to the Isaac Lab .sif image on the HPC. Default matches hpc/build_isaac_env.sh
# (SHARE=$HOME/hpc-share). Override here or via the SIF_IMAGE env var if you moved it.
: "${SIF_IMAGE:=$HOME/hpc-share/isaac/ghvic.sif}"
# Container runtime binary: `apptainer` on newer clusters, `singularity` on older ones.
: "${APPTAINER_BIN:=apptainer}"
# Python INSIDE the container. Our pip-based Isaac Sim 5.1 image has python (->3.11) on PATH;
# this is NOT the old nvcr container's /isaac-sim/python.sh. Exported to the worker as PYTHON.
: "${CONTAINER_PYTHON:=python}"
# Writable HOME bound into the container so Isaac Sim's kit/shader caches (GBs) land on the
# big share, not your small home NFS quota. Mirrors hpc/run.sh's --home. On Lustre is fine.
: "${ISAAC_CACHE_HOME:=$HOME/hpc-share/isaac_cache_home}"
# Extra bind mounts, space-separated, each "host:container" or just "host". The project
# root is always bound automatically; add Isaac cache/asset dirs here if they live
# outside the project and outside $HOME. Example:
#   APPTAINER_BINDS="/scratch/$USER/isaac_cache:/root/.cache /nfs/assets:/nfs/assets"
: "${APPTAINER_BINDS:=}"
export SIF_IMAGE APPTAINER_BIN CONTAINER_PYTHON ISAAC_CACHE_HOME APPTAINER_BINDS

# ===== Output locations (repo-relative defaults) =====
# Where runner.py writes experiment runs (matches sac_block_e2e.sh's LOGDIR default).
: "${LOGDIR:=$PROJECT_ROOT/runs}"
# Where per-job SLURM stdout/stderr (.out/.err) are collected by the submitter.
: "${EXP_LOG_DIR:=$PROJECT_ROOT/exp_logs}"
export LOGDIR EXP_LOG_DIR

# ===== SLURM resources =====
# These are passed by sbatch_launcher.bash as `sbatch` CLI flags, which OVERRIDE the
# fallback #SBATCH directives baked into hpc_batch.bash. Edit here to change resources
# for every job at once. Defaults mirror the RoboNuke/Continuous_Force_RL setup.
: "${HPC_ACCOUNT:=virl-grp}"                              # -A  sponsored account
: "${HPC_PARTITIONS:=dgxh,dgx2,tiamat,gpu,eecs2}"        # -p  partition list
: "${HPC_TIME:=0-09:00:00}"                              # --time  wall-clock limit
: "${HPC_GRES:=gpu:1}"                                   # --gres  GPUs to request
: "${HPC_MEM:=32G}"                                      # --mem  host memory
: "${HPC_CPUS:=12}"                                      # -c  cores/threads per task
: "${HPC_SIGNAL:=TERM@300}"                              # --signal  SIGTERM N s before limit
export HPC_ACCOUNT HPC_PARTITIONS HPC_TIME HPC_GRES HPC_MEM HPC_CPUS HPC_SIGNAL

# ===== Shared validation helper =====
# Fail loud and early if the container image is unusable. Called by both launchers
# (the submitter calls it on the login node so you find out BEFORE jobs are queued).
hpc_require_container() {
    if [[ -z "${SIF_IMAGE:-}" ]]; then
        echo "[hpc-env] SIF_IMAGE is not set — edit launchers/hpc/hpc_env.bash and point it at your .sif" >&2
        return 1
    fi
    if [[ ! -e "$SIF_IMAGE" ]]; then
        echo "[hpc-env] SIF_IMAGE does not exist: $SIF_IMAGE" >&2
        return 1
    fi
    if ! command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
        echo "[hpc-env] container runtime '$APPTAINER_BIN' not found on PATH — set APPTAINER_BIN (apptainer/singularity), or 'module load' it first" >&2
        return 1
    fi
    return 0
}
