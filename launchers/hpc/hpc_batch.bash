#!/usr/bin/env bash
#SBATCH -J HPC_BATCH                       # job name (overridden by sbatch -J from the submitter)
#SBATCH -A virl-grp                        # sponsored account (overridden by sbatch -A)
#SBATCH -p tiamat,dgxh,gpu      # partitions (overridden by sbatch -p)
#SBATCH --time=0-12:00:00                  # wall-clock limit (overridden by sbatch --time)
#SBATCH --gres=gpu:1                       # GPUs (overridden by sbatch --gres)
#SBATCH --mem=64G                          # host memory (overridden by sbatch --mem)
#SBATCH -c 12                              # cores/threads (overridden by sbatch -c)
#SBATCH --signal=TERM@300                  # SIGTERM 300s before the limit (overridden by sbatch --signal)
#
# launchers/hpc/hpc_batch.bash — the per-config SLURM job. Runs on a compute node.
#
# Enters the Apptainer/Singularity container and runs the existing, env-agnostic worker
# launchers/sac_block_e2e.sh INSIDE it (train -> verify -> eval -> record). All container
# and resource settings come from launchers/hpc/hpc_env.bash, which this script sources.
#
# Usage (normally invoked by sbatch_launcher.bash, not by hand):
#   sbatch [resource flags] hpc_batch.bash --config <path> --exp_name <name> [-- <forwarded args...>]
#
# Everything after the literal `--` is forwarded VERBATIM to sac_block_e2e.sh, so any of
# its flags pass straight through:  -- --no_eval --experiment_directory FOO --record_config f.yaml
#
# The #SBATCH directives above are only FALLBACK defaults so the script is runnable
# standalone; the submitter passes the hpc_env.bash values as sbatch CLI flags, which win.
#
# Fail loud, fail fast.
set -Eeuo pipefail
trap 'echo "[hpc-batch] FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

# ===== Load central config =====
# Under sbatch the script is COPIED into a spool dir, so BASH_SOURCE no longer points at
# launchers/hpc and a sibling source would miss. sbatch_launcher.bash passes the real dir
# via HPC_LAUNCHER_DIR (--export); fall back to BASH_SOURCE for standalone/manual runs.
SCRIPT_DIR="${HPC_LAUNCHER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=hpc_env.bash
source "$SCRIPT_DIR/hpc_env.bash"

# ===== Args =====
CONFIG_PATH=""
EXP_NAME=""
FORWARD=()                 # everything after `--`, passed through to sac_block_e2e.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   [[ $# -ge 2 ]] || { echo "[hpc-batch] --config requires a value" >&2; exit 2; }
                    CONFIG_PATH="$2"; shift 2 ;;
        --exp_name) [[ $# -ge 2 ]] || { echo "[hpc-batch] --exp_name requires a value" >&2; exit 2; }
                    EXP_NAME="$2"; shift 2 ;;
        --)         shift; FORWARD=("$@"); break ;;
        *)          echo "[hpc-batch] unknown argument before '--': $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CONFIG_PATH" ]] || { echo "[hpc-batch] --config is required" >&2; exit 2; }
[[ -n "$EXP_NAME"    ]] || { echo "[hpc-batch] --exp_name is required" >&2; exit 2; }

# Resolve a project-root-relative config to absolute (the container sees the same path).
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi

# ===== Sanity =====
hpc_require_container
WORKER="$PROJECT_ROOT/launchers/sac_block_e2e.sh"
[[ -f "$WORKER"      ]] || { echo "[hpc-batch] worker not found: $WORKER" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "[hpc-batch] config not found: $CONFIG_PATH" >&2; exit 1; }
mkdir -p "$LOGDIR"

echo "=== HPC Batch Job ==="
echo "  Job name:   ${SLURM_JOB_NAME:-<none>}"
echo "  Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "  Node:       $(hostname)"
echo "  Container:  $APPTAINER_BIN exec  ($SIF_IMAGE)"
echo "  Worker:     $WORKER"
echo "  Config:     $CONFIG_PATH"
echo "  Exp name:   $EXP_NAME"
echo "  LOGDIR:     $LOGDIR"
echo "  Forwarded:  ${FORWARD[*]+${FORWARD[*]}}"
echo ""

# ===== Bind mounts =====
# Always bind the project at the same path so sac_block_e2e.sh resolves PROJECT_ROOT
# identically inside and outside the container. Bind LOGDIR separately only if it lives
# outside the project tree. Then any user-specified extra binds from hpc_env.bash.
BIND_ARGS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
case "$LOGDIR" in
    "$PROJECT_ROOT"/*|"$PROJECT_ROOT") ;;                 # already covered by the project bind
    *) BIND_ARGS+=(--bind "$LOGDIR:$LOGDIR") ;;
esac
for _b in $APPTAINER_BINDS; do
    BIND_ARGS+=(--bind "$_b")
done

# ===== Run worker inside the container =====
# --nv exposes the GPU. We export into the container: PYTHON (the in-container python the
# worker invokes), LOGDIR (so outputs land where we bound them), and TORCHDYNAMO_DISABLE
# (the worker sets this too, but set it here so it's present from process start).
#
# Pip-based Isaac Sim 5.1 in a read-only .sif needs (validated via hpc/run.sh):
#   --writable-tmpfs       : ephemeral overlay so Isaac Sim can write its EULA_ACCEPTED
#                            marker into the otherwise read-only image.
#   OMNI_KIT_ACCEPT_EULA   : auto-accept the EULA so a batch job never hangs on the
#                            interactive Yes/No prompt.
#   --home CACHE:/root     : kit/shader caches (GBs) go to the big share, not home NFS quota.
#
# `exec` replaces this bash process with the container process so SLURM's --signal=TERM@300
# is delivered straight to it (no forwarding shim), matching the RoboNuke pattern. The
# `${FORWARD[@]+...}` guard keeps an EMPTY forwarded-args array from tripping `set -u` on
# the older bash found on many clusters.
mkdir -p "$ISAAC_CACHE_HOME"

# wandb: online needs an API key, but unattended jobs can't do the interactive login and
# --containall hides host creds. Default to OFFLINE so a wandb:true config logs locally
# instead of crashing on "No API key configured"; if WANDB_API_KEY is set, go online and
# pass it in. An explicit WANDB_MODE always wins.
WANDB_ENV=(--env WANDB_MODE="${WANDB_MODE:-offline}")
WANDB_DESC="offline (no key)"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    if [[ "${#WANDB_API_KEY}" -ge 40 ]]; then
        WANDB_ENV=(--env WANDB_API_KEY="$WANDB_API_KEY" --env WANDB_MODE="${WANDB_MODE:-online}")
        WANDB_DESC="online (key len ${#WANDB_API_KEY})"
    else
        # A malformed key in online mode makes wandb.init raise and KILL training. Degrade
        # to offline instead of crashing the whole job over a bad paste.
        echo "[hpc-batch] WARNING: WANDB_API_KEY is ${#WANDB_API_KEY} chars (wandb needs 40+) — ignoring it, using offline. Fix it in hpc_env.bash for online logging." >&2
        WANDB_DESC="offline (key too short: ${#WANDB_API_KEY})"
    fi
fi

echo "[hpc-batch] launching container worker...  (wandb: $WANDB_DESC)"
exec "$APPTAINER_BIN" exec --nv --writable-tmpfs \
    --home "$ISAAC_CACHE_HOME:/root" \
    "${BIND_ARGS[@]}" \
    --env PYTHON="$CONTAINER_PYTHON" \
    --env LOGDIR="$LOGDIR" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env TORCHDYNAMO_DISABLE=1 \
    --env PYTHONUNBUFFERED=1 \
    "${WANDB_ENV[@]}" \
    "$SIF_IMAGE" \
    bash "$WORKER" "$CONFIG_PATH" "$EXP_NAME" ${FORWARD[@]+"${FORWARD[@]}"}
