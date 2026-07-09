#!/usr/bin/env bash
#SBATCH -J HPC_RECORD                      # job name (overridden by sbatch -J from the submitter)
#SBATCH -A virl-grp                        # sponsored account (overridden by sbatch -A)
#SBATCH -p tiamat,dgxh,gpu      # partitions (overridden by sbatch -p)
#SBATCH --time=0-02:00:00                  # wall-clock limit (overridden by sbatch --time)
#SBATCH --gres=gpu:1                       # GPUs (overridden by sbatch --gres)
#SBATCH --mem=32G                          # host memory (overridden by sbatch --mem)
#SBATCH -c 12                              # cores/threads (overridden by sbatch -c)
#SBATCH --signal=TERM@120                  # SIGTERM 120s before the limit (overridden by sbatch --signal)
#
# launchers/hpc/hpc_record.bash — the per-AGENT SLURM job. Runs on a compute node.
#
# The recording analog of hpc_batch.bash: instead of running sac_block_e2e.sh (train ->
# eval -> record) it enters the Apptainer/Singularity container and runs learning/record.py
# INSIDE it for ONE trained agent. record.py reloads that agent's snapshotted config.yaml,
# builds the env, loads its checkpoint (best by default), rolls out, and writes an
# agent-specific video under <agent_dir>/videos/. All container and resource settings come
# from launchers/hpc/hpc_env.bash, which this script sources.
#
# Usage (normally invoked by record_submit.bash, not by hand):
#   sbatch [resource flags] hpc_record.bash \
#          --agent_dir <path> --record_config <overlay.yaml> [-- <forwarded record.py args...>]
#
# Everything after the literal `--` is forwarded VERBATIM to learning/record.py, so any of
# its flags pass straight through:  -- --checkpoint_step best --num_trajectories 48
#
# The #SBATCH directives above are only FALLBACK defaults so the script is runnable
# standalone; the submitter passes the hpc_env.bash values as sbatch CLI flags, which win.
#
# Fail loud, fail fast.
set -Eeuo pipefail
trap 'echo "[hpc-record] FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

# ===== Load central config =====
# Under sbatch the script is COPIED into a spool dir, so BASH_SOURCE no longer points at
# launchers/hpc and a sibling source would miss. record_submit.bash passes the real dir
# via HPC_LAUNCHER_DIR (--export); fall back to BASH_SOURCE for standalone/manual runs.
SCRIPT_DIR="${HPC_LAUNCHER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=hpc_env.bash
source "$SCRIPT_DIR/hpc_env.bash"

# ===== Args =====
AGENT_DIR=""
RECORD_CONFIG=""
FORWARD=()                 # everything after `--`, passed through to learning/record.py
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent_dir)     [[ $# -ge 2 ]] || { echo "[hpc-record] --agent_dir requires a value" >&2; exit 2; }
                         AGENT_DIR="$2"; shift 2 ;;
        --record_config) [[ $# -ge 2 ]] || { echo "[hpc-record] --record_config requires a value" >&2; exit 2; }
                         RECORD_CONFIG="$2"; shift 2 ;;
        --)              shift; FORWARD=("$@"); break ;;
        *)               echo "[hpc-record] unknown argument before '--': $1" >&2; exit 2 ;;
    esac
done

[[ -n "$AGENT_DIR"     ]] || { echo "[hpc-record] --agent_dir is required" >&2; exit 2; }
[[ -n "$RECORD_CONFIG" ]] || { echo "[hpc-record] --record_config is required" >&2; exit 2; }

# Resolve project-root-relative paths to absolute (the container sees the same paths).
[[ "$AGENT_DIR"     != /* ]] && AGENT_DIR="$PROJECT_ROOT/$AGENT_DIR"
[[ "$RECORD_CONFIG" != /* ]] && RECORD_CONFIG="$PROJECT_ROOT/$RECORD_CONFIG"

# ===== Sanity =====
hpc_require_container
RECORDER="$PROJECT_ROOT/learning/record.py"
[[ -f "$RECORDER"      ]] || { echo "[hpc-record] recorder not found: $RECORDER" >&2; exit 1; }
[[ -d "$AGENT_DIR"     ]] || { echo "[hpc-record] agent_dir not found: $AGENT_DIR" >&2; exit 1; }
[[ -f "$RECORD_CONFIG" ]] || { echo "[hpc-record] record config not found: $RECORD_CONFIG" >&2; exit 1; }
[[ -f "$AGENT_DIR/config.yaml" ]] \
    || { echo "[hpc-record] no config.yaml in agent_dir (record.py needs the snapshotted config): $AGENT_DIR" >&2; exit 1; }

echo "=== HPC Record Job ==="
echo "  Job name:   ${SLURM_JOB_NAME:-<none>}"
echo "  Job ID:     ${SLURM_JOB_ID:-<none>}"
echo "  Node:       $(hostname)"
echo "  Container:  $APPTAINER_BIN exec  ($SIF_IMAGE)"
echo "  Recorder:   $RECORDER"
echo "  Agent dir:  $AGENT_DIR"
echo "  Record cfg: $RECORD_CONFIG"
echo "  Forwarded:  ${FORWARD[*]+${FORWARD[*]}}"
echo ""

# ===== Bind mounts =====
# Always bind the project at the same path so record.py resolves paths identically inside
# and outside the container. Bind LOGDIR separately only if it lives outside the project
# tree (the agent dir + its videos output live under LOGDIR). Then any user-specified
# extra binds from hpc_env.bash.
BIND_ARGS=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
case "$LOGDIR" in
    "$PROJECT_ROOT"/*|"$PROJECT_ROOT") ;;                 # already covered by the project bind
    *) BIND_ARGS+=(--bind "$LOGDIR:$LOGDIR") ;;
esac
# The agent dir may live outside both PROJECT_ROOT and LOGDIR (e.g. a custom runs path); bind
# it too so the container can read the checkpoint and write the video there.
case "$AGENT_DIR" in
    "$PROJECT_ROOT"/*|"$LOGDIR"/*) ;;
    *) BIND_ARGS+=(--bind "$AGENT_DIR:$AGENT_DIR") ;;
esac
for _b in $APPTAINER_BINDS; do
    BIND_ARGS+=(--bind "$_b")
done

# ===== Run the recorder inside the container =====
# Mirrors hpc_batch.bash's container invocation (see its comments for the --writable-tmpfs /
# --home / EULA rationale). Recording renders a TiledCamera, so --nv (GPU) is required and
# --enable_cameras is auto-forced by runner.py when the record overlay flips recorder.enabled
# on. No wandb here (recording never logs to wandb).
#
# `exec` replaces this bash process with the container process so SLURM's --signal is
# delivered straight to it. The `${FORWARD[@]+...}` guard keeps an EMPTY forwarded-args
# array from tripping `set -u` on the older bash found on many clusters.
mkdir -p "$ISAAC_CACHE_HOME"

echo "[hpc-record] launching container recorder..."
exec "$APPTAINER_BIN" exec --nv --writable-tmpfs \
    --home "$ISAAC_CACHE_HOME:/root" \
    "${BIND_ARGS[@]}" \
    --env PYTHON="$CONTAINER_PYTHON" \
    --env LOGDIR="$LOGDIR" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env TORCHDYNAMO_DISABLE=1 \
    --env PYTHONUNBUFFERED=1 \
    "$SIF_IMAGE" \
    "$CONTAINER_PYTHON" "$RECORDER" \
        --agent_dir "$AGENT_DIR" \
        --record_config "$RECORD_CONFIG" \
        --headless ${FORWARD[@]+"${FORWARD[@]}"}
