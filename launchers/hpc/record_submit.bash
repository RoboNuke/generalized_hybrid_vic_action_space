#!/usr/bin/env bash
# launchers/hpc/record_submit.bash — submit one SLURM job per AGENT to record its policy.
#
# The HPC analog of launchers/record_group.bash + launchers/record_experiment.bash: instead
# of looping over agents sequentially in-process, it submits an independent `sbatch` job per
# agent, each of which enters the container and runs learning/record.py (via hpc_record.bash).
#
# It handles BOTH "record group" and "record experiment" granularity with one command: it
# scans the path you give it for every agent folder beneath it (any dir with
# checkpoints/ckpt_*.pt) and submits one job each. So:
#   * a GROUP path      (runs/glued_surface)             -> every agent of every experiment
#   * an EXPERIMENT path (runs/glued_surface/1_fixed.yaml) -> every agent of that experiment
#   * a single AGENT     (runs/glued_surface/1_fixed.yaml/0) -> just that agent
#
# Usage:
#   record_submit.bash <record_config> <path> [--record_tag TAG] [--skip_existing] \
#                      [extra args forwarded to learning/record.py ...]
#
#   e.g.
#   record_submit.bash configs/exp_cfgs/glued_surface/_record.yaml runs/glued_surface
#
# By default each job records the agent's BEST checkpoint (record.py --checkpoint_step best);
# override by forwarding your own, e.g. `... -- --checkpoint_step 400` or `--num_trajectories 48`.
#
# Launcher-only flags (NOT forwarded to record.py):
#   --record_tag TAG   Names the jobs and the exp_logs/<TAG>/ subdir. Defaults to the scan
#                      path's basename (e.g. glued_surface).
#   --skip_existing    Skip agents that already have a video (videos/*.mp4 or *.gif) so a
#                      re-submit only fills the gaps.
#
# Everything else is forwarded VERBATIM to learning/record.py through the `--` boundary.
#
# A failing single submission is logged and does not abort the batch.
set -uo pipefail

# ===== Load central config (paths, container, SLURM resources) =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_env.bash
source "$SCRIPT_DIR/hpc_env.bash"

# Record jobs are short (a handful of rollouts), so default to a tighter wall-clock than the
# training jobs — still overridable via the HPC_RECORD_TIME env var at submit time.
: "${HPC_RECORD_TIME:=0-02:00:00}"

# ===== Args =====
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <record_config> <path> [--record_tag TAG] [--skip_existing] [extra args for record.py ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/glued_surface/_record.yaml runs/glued_surface" >&2
    exit 2
fi
RECORD_CONFIG="$1"
SCAN_PATH="$2"
shift 2

RECORD_TAG=""
SKIP_EXISTING=0
PASSTHROUGH=()             # forwarded to learning/record.py after `--`
while [[ $# -gt 0 ]]; do
    case "$1" in
        --record_tag)
            [[ $# -ge 2 ]] || { echo "[hpc-rec-submit] --record_tag requires a value" >&2; exit 2; }
            RECORD_TAG="$2"; shift ;;
        --skip_existing) SKIP_EXISTING=1 ;;
        *) PASSTHROUGH+=("$1") ;;
    esac
    shift
done

# ===== Derived paths =====
PROJECT_ROOT_LOCAL="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/hpc_record.bash"

# Resolve project-root-relative inputs to absolute (jobs bind the project at the same path).
[[ "$RECORD_CONFIG" != /* ]] && RECORD_CONFIG="$PROJECT_ROOT_LOCAL/$RECORD_CONFIG"
[[ "$SCAN_PATH"     != /* ]] && SCAN_PATH="$PROJECT_ROOT_LOCAL/$SCAN_PATH"
SCAN_PATH="${SCAN_PATH%/}"

# Default the record tag to the scan path's basename.
[[ -n "$RECORD_TAG" ]] || RECORD_TAG="$(basename -- "$SCAN_PATH")"
SLURM_OUT_DIR="$EXP_LOG_DIR/record_$RECORD_TAG"

# Default to the best checkpoint unless the caller already passed a --checkpoint_step. record.py
# uses argparse (last wins), so putting our default FIRST lets an explicit override still win.
FORWARD_DEFAULTS=()
_has_ckpt_step=0
for a in ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; do
    [[ "$a" == "--checkpoint_step" ]] && _has_ckpt_step=1
done
[[ "$_has_ckpt_step" -eq 0 ]] && FORWARD_DEFAULTS=(--checkpoint_step best)

# ===== Sanity (fail BEFORE queuing anything) =====
[[ -f "$JOB_SCRIPT"    ]] || { echo "[hpc-rec-submit] job script not found: $JOB_SCRIPT" >&2; exit 1; }
[[ -f "$RECORD_CONFIG" ]] || { echo "[hpc-rec-submit] record config not found: $RECORD_CONFIG" >&2; exit 1; }
[[ -d "$SCAN_PATH"     ]] || { echo "[hpc-rec-submit] scan path not found: $SCAN_PATH" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "[hpc-rec-submit] 'sbatch' not found — are you on an HPC login node?" >&2; exit 1; }
hpc_require_container || exit 1
mkdir -p "$SLURM_OUT_DIR"

# ===== Discover agent dirs (any dir holding checkpoints/ckpt_*.pt beneath SCAN_PATH) =====
# Works whether SCAN_PATH is a group, an experiment, or a single agent dir.
AGENT_DIRS=()
while IFS= read -r ckpt_dir; do
    if compgen -G "$ckpt_dir/ckpt_*.pt" >/dev/null 2>&1; then
        AGENT_DIRS+=("$(dirname "$ckpt_dir")")
    fi
done < <(find "$SCAN_PATH" -type d -name checkpoints -print | sort)

if [[ ${#AGENT_DIRS[@]} -eq 0 ]]; then
    echo "[hpc-rec-submit] no agent folders with checkpoints/ckpt_*.pt found under $SCAN_PATH" >&2
    exit 1
fi
IFS=$'\n' AGENT_DIRS=($(sort -u <<<"${AGENT_DIRS[*]}")); unset IFS

echo "[hpc-rec-submit] scan path: $SCAN_PATH"
echo "[hpc-rec-submit] agents found: ${#AGENT_DIRS[@]}   record tag: $RECORD_TAG   slurm logs: $SLURM_OUT_DIR"
echo "[hpc-rec-submit] record config: $RECORD_CONFIG"
echo "[hpc-rec-submit] forwarded to record.py: ${FORWARD_DEFAULTS[*]+${FORWARD_DEFAULTS[*]}} ${PASSTHROUGH[*]+${PASSTHROUGH[*]}}"
echo ""

# ===== Resource flags from hpc_env.bash (override hpc_record.bash's #SBATCH fallbacks) =====
SBATCH_RES=(
    -A "$HPC_ACCOUNT"
    -p "$HPC_PARTITIONS"
    --time="$HPC_RECORD_TIME"
    --gres="$HPC_GRES"
    --mem="$HPC_MEM"
    -c "$HPC_CPUS"
    --signal="$HPC_SIGNAL"
    # Pass the REAL launcher dir into the job env: sbatch copies the job script to a spool
    # dir, so hpc_record.bash can't find hpc_env.bash via BASH_SOURCE. ALL also propagates
    # any HPC_* env overrides used at submit time.
    --export="ALL,HPC_LAUNCHER_DIR=$SCRIPT_DIR"
)

# ===== Submit one job per agent =====
SUBMITTED=()
SKIPPED=()
FAILED=()
for agent_dir in "${AGENT_DIRS[@]}"; do
    # A stable, filesystem-safe label from the agent's path relative to the scan root.
    if [[ "$agent_dir" == "$SCAN_PATH" ]]; then
        label="$(basename -- "$agent_dir")"
    else
        label="${agent_dir#"$SCAN_PATH"/}"
    fi
    safe_label="${label//\//__}"          # 1_fixed.yaml/0 -> 1_fixed.yaml__0

    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        if compgen -G "$agent_dir/videos/*.mp4" >/dev/null 2>&1 \
           || compgen -G "$agent_dir/videos/*.gif" >/dev/null 2>&1; then
            echo "[hpc-rec-submit] SKIP (video exists): $label"
            SKIPPED+=("$label")
            continue
        fi
    fi

    job_name="record_${RECORD_TAG}_${safe_label}"
    out_pat="$SLURM_OUT_DIR/${safe_label}_%j.out"
    err_pat="$SLURM_OUT_DIR/${safe_label}_%j.err"

    echo "[hpc-rec-submit] submitting: $job_name  (agent_dir=$agent_dir)"
    if sbatch "${SBATCH_RES[@]}" \
            -J "$job_name" \
            -o "$out_pat" \
            -e "$err_pat" \
            "$JOB_SCRIPT" \
            --agent_dir "$agent_dir" --record_config "$RECORD_CONFIG" \
            -- ${FORWARD_DEFAULTS[@]+"${FORWARD_DEFAULTS[@]}"} ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
        SUBMITTED+=("$job_name")
    else
        echo "[hpc-rec-submit] FAILED to submit: $job_name — continuing" >&2
        FAILED+=("$job_name")
    fi
done

# ===== Summary =====
echo ""
echo "[hpc-rec-submit] ===================================================================="
echo "[hpc-rec-submit] DONE. ${#SUBMITTED[@]} submitted, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped (of ${#AGENT_DIRS[@]} agents)"
for n in ${SUBMITTED[@]+"${SUBMITTED[@]}"}; do echo "[hpc-rec-submit]   SUBMITTED: $n"; done
for n in ${SKIPPED[@]+"${SKIPPED[@]}"};   do echo "[hpc-rec-submit]   SKIP:      $n"; done
for n in ${FAILED[@]+"${FAILED[@]}"};     do echo "[hpc-rec-submit]   FAIL:      $n"; done
echo "[hpc-rec-submit] ===================================================================="

[[ ${#FAILED[@]} -eq 0 ]]
