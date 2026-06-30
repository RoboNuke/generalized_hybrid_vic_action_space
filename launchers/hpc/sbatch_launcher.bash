#!/usr/bin/env bash
# launchers/hpc/sbatch_launcher.bash — submit one SLURM job per config in a folder.
#
# The HPC analog of launchers/exp_file_launcher.bash: instead of running each config
# sequentially in-process, it submits an independent `sbatch` job per config, each of
# which enters the container and runs sac_block_e2e.sh (via hpc_batch.bash).
#
# Usage:
#   sbatch_launcher.bash <config_folder> [--experiment_tag TAG] [--skip_existing] \
#                        [extra args forwarded to sac_block_e2e.sh ...]
#
# For each *.yaml directly inside <config_folder> (non-recursive, sorted) it submits:
#   sbatch <resources> -J <TAG>_<exp> -o/-e exp_logs/<TAG>/<exp>_%j.{out,err} \
#          launchers/hpc/hpc_batch.bash --config <cfg> --exp_name <exp> -- <extra args...>
# where <exp> is the filename without the .yaml suffix.
#
# Launcher-only flags (NOT forwarded to the worker):
#   --experiment_tag TAG   Names the jobs and the exp_logs/<TAG>/ subdir. Defaults to the
#                          config folder's basename.
#   --skip_existing        Resolve each config's output dir the way runner.py does and SKIP
#                          submitting configs whose run dir already exists and is non-empty
#                          (re-submit a folder to fill only the gaps).
#
# Everything else after the folder is forwarded VERBATIM to sac_block_e2e.sh through the
# `--` boundary, so e.g. `... --no_eval` skips eval on every submitted job, and
# `... --experiment_directory FAM` overrides the family dir (also honored by --skip_existing).
#
# Underscore-prefixed configs (e.g. _record.yaml) are overlays, not trainable configs, and
# are skipped — same rule as exp_file_launcher.bash.
#
# A failing single submission is logged and does not abort the batch.
set -uo pipefail

# ===== Load central config (paths, container, SLURM resources) =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_env.bash
source "$SCRIPT_DIR/hpc_env.bash"

# ===== Args =====
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_folder> [--experiment_tag TAG] [--skip_existing] [extra args for sac_block_e2e.sh ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/sac_PiH --experiment_tag GAS_v1" >&2
    exit 2
fi
CONFIG_FOLDER="$1"
shift

EXPERIMENT_TAG=""
SKIP_EXISTING=0
EXP_DIR_OVERRIDE=""        # mirrors exp_file_launcher: needed to resolve --skip_existing targets
PASSTHROUGH=()             # forwarded to sac_block_e2e.sh after `--`
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment_tag)
            [[ $# -ge 2 ]] || { echo "[hpc-batch-submit] --experiment_tag requires a value" >&2; exit 2; }
            EXPERIMENT_TAG="$2"; shift ;;
        --skip_existing) SKIP_EXISTING=1 ;;
        --experiment_directory)
            [[ $# -ge 2 ]] || { echo "[hpc-batch-submit] --experiment_directory requires a value" >&2; exit 2; }
            EXP_DIR_OVERRIDE="$2"
            PASSTHROUGH+=("$1" "$2")              # also forward it to the worker
            shift ;;
        *) PASSTHROUGH+=("$1") ;;
    esac
    shift
done

# ===== Derived paths =====
PROJECT_ROOT_LOCAL="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/hpc_batch.bash"
PYTHON="${PYTHON:-python}"     # login-node python, only for --skip_existing YAML resolution
# Default the experiment tag to the config folder's basename.
[[ -n "$EXPERIMENT_TAG" ]] || EXPERIMENT_TAG="$(basename -- "$(cd -- "$CONFIG_FOLDER" 2>/dev/null && pwd || echo "$CONFIG_FOLDER")")"
SLURM_OUT_DIR="$EXP_LOG_DIR/$EXPERIMENT_TAG"

# ===== Sanity (fail BEFORE queuing anything) =====
[[ -d "$CONFIG_FOLDER" ]] || { echo "[hpc-batch-submit] config folder not found: $CONFIG_FOLDER" >&2; exit 1; }
[[ -f "$JOB_SCRIPT"    ]] || { echo "[hpc-batch-submit] job script not found: $JOB_SCRIPT" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "[hpc-batch-submit] 'sbatch' not found — are you on an HPC login node?" >&2; exit 1; }
hpc_require_container || exit 1
mkdir -p "$SLURM_OUT_DIR"

# ===== Collect configs (non-recursive, skip underscore overlays, sorted) =====
shopt -s nullglob
CONFIGS=("$CONFIG_FOLDER"/*.yaml)
shopt -u nullglob
FILTERED=()
for c in "${CONFIGS[@]}"; do
    b="$(basename -- "$c")"
    if [[ "$b" == _* ]]; then
        echo "[hpc-batch-submit] skipping overlay config (underscore-prefixed): $b"
        continue
    fi
    FILTERED+=("$c")
done
CONFIGS=("${FILTERED[@]}")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[hpc-batch-submit] no trainable *.yaml files found in $CONFIG_FOLDER" >&2
    exit 1
fi
IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS

echo "[hpc-batch-submit] found ${#CONFIGS[@]} config(s) in $CONFIG_FOLDER"
echo "[hpc-batch-submit] experiment tag: $EXPERIMENT_TAG   slurm logs: $SLURM_OUT_DIR"
echo "[hpc-batch-submit] forwarded to worker: ${PASSTHROUGH[*]+${PASSTHROUGH[*]}}"
echo ""

# ===== --skip_existing helper (resolve output dir exactly as runner.py does) =====
resolve_exp_dir() {
    local config_path="$1" exp_name="$2"
    "$PYTHON" - "$config_path" "$LOGDIR" "$EXP_DIR_OVERRIDE" "$exp_name" "$PROJECT_ROOT_LOCAL" <<'PY'
import os, sys, yaml
config_path, log_root, exp_dir_override, exp_name, project_root = sys.argv[1:6]
cfg = yaml.safe_load(open(config_path)) or {}
agent_type = str(cfg.get("runner_cfg", {}).get("agent_type", "sac")).lower()
cfg_key = "ppo_cfg" if agent_type == "ppo" else "sac_cfg"
family = (exp_dir_override or "").strip()
if not family:
    experiment = (cfg.get(cfg_key) or {}).get("experiment") or {}
    family = str(experiment.get("directory") or "").strip()
log_root_basename = os.path.basename(os.path.normpath(log_root)) if log_root else ""
family_basename = os.path.basename(os.path.normpath(family)) if family else ""
if not family or family_basename == log_root_basename:
    final_directory = log_root
else:
    final_directory = os.path.join(log_root, family)
if not os.path.isabs(final_directory):
    final_directory = os.path.join(project_root, final_directory)
print(os.path.join(final_directory, exp_name))
PY
}

# ===== Resource flags from hpc_env.bash (override hpc_batch.bash's #SBATCH fallbacks) =====
SBATCH_RES=(
    -A "$HPC_ACCOUNT"
    -p "$HPC_PARTITIONS"
    --time="$HPC_TIME"
    --gres="$HPC_GRES"
    --mem="$HPC_MEM"
    -c "$HPC_CPUS"
    --signal="$HPC_SIGNAL"
    # Pass the REAL launcher dir into the job env: sbatch copies the job script to a spool
    # dir, so hpc_batch.bash can't find hpc_env.bash via BASH_SOURCE. ALL also propagates
    # any HPC_* env overrides used at submit time.
    --export="ALL,HPC_LAUNCHER_DIR=$SCRIPT_DIR"
)

# ===== Submit one job per config =====
SUBMITTED=()
SKIPPED=()
FAILED=()
for config_path in "${CONFIGS[@]}"; do
    base="$(basename -- "$config_path")"
    exp_name="${base%.yaml}"

    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        exp_dir="$(resolve_exp_dir "$config_path" "$exp_name")" || exp_dir=""
        if [[ -n "$exp_dir" && -d "$exp_dir" ]] && compgen -G "$exp_dir/*" >/dev/null; then
            echo "[hpc-batch-submit] SKIP (exists): $exp_name -> $exp_dir"
            SKIPPED+=("$exp_name")
            continue
        fi
    fi

    job_name="${EXPERIMENT_TAG}_${exp_name}"
    out_pat="$SLURM_OUT_DIR/${exp_name}_%j.out"
    err_pat="$SLURM_OUT_DIR/${exp_name}_%j.err"

    echo "[hpc-batch-submit] submitting: $job_name  (config=$config_path)"
    if sbatch "${SBATCH_RES[@]}" \
            -J "$job_name" \
            -o "$out_pat" \
            -e "$err_pat" \
            "$JOB_SCRIPT" \
            --config "$config_path" --exp_name "$exp_name" -- ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
        SUBMITTED+=("$job_name")
    else
        echo "[hpc-batch-submit] FAILED to submit: $job_name — continuing" >&2
        FAILED+=("$job_name")
    fi
done

# ===== Summary =====
echo ""
echo "[hpc-batch-submit] ===================================================================="
echo "[hpc-batch-submit] DONE. ${#SUBMITTED[@]} submitted, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped (of ${#CONFIGS[@]} total)"
for n in ${SUBMITTED[@]+"${SUBMITTED[@]}"}; do echo "[hpc-batch-submit]   SUBMITTED: $n"; done
for n in ${SKIPPED[@]+"${SKIPPED[@]}"};   do echo "[hpc-batch-submit]   SKIP:      $n"; done
for n in ${FAILED[@]+"${FAILED[@]}"};     do echo "[hpc-batch-submit]   FAIL:      $n"; done
echo "[hpc-batch-submit] ===================================================================="

[[ ${#FAILED[@]} -eq 0 ]]
