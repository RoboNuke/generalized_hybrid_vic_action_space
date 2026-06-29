#!/usr/bin/env bash
# launchers/exp_file_launcher.bash — run sac_block_e2e.sh over every config in a folder.
#
# Usage:
#   exp_file_launcher.bash <config_folder> [--skip_existing] [extra args passed through to sac_block_e2e.sh ...]
#
# For each *.yaml file directly inside <config_folder> it calls:
#   bash launchers/sac_block_e2e.sh <config_path> <exp_name> --record [extra args...]
# where <exp_name> is the config filename with the .yaml suffix and the folder
# path stripped (e.g. configs/exp_cfgs/sac_PiH/VIC.yaml -> VIC).
#
# --skip_existing (launcher-only flag, not forwarded): before running a config,
# resolve the experiment output dir the same way runner.py does
# (<logdir>/<family>/<exp_name>, with the legacy collapse) and SKIP the config if
# that folder already exists and is non-empty. Lets you re-invoke a batch to fill
# in only the runs that have not been done yet, without re-training finished ones.
#
# --record is passed so each run, after training (and eval), records a best-policy
# (ckpt_best.pt) grid GIF per agent into <EXP_DIR>/<i>/videos/, using <config_folder>/_record.yaml
# as the recorder overlay. Underscore-prefixed configs (e.g. _record.yaml) are NOT trained —
# they are overlays, so they are skipped when collecting configs.
#
# Runs sequentially, one at a time. If any single run errors out, it is logged
# and the launcher continues to the next config rather than aborting the batch.
#
# Unlike the per-run launcher we do NOT set -e here: a failing child must not
# kill the loop. We track failures and report a summary (and nonzero exit) at the end.
set -uo pipefail

# ===== Args =====
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_folder> [--skip_existing] [extra args for sac_block_e2e.sh ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/sac_PiH" >&2
    exit 2
fi
CONFIG_FOLDER="$1"
shift
# Pull out the launcher-only --skip_existing flag; everything else is forwarded to
# the per-run launcher. We also note --experiment_directory (if present) because it
# overrides the experiment "family" dir and is needed to resolve the skip target.
SKIP_EXISTING=0
EXP_DIR_OVERRIDE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_existing) SKIP_EXISTING=1 ;;
        --experiment_directory)
            [[ $# -ge 2 ]] || { echo "[batch] --experiment_directory requires a value" >&2; exit 2; }
            EXP_DIR_OVERRIDE="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# ===== Derived paths =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Mirror sac_block_e2e.sh: LOGDIR defaults to <project_root>/runs and can be
# overridden via the LOGDIR env var. Used only to resolve --skip_existing targets.
LOGDIR="${LOGDIR:-$PROJECT_ROOT/runs}"
PYTHON="${PYTHON:-python}"
PER_RUN_LAUNCHER="$SCRIPT_DIR/sac_block_e2e.sh"

[[ -d "$CONFIG_FOLDER" ]] || { echo "[batch] config folder not found: $CONFIG_FOLDER" >&2; exit 1; }
[[ -f "$PER_RUN_LAUNCHER" ]] || { echo "[batch] per-run launcher not found: $PER_RUN_LAUNCHER" >&2; exit 1; }

# ===== Collect configs =====
# Only *.yaml directly in the folder (non-recursive), sorted for deterministic order.
shopt -s nullglob
CONFIGS=("$CONFIG_FOLDER"/*.yaml)
shopt -u nullglob
# Drop overlay/helper configs (underscore-prefixed, e.g. _record.yaml): they are recorder/eval
# overlays merged onto a base config, not standalone training configs.
FILTERED=()
for c in "${CONFIGS[@]}"; do
    b="$(basename -- "$c")"
    if [[ "$b" == _* ]]; then
        echo "[batch] skipping overlay config (underscore-prefixed): $b"
        continue
    fi
    FILTERED+=("$c")
done
CONFIGS=("${FILTERED[@]}")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[batch] no trainable *.yaml files found in $CONFIG_FOLDER" >&2
    exit 1
fi
IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS

echo "[batch] found ${#CONFIGS[@]} config(s) in $CONFIG_FOLDER"

# ===== --skip_existing helper =====
# Resolve the per-run output dir for a config exactly as runner.py does:
#   <logdir>/<family>/<exp_name>, where family = sac_cfg/ppo_cfg.experiment.directory
#   (overridden by --experiment_directory), with the legacy collapse when the family
#   basename equals the logdir basename. Prints the resolved path on stdout.
resolve_exp_dir() {
    local config_path="$1" exp_name="$2"
    "$PYTHON" - "$config_path" "$LOGDIR" "$EXP_DIR_OVERRIDE" "$exp_name" "$PROJECT_ROOT" <<'PY'
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

# ===== Run each config =====
FAILED=()
PASSED=()
SKIPPED=()
for config_path in "${CONFIGS[@]}"; do
    base="$(basename -- "$config_path")"   # strip folder path
    exp_name="${base%.yaml}"               # strip .yaml suffix

    # --skip_existing: if the resolved output dir already exists and is non-empty,
    # this config has already been run — skip it. If resolution fails (bad YAML,
    # python error), fall through and run it rather than silently skipping.
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        exp_dir="$(resolve_exp_dir "$config_path" "$exp_name")" || exp_dir=""
        if [[ -n "$exp_dir" && -d "$exp_dir" ]] && compgen -G "$exp_dir/*" >/dev/null; then
            echo "[batch] SKIP (exists): $exp_name -> $exp_dir"
            SKIPPED+=("$exp_name")
            continue
        fi
    fi

    echo ""
    echo "[batch] ===================================================================="
    echo "[batch] === RUN: $config_path  (exp_name=$exp_name)"
    echo "[batch] ===================================================================="

    rc=0
    # --record: after training+eval, record each agent's best policy (ckpt_best.pt) to a GIF.
    # The per-run launcher defaults the record overlay to <config_dir>/_record.yaml.
    bash "$PER_RUN_LAUNCHER" "$config_path" "$exp_name" --record "${EXTRA_ARGS[@]}" || rc=$?

    if [[ "$rc" -eq 0 ]]; then
        echo "[batch] OK: $exp_name"
        PASSED+=("$exp_name")
    else
        echo "[batch] FAILED (exit $rc): $exp_name — continuing to next config" >&2
        FAILED+=("$exp_name (exit $rc)")
    fi
done

# ===== Summary =====
echo ""
echo "[batch] ===================================================================="
echo "[batch] DONE. ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped (of ${#CONFIGS[@]} total)"
for name in "${PASSED[@]}"; do echo "[batch]   PASS: $name"; done
for name in "${FAILED[@]}"; do echo "[batch]   FAIL: $name"; done
for name in "${SKIPPED[@]}"; do echo "[batch]   SKIP: $name"; done
echo "[batch] ===================================================================="

# Nonzero exit if any run failed, but only after attempting all of them.
[[ ${#FAILED[@]} -eq 0 ]]
