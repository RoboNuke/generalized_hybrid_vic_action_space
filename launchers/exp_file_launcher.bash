#!/usr/bin/env bash
# launchers/exp_file_launcher.bash — run sac_block_e2e.sh over every config in a folder.
#
# Usage:
#   exp_file_launcher.bash <config_folder> [extra args passed through to sac_block_e2e.sh ...]
#
# For each *.yaml file directly inside <config_folder> it calls:
#   bash launchers/sac_block_e2e.sh <config_path> <exp_name> --record [extra args...]
# where <exp_name> is the config filename with the .yaml suffix and the folder
# path stripped (e.g. configs/exp_cfgs/sac_PiH/VIC.yaml -> VIC).
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
    echo "Usage: $0 <config_folder> [extra args for sac_block_e2e.sh ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/sac_PiH" >&2
    exit 2
fi
CONFIG_FOLDER="$1"
shift
EXTRA_ARGS=("$@")

# ===== Derived paths =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

# ===== Run each config =====
FAILED=()
PASSED=()
for config_path in "${CONFIGS[@]}"; do
    base="$(basename -- "$config_path")"   # strip folder path
    exp_name="${base%.yaml}"               # strip .yaml suffix

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
echo "[batch] DONE. ${#PASSED[@]} passed, ${#FAILED[@]} failed (of ${#CONFIGS[@]} total)"
for name in "${PASSED[@]}"; do echo "[batch]   PASS: $name"; done
for name in "${FAILED[@]}"; do echo "[batch]   FAIL: $name"; done
echo "[batch] ===================================================================="

# Nonzero exit if any run failed, but only after attempting all of them.
[[ ${#FAILED[@]} -eq 0 ]]
