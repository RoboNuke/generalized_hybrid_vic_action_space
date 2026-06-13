#!/usr/bin/env bash
# launchers/record_group.bash — record EVERY agent of EVERY run under a log dir.
#
# Usage:
#   record_group.bash <record_config> <group_path> [extra args for record.py ...]
#
#   e.g.
#   record_group.bash configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml runs/full_ctrl
#
# <group_path> contains one subfolder per experiment (1_fixed/, 5_MATCH/, ...), each
# of which contains per-agent subfolders 0/, 1/, ... For each experiment it calls
# record_experiment.bash, which in turn records each agent.
#
# Runs sequentially; a failing experiment is logged and the loop continues. Set
# PYTHON=/isaac-sim/python.sh before invoking (forwarded via the environment).
set -uo pipefail

# ===== Args =====
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <record_config> <group_path> [extra args for record.py ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml runs/full_ctrl" >&2
    exit 2
fi
RECORD_CONFIG="$1"
GROUP_PATH="$2"
shift 2
EXTRA_ARGS=("$@")

# ===== Derived paths =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PER_EXP_LAUNCHER="$SCRIPT_DIR/record_experiment.bash"

[[ "$RECORD_CONFIG" != /* ]] && RECORD_CONFIG="$PROJECT_ROOT/$RECORD_CONFIG"
[[ "$GROUP_PATH"    != /* ]] && GROUP_PATH="$PROJECT_ROOT/$GROUP_PATH"

# ===== Sanity =====
[[ -f "$PER_EXP_LAUNCHER" ]] || { echo "[rec-grp] per-exp launcher not found: $PER_EXP_LAUNCHER" >&2; exit 1; }
[[ -f "$RECORD_CONFIG" ]]    || { echo "[rec-grp] record config not found: $RECORD_CONFIG" >&2; exit 1; }
[[ -d "$GROUP_PATH" ]]       || { echo "[rec-grp] group path not found: $GROUP_PATH" >&2; exit 1; }

# ===== Find experiment subfolders (immediate subdirs holding agent checkpoints) =====
EXP_DIRS=()
shopt -s nullglob
for d in "$GROUP_PATH"/*/; do
    if compgen -G "${d}*/checkpoints/ckpt_*.pt" >/dev/null; then
        EXP_DIRS+=("${d%/}")
    fi
done
shopt -u nullglob
if [[ ${#EXP_DIRS[@]} -eq 0 ]]; then
    echo "[rec-grp] no experiment subfolders with agent checkpoints found under $GROUP_PATH" >&2
    exit 1
fi
IFS=$'\n' EXP_DIRS=($(sort <<<"${EXP_DIRS[*]}")); unset IFS

echo "[rec-grp] group=$GROUP_PATH  experiments=${#EXP_DIRS[@]}  record_config=$RECORD_CONFIG"

# ===== Record each experiment =====
PASSED=()
FAILED=()
for exp_dir in "${EXP_DIRS[@]}"; do
    name="$(basename "$exp_dir")"
    echo ""
    echo "[rec-grp] ===================================================================="
    echo "[rec-grp] === EXPERIMENT: $name ($exp_dir)"
    echo "[rec-grp] ===================================================================="
    rc=0
    bash "$PER_EXP_LAUNCHER" "$RECORD_CONFIG" "$exp_dir" "${EXTRA_ARGS[@]}" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "[rec-grp] OK: $name"
        PASSED+=("$name")
    else
        echo "[rec-grp] FAILED (exit $rc): $name — continuing" >&2
        FAILED+=("$name (exit $rc)")
    fi
done

# ===== Summary =====
echo ""
echo "[rec-grp] ===================================================================="
echo "[rec-grp] DONE for $GROUP_PATH. ${#PASSED[@]} experiments ok, ${#FAILED[@]} failed"
for n in "${PASSED[@]}"; do echo "[rec-grp]   OK  : $n"; done
for n in "${FAILED[@]}"; do echo "[rec-grp]   FAIL: $n"; done
echo "[rec-grp] ===================================================================="

[[ ${#FAILED[@]} -eq 0 ]]
