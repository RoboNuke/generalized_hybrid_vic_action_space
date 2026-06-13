#!/usr/bin/env bash
# launchers/record_experiment.bash — record EVERY agent of ONE trained run.
#
# Usage:
#   record_experiment.bash <record_config> <exp_path> [extra args for record.py ...]
#
#   e.g.
#   record_experiment.bash configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml \
#                          runs/full_ctrl/1_fixed
#
# <exp_path> is a single experiment dir containing per-agent subfolders 0/, 1/, ...
# (each with checkpoints/ckpt_*.pt and config.yaml). For each agent subfolder it
# calls learning/record.py, producing an agent-specific GIF at <agent>/videos/.
#
# Runs sequentially; a failing agent is logged and the loop continues. Set
# PYTHON=/isaac-sim/python.sh before invoking.
set -uo pipefail

# ===== Args =====
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <record_config> <exp_path> [extra args for record.py ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/fixed_angle_peg_FORGE/_record.yaml runs/full_ctrl/1_fixed" >&2
    exit 2
fi
RECORD_CONFIG="$1"
EXP_PATH="$2"
shift 2
EXTRA_ARGS=("$@")

# ===== Derived paths =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RECORD_PY="$PROJECT_ROOT/learning/record.py"

[[ "$RECORD_CONFIG" != /* ]] && RECORD_CONFIG="$PROJECT_ROOT/$RECORD_CONFIG"
[[ "$EXP_PATH"      != /* ]] && EXP_PATH="$PROJECT_ROOT/$EXP_PATH"

# ===== Sanity =====
[[ -f "$RECORD_PY" ]]     || { echo "[rec-exp] record.py not found: $RECORD_PY" >&2; exit 1; }
[[ -f "$RECORD_CONFIG" ]] || { echo "[rec-exp] record config not found: $RECORD_CONFIG" >&2; exit 1; }
[[ -d "$EXP_PATH" ]]      || { echo "[rec-exp] exp path not found: $EXP_PATH" >&2; exit 1; }

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null \
    || { echo "[rec-exp] python interpreter '$PYTHON' not found — set PYTHON=/path/to/python (e.g. /isaac-sim/python.sh)" >&2; exit 1; }

# ===== Find agent subfolders (immediate subdirs with checkpoints/ckpt_*.pt) =====
AGENT_DIRS=()
shopt -s nullglob
for d in "$EXP_PATH"/*/; do
    if compgen -G "${d}checkpoints/ckpt_*.pt" >/dev/null; then
        AGENT_DIRS+=("${d%/}")
    fi
done
shopt -u nullglob
if [[ ${#AGENT_DIRS[@]} -eq 0 ]]; then
    echo "[rec-exp] no agent subfolders with checkpoints found under $EXP_PATH" >&2
    exit 1
fi
IFS=$'\n' AGENT_DIRS=($(sort <<<"${AGENT_DIRS[*]}")); unset IFS

echo "[rec-exp] exp=$EXP_PATH  agents=${#AGENT_DIRS[@]}  record_config=$RECORD_CONFIG"

# ===== Record each agent =====
PASSED=()
FAILED=()
for agent_dir in "${AGENT_DIRS[@]}"; do
    name="$(basename "$(dirname "$agent_dir")")/$(basename "$agent_dir")"  # e.g. 1_fixed/0
    echo ""
    echo "[rec-exp] === RECORD agent: $name ($agent_dir)"
    rc=0
    "$PYTHON" "$RECORD_PY" \
        --agent_dir "$agent_dir" \
        --record_config "$RECORD_CONFIG" \
        --headless \
        "${EXTRA_ARGS[@]}" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "[rec-exp] OK: $name"
        PASSED+=("$name")
    else
        echo "[rec-exp] FAILED (exit $rc): $name — continuing" >&2
        FAILED+=("$name (exit $rc)")
    fi
done

# ===== Summary =====
echo ""
echo "[rec-exp] DONE for $EXP_PATH. ${#PASSED[@]} ok, ${#FAILED[@]} failed"
for n in "${PASSED[@]}"; do echo "[rec-exp]   OK  : $n"; done
for n in "${FAILED[@]}"; do echo "[rec-exp]   FAIL: $n"; done

[[ ${#FAILED[@]} -eq 0 ]]
