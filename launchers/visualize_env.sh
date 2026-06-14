#!/usr/bin/env bash
# launchers/visualize_env.sh — boot a config in the Isaac Sim GUI to eyeball the env.
#
# Always windowed (the underlying tool refuses --headless). Optionally drives a
# trained policy; otherwise random actions. Builds the env EXACTLY like
# launchers/sac_block_e2e.sh / learning/runner.py (via learning.env_setup.build_env).
#
# Usage:
#   visualize_env.sh <config_path> [--checkpoint <dir>] [--checkpoint_step <n>] \
#                    [--seed <n>] [--task <id>]
#
# Always spawns exactly one env.
# In-window controls: [r] reset (_reset_idx)  [p] pause/resume  [q] quit
#
# Like sac_block_e2e.sh, this does NOT manage python envs — activate the right
# one first, or point PYTHON at the container wrapper (e.g. PYTHON=/isaac-sim/python.sh).
set -Eeuo pipefail
trap 'echo "[visualize] FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_path> [--checkpoint <dir>] [--checkpoint_step <n>] [--seed <n>] [--task <id>]" >&2
    echo "  e.g. $0 configs/exp_cfgs/fixed_angle_peg_FORGE/4-1_rotated_fixed.yaml" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TOOL="$PROJECT_ROOT/utilities/visualize_env.py"
[[ -f "$TOOL" ]] || { echo "[visualize] tool not found: $TOOL" >&2; exit 1; }

CONFIG_PATH="$1"; shift
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi
[[ -f "$CONFIG_PATH" ]] || { echo "[visualize] config not found: $CONFIG_PATH" >&2; exit 1; }

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null \
    || { echo "[visualize] python interpreter '$PYTHON' not found — set PYTHON=/path/to/python (e.g. /isaac-sim/python.sh) or put one on PATH" >&2; exit 1; }

echo "[visualize] python=$(command -v "$PYTHON")  config=$CONFIG_PATH"
exec "$PYTHON" "$TOOL" "$CONFIG_PATH" "$@"
