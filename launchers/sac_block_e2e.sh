#!/usr/bin/env bash
# launchers/sac_block_e2e.sh — full train -> save -> load -> eval smoke test.
#
# Usage:
#   sac_block_e2e.sh <config_path> <experiment_name> [--no_eval] [--experiment_directory <dir>]
#
# Reads task / num_envs / num_agents / total_timesteps / eval_timesteps / memory_size
# from runner_cfg in the supplied YAML. Override anything one-off via runner CLI flags
# in the python invocations below.
#
# Flags:
#   --no_eval                      Skip the post-training eval pass (still verifies checkpoints exist).
#   --experiment_directory <dir>   Override sac_cfg.experiment.directory (the "family" subdir
#                                  under <logdir>); lets you save runs to different places.
#   --record                       After training (and eval), record a best-policy grid GIF for
#                                  each agent (loads ckpt_best.pt) into <EXP_DIR>/<i>/videos/.
#   --record_config <overlay>      Record overlay YAML. Defaults to <config_dir>/_record.yaml.
#
# Fail loud, fail fast: any silent miss is a bug, not an expected outcome.
set -Eeuo pipefail
trap 'echo "[launcher] FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

# We never use torch.compile. Disabling TorchDynamo sidesteps the lazy `import torch._dynamo`
# (triggered by the first optimizer construction in torch 2.8) that can hit a re-entrant/concurrent
# circular-import crash under Omniverse threads. runner.py also pre-imports torch._dynamo as a
# second layer of defense.
export TORCHDYNAMO_DISABLE=1

# ===== Args =====
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config_path> <experiment_name> [--no_eval] [--experiment_directory <dir>]" >&2
    echo "  e.g. $0 configs/exp_cfgs/cartpole.yaml cartpole_run1" >&2
    exit 2
fi
CONFIG_PATH="$1"
EXPERIMENT_NAME="$2"
shift 2
RUN_EVAL=1
EXPERIMENT_DIRECTORY=""
RECORD=0
RECORD_CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_eval) RUN_EVAL=0 ;;
        --experiment_directory)
            [[ $# -ge 2 ]] || { echo "[launcher] --experiment_directory requires a value" >&2; exit 2; }
            EXPERIMENT_DIRECTORY="$2"; shift ;;
        --record) RECORD=1 ;;
        --record_config)
            [[ $# -ge 2 ]] || { echo "[launcher] --record_config requires a value" >&2; exit 2; }
            RECORD_CONFIG="$2"; shift ;;
        *) echo "[launcher] unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ===== Derived paths =====
# Resolve PROJECT_ROOT from the script's own location so this works in any
# clone path (HPC home != local home). LOGDIR follows project root by default;
# override via env var if needed (LOGDIR=... ./launchers/sac_block_e2e.sh ...).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/runs}"

RUNNER="$PROJECT_ROOT/learning/runner.py"
# Final per-run output dir mirrors runner.py: <logdir>/<family>/<experiment_name>.
# The family subdir is sac_cfg.experiment.directory, which --experiment_directory
# overrides. Replicate the runner's legacy collapse: if family basename equals the
# logdir basename, the family level is dropped (runs/runs/<exp> -> runs/<exp>).
EXP_FAMILY_DIR="$LOGDIR"
if [[ -n "$EXPERIMENT_DIRECTORY" \
      && "$(basename "$EXPERIMENT_DIRECTORY")" != "$(basename "$LOGDIR")" ]]; then
    EXP_FAMILY_DIR="$LOGDIR/$EXPERIMENT_DIRECTORY"
fi
EXP_DIR="$EXP_FAMILY_DIR/$EXPERIMENT_NAME"
EVAL_EXP_NAME="${EXPERIMENT_NAME}_eval"

# Resolve config to absolute (allow caller to pass a project-root-relative path).
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi

# ===== Sanity =====
# We assume the caller has already activated the right python env (conda env,
# apptainer shell, venv, etc.) — the launcher does NOT manage environments.
[[ -f "$RUNNER" ]] || { echo "[launcher] runner not found: $RUNNER" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "[launcher] config not found: $CONFIG_PATH" >&2; exit 1; }
# Resolve python: PYTHON env var (e.g. PYTHON=/isaac-sim/python.sh) wins,
# else fall back to `python` on PATH. Set in your shell or sbatch script to
# point at the container's python wrapper.
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null \
    || { echo "[launcher] python interpreter '$PYTHON' not found — set PYTHON=/path/to/python (e.g. /isaac-sim/python.sh) or put one on PATH" >&2; exit 1; }

# ===== Read num_agents from YAML for the post-train checkpoint check =====
# All other runner_cfg fields (task, num_envs, etc.) flow through to runner.py
# implicitly via --config; only num_agents is needed bash-side to walk per-agent
# checkpoint dirs.
NUM_AGENTS="$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG_PATH'))['runner_cfg']['num_agents'])")"
[[ "$NUM_AGENTS" =~ ^[0-9]+$ ]] \
    || { echo "[launcher] could not read runner_cfg.num_agents from $CONFIG_PATH (got '$NUM_AGENTS')" >&2; exit 1; }

echo "[launcher] python=$(command -v "$PYTHON")  config=$CONFIG_PATH  experiment=$EXPERIMENT_NAME  num_agents=$NUM_AGENTS"

# Optional --experiment_directory passthrough: only forward the flag when the
# caller set it, so an empty value falls back to the YAML's experiment.directory.
EXP_DIR_FLAG=()
if [[ -n "$EXPERIMENT_DIRECTORY" ]]; then
    EXP_DIR_FLAG=(--experiment_directory "$EXPERIMENT_DIRECTORY")
fi

# ===== Train =====
# Ctrl-C (SIGINT, exit 130) is treated as "interrupted, proceed with whatever was last flushed
# to disk". Any other nonzero exit (OOM=137, segfault=139, ValueError from runner, etc.) is a
# hard failure: we SKIP eval but still fall through to RECORD (so videos of any best checkpoint
# are produced), then re-surface the failure code at the very end. The `|| TRAIN_RC=$?` form
# neutralizes `set -e` and the ERR trap for this one command so we can branch on the code.
echo "[launcher] === TRAIN (config=$CONFIG_PATH) ==="
TRAIN_RC=0
"$PYTHON" "$RUNNER" \
    --config "$CONFIG_PATH" \
    --experiment_name "$EXPERIMENT_NAME" \
    --logdir "$LOGDIR" \
    "${EXP_DIR_FLAG[@]}" \
    --mode train \
    --headless || TRAIN_RC=$?

TRAIN_HARD_FAIL=0
case "$TRAIN_RC" in
    0)   echo "[launcher] training completed normally" ;;
    130) echo "[launcher] training interrupted by Ctrl-C (exit 130); proceeding with last saved checkpoints" ;;
    *)   echo "[launcher] training failed with exit $TRAIN_RC (not Ctrl-C) — skipping eval, but STILL recording any best checkpoints below" >&2
         TRAIN_HARD_FAIL=1 ;;
esac

# ===== Verify checkpoints + Eval (only when training did NOT hard-fail) =====
# On a hard failure (OOM/segfault/runner error) we skip the checkpoint-existence guard and eval
# — the env may be in a bad state and eval would likely fail too — but we deliberately drop
# through to RECORD below so videos of any best checkpoint are still produced. The guard only
# aborts on a CLEAN run that silently wrote no checkpoints (the real bug it exists to catch).
if [[ "$TRAIN_HARD_FAIL" -eq 0 ]]; then
    # sac.write_checkpoint writes one file per agent at $EXP_DIR/<i>/checkpoints/ckpt_<step>.pt.
    echo "[launcher] verifying per-agent checkpoints under $EXP_DIR"
    [[ -d "$EXP_DIR" ]] || { echo "[launcher] experiment dir was not created: $EXP_DIR" >&2; exit 1; }
    for i in $(seq 0 $((NUM_AGENTS - 1))); do
        agent_ckpt_dir="$EXP_DIR/$i/checkpoints"
        [[ -d "$agent_ckpt_dir" ]] \
            || { echo "[launcher] missing checkpoint dir for agent $i: $agent_ckpt_dir" >&2; exit 1; }
        if ! compgen -G "$agent_ckpt_dir/ckpt_*.pt" >/dev/null; then
            echo "[launcher] no ckpt_*.pt files for agent $i in $agent_ckpt_dir" >&2
            exit 1
        fi
        latest_for_agent="$(ls -1 "$agent_ckpt_dir"/ckpt_*.pt | tail -1)"
        echo "[launcher]   agent $i: $latest_for_agent"
    done

    # Pass the experiment dir as --checkpoint; the runner walks 0/, 1/, ... and resolves the
    # latest ckpt per agent. A fresh experiment name keeps eval's tensorboard events out of the
    # training agent dirs. `--mode eval` uses runner_cfg.eval_timesteps.
    if [[ "$RUN_EVAL" -eq 1 ]]; then
        echo "[launcher] === EVAL (config=$CONFIG_PATH, checkpoint=$EXP_DIR) ==="
        "$PYTHON" "$RUNNER" \
            --config "$CONFIG_PATH" \
            --experiment_name "$EVAL_EXP_NAME" \
            --logdir "$LOGDIR" \
            "${EXP_DIR_FLAG[@]}" \
            --checkpoint "$EXP_DIR" \
            --mode eval \
            --headless
        echo "[launcher] done. train=$EXP_DIR  eval=$EXP_FAMILY_DIR/$EVAL_EXP_NAME"
    else
        echo "[launcher] === EVAL skipped (--no_eval) ==="
        echo "[launcher] done. train=$EXP_DIR"
    fi
fi

# ===== Record (optional) — runs even after a hard training failure =====
# Records the BEST policy (ckpt_best.pt) of each agent to a grid GIF under <EXP_DIR>/<i>/videos/.
# A fresh recorder process reloads ckpt_best.pt, so as long as a best checkpoint exists the video
# is produced regardless of how training ended (OOM, crash, Ctrl-C, or clean). Per-agent and
# non-fatal: agents with no ckpt_best.pt are skipped; render failures warn. Recording NEVER
# changes the script's exit code (the training outcome below owns that).
if [[ "$RECORD" -eq 1 ]]; then
    RECORDER="$PROJECT_ROOT/learning/record.py"
    # Default the overlay to <config_dir>/_record.yaml when not explicitly given.
    if [[ -z "$RECORD_CONFIG" ]]; then
        RECORD_CONFIG="$(dirname "$CONFIG_PATH")/_record.yaml"
    fi
    echo "[launcher] === RECORD (best policy per agent, overlay=$RECORD_CONFIG) ==="
    if [[ ! -f "$RECORDER" ]]; then
        echo "[launcher] recorder not found: $RECORDER — skipping recording" >&2
    elif [[ ! -f "$RECORD_CONFIG" ]]; then
        echo "[launcher] record overlay not found: $RECORD_CONFIG — skipping recording" >&2
    else
        for i in $(seq 0 $((NUM_AGENTS - 1))); do
            best_ckpt="$EXP_DIR/$i/checkpoints/ckpt_best.pt"
            if [[ ! -f "$best_ckpt" ]]; then
                echo "[launcher]   agent $i: no ckpt_best.pt — nothing to record, skipping" >&2
                continue
            fi
            echo "[launcher]   recording agent $i (ckpt_best.pt) -> $EXP_DIR/$i/videos/"
            rec_rc=0
            "$PYTHON" "$RECORDER" \
                --agent_dir "$EXP_DIR/$i" \
                --record_config "$RECORD_CONFIG" \
                --checkpoint_step best \
                --headless || rec_rc=$?
            if [[ "$rec_rc" -ne 0 ]]; then
                echo "[launcher]   WARNING: recording agent $i failed (exit $rec_rc) — continuing" >&2
            fi
        done
    fi
fi

# ===== Final exit code =====
# Surface a hard training failure to the caller (the batch marks it FAILED) — but only AFTER
# recording above has had its chance to produce videos of the best checkpoints.
if [[ "$TRAIN_HARD_FAIL" -eq 1 ]]; then
    echo "[launcher] exiting with training failure code $TRAIN_RC (recording, if any, ran first)" >&2
    exit "$TRAIN_RC"
fi
