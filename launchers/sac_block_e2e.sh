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
WANDB_TAG_FLAGS=()   # collected --wandb_tag flags, forwarded verbatim to runner.py
FIRST_WANDB_TAG=""   # first tag value, used to narrow the post-train per-step eval's run query
OVERLAY_FLAGS=()     # collected --overlay flags, forwarded verbatim to runner.py (train + eval)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_eval) RUN_EVAL=0 ;;
        --experiment_directory)
            [[ $# -ge 2 ]] || { echo "[launcher] --experiment_directory requires a value" >&2; exit 2; }
            EXPERIMENT_DIRECTORY="$2"; shift ;;
        --wandb_tag)
            [[ $# -ge 2 ]] || { echo "[launcher] --wandb_tag requires a value" >&2; exit 2; }
            WANDB_TAG_FLAGS+=("--wandb_tag" "$2")
            [[ -z "$FIRST_WANDB_TAG" ]] && FIRST_WANDB_TAG="$2"
            shift ;;
        --overlay)
            # A deep-merged YAML overlay applied over --config by runner.py BEFORE validation.
            # Repeatable; forwarded to BOTH train and eval so the env matches (e.g. a sweep
            # launcher pinning runner_cfg.rel_grasp_rot_init_deg per job). NOT passed to the
            # record step, which reloads the agent's snapshotted (already-merged) config.yaml.
            [[ $# -ge 2 ]] || { echo "[launcher] --overlay requires a value" >&2; exit 2; }
            OVERLAY_FLAGS+=("--overlay" "$2"); shift ;;
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
# EXP_FAMILY_DIR / EXP_DIR / WANDB_PROJECT are computed below — AFTER the config's
# sac_cfg.experiment.directory is read — so the worker's checkpoint/eval paths match
# runner.py's output dir even when --experiment_directory was not passed.

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

# ===== Resolve the output dir EXACTLY like runner.py (<logdir>/<family>/<exp_name>) =====
# Family subdir: --experiment_directory wins; otherwise fall back to the config's own
# sac_cfg.experiment.directory (what runner.py uses). Without this, a caller that omits
# --experiment_directory (e.g. the SLURM batch path) would look in <logdir>/<exp_name>
# while runner.py wrote to <logdir>/<family>/<exp_name>. We use a SEPARATE var (FAMILY) so
# this fallback does NOT change what is forwarded to runner.py via EXP_DIR_FLAG below.
FAMILY="$EXPERIMENT_DIRECTORY"
if [[ -z "$FAMILY" ]]; then
    FAMILY="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')) or {}; e=(c.get('sac_cfg') or {}).get('experiment') or {}; print(e.get('directory') or '')" 2>/dev/null || true)"
fi
EXP_FAMILY_DIR="$LOGDIR"
if [[ -n "$FAMILY" && "$(basename "$FAMILY")" != "$(basename "$LOGDIR")" ]]; then
    EXP_FAMILY_DIR="$LOGDIR/$FAMILY"
fi
EXP_DIR="$EXP_FAMILY_DIR/$EXPERIMENT_NAME"

# wandb identity for the post-train PER-STEP eval (record.py from-wandb pipeline). Mirrors
# make_wandb_run: project = wandb_kwargs.project if set, else the basename of the experiment
# family dir; entity from wandb_kwargs.entity. The just-trained runs live in this project under
# group = EXPERIMENT_NAME (runs "<EXPERIMENT_NAME>_agentN"), which is how record.py selects them.
WANDB_ENTITY="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')) or {}; e=(c.get('sac_cfg') or {}).get('experiment') or {}; wk=e.get('wandb_kwargs') or {}; print(wk.get('entity') or '')" 2>/dev/null || true)"
WANDB_PROJECT="$("$PYTHON" -c "import os,yaml; c=yaml.safe_load(open('$CONFIG_PATH')) or {}; e=(c.get('sac_cfg') or {}).get('experiment') or {}; wk=e.get('wandb_kwargs') or {}; print(wk.get('project') or os.path.basename('$FAMILY'.rstrip('/')))" 2>/dev/null || true)"

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
    "${WANDB_TAG_FLAGS[@]}" \
    "${OVERLAY_FLAGS[@]}" \
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

    # PER-STEP eval: instead of the old aggregate-metrics eval (runner.py --mode eval into a
    # separate "<name>_eval" wandb run), run the from-wandb per-step pipeline (learning/record.py
    # --mode eval). For each just-trained agent run it downloads its ckpt_best.pt + runtime_config,
    # rolls out the deterministic policy while capturing a full per-step/per-env trace, and uploads
    # that trace as eval_<ts>.parquet to the ORIGINAL training run's Files — NO separate eval run is
    # created. Runs are selected by this experiment's wandb group (EXPERIMENT_NAME), narrowed to
    # "<EXPERIMENT_NAME>_agent*". Requires the training runs to be on wandb (ckpt_best.pt is
    # uploaded live during training); if a run never hit a best it is skipped by record.py.
    if [[ "$RUN_EVAL" -eq 1 ]]; then
        echo "[launcher] === EVAL (per-step trace, group=$EXPERIMENT_NAME, project=$WANDB_PROJECT) ==="
        REC_TAG_FLAG=()
        [[ -n "$FIRST_WANDB_TAG" ]] && REC_TAG_FLAG=(--wandb_tag "$FIRST_WANDB_TAG")
        REC_ENTITY_FLAG=()
        [[ -n "$WANDB_ENTITY" ]] && REC_ENTITY_FLAG=(--wandb_entity "$WANDB_ENTITY")
        # NON-FATAL: unlike the old in-process eval, this pipeline needs the just-trained runs to be
        # ONLINE on wandb (it downloads ckpt_best.pt + runtime_config from them and uploads the trace
        # back). A wandb hiccup / offline run must NOT fail an already-successful training — warn and
        # continue (re-run record.py --mode eval to retry). `|| EVAL_RC=$?` neutralizes set -e here.
        EVAL_RC=0
        "$PYTHON" "$PROJECT_ROOT/learning/record.py" \
            --mode eval \
            "${REC_ENTITY_FLAG[@]}" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_group "$EXPERIMENT_NAME" \
            --wandb_run_filter "${EXPERIMENT_NAME}_agent" \
            "${REC_TAG_FLAG[@]}" \
            --headless || EVAL_RC=$?
        if [[ "$EVAL_RC" -ne 0 ]]; then
            echo "[launcher] WARNING: per-step eval failed (exit $EVAL_RC) — training is preserved. "\
"Re-run: learning/record.py --mode eval --wandb_project $WANDB_PROJECT --wandb_group $EXPERIMENT_NAME" >&2
        else
            echo "[launcher] done. train=$EXP_DIR  eval=per-step traces uploaded to the training runs"
        fi
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
