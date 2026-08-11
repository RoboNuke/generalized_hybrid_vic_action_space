#!/usr/bin/env bash
# launchers/eval_sweep.bash — sweep ONE parameter over a value list at EVAL time,
# driven by a YAML SWEEP CONFIG.
#
# The eval analogue of hpc/sweep_launcher.bash (which sweeps at TRAIN time on SLURM). This
# one is LOCAL: it drives learning/record.py --mode eval, which downloads each wandb run's
# ckpt_best.pt + runtime_config.yaml, re-runs eval, and uploads a per-step trace to the
# ORIGINAL run's Files. For each swept value it generates a one-off deep-merge overlay and
# hands it to record.py --overlay, so ONLY the swept parameter (plus the fixed `constants`)
# changes; everything else stays exactly as the run trained. Each value's trace uploads as:
#
#     eval_<label>_<value>_<ts>.parquet
#
# so the swept value is visible in the filename.
#
# WHAT COMES FROM WHERE
#   * The SWEEP itself (label, parameter, values) and any fixed CONSTANT overrides come from a
#     YAML --sweep_config (below).
#   * wandb SELECTION + eval knobs stay on the CLI (--wandb_tag/--wandb_group/--num_envs/...).
#
# Usage:
#   eval_sweep.bash --sweep_config <sweep.yaml> \
#       --wandb_tag <TAG> [--wandb_group G ...] [--wandb_run_filter SUBSTR] \
#       [--wandb_entity hur] [--wandb_project pitch_sweep] \
#       [--num_envs N] [--eval_timesteps K] \
#       [-- extra args forwarded verbatim to record.py (e.g. --no_upload --keep_local) ...]
#
# SWEEP CONFIG (YAML). Required keys: label, sweep_param, sweep_values. Optional: constants.
#
#   label:        bumpy_speed                       # folded into the trace filename
#   sweep_param:  task.desired_speed_cm_s           # the ONE dotted path that varies
#   sweep_values: [3, 5, 8]                         # scalars or lists ([0.0,45.0,0.0]); or, for
#                                                   # explicit per-value tags:
#                                                   #   - {label: slow, value: 3}
#                                                   #   - {label: fast, value: 8}
#   constants:                                      # FIXED overrides applied to EVERY eval
#     runner_cfg.task: Isaac-FlatSurfaceFollow-Bumpy-Direct-v0   # e.g. switch flat -> bumpy env
#     task.bump_max_height: 0.006                   # e.g. a difficulty knob
#
# DOTTED PATH ROUTING (auto, for BOTH sweep_param and every constant):
#   * runner_cfg.env_cfg_overrides.<k>  -> flat env override key <k>
#   * task.* (or any non-config-header first segment) -> runner_cfg.env_cfg_overrides (env param)
#   * a config header (runner_cfg/sac_cfg/ppo_cfg/model_cfg/controller_cfg/noise_cfg/sensor_cfg/
#     loss_cfg/reset_curriculum_cfg/keypoint_servo_cfg) -> nested config field
#   The sweep_param wins over a constant that targets the same path.
#
# Env overrides: SWEEP_PY (python w/ pyyaml for overlay gen, default python3),
#                EVAL_PY (how to run record.py, default "conda run -n general python").
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SWEEP_PY="${SWEEP_PY:-python3}"
EVAL_PY="${EVAL_PY:-conda run -n general python}"

usage() { sed -n '2,60p' "${BASH_SOURCE[0]}" >&2; }

SWEEP_CONFIG=""
WANDB_TAG=""
WANDB_ENTITY="hur"
WANDB_PROJECT="pitch_sweep"
WANDB_GROUPS=()
WANDB_RUN_FILTER=""
NUM_ENVS=""
EVAL_TIMESTEPS=""
EXTRA=()          # forwarded verbatim to record.py (e.g. --no_upload --keep_local --device cuda:0)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep_config)     SWEEP_CONFIG="$2"; shift 2 ;;
        --wandb_tag)        WANDB_TAG="$2"; shift 2 ;;
        --wandb_entity)     WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_project)    WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_group)      WANDB_GROUPS+=("$2"); shift 2 ;;
        --wandb_run_filter) WANDB_RUN_FILTER="$2"; shift 2 ;;
        --num_envs)         NUM_ENVS="$2"; shift 2 ;;
        --eval_timesteps)   EVAL_TIMESTEPS="$2"; shift 2 ;;
        --)                 shift; EXTRA+=("$@"); break ;;
        -h|--help)          usage; exit 0 ;;
        *)                  EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$SWEEP_CONFIG" ]] || { echo "[eval-sweep] --sweep_config is required" >&2; usage; exit 2; }
[[ -f "$SWEEP_CONFIG" ]] || { echo "[eval-sweep] sweep config not found: $SWEEP_CONFIG" >&2; exit 1; }
[[ -n "$WANDB_TAG"    ]] || { echo "[eval-sweep] --wandb_tag is required" >&2; usage; exit 2; }
$SWEEP_PY -c "import yaml" 2>/dev/null || {
    echo "[eval-sweep] SWEEP_PY ($SWEEP_PY) lacks pyyaml (needed to read the sweep config). "\
"Set SWEEP_PY=..., e.g. SWEEP_PY='conda run -n general python'." >&2; exit 1; }

OVERLAY_DIR="$PROJECT_ROOT/runs/_eval_sweep_overlays/${WANDB_PROJECT}_${WANDB_TAG}_$(basename -- "${SWEEP_CONFIG%.*}")"
mkdir -p "$OVERLAY_DIR"

# ---- Read the sweep config, generate one overlay per value (sweep_param=value + constants,
#      auto-routed), and print one "<overlay_path>\t<trace_label>" line per value. ----
GEN_TSV="$OVERLAY_DIR/_pairs.tsv"
if ! "$SWEEP_PY" - "$SWEEP_CONFIG" "$OVERLAY_DIR" > "$GEN_TSV" <<'PY'
import os, re, sys, yaml
cfg_path, out_dir = sys.argv[1:3]
cfg = yaml.safe_load(open(cfg_path)) or {}

label       = str(cfg.get("label", "")).strip()
sweep_param = cfg.get("sweep_param")
values      = cfg.get("sweep_values")
constants   = cfg.get("constants") or {}
if not label:               sys.exit("[eval-sweep] sweep config missing 'label'")
if not sweep_param:         sys.exit("[eval-sweep] sweep config missing 'sweep_param'")
if not isinstance(values, list) or not values:
    sys.exit("[eval-sweep] sweep config 'sweep_values' must be a non-empty list")
if not isinstance(constants, dict):
    sys.exit("[eval-sweep] sweep config 'constants' must be a mapping")

HEADERS = {"runner_cfg","sac_cfg","ppo_cfg","model_cfg","controller_cfg","noise_cfg",
           "sensor_cfg","loss_cfg","reset_curriculum_cfg","keypoint_servo_cfg"}

def set_dotted(overlay, path, value):
    """Route a dotted path to the right place in the overlay (env override vs config field)."""
    path = str(path)
    if path.startswith("runner_cfg.env_cfg_overrides."):
        key = path[len("runner_cfg.env_cfg_overrides."):]
        overlay.setdefault("runner_cfg", {}).setdefault("env_cfg_overrides", {})[key] = value
    elif path.split(".")[0] in HEADERS:
        cur = overlay
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    else:                                   # env param, e.g. task.*
        overlay.setdefault("runner_cfg", {}).setdefault("env_cfg_overrides", {})[path] = value

def san(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

print(f"[eval-sweep] sweep '{label}': {sweep_param} over {len(values)} value(s)", file=sys.stderr)
if constants:
    print(f"[eval-sweep] constants: {', '.join(f'{k}={v}' for k, v in constants.items())}", file=sys.stderr)

for entry in values:
    if isinstance(entry, dict):                       # {label: <tag>, value: <v>}
        v = entry.get("value")
        vlabel = entry.get("label")
        token = f"{san(vlabel)}_{san(v)}" if vlabel is not None else san(v)
    else:
        v = entry
        token = san(v)
    overlay = {}
    for cpath, cval in constants.items():             # fixed constants first ...
        set_dotted(overlay, cpath, cval)
    set_dotted(overlay, sweep_param, v)               # ... swept param wins on conflict
    trace_label = f"{san(label)}_{token}"
    out_path = os.path.join(out_dir, f"{trace_label}.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(overlay, f, default_flow_style=False, sort_keys=False)
    print(f"{out_path}\t{trace_label}")
PY
then
    echo "[eval-sweep] FAILED to read sweep config / generate overlays (see error above)" >&2
    exit 1
fi

echo "[eval-sweep] config: $SWEEP_CONFIG"
echo "[eval-sweep] target: ${WANDB_ENTITY}/${WANDB_PROJECT} tag=$WANDB_TAG"\
"${WANDB_GROUPS[*]+ groups=${WANDB_GROUPS[*]}}${WANDB_RUN_FILTER:+ filter=$WANDB_RUN_FILTER}"
echo "[eval-sweep] overlays: $OVERLAY_DIR"
echo ""

OK=(); FAIL=()
while IFS=$'\t' read -r overlay_path trace_label; do
    [[ -n "$overlay_path" && -f "$overlay_path" ]] || { echo "[eval-sweep] bad overlay line: '$overlay_path'" >&2; FAIL+=("$trace_label"); continue; }

    cmd=($EVAL_PY "$PROJECT_ROOT/learning/record.py" --mode eval
         --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tag "$WANDB_TAG"
         --overlay "$overlay_path" --trace_label "$trace_label" --headless)
    for g in ${WANDB_GROUPS[@]+"${WANDB_GROUPS[@]}"}; do cmd+=(--wandb_group "$g"); done
    [[ -n "$WANDB_RUN_FILTER" ]] && cmd+=(--wandb_run_filter "$WANDB_RUN_FILTER")
    [[ -n "$NUM_ENVS" ]]        && cmd+=(--num_envs "$NUM_ENVS")
    [[ -n "$EVAL_TIMESTEPS" ]]  && cmd+=(--eval_timesteps "$EVAL_TIMESTEPS")
    cmd+=(${EXTRA[@]+"${EXTRA[@]}"})

    echo "[eval-sweep] === $trace_label (overlay: $(basename -- "$overlay_path")) ==="
    echo "[eval-sweep] run: ${cmd[*]}"
    if "${cmd[@]}"; then OK+=("$trace_label"); else echo "[eval-sweep] $trace_label FAILED" >&2; FAIL+=("$trace_label"); fi
done < "$GEN_TSV"

echo ""
echo "[eval-sweep] DONE. ${#OK[@]} value(s) ok, ${#FAIL[@]} failed."
[[ ${#FAIL[@]} -eq 0 ]]
