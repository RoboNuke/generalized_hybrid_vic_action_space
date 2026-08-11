#!/usr/bin/env bash
# launchers/eval_sweep.bash — sweep ONE parameter over a value list at EVAL time.
#
# The eval analogue of hpc/sweep_launcher.bash (which sweeps at TRAIN time on SLURM). This
# one is LOCAL: it drives learning/record.py --mode eval, which downloads each wandb run's
# ckpt_best.pt + runtime_config.yaml, re-runs eval, and uploads a per-step trace to the
# ORIGINAL run's Files. For each swept value it generates a one-field deep-merge overlay and
# hands it to record.py --overlay, so ONLY that parameter changes; every other setting stays
# exactly as the run trained. Each value's trace is uploaded as:
#
#     eval_<label>_<value>_<ts>.parquet         (rec_… in record mode is unaffected; this is eval)
#
# so the swept value is visible in the filename.
#
# Usage:
#   eval_sweep.bash \
#       --sweep_param <dotted.path> \
#       --sweep_value <LABEL=YAML_VALUE> [--sweep_value ...] \
#       --wandb_tag <TAG> [--wandb_group G ...] [--wandb_run_filter SUBSTR] \
#       [--wandb_entity hur] [--wandb_project pitch_sweep] \
#       [--num_envs N] [--eval_timesteps K] [--no_upload] [--keep_local] \
#       [-- extra args forwarded verbatim to record.py ...]
#
# Example — sweep the desired drag speed over 3/5/8 cm/s for one group:
#   eval_sweep.bash \
#       --sweep_param task.desired_speed_cm_s \
#       --sweep_value 3 --sweep_value 5 --sweep_value 8 \
#       --wandb_tag pitch_sweep --wandb_group VICES_0 --num_envs 512
#
#   -> for each VICES_0 seed, three evals uploading eval_3_<ts>/eval_5_<ts>/eval_8_<ts>.parquet.
#
# --sweep_param PATH   Dotted path to override. AUTO-DETECTED:
#                        * starts with a config header (runner_cfg/sac_cfg/ppo_cfg/model_cfg/
#                          controller_cfg/noise_cfg/sensor_cfg/loss_cfg/reset_curriculum_cfg/
#                          keypoint_servo_cfg) -> nested directly in the overlay;
#                        * anything else (e.g. task.*) -> an ENV override, placed under
#                          runner_cfg.env_cfg_overrides as a FLAT dotted key (the format that dict
#                          uses). runner_cfg.env_cfg_overrides.<k> is also accepted and normalized.
# --sweep_value SPEC   LABEL=YAML_VALUE (explicit label) or bare YAML_VALUE (value doubles as label).
#                        YAML so scalars (5, 0.08, true) and lists ([0.0,45.0,0.0], no spaces) work.
#                        REPEATABLE; at least one required.
#
# Env overrides: SWEEP_PY (python w/ pyyaml for overlay gen, default python3),
#                EVAL_PY (how to run record.py, default "conda run -n general python").
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SWEEP_PY="${SWEEP_PY:-python3}"
EVAL_PY="${EVAL_PY:-conda run -n general python}"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" >&2; }

SWEEP_PARAM=""
SWEEP_SPECS=()
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
        --sweep_param)     SWEEP_PARAM="$2"; shift 2 ;;
        --sweep_value)     SWEEP_SPECS+=("$2"); shift 2 ;;
        --wandb_tag)       WANDB_TAG="$2"; shift 2 ;;
        --wandb_entity)    WANDB_ENTITY="$2"; shift 2 ;;
        --wandb_project)   WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_group)     WANDB_GROUPS+=("$2"); shift 2 ;;
        --wandb_run_filter) WANDB_RUN_FILTER="$2"; shift 2 ;;
        --num_envs)        NUM_ENVS="$2"; shift 2 ;;
        --eval_timesteps)  EVAL_TIMESTEPS="$2"; shift 2 ;;
        --)                shift; EXTRA+=("$@"); break ;;
        -h|--help)         usage; exit 0 ;;
        *)                 EXTRA+=("$1"); shift ;;
    esac
done

[[ -n "$SWEEP_PARAM" ]] || { echo "[eval-sweep] --sweep_param is required" >&2; usage; exit 2; }
[[ ${#SWEEP_SPECS[@]} -ge 1 ]] || { echo "[eval-sweep] at least one --sweep_value is required" >&2; usage; exit 2; }
[[ -n "$WANDB_TAG" ]] || { echo "[eval-sweep] --wandb_tag is required" >&2; usage; exit 2; }
$SWEEP_PY -c "import yaml" 2>/dev/null || {
    echo "[eval-sweep] SWEEP_PY ($SWEEP_PY) lacks pyyaml (needed to generate overlays). "\
"Set SWEEP_PY=..., e.g. SWEEP_PY='conda run -n general python'." >&2; exit 1; }

OVERLAY_DIR="$PROJECT_ROOT/runs/_eval_sweep_overlays/${WANDB_PROJECT}_${WANDB_TAG}"
mkdir -p "$OVERLAY_DIR"

# ---- overlay generator: pins ONE dotted path to a value, auto-routing env params to
#      runner_cfg.env_cfg_overrides (flat dotted key) vs. config-header paths (nested).
#      Prints two lines: <overlay_path> then <trace_label>. ----
gen_overlay() {
    local spec="$1"
    "$SWEEP_PY" - "$SWEEP_PARAM" "$spec" "$OVERLAY_DIR" <<'PY'
import os, re, sys, yaml
param, spec, out_dir = sys.argv[1:4]
if "=" in spec:
    label, value_str = spec.split("=", 1)
else:
    label, value_str = spec, spec
label = label.strip()
value = yaml.safe_load(value_str)

HEADERS = {"runner_cfg","sac_cfg","ppo_cfg","model_cfg","controller_cfg","noise_cfg",
           "sensor_cfg","loss_cfg","reset_curriculum_cfg","keypoint_servo_cfg"}
overlay = {}
if param.startswith("runner_cfg.env_cfg_overrides."):
    key = param[len("runner_cfg.env_cfg_overrides."):]     # flat dotted env key
    overlay = {"runner_cfg": {"env_cfg_overrides": {key: value}}}
elif param.split(".")[0] in HEADERS:
    cur = overlay                                           # nest the config-header path directly
    keys = param.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
else:
    overlay = {"runner_cfg": {"env_cfg_overrides": {param: value}}}  # env param, e.g. task.*

def _san(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
val_tok = _san(value_str)
trace_label = val_tok if _san(label) == val_tok else f"{_san(label)}_{val_tok}"

out_path = os.path.join(out_dir, f"{re.sub(r'[^A-Za-z0-9._-]+','_',param)}__{trace_label}.yaml")
with open(out_path, "w") as f:
    yaml.safe_dump(overlay, f, default_flow_style=False, sort_keys=False)
print(out_path)
print(trace_label)
PY
}

echo "[eval-sweep] param '$SWEEP_PARAM' over ${#SWEEP_SPECS[@]} value(s): ${SWEEP_SPECS[*]}"
echo "[eval-sweep] target: ${WANDB_ENTITY}/${WANDB_PROJECT} tag=$WANDB_TAG"\
"${WANDB_GROUPS[*]+ groups=${WANDB_GROUPS[*]}}${WANDB_RUN_FILTER:+ filter=$WANDB_RUN_FILTER}"
echo "[eval-sweep] overlays: $OVERLAY_DIR"
echo ""

OK=(); FAIL=()
for spec in "${SWEEP_SPECS[@]}"; do
    gen="$(gen_overlay "$spec")" || { echo "[eval-sweep] FAILED overlay gen for '$spec'" >&2; FAIL+=("$spec"); continue; }
    overlay_path="$(sed -n '1p' <<<"$gen")"
    trace_label="$(sed -n '2p' <<<"$gen")"
    [[ -f "$overlay_path" ]] || { echo "[eval-sweep] FAILED: no overlay for '$spec'" >&2; FAIL+=("$spec"); continue; }

    cmd=($EVAL_PY "$PROJECT_ROOT/learning/record.py" --mode eval
         --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --wandb_tag "$WANDB_TAG"
         --overlay "$overlay_path" --trace_label "$trace_label" --headless)
    for g in ${WANDB_GROUPS[@]+"${WANDB_GROUPS[@]}"}; do cmd+=(--wandb_group "$g"); done
    [[ -n "$WANDB_RUN_FILTER" ]] && cmd+=(--wandb_run_filter "$WANDB_RUN_FILTER")
    [[ -n "$NUM_ENVS" ]]        && cmd+=(--num_envs "$NUM_ENVS")
    [[ -n "$EVAL_TIMESTEPS" ]]  && cmd+=(--eval_timesteps "$EVAL_TIMESTEPS")
    cmd+=(${EXTRA[@]+"${EXTRA[@]}"})

    echo "[eval-sweep] === value '$spec' -> ${SWEEP_PARAM} (label=$trace_label) ==="
    echo "[eval-sweep] overlay: $overlay_path"
    echo "[eval-sweep] run: ${cmd[*]}"
    if "${cmd[@]}"; then OK+=("$spec"); else echo "[eval-sweep] value '$spec' FAILED" >&2; FAIL+=("$spec"); fi
done

echo ""
echo "[eval-sweep] DONE. ${#OK[@]} value(s) ok, ${#FAIL[@]} failed."
[[ ${#FAIL[@]} -eq 0 ]]
