#!/usr/bin/env bash
# launchers/hpc/sweep_launcher.bash — submit one SLURM job per (config × swept value).
#
# A generalization of sbatch_launcher.bash. Instead of one job per config, it takes ONE
# swept parameter and a LIST of values, and submits an independent `sbatch` job for every
# (config, value) pair. Each job trains the config with that parameter pinned to the value,
# via a generated deep-merge overlay handed to runner.py's --overlay.
#
# This lets you keep ONE config per method (e.g. a single VICES.yaml) and sweep the grasp
# pitch across jobs, instead of maintaining VICES_0/15/30/45.yaml by hand.
#
# Usage:
#   sweep_launcher.bash <config_folder> \
#       --sweep_param <dotted.config.path> \
#       --sweep_value <LABEL=YAML_VALUE>  [--sweep_value ...] \
#       [--experiment_tag TAG] [--skip_existing] \
#       [extra args forwarded to sac_block_e2e.sh ...]
#
# Example — sweep cylinder grasp pitch over 0/15/30/45 deg for every config in the folder:
#   sweep_launcher.bash configs/exp_cfgs/simp_low-F_high-P \
#       --sweep_param runner_cfg.rel_grasp_rot_init_deg \
#       --sweep_value "0=[0.0,0.0,0.0]"  --sweep_value "15=[0.0,15.0,0.0]" \
#       --sweep_value "30=[0.0,30.0,0.0]" --sweep_value "45=[0.0,45.0,0.0]" \
#       --wandb_tag pitch_sweep
#
# Sweep flags (launcher-only, NOT forwarded to the worker):
#   --sweep_param PATH   Dotted path into the config YAML to override, e.g.
#                        runner_cfg.rel_grasp_rot_init_deg or
#                        runner_cfg.env_cfg_overrides.task.force_desired_max . REQUIRED.
#   --sweep_value SPEC   One swept value, REPEATABLE (at least one required). SPEC is either
#                        LABEL=YAML_VALUE (explicit short label) or just YAML_VALUE (the value
#                        doubles as the label). YAML_VALUE is parsed as YAML, so scalars
#                        (80, 0.08, true) and lists ([0.0,45.0,0.0]) both work. Write list
#                        values WITHOUT spaces. The LABEL is what gets appended to the group/run.
#
# Naming: for a config whose original wandb group is <G> (its explicit
# sac_cfg/ppo_cfg.experiment.wandb_kwargs.group, or the filename stem if none), each value's
# job trains under experiment_name = group = "<G>_<LABEL>" and per-agent run "<G>_<LABEL>_agentN".
# Both the group and the run name are set explicitly in the generated overlay + via --exp_name.
#
# Launcher-only flags shared with sbatch_launcher.bash:
#   --experiment_tag TAG   Names the jobs and the exp_logs/<TAG>/ subdir. Defaults to the
#                          config folder's basename.
#   --skip_existing        SKIP submitting a (config,value) whose run dir already exists and
#                          is non-empty (re-run a folder to fill only the gaps).
#
# Everything else after the folder is forwarded VERBATIM to sac_block_e2e.sh through the `--`
# boundary (e.g. --no_eval, --wandb_tag X, --experiment_directory FAM), same as sbatch_launcher.
#
# Underscore-prefixed configs (overlays, e.g. _record.yaml) are skipped. Generated sweep
# overlays are written under exp_logs/<TAG>/sweep_overlays/ (regenerated each submit; they
# double as a record of exactly what each job ran). A failing single submission is logged and
# does not abort the batch.
set -uo pipefail

# ===== Load central config (paths, container, SLURM resources) =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc_env.bash
source "$SCRIPT_DIR/hpc_env.bash"

usage() {
    echo "Usage: $0 <config_folder> --sweep_param <dotted.path> --sweep_value <LABEL=YAML> [--sweep_value ...] \\" >&2
    echo "          [--experiment_tag TAG] [--skip_existing] [extra args for sac_block_e2e.sh ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/simp_low-F_high-P \\" >&2
    echo "          --sweep_param runner_cfg.rel_grasp_rot_init_deg \\" >&2
    echo "          --sweep_value '0=[0.0,0.0,0.0]' --sweep_value '45=[0.0,45.0,0.0]'" >&2
}

# ===== Args =====
if [[ $# -lt 1 ]]; then usage; exit 2; fi
CONFIG_FOLDER="$1"
shift

EXPERIMENT_TAG=""
SKIP_EXISTING=0
EXP_DIR_OVERRIDE=""        # mirrors sbatch_launcher: needed to resolve --skip_existing targets
SWEEP_PARAM=""
SWEEP_SPECS=()             # each "LABEL=YAML" or "YAML"
PASSTHROUGH=()             # forwarded to sac_block_e2e.sh after `--`
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep_param)
            [[ $# -ge 2 ]] || { echo "[sweep] --sweep_param requires a value" >&2; exit 2; }
            SWEEP_PARAM="$2"; shift ;;
        --sweep_value)
            [[ $# -ge 2 ]] || { echo "[sweep] --sweep_value requires a value" >&2; exit 2; }
            SWEEP_SPECS+=("$2"); shift ;;
        --experiment_tag)
            [[ $# -ge 2 ]] || { echo "[sweep] --experiment_tag requires a value" >&2; exit 2; }
            EXPERIMENT_TAG="$2"; shift ;;
        --skip_existing) SKIP_EXISTING=1 ;;
        --experiment_directory)
            [[ $# -ge 2 ]] || { echo "[sweep] --experiment_directory requires a value" >&2; exit 2; }
            EXP_DIR_OVERRIDE="$2"
            PASSTHROUGH+=("$1" "$2")              # also forward it to the worker
            shift ;;
        *) PASSTHROUGH+=("$1") ;;
    esac
    shift
done

[[ -n "$SWEEP_PARAM" ]] || { echo "[sweep] --sweep_param is required" >&2; usage; exit 2; }
[[ ${#SWEEP_SPECS[@]} -ge 1 ]] || { echo "[sweep] at least one --sweep_value is required" >&2; usage; exit 2; }

# ===== Derived paths =====
PROJECT_ROOT_LOCAL="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/hpc_batch.bash"
PYTHON="${PYTHON:-python}"     # login-node python (needs pyyaml) for overlay generation + skip resolution
# Default the experiment tag to the config folder's basename.
[[ -n "$EXPERIMENT_TAG" ]] || EXPERIMENT_TAG="$(basename -- "$(cd -- "$CONFIG_FOLDER" 2>/dev/null && pwd || echo "$CONFIG_FOLDER")")"
SLURM_OUT_DIR="$EXP_LOG_DIR/$EXPERIMENT_TAG"
OVERLAY_DIR="$SLURM_OUT_DIR/sweep_overlays"

# ===== Sanity (fail BEFORE queuing anything) =====
[[ -d "$CONFIG_FOLDER" ]] || { echo "[sweep] config folder not found: $CONFIG_FOLDER" >&2; exit 1; }
[[ -f "$JOB_SCRIPT"    ]] || { echo "[sweep] job script not found: $JOB_SCRIPT" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "[sweep] 'sbatch' not found — are you on an HPC login node?" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[sweep] python '$PYTHON' not found (needed to generate overlays) — set PYTHON=..." >&2; exit 1; }
"$PYTHON" -c "import yaml" 2>/dev/null || { echo "[sweep] python '$PYTHON' lacks pyyaml (needed to generate overlays)" >&2; exit 1; }
hpc_require_container || exit 1
mkdir -p "$SLURM_OUT_DIR" "$OVERLAY_DIR"

# ===== Collect configs (non-recursive, skip underscore overlays, sorted) =====
shopt -s nullglob
CONFIGS=("$CONFIG_FOLDER"/*.yaml)
shopt -u nullglob
FILTERED=()
for c in "${CONFIGS[@]}"; do
    b="$(basename -- "$c")"
    if [[ "$b" == _* ]]; then
        echo "[sweep] skipping overlay config (underscore-prefixed): $b"
        continue
    fi
    FILTERED+=("$c")
done
CONFIGS=("${FILTERED[@]}")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[sweep] no trainable *.yaml files found in $CONFIG_FOLDER" >&2
    exit 1
fi
IFS=$'\n' CONFIGS=($(sort <<<"${CONFIGS[*]}")); unset IFS

echo "[sweep] found ${#CONFIGS[@]} config(s) in $CONFIG_FOLDER"
echo "[sweep] sweeping '$SWEEP_PARAM' over ${#SWEEP_SPECS[@]} value(s): ${SWEEP_SPECS[*]}"
echo "[sweep] => ${#CONFIGS[@]}x${#SWEEP_SPECS[@]} = $(( ${#CONFIGS[@]} * ${#SWEEP_SPECS[@]} )) job(s)"
echo "[sweep] experiment tag: $EXPERIMENT_TAG   slurm logs: $SLURM_OUT_DIR   overlays: $OVERLAY_DIR"
echo "[sweep] forwarded to worker: ${PASSTHROUGH[*]+${PASSTHROUGH[*]}}"
echo ""

# ===== Overlay generator (login-node python). Given a config + sweep param + one value, it:
#   * resolves the config's ORIGINAL wandb group (explicit sac/ppo_cfg.experiment.wandb_kwargs.group,
#     else the filename stem),
#   * writes an overlay pinning the swept param AND setting group = "<orig>_<label>",
#   * prints two lines to stdout: "<new_name>" then "<overlay_path>".
# Splits a bare LABEL=YAML on the FIRST '='; a spec with no '=' uses the whole value as label. =====
gen_overlay() {
    local config_path="$1" spec="$2" out_dir="$3"
    "$PYTHON" - "$config_path" "$SWEEP_PARAM" "$spec" "$out_dir" <<'PY'
import os, re, sys, yaml

config_path, sweep_param, spec, out_dir = sys.argv[1:5]

# LABEL=YAML  (split on first '=');  or bare YAML (label == value string).
if "=" in spec:
    label, value_str = spec.split("=", 1)
else:
    label, value_str = spec, spec
label = label.strip()
value = yaml.safe_load(value_str)   # scalars, lists, bools, etc.

cfg = yaml.safe_load(open(config_path)) or {}
agent_type = str((cfg.get("runner_cfg") or {}).get("agent_type", "sac")).lower()
cfg_key = "ppo_cfg" if agent_type == "ppo" else "sac_cfg"

# Original group: explicit wandb_kwargs.group if set, else the filename stem.
exp = ((cfg.get(cfg_key) or {}).get("experiment") or {})
wk = (exp.get("wandb_kwargs") or {})
orig_group = str(wk.get("group") or "").strip()
if not orig_group:
    orig_group = os.path.splitext(os.path.basename(config_path))[0]
new_name = f"{orig_group}_{label}"

# Build the overlay: nest the swept param under its dotted path, and pin the group explicitly
# (so it holds even for configs that set their own group — runner.py only setdefaults it).
overlay = {}
cur = overlay
keys = sweep_param.split(".")
for k in keys[:-1]:
    cur = cur.setdefault(k, {})
cur[keys[-1]] = value
overlay.setdefault(cfg_key, {}).setdefault("experiment", {}).setdefault("wandb_kwargs", {})["group"] = new_name

safe = re.sub(r"[^A-Za-z0-9._-]", "_", new_name)
out_path = os.path.join(out_dir, f"{safe}.yaml")
with open(out_path, "w") as f:
    yaml.safe_dump(overlay, f, default_flow_style=False, sort_keys=False)

print(new_name)
print(out_path)
PY
}

# ===== --skip_existing helper (resolve output dir exactly as runner.py does) =====
# Prints <logdir>/<family>/<exp_name>. Family = --experiment_directory override, else the
# config's sac_cfg.experiment.directory. Pure-bash path assembly (mirrors sbatch_launcher).
resolve_exp_dir() {
    local config_path="$1" exp_name="$2" family
    if [[ -n "$EXP_DIR_OVERRIDE" ]]; then
        family="$EXP_DIR_OVERRIDE"
    else
        family="$("$PYTHON" - "$config_path" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
key = "ppo_cfg" if str(cfg.get("runner_cfg", {}).get("agent_type", "sac")).lower() == "ppo" else "sac_cfg"
exp = (cfg.get(key) or {}).get("experiment") or {}
print(str(exp.get("directory") or "").strip())
PY
)" || { echo "[sweep] WARNING: skip-check could not read experiment.directory from $config_path — will NOT skip $exp_name" >&2; return 1; }
    fi
    family="${family%/}"
    local log_root="${LOGDIR%/}" final_dir
    if [[ -n "$family" && "$(basename -- "$family")" != "$(basename -- "$log_root")" ]]; then
        final_dir="$log_root/$family"
    else
        final_dir="$log_root"
    fi
    [[ "$final_dir" == /* ]] || final_dir="$PROJECT_ROOT_LOCAL/$final_dir"
    printf '%s\n' "$final_dir/$exp_name"
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
    --export="ALL,HPC_LAUNCHER_DIR=$SCRIPT_DIR"
)

# ===== Submit one job per (config, value) =====
SUBMITTED=()
SKIPPED=()
FAILED=()
for config_path in "${CONFIGS[@]}"; do
    for spec in "${SWEEP_SPECS[@]}"; do
        # Generate the overlay + resolve the new group/run name for this pair.
        if ! gen_out="$(gen_overlay "$config_path" "$spec" "$OVERLAY_DIR")"; then
            echo "[sweep] FAILED to generate overlay for $(basename -- "$config_path") value '$spec' — skipping" >&2
            FAILED+=("$(basename -- "${config_path%.yaml}")/$spec")
            continue
        fi
        exp_name="$(printf '%s\n' "$gen_out" | sed -n '1p')"
        overlay_path="$(printf '%s\n' "$gen_out" | sed -n '2p')"
        if [[ -z "$exp_name" || -z "$overlay_path" || ! -f "$overlay_path" ]]; then
            echo "[sweep] FAILED: overlay generation returned no path for $(basename -- "$config_path") value '$spec' — skipping" >&2
            FAILED+=("$(basename -- "${config_path%.yaml}")/$spec")
            continue
        fi

        if [[ "$SKIP_EXISTING" -eq 1 ]]; then
            exp_dir="$(resolve_exp_dir "$config_path" "$exp_name")" || exp_dir=""
            if [[ -n "$exp_dir" && -d "$exp_dir" ]] && compgen -G "$exp_dir/*" >/dev/null 2>&1; then
                echo "[sweep] SKIP (exists): $exp_name -> $exp_dir"
                SKIPPED+=("$exp_name")
                continue
            else
                echo "[sweep] skip-check: $exp_name -> ${exp_dir:-<unresolved>} $([[ -d "$exp_dir" ]] && echo '(dir empty)' || echo '(not present)') — submitting"
            fi
        fi

        job_name="${EXPERIMENT_TAG}_${exp_name}"
        out_pat="$SLURM_OUT_DIR/${exp_name}_%j.out"
        err_pat="$SLURM_OUT_DIR/${exp_name}_%j.err"

        echo "[sweep] submitting: $job_name  (config=$(basename -- "$config_path"), $SWEEP_PARAM=$spec)"
        if sbatch "${SBATCH_RES[@]}" \
                -J "$job_name" \
                -o "$out_pat" \
                -e "$err_pat" \
                "$JOB_SCRIPT" \
                --config "$config_path" --exp_name "$exp_name" \
                -- --overlay "$overlay_path" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
            SUBMITTED+=("$job_name")
        else
            echo "[sweep] FAILED to submit: $job_name — continuing" >&2
            FAILED+=("$job_name")
        fi
    done
done

# ===== Summary =====
echo ""
echo "[sweep] ===================================================================="
echo "[sweep] DONE. ${#SUBMITTED[@]} submitted, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
for n in ${SUBMITTED[@]+"${SUBMITTED[@]}"}; do echo "[sweep]   SUBMITTED: $n"; done
for n in ${SKIPPED[@]+"${SKIPPED[@]}"};   do echo "[sweep]   SKIP:      $n"; done
for n in ${FAILED[@]+"${FAILED[@]}"};     do echo "[sweep]   FAIL:      $n"; done
echo "[sweep] ===================================================================="

[[ ${#FAILED[@]} -eq 0 ]]
