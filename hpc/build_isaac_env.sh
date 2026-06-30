#!/usr/bin/env bash
# =============================================================================
# build_isaac_env.sh — build an Apptainer/Singularity image reproducing the
# local `general` env (newest Isaac Lab / Isaac Sim 5.1) for this repo on HPC.
#
# WHY a container: the newest Isaac Lab does not install cleanly as a plain
# conda env on the cluster (the approach used for older Isaac Lab). This image
# bakes the whole stack so `apptainer run` just works on a GPU node.
#
# Reproduces the `general` env faithfully:
#   * Ubuntu 22.04 + CUDA 12.8, Python 3.11
#   * torch 2.7.0+cu128            (PyTorch cu128 index)
#   * Isaac Sim 5.1.0 wheels       (public pypi.nvidia.com — no NGC login)
#   * Isaac Lab EDITABLE from upstream source @ pinned commit (newest)
#   * everything else from requirements.lock.txt (audited against the live env)
#   * this repo cloned to /opt/ghvic
#
# RUN ON AN HPC COMPUTE NODE (login nodes OOM mksquashfs). Using the real
# Continuous_Force_RL partitions/account:
#   srun -A virl-grp -p dgxh,dgx2,tiamat,gpu,eecs2 --time=2:00:00 \
#        --mem=32G --cpus-per-task=4 --pty bash
#   bash hpc/build_isaac_env.sh
# =============================================================================
set -euo pipefail

# ============================ CONFIG (all baked here) ========================
SHARE="${SHARE:-$HOME/hpc-share}"                       # big/Lustre dir for the .sif + caches
IMG="${IMG:-$SHARE/isaac/ghvic.sif}"                    # output image path
BASE_IMAGE="${BASE_IMAGE:-docker://nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
PYVER="${PYVER:-3.11}"

# --- code sources ---
REPO_URL="${REPO_URL:-https://github.com/RoboNuke/generalized_hybrid_vic_action_space.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
ISAACLAB_URL="${ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-e06a0674a9}"        # = local /home/hunter/IsaacLab HEAD (newest)

# --- Slurm defaults baked into the emitted train.slurm (from Continuous_Force_RL) ---
SLURM_ACCOUNT="${SLURM_ACCOUNT:-virl-grp}"
SLURM_PARTITIONS="${SLURM_PARTITIONS:-dgxh,dgx2,tiamat,gpu,eecs2}"
SLURM_TIME="${SLURM_TIME:-9:00:00}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_CPUS="${SLURM_CPUS:-12}"
# =============================================================================

BUILD="${BUILD:-/tmp/ghvic_build_$USER}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_$USER}"   # MUST be local disk, not Lustre
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$APPTAINER_TMPDIR/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$BUILD" "$(dirname "$IMG")"

echo "[build] image    : $IMG"
echo "[build] base     : $BASE_IMAGE"
echo "[build] repo     : $REPO_URL ($REPO_BRANCH)"
echo "[build] isaaclab : $ISAACLAB_URL @ $ISAACLAB_COMMIT"
echo "[build] tmpdir   : $APPTAINER_TMPDIR"
echo

# ----- Apptainer definition --------------------------------------------------
DEF="$BUILD/ghvic.def"
cat > "$DEF" <<DEF
Bootstrap: docker
From: ${BASE_IMAGE#docker://}

%environment
    export OMNI_KIT_ALLOW_ROOT=1
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y
    export PYTHONNOUSERSITE=1

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # system deps: graphics + build libs Isaac Sim / kit need at runtime
    apt-get update
    apt-get install -y --no-install-recommends \\
        software-properties-common git git-lfs curl wget ca-certificates build-essential cmake \\
        libgl1 libglu1-mesa libegl1 libgomp1 libglib2.0-0 \\
        libxrandr2 libxinerama1 libxcursor1 libxi6 libxext6 libxrender1 libsm6
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y --no-install-recommends \\
        python${PYVER} python${PYVER}-dev python${PYVER}-venv python${PYVER}-distutils
    rm -rf /var/lib/apt/lists/*

    # pip on python ${PYVER}
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYVER}
    ln -sf /usr/bin/python${PYVER} /usr/local/bin/python
    ln -sf /usr/bin/python${PYVER} /usr/local/bin/python3
    python -m pip install --upgrade pip wheel "setuptools<80" packaging
    # Legacy sdists (e.g. flatdict, a transitive Isaac Lab dep) do \`import pkg_resources\`
    # in setup.py; setuptools 80+ drops pkg_resources from the PEP-517 build-isolation
    # overlay, breaking the wheel build. Constrain build envs to setuptools<80 too.
    printf 'setuptools<80\n' > /opt/pip-constraints.txt
    export PIP_CONSTRAINT=/opt/pip-constraints.txt

    # this repo (carries requirements.lock.txt + the split helper)
    git clone --branch ${REPO_BRANCH} ${REPO_URL} /opt/ghvic
    python /opt/ghvic/hpc/split_requirements.py /opt/ghvic/requirements.lock.txt /opt/reqs

    # 1) torch (cu128)
    python -m pip install -r /opt/reqs/torch.txt --index-url https://download.pytorch.org/whl/cu128

    # 2) Isaac Sim 5.1 wheels (public NVIDIA index)
    python -m pip install -r /opt/reqs/isaacsim.txt --extra-index-url https://pypi.nvidia.com

    # 3) Isaac Lab: newest upstream @ pinned commit, EDITABLE (mirrors local layout)
    git clone ${ISAACLAB_URL} /opt/IsaacLab
    git -C /opt/IsaacLab checkout ${ISAACLAB_COMMIT}
    for ext in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
        python -m pip install -e /opt/IsaacLab/source/\$ext
    done

    # 4) everything else (re-pinned last so versions match \`general\`)
    python -m pip install -r /opt/reqs/rest.txt

    # NOTE: verification is intentionally NOT done here. A failed check in %post would
    # abort the whole build (no .sif written) and force a full rebuild. pip install
    # success above IS the build criterion. Verify the image separately on a GPU node:
    #   ~/hpc-share/isaac/run.sh python hpc/verify_env.py
    chmod -R a+rX /opt/ghvic /opt/IsaacLab

%runscript
    cd /opt/ghvic
    exec "\$@"

%labels
    org.repo generalized_hybrid_vic_action_space
    org.isaaclab_commit ${ISAACLAB_COMMIT}
DEF
echo "[build] wrote $DEF"

# ----- build -----------------------------------------------------------------
apptainer build --fakeroot \
    --mksquashfs-args "-mem 8G -processors 4" \
    "$IMG" "$DEF" 2>&1 | tee "$BUILD/build.log"

# ----- runtime wrapper (writable HOME for Isaac caches; host repo for live edits) ----
RUN="$(dirname "$IMG")/run.sh"
cat > "$RUN" <<RUNEOF
#!/usr/bin/env bash
# run.sh — execute a command inside the image with a writable HOME (Isaac Sim
# caches) and your host working copy bind-mounted over the baked-in clone so
# code edits + run outputs persist on the host.
#   ./run.sh python learning/runner.py --config configs/exp_cfgs/eef_glued_peg/1_fixed.yaml --headless ...
#   ./run.sh bash launchers/exp_file_launcher.bash configs/exp_cfgs/eef_glued_peg
set -euo pipefail
SHARE="\${SHARE:-$SHARE}"
IMG="\${IMG:-$IMG}"
HOST_REPO="\${HOST_REPO:-\$SHARE/generalized_hybrid_vic_action_space}"   # your editable checkout
CACHE="\${CACHE:-\$SHARE/isaac_cache_home}"                             # writable HOME for OV/kit caches
mkdir -p "\$CACHE"
BINDS=(--bind "\$SHARE:/work" --bind /tmp:/tmp)
[ -d "\$HOST_REPO" ] && BINDS+=(--bind "\$HOST_REPO:/opt/ghvic")
exec apptainer run --nv --containall --home "\$CACHE":/root "\${BINDS[@]}" "\$IMG" "\$@"
RUNEOF
chmod +x "$RUN"

# ----- Slurm template (real Continuous_Force_RL directives; conda swapped for the container) ----
SLURM="$(dirname "$IMG")/train.slurm"
cat > "$SLURM" <<SLURMEOF
#!/bin/bash
#SBATCH --job-name=ghvic
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITIONS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --signal=SIGTERM@300
#SBATCH --output=logs/%j.out
set -euo pipefail
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
"\$DIR/run.sh" python learning/runner.py \\
    --config configs/exp_cfgs/eef_glued_peg/1_fixed.yaml \\
    --headless --experiment_name 1_fixed --experiment_directory eef_PiH_test
SLURMEOF

echo
echo "[build] DONE"
echo "  image : $IMG"
echo "  run   : $RUN"
echo "  slurm : $SLURM   (account=$SLURM_ACCOUNT partitions=$SLURM_PARTITIONS)"
echo "  log   : $BUILD/build.log"
echo
echo "Verify the image on a GPU node (separate from the build — re-runnable):"
echo "  srun -A $SLURM_ACCOUNT -p $SLURM_PARTITIONS --gres=gpu:1 --time=0:20:00 --pty bash"
echo "  $RUN python hpc/verify_env.py"
