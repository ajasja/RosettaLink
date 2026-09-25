#!/bin/bash
#SBATCH --job-name=rosettalink_boltz
#SBATCH --partition=gpu-a40,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm-%A_%a_%x.out
#SBATCH --error=logs/slurm-%A_%a_%x.err

set -euo pipefail

# Relative paths resolve against the directory sbatch was invoked from.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# sbatch runs a non-interactive shell, which has not sourced conda's hook.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Boltz2 runs as a CLI subprocess, so it only needs to be on PATH in this env.
CONDA_ENV=${CONDA_ENV:-/home/folivieri/miniforge3/envs/rosettalink}
conda activate "$CONDA_ENV"

echo "python: $(which python)"
echo "env:    $CONDA_PREFIX"

command -v boltz >/dev/null \
    || { echo "ERROR: boltz not on PATH in $CONDA_PREFIX"; exit 1; }
# Triton compiles boltz's CUDA kernels at runtime and uses CC when set.
[ -n "${CC:-}" ] || echo "WARNING: CC unset; boltz kernels will build with the system gcc"

export PIPELINE_OUTPUT_DIR=output/${SLURM_JOB_ID:-manual}
python /home/folivieri/RosettaLink/examples/demo_full_pipeline_boltz_binderDesign.py
