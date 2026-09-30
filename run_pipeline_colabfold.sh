#!/bin/bash
#SBATCH --job-name=rosettalink_colabfold
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

# ColabFold runs from a singularity image, so this env needs only PyRosetta.
CONDA_ENV=${CONDA_ENV:-/home/folivieri/miniforge3/envs/rosettalink}
conda activate "$CONDA_ENV"

echo "python: $(which python)"
echo "env:    $CONDA_PREFIX"

DEMO=/home/folivieri/RosettaLink/examples/demo_full_pipeline_colabfold.py

export PIPELINE_OUTPUT_DIR=output/${SLURM_JOB_ID:-manual}
python "$DEMO"
