#!/bin/bash
#SBATCH --job-name=rosettalink_esmfold
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

# ESMFold2 is imported in process, so this env holds esm alongside PyRosetta.
CONDA_ENV=${CONDA_ENV:-/home/folivieri/miniforge3/envs/esmfold2}
conda activate "$CONDA_ENV"

echo "python: $(which python)"
echo "env:    $CONDA_PREFIX"

# Checked here so a wrong env fails in seconds rather than after RFDiffusion
# and LigandMPNN have run.
python -c "import esm.models.esmfold2" 2>/dev/null \
    || { echo "ERROR: $CONDA_PREFIX has no esm.models.esmfold2 (needs esm 3.4.0+)"; exit 1; }

export PIPELINE_OUTPUT_DIR=output/${SLURM_JOB_ID:-manual}
python /home/folivieri/RosettaLink/examples/demo_full_pipeline_ESM_binderDesign.py
