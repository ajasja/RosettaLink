#!/bin/bash
#SBATCH --job-name=rosettalink_pipeline
#SBATCH --partition=gpu-a40,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm-%A_%a_%x.out
#SBATCH --error=logs/slurm-%A_%a_%x.err

# Abort on any error instead of silently continuing in the wrong environment
# (this is exactly what let a failed `conda activate` below go unnoticed).
set -euo pipefail

# Anchor all relative paths (examples/..., logs/...) to the directory `sbatch`
# was invoked from. NOT `dirname "${BASH_SOURCE[0]}"`: sbatch copies this
# script into its own spool location on the compute node and runs that copy,
# so BASH_SOURCE points there instead - a directory the job's user can't
# write to, which is what caused "mkdir: cannot create directory 'logs':
# Permission denied". $SLURM_SUBMIT_DIR is the one sbatch itself guarantees.
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# `sbatch` runs this in a non-interactive shell that hasn't sourced conda's
# shell hook, so a bare `conda activate` fails with
# "CondaError: Run 'conda init' before 'conda activate'" - and, without
# `set -e` above, that failure was silent: the job went on to run whatever
# `python` was first on PATH (miniforge3's base env) instead of `prosculpt`.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rosettalink

echo "python: $(which python)"
echo "rosettalink: $(python -c 'import rosettalink, os; print(os.path.dirname(rosettalink.__file__))')"

python examples/demo_full_pipeline.py
