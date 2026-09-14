#!/bin/bash
#SBATCH --job-name=hbdesigner
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%A_%a_%x.out
#SBATCH --error=logs/slurm-%A_%a_%x.err

# gpu-a40 only: the torch_scatter and torch_cluster extensions in
# hbdesigner_gpu were compiled for sm_86 without PTX, so they will not load on
# the l40s (sm_89) or gtx1000 nodes. Rebuild them with a wider
# TORCH_CUDA_ARCH_LIST to use other partitions.
#
# cpus-per-task matches N_WORKERS in the demo, which HBDesigner uses for
# parallel packing and scoring.

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
# `python` was first on PATH (miniforge3's base env) instead of `rosettalink`.
source "$(conda info --base)/etc/profile.d/conda.sh"

# The demo needs PyRosetta; HBDesigner itself runs in its own environment,
# reached by the mover through cmd_header.
conda activate rosettalink

echo "python:      $(which python)"
echo "hbdesigner:  $(conda run -n hbdesigner_gpu which run_hbdesigner)"

python examples/demo_hbdesigner_extended.py
