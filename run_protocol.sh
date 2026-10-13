#!/bin/bash
#SBATCH --job-name=rosettalink
#SBATCH --partition=gpu-a40,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%A_%x.out
#SBATCH --error=logs/slurm-%A_%x.err

# Runs any RosettaScripts XML through rosetta_link_scripts.
#
#   sbatch run_protocol.sh examples/redesign_colabfold.xml \
#       -s examples/input_data/insulin_target.pdb \
#       -out:path output/redesign
#
# Every argument after the protocol is passed through to the runner. The
# external programs are reached through rosettalink.config.yaml, so this
# environment needs only PyRosetta.
#
# CONDA_ENV picks the environment, PROXY_SETUP a script to source before the
# run. A compute node has no route to the internet until something exports
# proxy settings, which a protocol needs whenever it asks ColabFold for an
# msa_mode other than single_sequence:
#
#   PROXY_SETUP=/home/aljubetic/bin/setup_proxy_settings.sh \
#       sbatch run_protocol.sh examples/binder_design.xml -s target.pdb

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-/home/folivieri/miniforge3/envs/rosettalink}"

if [ -n "${PROXY_SETUP:-}" ]; then
    source "$PROXY_SETUP"
    echo "proxy:    $PROXY_SETUP (https_proxy=${https_proxy:-unset})"
fi

PROTOCOL=$1
shift

echo "python:   $(which python)"
echo "protocol: $PROTOCOL"
echo "args:     $*"

python -m rosettalink.scripts.run_xml --protocol "$PROTOCOL" "$@"
