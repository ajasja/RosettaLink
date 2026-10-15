#!/bin/bash
#SBATCH --job-name=rosettalink
#SBATCH --partition=gpu-a40,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%A_%x.out
#SBATCH --error=logs/slurm-%A_%x.err

# Runs a RosettaScripts XML through rosetta_link_scripts on a slurm node.
# The protocol comes first; every argument after it is passed through to the
# runner:
#
#   mkdir -p logs
#   CONDA_ENV=my_rosettalink_env sbatch run_protocol.sh \
#       examples/redesign_colabfold.xml \
#       -s examples/input_data/insulin_target.pdb -out:path output/redesign
#
# CONDA_ENV is the conda environment holding PyRosetta and rosettalink, by
# name or path. The external programs are reached through
# rosettalink.config.yaml.
#
# PROXY_SETUP is a script to source before the run, for a compute node that
# reaches the internet only through a proxy. ColabFold needs the internet for
# any msa_mode other than single_sequence.
#
# The job runs in the submit directory, and the slurm logs go to logs/ there,
# which must exist before sbatch is called. Pass --output and --error to
# sbatch to put them elsewhere.

set -euo pipefail

if [ -z "${CONDA_ENV:-}" ]; then
    echo "Set CONDA_ENV to the conda environment holding PyRosetta and rosettalink" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-.}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

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
