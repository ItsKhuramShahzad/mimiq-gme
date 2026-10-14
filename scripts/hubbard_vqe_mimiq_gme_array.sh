#!/bin/bash
# GME of the VQE ground states vs ED on a SLURM cluster: one array task per VQE state (28 states).
# No MIMIQ needed: the states were exported on the laptop with
#     python scripts/hubbard_vqe_mimiq_gme.py --export-states --which all
# and copied to results/hubbard_vqe_mimiq/states/ on the cluster.
#
# Needs: numpy, scipy, matplotlib, openfermion, and the thesis code hubbard_gme.
# Submit from the mimiq_gme folder:   mkdir -p logs && sbatch scripts/hubbard_vqe_mimiq_gme_array.sh
# Afterwards:                          python scripts/hubbard_vqe_mimiq_gme.py --summary
#
#SBATCH --job-name=hubbard-vqe-gme
#SBATCH --array=0-27
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hubbard_vqe_mimiq_gme_%A_%a.log
# fill in for your cluster:
##SBATCH --account=<account>
##SBATCH --partition=<partition>

set -euo pipefail
# activate the environment that runs the thesis code, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HUBBARD_GME_SRC="${HUBBARD_GME_SRC:-$HOME/hubbard_gme/src}"     # thesis code on the cluster
# optional, adds the thesis column:  export THESIS_USCAN_PKL=/path/to/uscan_v1_L6_Ne6_t1_..._sz0_restarts10_iter200_...pkl

STATES=results/hubbard_vqe_mimiq/states
mapfile -t TAGS < <(ls "$STATES"/*.npz | sort | xargs -n1 basename | sed 's/\.npz$//')
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID: $TAG on $(hostname)"
python -u scripts/hubbard_vqe_mimiq_gme.py --which all --states "$STATES" --tag "$TAG"
