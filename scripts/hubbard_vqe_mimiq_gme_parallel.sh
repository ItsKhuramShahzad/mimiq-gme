#!/bin/bash
# GME of the VQE ground states vs ED on a multi-core workstation: one process per VQE state, N at a time.
# No MIMIQ needed (states exported on the laptop with --export-states into results/hubbard_vqe_mimiq/states/).
# Needs an env with numpy, scipy, matplotlib, openfermion (e.g. qchem) and the thesis code hubbard_gme.
#
# Run from this folder:   conda activate qchem && N=14 bash scripts/hubbard_vqe_mimiq_gme_parallel.sh
# Results:                results/hubbard_vqe_mimiq/gme/*.json, logs in logs/; table at the end (--summary)
set -euo pipefail
cd "$(dirname "$0")/.."
N="${N:-14}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HUBBARD_GME_SRC="${HUBBARD_GME_SRC:-$HOME/Dropbox/PhDWork/hubbard_gme/src}"
TH="$HOME/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs/uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl"
export THESIS_USCAN_PKL="${THESIS_USCAN_PKL:-$TH}"
[ -f "$THESIS_USCAN_PKL" ] || echo "note: thesis U-scan file not found, the thesis column stays empty (ED is computed anyway)"
mkdir -p logs
STATES=results/hubbard_vqe_mimiq/states
echo "$(ls $STATES/*.npz | wc -l) states, $N at a time, thesis code: $HUBBARD_GME_SRC"
ls "$STATES"/*.npz | sort | xargs -n1 basename | sed 's/\.npz$//' | \
    xargs -P "$N" -I{} sh -c "python -u scripts/hubbard_vqe_mimiq_gme.py --which all --states $STATES --tag {} > logs/gme_{}.log 2>&1 && echo 'done {}'"
python scripts/hubbard_vqe_mimiq_gme.py --summary
