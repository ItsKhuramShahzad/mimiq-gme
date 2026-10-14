"""
HVA VQE along a slow ramp in U/t (warm start): every point starts from the optimized angles of the previous one.

    U/t = 1 (start theta = 0, the U = 0 ground state)  ->  2  ->  3  ->  4  ->  ...
          theta_opt(U_1) is the start of U_2, theta_opt(U_2) the start of U_3, ...

The HVA circuit does not depend on U (U enters only through the measured energy), so the angles of one
point are a valid circuit at the next point, and the ground state changes smoothly with U (the gap of the
finite ring stays open). Each result is saved as it finishes to
    results/hubbard_vqe_mimiq/ramp/hubbard_L6_t1_V0_hva-p<p>-ramp_mimiq-exaqt/hubbard_L6_t1_U<U>_V0_hva-p<p>-ramp_mimiq-exaqt.pkl
with the fidelity of the warm start ("start_fidelity") next to the fidelity after the optimization.

Run from the mimiq_gme folder:
    python scripts/hubbard_vqe_mimiq_ramp.py --layers 8                        # U/t = 1 2 3 4 5 6 8 10
    python scripts/hubbard_vqe_mimiq_ramp.py --layers 4 --U 1 2 3 4 5 6
    python scripts/hubbard_vqe_mimiq_ramp.py --layers 8 --start results/hubbard_vqe_mimiq/point/hubbard_L6_t1_U1_V0_hva-p8_mimiq-exaqt.pkl --U 2 3 4
    python scripts/hubbard_vqe_mimiq_ramp.py --layers 8 --summary
"""

import argparse
import glob
import os
import pickle
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from openfermion import get_sparse_operator
from openfermion.linalg import jw_number_restrict_operator
from scipy.sparse.linalg import eigsh

from hubbard_vqe_mimiq.core import RESULTS_DIR, result_name, hubbard_hamiltonian, exact_ground_state, state_vector
from hubbard_vqe_mimiq.hva import hva_ansatz, run_vqe_hva

p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=6)
p.add_argument("--t", type=float, default=1.0)
p.add_argument("--V", type=float, default=0.0)
p.add_argument("--U", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
p.add_argument("--layers", type=int, default=8)
p.add_argument("--max-cycles", type=int, default=15, help="COBYLA chunks of 600 evaluations per point")
p.add_argument("--every", type=int, default=0, help="print progress every N energy evaluations (0 = off)")
p.add_argument("--start", default=None, help="pkl of a finished HVA run: its theta_opt starts the first point")
p.add_argument("--summary", action="store_true")
args = p.parse_args()

tag = f"hva-p{args.layers}-ramp"
out_dir = os.path.join(RESULTS_DIR, "ramp", f"hubbard_L{args.L}_t{args.t:g}_V{args.V:g}_{tag}_mimiq-exaqt")
os.makedirs(out_dir, exist_ok=True)


def summary():
    rows = [pickle.load(open(f, "rb")) for f in glob.glob(os.path.join(out_dir, "*.pkl"))]
    rows.sort(key=lambda r: r["U_over_t"])
    header = (f"{'U/t':>5} {'E0 (ED)':>10} {'gap':>6} {'E_VQE':>10} {'E_VQE-E0':>9} {'dE/gap':>7} "
              f"{'F_theta0':>8} {'F_start':>8} {'F_VQE':>8} {'evals':>6} {'min':>5}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['U_over_t']:5.1f} {r['E0']:10.5f} {r['gap']:6.3f} {r['E_VQE']:10.5f} {r['dE']:9.2e} "
              f"{r['dE'] / r['gap']:7.4f} {r['fidelity_HF']:8.4f} {r['start_fidelity']:8.4f} {r['fidelity']:8.5f} "
              f"{r['n_evaluations']:6d} {r['runtime_s'] / 60:5.1f}")


if args.summary:
    summary()
    sys.exit()

n_qubits, n_electrons = 2 * args.L, args.L
ansatz = hva_ansatz(args.L, args.t, args.layers, with_V=abs(args.V) > 0)
n_params = ansatz.per_layer * args.layers
theta = np.zeros(n_params)
if args.start:
    with open(args.start, "rb") as fh:
        theta = np.asarray(pickle.load(fh)["theta_opt"], dtype=float)
    if theta.shape != (n_params,):
        sys.exit(f"--start has {theta.size} angles, HVA p = {args.layers} needs {n_params}")

print(f"HVA ramp, p = {args.layers} ({n_params} parameters), L = {args.L}, t = {args.t}, V = {args.V}, "
      f"U/t = {' -> '.join(f'{u:g}' for u in args.U)}; start: {args.start or 'theta = 0'}\n", flush=True)
for U in args.U:
    H = hubbard_hamiltonian(args.L, args.t, U, args.V)
    E0, psi0 = exact_ground_state(H, n_qubits, n_electrons)
    start_fidelity = float(abs(np.vdot(psi0, state_vector(ansatz(n_qubits, n_electrons, theta), n_qubits))) ** 2)

    r = run_vqe_hva(L=args.L, t=args.t, U=U, V=args.V, n_layers=args.layers, max_cycles=args.max_cycles,
                    progress_every=args.every, theta_start=theta)
    w = np.sort(eigsh(jw_number_restrict_operator(get_sparse_operator(H, n_qubits=n_qubits), n_electrons, n_qubits),
                      k=2, which="SA")[0])
    r.update(start_fidelity=start_fidelity, theta_start=theta.copy(), gap=float(w[1] - w[0]),
             ramp=list(args.U), ramp_start=args.start)
    with open(os.path.join(out_dir, result_name(args.L, args.t, U, args.V, tag) + ".pkl"), "wb") as fh:
        pickle.dump(r, fh)
    print(f"U/t = {U:5.2f}: start fidelity {start_fidelity:.4f} (theta = 0: {r['fidelity_HF']:.4f}) -> "
          f"fidelity {r['fidelity']:.5f}, E_VQE - E0 = {r['dE']:.2e} ({r['dE'] / r['gap']:.4f} of the gap), "
          f"{r['n_evaluations']} evaluations, {r['runtime_s'] / 60:.1f} min", flush=True)
    theta = np.asarray(r["theta_opt"], dtype=float)

print()
summary()
