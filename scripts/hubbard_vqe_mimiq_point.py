"""
VQE for ONE point of the Hubbard ring on MIMIQ, with progress printed while it runs, and the
comparison with exact diagonalisation (ED) at the end. The result is saved to
    results/hubbard_vqe_mimiq/point/hubbard_L<L>_t<t>_U<U>_V<V>_<ansatz>_mimiq-exaqt.pkl
with <ansatz> = uccsd, lucj-local-r2, hva-p4, ...

Run from the mimiq_gme folder:
    python scripts/hubbard_vqe_mimiq_point.py --U 10                  # t = 1, V = 0, L = 6
    python scripts/hubbard_vqe_mimiq_point.py --U 4 --every 50        # progress every 50 evaluations
    python scripts/hubbard_vqe_mimiq_point.py --U 10 --ansatz lucj    # LUCJ (local, 2 repetitions) instead of UCCSD
    python scripts/hubbard_vqe_mimiq_point.py --U 10 --ansatz hva --layers 4   # Hamiltonian variational ansatz, 4 layers
"""

import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hubbard_vqe_mimiq.core import RESULTS_DIR, result_name, run_vqe

p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=6)
p.add_argument("--t", type=float, default=1.0)
p.add_argument("--U", type=float, default=10.0)
p.add_argument("--V", type=float, default=0.0)
p.add_argument("--every", type=int, default=100, help="print progress every N energy evaluations")
p.add_argument("--ansatz", choices=["uccsd", "lucj", "hva"], default="uccsd")
p.add_argument("--layers", type=int, default=4, help="HVA layers")
p.add_argument("--reps", type=int, default=2, help="LUCJ repetitions")
p.add_argument("--pairs", choices=["local", "full"], default="local", help="LUCJ interaction pairs")
args = p.parse_args()

start = time.time()
if args.ansatz == "hva":
    from hubbard_vqe_mimiq.hva import run_vqe_hva
    r = run_vqe_hva(L=args.L, t=args.t, U=args.U, V=args.V, n_layers=args.layers, progress_every=args.every)
elif args.ansatz == "lucj":
    from hubbard_vqe_mimiq.lucj import run_vqe_lucj
    r = run_vqe_lucj(L=args.L, t=args.t, U=args.U, V=args.V, n_reps=args.reps, pairs_kind=args.pairs,
                     progress_every=args.every)
else:
    r = run_vqe(L=args.L, t=args.t, U=args.U, V=args.V, verbose=True, progress_every=args.every)
r["wall_time_s"] = time.time() - start

out_dir = os.path.join(RESULTS_DIR, "point")
os.makedirs(out_dir, exist_ok=True)
tag = {"uccsd": "uccsd", "lucj": f"lucj-{args.pairs}-r{args.reps}", "hva": f"hva-p{args.layers}"}[args.ansatz]
path = os.path.join(out_dir, result_name(args.L, args.t, args.U, args.V, tag) + ".pkl")
with open(path, "wb") as fh:
    pickle.dump(r, fh)

print(f"\nResult (U/t = {r['U_over_t']:g}, {r.get('ansatz', 'UCCSD')}, {r['n_parameters']} parameters), "
      f"compared with exact diagonalisation:")
print(f"  E0 (ED)          = {r['E0']:.8f}")
print(f"  E_HF (start)     = {r['E_HF']:.8f}")
print(f"  E_VQE            = {r['E_VQE']:.8f}")
print(f"  E_VQE - E0       = {r['dE']:.3e}   (relative {r['rel_error']:.2e})")
print(f"  correlation      = {100 * r['correlation_recovered']:.2f} % of (E_HF - E0) recovered")
print(f"  fidelity         = {r['fidelity']:.6f}   (Hartree-Fock: {r['fidelity_HF']:.6f})")
print(f"  evaluations      = {r['n_evaluations']}, cycles = {r['cycles']}, converged = {r['converged']}")
print(f"  time             = {r['wall_time_s'] / 60:.1f} min   -> saved {path}")
