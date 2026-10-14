"""
VQE for the ground state of the Hubbard ring (L = 6, half filling) along U/t, on MIMIQ, and the
comparison with exact diagonalisation (ED).

Each point runs in its own process (one core each, Exaqt limited to one thread) and saves its own
.pkl as soon as it finishes, so partial results can be read at any time with --summary.

Run from the mimiq_gme folder:
    python scripts/hubbard_vqe_mimiq_uscan.py                         # thesis U-scan: t = 1, V = 0, U = 1 ... 20
    python scripts/hubbard_vqe_mimiq_uscan.py --U 1 4 10              # chosen points only
    python scripts/hubbard_vqe_mimiq_uscan.py --summary               # table of the points finished so far
"""

import argparse
import glob
import os
import pickle
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")            # one core per point (Exaqt threads)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hubbard_vqe_mimiq", "uscan")


def run_point(args):
    """one VQE run; the result is saved to OUT_DIR/hubbard_L<L>_t<t>_U<U>_V<V>_uccsd_mimiq-exaqt.pkl"""
    L, t, U, V = args
    from hubbard_vqe_mimiq.core import result_name, run_vqe       # imported in the worker process
    start = time.time()
    result = run_vqe(L=L, t=t, U=U, V=V)
    result["wall_time_s"] = time.time() - start
    with open(os.path.join(OUT_DIR, result_name(L, t, U, V, "uccsd") + ".pkl"), "wb") as fh:
        pickle.dump(result, fh)
    print(f"done U/t = {U / t:5.2f}: dE = {result['dE']:.2e}, fidelity = {result['fidelity']:.5f}, "
          f"{result['n_evaluations']} evaluations, {result['wall_time_s'] / 60:.1f} min", flush=True)
    return result


def summary():
    """table of all the finished points, compared with ED"""
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*_uccsd_mimiq-exaqt.pkl"))):
        with open(path, "rb") as fh:
            rows.append(pickle.load(fh))
    rows.sort(key=lambda r: r["U_over_t"])
    header = (f"{'U/t':>5} {'E0 (ED)':>11} {'E_HF':>9} {'E_VQE':>11} {'E_VQE-E0':>10} {'rel.err':>9} "
              f"{'corr.%':>7} {'F_HF':>7} {'F_VQE':>8} {'evals':>6} {'conv':>5} {'min':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['U_over_t']:5.1f} {r['E0']:11.6f} {r['E_HF']:9.4f} {r['E_VQE']:11.6f} {r['dE']:10.2e} "
              f"{r['rel_error']:9.2e} {100 * r['correlation_recovered']:7.2f} {r['fidelity_HF']:7.4f} "
              f"{r['fidelity']:8.5f} {r['n_evaluations']:6d} {str(r['converged']):>5} {r['wall_time_s'] / 60:6.1f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=6)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--V", type=float, default=0.0)
    p.add_argument("--U", type=float, nargs="+", default=[float(u) for u in range(1, 21)])
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summary:
        summary()
        return
    points = [(args.L, args.t, U, args.V) for U in args.U]
    print(f"{len(points)} points, L = {args.L}, t = {args.t}, V = {args.V}; results in {OUT_DIR}", flush=True)
    with Pool(processes=min(len(points), max(1, os.cpu_count() - 2))) as pool:
        pool.map(run_point, points, chunksize=1)
    print()
    summary()


if __name__ == "__main__":
    main()
