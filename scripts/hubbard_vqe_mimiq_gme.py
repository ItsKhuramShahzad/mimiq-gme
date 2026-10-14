"""
GME of the VQE ground states of the Hubbard ring, compared with exact diagonalization (ED) and with the
thesis U-scan.

For every finished VQE result:
  1. n-site RDM of the sites 1..n, built exactly as in the thesis (hubbard_gme.thermal_pipeline):
        rho[bra, ket] = <psi| prod_b c+_{2 s_b + sigma_b} c_{2 s_b + sigma'_b} |psi>,   normalized to trace 1
     (bra, ket = spin configurations of the n sites; the trace is the weight of single occupation)
  2. C_GME with the thesis solver and the thesis settings (compute_gme: GCG convex roof, exact formula if pure)
  3. the same for the exact ground state (ED), and the thesis value read from the U-scan file (if available)
Also reported: fidelity, max |rho_VQE - rho_ED|, trace distance, and how much the VQE state breaks the
translation symmetry (largest difference between the RDMs of the L clusters). The HVA layers treat the bonds
(1,2),(3,4),(5,6) and (2,3),(4,5),(6,1) differently, so its state is only invariant under a shift by two sites;
C_VQE_avg is the GME of the RDM averaged over the L cluster positions, the fair comparison with the
translation-invariant ED state.

Two steps, so that the slow part (GCG) can run on a cluster without MIMIQ:
  laptop (MIMIQ):   python scripts/hubbard_vqe_mimiq_gme.py --export-states --which all
                    -> results/hubbard_vqe_mimiq/states/<result>.npz   (state vector, 4096 amplitudes each)
  anywhere (numpy, scipy, openfermion, matplotlib + hubbard_gme; no MIMIQ):
                    python scripts/hubbard_vqe_mimiq_gme.py --states results/hubbard_vqe_mimiq/states --match <result>
                    (one result per job: scripts/hubbard_vqe_mimiq_gme_array.sh)
  table of everything computed so far:
                    python scripts/hubbard_vqe_mimiq_gme.py --summary
Without --states the states are rebuilt from the saved angles on MIMIQ (laptop).

Paths on another machine: HUBBARD_GME_SRC (thesis code, .../hubbard_gme/src), THESIS_USCAN_PKL (optional,
the thesis U-scan file; without it the thesis column is empty, the ED value is computed anyway).
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.environ.get("HUBBARD_GME_SRC", "/home/anjum/PhDWork/hubbard_gme/src"))

import numpy as np
from openfermion import FermionOperator, get_sparse_operator, reorder
from openfermion.linalg import jw_get_ground_state_at_particle_number

from hubbard_gme.hamiltonian import build_hubbard_hamiltonian
from hubbard_gme.thermal_pipeline import compute_gme

RESULTS_DIR = os.path.join(ROOT, "results", "hubbard_vqe_mimiq")
THESIS_USCAN = os.environ.get(
    "THESIS_USCAN_PKL",
    "/mnt/d/Dropbox/PhDWork/hubbard_gme/results/rdm_pkl_results/thermal_runs/"
    "uscan_v1_L6_Ne6_t1_mode-uniform_nmin2_solver-gcg_sz0_restarts10_iter200_kmax4_tol1em09_U1to20_N20-d03a74da789e.pkl")
GCG = dict(solver="gcg", n_restarts=10, max_iter=200, tol=1e-9, k_max_extra=4)   # the thesis settings of that file
PATTERNS = {"ramp-p8": "ramp/*hva-p8-ramp*/*.pkl", "ramp-p4": "ramp/*hva-p4-ramp*/*.pkl", "point": "point/*.pkl"}


# ------------------------------------------------------------------------------------------
# states
# ------------------------------------------------------------------------------------------
def tag_of(path):
    return os.path.basename(path).replace("_mimiq-exaqt.pkl", "").replace(".npz", "")


def group_of(tag):
    return "ramp-p8" if "hva-p8-ramp" in tag else "ramp-p4" if "hva-p4-ramp" in tag else "point"


def vqe_state(r):
    """state vector (OpenFermion order) of a saved VQE result, rebuilt on MIMIQ, and its qubit layout"""
    from hubbard_vqe_mimiq.core import hopping_orbitals, make_ansatz, state_vector
    from hubbard_vqe_mimiq.hva import hva_ansatz
    L, t, name = r["L"], r["t"], r.get("ansatz", "UCCSD")
    nq, ne = 2 * L, L
    if name.startswith("HVA"):
        layout = r.get("layout", "interleaved")
        ansatz = hva_ansatz(L, t, int(name.split("p")[1]), with_V=abs(r["V"]) > 0, layout=layout,
                            native_hopping=r.get("native_hopping", False))
    elif name.startswith("LUCJ"):
        from hubbard_vqe_mimiq.lucj import lucj_ansatz
        layout = "blocked"                                   # ffsim: all spin up, then all spin down
        _, kind, reps = name.split()
        ansatz = lucj_ansatz(hopping_orbitals(L, t)[1], int(reps[1:]), kind)
    else:
        layout = "interleaved"
        ansatz = make_ansatz(hopping_orbitals(L, t)[1])
    return state_vector(ansatz(nq, ne, r["theta_opt"]), nq), layout


META = ("L", "t", "U", "V", "fidelity", "dE")


def result_files(which, match):
    keys = list(PATTERNS) if which == "all" else [which]
    return sorted(f for k in keys for f in glob.glob(os.path.join(RESULTS_DIR, PATTERNS[k])) if match in os.path.basename(f))


def export_states(which, match, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in result_files(which, match):
        r = pickle.load(open(f, "rb"))
        psi, layout = vqe_state(r)
        path = os.path.join(out_dir, tag_of(f) + ".npz")
        np.savez(path, psi=psi, layout=layout, ansatz=r.get("ansatz", "UCCSD"), **{k: r[k] for k in META})
        print(f"saved {path}  (F = {r['fidelity']:.5f}, norm = {np.linalg.norm(psi):.12f})", flush=True)


def load_states(which, match, states_dir, exact=""):
    """(tag, metadata, psi, layout) from the exported .npz files, or rebuilt on MIMIQ from the result .pkl files"""
    if states_dir:
        for f in sorted(glob.glob(os.path.join(states_dir, "*.npz"))):
            tag = tag_of(f)
            if (exact and tag != exact) or match not in tag or (which != "all" and group_of(tag) != which):
                continue
            d = np.load(f)
            meta = {k: float(d[k]) for k in META}
            meta["L"] = int(meta["L"]); meta["ansatz"] = str(d["ansatz"])
            yield tag, meta, d["psi"], str(d["layout"])
    else:
        for f in result_files(which, match):
            r = pickle.load(open(f, "rb"))
            psi, layout = vqe_state(r)
            meta = {k: r[k] for k in META}; meta["ansatz"] = r.get("ansatz", "UCCSD")
            yield tag_of(f), meta, psi, layout


# ------------------------------------------------------------------------------------------
# RDMs
# ------------------------------------------------------------------------------------------
def to_layout(op, L, layout):
    """same operator with the modes renumbered from interleaved (2k + s) to blocked (s L + k)"""
    if layout == "interleaved":
        return op
    return reorder(op, lambda m, n: (m % 2) * (n // 2) + m // 2, num_modes=2 * L)


def exact_ground_state(L, t, U, V):
    H = build_hubbard_hamiltonian(L, t, U, mode="uniform", V=V)
    return np.asarray(jw_get_ground_state_at_particle_number(get_sparse_operator(H, n_qubits=2 * L), L)[1])


OPS = {}


def rdm_operators(L, sites, layout):
    """sparse operators of the n-site RDM, same definition as hubbard_gme.thermal_pipeline._precompute_ops"""
    key = (L, tuple(sites), layout)
    if key not in OPS:
        n, ops = len(sites), {}
        for bra in range(2**n):
            bs = [(bra >> k) & 1 for k in range(n)]
            for ket in range(2**n):
                ks = [(ket >> k) & 1 for k in range(n)]
                terms = []
                for b, site in enumerate(sites):
                    terms += [(2 * site + bs[b], 1), (2 * site + ks[b], 0)]
                op = to_layout(FermionOperator(tuple(terms)), L, layout)
                ops[(bra, ket)] = get_sparse_operator(op, n_qubits=2 * L)
        OPS[key] = ops
    return OPS[key]


def rdm(psi, L, sites, layout="interleaved"):
    n = len(sites)
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for (bra, ket), op in rdm_operators(L, sites, layout).items():
        rho[bra, ket] = np.vdot(psi, op @ psi)
    tr = float(np.real(np.trace(rho)))
    return rho / tr, tr


def trace_distance(a, b):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b))))


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def summary(out_dir):
    rows = {}
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        for r in json.load(open(f)):
            rows[(r["file"], r["n"])] = r
    if not rows:
        print("no results yet"); return
    print(f"{'result':40s} {'U/t':>4} {'n':>2} {'F':>8} {'C_ED':>7} {'C_thesis':>8} {'C_VQE':>7} {'C_avg':>7} "
          f"{'|dC_avg|':>8} {'spread':>7}")
    for (tag, n), r in sorted(rows.items(), key=lambda kv: (group_of(kv[0][0]), kv[1]["U_over_t"], kv[0][0], kv[0][1])):
        th = f"{r['C_thesis']:.4f}" if r.get("C_thesis") is not None else "-"
        print(f"{tag:40s} {r['U_over_t']:4.0f} {n:2d} {r['fidelity']:8.5f} {r['C_ED']:7.4f} {th:>8} {r['C_VQE']:7.4f} "
              f"{r['C_VQE_avg']:7.4f} {abs(r['dC_avg']):8.4f} {r['cluster_spread']:7.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ramp-p8", "ramp-p4", "point", "all"], default="ramp-p8")
    ap.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--match", default="", help="only results whose name contains this text (e.g. uccsd)")
    ap.add_argument("--tag", default="", help="exactly one result, by its full name (used by the array job)")
    ap.add_argument("--states", default="", help="folder of exported .npz states (no MIMIQ needed)")
    ap.add_argument("--export-states", action="store_true", help="save the VQE state vectors and stop (needs MIMIQ)")
    ap.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "gme"))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary:
        summary(args.out_dir); return
    if args.export_states:
        export_states(args.which, args.match, os.path.join(RESULTS_DIR, "states")); return

    thesis = pickle.load(open(THESIS_USCAN, "rb"))["result"] if os.path.exists(THESIS_USCAN) else None
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.tag or f"{args.which}{'-' + args.match if args.match else ''}"
    out = os.path.join(args.out_dir, f"hubbard_L6_t1_V0_gme_{label}_mimiq-exaqt.json")
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {(r["file"], r["n"]) for r in rows}
    ed_cache, ed_gme = {}, {}

    for tag, r, psi, layout in load_states(args.which, args.match, args.states, args.tag):
        L, U, name = r["L"], r["U"], r["ansatz"]
        if U not in ed_cache:
            ed_cache[U] = exact_ground_state(L, r["t"], U, r["V"])
        psi0 = ed_cache[U]
        for n in args.n:
            if (tag, n) in done:
                continue
            sites = list(range(n))
            t0 = time.time()
            rho_v, tr_v = rdm(psi, L, sites, layout)
            rho_e, tr_e = rdm(psi0, L, sites)
            c_v = compute_gme(rho_v, n, **GCG)[0]
            if (U, n) not in ed_gme:                       # the ED value is computed once per (U, n)
                ed_gme[(U, n)] = compute_gme(rho_e, n, **GCG)[0]
            c_e = ed_gme[(U, n)]
            shifted = [rdm(psi, L, [(s + k) % L for s in sites], layout)[0] for k in range(L)]
            spread = max(float(np.max(np.abs(x - rho_v))) for x in shifted)
            rho_avg = sum(shifted) / L                     # translation-averaged RDM (the ED state is invariant)
            c_avg = compute_gme(rho_avg, n, **GCG)[0]
            c_th = None
            if thesis is not None and n in thesis["C_GME"]:
                idx = np.where(np.isclose(thesis["U_over_t"], U / r["t"]))[0]
                c_th = float(thesis["C_GME"][n][idx[0]]) if len(idx) else None
            row = {"file": tag, "ansatz": name, "U_over_t": U / r["t"], "n": n, "fidelity": r["fidelity"],
                   "dE": r["dE"], "C_VQE": c_v, "C_VQE_avg": c_avg, "C_ED": c_e, "C_thesis": c_th,
                   "dC": c_v - c_e, "dC_avg": c_avg - c_e,
                   "rho_max_diff": float(np.max(np.abs(rho_v - rho_e))), "trace_distance": trace_distance(rho_v, rho_e),
                   "trace_distance_avg": trace_distance(rho_avg, rho_e),
                   "single_occupation_VQE": tr_v, "single_occupation_ED": tr_e,
                   "cluster_spread": float(spread), "seconds": time.time() - t0}
            rows.append(row)
            with open(out, "w") as fh:
                json.dump(rows, fh, indent=1)
            print(f"{tag:42s} n={n}  F={r['fidelity']:.5f}  C_VQE={c_v:.4f}  C_ED={c_e:.4f}  "
                  f"thesis={'%.4f' % c_th if c_th is not None else '  -   '}  |dC|={abs(c_v - c_e):.4f}  "
                  f"C_avg={c_avg:.4f} |dC_avg|={abs(c_avg - c_e):.4f}  "
                  f"TD={row['trace_distance']:.1e}  spread={spread:.1e}  ({row['seconds']:.0f} s)", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
