# Entanglement of k-qubit clusters of the GHZ and W states (GCG convex roof, thesis code).
#
# Part 1: GHZ on 6 qubits, every cluster (sizes 2-5), from Pauli strings measured on MIMIQ.
# Part 2: GHZ on 20 qubits (circuit of ghz_large.py), clusters that do NOT contain qubit 0.
# Part 3: GHZ vs W, clusters of 2, 3, 4 qubits, N = 6 and 20 (exact state vectors;
#         both states are symmetric, so any k qubits give the same value).
#
# Result: every GHZ cluster has purity 0.5 and C = 0 (the coherence <0..0|rho|1..1> exists only in
# the full state); W keeps entanglement in every cluster (pairs: concurrence 2/N, exact).
#
# Run:  python examples/ghz_w_clusters.py

import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

from hubbard_gme.convex_roof import convex_roof_gcg, C_GME_pure_dm

from ghz_large import cluster_density_matrix                    # GHZ circuit + Pauli strings on MIMIQ
from w_state import w_vector


def ghz_vector(n_qubits):
    psi = np.zeros(2**n_qubits)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi


def reduced(psi, n_qubits, keep):
    """density matrix of the qubits in keep (qubit 0 = leftmost bit)"""
    tensor = psi.reshape([2] * n_qubits)
    rest = [q for q in range(n_qubits) if q not in keep]
    tensor = np.transpose(tensor, list(keep) + rest).reshape(2**len(keep), -1)
    return tensor @ tensor.conj().T


def part1_all_ghz_clusters(n_qubits=6):
    print(f"Part 1: GHZ, N = {n_qubits}, every cluster (Pauli strings on MIMIQ)")
    for k in (2, 3, 4, 5):
        values = []
        for cluster in combinations(range(n_qubits), k):
            rho = cluster_density_matrix(n_qubits, list(cluster))
            purity = np.trace(rho @ rho).real
            c = convex_roof_gcg(rho, k, n_restarts=3, verbose=False)["C_gcg"]
            values.append((cluster, purity, c))
        without_0 = sum(0 not in v[0] for v in values)
        print(f"  k = {k}: {len(values)} clusters ({without_0} without qubit 0) | purity "
              f"{min(v[1] for v in values):.3f}-{max(v[1] for v in values):.3f} | max C = {max(v[2] for v in values):.1e}")
    rho = cluster_density_matrix(n_qubits, list(range(n_qubits)))
    print(f"  all {n_qubits} qubits: purity {np.trace(rho @ rho).real:.6f}, C_GME = {C_GME_pure_dm(rho, n_qubits):.6f}\n")


def part2_ghz20_without_qubit0(n_qubits=20):
    print(f"Part 2: GHZ, N = {n_qubits}, clusters without qubit 0 (Pauli strings on MIMIQ)")
    clusters = [[0, 1], [0, 1, 2],                          # with qubit 0, for comparison
                [5, 6], [18, 19], [3, 15],                  # pairs
                [9, 10, 11], [17, 18, 19], [2, 10, 19],     # triples
                [10, 11, 12, 13], [1, 7, 13, 19]]           # quadruples
    for cluster in clusters:
        start = time.time()
        rho = cluster_density_matrix(n_qubits, cluster)
        purity = np.trace(rho @ rho).real
        c = convex_roof_gcg(rho, len(cluster), n_restarts=3, verbose=False)["C_gcg"]
        kind = "concurrence" if len(cluster) == 2 else "C_GME"
        print(f"  qubits {str(cluster):18} purity = {purity:.3f}  {kind} = {c:.1e}  "
              f"|<0..0|rho|1..1>| = {abs(rho[0, -1]):.1e}  ({time.time() - start:.1f} s)")
    print()


def part3_ghz_vs_w():
    print("Part 3: GHZ vs W, clusters of k qubits (exact state vectors)")
    for n in (6, 20):
        for name, psi in (("GHZ", ghz_vector(n)), ("W", w_vector(n))):
            row = []
            for k in (2, 3, 4):
                rho = reduced(psi, n, list(range(k)))
                c = convex_roof_gcg(rho, k, n_restarts=3, verbose=False)["C_gcg"]
                row.append(f"k={k}: C={c:.3f}")
            print(f"  N = {n:2d} {name:3}  " + " | ".join(row))


if __name__ == "__main__":
    part1_all_ghz_clusters()
    part2_ghz20_without_qubit0()
    part3_ghz_vs_w()
