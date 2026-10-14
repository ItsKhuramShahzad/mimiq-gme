# Entanglement of a large GHZ state (default 20 qubits) on MIMIQ.
#
# What scales and what does not:
#   - entropy / Schmidt rank / bond dimension at every cut: yes (MPS, GHZ has bond dimension 2)
#   - density matrix of SMALL clusters (2-4 qubits) from Pauli strings: yes (4^k strings per cluster)
#   - density matrix of the WHOLE state from Pauli strings: no (4^20 ~ 10^12 strings, 2^20 x 2^20 matrix)
#
# Run:  python examples/ghz_large.py          (20 qubits)
#       python examples/ghz_large.py 30       (any number of qubits)

import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

import exaqt
import mimiqcircuits as mc
from hubbard_gme.convex_roof import convex_roof_gcg

# functions from the small example
from ghz_entanglement import ghz_circuit, pauli_matrix, builtin_entanglement


def cluster_density_matrix(n_qubits, cluster):
    """
    Density matrix of the qubits in `cluster` only, from Pauli expectation values:
        rho_cluster = (1 / 2^k) * sum_P <P> P      over the 4^k Pauli strings on the k cluster qubits
    The GHZ state is prepared on all n_qubits; only strings on the cluster are measured.
    """
    k = len(cluster)
    strings = ["".join(s) for s in product("IXYZ", repeat=k)]
    measured = [s for s in strings if s != "I" * k]               # <I...I> = 1

    circuit = ghz_circuit(n_qubits)
    for z, letters in enumerate(measured):
        circuit.push(mc.ExpectationValue(mc.PauliString(letters)), *cluster, z)

    mimiq_state = exaqt.ExaqtState.zero(n_qubits, num_zvars=circuit.num_zvars())
    mimiq_state, _ = exaqt.evolve(mimiq_state, circuit)

    rho = pauli_matrix("I" * k).astype(complex)                   # the identity term, <I...I> = 1
    for z, letters in enumerate(measured):
        rho += mimiq_state.z[z].real * pauli_matrix(letters)
    return rho / 2**k


def main(n_qubits):
    print(f"GHZ state on {n_qubits} qubits: (|0...0> + |1...1>) / sqrt(2)\n")

    # Part 1: built-in measures on the MPS simulator (needs a free TensorWeaver licence seat)
    try:
        builtin_entanglement(n_qubits)
    except Exception as error:
        print(f"Part 1 skipped: {type(error).__name__}: {error}\n")

    # Part 2 + 3: small clusters, measured with Pauli strings, then entanglement with the thesis code
    print("Part 2 + 3: clusters from Pauli strings (Exaqt), entanglement with the thesis code")
    clusters = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, n_qubits // 2, n_qubits - 1]]
    for cluster in clusters:
        start = time.time()
        rho = cluster_density_matrix(n_qubits, cluster)
        purity = np.trace(rho @ rho).real
        result = convex_roof_gcg(rho, len(cluster), n_restarts=5, verbose=False)
        kind = "bipartite entanglement" if len(cluster) == 2 else "GME concurrence"
        print(f"  qubits {cluster}: {4**len(cluster) - 1:3d} strings, purity = {purity:.3f}, "
              f"{kind} = {result['C_gcg']:.2e}   ({time.time() - start:.1f} s)")

    print()
    print("Every cluster is (|0..0><0..0| + |1..1><1..1|)/2: classically correlated, no entanglement,")
    print("no matter how far apart the qubits are. All GHZ entanglement involves all qubits at once.")
    print(f"The GME of the full {n_qubits}-qubit state is 1, but it cannot be obtained from Pauli")
    print(f"strings: that needs all 4^{n_qubits} of them. For large n, entanglement witnesses or spin")
    print("inequalities (a few collective measurements) are the practical route.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
