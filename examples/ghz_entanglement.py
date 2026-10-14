# Entanglement of a GHZ state on MIMIQ: entropy, Schmidt rank, bond dimension and GME.
#
#   GHZ state on n qubits: (|00...0> + |11...1>) / sqrt(2)
#
# Part 1: MIMIQ's built-in entanglement measures (need the MPS simulator, TensorWeaver).
# Part 2: the density matrix rebuilt from Pauli expectation values only (Exaqt simulator).
# Part 3: GME concurrence with the thesis code (hubbard_gme).
#
# Run:  python examples/ghz_entanglement.py

import sys
from itertools import product

import numpy as np

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

import exaqt
import tensorweaver
import mimiqcircuits as mc
from hubbard_gme.convex_roof import C_GME_pure_dm, convex_roof_gcg

N_QUBITS = 4


def ghz_circuit(n_qubits):
    """
    GHZ state on a 1D chain of qubits:
      H on qubit 0                      -> (|0> + |1>) / sqrt(2) on qubit 0
      CX 0->1, CX 1->2, ..., CX (n-2)->(n-1)   -> every qubit copies its left neighbour
    Result: (|00...0> + |11...1>) / sqrt(2)
    """
    circuit = mc.Circuit()
    circuit.push(mc.GateH(), 0)
    for q in range(n_qubits - 1):
        circuit.push(mc.GateCX(), q, q + 1)
    return circuit


# ------------------------------------------------------------------------------------------
# Part 1: MIMIQ's built-in entanglement measures
# ------------------------------------------------------------------------------------------
def builtin_entanglement(n_qubits):
    """
    For every cut k, MIMIQ splits the qubits into  A = {0, ..., k-1}  and  B = {k, ..., n-1}.
    Each measure is pushed on ONE qubit (k, the cut) and writes its result in one z-register slot:
        VonNeumannEntropy : S(rho_A) = -Tr(rho_A log2 rho_A)   (0 = product, 1 = one maximally entangled bit)
        SchmidtRank       : number of terms in the Schmidt decomposition (1 = product, > 1 = entangled)
        BondDim           : bond dimension of the MPS at that cut (equals the Schmidt rank here)
    These operations exist only for the MPS simulator (TensorWeaver), not for Exaqt.
    """
    circuit = ghz_circuit(n_qubits)

    z = 0
    slots = {"entropy": [], "schmidt": [], "bonddim": []}
    for k in range(n_qubits):
        circuit.push(mc.VonNeumannEntropy(), k, z)
        slots["entropy"].append(z)
        z += 1
    for k in range(n_qubits):
        circuit.push(mc.SchmidtRank(), k, z)
        slots["schmidt"].append(z)
        z += 1
    for k in range(n_qubits):
        circuit.push(mc.BondDim(), k, z)
        slots["bonddim"].append(z)
        z += 1

    simulator = tensorweaver.TwSimulator()
    result = simulator.execute(circuit, nsamples=1)
    zvalues = np.real(result.zstates[0])

    print("Part 1: MIMIQ built-in measures (TensorWeaver, MPS)")
    print("  cut k (A = qubits 0..k-1):", list(range(n_qubits)))
    print("  entropy  S(rho_A)        :", [round(float(zvalues[i]), 6) for i in slots["entropy"]])
    print("  Schmidt rank             :", [int(round(zvalues[i])) for i in slots["schmidt"]])
    print("  bond dimension           :", [int(round(zvalues[i])) for i in slots["bonddim"]])
    print()


# ------------------------------------------------------------------------------------------
# Part 2: density matrix from Pauli expectation values
# ------------------------------------------------------------------------------------------
PAULI_MATRIX = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


def pauli_matrix(letters):
    """Matrix of a Pauli string, qubit 0 = leftmost factor (e.g. "XZ" = X on qubit 0, Z on qubit 1)."""
    matrix = np.array([[1.0]])
    for letter in letters:
        matrix = np.kron(matrix, PAULI_MATRIX[letter])
    return matrix


def density_matrix_from_paulis(n_qubits):
    """
    Any n-qubit density matrix can be written with Pauli strings:
        rho = (1 / 2^n) * sum_P <P> P          (4^n strings, including the identity)
    We only measure the numbers <P> on MIMIQ (no state read-out), then add them up here.
    """
    all_strings = ["".join(s) for s in product("IXYZ", repeat=n_qubits)]
    measured = [s for s in all_strings if s != "I" * n_qubits]      # <I...I> = 1, no need to measure

    # one circuit: prepare GHZ, then one expectation value per Pauli string
    circuit = ghz_circuit(n_qubits)
    for z, letters in enumerate(measured):
        circuit.push(mc.ExpectationValue(mc.PauliString(letters)), *range(n_qubits), z)

    mimiq_state = exaqt.ExaqtState.zero(n_qubits, num_zvars=circuit.num_zvars())
    mimiq_state, _ = exaqt.evolve(mimiq_state, circuit)

    expectation = {"I" * n_qubits: 1.0}
    for z, letters in enumerate(measured):
        expectation[letters] = mimiq_state.z[z].real

    rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for letters, value in expectation.items():
        rho += value * pauli_matrix(letters)
    rho /= 2**n_qubits

    nonzero = {s: round(v, 6) for s, v in expectation.items() if abs(v) > 1e-9}
    print("Part 2: density matrix from Pauli expectation values (Exaqt)")
    print(f"  measured {len(measured)} Pauli strings; the non-zero ones:")
    print("  ", nonzero)
    return rho


def reduced_density_matrix(rho, n_qubits, keep):
    """Trace out every qubit not in `keep` (qubit 0 = leftmost factor)."""
    tensor = rho.reshape([2] * (2 * n_qubits))
    traced = [q for q in range(n_qubits) if q not in keep]
    for q in sorted(traced, reverse=True):                 # trace the highest index first
        n_left = tensor.ndim // 2
        tensor = np.trace(tensor, axis1=q, axis2=q + n_left)
    dim = 2**len(keep)
    return tensor.reshape(dim, dim)


def entropy_and_schmidt_rank(rho, n_qubits, k):
    """Same quantities as MIMIQ's built-in ones, from rho: cut A = qubits 0..k-1."""
    if k == 0:
        return 0.0, 1
    rho_A = reduced_density_matrix(rho, n_qubits, list(range(k)))
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    schmidt_rank = len(eigenvalues)                         # pure state: rank of rho_A = Schmidt rank
    return entropy, schmidt_rank


# ------------------------------------------------------------------------------------------
# Part 3: GME with the thesis code
# ------------------------------------------------------------------------------------------
def main():
    builtin_entanglement(N_QUBITS)

    rho = density_matrix_from_paulis(N_QUBITS)
    purity = np.trace(rho @ rho).real
    print(f"  purity Tr(rho^2) = {purity:.6f}   (1 = pure state)")
    print("  same measures, computed from rho:")
    for k in range(N_QUBITS):
        entropy, rank = entropy_and_schmidt_rank(rho, N_QUBITS, k)
        print(f"    cut k = {k}:  entropy = {entropy:.6f}   Schmidt rank = {rank}")
    print()

    # GME is defined for 3 or more parties. For 2 parties there is only one bipartition (A | B),
    # so the same formula gives the ordinary bipartite entanglement (the concurrence).
    print("Part 3: entanglement with the thesis code")

    c_gme = C_GME_pure_dm(rho, N_QUBITS)
    print(f"  4 qubits, pure state   : GME concurrence C_GME = {c_gme:.6f}   (GHZ: 1)")

    rho_3 = reduced_density_matrix(rho, N_QUBITS, [0, 1, 2])
    purity_3 = np.trace(rho_3 @ rho_3).real
    gcg_3 = convex_roof_gcg(rho_3, 3, n_restarts=5, verbose=False)
    print(f"  3 qubits (0, 1, 2), mixed: purity = {purity_3:.6f}, "
          f"GME concurrence (GCG convex roof) = {gcg_3['C_gcg']:.2e}")
    print("      (|000><000| + |111><111|)/2: classically correlated, no GME")

    rho_2 = reduced_density_matrix(rho, N_QUBITS, [0, 1])
    purity_2 = np.trace(rho_2 @ rho_2).real
    gcg_2 = convex_roof_gcg(rho_2, 2, n_restarts=5, verbose=False)
    print(f"  2 qubits (0, 1), mixed : purity = {purity_2:.6f}, "
          f"bipartite entanglement (concurrence, convex roof) = {gcg_2['C_gcg']:.2e}")
    print("      (|00><00| + |11><11|)/2: not entangled. With 2 parties this is bipartite, not GME.")


if __name__ == "__main__":
    main()
