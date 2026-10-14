# GHZ states of 6, 8 and 10 qubits: cut measures (MIMIQ cloud) and the GME concurrence of the
# full state, from measurements (Pauli strings) and from the state vector.
#
# Run:  python examples/ghz_sizes.py
# The cloud part needs today's login token in ~/.mimiq/qperfect_token.json (else it is skipped).

import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

import exaqt
import mimiqcircuits as mc
from hubbard_gme.convex_roof import C_GME_pure_dm

from ghz_entanglement import ghz_circuit, pauli_matrix

SIZES = [6, 8, 10]
MAX_QUBITS_FROM_MEASUREMENTS = 8          # 4^10 ~ 10^6 Pauli strings is too heavy for the laptop
TOKEN = "/home/anjum/.mimiq/qperfect_token.json"


def cut_measures_on_cloud(n_qubits, connection):
    """Entropy, Schmidt rank and bond dimension at every cut, on the MIMIQ cloud (MPS)."""
    circuit = ghz_circuit(n_qubits)
    for k in range(n_qubits):
        circuit.push(mc.VonNeumannEntropy(), k, k)
        circuit.push(mc.SchmidtRank(), k, n_qubits + k)
        circuit.push(mc.BondDim(), k, 2 * n_qubits + k)
    job = connection.submit(circuit, algorithm="mps", nsamples=1)
    z = np.real(connection.get_result(job).zstates[0])
    entropy = np.round(z[:n_qubits], 6)
    schmidt = np.round(z[n_qubits:2 * n_qubits]).astype(int)
    bonddim = np.round(z[2 * n_qubits:]).astype(int)
    return entropy, schmidt, bonddim


def density_matrix_from_measurements(n_qubits):
    """rho = (1/2^n) sum_P <P> P, with every <P> measured on MIMIQ (Exaqt), no state read-out."""
    strings = ["".join(s) for s in product("IXYZ", repeat=n_qubits)]
    measured = [s for s in strings if s != "I" * n_qubits]

    circuit = ghz_circuit(n_qubits)
    for z, letters in enumerate(measured):
        circuit.push(mc.ExpectationValue(mc.PauliString(letters)), *range(n_qubits), z)
    mimiq_state = exaqt.ExaqtState.zero(n_qubits, num_zvars=circuit.num_zvars())
    mimiq_state, _ = exaqt.evolve(mimiq_state, circuit)

    rho = np.eye(2**n_qubits, dtype=complex)                    # the identity term, <I...I> = 1
    for z, letters in enumerate(measured):
        value = mimiq_state.z[z].real
        if abs(value) > 1e-12:                                  # skip the zeros (most of them for GHZ)
            rho += value * pauli_matrix(letters)
    return rho / 2**n_qubits, len(measured)


def density_matrix_from_state_vector(n_qubits):
    """rho = |psi><psi| from the simulator's state vector (only possible on a simulator)."""
    mimiq_state, _ = exaqt.evolve(exaqt.ExaqtState.zero(n_qubits), ghz_circuit(n_qubits))
    psi = np.asarray(mimiq_state.q.amplitudes())
    psi = psi.reshape([2] * n_qubits).transpose(range(n_qubits - 1, -1, -1)).reshape(-1)  # qubit 0 = leftmost
    return np.outer(psi, psi.conj())


def main():
    connection = None
    try:
        connection = mc.MimiqConnection(mc.QPERFECT_CLOUD)
        connection.loadtoken(TOKEN)
    except Exception as error:
        print(f"cloud login not available ({type(error).__name__}); cut measures skipped\n")
        connection = None

    for n in SIZES:
        print(f"===== GHZ, N = {n} qubits ({2**(n - 1) - 1} bipartitions)")

        if connection is not None:
            try:
                entropy, schmidt, bonddim = cut_measures_on_cloud(n, connection)
                print(f"  cut measures (cloud, MPS), k = 0..{n - 1}:")
                print(f"    entropy      : {entropy.tolist()}")
                print(f"    Schmidt rank : {schmidt.tolist()}")
                print(f"    bond dim     : {bonddim.tolist()}")
            except Exception as error:
                print(f"  cut measures skipped: {type(error).__name__}: {error}")

        if n <= MAX_QUBITS_FROM_MEASUREMENTS:
            start = time.time()
            rho, n_strings = density_matrix_from_measurements(n)
            purity = np.trace(rho @ rho).real
            print(f"  from measurements : {n_strings} Pauli strings, purity = {purity:.6f}, "
                  f"C_GME = {C_GME_pure_dm(rho, n):.6f}   ({time.time() - start:.1f} s)")
        else:
            print(f"  from measurements : skipped ({4**n - 1} Pauli strings)")

        start = time.time()
        rho = density_matrix_from_state_vector(n)
        print(f"  from state vector : C_GME = {C_GME_pure_dm(rho, n):.6f}   ({time.time() - start:.1f} s)")
        print()


if __name__ == "__main__":
    main()
