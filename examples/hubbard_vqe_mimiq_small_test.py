# Small end-to-end test of the planned VQE procedure for the Hubbard ring on MIMIQ (Exaqt):
#
#   |0...0> -> X gates (Hartree-Fock in the U = 0 orbitals) -> spin UCCSD(theta) (mimiq-vqe)
#           -> rotation to sites (Givens, tests/test_rotation_to_site.py) -> <H_site> -> COBYLA
#
# Checks: (1) theta = 0 gives E_HF = -8t + 1.5U exactly; (2) a short COBYLA run lowers the energy
# towards E0, with the fidelity against exact diagonalisation.
# Result on 13 Oct (L = 6, U = 1, 400 steps): E(theta=0) - E_HF = 6.6e-14; E - E0 = 1.7e-2,
# fidelity 0.996, still decreasing (not converged at 400 steps), about 0.36 s per evaluation.
#
# Workarounds used here, NOT yet fixed in tests/test_rotation_to_site.py (push_givens_rotations, line 77):
#   1. mc.Rpauli is a typo: the class is mc.RPauli
#   2. the angle 2*coeff is complex (zero imaginary part): MIMIQ needs 2*coeff.real
#   3. Exaqt cannot run RPauli gates: circuit.decompose() before exaqt.evolve
# The energy is computed from the state vector here (test only); in the real runs it is measured
# from the Pauli strings of H_site.
#
# Run:  python examples/hubbard_vqe_mimiq_small_test.py [U] [maxiter]      (default U = 1, 400 steps)

import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/anjum/PhDWork/Strasbourg_Secondment/mimiq_gme/tests")
sys.path.insert(0, "/home/anjum/PhDWork/Strasbourg_Secondment/mimiq_vqe")
sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

import exaqt
import mimiqcircuits as mc
from openfermion import jordan_wigner, get_sparse_operator
from openfermion.linalg import jw_get_ground_state_at_particle_number

mc.Rpauli = lambda pauli, angle: mc.RPauli(pauli, float(np.real(angle)))   # workarounds 1 and 2

from test_hubbard_orbitals import hopping_orbitals, build_hubbard_hamiltonian
from test_rotation_to_site import orbital_to_sites_circuit
from src.mimiq_ansatz import build_uccsd_spin, spin_excitations

L, N_QUBITS, N_ELECTRONS, T = 6, 12, 6, 1.0


def main(U, maxiter):
    H = get_sparse_operator(jordan_wigner(build_hubbard_hamiltonian(L, T, U, mode="uniform", V=0.0)),
                            n_qubits=N_QUBITS)
    E0, psi0 = jw_get_ground_state_at_particle_number(H, N_ELECTRONS)
    _, C = hopping_orbitals(L, T)
    rotation = orbital_to_sites_circuit(C)
    n_params = len(spin_excitations(N_QUBITS, N_ELECTRONS))

    def state(theta):
        circuit = build_uccsd_spin(N_QUBITS, N_ELECTRONS, theta)
        circuit.append(rotation)
        circuit = circuit.decompose()                                         # workaround 3
        st, _ = exaqt.evolve(exaqt.ExaqtState.zero(N_QUBITS), circuit)
        a = np.asarray(st.q.amplitudes())
        return a.reshape([2] * N_QUBITS).transpose(range(N_QUBITS - 1, -1, -1)).reshape(-1)  # qubit 0 = leftmost

    def energy(theta):
        psi = state(theta)
        return float(np.real(np.vdot(psi, H @ psi)))

    start = time.time()
    e_hf = energy(np.zeros(n_params))
    print(f"U = {U}: {n_params} parameters | E(theta=0) = {e_hf:.10f}, E_HF = {-8 * T + 1.5 * U:.10f}, "
          f"diff = {e_hf - (-8 * T + 1.5 * U):.1e} | E0 = {E0:.6f}")

    history = []
    def cost(theta):
        e = energy(theta); history.append(e); return e

    res = minimize(cost, np.zeros(n_params), method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2, "tol": 1e-10})
    fidelity = abs(np.vdot(psi0, state(res.x))) ** 2
    print(f"COBYLA {len(history)} evaluations ({time.time() - start:.0f} s): E = {res.fun:.6f}, "
          f"E - E0 = {res.fun - E0:.2e}, fidelity = {fidelity:.5f}")


if __name__ == "__main__":
    U = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    main(U, maxiter)
