"""
VQE for the Hubbard ring with the LUCJ ansatz (local unitary cluster Jastrow) on MIMIQ.

Same steps as core.py (UCCSD), with three differences:
  - qubit layout: ffsim puts all spin-up modes first, then all spin-down
        site k (counted from 0 in the code), spin up   -> qubit k
        site k,                          spin down -> qubit L + k
    (core.py uses the interleaved layout 2k + spin). The Hamiltonian is the same thesis Hamiltonian,
    only its modes are renumbered to this layout (still 42 Pauli strings at (1, 4, 0)).
  - ansatz: Hartree-Fock in the hopping orbitals, then the LUCJ operator
        |Psi_orb(theta)> = prod_k  U_k exp(i J_k) U_k^dagger |Phi_0>      (k = 1 ... n_reps)
    with U_k orbital rotations and J_k diagonal (Jastrow) operators; "local" couples spin-up/up
    pairs (p, p+1) and spin-up/down pairs (p, p). Exact gates, no Trotter step.
  - rotation to the sites: ffsim's exact orbital rotation with the matrix C of the hopping orbitals,
        a+_{p,s} -> sum_k C[k, p] c+_{k,s}         (the same R_C as in the report, Eq. 16)

At theta = 0 every LUCJ factor is the identity, so the circuit gives Hartree-Fock: E = E_HF.
"""

import time

import numpy as np
from openfermion import jordan_wigner, get_sparse_operator, reorder
from openfermion.linalg import jw_get_ground_state_at_particle_number
from openfermion.utils import up_then_down

import exaqt
import ffsim
import mimiq_qiskit
from qiskit import QuantumCircuit

from hubbard_vqe_mimiq.core import hubbard_hamiltonian, hopping_orbitals
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import vqe_until_converged
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from src.mimiq_lucj import lucj_pairs, lucj_num_parameters


def to_up_then_down(H_interleaved, L):
    """renumber the modes 2k + spin -> (k for spin up, L + k for spin down); same operator"""
    return reorder(H_interleaved, up_then_down, num_modes=2 * L)


def lucj_ansatz(C, n_reps=2, pairs_kind="local"):
    """ansatz(n_qubits, n_electrons, params) -> MIMIQ circuit:
    Hartree-Fock (orbitals) -> LUCJ(params) -> orbital rotation C back to the sites"""

    def ansatz(n_qubits, n_electrons, params):
        norb = n_qubits // 2
        nelec = (n_electrons // 2, n_electrons // 2)
        op = ffsim.UCJOpSpinBalanced.from_parameters(
            np.asarray(params, dtype=float), norb=norb, n_reps=n_reps,
            interaction_pairs=lucj_pairs(norb, pairs_kind), with_final_orbital_rotation=True)
        qc = QuantumCircuit(n_qubits)
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), range(n_qubits))
        qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), range(n_qubits))
        qc.append(ffsim.qiskit.OrbitalRotationJW(norb, C), range(n_qubits))      # orbitals -> sites
        return mimiq_qiskit.qiskit_to_mimiq(qc.decompose(reps=2))

    return ansatz


def state_vector(circuit, n_qubits):
    """state vector (OpenFermion order: qubit 0 = leftmost bit); simulator only, for the fidelity"""
    state, _ = exaqt.evolve(exaqt.ExaqtState.zero(n_qubits), circuit.decompose())
    psi = np.asarray(state.q.amplitudes())
    return psi.reshape([2] * n_qubits).transpose(range(n_qubits - 1, -1, -1)).reshape(-1)


def run_vqe_lucj(L=6, t=1.0, U=1.0, V=0.0, n_reps=2, pairs_kind="local", seed=12345, max_cycles=25,
                 chunk_maxiter=600, eps_E=1e-6, patience=3, progress_every=0):
    """LUCJ VQE for one point (t, U, V) at half filling, started from Hartree-Fock (theta = 0)."""
    from hubbard_vqe_mimiq.core import with_progress
    n_qubits, n_electrons = 2 * L, L
    H = to_up_then_down(hubbard_hamiltonian(L, t, U, V), L)
    E0, psi0 = jw_get_ground_state_at_particle_number(get_sparse_operator(H, n_qubits=n_qubits), n_electrons)
    H_mimiq, constant, stats = openfermion_to_mimiq_hamiltonian(jordan_wigner(H))
    _, C = hopping_orbitals(L, t)
    ansatz = lucj_ansatz(C, n_reps, pairs_kind)
    n_params = lucj_num_parameters(L, n_reps, pairs_kind)

    energy_fn, _ = make_energy_fn(H_mimiq, constant=0.0, n_qubits=n_qubits, n_electrons=n_electrons,
                                  ansatz=ansatz, target="exaqt-cpu")
    theta0 = np.zeros(n_params)
    E_start = energy_fn(theta0) + constant
    if progress_every:
        print(f"LUCJ ({pairs_kind}, n_reps = {n_reps}): L = {L}, t = {t}, U = {U}, V = {V} (U/t = {U / t:g}): "
              f"{n_qubits} qubits, {n_params} parameters, {stats['n_terms_kept']} Pauli strings | "
              f"E0 (ED) = {E0:.6f}, E_HF = {E_start:.6f}", flush=True)
        energy_fn = with_progress(energy_fn, constant, E0, progress_every)

    start = time.time()
    out = vqe_until_converged(energy_fn, theta0, np.random.default_rng(seed), eps_E=eps_E, patience=patience,
                              max_cycles=max_cycles, chunk_maxiter=chunk_maxiter, verbose_cycles=bool(progress_every))
    runtime = time.time() - start

    E_vqe = out["E_nc_opt"] + constant
    psi = state_vector(ansatz(n_qubits, n_electrons, out["theta_opt"]), n_qubits)
    psi_hf = state_vector(ansatz(n_qubits, n_electrons, theta0), n_qubits)
    return {
        "ansatz": f"LUCJ {pairs_kind} r{n_reps}", "L": L, "t": t, "U": U, "V": V, "U_over_t": U / t,
        "n_qubits": n_qubits, "n_parameters": n_params, "n_pauli_strings": stats["n_terms_kept"],
        "E0": float(E0), "E_HF": E_start, "E_VQE": E_vqe, "dE": E_vqe - E0,
        "rel_error": abs(E_vqe - E0) / abs(E0) if E0 != 0 else None,
        "correlation_recovered": (E_start - E_vqe) / (E_start - E0) if E_start != E0 else None,
        "fidelity": float(abs(np.vdot(psi0, psi)) ** 2), "fidelity_HF": float(abs(np.vdot(psi0, psi_hf)) ** 2),
        "n_evaluations": out["nfev"], "cycles": out["cycles"], "converged": out["converged"],
        "runtime_s": runtime, "theta_opt": out["theta_opt"],
        "best_energy_per_cycle": [e + constant for e in out["best_energy_per_cycle"]],
    }
