"""
VQE for the Hubbard ring on MIMIQ: all the steps in one place.

The equations are in docs/vqe_hubbard_mimiq.tex; the equation numbers below refer to that note.

One VQE run, in words:
  1. build the Hubbard Hamiltonian on the sites (thesis code), Eq. (1)
  2. map it to qubits with Jordan-Wigner, qubit 2j + spin for site j, Eqs. (2)-(6)
  3. find the hopping (U = 0) orbitals, Eq. (8)
  4. circuit: Hartree-Fock in these orbitals (X gates) -> UCCSD(theta) -> rotation back to the
     sites, Eq. (17)
  5. measure the energy <H> on MIMIQ, Eq. (22)
  6. let COBYLA change theta until the energy stops going down (mimiq-vqe driver)
  7. compare with exact diagonalisation (ED): energy error and fidelity

Qubit order: MIMIQ keeps qubit 0 as the lowest bit, OpenFermion as the highest bit. Gates and
Pauli strings use qubit numbers and are the same in both; only state vectors are reordered.
"""

import os
import sys
import time

import numpy as np

# the thesis code (hubbard_gme) and mimiq-vqe; set HUBBARD_GME_SRC / MIMIQ_VQE_DIR on another machine
sys.path.insert(0, os.environ.get("HUBBARD_GME_SRC", "/home/anjum/PhDWork/hubbard_gme/src"))
sys.path.insert(0, os.environ.get("MIMIQ_VQE_DIR", "/home/anjum/PhDWork/Strasbourg_Secondment/mimiq_vqe"))

import exaqt
import mimiqcircuits as mc
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, reorder
from openfermion.linalg import jw_get_ground_state_at_particle_number

from hubbard_gme.hamiltonian import build_hubbard_hamiltonian
from src.mimiq_ansatz import build_uccsd_spin, spin_excitations
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import vqe_until_converged
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian

# every energy in this package is measured on the MIMIQ Exaqt simulator (a CUDA-Q version would be hubbard_vqe_cudaq)
BACKEND = "mimiq-exaqt"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hubbard_vqe_mimiq")


def result_name(L, t, U, V, ansatz):
    """file name of one result: model, parameters, ansatz and backend,
    e.g. result_name(6, 1, 10, 0, "hva-p8") -> hubbard_L6_t1_U10_V0_hva-p8_mimiq-exaqt"""
    return f"hubbard_L{L}_t{t:g}_U{U:g}_V{V:g}_{ansatz}_{BACKEND}"


# ------------------------------------------------------------------------------------------
# 1-2. Hamiltonian and exact ground state
# ------------------------------------------------------------------------------------------
def hubbard_hamiltonian(L, t, U, V=0.0):
    """Hubbard ring on the sites, exactly as in the thesis (Eq. 1). Returns a FermionOperator."""
    return build_hubbard_hamiltonian(L, t, U, mode="uniform", V=V)


# ------------------------------------------------------------------------------------------
# Qubit order. "interleaved": site k (from 0), spin s -> qubit 2k + s (thesis, UCCSD, default).
# "blocked": all spin up first, then all spin down -> qubit s L + k. In the blocked order two
# neighbouring sites with the same spin are neighbouring qubits, so a hop needs no Z string
# (except across the ring bond): shorter gates on hardware.
# ------------------------------------------------------------------------------------------
LAYOUTS = ("interleaved", "blocked")


def mode_index(k, spin, L, layout="interleaved"):
    """qubit of site (or orbital) k, counted from 0, with spin 0 (up) or 1 (down)"""
    if layout == "interleaved":
        return 2 * k + spin
    if layout == "blocked":
        return spin * L + k
    raise ValueError(f"unknown layout {layout!r}; use one of {LAYOUTS}")


def to_layout(op, L, layout="interleaved"):
    """the same fermion operator with its modes renumbered from the interleaved to the given order"""
    if layout == "interleaved":
        return op
    mode_index(0, 0, L, layout)                            # checks the name
    return reorder(op, lambda m, n: mode_index(m // 2, m % 2, n // 2, layout), num_modes=2 * L)


def exact_ground_state(H, n_qubits, n_electrons):
    """Exact diagonalisation (ED) at fixed electron number: E0 and the state (OpenFermion order)."""
    E0, psi0 = jw_get_ground_state_at_particle_number(get_sparse_operator(H, n_qubits=n_qubits), n_electrons)
    return float(E0), np.asarray(psi0)


def mimiq_hamiltonian(H):
    """Jordan-Wigner Pauli strings as a MIMIQ Hamiltonian (Eq. 6); the identity part is returned apart."""
    H_mimiq, constant, stats = openfermion_to_mimiq_hamiltonian(jordan_wigner(H))
    return H_mimiq, constant, stats["n_terms_kept"]


# ------------------------------------------------------------------------------------------
# 3. Hopping (U = 0) orbitals and Hartree-Fock
# ------------------------------------------------------------------------------------------
def hopping_orbitals(L, t):
    """Eigenvectors of the hopping matrix of the ring, lowest energy first (Eq. 8).
    Returns the energies e_p and C, with c_{j,s} = sum_p C[j, p] a_{p,s}."""
    h = np.zeros((L, L))
    for j in range(L):
        k = (j + 1) % L
        h[j, k] = h[k, j] = -t
    energies, C = np.linalg.eigh(h)
    return energies, C


def hartree_fock_energy(L, t, U, n_electrons):
    """E_HF = 2 * (sum of the filled orbital energies) + U L / 4, at half filling and V = 0 (Eq. 10)."""
    energies, _ = hopping_orbitals(L, t)
    return 2 * float(np.sum(energies[: n_electrons // 2])) + U * L / 4


# ------------------------------------------------------------------------------------------
# 4. Rotation from the orbitals back to the sites (Eqs. 16-21)
# ------------------------------------------------------------------------------------------
def givens_rotations(C):
    """Split C into rotations of neighbouring orbitals (p, p+1): G_m ... G_1 C = D (Eq. 18).
    Returns the list of (p, angle) and the signs on the diagonal of D."""
    L = C.shape[0]
    R = C.copy()
    rotations = []
    for col in range(L):
        for row in range(L - 1, col, -1):
            a, b = R[row - 1, col], R[row, col]
            if abs(b) < 1e-12:
                continue                                   # already zero
            angle = np.arctan2(b, a)
            G = np.eye(L)
            G[row - 1, row - 1] = G[row, row] = np.cos(angle)
            G[row - 1, row] = np.sin(angle)
            G[row, row - 1] = -np.sin(angle)
            R = G @ R
            rotations.append((row - 1, angle))
    signs = np.round(np.diag(R)).astype(int)
    return rotations, signs


def push_givens(circuit, p, angle, L=None, layout="interleaved"):
    """Rotate orbitals p and p+1 by angle, for both spins: exp[angle (a+_{p+1} a_p - a+_p a_{p+1})].
    Jordan-Wigner gives two commuting Pauli strings per spin (Eq. 20): two exact RPauli gates
    (with a Z on the other spin in between in the interleaved order, none in the blocked order)."""
    for spin in (0, 1):
        i, j = mode_index(p, spin, L, layout), mode_index(p + 1, spin, L, layout)
        generator = FermionOperator(((j, 1), (i, 0)), angle) - FermionOperator(((i, 1), (j, 0)), angle)
        hermitian = jordan_wigner(1j * generator)          # exp(generator) = exp(-i * hermitian)
        for term, coeff in hermitian.terms.items():
            qubits = [q for q, _ in term]
            paulis = "".join(letter for _, letter in term)
            # RPauli(P, phi) = exp(-i phi P / 2), so phi = 2 * coeff (a real number)
            circuit.push(mc.RPauli(mc.PauliString(paulis), 2 * float(np.real(coeff))), *qubits)


def rotation_to_sites(C, layout="interleaved"):
    """Circuit R_C that turns a state written in the orbitals into the same state on the sites (Eq. 16).
    C = G_1^T ... G_m^T D: the sign matrix D acts first, then the rotations in reverse order."""
    L = C.shape[0]
    rotations, signs = givens_rotations(C)
    circuit = mc.Circuit()
    for p, sign in enumerate(signs):
        if sign < 0:                                       # D_pp = -1: change the sign of orbital p
            circuit.push(mc.GateZ(), mode_index(p, 0, L, layout))
            circuit.push(mc.GateZ(), mode_index(p, 1, L, layout))
    for p, angle in reversed(rotations):
        push_givens(circuit, p, angle, L, layout)
    return circuit


# ------------------------------------------------------------------------------------------
# 4. The full circuit: Hartree-Fock -> UCCSD(theta) -> rotation to sites (Eq. 17)
# ------------------------------------------------------------------------------------------
def number_of_parameters(n_qubits, n_electrons):
    """Spin-conserving singles and doubles (Eq. 12): 117 for 6 electrons on 6 sites."""
    return len(spin_excitations(n_qubits, n_electrons))


def make_ansatz(C):
    """Return a function ansatz(n_qubits, n_electrons, params) -> circuit, as make_energy_fn expects.
    UCCSD (mimiq-vqe, first-order Trotter, includes the Hartree-Fock X gates) followed by the rotation."""
    rotation = rotation_to_sites(C)

    def ansatz(n_qubits, n_electrons, params):
        circuit = build_uccsd_spin(n_qubits, n_electrons, params)
        circuit.append(rotation)
        return circuit

    return ansatz


def state_vector(circuit, n_qubits):
    """State vector of a circuit on Exaqt, in OpenFermion order (qubit 0 = leftmost bit).
    Only possible on a simulator; used for the fidelity, never for the energy."""
    state, _ = exaqt.evolve(exaqt.ExaqtState.zero(n_qubits), circuit.decompose())
    psi = np.asarray(state.q.amplitudes())
    return psi.reshape([2] * n_qubits).transpose(range(n_qubits - 1, -1, -1)).reshape(-1)


def block_entropy(psi, n_qubits, n_first):
    """von Neumann entropy (in bits) of the first n_first qubits of a pure state (OpenFermion order),
    from its Schmidt values. In the interleaved order the sites 1 ... m are the first 2m qubits; their
    Jordan-Wigner strings involve no later qubit, so this is the entropy of the m sites."""
    s = np.linalg.svd(np.asarray(psi).reshape(2**n_first, 2 ** (n_qubits - n_first)), compute_uv=False)
    w = s**2
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log2(w)))


def site_entropy(psi, L, n_sites):
    """entanglement entropy (bits) of the sites 1 ... n_sites with the rest of the ring (interleaved order)"""
    return block_entropy(psi, 2 * L, 2 * n_sites)


def load_state(psi, n_qubits):
    """Load an OpenFermion-order state vector into an Exaqt state (bits reversed for MIMIQ)."""
    index = np.arange(2**n_qubits)
    reversed_index = np.array([int(format(i, f"0{n_qubits}b")[::-1], 2) for i in index])
    amplitudes = np.zeros(2**n_qubits, dtype=np.complex128)
    amplitudes[reversed_index] = psi
    state = exaqt.ExaqtState.zero(n_qubits)
    state.q = exaqt.ExaqtSV.from_amplitudes(n_qubits, np.ascontiguousarray(amplitudes))
    return state


# ------------------------------------------------------------------------------------------
# 5-7. One VQE run and the comparison with ED
# ------------------------------------------------------------------------------------------
def with_progress(energy_fn, constant, E0, every):
    """wrap energy_fn so that every `every` evaluations it prints the energy, the best so far and its
    distance to the exact E0 (the driver still sees the same function and its timing lists)"""
    state = {"n": 0, "best": np.inf, "start": time.time()}

    def wrapped(theta):
        e = energy_fn(theta)
        state["n"] += 1
        state["best"] = min(state["best"], e)
        if every and state["n"] % every == 0:
            best = state["best"] + constant
            print(f"  eval {state['n']:6d} | {time.time() - state['start']:7.0f} s | E = {e + constant:11.6f} | "
                  f"best = {best:11.6f} | best - E0 = {best - E0:.3e}", flush=True)
        return e

    wrapped.quantum_times = energy_fn.quantum_times
    wrapped.exaqt_timings = energy_fn.exaqt_timings
    return wrapped


def run_vqe(L=6, t=1.0, U=1.0, V=0.0, seed=12345, max_cycles=25, chunk_maxiter=600, eps_E=1e-6,
            patience=3, verbose=False, progress_every=0):
    """
    VQE for one point (t, U, V) at half filling, started from Hartree-Fock (theta = 0).
    The energy is measured on MIMIQ (Exaqt) as <H>; the state vector is used only at the end,
    for the fidelity with the exact ground state. Returns a dict with all the numbers.
    """
    n_qubits, n_electrons = 2 * L, L
    H = hubbard_hamiltonian(L, t, U, V)
    E0, psi0 = exact_ground_state(H, n_qubits, n_electrons)
    H_mimiq, constant, n_strings = mimiq_hamiltonian(H)
    _, C = hopping_orbitals(L, t)
    ansatz = make_ansatz(C)
    n_params = number_of_parameters(n_qubits, n_electrons)

    # energy on MIMIQ without the constant (the driver works with E - constant)
    energy_fn, _ = make_energy_fn(H_mimiq, constant=0.0, n_qubits=n_qubits, n_electrons=n_electrons,
                                  ansatz=ansatz, target="exaqt-cpu")
    theta0 = np.zeros(n_params)
    E_start = energy_fn(theta0) + constant
    if progress_every:
        print(f"L = {L}, t = {t}, U = {U}, V = {V} (U/t = {U / t:g}): {n_qubits} qubits, {n_params} parameters, "
              f"{n_strings} Pauli strings | E0 (ED) = {E0:.6f}, E_HF = {E_start:.6f}", flush=True)
        energy_fn = with_progress(energy_fn, constant, E0, progress_every)

    start = time.time()
    out = vqe_until_converged(energy_fn, theta0, np.random.default_rng(seed), eps_E=eps_E, patience=patience,
                              max_cycles=max_cycles, chunk_maxiter=chunk_maxiter, verbose_cycles=verbose)
    runtime = time.time() - start

    E_vqe = out["E_nc_opt"] + constant
    psi = state_vector(ansatz(n_qubits, n_electrons, out["theta_opt"]), n_qubits)
    fidelity = float(abs(np.vdot(psi0, psi)) ** 2)
    psi_hf = state_vector(ansatz(n_qubits, n_electrons, theta0), n_qubits)

    return {
        "L": L, "t": t, "U": U, "V": V, "U_over_t": U / t,
        "n_qubits": n_qubits, "n_parameters": n_params, "n_pauli_strings": n_strings,
        "E0": E0, "E_HF": E_start, "E_VQE": E_vqe,
        "dE": E_vqe - E0, "rel_error": abs(E_vqe - E0) / abs(E0) if E0 != 0 else None,
        "correlation_recovered": (E_start - E_vqe) / (E_start - E0) if E_start != E0 else None,
        "fidelity": fidelity, "fidelity_HF": float(abs(np.vdot(psi0, psi_hf)) ** 2),
        "n_evaluations": out["nfev"], "cycles": out["cycles"], "converged": out["converged"],
        "runtime_s": runtime, "theta_opt": out["theta_opt"],
        "best_energy_per_cycle": [e + constant for e in out["best_energy_per_cycle"]],
    }
