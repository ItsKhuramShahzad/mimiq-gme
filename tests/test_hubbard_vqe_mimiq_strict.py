# Strict tests for hubbard_vqe_mimiq: Hamiltonian, qubit conversion, bit order, orbitals and rotation,
# the three ansatze (UCCSD, LUCJ, HVA), the fidelity, the VQE runs and the file names.
# Every check is against an independent answer (a Hamiltonian written by hand, exact diagonalisation,
# an exact matrix exponential, a known limit, or a value recorded in a finished run), mostly to 1e-10.
#
# Run:  python -m pytest -q tests/test_hubbard_vqe_mimiq_strict.py          (a few minutes)

import itertools
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exaqt
import mimiqcircuits as mc
from openfermion import (FermionOperator, QubitOperator, commutator, get_sparse_operator, hermitian_conjugated,
                         jordan_wigner, normal_ordered, number_operator, s_squared_operator, sz_operator)
from openfermion.linalg import jw_number_restrict_operator

from hubbard_vqe_mimiq.core import (BACKEND, result_name, hubbard_hamiltonian, exact_ground_state, mimiq_hamiltonian,
                                    hopping_orbitals, hartree_fock_energy, rotation_to_sites, number_of_parameters,
                                    make_ansatz, state_vector, load_state, run_vqe)
from hubbard_vqe_mimiq.hva import (onsite_operator, neighbour_operator, hopping_operator, pauli_terms,
                                   layer_operators, push_exp, hva_ansatz, run_vqe_hva)
from hubbard_vqe_mimiq.lucj import to_up_then_down, lucj_ansatz, run_vqe_lucj
from src.mimiq_backend import make_energy_fn
from src.mimiq_lucj import lucj_num_parameters

L, NQ, NE = 6, 12, 6
TOL = 1e-10
POINTS = [(1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (1.0, 10.0, 0.0), (0.05, 10.0, 0.0), (1.0, 10.0, 5.0)]   # (t, U, V)
RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------------------------------
# helpers (independent of the package)
# ------------------------------------------------------------------------------------------
def n_op(mode):
    return FermionOperator(((mode, 1), (mode, 0)))


def hand_written_hamiltonian(L, t, U, V):
    """H = -t sum_{k,s} (c+_{k,s} c_{k+1,s} + h.c.) + U sum_k n_{k,up} n_{k,dn} + V sum_k n_k n_{k+1}, ring,
    mode 2k + s (s = 0 up, 1 down): written here from the formula, not from the thesis code"""
    H = FermionOperator()
    for k in range(L):
        j = (k + 1) % L
        for s in (0, 1):
            H += FermionOperator(((2 * k + s, 1), (2 * j + s, 0)), -t) + FermionOperator(((2 * j + s, 1), (2 * k + s, 0)), -t)
        H += U * n_op(2 * k) * n_op(2 * k + 1)
        H += V * (n_op(2 * k) + n_op(2 * k + 1)) * (n_op(2 * j) + n_op(2 * j + 1))
    return H


def same_operator(A, B, tol=1e-12):
    """two FermionOperators are equal after normal ordering"""
    D = normal_ordered(A - B)
    D.compress(abs_tol=tol)
    return len(D.terms) == 0


def sparse(op, n=NQ):
    return get_sparse_operator(op, n_qubits=n)


def expectation(op_sparse, psi):
    return float(np.real(np.vdot(psi, op_sparse @ psi)))


def sector_weight(psi, op_sparse, value):
    """weight of psi in the eigenspace op = value of a diagonal operator (number, Sz)"""
    diag = op_sparse.diagonal().real
    return float(np.sum(np.abs(psi[np.abs(diag - value) < 1e-9]) ** 2))


def lowest_two(H):
    """E0 and E1 in the N = 6 sector"""
    w = np.sort(eigsh(jw_number_restrict_operator(sparse(H), NE, NQ), k=2, which="SA")[0])
    return w[0], w[1]


def evolve_state(psi, circuit):
    """apply a MIMIQ circuit to an OpenFermion-order state on Exaqt, return the OpenFermion-order result"""
    state, _ = exaqt.evolve(load_state(psi, NQ), circuit.decompose())
    out = np.asarray(state.q.amplitudes())
    return out.reshape([2] * NQ).transpose(range(NQ - 1, -1, -1)).reshape(-1)


def random_state(n=NQ):
    psi = RNG.normal(size=2**n) + 1j * RNG.normal(size=2**n)
    return psi / np.linalg.norm(psi)


def fidelity(a, b):
    return float(abs(np.vdot(a, b)) ** 2)


def mimiq_energy(H, ansatz_fn, theta):
    """<H> measured on MIMIQ (Exaqt) for ansatz_fn(theta), constant added back"""
    H_mimiq, constant, _ = mimiq_hamiltonian(H)
    energy_fn, _ = make_energy_fn(H_mimiq, 0.0, NQ, NE, ansatz=ansatz_fn)
    return energy_fn(theta) + constant


# ==========================================================================================
# 1. Hamiltonian
# ==========================================================================================
@pytest.mark.parametrize("t, U, V", POINTS)
def test_hamiltonian_equals_hand_written_formula(t, U, V):
    """the thesis builder gives exactly the Hubbard ring of the formula (signs, ring bond, V as n_k n_{k+1})"""
    assert same_operator(hubbard_hamiltonian(L, t, U, V), hand_written_hamiltonian(L, t, U, V))


@pytest.mark.parametrize("t, U, V", POINTS)
def test_hamiltonian_is_hermitian(t, U, V):
    H = hubbard_hamiltonian(L, t, U, V)
    assert same_operator(H, hermitian_conjugated(H))
    Hs = sparse(H)
    assert sp.linalg.norm(Hs - Hs.getH()) < 1e-12


@pytest.mark.parametrize("t, U, V", POINTS)
def test_hamiltonian_conserves_number_sz_and_s2(t, U, V):
    """[H, N] = [H, Sz] = [H, S^2] = 0 (mode 2k + s is OpenFermion's up/down convention)"""
    Hs = sparse(hubbard_hamiltonian(L, t, U, V))
    for op in (number_operator(NQ), sz_operator(L), s_squared_operator(L)):
        O = sparse(op)
        assert sp.linalg.norm(Hs @ O - O @ Hs) < 1e-10


@pytest.mark.parametrize("t, U, V", POINTS)
def test_hamiltonian_is_translation_invariant(t, U, V):
    """shifting every site by one (k -> k + 1 mod L) leaves H unchanged: the ring has no ends"""
    H = hubbard_hamiltonian(L, t, U, V)
    shifted = FermionOperator()
    for term, coeff in H.terms.items():
        shifted += FermionOperator(tuple(((m + 2) % (2 * L), a) for m, a in term), coeff)
    assert same_operator(H, shifted)


def test_hamiltonian_scaling_with_t():
    """H(t, U, V) = t H(1, U/t, V/t): equal energies after dividing by t"""
    E_a, psi_a = exact_ground_state(hubbard_hamiltonian(L, 0.05, 10.0, 0.0), NQ, NE)
    E_b, psi_b = exact_ground_state(hubbard_hamiltonian(L, 1.0, 200.0, 0.0), NQ, NE)
    assert abs(E_a / 0.05 - E_b) < 1e-8
    assert fidelity(psi_a, psi_b) > 1 - 1e-8


def test_limit_U_zero_is_free_fermions():
    """U = V = 0: E0 = 2 (e_1 + e_2 + e_3) = 2 (-2 - 1 - 1) t = -8t"""
    for t in (1.0, 0.3):
        assert abs(exact_ground_state(hubbard_hamiltonian(L, t, 0.0), NQ, NE)[0] + 8 * t) < TOL


def test_limit_t_zero_is_atomic():
    """t = 0, V = 0: one electron per site costs nothing, E0 = 0 exactly; double occupancy costs U"""
    H = hubbard_hamiltonian(L, 0.0, 4.0)
    w = np.sort(np.linalg.eigvalsh(jw_number_restrict_operator(sparse(H), NE, NQ).toarray()))
    assert abs(w[0]) < TOL
    assert abs(w[2**L] - 4.0) < TOL                 # 2^6 spin states at E = 0, then one doublon at E = U
    assert abs(w[2**L - 1]) < TOL


def test_limit_large_U_is_heisenberg_ring():
    """U >> t: E0 = J (E_Heisenberg - L/4) + O(t^4/U^3), J = 4 t^2 / U, Heisenberg ring sum_k S_k . S_{k+1}"""
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2

    def site(op, k):
        return np.kron(np.kron(np.eye(2**k), op), np.eye(2 ** (L - k - 1)))

    Hh = sum(site(s, k) @ site(s, (k + 1) % L) for k in range(L) for s in (sx, sy, sz))
    E_heis = np.linalg.eigvalsh(Hh)[0]
    assert abs(E_heis - (-2.8027756377319946)) < 1e-10      # known 6-site ring value
    U = 200.0
    E0 = exact_ground_state(hubbard_hamiltonian(L, 1.0, U), NQ, NE)[0]
    assert abs(E0 / ((4 / U) * (E_heis - L / 4)) - 1) < 2e-3


def test_ground_state_is_a_singlet():
    """V = 0 half-filled ring: the ground state has S^2 = 0 and Sz = 0"""
    S2, SZ = sparse(s_squared_operator(L)), sparse(sz_operator(L))
    for U in (1.0, 4.0, 10.0):
        _, psi = exact_ground_state(hubbard_hamiltonian(L, 1.0, U), NQ, NE)
        assert abs(expectation(S2, psi)) < 1e-9
        assert abs(expectation(SZ, psi)) < 1e-9


@pytest.mark.parametrize("t, U, V", POINTS)
def test_exact_ground_state_is_an_eigenvector(t, U, V):
    """ED: H psi0 = E0 psi0, |psi0| = 1, 6 electrons, E0 equals the dense diagonalisation of the N = 6 block"""
    H = hubbard_hamiltonian(L, t, U, V)
    E0, psi0 = exact_ground_state(H, NQ, NE)
    Hs = sparse(H)
    assert abs(np.linalg.norm(psi0) - 1) < 1e-12
    assert np.linalg.norm(Hs @ psi0 - E0 * psi0) < 1e-9
    assert sector_weight(psi0, sparse(number_operator(NQ)), NE) > 1 - 1e-12
    dense = np.linalg.eigvalsh(jw_number_restrict_operator(Hs, NE, NQ).toarray())[0]
    assert abs(E0 - dense) < 1e-9


@pytest.mark.parametrize("U, E0, gap", [(1.0, -6.60115829, 1.65624142), (10.0, -1.66436273, 0.27386619)])
def test_recorded_energies_and_gaps(U, E0, gap):
    """values used in the reports (regression): E0 and the gap E1 - E0, t = 1, V = 0"""
    e0, e1 = lowest_two(hubbard_hamiltonian(L, 1.0, U))
    assert abs(e0 - E0) < 1e-8
    assert abs((e1 - e0) - gap) < 1e-8


def test_uniform_epsilon_only_shifts_the_energy():
    """the paper's eps sum_k n_k: at fixed N = 6 it shifts every energy by 6 eps, the state is unchanged"""
    eps = 0.37
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    H_eps = H + eps * sum((n_op(m) for m in range(NQ)), FermionOperator())
    E, psi = exact_ground_state(H, NQ, NE)
    E_eps, psi_eps = exact_ground_state(H_eps, NQ, NE)
    assert abs(E_eps - E - eps * NE) < TOL
    assert fidelity(psi, psi_eps) > 1 - TOL


# ==========================================================================================
# 2. Conversion to qubits (Jordan-Wigner) and to MIMIQ
# ==========================================================================================
def test_pauli_terms_by_hand():
    """Jordan-Wigner coefficients checked one by one (t = 1.3, U = 2.6, V = 0.7)"""
    t, U, V = 1.3, 2.6, 0.7
    Q = jordan_wigner(hubbard_hamiltonian(L, t, U, V))
    c = lambda *ops: Q.terms.get(tuple(ops), 0.0)
    # hopping site 0 -> 1, spin up: qubits 0 and 2 with Z on 1:  -t/2 (X0 Z1 X2 + Y0 Z1 Y2)
    assert abs(c((0, "X"), (1, "Z"), (2, "X")) + t / 2) < 1e-12
    assert abs(c((0, "Y"), (1, "Z"), (2, "Y")) + t / 2) < 1e-12
    # the ring bond 5 -> 0 (qubits 0 and 10) carries a Z string on qubits 1..9, magnitude t/2
    ring = [(0, "X")] + [(q, "Z") for q in range(1, 10)] + [(10, "X")]
    assert abs(abs(c(*ring)) - t / 2) < 1e-12
    # on-site U n_up n_dn = U/4 (1 - Z0 - Z1 + Z0 Z1);  V n_k n_{k+1} adds V/4 Z Z between neighbours
    assert abs(c((0, "Z"), (1, "Z")) - U / 4) < 1e-12
    assert abs(c((0, "Z"), (2, "Z")) - V / 4) < 1e-12
    assert all(abs(np.imag(v)) < 1e-14 for v in Q.terms.values())          # real coefficients only


@pytest.mark.parametrize("t, U, V, n", [(1.0, 4.0, 0.0, 42), (1.0, 10.0, 5.0, 66), (1.0, 0.0, 0.0, 24)])
def test_pauli_counts_and_constant(t, U, V, n):
    """42 strings (24 hopping + 18 on-site) at V = 0, 66 with V; constant = U L/4 + V (number of V pairs)"""
    H_mimiq, constant, n_strings = mimiq_hamiltonian(hubbard_hamiltonian(L, t, U, V))
    assert n_strings == n
    assert H_mimiq.num_qubits() == NQ
    assert abs(constant - (U * L / 4 + V * L)) < 1e-12        # V n_k n_{k+1}: 4 spin pairs x V/4 = V per bond


@pytest.mark.parametrize("t, U, V", [(1.0, 4.0, 0.0), (0.7, 3.0, 1.5)])
def test_mimiq_hamiltonian_on_a_random_circuit(t, U, V):
    """<H> on MIMIQ = <psi|H|psi> from OpenFermion for a random circuit that mixes all particle numbers:
    checks every Pauli string, its coefficient and the qubit order of the conversion"""
    H = hubbard_hamiltonian(L, t, U, V)
    angles = RNG.uniform(0, 2 * np.pi, size=(2, NQ))

    def circuit_fn(n_qubits, n_electrons, params):
        c = mc.Circuit()
        for q in range(NQ):
            c.push(mc.GateRY(angles[0, q]), q)
            c.push(mc.GateRZ(angles[1, q]), q)
        for q in range(NQ - 1):
            c.push(mc.GateCX(), q, q + 1)
        return c

    psi = state_vector(circuit_fn(NQ, NE, []), NQ)
    assert sector_weight(psi, sparse(number_operator(NQ)), NE) < 0.9          # really a mixed-N state
    assert abs(mimiq_energy(H, circuit_fn, np.zeros(0)) - expectation(sparse(H), psi)) < TOL


def test_bit_order_single_qubit():
    """X on qubit q -> OpenFermion index 2^(n-1-q) (qubit 0 = leftmost bit); n_q measured = 1"""
    for q in (0, 1, 5, 11):
        c = mc.Circuit()
        c.push(mc.GateX(), q)
        psi = state_vector(c, NQ)
        assert abs(psi[2 ** (NQ - 1 - q)] - 1) < 1e-12
        assert abs(np.linalg.norm(psi) - 1) < 1e-12
        assert abs(expectation(sparse(n_op(q)), psi) - 1) < 1e-12


def test_load_state_round_trip():
    """load_state (OpenFermion -> MIMIQ order) followed by the read-out gives the same vector"""
    psi = random_state()
    assert np.allclose(evolve_state(psi, mc.Circuit()), psi, atol=1e-12)


# ==========================================================================================
# 3. Orbitals, Hartree-Fock and the rotation to the sites
# ==========================================================================================
@pytest.mark.parametrize("t", [1.0, 0.05, 2.0])
def test_hopping_orbitals_diagonalise_the_hopping_matrix(t):
    energies, C = hopping_orbitals(L, t)
    h = np.zeros((L, L))
    for k in range(L):
        h[k, (k + 1) % L] = h[(k + 1) % L, k] = -t
    assert np.allclose(C.T @ h @ C, np.diag(energies), atol=1e-12)
    assert np.allclose(C.T @ C, np.eye(L), atol=1e-12)
    assert np.allclose(energies, t * np.array([-2, -1, -1, 1, 1, 2]), atol=1e-12)
    assert np.all(np.isreal(C))


@pytest.mark.parametrize("t, U", [(1.0, 0.0), (1.0, 4.0), (0.5, 7.0)])
def test_hartree_fock_energy_formula(t, U):
    assert abs(hartree_fock_energy(L, t, U, NE) - (-8 * t + 1.5 * U)) < 1e-12


@pytest.mark.parametrize("p", range(L))
@pytest.mark.parametrize("spin", [0, 1])
def test_rotation_moves_one_electron_from_orbital_to_sites(p, spin):
    """one electron in orbital p -> amplitudes C[k, p] on the sites k (same spin), exactly (signs included)"""
    _, C = hopping_orbitals(L, 1.0)
    c = mc.Circuit()
    c.push(mc.GateX(), 2 * p + spin)
    c.append(rotation_to_sites(C))
    psi = state_vector(c, NQ)
    expected = np.zeros(2**NQ, dtype=complex)
    for k in range(L):
        expected[2 ** (NQ - 1 - (2 * k + spin))] = C[k, p]
    assert np.allclose(psi, expected, atol=1e-10)


def test_rotation_conserves_number_and_sz():
    """a random 6-electron Sz = 0 state stays in that sector after the rotation"""
    _, C = hopping_orbitals(L, 1.0)
    N, SZ = sparse(number_operator(NQ)), sparse(sz_operator(L))
    psi = random_state()
    mask = (np.abs(N.diagonal() - NE) < 1e-9) & (np.abs(SZ.diagonal()) < 1e-9)
    psi = np.where(mask, psi, 0)
    psi /= np.linalg.norm(psi)
    out = evolve_state(psi, rotation_to_sites(C))
    assert abs(np.linalg.norm(out) - 1) < 1e-12
    assert sector_weight(out, N, NE) > 1 - 1e-12
    assert sector_weight(out, SZ, 0) > 1 - 1e-12


def test_hartree_fock_on_sites_is_the_U0_ground_state():
    """X on the 6 lowest spin-orbitals + rotation = exact U = 0 ground state (closed shell for L = 6)"""
    _, C = hopping_orbitals(L, 1.0)
    c = mc.Circuit()
    for q in range(NE):
        c.push(mc.GateX(), q)
    c.append(rotation_to_sites(C))
    _, psi0 = exact_ground_state(hubbard_hamiltonian(L, 1.0, 0.0), NQ, NE)
    assert fidelity(state_vector(c, NQ), psi0) > 1 - TOL


# ==========================================================================================
# 4. UCCSD
# ==========================================================================================
def test_uccsd_parameter_count():
    assert number_of_parameters(NQ, NE) == 117


def test_uccsd_state_is_normalised_and_in_the_right_sector():
    _, C = hopping_orbitals(L, 1.0)
    psi = state_vector(make_ansatz(C)(NQ, NE, RNG.normal(0, 0.3, 117)), NQ)
    assert abs(np.linalg.norm(psi) - 1) < 1e-12
    assert sector_weight(psi, sparse(number_operator(NQ)), NE) > 1 - 1e-10
    assert sector_weight(psi, sparse(sz_operator(L)), 0) > 1 - 1e-10


@pytest.mark.parametrize("t, U, V", [(1.0, 4.0, 0.0), (1.0, 10.0, 5.0)])
def test_uccsd_mimiq_energy_is_variational(t, U, V):
    """MIMIQ energy = state-vector energy, and never below E0, for random angles"""
    H = hubbard_hamiltonian(L, t, U, V)
    _, C = hopping_orbitals(L, t)
    ansatz = make_ansatz(C)
    E0 = exact_ground_state(H, NQ, NE)[0]
    for scale in (0.05, 0.5):
        theta = RNG.normal(0, scale, 117)
        E = mimiq_energy(H, ansatz, theta)
        assert abs(E - expectation(sparse(H), state_vector(ansatz(NQ, NE, theta), NQ))) < TOL
        assert E >= E0 - TOL


# ==========================================================================================
# 5. LUCJ (ffsim, up-then-down layout)
# ==========================================================================================
def test_lucj_parameter_count():
    assert lucj_num_parameters(L, 2, "local") == 130


def test_up_then_down_is_only_a_relabelling():
    """same spectrum and the same Pauli count after renumbering the modes"""
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    H2 = to_up_then_down(H, L)
    assert abs(exact_ground_state(H, NQ, NE)[0] - exact_ground_state(H2, NQ, NE)[0]) < TOL
    assert mimiq_hamiltonian(H)[2] == mimiq_hamiltonian(H2)[2]
    # mode 2k + s -> k + s L
    assert same_operator(to_up_then_down(n_op(2 * 3 + 1), L), n_op(3 + L))


@pytest.mark.parametrize("U", [1.0, 10.0])
def test_lucj_theta_zero_is_hartree_fock(U):
    """theta = 0: E = E_HF exactly and the state is the U = 0 ground state (fidelity 1)"""
    _, C = hopping_orbitals(L, 1.0)
    ansatz = lucj_ansatz(C, 2, "local")
    theta = np.zeros(130)
    H = to_up_then_down(hubbard_hamiltonian(L, 1.0, U), L)
    assert abs(mimiq_energy(H, ansatz, theta) - (-8 + 1.5 * U)) < TOL
    _, psi0 = exact_ground_state(to_up_then_down(hubbard_hamiltonian(L, 1.0, 0.0), L), NQ, NE)
    assert fidelity(state_vector(ansatz(NQ, NE, theta), NQ), psi0) > 1 - TOL


def test_lucj_random_angles():
    """normalised, 6 electrons, Sz = 0, MIMIQ energy = state-vector energy >= E0"""
    _, C = hopping_orbitals(L, 1.0)
    ansatz = lucj_ansatz(C, 2, "local")
    H = to_up_then_down(hubbard_hamiltonian(L, 1.0, 4.0), L)
    E0 = exact_ground_state(H, NQ, NE)[0]
    theta = RNG.normal(0, 0.3, 130)
    psi = state_vector(ansatz(NQ, NE, theta), NQ)
    assert abs(np.linalg.norm(psi) - 1) < 1e-12
    assert sector_weight(psi, sparse(number_operator(NQ)), NE) > 1 - 1e-10
    assert sector_weight(psi, sparse(to_up_then_down(sz_operator(L), L)), 0) > 1 - 1e-10
    E = mimiq_energy(H, ansatz, theta)
    assert abs(E - expectation(sparse(H), psi)) < TOL
    assert E >= E0 - TOL


# ==========================================================================================
# 6. HVA
# ==========================================================================================
@pytest.mark.parametrize("t, U, V", POINTS)
def test_hva_layers_rebuild_the_hamiltonian(t, U, V):
    """H = -t (H_even + H_odd) + U H_U + V H_V: the layers are exactly the Hamiltonian's own terms"""
    groups = {name: op for name, op, _ in layer_operators(L, with_V=True)}
    assert same_operator(hubbard_hamiltonian(L, t, U, V),
                         -t * (groups["even"] + groups["odd"]) + U * groups["U"] + V * groups["V"])
    assert same_operator(groups["even"] + groups["odd"], hopping_operator(L, range(L)))
    assert same_operator(groups["U"], onsite_operator(L)) and same_operator(groups["V"], neighbour_operator(L))


def test_hva_strings_commute_inside_each_layer():
    """inside a layer every pair of Pauli strings commutes, so the RPauli product is exact"""
    for name, _, terms in layer_operators(L, with_V=True):
        strings = [QubitOperator(tuple(zip(q, letters))) for letters, q, _ in terms]
        assert len(strings) > 0
        for a, b in itertools.combinations(strings, 2):
            assert commutator(a, b).induced_norm() < 1e-14, name


@pytest.mark.parametrize("angle", [0.0, 0.37, -1.2, 2.9])
def test_hva_layer_is_the_exact_exponential(angle):
    """push_exp(group, angle) = exp(-i angle H_group) (identity part dropped), on a random state"""
    psi = random_state()
    for name, op, terms in layer_operators(L, with_V=True):
        Q = jordan_wigner(op)
        Q.terms.pop((), None)                               # pauli_terms drops the identity (a global phase)
        exact = expm_multiply(-1j * angle * sparse(Q), psi)
        c = mc.Circuit()
        push_exp(c, terms, angle)
        assert np.allclose(evolve_state(psi, c), exact, atol=1e-10), name


def test_hva_parameter_count_and_errors():
    assert hva_ansatz(L, 1.0, 4).per_layer == 3
    assert hva_ansatz(L, 1.0, 4, with_V=True).per_layer == 4
    with pytest.raises(ValueError):
        hva_ansatz(L, 1.0, 2)(NQ, NE, np.zeros(5))          # needs 6
    with pytest.raises(ValueError):
        layer_operators(5)                                  # odd ring: no even/odd bond split


@pytest.mark.parametrize("U", [1.0, 4.0, 10.0])
def test_hva_theta_zero_is_the_U0_ground_state(U):
    ansatz = hva_ansatz(L, 1.0, 4)
    _, psi0 = exact_ground_state(hubbard_hamiltonian(L, 1.0, 0.0), NQ, NE)
    assert fidelity(state_vector(ansatz(NQ, NE, np.zeros(12)), NQ), psi0) > 1 - TOL
    assert abs(mimiq_energy(hubbard_hamiltonian(L, 1.0, U), ansatz, np.zeros(12)) - (-8 + 1.5 * U)) < TOL


@pytest.mark.parametrize("t, U, V, with_V", [(1.0, 4.0, 0.0, False), (1.0, 10.0, 5.0, True)])
def test_hva_random_angles(t, U, V, with_V):
    """normalised, 6 electrons, singlet (S^2 = 0: every layer commutes with S^2), energy exact and >= E0"""
    ansatz = hva_ansatz(L, t, 4, with_V=with_V)
    H = hubbard_hamiltonian(L, t, U, V)
    E0 = exact_ground_state(H, NQ, NE)[0]
    theta = RNG.normal(0, 0.7, 4 * ansatz.per_layer)
    psi = state_vector(ansatz(NQ, NE, theta), NQ)
    assert abs(np.linalg.norm(psi) - 1) < 1e-12
    assert sector_weight(psi, sparse(number_operator(NQ)), NE) > 1 - 1e-10
    assert abs(expectation(sparse(s_squared_operator(L)), psi)) < 1e-9
    E = mimiq_energy(H, ansatz, theta)
    assert abs(E - expectation(sparse(H), psi)) < TOL
    assert E >= E0 - TOL


def test_hva_theta_zero_is_a_stationary_point():
    """the gradient of the HVA energy vanishes at theta = 0 (why COBYLA needs the jitter), checked by
    central differences on the state vector"""
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    Hs = sparse(H)
    ansatz = hva_ansatz(L, 1.0, 2)
    energy = lambda th: expectation(Hs, state_vector(ansatz(NQ, NE, th), NQ))
    h = 1e-4
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        assert abs((energy(e) - energy(-e)) / (2 * h)) < 1e-7


# ==========================================================================================
# 7. Fidelity
# ==========================================================================================
def test_fidelity_definition():
    """F(psi, psi) = 1, invariant under a global phase, 0 for orthogonal states (ground and first excited)"""
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    w, v = eigsh(jw_number_restrict_operator(sparse(H), NE, NQ), k=2, which="SA")
    a, b = v[:, 0], v[:, 1]
    assert abs(fidelity(a, a) - 1) < 1e-12
    assert abs(fidelity(a, np.exp(1.234j) * a) - 1) < 1e-12
    assert fidelity(a, b) < 1e-12


def test_fidelity_and_energy_obey_the_gap_bound():
    """for any 6-electron state: 1 - F >= (E - E0) / (E_max - E0) and 1 - F <= (E - E0) / (E1 - E0)"""
    H = hubbard_hamiltonian(L, 1.0, 10.0)
    Hs = sparse(H)
    E0, psi0 = exact_ground_state(H, NQ, NE)
    E1 = lowest_two(H)[1]
    E_max = -eigsh(-jw_number_restrict_operator(Hs, NE, NQ), k=1, which="SA")[0][0]
    ansatz = hva_ansatz(L, 1.0, 4)
    for scale in (0.1, 0.5, 1.5):
        psi = state_vector(ansatz(NQ, NE, RNG.normal(0, scale, 12)), NQ)
        E, F = expectation(Hs, psi), fidelity(psi0, psi)
        assert 1 - F <= (E - E0) / (E1 - E0) + 1e-10
        assert 1 - F >= (E - E0) / (E_max - E0) - 1e-10


# ==========================================================================================
# 8. Short VQE runs: the returned numbers are consistent with the returned angles
# ==========================================================================================
def check_result(r, n_params):
    assert r["n_parameters"] == n_params == len(r["theta_opt"])
    assert r["E_VQE"] >= r["E0"] - 1e-9                    # variational principle
    assert r["E_VQE"] <= r["E_HF"] + 1e-12                 # the best energy is never worse than the start
    assert abs(r["dE"] - (r["E_VQE"] - r["E0"])) < 1e-14
    assert abs(r["rel_error"] - abs(r["dE"]) / abs(r["E0"])) < 1e-14
    assert 0 <= r["fidelity"] <= 1 + 1e-12 and 0 <= r["fidelity_HF"] <= 1 + 1e-12
    assert r["n_evaluations"] > 0


def test_short_hva_run_is_consistent():
    """the reported energy and fidelity are those of the reported angles"""
    r = run_vqe_hva(L=6, t=1.0, U=4.0, n_layers=2, max_cycles=1, chunk_maxiter=150)
    check_result(r, 6)
    ansatz = hva_ansatz(L, 1.0, 2)
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    E0, psi0 = exact_ground_state(H, NQ, NE)
    psi = state_vector(ansatz(NQ, NE, r["theta_opt"]), NQ)
    assert abs(expectation(sparse(H), psi) - r["E_VQE"]) < 1e-9
    assert abs(fidelity(psi0, psi) - r["fidelity"]) < 1e-9
    assert abs(r["E0"] - E0) < 1e-12


def test_hva_warm_start_is_used():
    """theta_start: the run never ends above the energy of its start angles"""
    theta = RNG.normal(0, 0.2, 6)
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    E_start = expectation(sparse(H), state_vector(hva_ansatz(L, 1.0, 2)(NQ, NE, theta), NQ))
    r = run_vqe_hva(L=6, t=1.0, U=4.0, n_layers=2, max_cycles=1, chunk_maxiter=60, theta_start=theta)
    assert r["E_VQE"] <= E_start + 1e-9


def test_short_lucj_run_is_consistent():
    r = run_vqe_lucj(L=6, t=1.0, U=1.0, max_cycles=1, chunk_maxiter=60)
    check_result(r, 130)


def test_short_uccsd_run_is_consistent():
    r = run_vqe(L=6, t=1.0, U=1.0, max_cycles=1, chunk_maxiter=60)
    check_result(r, 117)
    assert abs(r["E_HF"] - (-8 + 1.5)) < TOL
    assert abs(r["fidelity_HF"] - 0.979685) < 1e-6         # recorded Hartree-Fock fidelity at U/t = 1


def test_recorded_hartree_fock_fidelity_at_U10():
    _, C = hopping_orbitals(L, 1.0)
    psi_hf = state_vector(hva_ansatz(L, 1.0, 1)(NQ, NE, np.zeros(3)), NQ)
    _, psi0 = exact_ground_state(hubbard_hamiltonian(L, 1.0, 10.0), NQ, NE)
    assert abs(fidelity(psi0, psi_hf) - 0.264546) < 1e-6


# ==========================================================================================
# 9. File names
# ==========================================================================================
def test_result_names():
    assert BACKEND == "mimiq-exaqt"
    assert result_name(6, 1.0, 10.0, 0.0, "hva-p8") == "hubbard_L6_t1_U10_V0_hva-p8_mimiq-exaqt"
    assert result_name(6, 0.05, 2.5, 1.0, "lucj-local-r2") == "hubbard_L6_t0.05_U2.5_V1_lucj-local-r2_mimiq-exaqt"
    name = result_name(8, 1, 4, 0, "uccsd")
    assert name.startswith("hubbard_") and name.endswith("_" + BACKEND) and " " not in name
