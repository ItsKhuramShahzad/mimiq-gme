# Tests for the hardware-oriented options: the blocked qubit order, the native CX-CRX-CX hopping block and
# the entanglement entropy. Every check is against an exact answer, mostly to 1e-10.
# Run:  python -m pytest -q tests/test_hubbard_vqe_mimiq_hardware.py

import os
import sys

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mimiqcircuits as mc
from openfermion import get_sparse_operator, number_operator, sz_operator

from hubbard_vqe_mimiq.core import (hubbard_hamiltonian, exact_ground_state, hopping_orbitals, rotation_to_sites,
                                    state_vector, mode_index, to_layout, block_entropy, site_entropy)
from hubbard_vqe_mimiq.hva import hva_ansatz, layer_operators, push_exp, push_hop_native, push_hopping_native

L, NQ, NE = 6, 12, 6
TOL = 1e-10
RNG = np.random.default_rng(7)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])


def energy(H, psi):
    return float(np.real(np.vdot(psi, get_sparse_operator(H, n_qubits=NQ) @ psi)))


def unitary_2q(circuit):
    """the 4 x 4 unitary of a 2-qubit circuit, OpenFermion order"""
    U = np.zeros((4, 4), dtype=complex)
    for b in range(4):
        c = mc.Circuit()
        for q in range(2):
            if (b >> (1 - q)) & 1:
                c.push(mc.GateX(), q)
        c.append(circuit)
        U[:, b] = state_vector(c, 2)
    return U


# ------------------------------------------------------------------------------------------
# qubit order
# ------------------------------------------------------------------------------------------
def test_mode_index():
    assert [mode_index(k, s, L, "interleaved") for k in range(L) for s in (0, 1)] == list(range(NQ))
    assert [mode_index(k, 0, L, "blocked") for k in range(L)] == list(range(L))
    assert [mode_index(k, 1, L, "blocked") for k in range(L)] == list(range(L, NQ))
    with pytest.raises(ValueError):
        mode_index(0, 0, L, "zigzag")


def test_blocked_hopping_has_no_z_string_except_the_ring_bond():
    """blocked order: hops between neighbouring sites are XX + YY on neighbouring qubits; only the ring
    bond (site L -> site 1) carries a Z string; interleaved: every hop has a Z on the other spin"""
    for name, _, terms in layer_operators(L, layout="blocked"):
        if name not in ("even", "odd"):
            continue
        for letters, qubits, c in terms:
            ring = {qubits[0] % L, qubits[-1] % L} == {0, L - 1}
            assert (len(qubits) == L) if ring else (letters in ("XX", "YY") and qubits[1] - qubits[0] == 1)
            assert abs(abs(c) - 0.5) < 1e-14
    for name, _, terms in layer_operators(L, layout="interleaved"):
        if name in ("even", "odd"):
            assert all(len(q) >= 3 for _, q, _ in terms)


@pytest.mark.parametrize("U", [1.0, 4.0, 10.0])
def test_blocked_hamiltonian_has_the_same_spectrum(U):
    H = hubbard_hamiltonian(L, 1.0, U)
    assert abs(exact_ground_state(H, NQ, NE)[0] - exact_ground_state(to_layout(H, L, "blocked"), NQ, NE)[0]) < TOL


def test_blocked_rotation_moves_one_electron_to_the_sites():
    """blocked order: one electron in orbital p -> amplitudes C[k, p] on the sites k, signs included"""
    _, C = hopping_orbitals(L, 1.0)
    for p in range(L):
        for spin in (0, 1):
            c = mc.Circuit()
            c.push(mc.GateX(), mode_index(p, spin, L, "blocked"))
            c.append(rotation_to_sites(C, "blocked"))
            expected = np.zeros(2**NQ, dtype=complex)
            for k in range(L):
                expected[2 ** (NQ - 1 - mode_index(k, spin, L, "blocked"))] = C[k, p]
            assert np.allclose(state_vector(c, NQ), expected, atol=TOL)


# ------------------------------------------------------------------------------------------
# native hopping block
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.0, 0.3, -1.1, 2.4])
def test_native_hop_is_the_exact_exponential(phi):
    """CX - CRX(4 phi) - CX = exp(-i phi (XX + YY)) exactly, including the global phase"""
    c = mc.Circuit()
    push_hop_native(c, 0, 1, phi)
    assert np.allclose(unitary_2q(c), expm(-1j * phi * (np.kron(X, X) + np.kron(Y, Y))), atol=TOL)


@pytest.mark.parametrize("name", ["even", "odd"])
def test_native_hopping_layer_equals_rpauli_layer(name):
    """the whole hopping layer with native blocks gives the same state as with RPauli gates"""
    terms = {n: t for n, _, t in layer_operators(L, layout="blocked")}[name]
    start = hva_ansatz(L, 1.0, 1, layout="blocked")(NQ, NE, RNG.normal(0, 0.5, 3))
    a, b = start.deepcopy(), start.deepcopy()
    push_exp(a, terms, 0.77)
    push_hopping_native(b, terms, 0.77)
    assert abs(np.vdot(state_vector(a, NQ), state_vector(b, NQ)) - 1) < TOL


# ------------------------------------------------------------------------------------------
# the same physics in every version of the circuit
# ------------------------------------------------------------------------------------------
VERSIONS = [("interleaved", False), ("blocked", False), ("blocked", True)]


@pytest.mark.parametrize("U", [1.0, 10.0])
def test_all_versions_give_the_same_energy_and_fidelity(U):
    """for the same angles every qubit order / gate choice gives the same energy and fidelity"""
    theta = RNG.normal(0, 0.6, 12)
    results = []
    for layout, native in VERSIONS:
        H = to_layout(hubbard_hamiltonian(L, 1.0, U), L, layout)
        _, psi0 = exact_ground_state(H, NQ, NE)
        psi = state_vector(hva_ansatz(L, 1.0, 4, layout=layout, native_hopping=native)(NQ, NE, theta), NQ)
        results.append((energy(H, psi), abs(np.vdot(psi0, psi)) ** 2))
    for E, F in results[1:]:
        assert abs(E - results[0][0]) < TOL and abs(F - results[0][1]) < 1e-9


@pytest.mark.parametrize("layout, native", VERSIONS)
def test_theta_zero_and_conservation(layout, native):
    """theta = 0: E_HF = -8t + 1.5U; random angles: 6 electrons, Sz = 0, in every version"""
    H = to_layout(hubbard_hamiltonian(L, 1.0, 4.0), L, layout)
    ansatz = hva_ansatz(L, 1.0, 2, layout=layout, native_hopping=native)
    assert abs(energy(H, state_vector(ansatz(NQ, NE, np.zeros(6)), NQ)) - (-8 + 6)) < TOL
    psi = state_vector(ansatz(NQ, NE, RNG.normal(0, 0.8, 6)), NQ)
    N = get_sparse_operator(number_operator(NQ), n_qubits=NQ)
    SZ = get_sparse_operator(to_layout(sz_operator(L), L, layout), n_qubits=NQ)
    assert abs(energy(number_operator(NQ), psi) - NE) < TOL
    assert abs(np.real(np.vdot(psi, N @ N @ psi)) - NE**2) < 1e-8
    assert abs(np.real(np.vdot(psi, SZ @ psi))) < TOL


def count_two_qubit_gates(circuit):
    return sum(1 for inst in circuit.decompose() if inst.num_qubits() == 2)


def test_blocked_order_halves_the_two_qubit_gates_of_the_hopping():
    """per hopping layer: interleaved 160 two-qubit gates, blocked 80; the native block keeps 80 but needs
    far fewer gates in total (recorded values for L = 6)"""
    counts = {}
    for layout, native in VERSIONS:
        terms = {n: t for n, _, t in layer_operators(L, layout=layout)}
        c = mc.Circuit()
        for name in ("even", "odd"):
            (push_hopping_native if native else push_exp)(c, terms[name], 0.3)
        counts[(layout, native)] = (len(list(c.decompose())), count_two_qubit_gates(c))
    assert counts[("interleaved", False)] == (472, 160)
    assert counts[("blocked", False)] == (392, 80)
    assert counts[("blocked", True)] == (162, 80)


# ------------------------------------------------------------------------------------------
# entanglement entropy
# ------------------------------------------------------------------------------------------
def test_entropy_of_simple_states():
    """product state: 0; Bell pair across the cut: 1 bit; two Bell pairs: 2 bits"""
    zero = np.zeros(2**4)
    zero[0] = 1
    assert abs(block_entropy(zero, 4, 2)) < 1e-12
    bell = np.zeros(4)
    bell[0] = bell[3] = 1 / np.sqrt(2)                     # (|00> + |11>)/sqrt2
    assert abs(block_entropy(bell, 2, 1) - 1) < 1e-12
    two = np.kron(bell, bell).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(-1)   # qubits (0,2), (1,3) paired
    assert abs(block_entropy(two, 4, 2) - 2) < 1e-12


def test_entropy_of_a_pure_state_is_symmetric():
    """S(A) = S(B) for a pure state: first m qubits vs the last 2L - m (via the reversed vector)"""
    _, psi = exact_ground_state(hubbard_hamiltonian(L, 1.0, 4.0), NQ, NE)
    reversed_psi = psi.reshape([2] * NQ).transpose(range(NQ - 1, -1, -1)).reshape(-1)
    for m in (2, 4, 6):
        assert abs(block_entropy(psi, NQ, m) - block_entropy(reversed_psi, NQ, NQ - m)) < 1e-10


def test_entropy_of_the_hartree_fock_state():
    """the U = 0 ground state (a Slater determinant): S(half ring) from the state vector equals the
    correlation-matrix formula S = -sum [n log n + (1-n) log(1-n)], twice (one per spin)"""
    _, C = hopping_orbitals(L, 1.0)
    psi = state_vector(hva_ansatz(L, 1.0, 1)(NQ, NE, np.zeros(3)), NQ)
    occ = C[:, :NE // 2]
    G = occ[:3, :] @ occ[:3, :].T                          # one-spin correlation matrix of sites 1..3
    n = np.clip(np.linalg.eigvalsh(G), 1e-15, 1 - 1e-15)
    S_one_spin = float(-np.sum(n * np.log2(n) + (1 - n) * np.log2(1 - n)))
    assert abs(site_entropy(psi, L, 3) - 2 * S_one_spin) < 1e-10
