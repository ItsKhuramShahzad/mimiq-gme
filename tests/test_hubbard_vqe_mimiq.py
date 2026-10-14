# Tests for hubbard_vqe_mimiq/core.py: every step of the VQE checked against an exact answer.
# Run:  python -m pytest -q tests/test_hubbard_vqe_mimiq.py        (about 2 minutes)

import sys

import numpy as np
import pytest
from openfermion import FermionOperator, normal_ordered

sys.path.insert(0, "/home/anjum/PhDWork/Strasbourg_Secondment/mimiq_gme")

from hubbard_vqe_mimiq.core import (hubbard_hamiltonian, exact_ground_state, mimiq_hamiltonian, hopping_orbitals,
                              hartree_fock_energy, givens_rotations, rotation_to_sites, number_of_parameters,
                              make_ansatz, state_vector, load_state, run_vqe)
from src.mimiq_backend import make_energy_fn
import exaqt
from openfermion import get_sparse_operator

L, N_QUBITS, N_ELECTRONS = 6, 12, 6
POINTS = [(1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (1.0, 10.0, 0.0), (0.05, 10.0, 0.0), (1.0, 10.0, 5.0)]   # (t, U, V)


def to_orbital_basis(H_sites, C):
    """the site Hamiltonian written in the orbitals: c_{j,s} = sum_p C[j,p] a_{p,s} (test tool only)"""
    H_orbitals = FermionOperator()
    for term, coeff in H_sites.terms.items():
        new_term = FermionOperator((), coeff)
        for spin_orbital, action in term:
            site, spin = divmod(spin_orbital, 2)
            replacement = FermionOperator()
            for p in range(C.shape[0]):
                replacement += FermionOperator(((2 * p + spin, action),), C[site, p])
            new_term *= replacement
        H_orbitals += new_term
    return normal_ordered(H_orbitals)


def test_pauli_string_counts():
    """Eq. (6): 42 strings at (1, 4, 0) and 66 at (1, 10, 5), plus a constant"""
    assert mimiq_hamiltonian(hubbard_hamiltonian(L, 1.0, 4.0))[2] == 42
    assert mimiq_hamiltonian(hubbard_hamiltonian(L, 1.0, 10.0, 5.0))[2] == 66


@pytest.mark.parametrize("t", [1.0, 0.05])
def test_orbital_energies(t):
    """Eq. (8): -2t, -t, -t, t, t, 2t for the 6-site ring; C is orthogonal"""
    energies, C = hopping_orbitals(L, t)
    assert np.allclose(energies, t * np.array([-2, -1, -1, 1, 1, 2]))
    assert np.allclose(C.T @ C, np.eye(L))


def test_number_of_parameters():
    """Eq. (12): 18 singles + 99 doubles = 117"""
    assert number_of_parameters(N_QUBITS, N_ELECTRONS) == 117


def test_givens_reproduces_C():
    """Eq. (18): the rotations bring C to a diagonal matrix of signs"""
    _, C = hopping_orbitals(L, 1.0)
    rotations, signs = givens_rotations(C)
    R = C.copy()
    for p, angle in rotations:
        G = np.eye(L)
        G[p, p] = G[p + 1, p + 1] = np.cos(angle)
        G[p, p + 1], G[p + 1, p] = np.sin(angle), -np.sin(angle)
        R = G @ R
    assert len(rotations) == L * (L - 1) // 2
    assert np.allclose(R, np.diag(signs), atol=1e-10)


@pytest.mark.parametrize("t, U, V", POINTS)
def test_rotation_maps_orbital_ground_state_to_sites(t, U, V):
    """Eq. (16): exact ground state written in the orbitals -> rotation circuit -> exact ground state on sites"""
    H = hubbard_hamiltonian(L, t, U, V)
    _, C = hopping_orbitals(L, t)
    E_sites, psi_sites = exact_ground_state(H, N_QUBITS, N_ELECTRONS)
    E_orb, psi_orb = exact_ground_state(to_orbital_basis(H, C), N_QUBITS, N_ELECTRONS)
    assert abs(E_sites - E_orb) < 1e-9
    state, _ = exaqt.evolve(load_state(psi_orb, N_QUBITS), rotation_to_sites(C).decompose())
    psi = np.asarray(state.q.amplitudes()).reshape([2] * N_QUBITS).transpose(range(N_QUBITS - 1, -1, -1)).reshape(-1)
    assert abs(np.vdot(psi_sites, psi)) ** 2 > 1 - 1e-10


@pytest.mark.parametrize("t, U", [(1.0, 1.0), (1.0, 4.0), (1.0, 10.0), (0.05, 10.0)])
def test_hartree_fock_energy_on_mimiq(t, U):
    """Eq. (10): theta = 0 gives E_HF = -8t + 1.5U, measured as <H> on MIMIQ"""
    H_mimiq, constant, _ = mimiq_hamiltonian(hubbard_hamiltonian(L, t, U))
    _, C = hopping_orbitals(L, t)
    energy_fn, _ = make_energy_fn(H_mimiq, 0.0, N_QUBITS, N_ELECTRONS, ansatz=make_ansatz(C))
    E = energy_fn(np.zeros(117)) + constant
    assert abs(E - hartree_fock_energy(L, t, U, N_ELECTRONS)) < 1e-10
    assert abs(E - (-8 * t + 1.5 * U)) < 1e-10


def test_mimiq_energy_equals_state_vector_energy():
    """Eq. (22): the energy measured on MIMIQ equals <psi|H|psi> from the state vector, for any theta"""
    H = hubbard_hamiltonian(L, 1.0, 4.0)
    H_mimiq, constant, _ = mimiq_hamiltonian(H)
    _, C = hopping_orbitals(L, 1.0)
    ansatz = make_ansatz(C)
    energy_fn, _ = make_energy_fn(H_mimiq, 0.0, N_QUBITS, N_ELECTRONS, ansatz=ansatz)
    theta = np.random.default_rng(1).normal(0, 0.1, 117)
    psi = state_vector(ansatz(N_QUBITS, N_ELECTRONS, theta), N_QUBITS)
    E_sv = float(np.real(np.vdot(psi, get_sparse_operator(H, n_qubits=N_QUBITS) @ psi)))
    assert abs(energy_fn(theta) + constant - E_sv) < 1e-10
    assert abs(np.linalg.norm(psi) - 1) < 1e-12


def test_short_vqe_lowers_the_energy():
    """a short run (one chunk of 300 COBYLA steps) goes below Hartree-Fock and stays above E0"""
    r = run_vqe(L=6, t=1.0, U=1.0, max_cycles=1, chunk_maxiter=300)
    assert r["E_VQE"] < r["E_HF"] - 1e-3
    assert r["E_VQE"] >= r["E0"] - 1e-9                         # variational principle
    assert r["fidelity"] > r["fidelity_HF"]
