# hubbard hamiltonian on mimiq, exaqt ground state loaded into mimiq, <H> = E0.

import sys
import numpy as np
import pytest

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")
sys.path.insert(0, "/home/anjum/PhDWork/Strasbourg_Secondment/mimiq_vqe")

import exaqt
import mimiqcircuits as mc
from itertools import combinations
from openfermion import QubitOperator, jordan_wigner, get_sparse_operator , FermionOperator
from openfermion.linalg import jw_get_ground_state_at_particle_number
from hubbard_gme.hamiltonian import build_hubbard_hamiltonian, compute_ground_and_excited_states
from src.mimiq_hamiltonian import openfermion_to_mimiq_hamiltonian
from hubbard_gme.rdm import compute_general_nRDM



def to_exaqt_order(psi, n_qubits):
    """
    Openfermion state (qubits o = leftmost bit) -> Exaqt order (qubit 0 = lowest bit)

    Args:
        psi: Openfermion state vector
        n_qubits: number of qubits"""
        
        
    return psi.reshape([2]*n_qubits).transpose(range(n_qubits-1, -1, -1)).reshape(-1)

def load_state(psi_openfermion, n_qubits, num_zvars=0):
    """
    Put a given state vector inot exaqt state, ready for a circuit
    """
    st = exaqt.ExaqtState.zero(n_qubits, num_zvars=num_zvars)
    amps = np.ascontiguousarray(to_exaqt_order(psi_openfermion, n_qubits    ), dtype=np.complex128)
    st.q= exaqt.ExaqtSV.from_amplitudes(n_qubits, amps)
    
    return st


def test_bit_order_random_state():
    """
    
    A non symmetric check: one pauli sting on a randon state, mimiq vs openfermion"""
    
    nq=8
    
    rng = np.random.default_rng(1)
    psi= rng.normal(size=2**nq)+1j*rng.normal(size=2**nq)
    
    psi/= np.linalg.norm(psi)
    ref = np.vdot(psi, get_sparse_operator(QubitOperator("Z0 X1 Y5"), nq)@ psi).real
    got = load_state(psi, nq).q.expectation_pauli("ZXY", [0, 1, 5]).real
    assert got ==pytest.approx(ref, abs=1e-12)
    
@pytest.mark.parametrize("L, U", [(4, 4.0), (6, 4.0)])     # correct
def test_hubbard_energy_on_mimiq(L, U):
    t, n_electrons, nq = 1.0, L, 2*L     # correct: half filling, L electrons on 2L qubits
    
    # hamiltonin: hubbard_hme -> jordan_wigner -> mimiq
    H_qubit = jordan_wigner(build_hubbard_hamiltonian(L, t, U, mode="uniform"))
    H_mimiq, c0, _= openfermion_to_mimiq_hamiltonian(H_qubit)
    
    #exact reference form hubbard gme and the exact state as full 2^nq vector
    E0_ref = compute_ground_and_excited_states(L, t, U,"uniform", n_electrons)[0]
    _, psi= jw_get_ground_state_at_particle_number(get_sparse_operator(H_qubit, nq), n_electrons)
    
    # <H> on mimiq: load the state, measure with push_expval
    
    circuit = mc.Circuit()
    circuit.push_expval(H_mimiq, *range(nq))
    st = load_state(psi, nq, num_zvars=circuit.num_zvars())
    st, _ = exaqt.evolve(st, circuit)
    assert st.z[0].real +c0 == pytest.approx(E0_ref, abs=1e-10)
    
    
def rdm_pauli_term(sites):
    """
    for every element (bra, ket) of n-site rdm: its jordanwigner expansion 
    {pauli_terms: coeff}. 
    same element order as hubbard_gme: sites s ahs up=2s and down=2s+1; bit k of index is the spin of body k
    """
    
    n = len(sites)
    dim = 2**n
    out = {}
    for bra in range(dim):
        for ket in range(dim):
            ops = []
            for body, site in enumerate(sites):
                ops.append((2*site+ ((bra>>body) & 1), 1)) # c^dagger (bra spin )
                ops.append((2*site+ ((ket>>body) & 1), 0)) # c (ket spin)
            out[(bra, ket)] = jordan_wigner(FermionOperator(tuple(ops))).terms
    return out
def measure_on_mimiq(psi, n_qubits, strings):
    """
    <P> for every paulu string, measured in one circuit  on exaqt (one expectation valuer per string)"""
    
    strings = [s for s in strings if s !=()]
    circuit = mc.Circuit()
    for z, term in enumerate(strings):
        qubits = [q for q, p in term]
        paulis = "". join(p for q, p in term)
        circuit.push(mc.ExpectationValue(mc.PauliString(paulis)), *qubits, z)
    st = load_state (psi, n_qubits, num_zvars=circuit.num_zvars())
    st, _ = exaqt.evolve(st, circuit)
    
    values = {term:  st.z[z] for z , term in enumerate(strings)}
    values[()] = 1.0
    return values

def rdm_from_expectation(terms_per_element, values, n):
    """
    rho [bra, ket] = sum_k c_k <P_k>, then normalised to trace 1.
    """
    
    dim = 2**n
    rho = np.zeros((dim,dim), dtype=complex)
    for (bra, ket), terms in terms_per_element.items():
        rho[bra, ket] = sum(coeff* values[term] for term, coeff in terms.items())
    return rho/np.trace(rho).real



@pytest.mark.parametrize("L, n", [(4,1), (4,2), (4,3), (4,4), (6,1), (6,2), (6,3), (6,4)])
def test_rdm_from_mimiq_pauli_strings(L, n):
    U, t , n_electrons, nq = 4.0, 1.0, L, 2*L
    H_qubit = jordan_wigner(build_hubbard_hamiltonian(L, t, U, mode="uniform"))
   
   # the exact gound state: full 2^nq vector for mimiq, number preserving vector for hubbard_gme
    _, psi = jw_get_ground_state_at_particle_number(get_sparse_operator(H_qubit, nq), n_electrons)
    
    psi_hubbard = compute_ground_and_excited_states(L, t, U,"uniform", n_electrons)[1]
    
    cluster = list(combinations(range(L), n))
    terms = {cl: rdm_pauli_term(cl) for cl in cluster}
    union = set().union(*(term for per_cluster in terms.values() for term in per_cluster.values()))
    values = measure_on_mimiq(psi, nq, union)
    
    for cl in cluster:
        rho_mimiq = rdm_from_expectation(terms[cl], values, n)
        rho_hubbard = compute_general_nRDM(list(cl), psi_hubbard, nq, n_electrons)
        rho_hubbard = rho_hubbard/np.trace(rho_hubbard).real
        assert np.max(np.abs(rho_mimiq-rho_hubbard)) < 1e-10