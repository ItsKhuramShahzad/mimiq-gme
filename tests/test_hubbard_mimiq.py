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
    Openfermion state (qubits 0 = leftmost bit) -> Exaqt order (qubit 0 = lowest bit)

    Args:
        psi: Openfermion state vector
        n_qubits: number of qubits"""
        
    new_psi = np.zeros_like(psi) # a emply vector of the same shape as psi
    for i in range(2**n_qubits): # 
        bits = format(i, f"0{n_qubits}b") # position i in bits, e.g 4 -> 100, write number in binary
        new_index = int(bits[::-1], 2)  # reverse the bits   → 4   (read "100" as binary)
        new_psi[new_index] = psi[i]
    return new_psi

def openfermion_state_to_mimiq(psi_openfermion, n_qubits, num_zvars=0):
    """
 Load a state vector from openferion into mimiq (Exaqt) simulator state.
    Args:also 
        psi_openfermion: state vector with 2**n_qubits amplitudes, in openfermion order (qubit 0 = leftmost bit)
        n_qubits: number of qubits (2 per site for hubbard_gme)
        num_zvars: number of z-variables, one per expectation value the circuit will measure.
        
    Returns:
        ExaqtState: mimiq state holding psi, ready for exaqt.evolve() with a circuit. The state is in Exaqt order (qubit 0 = lowest bit).
        
    """
    
    # an empty MIMIQ state: qubits in |00...0>, num_zvars z-variables set to 0 (one per expectation value)

    mimiq_state = exaqt.ExaqtState.zero(n_qubits, num_zvars=num_zvars)
    
    #reorder the amplitudes to mimiq bit order (qubit 0 = lowest bit) and convert to complex128
    amplitudes_mimiq_order = to_exaqt_order(psi_openfermion, n_qubits)
    psi_mimiq  = np.ascontiguousarray(amplitudes_mimiq_order, dtype=np.complex128)
    
    # load the amplitudes into the mimiq state  replace |00...0> by our state (written directly into memory, no gates)
    mimiq_state.q= exaqt.ExaqtSV.from_amplitudes(n_qubits, psi_mimiq)
    
    return mimiq_state



def test_bit_order_random_state():
    """
    
    A non symmetric check: one pauli sting on a randon state, mimiq vs openfermion"""
    
    nq=8
    
    rng = np.random.default_rng(1)
    psi= rng.normal(size=2**nq)+1j*rng.normal(size=2**nq)
    
    psi/= np.linalg.norm(psi)
    ref = np.vdot(psi, get_sparse_operator(QubitOperator("Z0 X1 Y5"), nq)@ psi).real
    got = openfermion_state_to_mimiq(psi, nq).q.expectation_pauli("ZXY", [0, 1, 5]).real
    assert got ==pytest.approx(ref, abs=1e-12)
    
# @pytest.mark.parametrize("L, U", [(4, 4.0), (6, 4.0)])     # correct
# def test_hubbard_energy_on_mimiq(L, U):
    
#     t, n_electrons, nq = 1.0, L, 2*L     # correct: half filling, L electrons on 2L qubits
    
#     # hamiltonin: hubbard_hme -> jordan_wigner -> mimiq
#     H_qubit = jordan_wigner(build_hubbard_hamiltonian(L, t, U, mode="uniform"))
#     H_mimiq, c0, _= openfermion_to_mimiq_hamiltonian(H_qubit)
#     print(H_mimiq.matrix())
#     #exact reference form hubbard gme and the exact state as full 2^nq vector
#     E0_ref = compute_ground_and_excited_states(L, t, U,"uniform", n_electrons)[0]
#     _, psi= jw_get_ground_state_at_particle_number(get_sparse_operator(H_qubit, nq), n_electrons)
    
#     # <H> on mimiq: load the state, measure with push_expval
    
#     circuit = mc.Circuit()
#     circuit.push_expval(H_mimiq, *range(nq))
#     st = openfermion_state_to_mimiq(psi, nq, num_zvars=circuit.num_zvars())
#     st, _ = exaqt.evolve(st, circuit)
#     assert st.z[0].real +c0 == pytest.approx(E0_ref, abs=1e-10)
    
    
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
    st = openfermion_state_to_mimiq (psi, n_qubits, num_zvars=circuit.num_zvars())
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
        
@pytest.mark.parametrize("qubit", [0,1,2])
def test_qubit_position_openfermion_vs_mimiq(qubit):
    """
    Test the position of a qubit in the state vector of openfermion and mimiq.

    Flip one qubit from |000> in both pragram and check where the amplitudes lands:
    
    openfermion puts qubit 0 as higest bit, Exaqt as lowest bit. 
    and to_exaqt_order turns the opnefermion vector into exaqt order.

    
    """
    
    n=3; 
    
    #openfermion : |000> is position 0,, apply X on this qubit
    
    start = np.zeros(2**n, dtype=complex)
    start[0] = 1.0
    
    psi_of = get_sparse_operator(QubitOperator(f"X{qubit}"), n) @ start  # @=> product
    
    #Exaqt : same operation as a circuit
    circuit = mc.Circuit()
    circuit.push(mc.GateX(), qubit)
    circuit.push(mc.GateID(), n-1) # to make sure the circuit has the right number of qubits
    
    st, _ = exaqt.evolve(exaqt.ExaqtState.zero(n), circuit)
    psi_exq= np.asarray(st.q.amplitudes(), dtype=complex)
    
    # where each program stores "this qubit is 1"
    
    assert np.flatnonzero(np.abs(psi_of)> 0.5)[0] == 2**(n-1-qubit)  # openfermion: qubit 0 is highest bit
    assert np.flatnonzero(np.abs(psi_exq)> 0.5)[0] == 2**(qubit)  # exaqt: qubit 0 is lowest bit
    
    # our conversation gives excatly the exaqt vector
    
    assert np.allclose(to_exaqt_order(psi_of, n), psi_exq)
    