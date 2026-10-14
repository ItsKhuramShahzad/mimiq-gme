# step1: the hubbard hamiltonina written in the U =0 orbitlas (orbitals of the hopping term).
# in these orbitls "fill the lowest orbitals " is a good hartree fock start for VQE.
import sys
import numpy as np
import pytest

sys.path.insert(0, "/home/anjum/PhDWork/hubbard_gme/src")

from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, normal_ordered
from openfermion.linalg import jw_get_ground_state_at_particle_number
from hubbard_gme.hamiltonian import build_hubbard_hamiltonian

def hopping_orbitals(L, t):
    """
    U =0 orbitals of the ring: eigenvectors of the hopping maxtix.(sorted by energy, lowest first)
    
    Returns:
        energies: orbital energies, lowest first
        C: C[j,p] = weight of orbitals p on site j
    """
    
    h = np.zeros((L, L))
    
    for j in range(L):
        k = (j+1)%L
        h[j, k] = -t
        h[k, j] = -t
    
    energies, C = np.linalg.eigh(h)
    return energies, C

def to_orbital_basis(H_sites, C):
    
    """
    Rewrite a site-basis FermoinOperator in the U =0 orbitals.
    Each site operator is a combination of orbital operators, with same spin:
    c_{j, sigma} = sum_p C[j,p] a_{p,s}} (spin-orbital index =2*site+spin)
    """
    L =C.shape[0]
    H_orbitals = FermionOperator()
    for term, coeff in H_sites.terms.items():
        new_term = FermionOperator((), coeff)
        for spin_orbital, action in term:
            site, spin = divmod(spin_orbital, 2) 
            replacement = FermionOperator()
            for p in range(L):
                replacement+= FermionOperator(((2*p+spin, action),), C[site, p])
            new_term *= replacement
        H_orbitals += new_term
    return normal_ordered(H_orbitals)

def hartree_fock_state(n_qubits, n_electrons):
    """
    Generate a Hartree-Fock state in the orbital basis.
    Hartree_fock in orbital basis: the lowesn_eletron spin orbitals filled 
    (orbital 0 up and down , orbital 1 up and down, etc)
    Openfermion order: qubit 0 =leftmost bit.
    
    
    """
    bits ="1" * n_electrons + "0" * (n_qubits - n_electrons)
    
    psi = np.zeros(2**n_qubits)
    psi[int(bits, 2)] = 1.0
    return psi

# existing work pints used for local test (T, U, V)
POINTS = [(1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (1.0, 10.0, 0.0), (0.05, 10.0, 0.0), (1.0, 10.0, 5.0)]

@pytest.mark.parametrize("t, U, V", POINTS)
def test_same_ground_state_energy_in_both_bases (t, U, V):
    """
    A change of basis dose not chage the energies: 
    
    E0 in sites = E0 in U=0 orbitals.
    """
    
    L, n_electrons, n_qubits =6,6,12
    
    H_sites = build_hubbard_hamiltonian(L, t, U, mode="uniform", V=V)
    _, C = hopping_orbitals(L, t)
    H_orbitals = to_orbital_basis(H_sites, C)
    
    E0_sites, _ = jw_get_ground_state_at_particle_number(get_sparse_operator(jordan_wigner(H_sites), n_qubits), n_electrons)
    E_orbitals, _ = jw_get_ground_state_at_particle_number(get_sparse_operator(jordan_wigner(H_orbitals), n_qubits), n_electrons)
    
    assert E0_sites == pytest.approx(E_orbitals, abs=1e-10)
    
@pytest.mark.parametrize("t,U", [(1.0, 1.0), (1.0, 4.0), (1.0, 10.0), (0.05, 10.0)])
def test_hartree_fock_energy(t, U):
    """
    Hartree-Fock energy for V=0, known on paper:
            kinetic: 2 electrons in each of the orbitals with energies -2t, -t, -t  ->  -8t
            on-site: each site has n_up = n_down = 1/2 on average                    ->  6 * U/4 = 1.5 U

        
        
        """
    
    L, n_electrons, n_qubits = 6,6,12
    
    energies, C= hopping_orbitals(L, t)
    assert np.allclose(energies[:3], [-2*t, -1*t, -t]) # lowest 3 orbitals: closed shell
    
    
    H_orbitals = to_orbital_basis(build_hubbard_hamiltonian(L, t, U, mode="uniform"), C)
    H_matrix = get_sparse_operator(jordan_wigner(H_orbitals), n_qubits)
    hf= hartree_fock_state(n_qubits, n_electrons)
    E_hf = hf@(H_matrix@hf)
    
    assert E_hf == pytest.approx(-8*t + 1.5*U, abs=1e-10)
    
    