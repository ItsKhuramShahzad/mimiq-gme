# ghzstate prepared with gates on mimiq then nbrdm form pauli strings then gme analysis
# ghz has known answer, so it checks the whole chian on a state by a circuit.

import sys
import numpy as np
import pytest
import mimiqcircuits as mc
import exaqt
sys.path.insert(0, "home/anjum/PhDWork/hubbard_gme/src")
sys.path.insert(0, "home/anjum/PhDWork/Strasbourg_Secondment/mimiq_vqe")

from hubbard_gme.convef_roof import C_GME_pure_dm, convex_roof_gcg

# helper already written and tested in test_hubbard_mimiq.py

from test_hubbard_mimiq import rdm_pauli_terms, rdm_from_expectation

def ghz_circuit(n_sites):
    """
    Circuit that prepares the GHZ spin state (|up up ... up> + |down down ... down>) / sqrt(2),
    one electron per site.
    Qubit layout as in hubbard_gme: site k spin up = qubit 2k, site k spin down = qubit 2k+1.
    Start: all qubits 0 (no electrons).
    """
    
    circuit = mc.Circuit()
    
    # qubits 0 (site 0 up) in superposition (|0> + |1>) / sqrt(2)
    circuit.push(mc.GateH(),0)
    
    # spin up on every other site copies qubit 0 -> branch all up
    
    for site in range(1, n_sites):
        circuit.push(mc.GateCX(), 0,2*site)    
        
    # spin down on every site is oppsote of qubit 0,  branch all down
    for site in range(n_sites):
        circuit.push(mc.GateCX(), 0,2*site+1)
        circuit.push(mc.GateX(), 2*site+1)
        
    return circuit