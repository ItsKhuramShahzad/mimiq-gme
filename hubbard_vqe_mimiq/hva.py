"""
VQE for the half-filled Hubbard ring with the Hamiltonian variational ansatz (HVA) on MIMIQ.
(Wecker, Hastings, Troyer, PRA 92, 042303 (2015); Cade et al., PRB 102, 235122 (2020).)

    |Psi(theta)> = prod_{l=1..p}  e^{-i theta_odd^l H_odd}  e^{-i theta_even^l H_even}  e^{-i theta_U^l H_U}  |Psi_{U=0}>

  |Psi_{U=0}>  exact ground state at U = 0 on the sites: Hartree-Fock in the hopping orbitals (X gates)
               followed by the rotation to the sites (core.rotation_to_sites, tested to fidelity 1)
  H_U          sum_k n_{k,up} n_{k,dn}                          (only Z strings: all commute)
  H_even       hopping on the bonds (1,2), (3,4), (5,6)          (disjoint bonds: all terms commute)
  H_odd        hopping on the bonds (2,3), (4,5), (6,1)          (the ring needs L even)
  H_V          sum_k n_k n_{k+1}, one more layer with theta_V when V != 0 (Z strings, commutes with H_U)

Inside every layer all Pauli strings commute, so each exponential is a product of exact RPauli gates:
no Trotter error. The parameters are the layer angles: 3 per layer (4 with V). The model parameters t, U, V
enter only through the measured energy, the operators in the layers are taken with unit coefficients.
Qubit order (core.mode_index): "interleaved" (default, qubit 2k + spin, as core.py) or "blocked" (all
spin up, then all spin down: qubit spin L + k). In the blocked order a hop between neighbouring sites is
a hop between neighbouring qubits, with no Z string except across the ring bond; with native_hopping=True
such a hop is applied as CX - CRX - CX, which is exactly exp(-i theta/4 (XX + YY)) (push_hop_native).
The physics does not depend on the order: for the same angles both orders give the same state and energy.
At theta = 0 the circuit is the U = 0 ground state, so E(0) = E_HF.
"""

import time

import numpy as np
from openfermion import FermionOperator, jordan_wigner

import mimiqcircuits as mc

from hubbard_vqe_mimiq.core import (hubbard_hamiltonian, exact_ground_state, mimiq_hamiltonian, hopping_orbitals,
                              rotation_to_sites, state_vector, with_progress, mode_index, to_layout)
from src.mimiq_backend import make_energy_fn
from src.mimiq_driver import vqe_until_converged


# ------------------------------------------------------------------------------------------
# the layer operators (unit coefficients) and their Pauli strings
# ------------------------------------------------------------------------------------------
def onsite_operator(L):
    """H_U = sum_k n_{k,up} n_{k,dn}"""
    return sum((FermionOperator(((2 * k, 1), (2 * k, 0))) * FermionOperator(((2 * k + 1, 1), (2 * k + 1, 0)))
                for k in range(L)), FermionOperator())


def neighbour_operator(L):
    """H_V = sum_k n_k n_{k+1} (ring)"""
    n = lambda k: FermionOperator(((2 * k, 1), (2 * k, 0))) + FermionOperator(((2 * k + 1, 1), (2 * k + 1, 0)))
    return sum((n(k) * n((k + 1) % L) for k in range(L)), FermionOperator())


def hopping_operator(L, bonds):
    """sum over the given bonds (k, k+1) and both spins of (c+_k c_{k+1} + h.c.), coefficient +1"""
    H = FermionOperator()
    for k in bonds:
        j = (k + 1) % L
        for s in (0, 1):
            H += FermionOperator(((2 * k + s, 1), (2 * j + s, 0))) + FermionOperator(((2 * j + s, 1), (2 * k + s, 0)))
    return H


def pauli_terms(fermion_op):
    """Jordan-Wigner Pauli strings as (letters, qubits, real coefficient); the identity (a global phase) is dropped"""
    terms = []
    for term, coeff in jordan_wigner(fermion_op).terms.items():
        if not term or abs(coeff) < 1e-14:
            continue
        terms.append(("".join(p for _, p in term), [q for q, _ in term], float(np.real(coeff))))
    return terms


def layer_operators(L, with_V=False, layout="interleaved"):
    """the commuting groups of one HVA layer, in the order they are applied (operators in the given qubit order)"""
    if L % 2:
        raise ValueError("the even/odd bond split needs an even number of sites L")
    groups = [("U", onsite_operator(L))]
    if with_V:
        groups.append(("V", neighbour_operator(L)))
    groups += [("even", hopping_operator(L, range(0, L, 2))), ("odd", hopping_operator(L, range(1, L, 2)))]
    groups = [(name, to_layout(op, L, layout)) for name, op in groups]
    return [(name, op, pauli_terms(op)) for name, op in groups]


def push_exp(circuit, terms, angle):
    """exp(-i angle sum_m c_m P_m) for commuting P_m: one RPauli(P_m, 2 angle c_m) per string, exact"""
    for letters, qubits, c in terms:
        circuit.push(mc.RPauli(mc.PauliString(letters), 2 * angle * c), *qubits)


def push_hop_native(circuit, q1, q2, phi):
    """exp(-i phi (X_q1 X_q2 + Y_q1 Y_q2)) as CX(q1, q2) - CRX(4 phi)(control q2, target q1) - CX(q1, q2):
    one number-conserving two-qubit block (the hop of one electron between two neighbouring qubits)"""
    circuit.push(mc.GateCX(), q1, q2)
    circuit.push(mc.GateCRX(4 * phi), q2, q1)
    circuit.push(mc.GateCX(), q1, q2)


def push_hopping_native(circuit, terms, angle):
    """exp(-i angle sum_m c_m P_m) for a hopping group: every XX + YY pair on two neighbouring qubits as one
    native block, the remaining strings (the ring bond, with its Z string) as RPauli gates"""
    pairs, rest = {}, []
    for letters, qubits, c in terms:
        if letters in ("XX", "YY") and abs(qubits[1] - qubits[0]) == 1:
            pairs.setdefault(tuple(qubits), {})[letters] = c
        else:
            rest.append((letters, qubits, c))
    for (q1, q2), cs in pairs.items():
        if set(cs) == {"XX", "YY"} and abs(cs["XX"] - cs["YY"]) < 1e-14:
            push_hop_native(circuit, q1, q2, angle * cs["XX"])
        else:
            rest += [(letters, [q1, q2], c) for letters, c in cs.items()]
    push_exp(circuit, rest, angle)


# ------------------------------------------------------------------------------------------
# the ansatz
# ------------------------------------------------------------------------------------------
def hva_ansatz(L, t, n_layers, with_V=False, layout="interleaved", native_hopping=False):
    """ansatz(n_qubits, n_electrons, params) -> circuit, as make_energy_fn expects.
    params = [theta_U^1, (theta_V^1), theta_even^1, theta_odd^1, theta_U^2, ...]
    layout: qubit order ("interleaved" or "blocked"); native_hopping: hops between neighbouring qubits as
    CX - CRX - CX blocks instead of two RPauli gates (same unitary)."""
    _, C = hopping_orbitals(L, t)
    rotation = rotation_to_sites(C, layout)
    groups = layer_operators(L, with_V, layout)
    per_layer = len(groups)

    def ansatz(n_qubits, n_electrons, params):
        params = np.asarray(params, dtype=float)
        if params.shape != (per_layer * n_layers,):
            raise ValueError(f"expected {per_layer * n_layers} parameters, got {params.shape}")
        circuit = mc.Circuit()
        for p in range(n_electrons // 2):                  # Hartree-Fock in the hopping orbitals
            for spin in (0, 1):
                circuit.push(mc.GateX(), mode_index(p, spin, L, layout))
        circuit.append(rotation)                           # -> exact U = 0 ground state on the sites
        for layer in range(n_layers):
            for g, (name, _, terms) in enumerate(groups):
                angle = params[layer * per_layer + g]
                if native_hopping and name in ("even", "odd"):
                    push_hopping_native(circuit, terms, angle)
                else:
                    push_exp(circuit, terms, angle)
        return circuit

    ansatz.per_layer = per_layer
    return ansatz


def run_vqe_hva(L=6, t=1.0, U=1.0, V=0.0, n_layers=4, seed=12345, max_cycles=25, chunk_maxiter=600, eps_E=1e-8,
                patience=3, progress_every=0, theta_start=None, layout="interleaved", native_hopping=False):
    """HVA VQE for one point (t, U, V) at half filling. Start: theta = 0 (the U = 0 ground state) unless given.
    The fidelity is computed in the same qubit order as the circuit."""
    n_qubits, n_electrons = 2 * L, L
    H = to_layout(hubbard_hamiltonian(L, t, U, V), L, layout)
    E0, psi0 = exact_ground_state(H, n_qubits, n_electrons)
    H_mimiq, constant, n_strings = mimiq_hamiltonian(H)
    ansatz = hva_ansatz(L, t, n_layers, with_V=abs(V) > 0, layout=layout, native_hopping=native_hopping)
    n_params = ansatz.per_layer * n_layers

    energy_fn, _ = make_energy_fn(H_mimiq, constant=0.0, n_qubits=n_qubits, n_electrons=n_electrons,
                                  ansatz=ansatz, target="exaqt-cpu")
    theta0 = np.zeros(n_params) if theta_start is None else np.asarray(theta_start, dtype=float)
    E_start = energy_fn(np.zeros(n_params)) + constant
    if progress_every:
        print(f"HVA (p = {n_layers} layers): L = {L}, t = {t}, U = {U}, V = {V} (U/t = {U / t:g}): {n_qubits} qubits, "
              f"{n_params} parameters, {n_strings} Pauli strings | E0 (ED) = {E0:.6f}, E(U=0 state) = {E_start:.6f}",
              flush=True)
        energy_fn = with_progress(energy_fn, constant, E0, progress_every)

    start = time.time()
    out = vqe_until_converged(energy_fn, theta0, np.random.default_rng(seed), eps_E=eps_E, patience=patience,
                              max_cycles=max_cycles, chunk_maxiter=chunk_maxiter, verbose_cycles=bool(progress_every))
    runtime = time.time() - start

    E_vqe = out["E_nc_opt"] + constant
    psi = state_vector(ansatz(n_qubits, n_electrons, out["theta_opt"]), n_qubits)
    psi_hf = state_vector(ansatz(n_qubits, n_electrons, np.zeros(n_params)), n_qubits)
    return {
        "ansatz": f"HVA p{n_layers}", "layout": layout, "native_hopping": native_hopping, "L": L, "t": t, "U": U, "V": V, "U_over_t": U / t,
        "n_qubits": n_qubits, "n_parameters": n_params, "n_pauli_strings": n_strings,
        "E0": E0, "E_HF": E_start, "E_VQE": E_vqe, "dE": E_vqe - E0,
        "rel_error": abs(E_vqe - E0) / abs(E0) if E0 != 0 else None,
        "correlation_recovered": (E_start - E_vqe) / (E_start - E0) if E_start != E0 else None,
        "fidelity": float(abs(np.vdot(psi0, psi)) ** 2), "fidelity_HF": float(abs(np.vdot(psi0, psi_hf)) ** 2),
        "n_evaluations": out["nfev"], "cycles": out["cycles"], "converged": out["converged"],
        "runtime_s": runtime, "theta_opt": out["theta_opt"],
        "best_energy_per_cycle": [e + constant for e in out["best_energy_per_cycle"]],
    }
