# Optimal spin-squeezing inequalities (Toth, Knapp, Guehne, Briegel, PRL 99, 250405 (2007)),
# with the collective-spin moments measured on MIMIQ (Exaqt) from Pauli strings only.
#
# Collective spin J_k = 1/2 sum_i sigma_k^i (k = x, y, z). Every separable N-qubit state satisfies
#   (a) <Jx^2> + <Jy^2> + <Jz^2>            <= N(N+2)/4     (holds for every qubit state)
#   (b) (dJx)^2 + (dJy)^2 + (dJz)^2          >= N/2          (violated by total-spin singlets)
#   (c) <Jk^2> + <Jl^2> - N/2                <= (N-1)(dJm)^2
#   (d) (N-1)[(dJk)^2 + (dJl)^2]             >= <Jm^2> + N(N-2)/4
#   for every permutation (k, l, m) of (x, y, z). Any violation => entangled.
#
# Measured: <J_k> = 1/2 sum_i <sigma_k^i>,  <J_k^2> = N/4 + 1/2 sum_{i<j} <sigma_k^i sigma_k^j>
#   -> 3N(N+1)/2 one- and two-qubit Pauli strings (630 at N = 20), not 4^N.
# Result: W violates (c) by N - 1; GHZ sits exactly on the boundary (not detected);
#         a total-spin singlet (Heisenberg ring) violates (b) by N/2.
#
# Run:  python examples/spin_squeezing.py

import itertools
import time

import numpy as np

import exaqt
import mimiqcircuits as mc

from ghz_entanglement import ghz_circuit
from w_state import w_circuit


def collective_moments(circuit, n_qubits):
    """<J_k> and <J_k^2> for k = X, Y, Z from one- and two-qubit Pauli strings measured on MIMIQ"""
    slots = []
    z = 0
    for k in "XYZ":
        for i in range(n_qubits):
            circuit.push(mc.ExpectationValue(mc.PauliString(k)), i, z)
            slots.append((k, 1, z)); z += 1
        for i, j in itertools.combinations(range(n_qubits), 2):
            circuit.push(mc.ExpectationValue(mc.PauliString(k + k)), i, j, z)
            slots.append((k, 2, z)); z += 1
    state, _ = exaqt.evolve(exaqt.ExaqtState.zero(n_qubits, num_zvars=z), circuit)
    m1 = {k: 0.5 * sum(state.z[s].real for kk, w, s in slots if kk == k and w == 1) for k in "XYZ"}
    m2 = {k: n_qubits / 4 + 0.5 * sum(state.z[s].real for kk, w, s in slots if kk == k and w == 2) for k in "XYZ"}
    return m1, m2, z


def check_inequalities(m1, m2, n):
    """value of each inequality and whether it is violated (strongest permutation for (c), (d))"""
    var = {k: m2[k] - m1[k] ** 2 for k in "XYZ"}
    perms = list(itertools.permutations("XYZ"))
    c = max(((m2[k] + m2[l] - n / 2, (n - 1) * var[m], f"{k}{l}|{m}") for k, l, m in perms), key=lambda t: t[0] - t[1])
    d = max(((m2[m] + n * (n - 2) / 4, (n - 1) * (var[k] + var[l]), f"{k}{l}|{m}") for k, l, m in perms),
            key=lambda t: t[0] - t[1])
    return var, {
        "a": (sum(m2.values()), n * (n + 2) / 4, sum(m2.values()) - n * (n + 2) / 4 > 1e-9),
        "b": (sum(var.values()), n / 2, n / 2 - sum(var.values()) > 1e-9),
        "c": (c[0], c[1], c[0] - c[1] > 1e-9, c[2]),
        "d": (d[1], d[0], d[0] - d[1] > 1e-9, d[2]),
    }


def singlet_check(n=6):
    """total-spin singlet: ground state of the Heisenberg ring, exact (violates (b) by N/2)"""
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    op = lambda P, i: np.kron(np.kron(np.eye(2**i), P), np.eye(2**(n - i - 1)))
    H = sum(op(P, i) @ op(P, (i + 1) % n) for i in range(n) for P in (X, Y, Z))
    psi = np.linalg.eigh(H)[1][:, 0]
    J = [sum(op(P, i) for i in range(n)) / 2 for P in (X, Y, Z)]
    variances = [np.real(psi.conj() @ Jk @ Jk @ psi) - np.real(psi.conj() @ Jk @ psi) ** 2 for Jk in J]
    print(f"singlet (Heisenberg ring, N = {n}, exact): sum of variances = {sum(variances):.2e} "
          f">= N/2 = {n / 2}?  violated by {n / 2 - sum(variances):.3f} -> entangled")


def main():
    for n in (4, 6, 20):
        for name, circuit in (("GHZ", ghz_circuit(n)), ("W", w_circuit(n))):
            start = time.time()
            m1, m2, n_strings = collective_moments(circuit, n)
            var, ineq = check_inequalities(m1, m2, n)
            verdict = "ENTANGLED" if any(v[2] for v in ineq.values()) else "not detected"
            print(f"N = {n:2d} {name:3} {n_strings} strings ({time.time() - start:.1f} s): "
                  f"variances = ({var['X']:.2f}, {var['Y']:.2f}, {var['Z']:.2f})")
            print(f"     (a) {ineq['a'][0]:.2f} <= {ineq['a'][1]:.2f} | (b) {ineq['b'][0]:.2f} >= {ineq['b'][1]:.2f} | "
                  f"(c) [{ineq['c'][3]}] {ineq['c'][0]:.2f} <= {ineq['c'][1]:.2f} | "
                  f"(d) [{ineq['d'][3]}] {ineq['d'][0]:.2f} >= {ineq['d'][1]:.2f}  => {verdict}")
    singlet_check()


if __name__ == "__main__":
    main()
