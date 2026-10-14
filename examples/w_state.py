# W state on MIMIQ: one excitation shared equally by all n qubits.
#
#   |W_n> = (|10...0> + |01...0> + ... + |00...1>) / sqrt(n)
#
# Circuit (chain): X on qubit 0, then for i = 0 ... n-2
#   controlled-RY(theta_i) from qubit i to qubit i+1, then CX from qubit i+1 to qubit i.
# At step i the excitation stays on qubit i with probability cos^2(theta_i/2) and moves on with
# sin^2(theta_i/2). For equal weights qubit i keeps 1/(n-i) of what arrives:
#   cos^2(theta_i/2) = 1/(n-i)   ->   theta_i = 2 arccos(sqrt(1/(n-i)))   (= 2 arctan(sqrt(n-i-1)))
# n = 4: 120, 109.5, 90 degrees. The last angle is always 90 degrees.
#
# Run:  python examples/w_state.py

import numpy as np

import exaqt
import mimiqcircuits as mc


def w_circuit(n_qubits):
    """W state: one excitation shared equally by all n qubits (1 X + (n-1) x (CRY + CX))"""
    circuit = mc.Circuit()
    circuit.push(mc.GateX(), 0)                                  # excitation starts on qubit 0
    for i in range(n_qubits - 1):
        theta = 2 * np.arccos(np.sqrt(1 / (n_qubits - i)))      # keep 1/(n-i) of it on qubit i ...
        circuit.push(mc.GateCRY(theta), i, i + 1)                # ... pass the rest on to qubit i+1
        circuit.push(mc.GateCX(), i + 1, i)                      # clear qubit i when it moved
    return circuit


def w_vector(n_qubits):
    """exact W state, qubit 0 = leftmost bit (OpenFermion order)"""
    psi = np.zeros(2**n_qubits)
    for q in range(n_qubits):
        psi[1 << (n_qubits - 1 - q)] = 1.0
    return psi / np.sqrt(n_qubits)


def state_vector(circuit, n_qubits):
    """state vector from Exaqt, reordered so that qubit 0 is the leftmost bit"""
    state, _ = exaqt.evolve(exaqt.ExaqtState.zero(n_qubits), circuit)
    psi = np.asarray(state.q.amplitudes())
    return psi.reshape([2] * n_qubits).transpose(range(n_qubits - 1, -1, -1)).reshape(-1)


def main():
    n = 4
    angles = [np.degrees(2 * np.arccos(np.sqrt(1 / (n - i)))) for i in range(n - 1)]
    print(f"angles for n = {n}: " + ", ".join(f"{a:.1f}" for a in angles) + " degrees")
    for n in (4, 6, 20):
        fidelity = abs(np.vdot(w_vector(n), state_vector(w_circuit(n), n))) ** 2
        print(f"n = {n:2d}: fidelity with W = {fidelity:.12f}, gates = {len(w_circuit(n))}")


if __name__ == "__main__":
    main()
