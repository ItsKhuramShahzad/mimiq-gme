# mimiq-gme

Genuine multipartite entanglement (GME) from quantum states prepared on the
[MIMIQ](https://qperfect.io) simulator.

On a quantum computer the state vector cannot be read out; only expectation values
⟨P⟩ of Pauli strings can be measured. This repository rebuilds the n-site reduced
density matrices (nBRDMs) of a state from such measurements alone, and passes them
to a classical convex-roof solver for the GME concurrence C_GME.

## How it works

Each element of the n-site RDM is a fermionic expectation value. The Jordan-Wigner
transformation turns it into a weighted sum of Pauli strings:

```
rho[s, s'] = < c†_{2 s1 + σ1} c_{2 s1 + σ1'} ... c†_{2 sn + σn} c_{2 sn + σn'} >
           = Σ_k c_k <P_k>
```

1. **Hamiltonian:** the Hubbard ring, mapped to qubits by Jordan-Wigner and converted to
   a MIMIQ Hamiltonian.
2. **State on MIMIQ:** the exact ground state is loaded into the Exaqt simulator (later:
   states prepared by VQE, and QPerfect's digital twin).
3. **Measurement:** every unique Pauli string needed by all clusters is measured once,
   in one MIMIQ circuit.
4. **Assembly:** each RDM element is summed from the measured values and the matrix is
   normalised to trace 1.
5. **GME:** the RDMs go to the GCG convex-roof solver (next step).

Conventions: site s has spin-orbitals 2s (up) and 2s+1 (down). MIMIQ counts qubit 0 as
the lowest bit, OpenFermion as the highest, so states are bit-reversed when loaded.

## Results so far

Half-filled Hubbard ring, t = 1, U = 4. Every quantity measured on MIMIQ equals the
reference computed directly from the state.

| Ring | Qubits | E0 (reference) | ⟨H⟩ on MIMIQ | Difference |
|---|---|---|---|---|
| L = 4 | 8 | −2.1027484835 | −2.1027484835 | 2e-15 |
| L = 6 | 12 | −3.6687061789 | −3.6687061789 | 3e-15 |

Largest difference between the RDM from MIMIQ measurements and the reference RDM,
over all clusters of n sites:

| Ring | n = 1 | n = 2 | n = 3 | n = 4 |
|---|---|---|---|---|
| L = 4 | 4e-15 | 4e-15 | 4e-15 | 4e-15 |
| L = 6 | 2e-16 | 3e-16 | 4e-16 | 3e-16 |

Unique Pauli strings measured for L = 6: 37 (n = 1), 577 (n = 2), 4,897 (n = 3),
24,337 (n = 4).

## VQE for the Hubbard ring

The ground state of the half-filled Hubbard ring (L = 6, 12 qubits) is prepared by VQE on
MIMIQ and compared with exact diagonalization (ED).

**Reports (start here):**
- [`docs/vqe_hubbard_mimiq_short.pdf`](docs/vqe_hubbard_mimiq_short.pdf): the procedure in
  short, with equations, results and references (4 pages).
- [`docs/vqe_hubbard_mimiq.pdf`](docs/vqe_hubbard_mimiq.pdf): every equation in detail.

**Procedure:** Jordan-Wigner mapping (42 Pauli strings) → starting state = U = 0 ground state
(Hartree-Fock in the hopping orbitals, rotated exactly to the sites) → ansatz (UCCSD, LUCJ or
the Hamiltonian variational ansatz, HVA) → energy ⟨H⟩ measured on MIMIQ → COBYLA.
For large U/t the HVA is started from the optimal angles of the previous U/t (warm start).

**Main result:** HVA with 8 layers (24 parameters) and the warm start reaches a fidelity
F ≥ 0.9992 with the exact ground state for U/t = 1 … 10.

| U/t | UCCSD (117 par.) F | LUCJ (130 par.) F | HVA p = 8, warm start F |
|---|---|---|---|
| 1 | 0.99997 | 0.9990 | 0.99954 |
| 10 | 0.942 | 0.634 | 0.99937 |

**Code:** `hubbard_vqe_mimiq/` (`core.py`: Hamiltonian, qubit mapping, orbitals, rotation,
UCCSD; `hva.py`; `lucj.py`), scripts in `scripts/hubbard_vqe_mimiq_*.py`.

**Tests** (132, all passing):

```bash
python -m pytest -q tests/test_hubbard_vqe_mimiq.py tests/test_hubbard_vqe_mimiq_strict.py \
                    tests/test_hubbard_vqe_mimiq_hardware.py
```

## Requirements

- Python 3.11 with `mimiqcircuits`, `mimiq-exaqt`, `openfermion`, `numpy`, `pytest`
- [mimiq-vqe](https://github.com/ItsKhuramShahzad/mimiq-vqe), for the conversion of
  OpenFermion operators to MIMIQ Hamiltonians
- `hubbard_gme`, the GME thesis code (Hamiltonian, reference RDMs, GCG solver), not yet
  public

The paths to both are set at the top of `tests/test_hubbard_mimiq.py` and
`hubbard_vqe_mimiq/core.py`; change them to your local copies.

## Run the tests

```bash
python -m pytest tests/test_hubbard_mimiq.py -v
```

| Test | Checks |
|---|---|
| `test_bit_order_random_state` | qubit order between OpenFermion and MIMIQ, on a random state |
| `test_hubbard_energy_on_mimiq` | ⟨H⟩ on MIMIQ equals the exact E0 (L = 4, 6) |
| `test_rdm_from_mimiq_pauli_strings` | RDMs from MIMIQ measurements equal the reference RDMs (L = 4, 6; n = 1 to 4) |

## Next steps

1. nBRDMs and C_GME of the VQE states (HVA, warm start), compared with ED.
2. Molecules (Benzene and the other benchmark molecules from mimiq-vqe).
3. Finite-shot measurements and grouping of commuting Pauli strings.
4. QPerfect's digital twin (a cheaper HVA circuit with the qubits ordered spin up first, then spin down, is ready).

## Author

Khuram Shahzad
