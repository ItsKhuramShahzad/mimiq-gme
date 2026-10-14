"""
Two tables for the HVA circuits of the Hubbard ring on MIMIQ:

1. Gate cost of the same HVA circuit in three versions (same state, same energy):
     interleaved qubit order + RPauli   (the circuits used for the VQE runs)
     blocked qubit order + RPauli       (all spin up, then all spin down: hops between neighbouring qubits)
     blocked + native CX-CRX-CX hops
2. Entanglement entropy (bits) of the first 1, 2, 3 sites with the rest of the ring, for the HVA ramp
   states and for the exact ground state (ED).

Run from the mimiq_gme folder (after the ramp has finished):
    python scripts/hubbard_vqe_mimiq_hardware_entropy.py              # HVA p = 8 ramp
    python scripts/hubbard_vqe_mimiq_hardware_entropy.py --layers 4
"""

import argparse
import glob
import os
import pickle
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mimiqcircuits as mc

from hubbard_vqe_mimiq.core import (RESULTS_DIR, hubbard_hamiltonian, exact_ground_state, state_vector, site_entropy)
from hubbard_vqe_mimiq.hva import hva_ansatz, layer_operators, push_exp, push_hopping_native

p = argparse.ArgumentParser()
p.add_argument("--L", type=int, default=6)
p.add_argument("--layers", type=int, default=8)
args = p.parse_args()
L, NQ, NE = args.L, 2 * args.L, args.L
VERSIONS = [("interleaved", False), ("blocked", False), ("blocked", True)]


def counts(circuit):
    gates = list(circuit.decompose())
    return len(gates), sum(1 for g in gates if g.num_qubits() == 2)


# 1. gate cost ------------------------------------------------------------------------------
theta = np.full(3 * args.layers, 0.3)
cost = {}
print(f"Gate cost, HVA L = {L}, p = {args.layers} (after decomposition into one- and two-qubit gates)\n")
print(f"{'version':32s} {'hopping layer':>16s} {'U layer':>10s} {'full circuit':>16s}")
print(f"{'':32s} {'total / 2-qubit':>16s} {'2-qubit':>10s} {'total / 2-qubit':>16s}")
for layout, native in VERSIONS:
    terms = {n: t for n, _, t in layer_operators(L, layout=layout)}
    hop = mc.Circuit()
    for name in ("even", "odd"):
        (push_hopping_native if native else push_exp)(hop, terms[name], 0.3)
    u_layer = mc.Circuit()
    push_exp(u_layer, terms["U"], 0.3)
    full = hva_ansatz(L, 1.0, args.layers, layout=layout, native_hopping=native)(NQ, NE, theta)
    cost[(layout, native)] = {"hopping_layer": counts(hop), "U_layer": counts(u_layer), "full": counts(full)}
    label = f"{layout} + {'native CX-CRX-CX' if native else 'RPauli'}"
    c = cost[(layout, native)]
    print(f"{label:32s} {c['hopping_layer'][0]:7d} / {c['hopping_layer'][1]:<6d} {c['U_layer'][1]:>10d} "
          f"{c['full'][0]:7d} / {c['full'][1]:<6d}")

# 2. entanglement entropy -------------------------------------------------------------------
ramp_dir = os.path.join(RESULTS_DIR, "ramp", f"hubbard_L{L}_t1_V0_hva-p{args.layers}-ramp_mimiq-exaqt")
rows = sorted((pickle.load(open(f, "rb")) for f in glob.glob(os.path.join(ramp_dir, "*.pkl"))),
              key=lambda r: r["U_over_t"])
if not rows:
    sys.exit(f"\nno ramp results in {ramp_dir}")
ansatz = hva_ansatz(L, 1.0, args.layers)
table = []
print(f"\nEntanglement entropy S (bits) of sites 1..m with the rest of the ring: HVA p = {args.layers} ramp vs ED\n")
print(f"{'U/t':>5} {'F':>8} | " + " | ".join(f"S(m={m}) ED    VQE    diff" for m in (1, 2, L // 2)))
for r in rows:
    _, psi0 = exact_ground_state(hubbard_hamiltonian(L, 1.0, r["U"]), NQ, NE)
    psi = state_vector(ansatz(NQ, NE, r["theta_opt"]), NQ)
    row = {"U_over_t": r["U_over_t"], "fidelity": r["fidelity"]}
    cells = []
    for m in (1, 2, L // 2):
        s_ed, s_vqe = site_entropy(psi0, L, m), site_entropy(psi, L, m)
        row[f"S{m}_ED"], row[f"S{m}_VQE"] = s_ed, s_vqe
        cells.append(f"{s_ed:9.4f} {s_vqe:6.4f} {s_vqe - s_ed:+.0e}")
    table.append(row)
    print(f"{r['U_over_t']:5.1f} {r['fidelity']:8.5f} | " + " | ".join(cells))

out = os.path.join(RESULTS_DIR, "analysis", f"hubbard_L{L}_t1_V0_hva-p{args.layers}-ramp_hardware-entropy_mimiq-exaqt.pkl")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as fh:
    pickle.dump({"gate_cost": cost, "entropy": table, "layers": args.layers, "L": L}, fh)
print(f"\nsaved {out}")
