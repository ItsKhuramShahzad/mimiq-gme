# Entanglement witnesses for GHZ and W states up to 500 qubits on the MIMIQ cloud (MPS simulator),
# everything estimated from measurement samples, as on a real device.
#
# Three measurement settings per state: all qubits in the Z, X and Y basis.
#   GHZ witness (Toth & Guehne, PRL 94, 060501 (2005)):  B = 2 P_Z + <X^N>,  biseparable => B <= 2,
#       P_Z = P(0...0) + P(1...1) from the Z samples, <X^N> = parity of the X samples.  B > 2 => GME.
#   Spin-squeezing inequalities (Toth, Knapp, Guehne, Briegel, PRL 99, 250405 (2007)), J_k = 1/2 sum_i sigma_k^i:
#       (a) <Jx^2> + <Jy^2> + <Jz^2>          <= N(N+2)/4
#       (b) (dJx)^2 + (dJy)^2 + (dJz)^2        >= N/2
#       (c) <Jk^2> + <Jl^2> - N/2              <= (N-1)(dJm)^2
#       (d) (N-1)[(dJk)^2 + (dJl)^2]           >= <Jm^2> + N(N-2)/4
#     every separable state satisfies all four (any permutation k, l, m); a violation => entangled
#     (not necessarily GME). <J_k>, <J_k^2> are the sample means of J_k and J_k^2 in basis k.
# Every margin gets a bootstrap standard error; "violated" means margin > 3 standard errors.
# Exact entropy across the contiguous cuts (MPS, no sampling) is compared with the analytic value.
# Noise: optional dephasing (Z with probability q) on every qubit after the state preparation.
#
# Run:  python examples/ghz_w_witnesses_cloud.py               (needs a valid ~/.mimiq/qperfect_token.json)
#       python examples/ghz_w_witnesses_cloud.py --sizes 10 50 --shots 1000

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import mimiqcircuits as mc

TOKEN = os.path.expanduser("~/.mimiq/qperfect_token.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "witness")
RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------------------------------
# states
# ------------------------------------------------------------------------------------------
def ghz_circuit(n):
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.GateCX(), 0, range(1, n))
    return c


def w_circuit(n):
    """|W> = (|10...0> + ... + |0...01>)/sqrt(N): cascade of CRY(2 arccos sqrt(1/(N-i))) and CX"""
    c = mc.Circuit()
    c.push(mc.GateX(), 0)
    for i in range(n - 1):
        c.push(mc.GateCRY(2 * math.acos(math.sqrt(1 / (n - i)))), i, i + 1)
        c.push(mc.GateCX(), i + 1, i)
    return c


STATES = {"GHZ": ghz_circuit, "W": w_circuit}


def measured(make, n, basis, q):
    c = make(n)
    # dephasing noise. It commutes with a Z-basis measurement, so it cannot change the Z samples and is left out
    # there (exact). This also avoids a MIMIQ cloud error seen for N = 200 ("invalid argument #4 to LAPACK call",
    # GHZ + PauliZ(0.01) on every qubit + Z measurement, MPS; N = 100 and the X, Y bases run fine).
    if q > 0 and basis != "Z":
        for k in range(n):
            c.push(mc.PauliZ(q), k)
    for k in range(n):
        if basis == "Y":
            c.push(mc.GateSDG(), k)
        if basis in ("X", "Y"):
            c.push(mc.GateH(), k)
    c.push(mc.Measure(), range(n), range(n))
    return c


# ------------------------------------------------------------------------------------------
# estimators
# ------------------------------------------------------------------------------------------
def spins(samples, n):
    """per shot: J = 1/2 sum_i sigma_i with sigma = +1 for bit 0, -1 for bit 1; also the parity and all-equal flag"""
    bits = np.array([[c == "1" for c in s[:n]] for s in samples], dtype=np.int8)
    ones = bits.sum(axis=1)
    return {"J": 0.5 * (n - 2 * ones), "parity": 1 - 2 * (ones % 2), "alleq": (ones == 0) | (ones == n)}


def margins(data, n):
    """violation margin of each test (> 0 means violated); uses the strongest permutation for (c), (d)"""
    m1 = {k: data[k]["J"].mean() for k in "XYZ"}
    m2 = {k: (data[k]["J"] ** 2).mean() for k in "XYZ"}
    var = {k: m2[k] - m1[k] ** 2 for k in "XYZ"}
    perms = list(itertools.permutations("XYZ"))
    out = {
        "B": 2 * data["Z"]["alleq"].mean() + data["X"]["parity"].mean(),
        "a": sum(m2.values()) - n * (n + 2) / 4,
        "b": n / 2 - sum(var.values()),
        "c": max(m2[k] + m2[l] - n / 2 - (n - 1) * var[m] for k, l, m in perms),
        "d": max(m2[m] + n * (n - 2) / 4 - (n - 1) * (var[k] + var[l]) for k, l, m in perms),
    }
    out["B_margin"] = out["B"] - 2
    return out, m1, m2, var


def bootstrap(data, n, n_boot=300):
    """standard error of every margin, resampling the shots of each basis independently"""
    draws = []
    for _ in range(n_boot):
        res = {k: {f: v[idx] for f, v in data[k].items()}
               for k in "XYZ" for idx in [RNG.integers(0, len(data[k]["J"]), len(data[k]["J"]))]}
        draws.append(margins(res, n)[0])
    return {key: float(np.std([d[key] for d in draws])) for key in draws[0]}


def predicted(state, n, q):
    """analytic values (dephasing q): GHZ witness margin, (c) margin; exact C_GME of the pure state"""
    f = (1 - 2 * q)
    if state == "GHZ":
        return {"B_margin": f**n, "c": 0.0, "C_GME": 1.0}
    return {"B_margin": -2.0 if n >= 3 else None, "c": (n - 1) * f**2, "C_GME": 2 * math.sqrt(n - 1) / n}


# ------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------
def run_witnesses(conn, state, n, q, shots):
    circuits = [measured(STATES[state], n, b, q) for b in "ZXY"]
    t = time.time()
    job = conn.submit(circuits, algorithm="mps", nsamples=shots, bonddim=16, label=f"witness-{state}-{n}-q{q}")
    results = conn.get_results(job)
    for r in results:
        if not hasattr(r, "cstates"):                      # the cloud returned an error object
            raise RuntimeError(f"{type(r).__name__}: {r}")
    data = {b: spins([s.to01() for s in r.cstates], n) for b, r in zip("ZXY", results)}
    m, m1, m2, var = margins(data, n)
    se = bootstrap(data, n)
    return {"state": state, "N": n, "q": q, "shots": shots, "seconds": time.time() - t,
            "margins": m, "se": se, "J": m1, "J2": m2, "var": var, "predicted": predicted(state, n, q)}


def run_cut_entropy(conn, state, n):
    """exact von Neumann entropy (bits) of the first k qubits vs the rest, k = 1..N-1, from the MPS
    (VonNeumannEntropy pushed on qubit k measures the cut between qubit k-1 and qubit k)"""
    c = STATES[state](n)
    for k in range(1, n):
        c.push(mc.VonNeumannEntropy(), k, k - 1)
    job = conn.submit(c, algorithm="mps", nsamples=1, bonddim=16, label=f"entropy-{state}-{n}")
    z = np.real(conn.get_result(job).zstates[0])[: n - 1]
    k = np.arange(1, n)                                    # size of the left block
    if state == "GHZ":
        exact = np.ones(n - 1)
    else:
        p = k / n
        exact = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return {"state": state, "N": n, "entropy": z.tolist(), "exact": exact.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    ap.add_argument("--shots", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = mc.MimiqConnection(mc.QPERFECT_CLOUD)
    conn.loadtoken(TOKEN)

    path = os.path.join(OUT_DIR, f"ghz_w_witnesses_mimiq-cloud_shots{args.shots}.json")
    saved = json.load(open(path)) if os.path.exists(path) else {"witnesses": [], "cuts": [], "failed": []}
    rows, cuts, failed = saved["witnesses"], saved["cuts"], saved.get("failed", [])
    done = {(r["state"], r["N"], r["q"]) for r in rows}
    done_cuts = {(c["state"], c["N"]) for c in cuts}

    def save():
        with open(path, "w") as fh:
            json.dump({"witnesses": rows, "cuts": cuts, "failed": failed}, fh, default=float)

    for state in STATES:
        for n in args.sizes:
            for q in args.noise:
                if (state, n, q) in done:
                    continue
                try:
                    r = run_witnesses(conn, state, n, q, args.shots)
                except Exception as error:
                    failed.append({"state": state, "N": n, "q": q, "error": str(error)[:500]})
                    print(f"{state:3} N={n:3d} q={q:.2f} | FAILED: {str(error)[:300]}", flush=True)
                    save()
                    continue
                rows.append(r)
                save()
                m, se = r["margins"], r["se"]
                flag = lambda key: "VIOL" if m[key] > 3 * se[key] else "ok"
                print(f"{state:3} N={n:3d} q={q:.2f} | B = {m['B']:.3f} ± {se['B']:.3f} "
                      f"({'GME' if m['B_margin'] > 3 * se['B_margin'] else 'not det.'}) | "
                      f"(a) {flag('a')} (b) {flag('b')} (c) {m['c']:+.1f} ± {se['c']:.1f} {flag('c')} (d) {flag('d')} "
                      f"| pred B-2 = {r['predicted']['B_margin']}, (c) = {r['predicted']['c']:.1f} | {r['seconds']:.0f} s",
                      flush=True)
            if (state, n) in done_cuts:
                continue
            cuts.append(run_cut_entropy(conn, state, n))
            save()
            err = np.max(np.abs(np.array(cuts[-1]["entropy"]) - np.array(cuts[-1]["exact"])))
            print(f"{state:3} N={n:3d} cut entropies (MPS, {n - 1} cuts): max |S - S_exact| = {err:.1e}", flush=True)

    save()
    print(f"saved {path}")


if __name__ == "__main__":
    main()
