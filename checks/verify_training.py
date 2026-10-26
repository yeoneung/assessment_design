"""Independent checks of shared-budget allocation (shared-budget direct training).

Run: python3 checks/verify_training.py
Requires NumPy and SciPy. Inputs are synthetic, not empirical estimates.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog


def formula(H, b, delta, G0, g, r, K, learning, work, tau):
    if H == 0:
        return 0.0
    time = 1.0 + work + r / (G0 + g * H)
    return (tau * b - delta) * H + max(learning - tau * time, 0.0) * min(K, b * H / time)


def allocation_lp(H, b, delta, G0, g, r, K, learning, work, tau):
    time = 1.0 + work + r / (G0 + g * H)
    result = linprog(
        [-learning, -tau],
        A_ub=[[time, 1.0], [1.0, 0.0]],
        b_ub=[b * H, K],
        bounds=[(0.0, None), (0.0, None)],
        method="highs",
    )
    assert result.success, result.message
    return -result.fun - delta * H, result.x, time


rng = np.random.default_rng(20261025)
max_error = 0.0
negative_cases = 0
regimes = {"training_only": 0, "recognition_time_limited": 0, "recognition_cap_limited": 0}
for i in range(200):
    b, delta = rng.uniform(0.3, 2.0, 2)
    work = rng.uniform(0.0, 3.0)
    if i < 100:
        learning = delta * (1.0 + work) / b * rng.uniform(0.0, 1.0)
        tau = delta / b * rng.uniform(0.0, 0.99)
    else:
        learning = rng.uniform(0.1, 8.0)
        tau = rng.uniform(0.0, 2.0)
    pars = dict(
        H=rng.uniform(0.01, 12.0), b=b, delta=delta,
        G0=0.0 if i % 5 == 0 else rng.uniform(0.0, 1.0),
        g=b + rng.uniform(0.0, 3.0), r=rng.uniform(0.01, 8.0),
        K=rng.uniform(0.1, 5.0), learning=learning, work=work, tau=tau,
    )
    closed = formula(**pars)
    lp, (recognized, trained), time = allocation_lp(**pars)
    max_error = max(max_error, abs(closed - lp))
    assert abs(closed - lp) < 1e-9
    assert time * recognized + trained <= b * pars["H"] + 1e-8
    assert recognized <= pars["K"] + 1e-8
    assert recognized <= pars["G0"] + pars["g"] * pars["H"] + 1e-8
    if learning < tau * time:
        regimes["training_only"] += 1
        assert recognized < 1e-8
    elif b * pars["H"] / time < pars["K"]:
        regimes["recognition_time_limited"] += 1
    else:
        regimes["recognition_cap_limited"] += 1
    if tau * b < delta and learning * b <= delta * (1.0 + work):
        negative_cases += 1
        assert closed < 0.0
assert all(count > 0 for count in regimes.values())
assert negative_cases >= 100

# Equality of the two productivity rates, and training with zero productivity.
edge = dict(H=2.0, b=1.0, delta=1.0, G0=0.0, g=2.0,
            r=1.0, K=1.0, learning=2.25, work=1.0, tau=1.0)
closed = formula(**edge)
lp, _, _ = allocation_lp(**edge)
assert abs(closed - lp) < 1e-10
edge["tau"] = 0.0
assert abs(formula(**edge) - allocation_lp(**edge)[0]) < 1e-10
assert formula(**dict(edge, H=0.0)) == 0.0

# The example in the proof: neither direct training alone nor assessed research
# alone is viable at this pressure, but their combination has a stable high stock.
example = dict(H=5.0, b=1.0, delta=1.0, G0=0.2, g=2.0,
               r=7.0, K=1.0, learning=3.0, work=1.0, tau=0.9)
M = example["learning"] * example["b"] - example["delta"] * (1.0 + example["work"])
Hs = example["learning"] * example["K"] / example["delta"]
critical = M / example["delta"] * (example["G0"] + example["g"] * Hs)
assert abs(critical - 6.2) < 1e-12
assert example["r"] > critical and example["tau"] * example["b"] < example["delta"]
at_five = formula(**example)
assert abs(at_five - 0.08235294117647074) < 1e-12
assert abs(at_five - allocation_lp(**example)[0]) < 1e-12
root = brentq(lambda h: formula(**dict(example, H=h)), 6.0, 10.0, xtol=1e-13)
assert abs(formula(**dict(example, H=root))) < 1e-11
left = allocation_lp(**dict(example, H=root - 0.01))[0]
right = allocation_lp(**dict(example, H=root + 0.01))[0]
assert left > 0.0 > right
_, (recognized, trained), _ = allocation_lp(**dict(example, H=root))
assert abs(recognized - example["K"]) < 1e-10 and trained > 0.0
derivative = -0.1 + 12.6 / (0.2 + 2.0 * root) ** 2
assert derivative < 0.0

lines = [
    "Shared-budget direct-training verification (shared-budget allocation)",
    f"200 independent allocation LPs: maximum drift discrepancy {max_error:.3e}",
    f"Allocation regimes: {regimes}",
    f"Subcritical training and nonpositive M: strictly negative drift in {negative_cases} cases",
    "Equal productivity, zero training productivity, and zero-stock cases passed",
    f"Example assessment-only threshold: {critical:.12f}",
    f"Example routine pressure: {example['r']:.12f}",
    f"Example drift at H=5: {at_five:.12f}",
    f"Example stable high stock: {root:.12f}",
    f"Example derivative at stable high stock: {derivative:.12f}",
    "Independent LPs verify positive drift below and negative drift above the high stock",
    "ALL CHECKS PASSED",
]
report = "\n".join(lines) + "\n"
Path(__file__).with_name("training_validation.txt").write_text(report)
print(report, end="")
