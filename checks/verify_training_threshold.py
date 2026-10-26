"""Independent checks of the shared-budget training threshold and policy optimization.

Run from the package root: python3 checks/verify_training_threshold.py
Requires NumPy and SciPy. Checks the closed-form boundary against the original
allocation LP, and the finite-candidate design optimum against a dense grid.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, linprog, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_comparison import benchmark, critical_pressure, design_threshold, optimal_design


def allocation_drift(H, r, y, q, tau, b, delta, G0, g, K):
    """Independent original resource-allocation LP, with no drift formula."""
    T = q + r / (G0 + g * H)
    result = linprog([-y, -tau], A_ub=[[T, 1.0], [1.0, 0.0]],
                     b_ub=[b * H, K], bounds=[(0.0, None), (0.0, None)],
                     method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                                "dual_feasibility_tolerance": 1e-9})
    assert result.success, result.message
    return -result.fun - delta * H


def max_relative_drift(r, pars):
    """Search stock independently, using LP objective values only.

    No positive drift can occur above y*K/(delta-tau*b), since recognition
    produces at most y*K and all training uses at most b*H expert hours.
    Below Hs, a viable design's maximal per-capita reproduction is increasing;
    the stock grid nonetheless includes small stocks as a numerical check.
    """
    Hs = pars["y"] * pars["K"] / pars["delta"]
    upper = pars["y"] * pars["K"] / (pars["delta"] - pars["tau"] * pars["b"])
    grid = np.unique(np.r_[np.geomspace(Hs * 0.01, max(upper, Hs), 35), Hs, upper])
    fn = lambda H: allocation_drift(H, r, **pars) / H
    values = [fn(H) for H in grid]
    best = max(values)
    for i in range(len(grid)):
        left, right = max(i - 1, 0), min(i + 1, len(grid) - 1)
        if values[i] >= values[left] and values[i] >= values[right]:
            result = minimize_scalar(lambda H: -fn(H), bounds=(grid[left], grid[right]),
                                     method="bounded", options={"xatol": 1e-10})
            best = max(best, -result.fun)
    return best


rng = np.random.default_rng(20261025)
boundary_error = 0.0
threshold_error = 0.0
regimes = {"unchanged_threshold": 0, "training_improves_threshold": 0}
checked = []
for i in range(60):
    b, delta, K = rng.uniform(0.5, 1.5, 3)
    q = rng.uniform(1.0, 3.0)
    y = delta * q / b * rng.uniform(1.05, 2.5)
    tau = (delta / b) * (0.0 if i % 10 == 0 else rng.uniform(0.0, 0.97))
    pars = dict(y=y, q=q, tau=tau, b=b, delta=delta,
                G0=0.0 if i % 4 == 0 else rng.uniform(0.0, 1.0),
                g=b + rng.uniform(0.0, 2.0), K=K)
    threshold = critical_pressure(**pars)
    Hs = y * K / delta
    if tau > 0.0:
        alpha = delta - tau * b
        vertex = (pars["g"] * K * (y - tau * q) - alpha * pars["G0"]) / (2.0 * pars["g"] * alpha)
    else:
        vertex = Hs
    stock = max(Hs, vertex)
    regimes["training_improves_threshold" if vertex > Hs else "unchanged_threshold"] += 1
    equality = allocation_drift(stock, threshold, **pars)
    boundary_error = max(boundary_error, abs(equality))
    assert abs(equality) <= 1e-8 * max(1.0, delta * stock)
    assert max_relative_drift(threshold * (1.0 - 1e-4), pars) > 0.0
    assert max_relative_drift(threshold * (1.0 + 1e-4), pars) < 0.0
    # Recover a subset of thresholds with nested LP/stock optimization and
    # bisection in pressure. This routine never uses the closed-form vertex.
    if i < 12:
        found = brentq(lambda r: max_relative_drift(r, pars), threshold * 0.8,
                       threshold * 1.2, xtol=1e-9)
        err = abs(found - threshold) / max(1.0, threshold)
        threshold_error = max(threshold_error, err)
        assert err < 2e-7, (i, found, threshold, err, pars)
    checked.append(pars)
assert all(regimes.values())

# M<=0, including exact equality, cannot be rescued by subcritical training.
negative_count = 0
for i in range(30):
    pars = checked[i].copy()
    pars["y"] = pars["delta"] * pars["q"] / pars["b"] * (1.0 if i % 3 == 0 else rng.uniform(0.1, 0.99))
    assert critical_pressure(**pars) < 1e-12
    for H in (0.01, 0.5, 2.0, 20.0):
        assert allocation_drift(H, 1.0, **pars) < 0.0
        negative_count += 1

# The switching equality vertex=Hs is also handled without a discontinuity.
edge = dict(y=1.95, q=1.35, b=1.0, delta=1.0, G0=0.2, g=2.0, K=1.0)
switch_tau = (edge["K"] * edge["y"] + edge["delta"] * edge["G0"] / edge["g"]) / (
    2.0 * edge["b"] * edge["y"] * edge["K"] / edge["delta"]
    - edge["K"] * edge["q"] + edge["b"] * edge["G0"] / edge["g"])
baseline = critical_pressure(tau=0.0, **edge)
assert abs(critical_pressure(tau=switch_tau, **edge) - baseline) < 1e-12
assert abs(allocation_drift(edge["y"] * edge["K"] / edge["delta"], baseline,
                            tau=switch_tau, **edge)) < 1e-12


pars, exposures, workloads = benchmark()
grid_error = 0.0
for tau in np.linspace(0.0, 0.99, 100):
    z, value, _ = optimal_design(float(tau), pars, exposures, workloads)
    grid_values = np.array([design_threshold(float(zz), float(tau), pars, exposures, workloads)
                           for zz in np.linspace(0.0, exposures[-1], 20001)])
    assert value + 1e-10 >= np.max(grid_values)
    grid_error = max(grid_error, value - np.max(grid_values))
    assert value - np.max(grid_values) < 0.004
    assert abs(value - design_threshold(z, float(tau), pars, exposures, workloads)) < 1e-12

# Additional synthetic parameters retain the same finite candidate logic.
for i in range(30):
    trial = pars.copy()
    trial["eta"] = rng.uniform(1.5, 5.0)
    trial["G0"] = rng.uniform(0.0, 1.0)
    trial["kbar"] = rng.uniform(0.3, 1.0)
    tau = float(rng.uniform(0.0, 0.98))
    z, value, _ = optimal_design(tau, trial, exposures, workloads)
    coarse = max(design_threshold(float(zz), tau, trial, exposures, workloads)
                 for zz in np.linspace(0.0, exposures[-1], 10001))
    assert value + 1e-9 >= coarse

lines = [
    "Shared-budget training: sustainable-pressure and policy-choice verification",
    f"60 fixed-design thresholds checked against original allocation LPs; regimes: {regimes}",
    f"Largest LP drift error at the analytical threshold: {boundary_error:.3e}",
    "Positive LP drift below every threshold; globally searched negative per-capita drift above every threshold",
    f"12 independent nested LP/stock/pressure solutions: maximum relative threshold error {threshold_error:.3e}",
    f"M<=0 and subcritical training: {negative_count} strictly negative LP drift checks",
    "Zero training productivity, G0=0, M=0, and threshold-branch equality checks passed",
    f"100 training productivities: finite candidate optimum dominates 20,001-point design grid; largest discretization gap {grid_error:.3e}",
    "30 additional synthetic learning-technology parameters: finite candidate optimum dominates 10,001-point grid",
    "ALL CHECKS PASSED",
]
report = "\n".join(lines) + "\n"
(ROOT / "checks" / "training_threshold_validation.txt").write_text(report)
print(report, end="")
