"""Shared-budget direct training: critical pressures and assessment choices.

Run: python3 training_comparison.py
Uses the unchanged synthetic benchmark in results.json. Requires NumPy.
All reported thresholds are suprema: a stable positive stock requires strict
inequality between routine pressure and the reported threshold.
"""
from pathlib import Path
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parent


def critical_pressure(y, q, tau, b, delta, G0, g, K):
    """Exact fixed-design threshold; q is one plus assessment workload."""
    if not (0.0 <= tau < delta / b):
        raise ValueError("Direct training must be subcritical: 0 <= tau < delta/b")
    M = b * y - delta * q
    if M <= 0.0:
        return 0.0
    Hs = y * K / delta
    baseline = M / delta * (G0 + g * Hs)
    if tau == 0.0:
        return baseline
    alpha = delta - tau * b
    vertex = (g * K * (y - tau * q) - alpha * G0) / (2.0 * g * alpha)
    if vertex <= Hs:
        return baseline
    return (g * K * (y - tau * q) + alpha * G0) ** 2 / (4.0 * g * alpha * tau * K)


def benchmark():
    data = json.loads((ROOT / "results.json").read_text())
    pars = data["parameters"]
    return pars, np.r_[0.0, pars["L"] * np.array(pars["d"])], np.r_[0.0, pars["t"]]


def design_threshold(z, tau, pars, exposures, workloads):
    y = pars["eta"] * min(max(z / pars["kbar"], 0.0), 1.0)
    q = 1.0 + float(np.interp(z, exposures, workloads))
    return critical_pressure(y, q, tau, **{k: pars[k] for k in ("b", "delta", "G0", "g", "K")})


def optimal_design(tau, pars, exposures, workloads):
    """Global optimum from a finite candidate set for uniform learning costs.

    On each linear piece of F and C, M and vertex-Hs are affine. Add their
    roots to the piece boundaries. The assessment-only branch is quadratic;
    the training-improved branch is the square of an affine function divided
    by a positive constant and is convex. Hence only endpoints and interior
    stationary points of concave assessment-only pieces need be checked.
    """
    b, delta, G0, g, K, eta, kbar = [pars[k] for k in ("b", "delta", "G0", "g", "K", "eta", "kbar")]
    if not 0.0 <= tau < delta / b:
        raise ValueError("Direct training must be subcritical")
    cuts = sorted(set([*map(float, exposures), min(kbar, float(exposures[-1]))]))
    candidates = list(cuts)
    alpha = delta - tau * b
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        q1 = float((np.interp(hi, exposures, workloads) - np.interp(lo, exposures, workloads)) / (hi - lo))
        q0 = 1.0 + float(np.interp(lo, exposures, workloads)) - q1 * lo
        y1, y0 = (eta / kbar, 0.0) if (lo + hi) / 2.0 < kbar else (0.0, eta)
        m1, m0 = b * y1 - delta * q1, b * y0 - delta * q0
        local = [lo, hi]
        if m1 != 0.0 and lo < -m0 / m1 < hi:
            local.append(-m0 / m1)
        if tau > 0.0:
            v1 = g * K * (y1 - tau * q1) - 2.0 * g * alpha * K * y1 / delta
            v0 = g * K * (y0 - tau * q0) - alpha * G0 - 2.0 * g * alpha * K * y0 / delta
            if v1 != 0.0 and lo < -v0 / v1 < hi:
                local.append(-v0 / v1)
        local.sort()
        candidates.extend(local)
        for left, right in zip(local[:-1], local[1:]):
            middle = (left + right) / 2.0
            if m1 * middle + m0 <= 0.0:
                continue
            if tau > 0.0 and v1 * middle + v0 > 0.0:
                continue  # convex training branch: endpoints suffice
            s1, s0 = g * K * y1 / delta, G0 + g * K * y0 / delta
            a2 = m1 * s1 / delta
            a1 = (m1 * s0 + m0 * s1) / delta
            if a2 < 0.0:
                stationary = -a1 / (2.0 * a2)
                if left < stationary < right:
                    candidates.append(stationary)
    candidates = sorted(set(candidates))
    vals = [design_threshold(z, tau, pars, exposures, workloads) for z in candidates]
    i = int(np.argmax(vals))
    return candidates[i], vals[i], candidates


def run():
    pars, exposures, workloads = benchmark()
    z_baseline, baseline_optimum, _ = optimal_design(0.0, pars, exposures, workloads)
    rows = []
    for tau in (0.0, 0.5, 0.8, 0.9, 0.95):
        z, threshold, _ = optimal_design(tau, pars, exposures, workloads)
        row = {
            "tau": tau,
            "training_only": 0.0,
            "task_2": design_threshold(0.5, tau, pars, exposures, workloads),
            "baseline_resilient": design_threshold(z_baseline, tau, pars, exposures, workloads),
            "cheapest_universal": design_threshold(pars["kbar"], tau, pars, exposures, workloads),
            "joint_optimum": threshold,
            "optimal_exposure": z,
            "gain_over_assessment_only_optimum": threshold / baseline_optimum - 1.0,
        }
        # All optimal benchmark designs are on the task-2/task-3 frontier.
        assert 0.5 - 1e-12 <= z <= 0.9 + 1e-12
        row["task_2_probability"] = (0.9 - z) / 0.4
        row["task_3_probability"] = (z - 0.5) / 0.4
        rows.append(row)
    output = {
        "notice": "All inputs are synthetic. Each column permits optimal allocation of the same expert-time budget. Strictly smaller pressure is required for a stable positive stock.",
        "parameters": pars,
        "baseline_resilient_exposure": z_baseline,
        "assessment_only_optimum": baseline_optimum,
        "rows": rows,
    }
    (ROOT / "training_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    with (ROOT / "training_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("tau    task 2   baseline design   universal   joint optimum   exposure")
    for row in rows:
        print("{tau:.4f}  {task_2:.4f}   {baseline_resilient:.4f}            {cheapest_universal:.4f}      {joint_optimum:.4f}          {optimal_exposure:.4f}".format(**row))


if __name__ == "__main__":
    run()
