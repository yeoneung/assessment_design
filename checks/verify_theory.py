"""Independent numerical checks of the analytical assessment-design results.

Run from any working directory with NumPy and SciPy installed.
This verifies identities and optimizers. It is not an empirical calibration.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.optimize import linprog, minimize_scalar

L = 2.0
gaps = np.array([0.1, 0.25, 0.45])
work = np.array([0.06, 0.35, 2.2])
exposure = L * gaps

@lru_cache(maxsize=None)
def frontier(z):
    result = linprog(work,
        A_ub=np.array([-exposure, np.ones(len(work))]),
        b_ub=np.array([-z, 1.0]), bounds=[(0, None)] * len(work),
        method="highs")
    if not result.success:
        raise AssertionError(result.message)
    assert abs(exposure @ result.x - z) < 1e-7
    assert result.x.sum() <= 1.0 + 1e-8
    return result.fun, result.x

def evaluate(z, eta, b, delta, g, G0, K, kbar):
    C, _ = frontier(float(z))
    F = min(z / kbar, 1.0)
    M = eta * b * F - delta * (1.0 + C)
    return max(M, 0.0) / delta * (G0 + g * eta * F * K / delta)

def candidate_set(eta, b, delta, g, G0, K, kbar):
    Z = min(kbar, exposure.max())
    endpoints = sorted(set([0.0, Z] + [float(z) for z in exposure if z < Z]))
    candidates = endpoints.copy()
    w = g * eta * K / (delta * kbar)
    for lo, hi in zip(endpoints[:-1], endpoints[1:]):
        clo, chi = frontier(lo)[0], frontier(hi)[0]
        beta = (chi - clo) / (hi - lo)
        alpha = clo - beta * lo
        u = eta * b / kbar - delta * beta
        v = delta * (1.0 + alpha)
        if u < 0:
            critical = (v * w - u * G0) / (2.0 * u * w)
            if lo < critical < hi and u * critical - v > 0:
                candidates.append(critical)
    return Z, candidates

pars = dict(eta=3.12, b=1.0, delta=1.0, g=2.0, G0=0.2, K=1.0, kbar=0.8)
Z, candidates = candidate_set(**pars)
zstar = max(candidates, key=lambda z: evaluate(z, **pars))
C, probabilities = frontier(float(zstar))
F = zstar / pars["kbar"]
M = pars["eta"] * pars["b"] * F - pars["delta"] * (1.0 + C)
resilience = evaluate(zstar, **pars)
r = 1.5
Hc = (pars["delta"] * r / M - pars["G0"]) / pars["g"]
Hs = pars["eta"] * F * pars["K"] / pars["delta"]
v = pars["eta"] * pars["b"] * F
q = 1.0 + C
support = (math.sqrt(v * r) - math.sqrt(M * (q * pars["G0"] + r)))**2 / (pars["g"] * q * q)

def negative_drift(h):
    G = pars["G0"] + pars["g"] * h
    return pars["delta"] * h - v * h * G / (q * G + r)

numerical_support = minimize_scalar(lambda h: -negative_drift(h), bounds=(0, Hc), method="bounded")
assert abs(support + numerical_support.fun) < 1e-9
assert abs(zstar - 0.6509725906277629) < 1e-10
assert abs(resilience - 2.588892849248452) < 1e-9
print("Baseline independent linear-programming checks")
for name, value in [("optimal exposure", zstar), ("assessment work", C), ("learning share", F), ("net reproduction", M), ("resilience", resilience), ("unstable threshold", Hc), ("high equilibrium", Hs), ("training-support threshold", support)]:
    print(f"{name}: {value:.12f}")
print("challenge probabilities:", probabilities)
print("support formula agrees with direct scalar optimization")

rng = np.random.default_rng(13)
max_positive_gap = 0.0
for case in range(100):
    b = rng.uniform(0.2, 1.5)
    local = dict(eta=rng.uniform(0.5, 6.0), b=b, delta=rng.uniform(0.2, 2.0),
        g=max(b, rng.uniform(0.3, 4.0)), G0=rng.uniform(0, 1.0),
        K=rng.uniform(0.1, 2.0), kbar=rng.uniform(0.1, 2.0))
    Z, candidates = candidate_set(**local)
    finite_max = max(evaluate(z, **local) for z in candidates)
    dense_max = max(evaluate(float(z), **local) for z in np.linspace(0, Z, 151))
    assert dense_max <= finite_max + 1e-8, (case, dense_max, finite_max)
    max_positive_gap = max(max_positive_gap, finite_max - dense_max)
print("100 randomized parameter cases: finite candidate optimum weakly exceeds every dense-grid LP value")
print(f"Largest finite-minus-grid gap: {max_positive_gap:.12f}")

for z in np.linspace(0, Z, 31):
    c, _ = frontier(float(z))
    f = min(z / pars["kbar"], 1.0)
    m = pars["eta"] * pars["b"] * f - pars["delta"] * (1 + c)
    for h in np.linspace(0, 10, 41):
        G = pars["G0"] + pars["g"] * h
        raw = pars["b"] * h * G / ((1 + c) * G + r)
        direct = pars["eta"] * f * raw - pars["delta"] * h
        factored = h * (m * G - pars["delta"] * r) / ((1 + c) * G + r)
        assert abs(direct - factored) < 1e-9
print("Factored drift identity verified on 1271 state-policy pairs")
# Universal-learning rule: with concave F on [0, kbar] and convex piecewise affine C,
# the stopping inequality puts every maximizer of R strictly below kbar, and when
# it fails kbar maximizes R.
rng_c = np.random.default_rng(20261025)
agree_c = 0
cases_c = 0
for _ in range(3000):
    kb = rng_c.uniform(0.3, 1.0)
    zmax = kb * rng_c.uniform(1.0, 1.4)
    ph = rng_c.uniform(0.0, 0.3)
    q = rng_c.uniform(1.0, 3.0)
    nk = rng_c.integers(1, 4)
    knots = np.sort(rng_c.uniform(0, zmax, nk))
    slopes = np.sort(rng_c.uniform(0.2, 3.0, nk + 1))
    eta_, b_, dl_ = rng_c.uniform(1, 3), rng_c.uniform(1, 3), rng_c.uniform(0.2, 0.8)
    g_, G0_, K_ = rng_c.uniform(0.5, 4), rng_c.uniform(0, 1), rng_c.uniform(0.5, 3)
    zz = np.linspace(0, zmax, 40001)
    zz = np.sort(np.append(zz, kb))
    Fz = ph + (1 - ph) * np.minimum(zz / kb, 1.0) ** (1.0 / q)
    edges = np.concatenate([[0.0], knots, [zmax]])
    Cz = np.zeros_like(zz)
    for i in range(len(edges) - 1):
        Cz += slopes[i] * np.clip(zz - edges[i], 0, edges[i + 1] - edges[i])
    Mz = eta_ * b_ * Fz - dl_ * (1 + Cz)
    Rz = np.maximum(Mz, 0) / dl_ * (G0_ + g_ * eta_ * Fz * K_ / dl_)
    Ck = float(np.interp(kb, zz, Cz))
    Mk = eta_ * b_ - dl_ * (1 + Ck)
    if Mk <= 0:
        continue
    fm = (1 - ph) / (q * kb)
    bm = slopes[np.searchsorted(knots, kb, side='left')]
    lhs = dl_ * bm
    rhs = fm * (eta_ * b_ + g_ * eta_ * K_ * Mk / (dl_ * G0_ + g_ * eta_ * K_))
    if abs(lhs - rhs) < 0.02 * max(lhs, rhs):
        continue
    cases_c += 1
    Rk = Mk / dl_ * (G0_ + g_ * eta_ * K_ / dl_)
    if lhs > rhs:
        ok = (zz[np.argmax(Rz)] < kb) and (Rz.max() > Rk)
    else:
        ok = Rz.max() <= Rk * (1 + 1e-12)
    agree_c += bool(ok)
assert agree_c == cases_c, (agree_c, cases_c)
print(f"Universal-learning rule verified on {cases_c} random concave learning distributions and convex frontiers")
# AI-review comparison: if rho > delta/b and eta*b*F(z) <= delta*C(z), the critical
# pressure of the design is strictly increasing in the human time per inspection.
rng_d = np.random.default_rng(7)
cnt_d = 0
for _ in range(20000):
    eta_, b_, dl_, g_, G0_, K_ = rng_d.uniform(1, 3), rng_d.uniform(0.5, 2), rng_d.uniform(0.2, 1.0), rng_d.uniform(1, 4), rng_d.uniform(0, 1), rng_d.uniform(0.5, 3)
    rho_ = dl_ / b_ * rng_d.uniform(1.01, 3.0)
    Fz, Cz = rng_d.uniform(0, 1), rng_d.uniform(0, 3)
    if eta_ * b_ * Fz > dl_ * Cz:
        continue
    cgrid = np.linspace(b_ / g_, 1.0, 200) if b_ / g_ < 1.0 else None
    if cgrid is None:
        continue
    num = np.maximum(eta_ * b_ * Fz - dl_ * Cz - cgrid * (dl_ - rho_ * b_), 0.0)
    RA = num / (dl_ * cgrid) * (G0_ + g_ * (eta_ * Fz + rho_ * cgrid) * K_ / dl_)
    pos = RA > 0
    assert np.all(np.diff(RA[pos]) > 0)
    cnt_d += 1
print(f"AI-review comparison verified on {cnt_d} random designs with rho > delta/b")
# Feedback-policy result: some feedback rule is viable if and only if r < R*, and the
# positive set of the envelope drift is the union of the viable intervals (Hc, Hs).
rng_f = np.random.default_rng(314)
cnt_f = 0
for _ in range(60):
    Lf = rng_f.uniform(1.0, 3.0)
    gf = np.sort(rng_f.uniform(0.05, 0.6, 3))
    tf = np.sort(rng_f.uniform(0.02, 3.0, 3))
    hx = np.concatenate([[0.0], Lf * gf])
    hc_ = np.concatenate([[0.0], tf])
    # lower hull
    pts = list(zip(hx, hc_)); hull = []
    for pt in pts:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (pt[0] - hull[-2][0]) >= (pt[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(pt)
    hx = np.array([q[0] for q in hull]); hc_ = np.array([q[1] for q in hull])
    bf = rng_f.uniform(0.3, 1.5); gg = max(bf, rng_f.uniform(0.3, 4.0)); ef = rng_f.uniform(1.0, 6.0)
    df = rng_f.uniform(0.3, 2.0); G0f = rng_f.uniform(0.0, 1.0); Kf = rng_f.uniform(0.3, 2.0)
    kbf = hx[-1] * rng_f.uniform(0.2, 1.5)
    zg = np.linspace(0.0, hx[-1], 1501)
    Fz = np.clip(zg / kbf, 0, 1); Cz = np.interp(zg, hx, hc_); qz = 1 + Cz; Mz = ef * bf * Fz - df * qz
    Rz = np.maximum(Mz, 0) / df * (G0f + gg * ef * Fz * Kf / df)
    Rs = Rz.max()
    if Rs <= 1e-6:
        continue
    Hg = np.linspace(1e-6, 1.2 * ef * Kf / df, 1501)

    def Dst(r):
        G = G0f + gg * Hg[:, None]
        Xi = bf * Hg[:, None] * G / (qz[None, :] * G + r)
        return (ef * Fz[None, :] * np.minimum(Kf, Xi) - df * Hg[:, None]).max(axis=1)
    assert Dst(0.98 * Rs).max() > 0
    assert Dst(1.02 * Rs).max() <= 1e-12
    r0 = 0.5 * Rs
    Dp = Dst(r0) > 1e-12
    viab = Rz > r0
    Hc_ = np.maximum((df * r0 / np.where(Mz > 0, Mz, np.inf) - G0f) / gg, 0.0)
    Hs_ = ef * Fz * Kf / df
    union = np.zeros_like(Dp)
    for j in np.where(viab)[0]:
        union |= (Hg > Hc_[j]) & (Hg < Hs_[j])
    mism = np.mean(union != Dp)
    assert mism < 0.01, mism
    cnt_f += 1
print(f"Feedback-policy result verified on {cnt_f} random technologies")
print("ALL CHECKS PASSED")
