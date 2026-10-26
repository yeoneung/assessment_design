"""Synthetic assessment and training examples.

Run: python3 numerics.py
Requires NumPy, SciPy, Matplotlib, and PyMuPDF.
"""
from pathlib import Path
import csv
import json
import math
import subprocess

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, linprog, minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
(ROOT / "checks").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
NAVY = "#1B365D"
TEAL = "#2C7A7B"
OCHRE = "#C8902E"
BRICK = "#A23B2C"
SLATE = "#5C6B77"
GRAY = "#9AA5AE"
PALE = "#EEF1F4"
INK = "#1F2A33"
SOFT_TEAL = "#A9D1CB"
SOFT_OCHRE = "#F2DCA9"
SOFT_GRAY = "#E6EAED"
WIDTH = 6.5

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
L = 2.0
gaps = np.array([0.10, 0.25, 0.45])
p1 = np.array([0.90, 0.90, 0.90])
p0 = p1 - gaps
work = np.array([0.06, 0.35, 2.20])
x = np.concatenate([[0.0], L * gaps])          # exposures of hull vertices
costs = np.concatenate([[0.0], work])            # workloads of hull vertices
b, g, G0, eta, delta, K, kbar = 1.0, 2.0, 0.2, 3.12, 1.0, 1.0, 0.8
rbase = 1.5
H0_map = 1.5
H_compare = 1.8


def C(z, xn=x, cn=costs):
    """Assessment frontier. The menu vertices are already on the lower hull."""
    return np.interp(z, xn, cn)


def F(z, k=kbar):
    return np.clip(np.asarray(z, dtype=float) / k, 0.0, 1.0)


def M(z, xn=x, k=kbar, eta_=None):
    e = eta if eta_ is None else eta_
    return e * b * F(z, k) - delta * (1.0 + C(z, xn))


def Rcrit(z, xn=x, k=kbar, eta_=None):
    e = eta if eta_ is None else eta_
    return np.maximum(M(z, xn, k, e), 0.0) / delta * (G0 + g * e * F(z, k) * K / delta)


def optimal(xn=x, k=kbar, eta_=None):
    """Finite candidate set of finite-candidate result (th:finite), including the saturation point k."""
    e = eta if eta_ is None else eta_
    cuts = sorted(set([*xn.tolist(), *([k] if 0.0 < k < xn[-1] else [])]))
    cuts = [c for c in cuts if c <= xn[-1] + 1e-15]
    cand = list(cuts)
    w = g * e * K / (delta * k)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if 0.5 * (lo + hi) >= k:
            continue
        beta = (C(hi, xn) - C(lo, xn)) / (hi - lo)
        alpha = C(lo, xn) - beta * lo
        u = e * b / k - delta * beta
        v = delta * (1.0 + alpha)
        if u < 0:
            zc = (v * w - u * G0) / (2.0 * u * w)
            if lo < zc < hi and u * zc - v > 0:
                cand.append(zc)
    vals = np.array([float(Rcrit(z, xn, k, e)) for z in cand])
    j = int(np.argmax(vals))
    return float(cand[j]), float(vals[j]), cand


def X(h, z, r=rbase):
    G = G0 + g * np.asarray(h, dtype=float)
    return np.minimum(K, b * np.asarray(h) * G / ((1.0 + C(z)) * G + r))


def drift(h, z, r=rbase, s=0.0):
    return s + eta * F(z) * X(h, z, r) - delta * np.asarray(h, dtype=float)


def Hc(z, r=rbase):
    m = float(M(z))
    return (delta * r / m - G0) / g if m > 0 else float("inf")


def Hs(z, s=0.0):
    return (s + eta * float(F(z)) * K) / delta


def fmt2(v, _pos=None):
    return r"$%.2f$" % (0.0 if abs(v) < 5e-13 else v)


FMT = FuncFormatter(fmt2)


def style(ax, xt=None, yt=None, grid="y"):
    if xt is not None:
        ax.xaxis.set_major_locator(FixedLocator(xt))
    if yt is not None:
        ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.xaxis.set_major_formatter(FMT)
    ax.yaxis.set_major_formatter(FMT)
    if grid:
        ax.grid(axis=grid, color=PALE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    for sp in ax.spines.values():
        sp.set_color(INK)


def panel_title(ax, text):
    ax.set_title(text, loc="left", color=INK, pad=6)


def save(fig, name):
    path = FIG / f"{name}.pdf"
    fig.savefig(path, metadata={'CreationDate': None, 'Author': None, 'Creator': None, 'Producer': None})
    plt.close(fig)
    import fitz
    with fitz.open(path) as document:
        document[0].get_pixmap(matrix=fitz.Matrix(220/72, 220/72)).save(FIG / f"{name}.png")


# ---------------------------------------------------------------------------
# Baseline optimum.
# ---------------------------------------------------------------------------
zstar, rstar, candidates = optimal()
cs, ls, ms = float(C(zstar)), float(F(zstar)), float(M(zstar))
aC = (zstar - x[2]) / (x[3] - x[2])
aB = 1.0 - aC
hc_star, hs_star = Hc(zstar), Hs(zstar)
v_fix, q_fix = eta * b * ls, 1.0 + cs
m_fix = q_fix * G0 + rbase
s_crit = (math.sqrt(v_fix * rbase) - math.sqrt(ms * m_fix)) ** 2 / (g * q_fix ** 2)
y_tan = math.sqrt(v_fix * rbase * m_fix / ms)
h_tan = (y_tan - m_fix) / (q_fix * g)
support = 1.25 * s_crit
R_B, R_full, R_Cpure = float(Rcrit(x[2])), float(Rcrit(kbar)), float(Rcrit(x[3]))
gain = rstar / max(R_B, R_full) - 1.0

# Zero equilibrium changes stability at r = M G0 / delta for the optimal design.
r_zero_switch = ms * G0 / delta

# ===========================================================================

# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
fig.subplots_adjust(wspace=0.34)

ax = axs[0]
hull = Polygon(np.column_stack([[0, x[3], x[2], x[1]], [0, costs[3], costs[2], costs[1]]]),
               closed=True, facecolor=PALE, edgecolor="none", zorder=1)
ax.add_patch(hull)
ax.plot([0, x[3]], [0, costs[3]], color=GRAY, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.plot(x, costs, color=NAVY, lw=1.8, zorder=3, solid_capstyle="round")
ax.scatter(x, costs, s=22, color=NAVY, zorder=4, edgecolor="white", linewidth=0.6)
ax.axvline(kbar, color=GRAY, lw=0.7, ls=":", zorder=2)
ax.scatter([kbar], [C(kbar)], s=26, facecolor="white", edgecolor=NAVY, lw=1.0, zorder=5, marker="D")
ax.scatter([zstar], [cs], s=95, marker="*", color=OCHRE, edgecolor=INK, linewidth=0.5, zorder=6)
lab = {1: (r"task 1", (0.165, 0.21)), 2: (r"task 2", (0.345, 0.46)), 3: (r"task 3", (0.775, 2.24))}
for j, (t, pos) in lab.items():
    ax.text(*pos, t, fontsize=8, color=INK, ha="left")
ax.annotate(r"no test", (0.0, 0.0), xytext=(-0.005, 0.46), fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.annotate(r"$(z^{\star},C(z^{\star}))$", (zstar, cs), xytext=(0.17, 1.12),
            fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.annotate(r"full learning, $z=\bar{k}$", (kbar, C(kbar)), xytext=(0.43, 1.98),
            fontsize=7.6, color=INK, arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.text(0.455, 0.80, r"tasks $2{+}3$", fontsize=7.2, color=SLATE, rotation=0)
ax.set(xlim=(-0.02, 0.95), ylim=(-0.07, 2.42),
       xlabel=r"Incentive exposure $z$", ylabel=r"Expert time per recognized result")
style(ax, xt=[0, 0.3, 0.6, 0.9], yt=[0, 0.8, 1.6, 2.4])
panel_title(ax, r"(a) Assessment frontier $C(z)$")
# Inset: the inexpensive part of the frontier.
ins = ax.inset_axes([0.665, 0.07, 0.32, 0.25])
ins.fill([0, 0.5, 0.2, 0], [0, 0.35, 0.06, 0], color=PALE)
ins.plot(x[:3], costs[:3], color=NAVY, lw=1.4)
ins.scatter(x[:3], costs[:3], s=12, color=NAVY, zorder=4)
ins.plot([0.2, 0.2], [0, 0.06], color=GRAY, lw=0.5, ls=":")
ins.set(xlim=(-0.02, 0.55), ylim=(-0.02, 0.40))
ins.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.5]))
ins.yaxis.set_major_locator(FixedLocator([0, 0.35]))
ins.xaxis.set_major_formatter(FMT)
ins.yaxis.set_major_formatter(FMT)
ins.tick_params(labelsize=5.8, length=2, width=0.4)
for sp in ins.spines.values():
    sp.set_linewidth(0.4)
    sp.set_color(SLATE)
ins.spines["top"].set_visible(True)
ins.spines["right"].set_visible(True)
ins.set_facecolor("white")

ax = axs[1]
zz = np.linspace(0, x[3], 2401)
rv = Rcrit(zz)
viable = rv > rbase
ax.fill_between(zz, rv, color=TEAL, alpha=0.10, lw=0, zorder=1)
for lo, hi in [(x[1], x[2])]:
    ax.axvspan(lo, hi, color=PALE, alpha=0.0)
ax.plot(zz, rv, color=TEAL, lw=1.8, zorder=3)
ax.axhline(rbase, color=SLATE, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(0.015, rbase + 0.07, r"$r=%.2f$" % rbase, fontsize=7.6, color=SLATE)
zv = zz[viable]
ax.plot([zv.min(), zv.max()], [0.0, 0.0], color=TEAL, lw=4.0, solid_capstyle="butt", zorder=4, clip_on=False)
ax.text(0.5 * (zv.min() + zv.max()), 0.10, r"viable designs", fontsize=7.4, color=TEAL, ha="center")
ax.axvline(kbar, color=GRAY, lw=0.7, ls=":", zorder=2)
ax.scatter([x[2]], [R_B], s=24, color=NAVY, zorder=5, edgecolor="white", lw=0.6)
ax.scatter([kbar], [R_full], s=26, facecolor="white", edgecolor=NAVY, lw=1.0, zorder=5, marker="D")
ax.scatter([x[3]], [R_Cpure], s=26, color=BRICK, marker="X", zorder=5, lw=0.4, edgecolor="white")
ax.scatter([zstar], [rstar], s=95, marker="*", color=OCHRE, edgecolor=INK, lw=0.5, zorder=6)
ax.annotate(r"$\mathcal{R}^{\star}=%.2f$" % rstar, (zstar, rstar), xytext=(0.12, 2.86),
            fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.annotate(r"task 2", (x[2], R_B), xytext=(0.30, 2.15), fontsize=7.6, color=INK,
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.annotate(r"task 3", (x[3], 0.0), xytext=(0.80, 0.62), fontsize=7.6, color=BRICK,
            arrowprops=dict(arrowstyle="-", color=BRICK, lw=0.6))
ax.set(xlim=(0, 0.95), ylim=(0, 3.15), xlabel=r"Incentive exposure $z$",
       ylabel=r"Critical routine pressure $\mathcal{R}(z)$")
style(ax, xt=[0, 0.3, 0.6, 0.9], yt=[0, 1.0, 2.0, 3.0])
panel_title(ax, r"(b) Sustainable routine pressure")
save(fig, "fig2_frontier")

# ===========================================================================

# ===========================================================================
policies = [("No assessment", 0.0), (r"Task 1 only", x[1]), (r"Task 2 only", x[2]),
            (r"Task 3 only", x[3]), ("Cheapest full learning", kbar),
            ("Resilient design", zstar)]
plain_names = ["No assessment", "Task 1 only", "Task 2 only", "Task 3 only",
               "Cheapest full learning", "Resilient design"]
rows = []
for (name, z), plain in zip(policies, plain_names):
    mz = float(M(z))
    hcz = Hc(z) if mz > 0 else None
    rows.append(dict(policy=plain, z=float(z), assessment_cost=float(C(z)), learning_share=float(F(z)),
                     M=mz, rcrit=float(Rcrit(z)), Hc=hcz, Hs=Hs(z) if mz > 0 else None,
                     throughput=float(X(H_compare, z)), expertise_drift=float(drift(H_compare, z))))

fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.45), sharey=True)
fig.subplots_adjust(wspace=0.16, left=0.215)
ypos = np.arange(len(rows))[::-1]
cols = [GRAY, "#8FA6BA", NAVY, BRICK, SLATE, OCHRE]
specs = [("learning_share", r"(a) Learning share", (0, 1.36), [0, 0.5, 1.0], r"$F(z)$"),
         ("throughput", r"(b) Recognized flow", (0, 1.42), [0, 0.5, 1.0], r"$X(H,z)$"),
         ("expertise_drift", r"(c) Expertise drift", (-2.85, 0.85), [-2.0, 0.0], r"$\dot H$")]
for ax, (key, title, xl, xt, xlab) in zip(axs, specs):
    vals = np.array([rw[key] for rw in rows])
    base = 0.0
    for yy, v, c in zip(ypos, vals, cols):
        ax.plot([base, v], [yy, yy], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.scatter([v], [yy], s=30, color=c, zorder=3, edgecolor="white", lw=0.6)
        off = 0.035 * (xl[1] - xl[0]) * (1 if v >= 0 else -1)
        ax.text(v + off, yy, r"$%.2f$" % v, va="center",
                ha="left" if v >= 0 else "right", fontsize=7.0, color=INK)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlim(*xl)
    ax.set_xlabel(xlab)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    style(ax, xt=xt, grid="x")
    panel_title(ax, title)
axs[0].set_yticks(ypos)
axs[0].set_yticklabels([p[0] for p in policies])
save(fig, "fig3_policies")

# ===========================================================================

# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
fig.subplots_adjust(wspace=0.32)
ax = axs[0]
zs = np.linspace(0, x[3], 721)
rs = np.linspace(0, 3.15, 631)
ZZ, RR = np.meshgrid(zs, rs)
MM = M(ZZ)
rc = Rcrit(ZZ)
reg = np.zeros(ZZ.shape, dtype=int)
exists = (MM > 0) & (RR < rc)
recover = exists & (RR < MM * (G0 + g * H0_map) / delta)
reg[exists] = 1
reg[recover] = 2
ax.pcolormesh(ZZ, RR, reg, cmap=ListedColormap([SOFT_GRAY, SOFT_OCHRE, SOFT_TEAL]), shading="nearest",
              rasterized=True, vmin=0, vmax=2, zorder=0)
ax.plot(zs, Rcrit(zs), color=NAVY, lw=1.3, zorder=2)
rec_line = np.minimum(Rcrit(zs), np.maximum(M(zs), 0) * (G0 + g * H0_map) / delta)
ax.plot(zs, rec_line, color=TEAL, lw=1.0, ls=(0, (4, 2)), zorder=2)
ax.scatter(zstar, rbase, marker="*", s=90, color=OCHRE, edgecolor=INK, lw=0.5, zorder=4)
ax.annotate(r"$r=\mathcal{R}(z)$", (0.735, float(Rcrit(0.735))), xytext=(0.74, 2.88), fontsize=7.6,
            color=NAVY, arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.5))
ax.annotate(r"$H_c(z)=H_0$", (0.42, float(rec_line[np.argmin(abs(zs - 0.42))])), xytext=(0.16, 1.55),
            fontsize=7.6, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.5))
ax.set(xlim=(0, x[3]), ylim=(0, 3.15), xlabel=r"Incentive exposure $z$", ylabel=r"Routine pressure $r$")
style(ax, xt=[0, 0.3, 0.6, 0.9], yt=[0, 1.0, 2.0, 3.0], grid=None)
ax.legend(handles=[Patch(facecolor=SOFT_TEAL, label=r"recovery from $H_0$"),
                   Patch(facecolor=SOFT_OCHRE, label=r"attractor exists, $H_0<H_c$"),
                   Patch(facecolor=SOFT_GRAY, label=r"no positive attractor")],
          loc="upper left", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.0,
          handlelength=1.2, borderpad=0.4)
panel_title(ax, r"(a) Viability and recovery, $H_0=%.2f$" % H0_map)

ax = axs[1]
rgrid = np.linspace(1e-4, rstar, 600)
crit = (delta * rgrid / ms - G0) / g
ax.plot(rgrid, np.full_like(rgrid, hs_star), color=NAVY, lw=1.9, zorder=3)
pos = crit > 0
ax.plot(rgrid[pos], crit[pos], color=BRICK, lw=1.5, ls=(0, (4, 2)), zorder=3)
ax.plot([0, r_zero_switch], [0, 0], color=GRAY, lw=1.5, ls=(0, (4, 2)))
ax.plot([r_zero_switch, 3.15], [0, 0], color=GRAY, lw=1.9)
ax.scatter([rstar], [hs_star], s=34, facecolor="white", edgecolor=INK, lw=0.9, zorder=5)
ax.annotate(r"fold at $\mathcal{R}^{\star}$", (rstar, hs_star), xytext=(2.05, 3.0), fontsize=7.6,
            color=INK, arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.6))
ax.axvline(rbase, color=SLATE, lw=0.6, ls=":", zorder=1)
for y0, y1 in [(hc_star - 0.12, 0.18), (hc_star + 0.12, hs_star - 0.14), (3.05, hs_star + 0.14)]:
    ax.annotate("", xy=(rbase, y1), xytext=(rbase, y0),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=0.7, mutation_scale=7))
ax.scatter([rbase], [hc_star], color=BRICK, s=20, zorder=5)
ax.scatter([rbase], [hs_star], color=NAVY, s=20, zorder=5)
ax.text(rbase + 0.06, hc_star - 0.05, r"$H_c=%.2f$" % hc_star, fontsize=7.4, color=BRICK, va="top")
ax.text(rbase + 0.06, hs_star + 0.08, r"$H_s=%.2f$" % hs_star, fontsize=7.4, color=NAVY)
ax.set(xlim=(0, 3.15), ylim=(-0.12, 3.25), xlabel=r"Routine pressure $r$", ylabel=r"Expertise stock $H$")
style(ax, xt=[0, 1.0, 2.0, 3.0], yt=[0, 1.0, 2.0, 3.0])
ax.legend(handles=[Line2D([], [], color=NAVY, lw=1.9, label=r"stable, $H_s$"),
                   Line2D([], [], color=BRICK, lw=1.5, ls=(0, (4, 2)), label=r"unstable, $H_c$"),
                   Line2D([], [], color=GRAY, lw=1.9, label=r"zero stock")],
          loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=True, framealpha=0.92, edgecolor="none",
          fontsize=7.0, handlelength=1.8, borderpad=0.4)
panel_title(ax, r"(b) Equilibria at $z^{\star}$")
save(fig, "fig4_phase")

# ===========================================================================

# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
fig.subplots_adjust(wspace=0.30)
ax = axs[0]
hh = np.linspace(0, 3.1, 1200)
ax.axhline(0, color=INK, lw=0.6)
ax.plot(hh, drift(hh, zstar), color=NAVY, lw=1.7, label=r"no support")
ax.plot(hh, drift(hh, zstar, s=s_crit), color=OCHRE, lw=1.0, ls=(0, (2, 1.5)),
        label=r"$s=s_c=%.2f$" % s_crit)
ax.plot(hh, drift(hh, zstar, s=support), color=OCHRE, lw=1.7, label=r"$s=%.2f$" % support)
ax.scatter([hc_star], [0], color=BRICK, s=22, zorder=5)
ax.scatter([hs_star], [0], color=NAVY, s=22, zorder=5)
ax.scatter([h_tan], [0], facecolor="white", edgecolor=OCHRE, s=24, lw=1.0, zorder=5)
ax.annotate(r"$H_c$", (hc_star, 0), xytext=(hc_star - 0.05, -0.30), fontsize=7.8, color=BRICK, ha="center",
            arrowprops=dict(arrowstyle="-", color=BRICK, lw=0.5))
ax.annotate(r"$H_s$", (hs_star, 0), xytext=(hs_star + 0.18, 0.28), fontsize=7.8, color=NAVY,
            arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.5))
ax.annotate(r"tangency $H_{\ast}$", (h_tan, 0), xytext=(0.12, 0.42), fontsize=7.6, color=OCHRE,
            arrowprops=dict(arrowstyle="-", color=OCHRE, lw=0.5))
for xa, xb in [(1.05, 0.62), (1.75, 2.15), (2.95, 2.70)]:
    ax.annotate("", xy=(xb, -0.045), xytext=(xa, -0.045),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=0.7, mutation_scale=7))
ax.set(xlim=(0, 3.1), ylim=(-0.62, 0.66), xlabel=r"Expertise stock $H$",
       ylabel=r"Drift $\dot H$")
style(ax, xt=[0, 1.0, 2.0, 3.0], yt=[-0.5, 0, 0.5])
ax.legend(loc="lower left", frameon=False, fontsize=7.2, handlelength=2.0)
panel_title(ax, r"(a) Drift at $z^{\star}$, $r=%.2f$" % rbase)

ax = axs[1]
T = 32.0
times = np.linspace(0, T, 801)
paths = [(0.60, "#B8C4CF"), (1.20, "#8FA6BA"), (1.65, TEAL), (2.80, NAVY)]
for h0, col in paths:
    sol = solve_ivp(lambda t, y: [float(drift(y[0], zstar))], (0, T), [h0], t_eval=times,
                    rtol=1e-10, atol=1e-12)
    ax.plot(times, sol.y[0], color=col, lw=1.5, label=r"$H_0=%.2f$" % h0)
sol = solve_ivp(lambda t, y: [float(drift(y[0], zstar, s=support))], (0, T), [0.60], t_eval=times,
                rtol=1e-10, atol=1e-12)
ax.plot(times, sol.y[0], color=OCHRE, lw=2.0, label=r"$H_0=0.60$, $s=%.2f$" % support)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), frameon=True, framealpha=0.95, edgecolor="none",
          fontsize=7.0, handlelength=1.6, borderpad=0.4)
ax.axhline(hc_star, color=BRICK, lw=0.7, ls=(0, (4, 2)))
ax.text(1.0, hc_star + 0.07, r"$H_c=%.2f$" % hc_star, fontsize=7.2, color=BRICK)
ax.set(xlim=(0, T), ylim=(0, 3.0), xlabel=r"Time $t$", ylabel=r"Expertise stock $H_t$")
style(ax, xt=[0, 10.0, 20.0, 30.0], yt=[0, 1.0, 2.0, 3.0])
panel_title(ax, r"(b) Path dependence and recovery")
save(fig, "fig5_dynamics")

# ===========================================================================

# ===========================================================================
Agrid = np.linspace(0, 4, 321)
scenarios = [("learning faster", 0.4, 0.1, TEAL), ("matched", 0.3, 0.3, NAVY),
             ("imitation faster", 0.1, 0.4, BRICK)]
scenario_rows = []
curves = {}
for lab_, kap, chi, col in scenarios:
    vals, learn = [], []
    for A in Agrid:
        xx = x * math.exp(-chi * A)
        kk = kbar * math.exp(-kap * A)
        zo, ro, _ = optimal(xx, kk)
        vals.append(ro)
        learn.append(float(F(zo, kk)))
        scenario_rows.append(dict(scenario=lab_, kappa=kap, chi=chi, A=float(A), z=zo,
                                  C=float(C(zo, xx)), learning=float(F(zo, kk)), rcrit=ro))
    curves[lab_] = (np.array(vals), np.array(learn))


def Psi(theta):
    """Resilience as a function of effective discriminability theta (comparative statics)."""
    return optimal(x * theta, kbar)[1]


fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.15))
fig.subplots_adjust(wspace=0.42, hspace=0.52)
ax = axs[0, 0]
for lab_, kap, chi, col in scenarios:
    ax.plot(Agrid, curves[lab_][0], color=col, lw=1.7,
            label=r"%s, $(\kappa,\chi)=(%.2f,%.2f)$" % (lab_, kap, chi))
ax.plot(Agrid, rbase * np.exp(0.15 * Agrid), color=SLATE, lw=1.0, ls=(0, (4, 2)),
        label=r"routine flow $1.5000\,e^{0.1500A}$")
ax.set(xlim=(0, 4), ylim=(0, 14), xlabel=r"AI capability index $A$", ylabel=r"$\mathcal{R}^{\star}(A)$")
style(ax, xt=[0, 1.0, 2.0, 3.0, 4.0], yt=[0, 5.0, 10.0])
panel_title(ax, r"(a) Sustainable routine pressure")

ax = axs[0, 1]
imit = curves["imitation faster"][0]
dead = Agrid[imit <= 1e-12]
if dead.size:
    ax.axvspan(dead.min(), 4, color=SOFT_GRAY, lw=0, zorder=0)
    ax.text(0.5 * (dead.min() + 4), 0.52, r"no viable design" "\n" r"under imitation faster",
            fontsize=7.0, color=SLATE, ha="center")
for lab_, kap, chi, col in scenarios:
    ax.plot(Agrid, curves[lab_][1], color=col, lw=1.7)
ax.set(xlim=(0, 4), ylim=(0, 1.06), xlabel=r"AI capability index $A$",
       ylabel=r"Learning share $F_A(z^{\star}_A)$")
style(ax, xt=[0, 1.0, 2.0, 3.0, 4.0], yt=[0, 0.5, 1.0])
panel_title(ax, r"(b) Learning share at the optimum")

ax = axs[1, 0]
kgrid = np.linspace(0, 0.5, 101)
A_fix = 2.0
Rmap = np.zeros((kgrid.size, kgrid.size))
theta_cache = {}
for i, chi in enumerate(kgrid):
    for j, kap in enumerate(kgrid):
        key = round((kap - chi) * A_fix, 10)
        if key not in theta_cache:
            theta_cache[key] = optimal(x * math.exp(-chi * A_fix), kbar * math.exp(-kap * A_fix))[1]
        Rmap[i, j] = theta_cache[key]
cmap = LinearSegmentedColormap.from_list("res", ["#F6F3EC", SOFT_TEAL, TEAL, NAVY])
pcm = ax.pcolormesh(kgrid, kgrid, Rmap, cmap=cmap, shading="gouraud", rasterized=True)
cs_ = ax.contour(kgrid, kgrid, Rmap, levels=[1.0, 2.5889, 5.0, 8.0, 11.0], colors="white",
                 linewidths=0.6)
ax.plot([0, 0.5], [0, 0.5], color=INK, lw=0.7, ls=(0, (3, 2)))
ax.text(0.30, 0.335, r"$\kappa=\chi$", fontsize=7.4, color=INK, rotation=45)
for lab_, kap, chi, col in scenarios:
    ax.scatter([kap], [chi], s=30, color=col, edgecolor="white", lw=0.8, zorder=5)
cb = fig.colorbar(pcm, ax=ax, fraction=0.05, pad=0.03)
cb.ax.yaxis.set_major_locator(FixedLocator([0, 4.0, 8.0, 12.0]))
cb.ax.yaxis.set_major_formatter(FMT)
cb.ax.tick_params(labelsize=6.8, width=0.5, length=2)
cb.outline.set_linewidth(0.4)
ax.set(xlim=(0, 0.5), ylim=(0, 0.5), xlabel=r"Decline rate of learning costs $\kappa$",
       ylabel=r"Erosion rate of gaps $\chi$")
style(ax, xt=[0, 0.25, 0.5], yt=[0, 0.25, 0.5], grid=None)
panel_title(ax, r"(c) $\mathcal{R}^{\star}$ at $A=2.0000$")

ax = axs[1, 1]
rng = np.random.default_rng(7)
draws = []
for _ in range(220):
    kap, chi, A = rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0, 4)
    val = optimal(x * math.exp(-chi * A), kbar * math.exp(-kap * A))[1]
    draws.append(((kap - chi) * A, val))
draws = np.array(draws)
ugrid = np.linspace(-2, 2, 401)
master = np.array([Psi(math.exp(u)) for u in ugrid])
ax.plot(ugrid, master, color=INK, lw=1.2, zorder=2, label=r"$\Psi(e^{(\kappa-\chi)A})$")
ax.scatter(draws[:, 0], draws[:, 1], s=9, color=OCHRE, alpha=0.75, edgecolor="none", zorder=3,
           label=r"random $(\kappa,\chi,A)$")
ax.axvline(0, color=SLATE, lw=0.6, ls=":")
ax.set(xlim=(-2, 2), ylim=(0, 14), xlabel=r"Effective index $(\kappa-\chi)A$",
       ylabel=r"$\mathcal{R}^{\star}$")
style(ax, xt=[-2.0, -1.0, 0, 1.0, 2.0], yt=[0, 5.0, 10.0])
ax.legend(loc="upper left", frameon=False, fontsize=7.2, handletextpad=0.4)
panel_title(ax, r"(d) Collapse onto one curve")
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=2, frameon=False, fontsize=7.4,
           columnspacing=1.4, handlelength=2.2)
save(fig, "fig6_ai")

# ===========================================================================
# Independent verification.
# ===========================================================================
checks = {}
# Frontier against linear programming.
err = 0.0
for z in np.linspace(0, x[-1], 181):
    res = linprog(work, A_ub=np.vstack([-L * gaps, np.ones(3)]), b_ub=[-z, 1.0],
                  bounds=[(0, None)] * 3, method="highs")
    assert res.success
    err = max(err, abs(res.fun - C(z)))
    assert abs(L * gaps @ res.x - z) < 1e-8
checks["frontier_lp_max_error"] = err
# Finite candidate optimum against dense grids and bounded scalar search.
gap_grid, gap_scalar = 0.0, 0.0
for scale in [1.0, 0.8, 0.5, 0.3]:
    for kk in [0.15, 0.35, 0.6, 0.8, 1.0]:
        xx = x * scale
        zo, ro, _ = optimal(xx, kk)
        grid = np.linspace(0, xx[-1], 100001)
        brute = float(np.max(Rcrit(grid, xx, kk)))
        assert ro >= brute - 1e-9
        gap_grid = max(gap_grid, ro - brute)
        cuts = sorted(set([*xx.tolist(), *([kk] if 0 < kk < xx[-1] else [])]))
        best = max(float(Rcrit(v, xx, kk)) for v in cuts)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            sc = minimize_scalar(lambda v: -float(Rcrit(v, xx, kk)), bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-13})
            best = max(best, -sc.fun)
        gap_scalar = max(gap_scalar, abs(ro - best))
        assert abs(ro - best) < 1e-7
checks["optimizer_grid_gap"] = gap_grid
checks["optimizer_scalar_gap"] = gap_scalar
# Drift factorization and resource identity.
rng = np.random.default_rng(2048)
ident = 0.0
for _ in range(2000):
    h = float(rng.uniform(1e-3, 8))
    z = float(rng.uniform(0, x[-1]))
    r = float(rng.uniform(0.01, 3))
    G = G0 + g * h
    raw = b * h * G / ((1 + C(z)) * G + r)
    lhs = eta * F(z) * raw - delta * h
    rhs = h * (M(z) * G - delta * r) / ((1 + C(z)) * G + r)
    ident = max(ident, abs(lhs - rhs))
    assert abs(raw * ((G + r) / G + C(z)) - b * h) < 1e-10
checks["drift_identity_max_error"] = ident
# Critical training support against direct minimization of the uncapped drift.
mn = minimize_scalar(lambda h: float(eta * ls * b * h * (G0 + g * h) / (q_fix * (G0 + g * h) + rbase)
                                     - delta * h), bounds=(0, hc_star), method="bounded",
                     options={"xatol": 1e-13})
checks["support_threshold_error"] = abs(s_crit + mn.fun)
checks["tangency_location_error"] = abs(h_tan - mn.x)
assert checks["support_threshold_error"] < 1e-9
# Scale invariance (comparative statics): random (kappa, chi, A) against Psi.
inv = 0.0
for u_, val in draws:
    inv = max(inv, abs(val - Psi(math.exp(u_))))
checks["scale_invariance_max_error"] = inv
assert inv < 1e-9
# Monotonicity of Psi.
th = np.linspace(0.2, 3.5, 166)
ps = np.array([Psi(t) for t in th])
checks["psi_min_increment"] = float(np.min(np.diff(ps)))
assert np.all(np.diff(ps) >= -1e-12)
# Comparative statics of R* in eta (AI comparative statics(a)).
cmp_ok = True
base_r = rstar
for name, fn in [("eta", lambda e: optimal(eta_=e)[1])]:
    vals = [fn(e) for e in [3.0, 3.04, 3.08, 3.12, 3.16]]
    cmp_ok &= bool(np.all(np.diff(vals) > 0))
checks["eta_monotone"] = cmp_ok
assert abs(zstar - 0.6509725906277629) < 1e-12
assert abs(rstar - 2.588892849248452) < 1e-9

# Local sensitivity to expertise formation.
sens = []
for e in [3.00, 3.04, 3.08, 3.12, 3.16]:
    zo, ro, _ = optimal(eta_=e)
    rb, rf = float(Rcrit(x[2], eta_=e)), float(Rcrit(kbar, eta_=e))
    sens.append(dict(eta=e, z_star=zo, learning_share=float(F(zo)), rcrit=ro, task_B=rb,
                     full_learning=rf, gain=ro / max(rb, rf) - 1.0))

results = {
    "notice": "All numerical inputs are illustrative and synthetic.",
    "parameters": dict(L=L, d=gaps.tolist(), p1=p1.tolist(), p0=p0.tolist(), t=work.tolist(), b=b, g=g,
                       G0=G0, eta=eta, delta=delta, K=K, kbar=kbar, r=rbase, H0_map=H0_map,
                       H_compare=H_compare),
    "optimum": dict(z=zstar, C=cs, F=ls, M=ms, rcrit=rstar, a_B=aB, a_C=aC, Hc=hc_star, Hs=hs_star,
                    s_crit=s_crit, tangency=h_tan, support=support, Hs_supported=Hs(zstar, support),
                    r_zero_switch=r_zero_switch, payment_per_result=float(
                        L * (aB * (p0[1] + ls * gaps[1]) + aC * (p0[2] + ls * gaps[2]))),
                    learning_cost_per_author=float(ls * ls * kbar / 2.0)),
    "benchmarks": dict(R_taskB=R_B, R_full_learning=R_full, R_taskC=R_Cpure, C_full=float(C(kbar)),
                       gain_over_best_benchmark=gain),
    "policies": rows,
    "sensitivity": sens,
    "checks": checks,
}
(ROOT / "results.json").write_text(json.dumps(results, indent=2))
with (ROOT / "policy_comparison.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
with (ROOT / "ai_scenarios.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(scenario_rows[0]))
    wr.writeheader()
    wr.writerows(scenario_rows)
with (ROOT / "sensitivity.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(sens[0]))
    wr.writeheader()
    wr.writerows(sens)
(ROOT / "checks" / "numerical_checks.txt").write_text(json.dumps(checks, indent=2) + "\nAll checks passed.\n")
print(json.dumps(results["optimum"], indent=2))
print(json.dumps(results["benchmarks"], indent=2))
print(json.dumps(checks, indent=2))
for rw in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rw.items()})
for s_ in sens:
    print({k: round(v, 4) for k, v in s_.items()})


# ===========================================================================

# ===========================================================================
def lower_hull(px, py):
    """Lower convex hull of the menu points, from (0, 0) to the largest exposure."""
    pts = sorted(zip(px, py))
    hull = []
    for p in pts:
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
            hull.pop()
        hull.append(p)
    hx = np.array([q[0] for q in hull])
    hy = np.array([q[1] for q in hull])
    keep = np.r_[True, np.diff(hx) > 1e-14]
    return hx[keep], hy[keep]


def solve_exact(hx, hy, kb, phi, p):
    """Exact resilient and recovery designs under uniform costs with voluntary share phi
    (finite-candidate result (th:finite) and recovery-policy result (cor:recovery) with the coefficients in the supporting assessment computation)."""
    et, bb, gg, g0, kk, de = p["eta"], p["b"], p["g"], p["G0"], p["K"], p["delta"]
    zmax = hx[-1]
    Z = min(zmax, kb)
    Cf = lambda z: float(np.interp(z, hx, hy))
    Ff = lambda z: phi + (1.0 - phi) * min(z / kb, 1.0)
    Mf = lambda z: et * bb * Ff(z) - de * (1.0 + Cf(z))
    Rf = lambda z: max(Mf(z), 0.0) / de * (g0 + gg * et * Ff(z) * kk / de)
    cuts = sorted(set([float(v) for v in hx if v < Z - 1e-14] + [Z]))
    w = gg * et * kk * (1.0 - phi) / (de * kb)
    g0t = g0 + gg * et * kk * phi / de
    stat = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        beta = (Cf(hi) - Cf(lo)) / (hi - lo)
        alpha = Cf(lo) - beta * lo
        u = et * bb * (1.0 - phi) / kb - de * beta
        v = de * (1.0 + alpha) - et * bb * phi
        if u < 0:
            zc = (v * w - u * g0t) / (2.0 * u * w)
            if lo < zc < hi and u * zc - v > 0:
                stat.append(zc)
    cand = cuts + stat
    vals = [Rf(z) for z in cand]
    j = int(np.argmax(vals))
    zs_, rs_ = cand[j], vals[j]
    best_cut = max(Rf(z) for z in cuts)
    interior = (j >= len(cuts)) and rs_ > best_cut * (1 + 1e-10) + 1e-14
    mvals = [Mf(z) for z in cuts]
    zm = cuts[int(np.argmax(mvals))]
    return dict(z=zs_, R=rs_, F=Ff(zs_), M=Mf(zs_), interior=interior, best_cut=best_cut, zM=zm,
                Mmax=max(mvals), R0=Rf(0.0), Rmax=Rf(zmax), Rfull=(Rf(kb) if kb <= zmax else None),
                Rf=Rf, Mf=Mf)


base_p = dict(eta=eta, b=b, g=g, G0=G0, K=K, delta=delta)
# Consistency of the exact solver with the baseline optimizer.
chk = solve_exact(x, costs, kbar, 0.0, base_p)
assert abs(chk["z"] - zstar) < 1e-12 and abs(chk["R"] - rstar) < 1e-12

# (i) Voluntary learning in the benchmark.
phi_rows = []
for phi in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4]:
    s_ = solve_exact(x, costs, kbar, phi, base_p)
    hc_s = (delta * rbase / s_["M"] - G0) / g if s_["M"] > 0 else None
    hc_m = (delta * rbase / s_["Mmax"] - G0) / g if s_["Mmax"] > 0 else None
    phi_rows.append(dict(phi=phi, R0=s_["R0"], z_star=s_["z"], R_star=s_["R"], F_star=s_["F"],
                         interior=bool(s_["interior"]), z_M=s_["zM"], Hc_star=hc_s, Hc_M=hc_m,
                         R_task3=s_["Rmax"]))
phi_threshold = delta / (eta * b)

# (ii) Random technologies.
rng = np.random.default_rng(20261024)
N_MC = 4000
draws_mc = []
for _ in range(N_MC):
    J = int(rng.integers(2, 6))
    Lr = rng.uniform(1.0, 3.0)
    dr = np.sort(rng.uniform(0.05, 0.60, J))
    tr = np.sort(rng.uniform(0.02, 3.00, J))
    hx, hy = lower_hull(np.r_[0.0, Lr * dr], np.r_[0.0, tr])
    br = rng.uniform(0.3, 1.5)
    pr = dict(eta=rng.uniform(1.0, 6.0), b=br, g=max(br, rng.uniform(0.3, 4.0)), G0=rng.uniform(0.0, 1.0),
              K=rng.uniform(0.3, 2.0), delta=rng.uniform(0.3, 2.0))
    kbr = rng.uniform(0.2, 1.5) * hx[-1]
    draws_mc.append((hx, hy, kbr, pr))

mc_rows = []
ratio_strongest = []
for phi in [0.0, 0.1, 0.2]:
    n = n_feas = n_multi = 0
    cnt = dict(necessary=0, single_task=0, strongest_fails=0, strongest_universal_fails=0, infeasible=0,
               trap=0, interior=0, differ=0)
    cnt_feas = dict(full_fails_given_feasible=0, partial_given_feasible=0, strongest_fails_given_feasible=0)
    interior_multi = 0
    gains = []
    for hx, hy, kbr, pr in draws_mc:
        s_ = solve_exact(hx, hy, kbr, phi, pr)
        if s_["R"] <= 0:
            continue
        n += 1
        zmx = hx[-1]
        single = len(hx) == 2
        feasible = kbr <= zmx
        cnt["necessary"] += s_["R0"] <= 0
        cnt["single_task"] += single
        cnt["strongest_fails"] += s_["Rmax"] <= 0
        cnt["strongest_universal_fails"] += feasible and s_["Rmax"] <= 0
        cnt["infeasible"] += not feasible
        # At r = R(z)/2 the threshold equals (g H_s(z) - G0)/(2g), so a trap exists iff G0 < g H_s(z).
        hc_half = (pr["delta"] * 0.5 * s_["R"] / s_["M"] - pr["G0"]) / pr["g"]
        hs_mc = pr["eta"] * s_["F"] * pr["K"] / pr["delta"]
        assert (hc_half > 0) == (pr["G0"] < pr["g"] * hs_mc) or abs(pr["G0"] - pr["g"] * hs_mc) < 1e-12
        cnt["trap"] += hc_half > 0
        cnt["interior"] += bool(s_["interior"])
        cnt["differ"] += abs(s_["z"] - s_["zM"]) > 1e-9
        if not single:
            n_multi += 1
            interior_multi += bool(s_["interior"])
        if feasible:
            n_feas += 1
            cnt_feas["full_fails_given_feasible"] += s_["Rfull"] <= 0
            cnt_feas["partial_given_feasible"] += s_["F"] < 1 - 1e-12
            cnt_feas["strongest_fails_given_feasible"] += s_["Rmax"] <= 0
        if s_["interior"]:
            gains.append(s_["R"] / s_["best_cut"] - 1.0)
        if phi == 0.0:
            ratio_strongest.append(s_["Rmax"] / s_["R"])
    row = dict(phi=phi, viable=n, feasible=n_feas, multi_task=n_multi)
    row.update({k: v / n for k, v in cnt.items()})
    row.update({k: v / n_feas for k, v in cnt_feas.items()})
    row["interior_given_multi_task"] = interior_multi / n_multi
    row["median_interior_gain"] = float(np.median(gains)) if gains else None
    row["max_interior_gain"] = float(np.max(gains)) if gains else None
    mc_rows.append(row)
ratio_strongest = np.array(ratio_strongest)

# Check the exact solver against a dense grid on a subset of draws.
mc_grid_gap = 0.0
mc_m_gap = 0.0
for phi in [0.0, 0.2]:
    for hx, hy, kbr, pr in draws_mc[:300]:
        s_ = solve_exact(hx, hy, kbr, phi, pr)
        zg = np.linspace(0, hx[-1], 20001)
        rg = max(s_["Rf"](z) for z in zg)
        mg = max(s_["Mf"](z) for z in zg)
        assert s_["R"] >= rg - 1e-10 and s_["Mmax"] >= mg - 1e-10
        mc_grid_gap = max(mc_grid_gap, s_["R"] - rg)
        mc_m_gap = max(mc_m_gap, s_["Mmax"] - mg)
checks["mc_exact_minus_grid_max"] = mc_grid_gap
checks["mc_M_exact_minus_grid_max"] = mc_m_gap

# (iii) A cap on the expected payment per recognized result.
def frontier_budget(z, pibar):
    rhs = pibar - z * float(F(z))
    if rhs < -1e-12:
        return np.inf, None
    res = linprog(work, A_ub=np.vstack([-L * gaps, np.ones(3), L * p0]), b_ub=[-z, 1.0, max(rhs, 0.0)],
                  bounds=[(0, None)] * 3, method="highs")
    return (res.fun, res.x) if res.success else (np.inf, None)


def R_budget(z, pibar):
    cz, _ = frontier_budget(z, pibar)
    if not np.isfinite(cz):
        return 0.0
    mz = eta * b * float(F(z)) - delta * (1.0 + cz)
    return max(mz, 0.0) / delta * (G0 + g * eta * float(F(z)) * K / delta)


budget_rows = []
for pibar in [1.5, 1.3, 1.1, 0.9]:
    zg = np.linspace(0, x[3], 1801)
    rv_ = np.array([R_budget(z, pibar) for z in zg])
    i = int(np.argmax(rv_))
    if rv_[i] > 0:
        lo_, hi_ = zg[max(i - 1, 0)], zg[min(i + 1, len(zg) - 1)]
        sc = minimize_scalar(lambda z: -R_budget(z, pibar), bounds=(lo_, hi_), method="bounded",
                             options={"xatol": 1e-12})
        zb = sc.x if -sc.fun >= rv_[i] else zg[i]
        cb, ab = frontier_budget(zb, pibar)
        budget_rows.append(dict(pibar=pibar, z=zb, R=R_budget(zb, pibar), F=float(F(zb)), C=cb,
                                a1=ab[0], a2=ab[1], a3=ab[2], assessed=ab.sum()))
    else:
        budget_rows.append(dict(pibar=pibar, z=0.0, R=0.0, F=0.0, C=0.0, a1=0.0, a2=0.0, a3=0.0, assessed=0.0))
pay_learners = L * ls * p1[0]
pay_nonlearners = L * (1 - ls) * (aB * p0[1] + aC * p0[2])


share0 = float(np.mean(ratio_strongest <= 1e-12))
med = float(np.median(ratio_strongest))
robust = dict(phi_threshold=phi_threshold, voluntary=phi_rows, monte_carlo=mc_rows,
              monte_carlo_draws=N_MC, strongest_ratio_median=med, strongest_ratio_zero_share=share0,
              budget=budget_rows, payments_to_learners=pay_learners, payments_to_nonlearners=pay_nonlearners)
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
(ROOT / "checks" / "numerical_checks.txt").write_text(json.dumps(checks, indent=2) + "\nAll checks passed.\n")
print(json.dumps(robust, indent=2))


# ===========================================================================

# idea production, soft capacity, and learning depth.
# ===========================================================================
ext = {}

f_minus = 1.0 / kbar
beta_minus = (work[2] - work[1]) / (x[3] - x[2])
M_full = float(M(kbar))
stop_threshold = f_minus * (eta * b + g * eta * K * M_full / (delta * G0 + g * eta * K))
ext["stop_rule"] = dict(delta_beta=delta * beta_minus, threshold=stop_threshold,
                        overexposure_time=float(C(x[3]) - C(kbar)), M_full=M_full, M_task3=float(M(x[3])),
                        gain_vs_full=rstar / R_full - 1.0, gain_vs_task2=rstar / R_B - 1.0)

# AI-assisted selection: inspection time cI, training from refereeing rho.
def R_ai(z, cI, rho):
    f = float(F(z))
    ma = eta * b * f - delta * float(C(z)) - cI * (delta - rho * b)
    return max(ma, 0.0) * (G0 + g * (eta * f + rho * cI) * K / delta) / (delta * cI)


def best_ai(cI, rho):
    zg = np.linspace(0, x[3], 3601)
    vals = np.array([R_ai(z, cI, rho) for z in zg])
    i = int(np.argmax(vals))
    lo_, hi_ = zg[max(i - 1, 0)], zg[min(i + 1, len(zg) - 1)]
    sc = minimize_scalar(lambda z: -R_ai(z, cI, rho), bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-12})
    zb = sc.x if -sc.fun >= vals[i] else zg[i]
    return zb, R_ai(zb, cI, rho), (eta * float(F(zb)) + rho * cI) * K / delta


ai_rows = []
cI_grid = np.linspace(b / g, 1.0, 101)
ai_curves = {}
for rho in [0.0, 0.3, 0.6]:
    curve = [best_ai(cI, rho) for cI in cI_grid]
    ai_curves[rho] = curve
    for cI in [1.0, 0.75, 0.5]:
        zb, rb, hsb = best_ai(cI, rho)
        ai_rows.append(dict(rho=rho, cI=cI, z=zb, R=rb, Hs=hsb))
ext["ai_review"] = ai_rows
assert abs(best_ai(1.0, 0.0)[1] - rstar) < 1e-9


# Numerical viability for general drifts: stable positive equilibrium exists.
H_grid = np.linspace(1e-5, 15.0, 30001)


def stable_exists(dr):
    d = dr(H_grid)
    return bool(np.any((d[:-1] > 0) & (d[1:] <= 0)))


def crit_numeric(dr_of_r, r_hi=64.0):
    if not stable_exists(dr_of_r(1e-6)):
        return 0.0
    lo_, hi_ = 1e-6, 1.0
    while stable_exists(dr_of_r(hi_)) and hi_ < r_hi:
        hi_ *= 2
    for _ in range(50):
        mid = 0.5 * (lo_ + hi_)
        if stable_exists(dr_of_r(mid)):
            lo_ = mid
        else:
            hi_ = mid
    return lo_



ai_check = 0.0
for (z_, cI, rho) in [(0.651, 1.0, 0.0), (0.8, 0.75, 0.3), (0.7, 0.5, 0.6), (0.55, 0.6, 0.3)]:
    f_ = float(F(z_)); q_ = cI + float(C(z_))

    def dr_ai(r, f_=f_, q_=q_, cI=cI, rho=rho):
        def dd(H):
            G_ = G0 + g * H
            Xi = b * H * G_ / (q_ * G_ + cI * r)
            return (eta * f_ + rho * cI) * np.minimum(K, Xi) - delta * H
        return dd
    num = crit_numeric(dr_ai)
    ai_check = max(ai_check, abs(num - R_ai(z_, cI, rho)) / max(1.0, R_ai(z_, cI, rho)))
checks["ai_review_closed_form_rel_error"] = ai_check
assert ai_check < 1e-3


def R_gamma(z, gam):
    mz = float(M(z))
    hs = eta * float(F(z)) * K / delta
    return max(mz, 0.0) * (G0 + g * hs ** gam) / delta


gam_rows = []
zz_s = np.linspace(0, x[3], 1801)
for gam in [0.6, 1.0, 1.4]:
    Hh = np.linspace(0, 50, 50001)
    assert np.all(G0 + g * Hh ** gam >= np.minimum(K, b * Hh) - 1e-12)
    vals = np.array([R_gamma(z, gam) for z in zz_s])
    i = int(np.argmax(vals))
    gam_rows.append(dict(gamma=gam, z=float(zz_s[i]), R=float(vals[i]), R_task3=R_gamma(x[3], gam)))
ext["idea_production"] = gam_rows
# check general formula against direct dynamics for gamma = 0.6
gam_check = 0.0
for z_ in [0.5, 0.65, 0.8]:
    f_ = float(F(z_)); q_ = 1 + float(C(z_))

    def dr_g(r, f_=f_, q_=q_):
        def dd(H):
            G_ = G0 + g * H ** 0.6
            Xi = b * H * G_ / (q_ * G_ + r)
            return eta * f_ * np.minimum(K, np.minimum(G_, Xi)) - delta * H
        return dd
    gam_check = max(gam_check, abs(crit_numeric(dr_g) - R_gamma(z_, 0.6)) / max(1.0, R_gamma(z_, 0.6)))
checks["idea_production_closed_form_rel_error"] = gam_check
assert gam_check < 1e-3

P_SOFT = 4.0


def R_soft(z):
    f_ = float(F(z)); q_ = 1 + float(C(z))

    def dr_s(r):
        def dd(H):
            G_ = G0 + g * H
            Xi = b * H * G_ / (q_ * G_ + r)
            Xs = (Xi ** -P_SOFT + K ** -P_SOFT) ** (-1.0 / P_SOFT)
            return eta * f_ * Xs - delta * H
        return dd
    return crit_numeric(dr_s)


zz_soft = np.linspace(0, x[3], 181)
soft_vals = np.array([R_soft(z) for z in zz_soft])
i_soft = int(np.argmax(soft_vals))
ext["soft_capacity"] = dict(p=P_SOFT, z=float(zz_soft[i_soft]), R=float(soft_vals[i_soft]),
                            R_task2=R_soft(x[2]), R_full=R_soft(kbar), R_task3=R_soft(x[3]))

# Learning depth: eta(z) = eta (1 + zeta z / z_max).
def R_depth(z, zeta):
    e = eta * (1 + zeta * z / x[3])
    mz = e * b * float(F(z)) - delta * (1 + float(C(z)))
    return max(mz, 0.0) / delta * (G0 + g * e * float(F(z)) * K / delta)


zeta_viable = brentq(lambda s: eta * (1 + s) * b - delta * (1 + work[2]), 0.0, 1.0)
zeta_beats = brentq(lambda s: R_depth(x[3], s) - R_depth(kbar, s), 0.05, 5.0)


def right_slope_at_full(s, h=1e-7):
    return (R_depth(kbar + h, s) - R_depth(kbar, s)) / h


zeta_right = brentq(right_slope_at_full, 0.05, 5.0)
depth_rows = []
for zeta in [0.0, 0.2, zeta_right, zeta_beats, 1.5]:
    vals = np.array([R_depth(z, zeta) for z in zz_s])
    i = int(np.argmax(vals))
    depth_rows.append(dict(zeta=zeta, z=float(zz_s[i]), R=float(vals[i]), R_full=R_depth(kbar, zeta),
                           R_task3=R_depth(x[3], zeta)))
ext["depth"] = dict(zeta_task3_viable=zeta_viable, zeta_task3_beats_full=zeta_beats, zeta_beyond_need_pays=zeta_right,
                    eta_ratio_right=(1 + zeta_right) / (1 + zeta_right * kbar / x[3]),
                    eta_ratio_at_threshold=(1 + zeta_beats) / (1 + zeta_beats * kbar / x[3]), rows=depth_rows)


fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.35))
fig.subplots_adjust(wspace=0.32, hspace=0.50)
ax = axs[0, 0]
zz = np.linspace(0, x[3], 1801)
for phi, col in [(0.0, NAVY), (0.1, TEAL), (0.2, OCHRE), (0.35, BRICK)]:
    s_ = solve_exact(x, costs, kbar, phi, base_p)
    ax.plot(zz, [s_["Rf"](z) for z in zz], color=col, lw=1.5, label=r"$\varphi_0=%.2f$" % phi)
    ax.scatter([s_["z"]], [s_["R"]], marker="*", s=60, color=col, edgecolor=INK, lw=0.4, zorder=5)
    if s_["R0"] > 0:
        ax.scatter([0.0], [s_["R0"]], s=20, facecolor="white", edgecolor=col, lw=1.0, zorder=5, clip_on=False)
ax.axvline(kbar, color=GRAY, lw=0.6, ls=":")
ax.set(xlim=(0, x[3]), ylim=(0, 5.6), xlabel=r"Incentive exposure $z$", ylabel=r"$\mathcal{R}(z)$")
style(ax, xt=[0, 0.3, 0.6, 0.9], yt=[0, 2.0, 4.0])
ax.legend(loc="upper left", frameon=False, fontsize=7.0, handlelength=1.6)
panel_title(ax, r"(a) Voluntary learning share $\varphi_0$")

ax = axs[0, 1]
share0 = float(np.mean(ratio_strongest <= 1e-12))
pos = ratio_strongest[ratio_strongest > 1e-12]
n_zero = int(np.sum(ratio_strongest <= 1e-12))
bins = np.linspace(0, 1, 21)
ax.bar([-0.06], [n_zero], width=0.04, color=BRICK, alpha=0.85, zorder=2)
ax.hist(pos, bins=bins, color=SOFT_TEAL, edgecolor=TEAL, lw=0.6, zorder=2)
ax.text(-0.03, n_zero * 1.02, r"$%.2f$ at zero" % share0, fontsize=7.2, color=BRICK, va="bottom", ha="left")
med = float(np.median(ratio_strongest))
ax.axvline(med, color=NAVY, lw=0.9, ls=(0, (4, 2)))
ax.text(med + 0.02, 0.45 * n_zero, r"median $%.2f$" % med, fontsize=7.2, color=NAVY)
ax.set(xlim=(-0.09, 1.0), ylim=(0, n_zero * 1.15), xlabel=r"$\mathcal{R}(z_{\max})/\mathcal{R}^{\star}$, random technologies",
       ylabel=r"Number of draws")
ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
ax.xaxis.set_major_formatter(FMT)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: r"$%d$" % v))
ax.grid(axis="y", color=PALE, lw=0.6, zorder=0)
ax.set_axisbelow(True)
panel_title(ax, r"(b) The strongest single task")

ax = axs[1, 0]
for gam, col in [(0.6, TEAL), (1.0, NAVY), (1.4, OCHRE)]:
    vals = np.array([R_gamma(z, gam) for z in zz])
    ax.plot(zz, vals, color=col, lw=1.5, label=r"$G_0+gH^{%.2f}$" % gam)
    j = int(np.argmax(vals))
    ax.scatter([zz[j]], [vals[j]], marker="*", s=60, color=col, edgecolor=INK, lw=0.4, zorder=5)
ax.plot(zz_soft, soft_vals, color=BRICK, lw=1.3, ls=(0, (4, 2)), label=r"soft cap, $p=%.2f$" % P_SOFT)
ax.scatter([zz_soft[i_soft]], [soft_vals[i_soft]], marker="*", s=60, color=BRICK, edgecolor=INK, lw=0.4, zorder=5)
ax.axvline(kbar, color=GRAY, lw=0.6, ls=":")
ax.set(xlim=(0, x[3]), ylim=(0, 4.4), xlabel=r"Incentive exposure $z$", ylabel=r"$\mathcal{R}(z)$")
style(ax, xt=[0, 0.3, 0.6, 0.9], yt=[0, 2.0, 4.0])
ax.legend(loc="upper left", frameon=False, fontsize=7.0, handlelength=1.8)
panel_title(ax, r"(c) Idea production and capacity")

ax = axs[1, 1]
zd = np.linspace(0.3, x[3], 1201)
for (zeta, col) in zip([0.0, 0.2, zeta_right, 1.5], [NAVY, TEAL, OCHRE, BRICK]):
    base_ = R_depth(kbar, zeta)
    vals = np.array([R_depth(z, zeta) for z in zd]) / base_
    ax.plot(zd, vals, color=col, lw=1.5, label=r"$\zeta=%.2f$" % zeta)
    j = int(np.argmax(vals))
    ax.scatter([zd[j]], [vals[j]], marker="*", s=60, color=col, edgecolor=INK, lw=0.4, zorder=5)
ax.axvline(kbar, color=GRAY, lw=0.6, ls=":")
ax.axhline(1.0, color=GRAY, lw=0.6, ls=":")
ax.set(xlim=(0.3, x[3]), ylim=(0, 1.2), xlabel=r"Incentive exposure $z$", ylabel=r"$\mathcal{R}(z)/\mathcal{R}(\bar{k})$")
style(ax, xt=[0.3, 0.6, 0.9], yt=[0, 0.5, 1.0])
ax.legend(loc="lower right", bbox_to_anchor=(0.88, 0.0), frameon=False, fontsize=7.0, handlelength=1.6)
panel_title(ax, r"(d) Deeper learning under harder tests")
save(fig, "fig8_robustness")


fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
fig.subplots_adjust(wspace=0.32)
for rho, col in [(0.0, NAVY), (0.3, TEAL), (0.6, OCHRE)]:
    curve = ai_curves[rho]
    axs[0].plot(cI_grid, [c_[1] for c_ in curve], color=col, lw=1.6, label=r"$\rho=%.2f$" % rho)
    axs[1].plot(cI_grid, [c_[2] for c_ in curve], color=col, lw=1.6, label=r"$\rho=%.2f$" % rho)
axs[0].scatter([1.0], [rstar], marker="*", s=70, color=OCHRE, edgecolor=INK, lw=0.4, zorder=5)
axs[0].set(xlim=(1.0, b / g), ylim=(0, 18), xlabel=r"Human time per inspection $c_I$",
           ylabel=r"$\mathcal{R}^{\star}_A$, tolerable $R/\mathcal{L}_A$")
style(axs[0], xt=[1.0, 0.75, 0.5], yt=[0, 5.0, 10.0, 15.0])
axs[0].legend(loc="upper left", frameon=False, fontsize=7.2)
panel_title(axs[0], r"(a) Resilience")
axs[1].set(xlim=(1.0, b / g), ylim=(2.4, 3.9), xlabel=r"Human time per inspection $c_I$",
           ylabel=r"High stock $H_s$ at the resilient design")
style(axs[1], xt=[1.0, 0.75, 0.5], yt=[2.5, 3.0, 3.5])
panel_title(axs[1], r"(b) Level of expertise")
save(fig, "fig7_ai_review")

robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
(ROOT / "checks" / "numerical_checks.txt").write_text(json.dumps(checks, indent=2) + "\nAll checks passed.\n")
print(json.dumps(ext, indent=2))

# Thresholds for the effect of AI review on the high stock under reoptimization.
def z_star_ai(cI, rho):
    return best_ai(cI, rho)[0]


rho_full = brentq(lambda rh: z_star_ai(1.0, rh) - (kbar - 1e-9), 0.0, 0.6, xtol=1e-10)
rho_level = brentq(lambda rh: best_ai(0.5, rh)[2] - best_ai(1.0, rh)[2], 0.05, rho_full, xtol=1e-10)
ext["ai_review_thresholds"] = dict(rho_universal_at_cI1=rho_full, rho_stock_equal=rho_level,
                                   Hs_rho01=(best_ai(1.0, 0.1)[2], best_ai(0.5, 0.1)[2]),
                                   Hs_rho02=(best_ai(1.0, 0.2)[2], best_ai(0.5, 0.2)[2]),
                                   max_interior_gain_phi0=mc_rows[0]["max_interior_gain"])
robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
print(json.dumps(ext["ai_review_thresholds"], indent=2))


# Joint condition under AI: a ten percent erosion of every gap with learning costs
# fixed, alone and together with halved human inspection time (rho = 0).
def best_joint(sd, cI):
    xs_ = x * sd
    zg = np.linspace(0, xs_[-1], 2001)

    def Rj(z):
        f = float(F(z))
        ma = eta * b * f - delta * float(C(z, xs_)) - cI * delta
        return max(ma, 0.0) * (G0 + g * eta * f * K / delta) / (delta * cI)

    cuts = sorted(set([*xs_.tolist(), *([kbar] if kbar < xs_[-1] else [])]))
    best_ = max((Rj(z), z) for z in zg)
    for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
        for zc in (lo_, hi_):
            best_ = max(best_, (Rj(zc), zc))
        sc = minimize_scalar(lambda z: -Rj(z), bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-12})
        best_ = max(best_, (-sc.fun, sc.x))
    return best_


# Tolerated misstatement of the unlearned pass probabilities at the resilient design.
w_star = (zstar - x[2]) / (x[3] - x[2])
a_res = np.array([0.0, 1.0 - w_star, w_star])
c_res = float(a_res @ work)


def R_fixed(eps):
    z = zstar - L * eps * a_res.sum()
    f = float(F(z))
    return max(eta * b * f - delta * (1.0 + c_res), 0.0) / delta * (G0 + g * eta * f * K / delta)


ext["joint_ai"] = dict(R_gaps_minus10=best_joint(0.9, 1.0)[0], R_gaps_minus10_cI_half=best_joint(0.9, 0.5)[0],
                       eps_at_rbase=brentq(lambda e: R_fixed(e) - rbase, 0.0, 0.2, xtol=1e-12),
                       eps_at_M0=brentq(lambda e: eta * b * float(F(zstar - L * e)) - delta * (1 + c_res), 0.0, 0.3, xtol=1e-12))
robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
print(json.dumps(ext["joint_ai"], indent=2))


# Timing of the draw: reconstruction alone with a surprise draw versus a draw announced in advance.
a_surprise = kbar / x[3]
ext["draw_timing"] = dict(a_surprise=a_surprise, work_surprise=a_surprise * work[2], work_advance=float(work[2]),
                          share_advance_at_a_surprise=a_surprise * float(F(x[3])))
robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
print(json.dumps(ext["draw_timing"], indent=2))



# ---------------------------------------------------------------------------
# Designs that respond to the stock: the envelope drift D*(H) = max_z drift.
# ---------------------------------------------------------------------------
ZG = np.unique(np.concatenate([np.linspace(0.0, x[3], 9001), x, [kbar]]))


def Dstar(h, r=rbase):
    d = eta * F(ZG) * X(h, ZG, r) - delta * h
    i = int(np.argmax(d))
    lo_, hi_ = ZG[max(i - 1, 0)], ZG[min(i + 1, len(ZG) - 1)]
    sc = minimize_scalar(lambda z: -float(drift(h, z, r)), bounds=(lo_, hi_), method="bounded",
                         options={"xatol": 1e-12})
    return max(float(d[i]), -float(sc.fun))


R_grid = Rcrit(ZG)
viab = R_grid > rbase
Mg_ = M(ZG)
Hc_env = (delta * rbase / Mg_[viab].max() - G0) / g
Hs_env = eta * K * F(ZG)[viab].max() / delta
hh_env = np.linspace(0.0, Hs_env, 4001)
neg = np.array([-Dstar(h) for h in hh_env])
j_env = int(np.argmax(neg))
sc_env = minimize_scalar(lambda h: Dstar(h), bounds=(hh_env[max(j_env - 1, 0)], hh_env[min(j_env + 1, 4000)]),
                         method="bounded", options={"xatol": 1e-12})
s_env = -float(sc_env.fun)
G_hs = G0 + g * hs_star
idle_share = 1.0 - K * ((G_hs + rbase) / G_hs + cs) / (b * hs_star)
pos_set = np.array([Dstar(h) for h in np.linspace(1e-6, 4.0, 4001)]) > 0
ext["state_dependent"] = dict(Hc_envelope=Hc_env, Hs_envelope=Hs_env, s_c_envelope=s_env,
                              H_trough_envelope=float(sc_env.x), s_c_zstar=s_crit, s_c_full=None,
                              idle_share_at_Hs=idle_share,
                              positive_set_connected=bool(np.all(np.diff(np.where(pos_set)[0]) == 1)),
                              maxD_below_Rstar=max(Dstar(h, rstar - 1e-3) for h in np.linspace(1e-4, 4.0, 2001)),
                              maxD_above_Rstar=max(Dstar(h, rstar + 1e-3) for h in np.linspace(1e-4, 4.0, 2001)))
nu_f, q_f = eta * b, 1.0 + float(C(kbar))
m_f = nu_f - delta * q_f
gam_f = q_f * G0 + rbase
ext["state_dependent"]["s_c_full"] = (nu_f * rbase + m_f * gam_f - 2 * math.sqrt(nu_f * rbase * m_f * gam_f)) / (q_f ** 2 * g)

# Temporary overload: deadline for returning the routine pressure to r0 = rbase.
from scipy.integrate import quad


def T_crit(r1):
    return quad(lambda h: 1.0 / (delta * h - eta * ls * float(X(h, zstar, r1))), hc_star, hs_star, limit=400)[0]


def T_crit_env(r1):
    ev = lambda t, y: y[0] - Hc_env
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(lambda t, y: [Dstar(y[0], r1)], (0.0, 200.0), [Hs_env], events=ev, rtol=1e-10, atol=1e-12)
    return float(sol.t_events[0][0])


d18 = eta * F(ZG) * X(1.8, ZG) - delta * 1.8
i18 = int(np.argmax(d18))
z18 = minimize_scalar(lambda z: -float(drift(1.8, z)), bounds=(ZG[i18 - 1], ZG[i18 + 1]), method="bounded",
                      options={"xatol": 1e-12}).x
ext["state_dependent"]["envelope_at_1p8"] = dict(drift=Dstar(1.8), z=float(z18))


def flood_outcome(r1, T):
    s1 = solve_ivp(lambda t, y: [float(drift(y[0], zstar, r1))], (0.0, T), [hs_star], rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(lambda t, y: [float(drift(y[0], zstar, rbase))], (0.0, 300.0), [s1.y[0, -1]], rtol=1e-10, atol=1e-12)
    return float(s2.y[0, -1])


ext["overload"] = dict(T_lower=math.log(hs_star / hc_star) / delta, T_r3=T_crit(3.0), T_r4=T_crit(4.0),
                       T_env_r4=T_crit_env(4.0), margin_rbase=hs_star - hc_star,
                       margin_formula=delta / (ms * g) * (rstar - rbase))
for r1_ in (3.0, 4.0):
    Tc_ = T_crit(r1_)
    assert flood_outcome(r1_, 0.99 * Tc_) > hs_star - 1e-6 and flood_outcome(r1_, 1.01 * Tc_) < 1e-6
checks["flood_deadline_simulation"] = "recovers at 0.99 T_crit and collapses at 1.01 T_crit for r1 = 3 and 4"

# AI assisted inspection that recognizes a routine candidate with probability eps.
def R_eps(z, cI, eps):
    f = float(F(z))
    c_ = float(C(z))
    ma = eta * b * f - delta * (cI + c_)
    if ma <= 0:
        return 0.0
    return ma / delta * (G0 / (cI + eps * c_) + g * eta * f * K / (delta * (cI + eps * c_) + eps * ma))


def best_eps(cI, eps):
    cuts = sorted(set([*x.tolist(), kbar]))
    best_ = max((R_eps(z, cI, eps), z) for z in cuts)
    for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
        sc = minimize_scalar(lambda z: -R_eps(z, cI, eps), bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-12})
        best_ = max(best_, (-sc.fun, sc.x))
    return best_


def Hs_eps(z, eps, r):
    f = float(F(z))
    A_ = eta * f * K * g - delta * (G0 + eps * r)
    return (A_ + math.sqrt(A_ * A_ + 4 * delta * g * eta * f * K * G0)) / (2 * delta * g)


eps_rel = 0.0
for (z_, cI_, e_) in [(0.65, 1.0, 0.1), (0.8, 0.5, 0.2), (0.5, 0.7, 0.4), (0.7, 0.5, 0.05)]:
    f_, c_ = float(F(z_)), float(C(z_))

    def dr_e(r, f_=f_, c_=c_, cI_=cI_, e_=e_):
        def d(hgrid):
            G = G0 + g * hgrid
            U = b * hgrid * G / ((cI_ + c_) * G + (cI_ + e_ * c_) * r)
            return eta * f_ * np.minimum(U, K * G / (G + e_ * r)) - delta * hgrid
        return d
    num = crit_numeric(dr_e)
    eps_rel = max(eps_rel, abs(num - R_eps(z_, cI_, e_)) / R_eps(z_, cI_, e_))
checks["ai_error_closed_form_rel_error"] = eps_rel
assert eps_rel < 1e-3
b01 = best_eps(0.5, 0.1)
ext["ai_errors"] = dict(R_cI05_eps01=b01[0], z_cI05_eps01=b01[1], Hs_eps01_r15=Hs_eps(b01[1], 0.1, rbase),
                        Hs_eps01_r3=Hs_eps(b01[1], 0.1, 3.0), R_cI1_eps01=best_eps(1.0, 0.1)[0],
                        eps_bar=brentq(lambda e: best_eps(0.5, e)[0] - rstar, 0.0, 1.0, xtol=1e-12),
                        rel_error=eps_rel)

# Max-min design when every unlearned pass probability may exceed its estimate by eps.
def best_gaps(dvec):
    hx_, hc_ = lower_hull(np.concatenate([[0.0], L * dvec]), np.concatenate([[0.0], work]))

    def Rg(z):
        f = min(z / kbar, 1.0)
        c_ = float(np.interp(z, hx_, hc_))
        return max(eta * b * f - delta * (1 + c_), 0.0) / delta * (G0 + g * eta * f * K / delta)
    cuts = sorted(set([*hx_.tolist(), *([kbar] if kbar < hx_[-1] else [])]))
    best_ = max((Rg(z), z) for z in cuts)
    for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
        sc = minimize_scalar(lambda z: -Rg(z), bounds=(lo_, hi_), method="bounded", options={"xatol": 1e-12})
        best_ = max(best_, (-sc.fun, sc.x))
    return best_


rb05 = best_gaps(np.maximum(gaps - 0.05, 0.0))
ext["robust_p0"] = dict(eps=0.05, R_maxmin=rb05[0], z_maxmin=rb05[1], R_nominal_worst=R_fixed(0.05))


eta_ax = np.linspace(2.0, 6.0, 201)
kb_ax = np.linspace(0.3, 1.2, 181)
TYPE = np.zeros((len(kb_ax), len(eta_ax)), dtype=int)
GAIN = np.full((len(kb_ax), len(eta_ax)), np.nan)
for i_, kb_ in enumerate(kb_ax):
    for j_, et_ in enumerate(eta_ax):
        zb_, rb_, _ = optimal(x, kb_, et_)
        alts = [float(Rcrit(v, x, kb_, et_)) for v in x[1:]]
        if kb_ < x[3]:
            alts.append(float(Rcrit(kb_, x, kb_, et_)))
        if rb_ <= 0:
            TYPE[i_, j_] = 0
        elif np.min(np.abs(x[1:] - zb_)) < 1e-9:
            TYPE[i_, j_] = 1
        elif kb_ < x[3] and abs(zb_ - kb_) < 1e-9:
            TYPE[i_, j_] = 3
        else:
            TYPE[i_, j_] = 2
        if max(alts) > 0:
            GAIN[i_, j_] = 100.0 * (rb_ / max(alts) - 1.0)
ext["regions"] = dict(share_none=float((TYPE == 0).mean()), share_single=float((TYPE == 1).mean()),
                      share_interior=float((TYPE == 2).mean()), share_full=float((TYPE == 3).mean()),
                      max_gain_percent=float(np.nanmax(GAIN)), share_gain_above_1=float(np.nanmean(GAIN > 1.0)))

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
fig.subplots_adjust(wspace=0.30)
cmap_t = ListedColormap([SOFT_GRAY, SOFT_TEAL, OCHRE, "#B9C7D8"])
axs[0].pcolormesh(eta_ax, kb_ax, TYPE, cmap=cmap_t, vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True)
axs[0].axhline(x[3], color=INK, lw=0.6, ls=":")
axs[0].text(5.95, x[3] + 0.02, r"$\bar{k}=z_{\max}$", fontsize=7.2, color=INK, ha="right")
axs[0].scatter([eta / delta], [kbar], marker="*", s=80, color="white", edgecolor=INK, lw=0.6, zorder=5)
axs[0].legend(handles=[Patch(facecolor=SOFT_GRAY, label=r"no viable design"),
                       Patch(facecolor=SOFT_TEAL, label=r"single task"),
                       Patch(facecolor=OCHRE, label=r"interior mixture"),
                       Patch(facecolor="#B9C7D8", label=r"cheapest universal learning")],
              loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, frameon=False, fontsize=7.0,
              handlelength=1.2, columnspacing=1.0)
axs[0].set(xlim=(2.0, 6.0), ylim=(0.3, 1.2), xlabel=r"Contribution relative to depreciation $\eta/\delta$",
           ylabel=r"Largest learning cost $\bar{k}$")
style(axs[0], xt=[2.0, 3.0, 4.0, 5.0, 6.0], yt=[0.3, 0.6, 0.9, 1.2], grid=None)
panel_title(axs[0], r"(a) Resilient design")
levels = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 8.5]
cmap_g = LinearSegmentedColormap.from_list("gain", ["#FFFFFF", SOFT_OCHRE, OCHRE, BRICK])
cf = axs[1].contourf(eta_ax, kb_ax, np.nan_to_num(GAIN, nan=0.0), levels=levels, cmap=cmap_g, extend="neither")
axs[1].contour(eta_ax, kb_ax, np.nan_to_num(GAIN, nan=0.0), levels=[1.0], colors=[INK], linewidths=0.5)
axs[1].axhline(x[3], color=INK, lw=0.6, ls=":")
axs[1].scatter([eta / delta], [kbar], marker="*", s=80, color="white", edgecolor=INK, lw=0.6, zorder=5)
cb = fig.colorbar(cf, ax=axs[1], fraction=0.05, pad=0.03, ticks=[0.0, 1.0, 2.0, 4.0, 8.0])
cb.ax.set_yticklabels([fmt2(v) for v in [0.0, 1.0, 2.0, 4.0, 8.0]], fontsize=7.0)
cb.set_label(r"Gain in percent", fontsize=7.6)
cb.outline.set_linewidth(0.4)
axs[1].set(xlim=(2.0, 6.0), ylim=(0.3, 1.2), xlabel=r"Contribution relative to depreciation $\eta/\delta$")
style(axs[1], xt=[2.0, 3.0, 4.0, 5.0, 6.0], yt=[0.3, 0.6, 0.9, 1.2], grid=None)
panel_title(axs[1], r"(b) Gain over the efficient alternatives")
save(fig, "fig9_regions")

robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
(ROOT / "checks" / "numerical_checks.txt").write_text(json.dumps(checks, indent=2) + "\nAll checks passed.\n")
for key in ["state_dependent", "overload", "ai_errors", "robust_p0", "regions"]:
    print(key, json.dumps(ext[key], indent=2))


# Smooth capacity at z*: stable stock and rate of return as routine pressure rises.
def soft_eq(r):
    q_ = 1 + float(C(zstar))
    f_ = float(F(zstar))

    def dd(H):
        G_ = G0 + g * H
        Xi = b * H * G_ / (q_ * G_ + r)
        return eta * f_ * (Xi ** -P_SOFT + K ** -P_SOFT) ** (-1.0 / P_SOFT) - delta * H
    hg = np.linspace(1e-4, 4.0, 40001)
    dv = dd(hg)
    idx = np.where((dv[:-1] > 0) & (dv[1:] <= 0))[0][-1]
    h0 = brentq(dd, hg[idx], hg[idx + 1], xtol=1e-14)
    slope = (dd(h0 + 1e-6) - dd(h0 - 1e-6)) / 2e-6
    return h0, -slope


ext["soft_warning"] = dict(r05=soft_eq(0.5), r13=soft_eq(1.3))
robust["extensions"] = ext
(ROOT / "robustness.json").write_text(json.dumps(robust, indent=2))
print("soft_warning", ext["soft_warning"])
