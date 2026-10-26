"""Compare simple assessment, recovery, training, and funded policies.

The original parameter files are read without modification. New comparison
results retain full precision in policy_revision.json; plots show two decimals.
"""
from pathlib import Path
import json
import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from training_comparison import benchmark, critical_pressure, design_threshold, optimal_design


ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
NAVY, TEAL, OCHRE, GRAY = '#1B365D', '#2C7A7B', '#C8902E', '#9AA5AE'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})


def run():
    p, zvertices, workloads = benchmark()
    b, g, G0, eta, delta, K, kbar, pressure = [
        p[k] for k in ('b', 'g', 'G0', 'eta', 'delta', 'K', 'kbar', 'r')]

    def learning(z):
        return min(z / kbar, 1.0)

    def cost(z):
        return float(np.interp(z, zvertices, workloads))

    def flow(H, z, r=pressure):
        G = G0 + g * H
        return min(K, b * H * G / ((1 + cost(z)) * G + r))

    def drift(H, z, r=pressure):
        return eta * learning(z) * flow(H, z, r) - delta * H

    def envelope(H, r=pressure):
        # For uniform costs, on each unsaturated frontier segment the
        # uncapped objective is a linear-fractional function. Its derivative
        # has constant sign; the capped objective is increasing. Endpoints
        # and time/cap switching points are consequently sufficient.
        candidates = list(zvertices) + [kbar]
        G = G0 + g * H
        if G > 0:
            required_cost = b * H / K - r / G - 1
            for lo, hi, clo, chi in zip(zvertices[:-1], zvertices[1:],
                                        workloads[:-1], workloads[1:]):
                if clo < required_cost < chi:
                    candidates.append(lo + (required_cost - clo) * (hi - lo) / (chi - clo))
        values = [drift(H, z, r) for z in candidates]
        return max(values)

    def fixed(z):
        f, c = learning(z), cost(z)
        M = eta * b * f - delta * (1 + c)
        Hc = (delta * pressure / M - G0) / g
        Hs = eta * f * K / delta
        B = p['L'] * float(np.interp(z, zvertices, np.r_[0, p['p0']]))
        payment = B + z * f
        nu, q = eta * b * f, 1 + c
        gamma = q * G0 + pressure
        support = (math.sqrt(nu * pressure) - math.sqrt(M * gamma)) ** 2 / (g * q*q)
        return dict(z=z, learning=f, work=c, M=M, Hc=Hc, Hs=Hs,
                    threshold=design_threshold(z, 0., p, zvertices, workloads),
                    reward=payment, nonlearner_reward=(1-f)*B,
                    support_threshold=support,
                    flood_r4=quad(lambda H: -1 / drift(H, z, 4.), Hc, Hs)[0],
                    flood_lower=math.log(Hs/Hc)/delta)

    zopt, ceiling, _ = optimal_design(0., p, zvertices, workloads)
    policies = {'task_2': fixed(.5), 'universal': fixed(kbar),
                'interior_example': fixed(zopt)}
    hc = policies['task_2']['Hc']
    hs = policies['universal']['Hs']
    flood = solve_ivp(lambda t, h: [envelope(float(h[0]), 4.)], (0, 100), [hs],
                      events=lambda t, h: h[0]-hc, rtol=1e-10, atol=1e-12)
    policies['feedback'] = dict(Hc=hc, Hs=hs, threshold=ceiling,
                                flood_r4=float(flood.t_events[0][0]))

    # Independent dense-menu checks for the finite envelope calculation.
    for H in np.linspace(.01, 4., 45):
        dense = max(drift(H, float(z)) for z in np.linspace(0, .9, 2001))
        assert envelope(H) >= dense - 1e-10
        assert envelope(H) - dense < .003

    budget_rows = []
    robustness = json.loads((ROOT/'robustness.json').read_text())
    optimized_caps = {row['pibar']: row['R'] for row in robustness['budget']}
    for cap in [None, 1.5, 1.3, 1.1, .9]:
        best = ceiling if cap is None else optimized_caps[cap]
        full_reward = policies['task_2']['reward']
        a = 1. if cap is None or cap >= full_reward else brentq(
            lambda a: p['L']*p['p0'][1]*a + (.5*a)**2/kbar - cap, 0., 1.)
        # A randomized transfer task uses cost a*t_2, not the lower-hull
        # cost at its exposure (which may mix different tasks).
        threshold = critical_pressure(eta*learning(.5*a), 1+a*p['t'][1], 0.,
                                      b,delta,G0,g,K)
        assert threshold <= best + 2e-6
        budget_rows.append(dict(cap=cap, task_2_probability=a,
                                task_2_threshold=threshold,
                                optimized_threshold=best))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), layout='constrained')
    H = np.linspace(0, 3.4, 501)
    for name, z, color in [('Transfer task', .5, NAVY),
                            ('Universal learning', kbar, OCHRE)]:
        axes[0].plot(H, [drift(float(h), z) for h in H], color=color, label=name)
    axes[0].plot(H, [envelope(float(h)) for h in H], color=TEAL,
                 linestyle='--', label='State-dependent policy')
    axes[0].axhline(0, color=GRAY, linewidth=.7)
    axes[0].set(xlabel='Expertise stock $H$', ylabel='Expertise growth',
                title='(a) Growth and recovery', ylim=(-.5,.45), xlim=(0,3.4))
    axes[0].legend(loc='upper left', frameon=False, fontsize=7.5)
    times = np.linspace(0, 18, 451)
    paths = {}
    for name, fun, color, ls in [
        ('Transfer task', lambda h: drift(h,.5), NAVY, '-'),
        ('Universal learning', lambda h: drift(h,kbar), OCHRE, '-'),
        ('State-dependent policy', envelope, TEAL, '--')]:
        sol = solve_ivp(lambda t, h: [fun(float(h[0]))], (0,18), [1.3],
                        t_eval=times, rtol=1e-9, atol=1e-11)
        axes[1].plot(times, sol.y[0], color=color, linestyle=ls, label=name)
        paths[name] = float(sol.y[0,-1])
    axes[1].axhline(hc, color=GRAY, linewidth=.7, linestyle=':')
    axes[1].set(xlabel='Time (mean lives of expertise)', ylabel='Expertise stock $H$',
                title='(b) Recovery from $H_0=1.30$', ylim=(0,3.4), xlim=(0,18))
    for ax in axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='y', color='#EEF1F4')
    fig.savefig(FIG/'fig_policy_recovery.pdf', metadata={'CreationDate': None, 'Author': None, 'Creator': None, 'Producer': None});fig.savefig(FIG/'fig_policy_recovery.png',dpi=180)
    plt.close(fig)

    taus = np.linspace(0,.96,301)
    opts = [optimal_design(float(t), p, zvertices, workloads) for t in taus]
    fig, axes = plt.subplots(1,2,figsize=(6.5,2.7),layout='constrained')
    for z,label,color in [(.5,'Transfer task',NAVY),(kbar,'Universal learning',OCHRE)]:
        axes[0].plot(taus,[design_threshold(z,float(t),p,zvertices,workloads) for t in taus],
                     label=label,color=color)
    axes[0].plot(taus,[v[1] for v in opts],label='Joint optimum',color=TEAL,linestyle='--')
    axes[0].set(xlabel=r'Training productivity $\tau$',ylabel='Sustainable routine pressure',
                title='(a) Training under the shared budget',xlim=(0,.96))
    axes[0].legend(frameon=False,fontsize=7.5)
    axes[1].plot(taus,[v[0] for v in opts],color=TEAL)
    axes[1].axhline(.5,color=NAVY,linestyle=':',linewidth=.8)
    axes[1].text(.04,.51,'Transfer task',color=NAVY,fontsize=8)
    axes[1].set(xlabel=r'Training productivity $\tau$',ylabel='Selected exposure $z$',
                title='(b) Assessment chosen jointly with training',xlim=(0,.96),ylim=(.47,.83))
    for ax in axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='y',color='#EEF1F4')
    fig.savefig(FIG/'fig_policy_training.pdf', metadata={'CreationDate': None, 'Author': None, 'Creator': None, 'Producer': None});fig.savefig(FIG/'fig_policy_training.png',dpi=180)
    plt.close(fig)

    def gap_pressure(eps):
        return critical_pressure(eta*min(p['L']*(p['d'][1]-eps)/kbar,1),
                                 1+p['t'][1],0.,b,delta,G0,g,K)
    task_2_gap_margin = brentq(lambda eps: gap_pressure(eps)-pressure,0.,p['d'][1])
    result = dict(parameters=p, reference_policy='task_2', policies=policies,
                  budget_rows=budget_rows, recovery_at_time_18=paths)
    result['task_2_gap_margin'] = task_2_gap_margin
    (ROOT/'policy_revision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(policies,indent=2))
    print('Policy illustrations and unrounded comparison results written.')


if __name__ == '__main__':
    run()
