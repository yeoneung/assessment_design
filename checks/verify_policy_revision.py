"""Check the weaker verification-capacity regime and revised policy comparisons.

The original inspection/recognition/training LP retains candidate-supply and
recognition constraints. It independently verifies the reduced formulas when
idea productivity g is below normalized expert capacity b.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_comparison import critical_pressure

rng = np.random.default_rng(20261026)
errors = []
for case in range(240):
    b, K = rng.uniform(.3,2.,2)
    g = b*rng.uniform(.05,.95)
    cI = 1. if case<120 else rng.uniform(.2,1.)
    cutoff = K*max(1-g*cI/b,0.)
    G0 = cutoff+rng.uniform(.001,1.)
    H, r = rng.uniform(.001,10.,2)
    c = rng.uniform(0.,2.)
    delta = rng.uniform(.3,2.)
    y = rng.uniform(.1,6.)
    tau = delta/b*rng.uniform(0.,.95) if cI==1. else 0.
    G = G0+g*H
    theta = G/(G+r)
    # Variables: inspection rate n, recognized flow X, training time u.
    lp = linprog([0.,-y,-tau],
                 A_ub=[[cI,c,1.],[-theta,1.,0.],[1.,0.,0.],[0.,1.,0.]],
                 b_ub=[b*H,0.,G+r,K],bounds=[(0.,None)]*3,method='highs')
    assert lp.success,lp.message
    time_per_result=cI/theta+c
    formula=(tau*b-delta)*H+max(y-tau*time_per_result,0.)*min(K,b*H/time_per_result)
    error=abs((-lp.fun-delta*H)-formula)
    errors.append(error)
    assert error<1e-8
    assert lp.x[1]<=G+1e-8
    # The bottleneck inequality is tightest at H=K*cI/b.
    Hswitch=K*cI/b
    assert abs((G0+g*Hswitch)-K-(G0-cutoff))<1e-12
    grid=np.r_[0.,np.geomspace(1e-5,50.,80),Hswitch]
    assert np.all(G0+g*grid>=np.minimum(K,b*grid/cI)-1e-12)

error_errors = []
for case in range(80):
    b, K = rng.uniform(.3, 2., 2)
    g = b*rng.uniform(.05, .95)
    cI = rng.uniform(.2, 1.)
    G0 = K*(1-g*cI/b)+rng.uniform(.001, 1.)
    H, r = rng.uniform(.001, 10., 2)
    c, eps, y = rng.uniform(0., 2.), rng.uniform(0., .5), rng.uniform(.1, 6.)
    G = G0+g*H
    theta = G/(G+r)
    recognition = theta+eps*(1-theta)
    # The original inspection LP includes false recognitions in both costs.
    lp = linprog([-y*theta], A_ub=[[cI+c*recognition], [recognition], [1.]],
                 b_ub=[b*H, K, G+r], bounds=[(0., None)], method='highs')
    assert lp.success, lp.message
    X = min(b*H*G/((cI+c)*G+(cI+eps*c)*r), K*G/(G+eps*r))
    error_errors.append(abs(-lp.fun-y*X))
    assert error_errors[-1] < 1e-8

boundary_errors = []
for case in range(30):
    b, delta, K = rng.uniform(.3, 2., 3)
    g = b*rng.uniform(.05, .95)
    G0 = K*(1-g/b)+rng.uniform(.001, 1.)
    q = rng.uniform(1., 3.)
    y = delta*q/b*rng.uniform(1.05, 2.)
    tau = delta/b*rng.uniform(.01, .98)
    alpha = delta-tau*b
    Hs = y*K/delta
    vertex = (g*K*(y-tau*q)-alpha*G0)/(2*g*alpha)
    peak = max(Hs, vertex)
    threshold = critical_pressure(y, q, tau, b, delta, G0, g, K)

    def original_drift(stock, pressure):
        G = G0+g*stock
        theta = G/(G+pressure)
        lp = linprog([0., -y, -tau],
                     A_ub=[[1., q-1, 1.], [-theta, 1., 0.],
                           [1., 0., 0.], [0., 1., 0.]],
                     b_ub=[b*stock, 0., G+pressure, K],
                     bounds=[(0., None)]*3, method='highs')
        assert lp.success, lp.message
        return -lp.fun-delta*stock

    boundary_errors.append(abs(original_drift(peak, threshold)))
    assert boundary_errors[-1] < 1e-8*max(1., delta*peak)
    assert original_drift(peak, threshold*(1-1e-4)) > 0.
    stocks = np.unique(np.r_[np.geomspace(.001*Hs, y*K/alpha, 50), peak])
    assert max(original_drift(H, threshold*(1+1e-4)) for H in stocks) < 0.

data=json.loads((ROOT/'policy_revision.json').read_text())
p=data['parameters']
task=data['policies']['task_2']
assert abs(task['Hc']-data['policies']['feedback']['Hc'])<1e-10
assert data['policies']['feedback']['Hs']>task['Hs']
assert task['flood_r4']>task['flood_lower']
for row in data['budget_rows']:
    a=row['task_2_probability']
    F=p['L']*p['d'][1]*a/p['kbar']
    reward=p['L']*p['p0'][1]*a+p['L']*p['d'][1]*a*F
    if row['cap'] is not None:
        assert reward<=row['cap']+1e-10
    M=p['eta']*p['b']*F-p['delta']*(1+p['t'][1]*a)
    threshold=max(M,0)/p['delta']*(p['G0']+p['g']*p['eta']*F*p['K']/p['delta'])
    assert abs(threshold-row['task_2_threshold'])<1e-10
    assert threshold<=row['optimized_threshold']+2e-6
eps=data['task_2_gap_margin']
F=p['L']*(p['d'][1]-eps)/p['kbar']
M=p['eta']*p['b']*F-p['delta']*(1+p['t'][1])
R=max(M,0)/p['delta']*(p['G0']+p['g']*p['eta']*F*p['K']/p['delta'])
assert abs(R-p['r'])<1e-9

report=(
    'Verification-capacity and policy-revision checks\n'
    f'240 independent original allocation LPs with g<b: maximum drift error {max(errors):.3e}\n'
    'Includes 120 shared-training cases and 120 AI-inspection cases.\n'
    f'80 original inspection LPs with review errors: maximum flow error {max(error_errors):.3e}\n'
    f'30 training-threshold boundary checks with g<b: maximum drift error {max(boundary_errors):.3e}\n'
    'Each training boundary also checked below and above critical pressure.\n'
    'Affine bottleneck condition and significant-supply constraints checked.\n'
    'Simple-task cash/reward caps use the actual randomized-task workload, not a cheaper frontier mixture.\n'
    'Task 2 and feedback recovery thresholds agree; feedback has the higher long-run stock.\n'
    'Flood and fixed-task performance-gap sensitivity calculations passed.\n'
    'ALL CHECKS PASSED\n')
(ROOT/'checks/policy_revision_validation.txt').write_text(report)
print(report,end='')
