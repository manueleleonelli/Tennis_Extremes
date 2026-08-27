"""Alternative era division: four equal blocks of twelve years.

The paper divides 1978-2025 at 1990, 2003 and 2022. The first boundary is a
structural break in the tour; the other two are the span of the three players
under study, which makes the comparison less than blind and gives the third era
twice the length of the others. This script repeats the era comparison on four
blocks of equal length, whose boundaries owe nothing to any player's career:

    1978-1989   1990-2001   2002-2013   2014-2025

The first is identical to the paper's first era and the second nearly so. The
third begins the year before Federer's first major title, and the third and
fourth between them cover the whole span of the three players, so the division
also asks whether both halves of that span look alike.

Needs out/04_estimands.pkl. Runs in seconds.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as CFG

R = pickle.load(open(CFG.OUT/"04_estimands.pkl","rb"))
D = pickle.load(open(CFG.OUT/"03_posterior.pkl","rb"))
yr, W0, NW = D['yr'], D['W0'], D['NW']
names, pids, code = D['names'], D['pids'], D['code']
M = R['CNT'][1].shape[1]

EQUAL = [(1978,1989),(1990,2001),(2002,2013),(2014,2025)]
PAPER = [(1978,1989),(1990,2002),(2003,2022),(2023,2025)]
LAGS  = (1,4,13,26,52)
WINS  = (4,13,26,52,104)

def block(Cn, lo, hi):
    sl = (yr>=lo)&(yr<=hi)&(np.arange(NW)>=W0)
    p3 = (Cn[sl]>=3).mean(1)
    tot = (Cn[sl]>=3).sum(0)
    lag=[]
    for k in LAGS:
        v=[core.lag_persistence(Cn[sl,d],k) for d in range(M)]
        v=[x for x in v if not np.isnan(x)]
        lag.append(np.median(v) if len(v)>5 else np.nan)
    blk=[np.median([core.block_persistence(Cn[sl,d],w) for d in range(M)]) for w in WINS]
    return dict(periods=int(sl.sum()), EN=float(Cn[sl].mean()),
                meanP3=float(p3.mean()), tot=float(np.median(tot)),
                lo=float(np.percentile(tot,2.5)), hi=float(np.percentile(tot,97.5)),
                wks50=int((p3>=0.5).sum()), lag=lag, blk=blk)

for name, eras in (("EQUAL-LENGTH ERAS", EQUAL), ("ERAS USED IN THE PAPER", PAPER)):
    print("="*78); print(name)
    for k in (2,1):
        print(f"\n threshold mean N_t = {k}")
        print(f"{'era':<12}{'periods':>8}{'E[N]':>7}{'meanP(N>=3)':>13}"
              f"{'tot':>7}{'95% CI':>14}{'wks p>=.5':>11}")
        for lo,hi in eras:
            b=block(R['CNT'][k],lo,hi)
            print(f"{lo}-{hi}{b['periods']:>8}{b['EN']:>7.2f}{b['meanP3']:>13.3f}"
                  f"{b['tot']:>7.0f}   [{b['lo']:.0f}, {b['hi']:.0f}]{b['wks50']:>11}")
        print(f"\n  lag persistence pi_k, k = {LAGS}")
        for lo,hi in eras:
            b=block(R['CNT'][k],lo,hi)
            print(f"   {lo}-{hi}: "+"  ".join("  . " if np.isnan(v) else f"{v:.2f}" for v in b['lag']))
        print(f"  block persistence beta_w, w = {WINS}")
        for lo,hi in eras:
            b=block(R['CNT'][k],lo,hi)
            print(f"   {lo}-{hi}: "+"  ".join(f"{v:.2f}" for v in b['blk']))
    print()

# who holds the top three in each equal-length era, averaged over draws
print("="*78); print("TOP-THREE OCCUPANCY, EQUAL-LENGTH ERAS")
try:
    S8 = pickle.load(open(CFG.OUT/"08_sensitivity.pkl","rb"))
    print("  (from 08_sensitivity.pkl, which uses the paper's eras; rerun 08 with")
    print("   EQUAL substituted for ERAS if you want this on the equal division)")
except FileNotFoundError:
    print("  run 08_sensitivity.py first if you want occupancy percentages")

print("\nLaTeX rows for the equal-length version of Table 8:")
for k in (2,1):
    print(f"\\multicolumn{{7}}{{l}}{{\\textit{{$\\bar N = {k}$}}}}\\\\")
    for lo,hi in EQUAL:
        b=block(R['CNT'][k],lo,hi)
        print(f"{lo}--{hi} & {b['periods']} & {b['EN']:.2f} & {b['meanP3']:.3f} & "
              f"{b['tot']:.0f} & [{b['lo']:.0f}, {b['hi']:.0f}] & {b['wks50']} \\\\")
