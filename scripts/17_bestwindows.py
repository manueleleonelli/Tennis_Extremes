"""Best decade against best other decade.

Comparing eras whose boundaries are chosen invites the objection that the
boundaries flatter one side. This script removes the objection by giving each
side its best shot: it finds the ten-year window that maximizes concurrent
dominance, then the best ten-year window disjoint from it, and compares the two
on every summary. Selection inflates both, but symmetrically, so the comparison
between them is fair even though neither figure should be read as typical.

Needs out/04_estimands.pkl, and out/traj.pkl for the occupancy columns.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as CFG

R = pickle.load(open(CFG.OUT/"04_estimands.pkl","rb"))
D = pickle.load(open(CFG.OUT/"03_posterior.pkl","rb"))
yr, W0, NW = D['yr'], D['W0'], D['NW']
M = R['CNT'][1].shape[1]
LEN = 10
LAGS = (1,4,13,26,52)

def stats(Cn, lo, hi):
    sl = (yr>=lo)&(yr<=hi)&(np.arange(NW)>=W0)
    if sl.sum() < 30:            # window falls outside the record
        return dict(EN=np.nan, meanP3=np.nan, wks50=0, tot=np.nan,
                    lag=[np.nan]*len(LAGS))
    p3 = (Cn[sl]>=3).mean(1)
    lag=[]
    for k in LAGS:
        v=[core.lag_persistence(Cn[sl,d],k) for d in range(M)]
        v=[x for x in v if not np.isnan(x)]
        lag.append(np.median(v) if len(v)>5 else np.nan)
    return dict(EN=float(Cn[sl].mean()), meanP3=float(p3.mean()),
                wks50=int((p3>=0.5).sum()),
                tot=float(np.median((Cn[sl]>=3).sum(0))), lag=lag)

wins = [(lo, lo+LEN-1) for lo in range(1978, 2026-LEN+1)]
for k in CFG.MEAN_COUNTS:
    Cn = R['CNT'][k]
    sc = {w: stats(Cn,*w)['meanP3'] for w in wins}
    sc = {w: v for w, v in sc.items() if not np.isnan(v)}
    best = max(sc, key=sc.get)
    rival = max((w for w in wins if w[1] < best[0] or w[0] > best[1]), key=sc.get)
    print("="*74)
    print(f"THRESHOLD mean N_t = {k}: best decade and best disjoint decade")
    rows=[]
    for lab, w in (("best", best), ("best disjoint", rival)):
        st = stats(Cn, *w)
        rows.append(dict(window=f"{w[0]}-{w[1]}", role=lab, **{
            'E[N]': round(st['EN'],2), 'meanP(N>=3)': round(st['meanP3'],3),
            'periods N>=3': int(st['tot']), 'wks p>=.5': st['wks50'],
            **{f'pi_{L}': (round(v,2) if not np.isnan(v) else None)
               for L,v in zip(LAGS, st['lag'])}}))
    print(pd.DataFrame(rows).to_string(index=False))

    try:
        T = pickle.load(open(CFG.OUT/"traj.pkl","rb"))
        occ, yrs, pids = T['occ_year'], np.array(T['years']), T['pids']
        nm = D['names']
        for lab, w in (("best", best), ("best disjoint", rival)):
            sel = (yrs>=w[0])&(yrs<=w[1])
            tot = occ[:, sel].sum(1)
            share = 100*tot/max(tot.sum()/3, 1)
            top = np.argsort(tot)[::-1][:5]
            print(f"  top three most often held in {w[0]}-{w[1]}: " +
                  ", ".join(f"{core.surname(nm.get(pids[i],''))} {share[i]:.0f}%" for i in top))
    except FileNotFoundError:
        print("  (run 09_trajectories.py for the occupancy columns)")
    print()
