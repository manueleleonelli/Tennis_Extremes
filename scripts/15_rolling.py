"""Continuous, era-free view of the two central summaries.

The eras used in the paper have boundaries fixed in advance, but two of them are
drawn around the careers of the players under study. This script recomputes mean
P(N_t>=3) and the one-year persistence pi_52 on every window of a given length,
so one can see whether the paper's window is a maximum of a continuous curve or
an artefact of where the boundaries fall.

Needs out/04_estimands.pkl. Runs in seconds.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tennisdom import core, config as CFG

plt.rcParams.update({'font.size':8.5,'font.family':'serif','mathtext.fontset':'cm',
 'axes.grid':True,'grid.alpha':.22,'grid.linewidth':.5,'figure.dpi':200,
 'savefig.bbox':'tight','axes.linewidth':.7,'legend.frameon':True,
 'legend.framealpha':.95,'legend.edgecolor':'0.8','legend.fancybox':False})
NAVY, RED, GREEN = '#1F3B63', '#B5443F', '#3F7A52'

R = pickle.load(open(CFG.OUT/"04_estimands.pkl","rb"))
D = pickle.load(open(CFG.OUT/"03_posterior.pkl","rb"))
yr, W0, NW = D['yr'], D['W0'], D['NW']
M = R['CNT'][1].shape[1]
PAPER = (2003, 2022)

def window_stats(Cn, lo, hi):
    sl = (yr>=lo)&(yr<=hi)&(np.arange(NW)>=W0)
    if sl.sum() < 52*4: return np.nan, np.nan
    rate = float((Cn[sl]>=3).mean(1).mean())
    lags = [core.lag_persistence(Cn[sl,d],52) for d in range(M)]
    lags = [x for x in lags if not np.isnan(x)]
    return rate, (float(np.median(lags)) if len(lags)>5 else np.nan)

tables={}
for WIN in (10,15,20):
    rows=[]
    for lo in range(1978, 2026-WIN+1):
        hi=lo+WIN-1
        r1,l1 = window_stats(R['CNT'][1], lo, hi)
        r2,l2 = window_stats(R['CNT'][2], lo, hi)
        rows.append(dict(start=lo,end=hi,mid=lo+(WIN-1)/2,
                         rate_N1=r1,lag_N1=l1,rate_N2=r2,lag_N2=l2))
    tables[WIN]=pd.DataFrame(rows)
    tables[WIN].to_csv(CFG.OUT/f"rolling_{WIN}y.csv", index=False)

W=20; T=tables[W]
print(f"{W}-YEAR ROLLING WINDOWS, posterior medians\n")
print(f"{'window':<12}{'meanP(N>=3)':>13}{'pi_52':>8}{'meanP(N>=3)':>14}{'pi_52':>8}")
print(f"{'':<12}{'Nbar=1':>13}{'':>8}{'Nbar=2':>14}{'':>8}")
for _,r in T.iterrows():
    star = "   <-- the paper's era" if (r.start,r.end)==PAPER else ""
    print(f"{int(r.start)}-{int(r.end)}{r.rate_N1:13.3f}{r.lag_N1:8.2f}"
          f"{r.rate_N2:14.3f}{r.lag_N2:8.2f}{star}")

print("\nWHERE THE PAPER'S WINDOW RANKS")
for c,lab in (("rate_N1","mean P(N>=3), Nbar=1"),("lag_N1","pi_52, Nbar=1"),
              ("rate_N2","mean P(N>=3), Nbar=2"),("lag_N2","pi_52, Nbar=2")):
    if T[c].notna().sum()==0:
        print(f"  {lab:<24} no window has a defined value"); continue
    b=T.loc[T[c].idxmax()]
    hit=T[(T.start==PAPER[0])&(T.end==PAPER[1])]
    ev=float(hit[c].iloc[0]) if len(hit) else np.nan
    rank=int((T[c]>ev).sum())+1
    print(f"  {lab:<24} max {b[c]:.3f} at {int(b.start)}-{int(b.end)};"
          f"  paper {ev:.3f}, rank {rank} of {int(T[c].notna().sum())}")

fig,axes=plt.subplots(1,2,figsize=(7.2,3.0)); fig.subplots_adjust(wspace=.30)
for ax,(c1,c2,lab,ttl) in zip(axes,[
        ("rate_N1","rate_N2",r"mean $P(N_t\geq3)$","(a)  how often"),
        ("lag_N1","lag_N2",r"$\pi_{52}$","(b)  how long")]):
    ax.plot(T.mid,T[c2],'-',c=NAVY,lw=1.5)
    ax.plot(T.mid,T[c1],'-',c=RED,lw=1.5)
    ax.axvline(PAPER[0]+(W-1)/2,color=GREEN,lw=6,alpha=.20)
    ax.set_xlabel(f"midpoint of {W}-year window"); ax.set_ylabel(lab)
    ax.set_title(ttl,fontsize=8.5,loc='left')
axes[0].legend([Line2D([0],[0],c=NAVY,lw=1.5),Line2D([0],[0],c=RED,lw=1.5),
                Line2D([0],[0],c=GREEN,lw=6,alpha=.20)],
               [r'$\bar N=2$',r'$\bar N=1$',"paper's window"],loc='upper left',fontsize=7)
plt.savefig(str(CFG.OUT/"fig9_rolling.pdf")); plt.close()
print(f"\nwrote fig9_rolling.pdf and rolling_{{10,15,20}}y.csv")
