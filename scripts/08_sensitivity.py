"""Section 6.3 top-five profile and Section 6.10 reference-band sensitivity.

Reuses the cached posterior draws from 03_posterior.py so that every number in
the paper comes from one set of N_DRAWS trajectories."""
import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as C

print("hyperparameters:", C.describe())
D=pickle.load(open(C.OUT/"03_posterior.pkl","rb"))
pl,wk,ptr,ws,we,samp=D['pl'],D['wk'],D['ptr'],D['ws'],D['we'],D['samp']
yr,W0,NW,code,names,pids=D['yr'],D['W0'],D['NW'],D['code'],D['names'],D['pids']
M=samp.shape[1]; NWa=NW-W0; msk=wk>=W0; SC=core.ELO_SCALE*C.PLATT
BANDS={'5--100':(4,100),'10--100':(9,100),'15--100':(14,100)}
ERAS=list(C.ERAS)
KEY=['D643','F324','N409','B058','M047','L018','C044','S402','A092','MC10']

peak={b:{p:np.zeros(M) for p in KEY} for b in BANDS}
CNT={b:{k:np.zeros((NW,M),np.int16) for k in C.MEAN_COUNTS} for b in BANDS}
PROF={e:np.zeros((5,M)) for e in ERAS}
OCC={e:np.zeros(len(pids)) for e in ERAS}

for d in range(M):
    v=samp[ptr,d].astype(float)*SC
    rels={}
    for t in range(W0,NW):
        a,b=ws[t],we[t]
        if b-a<C.MIN_ACTIVE: continue
        x=v[a:b]; sx=np.sort(x)[::-1]
        for bn,(lo,hi) in BANDS.items():
            rels.setdefault(bn,np.full_like(v,-1e9))[a:b]=x-sx[lo:hi].mean()
    for bn in BANDS:
        rel=rels[bn]; rv=rel[msk]; wkm=wk[msk]; plm=pl[msk]
        for p in KEY:
            i=code.get(p)
            if i is None: continue
            s=plm==i
            if s.any(): peak[bn][p][d]=rv[s].max()
        for k in C.MEAN_COUNTS:
            u=core.calibrate_threshold(rv,NWa,k)
            CNT[bn][k][:,d]=np.bincount(wkm[rv>u],minlength=NW)
    rel=rels['10--100']
    for e_ in ERAS:
        a_,b_=e_; acc=np.zeros(5); n=0
        for t in range(W0,NW):
            if not (a_<=yr[t]<=b_): continue
            aa,bb=ws[t],we[t]
            if bb-aa<C.MIN_ACTIVE: continue
            top=np.sort(rel[aa:bb])[::-1][:5]
            if len(top)==5:
                acc+=top; n+=1
            # occupancy accumulated over ALL draws, not one, so the reported
            # percentages are posterior means rather than a single realization
            for j in np.argsort(rel[aa:bb])[::-1][:3]:
                OCC[e_][pl[aa:bb][j]]+=1.0/M
        if n: PROF[e_][:,d]=acc/n
    if d%50==0: print("draw",d,flush=True)

q=lambda x:np.percentile(x,[2.5,50,97.5])
print("\n--- Table 4: top-five profile by era ---")
for k in range(5):
    print(f"rank {k+1}: "+"  ".join(
        f"{a}-{b}: {q(PROF[(a,b)][k])[1]:5.0f} [{q(PROF[(a,b)][k])[0]:.0f},{q(PROF[(a,b)][k])[2]:.0f}]"
        for a,b in ERAS))
print("\n--- top-three occupancy by era ---")
for e_ in ERAS:
    tot=OCC[e_].sum()
    print(f"  {e_[0]}-{e_[1]}: "+", ".join(
        f"{names.get(pids[i],'')} {100*OCC[e_][i]/max(tot/3,1):.0f}%"
        for i in np.argsort(OCC[e_])[::-1][:6]))
print("\n--- Table 14: reference band sensitivity ---")
print(f"{'player':<10}"+"".join(f"{b:>12}" for b in BANDS))
for p in KEY:
    print(f"{C.PLAYERS.get(p,p):<10}"+"".join(f"{np.median(peak[b][p]):>12.0f}" for b in BANDS))
for a_,b_ in [('D643','F324'),('D643','B058'),('F324','B058'),('N409','B058')]:
    print(f"  P({C.PLAYERS[a_]}>{C.PLAYERS[b_]}): "+"  ".join(
        f"{bn}={(peak[bn][a_]>peak[bn][b_]).mean():.2f}" for bn in BANDS))
for k in C.MEAN_COUNTS:
    print(f"  mean P(N>=3), threshold mean N_t={k}")
    for e_ in ERAS:
        sl=(yr>=e_[0])&(yr<=e_[1])&(np.arange(NW)>=W0)
        print(f"    {e_[0]}-{e_[1]}: "+"  ".join(
            f"{bn}={((CNT[bn][k][sl]>=3).mean(1)).mean():.3f}" for bn in BANDS))
print("  lag persistence pi_52, threshold mean N_t=1")
for e_ in ERAS:
    sl=(yr>=e_[0])&(yr<=e_[1])&(np.arange(NW)>=W0)
    out=[]
    for bn in BANDS:
        vals=[core.lag_persistence(CNT[bn][1][sl,d],52) for d in range(M)]
        vals=[v for v in vals if not np.isnan(v)]
        out.append(f"{bn}={np.median(vals):.2f}" if len(vals)>5 else f"{bn}=  .")
    print(f"    {e_[0]}-{e_[1]}: "+"  ".join(out))
pickle.dump(dict(peak=peak,CNT=CNT,PROF=PROF,OCC=OCC),open(C.OUT/"08_sensitivity.pkl","wb"))
