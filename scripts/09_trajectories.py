"""Posterior relative-strength trajectories for Figure 8. Reuses the cached
posterior draws so Figure 8 is consistent with every table."""
import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as C

print("hyperparameters:", C.describe())
D=pickle.load(open(C.OUT/"03_posterior.pkl","rb"))
pl,wk,ptr,ws,we,samp=D['pl'],D['wk'],D['ptr'],D['ws'],D['we'],D['samp']
yr,W0,NW,code,names,weeks=D['yr'],D['W0'],D['NW'],D['code'],D['names'],D['weeks']
pids=D['pids']
M=samp.shape[1]; NWa=NW-W0; msk=wk>=W0; SC=core.ELO_SCALE*C.PLATT
# the five players who most often held a top-three position in each era, found
# from the draws rather than fixed by hand, so the figure follows C.ERAS
OCC={e:np.zeros(len(pids)) for e in C.ERAS}
U={k:np.zeros(M) for k in C.MEAN_COUNTS}
ALL={}
for d in range(M):
    v=samp[ptr,d].astype(float)*SC
    rel=core.relative(v,ws,we,NW,t0=W0,band=C.BAND,min_active=C.MIN_ACTIVE)
    rv=rel[msk]
    for k in C.MEAN_COUNTS: U[k][d]=core.calibrate_threshold(rv,NWa,k)
    for t in range(W0,NW):
        a,b=ws[t],we[t]
        if b-a<C.MIN_ACTIVE: continue
        for e in C.ERAS:
            if e[0]<=yr[t]<=e[1]:
                for j in np.argsort(rel[a:b])[::-1][:3]:
                    OCC[e][pl[a:b][j]]+=1.0/M
    ALL[d]=rel
    if d%50==0: print("draw",d,flush=True)

SHOW={f"{a}-{b}":[pids[i] for i in np.argsort(OCC[(a,b)])[::-1][:8]] for a,b in C.ERAS}
want=sorted({p for v in SHOW.values() for p in v})
TR={p:np.full((M,NW),np.nan,np.float32) for p in want}
for d in range(M):
    rel=ALL[d]
    for p in want:
        i=code.get(p)
        if i is None: continue
        sel=pl==i
        if sel.any(): TR[p][d,wk[sel]]=rel[sel]
YRS=sorted(set(yr[W0:].tolist()))
yidx={y:i for i,y in enumerate(YRS)}
OCCY=np.zeros((len(pids),len(YRS)))
for d in range(M):
    rel=ALL[d]
    for t in range(W0,NW):
        a,b=ws[t],we[t]
        if b-a<C.MIN_ACTIVE: continue
        for j in np.argsort(rel[a:b])[::-1][:3]:
            OCCY[pl[a:b][j], yidx[yr[t]]]+=1.0/M
pickle.dump(dict(TR=TR,U1=U[1],U2=U[2],weeks=weeks,SHOW=SHOW,OCC=OCC,
                 occ_year=OCCY,years=YRS,pids=list(pids),
                 names={p:names.get(p,p) for p in want}),open(C.OUT/"traj.pkl","wb"))
print("median thresholds:",np.median(U[1]).round(0),np.median(U[2]).round(0))
