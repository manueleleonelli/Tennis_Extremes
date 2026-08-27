"""Computes every quantity reported in Sections 6.3 to 6.9 within each
posterior draw. Emits a single results pickle plus LaTeX tables."""
import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as C

print("hyperparameters:", C.describe())
D=pickle.load(open(C.OUT/"03_posterior.pkl","rb"))
pl,wk,ptr,ws,we,samp = D['pl'],D['wk'],D['ptr'],D['ws'],D['we'],D['samp']
yr,W0,NW,code,names,pids = D['yr'],D['W0'],D['NW'],D['code'],D['names'],D['pids']
M=samp.shape[1]; NWa=NW-W0; msk=wk>=W0
SC=core.ELO_SCALE*C.PLATT
ERAS=[(a,b) for a,b in C.ERAS]

R=dict(peak={p:np.zeros(M) for p in C.PLAYERS},
       CNT={k:np.zeros((NW,M),np.int16) for k in C.MEAN_COUNTS},
       SOJ={k:np.zeros((len(pids),M),np.int32) for k in C.MEAN_COUNTS},
       PROF={e:np.zeros((5,M)) for e in ERAS},
       GPD={e:[] for e in ERAS}, THETA={e:[] for e in ERAS},
       NEXC={e:[] for e in ERAS}, U={k:np.zeros(M) for k in C.MEAN_COUNTS})

for d in range(M):
    v = samp[ptr,d].astype(float)*SC
    rel = core.relative(v, ws, we, NW, t0=W0, band=C.BAND, min_active=C.MIN_ACTIVE)
    rv = rel[msk]; wkm = wk[msk]; plm = pl[msk]; yym = yr[wkm]
    for p in C.PLAYERS:
        i=code.get(p)
        if i is None: continue
        s=plm==i
        if s.any(): R['peak'][p][d]=rv[s].max()
    for k in C.MEAN_COUNTS:
        u=core.calibrate_threshold(rv,NWa,k); R['U'][k][d]=u
        ex=rv>u
        R['CNT'][k][:,d]=np.bincount(wkm[ex],minlength=NW)
        R['SOJ'][k][:,d]=np.bincount(plm[ex],minlength=len(pids))
    # top-five profile, and POT quantities at the mean-count-2 threshold
    u2=R['U'][2][d]
    for e_ in ERAS:
        a_,b_=e_; sl=(yym>=a_)&(yym<=b_)
        acc=np.zeros(5); n=0
        for t in range(W0,NW):
            if not (a_<=yr[t]<=b_): continue
            aa,bb=ws[t],we[t]
            if bb-aa<C.MIN_ACTIVE: continue
            top=np.sort(rel[aa:bb])[::-1][:5]
            if len(top)==5: acc+=top; n+=1
        if n: R['PROF'][e_][:,d]=acc/n
        x=rv[sl]; exc=x[x>u2]-u2; R['NEXC'][e_].append(len(exc))
        if len(exc)>=25:
            sig,xi=core.fit_gpd(exc); zeta=len(exc)/len(x)
            R['GPD'][e_].append((sig,xi,
                core.return_level(u2,sig,xi,zeta,10*52),
                core.return_level(u2,sig,xi,zeta,50*52)))
        else: R['GPD'][e_].append((np.nan,)*4)
        Mt=np.full(NW,-1e9); np.maximum.at(Mt,wkm,rv)
        Ms=Mt[(yr>=a_)&(yr<=b_)&(np.arange(NW)>=W0)]
        idx=np.where(Ms>u2)[0]
        R['THETA'][e_].append(core.extremal_index(idx,len(idx)))
    if d%50==0: print("draw",d,flush=True)
pickle.dump(R,open(C.OUT/"04_estimands.pkl","wb"))

q=lambda x:np.percentile(np.asarray(x,float)[~np.isnan(np.asarray(x,float))],[2.5,50,97.5])
print("\n--- Table 6: peak relative strength ---")
for p,nm in sorted(C.PLAYERS.items(), key=lambda kv:-np.median(R['peak'][kv[0]])):
    lo,md,hi=q(R['peak'][p]); print(f"{nm:<10}{md:6.0f} [{lo:5.0f},{hi:5.0f}]")
print("\n--- Table 7: pairwise ordering probabilities ---")
K=['D643','F324','N409','B058','M047','L018']
print("          "+" ".join(f"{C.PLAYERS[k][:5]:>6}" for k in K))
for a_ in K:
    print(f"{C.PLAYERS[a_]:<10}"+" ".join("   -  " if a_==b_ else
          f"{(R['peak'][a_]>R['peak'][b_]).mean():6.2f}" for b_ in K))
print("\n--- Table 4: top-five profile ---")
for k in range(5):
    print(f"rank {k+1}: "+"  ".join(
      f"{a}-{b}: {q(R['PROF'][(a,b)][k])[1]:5.0f}" for a,b in ERAS[:3]))
print("\n--- Table 8: concurrence, Table 9: persistence ---")
for k in C.MEAN_COUNTS:
    Cn=R['CNT'][k]; p3=(Cn>=3).mean(1)
    print(f"threshold mean N_t={k}  (median u={np.median(R['U'][k]):.0f})")
    for a_,b_ in ERAS:
        sl=(yr>=a_)&(yr<=b_)&(np.arange(NW)>=W0)
        tw=(Cn[sl]>=3).sum(0)
        lags=[np.nanmedian([core.lag_persistence(Cn[sl,d],L) for d in range(M)])
              for L in (1,4,13,26,52)]
        blk=[np.median([core.block_persistence(Cn[sl,d],w) for d in range(M)])
             for w in (4,13,26,52,104)]
        print(f"  {a_}-{b_}: E[N]={Cn[sl].mean():.2f} meanP(N>=3)={p3[sl].mean():.3f} "
              f"tot={np.median(tw):.0f} [{np.percentile(tw,2.5):.0f},{np.percentile(tw,97.5):.0f}] "
              f"wksP>=.5={(p3[sl]>=0.5).sum()}")
        print(f"      pi_k  = "+" ".join(f"{v:.2f}" for v in lags))
        print(f"      beta_w= "+" ".join(f"{v:.2f}" for v in blk))
print("\n--- Table 10: sojourn (mean N_t = 2) ---")
med=np.median(R['SOJ'][2],axis=1)
for i in np.argsort(med)[::-1][:10]:
    lo,hi=np.percentile(R['SOJ'][2][i],[2.5,97.5])
    print(f"  {names.get(pids[i],''):<20}{med[i]:6.0f} [{lo:.0f},{hi:.0f}]")
print("\n--- Table 11: GPD by era, and extremal index ---")
for e_ in ERAS[:3]:
    g=np.array(R['GPD'][e_],float)
    print(f"  {e_[0]}-{e_[1]}: n={np.median(R['NEXC'][e_]):.0f} "
          f"sigma={q(g[:,0])[1]:.0f} {q(g[:,0])[[0,2]].round(0)} "
          f"xi={q(g[:,1])[1]:+.2f} {q(g[:,1])[[0,2]].round(2)} "
          f"rl10={q(g[:,2])[1]:.0f} rl50={q(g[:,3])[1]:.0f}")
    th=q(R['THETA'][e_]); print(f"      extremal index {th[1]:.3f} [{th[0]:.3f},{th[2]:.3f}]")
