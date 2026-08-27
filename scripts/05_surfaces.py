"""Section 6.9. Single multivariate model with one state component per surface;
selects rho, samples peaks by multivariate FFBS, computes chi and eta."""
import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as C

print("hyperparameters:", C.describe())
m = core.load_matches(C.DATA)
m = m[(pd.to_datetime(m.tourney_date).dt.year>=C.YEAR_START) & m.surface.isin(core.SURFACES)]
e = core.encode(m)
kw = dict(wi=e['wi'],li=e['li'],wt=e['wt'],bo=e['bo'],sf=e['sf'],NP=e['NP'],NW=e['NW'])
print(f"{len(m)} matches, {e['NP']} players, {e['NW']} periods")
best=None
for rho in (0.0,0.4,0.7,0.9,0.97):
    for tau in (0.04,0.05,0.08):
        ll,acc,_=core.dbt_filter_surface(**kw,tau=tau,rho=rho,sigma0=C.SIGMA0,a5=C.A5)
        print(f"  rho={rho} tau={tau}: {ll:,.0f} acc={acc:.4f}")
        if best is None or ll>best[0]: best=(ll,tau,rho,acc)
ll,TAU,RHO,ACC=best; print(f"-> tau={TAU} rho={RHO} acc={ACC:.4f}")

cnt={s:np.zeros(e['NP'],int) for s in range(3)}
for s in range(3):
    idx=e['sf']==s
    c=pd.Series(np.concatenate([e['wi'][idx],e['li'][idx]])).value_counts()
    cnt[s][c.index.values]=c.values
elig=np.where((cnt[0]>=40)&(cnt[1]>=40)&(cnt[2]>=40))[0]
print("eligible on all three surfaces:",len(elig))
_,_,store=core.dbt_filter_surface(**kw,tau=TAU,rho=RHO,sigma0=C.SIGMA0,a5=C.A5,
                                  collect_for=list(elig))
rng=np.random.default_rng(C.SEED); M=200
peaks=np.full((len(elig),3,M),-1e9)
for ii,i in enumerate(elig):
    if len(store[i])>=2:
        peaks[ii]=core.ffbs_mv(store[i],TAU,RHO,M,rng).max(axis=0).T
rk=lambda x:(np.argsort(np.argsort(x))+1)/(len(x)+1)
n=len(elig); res={q:{'chi':{},'eta':{},'a3':[]} for q in (0.90,0.95)}
mem={q:np.zeros(n) for q in (0.90,0.95)}; pct=np.zeros((n,3,M))
for q in (0.90,0.95):
    for k in ('HC','HG','CG'): res[q]['chi'][k]=[]; res[q]['eta'][k]=[]
for d in range(M):
    u=[rk(peaks[:,s,d]) for s in range(3)]
    for s in range(3): pct[:,s,d]=u[s]
    for q in (0.90,0.95):
        for k,(a,b) in {'HC':(u[0],u[1]),'HG':(u[0],u[2]),'CG':(u[1],u[2])}.items():
            res[q]['chi'][k].append(((a>q)&(b>q)).mean()/(1-q))
            t=np.minimum(a,b); pr=(t>q).mean()
            res[q]['eta'][k].append(np.log(1-q)/np.log(pr) if pr>0 else np.nan)
        sel=(u[0]>q)&(u[1]>q)&(u[2]>q); res[q]['a3'].append(sel.sum()); mem[q]+=sel
q_=lambda x:np.nanpercentile(np.asarray(x,float),[2.5,50,97.5])
print("\n--- Table 12: cross-surface tail dependence ---")
for q in (0.90,0.95):
    print(f" q={q}")
    for k,l in [('HC','hard-clay'),('HG','hard-grass'),('CG','clay-grass')]:
        c=q_(res[q]['chi'][k]); et=q_(res[q]['eta'][k])
        print(f"   chi({l:<11})={c[1]:.2f} [{c[0]:.2f},{c[2]:.2f}]  eta={et[1]:.2f} [{et[0]:.2f},{et[2]:.2f}]")
    a=q_(res[q]['a3']); print(f"   all three: {a[1]:.0f} [{a[0]:.0f},{a[2]:.0f}] (indep {n*(1-q)**3:.2f})")
print("\n--- Table 13: membership and surface percentiles ---")
inv={v:k for k,v in e['code'].items()}
for q in (0.90,0.95):
    fr=mem[q]/M
    print(f" q={q}: "+", ".join(f"{e['names'].get(inv[elig[i]],'')} {fr[i]:.2f}"
          for i in np.argsort(fr)[::-1][:9] if fr[i]>0.05))
for p in ['D643','F324','N409','B058','M047','L018','C044','MC10','S402','A092']:
    i=e['code'].get(p)
    if i is None or i not in elig: continue
    j=int(np.where(elig==i)[0][0])
    print(f"  {e['names'].get(p,''):<20}"+"  ".join(
        f"{core.SURFACES[s].lower()}={np.median(pct[j,s]):.3f}" for s in range(3)))
pickle.dump(dict(peaks=peaks,elig=elig,res=res,mem=mem,pct=pct,TAU=TAU,RHO=RHO,
                 pids=list(e['pids']),names=e['names']),open(C.OUT/"05_surfaces.pkl","wb"))
