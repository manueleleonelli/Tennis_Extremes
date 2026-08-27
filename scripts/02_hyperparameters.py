"""Section 6.2. Selects tau, sigma0 and a5 by prequential likelihood (16),
computes the curvature-based posterior SD for tau, and the reliability
diagram plus Platt recalibration. Produces Table 5 and Figure 4 inputs."""
import sys, json, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import optimize
from tennisdom import core, config as C

m = core.load_matches(C.DATA); e = core.encode(m)
kw = dict(wi=e['wi'], li=e['li'], wt=e['wt'], bo=e['bo'], NP=e['NP'], NW=e['NW'])

print("=== grid over sigma0, tau, a5 (prequential log-likelihood) ===")
best = None
for sig0 in (1.0, 1.5, 2.0):
    for a5 in (1.00, 1.15, 1.30):
        for tau in (0.040, 0.044, 0.048):
            ll, acc, _ = core.dbt_filter(**kw, tau=tau, sigma0=sig0, a5=a5, collect=False)
            if best is None or ll > best[0]:
                best = (ll, tau, sig0, a5, acc)
            print(f"  sigma0={sig0} a5={a5:.2f} tau={tau:.3f}: {ll:,.0f} acc={acc:.4f}")
ll, TAU, SIG0, A5, ACC = best
print(f"-> tau={TAU} sigma0={SIG0} a5={A5} acc={ACC:.4f}")

print("\n=== curvature of the prequential likelihood in tau ===")
g = np.array([TAU-0.008, TAU-0.004, TAU, TAU+0.004, TAU+0.008])
lls = np.array([core.dbt_filter(**kw, tau=t, sigma0=SIG0, a5=A5, collect=False)[0] for t in g])
for t,v in zip(g,lls): print(f"  tau={t:.4f}  ll={v:,.1f}")
c = np.polyfit(g, lls, 2); mode = -c[1]/(2*c[0]); sd = 1/np.sqrt(-2*c[0])
print(f"  mode={mode:.4f}  posterior SD={sd:.5f}  95% CI [{mode-1.96*sd:.4f},{mode+1.96*sd:.4f}]")

print("\n=== identifiability check: extra dispersion runs along a ridge ===")
def with_kappa(tau, kap):
    Q3=core.Q3; NP,NW=e['NP'],e['NW']
    st=np.searchsorted(e['wt'],np.arange(NW),'left'); en=np.searchsorted(e['wt'],np.arange(NW),'right')
    mu=np.zeros(NP);var=np.full(NP,SIG0**2);seen=np.zeros(NP,bool);ll=0.;k2=kap**2
    sc=np.zeros(NP);inf=np.zeros(NP)
    for t in range(NW):
        var[seen]+=tau**2; a,b=st[t],en[t]
        if a==b: continue
        W,L=e['wi'][a:b],e['li'][a:b]; A=np.where(e['bo'][a:b]>=5,A5,1.)
        gW=A/np.sqrt(1+Q3*var[W]+k2); gL=A/np.sqrt(1+Q3*var[L]+k2)
        gp=A/np.sqrt(1+Q3*(var[W]+var[L])+k2)
        p=1/(1+np.exp(-gp*(mu[W]-mu[L]))); msk=seen[W]&seen[L]
        if msk.any(): ll+=np.log(np.clip(p[msk],1e-12,1)).sum()
        act=np.unique(np.concatenate([W,L])); mp=mu[act].copy(); vp=var[act].copy()
        pos=np.full(NP,-1,np.int64); pos[act]=np.arange(len(act)); th=mp.copy()
        for _ in range(8):
            tW,tL=th[pos[W]],th[pos[L]]
            EW=1/(1+np.exp(-gL*(tW-tL)));EL=1/(1+np.exp(-gW*(tL-tW)))
            sc[:]=0;inf[:]=0
            np.add.at(sc,W,gL*(1-EW));np.add.at(sc,L,-gW*EL)
            np.add.at(inf,W,gL**2*EW*(1-EW));np.add.at(inf,L,gW**2*EL*(1-EL))
            th=th+(sc[act]-(th-mp)/vp)/(inf[act]+1/vp)
        I=np.maximum(inf[act],1e-10); mu[act]=th; var[act]=1/(1/vp+I); seen[act]=True
    return ll
for kap in (0.0,0.8,1.6):
    for tau in (0.05,0.09):
        print(f"  kappa={kap} tau={tau}: {with_kappa(tau,kap):,.0f}")

print("\n=== reliability and Platt recalibration ===")
Q3=core.Q3; NP,NW=e['NP'],e['NW']
st=np.searchsorted(e['wt'],np.arange(NW),'left'); en=np.searchsorted(e['wt'],np.arange(NW),'right')
mu=np.zeros(NP);var=np.full(NP,SIG0**2);seen=np.zeros(NP,bool)
sc=np.zeros(NP);inf=np.zeros(NP);PR=[];YR=[]
yrm=pd.to_datetime(m.tourney_date).dt.year.values[e['order']]
for t in range(NW):
    var[seen]+=TAU**2; a,b=st[t],en[t]
    if a==b: continue
    W,L=e['wi'][a:b],e['li'][a:b]; A=np.where(e['bo'][a:b]>=5,A5,1.)
    gW=A/np.sqrt(1+Q3*var[W]); gL=A/np.sqrt(1+Q3*var[L])
    gp=A/np.sqrt(1+Q3*(var[W]+var[L]))
    p=1/(1+np.exp(-gp*(mu[W]-mu[L]))); msk=seen[W]&seen[L]
    if msk.any(): PR.append(p[msk]); YR.append(yrm[a:b][msk])
    act=np.unique(np.concatenate([W,L])); mp=mu[act].copy(); vp=var[act].copy()
    pos=np.full(NP,-1,np.int64); pos[act]=np.arange(len(act)); th=mp.copy()
    for _ in range(8):
        tW,tL=th[pos[W]],th[pos[L]]
        EW=1/(1+np.exp(-gL*(tW-tL)));EL=1/(1+np.exp(-gW*(tL-tW)))
        sc[:]=0;inf[:]=0
        np.add.at(sc,W,gL*(1-EW));np.add.at(sc,L,-gW*EL)
        np.add.at(inf,W,gL**2*EW*(1-EW));np.add.at(inf,L,gW**2*EL*(1-EL))
        th=th+(sc[act]-(th-mp)/vp)/(inf[act]+1/vp)
    I=np.maximum(inf[act],1e-10); mu[act]=th; var[act]=1/(1/vp+I); seen[act]=True
p=np.concatenate(PR); yy=np.concatenate(YR)
print(f"  n={len(p)}  accuracy={(p>0.5).mean():.4f}  Brier={np.mean((1-p)**2):.4f}")
pp=np.concatenate([p,1-p]); oo=np.concatenate([np.ones(len(p)),np.zeros(len(p))])
bins=[0.5,0.6,0.7,0.8,0.9,1.0]; rel_rows=[]
for i in range(5):
    s=(pp>=bins[i])&(pp<bins[i+1])
    rel_rows.append((bins[i],float(pp[s].mean()),float(oo[s].mean())))
    print(f"  [{bins[i]:.1f},{bins[i+1]:.1f}) pred={pp[s].mean():.3f} obs={oo[s].mean():.3f}")
lg=np.log(np.clip(pp,1e-9,1-1e-9)/(1-np.clip(pp,1e-9,1-1e-9)))
yy2=np.concatenate([yy,yy]); tr=(yy2%2==1); te=~tr
f=lambda par:-np.sum(oo[tr]*(par[0]+par[1]*lg[tr])-np.log1p(np.exp(par[0]+par[1]*lg[tr])))
r=optimize.minimize(f,[0.,1.],method='Nelder-Mead'); a_,b_=r.x
pc=1/(1+np.exp(-(a_+b_*lg[te])))
print(f"  Platt a={a_:.3f} b={b_:.3f}  test Brier {np.mean((oo[te]-pp[te])**2):.4f} -> {np.mean((oo[te]-pc)**2):.4f}")
print("  reliability after recalibration (test years):")
for i in range(5):
    sel=te&(pp>=bins[i])&(pp<bins[i+1])
    if sel.sum()>50:
        z2=a_+b_*lg[sel]; pcc=1/(1+np.exp(-z2))
        print(f"    [{bins[i]:.1f},{bins[i+1]:.1f}) pred={pcc.mean():.3f} obs={oo[sel].mean():.3f}"
              f"  diff={oo[sel].mean()-pcc.mean():+.3f}")
print(f"\n  >>> written to out/02_hyper.json; config.py now reports:")
import importlib; importlib.reload(C); print("      " + C.describe())
json.dump({"tau":float(mode),"tau_sd":float(sd),"sigma0":SIG0,"a5":A5,
           "accuracy":float(ACC),"brier":float(np.mean((1-p)**2)),
           "platt_a":float(a_),"platt_b":float(b_),"reliability":rel_rows},
          open(C.OUT/"02_hyper.json","w"),indent=1)
