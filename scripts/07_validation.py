"""Section 5.2. Polya-Gamma Gibbs sampler compared against the Laplace/FFBS
approximation on a 2000-2010 subsample. Table 3. Runtime about 20 minutes.

Reports the comparison twice: on the raw latent scale, and after centring each
period on the contemporaneous field as in equation (3). The centred figures are
the ones the paper quotes, because every estimand is computed on that scale and
a common shift across players cancels there."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
Q3=3.0/np.pi**2; S=400/np.log(10)
TAU=0.0471; SIG0=1.0

from tennisdom import core, config as C
_m = core.load_matches(C.DATA); _e = core.encode(_m)
names = _e['names']; m = _m
d=m[(pd.to_datetime(m.tourney_date).dt.year>=2000)&(pd.to_datetime(m.tourney_date).dt.year<=2010)]
cnt=pd.concat([d.winner_id,d.loser_id]).value_counts()
keep=set(cnt[cnt>=120].index)
d=d[d.winner_id.isin(keep)&d.loser_id.isin(keep)]
pl=pd.Index(sorted(set(d.winner_id)|set(d.loser_id))); code={p:i for i,p in enumerate(pl)}
NP=len(pl)
wkv=d.tourney_date.values.astype('datetime64[W]')
wks=np.array(sorted(set(wkv.tolist())),dtype='datetime64[W]'); wm={w:i for i,w in enumerate(wks)}
tt=np.array([wm[w] for w in wkv],np.int32); NW=len(wks)
W=d.winner_id.map(code).values.astype(np.int32); L=d.loser_id.map(code).values.astype(np.int32)
o=np.argsort(tt,kind='stable'); W,L,tt=W[o],L[o],tt[o]
print(f"validation subsample: {len(W)} matches, {NP} players, {NW} weeks")

# ---- observation index: for each player, the periods in which they played ----
obs={i:sorted(set(tt[(W==i)|(L==i)].tolist())) for i in range(NP)}

# ---- Polya-Gamma sampler, truncated sum-of-gammas representation ----
KTR=120
kk=(np.arange(1,KTR+1)-0.5)**2
def rpg(z,rng):
    n=len(z)
    g=rng.exponential(size=(n,KTR))
    den=kk[None,:]+ (z**2)[:,None]/(4*np.pi**2)
    return (g/den).sum(1)/(2*np.pi**2)

def ffbs_player(pseudo, tau, rng):
    """pseudo: list of (period, y, v). Random-walk state, returns sampled path dict."""
    ts=np.array([p[0] for p in pseudo]); ys=np.array([p[1] for p in pseudo])
    vs=np.array([p[2] for p in pseudo]); K=len(ts)
    mf=np.empty(K); Pf=np.empty(K)
    mprev=0.0; Pprev=SIG0**2
    for k in range(K):
        if k>0: Pprev=Pf[k-1]+(ts[k]-ts[k-1])*tau**2; mprev=mf[k-1]
        Pf[k]=1.0/(1.0/Pprev+1.0/vs[k]); mf[k]=Pf[k]*(mprev/Pprev+ys[k]/vs[k])
    x=np.empty(K); x[K-1]=rng.normal(mf[K-1],np.sqrt(Pf[K-1]))
    for k in range(K-2,-1,-1):
        Q=(ts[k+1]-ts[k])*tau**2
        J=Pf[k]/(Pf[k]+Q)
        x[k]=rng.normal(mf[k]+J*(x[k+1]-mf[k]),np.sqrt(Pf[k]*Q/(Pf[k]+Q)))
    return ts,x

rng=np.random.default_rng(C.SEED+1)
TH=np.zeros((NP,NW))            # current state, forward filled
paths={i:(np.array(obs[i]),np.zeros(len(obs[i]))) for i in range(NP)}
def refill(i):
    ts,x=paths[i]
    j=np.searchsorted(ts,np.arange(NW),'right')-1
    j=np.clip(j,0,len(ts)-1)
    TH[i]=x[j]
for i in range(NP): refill(i)

NIT,BURN=1200,400
keepdraws=[]
for it in range(NIT):
    psi=TH[W,tt]-TH[L,tt]
    om=rpg(psi,rng)
    om=np.maximum(om,1e-8)
    for i in rng.permutation(NP):
        aw=W==i; al=L==i
        per=[];yy=[];vv=[]
        # accumulate per period
        prec={}; num={}
        for arr,sgn,opp in ((np.where(aw)[0],0.5,L),(np.where(al)[0],-0.5,W)):
            if len(arr)==0: continue
            t_=tt[arr]; o_=TH[opp[arr],t_]; w_=om[arr]
            contrib=w_*o_+sgn
            for t1,p1,n1 in zip(t_,w_,contrib):
                prec[t1]=prec.get(t1,0.0)+p1; num[t1]=num.get(t1,0.0)+n1
        ts=np.array(sorted(prec))
        if len(ts)==0: continue
        pseudo=[(int(t1),num[t1]/prec[t1],1.0/prec[t1]) for t1 in ts]
        paths[i]=ffbs_player(pseudo,TAU,rng)
        refill(i)
    if it>=BURN and it%4==0:
        keepdraws.append(TH.copy())
    if it%100==0: print("iter",it,flush=True)

G=np.array(keepdraws)   # (ndraw, NP, NW)
print("Gibbs draws:",G.shape)

# ---- Laplace/FFBS approximation on the same subsample ----
from dbtlib import dbt_filter, flatten, ffbs as ffbs_lap, panel_index
bo=np.full(len(W),3)
_,_,store=dbt_filter(W,L,tt,NP,NW,TAU,SIG0)
T2,M2,P2,off2=flatten(store,NP)
samp=ffbs_lap(T2,M2,P2,off2,NP,TAU,len(G),np.random.default_rng(C.SEED+2))
LA=np.zeros((len(G),NP,NW))
for i in range(NP):
    a,b=off2[i],off2[i+1]
    if b==a: continue
    ts=T2[a:b]; j=np.clip(np.searchsorted(ts,np.arange(NW),'right')-1,0,b-a-1)
    LA[:,i,:]=samp[a:b][j].T

print("\n"+"="*74)
print("VALIDATION OF THE GAUSSIAN FILTERING APPROXIMATION")
top=np.argsort([-(cnt.get(p,0)) for p in pl])[:10]
print(f"{'player':<22}{'Gibbs mean':>12}{'Laplace mean':>14}{'Gibbs sd':>10}{'Lap. sd':>9}")
for i in top:
    gm=G[:,i,:].mean(); lm=LA[:,i,:].mean()
    gs=G[:,i,:].std(0).mean(); ls=LA[:,i,:].std(0).mean()
    print(f"{names.get(pl[i],''):<22}{gm*S:12.1f}{lm*S:14.1f}{gs*S:10.1f}{ls*S:9.1f}")
gm=G.mean(0); lm=LA.mean(0)
mask=np.zeros((NP,NW),bool)
for i in range(NP):
    mask[i,min(obs[i]):max(obs[i])+1]=True
print(f"\n  correlation of posterior means over active cells: "
      f"{np.corrcoef(gm[mask],lm[mask])[0,1]:.4f}")
print(f"  mean |difference| in posterior mean: {np.abs(gm[mask]-lm[mask]).mean()*S:.1f} rating points")
print(f"  ratio of mean posterior s.d. (Laplace / Gibbs): "
      f"{LA.std(0)[mask].mean()/G.std(0)[mask].mean():.3f}")
# peak functional
pkG=np.array([[G[dd,i,mask[i]].max() for i in range(NP)] for dd in range(len(G))])
pkL=np.array([[LA[dd,i,mask[i]].max() for i in range(NP)] for dd in range(len(G))])
print(f"  peak strength: correlation of posterior medians "
      f"{np.corrcoef(np.median(pkG,0),np.median(pkL,0))[0,1]:.4f}")
print(f"  peak strength: mean |difference| in posterior median "
      f"{np.abs(np.median(pkG,0)-np.median(pkL,0)).mean()*S:.1f} points")
print(f"  peak strength: ratio of posterior s.d. (Laplace / Gibbs) "
      f"{pkL.std(0).mean()/pkG.std(0).mean():.3f}")


# ---------------------------------------------------------------- centred scale
act = (G.std(0) > 1e-9) | (LA.std(0) > 1e-9)
def centre(X):
    Y = X.copy()
    for d_ in range(X.shape[0]):
        for t_ in range(X.shape[2]):
            s_ = act[:, t_]
            if s_.sum() > 5:
                Y[d_, s_, t_] -= X[d_, s_, t_].mean()
    return Y
Gc, Lc = centre(G), centre(LA)
gm, lm = Gc.mean(0), Lc.mean(0)
print("\n" + "=" * 74)
print("ON THE RELATIVE SCALE OF EQUATION (3), WHICH IS WHAT THE PAPER REPORTS")
print(f"  correlation of posterior means        {np.corrcoef(gm[act], lm[act])[0,1]:.4f}")
print(f"  mean |difference| in posterior mean   {np.abs(gm[act]-lm[act]).mean()*S:.1f} rating points")
print(f"  90th percentile |difference|          {np.percentile(np.abs(gm[act]-lm[act]),90)*S:.1f} points")
print(f"  ratio of posterior s.d. (Lap/Gibbs)   {Lc.std(0)[act].mean()/Gc.std(0)[act].mean():.3f}")
pkG = np.array([[Gc[d_, i, act[i]].max() for i in range(NP)] for d_ in range(G.shape[0])])
pkL = np.array([[Lc[d_, i, act[i]].max() for i in range(NP)] for d_ in range(G.shape[0])])
print(f"  peak: correlation of posterior medians {np.corrcoef(np.median(pkG,0), np.median(pkL,0))[0,1]:.4f}")
print(f"  peak: mean |difference|                {np.abs(np.median(pkG,0)-np.median(pkL,0)).mean()*S:.1f} points")
print(f"  peak: ratio of posterior s.d.          {pkL.std(0).mean()/pkG.std(0).mean():.3f}")
pickle.dump({'G': G, 'LA': LA, 'pl': list(pl)}, open(str(C.OUT / 'pgvalid.pkl'), 'wb'))
