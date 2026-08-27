"""Section 5.1. Known-truth study of the functionals. Table 1, Table 2, Figure 3.
Runtime about 25 minutes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from tennisdom import config as C
from dbtlib import dbt_filter, flatten, ffbs, panel_index

NPL,NWK,TAU,SIG0=250,400,0.05,2.0
MPW,APW=70,48; M,REPS=120,25; S=400/np.log(10); TARGET=0.20

def simulate(rng):
    th=np.empty((NPL,NWK));th[:,0]=rng.normal(0,SIG0,NPL)
    for t in range(1,NWK): th[:,t]=th[:,t-1]+rng.normal(0,TAU,NPL)
    wi=[];li=[];wt=[]
    for t in range(NWK):
        act=rng.choice(NPL,APW,replace=False)
        a=rng.choice(act,MPW);b=rng.choice(act,MPW);ok=a!=b;a,b=a[ok],b[ok]
        p=1/(1+np.exp(-(th[a,t]-th[b,t])));w=rng.random(len(a))<p
        wi.append(np.where(w,a,b));li.append(np.where(w,b,a));wt.append(np.full(len(a),t))
    return th,np.concatenate(wi).astype(np.int32),np.concatenate(li).astype(np.int32),\
           np.concatenate(wt).astype(np.int32)

def elo_panel(wi,li,wt):
    o=np.argsort(wt,kind='stable');wi,li,wt=wi[o],li[o],wt[o]
    R=np.zeros(NPL);n=np.zeros(NPL);T=[];Mn=[];PL=[]
    for i in range(len(wi)):
        a,b=wi[i],li[i];e=1/(1+np.exp(-(R[a]-R[b])))
        ka=(250/(n[a]+5)**0.4)/S;kb=(250/(n[b]+5)**0.4)/S
        R[a]+=ka*(1-e);R[b]-=kb*(1-e);n[a]+=1;n[b]+=1
        T+= [wt[i],wt[i]];Mn+=[R[a],R[b]];PL+=[a,b]
    df=pd.DataFrame({'t':T,'m':Mn,'p':PL}).groupby(['p','t'],as_index=False).last().sort_values(['p','t'])
    off=np.zeros(NPL+1,np.int64)
    off[1:]=np.cumsum(df.groupby('p').size().reindex(range(NPL),fill_value=0).values)
    return df.t.values.astype(np.int32),df.m.values,off

def relative(v,ws,we,NW,lo=9,hi=100):
    r=np.full_like(v,-1e9)
    for t in range(NW):
        a,b=ws[t],we[t]
        if b-a<hi+20: continue
        x=v[a:b];r[a:b]=x-np.sort(x)[::-1][lo:hi].mean()
    return r

def calibU(rel,wk,NW,target):
    lo,hi=0.5,6.0
    for _ in range(40):
        mid=(lo+hi)/2
        if (np.bincount(wk[rel>mid],minlength=NW)>=3).mean()>target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def runs_of(b):
    out=[];c=0
    for v in b:
        if v: c+=1
        elif c: out.append(c);c=0
    if c: out.append(c)
    return out

def funcs(b):
    """b: boolean exceedance-configuration series."""
    r=runs_of(b)
    L=max(r) if r else 0
    # gap-tolerant run: merge runs separated by <= g zeros
    def tol(g):
        bb=b.copy();z=0;start=None
        out=[];c=0;gap=0
        for v in bb:
            if v: c+=1;gap=0
            else:
                gap+=1
                if gap<=g and c>0: c+=1
                else:
                    if c: out.append(c-min(gap-1,g));c=0
                    gap=0
        if c: out.append(c)
        return max(out) if out else 0
    mcs=np.mean(r) if r else 0.0
    nb=b.astype(int)
    p1=(nb[:-1]&nb[1:]).sum()/max(nb[:-1].sum(),1)
    p8=(nb[:-8]&nb[8:]).sum()/max(nb[:-8].sum(),1)
    d=dict(L=L,L4=tol(4),L8=tol(8),mcs=mcs,p1=p1,p8=p8,tot=int(nb.sum()))
    for w in (4,13,26,52):                      # block persistence probability
        if len(nb)>=w:
            blk=np.convolve(nb,np.ones(w,int),'valid')==w
            d[f'B{w}']=blk.mean()
        else: d[f'B{w}']=0.0
    return d

rows=[]
for rep in range(REPS):
    rng=np.random.default_rng(C.SEED+rep)
    th,wi,li,wt=simulate(rng)
    _,_,store=dbt_filter(wi,li,wt,NPL,NWK,TAU,SIG0)
    T,Mn,P,off=flatten(store,NPL)
    pl,wk,pt,ws,we=panel_index(T,off,NPL,NWK)
    relT=relative(th[pl,wk],ws,we,NWK); U=calibU(relT,wk,NWK,TARGET)
    bt=np.bincount(wk[relT>U],minlength=NWK)>=3
    ft=funcs(bt)
    eT,eM,eoff=elo_panel(wi,li,wt)
    epl,ewk,ept,ews,ewe=panel_index(eT,eoff,NPL,NWK)
    relE=relative(eM[ept],ews,ewe,NWK); Ue=calibU(relE,ewk,NWK,TARGET)
    fe=funcs(np.bincount(ewk[relE>Ue],minlength=NWK)>=3)
    # posterior-mean trajectory treated as observed, the second plug-in
    relM=relative(Mn[pt],ws,we,NWK); Um=calibU(relM,wk,NWK,TARGET)
    fm=funcs(np.bincount(wk[relM>Um],minlength=NWK)>=3)
    samp=ffbs(T,Mn,P,off,NPL,TAU,M,rng)
    acc={k:[] for k in ft}
    IND=np.zeros((NWK,M),bool)
    for d in range(M):
        rd=relative(samp[pt,d].astype(float),ws,we,NWK)
        Ud=calibU(rd,wk,NWK,TARGET)
        b=np.bincount(wk[rd>Ud],minlength=NWK)>=3
        IND[:,d]=b
        fd=funcs(b)
        for k in ft: acc[k].append(fd[k])
    # the posterior probability curve, longest run where P(N>=3 | data) >= 1/2
    curve=funcs(IND.mean(1)>=0.5)['L']
    row={'rep':rep,'L_pmean':fm['L'],'L_curve':curve}
    for k in ft:
        row[f'{k}_true']=ft[k];row[f'{k}_elo']=fe[k];row[f'{k}_post']=np.median(acc[k])
    rows.append(row)
    if rep%5==0: print("rep",rep,flush=True)

R=pd.DataFrame(rows); R.to_csv(C.OUT/'sim3.csv',index=False)
print("\nLongest run, all four estimators, ratio to truth (Figure 3, left panel):")
for c,lab in [('L_elo','Elo plug-in'),('L_pmean','posterior mean'),
              ('L_post','posterior median'),('L_curve','probability curve')]:
    r_=R[c]/np.maximum(R['L_true'],1)
    print(f"  {lab:<20} mean {r_.mean():.2f}  median {r_.median():.2f}  sd {r_.std():.2f}"
          f"  overstates {int((R[c]>R['L_true']).sum())}/{len(R)}")
# ratios are meaningful only where the truth is bounded away from zero; the
# block probabilities are reported as levels below instead
LBL={'L':'longest run','mcs':'mean run length','L4':'run, gap tol. 4',
     'L8':'run, gap tol. 8','p1':'lag persistence pi_1','p8':'lag persistence pi_8',
     'tot':'rate functional'}
print("\n"+"="*80)
print(f"{'functional':<26}{'Elo/truth':>22}{'posterior/truth':>24}")
print(f"{'':<26}{'mean':>8}{'med':>7}{'sd':>7}{'mean':>9}{'med':>7}{'sd':>7}")
for k,l in LBL.items():
    a=R[f'{k}_elo']/np.maximum(R[f'{k}_true'],1e-9)
    b=R[f'{k}_post']/np.maximum(R[f'{k}_true'],1e-9)
    a=a.replace([np.inf,-np.inf],np.nan).dropna();b=b.replace([np.inf,-np.inf],np.nan).dropna()
    print(f"{l:<26}{a.mean():8.2f}{a.median():7.2f}{a.std():7.2f}"
          f"{b.mean():9.2f}{b.median():7.2f}{b.std():7.2f}")
(C.OUT/'06_ran.flag').write_text('ok')
print('flag written; tables 1 and 2 are now from your own run')

print("\n" + "="*80)
print("BLOCK PERSISTENCE (Table 2), reported as levels since the truth nears zero")
print(f"{'window w':>10}{'truth':>9}{'Elo':>9}{'posterior':>11}{'|Elo-truth|':>13}{'|post-truth|':>14}")
for w_ in (4,13,26,52):
    t_,e_,p_=R[f'B{w_}_true'],R[f'B{w_}_elo'],R[f'B{w_}_post']
    print(f"{w_:>10}{t_.mean():9.3f}{e_.mean():9.3f}{p_.mean():11.3f}"
          f"{(e_-t_).abs().mean():13.3f}{(p_-t_).abs().mean():14.3f}")
