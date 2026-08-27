"""Figure 7, the lag and block persistence profiles."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pickle,matplotlib
from tennisdom import core
from tennisdom import config as CFG
matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':8.5,'font.family':'serif','mathtext.fontset':'cm',
    'axes.grid':True,'grid.alpha':.22,'grid.linewidth':.5,'figure.dpi':200,
    'savefig.bbox':'tight','axes.linewidth':.7,'legend.frameon':True,
    'legend.framealpha':.95,'legend.edgecolor':'0.8','legend.fancybox':False})
NAVY,RED,GREEN='#1F3B63','#B5443F','#3F7A52'
F=pickle.load(open(CFG.OUT/'04_estimands.pkl','rb'));m=core.load_matches(CFG.DATA)
wkv=m.tourney_date.values.astype('datetime64[W]')
weeks=np.array(sorted(set(wkv.tolist())),dtype='datetime64[W]')
yr=pd.to_datetime(weeks).year.values;NW=len(weeks);W0=int(np.searchsorted(yr,1978))
_PAL=[NAVY,RED,GREEN,'#C7803A']
ERAS=[(a,b,f"{a}\u2013{b}",_PAL[i%len(_PAL)]) for i,(a,b) in enumerate(CFG.ERAS)]
LAGS=[1,4,13,26,52,104];WINS=[4,13,26,52,104]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1));fig.subplots_adjust(wspace=.30)
k=1;C=F['CNT'][k];M=C.shape[1]
for a,b,nm,c in ERAS:
    sl=(yr>=a)&(yr<=b)&(np.arange(NW)>=W0)
    md=[];lo=[];hi=[]
    for lag in LAGS:
        v=[]
        for d in range(M):
            nb=(C[sl,d]>=3).astype(int)
            v.append((nb[:-lag]&nb[lag:]).sum()/nb[:-lag].sum() if nb[:-lag].sum()>=5 else 0.0)
        q=np.percentile(v,[2.5,50,97.5]);lo.append(q[0]);md.append(q[1]);hi.append(q[2])
    axes[0].plot(LAGS,md,'o-',c=c,ms=3.6,lw=1.3,mec='w',mew=.6)
    axes[0].fill_between(LAGS,lo,hi,color=c,alpha=.13,lw=0)
    md2=[];lo2=[];hi2=[]
    for w in WINS:
        v=[]
        for d in range(M):
            nb=(C[sl,d]>=3).astype(int)
            v.append((np.convolve(nb,np.ones(w,int),'valid')==w).mean() if len(nb)>=w else 0.0)
        q=np.percentile(v,[2.5,50,97.5]);lo2.append(q[0]);md2.append(q[1]);hi2.append(q[2])
    axes[1].plot(WINS,md2,'o-',c=c,ms=3.6,lw=1.3,mec='w',mew=.6)
    axes[1].fill_between(WINS,lo2,hi2,color=c,alpha=.13,lw=0)
axes[0].set_xscale('log');axes[0].set_xticks(LAGS);axes[0].set_xticklabels(LAGS)
axes[0].set_xlabel('lag $k$ (weekly periods)');axes[0].set_ylabel(r'$P(N_{t+k}\geq3\mid N_t\geq3)$')
axes[0].set_title('(a)  lag persistence profile',fontsize=8.5,loc='left');axes[0].set_ylim(0,1)
axes[1].set_xscale('log');axes[1].set_xticks(WINS);axes[1].set_xticklabels(WINS)
axes[1].set_xlabel('window $w$ (weekly periods)');axes[1].set_ylabel(r'$P(N\geq3$ throughout$)$')
axes[1].set_title('(b)  block persistence profile',fontsize=8.5,loc='left')
axes[0].legend([Line2D([0],[0],c=c,marker='o',ms=3.6,lw=1.3) for _,_,_,c in ERAS],
               [nm for _,_,nm,_ in ERAS],loc='lower left',fontsize=7.5)
plt.savefig(str(CFG.OUT/'fig7_persistence.pdf'));plt.close();print("ok")
