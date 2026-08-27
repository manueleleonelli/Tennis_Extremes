"""Figures 1, 2, 3, 4, 5 and 6. Requires 04_estimands.pkl."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle, matplotlib
matplotlib.use('Agg')
from tennisdom import core, config as CFG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_PARTICLES = {"del","de","van","von","der","di","da","la","le"}
def _surname(full):
    """Short display name keeping particles. Local copy so this script does not
    depend on the version of tennisdom.core that happens to be installed."""
    parts = str(full).split()
    if not parts:
        return str(full)
    if len(parts) >= 2 and parts[-2].lower() in _PARTICLES:
        return " ".join(parts[-2:])
    return parts[-1]


plt.rcParams.update({'font.size':8.5,'font.family':'serif','mathtext.fontset':'cm',
                     'axes.grid':True,'grid.alpha':0.22,'grid.linewidth':.5,
                     'figure.dpi':200,'savefig.bbox':'tight','axes.linewidth':.7,
                     'xtick.major.width':.7,'ytick.major.width':.7,
                     'legend.frameon':True,'legend.framealpha':.95,
                     'legend.edgecolor':'0.8','legend.fancybox':False})
JIT=np.random.default_rng(CFG.SEED)
NAVY,RED,GREEN,ORANGE,PURPLE='#1F3B63','#B5443F','#3F7A52','#C7803A','#6B5B95'
F=pickle.load(open(str(CFG.OUT/'04_estimands.pkl'),'rb'))
m=core.load_matches(CFG.DATA)
pids=pd.Index(sorted(set(m.winner_id)|set(m.loser_id)))
wkv=m.tourney_date.values.astype('datetime64[W]')
weeks=np.array(sorted(set(wkv.tolist())),dtype='datetime64[W]')
yr=pd.to_datetime(weeks).year.values; NW=len(weeks); W0=int(np.searchsorted(yr,1978))
T=pd.to_datetime(weeks)

# ================= FIG 1: concurrence curve =================
fig=plt.figure(figsize=(7.2,5.1))
gs=fig.add_gridspec(3,1,height_ratios=[.16,1,1],hspace=.30)
axe=fig.add_subplot(gs[0]); axes=[fig.add_subplot(gs[1]),fig.add_subplot(gs[2])]
_PAL=[NAVY,RED,GREEN,ORANGE,PURPLE]
ERAS=[(a,b,f"{a}\u2013{b}",_PAL[i%len(_PAL)]) for i,(a,b) in enumerate(CFG.ERAS)]
for a,b,nm,c in ERAS:
    axe.axvspan(pd.Timestamp(f'{a}-01-01'),pd.Timestamp(f'{b}-12-31'),color=c,alpha=.55,lw=0)
    if nm:
        axe.text(pd.Timestamp(f'{(a+b)//2}-07-01'),.5,nm,ha='center',va='center',
                 fontsize=7.5,color='w',fontweight='bold')
axe.set_xlim(T[W0],T[-1]);axe.set_ylim(0,1);axe.axis('off')
for ax,k,lab in zip(axes,[2,1],
        [r'(a)  threshold calibrated to $\bar N=2$ competitors per period',
         r'(b)  threshold calibrated to $\bar N=1$ competitor per period']):
    C=F['CNT'][k]; p3=(C>=3).mean(1); p2=(C>=2).mean(1)
    ax.fill_between(T[W0:],0,p2[W0:],color='0.83',lw=0)
    ax.fill_between(T[W0:],0,p3[W0:],color=NAVY,lw=0)
    ax.axhline(.5,ls=(0,(4,3)),lw=.8,c='0.25')
    for a_,b_,_,_ in ERAS:
        ax.axvline(pd.Timestamp(f'{a_}-01-01'),color='0.6',lw=.5,ls=':')
    ax.set_ylim(0,1.0);ax.set_xlim(T[W0],T[-1])
    ax.set_ylabel('posterior probability')
    ax.set_title(lab,fontsize=8.5,loc='left',pad=3)
    ax.set_yticks([0,.25,.5,.75,1])
axes[0].set_xticklabels([])
handles=[plt.Rectangle((0,0),1,1,fc=NAVY,ec='none'),
         plt.Rectangle((0,0),1,1,fc='0.83',ec='none'),
         Line2D([0],[0],ls=(0,(4,3)),c='0.25',lw=.8)]
axes[1].legend(handles,[r'$P(N_t\geq 3\mid \mathbf{y})$',r'$P(N_t\geq 2\mid \mathbf{y})$',
                        'probability $1/2$'],
               loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=8)
plt.savefig(str(CFG.OUT/'fig1_concurrence.pdf'));plt.close()

# ================= FIG 2: caterpillar =================
pk=F['peak']
FIG2 = ['D643','B058','M047','L018','F324','N409','C044','S0AG','MC10','A0E2','B028','S402']
lab={k:CFG.PLAYERS[k] for k in FIG2}
BIG3={'D643','F324','N409'}
order=sorted([k for k in FIG2 if k in pk],key=lambda p:np.median(pk[p]))
fig,ax=plt.subplots(figsize=(5.0,3.6))
for i,p in enumerate(order):
    q=np.percentile(pk[p],[2.5,25,50,75,97.5])
    c=RED if p in BIG3 else NAVY
    ax.plot([q[0],q[4]],[i,i],c=c,lw=.9,alpha=.55)
    ax.plot([q[1],q[3]],[i,i],c=c,lw=3.4,solid_capstyle='butt')
    ax.plot(q[2],i,'o',c='w',mec=c,mew=1.1,ms=4.6,zorder=5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([lab[p] for p in order])
for t,p in zip(ax.get_yticklabels(),order):
    if p in BIG3: t.set_color(RED)
ax.set_ylim(-.8,len(order)-.2)
ax.set_xlabel('peak relative strength (calibrated rating points above field)')
ax.legend([Line2D([0],[0],c=RED,lw=3.4),Line2D([0],[0],c=NAVY,lw=3.4)],
          ['Federer, Nadal, Djokovic','other competitors'],
          loc='lower right',fontsize=7.5)
plt.savefig(str(CFG.OUT/'fig2_peaks.pdf'));plt.close()

# ================= FIG 3: simulation =================
R=pd.read_csv(str(CFG.OUT/'sim3.csv'))
fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={'width_ratios':[1.35,1]})
fig.subplots_adjust(wspace=.30)
d=[R.L_elo/np.maximum(R.L_true,1),R.L_pmean/np.maximum(R.L_true,1),
   R.L_post/np.maximum(R.L_true,1),R.L_curve/np.maximum(R.L_true,1)]
bp=axes[0].boxplot(d,patch_artist=True,widths=.55,showfliers=False,
                   medianprops=dict(color='k',lw=1.1))
for b,c in zip(bp['boxes'],[RED,ORANGE,NAVY,PURPLE]):
    b.set_facecolor(c);b.set_alpha(.7);b.set_edgecolor('0.3');b.set_linewidth(.7)
for i,dd in enumerate(d):
    axes[0].plot(JIT.normal(i+1,.055,len(dd)),dd,'.',c='0.25',ms=2.2,alpha=.5,zorder=3)
axes[0].set_xticks([1,2,3,4])
axes[0].set_xticklabels(['Elo\nplug-in','posterior\nmean','posterior\nmedian','probability\ncurve'],fontsize=7.6)
axes[0].axhline(1,ls=(0,(4,3)),c='k',lw=.9)
axes[0].set_ylabel('estimate / truth')
axes[0].set_title('(a)  longest unbroken run of $N_t\\geq3$',fontsize=8.5,loc='left')
rw=R.tot_elo/np.maximum(R.tot_true,1); rr=R.L_elo/np.maximum(R.L_true,1)
bp2=axes[1].boxplot([rw,rr],patch_artist=True,widths=.45,showfliers=False,
                    medianprops=dict(color='k',lw=1.1))
for b,c in zip(bp2['boxes'],[GREEN,RED]):
    b.set_facecolor(c);b.set_alpha(.7);b.set_edgecolor('0.3');b.set_linewidth(.7)
for i,dd in enumerate([rw,rr]):
    axes[1].plot(JIT.normal(i+1,.045,len(dd)),dd,'.',c='0.25',ms=2.2,alpha=.5,zorder=3)
axes[1].set_xticks([1,2])
axes[1].set_xticklabels(['total periods\n(rate)','longest run\n(persistence)'],fontsize=7.6)
axes[1].axhline(1,ls=(0,(4,3)),c='k',lw=.9)
axes[1].set_ylabel('Elo plug-in / truth')
axes[1].set_title('(b)  matched marginal exceedance rate',fontsize=8.5,loc='left')
plt.savefig(str(CFG.OUT/'fig3_simulation.pdf'));plt.close()

# ================= FIG 4: reliability =================
import json as _json
_h=_json.load(open(CFG.OUT/'02_hyper.json'))
_rel=np.array(_h['reliability'])              # (bin_lo, mean predicted, observed)
mid=_rel[:,1]; raw=_rel[:,2]
_a,_b=_h['platt_a'],_h['platt_b']
_lg=np.log(mid/(1-mid)); recp=1/(1+np.exp(-(_a+_b*_lg))); rec=raw
fig,ax=plt.subplots(figsize=(3.7,3.5))
ax.plot([.5,1],[.5,1],ls=(0,(4,3)),c='0.35',lw=.9)
ax.plot(mid,raw,'o-',c=RED,ms=4.5,lw=1.2,mec='w',mew=.7)
ax.plot(recp,rec,'s-',c=NAVY,ms=4.2,lw=1.2,mec='w',mew=.7)
ax.set_xlabel('predicted probability');ax.set_ylabel('observed frequency')
ax.set_xlim(.5,1);ax.set_ylim(.5,1)
ax.set_xticks([.5,.6,.7,.8,.9,1.0]);ax.set_yticks([.5,.6,.7,.8,.9,1.0])
ax.legend([Line2D([0],[0],c=RED,marker='o',ms=4.5,lw=1.2),
           Line2D([0],[0],c=NAVY,marker='s',ms=4.2,lw=1.2),
           Line2D([0],[0],ls=(0,(4,3)),c='0.35',lw=.9)],
          ['fitted model','after recalibration','perfect calibration'],
          loc='upper left',fontsize=7.5)
plt.savefig(str(CFG.OUT/'fig4_reliability.pdf'));plt.close()

# ================= FIG 5: surfaces =================
_p5 = CFG.OUT / '05_surfaces.pkl'
if not _p5.exists():
    print("  skipping fig5: run 05_surfaces.py first")
else:
    S5 = pickle.load(open(_p5, 'rb'))
    _M5 = S5['peaks'].shape[2]
    _q = lambda x: np.nanpercentile(np.asarray(x, float), [2.5, 50, 97.5])
    chi = {qq: {lab: _q(S5['res'][qq]['chi'][k])
                for k, lab in (('HC', 'hard\u2013clay'), ('HG', 'hard\u2013grass'),
                               ('CG', 'clay\u2013grass'))}
           for qq in (0.90, 0.95)}
    _sp, _sn = S5['pids'], S5['names']          # the surface model's own index
    _fr = {qq: S5['mem'][qq] / _M5 for qq in (0.90, 0.95)}
    _ord = np.lexsort((-_fr[0.95], -_fr[0.90]))[:9]   # ties broken on q=0.95
    memb = [(_surname(_sn.get(_sp[S5['elig'][i]], '')),
             _fr[0.90][i], _fr[0.95][i]) for i in _ord]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=.38)
    keys = list(chi[0.90])
    for off, (qq, c, mk) in enumerate([(0.90, NAVY, 'o'), (0.95, RED, 's')]):
        for i, k in enumerate(keys):
            lo, v, hi = chi[qq][k]
            axes[0].errorbar(i + (off - .5) * .20, v, yerr=[[v - lo], [hi - v]],
                             fmt=mk, ms=4.3, c=c, capsize=2.5, lw=1.1, mec='w', mew=.6)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(keys)
    axes[0].set_ylabel(r'$\chi$'); axes[0].set_ylim(0, .9); axes[0].set_xlim(-.5, 2.5)
    axes[0].legend([Line2D([0], [0], c=NAVY, marker='o', ms=4.3, lw=1.1),
                    Line2D([0], [0], c=RED, marker='s', ms=4.3, lw=1.1)],
                   ['$q=0.90$', '$q=0.95$'], loc='lower left', fontsize=7.5)
    axes[0].set_title('(a)  cross-surface upper tail dependence', fontsize=8.5, loc='left')
    yy = np.arange(len(memb))
    axes[1].barh(yy - .19, [b_ for _, b_, _ in memb], height=.36, color=NAVY, ec='none')
    axes[1].barh(yy + .19, [c_ for _, _, c_ in memb], height=.36, color=RED, ec='none')
    axes[1].set_yticks(yy); axes[1].set_yticklabels([a_ for a_, _, _ in memb])
    for t, (nm, _, _) in zip(axes[1].get_yticklabels(), memb):
        if nm in ('Djokovic', 'Federer', 'Nadal'):
            t.set_color(RED)
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel('posterior probability')
    axes[1].legend([plt.Rectangle((0, 0), 1, 1, fc=NAVY), plt.Rectangle((0, 0), 1, 1, fc=RED)],
                   ['$q=0.90$', '$q=0.95$'], loc='lower right', fontsize=7.5)
    axes[1].set_title('(b)  extreme on all three surfaces', fontsize=8.5, loc='left')
    plt.savefig(str(CFG.OUT / 'fig5_surfaces.pdf')); plt.close()

# ================= FIG 6: expected count =================
fig,ax=plt.subplots(figsize=(7.2,2.6))
for k,c,ls,l in [(2,NAVY,'-',r'$\bar N=2$'),(1,RED,'-',r'$\bar N=1$')]:
    s=pd.Series(F['CNT'][k].mean(1),index=T)
    a=s.groupby(s.index.year).mean(); a=a[a.index>=1978]
    ax.plot(a.index,a.values,c=c,lw=1.5,ls=ls,label=l)
ax.axhline(3,ls=(0,(4,3)),c='0.35',lw=.8)
ax.text(2024.5,3.06,'three concurrent',fontsize=7,color='0.35',va='bottom',ha='right')
for a_,b_ in CFG.ERAS:
    ax.axvspan(a_,b_,color='0.5',alpha=.05,lw=0)
    ax.axvline(a_,color='0.6',lw=.5,ls=':')
ax.set_ylabel(r'$E[N_t\mid\mathbf{y}]$');ax.set_xlabel('year')
ax.set_xlim(1978,2025);ax.set_ylim(0,4.15)
ax.legend(loc='lower right',fontsize=8,ncol=2)
plt.savefig(str(CFG.OUT/'fig6_ent.pdf'));plt.close()
print("figures rebuilt")
