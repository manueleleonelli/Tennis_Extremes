"""Figure 8, posterior strength trajectories by era.

Laid out from C.ERAS: a single row for three eras or fewer, otherwise a grid of
two rows. The five players drawn in each panel are those chosen by
09_trajectories.py on the basis of how often they held a top-three position.
"""
import sys, pathlib, warnings, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle, matplotlib
from tennisdom import core, config as C
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size':8.5,'font.family':'serif','mathtext.fontset':'cm',
 'axes.grid':True,'grid.alpha':.20,'grid.linewidth':.5,'figure.dpi':200,
 'savefig.bbox':'tight','axes.linewidth':.7,'legend.frameon':True,
 'legend.framealpha':.93,'legend.edgecolor':'0.8','legend.fancybox':False})

D = pickle.load(open(C.OUT/'traj.pkl','rb'))
TR, U1, U2, weeks, SHOW, NM = D['TR'], D['U1'], D['U2'], D['weeks'], D['SHOW'], D['names']
T = pd.to_datetime(weeks); yr = T.year.values
PAL = ['#1F3B63','#B5443F','#3F7A52','#C7803A',
       '#6B5B95','#8C8C8C','#4E8FA8','#8B5E3C']
NSHOW = 8
ERAS = [(f'{a}-{b}', a, b) for a, b in C.ERAS]
n = len(ERAS)
nrow = 1 if n <= 3 else 2
ncol = math.ceil(n/nrow)
span = [b-a+1 for _, a, b in ERAS]
equal = max(span) - min(span) <= 1
kw = {'wspace':0.06} if equal else {'wspace':0.06,'width_ratios':span[:ncol]}
fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 3.6*nrow), sharey=True, gridspec_kw=kw)
axes = np.atleast_1d(axes).ravel()
u1, u2 = np.median(U1), np.median(U2)

for ax, (lab, a, b) in zip(axes, ERAS):
    sl = (yr>=a)&(yr<=b)
    ax.axhline(u1, color='0.35', lw=.8, ls=(0,(4,3)), zorder=1)
    ax.axhline(u2, color='0.55', lw=.7, ls=(0,(1,2)), zorder=1)
    for c, p in zip(PAL, SHOW[lab][:NSHOW]):
        if p not in TR: continue
        y  = np.nanmedian(TR[p], axis=0)[sl]
        lo = np.nanpercentile(TR[p], 10, axis=0)[sl]
        hi = np.nanpercentile(TR[p], 90, axis=0)[sl]
        x  = T[sl]; ok = ~np.isnan(y)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=c, alpha=.10, lw=0)
        ax.plot(x[ok], y[ok], c=c, lw=1.25)
    ax.set_xlim(pd.Timestamp(f'{a}-01-01'), pd.Timestamp(f'{b}-12-31'))
    ax.set_title(lab, fontsize=8.5, pad=3)
    step = 5 if (b-a) > 12 else 4
    ticks = list(range(a+1, b+1, step))
    ax.set_xticks([pd.Timestamp(f'{v}-01-01') for v in ticks])
    ax.set_xticklabels([str(v) for v in ticks])
    ax.legend([Line2D([0],[0],c=c,lw=1.25) for c,_ in zip(PAL, SHOW[lab][:NSHOW])],
              [core.surname(NM.get(p,p)) for p in SHOW[lab][:NSHOW]],
              loc='upper left', fontsize=6.0, handlelength=1.0, ncol=2,
              borderpad=.28, labelspacing=.16, columnspacing=.8)
for ax in axes[len(ERAS):]:
    ax.axis('off')
for r in range(nrow):
    axes[r*ncol].set_ylabel('relative strength (points above field)')
axes[0].set_ylim(-30, 740)
# label the thresholds in the panel with the most clear space above its lines
_i = 1 if len(ERAS) > 1 else 0
_xa = pd.Timestamp(f'{ERAS[_i][2]}-09-01')
axes[_i].text(_xa, u1 + 16, r'$\bar N=1$', fontsize=7, color='0.35', ha='right')
axes[_i].text(_xa, u2 + 16, r'$\bar N=2$', fontsize=7, color='0.55', ha='right')
plt.savefig(str(C.OUT/'fig8_trajectories.pdf')); plt.close(); print('ok')
