"""Writes the body of every LaTeX table in the paper from whatever pickles are
present in out/, so nothing has to be transcribed by hand.

Run after 04 (and 05, 08 if you have them). Output goes to out/tables/.
Paste each file into the matching table in the manuscript.
"""
import sys, pathlib, pickle, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from tennisdom import core, config as C

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



TD = C.OUT / "tables"; TD.mkdir(exist_ok=True)
def w(name, txt):
    (TD / name).write_text(txt.rstrip() + "\n"); print(f"  wrote tables/{name}")
def q(x, p=(2.5, 50, 97.5)):
    a = np.asarray(x, float); a = a[~np.isnan(a)]
    return np.percentile(a, p) if len(a) else [np.nan]*3

R = pickle.load(open(C.OUT / "04_estimands.pkl", "rb"))
D = pickle.load(open(C.OUT / "03_posterior.pkl", "rb"))
yr, W0, NW, code, names, pids = D['yr'], D['W0'], D['NW'], D['code'], D['names'], D['pids']
M = R['CNT'][2].shape[1]
E3 = list(C.ERAS)
ERAS = [e for e in C.ERAS]
print("writing LaTeX table bodies")

# ---- Table 6: peaks ----
order = sorted(C.PLAYERS, key=lambda p: -np.median(R['peak'][p]))
w("tab_peaks.tex", "\n".join(
    f"{C.PLAYERS[p]:<10} & {q(R['peak'][p])[1]:.0f} & {q(R['peak'][p])[0]:.0f} & {q(R['peak'][p])[2]:.0f} \\\\"
    for p in order))

# ---- Table 7: pairwise ----
K = ['D643','F324','N409','B058','M047','L018']
rows = []
for a in K:
    cells = ["--" if a==b else f"{(R['peak'][a]>R['peak'][b]).mean():.2f}" for b in K]
    rows.append(f"{C.PLAYERS[a]:<9} & " + " & ".join(cells) + " \\\\")
w("tab_pairwise.tex", "\n".join(rows))

# ---- Table 8: concurrence ----
rows = []
for k in (2, 1):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{$\\bar N = {k}$ player{'s' if k>1 else ''} per period}}}}\\\\")
    Cn = R['CNT'][k]; p3 = (Cn >= 3).mean(1)
    for a, b in ERAS:
        sl = (yr>=a)&(yr<=b)&(np.arange(NW)>=W0)
        tw = (Cn[sl] >= 3).sum(0)
        rows.append(f"{a}--{b} & {int(sl.sum())} & {Cn[sl].mean():.2f} & {p3[sl].mean():.3f} & "
                    f"{np.median(tw):.0f} & [{np.percentile(tw,2.5):.0f}, {np.percentile(tw,97.5):.0f}] & "
                    f"{int((p3[sl]>=0.5).sum())} \\\\")
    if k == 2: rows.append("\\midrule")
w("tab_conc.tex", "\n".join(rows))

# ---- Table 9: persistence profiles at mean N = 1 ----
Cn = R['CNT'][1]; lag_rows, blk_rows = [], []
for a, b in E3:
    sl = (yr>=a)&(yr<=b)&(np.arange(NW)>=W0)
    lags = []
    for L in (1,4,13,26,52):
        v = [core.lag_persistence(Cn[sl,d], L) for d in range(M)]
        v = [x for x in v if not np.isnan(x)]
        lags.append(f"{np.median(v):.2f}" if len(v)>5 else "--")
    blk = [f"{np.median([core.block_persistence(Cn[sl,d],ww) for d in range(M)]):.2f}"
           for ww in (4,13,26,52,104)]
    lag_rows.append(f"{a}--{b} & " + " & ".join(lags) + " \\\\")
    blk_rows.append(f"{a}--{b} & " + " & ".join(blk) + " \\\\")
w("tab_persist.tex", "\n".join(lag_rows) + "\n\\midrule\n" + "\n".join(blk_rows))

# ---- Table 10: sojourn ----
soj = R['SOJ'][2]; med = np.median(soj, axis=1)
rows = []
for i in np.argsort(med)[::-1][:10]:
    lo, hi = np.percentile(soj[i], [2.5, 97.5])
    rows.append(f"{_surname(names.get(pids[i],'')):<9} & {np.median(soj[i]):.0f} & {lo:.0f} & {hi:.0f} \\\\")
w("tab_sojourn.tex", "\n".join(rows))

# ---- Table 11: GPD, with return-level intervals ----
rows = []
for e in E3:
    g = np.array(R['GPD'][e], float)
    s_, x_, r10, r50 = (q(g[:,i]) for i in range(4))
    rows.append(f"{e[0]}--{e[1]} & {np.median(R['NEXC'][e]):.0f} & "
                f"{s_[1]:.0f} [{s_[0]:.0f}, {s_[2]:.0f}] & "
                f"${x_[1]:+.2f}$ [${x_[0]:+.2f}$, ${x_[2]:+.2f}$] & "
                f"{r10[1]:.0f} [{r10[0]:.0f}, {r10[2]:.0f}] & "
                f"{r50[1]:.0f} [{r50[0]:.0f}, {r50[2]:.0f}] \\\\")
w("tab_gpd.tex", "\n".join(rows))
print("\nextremal index, for the prose in Section 6.4:")
for e in E3:
    t = q(R['THETA'][e]); print(f"  {e[0]}-{e[1]}: {t[1]:.3f} [{t[0]:.3f}, {t[2]:.3f}]")

# ---- optional: surfaces ----
p5 = C.OUT / "05_surfaces.pkl"
if p5.exists():
    S = pickle.load(open(p5, "rb")); n = len(S['elig'])
    inv = {v:k for k,v in code.items()}
    rows = []
    for k, lab in (("HC","Hard--clay"), ("HG","Hard--grass"), ("CG","Clay--grass")):
        cells = []
        for qq in (0.90, 0.95):
            c = q(S['res'][qq]['chi'][k]); et = q(S['res'][qq]['eta'][k])
            cells += [f"{c[1]:.2f} [{c[0]:.2f}, {c[2]:.2f}]", f"{et[1]:.2f} [{et[0]:.2f}, {et[2]:.2f}]"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    a90, a95 = q(S['res'][0.90]['a3']), q(S['res'][0.95]['a3'])
    rows.append("\\midrule")
    rows.append(f"Extreme on all three & \\multicolumn{{2}}{{c}}{{{a90[1]:.0f} [{a90[0]:.0f}, {a90[2]:.0f}]}} & "
                f"\\multicolumn{{2}}{{c}}{{{a95[1]:.0f} [{a95[0]:.0f}, {a95[2]:.0f}]}} \\\\")
    rows.append(f"Under independence & \\multicolumn{{2}}{{c}}{{{n*0.1**3:.2f}}} & \\multicolumn{{2}}{{c}}{{{n*0.05**3:.2f}}} \\\\")
    rows.append(f"Under perfect dependence & \\multicolumn{{2}}{{c}}{{{n*0.1:.0f}}} & \\multicolumn{{2}}{{c}}{{{n*0.05:.0f}}} \\\\")
    w("tab_surf.tex", "\n".join(rows))
    Mm = {qq: S['mem'][qq]/S['peaks'].shape[2] for qq in (0.90,0.95)}
    spids, snames = S['pids'], S['names']       # the surface model's own index
    ordr = np.lexsort((-Mm[0.95], -Mm[0.90]))[:10]    # ties broken on q=0.95
    rows = []
    for i in ordr:
        pid = spids[S['elig'][i]]
        nm = _surname(snames.get(pid, pid))
        pc = [np.median(S['pct'][i, s]) for s in range(3)]
        rows.append(f"{nm:<11} & {Mm[0.90][i]:.2f} & {Mm[0.95][i]:.2f} & "
                    f"{pc[0]:.3f} & {pc[1]:.3f} & {pc[2]:.3f} \\\\")
    w("tab_surfmemb.tex", "\n".join(rows))
    print(f"  rho = {S['RHO']}")
else:
    print("  (no 05_surfaces.pkl; skipping Tables 12 and 13)")

# ---- optional: sensitivity and profile ----
p8 = C.OUT / "08_sensitivity.pkl"
if p8.exists():
    S8 = pickle.load(open(p8, "rb"))
    rows = []
    for k in range(5):
        cells = []
        for e in E3:
            v = q(S8['PROF'][e][k]); cells.append(f"{v[1]:.0f} [{v[0]:.0f}, {v[2]:.0f}]")
        rows.append(f"{k+1} & " + " & ".join(cells) + " \\\\")
    w("tab_prof.tex", "\n".join(rows))
    BN = list(S8['peak'].keys())
    rows = ["\\multicolumn{4}{l}{\\textit{Peak relative strength}}\\\\"]
    for p in ['D643','B058','M047','L018','F324','N409','S402']:
        rows.append(f"{C.PLAYERS[p]:<9} & " + " & ".join(
            f"{np.median(S8['peak'][b][p]):.0f}" for b in BN) + " \\\\")
    rows.append("\\midrule\n\\multicolumn{4}{l}{\\textit{Posterior probability of ordering}}\\\\")
    for a,b in [('D643','F324'),('D643','B058'),('F324','B058'),('N409','B058')]:
        rows.append(f"$P(\\text{{{C.PLAYERS[a]}}}>\\text{{{C.PLAYERS[b]}}})$ & " + " & ".join(
            f"{(S8['peak'][bn][a]>S8['peak'][bn][b]).mean():.2f}" for bn in BN) + " \\\\")
    for kk in C.MEAN_COUNTS:
        rows.append(f"\\midrule\n\\multicolumn{{4}}{{l}}{{\\textit{{Mean $P(N_t\\geq3)$, threshold $\\bar N={kk}$}}}}\\\\")
        for a,b in E3:
            sl=(yr>=a)&(yr<=b)&(np.arange(NW)>=W0)
            rows.append(f"{a}--{b} & " + " & ".join(
                f"{((S8['CNT'][bn][kk][sl]>=3).mean(1)).mean():.3f}" for bn in BN) + " \\\\")
    rows.append("\\midrule\n\\multicolumn{4}{l}{\\textit{Lag persistence $\\pi_{52}$, threshold $\\bar N=1$}}\\\\")
    for a,b in C.ERAS:
        sl=(yr>=a)&(yr<=b)&(np.arange(NW)>=W0)
        cells=[]
        for bn in BN:
            v=[core.lag_persistence(S8['CNT'][bn][1][sl,d],52) for d in range(S8['CNT'][bn][1].shape[1])]
            v=[x for x in v if not np.isnan(x)]
            cells.append(f"{np.median(v):.2f}" if v else "--")
        rows.append(f"{a}--{b} & " + " & ".join(cells) + " \\\\")
    w("tab_sens.tex", "\n".join(rows))
    print("\ntop-three occupancy, for the prose in Section 6.3:")
    for e in E3:
        tot_=S8['OCC'][e].sum()
        print(f"  {e[0]}-{e[1]}: " + ", ".join(
            f"{_surname(names.get(pids[i],''))} {100*S8['OCC'][e][i]/max(tot_/3,1):.0f}%"
            for i in np.argsort(S8['OCC'][e])[::-1][:6]))
else:
    print("  (no 08_sensitivity.pkl; skipping Tables 4 and 14)")

# ---- optional: simulation ----
sim = C.OUT / "sim3.csv"
if sim.exists():
    if not (C.OUT / "06_ran.flag").exists():
        print("\n  *** WARNING: sim3.csv exists but no 06_ran.flag, so this file was\n"
              "      not produced by 06_simulation.py in this working directory.\n"
              "      Do not trust Tables 1 and 2 until you have run it. ***")
    Rs = pd.read_csv(sim)
    lbl = [('L','Longest run'),('mcs','Mean run length'),('L4','Tolerance $g=4$'),
           ('L8','Tolerance $g=8$'),('p1','Lag persistence $\\pi_1$'),
           ('p8','Lag persistence $\\pi_8$'),('tot','Rate functional')]
    rows=[]
    for key,name in lbl:
        a=(Rs[f'{key}_elo']/np.maximum(Rs[f'{key}_true'],1e-9)).replace([np.inf,-np.inf],np.nan).dropna()
        b=(Rs[f'{key}_post']/np.maximum(Rs[f'{key}_true'],1e-9)).replace([np.inf,-np.inf],np.nan).dropna()
        rows.append(f"{name:<27} & {a.mean():.2f} & {a.median():.2f} & {a.std():.2f} & "
                    f"{b.mean():.2f} & {b.median():.2f} & {b.std():.2f} \\\\")
    w("tab_sim.tex", "\n".join(rows))
    rows=[]
    for ww in (4,13,26,52):
        t,e_,p_=Rs[f'B{ww}_true'],Rs[f'B{ww}_elo'],Rs[f'B{ww}_post']
        rows.append(f"{ww} & {t.mean():.3f} & {e_.mean():.3f} & {p_.mean():.3f} & "
                    f"{(e_-t).abs().mean():.3f} & {(p_-t).abs().mean():.3f} \\\\")
    w("tab_simblock.tex", "\n".join(rows))
else:
    print("  (no sim3.csv; skipping Tables 1 and 2)")
print(f"\nall written to {TD}")
