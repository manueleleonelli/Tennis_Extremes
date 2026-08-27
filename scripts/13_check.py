"""Compares a fresh pipeline run against the numbers currently in the
manuscript and flags anything that has moved. Run after 04 (and optionally 05).

Green OK  : within tolerance, so the manuscript value stands.
Yellow ~  : moved by more than tolerance but less than twice it. Monte Carlo
            noise is plausible; regenerate the table and move on.
Red  DIFF : moved substantially. Investigate before submitting.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pickle
from tennisdom import core, config as C

PV = json.load(open(pathlib.Path(__file__).resolve().parents[1] /
                    "tennisdom" / "paper_values.json"))
R = pickle.load(open(C.OUT / "04_estimands.pkl", "rb"))
D = pickle.load(open(C.OUT / "03_posterior.pkl", "rb"))
yr, W0, NW = D['yr'], D['W0'], D['NW']
pids, names, code = D['pids'], D['names'], D['code']
M = R['CNT'][2].shape[1]
ERAS = {f"{a}-{b}": (a, b) for a, b in C.ERAS}
rows, nbad, nwarn = [], 0, 0

def cmp(label, paper, fresh, tol):
    global nbad, nwarn
    if paper is None or fresh is None or (isinstance(fresh, float) and np.isnan(fresh)):
        rows.append((label, paper, fresh, "  --")); return
    d = abs(fresh - paper)
    if d <= tol:
        tag = "  OK"
    elif d <= 2 * tol:
        tag = "   ~"; nwarn += 1
    else:
        tag = "DIFF"; nbad += 1
    rows.append((label, paper, fresh, tag))

inv = {v: k for k, v in C.PLAYERS.items()}
for nm, val in PV.get("peaks_median", {}).items():
    p = inv.get(nm)
    cmp(f"peak {nm}", val, float(np.median(R['peak'][p])) if p in R['peak'] else None, 12)
for k, val in PV.get("pairwise", {}).items():
    a, b = k.split(">")
    cmp(f"P({k})", val, float((R['peak'][inv[a]] > R['peak'][inv[b]]).mean()), 0.04)
soj = R['SOJ'][2]
for nm, val in PV.get("sojourn_median_meanN2", {}).items():
    i = code.get(inv[nm])
    cmp(f"sojourn {nm}", val, float(np.median(soj[i])) if i is not None else None, 25)
for tag, key in (("meanN2", 2), ("meanN1", 1)):
    Cn = R['CNT'][key]; p3 = (Cn >= 3).mean(1)
    for era, (a, b) in ERAS.items():
        sl = (yr >= a) & (yr <= b) & (np.arange(NW) >= W0)
        ref = PV["concurrence"].get(tag, {}).get(era)
        if ref is None: continue
        cmp(f"{tag} {era} E[N]", ref["E_N"], float(Cn[sl].mean()), 0.08)
        cmp(f"{tag} {era} meanP3", ref["meanP3"], float(p3[sl].mean()), 0.02)
        cmp(f"{tag} {era} total", ref["total"], float(np.median((Cn[sl] >= 3).sum(0))), 20)
        cmp(f"{tag} {era} wks p>=.5", ref["wks_p50"], float((p3[sl] >= 0.5).sum()), 25)
Cn = R['CNT'][1]
for era in [k for k in ERAS if k in PV["persistence_meanN1"]["lag"]]:
    a, b = ERAS[era]; sl = (yr >= a) & (yr <= b) & (np.arange(NW) >= W0)
    for L, val in zip(PV["persistence_meanN1"]["lags"], PV["persistence_meanN1"]["lag"][era]):
        v = [core.lag_persistence(Cn[sl, d], L) for d in range(M)]
        v = [x for x in v if not np.isnan(x)]
        cmp(f"pi_{L} {era}", val, float(np.median(v)) if v else None, 0.04)
    for w, val in zip(PV["persistence_meanN1"]["windows"], PV["persistence_meanN1"]["block"][era]):
        cmp(f"beta_{w} {era}", val,
            float(np.median([core.block_persistence(Cn[sl, d], w) for d in range(M)])), 0.03)
for era, ref in PV.get("gpd", {}).items():
    g = np.array(R['GPD'][ERAS[era]], float)
    cmp(f"gpd {era} n", ref["n"], float(np.median(R['NEXC'][ERAS[era]])), 40)
    cmp(f"gpd {era} sigma", ref["sigma"], float(np.nanmedian(g[:, 0])), 8)
    cmp(f"gpd {era} xi", ref["xi"], float(np.nanmedian(g[:, 1])), 0.05)
    cmp(f"gpd {era} rl50", ref["rl50"], float(np.nanmedian(g[:, 3])), 12)
for era, vals in PV.get("top5_profile", {}).items():
    for k, val in enumerate(vals):
        cmp(f"profile {era} rank{k+1}", val, float(np.median(R['PROF'][ERAS[era]][k])), 10)

w = max(len(r[0]) for r in rows) + 2
print(f"{'quantity':<{w}}{'paper':>9}{'fresh':>9}   status")
print("-" * (w + 28))
for lab, p, f, tag in rows:
    ps = "   --" if p is None else f"{p:9.3f}" if abs(p) < 10 else f"{p:9.1f}"
    fs = "   --" if f is None else f"{f:9.3f}" if abs(f) < 10 else f"{f:9.1f}"
    print(f"{lab:<{w}}{ps}{fs}   {tag}")
print("-" * (w + 28))
print(f"{len(rows)} checks: {nbad} flagged DIFF, {nwarn} marginal")
print("\nNotes:")
for k, v in PV.get("known_discrepancies", {}).items():
    print(f"  {k}: {v}")
if (C.OUT / "05_surfaces.pkl").exists():
    S = pickle.load(open(C.OUT / "05_surfaces.pkl", "rb"))
    print(f"\nsurfaces: rho fresh={S['RHO']} paper={PV['surfaces']['rho']}")
    for q, key in ((0.90, "chi_q90"), (0.95, "chi_q95")):
        for k, lab in (("HC", "hard-clay"), ("HG", "hard-grass"), ("CG", "clay-grass")):
            print(f"  chi({lab}) q={q}: paper={PV['surfaces'][key][lab]:.2f} "
                  f"fresh={np.median(S['res'][q]['chi'][k]):.2f}")
else:
    print("\n(run 05_surfaces.py to check the surface numbers too)")
if PV.get("pending"):
    print("\npending:", PV["pending"])
