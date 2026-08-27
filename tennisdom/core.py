"""
Core routines for "How exceptional was the Big Three era?".

Everything the paper reports is built from this module. Three groups of
functions:

  * data           : load_matches, encode
  * model          : dbt_filter, dbt_filter_surface, flatten, ffbs, ffbs_mv
  * estimands      : panel_index, relative, calibrate_threshold, and the
                     functionals of Section 3 (rate, window-average, boundary)

Conventions
-----------
Latent strengths live on the natural logit scale. To express them in
"rating points" comparable to a conventional Elo scale, multiply by
ELO_SCALE = 400/log(10) = 173.72. The calibration slope PLATT estimated in
script 03 is applied on top of that wherever strengths are reported.
"""
import numpy as np
import pandas as pd

Q3 = 3.0 / np.pi**2          # variance inflation constant of the logistic link
ELO_SCALE = 400.0 / np.log(10)
SURFACES = ["Hard", "Clay", "Grass"]


# --------------------------------------------------------------------- data
PARTICLES = {"del", "de", "van", "von", "der", "di", "da", "la", "le"}


def surname(full):
    """Short display name, keeping nobiliary particles: 'del Potro', not 'Potro'."""
    parts = str(full).split()
    if not parts:
        return str(full)
    if len(parts) >= 2 and parts[-2].lower() in PARTICLES:
        return " ".join(parts[-2:])
    return parts[-1]


def load_matches(path, year_min=1968, year_max=2025):
    """Read the match archive and return a chronologically sorted frame."""
    m = pd.read_csv(path, low_memory=False)
    m["tourney_date"] = pd.to_datetime(m["tourney_date"], format="%Y%m%d",
                                       errors="coerce")
    m = m.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    y = m["tourney_date"].dt.year
    m = m[(y >= year_min) & (y <= year_max)]
    return m.sort_values(["tourney_date", "tourney_id", "match_num"]
                         if "match_num" in m.columns
                         else ["tourney_date", "tourney_id"]).reset_index(drop=True)


def encode(m):
    """Map player ids and dates to integer codes and weekly rating periods."""
    pids = pd.Index(sorted(set(m.winner_id) | set(m.loser_id)))
    code = {p: i for i, p in enumerate(pids)}
    wkv = m.tourney_date.values.astype("datetime64[W]")
    weeks = np.array(sorted(set(wkv.tolist())), dtype="datetime64[W]")
    wmap = {w: i for i, w in enumerate(weeks)}
    d = dict(
        pids=pids, code=code, weeks=weeks, NP=len(pids), NW=len(weeks),
        wi=m.winner_id.map(code).values.astype(np.int32),
        li=m.loser_id.map(code).values.astype(np.int32),
        wt=np.array([wmap[w] for w in wkv], np.int32),
        bo=pd.to_numeric(m.best_of, errors="coerce").fillna(3).values,
        names=pd.concat([
            m[["winner_id", "winner_name"]].rename(
                columns={"winner_id": "pid", "winner_name": "name"}),
            m[["loser_id", "loser_name"]].rename(
                columns={"loser_id": "pid", "loser_name": "name"})
        ]).drop_duplicates("pid").set_index("pid")["name"],
    )
    if "surface" in m.columns:
        d["sf"] = m.surface.map({s: i for i, s in enumerate(SURFACES)}).values
    o = np.argsort(d["wt"], kind="stable")
    for k in ("wi", "li", "wt", "bo"):
        d[k] = d[k][o]
    if "sf" in d:
        d["sf"] = d["sf"][o]
    d["order"] = o
    return d


# -------------------------------------------------------------------- model
def dbt_filter(wi, li, wt, bo, NP, NW, tau, sigma0=1.0, a5=1.15,
               n_newton=8, collect=True, single_step=False):
    """Dynamic Bradley-Terry filter, equations (9) to (13) of the paper.

    Within each weekly rating period the posterior mode solves (11). Setting
    single_step=True reproduces the Glicko update (12), i.e. one Fisher
    scoring step from the prior mean; the default iterates to convergence.

    Returns the prequential log-likelihood, the predictive accuracy, and (if
    collect) per player a list of (period, posterior mode, posterior variance).
    """
    st = np.searchsorted(wt, np.arange(NW), "left")
    en = np.searchsorted(wt, np.arange(NW), "right")
    mu = np.zeros(NP)
    var = np.full(NP, sigma0**2)
    seen = np.zeros(NP, bool)
    tau2 = tau**2
    ll = 0.0
    ncorr = 0
    ntot = 0
    store = [[] for _ in range(NP)] if collect else None
    sc = np.zeros(NP)
    inf = np.zeros(NP)
    for t in range(NW):
        var[seen] += tau2                                   # state propagation
        a, b = st[t], en[t]
        if a == b:
            continue
        W, L = wi[a:b], li[a:b]
        A = np.where(bo[a:b] >= 5, a5, 1.0)                  # format effect
        gW = A / np.sqrt(1 + Q3 * var[W])                    # equation (10)
        gL = A / np.sqrt(1 + Q3 * var[L])
        gp = A / np.sqrt(1 + Q3 * (var[W] + var[L]))
        p = 1 / (1 + np.exp(-gp * (mu[W] - mu[L])))          # one-step-ahead
        msk = seen[W] & seen[L]
        if msk.any():
            ll += np.log(np.clip(p[msk], 1e-12, 1)).sum()
            ncorr += int((p[msk] > 0.5).sum())
            ntot += int(msk.sum())
        act = np.unique(np.concatenate([W, L]))
        mp = mu[act].copy()
        vp = var[act].copy()
        pos = np.full(NP, -1, np.int64)
        pos[act] = np.arange(len(act))
        th = mp.copy()
        n_it = 1 if single_step else n_newton
        for _ in range(n_it):
            tW, tL = th[pos[W]], th[pos[L]]
            EW = 1 / (1 + np.exp(-gL * (tW - tL)))
            EL = 1 / (1 + np.exp(-gW * (tL - tW)))
            sc[:] = 0.0
            inf[:] = 0.0
            np.add.at(sc, W, gL * (1 - EW))
            np.add.at(sc, L, -gW * EL)
            np.add.at(inf, W, gL**2 * EW * (1 - EW))
            np.add.at(inf, L, gW**2 * EL * (1 - EL))
            th = th + (sc[act] - (th - mp) / vp) / (inf[act] + 1 / vp)
        I = np.maximum(inf[act], 1e-10)
        vpost = 1 / (1 / vp + I)
        if collect:
            for k, i in enumerate(act):
                store[i].append((t, th[k], vpost[k]))
        mu[act] = th
        var[act] = vpost
        seen[act] = True
    return ll, ncorr / max(ntot, 1), store


def dbt_filter_surface(wi, li, wt, bo, sf, NP, NW, tau, rho,
                       sigma0=1.0, a5=1.15, n_newton=8, collect_for=None):
    """Multivariate version: one state component per surface, equation (2).

    A match informs the component of its own surface directly through the same
    Laplace step, and the remaining components through the Kalman update (13).
    """
    R = np.full((3, 3), rho) + np.eye(3) * (1 - rho)
    Q = tau**2 * R
    Mu = np.zeros((NP, 3))
    Pv = np.tile(sigma0**2 * R, (NP, 1, 1))
    seen = np.zeros(NP, bool)
    ll = 0.0
    ncorr = 0
    ntot = 0
    st = np.searchsorted(wt, np.arange(NW), "left")
    en = np.searchsorted(wt, np.arange(NW), "right")
    store = {i: [] for i in (collect_for if collect_for is not None else [])}
    for t in range(NW):
        Pv[seen] += Q
        a, b = st[t], en[t]
        if a == b:
            continue
        for s in range(3):
            sel = np.where(sf[a:b] == s)[0]
            if len(sel) == 0:
                continue
            W = wi[a:b][sel]
            L = li[a:b][sel]
            A = np.where(bo[a:b][sel] >= 5, a5, 1.0)
            vW, vL = Pv[W, s, s], Pv[L, s, s]
            gW = A / np.sqrt(1 + Q3 * vW)
            gL = A / np.sqrt(1 + Q3 * vL)
            gp = A / np.sqrt(1 + Q3 * (vW + vL))
            p = 1 / (1 + np.exp(-gp * (Mu[W, s] - Mu[L, s])))
            msk = seen[W] & seen[L]
            if msk.any():
                ll += np.log(np.clip(p[msk], 1e-12, 1)).sum()
                ncorr += int((p[msk] > 0.5).sum())
                ntot += int(msk.sum())
            act = np.unique(np.concatenate([W, L]))
            pos = np.full(NP, -1, np.int64)
            pos[act] = np.arange(len(act))
            mp = Mu[act, s].copy()
            vp = Pv[act, s, s].copy()
            th = mp.copy()
            sc = np.zeros(NP)
            inf = np.zeros(NP)
            for _ in range(n_newton):
                tW, tL = th[pos[W]], th[pos[L]]
                EW = 1 / (1 + np.exp(-gL * (tW - tL)))
                EL = 1 / (1 + np.exp(-gW * (tL - tW)))
                sc[:] = 0.0
                inf[:] = 0.0
                np.add.at(sc, W, gL * (1 - EW))
                np.add.at(sc, L, -gW * EL)
                np.add.at(inf, W, gL**2 * EW * (1 - EW))
                np.add.at(inf, L, gW**2 * EL * (1 - EL))
                th = th + (sc[act] - (th - mp) / vp) / (inf[act] + 1 / vp)
            I = np.maximum(inf[act], 1e-10)
            v = 1.0 / I
            y = th + (v / vp) * (th - mp)                     # equation (14)
            Pa = Pv[act]
            K = Pa[:, :, s] / (Pa[:, s, s] + v)[:, None]      # equation (13)
            Mu[act] = Mu[act] + K * (y - Mu[act, s])[:, None]
            Pv[act] = Pa - K[:, :, None] * Pa[:, s, :][:, None, :]
            seen[act] = True
            for i in store:
                if pos[i] >= 0:
                    store[i].append((t, Mu[i].copy(), Pv[i].copy()))
    return ll, ncorr / max(ntot, 1), store


def flatten(store, NP):
    """Flatten per-player filter output into arrays with an offset index."""
    off = np.zeros(NP + 1, np.int64)
    T, M, P = [], [], []
    for i, s in enumerate(store):
        off[i + 1] = off[i] + len(s)
        for (t, m, p) in s:
            T.append(t)
            M.append(m)
            P.append(p)
    return np.array(T, np.int32), np.array(M), np.array(P), off


def ffbs(T, M, P, off, NP, tau, ndraw, rng):
    """Forward-filtering backward-sampling, equation (15), scalar state."""
    out = np.empty((len(T), ndraw), np.float32)
    tau2 = tau**2
    for i in range(NP):
        a, b = off[i], off[i + 1]
        if a == b:
            continue
        tt, mm, pp = T[a:b], M[a:b], P[a:b]
        K = b - a
        x = rng.normal(mm[K - 1], np.sqrt(pp[K - 1]), ndraw)
        out[b - 1] = x
        for k in range(K - 2, -1, -1):
            Q = (tt[k + 1] - tt[k]) * tau2
            J = pp[k] / (pp[k] + Q)
            x = rng.normal(mm[k] + J * (x - mm[k]),
                           np.sqrt(pp[k] * Q / (pp[k] + Q)))
            out[a + k] = x
    return out


def ffbs_mv(seq, tau, rho, ndraw, rng):
    """Backward sampling for the multivariate state; returns (K, ndraw, 3)."""
    R = np.full((3, 3), rho) + np.eye(3) * (1 - rho)
    Qm = tau**2 * R
    ts = np.array([x[0] for x in seq])
    ms = np.array([x[1] for x in seq])
    Ps = np.array([x[2] for x in seq])
    K = len(ts)
    X = np.empty((K, ndraw, 3))
    L = np.linalg.cholesky(Ps[K - 1] + 1e-10 * np.eye(3))
    X[K - 1] = ms[K - 1] + rng.standard_normal((ndraw, 3)) @ L.T
    for k in range(K - 2, -1, -1):
        S = Ps[k] + (ts[k + 1] - ts[k]) * Qm
        Si = np.linalg.inv(S)
        J = Ps[k] @ Si
        C = Ps[k] - Ps[k] @ Si @ Ps[k]
        C = (C + C.T) / 2 + 1e-10 * np.eye(3)
        Lc = np.linalg.cholesky(C)
        X[k] = ms[k] + (X[k + 1] - ms[k]) @ J.T + \
            rng.standard_normal((ndraw, 3)) @ Lc.T
    return X


# ----------------------------------------------------------------- estimands
def panel_index(T, off, NP, NW, active_weeks=52):
    """Map (player, period) cells to the last filtered observation at or before
    that period, dropping players inactive for more than active_weeks."""
    pl, wk, ptr = [], [], []
    for i in range(NP):
        a, b = off[i], off[i + 1]
        if b == a:
            continue
        tt = T[a:b]
        lo, hi = tt[0], min(tt[-1] + active_weeks, NW - 1)
        g = np.arange(lo, hi + 1)
        j = np.searchsorted(tt, g, "right") - 1
        keep = (g - tt[j]) <= active_weeks
        g, j = g[keep], j[keep]
        pl.append(np.full(len(g), i, np.int32))
        wk.append(g.astype(np.int32))
        ptr.append((a + j).astype(np.int64))
    pl = np.concatenate(pl)
    wk = np.concatenate(wk)
    ptr = np.concatenate(ptr)
    o = np.lexsort((pl, wk))
    pl, wk, ptr = pl[o], wk[o], ptr[o]
    return (pl, wk, ptr,
            np.searchsorted(wk, np.arange(NW), "left"),
            np.searchsorted(wk, np.arange(NW), "right"))


def relative(values, wstart, wend, NW, t0=0, band=(9, 100), min_active=130):
    """Relative strength, equation (3): subtract the mean of the players ranked
    band[0]+1 to band[1] among those active in the period."""
    rel = np.full_like(values, -1e9)
    for t in range(t0, NW):
        a, b = wstart[t], wend[t]
        if b - a < min_active:
            continue
        x = values[a:b]
        rel[a:b] = x - np.sort(x)[::-1][band[0]:band[1]].mean()
    return rel


def calibrate_threshold(rel_window, n_periods, mean_count):
    """Threshold such that the mean exceedance count equals mean_count exactly,
    as described in Section 3.4."""
    nk = int(mean_count * n_periods)
    return np.partition(rel_window, -nk)[-nk]


# --- window-average functionals (Definition 1) ---
def rate(counts, c=3):
    return float((counts >= c).mean())


def lag_persistence(counts, k, c=3):
    b = (counts >= c).astype(int)
    denom = b[:-k].sum()
    return float((b[:-k] & b[k:]).sum() / denom) if denom >= 5 else np.nan


def block_persistence(counts, w, c=3):
    b = (counts >= c).astype(int)
    if len(b) < w:
        return 0.0
    return float((np.convolve(b, np.ones(w, int), "valid") == w).mean())


# --- boundary functionals (Definition 2) ---
def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def run_lengths(mask):
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return out


def gap_tolerant_run(mask, g):
    """Longest run when runs separated by at most g gaps are merged."""
    out, c, gap = [], 0, 0
    for v in mask:
        if v:
            c += 1
            gap = 0
        else:
            gap += 1
            if gap <= g and c > 0:
                c += 1
            else:
                if c:
                    out.append(c - min(gap - 1, g))
                c = 0
                gap = 0
    if c:
        out.append(c)
    return max(out) if out else 0


def extremal_index(exceed_positions, N):
    """Ferro-Segers intervals estimator."""
    if N < 4:
        return np.nan
    Tg = np.diff(exceed_positions)
    if Tg.max() <= 2:
        num, den = 2 * Tg.sum()**2, (N - 1) * np.sum(Tg**2)
    else:
        num, den = 2 * np.sum(Tg - 1)**2, (N - 1) * np.sum((Tg - 1) * (Tg - 2))
    return min(num / den, 1.0) if den > 0 else np.nan


# --- peaks over threshold (equation 4) ---
def gpd_nll(par, x):
    sc, sh = par
    if sc <= 0 or abs(sh) < 1e-8:      # guard the exponential limit
        if sc <= 0:
            return 1e10
        return len(x) * np.log(sc) + np.sum(x) / sc
    z = 1 + sh * x / sc
    if np.any(z <= 0):
        return 1e10
    return len(x) * np.log(sc) + (1 + 1 / sh) * np.sum(np.log(z))


def fit_gpd(excesses):
    from scipy import optimize
    r = optimize.minimize(gpd_nll, [np.std(excesses) + 1e-6, 0.0],
                          args=(excesses,), method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 3000})
    return r.x


def return_level(u, sigma, xi, zeta, m_periods):
    if abs(xi) < 1e-6:
        return u + sigma * np.log(m_periods * zeta)
    return u + sigma / xi * ((m_periods * zeta)**xi - 1)


def rpg(z, rng, trunc=120):
    """Polya-Gamma PG(1, z) by the truncated sum-of-gammas representation."""
    kk = (np.arange(1, trunc + 1) - 0.5)**2
    g = rng.exponential(size=(len(z), trunc))
    den = kk[None, :] + (z**2)[:, None] / (4 * np.pi**2)
    return (g / den).sum(1) / (2 * np.pi**2)
