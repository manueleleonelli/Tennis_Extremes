"""Fits the model at the selected hyperparameters, draws N_DRAWS trajectories
by FFBS, and caches everything downstream scripts need."""
import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pickle
from tennisdom import core, config as C

print("hyperparameters:", C.describe())
m = core.load_matches(C.DATA); e = core.encode(m)
_,acc,store = core.dbt_filter(e['wi'],e['li'],e['wt'],e['bo'],e['NP'],e['NW'],
                              tau=C.TAU, sigma0=C.SIGMA0, a5=C.A5)
print("predictive accuracy", round(acc,4))
T,M,P,off = core.flatten(store, e['NP'])
pl,wk,ptr,ws,we = core.panel_index(T,off,e['NP'],e['NW'],C.ACTIVE_WEEKS)
rng = np.random.default_rng(C.SEED)
samp = core.ffbs(T,M,P,off,e['NP'],C.TAU,C.N_DRAWS,rng)
yr = pd.to_datetime(e['weeks']).year.values
W0 = int(np.searchsorted(yr, C.YEAR_START))
print("panel cells", len(pl), "| window periods", e['NW']-W0)
pickle.dump(dict(T=T,M=M,P=P,off=off,pl=pl,wk=wk,ptr=ptr,ws=ws,we=we,
                 samp=samp,weeks=e['weeks'],yr=yr,W0=W0,
                 pids=list(e['pids']),code=e['code'],names=e['names'],NW=e['NW'],NP=e['NP']),
            open(C.OUT/"03_posterior.pkl","wb"))
print("cached ->", C.OUT/"03_posterior.pkl")
