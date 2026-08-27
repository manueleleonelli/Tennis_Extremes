"""Section 6.1. Data summary; emits Table 5 (selected hyperparameters) inputs."""
import sys, json; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from tennisdom import core, config as C

m = core.load_matches(C.DATA)
e = core.encode(m)
print(f"matches                 {len(m):,}")
print(f"players                 {e['NP']:,}")
print(f"weekly periods (all)    {e['NW']:,}")
print(f"surface counts:\n{m.surface.value_counts(dropna=False).to_string()}")
print(f"best_of counts:\n{pd.to_numeric(m.best_of,errors='coerce').value_counts().to_string()}")
w0 = int(np.searchsorted(pd.to_datetime(e['weeks']).year.values, C.YEAR_START))
print(f"periods in window       {e['NW']-w0:,}")
sub = m[pd.to_datetime(m.tourney_date).dt.year >= C.YEAR_START]
s3 = sub[sub.surface.isin(core.SURFACES)]
print(f"three-surface matches   {len(s3):,}")
print(f"three-surface players   {len(set(s3.winner_id)|set(s3.loser_id)):,}")
json.dump({"matches": len(m), "players": int(e['NP']),
           "periods_window": int(e['NW']-w0), "surface_matches": len(s3)},
          open(C.OUT/"01_data.json","w"), indent=1)
