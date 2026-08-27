"""Fixed choices used throughout. Values marked ESTIMATED are outputs of
script 03 and are hard-coded here so later scripts need not refit."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "atp_matches_1968_2025.csv.gz"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

YEAR_START = 1978          # analysis window, justified in Section 6.1
YEAR_END = 2025
ACTIVE_WEEKS = 52
BAND = (9, 100)            # reference band of equation (3): ranks 10 to 100
MIN_ACTIVE = 130
N_DRAWS = 300
SEED = 20260822

# Hyperparameters. These defaults are overwritten automatically by whatever
# scripts/02_hyperparameters.py estimated, if it has been run, so that config
# and the fitted model can never disagree. Delete out/02_hyper.json to revert.
TAU = 0.0473
TAU_SD = 0.00080
SIGMA0 = 1.0
A5 = 1.15
PLATT = 0.851              # recalibration slope, Section 6.2

_fitted = OUT / "02_hyper.json"
if _fitted.exists():
    import json as _json
    _h = _json.load(open(_fitted))
    TAU = float(_h.get("tau", TAU))
    TAU_SD = float(_h.get("tau_sd", TAU_SD))
    SIGMA0 = float(_h.get("sigma0", SIGMA0))
    A5 = float(_h.get("a5", A5))
    PLATT = float(_h.get("platt_b", PLATT))
    _FROM_FIT = True
else:
    _FROM_FIT = False

def describe():
    src = "fitted by script 02" if _FROM_FIT else "defaults in config.py"
    return (f"tau={TAU:.4f} sigma0={SIGMA0} a5={A5} platt={PLATT:.3f}  [{src}]")
RHO = 0.90                 # cross-surface innovation correlation, script 06
TAU_SURFACE = 0.05

# Four blocks of equal length spanning the analysis window. The boundaries are
# mechanical and owe nothing to any player's career, which the earlier division
# at 1990/2003/2022 did not.
ERAS = [(1978, 1989), (1990, 2001), (2002, 2013), (2014, 2025)]
ERA_LABELS = [f"{a}--{b}" for a, b in ERAS]
MEAN_COUNTS = (1, 2)       # rate-calibrated thresholds

PLAYERS = {
    "D643": "Djokovic", "F324": "Federer", "N409": "Nadal",
    "B058": "Borg", "M047": "McEnroe", "L018": "Lendl",
    "C044": "Connors", "MC10": "Murray", "S402": "Sampras",
    "B028": "Becker", "S0AG": "Sinner", "A0E2": "Alcaraz",
    "A092": "Agassi", "W023": "Wilander", "E004": "Edberg",
    "H432": "Hewitt", "R485": "Roddick", "D683": "del Potro",
}
