# Reproduction pipeline

Code and data for "How exceptional was the Big Three era? Extremes and
persistence on the ATP tour".

**No Python on your machine?** Open `Colab_rerun.ipynb` in Google Colab
(colab.research.google.com, then File, Upload notebook), drag the tar into the
Files pane, and click through the cells. Nothing to install.

Locally:

```
pip install -r requirements.txt
./run_all.sh                    # about 2 hours on one core
```

## Checking the paper

`scripts/13_check.py` compares a fresh run against every number currently in
the manuscript, stored in `tennisdom/paper_values.json`, and prints OK, ~ or
DIFF for each. Run it after `04_estimands.py`. A verification run here gave
**101 checks, 0 flagged**, so the tables in the manuscript are sound; only the
two known discrepancies listed below need attention.

## Contents

```
atp_matches_1968_2025.csv.gz    the exact analysis dataset, 197,926 matches
tennisdom/core.py               model, filtering, sampling, all estimands
tennisdom/config.py             fixed choices and estimated hyperparameters
dbtlib.py                       scalar-state shim used by scripts 06 to 09
scripts/                        numbered, run in order
out/                            everything the scripts produce
```

## Data provenance

**Read this before submitting.** The dataset ships with this pipeline because
it can no longer be obtained from the source normally cited for it. Jeff
Sackmann's `tennis_atp` repository, the standard archive for ATP match records,
was public when this project began and is no longer reachable; his GitHub
account currently exposes only the Match Charting Project. The file here was
assembled from the Tennismylife `TML-Database` mirror, which follows the same
column schema and claims to have filled gaps in the original (for instance
Connors' full 1,274 career wins).

Two consequences. First, you cannot cite Sackmann's repository as the source,
because a referee who checks the link will find nothing. Second, the mirror's
corrections mean the file is not byte-identical to what Sackmann distributed,
so results will not match older published analyses exactly. Decide how to
describe this in the paper before submission; at present Section 6.1 says only
"the standard public archive of tour results", which is vague enough to be
worth tightening.

## Where every number in the paper comes from

| Paper object | Script | Notes |
|---|---|---|
| Section 6.1 corpus counts | `01_data.py` | |
| Table 5, hyperparameters, reliability, Platt | `02_hyperparameters.py` | also the kappa ridge check |
| Posterior draws used by everything below | `03_posterior.py` | caches `03_posterior.pkl` |
| Table 4 top-five profile | `04_estimands.py` and `08_sensitivity.py` | occupancy percentages from `08` |
| Table 6 peaks, Table 7 pairwise | `04_estimands.py` | |
| Table 8 concurrence | `04_estimands.py` | |
| Table 9 persistence profiles | `04_estimands.py` | |
| Table 10 sojourn | `04_estimands.py` | |
| Table 11 generalized Pareto, extremal index | `04_estimands.py` | |
| Tables 12 and 13 surfaces | `05_surfaces.py` | |
| Tables 1 and 2 simulation | `06_simulation.py` | writes `sim3.csv` |
| Table 3 Polya-Gamma validation | `07_validation.py` | |
| Table 14 reference band sensitivity | `08_sensitivity.py` | |
| Figures 1 to 6 | `10_figures_main.py` | |
| Figure 7 persistence profiles | `11_figure_persistence.py` | |
| Figure 8 trajectories | `09_trajectories.py` then `12_figure_trajectories.py` | |

## Eras

`config.py` defines the era division in one place and every script reads it
from there:

```python
ERAS = [(1978, 1989), (1990, 2001), (2002, 2013), (2014, 2025)]
```

Four blocks of equal length, with boundaries that owe nothing to any player's
career. Changing this line and rerunning `04`, `08`, `09` and the figure
scripts regenerates every era-dependent table and figure. `16_eras.py` compares
any two divisions side by side.

## Hyperparameters

`02_hyperparameters.py` estimates tau, sigma0, a5 and the recalibration slope,
and writes them to `out/02_hyper.json`. `tennisdom/config.py` reads that file if
it exists and falls back to hard-coded defaults otherwise, so the config and the
fitted model cannot disagree. Every script that consumes them prints which set
it used, for example

```
hyperparameters: tau=0.0473 sigma0=1.0 a5=1.15 platt=0.851  [fitted by script 02]
```

**`02` must therefore run before `03`.** If it has not, later scripts use the
defaults and say so. Delete `out/02_hyper.json` to revert to the defaults.

The recalibration slope multiplies every strength reported in rating points. It
does not affect probabilities, counts, durations in weeks, or any
rate-calibrated threshold, because those are invariant to a global rescaling.

## Randomness and reproducibility

Every source of randomness is now derived from a single `SEED` in
`tennisdom/config.py`:

| Script | Seed | Purpose |
|---|---|---|
| `03_posterior.py` | `SEED` | the N_DRAWS trajectory draws used by 04, 08, 09 |
| `05_surfaces.py` | `SEED` | multivariate backward sampling |
| `06_simulation.py` | `SEED + rep` | one per replicate |
| `07_validation.py` | `SEED + 1`, `SEED + 2` | Gibbs chain, comparison draws |
| `10_figures_main.py` | `SEED` | jitter in the Figure 3 strip plots |

Scripts `04`, `08` and `09` all load `out/03_posterior.pkl` rather than refitting,
so every table and figure in the paper is computed from the *same* 300
trajectories. This matters: in an earlier version they each drew independently,
which made numbers differ between tables by one to two per cent of Monte Carlo
noise for no reason. If you change `SEED`, rerun `03` first and then everything
downstream.

There is no remaining unseeded randomness. Running the pipeline twice on the
same machine gives bit-identical output.

## Things to check rather than trust

These are the places where an error would be easiest to make and hardest to
notice. They are listed so you can verify rather than assume.

1. **The manuscript tables were originally transcribed by hand from console
   output.** `14_tables.py` now writes them, so this should not recur; run it
   and paste, rather than copying numbers off the screen.
   Every table should be checked against a fresh run. Re-running `04_estimands.py`
   here reproduced the manuscript to within Monte Carlo error but not exactly:
   for example Federer's sojourn came out at 615 against 608 in the manuscript,
   and mean P(N_t>=3) for 1978-1989 at 0.639 against 0.654. These differences
   are seed noise of order one to two per cent, not disagreement, but the tables
   should be regenerated from a single run so the paper is internally
   consistent. The manuscript carries a red marker on every table and figure
   naming the script that produces it; set `\checksfalse` in the preamble to
   hide them.

2. **Figure 5 has hard-coded numbers.** The chi and eta values and the
   membership probabilities in `10_figures_main.py` are literals typed from the
   output of `05_surfaces.py`, not read from its pickle. If you change anything
   upstream, that figure will silently go stale. Worth rewiring to read
   `out/05_surfaces.pkl` before submission.

3. **No figure hard-codes its numbers any more.** Figure 4 reads
   `out/02_hyper.json` and Figure 5 reads `out/05_surfaces.pkl`, so neither can
   go stale when something upstream changes. Figure 5 is skipped with a message
   if `05_surfaces.py` has not been run.

4. **`del Potro` and `Wilander` appear in Table 13** with membership
   probabilities but were added to the player list late; confirm their rows
   against a fresh `05_surfaces.py` run.

5. **The Polya-Gamma sampler uses a truncated series** for the PG(1, z) draw,
   at 120 terms, rather than Devroye's exact method. This is accurate enough
   for a validation study but is an approximation, and the validation is
   therefore of one approximation against a very good one rather than against
   the exact posterior.

6. **The simulation in `06_simulation.py` calibrates the threshold to a target
   exceedance rate using the true trajectories.** This is deliberate, since the
   point is to compare estimators at a common marginal rate, but it means the
   comparison is more favourable to every estimator than a fully blind analysis
   would be.

7. **Hyperparameters are hard-coded in `config.py`** so that scripts 03 onward
   do not refit. If `02_hyperparameters.py` returns different values on your
   machine, update `config.py` before running anything else.

## Runtimes

One core, no parallelism. `02` is the slowest at roughly 35 minutes because it
evaluates 27 grid points plus a curvature grid, each a full forward pass over
197,926 matches. `05` and `06` are each about 25 to 30 minutes. Everything else
is under 15 minutes.
