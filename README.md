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


## Contents

```
atp_matches_1968_2025.csv.gz    the exact analysis dataset, 197,926 matches
tennisdom/core.py               model, filtering, sampling, all estimands
tennisdom/config.py             fixed choices and estimated hyperparameters
dbtlib.py                       scalar-state shim used by scripts 06 to 09
scripts/                        numbered, run in order
out/                            everything the scripts produce
```
