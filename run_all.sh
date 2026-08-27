#!/usr/bin/env bash
# Full pipeline. About two hours on one core.
# 02 must precede 03: it writes out/02_hyper.json, which config.py then reads.
set -e
cd "$(dirname "$0")/scripts"
python3 01_data.py                 | tee ../out/01_data.log
python3 02_hyperparameters.py      | tee ../out/02_hyper.log      # ~35 min
python3 03_posterior.py            | tee ../out/03_posterior.log  # ~6 min
python3 04_estimands.py            | tee ../out/04_estimands.log  # ~12 min
python3 05_surfaces.py             | tee ../out/05_surfaces.log   # ~30 min
python3 06_simulation.py           | tee ../out/06_simulation.log # ~25 min
python3 07_validation.py           | tee ../out/07_validation.log # ~20 min
python3 08_sensitivity.py          | tee ../out/08_sensitivity.log
python3 09_trajectories.py         | tee ../out/09_traj.log
python3 13_check.py                | tee ../out/13_check.log
python3 14_tables.py               | tee ../out/14_tables.log
python3 15_rolling.py              | tee ../out/15_rolling.log
python3 16_eras.py                 | tee ../out/16_eras.log
python3 17_bestwindows.py          | tee ../out/17_best.log
python3 10_figures_main.py
python3 11_figure_persistence.py
python3 12_figure_trajectories.py
echo "done; outputs in out/ and LaTeX table bodies in out/tables/"
