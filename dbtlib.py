"""Compatibility shim: the scalar-state signature used by the simulation and
validation scripts, which have no surface or format structure. Delegates to
tennisdom.core so there is exactly one implementation of the filter."""
import numpy as np
from tennisdom.core import (dbt_filter as _f, flatten, ffbs, panel_index,
                            longest_run, Q3)

def dbt_filter(wi, li, wt, NP, NW, tau, sig0=2.0, nit=8, collect=True):
    bo = np.full(len(wi), 3)
    return _f(wi, li, wt, bo, NP, NW, tau=tau, sigma0=sig0, a5=1.0,
              n_newton=nit, collect=collect)
