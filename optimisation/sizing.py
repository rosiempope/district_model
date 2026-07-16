"""
sizing.py
==========
Given a demand profile and a way to build ONE resizable source at
different scales, find out how much capacity that source needs to cover
the demand — answers "how many ASHP units / how big a boiler would I
need", on its own or alongside a fixed set of other sources.

Deliberately source-agnostic: pass in a builder function rather than
hardcoding ASHPArray, so this works for any resizable source —
ASHPArray.resize(), or a custom builder you write for DataCentre/EfWChp/
boilers at different scales. This is what makes it modular in the sense
you actually want: "swap in whatever sources are available for this
particular zone/estate and see what's needed" doesn't require touching
this file at all, just calling it with different arguments.

Usage
-----
    from optimisation.sizing import capacity_sweep, find_required_capacity
    from components.ASHP import ASHPArray

    # "How many ASHP units do I need, alone, to cover this demand?"
    result = find_required_capacity(
        demand_kW=my_demand_kW,
        build_source=lambda n: ASHPArray.from_preset(
            "ealing_phase1", weather_df
        ).resize(n_units=n),
        candidate_values=range(1, 16),
        unmet_tolerance_pct=1.0,
    )
    print(result["required_value"], "units ->", result["required_capacity_MW"], "MW")
    print(result["sweep_df"])   # full sweep, e.g. for a plot

    # Same question, but WITH a backup boiler already available — shows
    # how much smaller the ASHP fleet can be when backup exists:
    result_with_backup = find_required_capacity(
        demand_kW=my_demand_kW,
        build_source=lambda n: ASHPArray.from_preset(
            "ealing_phase1", weather_df
        ).resize(n_units=n),
        candidate_values=range(1, 16),
        other_sources=[my_gas_boiler],
    )
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Callable, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimisation.dispatch import run_dispatch


def capacity_sweep(
    demand_kW: np.ndarray,
    build_source: Callable[[float], object],
    candidate_values: Sequence[float],
    other_sources: Optional[list] = None,
    storage=None,
) -> pd.DataFrame:
    """
    Run dispatch once per candidate scale value, varying ONE source's
    size while holding demand, other_sources, and storage fixed. Returns
    one row per candidate so you can inspect/plot how cost and unmet
    demand change with scale.

    Parameters
    ----------
    demand_kW        : 8760-length demand profile (kW)
    build_source      : callable(value) -> a FRESH source object at that
                        scale each time it's called (e.g.
                        `lambda n: ASHPArray.from_preset("ealing_phase1",
                        weather_df).resize(n_units=n)`)
    candidate_values  : the scale values to sweep (n_units, capacity_MW,
                        whatever build_source understands)
    other_sources     : any OTHER sources also available (e.g. a backup
                        boiler), held fixed across the sweep. None/[]
                        tests the swept source completely alone.
    storage           : optional ThermalStorage, held fixed across the
                        sweep (reset to the same starting state for
                        every run automatically by run_dispatch()).

    Returns
    -------
    DataFrame: value, capacity_MW, pct_demand_unmet, peak_unmet_MW,
    annual_cost_GBP — sorted by capacity_MW ascending.
    """
    other_sources = other_sources or []
    rows = []
    for value in candidate_values:
        source = build_source(value)
        result = run_dispatch(demand_kW, [source] + other_sources, storage=storage)
        s = result.summary()
        rows.append({
            "value":            value,
            "capacity_MW":      source.capacity_MW,
            "pct_demand_unmet": s["pct_demand_unmet"],
            "peak_unmet_MW":    s["peak_unmet_MW"],
            "annual_cost_GBP":  s["total_annual_opex_GBP"],
        })
    return pd.DataFrame(rows).sort_values("capacity_MW").reset_index(drop=True)


def find_required_capacity(
    demand_kW: np.ndarray,
    build_source: Callable[[float], object],
    candidate_values: Sequence[float],
    unmet_tolerance_pct: float = 1.0,
    other_sources: Optional[list] = None,
    storage=None,
) -> dict:
    """
    Convenience wrapper around capacity_sweep(): finds the SMALLEST
    candidate that brings % unmet demand at or below unmet_tolerance_pct.

    Returns
    -------
    dict with keys: required_value, required_capacity_MW (both None if
    NO candidate in the sweep met the tolerance — widen candidate_values),
    plus sweep_df (the full sweep, for inspection/plotting).
    """
    df = capacity_sweep(demand_kW, build_source, candidate_values, other_sources, storage)
    meeting = df[df["pct_demand_unmet"] <= unmet_tolerance_pct]
    if len(meeting) == 0:
        return {"required_value": None, "required_capacity_MW": None, "sweep_df": df}
    required = meeting.iloc[0]
    return {
        "required_value":       required["value"],
        "required_capacity_MW": required["capacity_MW"],
        "sweep_df":             df,
    }
