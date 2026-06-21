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


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  sizing.py — self-test")
    print("=" * 70)

    from components.ASHP import ASHPArray
    from components.peak_demand_option import GasBoiler
    from profiles.demand_synthesis import synthesise_network

    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    dates = pd.date_range("2023-01-01", periods=8760, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    scenario = {
        "demand_nodes": [
            {"name": "Perceval House",       "type": "office",      "floor_area_m2": 8500},
            {"name": "High Street Retail",   "type": "retail",      "floor_area_m2": 3000},
            {"name": "Ealing Hospital Wing", "type": "hospital",    "floor_area_m2": 12000},
            {"name": "Dickens Yard Ph1",     "type": "residential", "units": 350},
            {"name": "Broadway Hotel",       "type": "hotel",       "floor_area_m2": 5000},
            {"name": "Ellen Wilkinson Sch",  "type": "school",      "floor_area_m2": 6000},
        ]
    }
    net = synthesise_network(weather_df, scenario)
    demand_kW = net["total_heat_kW"]
    print(f"\n  Demand: peak {demand_kW.max()/1000:.2f} MW, annual {demand_kW.sum()/1000:,.0f} MWh")

    # --- "How many ASHP units do I need, ALONE, no backup?" ---
    print("\n  ASHP alone — sweeping n_units (Ealing Phase 1 unit size, 0.7 MW each):")
    build_ashp = lambda n: ASHPArray.from_preset("ealing_phase1", weather_df, n_units=n)
    result_alone = find_required_capacity(
        demand_kW, build_ashp, candidate_values=range(1, 16), unmet_tolerance_pct=1.0,
    )
    print(result_alone["sweep_df"].to_string(index=False))
    print(f"\n  -> {result_alone['required_value']} units "
          f"({result_alone['required_capacity_MW']:.1f} MW) needed for <=1% unmet, alone")

    # --- Same question, but WITH a backup boiler available ---
    print("\n  ASHP + backup gas boiler available — same sweep:")
    backup = GasBoiler.from_preset("ealing_phase1")
    result_with_backup = find_required_capacity(
        demand_kW, build_ashp, candidate_values=range(1, 16),
        unmet_tolerance_pct=1.0, other_sources=[backup],
    )
    print(result_with_backup["sweep_df"].to_string(index=False))
    print(f"\n  -> {result_with_backup['required_value']} units "
          f"({result_with_backup['required_capacity_MW']:.1f} MW) needed for <=1% unmet, with backup")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    df = result_alone["sweep_df"]
    assert (df["pct_demand_unmet"].values[:-1] >= df["pct_demand_unmet"].values[1:] - 1e-9).all(), \
        "% unmet demand should be non-increasing as capacity grows"
    assert result_alone["required_capacity_MW"] is not None, "Should find a sufficient capacity within the swept range"
    assert (
        result_with_backup["required_capacity_MW"] <= result_alone["required_capacity_MW"]
    ), "Required ASHP capacity should be <= when backup boiler is also available"
    print("  ✓ % unmet demand falls monotonically as swept capacity increases")
    print("  ✓ Found a sufficient capacity within the candidate range")
    print("  ✓ Required ASHP capacity is smaller (or equal) when backup boiler is available")
    print()