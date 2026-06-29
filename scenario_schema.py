"""
test_sizing.py
======================
Self-test / demonstration suite for optimisation.sizing
(find_required_capacity(), capacity_sweep()). Moved out of sizing.py
itself as part of a project-wide split separating logic files from
their self-tests.

Run directly: python3 tests/test_sizing.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from optimisation.sizing import capacity_sweep, find_required_capacity


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