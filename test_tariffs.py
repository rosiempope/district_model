"""
test_demand_synthesis.py
======================
Self-test / demonstration suite for profiles.demand_synthesis
(synthesise_network(), _cooling_profile() — including the real
internal-gains-floor fix — and the full building-mix worked example).
Moved out of demand_synthesis.py itself as part of a project-wide split
separating logic files from their self-tests.

Run directly: python3 tests/test_demand_synthesis.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from profiles.demand_synthesis import _cooling_profile, synthesise_network


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  demand_synthesis.py — self-test (synthetic weather)")
    print("="*65)
 
    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    dates      = pd.date_range("2023-01-01", periods=8760, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    scenario = {
        "demand_nodes": [
            {"name": "Perceval House",       "type": "office",              "floor_area_m2": 8500},
            {"name": "High Street Retail",   "type": "retail",              "floor_area_m2": 3000},
            {"name": "Ealing Hospital Wing", "type": "hospital",            "floor_area_m2": 12000},
            {"name": "Dickens Yard Ph1",     "type": "residential",         "units": 350},
            {"name": "Broadway Hotel",       "type": "hotel",               "floor_area_m2": 5000},
            {"name": "Ellen Wilkinson Sch",  "type": "school",              "floor_area_m2": 6000},
        ]
    }

    network = synthesise_network(weather_df, scenario)
 
    print("\n  Per-building summary:")
    print(network["summary_df"].to_string(index=False))
 
    hh = network["total_heat_kW"]
    cc = network["total_cooling_kW"]
    jan_heat = hh[:744].mean()
    jul_heat = hh[4344:5088].mean()
    jul_cool = cc[4344:5088].mean()
    jan_cool = cc[:744].mean()

    print(f"\n  Network totals:")
    print(f"    Annual space heat : {network['annual_heat_MWh']:>8.0f} MWh")
    print(f"    Annual DHW        : {network['annual_dhw_MWh']:>8.0f} MWh")
    print(f"    Annual cooling    : {network['annual_cool_MWh']:>8.0f} MWh")
    print(f"    Peak heat demand  : {network['peak_heat_kW']:>8.1f} kW")
    print(f"    Peak cooling      : {network['peak_cool_kW']:>8.1f} kW")
    print(f"    Cool:Heat ratio   : {network['annual_cool_MWh']/(network['annual_heat_MWh']+network['annual_dhw_MWh']):.2f}  (expect ~0.05-0.15 for UK)")

    print(f"\n  Seasonal sanity:")
    print(f"    Jan heat: {jan_heat:.0f} kW  |  Jul heat: {jul_heat:.0f} kW  → {'✓ winter peak' if jan_heat > jul_heat else '✗ FAIL'}")
    print(f"    Jan cool: {jan_cool:.1f} kW  |  Jul cool: {jul_cool:.1f} kW  → {'✓ summer peak' if jul_cool > jan_cool else '✗ FAIL'}")
    # NOTE: this used to read "(expect majority)" zero-cooling hours --
    # that expectation matched the OLD (buggy) model, where cooling only
    # ever activated on genuinely hot days. The corrected model
    # (_cooling_profile(), see its docstring) includes a real,
    # literature-grounded internal-gains floor present during ALL
    # occupied hours, independent of outdoor temperature -- so cooling
    # demand during occupied hours is now correctly near-continuous, not
    # mostly zero. Zero-cooling hours should now correspond to
    # UNOCCUPIED hours instead (when occ_modifier approaches
    # base_load_frac, not exactly zero unless base_load_frac=0) -- a
    # genuinely different, more realistic expectation than before.
    print(f"    Hours with cooling < 1% of peak: {(cc < 0.01 * cc.max()).sum()} / 8760  "
          f"(expect roughly the UNOCCUPIED hour count, not 'majority of the year' --")
    print(f"      see the note above this line for why that old expectation no longer applies)")
    print()