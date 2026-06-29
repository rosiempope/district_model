"""
test_climate_scenarios.py
======================
Self-test / demonstration suite for profiles.climate_scenarios
(apply_climate_scenario(), the seasonal UHI offset model, and the
SCENARIOS table). Moved out of climate_scenarios.py itself as part of
a project-wide split separating logic files from their self-tests.

Run directly: python3 tests/test_climate_scenarios.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from profiles.climate_scenarios import (
    N_HOURS, DELTAS, SCENARIOS, _seasonal_uhi_offset, apply_climate_scenario,
)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  climate_scenarios.py — self-test")
    print("=" * 70)

    # Synthetic London-like weather (same approach as other modules' self-tests)
    np.random.seed(42)
    hours = np.arange(N_HOURS)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / N_HOURS)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, N_HOURS)
    )
    dates = pd.date_range("2023-01-01", periods=N_HOURS, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    heat_base_C = 15.5
    baseline_HDD = np.clip(heat_base_C - T, 0, None).sum()

    print(f"\n  Baseline annual HDD_h: {baseline_HDD:,.0f}")
    print(f"\n  {'Scenario':<15} {'Jan mean C':>11} {'Jul mean C':>11} "
          f"{'Annual HDD_h':>13} {'HDD reduction':>14}")
    print("  " + "-" * 70)

    results = {}
    for s in DELTAS:
        w = apply_climate_scenario(weather_df, s)
        T_s = w["temp_drybulb_C"].values
        jan_mean = T_s[:744].mean()
        jul_mean = T_s[4344:5088].mean()
        HDD_s = np.clip(heat_base_C - T_s, 0, None).sum()
        reduction_pct = (1 - HDD_s / baseline_HDD) * 100
        results[s] = HDD_s
        print(f"  {s:<15} {jan_mean:>11.2f} {jul_mean:>11.2f} "
              f"{HDD_s:>13,.0f} {reduction_pct:>13.1f}%")

    print("\n  Literature cross-check (Staffell et al. 2019, European HDD trends):")
    print("    RCP4.5 by 2100: ~24% reduction   |   RCP8.5 by 2100: ~42% reduction")
    print("    A 2050 (mid-century) estimate should sit BELOW its own pathway's")
    print("    2100 figure — check this holds for both scenarios above.")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert results["baseline"] == baseline_HDD, "Baseline should leave HDD unchanged"
    assert results["2050_central"] < baseline_HDD, "2050_central should reduce HDD"
    assert results["2050_high"] < results["2050_central"], \
        "2050_high should reduce HDD more than 2050_central"

    central_reduction = (1 - results["2050_central"] / baseline_HDD) * 100
    high_reduction     = (1 - results["2050_high"] / baseline_HDD) * 100
    assert central_reduction < 24.0, \
        f"2050_central ({central_reduction:.1f}%) should sit below the RCP4.5-by-2100 figure (~24%)"
    assert high_reduction < 42.0, \
        f"2050_high ({high_reduction:.1f}%) should sit below the RCP8.5-by-2100 figure (~42%)"

    # UHI should taper to ~0 in deep winter and peak in summer for 2050_high
    uhi_high = _seasonal_uhi_offset(DELTAS["2050_high"], 2.5)
    assert uhi_high[1] == 0.0 and uhi_high[12] == 0.0, "UHI should be zero in midwinter"
    assert uhi_high[7] == 2.5, "UHI should peak in midsummer"

    print("  ✓ Baseline scenario leaves weather unchanged")
    print("  ✓ Both scenarios reduce annual HDD, with 2050_high reducing more")
    print("  ✓ Both scenarios sit BELOW their pathway's full-century literature figure")
    print("  ✓ UHI offset correctly tapers to zero in winter, peaks in summer")
    print()