"""
test_thermal_storage.py
======================
Self-test / demonstration suite for components.thermal_storage
(ThermalStorage, the CAPEX estimator, and the volume<->energy
converters). Moved out of thermal_storage.py itself as part of a
project-wide split separating logic files from their self-tests.

Run directly: python3 tests/test_thermal_storage.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from components.thermal_storage import (
    N_HOURS, estimate_storage_capex, mwh_to_m3, m3_to_mwh, ThermalStorage,
)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  thermal_storage.py — self-test")
    print("="*70)
 
    # --- Test unit conversions ---
    print("\n  Unit conversion sanity check:")
    test_vol = 1000  # m3
    test_energy = m3_to_mwh(test_vol, delta_T_K=40)
    print(f"    {test_vol} m³ at 40K delta-T = {test_energy:.2f} MWh")
    back_to_vol = mwh_to_m3(test_energy, delta_T_K=40)
    print(f"    Round-trip back to volume: {back_to_vol:.1f} m³ (should match {test_vol})")
    assert abs(back_to_vol - test_vol) < 0.1, "Unit conversion round-trip failed"
 
    # --- Test operational buffer sizing ---
    print("\n  Operational buffer (ASHP, rule-of-thumb sizing):")
    buffer = ThermalStorage.from_buffer_rule(
        name="Ealing Phase 1 ASHP buffer",
        connected_capacity_MW=2.8,
        litres_per_kW=40,
    )
    for k, v in buffer.summary().items():
        print(f"    {k:<28} {v}")
 
    # --- Test cost curve scaling ---
    print("\n  Cost curve scaling (cost per MWh should FALL as capacity grows):")
    for cap in [0.5, 5, 50, 500]:
        capex = estimate_storage_capex(cap)
        print(f"    Capacity={cap:>6.1f} MWh  →  Total capex=£{capex:>12,.0f}  (£{capex/cap:>8,.0f}/MWh)")
 
    # --- Test strategic storage charge/discharge dynamics ---
    print("\n  Strategic storage — simulated 48-hour surplus/deficit cycle:")
    store = ThermalStorage(
        name="Strategic diurnal store",
        capacity_MWh=20,
        max_charge_MW=5,
        max_discharge_MW=5,
    )
 
    # Surplus (EfW/DC baseload > demand) for 8h, deficit for 8h, mild surplus for 8h
    test_pattern = np.tile(
        np.concatenate([np.full(8, 3.0), np.full(8, -4.0), np.full(8, 1.0)]), 2
    )
    result = store.run_series(test_pattern)
 
    print(f"    Hours simulated:        {len(test_pattern)}")
    print(f"    Min SoC:                {result['soc_MWh'].min():.2f} MWh (should be >= 0)")
    print(f"    Max SoC:                {result['soc_MWh'].max():.2f} MWh (should be <= {store.capacity_MWh})")
    print(f"    Total curtailed:        {result['unmet_surplus_MW'].sum():.2f} MWh (surplus that couldn't be stored)")
    print(f"    Total unmet demand:     {result['shortfall_MW'].sum():.2f} MWh (shortfall the store couldn't cover)")
 
    # --- Test reset ---
    print("\n  Reset test:")
    store.reset(initial_soc_fraction=0.5)
    print(f"    SoC after reset: {store.soc_MWh:.2f} MWh (should be {store.capacity_MWh * 0.5:.2f})")
 
    # --- Test full year run (8760 hours) for shape validation ---
    print("\n  Full year test — synthetic surplus pattern (winter deficit, summer surplus):")
    np.random.seed(0)
    hours = np.arange(N_HOURS)
    # EfW/DC baseload roughly constant; demand has winter peak — net surplus
    # is positive in summer (baseload > low summer demand), negative in
    # winter (baseload < high winter demand)
    seasonal_net = 2.0 * np.cos(2 * np.pi * (hours - 4200) / 8760) + np.random.normal(0, 0.5, N_HOURS)
 
    annual_store = ThermalStorage(
        name="Annual test store",
        capacity_MWh=15,
        max_charge_MW=3,
        max_discharge_MW=3,
    )
    annual_result = annual_store.run_series(seasonal_net)
 
    print(f"    Annual curtailed surplus: {annual_result['unmet_surplus_MW'].sum():>10,.0f} MWh")
    print(f"    Annual unmet demand:      {annual_result['shortfall_MW'].sum():>10,.0f} MWh")
    print(f"    Mean SoC fraction:        {annual_store.summary()['mean_soc_fraction']:.2f}")
 
    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert min(annual_store.soc_history) >= 0, "SoC went negative"
    assert max(annual_store.soc_history) <= annual_store.capacity_MWh + 0.001, "SoC exceeded capacity"
    assert estimate_storage_capex(500) / 500 < estimate_storage_capex(0.5) / 0.5, \
        "Cost per MWh should fall with scale"
    print("  ✓ SoC always within [0, capacity] bounds")
    print("  ✓ Cost per MWh falls with scale (economies of scale confirmed)")
    print("  ✓ Unit conversions (MWh <-> m³) round-trip correctly")
    print()