"""
test_datacentre_source.py
======================
Self-test / demonstration suite for components.datacentre_source
(DataCentre, the availability profile, and the DC_PRESETS table). Moved
out of datacentre_source.py itself as part of a project-wide split
separating logic files from their self-tests.

Run directly: python3 tests/test_datacentre_source.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from components.datacentre_source import DC_PRESETS, DataCentre


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  source.py — self-test")
    print("="*65)

    # Test all presets
    print("\n  All DC presets:")
    print(f"  {'Preset':<35} {'Capacity MW':>12} {'Annual MWh':>12} {'T supply °C':>12}")
    print("  " + "-"*72)
    for key in DC_PRESETS:
        dc = DataCentre.from_preset(key)
        s  = dc.summary()
        print(f"  {key:<35} {s['capacity_MW']:>12.1f} {s['annual_heat_available_MWh']:>12.0f} {s['supply_temp_nominal_C']:>11.1f}°C")

    # Detailed test for Redwire (primary Ealing scenario)
    print("\n  Redwire DC (Ealing town centre):")
    redwire = DataCentre.from_preset("redwire_ealing")
    for k, v in redwire.summary().items():
        print(f"    {k:<36} {v}")

    # Test custom DC
    print("\n  Custom DC (user-defined 17 MW at 28°C, OPDC-style):")
    custom = DataCentre(
        name="OPDC data centre cluster",
        it_load_MW=68.0,
        heat_offtake_fraction=0.25,
        supply_temp_C=28.0,
        availability_factor=0.95,
    )
    print(f"    {custom}")
    print(f"    Annual heat available: {custom.supply_MW.sum():,.0f} MWh")


    # Sanity checks
    print("\n  Sanity checks:")
    dc = DataCentre.from_preset("redwire_ealing")
    assert len(dc.supply_MW)     == 8760, "supply_MW wrong length"
    assert len(dc.supply_temp_C) == 8760, "supply_temp_C wrong length"
    assert len(dc.marginal_cost) == 8760, "marginal_cost wrong length"
    assert dc.supply_MW.max() <= dc.capacity_MW + 0.001, "supply exceeds capacity"
    assert dc.supply_MW.min() >= 0, "negative supply"
    assert abs(dc._avail.mean() - dc.availability_factor) < 0.01, "availability mismatch"
    print("  ✓ All array shapes correct")
    print("  ✓ Supply never exceeds capacity")
    print("  ✓ Availability factor within tolerance")
    print()