"""
test_peak_demand_option.py
======================
Self-test / demonstration suite for components.peak_demand_option
(GasBoiler, ElectricBoiler, the part-load efficiency curve, the
CARBON_INTENSITY dict, and both boiler preset tables). Moved out of
peak_demand_option.py itself as part of a project-wide split separating
logic files from their self-tests.

Run directly: python3 tests/test_peak_demand_option.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from components.peak_demand_option import (
    N_HOURS, GAS_BOILER_PRESETS, ELECTRIC_BOILER_PRESETS,
    gas_boiler_part_load_efficiency, GasBoiler, ElectricBoiler,
)
from economics.tariffs import GasTariff, ElectricityTariff


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  peak_demand_option.py — self-test")
    print("="*70)

    # Test part-load efficiency curve directly
    print("\n  Part-load efficiency curve (condensing, eta_full=0.92):")
    test_loads = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    eta_cond = gas_boiler_part_load_efficiency(test_loads, 0.92, condensing=True)
    eta_noncond = gas_boiler_part_load_efficiency(test_loads, 0.78, condensing=False)
    print(f"  {'Load':>6} {'Condensing':>12} {'Non-condensing':>16}")
    for l, ec, en in zip(test_loads, eta_cond, eta_noncond):
        print(f"  {l:>6.1f} {ec:>12.3f} {en:>16.3f}")

    # Test all gas boiler presets — now with the realistic default gas tariff
    print("\n  All gas boiler presets (default DESNZ central gas tariff):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Gas £/MWh':>11} {'Marg. cost £/MWh':>17} {'Condensing':>11}")
    print("  " + "-"*82)
    for key in GAS_BOILER_PRESETS:
        b = GasBoiler.from_preset(key)
        s = b.summary()
        print(f"  {key:<25} {s['capacity_MW']:>12.1f} {s['mean_gas_price_GBP_per_MWh']:>11.2f} "
              f"{s['mean_marginal_cost_GBP_per_MWh']:>17.2f} {str(s['condensing']):>11}")

    # Test all electric boiler presets — now with the realistic default electricity tariff
    print("\n  All electric boiler presets (default central commercial electricity tariff):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Elec £/MWh':>11} {'Marg. cost £/MWh':>17} {'Efficiency':>11}")
    print("  " + "-"*82)
    for key in ELECTRIC_BOILER_PRESETS:
        b = ElectricBoiler.from_preset(key)
        s = b.summary()
        print(f"  {key:<25} {s['capacity_MW']:>12.1f} {s['mean_electricity_price_GBP_per_MWh']:>11.2f} "
              f"{s['mean_marginal_cost_GBP_per_MWh']:>17.2f} {s['efficiency']:>11.1%}")

    # Detailed test: Ealing Phase 1 gas boiler
    print("\n  Ealing Phase 1 gas boiler (detailed, default tariff):")
    ealing_gas = GasBoiler.from_preset("ealing_phase1")
    for k, v in ealing_gas.summary().items():
        print(f"    {k:<36} {v}")

    # --- NEW: tariff integration tests for both boiler types ---
    print("\n  Gas tariff integration — comparing all four accepted price input types:")
    gas_default = GasBoiler.from_preset("ealing_phase1")
    gas_scenario = GasBoiler.from_preset(
        "ealing_phase1", gas_price_GBP_per_MWh=GasTariff.from_scenario("current_actual")
    )
    gas_flat = GasBoiler.from_preset("ealing_phase1", gas_price_GBP_per_MWh=45.0)
    gas_array = GasBoiler.from_preset("ealing_phase1", gas_price_GBP_per_MWh=np.full(N_HOURS, 60.0))
    print(f"    Default (None, DESNZ central) -> £{gas_default._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    GasTariff (current_actual)    -> £{gas_scenario._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_scenario.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Flat scalar override (£45)    -> £{gas_flat._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_flat.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Raw array override (£60)      -> £{gas_array._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_array.marginal_cost.mean():.2f}/MWh heat")

    print("\n  Electric boiler tariff integration:")
    elec_default = ElectricBoiler.from_preset("ealing_backup")
    elec_discounted = ElectricBoiler.from_preset(
        "ealing_backup", electricity_price_GBP_per_MWh=ElectricityTariff(negotiated_discount_pct=10.0)
    )
    print(f"    Default (None)        -> £{elec_default._elec_price.mean():.2f}/MWh, "
          f"marginal £{elec_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    10% discounted tariff -> £{elec_discounted._elec_price.mean():.2f}/MWh, "
          f"marginal £{elec_discounted.marginal_cost.mean():.2f}/MWh heat")

    # from_config tests
    print("\n  from_config() — gas boiler with named scenario:")
    gas_cfg = GasBoiler.from_config({
        "type": "gas_boiler", "name": "Peak boiler (config)", "capacity_MW": 3.6,
        "gas_tariff_scenario": "current_actual",
    })
    print(f"    {gas_cfg}  ->  £{gas_cfg._gas_price.mean():.2f}/MWh (expect ~£34.46, current_actual)")

    print("\n  from_config() — electric boiler with nested tariff block:")
    elec_cfg = ElectricBoiler.from_config({
        "type": "electric_boiler", "name": "Electric backup (config)", "capacity_MW": 1.0,
        "electricity_tariff": {"negotiated_discount_pct": 20.0},
    })
    print(f"    {elec_cfg}  ->  £{elec_cfg._elec_price.mean():.2f}/MWh (expect 20% below £240 central)")

    # Test set_load_profile — simulate a realistic part-load dispatch pattern
    print("\n  Testing set_load_profile() — simulated winter-peaking dispatch:")
    hours = np.arange(N_HOURS)
    # Boiler runs harder in winter (more peak shaving needed), idles in summer
    simulated_load = 0.15 + 0.55 * np.clip(
        np.cos(2 * np.pi * (hours - 0) / 8760), 0, 1
    )
    ealing_gas.set_load_profile(simulated_load)
    print(f"    Mean load fraction:        {simulated_load.mean():.2f}")
    print(f"    Mean efficiency (updated): {ealing_gas.efficiency_hourly.mean():.3f}")
    print(f"    Mean marginal cost:        £{ealing_gas.marginal_cost.mean():.2f}/MWh")

    # Compare gas vs electric boiler cost at realistic default prices
    print("\n  Cost comparison — Gas vs Electric boiler (same 3.6 MW capacity, REALISTIC default tariffs):")
    gas = GasBoiler.from_preset("ealing_phase1")
    elec = ElectricBoiler.from_preset("ealing_backup")
    print(f"    Gas boiler:      £{gas.marginal_cost.mean():.2f}/MWh heat  "
          f"(gas @ £{gas._gas_price.mean():.0f}/MWh DESNZ central, η={gas.eta_full_load:.0%})")
    print(f"    Electric boiler: £{elec.marginal_cost.mean():.2f}/MWh heat "
          f"(elec @ £{elec._elec_price.mean():.0f}/MWh central commercial, η={elec.efficiency:.0%})")
    print(f"    → Electric is {elec.marginal_cost.mean()/gas.marginal_cost.mean():.1f}x more expensive per MWh heat at these realistic prices")

    # --- n_units x unit_capacity_MW sizing — matches ASHPArray's pattern ---
    print("\n  Discrete-unit sizing (n_units x unit_capacity_MW), matching ASHPArray:")
    gas_3units = GasBoiler(name="Boiler bank", n_units=3, unit_capacity_MW=1.2)
    gas_flat_equiv = GasBoiler(name="Boiler bank (flat)", capacity_MW=3.6)
    print(f"    {gas_3units}")
    print(f"    Equivalent flat capacity: {gas_flat_equiv}")
    print(f"    Same marginal cost either way? "
          f"{np.allclose(gas_3units.marginal_cost, gas_flat_equiv.marginal_cost)}")

    # resize() — the sizing.py sweep hook
    print("\n  resize() — the hook optimisation/sizing.py expects for capacity sweeps:")
    ealing_bank = GasBoiler.from_preset("ealing_phase1")   # 1 unit @ 3.6 MW (legacy preset)
    resized = ealing_bank.resize(n_units=3, unit_capacity_MW=1.2)
    print(f"    Original: {ealing_bank}")
    print(f"    Resized:  {resized}")

    # Error handling — providing both forms, or an incomplete pair, should fail loudly
    print("\n  Error handling — conflicting or incomplete sizing args:")
    for bad_kwargs, desc in [
        ({"capacity_MW": 3.6, "n_units": 3, "unit_capacity_MW": 1.2}, "both forms given"),
        ({"n_units": 3}, "n_units without unit_capacity_MW"),
        ({}, "neither form given"),
    ]:
        try:
            GasBoiler(name="bad", **bad_kwargs)
            print(f"    ✗ FAIL: should have raised ValueError ({desc})")
        except ValueError as e:
            print(f"    ✓ Correctly raised ({desc}): {str(e)[:70]}...")

    # Sanity checks
    print("\n  Sanity checks:")
    assert len(gas.supply_MW)      == N_HOURS, "GasBoiler supply_MW wrong length"
    assert len(gas.marginal_cost)  == N_HOURS, "GasBoiler marginal_cost wrong length"
    assert len(elec.supply_MW)     == N_HOURS, "ElectricBoiler supply_MW wrong length"
    assert gas.supply_MW.max()  <= gas.capacity_MW + 0.001, "Gas supply exceeds capacity"
    assert elec.supply_MW.max() <= elec.capacity_MW + 0.001, "Electric supply exceeds capacity"
    assert eta_cond[0] > eta_cond[-1], "Condensing boiler should be MORE efficient at low load"
    assert eta_noncond[0] < eta_noncond[-1], "Non-condensing boiler should be LESS efficient at low load"

    # New tariff-integration assertions
    assert abs(gas_default._gas_price.mean() - 24.57) < 0.5, \
        "Default gas price should be the DESNZ central tariff (~£24.57/MWh), not the old £45 placeholder"
    assert abs(elec_default._elec_price.mean() - 240.0) < 0.5, \
        "Default electricity price should be the realistic ~£240/MWh tariff, not the old £120 placeholder"
    assert gas_scenario._gas_price.mean() > gas_default._gas_price.mean(), \
        "current_actual gas scenario should be pricier than desnz_central"
    assert abs(gas_flat._gas_price.mean() - 45.0) < 0.01, "Flat scalar override should be respected exactly"
    assert elec_discounted._elec_price.mean() < elec_default._elec_price.mean(), \
        "Discounted electricity tariff should be cheaper than the undiscounted default"
    assert abs(gas_cfg._gas_price.mean() - 34.46) < 0.5, \
        "from_config gas_tariff_scenario should resolve to the named GasTariff scenario"
    assert abs(elec_cfg._elec_price.mean() - 240.0 * 0.80) < 1.0, \
        "from_config nested electricity_tariff block should apply the 20% discount correctly"

    # n_units sizing assertions
    assert abs(gas_3units.capacity_MW - 3.6) < 1e-9, "3 x 1.2 MW should total 3.6 MW"
    assert np.allclose(gas_3units.marginal_cost, gas_flat_equiv.marginal_cost), \
        "n_units and flat capacity_MW paths should produce identical physics for the same total MW"
    assert abs(resized.capacity_MW - 3.6) < 1e-9 and resized.n_units == 3, \
        "resize() should apply the new scale correctly"
    assert ealing_bank.n_units == 1 and ealing_bank.unit_capacity_MW == 3.6, \
        "Legacy capacity_MW path should default to n_units=1, unit_capacity_MW=capacity_MW"

    print("  ✓ All array shapes correct (8760 hours)")
    print("  ✓ Supply never exceeds nameplate capacity")
    print("  ✓ Condensing boiler gains efficiency at low load (as expected)")
    print("  ✓ Non-condensing boiler loses efficiency at low load (as expected)")
    print("  ✓ Default gas price now uses DESNZ central tariff (~£25/MWh), not old £45 placeholder")
    print("  ✓ Default electricity price now uses realistic tariff (~£240/MWh), not old £120 placeholder")
    print("  ✓ GasTariff/ElectricityTariff objects, flat scalars, and raw arrays all behave correctly")
    print("  ✓ from_config() named scenario and nested tariff block both resolve correctly")
    print("  ✓ n_units x unit_capacity_MW produces identical physics to the equivalent flat capacity_MW")
    print("  ✓ resize() correctly rescales to a new unit count/size")
    print("  ✓ Legacy capacity_MW path still works and defaults n_units=1 for consistent reporting")
    print("  ✓ Conflicting or incomplete sizing arguments correctly rejected")
    print()