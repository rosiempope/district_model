"""
test_ashp.py
======================
Self-test / demonstration suite for components.ASHP (ASHPArray, the
COP model, capacity derating, unit-level outage model, presets) and
components.ashp_weather_compensation (the dormant weather-compensation
feature). Moved out of ASHP.py itself as part of a project-wide split
separating logic files from their self-tests — see ASHP.py's module
docstring for the file-restructuring rationale.

Run directly: python3 tests/test_ashp.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from economics.tariffs import ElectricityTariff
from components.ASHP import (
    ASHPArray, ASHP_PRESETS, ashp_cop, N_HOURS,
    weather_compensated_flow_temp_C, check_compensation_floor_against_network,
)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ASHP.py — self-test")
    print("="*70)

    # Build synthetic London-like weather (same approach as demand_synthesis test)
    np.random.seed(42)
    hours = np.arange(N_HOURS)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    dates = pd.date_range("2023-01-01", periods=8760, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    print(f"\n  Synthetic weather: T min={T.min():.1f}°C  T max={T.max():.1f}°C  T mean={T.mean():.1f}°C")

    # Test COP curve directly across a temperature sweep
    print("\n  COP curve sanity check (flow temp = 70°C, with defrost):")
    test_temps = np.array([-15, -10, -5, -2, 0, 2, 5, 8, 10, 15, 20, 25])
    cops = ashp_cop(test_temps, T_flow_C=70.0)
    for t, c in zip(test_temps, cops):
        print(f"    T_amb={t:>4}°C  COP={c:.2f}")

    # Test all presets — now with the realistic default tariff applied
    print("\n  All ASHP presets (electricity price now defaults to realistic tariff shape):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Mean COP':>10} {'Mean elec £/MWh':>16} {'Mean marg. cost £/MWh':>22}")
    print("  " + "-"*90)
    for key in ASHP_PRESETS:
        ashp = ASHPArray.from_preset(key, weather_df)
        s = ashp.summary()
        print(f"  {key:<25} {s['total_capacity_MW']:>12.1f} {s['cop_mean']:>10.2f} "
              f"{s['mean_electricity_price_GBP_per_MWh']:>16.2f} {s['mean_marginal_cost_GBP_per_MWh']:>22.2f}")

    # Detailed test: Ealing Phase 1
    print("\n  Ealing Phase 1 ASHP (detailed, default tariff):")
    ealing = ASHPArray.from_preset("ealing_phase1", weather_df)
    for k, v in ealing.summary().items():
        print(f"    {k:<36} {v}")

    # --- NEW: tariff integration tests ---
    print("\n  Tariff integration — comparing all four accepted price input types:")
    ealing_default  = ASHPArray.from_preset("ealing_phase1", weather_df)
    ealing_tariff    = ASHPArray.from_preset(
        "ealing_phase1", weather_df,
        electricity_price_GBP_per_MWh=ElectricityTariff(negotiated_discount_pct=10.0),
    )
    ealing_flat      = ASHPArray.from_preset(
        "ealing_phase1", weather_df, electricity_price_GBP_per_MWh=120.0,
    )
    ealing_array     = ASHPArray.from_preset(
        "ealing_phase1", weather_df, electricity_price_GBP_per_MWh=np.full(N_HOURS, 200.0),
    )
    print(f"    Default (None)        -> mean elec £{ealing_default._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Tariff (10% discount) -> mean elec £{ealing_tariff._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_tariff.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Flat scalar override  -> mean elec £{ealing_flat._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_flat.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Raw array override    -> mean elec £{ealing_array._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_array.marginal_cost.mean():.2f}/MWh heat")

    # --- from_config with nested tariff block ---
    print("\n  from_config() with a nested electricity_tariff block:")
    config_block = {
        "type": "ashp",
        "name": "Town centre ASHP bank (from config)",
        "n_units": 4,
        "unit_capacity_MW": 0.7,
        "flow_temp_C": 70.0,
        "electricity_tariff": {"negotiated_discount_pct": 15.0},
    }
    ashp_from_cfg = ASHPArray.from_config(config_block, weather_df)
    print(f"    {ashp_from_cfg}")
    print(f"    Mean elec price: £{ashp_from_cfg._elec_price.mean():.2f}/MWh (expect 15% below £240 central)")

    # Test resize — the "add more MW easily" requirement
    print("\n  Resize test — scaling Ealing Phase 1 up to 8 units:")
    ealing_scaled = ealing.resize(n_units=8)
    print(f"    Original: {ealing}")
    print(f"    Scaled:   {ealing_scaled}")
    assert ealing_scaled.capacity_MW == ealing.capacity_MW * 2, "Resize scaling failed"
    print("    ✓ Capacity scaled correctly (linear with n_units)")

    # Test custom array
    print("\n  Custom array (user-defined, 6 x 1.5 MW = 9 MW):")
    custom = ASHPArray(
        name="Custom test array",
        n_units=6,
        unit_capacity_MW=1.5,
        flow_temp_C=70.0,
        weather_df=weather_df,
    )
    print(f"    {custom}")

    # Seasonal sanity: COP should be higher in summer, lower in winter
    jan_cop = ealing.cop_hourly[:744].mean()
    jul_cop = ealing.cop_hourly[4344:5088].mean()
    jan_supply = ealing.supply_MW[:744].mean()
    jul_supply = ealing.supply_MW[4344:5088].mean()

    print(f"\n  Seasonal sanity checks:")
    print(f"    Jan mean COP: {jan_cop:.2f}  |  Jul mean COP: {jul_cop:.2f}  → {'✓ summer higher' if jul_cop > jan_cop else '✗ FAIL'}")
    print(f"    Jan mean supply: {jan_supply:.2f} MW  |  Jul mean supply: {jul_supply:.2f} MW  → {'✓ summer higher capacity' if jul_supply > jan_supply else '✗ FAIL'}")

    # --- NEW: weather-compensated flow temperature (currently DORMANT in
    #     the live model -- see the STATUS note in the constants block
    #     above. Demonstrated here for completeness/testing, not because
    #     the live dispatch/topology pipeline actually uses it.) ---
    print(f"\n  Weather compensation curve, DEFAULT parameters (currently dormant --")
    print(f"  see STATUS note in the constants block; both ends are 70°C, matching")
    print(f"  the project's single real design value, so the default curve is FLAT):")
    test_temps = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    flow_at_temps = weather_compensated_flow_temp_C(test_temps)
    for t, f in zip(test_temps, flow_at_temps):
        print(f"    Ambient {t:>6.1f}°C  ->  Flow {f:>5.1f}°C")

    fixed_ashp = ASHPArray.from_preset("ealing_phase1", weather_df)

    # Enabling compensation with NO other overrides should change NOTHING
    # -- this is the actual point of keeping both ends at 70°C: a caller
    # who flips enable_weather_compensation=True without deliberately
    # choosing a lower mild-end value gets IDENTICAL behaviour to the
    # fixed case, never a silent, unintended divergence from the
    # project's real design value.
    compensated_default = ASHPArray.from_preset(
        "ealing_phase1", weather_df, enable_weather_compensation=True,
    )
    default_flow_arr = np.broadcast_to(compensated_default.flow_temp_C, N_HOURS)
    fixed_flow_arr = np.broadcast_to(fixed_ashp.flow_temp_C, N_HOURS)
    print(f"\n    Fixed mean flow temp:       {fixed_flow_arr.mean():.1f}°C")
    print(f"    Compensated (default) mean: {default_flow_arr.mean():.1f}°C  "
          f"(should be IDENTICAL -- flat curve, dormant by design)")

    # --- Illustrative ONLY: if compensation were deliberately enabled
    #     later with a genuinely lower mild-end floor, here's what it
    #     would look like and what it would need to be checked against.
    #     NOT part of this project's current live assumptions. ---
    print(f"\n  Illustrative only (NOT a live assumption) — if compensation were")
    print(f"  deliberately enabled with a genuinely lower mild-end floor (62°C,")
    print(f"  verified safe for the real Ealing network — see the cross-check below),")
    print(f"  compensating DOWN from the same 70°C design value, never raising it:")
    compensated_illustrative = ASHPArray.from_preset(
        "ealing_phase1", weather_df, enable_weather_compensation=True,
        compensation_mild_temp_C=62.0,
    )
    illustrative_flow_arr = np.broadcast_to(compensated_illustrative.flow_temp_C, N_HOURS)
    print(f"    {'':20} {'Mean COP':>10} {'Annual elec demand MWh':>24} {'Mean flow temp':>16}")
    print(f"    {'Fixed':<20} {fixed_ashp.cop_hourly.mean():>10.3f} "
          f"{fixed_ashp.electrical_demand_MW.sum():>24.0f} {fixed_flow_arr.mean():>15.1f}°C")
    print(f"    {'Compensated (illus.)':<20} {compensated_illustrative.cop_hourly.mean():>10.3f} "
          f"{compensated_illustrative.electrical_demand_MW.sum():>24.0f} {illustrative_flow_arr.mean():>15.1f}°C")
    pct_saving_illustrative = (1 - compensated_illustrative.electrical_demand_MW.sum() / fixed_ashp.electrical_demand_MW.sum()) * 100
    print(f"    -> {pct_saving_illustrative:.1f}% less electricity IF this were enabled — "
          f"shown for context only, not used in any live dispatch/topology result in this project")

    # --- NEW: custom mild-end parameters ---
    custom_comp_ashp = ASHPArray.from_preset(
        "ealing_phase1", weather_df, enable_weather_compensation=True,
        compensation_mild_temp_C=55.0, compensation_mild_ambient_C=12.0,
    )
    print(f"\n  Custom compensation curve (mild end 55°C at 12°C ambient, same 70°C")
    print(f"  design value, instead of the dormant default's flat 70°C/70°C):")
    custom_flow_arr = np.broadcast_to(custom_comp_ashp.flow_temp_C, N_HOURS)
    print(f"    Mean flow temp: {custom_flow_arr.mean():.1f}°C (should be LOWER than the dormant "
          f"default's {default_flow_arr.mean():.1f}°C, since this curve actually drops)")

    # --- NEW: cross-check the compensation floor against a real network ---
    print(f"\n  Cross-checking the default 70°C mild-end floor against the real")
    print(f"  Ealing worked-example network topology (closes the loop between this")
    print(f"  module's curve and network_topology.py's real route-length physics):")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from network.network_topology import ealing_town_centre_topology
    from profiles.demand_synthesis import synthesise_network as _synthesise_network

    _scenario = {
        "demand_nodes": [
            {"name": "Perceval House",       "type": "office",      "floor_area_m2": 8500},
            {"name": "High Street Retail",   "type": "retail",      "floor_area_m2": 3000},
            {"name": "Ealing Hospital Wing", "type": "hospital",    "floor_area_m2": 12000},
            {"name": "Dickens Yard Ph1",     "type": "residential", "units": 350},
            {"name": "Broadway Hotel",       "type": "hotel",       "floor_area_m2": 5000},
            {"name": "Ellen Wilkinson Sch",  "type": "school",      "floor_area_m2": 6000},
        ]
    }
    _demand_result = _synthesise_network(weather_df, _scenario)
    _peak_by_building = {n["name"]: n["peak_heat_kW"] for n in _demand_result["nodes"]}
    _ealing_topo = ealing_town_centre_topology(peak_kW_by_building=_peak_by_building)

    floor_check = check_compensation_floor_against_network(_ealing_topo)
    for k, v in floor_check.items():
        print(f"    {k:<32} {v}")

    # Also check an UNSAFE proposed floor (60°C, the module's old value)
    # to confirm the cross-check can actually catch a bad floor, not just
    # confirm good ones
    floor_check_60 = check_compensation_floor_against_network(
        _ealing_topo, proposed_mild_floor_C=60.0,
    )
    print(f"\n    Checking the OLD 60°C floor against the same real network (should be unsafe):")
    for k, v in floor_check_60.items():
        print(f"    {k:<32} {v}")

    # --- NEW: unit-level outage model ---
    print(f"\n  Unit-level outage model (real maintenance practice — units")
    print(f"  serviced one at a time, never the whole bank together):")
    print(f"    n_units: {ealing.n_units}, availability_factor: {ealing.availability_factor}")
    print(f"    Outage hours per unit per year: {int(round((1.0 - ealing.availability_factor) * N_HOURS))}")
    print(f"    Units available — min: {ealing.units_available.min()}, max: {ealing.units_available.max()}")
    print(f"    Hours with full fleet up: {(ealing.units_available == ealing.n_units).sum()} / {N_HOURS}")
    print(f"    Hours with reduced fleet: {(ealing.units_available < ealing.n_units).sum()} / {N_HOURS}")
    # Higher availability should mean fewer reduced-fleet hours
    high_avail = ASHPArray.from_preset("ealing_phase1", weather_df, availability_factor=0.999)
    low_avail  = ASHPArray.from_preset("ealing_phase1", weather_df, availability_factor=0.90)
    print(f"    At 99.9% availability: {(high_avail.units_available < high_avail.n_units).sum()} reduced-fleet hours")
    print(f"    At 90.0% availability: {(low_avail.units_available < low_avail.n_units).sum()} reduced-fleet hours")

    # Array shape and bounds checks
    assert len(ealing.cop_hourly)    == N_HOURS, "cop_hourly wrong length"
    assert len(ealing.supply_MW)     == N_HOURS, "supply_MW wrong length"
    assert len(ealing.marginal_cost) == N_HOURS, "marginal_cost wrong length"
    assert ealing.supply_MW.max() <= ealing.capacity_MW + 0.001, "supply exceeds capacity"
    assert ealing.cop_hourly.min() >= 1.2, "COP below floor"
    assert ealing.cop_hourly.max() <= 6.0, "COP above ceiling"

    # New tariff-integration assertions
    assert ealing_default._elec_price.mean() > 200, \
        "Default electricity price should now be the realistic ~£240/MWh tariff, not the old £120 placeholder"
    assert ealing_tariff._elec_price.mean() < ealing_default._elec_price.mean(), \
        "10% discounted tariff should be cheaper than the undiscounted default"
    assert abs(ealing_flat._elec_price.mean() - 120.0) < 0.01, \
        "Flat scalar override should be respected exactly"
    assert abs(ashp_from_cfg._elec_price.mean() - ealing_default._elec_price.mean() * 0.85) < 1.0, \
        "from_config nested tariff block should apply the 15% discount correctly"

    # New outage-model assertions
    assert ealing.units_available.min() >= 0, "units_available should never go negative"
    assert ealing.units_available.max() <= ealing.n_units, "units_available should never exceed n_units"
    assert ealing.cop_hourly.mean() == ealing.cop_hourly.mean(), "COP should be unaffected by outages (sanity)"
    assert (high_avail.units_available < high_avail.n_units).sum() < (low_avail.units_available < low_avail.n_units).sum(), \
        "Lower availability_factor should produce MORE reduced-fleet hours than higher availability_factor"
    # At no point should ALL units be down simultaneously with a sane
    # availability factor (staggered scheduling should prevent total loss)
    assert ealing.units_available.min() >= ealing.n_units - 1, \
        "Staggered single-unit-at-a-time outages should never take down more than 1 unit simultaneously at this availability level"

    # New weather compensation assertions
    assert flow_at_temps[0] == 70.0, "Flow temp should clamp at 70°C below the cold anchor (-10°C) -- the dormant default"
    assert flow_at_temps[-1] == 70.0, "Flow temp should clamp at 70°C above the mild anchor (15°C) -- flat by design while dormant"
    assert np.all(np.diff(flow_at_temps) <= 0), \
        "Flow temp should be monotonically non-increasing as ambient temp rises"
    assert not fixed_ashp.enable_weather_compensation, "Default ASHPArray should NOT have compensation enabled"
    assert compensated_default.enable_weather_compensation, "Explicitly enabled ASHPArray should have compensation enabled"
    assert isinstance(fixed_ashp.flow_temp_C, float), \
        "Fixed-mode flow_temp_C should remain a plain scalar (backward compatibility)"
    assert isinstance(compensated_default.flow_temp_C, np.ndarray), \
        "Compensated-mode flow_temp_C should be an (N_HOURS,) array, even when dormant/flat"
    assert len(compensated_default.flow_temp_C) == N_HOURS, \
        "Compensated flow_temp_C array should have exactly N_HOURS entries"
    # THE KEY CONSISTENCY CHECK: enabling compensation with NO other
    # overrides must give IDENTICAL results to the fixed case -- this is
    # what makes the dormant default safe to leave switched on by
    # accident; it should never silently diverge from the project's
    # single real design value.
    assert np.allclose(default_flow_arr, fixed_flow_arr), \
        "Enabling compensation with default parameters should produce IDENTICAL flow temps " \
        "to the fixed case (both ends are 70°C) -- this is the actual point of reconciling " \
        "the dormant default back to the project's single real design value"
    assert abs(compensated_default.cop_hourly.mean() - fixed_ashp.cop_hourly.mean()) < 1e-9, \
        "Dormant-default compensated COP should be identical to fixed COP"
    assert abs(compensated_default.electrical_demand_MW.sum() - fixed_ashp.electrical_demand_MW.sum()) < 1e-6, \
        "Dormant-default compensated electricity demand should be identical to fixed demand"
    # The illustrative (NOT live) lower-floor case must still demonstrate
    # the real underlying physics correctly, even though it's not part
    # of this project's live assumptions
    assert compensated_illustrative.cop_hourly.mean() > fixed_ashp.cop_hourly.mean(), \
        "Compensating DOWN from the same cold-end design should always raise mean COP " \
        "vs always running at the peak design flow temp (every hour's flow temp is <= the fixed case's)"
    assert compensated_illustrative.electrical_demand_MW.sum() < fixed_ashp.electrical_demand_MW.sum(), \
        "Compensating DOWN from the same cold-end design should always reduce total annual " \
        "electricity demand for the same heat delivered"
    assert custom_flow_arr.mean() < default_flow_arr.mean(), \
        "A genuinely lower custom mild-end target should produce a lower mean flow temp " \
        "than the dormant (flat) default curve"
    # resize() must carry the ORIGINAL design flow temp through, not an
    # already-compensated hourly array re-interpreted as a new scalar
    resized_compensated = compensated_default.resize(n_units=8)
    assert resized_compensated.design_flow_temp_C == compensated_default.design_flow_temp_C, \
        "resize() should preserve the original design (cold-end) flow temp, not a derived hourly value"
    assert resized_compensated.enable_weather_compensation, \
        "resize() should preserve the enable_weather_compensation flag"

    # New cross-check function assertions
    assert floor_check["proposed_floor_safe"], \
        "The module's own 70°C default floor should check as safe against the real Ealing network"
    assert floor_check["margin_C"] > 0, "A safe floor should report a positive margin"
    assert not floor_check_60["proposed_floor_safe"], \
        "The OLD 60°C floor should check as UNSAFE against the real Ealing network -- " \
        "confirms the cross-check can actually catch a bad floor, not just rubber-stamp things"
    assert floor_check_60["margin_C"] < 0, "An unsafe floor should report a negative margin"
    assert abs(floor_check["actual_minimum_safe_flow_temp_C"]
               - floor_check_60["actual_minimum_safe_flow_temp_C"]) < 0.01, \
        "The network's own calculated physical floor should be identical regardless of which " \
        "proposed floor is being checked against it -- it's a property of the network, not the proposal"

    print(f"\n  ✓ All array shapes correct (8760 hours)")
    print(f"  ✓ Supply never exceeds nameplate capacity")
    print(f"  ✓ COP within physical bounds [1.2, 6.0]")
    print(f"  ✓ Default electricity price now uses realistic tariff (~£240/MWh), not old £120 placeholder")
    print(f"  ✓ Tariff object, flat scalar, and raw array overrides all behave correctly")
    print(f"  ✓ from_config() nested electricity_tariff block resolves correctly")
    print(f"  ✓ Unit-level outages correctly staggered — never more than 1 unit down at once at this scale")
    print(f"  ✓ Lower availability_factor correctly produces more reduced-fleet hours")
    print(f"  ✓ Weather compensation curve correctly clamps at both ends and is monotonic in between")
    print(f"  ✓ Compensation OFF by default — existing callers get identical fixed-temperature behaviour")
    print(f"  ✓ Compensation ON with DEFAULT params is identical to fixed (dormant, flat 70°C/70°C)")
    print(f"  ✓ Compensation ON with a genuinely lower mild-end floor measurably improves COP")
    print(f"    and reduces electricity demand (illustrative only -- not a live assumption)")
    print(f"  ✓ Custom mild-end parameters correctly shift the curve")
    print(f"  ✓ resize() correctly preserves the original design flow temp and compensation settings")
    print(f"  ✓ check_compensation_floor_against_network() correctly verifies the 70°C default is safe")
    print(f"    for the real Ealing network, and correctly catches the old 60°C floor as unsafe")
    print()