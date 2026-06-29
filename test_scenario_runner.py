"""
test_economics.py
======================
Self-test / demonstration suite for economics.CAPEX, economics.OPEX,
and economics.metrics — the whole-scheme CAPEX/OPEX aggregation, the
three individual-system counterfactuals (gas boiler, individual ASHP,
individual AC), and the financial metrics (simple/discounted payback,
NPV, LCOH) built on top of them.

Run directly: python3 tests/test_economics.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from economics.CAPEX import aggregate_capex, individual_system_capex_GBP, INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW
from economics.OPEX import annual_om_cost_GBP, total_annual_opex_GBP, DEFAULT_OM_RATE
from economics.metrics import (
    counterfactual_gas_boiler_dispatch, counterfactual_individual_ashp_dispatch,
    counterfactual_individual_ac_dispatch, aggregate_counterfactual,
    simple_payback_years, discounted_cash_flow_series, npv, discounted_payback_years,
    levelised_cost_of_heat_GBP_per_kWh, DEFAULT_DISCOUNT_RATE, DEFAULT_PROJECT_LIFETIME_YEARS,
)
from components.ASHP import ASHPArray
from components.EfW import EfWChp
from components.datacentre_source import DataCentre
from components.peak_demand_option import GasBoiler, ElectricBoiler
from components.thermal_storage import ThermalStorage
from network.network_topology import ealing_town_centre_topology
from optimisation.dispatch import run_dispatch
from profiles.demand_synthesis import synthesise_network


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  test_economics.py — CAPEX / OPEX / metrics self-test")
    print("=" * 70)

    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    weather_df = pd.DataFrame({"temp_drybulb_C": T})

    # --- Build the real Ealing scenario (1x scale, matching network_topology.py's own worked example) ---
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
    network = synthesise_network(weather_df, scenario)
    demand_kW = network["total_heat_kW"]
    demand_cool_kW = network["total_cooling_kW"]

    # --- Real centralised scheme: source stack + network + storage ---
    print("\n  Building the real centralised scheme (sources + network + storage):")
    dc = DataCentre.from_preset("redwire_ealing", weather_df)
    efw = EfWChp.from_preset("newlincs_style")
    ashp = ASHPArray.from_preset("ealing_phase1", weather_df)
    gas = GasBoiler.from_preset("ealing_phase1")
    elec = ElectricBoiler.from_preset("ealing_backup")
    store = ThermalStorage(name="store", capacity_MWh=1.74, max_charge_MW=2.8, max_discharge_MW=2.8, delta_T_K=30.0)
    sources = [dc, efw, ashp, gas, elec]

    peak_by_building = {n["name"]: n["peak_heat_kW"] for n in network["nodes"]}
    topo = ealing_town_centre_topology(peak_kW_by_building=peak_by_building)
    sized = topo.size_all_segments(flow_temp_C=70.0, return_temp_C=40.0)

    result = run_dispatch(
        demand_kW, sources, storage=store,
        network_topology=topo, network_flow_temp_C=70.0, network_return_temp_C=40.0,
        network_sized_segments=sized, duty="heat",
    )

    capex_result = aggregate_capex(sources=sources, network_topology=topo, sized_segments=sized, storage=store)
    print(f"    Whole-scheme CAPEX: £{capex_result['grand_total_GBP']:,.0f}")
    print(f"    By category: {capex_result['by_category']}")

    opex_result = total_annual_opex_GBP(result, capex_GBP=capex_result["grand_total_GBP"])
    print(f"    Whole-scheme annual OPEX: £{opex_result['total_GBP']:,.0f} "
          f"(fuel/elec £{opex_result['fuel_electricity_GBP']:,.0f} + O&M £{opex_result['om_GBP']:,.0f})")

    # --- Counterfactual 1: individual gas boilers, no network ---
    print(f"\n  Counterfactual 1: individual gas boilers (no network):")
    cf_gas = aggregate_counterfactual(network["nodes"], counterfactual_gas_boiler_dispatch)
    print(f"    Total CAPEX: £{cf_gas['total_capex_GBP']:,.0f}")
    print(f"    Total annual OPEX: £{cf_gas['total_annual_opex_GBP']:,.0f}")
    for name, b in cf_gas["by_building"].items():
        print(f"      {name:<22}: CAPEX £{b['capex_GBP']:>9,.0f}, OPEX £{b['annual_opex_GBP']:>8,.0f}/yr")

    # --- Counterfactual 2: individual ASHPs, no network ---
    print(f"\n  Counterfactual 2: individual ASHPs (no network):")
    cf_ashp = aggregate_counterfactual(network["nodes"], counterfactual_individual_ashp_dispatch, weather_df=weather_df)
    print(f"    Total CAPEX: £{cf_ashp['total_capex_GBP']:,.0f}")
    print(f"    Total annual OPEX: £{cf_ashp['total_annual_opex_GBP']:,.0f}")

    # --- Counterfactual 3: individual AC, no network (cooling duty) ---
    print(f"\n  Counterfactual 3: individual AC units (no network, cooling duty):")
    cf_ac = aggregate_counterfactual(network["nodes"], counterfactual_individual_ac_dispatch, weather_df=weather_df)
    print(f"    Total CAPEX: £{cf_ac['total_capex_GBP']:,.0f}")
    print(f"    Total annual OPEX: £{cf_ac['total_annual_opex_GBP']:,.0f}")

    # --- Network vs individual gas boiler: the actual comparison ---
    print(f"\n  ── Network vs. individual gas boilers (the real comparison) ──")
    print(f"    Network:    CAPEX £{capex_result['grand_total_GBP']:>12,.0f}   OPEX £{opex_result['total_GBP']:>10,.0f}/yr")
    print(f"    Individual: CAPEX £{cf_gas['total_capex_GBP']:>12,.0f}   OPEX £{cf_gas['total_annual_opex_GBP']:>10,.0f}/yr")
    avoided_cost = cf_gas["total_annual_opex_GBP"] - opex_result["total_GBP"]
    print(f"    Annual avoided cost (network's OPEX advantage): £{avoided_cost:,.0f}/yr")

    # --- Financial metrics on this real comparison ---
    print(f"\n  Financial metrics (network's extra CAPEX vs. individual gas boilers' baseline):")
    extra_capex = capex_result["grand_total_GBP"] - cf_gas["total_capex_GBP"]
    print(f"    Network's extra CAPEX over individual: £{extra_capex:,.0f}")

    payback_simple = simple_payback_years(extra_capex, avoided_cost)
    print(f"    Simple payback: {payback_simple:.1f} years" if payback_simple else "    Simple payback: never (negative/zero avoided cost)")

    payback_discounted = discounted_payback_years(extra_capex, avoided_cost)
    print(f"    Discounted payback ({DEFAULT_DISCOUNT_RATE:.1%} discount rate): "
          f"{payback_discounted:.1f} years" if payback_discounted else
          f"    Discounted payback: never within {DEFAULT_PROJECT_LIFETIME_YEARS} years")

    npv_result = npv(extra_capex, avoided_cost)
    print(f"    NPV ({DEFAULT_DISCOUNT_RATE:.1%} discount rate, {DEFAULT_PROJECT_LIFETIME_YEARS}-year lifetime): £{npv_result:,.0f}")

    print(f"\n    NPV sensitivity across the real cited 9-12% discount rate range:")
    for r in [0.09, 0.105, 0.12]:
        npv_at_r = npv(extra_capex, avoided_cost, discount_rate=r)
        print(f"      at {r:.1%}: NPV = £{npv_at_r:,.0f}")

    # --- LCOH for the network itself ---
    annual_heat_kWh = float(demand_kW.sum())
    lcoh_network = levelised_cost_of_heat_GBP_per_kWh(
        capex_result["grand_total_GBP"], opex_result["total_GBP"], annual_heat_kWh,
    )
    lcoh_gas = levelised_cost_of_heat_GBP_per_kWh(
        cf_gas["total_capex_GBP"], cf_gas["total_annual_opex_GBP"], annual_heat_kWh,
    )
    print(f"\n  Levelised Cost of Heat comparison:")
    print(f"    Network:              £{lcoh_network*100:.2f} p/kWh")
    print(f"    Individual gas boiler: £{lcoh_gas*100:.2f} p/kWh")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert capex_result["grand_total_GBP"] > 0, "Whole-scheme CAPEX should be positive"
    assert capex_result["by_category"]["network_GBP"] > 0, "Network CAPEX should be a real, positive component"
    assert opex_result["om_GBP"] == round(capex_result["grand_total_GBP"] * DEFAULT_OM_RATE, 0), \
        "O&M cost should be exactly 1% of whole-scheme CAPEX (the real cited CHDU figure)"
    assert opex_result["total_GBP"] > opex_result["fuel_electricity_GBP"], \
        "Total OPEX (with O&M added) should exceed fuel/electricity OPEX alone"

    assert cf_gas["total_capex_GBP"] > 0, "Gas boiler counterfactual CAPEX should be positive"
    assert cf_gas["total_capex_GBP"] < capex_result["grand_total_GBP"], \
        "Individual gas boilers should have LOWER total CAPEX than the centralised scheme " \
        "(no network/pipework costs at all) -- a real, expected structural result"
    for name in peak_by_building:
        assert name in cf_gas["by_building"], f"{name} should appear in the gas boiler counterfactual breakdown"

    assert cf_ashp["total_capex_GBP"] > cf_gas["total_capex_GBP"], \
        "Individual ASHPs should cost MORE upfront than individual gas boilers " \
        "(£1,150/kW vs £111/kW -- a real, large difference, not a rounding artifact)"

    assert cf_ac["total_capex_GBP"] > 0, "AC counterfactual CAPEX should be positive"

    assert simple_payback_years(1000.0, 0.0) is None, \
        "Simple payback should return None (not raise or return infinity) for zero avoided cost"
    assert simple_payback_years(1000.0, -100.0) is None, \
        "Simple payback should return None for NEGATIVE avoided cost (the project never pays back)"
    assert simple_payback_years(1000.0, 100.0) == 10.0, \
        "Simple payback should be exactly CAPEX/annual_saving for a basic positive case"

    cash_flows_test = discounted_cash_flow_series(100.0, project_lifetime_years=5, discount_rate=0.10)
    assert len(cash_flows_test) == 5, "Cash flow series should have exactly project_lifetime_years entries"
    assert np.all(np.diff(cash_flows_test) < 0), \
        "Discounted cash flow should be monotonically DECREASING year over year " \
        "(money further in the future is worth less today)"
    assert abs(cash_flows_test[0] - 100.0/1.10) < 0.01, \
        "Year 1's discounted cash flow should be exactly CashFlow/(1+r)^1"

    npv_zero_capex = npv(0.0, 100.0, project_lifetime_years=5, discount_rate=0.10)
    assert npv_zero_capex > 0, "NPV with zero CAPEX and positive cash flow should always be positive"
    npv_huge_capex = npv(1e9, 100.0, project_lifetime_years=5, discount_rate=0.10)
    assert npv_huge_capex < 0, "NPV with enormous CAPEX relative to cash flow should be negative"

    # Discounted payback should always be >= simple payback for the same
    # inputs, since discounting only makes future money WORTH LESS,
    # never more -- it should take longer (or equal, in the limit of a
    # zero discount rate) to recover the same CAPEX
    test_capex, test_saving = 1000.0, 150.0
    simple_pb = simple_payback_years(test_capex, test_saving)
    discounted_pb = discounted_payback_years(test_capex, test_saving, project_lifetime_years=30)
    assert discounted_pb is not None and discounted_pb >= simple_pb, \
        "Discounted payback should always take AT LEAST as long as simple payback " \
        "for the same CAPEX/saving, since discounting only reduces future cash flow value"

    # A scenario where discounted payback never occurs within the project
    # lifetime, even though simple payback would suggest it eventually pays back
    never_payback = discounted_payback_years(1_000_000.0, 1000.0, project_lifetime_years=25, discount_rate=0.10)
    assert never_payback is None, \
        "Discounted payback should correctly return None when cumulative discounted " \
        "cash flow never reaches CAPEX within the assumed project lifetime"

    lcoh_test = levelised_cost_of_heat_GBP_per_kWh(
        capex_GBP=1000.0, annual_opex_GBP=100.0, annual_heat_delivered_kWh=10000.0,
        project_lifetime_years=10,
    )
    expected_lcoh = (1000.0 + 100.0*10) / (10000.0*10)
    assert abs(lcoh_test - expected_lcoh) < 1e-9, "LCOH should match the UK gov't definition exactly"

    try:
        levelised_cost_of_heat_GBP_per_kWh(1000.0, 100.0, 0.0)
        print("    ✗ FAIL: should have raised ValueError for zero heat delivered")
    except ValueError:
        print("    ✓ Correctly raised ValueError for zero annual_heat_delivered_kWh")

    print("  ✓ Whole-scheme CAPEX/OPEX aggregation produces real, positive, correctly-composed totals")
    print("  ✓ O&M cost is exactly 1% of CAPEX (the real cited CHDU/DECC figure)")
    print("  ✓ Individual gas boiler counterfactual correctly has LOWER CAPEX than the centralised")
    print("    scheme (no network costs) -- the real structural trade-off this comparison exists to show")
    print("  ✓ Individual ASHP correctly costs more upfront than individual gas boiler (real £/kW gap)")
    print("  ✓ Simple payback correctly handles zero/negative avoided cost (returns None, not an error)")
    print("  ✓ Discounted cash flow is monotonically decreasing, matching real discounting behaviour")
    print("  ✓ NPV correctly flips sign between a trivial-CAPEX and an enormous-CAPEX scenario")
    print("  ✓ Discounted payback always takes >= as long as simple payback, and correctly returns")
    print("    None when payback never occurs within the project lifetime")
    print("  ✓ LCOH matches the UK government's own cited definition exactly")
    print()
