"""
test_tariffs.py
======================
Self-test / demonstration suite for economics.tariffs (ElectricityTariff,
GasTariff, resolve_electricity_price(), resolve_gas_price(), and the
GAS_PRICE_SCENARIOS table). Moved out of tariffs.py itself as part of a
project-wide split separating logic files from their self-tests.

Run directly: python3 tests/test_tariffs.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from economics.tariffs import (
    N_HOURS, GAS_PRICE_SCENARIOS, ElectricityTariff, GasTariff,
    resolve_electricity_price, resolve_gas_price,
)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  tariffs.py — self-test")
    print("="*70)

    # --- Electricity: central commercial case ---
    print("\n  Electricity tariff — central commercial case (no discount):")
    elec = ElectricityTariff()
    for k, v in elec.summary().items():
        print(f"    {k:<32} {v}")

    # --- Electricity: with a negotiated discount ---
    print("\n  Electricity tariff — with 10% negotiated discount (e.g. EDF relationship):")
    elec_discounted = ElectricityTariff(negotiated_discount_pct=10.0)
    for k, v in elec_discounted.summary().items():
        print(f"    {k:<32} {v}")

    # --- Electricity: escalated to a future year ---
    print("\n  Electricity tariff — escalated to 2040 (default 1.5%/yr real-terms):")
    elec_2040 = elec.escalate_to_year(2040)
    print(f"    2026 average: {elec.summary()['actual_mean_p_per_kWh']} p/kWh")
    print(f"    2040 average: {elec_2040.summary()['actual_mean_p_per_kWh']} p/kWh")

    # --- Gas: both scenarios ---
    print("\n  Gas tariff — sensitivity pair:")
    for key in GAS_PRICE_SCENARIOS:
        gas = GasTariff.from_scenario(key)
        s = gas.summary()
        print(f"    {key:<18} {s['price_p_per_therm']:>6.1f} p/therm  =  £{s['price_GBP_per_MWh']:>6.2f}/MWh   ({GAS_PRICE_SCENARIOS[key]['reference']})")

    # --- Gas: escalated ---
    print("\n  Gas tariff — DESNZ central, escalated to 2040:")
    gas_central = GasTariff.from_scenario("desnz_central")
    gas_2040 = gas_central.escalate_to_year(2040)
    print(f"    2026: £{gas_central.summary()['price_GBP_per_MWh']:.2f}/MWh")
    print(f"    2040: £{gas_2040.summary()['price_GBP_per_MWh']:.2f}/MWh")

    # --- Integration check: feed into ASHP-style marginal cost calc ---
    print("\n  Integration check — ASHP marginal cost using real tariff shape:")
    test_cop = np.full(N_HOURS, 3.0)  # flat COP for isolation test
    ashp_marginal_cost = elec.price_GBP_per_MWh / test_cop
    print(f"    Mean ASHP marginal cost (COP=3.0 flat): £{ashp_marginal_cost.mean():.2f}/MWh heat")
    print(f"    Cheapest hour:  £{ashp_marginal_cost.min():.2f}/MWh heat")
    print(f"    Most expensive: £{ashp_marginal_cost.max():.2f}/MWh heat")

    # --- New: resolve_*_price() helper checks ---
    print("\n  resolve_electricity_price() — all four accepted input types:")
    none_arr   = resolve_electricity_price(None)
    tariff_arr = resolve_electricity_price(elec_discounted)
    scalar_arr = resolve_electricity_price(150.0)
    array_arr  = resolve_electricity_price(np.full(N_HOURS, 99.0))
    print(f"    None              -> mean £{none_arr.mean():.2f}/MWh   (default central tariff)")
    print(f"    Tariff object     -> mean £{tariff_arr.mean():.2f}/MWh  (10% discount applied)")
    print(f"    Scalar override   -> mean £{scalar_arr.mean():.2f}/MWh  (flat, no shape)")
    print(f"    Raw array         -> mean £{array_arr.mean():.2f}/MWh  (flat, no shape)")

    print("\n  resolve_gas_price() — all four accepted input types:")
    none_gas   = resolve_gas_price(None)
    tariff_gas = resolve_gas_price(GasTariff.from_scenario("current_actual"))
    scalar_gas = resolve_gas_price(50.0)
    print(f"    None              -> £{none_gas.mean():.2f}/MWh   (desnz_central default)")
    print(f"    Tariff object     -> £{tariff_gas.mean():.2f}/MWh  (current_actual scenario)")
    print(f"    Scalar override   -> £{scalar_gas.mean():.2f}/MWh  (flat override)")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert len(elec.price_GBP_per_MWh) == N_HOURS, "Electricity price array wrong length"
    assert len(gas_central.price_GBP_per_MWh) == N_HOURS, "Gas price array wrong length"
    assert elec.price_p_per_kWh.min() > 0, "Electricity price went non-positive"
    assert elec_discounted.annual_avg_p_per_kWh_effective < elec.annual_avg_p_per_kWh_effective, \
        "Discount should reduce effective price"
    assert elec_2040.annual_avg_p_per_kWh_effective > elec.annual_avg_p_per_kWh_effective, \
        "Escalation should increase future price"
    summary = elec.summary()
    assert summary["peak_to_overnight_ratio"] > 1.5, "Peak should be meaningfully higher than overnight"
    assert len(none_arr) == N_HOURS and len(tariff_arr) == N_HOURS, "resolve_electricity_price wrong length"
    assert len(none_gas) == N_HOURS, "resolve_gas_price wrong length"
    try:
        resolve_electricity_price(np.zeros(100))
        raise AssertionError("Should have rejected a wrong-length array")
    except ValueError:
        pass
    print("  ✓ All array shapes correct (8760 hours)")
    print("  ✓ Negotiated discount reduces effective price")
    print("  ✓ Escalation increases future price")
    print("  ✓ Evening peak meaningfully more expensive than overnight (matches Agile pattern)")
    print("  ✓ resolve_electricity_price() and resolve_gas_price() handle all input types")
    print("  ✓ Wrong-length array input correctly rejected")
    print()