"""Investor-facing data-centre waste-heat feasibility scenarios.

The cases use the Ealing-calibrated customer demand so that changes are caused
by the heat-source, route and commercial assumptions rather than by changing
the customer base between alternatives.
"""
from copy import deepcopy

from scenarios.ealing_report_validation import scenario_copy
from scenarios.feasibility_comparison import COMPACT_PRIVATE


BASE_ROUTE_M = 2_148.0
BASE_NETWORK_CAPEX_GBP = 10_461_831.0
BASE_NETWORK_LOSS_MWH = 974.614


def data_centre_case(
    name: str,
    *,
    route_m: float = 1_200.0,
    recoverable_heat_MW: float = 2.0,
    source_temperature_C: float = 40.0,
    source_availability: float = 0.97,
    waste_heat_fee_GBP_per_MWh: float = 0.0,
    booster_capacity_MW: float = 3.2,
    backup_capacity_MW: float = 7.5,
    grant_rate: float = 0.49,
    discount_rate: float = 0.105,
    connection_contribution_GBP_per_kW: float = 1_000.0,
    heat_tariff_p_per_kWh: float = 9.56,
    heat_tariff_mode: str = "counterfactual_bill_parity",
) -> dict:
    """Build one fully explicit, JSON-compatible waste-heat case."""
    scenario = scenario_copy()
    scenario["name"] = name
    route_ratio = float(route_m) / BASE_ROUTE_M
    scenario["network"].update({
        "length_m": float(route_m),
        "capex_GBP_override": BASE_NETWORK_CAPEX_GBP * route_ratio,
        "annual_heat_loss_MWh_override": BASE_NETWORK_LOSS_MWH * route_ratio,
    })

    sources = [
        {
            "type": "data_centre",
            "preset": "redwire_ealing",
            "name": "Recovered data-centre heat",
            # For this scenario contract capacity_MW is recoverable low-grade
            # source heat, before the booster heat pump.
            "capacity_MW": float(recoverable_heat_MW),
            "supply_temp_C": float(source_temperature_C),
            "availability_factor": float(source_availability),
            "waste_heat_cost_GBP_per_MWh": float(waste_heat_fee_GBP_per_MWh),
        },
        {
            "type": "booster_heat_pump",
            "preset": "generic_5MW",
            "name": "Data-centre booster heat pump",
            "capacity_MW": float(booster_capacity_MW),
            "n_units": 4,
            "depends_on": 0,
            "electricity_price_GBP_per_MWh": 180.5,
        },
    ]
    if backup_capacity_MW > 0:
        sources.append({
            "type": "gas_boiler",
            "preset": "ealing_phase1",
            "name": "Independent peak and reserve boiler",
            "capacity_MW": float(backup_capacity_MW),
            "eta_full_load": 0.90,
            "gas_price_GBP_per_MWh": 46.9,
            "capex_GBP_per_MW": 494_400.0 / 3.6,
        })
    scenario["sources"] = sources

    economics = scenario["economics"]
    economics["discount_rate"] = float(discount_rate)
    economics["tariffs"]["heat_tariff_mode"] = heat_tariff_mode
    economics["tariffs"]["heat_unit_rate_p_per_kWh"] = float(heat_tariff_p_per_kWh)
    economics["ghnf_grant"] = {
        "enabled": grant_rate > 0,
        "rate": float(grant_rate),
    }
    for building in scenario["demand"]["buildings"]:
        # Customer-specific values take precedence over the scenario default.
        if heat_tariff_mode == "manual":
            building["heat_unit_rate_p_per_kWh"] = float(heat_tariff_p_per_kWh)
        else:
            building.pop("heat_unit_rate_p_per_kWh", None)
        building["connection_charge_GBP_per_kW"] = float(
            connection_contribution_GBP_per_kW
        )
    return scenario


DATA_CENTRE_ONLY = data_centre_case(
    "DC1 - Data-centre-only service stress test",
    recoverable_heat_MW=5.5,
    source_temperature_C=40.0,
    booster_capacity_MW=7.5,
    backup_capacity_MW=0.0,
)
DATA_CENTRE_ONLY["description"] = (
    "Ample liquid-cooled waste heat but no independent peak/reserve plant. "
    "This deliberately tests whether a data centre can be the sole source."
)

AIR_COOLED_HYBRID = data_centre_case(
    "DC2 - Typical air-cooled data centre plus reserve",
    route_m=1_800.0,
    recoverable_heat_MW=3.6,
    source_temperature_C=30.0,
    source_availability=0.95,
    waste_heat_fee_GBP_per_MWh=5.0,
    booster_capacity_MW=6.8,
    backup_capacity_MW=7.5,
)
AIR_COOLED_HYBRID["description"] = (
    "Typical 30C low-grade heat, GBP5/MWh heat fee and a longer customer route. "
    "Technically resilient, but included to test commercial viability."
)

OPTIMISED_LIQUID_COOLED_HYBRID = data_centre_case(
    "DC3 - Compact liquid-cooled baseload hybrid",
    route_m=1_200.0,
    recoverable_heat_MW=2.0,
    source_temperature_C=40.0,
    source_availability=0.97,
    waste_heat_fee_GBP_per_MWh=0.0,
    booster_capacity_MW=3.2,
    backup_capacity_MW=7.5,
)
OPTIMISED_LIQUID_COOLED_HYBRID["description"] = (
    "Higher-grade 40C liquid-cooled heat sized to baseload, a compact route, "
    "independent reserve, high eligible grant and GBP1,000/kW contributions."
)

UNSUPPORTED_LIQUID_COOLED_HYBRID = data_centre_case(
    "DC4 - Same compact hybrid without grant or contributions",
    route_m=1_200.0,
    recoverable_heat_MW=2.0,
    source_temperature_C=40.0,
    source_availability=0.97,
    waste_heat_fee_GBP_per_MWh=0.0,
    booster_capacity_MW=3.2,
    backup_capacity_MW=7.5,
    grant_rate=0.0,
    connection_contribution_GBP_per_kW=0.0,
)
UNSUPPORTED_LIQUID_COOLED_HYBRID["description"] = (
    "Identical engineering to DC3, with the grant and customer capital "
    "contributions removed to isolate the funding-stack effect."
)


DATA_CENTRE_SCENARIOS = [
    DATA_CENTRE_ONLY,
    AIR_COOLED_HYBRID,
    OPTIMISED_LIQUID_COOLED_HYBRID,
    UNSUPPORTED_LIQUID_COOLED_HYBRID,
]

LIFETIME_COMPARISON_SCENARIOS = [
    deepcopy(COMPACT_PRIVATE),
    *deepcopy(DATA_CENTRE_SCENARIOS),
]


def scenario_copies():
    return deepcopy(DATA_CENTRE_SCENARIOS)
