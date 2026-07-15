"""Comparable dense-core, compact-private and extended-route scenarios."""
from copy import deepcopy
from scenarios.ealing_report_validation import scenario_copy


BASE_ROUTE_M = 2_148.0
BASE_NETWORK_CAPEX_GBP = 10_461_831.0
BASE_LOSS_MWH = 974.614


def _route_case(name, length_m, grant_rate, discount_rate, connection_charge_per_kW):
    scenario = scenario_copy()
    scenario["name"] = name
    ratio = float(length_m) / BASE_ROUTE_M
    scenario["network"]["length_m"] = float(length_m)
    scenario["network"]["capex_GBP_override"] = BASE_NETWORK_CAPEX_GBP * ratio
    scenario["network"]["annual_heat_loss_MWh_override"] = BASE_LOSS_MWH * ratio
    scenario["economics"]["discount_rate"] = float(discount_rate)
    scenario["economics"]["tariffs"]["heat_tariff_mode"] = "counterfactual_bill_parity"
    scenario["economics"]["ghnf_grant"] = {"enabled": True, "rate": float(grant_rate)}
    for building in scenario["demand"]["buildings"]:
        building.pop("heat_unit_rate_p_per_kWh", None)
        building["connection_charge_GBP_per_kW"] = float(connection_charge_per_kW)
    return scenario


DENSE_CORE_PUBLIC = _route_case(
    "F1 - Dense town-centre core / public appraisal",
    length_m=2_148, grant_rate=0.34, discount_rate=0.035,
    connection_charge_per_kW=600,
)
DENSE_CORE_PUBLIC["description"] = (
    "Ealing-calibrated 14.2 GWh anchor-load core, 6.6 MWh/m, 34% grant on "
    "eligible CAPEX, GBP600/kW customer contributions and a 3.5% real appraisal rate."
)

COMPACT_PRIVATE = _route_case(
    "F2 - Compact anchor cluster / private hurdle",
    length_m=1_800, grant_rate=0.49, discount_rate=0.105,
    connection_charge_per_kW=1_000,
)
COMPACT_PRIVATE["description"] = (
    "Same customers compressed to a 1.8 km route, 7.9 MWh/m, high eligible-grant "
    "case and GBP1,000/kW connection contribution at a 10.5% hurdle rate."
)

EXTENDED_ROUTE_REJECT = _route_case(
    "F3 - Extended lower-density route / reject",
    length_m=3_000, grant_rate=0.34, discount_rate=0.035,
    connection_charge_per_kW=600,
)
EXTENDED_ROUTE_REJECT["description"] = (
    "Same heat sales spread over 3.0 km, 4.7 MWh/m. Included to show the route "
    "length at which the otherwise identical network stops being attractive."
)

FEASIBILITY_SCENARIOS = [DENSE_CORE_PUBLIC, COMPACT_PRIVATE, EXTENDED_ROUTE_REJECT]


def scenario_copies():
    return deepcopy(FEASIBILITY_SCENARIOS)
