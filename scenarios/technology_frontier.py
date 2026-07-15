"""Comparable technology-mix scenarios for route/demand frontier screening."""
from __future__ import annotations

from copy import deepcopy

from scenarios.worked_scenarios import GAS_ONLY


BASE_PEAK_HEAT_MW = 8.7
BASE_PEAK_COOLING_MW = 4.11

MIX_NAMES = [
    "Gas boiler reference",
    "Electric boiler",
    "ASHP only",
    "ASHP + gas backup",
    "ASHP + electric backup",
    "EfW + ASHP + gas backup",
    "Data-centre heat + booster + gas backup",
    "Four-pipe ASHP + gas + chiller",
]


def _sources(mix: str, peak_MW: float):
    ashp = {"type": "ashp", "preset": "ealing_phase1", "name": "ASHP bank",
            "capacity_MW": peak_MW * 0.45, "n_units": 6, "flow_temp_C": 70.0,
            "electricity_price_GBP_per_MWh": 180.5}
    gas = {"type": "gas_boiler", "preset": "ealing_phase2", "name": "Gas backup",
           "capacity_MW": peak_MW * 1.05, "gas_price_GBP_per_MWh": 46.9}
    electric = {"type": "electric_boiler", "preset": "ealing_backup", "name": "Electric backup",
                "capacity_MW": peak_MW * 1.05, "electricity_price_GBP_per_MWh": 180.5}
    if mix == "Gas boiler reference":
        return [{**gas, "capacity_MW": peak_MW * 1.20}]
    if mix == "Electric boiler":
        return [{**electric, "capacity_MW": peak_MW * 1.20}]
    if mix == "ASHP only":
        return [{**ashp, "capacity_MW": peak_MW * 1.50, "n_units": 8}]
    if mix in {"ASHP + gas backup", "Four-pipe ASHP + gas + chiller"}:
        return [ashp, gas]
    if mix == "ASHP + electric backup":
        return [ashp, electric]
    if mix == "EfW + ASHP + gas backup":
        return [
            {"type": "efw_chp", "preset": "newlincs_style", "name": "EfW heat export",
             "capacity_MW": peak_MW * 0.35},
            {**ashp, "capacity_MW": peak_MW * 0.25, "n_units": 4},
            gas,
        ]
    if mix == "Data-centre heat + booster + gas backup":
        return [
            {"type": "data_centre", "preset": "redwire_ealing", "name": "Recovered waste heat",
             "capacity_MW": peak_MW * 0.25, "supply_temp_C": 40.0,
             "availability_factor": 0.97, "waste_heat_cost_GBP_per_MWh": 0.0},
            {"type": "booster_heat_pump", "preset": "generic_5MW", "name": "Booster heat pump",
             "capacity_MW": peak_MW * 0.40, "n_units": 4, "depends_on": 0,
             "electricity_price_GBP_per_MWh": 180.5},
            gas,
        ]
    raise ValueError(f"Unknown technology mix: {mix}")


def frontier_scenario(
    mix: str,
    demand_scale: float,
    route_km: float,
    *,
    support_case: bool = True,
) -> dict:
    """Return one JSON-compatible scenario on a common assumption basis."""
    if mix not in MIX_NAMES:
        raise ValueError(f"mix must be one of {MIX_NAMES}")
    scenario = deepcopy(GAS_ONLY)
    scenario["name"] = f"{mix} | demand x{demand_scale:g} | route {route_km:g} km"
    scenario["description"] = (
        "Common-customer technology frontier with gas/AC customer-bill parity. "
        "High-support boundary case uses 49% eligible grant and GBP1,000/kW connection contribution."
    )
    for building in scenario["demand"]["buildings"]:
        if building.get("floor_area_m2"):
            building["floor_area_m2"] = float(building["floor_area_m2"]) * demand_scale
        if building.get("units"):
            building["units"] = max(1, round(float(building["units"]) * demand_scale))
            building["connections"] = int(building["units"])
        building["connection_year"] = 1
        building["connection_probability"] = 1.0
        building.pop("heat_unit_rate_p_per_kWh", None)
        building.pop("standing_charge_GBP_per_connection_year", None)
        if support_case:
            building["connection_charge_GBP_per_kW"] = 1_000.0
        else:
            building.pop("connection_charge_GBP_per_kW", None)

    include_cooling = mix == "Four-pipe ASHP + gas + chiller"
    scenario["network"].update({
        "mode": "generic_length", "length_m": float(route_km) * 1000.0,
        "include_cooling": include_cooling,
        "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
        "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0,
    })
    peak = BASE_PEAK_HEAT_MW * float(demand_scale)
    scenario["sources"] = _sources(mix, peak)
    scenario["cooling_sources"] = ([{
        "type": "air_cooled_chiller", "preset": "generic_2MW_bank",
        "name": "Central chiller bank",
        "capacity_MW": BASE_PEAK_COOLING_MW * float(demand_scale) * 1.20,
        "n_units": 6, "chilled_water_temp_C": 6.0,
        "electricity_price_GBP_per_MWh": 180.5,
    }] if include_cooling else [])
    scenario["thermal_storage"] = {"enabled": False}

    economics = scenario["economics"]
    economics["discount_rate"] = 0.105
    economics["social_discount_rate"] = 0.035
    economics["counterfactual"] = "individual_gas_and_ac" if include_cooling else "individual_gas"
    economics["parasitic_electricity_price_GBP_per_MWh"] = 180.5
    economics["tariffs"]["heat_tariff_mode"] = "counterfactual_bill_parity"
    economics["tariffs"]["cooling_tariff_mode"] = "counterfactual_bill_parity"
    economics["ghnf_grant"] = {"enabled": support_case, "rate": 0.49}
    capex = economics["capex_items"]
    has_gas = any(source["type"] == "gas_boiler" for source in scenario["sources"])
    has_electric_plant = any(
        source["type"] in {"ashp", "electric_boiler", "booster_heat_pump"}
        for source in scenario["sources"]
    ) or include_cooling
    capex["gas_connection_GBP"] = 250_000.0 if has_gas else 0.0
    capex["electricity_connection_GBP"] = 1_500_000.0 if has_electric_plant else 500_000.0
    scenario.setdefault("screening", {}).update({
        "maximum_unmet_energy_fraction": 0.001,
        "maximum_carbon_gCO2e_per_kWh": 100.0,
        "investor_hurdle_rate": 0.105,
        "minimum_investor_npv_GBP": 0.0,
        "require_n_minus_one": False,
    })
    return scenario
