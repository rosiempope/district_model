"""UI-ready runner for 2-pipe heating and 4-pipe heating/cooling screening cases."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import synthesise_network
from optimisation.dispatch import run_dispatch
from network.network import size_network_from_demand
from economics.CAPEX import aggregate_capex
from economics.OPEX import annual_om_cost_GBP
from economics.metrics import (levelised_cost_of_heat_GBP_per_kWh, npv, irr, simple_payback_years,
                               discounted_payback_years, aggregate_counterfactual,
                               counterfactual_gas_boiler_dispatch, counterfactual_individual_ac_dispatch)
from components.ASHP import ASHPArray
from components.EfW import EfWChp
from components.datacentre_source import DataCentre
from components.booster_heat_pump import BoosterHeatPump
from components.peak_demand_option import GasBoiler, ElectricBoiler
from components.chiller import AirCooledChiller
from scenarios.scenario_schema import validate_or_raise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEATHER_CSV = ROOT / "profiles" / "weather_data.csv"
HEAT_CLASSES = {"ashp": ASHPArray, "efw_chp": EfWChp, "data_centre": DataCentre,
                "gas_boiler": GasBoiler, "electric_boiler": ElectricBoiler}
UNIT_TYPES = {"ashp", "booster_heat_pump", "air_cooled_chiller"}

def load_weather(weather=None):
    df = pd.read_csv(Path((weather or {}).get("csv_path", DEFAULT_WEATHER_CSV)))
    if len(df) != 8760:
        raise ValueError(f"Weather file must contain 8760 rows; got {len(df)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2021-01-01", periods=len(df), freq="h")
    return df

def _overrides(cfg):
    result = {k: v for k, v in cfg.items() if k not in {"type", "preset", "capacity_MW", "name", "depends_on", "dispatch_direct"}}
    result["name"] = cfg.get("name") or cfg["type"].replace("_", " ").title()
    if "capacity_MW" in cfg:
        if cfg["type"] in UNIT_TYPES:
            result.setdefault("n_units", 1)
            result.setdefault("unit_capacity_MW", cfg["capacity_MW"])
        elif cfg["type"] == "data_centre":
            result.setdefault("heat_offtake_MW", cfg["capacity_MW"])
        elif cfg["type"] == "efw_chp":
            result.setdefault("heat_capacity_MW", cfg["capacity_MW"])
        else:
            result.setdefault("capacity_MW", cfg["capacity_MW"])
    return result

def build_heat_sources(configs, weather, sink_temp_C):
    sources, data_centres, dc_cfgs, booster_cfgs = [], [], [], []
    for cfg in configs:
        if cfg["type"] == "booster_heat_pump":
            booster_cfgs.append(cfg)
            continue
        cls = HEAT_CLASSES[cfg["type"]]
        obj = (cls.from_preset(cfg["preset"], weather_df=weather, **_overrides(cfg))
               if cfg["type"] in {"ashp", "data_centre", "efw_chp"}
               else cls.from_preset(cfg["preset"], **_overrides(cfg)))
        if cfg["type"] == "data_centre":
            data_centres.append(obj); dc_cfgs.append(cfg)
        else:
            sources.append(obj)
    used_dc = {int(cfg.get("depends_on", 0)) for cfg in booster_cfgs}
    for i, dc in enumerate(data_centres):
        if dc_cfgs[i].get("dispatch_direct", False) or i not in used_dc:
            sources.append(dc)
    for cfg in booster_cfgs:
        dc = data_centres[int(cfg.get("depends_on", 0))]
        sources.append(BoosterHeatPump.from_preset(
            cfg["preset"], source_temp_C_hourly=dc.supply_temp_C,
            sink_temp_C=sink_temp_C, **_overrides(cfg)))
    return sources

def build_cooling_sources(configs, weather):
    return [AirCooledChiller.from_preset(c["preset"], weather_df=weather, **_overrides(c)) for c in configs]

def _distributed_network_load(base_kW, annual_MWh):
    if annual_MWh <= 0 or base_kW.sum() <= 0:
        return base_kW * 0.0
    return base_kW * (annual_MWh * 1000.0 / base_kW.sum())

def _combined_counterfactual(nodes, weather, include_cooling, om_rate):
    heat = aggregate_counterfactual(nodes, counterfactual_gas_boiler_dispatch, om_rate=om_rate)
    if not include_cooling:
        return heat
    cooling = aggregate_counterfactual(nodes, counterfactual_individual_ac_dispatch,
                                       weather_df=weather, om_rate=om_rate)
    return {"heating": heat, "cooling": cooling,
            "total_capex_GBP": round(heat["total_capex_GBP"] + cooling["total_capex_GBP"], 0),
            "total_annual_opex_GBP": round(heat["total_annual_opex_GBP"] + cooling["total_annual_opex_GBP"], 0)}

def run_scenario(scenario):
    cfg = validate_or_raise(scenario)
    weather = apply_climate_scenario(load_weather(cfg.get("weather")), cfg["climate_scenario"])
    demand = synthesise_network(weather, {"demand_nodes": cfg["demand"]["buildings"]})
    net_cfg = cfg["network"]
    include_cooling = bool(net_cfg["include_cooling"])

    network = None
    heat_loss_MWh = cool_gain_MWh = 0.0
    if net_cfg["mode"] == "generic_length":
        kwargs = {"heat_flow_temp_C": net_cfg["heat_flow_temp_C"], "heat_return_temp_C": net_cfg["heat_return_temp_C"],
                  "cool_flow_temp_C": net_cfg["cool_flow_temp_C"], "cool_return_temp_C": net_cfg["cool_return_temp_C"]} if include_cooling else {"flow_temp_C": net_cfg["heat_flow_temp_C"], "return_temp_C": net_cfg["heat_return_temp_C"]}
        network = size_network_from_demand(demand, net_cfg["length_m"], include_cooling=include_cooling, **kwargs)
        for duty in network.duties:
            if duty.duty_name == "heating": heat_loss_MWh = duty.annual_heat_loss_MWh
            if duty.duty_name == "cooling": cool_gain_MWh = abs(duty.annual_heat_loss_MWh)

    heat_sources = build_heat_sources(cfg["sources"], weather, net_cfg["heat_flow_temp_C"])
    heat_dispatch = run_dispatch(demand["total_heat_kW"] + _distributed_network_load(demand["total_heat_kW"], heat_loss_MWh), heat_sources, duty="heat")
    heat_summary = heat_dispatch.summary()

    cooling_sources, cooling_dispatch, cooling_summary = [], None, None
    if include_cooling:
        cooling_sources = build_cooling_sources(cfg["cooling_sources"], weather)
        cooling_dispatch = run_dispatch(demand["total_cooling_kW"] + _distributed_network_load(demand["total_cooling_kW"], cool_gain_MWh), cooling_sources, duty="cool")
        cooling_summary = cooling_dispatch.summary()

    all_sources = heat_sources + cooling_sources
    capex = aggregate_capex(sources=all_sources)
    network_capex = network.total_capex_GBP if network else 0.0
    total_capex = capex["grand_total_GBP"] + network_capex
    om = annual_om_cost_GBP(total_capex, cfg["economics"]["om_rate"])
    energy_cost = heat_summary["total_annual_opex_GBP"] + (cooling_summary["total_annual_opex_GBP"] if cooling_summary else 0.0)
    annual_opex = energy_cost + om

    counterfactual = None
    financial = {}
    if cfg["economics"]["counterfactual"] != "none":
        counterfactual = _combined_counterfactual(demand["nodes"], weather, include_cooling, cfg["economics"]["om_rate"])
        incremental_capex = total_capex - counterfactual["total_capex_GBP"]
        annual_saving = counterfactual["total_annual_opex_GBP"] - annual_opex
        life, rate = cfg["economics"]["project_lifetime_years"], cfg["economics"]["discount_rate"]
        financial = {"counterfactual": cfg["economics"]["counterfactual"],
                     "counterfactual_capex_GBP": counterfactual["total_capex_GBP"],
                     "counterfactual_annual_opex_GBP": counterfactual["total_annual_opex_GBP"],
                     "incremental_capex_GBP": round(incremental_capex, 0),
                     "annual_avoided_cost_GBP": round(annual_saving, 0),
                     "npv_vs_counterfactual_GBP": round(npv(incremental_capex, annual_saving, life, rate), 0),
                     "irr_vs_counterfactual": irr(incremental_capex, annual_saving, life),
                     "simple_payback_years": simple_payback_years(incremental_capex, annual_saving),
                     "discounted_payback_years": discounted_payback_years(incremental_capex, annual_saving, life, rate)}

    heat_delivered = demand["annual_heat_MWh"] + demand["annual_dhw_MWh"]
    cool_delivered = demand["annual_cool_MWh"] if include_cooling else 0.0
    total_delivered = heat_delivered + cool_delivered
    carbon_t = heat_summary["total_annual_carbon_tCO2"] + (cooling_summary["total_annual_carbon_tCO2"] if cooling_summary else 0.0)
    headline = {
        "system_type": "4_pipe_heating_cooling" if include_cooling else "2_pipe_heating",
        "annual_heat_demand_MWh": round(heat_delivered, 1), "annual_cooling_demand_MWh": round(cool_delivered, 1),
        "annual_heat_to_generate_MWh": heat_summary["annual_demand_MWh"],
        "annual_cooling_to_generate_MWh": cooling_summary["annual_demand_MWh"] if cooling_summary else 0.0,
        "peak_heat_MW": round(demand["peak_heat_kW"] / 1000, 3), "peak_cooling_MW": round(demand["peak_cool_kW"] / 1000, 3),
        "annual_network_heat_loss_MWh": round(heat_loss_MWh, 1), "annual_network_cooling_gain_MWh": round(cool_gain_MWh, 1),
        "annual_heat_by_source_MWh": heat_summary["annual_MWh_by_source"],
        "annual_cooling_by_source_MWh": cooling_summary["annual_MWh_by_source"] if cooling_summary else {},
        "annual_unmet_demand_MWh": heat_summary["annual_unmet_demand_MWh"], "peak_unmet_MW": heat_summary["peak_unmet_MW"],
        "annual_unmet_cooling_MWh": cooling_summary["annual_unmet_demand_MWh"] if cooling_summary else 0.0,
        "peak_unmet_cooling_MW": cooling_summary["peak_unmet_MW"] if cooling_summary else 0.0,
        "capex_total_GBP": round(total_capex, 0), "capex_sources_GBP": capex["by_category"]["sources_GBP"], "capex_network_GBP": round(network_capex, 0),
        "annual_energy_cost_GBP": round(energy_cost, 0), "annual_om_cost_GBP": round(om, 0), "annual_total_opex_GBP": round(annual_opex, 0),
        "lcoh_GBP_per_kWh": round(levelised_cost_of_heat_GBP_per_kWh(total_capex, annual_opex, heat_summary["annual_demand_MWh"] * 1000, cfg["economics"]["project_lifetime_years"]), 4),
        "levelised_energy_service_GBP_per_kWh": round(levelised_cost_of_heat_GBP_per_kWh(total_capex, annual_opex, total_delivered * 1000, cfg["economics"]["project_lifetime_years"]), 4),
        "annual_carbon_tCO2": round(carbon_t, 1), "carbon_intensity_kgCO2_per_kWh": round(carbon_t * 1000 / total_delivered, 4) if total_delivered else 0.0, "carbon_intensity_kgCO2_per_kWh_service": round(carbon_t * 1000 / total_delivered, 4) if total_delivered else 0.0,
    }
    return {"scenario_name": cfg["name"], "input": cfg, "headline": headline, "financial": financial,
            "counterfactual": counterfactual, "demand": demand, "weather": weather, "network": network,
            "heat_sources": heat_sources, "cooling_sources": cooling_sources,
            "heat_dispatch": heat_dispatch, "cooling_dispatch": cooling_dispatch, "capex": capex}

def comparison_table(results):
    cols = ["system_type", "annual_heat_demand_MWh", "annual_cooling_demand_MWh", "capex_total_GBP", "annual_total_opex_GBP", "levelised_energy_service_GBP_per_kWh", "annual_carbon_tCO2", "annual_unmet_demand_MWh", "annual_unmet_cooling_MWh"]
    return pd.DataFrame([{"scenario": r["scenario_name"], **{c: r["headline"][c] for c in cols}} for r in results])
