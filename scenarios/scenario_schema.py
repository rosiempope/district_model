"""UI-facing validation for JSON-compatible 2-pipe and 4-pipe scenarios."""
from __future__ import annotations
from copy import deepcopy
from profiles.demand_synthesis import BUILDING_TYPES

HEAT_SOURCE_TYPES = {"ashp", "gas_boiler", "electric_boiler", "data_centre", "booster_heat_pump", "efw_chp"}
COOLING_SOURCE_TYPES = {"air_cooled_chiller"}
NETWORK_MODES = {"none", "generic_length"}
CLIMATE_SCENARIOS = {"baseline", "2050_central", "2050_high"}
COUNTERFACTUALS = {"none", "individual_gas", "individual_gas_and_ac"}
DEFAULTS = {
    "climate_scenario": "baseline",
    "network": {"mode":"generic_length", "length_m":3000.0, "include_cooling":False,
                "heat_flow_temp_C":70.0, "heat_return_temp_C":40.0,
                "cool_flow_temp_C":6.0, "cool_return_temp_C":12.0},
    "cooling_sources": [],
    "economics": {"project_lifetime_years":25, "discount_rate":0.105,
                  "om_rate":0.01, "counterfactual":"individual_gas"},
}

def apply_defaults(scenario: dict) -> dict:
    cfg = deepcopy(scenario)
    for key, value in DEFAULTS.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {})
            for k, v in value.items():
                cfg[key].setdefault(k, v)
        else:
            cfg.setdefault(key, value)
    return cfg

def _validate_sources(items, allowed, path, errors, required=False):
    if required and (not isinstance(items, list) or not items):
        errors.append(f"{path}: add at least one source")
        return
    if not isinstance(items, list):
        errors.append(f"{path}: must be a list")
        return
    for i, source in enumerate(items):
        p = f"{path}[{i}]"
        if not isinstance(source, dict):
            errors.append(f"{p}: must be an object")
            continue
        if source.get("type") not in allowed:
            errors.append(f"{p}.type: choose one of {sorted(allowed)}")
        if not isinstance(source.get("preset"), str) or not source["preset"]:
            errors.append(f"{p}.preset: required text")
        if "capacity_MW" in source and (not isinstance(source["capacity_MW"], (int, float)) or source["capacity_MW"] <= 0):
            errors.append(f"{p}.capacity_MW: must be positive")

def validate_scenario(scenario: dict) -> list[str]:
    if not isinstance(scenario, dict):
        return ["scenario: must be an object"]
    cfg = apply_defaults(scenario)
    errors = []
    if not isinstance(cfg.get("name"), str) or not cfg["name"].strip():
        errors.append("name: required text")
    if cfg.get("climate_scenario") not in CLIMATE_SCENARIOS:
        errors.append(f"climate_scenario: choose one of {sorted(CLIMATE_SCENARIOS)}")
    demand = cfg.get("demand", {})
    if not isinstance(demand.get("buildings"), list) or not demand["buildings"]:
        errors.append("demand.buildings: add at least one building")
    else:
        for i, b in enumerate(demand["buildings"]):
            p = f"demand.buildings[{i}]"
            if not isinstance(b, dict):
                errors.append(f"{p}: must be an object")
                continue
            if b.get("type") not in BUILDING_TYPES:
                errors.append(f"{p}.type: choose one of {sorted(BUILDING_TYPES)}")
            if not isinstance(b.get("name"), str) or not b["name"].strip():
                errors.append(f"{p}.name: required text")
            if not any(isinstance(b.get(k), (int, float)) and b[k] > 0 for k in ("floor_area_m2", "units")):
                errors.append(f"{p}: provide positive floor_area_m2 or units")
    network = cfg["network"]
    if network.get("mode") not in NETWORK_MODES:
        errors.append(f"network.mode: choose one of {sorted(NETWORK_MODES)}")
    if network.get("mode") == "generic_length" and (not isinstance(network.get("length_m"), (int, float)) or network["length_m"] <= 0):
        errors.append("network.length_m: must be positive for generic_length")
    _validate_sources(cfg.get("sources"), HEAT_SOURCE_TYPES, "sources", errors, required=True)
    has_dc = any(isinstance(s, dict) and s.get("type") == "data_centre" for s in cfg.get("sources", []))
    for i, s in enumerate(cfg.get("sources", [])):
        if isinstance(s, dict) and s.get("type") == "booster_heat_pump" and not has_dc:
            errors.append(f"sources[{i}]: requires a data_centre source")
    cooling = bool(network.get("include_cooling"))
    _validate_sources(cfg.get("cooling_sources"), COOLING_SOURCE_TYPES, "cooling_sources", errors, required=cooling)
    econ = cfg.get("economics", {})
    if not isinstance(econ.get("project_lifetime_years"), (int, float)) or econ["project_lifetime_years"] <= 0:
        errors.append("economics.project_lifetime_years: must be positive")
    if not isinstance(econ.get("discount_rate"), (int, float)) or econ["discount_rate"] < 0:
        errors.append("economics.discount_rate: must be zero or positive")
    if econ.get("counterfactual") not in COUNTERFACTUALS:
        errors.append(f"economics.counterfactual: choose one of {sorted(COUNTERFACTUALS)}")
    if cooling and econ.get("counterfactual") == "individual_gas":
        errors.append("economics.counterfactual: use 'individual_gas_and_ac' for a 4-pipe comparison")
    return errors

def validate_or_raise(scenario: dict) -> dict:
    cfg = apply_defaults(scenario)
    errors = validate_scenario(cfg)
    if errors:
        raise ValueError("Invalid scenario:\n- " + "\n- ".join(errors))
    return cfg
