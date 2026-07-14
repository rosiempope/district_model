"""UI-facing validation for JSON-compatible 2-pipe and 4-pipe scenarios."""
from __future__ import annotations
from copy import deepcopy
from profiles.demand_synthesis import BUILDING_TYPES

HEAT_SOURCE_TYPES = {"ashp", "gas_boiler", "electric_boiler", "data_centre", "booster_heat_pump", "efw_chp"}
COOLING_SOURCE_TYPES = {"air_cooled_chiller"}
NETWORK_MODES = {"none", "generic_length", "tree"}
TREE_ROOT_ID = "EC"   # the implicit energy-centre root every tree hangs off
CLIMATE_SCENARIOS = {"baseline", "2050_central", "2050_high"}
COUNTERFACTUALS = {"none", "individual_gas", "individual_gas_and_ac"}
DEFAULTS = {
    "climate_scenario": "baseline",
    "network": {"mode":"generic_length", "length_m":3000.0, "include_cooling":False,
                "heat_flow_temp_C":70.0, "heat_return_temp_C":40.0,
                "cool_flow_temp_C":6.0, "cool_return_temp_C":12.0,
                "segments":[]},
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
                # deepcopy the default: inserting a shared mutable (list/
                # dict) straight from DEFAULTS would let later scenario
                # edits silently mutate DEFAULTS itself for every
                # subsequent scenario in the same session
                cfg[key].setdefault(k, deepcopy(v))
        else:
            cfg.setdefault(key, deepcopy(value))
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

def _validate_tree_segments(segments, buildings, errors) -> None:
    """
    Validate a tree-mode segment list BEFORE the runner tries to build a
    NetworkTopology from it, so problems surface as readable messages in
    the UI rather than tracebacks. Conventions:
      - the energy centre is the implicit root, id TREE_ROOT_ID ("EC")
      - every segment: {node_id, parent_id, length_m, building (optional)}
      - every building in demand.buildings must be served by exactly one
        segment (all demand is dispatched through the network, so a
        building with no route to the energy centre is inconsistent)
    """
    if not isinstance(segments, list) or not segments:
        errors.append("network.segments: add at least one pipe segment for the tree layout")
        return
    building_names = {b.get("name") for b in buildings if isinstance(b, dict)}
    seen_ids, seen_buildings = set(), {}
    for i, seg in enumerate(segments):
        p = f"network.segments[{i}]"
        if not isinstance(seg, dict):
            errors.append(f"{p}: must be an object")
            continue
        nid = seg.get("node_id")
        if not isinstance(nid, str) or not nid.strip():
            errors.append(f"{p}.node_id: required text (a unique segment ID, e.g. 'J1' or 'B2')")
        elif nid == TREE_ROOT_ID:
            errors.append(f"{p}.node_id: '{TREE_ROOT_ID}' is reserved for the energy centre root")
        elif nid in seen_ids:
            errors.append(f"{p}.node_id: '{nid}' appears more than once — segment IDs must be unique")
        else:
            seen_ids.add(nid)
        if not isinstance(seg.get("parent_id"), str) or not seg["parent_id"].strip():
            errors.append(f"{p}.parent_id: required — use '{TREE_ROOT_ID}' to connect directly to the energy centre")
        if not isinstance(seg.get("length_m"), (int, float)) or seg["length_m"] <= 0:
            errors.append(f"{p}.length_m: must be a positive length in metres")
        b = seg.get("building")
        if b not in (None, ""):
            if b not in building_names:
                errors.append(f"{p}.building: '{b}' is not one of the buildings in section 2 "
                              f"(available: {sorted(n for n in building_names if n)})")
            elif b in seen_buildings:
                errors.append(f"{p}.building: '{b}' is already served by segment "
                              f"'{seen_buildings[b]}' — each building connects once")
            else:
                seen_buildings[b] = seg.get("node_id")
    # parents must resolve to the root or another segment
    for i, seg in enumerate(segments):
        if isinstance(seg, dict):
            pid = seg.get("parent_id")
            if isinstance(pid, str) and pid.strip() and pid != TREE_ROOT_ID and pid not in seen_ids:
                errors.append(f"network.segments[{i}].parent_id: '{pid}' doesn't match any "
                              f"segment ID or '{TREE_ROOT_ID}'")
    unserved = building_names - set(seen_buildings)
    if not errors and unserved:
        errors.append(f"network.segments: these buildings have no connecting segment: "
                      f"{sorted(n for n in unserved if n)} — every building needs a route to the energy centre")


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
    if network.get("mode") == "tree":
        _validate_tree_segments(network.get("segments"), demand.get("buildings", []), errors)

    # Design temperatures must give a usable delta-T for pipe sizing —
    # without this, an equal (or inverted) flow/return pair reaches
    # pipe_catalog.size_pipe_for_peak() and surfaces as a raw traceback
    # in the UI instead of a readable validation message.
    hf, hr = network.get("heat_flow_temp_C"), network.get("heat_return_temp_C")
    if isinstance(hf, (int, float)) and isinstance(hr, (int, float)) and hf - hr < 5.0:
        errors.append("network: heat flow temperature must be at least 5°C above the return "
                      f"temperature (got flow {hf}°C / return {hr}°C)")
    if network.get("include_cooling"):
        cf_, cr = network.get("cool_flow_temp_C"), network.get("cool_return_temp_C")
        if isinstance(cf_, (int, float)) and isinstance(cr, (int, float)) and cr - cf_ < 2.0:
            errors.append("network: cooling return temperature must be at least 2°C above the "
                          f"flow temperature (got flow {cf_}°C / return {cr}°C)")
    _validate_sources(cfg.get("sources"), HEAT_SOURCE_TYPES, "sources", errors, required=True)
    src_list = cfg.get("sources", [])
    dc_positions = {i for i, s in enumerate(src_list) if isinstance(s, dict) and s.get("type") == "data_centre"}
    has_dc = bool(dc_positions)
    boosted_dc_positions = set()
    for i, s in enumerate(src_list):
        if isinstance(s, dict) and s.get("type") == "booster_heat_pump":
            if not has_dc:
                errors.append(f"sources[{i}]: requires a data_centre source")
                continue
            depends_on = s.get("depends_on", 0)
            if not isinstance(depends_on, int) or depends_on not in dc_positions:
                errors.append(
                    f"sources[{i}].depends_on: must be the position (0-based, counting "
                    f"ALL sources in this list) of a data_centre source; valid positions "
                    f"here are {sorted(dc_positions)}, got {depends_on!r}"
                )
            else:
                boosted_dc_positions.add(depends_on)

    # PHYSICS CHECK: data-centre waste heat is recovered at ~28-35°C —
    # well below a typical 70°C LTHW network flow temperature. Dispatching
    # it straight into network duty ("dispatch_direct") without a booster
    # heat pump to lift it is only physically valid if the network itself
    # runs cool enough to accept it directly (a genuine low-temperature/
    # ambient-loop-style scheme, not the fixed-direction 2-pipe/4-pipe
    # networks this model builds). 35°C is a generous upper bound for
    # that case. Otherwise every data_centre source must have a booster.
    DIRECT_DISPATCH_MAX_FLOW_TEMP_C = 35.0
    network_flow_temp_C = network.get("heat_flow_temp_C")
    for i in sorted(dc_positions):
        s = src_list[i]
        dispatches_direct = bool(s.get("dispatch_direct", False))
        network_cool_enough = (
            isinstance(network_flow_temp_C, (int, float))
            and network_flow_temp_C <= DIRECT_DISPATCH_MAX_FLOW_TEMP_C
        )
        if i not in boosted_dc_positions and not (dispatches_direct and network_cool_enough):
            if i not in boosted_dc_positions and dispatches_direct and not network_cool_enough:
                errors.append(
                    f"sources[{i}] (data_centre): 'dispatch direct' is only physically valid when "
                    f"the network flow temperature is \u2264{DIRECT_DISPATCH_MAX_FLOW_TEMP_C:.0f}\u00b0C "
                    f"(this network runs at {network_flow_temp_C}\u00b0C) \u2014 data-centre waste heat "
                    f"needs a booster heat pump to reach that flow temperature. Add a booster source "
                    f"with depends_on={i}, or lower the network flow temperature if this is genuinely "
                    f"a low-temperature scheme."
                )
            elif i not in boosted_dc_positions:
                errors.append(
                    f"sources[{i}] (data_centre): needs a booster heat pump to lift its ~30\u00b0C waste "
                    f"heat up to the network's {network_flow_temp_C}\u00b0C flow temperature \u2014 add a "
                    f"booster source with depends_on={i}."
                )
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
