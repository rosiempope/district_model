"""UI-facing validation for JSON-compatible 2-pipe and 4-pipe scenarios."""
from __future__ import annotations
from copy import deepcopy
from economics.connection_costs import SENSITIVITY_CASES
from network.design_temperature_limits import DHW_SYSTEM_TYPES
from network.route_difficulty import ROUTE_DIFFICULTY
from profiles.demand_synthesis import BUILDING_TYPES

HEAT_SOURCE_TYPES = {"ashp", "gas_boiler", "electric_boiler", "data_centre",
                     "booster_heat_pump", "efw_chp", "wshp", "gshp"}
COOLING_SOURCE_TYPES = {"air_cooled_chiller"}
NETWORK_MODES = {"none", "generic_length", "tree"}
TREE_ROOT_ID = "EC"   # the implicit energy-centre root every tree hangs off
CLIMATE_SCENARIOS = {"baseline", "2050_central", "2050_high"}
# What each building would do WITHOUT the network. This choice decides the
# answer: gas boilers are the cheap alternative policy is removing, individual
# heat pumps are the alternative that is actually legal long-term and the one
# heat network zoning is explicitly judged against ("the lowest-cost solution
# for decarbonising heating"). See economics/metrics.py.
COUNTERFACTUALS = {"none", "individual_gas", "individual_gas_and_ac",
                   "individual_ashp", "individual_ashp_and_ac"}
DEFAULTS = {
    "climate_scenario": "baseline",
    "screening": {
        "maximum_unmet_energy_fraction": 0.001,
        "maximum_carbon_gCO2e_per_kWh": 100.0,
        "investor_hurdle_rate": None,
        "minimum_investor_npv_GBP": 0.0,
        "require_n_minus_one": False,
        "maximum_required_heat_tariff_p_per_kWh": None,
    },
    "network": {"mode":"generic_length", "length_m":3000.0, "include_cooling":False,
                "heat_flow_temp_C":70.0, "heat_return_temp_C":40.0,
                "cool_flow_temp_C":6.0, "cool_return_temp_C":12.0,
                # How customers make domestic hot water. This is NOT a detail: it
                # sets the delivered-temperature floor, which sets the achievable
                # flow temperature, which sets heat-pump COP. An instantaneous HIU
                # is an HSE 'low risk' system needing 50C at the outlet; a stored
                # cylinder needs 60C stored plus a daily disinfection cycle. See
                # network/design_temperature_limits.py for the citations.
                "dhw_system":"instantaneous_hiu",
                # What the ground is like. The pipe curve alone treats a
                # six-lane junction and a greenfield site as the same
                # purchase; the DESNZ Birmingham report shows they differ by
                # 2.5x. See network/route_difficulty.py.
                "route_type":"suburban",
                "segments":[]},
    "cooling_sources": [],
    "thermal_storage": {"enabled": False},
    "economics": {
        "project_lifetime_years": 40,
        "discount_rate": 0.105,
        "social_discount_rate": 0.035,
        "financial_basis": "real",
        "base_year": 2026,
        "price_year": 2026,
        "om_rate": 0.01,
        "counterfactual": "individual_gas",
        "price_changes": {
            "electricity_real_rate": 0.0,
            "gas_real_rate": 0.0,
            "third_party_heat_real_rate": 0.0,
            "heat_tariff_real_rate": 0.0,
            "cooling_tariff_real_rate": 0.0,
            "other_opex_real_rate": 0.0,
        },
        "tariffs": {
            "heat_tariff_mode": "counterfactual_bill_parity",
            "cooling_tariff_mode": "counterfactual_bill_parity",
            "heat_unit_rate_p_per_kWh": 7.33,
            "cooling_unit_rate_p_per_kWh": 0.0,
            "standing_charge_GBP_per_connection_year": 106.0,
        },
        "capex_items": {
            "energy_centre_building_GBP": 0.0,
            "land_and_enabling_GBP": 0.0,
            "electricity_connection_GBP": 0.0,
            "gas_connection_GBP": 0.0,
            "controls_and_scada_GBP": 0.0,
            # How a connection is priced. "by_building_type" builds it up from
            # DECC components — per dwelling for residential, per kW of substation
            # for everything else. "flat_per_connection" is the old behaviour and
            # uses the two rates below; it charged the same figure to a flat and to
            # a railway station.
            "connection_cost_mode": "by_building_type",
            "connection_cost_case": "base",
            "customer_connection_GBP_per_connection": 0.0,
            "metering_GBP_per_connection": 0.0,
            "development_and_design_pct": 0.0,
            "commissioning_pct": 0.0,
            "contingency_pct": 0.0,
        },
        "annual_opex_items": {
            "billing_and_customer_service_GBP": 0.0,
            "insurance_and_rates_GBP": 0.0,
            "land_lease_GBP": 0.0,
            "water_treatment_GBP": 0.0,
            "operator_overhead_GBP": 0.0,
        },
        "replacement_overrides": {},
    },
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
    if cfg["screening"].get("investor_hurdle_rate") is None:
        cfg["screening"]["investor_hurdle_rate"] = cfg["economics"]["discount_rate"]
    return cfg

def _validate_sources(items, allowed, path, errors, required=False):
    if required and (not isinstance(items, list) or not items):
        errors.append(f"{path}: add at least one source")
        return
    if not isinstance(items, list):
        errors.append(f"{path}: must be a list")
        return
    names = set()
    for i, source in enumerate(items):
        p = f"{path}[{i}]"
        if not isinstance(source, dict):
            errors.append(f"{p}: must be an object")
            continue
        if source.get("type") not in allowed:
            errors.append(f"{p}.type: choose one of {sorted(allowed)}")
        if not isinstance(source.get("preset"), str) or not source["preset"]:
            errors.append(f"{p}.preset: required text")
        name = source.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{p}.name: required text")
        elif name in names:
            errors.append(f"{p}.name: '{name}' is duplicated; source names must be unique")
        else:
            names.add(name)
        if "capacity_MW" in source and (not isinstance(source["capacity_MW"], (int, float)) or source["capacity_MW"] <= 0):
            errors.append(f"{p}.capacity_MW: must be positive")
        if "n_units" in source and (
            not isinstance(source["n_units"], int) or isinstance(source["n_units"], bool)
            or source["n_units"] <= 0
        ):
            errors.append(f"{p}.n_units: must be a positive whole number")

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
            has_scale = any(isinstance(b.get(k), (int, float)) and b[k] > 0 for k in ("floor_area_m2", "units"))
            has_measured = (
                isinstance(b.get("annual_heat_kWh"), (int, float))
                and b["annual_heat_kWh"] >= 0
            )
            if not has_scale and not has_measured:
                errors.append(f"{p}: provide positive floor_area_m2/units or measured annual_heat_kWh")
            if "peak_total_heat_kW" in b and (
                not isinstance(b["peak_total_heat_kW"], (int, float))
                or b["peak_total_heat_kW"] <= 0
            ):
                errors.append(f"{p}.peak_total_heat_kW: must be positive")
            if "connections" in b and (
                not isinstance(b["connections"], int) or isinstance(b["connections"], bool)
                or b["connections"] < 0
            ):
                errors.append(f"{p}.connections: must be a non-negative whole number")
            if "connection_year" in b and (
                not isinstance(b["connection_year"], int) or b["connection_year"] < 1
            ):
                errors.append(f"{p}.connection_year: must be a positive whole-number year")
            probability = b.get("connection_probability", 1.0)
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                errors.append(f"{p}.connection_probability: must be between 0 and 1")
            if "bus_eligible" in b and not isinstance(b["bus_eligible"], bool):
                errors.append(
                    f"{p}.bus_eligible: must be true or false — marks the building "
                    "ineligible for the Boiler Upgrade Scheme (social housing and "
                    "most new-build homes are excluded)"
                )
    network = cfg["network"]
    if network.get("mode") not in NETWORK_MODES:
        errors.append(f"network.mode: choose one of {sorted(NETWORK_MODES)}")
    if network.get("mode") == "generic_length" and (not isinstance(network.get("length_m"), (int, float)) or network["length_m"] <= 0):
        errors.append("network.length_m: must be positive for generic_length")
    if network.get("mode") == "tree":
        _validate_tree_segments(network.get("segments"), demand.get("buildings", []), errors)

    if network.get("route_type", "suburban") not in ROUTE_DIFFICULTY:
        errors.append(
            f"network.route_type: choose one of {sorted(ROUTE_DIFFICULTY)} — a "
            "city-centre route costs about 2.5x a greenfield one for the same pipe"
        )
    if network.get("dhw_system") not in DHW_SYSTEM_TYPES:
        errors.append(
            f"network.dhw_system: choose one of {sorted(DHW_SYSTEM_TYPES)} — this sets the "
            "delivered-temperature floor (instantaneous HIUs are HSE 'low risk' and need 50°C "
            "at the outlet; a stored cylinder needs 60°C stored)"
        )

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
                if depends_on in boosted_dc_positions:
                    errors.append(
                        f"sources[{i}].depends_on: data-centre source {depends_on} is already allocated "
                        "to another booster; add an explicit source split instead of double-counting heat"
                    )
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
    storage = cfg.get("thermal_storage", {})
    if storage.get("enabled"):
        capacity = storage.get("capacity_MWh")
        volume = storage.get("volume_litres")
        if not ((isinstance(capacity, (int, float)) and capacity > 0)
                or (isinstance(volume, (int, float)) and volume > 0)):
            errors.append("thermal_storage: provide positive capacity_MWh or volume_litres")
        for key in ("max_charge_MW", "max_discharge_MW"):
            if key in storage and (not isinstance(storage[key], (int, float)) or storage[key] <= 0):
                errors.append(f"thermal_storage.{key}: must be positive")
    econ = cfg.get("economics", {})
    if (not isinstance(econ.get("project_lifetime_years"), int)
            or isinstance(econ.get("project_lifetime_years"), bool)
            or econ["project_lifetime_years"] <= 0):
        errors.append("economics.project_lifetime_years: must be a positive whole number")
    if not isinstance(econ.get("discount_rate"), (int, float)) or econ["discount_rate"] < 0:
        errors.append("economics.discount_rate: must be zero or positive")
    if not isinstance(econ.get("social_discount_rate"), (int, float)) or econ["social_discount_rate"] < 0:
        errors.append("economics.social_discount_rate: must be zero or positive")
    for year_name in ("base_year", "price_year"):
        if not isinstance(econ.get(year_name), int) or isinstance(econ.get(year_name), bool):
            errors.append(f"economics.{year_name}: must be a whole-number year")
    # NOTE the .get() defaults. apply_defaults() only merges ONE level deep —
    # DEFAULTS["economics"]["capex_items"] is two levels down, so a scenario that
    # supplies its own capex_items dict never receives these keys. Validating for
    # their presence would reject every existing scenario. The defaults here match
    # what _connection_capex() actually falls back to, so the two cannot drift.
    capex_items = econ.get("capex_items", {})
    if capex_items.get("connection_cost_mode", "by_building_type") not in {
        "by_building_type", "flat_per_connection"
    }:
        errors.append(
            "economics.capex_items.connection_cost_mode: choose 'by_building_type' "
            "(priced per dwelling / per kW) or 'flat_per_connection'"
        )
    if capex_items.get("connection_cost_case", "base") not in SENSITIVITY_CASES:
        errors.append(
            f"economics.capex_items.connection_cost_case: choose one of {sorted(SENSITIVITY_CASES)}"
        )
    if econ.get("counterfactual") not in COUNTERFACTUALS:
        errors.append(f"economics.counterfactual: choose one of {sorted(COUNTERFACTUALS)}")
    if econ.get("financial_basis") not in {"real", "nominal"}:
        errors.append("economics.financial_basis: choose 'real' or 'nominal'")
    if econ.get("financial_basis") == "nominal":
        errors.append("economics.financial_basis: nominal cash flows are not yet supported; use 'real'")
    for name, value in econ.get("price_changes", {}).items():
        if not isinstance(value, (int, float)) or value <= -1:
            errors.append(f"economics.price_changes.{name}: must be numeric and greater than -1")
    grant = econ.get("ghnf_grant", {})
    if grant.get("enabled") and (
        not isinstance(grant.get("rate", 0.40), (int, float))
        or not 0 <= grant.get("rate", 0.40) < 0.50
    ):
        errors.append("economics.ghnf_grant.rate: must be at least 0 and strictly below 0.50")
    if cooling and econ.get("counterfactual") in {"individual_gas", "individual_ashp"}:
        errors.append(
            "economics.counterfactual: a 4-pipe scenario needs a cooling counterfactual too — "
            "use 'individual_gas_and_ac' or 'individual_ashp_and_ac'"
        )
    tariff_mode = econ.get("tariffs", {}).get("heat_tariff_mode", "counterfactual_bill_parity")
    if tariff_mode not in {"counterfactual_bill_parity", "manual"}:
        errors.append("economics.tariffs.heat_tariff_mode: choose counterfactual_bill_parity or manual")
    if tariff_mode == "counterfactual_bill_parity" and econ.get("counterfactual") == "none":
        errors.append("economics.tariffs.heat_tariff_mode: gas-bill parity requires an individual-gas counterfactual")
    cooling_tariff_mode = econ.get("tariffs", {}).get("cooling_tariff_mode", "counterfactual_bill_parity")
    if cooling_tariff_mode not in {"counterfactual_bill_parity", "manual"}:
        errors.append("economics.tariffs.cooling_tariff_mode: choose counterfactual_bill_parity or manual")
    if cooling and cooling_tariff_mode == "counterfactual_bill_parity" and econ.get("counterfactual") not in {"individual_gas_and_ac", "individual_ashp_and_ac"}:
        errors.append("economics.tariffs.cooling_tariff_mode: cooling-bill parity requires a cooling counterfactual")
    screening = cfg.get("screening", {})
    if (not isinstance(screening.get("maximum_unmet_energy_fraction"), (int, float))
            or screening["maximum_unmet_energy_fraction"] < 0):
        errors.append("screening.maximum_unmet_energy_fraction: must be zero or positive")
    if not isinstance(screening.get("minimum_investor_npv_GBP"), (int, float)):
        errors.append("screening.minimum_investor_npv_GBP: must be numeric")
    if not isinstance(screening.get("maximum_carbon_gCO2e_per_kWh"), (int, float)) or screening["maximum_carbon_gCO2e_per_kWh"] < 0:
        errors.append("screening.maximum_carbon_gCO2e_per_kWh: must be zero or positive")
    if not isinstance(screening.get("investor_hurdle_rate"), (int, float)) or screening["investor_hurdle_rate"] < 0:
        errors.append("screening.investor_hurdle_rate: must be zero or positive")
    max_tariff = screening.get("maximum_required_heat_tariff_p_per_kWh")
    if max_tariff is not None and (not isinstance(max_tariff, (int, float)) or max_tariff < 0):
        errors.append("screening.maximum_required_heat_tariff_p_per_kWh: must be blank or zero/positive")
    return errors

def validate_or_raise(scenario: dict) -> dict:
    cfg = apply_defaults(scenario)
    errors = validate_scenario(cfg)
    if errors:
        raise ValueError("Invalid scenario:\n- " + "\n- ".join(errors))
    return cfg
