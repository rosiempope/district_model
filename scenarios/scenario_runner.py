"""UI-ready runner for 2-pipe heating and 4-pipe heating/cooling screening cases."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import synthesise_network, compute_climate_reference
from optimisation.dispatch import run_dispatch
from network.network import size_network_from_demand
from network.network_pumping import annual_pumping_electricity_MWh
from network.pipe_catalog import water_properties
from economics.CAPEX import aggregate_capex
from economics.metrics import (aggregate_counterfactual,
                               counterfactual_gas_boiler_dispatch,
                               counterfactual_individual_ac_dispatch,
                               counterfactual_individual_ashp_dispatch,
                               )


def _mass_flow_kg_s(load_kW, flow_temp_C, return_temp_C):
    props = water_properties(flow_temp_C)
    delta_T = abs(float(flow_temp_C) - float(return_temp_C))
    return np.asarray(load_kW, dtype=float) * 1000.0 / (props["cp_J_kgK"] * delta_T)


def _generic_pumping_hourly_MW(duty_result, load_kW, flow_temp_C, return_temp_C):
    """Hourly equivalent-trunk pumping load for generic-length screening."""
    props = water_properties(flow_temp_C)
    mass_flow = _mass_flow_kg_s(load_kW, flow_temp_C, return_temp_C)
    volumetric_flow = mass_flow / props["density_kg_m3"]
    round_trip_drop_Pa = (
        duty_result.pipe.pressure_gradient_Pa_per_m
        * duty_result.network_length_m * 2.0
    )
    return volumetric_flow * round_trip_drop_Pa / 0.75 / 1e6


def _generic_pumping_MWh(duty_result, load_kW, flow_temp_C, return_temp_C):
    return float(_generic_pumping_hourly_MW(
        duty_result, load_kW, flow_temp_C, return_temp_C
    ).sum())
from components.ASHP import ASHPArray, ASHP_PRESETS
from components.EfW import EfWChp, EFW_PRESETS
from components.datacentre_source import DataCentre, DC_PRESETS
from components.booster_heat_pump import BoosterHeatPump, BOOSTER_PRESETS
from components.peak_demand_option import GasBoiler, ElectricBoiler, CARBON_INTENSITY
from components.chiller import AirCooledChiller, CHILLER_PRESETS
from components.water_ground_source_hp import (
    GroundSourceHeatPump, GSHP_PRESETS, WaterSourceHeatPump, WSHP_PRESETS,
)
from components.thermal_storage import ThermalStorage, m3_to_mwh
from economics.tariffs import resolve_electricity_price
from economics.cashflow import (
    build_cashflow, discounted_levelised_cost_GBP_per_kWh, discount_factors,
)
from scenarios.scenario_schema import validate_or_raise
from scenarios.screening import evaluate_screening

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "2.7.1-streamlit-matrix-verified"
DEFAULT_WEATHER_CSV = ROOT / "profiles" / "weather_data.csv"
HEAT_CLASSES = {"ashp": ASHPArray, "efw_chp": EfWChp, "data_centre": DataCentre,
                "gas_boiler": GasBoiler, "electric_boiler": ElectricBoiler,
                "wshp": WaterSourceHeatPump, "gshp": GroundSourceHeatPump}
UNIT_TYPES = {"ashp", "booster_heat_pump", "air_cooled_chiller", "wshp", "gshp"}
PRESETS_BY_TYPE = {
    "ashp": ASHP_PRESETS,
    "booster_heat_pump": BOOSTER_PRESETS,
    "air_cooled_chiller": CHILLER_PRESETS,
    "data_centre": DC_PRESETS,
    "efw_chp": EFW_PRESETS,
    "wshp": WSHP_PRESETS,
    "gshp": GSHP_PRESETS,
}

ELECTRIC_SOURCE_TYPES = {
    "ashp", "booster_heat_pump", "air_cooled_chiller", "electric_boiler",
    "wshp", "gshp",
}

# CAPEX lines that the development/design, commissioning and contingency
# percentages are NOT applied to. Land and enabling is a transaction, not a
# designed and constructed scope — a design fee and a build contingency on top
# of a land price would be double-counting. Everything else in the delivered
# scope does carry the adders; see the note at adder_base_capex in
# run_scenario() for why this is not just plant and network.
EXCLUDED_FROM_CAPEX_ADDERS = {"land_and_enabling_GBP"}
REPLACEMENT_DEFAULTS = {
    "ashp": (15, 0.60),
    "wshp": (15, 0.60),
    "gshp": (20, 0.50),   # borehole field outlives the plant
    "booster_heat_pump": (15, 0.60),
    "air_cooled_chiller": (15, 0.60),
    "gas_boiler": (20, 0.50),
    "electric_boiler": (20, 0.50),
    "data_centre": (25, 0.50),
    "efw_chp": (25, 0.50),
}


def _n_minus_one_at_peak(sources, demand_MW):
    """Screen peak capacity after the largest credible source/unit outage.

    Arrays (ASHP/chiller) lose one unit. Single assets and booster heat pumps
    lose their full peak-hour output; treating a booster train as one
    contingency also captures loss of its common upstream waste-heat source.
    This is a capacity screen, not a dynamic resilience simulation.
    """
    load = np.asarray(demand_MW, dtype=float)
    if not len(load):
        return {"available_MW": 0.0, "firm_capacity_MW": 0.0,
                "margin_MW": 0.0, "compliant": True}
    peak_i = int(np.argmax(load))
    available = []
    outage_blocks = []
    for source in sources:
        supply = np.asarray(source.supply_MW, dtype=float)
        at_peak = max(0.0, float(supply[peak_i]))
        available.append(at_peak)
        n_units = max(1, int(getattr(source, "n_units", 1)))
        if getattr(source, "source_type", "") == "booster_heat_pump":
            outage_blocks.append(at_peak)
        else:
            outage_blocks.append(at_peak / n_units)
    total = sum(available)
    firm = max(0.0, total - max(outage_blocks, default=0.0))
    peak = float(load[peak_i])
    return {
        "available_MW": total,
        "firm_capacity_MW": firm,
        "margin_MW": firm - peak,
        "compliant": firm + 1e-9 >= peak,
    }


def _dispatch_costs_by_carrier(dispatch_result):
    values = {"electricity": 0.0, "gas": 0.0, "third_party_heat": 0.0}
    if dispatch_result is None:
        return values
    for source in dispatch_result.sources:
        dispatched = dispatch_result.dispatch_by_source_MW[source.name]
        cost = float((dispatched * source.marginal_cost).sum())
        if source.source_type in ELECTRIC_SOURCE_TYPES:
            values["electricity"] += cost
        elif source.source_type == "gas_boiler":
            values["gas"] += cost
        else:
            values["third_party_heat"] += cost
    return values


def _dispatch_carbon_tCO2(dispatch_result, carbon_factors):
    if dispatch_result is None:
        return 0.0
    total_kg = 0.0
    for source in dispatch_result.sources:
        output_MWh = dispatch_result.dispatch_by_source_MW[source.name]
        if source.source_type in ELECTRIC_SOURCE_TYPES and carbon_factors.get("electricity_kgCO2_per_kWh") is not None:
            factor = float(carbon_factors["electricity_kgCO2_per_kWh"])
            if hasattr(source, "cop_hourly"):
                intensity = factor / source.cop_hourly
            elif hasattr(source, "efficiency_hourly"):
                intensity = factor / source.efficiency_hourly
            else:
                intensity = factor
        elif source.source_type == "gas_boiler" and carbon_factors.get("gas_kgCO2_per_kWh") is not None:
            intensity = float(carbon_factors["gas_kgCO2_per_kWh"]) / source.efficiency_hourly
        else:
            intensity = source.carbon_intensity_kgCO2_per_kWh
        total_kg += float((output_MWh * intensity * 1000.0).sum())
    return total_kg / 1000.0


def _connection_count(building):
    explicit = building.get("connections")
    if explicit is not None:
        return max(0, int(explicit))
    if building.get("type") in {"residential", "residential_existing"} and building.get("units"):
        return max(1, int(building["units"]))
    return 1


def _replacement_series(assets, life, overrides):
    result = {}
    for asset in assets:
        asset_capex = float(asset.capacity_MW * asset.capex_GBP_per_MW)
        default_life, default_fraction = REPLACEMENT_DEFAULTS.get(asset.source_type, (25, 0.50))
        cfg = overrides.get(asset.source_type, {})
        interval = int(cfg.get("interval_years", default_life))
        fraction = float(cfg.get("capex_fraction", default_fraction))
        arr = np.zeros(life + 1)
        if interval > 0:
            for year in range(interval, life + 1, interval):
                arr[year] = asset_capex * fraction
        result[asset.name] = arr
    return result


def _build_thermal_storage(config, flow_temp_C, return_temp_C):
    if not config or not config.get("enabled", False):
        return None
    delta_t = float(config.get("delta_T_K", abs(flow_temp_C - return_temp_C)))
    capacity = config.get("capacity_MWh")
    if capacity is None:
        volume_litres = float(config.get("volume_litres", 0.0))
        capacity = m3_to_mwh(volume_litres / 1000.0, delta_t)
    capacity = float(capacity)
    if capacity <= 0:
        raise ValueError("thermal_storage capacity must be positive when enabled")
    return ThermalStorage(
        name=config.get("name", "Thermal store"),
        capacity_MWh=capacity,
        max_charge_MW=float(config.get("max_charge_MW", capacity)),
        max_discharge_MW=float(config.get("max_discharge_MW", capacity)),
        round_trip_efficiency=float(config.get("round_trip_efficiency", 0.95)),
        standing_loss_pct_per_hour=float(config.get("standing_loss_pct_per_hour", 0.0008)),
        initial_soc_fraction=float(config.get("initial_soc_fraction", 0.5)),
        delta_T_K=delta_t,
        capex_GBP=config.get("capex_GBP"),
        dispatch_strategy=config.get("dispatch_strategy", "displace_boiler"),
    )


def _escalated_series(base, life, rate=0.0, start_year=1, end_year=None):
    """Return an explicit years 0..life real cash-flow series."""
    arr = np.zeros(int(life) + 1)
    end = int(life if end_year is None else min(end_year, life))
    for year in range(max(1, int(start_year)), end + 1):
        arr[year] = float(base) * (1.0 + float(rate)) ** (year - int(start_year))
    return arr


def _customer_revenue_and_energy(
    buildings, nodes, economics, include_cooling, life, counterfactual=None
):
    """Build customer-specific tariff revenue and connected-energy series.

    Each building can override connection timing, connection probability and
    heat/cooling/fixed tariffs.  This avoids charging cooling at a domestic gas
    tariff and makes phased build-out visible in the cash-flow table.
    """
    tariffs = economics.get("tariffs", {})
    tariff_mode = tariffs.get("heat_tariff_mode", "counterfactual_bill_parity")
    cooling_tariff_mode = tariffs.get("cooling_tariff_mode", "counterfactual_bill_parity")
    changes = economics.get("price_changes", {})
    revenue, heat_energy, cooling_energy = {}, {}, {}
    year1_heat_bill = 0.0
    year1_counterfactual_bill = 0.0
    year1_cooling_bill = 0.0
    year1_counterfactual_cooling_bill = 0.0
    heat_counterfactual = (
        counterfactual.get("heating", counterfactual) if counterfactual else None
    )
    cooling_counterfactual = counterfactual.get("cooling") if counterfactual else None
    for building, node in zip(buildings, nodes):
        name = str(building.get("name", node["name"]))
        start = max(1, int(building.get("connection_year", 1)))
        probability = min(1.0, max(0.0, float(building.get("connection_probability", 1.0))))
        connections = _connection_count(building)
        heat_kWh = (float(node["annual_heat_kWh"]) + float(node["annual_dhw_kWh"])) * probability
        cool_kWh = float(node["annual_cool_kWh"]) * probability if include_cooling else 0.0
        heat_rate = float(building.get(
            "heat_unit_rate_p_per_kWh", tariffs.get("heat_unit_rate_p_per_kWh", 0.0)
        )) / 100.0
        cool_rate = float(building.get(
            "cooling_unit_rate_p_per_kWh", tariffs.get("cooling_unit_rate_p_per_kWh", 0.0)
        )) / 100.0
        fixed = float(building.get(
            "standing_charge_GBP_per_connection_year",
            tariffs.get("standing_charge_GBP_per_connection_year", 0.0),
        )) * connections * probability
        counterfactual_bill = None
        if heat_counterfactual:
            counterfactual_bill = heat_counterfactual.get("by_building", {}).get(name, {}).get(
                "annual_customer_bill_GBP"
            )
        if tariff_mode == "counterfactual_bill_parity":
            if counterfactual_bill is None:
                raise ValueError(
                    "Counterfactual-bill parity requires an individual-gas heating counterfactual."
                )
            base_heat_bill = float(counterfactual_bill) * probability
            heat_bill_rate = changes.get("gas_real_rate", 0.0)
        else:
            base_heat_bill = heat_kWh * heat_rate
            heat_bill_rate = changes.get("heat_tariff_real_rate", 0.0)
        revenue[f"{name} heat"] = _escalated_series(
            base_heat_bill, life, heat_bill_rate, start
        )
        if start == 1:
            year1_heat_bill += base_heat_bill + (
                fixed if tariff_mode != "counterfactual_bill_parity" else 0.0
            )
            if counterfactual_bill is not None:
                year1_counterfactual_bill += float(counterfactual_bill) * probability
        if include_cooling and cool_kWh:
            cf_cooling_bill = None
            if cooling_counterfactual:
                cf_cooling_bill = cooling_counterfactual.get("by_building", {}).get(name, {}).get(
                    "annual_customer_bill_GBP"
                )
            if cooling_tariff_mode == "counterfactual_bill_parity":
                if cf_cooling_bill is None:
                    raise ValueError(
                        "Cooling-bill parity requires an individual-AC cooling counterfactual."
                    )
                base_cooling_bill = float(cf_cooling_bill) * probability
                cooling_bill_rate = changes.get("electricity_real_rate", 0.0)
            else:
                base_cooling_bill = cool_kWh * cool_rate
                cooling_bill_rate = changes.get("cooling_tariff_real_rate", 0.0)
            revenue[f"{name} cooling"] = _escalated_series(
                base_cooling_bill, life, cooling_bill_rate, start
            )
            if start == 1:
                year1_cooling_bill += base_cooling_bill
                if cf_cooling_bill is not None:
                    year1_counterfactual_cooling_bill += float(cf_cooling_bill) * probability
        if fixed and tariff_mode != "counterfactual_bill_parity":
            revenue[f"{name} standing charge"] = _escalated_series(
                fixed, life, changes.get("heat_tariff_real_rate", 0.0), start
            )
        connection_charge = float(building.get("connection_charge_GBP", 0.0))
        if building.get("connection_charge_GBP_per_kW") is not None:
            reported_peak = float(building.get("peak_total_heat_kW", node["total_heat_kW"].max()))
            connection_charge += float(building["connection_charge_GBP_per_kW"]) * reported_peak
        if connection_charge:
            charge_series = np.zeros(int(life) + 1)
            if start <= life:
                charge_series[start] = connection_charge * probability
            revenue[f"{name} connection charge"] = charge_series
        heat_energy[name] = _escalated_series(heat_kWh, life, 0.0, start)
        cooling_energy[name] = _escalated_series(cool_kWh, life, 0.0, start)
    zeros = np.zeros(int(life) + 1)
    metadata = {
        "heat_tariff_mode": tariff_mode,
        "year1_district_heat_bill_GBP": year1_heat_bill,
        "year1_counterfactual_heat_bill_GBP": year1_counterfactual_bill,
        "year1_customer_bill_ratio": (
            year1_heat_bill / year1_counterfactual_bill
            if year1_counterfactual_bill > 0 else None
        ),
        "cooling_tariff_mode": cooling_tariff_mode,
        "year1_district_cooling_bill_GBP": year1_cooling_bill,
        "year1_counterfactual_cooling_bill_GBP": year1_counterfactual_cooling_bill,
        "year1_cooling_bill_ratio": (
            year1_cooling_bill / year1_counterfactual_cooling_bill
            if year1_counterfactual_cooling_bill > 0 else None
        ),
    }
    return (
        revenue,
        sum(heat_energy.values(), zeros.copy()),
        sum(cooling_energy.values(), zeros.copy()),
        metadata,
    )

def load_weather(weather=None):
    df = pd.read_csv(Path((weather or {}).get("csv_path", DEFAULT_WEATHER_CSV)))
    if len(df) != 8760:
        raise ValueError(f"Weather file must contain 8760 rows; got {len(df)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2021-01-01", periods=len(df), freq="h")
    return df

def _overrides(cfg):
    result = {
        k: v for k, v in cfg.items()
        if k not in {
            "type", "preset", "capacity_MW", "name", "depends_on",
            "dispatch_direct", "role", "rationale",
        }
    }
    result["name"] = cfg.get("name") or cfg["type"].replace("_", " ").title()
    if "capacity_MW" in cfg:
        if cfg["type"] in UNIT_TYPES:
            # Public scenario contract: capacity_MW always means TOTAL
            # installed capacity. Components use n_units * unit_capacity_MW,
            # so derive the per-unit value exactly once here.
            preset_defaults = PRESETS_BY_TYPE.get(cfg["type"], {}).get(cfg.get("preset"), {})
            n_units = int(result.get("n_units", cfg.get("n_units", preset_defaults.get("n_units", 1))))
            if n_units <= 0:
                raise ValueError("n_units must be a positive integer")
            result["n_units"] = n_units
            result["unit_capacity_MW"] = float(cfg["capacity_MW"]) / n_units
        elif cfg["type"] == "data_centre":
            result.setdefault("heat_offtake_MW", cfg["capacity_MW"])
        elif cfg["type"] == "efw_chp":
            result.setdefault("heat_capacity_MW", cfg["capacity_MW"])
        else:
            result.setdefault("capacity_MW", cfg["capacity_MW"])
    return result

def build_heat_sources(configs, weather, sink_temp_C, return_assets=False):
    """
    Build every heating source, including the BoosterHeatPump special
    case (it needs another already-built data_centre source's output
    temperature as its own input, not just a preset/override dict).

    depends_on FIX: this used to index into a separate, DC-only filtered
    list ("data_centres"), while the UI's own tooltip described it as
    "Data-centre source index (0 = first heat source)" -- i.e. a
    position in the FULL list of sources shown on screen ("Heating
    source 1", "Heating source 2", ...). Following the UI's own
    instructions (e.g. ASHP at position 0, data_centre at position 1,
    booster with depends_on=1) crashed with an unhandled IndexError,
    since position 1 doesn't exist in a list containing only ONE data
    centre. Fixed here: depends_on now genuinely indexes into the SAME
    full `configs` list the UI displays, with a clear ValueError (not a
    raw IndexError) if it points at something that isn't a real
    data_centre source.
    """
    sources = []
    assets = []
    dc_by_position = {}     # position in `configs` -> (DataCentre object, its config dict)
    booster_cfgs = []       # (position in `configs`, config dict)

    for i, cfg in enumerate(configs):
        if cfg["type"] == "booster_heat_pump":
            booster_cfgs.append((i, cfg))
            continue
        cls = HEAT_CLASSES[cfg["type"]]
        if cfg["type"] in {"ashp", "data_centre", "efw_chp"}:
            obj = cls.from_preset(cfg["preset"], weather_df=weather, **_overrides(cfg))
        elif cfg["type"] in {"wshp", "gshp"}:
            # Water/ground-source heat pumps lift from their own source to the
            # NETWORK flow temperature, so they need the sink like the booster
            # does — unlike an ASHP, whose flow_temp_C is its own preset field.
            obj = cls.from_preset(
                cfg["preset"], sink_temp_C=sink_temp_C, weather_df=weather, **_overrides(cfg)
            )
        else:
            obj = cls.from_preset(cfg["preset"], **_overrides(cfg))
        if cfg["type"] == "data_centre":
            dc_by_position[i] = (obj, cfg)
            assets.append(obj)
        else:
            sources.append(obj)
            assets.append(obj)

    used_dc_positions = set()
    for i, cfg in booster_cfgs:
        depends_on = int(cfg.get("depends_on", 0))
        if depends_on not in dc_by_position:
            raise ValueError(
                f"Heating source {i} (booster_heat_pump) has depends_on={depends_on}, "
                f"which must be the position of a 'data_centre' source in THIS "
                f"scenario's heating source list (counting all heating sources, "
                f"0-based — the same numbering shown on screen as 'Heating source "
                f"1', 'Heating source 2', etc., minus 1). No data_centre source "
                f"was found at position {depends_on}."
            )
        used_dc_positions.add(depends_on)

    for pos, (dc, dc_cfg) in dc_by_position.items():
        if dc_cfg.get("dispatch_direct", False) or pos not in used_dc_positions:
            sources.append(dc)

    for i, cfg in booster_cfgs:
        depends_on = int(cfg.get("depends_on", 0))
        dc, _ = dc_by_position[depends_on]
        booster = BoosterHeatPump.from_preset(
            cfg["preset"], source_temp_C_hourly=dc.supply_temp_C,
            source_heat_available_MW=dc.supply_MW,
            source_heat_cost_GBP_per_MWh=dc.waste_heat_cost_GBP_per_MWh,
            sink_temp_C=sink_temp_C, **_overrides(cfg))
        sources.append(booster)
        assets.append(booster)
    return (sources, assets) if return_assets else sources

def build_cooling_sources(configs, weather):
    return [AirCooledChiller.from_preset(c["preset"], weather_df=weather, **_overrides(c)) for c in configs]

def _build_tree_topology(segments, demand, include_cooling):
    """
    Build a NetworkTopology from the UI's plain segment dicts, attaching
    each served building's REAL peaks (heating peak taken from the full
    total_heat_kW array, i.e. INCLUDING DHW — the network genuinely
    carries hot-water demand too, so branch pipes must be sized for it).

    Returns (topology, detail_rows) where detail_rows is a UI-ready list
    of per-segment dicts (DN/capex filled in later by _fill_segment_detail).
    """
    from network.network_topology import NetworkTopology
    from scenarios.scenario_schema import TREE_ROOT_ID

    peaks_by_building = {}
    cool_peaks_by_building = {}
    for node in demand["nodes"]:
        peaks_by_building[node["name"]] = float(np.asarray(node["total_heat_kW"]).max())
        cool_peaks_by_building[node["name"]] = float(node["peak_cool_kW"]) if include_cooling else 0.0

    topo = NetworkTopology(name="Scenario tree network")
    topo.add_node(TREE_ROOT_ID, parent_id=None, length_m=0.0, building_name="Energy centre")

    # add_node() requires parents before children — order the segments
    # accordingly (the schema has already checked every parent exists)
    remaining = list(segments)
    added = {TREE_ROOT_ID}
    ordered = []
    while remaining:
        progress = False
        for seg in list(remaining):
            if seg["parent_id"] in added:
                ordered.append(seg)
                added.add(seg["node_id"])
                remaining.remove(seg)
                progress = True
        if not progress:
            stuck = sorted(s["node_id"] for s in remaining)
            raise ValueError(f"Network segments {stuck} form a loop or reference each other "
                             f"circularly — a heat network tree cannot contain loops.")

    detail_rows = []
    for seg in ordered:
        building = seg.get("building") or None
        topo.add_node(
            seg["node_id"], parent_id=seg["parent_id"], length_m=float(seg["length_m"]),
            peak_kW=peaks_by_building.get(building, 0.0) if building else 0.0,
            peak_cool_kW=cool_peaks_by_building.get(building, 0.0) if building else 0.0,
            building_name=building,
        )
        detail_rows.append({"Segment": seg["node_id"], "Connects to": seg["parent_id"],
                            "Length (m)": float(seg["length_m"]),
                            "Serves": building or "junction"})
    topo.validate()
    return topo, detail_rows


def _delivered_temperature_check(topo, sized_heat, net_cfg):
    """Does heat arrive hot enough to make DHW, after real route loss?

    Only meaningful in tree mode: generic_length has no route to propagate a
    temperature along, so there is nothing to check and this is not called.

    Returns a UI/audit-ready dict, or None when the network serves no building
    with a heating peak (in which case there is genuinely nothing to check —
    see check_minimum_delivered_temperature()'s note on vacuous truth).
    """
    from network.design_temperature_limits import (
        check_flow_temp_against_cp1,
        check_return_temp_against_cp1,
        delivered_temp_basis,
        minimum_delivered_temp_C,
    )

    dhw_system = net_cfg.get("dhw_system", "instantaneous_hiu")
    floor_C = minimum_delivered_temp_C(dhw_system)
    flow_C = float(net_cfg["heat_flow_temp_C"])
    result = topo.check_minimum_delivered_temperature(
        sized_heat, source_flow_temp_C=flow_C, min_temp_C=floor_C,
    )
    if result["all_compliant"] is None:
        return None
    return {
        "dhw_system": dhw_system,
        "minimum_delivered_temp_C": floor_C,
        "basis": delivered_temp_basis(dhw_system),
        "worst_case_building": result["worst_case_building"],
        "worst_case_delivered_temp_C": result["worst_case_delivered_temp_C"],
        "margin_C": round(float(result["worst_case_delivered_temp_C"]) - floor_C, 2),
        "buildings_checked": result["buildings_checked"],
        "compliant": bool(result["all_compliant"]),
        "by_building": result["by_building"],
        "cp1_flow": check_flow_temp_against_cp1(flow_C),
        "cp1_return": check_return_temp_against_cp1(float(net_cfg["heat_return_temp_C"])),
    }


def _fill_segment_detail(detail_rows, sized_segments, duty):
    """Attach the sized pipe results for one duty onto the UI detail rows."""
    label = "Heat" if duty == "heat" else "Cooling"
    for row in detail_rows:
        s = sized_segments.get(row["Segment"])
        if s is None:
            row[f"{label} peak (kW)"] = 0.0
            row[f"{label} pipe"] = "—"
            row[f"{label} CAPEX (£)"] = 0.0
        else:
            row[f"{label} peak (kW)"] = round(s.peak_kW, 0)
            row[f"{label} pipe"] = f"DN{s.pipe.DN}"
            row[f"{label} CAPEX (£)"] = round(s.capex_GBP, 0)


def _combined_counterfactual(nodes, buildings, weather, include_cooling, om_rate,
                             counterfactual="individual_gas",
                             apply_bus_grant=True,
                             counterfactual_electricity_price_p_per_kWh=None):
    enriched_nodes = [
        {**node, "connections": _connection_count(building)}
        for node, building in zip(nodes, buildings)
    ]
    if counterfactual in {"individual_ashp", "individual_ashp_and_ac"}:
        from functools import partial
        # Every building installs its own heat pump instead. This is the
        # comparison heat network zoning actually rests on — the Energy Act 2023
        # framework designates zones where heat networks are "the lowest-cost
        # solution for DECARBONISING heat", not where they beat a gas boiler.
        # The BUS grant is applied where eligible (45 kWth cap), because it is
        # what a real customer would receive.
        fn = partial(
            counterfactual_individual_ashp_dispatch,
            apply_bus_grant=apply_bus_grant,
            electricity_price_p_per_kWh=counterfactual_electricity_price_p_per_kWh,
        )
        heat = aggregate_counterfactual(
            enriched_nodes, fn, weather_df=weather, om_rate=om_rate,
        )
    else:
        heat = aggregate_counterfactual(
            enriched_nodes, counterfactual_gas_boiler_dispatch, om_rate=om_rate
        )
    if not include_cooling:
        return heat
    cooling = aggregate_counterfactual(enriched_nodes, counterfactual_individual_ac_dispatch,
                                       weather_df=weather, om_rate=om_rate)
    return {"heating": heat, "cooling": cooling,
            "total_capex_GBP": round(heat["total_capex_GBP"] + cooling["total_capex_GBP"], 0),
            "total_annual_opex_GBP": round(heat["total_annual_opex_GBP"] + cooling["total_annual_opex_GBP"], 0)}

def run_scenario(scenario):
    cfg = validate_or_raise(scenario)
    raw_weather = load_weather(cfg.get("weather"))
    climate_reference = compute_climate_reference(
        apply_climate_scenario(raw_weather, "baseline")
    )
    weather = apply_climate_scenario(raw_weather, cfg["climate_scenario"])
    demand = synthesise_network(
        weather, {**cfg["demand"], "demand_nodes": cfg["demand"]["buildings"]},
        climate_reference=climate_reference,
    )
    net_cfg = cfg["network"]
    include_cooling = bool(net_cfg["include_cooling"])

    network = None
    network_detail = None            # per-segment breakdown for tree mode (UI table)
    delivered_temperature = None     # tree mode only — generic_length has no route to propagate along
    heat_loss_MWh = cool_gain_MWh = 0.0
    heat_loss_kW_hourly = cool_gain_kW_hourly = 0.0   # scalar 0 broadcasts fine
    network_capex = 0.0
    sized_heat = sized_cool = None

    if net_cfg["mode"] == "generic_length":
        kwargs = {"heat_flow_temp_C": net_cfg["heat_flow_temp_C"], "heat_return_temp_C": net_cfg["heat_return_temp_C"],
                  "cool_flow_temp_C": net_cfg["cool_flow_temp_C"], "cool_return_temp_C": net_cfg["cool_return_temp_C"]} if include_cooling else {"flow_temp_C": net_cfg["heat_flow_temp_C"], "return_temp_C": net_cfg["heat_return_temp_C"]}
        network = size_network_from_demand(demand, net_cfg["length_m"], include_cooling=include_cooling, **kwargs)
        for duty in network.duties:
            if duty.duty_name == "heating": heat_loss_MWh = duty.annual_heat_loss_MWh
            if duty.duty_name == "cooling": cool_gain_MWh = abs(duty.annual_heat_loss_MWh)
        network_capex = network.total_capex_GBP
        # Buried-pipe standing loss is driven by pipe-to-ground delta-T,
        # which is roughly constant across the year — NOT by demand. The
        # previous demand-proportional distribution concentrated the whole
        # annual loss into winter hours and made summer network load
        # (DHW-only weeks, where standing loss is a LARGE share of load)
        # look better than it really is. Spread it as a constant kW.
        heat_loss_kW_hourly = heat_loss_MWh * 1000.0 / 8760.0
        cool_gain_kW_hourly = cool_gain_MWh * 1000.0 / 8760.0

    elif net_cfg["mode"] == "tree":
        topo, network_detail = _build_tree_topology(net_cfg["segments"], demand, include_cooling)
        sized_heat = topo.size_all_segments(net_cfg["heat_flow_temp_C"], net_cfg["heat_return_temp_C"], duty="heat")
        network_capex = topo.total_capex_GBP(sized_heat)
        # Does the heat actually ARRIVE hot enough? Until now nothing in the
        # engine asked. check_minimum_delivered_temperature() and
        # minimum_safe_flow_temp_C() existed, were sophisticated, and were called
        # by nothing except a dormant module — so a scenario could drop its flow
        # temperature, collect the resulting heat-pump COP gain and a better NPV,
        # and never be told its customers were not getting hot water.
        delivered_temperature = _delivered_temperature_check(
            topo, sized_heat, net_cfg,
        )
        heat_losses = topo.network_heat_loss_kW_hourly(sized_heat, net_cfg["heat_flow_temp_C"])
        heat_loss_kW_hourly = heat_losses["total_kW_hourly"]
        heat_loss_MWh = heat_losses["annual_total_MWh"]
        _fill_segment_detail(network_detail, sized_heat, duty="heat")
        if include_cooling:
            sized_cool = topo.size_all_segments(net_cfg["cool_flow_temp_C"], net_cfg["cool_return_temp_C"], duty="cool")
            network_capex += topo.total_capex_GBP(sized_cool)
            cool_gains = topo.network_heat_loss_kW_hourly(sized_cool, net_cfg["cool_flow_temp_C"])
            cool_gain_kW_hourly = cool_gains["total_kW_hourly"]
            cool_gain_MWh = cool_gains["annual_total_MWh"]
            _fill_segment_detail(network_detail, sized_cool, duty="cool")
        network = topo

    # A feasibility report or hydraulic model can provide better loss/CAPEX
    # values than the screening pipe curve. Explicit overrides remain visible
    # in the scenario and are useful for calibration/assurance cases.
    if net_cfg.get("annual_heat_loss_MWh_override") is not None:
        heat_loss_MWh = float(net_cfg["annual_heat_loss_MWh_override"])
        heat_loss_kW_hourly = heat_loss_MWh * 1000.0 / 8760.0
    if net_cfg.get("annual_cooling_gain_MWh_override") is not None:
        cool_gain_MWh = float(net_cfg["annual_cooling_gain_MWh_override"])
        cool_gain_kW_hourly = cool_gain_MWh * 1000.0 / 8760.0
    if net_cfg.get("capex_GBP_override") is not None:
        network_capex = float(net_cfg["capex_GBP_override"])

    thermal_storage = _build_thermal_storage(
        cfg.get("thermal_storage"), net_cfg["heat_flow_temp_C"], net_cfg["heat_return_temp_C"]
    )
    heat_sources, heat_assets = build_heat_sources(
        cfg["sources"], weather, net_cfg["heat_flow_temp_C"], return_assets=True
    )
    heat_dispatch = run_dispatch(
        demand["total_heat_kW"] + heat_loss_kW_hourly,
        heat_sources, storage=thermal_storage, duty="heat",
    )
    heat_summary = heat_dispatch.summary()

    cooling_sources, cooling_dispatch, cooling_summary = [], None, None
    if include_cooling:
        cooling_sources = build_cooling_sources(cfg["cooling_sources"], weather)
        cooling_dispatch = run_dispatch(demand["total_cooling_kW"] + cool_gain_kW_hourly, cooling_sources, duty="cool")
        cooling_summary = cooling_dispatch.summary()

    # Paired data-centre recovery equipment is a capital/O&M asset even when
    # its low-grade heat is consumed through a booster rather than dispatched
    # directly to the network.
    all_sources = heat_assets + cooling_sources
    capex = aggregate_capex(sources=all_sources, storage=thermal_storage)
    econ_cfg = cfg["economics"]
    capex_cfg = econ_cfg.get("capex_items", {})
    total_connections = sum(_connection_count(b) for b in cfg["demand"]["buildings"])
    fixed_capex_items = {
        key: float(value)
        for key, value in capex_cfg.items()
        if key.endswith("_GBP") and "per_connection" not in key
    }
    fixed_capex_items["customer_connections_GBP"] = (
        float(capex_cfg.get("customer_connection_GBP_per_connection", 0.0))
        * total_connections
    )
    fixed_capex_items["metering_GBP"] = (
        float(capex_cfg.get("metering_GBP_per_connection", 0.0))
        * total_connections
    )

    # Design, commissioning and contingency apply to the whole delivered scope,
    # not just plant and network.
    #
    # This previously used base_capex = sources + storage + network only, which
    # left every fixed and per-connection item carrying ZERO contingency: the
    # energy-centre building, the electrical/gas connections, controls/SCADA,
    # customer connections and metering. On the worked scenarios that is ~£9.4m
    # of a ~£21m scheme — and customer connections (the single largest line at
    # £4.5m) are precisely the sort of scope that overruns, since they depend on
    # what is found inside each building. Excluding them understated CAPEX by
    # roughly 14% and flattered every NPV in the study pack.
    #
    # Land is the one deliberate exclusion (EXCLUDED_FROM_CAPEX_ADDERS below): a
    # land purchase or enabling payment is a transaction, not a designed,
    # commissioned or buildable scope, so a design fee and a construction
    # contingency on top of it would be double-counting.
    adder_base_capex = (
        capex["grand_total_GBP"]
        + network_capex
        + sum(v for k, v in fixed_capex_items.items() if k not in EXCLUDED_FROM_CAPEX_ADDERS)
    )
    percentage_capex_items = {
        "development_and_design_GBP": adder_base_capex * float(capex_cfg.get("development_and_design_pct", 0.0)),
        "commissioning_GBP": adder_base_capex * float(capex_cfg.get("commissioning_pct", 0.0)),
        "contingency_GBP": adder_base_capex * float(capex_cfg.get("contingency_pct", 0.0)),
    }
    project_capex_items = {
        "sources_GBP": float(capex["by_category"]["sources_GBP"]),
        "network_GBP": float(network_capex),
        "thermal_storage_GBP": float(capex["by_category"]["storage_GBP"]),
        **fixed_capex_items,
        **percentage_capex_items,
    }
    total_capex = sum(project_capex_items.values())
    # Per-technology O&M (replaces the flat CHDU 1% of total CAPEX)
    from economics.om_rates import total_annual_om_GBP as _tech_om
    om_detail = _tech_om(all_sources, network_capex)
    om = om_detail["total_GBP"]

    heat_load_kW = heat_dispatch.demand_MW * 1000.0
    cooling_load_kW = cooling_dispatch.demand_MW * 1000.0 if cooling_dispatch else None
    pumping_hourly_MW = np.zeros(8760)
    if net_cfg["mode"] == "tree":
        heat_pumping = annual_pumping_electricity_MWh(
            network, sized_heat,
            _mass_flow_kg_s(heat_load_kW, net_cfg["heat_flow_temp_C"], net_cfg["heat_return_temp_C"]),
            density_kg_m3=water_properties(net_cfg["heat_flow_temp_C"])["density_kg_m3"],
        )
        pumping_hourly_MW += heat_pumping["hourly_pumping_MW"]
        if include_cooling:
            cooling_pumping = annual_pumping_electricity_MWh(
                network, sized_cool,
                _mass_flow_kg_s(cooling_load_kW, net_cfg["cool_flow_temp_C"], net_cfg["cool_return_temp_C"]),
                density_kg_m3=water_properties(net_cfg["cool_flow_temp_C"])["density_kg_m3"],
            )
            pumping_hourly_MW += cooling_pumping["hourly_pumping_MW"]
    elif net_cfg["mode"] == "generic_length":
        heat_duty = next(d for d in network.duties if d.duty_name == "heating")
        pumping_hourly_MW += _generic_pumping_hourly_MW(
            heat_duty, heat_load_kW,
            net_cfg["heat_flow_temp_C"], net_cfg["heat_return_temp_C"],
        )
        if include_cooling:
            cool_duty = next(d for d in network.duties if d.duty_name == "cooling")
            pumping_hourly_MW += _generic_pumping_hourly_MW(
                cool_duty, cooling_load_kW,
                net_cfg["cool_flow_temp_C"], net_cfg["cool_return_temp_C"],
            )
    hydraulic_pumping_MWh = float(pumping_hourly_MW.sum())
    parasitic_override = econ_cfg.get("total_parasitic_electricity_MWh_override")
    if parasitic_override is None:
        parasitic_hourly_MW = pumping_hourly_MW
    elif hydraulic_pumping_MWh > 0:
        parasitic_hourly_MW = pumping_hourly_MW * (
            float(parasitic_override) / hydraulic_pumping_MWh
        )
    else:
        parasitic_hourly_MW = np.full(8760, float(parasitic_override) / 8760.0)
    parasitic_electricity_MWh = float(parasitic_hourly_MW.sum())
    additional_auxiliary_MWh = max(0.0, parasitic_electricity_MWh - hydraulic_pumping_MWh)
    pumping_MWh = parasitic_electricity_MWh
    parasitic_price = econ_cfg.get("parasitic_electricity_price_GBP_per_MWh")
    pumping_cost = float((parasitic_hourly_MW * resolve_electricity_price(parasitic_price)).sum())
    energy_cost = (
        heat_summary["total_annual_opex_GBP"]
        + (cooling_summary["total_annual_opex_GBP"] if cooling_summary else 0.0)
        + pumping_cost
    )
    additional_opex_items = {
        key: float(value)
        for key, value in econ_cfg.get("annual_opex_items", {}).items()
    }
    additional_opex = sum(additional_opex_items.values())
    annual_opex = energy_cost + om + additional_opex

    heat_delivered = demand["annual_heat_MWh"] + demand["annual_dhw_MWh"]
    cool_delivered = demand["annual_cool_MWh"] if include_cooling else 0.0
    total_delivered = heat_delivered + cool_delivered
    carbon_factors = econ_cfg.get("carbon_factors", {})
    electricity_carbon = float(carbon_factors.get(
        "electricity_kgCO2_per_kWh", CARBON_INTENSITY["electric"]
    ))
    carbon_t = (
        _dispatch_carbon_tCO2(heat_dispatch, carbon_factors)
        + _dispatch_carbon_tCO2(cooling_dispatch, carbon_factors)
        + pumping_MWh * electricity_carbon
    )
    # tCO2 / MWh is numerically equal to kgCO2 / kWh.
    carbon_intensity_kg = carbon_t / total_delivered if total_delivered else 0.0

    # GHNF grant is a financing inflow in year 0, not an OPEX reduction.
    from economics.grant import apply_ghnf_grant
    grant_cfg = econ_cfg.get("ghnf_grant", {})
    grant_result = None
    effective_capex = total_capex
    if grant_cfg.get("enabled", False) and carbon_intensity_kg <= 0.100:
        grant_result = apply_ghnf_grant(
            total_capex_GBP=total_capex,
            network_capex_GBP=network_capex,
            source_capex_GBP=capex["by_category"]["sources_GBP"],
            grant_rate=grant_cfg.get("rate", 0.40),
            eligible_capex_GBP=grant_cfg.get("eligible_capex_GBP"),
            annual_thermal_delivered_kWh=total_delivered * 1000.0,
        )
        grant_result["carbon_eligibility"] = "passes 100 gCO2e/kWh screening threshold"
        effective_capex = grant_result["net_capex_GBP"]
    elif grant_cfg.get("enabled", False):
        grant_result = {
            "eligible_capex_GBP": 0.0,
            "grant_rate": float(grant_cfg.get("rate", 0.40)),
            "grant_GBP": 0.0,
            "net_capex_GBP": round(total_capex, 0),
            "output_based_cap_GBP": None,
            "output_cap_basis": "4.5p/kWh of thermal energy delivered over 15 years",
            "carbon_eligibility": (
                f"fails 100 gCO2e/kWh screening threshold ({carbon_intensity_kg*1000:.1f} gCO2e/kWh)"
            ),
        }

    # All NPV, IRR, payback and chart values below are derived from the same
    # explicit annual cash-flow tables.  This removes the former inconsistency
    # where the KPI and chart silently escalated different quantities.
    counterfactual = None
    financial = {}
    life = int(econ_cfg["project_lifetime_years"])
    rate = float(econ_cfg["discount_rate"])
    social_rate = float(econ_cfg.get("social_discount_rate", 0.035))
    changes = econ_cfg.get("price_changes", {})
    if econ_cfg["counterfactual"] != "none":
        counterfactual = _combined_counterfactual(
            demand["nodes"], cfg["demand"]["buildings"], weather,
            include_cooling, econ_cfg["om_rate"],
            counterfactual=econ_cfg["counterfactual"],
            apply_bus_grant=econ_cfg.get("apply_bus_grant", True),
            counterfactual_electricity_price_p_per_kWh=econ_cfg.get(
                "counterfactual_electricity_price_p_per_kWh"
            ),
        )
    revenue_items, connected_heat_kWh, connected_cool_kWh, revenue_meta = _customer_revenue_and_energy(
        cfg["demand"]["buildings"], demand["nodes"], econ_cfg, include_cooling, life,
        counterfactual=counterfactual,
    )
    connected_total_kWh = connected_heat_kWh + connected_cool_kWh
    full_energy_kWh = max(total_delivered * 1000.0, 1e-9)
    service_factor = np.clip(connected_total_kWh / full_energy_kWh, 0.0, 1.0)

    carrier_costs = _dispatch_costs_by_carrier(heat_dispatch)
    cooling_carriers = _dispatch_costs_by_carrier(cooling_dispatch)
    for carrier in carrier_costs:
        carrier_costs[carrier] += cooling_carriers[carrier]
    carrier_costs["electricity"] += pumping_cost
    carrier_rates = {
        "electricity": float(changes.get("electricity_real_rate", 0.0)),
        "gas": float(changes.get("gas_real_rate", 0.0)),
        "third_party_heat": float(changes.get("third_party_heat_real_rate", 0.0)),
    }
    opex_items = {}
    for carrier, base in carrier_costs.items():
        opex_items[f"{carrier} energy"] = (
            _escalated_series(base, life, carrier_rates[carrier]) * service_factor
        )
    other_rate = float(changes.get("other_opex_real_rate", 0.0))
    opex_items["technology and network O&M"] = _escalated_series(om, life, other_rate)
    for name, value in additional_opex_items.items():
        opex_items[name] = _escalated_series(value, life, other_rate)
    repex_items = _replacement_series(
        all_sources, life, econ_cfg.get("replacement_overrides", {})
    )
    grant_items = {"GHNF": grant_result["grant_GBP"]} if grant_result else {}
    investor = build_cashflow(
        life_years=life,
        discount_rate=rate,
        capex=project_capex_items,
        revenues=revenue_items,
        opex=opex_items,
        repex=repex_items,
        grants=grant_items,
    )
    investor["annual_revenue_GBP"] = round(sum(x[1] for x in revenue_items.values()), 0)
    investor["annual_net_cashflow_GBP"] = round(investor["net_cashflow_GBP"][1], 0)
    investor.update(revenue_meta)
    investor["equivalent_year1_heat_tariff_p_per_kWh"] = (
        round(revenue_meta["year1_district_heat_bill_GBP"] / connected_heat_kWh[1] * 100.0, 3)
        if connected_heat_kWh[1] > 0 else None
    )
    investor["customer_bill_compliant"] = (
        revenue_meta["year1_customer_bill_ratio"] is None
        or revenue_meta["year1_customer_bill_ratio"] <= 1.000001
    )
    investor["cooling_bill_compliant"] = (
        revenue_meta["year1_cooling_bill_ratio"] is None
        or revenue_meta["year1_cooling_bill_ratio"] <= 1.000001
    )
    investor["revenue_basis"] = (
        "Heat bills are held at the modelled individual-gas customer bill by building; "
        "gas-price changes therefore move both sides together. Cooling uses the selected separate "
        "manual or individual-AC bill-parity basis."
        if revenue_meta["heat_tariff_mode"] == "counterfactual_bill_parity"
        else "Manual heat and cooling unit tariffs plus standing charges, by customer; "
             "connection year and connection probability are applied where provided."
    )
    financial["investor"] = investor
    year1_investor_opex = sum(values[1] for values in opex_items.values())
    opex_reconciliation = {
        "full_buildout_heat_source_energy_GBP": heat_summary["total_annual_opex_GBP"],
        "full_buildout_cooling_source_energy_GBP": (
            cooling_summary["total_annual_opex_GBP"] if cooling_summary else 0.0
        ),
        "full_buildout_parasitic_electricity_GBP": pumping_cost,
        "full_buildout_energy_total_GBP": energy_cost,
        "technology_and_network_om_GBP": om,
        "additional_overhead_opex_GBP": additional_opex,
        "full_buildout_total_opex_GBP": annual_opex,
        "year1_connection_weighted_energy_factor": float(service_factor[1]),
        "year1_investor_opex_GBP": year1_investor_opex,
        "full_buildout_reconciliation_residual_GBP": (
            annual_opex - energy_cost - om - additional_opex
        ),
        "basis": (
            "Full-buildout OPEX uses the complete hourly demand and dispatch. Investor-year energy "
            "cost is connection-weighted; fixed O&M and overhead remain payable in full."
        ),
    }
    financial["opex_reconciliation"] = opex_reconciliation

    # Gross project cost series supports discounted LCOH/LCO-service. Grant is
    # intentionally excluded because it transfers who pays, not resource cost.
    investor_rows = investor["annual_table"]
    project_cost_series = np.asarray([
        row["capex_GBP"] + row["repex_GBP"] + row["opex_GBP"] - row["residual_value_GBP"]
        for row in investor_rows
    ])
    factors = discount_factors(life, rate)
    grant_series = np.zeros(life + 1)
    grant_series[0] = grant_result["grant_GBP"] if grant_result else 0.0
    non_heat_revenue = sum(
        (values for name, values in revenue_items.items()
         if not name.endswith(" heat")),
        np.zeros(life + 1),
    )
    discounted_heat_kWh = float((connected_heat_kWh * factors).sum())
    required_heat_tariff = max(0.0, float(
        ((project_cost_series - grant_series - non_heat_revenue) * factors).sum()
        / discounted_heat_kWh
    )) if discounted_heat_kWh else None
    investor["required_heat_tariff_p_per_kWh_for_zero_NPV"] = (
        None if required_heat_tariff is None else round(required_heat_tariff * 100.0, 3)
    )
    investor["peak_funding_requirement_GBP"] = round(
        max(0.0, -min(investor["cumulative_undiscounted_GBP"])), 0
    )
    discounted_lcoh = discounted_levelised_cost_GBP_per_kWh(
        costs_GBP=project_cost_series,
        delivered_kWh=connected_heat_kWh,
        discount_rate=rate,
    )
    discounted_lcos = discounted_levelised_cost_GBP_per_kWh(
        costs_GBP=project_cost_series,
        delivered_kWh=connected_total_kWh,
        discount_rate=rate,
    )

    if econ_cfg["counterfactual"] != "none":
        cf_opex_items = {}
        if include_cooling:
            cf_opex_items["individual gas heating"] = _escalated_series(
                counterfactual["heating"]["total_annual_opex_GBP"], life,
                changes.get("gas_real_rate", 0.0),
            ) * np.clip(connected_heat_kWh / max(heat_delivered * 1000.0, 1e-9), 0.0, 1.0)
            cf_opex_items["individual cooling"] = _escalated_series(
                counterfactual["cooling"]["total_annual_opex_GBP"], life,
                changes.get("electricity_real_rate", 0.0),
            ) * np.clip(connected_cool_kWh / max(cool_delivered * 1000.0, 1e-9), 0.0, 1.0)
        else:
            cf_opex_items["individual gas heating"] = _escalated_series(
                counterfactual["total_annual_opex_GBP"], life,
                changes.get("gas_real_rate", 0.0),
            ) * np.clip(connected_heat_kWh / max(heat_delivered * 1000.0, 1e-9), 0.0, 1.0)
        cf_avoided_repex = np.zeros(life + 1)
        heat_cf_capex = float(counterfactual.get("heating", counterfactual)["total_capex_GBP"])
        for year in range(20, life + 1, 20):
            cf_avoided_repex[year] += heat_cf_capex * 0.50
        if include_cooling:
            cool_cf_capex = float(counterfactual["cooling"]["total_capex_GBP"])
            for year in range(15, life + 1, 15):
                cf_avoided_repex[year] += cool_cf_capex * 0.60
        avoided_costs = {f"avoided {k}": v for k, v in cf_opex_items.items()}
        avoided_costs["avoided counterfactual replacement"] = cf_avoided_repex
        social = build_cashflow(
            life_years=life,
            discount_rate=social_rate,
            capex={"incremental project CAPEX": total_capex - counterfactual["total_capex_GBP"]},
            revenues=avoided_costs,
            opex=opex_items,
            repex=repex_items,
        )
        social["basis"] = (
            "Whole-system resource-cost comparison at the social discount rate; "
            "grant and customer tariff transfers are excluded."
        )
        financial["social"] = social
        financial.update({
            "counterfactual": econ_cfg["counterfactual"],
            "counterfactual_capex_GBP": counterfactual["total_capex_GBP"],
            "counterfactual_annual_opex_GBP": counterfactual["total_annual_opex_GBP"],
            "incremental_capex_GBP": round(total_capex - counterfactual["total_capex_GBP"], 0),
            "annual_avoided_cost_GBP": round(counterfactual["total_annual_opex_GBP"] - annual_opex, 0),
            "npv_vs_counterfactual_GBP": social["npv_GBP"],
            "irr_vs_counterfactual": social["irr"],
            "simple_payback_years": social["simple_payback_years"],
            "discounted_payback_years": social["discounted_payback_years"],
            "cashflow_years": social["cashflow_years"],
            "cumulative_discounted_GBP": social["cumulative_discounted_GBP"],
            "cumulative_undiscounted_GBP": social["cumulative_undiscounted_GBP"],
            "annual_table": social["annual_table"],
        })
    headline = {
        "system_type": "4_pipe_heating_cooling" if include_cooling else "2_pipe_heating",
        "annual_heat_demand_MWh": round(heat_delivered, 1), "annual_cooling_demand_MWh": round(cool_delivered, 1),
        "annual_heat_to_generate_MWh": heat_summary["annual_demand_MWh"],
        "annual_cooling_to_generate_MWh": cooling_summary["annual_demand_MWh"] if cooling_summary else 0.0,
        "peak_heat_MW": round(demand["peak_heat_kW"] / 1000, 3), "peak_cooling_MW": round(demand["peak_cool_kW"] / 1000, 3),
        "peak_heat_to_generate_MW": round(float(heat_dispatch.demand_MW.max()), 3),
        "peak_cooling_to_generate_MW": round(float(cooling_dispatch.demand_MW.max()), 3) if cooling_dispatch else 0.0,
        "annual_network_heat_loss_MWh": round(heat_loss_MWh, 1), "annual_network_cooling_gain_MWh": round(cool_gain_MWh, 1),
        "annual_pumping_electricity_MWh": round(hydraulic_pumping_MWh, 1),
        "annual_additional_auxiliary_electricity_MWh": round(additional_auxiliary_MWh, 1),
        "annual_total_parasitic_electricity_MWh": round(parasitic_electricity_MWh, 1),
        "annual_pumping_cost_GBP": round(pumping_cost, 0),
        "thermal_storage_capacity_MWh": round(thermal_storage.capacity_MWh, 3) if thermal_storage else 0.0,
        "thermal_storage_charge_MWh": heat_summary.get("storage_annual_charge_MWh", 0.0),
        "thermal_storage_discharge_MWh": heat_summary.get("storage_annual_discharge_MWh", 0.0),
        "annual_heat_by_source_MWh": heat_summary["annual_MWh_by_source"],
        "annual_cooling_by_source_MWh": cooling_summary["annual_MWh_by_source"] if cooling_summary else {},
        "annual_unmet_demand_MWh": heat_summary["annual_unmet_demand_MWh"], "peak_unmet_MW": heat_summary["peak_unmet_MW"],
        "annual_unmet_cooling_MWh": cooling_summary["annual_unmet_demand_MWh"] if cooling_summary else 0.0,
        "peak_unmet_cooling_MW": cooling_summary["peak_unmet_MW"] if cooling_summary else 0.0,
        "capex_total_GBP": round(total_capex, 0), "effective_capex_GBP": round(effective_capex, 0), "capex_sources_GBP": capex["by_category"]["sources_GBP"], "capex_network_GBP": round(network_capex, 0),
        "capex_breakdown_GBP": {k: round(v, 0) for k, v in project_capex_items.items()},
        "annual_energy_cost_GBP": round(energy_cost, 0), "annual_om_cost_GBP": round(om, 0),
        "annual_additional_opex_GBP": round(additional_opex, 0),
        "annual_total_opex_GBP": round(annual_opex, 0),
        "lcoh_GBP_per_kWh": round(discounted_lcoh, 4),
        "levelised_energy_service_GBP_per_kWh": round(discounted_lcos, 4),
        "lco_method": "Discounted project costs divided by discounted connected customer energy",
        "heat_energy_balance_residual_MWh": round(
            sum(float(values.sum()) for values in heat_dispatch.dispatch_by_source_MW.values())
            + float(heat_dispatch.storage_discharge_MW.sum())
            - float(heat_dispatch.storage_charge_MW.sum())
            + float(heat_dispatch.unmet_demand_MW.sum())
            - float(heat_dispatch.demand_MW.sum()), 9
        ),
        "cooling_energy_balance_residual_MWh": round(
            (sum(float(values.sum()) for values in cooling_dispatch.dispatch_by_source_MW.values())
             + float(cooling_dispatch.unmet_demand_MW.sum())
             - float(cooling_dispatch.demand_MW.sum()))
            if cooling_dispatch else 0.0, 9
        ),
        "annual_carbon_tCO2": round(carbon_t, 1), "carbon_intensity_kgCO2_per_kWh": round(carbon_intensity_kg, 4), "carbon_intensity_kgCO2_per_kWh_service": round(carbon_intensity_kg, 4),
    }
    if network_detail is not None:
        headline["network_total_length_m"] = round(network.total_length_m(), 0)
    elif net_cfg["mode"] == "generic_length":
        headline["network_total_length_m"] = round(float(net_cfg["length_m"]), 0)
    else:
        headline["network_total_length_m"] = 0.0

    route_m = float(headline["network_total_length_m"])
    headline["linear_heat_density_MWh_per_m_year"] = round(heat_delivered / route_m, 3) if route_m else None
    headline["linear_cooling_density_MWh_per_m_year"] = round(cool_delivered / route_m, 3) if route_m and include_cooling else None
    headline["network_heat_loss_fraction"] = round(heat_loss_MWh / max(heat_delivered, 1e-9), 5)
    headline["network_capex_GBP_per_m"] = round(network_capex / route_m, 0) if route_m else None
    headline["capex_GBP_per_kW_peak_service"] = round(
        total_capex / max((headline["peak_heat_MW"] + headline["peak_cooling_MW"]) * 1000.0, 1e-9), 0
    )
    headline["unmet_heat_fraction"] = round(
        heat_summary["annual_unmet_demand_MWh"] / max(heat_summary["annual_demand_MWh"], 1e-9), 8
    )
    headline["unmet_cooling_fraction"] = round(
        (cooling_summary["annual_unmet_demand_MWh"] / max(cooling_summary["annual_demand_MWh"], 1e-9))
        if cooling_summary else 0.0, 8
    )

    screening_cfg = cfg.get("screening", {})
    unmet_fraction_limit = float(screening_cfg.get("maximum_unmet_energy_fraction", 0.001))
    heat_unmet_limit = max(0.5, heat_summary["annual_demand_MWh"] * unmet_fraction_limit)
    cool_unmet_limit = max(0.5, (cooling_summary["annual_demand_MWh"] if cooling_summary else 0.0) * unmet_fraction_limit)
    headline["service_compliant"] = (
        heat_summary["annual_unmet_demand_MWh"] <= heat_unmet_limit
        and (not cooling_summary or cooling_summary["annual_unmet_demand_MWh"] <= cool_unmet_limit)
    )
    carbon_threshold_g = float(screening_cfg.get("maximum_carbon_gCO2e_per_kWh", 100.0))
    headline["carbon_threshold_gCO2e_per_kWh"] = carbon_threshold_g
    headline["carbon_compliant"] = carbon_intensity_kg * 1000.0 <= carbon_threshold_g

    heat_n1 = _n_minus_one_at_peak(heat_sources, heat_dispatch.demand_MW)
    cool_n1 = _n_minus_one_at_peak(cooling_sources, cooling_dispatch.demand_MW) if cooling_dispatch else None
    headline.update({
        "peak_available_heat_capacity_MW": round(heat_n1["available_MW"], 3),
        "n_minus_one_heat_firm_capacity_MW": round(heat_n1["firm_capacity_MW"], 3),
        "n_minus_one_heat_margin_MW": round(heat_n1["margin_MW"], 3),
        "n_minus_one_heat_compliant": heat_n1["compliant"],
        "n_minus_one_cooling_margin_MW": round(cool_n1["margin_MW"], 3) if cool_n1 else None,
        "n_minus_one_cooling_compliant": cool_n1["compliant"] if cool_n1 else True,
        "n_minus_one_compliant": heat_n1["compliant"] and (cool_n1 is None or cool_n1["compliant"]),
        "n_minus_one_basis": "Peak-hour capacity after the largest credible source/unit outage; excludes repair duration, network outages and storage autonomy.",
    })

    # Delivered temperature. None in generic_length mode (no route to propagate a
    # temperature along) — reported as None rather than True, so "not assessed"
    # never reads as "passed".
    headline["dhw_system"] = net_cfg.get("dhw_system", "instantaneous_hiu")
    if delivered_temperature is not None:
        headline.update({
            "minimum_delivered_temp_C": delivered_temperature["minimum_delivered_temp_C"],
            "worst_case_delivered_temp_C": delivered_temperature["worst_case_delivered_temp_C"],
            "delivered_temp_margin_C": delivered_temperature["margin_C"],
            "delivered_temp_compliant": delivered_temperature["compliant"],
            "delivered_temp_basis": delivered_temperature["basis"],
        })
    else:
        headline.update({
            "minimum_delivered_temp_C": None,
            "worst_case_delivered_temp_C": None,
            "delivered_temp_margin_C": None,
            "delivered_temp_compliant": None,
            "delivered_temp_basis": (
                "Not assessed: delivered temperature requires tree mode, which has real "
                "route lengths to propagate a temperature along. Generic-length mode does not."
            ),
        })

    warnings = []
    if net_cfg["mode"] == "generic_length":
        warnings.append("Generic-length mode is an equivalent single trunk with high network CAPEX/pumping uncertainty; use tree mode for investment screening.")
    if any(
        b.get("annual_heat_kWh") is None or (include_cooling and b.get("annual_cool_kWh") is None)
        for b in cfg["demand"]["buildings"]
    ):
        warnings.append("One or more customer demands use archetype benchmarks rather than measured/calibrated data.")
    if not any(float(v) for v in capex_cfg.values()):
        warnings.append("All user-entered project CAPEX additions are zero; connection, building, enabling, utility and contingency costs may be missing.")
    if not any(float(v) for v in additional_opex_items.values()):
        warnings.append("All user-entered annual overhead OPEX lines are zero; billing, rates, insurance and operator overhead may be missing.")
    if include_cooling and not float(econ_cfg.get("tariffs", {}).get("cooling_unit_rate_p_per_kWh", 0.0)):
        warnings.append("Cooling is delivered but the default cooling tariff is zero.")
    if not headline["service_compliant"]:
        warnings.append(
            f"Design fails the screening service gate: annual unmet energy exceeds {unmet_fraction_limit*100:.3g}% "
            "(minimum 0.5 MWh tolerance)."
        )
    if not headline["carbon_compliant"]:
        warnings.append(f"Design exceeds the {carbon_threshold_g:g} gCO2e/kWh screening carbon threshold.")
    if delivered_temperature is not None:
        if not delivered_temperature["compliant"]:
            warnings.append(
                f"Heat does not arrive hot enough: {delivered_temperature['worst_case_building']} "
                f"receives {delivered_temperature['worst_case_delivered_temp_C']:.1f}°C against a "
                f"{delivered_temperature['minimum_delivered_temp_C']:.0f}°C floor "
                f"({delivered_temperature['basis']})."
            )
        if not delivered_temperature["cp1_flow"]["within_cp1_envelope"]:
            warnings.append(
                f"Flow temperature {net_cfg['heat_flow_temp_C']:.0f}°C is outside CP1 2020's "
                f"{delivered_temperature['cp1_flow']['cp1_min_C']:.0f}-"
                f"{delivered_temperature['cp1_flow']['cp1_max_new_scheme_C']:.0f}°C envelope for new schemes."
            )
        elif delivered_temperature["cp1_flow"]["at_cp1_ceiling"]:
            warnings.append(
                f"Flow temperature {net_cfg['heat_flow_temp_C']:.0f}°C sits at CP1 2020's maximum for new "
                "schemes. CP1's direction of travel is downward; a lower flow temperature would raise "
                "heat-pump COP at no pipe-sizing cost (pipe size follows delta-T, not absolute temperature)."
            )
        if not delivered_temperature["cp1_return"]["meets_best_practice"]:
            warnings.append(
                f"Return temperature {net_cfg['heat_return_temp_C']:.0f}°C is "
                f"{delivered_temperature['cp1_return']['excess_over_best_practice_K']:.0f}K above CP1 2020's "
                f"<{delivered_temperature['cp1_return']['cp1_best_practice_vwart_C']:.0f}°C VWART best practice."
            )
    elif net_cfg["mode"] == "tree":
        warnings.append(
            "Delivered temperature was not assessed: no building on this network carries a heating peak."
        )
    warnings.append("Long-term grid-carbon, demand and climate trajectories are not yet applied year by year; the annual operating case is repeated in the 40-year cash flow.")
    warnings.append("Screening results remain unassured until independently reconciled and reviewed by engineering and project-finance specialists.")

    scenario_json = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    audit = {
        "model_version": MODEL_VERSION,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_sha256": hashlib.sha256(scenario_json.encode("utf-8")).hexdigest(),
        "financial_basis": econ_cfg.get("financial_basis", "real"),
        "price_year": econ_cfg.get("price_year"),
        "warnings": warnings,
    }

    result = {"scenario_name": cfg["name"], "input": cfg, "headline": headline, "financial": financial,
              "grant": grant_result, "om_detail": om_detail,
              "audit": audit,
              "network_detail": network_detail,
              "delivered_temperature": delivered_temperature,
              "counterfactual": counterfactual, "demand": demand, "weather": weather, "network": network,
              "heat_sources": heat_sources, "cooling_sources": cooling_sources,
              "heat_dispatch": heat_dispatch, "cooling_dispatch": cooling_dispatch,
              "thermal_storage": thermal_storage, "capex": capex}
    result["screening"] = evaluate_screening(result)
    return result

def comparison_table(results):
    cols = ["system_type", "annual_heat_demand_MWh", "annual_cooling_demand_MWh", "capex_total_GBP", "annual_total_opex_GBP", "levelised_energy_service_GBP_per_kWh", "annual_carbon_tCO2", "annual_unmet_demand_MWh", "annual_unmet_cooling_MWh"]
    return pd.DataFrame([{"scenario": r["scenario_name"], **{c: r["headline"][c] for c in cols}} for r in results])
