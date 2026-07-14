"""UI-ready runner for 2-pipe heating and 4-pipe heating/cooling screening cases."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import synthesise_network
from optimisation.dispatch import run_dispatch
from network.network import size_network_from_demand
from economics.CAPEX import aggregate_capex
from economics.OPEX import annual_om_cost_GBP
from economics.metrics import (levelised_cost_of_heat_GBP_per_kWh, npv, irr, simple_payback_years,
                               discounted_payback_years, aggregate_counterfactual,
                               counterfactual_gas_boiler_dispatch, counterfactual_individual_ac_dispatch,
                               annual_revenue_GBP, discounted_cash_flow_series)


def _cumulative_position_escalated(capex_GBP, annual_cashflow_GBP, life_years, rate, annual_escalation=0.0):
    """Cumulative cash position with annual escalation of the cash flow.
    Each year's cash flow grows by (1 + annual_escalation) compound."""
    years = np.arange(1, life_years + 1)
    escalated = annual_cashflow_GBP * (1 + annual_escalation) ** years
    if rate > 0:
        discounted = escalated / (1 + rate) ** years
    else:
        discounted = escalated
    cum = np.concatenate([[0.0], np.cumsum(discounted)])
    return [round(v - capex_GBP, 0) for v in cum]


def _cumulative_position(capex_GBP, annual_cashflow_GBP, life_years, rate):
    """Cumulative cash position by year: [-capex at year 0, then + each
    year's (optionally discounted) cash flow] — the series the payback
    line chart plots. rate=0 gives the undiscounted (simple) track."""
    flows = discounted_cash_flow_series(annual_cashflow_GBP, life_years, rate)
    return [round(v, 0) for v in (-capex_GBP + np.concatenate([[0.0], np.cumsum(flows)]))]
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
    dc_by_position = {}     # position in `configs` -> (DataCentre object, its config dict)
    booster_cfgs = []       # (position in `configs`, config dict)

    for i, cfg in enumerate(configs):
        if cfg["type"] == "booster_heat_pump":
            booster_cfgs.append((i, cfg))
            continue
        cls = HEAT_CLASSES[cfg["type"]]
        obj = (cls.from_preset(cfg["preset"], weather_df=weather, **_overrides(cfg))
               if cfg["type"] in {"ashp", "data_centre", "efw_chp"}
               else cls.from_preset(cfg["preset"], **_overrides(cfg)))
        if cfg["type"] == "data_centre":
            dc_by_position[i] = (obj, cfg)
        else:
            sources.append(obj)

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
        sources.append(BoosterHeatPump.from_preset(
            cfg["preset"], source_temp_C_hourly=dc.supply_temp_C,
            sink_temp_C=sink_temp_C, **_overrides(cfg)))
    return sources

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
    network_detail = None            # per-segment breakdown for tree mode (UI table)
    heat_loss_MWh = cool_gain_MWh = 0.0
    heat_loss_kW_hourly = cool_gain_kW_hourly = 0.0   # scalar 0 broadcasts fine
    network_capex = 0.0

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

    heat_sources = build_heat_sources(cfg["sources"], weather, net_cfg["heat_flow_temp_C"])
    heat_dispatch = run_dispatch(demand["total_heat_kW"] + heat_loss_kW_hourly, heat_sources, duty="heat")
    heat_summary = heat_dispatch.summary()

    cooling_sources, cooling_dispatch, cooling_summary = [], None, None
    if include_cooling:
        cooling_sources = build_cooling_sources(cfg["cooling_sources"], weather)
        cooling_dispatch = run_dispatch(demand["total_cooling_kW"] + cool_gain_kW_hourly, cooling_sources, duty="cool")
        cooling_summary = cooling_dispatch.summary()

    all_sources = heat_sources + cooling_sources
    capex = aggregate_capex(sources=all_sources)
    total_capex = capex["grand_total_GBP"] + network_capex
    n_buildings = len(demand["nodes"])
    # Per-technology O&M (replaces the flat CHDU 1% of total CAPEX)
    from economics.om_rates import total_annual_om_GBP as _tech_om
    om_detail = _tech_om(all_sources, network_capex)
    om = om_detail["total_GBP"]
    energy_cost = heat_summary["total_annual_opex_GBP"] + (cooling_summary["total_annual_opex_GBP"] if cooling_summary else 0.0)
    annual_opex = energy_cost + om

    # GHNF grant (optional — reduces effective CAPEX for financial metrics)
    from economics.grant import apply_ghnf_grant
    grant_cfg = cfg["economics"].get("ghnf_grant", {})
    grant_result = None
    effective_capex = total_capex
    if grant_cfg.get("enabled", False):
        grant_result = apply_ghnf_grant(
            total_capex_GBP=total_capex,
            network_capex_GBP=network_capex,
            source_capex_GBP=capex["by_category"]["sources_GBP"],
            grant_rate=grant_cfg.get("rate", 0.40),
        )
        effective_capex = grant_result["net_capex_GBP"]

    counterfactual = None
    financial = {}
    life, rate = cfg["economics"]["project_lifetime_years"], cfg["economics"]["discount_rate"]
    elec_esc = cfg["economics"].get("electricity_escalation_pct", 1.5) / 100.0
    gas_esc = cfg["economics"].get("gas_escalation_pct", 1.0) / 100.0
    if cfg["economics"]["counterfactual"] != "none":
        counterfactual = _combined_counterfactual(demand["nodes"], weather, include_cooling, cfg["economics"]["om_rate"])
        incremental_capex = effective_capex - counterfactual["total_capex_GBP"]
        annual_saving = counterfactual["total_annual_opex_GBP"] - annual_opex
        financial = {"counterfactual": cfg["economics"]["counterfactual"],
                     "counterfactual_capex_GBP": counterfactual["total_capex_GBP"],
                     "counterfactual_annual_opex_GBP": counterfactual["total_annual_opex_GBP"],
                     "incremental_capex_GBP": round(incremental_capex, 0),
                     "annual_avoided_cost_GBP": round(annual_saving, 0),
                     "npv_vs_counterfactual_GBP": round(npv(incremental_capex, annual_saving, life, rate), 0),
                     "irr_vs_counterfactual": irr(incremental_capex, annual_saving, life),
                     "simple_payback_years": simple_payback_years(incremental_capex, annual_saving),
                     "discounted_payback_years": discounted_payback_years(incremental_capex, annual_saving, life, rate),
                     # Year-by-year cumulative cash position (year 0 = the
                     # incremental CAPEX outlay) — this is what the UI's
                     # payback-over-project-life line chart plots. Both a
                     # discounted and an undiscounted track are provided so
                     # the chart can show simple vs discounted payback on
                     # the same axes.
                     "cashflow_years": list(range(0, life + 1)),
                     "cumulative_discounted_GBP": _cumulative_position_escalated(incremental_capex, annual_saving, life, rate, elec_esc),
                     "cumulative_undiscounted_GBP": _cumulative_position_escalated(incremental_capex, annual_saving, life, 0.0, elec_esc)}

    # INVESTOR view — a genuinely different question from the avoided-cost
    # comparison above. The avoided-cost NPV asks "is the district scheme
    # cheaper for society than every building going individual?" and mixes
    # the customers' avoided retail bills with the scheme's own costs. An
    # investor (Dalkia) instead asks "does the scheme's own REVENUE cover
    # its own CAPEX and OPEX at my cost of capital?". Revenue uses the
    # existing gas-parity tariff mechanism (economics.tariffs.
    # customer_heat_revenue_GBP — Ofgem cap unit rate + one standing charge
    # per connected building), which until now was built but never called.
    heat_delivered_for_revenue_MWh = demand["annual_heat_MWh"] + demand["annual_dhw_MWh"]
    cool_delivered_for_revenue_MWh = demand["annual_cool_MWh"] if include_cooling else 0.0
    revenue = annual_revenue_GBP(heat_delivered_for_revenue_MWh + cool_delivered_for_revenue_MWh, n_buildings)
    investor_cashflow = revenue["total_revenue_GBP"] - annual_opex
    financial["investor"] = {
        "annual_revenue_GBP": revenue["total_revenue_GBP"],
        "revenue_basis": "Gas-parity tariff (Ofgem cap unit rate on all delivered kWh incl. cooling, "
                         "plus one standing charge per connected building)",
        "annual_net_cashflow_GBP": round(investor_cashflow, 0),
        "npv_GBP": round(npv(effective_capex, investor_cashflow, life, rate), 0),
        "irr": irr(effective_capex, investor_cashflow, life),
        "simple_payback_years": simple_payback_years(effective_capex, investor_cashflow),
        "discounted_payback_years": discounted_payback_years(effective_capex, investor_cashflow, life, rate),
        "cashflow_years": list(range(0, life + 1)),
        "cumulative_discounted_GBP": _cumulative_position_escalated(effective_capex, investor_cashflow, life, rate, elec_esc),
        "cumulative_undiscounted_GBP": _cumulative_position_escalated(effective_capex, investor_cashflow, life, 0.0, elec_esc),
    }

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
        "capex_total_GBP": round(total_capex, 0), "effective_capex_GBP": round(effective_capex, 0), "capex_sources_GBP": capex["by_category"]["sources_GBP"], "capex_network_GBP": round(network_capex, 0),
        "annual_energy_cost_GBP": round(energy_cost, 0), "annual_om_cost_GBP": round(om, 0), "annual_total_opex_GBP": round(annual_opex, 0),
        # LCOH denominator FIX: previously divided by heat GENERATED
        # (building demand + network losses) rather than heat DELIVERED to
        # customers. The gov.uk definition this project cites divides by
        # "total energy demand" — the delivered figure — and the sibling
        # levelised_energy_service metric below already did. Losses now
        # correctly INCREASE LCOH (same cost, less useful heat) instead of
        # partially hiding inside the denominator.
        "lcoh_GBP_per_kWh": round(levelised_cost_of_heat_GBP_per_kWh(total_capex, annual_opex, heat_delivered * 1000, cfg["economics"]["project_lifetime_years"]), 4),
        "levelised_energy_service_GBP_per_kWh": round(levelised_cost_of_heat_GBP_per_kWh(total_capex, annual_opex, total_delivered * 1000, cfg["economics"]["project_lifetime_years"]), 4),
        "annual_carbon_tCO2": round(carbon_t, 1), "carbon_intensity_kgCO2_per_kWh": round(carbon_t * 1000 / total_delivered, 4) if total_delivered else 0.0, "carbon_intensity_kgCO2_per_kWh_service": round(carbon_t * 1000 / total_delivered, 4) if total_delivered else 0.0,
    }
    if network_detail is not None:
        headline["network_total_length_m"] = round(network.total_length_m(), 0)
    elif net_cfg["mode"] == "generic_length":
        headline["network_total_length_m"] = round(float(net_cfg["length_m"]), 0)
    else:
        headline["network_total_length_m"] = 0.0

    return {"scenario_name": cfg["name"], "input": cfg, "headline": headline, "financial": financial,
            "grant": grant_result, "om_detail": om_detail,
            "network_detail": network_detail,
            "counterfactual": counterfactual, "demand": demand, "weather": weather, "network": network,
            "heat_sources": heat_sources, "cooling_sources": cooling_sources,
            "heat_dispatch": heat_dispatch, "cooling_dispatch": cooling_dispatch, "capex": capex}

def comparison_table(results):
    cols = ["system_type", "annual_heat_demand_MWh", "annual_cooling_demand_MWh", "capex_total_GBP", "annual_total_opex_GBP", "levelised_energy_service_GBP_per_kWh", "annual_carbon_tCO2", "annual_unmet_demand_MWh", "annual_unmet_cooling_MWh"]
    return pd.DataFrame([{"scenario": r["scenario_name"], **{c: r["headline"][c] for c in cols}} for r in results])
