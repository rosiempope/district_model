"""Streamlit UI for the district heating/cooling screening model.

Run from the repository root:
    streamlit run app.py

The UI only creates JSON-compatible scenario dictionaries.  The same contract
can later be submitted by a React/FastAPI application to `run_scenario()`.
"""
from __future__ import annotations

import copy
import io
import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from profiles.demand_synthesis import BUILDING_TYPES
from components.ASHP import ASHP_PRESETS
from components.EfW import EFW_PRESETS
from components.datacentre_source import DC_PRESETS
from components.booster_heat_pump import BOOSTER_PRESETS
from components.peak_demand_option import GAS_BOILER_PRESETS, ELECTRIC_BOILER_PRESETS
from components.chiller import CHILLER_PRESETS
from scenarios.scenario_runner import run_scenario
from optimisation.auto_size import recommend_sizing, DIVERSITY_FACTORS
from scenarios.scenario_schema import apply_defaults, validate_scenario
from scenarios.worked_scenarios import WORKED_SCENARIOS

st.set_page_config(page_title="District energy screening tool", page_icon="⚡", layout="wide")

HEAT_PRESETS = {
    "ashp": ASHP_PRESETS,
    "gas_boiler": GAS_BOILER_PRESETS,
    "electric_boiler": ELECTRIC_BOILER_PRESETS,
    "efw_chp": EFW_PRESETS,
    "data_centre": DC_PRESETS,
    "booster_heat_pump": BOOSTER_PRESETS,
}
HEAT_TYPE_LABELS = {
    "ashp": "Air-source heat pump",
    "gas_boiler": "Gas boiler",
    "electric_boiler": "Electric boiler",
    "efw_chp": "Energy-from-waste heat export",
    "data_centre": "Data-centre waste heat",
    "booster_heat_pump": "Booster heat pump",
}
COOL_PRESETS = {"air_cooled_chiller": CHILLER_PRESETS}


def _safe_name(value: str) -> str:
    return value.replace("_", " ").title()


def _template_map() -> dict[str, dict[str, Any]]:
    return {scenario["name"]: copy.deepcopy(scenario) for scenario in WORKED_SCENARIOS}


def _new_scenario() -> dict[str, Any]:
    base = copy.deepcopy(WORKED_SCENARIOS[2])
    base["name"] = "New screening scenario"
    return apply_defaults(base)


def _clear_editor_widget_state() -> None:
    """Clear widget values when loading a different JSON/template scenario."""
    prefixes = ("heat_", "cool_", "remove_heat_", "remove_cool_", "building_editor", "tree_seg_", "remove_seg_")
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes) or key == "building_editor":
            del st.session_state[key]


def init_state() -> None:
    if "scenario" not in st.session_state:
        st.session_state.scenario = _new_scenario()
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def scenario_to_json_bytes(scenario: dict[str, Any]) -> bytes:
    return json.dumps(scenario, indent=2).encode("utf-8")


def result_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    h = result["headline"]
    f = result.get("financial", {})
    return {
        "Scenario": result["scenario_name"],
        "System": h["system_type"],
        "CAPEX (£m)": h["capex_total_GBP"] / 1e6,
        "Annual OPEX (£m)": h["annual_total_opex_GBP"] / 1e6,
        "LCO service (£/kWh)": h["levelised_energy_service_GBP_per_kWh"],
        "Carbon (gCO₂e/kWh)": h["carbon_intensity_kgCO2_per_kWh_service"] * 1000,
        "Unmet heat (MWh)": h["annual_unmet_demand_MWh"],
        "Unmet cooling (MWh)": h["annual_unmet_cooling_MWh"],
        "NPV vs counterfactual (£m)": (f.get("npv_vs_counterfactual_GBP") or 0) / 1e6,
        "Discounted payback (yrs)": f.get("discounted_payback_years"),
    }


def _source_editor(prefix: str, source: dict[str, Any], allowed: dict[str, dict], cooling: bool = False) -> dict[str, Any]:
    """Render one source editor and return a serialisable source config."""
    cols = st.columns([2.2, 2.0, 1.3, 1.2, 1.2])
    types = list(allowed)
    current_type = source.get("type", types[0])
    type_index = types.index(current_type) if current_type in types else 0
    source_type = cols[0].selectbox("Technology", types, index=type_index,
                                    format_func=lambda x: _safe_name(x), key=f"{prefix}_type")
    presets = list(allowed[source_type])
    current_preset = source.get("preset", presets[0])
    preset_index = presets.index(current_preset) if current_preset in presets else 0
    preset = cols[1].selectbox("Preset", presets, index=preset_index, key=f"{prefix}_preset")
    name = cols[2].text_input("Name", value=source.get("name", _safe_name(source_type)), key=f"{prefix}_name")
    capacity = cols[3].number_input("Total capacity (MW)", min_value=0.01,
                                     value=number(source.get("capacity_MW"), 1.0), step=0.1,
                                     key=f"{prefix}_capacity")
    if source_type in {"ashp", "booster_heat_pump", "air_cooled_chiller"}:
        n_units = int(cols[4].number_input("Units", min_value=1, value=int(source.get("n_units", 1)), step=1,
                                            key=f"{prefix}_units"))
    else:
        cols[4].caption("Single total-capacity asset")
        n_units = None
    edited = {"type": source_type, "preset": preset, "name": name, "capacity_MW": float(capacity)}
    if n_units is not None:
        edited["n_units"] = n_units

    with st.expander("Technology-specific settings", expanded=False):
        if source_type == "ashp":
            edited["flow_temp_C"] = st.number_input("ASHP flow temperature (°C)", min_value=30.0, max_value=90.0,
                                                       value=number(source.get("flow_temp_C"), 70.0), step=1.0,
                                                       key=f"{prefix}_flow")
        if source_type == "data_centre":
            edited["dispatch_direct"] = st.checkbox(
                "Dispatch raw waste heat directly (no booster)",
                value=bool(source.get("dispatch_direct", False)), key=f"{prefix}_direct",
                help="Only valid if the network flow temperature (section 3) is 35°C or below — "
                     "data-centre waste heat is recovered at ~28-35°C and normally needs a booster "
                     "heat pump to reach a typical 70°C network flow temperature. Leave this off and "
                     "add a booster source below with 'Data-centre source position' pointing at this "
                     "source for a standard network.",
            )
        if source_type == "booster_heat_pump":
            edited["depends_on"] = int(st.number_input(
                "Data-centre source position (counting ALL heating sources above, starting at 0 — must point at a 'Data-centre waste heat' source)",
                min_value=0, value=int(source.get("depends_on", 0)), step=1, key=f"{prefix}_depends"))
        if cooling:
            edited["chilled_water_temp_C"] = st.number_input("Chilled-water flow temperature (°C)", min_value=2.0, max_value=15.0,
                                                               value=number(source.get("chilled_water_temp_C"), 6.0), step=0.5,
                                                               key=f"{prefix}_chw")
    return edited


def edit_scenario() -> dict[str, Any]:
    scenario = copy.deepcopy(apply_defaults(st.session_state.scenario))

    st.subheader("1. Scenario, climate and economics")
    c1, c2, c3, c4 = st.columns(4)
    scenario["name"] = c1.text_input("Scenario name", value=scenario["name"])
    climate_options = ["baseline", "2050_central", "2050_high"]
    scenario["climate_scenario"] = c2.selectbox("Climate scenario", climate_options,
                                                  index=climate_options.index(scenario["climate_scenario"]))
    econ = scenario["economics"]
    econ["project_lifetime_years"] = int(c3.number_input("Project lifetime (years)", min_value=1, max_value=60,
                                                           value=int(econ["project_lifetime_years"]), step=1))
    econ["discount_rate"] = c4.number_input("Discount rate", min_value=0.0, max_value=0.5,
                                              value=number(econ["discount_rate"], 0.105), step=0.005,
                                              format="%.3f")
    c5, c6 = st.columns(2)
    econ["om_rate"] = c5.number_input("O&M rate (legacy flat)", min_value=0.0, max_value=0.2,
                                        value=number(econ["om_rate"], 0.01), step=0.001, format="%.3f",
                                        help="Flat rate used for counterfactual only; the scheme itself uses per-technology rates automatically")
    c7, c8, c9 = st.columns(3)
    grant_cfg = econ.setdefault("ghnf_grant", {"enabled": False, "rate": 0.40})
    grant_cfg["enabled"] = c7.toggle("GHNF grant", value=bool(grant_cfg.get("enabled", False)),
                                      help="Green Heat Network Fund — reduces effective CAPEX by up to 50%")
    if grant_cfg["enabled"]:
        grant_cfg["rate"] = c8.number_input("Grant rate", min_value=0.0, max_value=0.50,
                                             value=number(grant_cfg.get("rate"), 0.40), step=0.05, format="%.2f",
                                             help="GHNF awards typically 30-50% of eligible CAPEX")
    c9_esc1, c9_esc2 = st.columns(2)
    econ["electricity_escalation_pct"] = c9_esc1.number_input("Electricity escalation (%/yr)", min_value=0.0, max_value=5.0,
                                                              value=number(econ.get("electricity_escalation_pct"), 1.5), step=0.5,
                                                              help="Real-terms annual price escalation for energy costs in the NPV cash-flow series")
    econ["gas_escalation_pct"] = c9_esc2.number_input("Gas escalation (%/yr)", min_value=0.0, max_value=5.0,
                                                       value=number(econ.get("gas_escalation_pct"), 1.0), step=0.5)

    st.subheader("2. Buildings and demand")
    st.caption("Enter a positive floor area for commercial buildings. For residential archetypes, you may use either floor area or unit count.")
    buildings = pd.DataFrame(scenario["demand"]["buildings"])
    for col in ["name", "type", "floor_area_m2", "units"]:
        if col not in buildings:
            buildings[col] = None
    buildings = buildings[["name", "type", "floor_area_m2", "units"]]
    edited_buildings = st.data_editor(
        buildings,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Building name", required=True),
            "type": st.column_config.SelectboxColumn("Building type", options=list(BUILDING_TYPES), required=True),
            "floor_area_m2": st.column_config.NumberColumn("Floor area (m²)", min_value=0.0, step=100.0),
            "units": st.column_config.NumberColumn("Dwellings / units", min_value=0.0, step=1.0),
        },
        key="building_editor",
    )
    scenario["demand"]["buildings"] = [
        {k: (None if pd.isna(v) else v) for k, v in row.items() if not (k in {"floor_area_m2", "units"} and pd.isna(v))}
        for row in edited_buildings.to_dict("records")
    ]

    st.subheader("3. Network configuration")
    net = scenario["network"]
    net["mode"] = st.radio(
        "Network approach",
        ["generic_length", "tree", "none"],
        index=["generic_length", "tree", "none"].index(net.get("mode", "generic_length")),
        format_func={"generic_length": "Single trunk (total length)", "tree": "Tree topology (branch lengths)", "none": "No network"}.get,
        horizontal=True,
    )

    if net["mode"] == "generic_length":
        t1, t2, t3, t4, t5 = st.columns(5)
        net["length_m"] = t1.number_input("Total network length (m)", min_value=0.0, value=number(net.get("length_m"), 3000.0), step=100.0, help="Total route length of the main and branch pipes. Used to size one representative trunk per duty.")
        heat_col_temp, return_col_temp, cool_col_1, cool_col_2 = t2, t3, t4, t5
    elif net["mode"] == "tree":
        st.caption("Define each pipe segment. Every building must be connected. Add junction nodes to model branching.")
        building_names_for_tree = [""] + [b.get("name","") for b in scenario.get("demand",{}).get("buildings",[]) if b.get("name")]
        segs = net.setdefault("segments", [])
        from scenarios.scenario_schema import TREE_ROOT_ID
        kept_segs = []
        for i, seg in enumerate(segs):
            with st.container(border=True):
                sc1, sc2, sc3, sc4, sc5 = st.columns([2, 2, 1.5, 2, 0.8])
                nid = sc1.text_input("Segment ID", value=seg.get("node_id",""), key=f"tree_seg_{i}_id",
                                     help="Unique label, e.g. J1, B2, MAIN-S")
                pid = sc2.text_input("Connects to (parent ID or EC)", value=seg.get("parent_id", TREE_ROOT_ID), key=f"tree_seg_{i}_pid",
                                     help=f"Use '{TREE_ROOT_ID}' to connect directly to the energy centre")
                length = sc3.number_input("Length (m)", min_value=1.0, value=float(seg.get("length_m", 100.0)), step=10.0, key=f"tree_seg_{i}_len")
                bld_idx = building_names_for_tree.index(seg.get("building","")) if seg.get("building") in building_names_for_tree else 0
                bld = sc4.selectbox("Serves building", building_names_for_tree, index=bld_idx, key=f"tree_seg_{i}_bld",
                                    help="Leave blank for a junction segment with no building at the end")
                remove_seg = sc5.button("✕", key=f"remove_seg_{i}", help="Remove this segment")
                if remove_seg:
                    st.session_state.scenario["network"]["segments"].pop(i)
                    st.rerun()
                kept_segs.append({"node_id": nid, "parent_id": pid, "length_m": float(length), "building": bld if bld else None})
        net["segments"] = kept_segs
        if st.button("+ Add segment"):
            st.session_state.scenario["network"].setdefault("segments", []).append(
                {"node_id": f"S{len(segs)+1}", "parent_id": TREE_ROOT_ID, "length_m": 100.0, "building": None}
            )
            st.rerun()
        n1_temp, n2_temp = st.columns(2)
        heat_col_temp, return_col_temp = n1_temp, n2_temp
        cool_col_1, cool_col_2 = st.columns(2)
    else:
        heat_col_temp, return_col_temp, cool_col_1, cool_col_2 = st.columns(4)

    net["heat_flow_temp_C"] = heat_col_temp.number_input("Heat flow (°C)", min_value=30.0, max_value=100.0, value=number(net["heat_flow_temp_C"], 70.0), step=1.0)
    net["heat_return_temp_C"] = return_col_temp.number_input("Heat return (°C)", min_value=15.0, max_value=90.0, value=number(net["heat_return_temp_C"], 40.0), step=1.0)
    net["include_cooling"] = cool_col_1.toggle("Include cooling / 4-pipe", value=bool(net["include_cooling"]))
    if net["include_cooling"]:
        net["cool_flow_temp_C"] = cool_col_2.number_input("Cooling flow (°C)", min_value=2.0, max_value=15.0, value=number(net.get("cool_flow_temp_C"), 6.0), step=0.5)
        cool_ret_c1, cool_ret_c2 = st.columns(2)
        net["cool_return_temp_C"] = cool_ret_c1.number_input("Cooling return (°C)", min_value=4.0, max_value=22.0, value=number(net.get("cool_return_temp_C"), 12.0), step=0.5)
        econ["counterfactual"] = "individual_gas_and_ac"
        st.info("The fair counterfactual is now individual gas boilers plus individual AC.")
    else:
        counterfactual_options = ["individual_gas", "none"]
        econ["counterfactual"] = st.selectbox("Counterfactual", counterfactual_options,
                                                index=counterfactual_options.index(econ.get("counterfactual", "individual_gas")) if econ.get("counterfactual") in counterfactual_options else 0,
                                                format_func=lambda x: "Individual gas boilers" if x == "individual_gas" else "No counterfactual")

    st.subheader("4. Heating technologies")
    auto_col1, auto_col2 = st.columns([3, 1])
    auto_col1.caption("Order matters: sources are dispatched in the order shown. Put low-cost/low-carbon base-load sources before peak boilers.")
    if auto_col2.button("Auto-size from demand", type="secondary", help="Let the model recommend technology capacities based on the demand profile above"):
        import copy as _cp
        _auto_sc = _cp.deepcopy(scenario)
        try:
            from profiles.demand_synthesis import synthesise_network as _synth
            from profiles.climate_scenarios import apply_climate_scenario as _apply_clim
            from scenarios.scenario_runner import load_weather as _load_w
            _w = _apply_clim(_load_w(), _auto_sc["climate_scenario"])
            _d = _synth(_w, {"demand_nodes": _auto_sc["demand"]["buildings"]})
            _tech_types = list({s.get("type","ashp") for s in _auto_sc.get("sources", [])}) or ["ashp", "gas_boiler"]
            _inc_cool = bool(net.get("include_cooling"))
            _bld_types = [b.get("type") for b in _auto_sc["demand"]["buildings"]]
            rec = recommend_sizing(
                demand_kW=_d["total_heat_kW"], peak_demand_kW=_d["peak_heat_kW"],
                technology_types=_tech_types, weather_df=_w,
                network_flow_temp_C=net["heat_flow_temp_C"],
                include_cooling=_inc_cool,
                cooling_demand_kW=_d["total_cooling_kW"] if _inc_cool else None,
                peak_cooling_kW=_d["peak_cool_kW"] if _inc_cool else 0.0,
                n_buildings=len(_auto_sc["demand"]["buildings"]),
                building_types=_bld_types,
            )
            # Apply recommended sources to the scenario
            new_heat_sources = []
            # UNIT_TYPES in the runner (ashp, booster_heat_pump, air_cooled_chiller)
            # map capacity_MW → unit_capacity_MW in _overrides(). If we pass both
            # capacity_MW and n_units, the runner treats capacity_MW as the PER-UNIT
            # size (giving n × capacity_MW total, which is wrong). For these types,
            # compute the per-unit size explicitly.
            _unit_types = {"ashp", "booster_heat_pump", "air_cooled_chiller"}
            for rs in rec["sources"]:
                stype = rs["type"]
                type_presets = HEAT_PRESETS.get(stype, {})
                preset = list(type_presets.keys())[0] if type_presets else "ealing_phase1"
                s_cfg = {"type": stype, "preset": preset, "name": stype.replace("_", " ").title(),
                         "capacity_MW": rs["capacity_MW"]}
                if "n_units" in rs and stype in _unit_types:
                    # Runner expects capacity_MW as the per-unit size for these types
                    s_cfg["capacity_MW"] = round(rs["capacity_MW"] / rs["n_units"], 3)
                    s_cfg["n_units"] = rs["n_units"]
                elif "n_units" in rs:
                    s_cfg["n_units"] = rs["n_units"]
                if "flow_temp_C" in rs: s_cfg["flow_temp_C"] = rs["flow_temp_C"]
                if "depends_on" in rs: s_cfg["depends_on"] = rs["depends_on"]
                if "dispatch_direct" in rs: s_cfg["dispatch_direct"] = rs["dispatch_direct"]
                s_cfg["rationale"] = rs.get("rationale", "")
                new_heat_sources.append(s_cfg)
            st.session_state.scenario["sources"] = new_heat_sources
            if _inc_cool and rec.get("cooling_sources"):
                new_cool = []
                for cs in rec["cooling_sources"]:
                    new_cool.append({"type": cs["type"], "preset": "generic_2MW_bank",
                                     "name": "Central chiller bank", "capacity_MW": cs["capacity_MW"],
                                     "n_units": cs.get("n_units", 1)})
                st.session_state.scenario["cooling_sources"] = new_cool
            # CRITICAL: clear the cached widget values for source editors,
            # otherwise Streamlit's own widget state (heat_0_type, heat_0_capacity,
            # etc.) from the PREVIOUS render overwrites the new sources on rerun —
            # the widgets show old values, _source_editor returns those, and
            # line `scenario["sources"] = retained_heat` silently reverts the change.
            _clear_editor_widget_state()
            st.session_state["_auto_size_notes"] = "\n\n".join(rec["sizing_notes"])
            st.rerun()
        except Exception as exc:
            st.exception(exc)
    # Show auto-size rationale if it just ran (survives the rerun via session state)
    if st.session_state.get("_auto_size_notes"):
        st.info(st.session_state.pop("_auto_size_notes"))
    heat_sources = scenario.get("sources", [])
    retained_heat = []
    for i, source in enumerate(heat_sources):
        with st.container(border=True):
            head, remove_col = st.columns([5, 1])
            head.markdown(f"**Heating source {i + 1}**")
            remove = remove_col.button("Remove", key=f"remove_heat_{i}")
            if remove:
                st.session_state.scenario["sources"].pop(i)
                st.rerun()
            retained_heat.append(_source_editor(f"heat_{i}", source, HEAT_PRESETS))
    if st.button("Add heating source"):
        st.session_state.scenario.setdefault("sources", []).append({"type": "ashp", "preset": "ealing_phase1", "name": "ASHP bank", "capacity_MW": 1.0, "n_units": 1, "flow_temp_C": net["heat_flow_temp_C"]})
        st.rerun()
    scenario["sources"] = retained_heat

    if net["include_cooling"]:
        st.subheader("5. Cooling technologies")
        cooling_sources = scenario.get("cooling_sources", [])
        retained_cool = []
        for i, source in enumerate(cooling_sources):
            with st.container(border=True):
                head, remove_col = st.columns([5, 1])
                head.markdown(f"**Cooling source {i + 1}**")
                remove = remove_col.button("Remove", key=f"remove_cool_{i}")
                if remove:
                    st.session_state.scenario["cooling_sources"].pop(i)
                    st.rerun()
                retained_cool.append(_source_editor(f"cool_{i}", source, COOL_PRESETS, cooling=True))
        if st.button("Add cooling source"):
            st.session_state.scenario.setdefault("cooling_sources", []).append({"type": "air_cooled_chiller", "preset": "generic_2MW_bank", "name": "Central chiller bank", "capacity_MW": 2.0, "n_units": 1, "chilled_water_temp_C": net["cool_flow_temp_C"]})
            st.rerun()
        scenario["cooling_sources"] = retained_cool
    else:
        scenario["cooling_sources"] = []

    return scenario


def _payback_chart(years, cumulative_disc, cumulative_undisc, title):
    """Line chart of cumulative cash position (discounted + undiscounted)."""
    import altair as alt
    disc_df   = pd.DataFrame({"Year": years, "Value (£m)": [v/1e6 for v in cumulative_disc],   "Track": "Discounted"})
    undisc_df = pd.DataFrame({"Year": years, "Value (£m)": [v/1e6 for v in cumulative_undisc], "Track": "Undiscounted"})
    df = pd.concat([disc_df, undisc_df], ignore_index=True)
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="red", strokeDash=[4, 3], strokeWidth=1)
        .encode(y="y:Q")
    )
    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Year:Q", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Value (£m):Q", title="Cumulative cash position (£m)"),
            color=alt.Color("Track:N", scale=alt.Scale(
                domain=["Discounted", "Undiscounted"],
                range=["#1f77b4", "#aec7e8"])),
            tooltip=["Year:Q", "Track:N", alt.Tooltip("Value (£m):Q", format=".2f")],
        )
    )
    return (line + zero_line).properties(title=title, height=280)


def show_result(result: dict[str, Any]) -> None:
    h, f = result["headline"], result.get("financial", {})
    st.success(f"Completed: {result['scenario_name']}")

    # -- Headline KPIs --
    st.subheader("Headline result")
    m = st.columns(6)
    m[0].metric("System", "4-pipe" if h["system_type"].startswith("4_") else "2-pipe")
    m[1].metric("Total CAPEX", f"\u00a3{h['capex_total_GBP']/1e6:.2f}m")
    m[2].metric("Annual OPEX", f"\u00a3{h['annual_total_opex_GBP']/1e6:.2f}m")
    m[3].metric("LCoH", f"\u00a3{h['levelised_energy_service_GBP_per_kWh']:.3f}/kWh")
    m[4].metric("Carbon", f"{h['carbon_intensity_kgCO2_per_kWh_service']*1000:.0f} gCO\u2082e/kWh")
    m[5].metric("Network length", f"{h.get('network_total_length_m', 0):,.0f} m")

    if h["annual_unmet_demand_MWh"] > 0.5:
        st.warning(f"\u26a0\ufe0f Unmet heat: {h['annual_unmet_demand_MWh']:.1f} MWh/yr")
    if h["annual_unmet_cooling_MWh"] > 0.5:
        st.warning(f"\u26a0\ufe0f Unmet cooling: {h['annual_unmet_cooling_MWh']:.1f} MWh/yr")

    # -- Payback line charts --
    st.subheader("Project cash position over life")
    view_tab1, view_tab2 = st.tabs(["Society vs counterfactual", "Investor (revenue-based)"])

    with view_tab1:
        if f and f.get("cashflow_years"):
            payback = f.get("discounted_payback_years")
            pb_str = "No payback within appraisal life" if payback is None else f"{payback:.1f} years"
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avoided-cost NPV", f"\u00a3{f.get('npv_vs_counterfactual_GBP', 0)/1e6:.2f}m")
            col_b.metric("Discounted payback", pb_str)
            irr_val = f.get("irr_vs_counterfactual")
            col_c.metric("IRR", "\u2014" if irr_val is None else f"{irr_val*100:.1f}%")
            ch = _payback_chart(
                f["cashflow_years"],
                f["cumulative_discounted_GBP"],
                f["cumulative_undiscounted_GBP"],
                "Cumulative position vs. everyone-goes-individual counterfactual",
            )
            st.altair_chart(ch, use_container_width=True)
            st.caption("Red line = break-even. Positive = district scheme cheaper than individual systems.")
        else:
            st.info("Select a counterfactual in section 1 to enable this view.")

    with view_tab2:
        inv = f.get("investor", {}) if f else {}
        if inv:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Investor NPV", f"\u00a3{inv.get('npv_GBP', 0)/1e6:.2f}m")
            pb_inv = inv.get("discounted_payback_years")
            col_b.metric("Discounted payback", "No payback" if pb_inv is None else f"{pb_inv:.1f} years")
            irr_inv = inv.get("irr")
            col_c.metric("IRR", "\u2014" if irr_inv is None else f"{irr_inv*100:.1f}%")
            st.caption(f"Revenue basis: {inv.get('revenue_basis','')}")
            ch2 = _payback_chart(
                inv["cashflow_years"],
                inv["cumulative_discounted_GBP"],
                inv["cumulative_undiscounted_GBP"],
                "Cumulative investor position (scheme revenue minus OPEX vs total CAPEX)",
            )
            st.altair_chart(ch2, use_container_width=True)
        else:
            st.info("Investor view not available.")

    # -- Energy & network detail (collapsed) --
    with st.expander("Energy service detail", expanded=False):
        ec1, ec2 = st.columns(2)
        ec1.metric("Heat demand", f"{h['annual_heat_demand_MWh']:,.0f} MWh/year")
        ec1.metric("Network heat loss", f"{h['annual_network_heat_loss_MWh']:.0f} MWh/year")
        if h["annual_cooling_demand_MWh"] > 0:
            ec2.metric("Cooling demand", f"{h['annual_cooling_demand_MWh']:,.0f} MWh/year")
            ec2.metric("Network cooling gain", f"{h.get('annual_network_cooling_gain_MWh', 0):.0f} MWh/year")

    with st.expander("Source generation mix", expanded=False):
        rows = []
        for duty, values in [("Heating", h["annual_heat_by_source_MWh"]), ("Cooling", h["annual_cooling_by_source_MWh"])]:
            rows.extend({"Duty": duty, "Source": source, "MWh/year": value} for source, value in values.items())
        source_df = pd.DataFrame(rows)
        if not source_df.empty:
            st.dataframe(source_df, use_container_width=True, hide_index=True)

    with st.expander("Monthly demand profile", expanded=False):
        index = pd.DatetimeIndex(result["demand"]["datetime_index"])
        heat_m = pd.Series(result["demand"]["total_heat_kW"], index=index).groupby(index.month).sum() / 1000
        cool_m = pd.Series(result["demand"]["total_cooling_kW"], index=index).groupby(index.month).sum() / 1000
        monthly = pd.DataFrame({"Heating + DHW (MWh)": heat_m, "Cooling (MWh)": cool_m})
        monthly.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        st.bar_chart(monthly)

    # -- Grant info --
    if result.get("grant"):
        g = result["grant"]
        with st.expander(f"GHNF grant: \u00a3{g['grant_GBP']/1e6:.2f}m ({g['grant_rate']*100:.0f}%)", expanded=False):
            st.metric("Eligible CAPEX", f"\u00a3{g['eligible_capex_GBP']/1e6:.2f}m")
            st.metric("Grant amount", f"\u00a3{g['grant_GBP']/1e6:.2f}m")
            st.metric("Net CAPEX after grant", f"\u00a3{g['net_capex_GBP']/1e6:.2f}m")

    # -- Feasibility verdict --
    st.subheader("Feasibility verdict")
    inv = f.get("investor", {}) if f else {}
    inv_irr = inv.get("irr")
    inv_pb = inv.get("discounted_payback_years")
    life_yrs = result["input"]["economics"]["project_lifetime_years"]
    unmet_pct = h["annual_unmet_demand_MWh"] / max(h["annual_heat_demand_MWh"], 0.1) * 100

    verdict_parts = []
    if inv_irr is not None and inv_irr > 0.09:
        verdict_parts.append(("\u2705", f"IRR {inv_irr*100:.1f}% exceeds typical 9% hurdle rate"))
    elif inv_irr is not None:
        verdict_parts.append(("\u26a0\ufe0f", f"IRR {inv_irr*100:.1f}% is below 9% hurdle rate"))
    else:
        verdict_parts.append(("\u274c", "No positive IRR — scheme does not recover CAPEX from revenue"))
    if inv_pb is not None and inv_pb <= life_yrs:
        verdict_parts.append(("\u2705", f"Discounted payback in {inv_pb:.1f} years (within {life_yrs}-year life)"))
    else:
        verdict_parts.append(("\u274c", f"No payback within {life_yrs}-year project life"))
    if unmet_pct < 1:
        verdict_parts.append(("\u2705", "Plant capacity meets >99% of demand"))
    else:
        verdict_parts.append(("\u26a0\ufe0f", f"{unmet_pct:.1f}% of demand unmet — consider adding capacity"))

    for icon, text in verdict_parts:
        st.markdown(f"{icon} {text}")

    nd = result.get("network_detail")
    if nd:
        with st.expander("Network segment detail", expanded=True):
            st.dataframe(pd.DataFrame(nd), use_container_width=True, hide_index=True)

    # Downloads
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download scenario JSON", scenario_to_json_bytes(result["input"]), "scenario.json", "application/json")
    csv_bytes = pd.DataFrame([result_summary_row(result)]).to_csv(index=False).encode("utf-8")
    dl2.download_button("Download result CSV", csv_bytes, "scenario_result.csv", "text/csv")


def show_comparison() -> None:
    st.subheader("Scenario comparison")
    results = st.session_state.comparison_results
    if not results:
        st.info("Run a scenario, then select “Add result to comparison”, or use “Run all worked scenarios”.")
        return
    rows = [result_summary_row(r) for r in results]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    numeric_cols = [c for c in df.columns if c not in {"Scenario", "System"}]
    metric = st.selectbox("Graph metric", numeric_cols, index=numeric_cols.index("NPV vs counterfactual (£m)") if "NPV vs counterfactual (£m)" in numeric_cols else 0)
    chart = df.set_index("Scenario")[[metric]]
    st.bar_chart(chart)
    st.download_button("Download comparison CSV", df.to_csv(index=False).encode("utf-8"), "scenario_comparison.csv", "text/csv")


def main() -> None:
    init_state()
    st.title("District energy screening tool")
    st.caption("Editable 2-pipe heating and 4-pipe heating/cooling screening scenarios. Inputs remain plain JSON-compatible data for future API/UI use.")

    with st.sidebar:
        st.header("Scenario controls")
        templates = _template_map()
        selected = st.selectbox("Load example", ["New blank scenario", *templates.keys()])
        if st.button("Load selected scenario", use_container_width=True):
            _clear_editor_widget_state()
            st.session_state.scenario = _new_scenario() if selected == "New blank scenario" else apply_defaults(templates[selected])
            st.session_state.last_result = None
            st.rerun()
        st.divider()
        if st.button("Run all worked scenarios", use_container_width=True):
            with st.spinner("Running worked scenarios..."):
                st.session_state.comparison_results = [run_scenario(copy.deepcopy(s)) for s in WORKED_SCENARIOS]
            st.success("Worked scenarios added to comparison.")
        if st.button("Clear comparison", use_container_width=True):
            st.session_state.comparison_results = []

    tab_inputs, tab_result, tab_compare, tab_contract = st.tabs(["Build scenario", "Results", "Compare scenarios", "UI/API contract"])

    with tab_inputs:
        edited = edit_scenario()
        run_clicked = st.button("Validate and run scenario", type="primary", use_container_width=True)
        st.download_button("Download current input JSON", scenario_to_json_bytes(edited), "scenario_input.json", "application/json")
        if run_clicked:
            errors = validate_scenario(edited)
            if errors:
                st.error("Please correct the highlighted scenario issues:")
                for error in errors:
                    st.write(f"• {error}")
            else:
                with st.spinner("Running hourly demand, dispatch and economics..."):
                    try:
                        st.session_state.scenario = edited
                        st.session_state.last_result = run_scenario(edited)
                    except Exception as exc:  # report user-readable model exception
                        st.exception(exc)
                    else:
                        st.success("Scenario completed. Open the Results tab.")

    with tab_result:
        if st.session_state.last_result is None:
            st.info("Build and run a scenario first.")
        else:
            show_result(st.session_state.last_result)
            if st.button("Add result to comparison"):
                current_name = st.session_state.last_result["scenario_name"]
                st.session_state.comparison_results = [r for r in st.session_state.comparison_results if r["scenario_name"] != current_name]
                st.session_state.comparison_results.append(copy.deepcopy(st.session_state.last_result))
                st.success("Added to comparison.")

    with tab_compare:
        show_comparison()

    with tab_contract:
        st.markdown("""
### Future UI/API contract

This Streamlit UI sends a plain, JSON-compatible scenario dictionary to
`scenarios.scenario_runner.run_scenario()`. A future React form can submit the
same JSON to a FastAPI endpoint without changing the model layer.

**Two-pipe:** set `network.include_cooling` to `false` and use `individual_gas`.

**Four-pipe:** set `network.include_cooling` to `true`, provide at least one
cooling source, and use `individual_gas_and_ac` as the counterfactual.

The runner validates required fields before it performs any calculations.
""")
        st.code(json.dumps(apply_defaults(st.session_state.scenario), indent=2), language="json")


if __name__ == "__main__":
    main()
