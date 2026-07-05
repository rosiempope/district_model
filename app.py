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

import matplotlib.pyplot as plt
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
    prefixes = ("heat_", "cool_", "remove_heat_", "remove_cool_", "building_editor")
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
            edited["dispatch_direct"] = st.checkbox("Dispatch direct heat as well as using a booster", value=bool(source.get("dispatch_direct", False)), key=f"{prefix}_direct")
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
    econ["om_rate"] = c5.number_input("Annual O&M rate of total CAPEX", min_value=0.0, max_value=0.2,
                                        value=number(econ["om_rate"], 0.01), step=0.001, format="%.3f")

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
    n1, n2, n3, n4, n5 = st.columns(5)
    net["mode"] = n1.selectbox("Network approach", ["generic_length", "none"], index=["generic_length", "none"].index(net["mode"]))
    net["length_m"] = n2.number_input("Network length (m)", min_value=0.0, value=number(net.get("length_m"), 3000.0), step=100.0)
    net["heat_flow_temp_C"] = n3.number_input("Heat flow (°C)", min_value=30.0, max_value=100.0, value=number(net["heat_flow_temp_C"], 70.0), step=1.0)
    net["heat_return_temp_C"] = n4.number_input("Heat return (°C)", min_value=15.0, max_value=90.0, value=number(net["heat_return_temp_C"], 40.0), step=1.0)
    net["include_cooling"] = n5.toggle("Include cooling / 4-pipe", value=bool(net["include_cooling"]))
    if net["include_cooling"]:
        co1, co2 = st.columns(2)
        net["cool_flow_temp_C"] = co1.number_input("Cooling flow (°C)", min_value=2.0, max_value=15.0, value=number(net.get("cool_flow_temp_C"), 6.0), step=0.5)
        net["cool_return_temp_C"] = co2.number_input("Cooling return (°C)", min_value=4.0, max_value=22.0, value=number(net.get("cool_return_temp_C"), 12.0), step=0.5)
        econ["counterfactual"] = "individual_gas_and_ac"
        st.info("The fair counterfactual is now individual gas boilers plus individual AC.")
    else:
        counterfactual_options = ["individual_gas", "none"]
        econ["counterfactual"] = st.selectbox("Counterfactual", counterfactual_options,
                                                index=counterfactual_options.index(econ.get("counterfactual", "individual_gas")) if econ.get("counterfactual") in counterfactual_options else 0,
                                                format_func=lambda x: "Individual gas boilers" if x == "individual_gas" else "No counterfactual")

    st.subheader("4. Heating technologies")
    st.caption("Order matters: sources are dispatched in the order shown. Put low-cost/low-carbon base-load sources before peak boilers.")
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


def show_result(result: dict[str, Any]) -> None:
    h, f = result["headline"], result.get("financial", {})
    st.success(f"Completed: {result['scenario_name']}")
    st.subheader("Headline result")
    metrics = st.columns(6)
    metrics[0].metric("System", "4-pipe" if h["system_type"].startswith("4_") else "2-pipe")
    metrics[1].metric("Total CAPEX", f"£{h['capex_total_GBP']/1e6:.2f}m")
    metrics[2].metric("Annual OPEX", f"£{h['annual_total_opex_GBP']/1e6:.2f}m")
    metrics[3].metric("Carbon intensity", f"{h['carbon_intensity_kgCO2_per_kWh_service']*1000:.0f} gCO₂e/kWh")
    metrics[4].metric("Unmet heat", f"{h['annual_unmet_demand_MWh']:.1f} MWh")
    metrics[5].metric("Unmet cooling", f"{h['annual_unmet_cooling_MWh']:.1f} MWh")

    left, right = st.columns(2)
    with left:
        st.markdown("**Financial comparison**")
        if f:
            st.metric("NPV vs counterfactual", f"£{f.get('npv_vs_counterfactual_GBP', 0)/1e6:.2f}m")
            payback = f.get("discounted_payback_years")
            st.metric("Discounted payback", "No payback within appraisal life" if payback is None else f"{payback:.1f} years")
        else:
            st.info("No counterfactual selected.")
    with right:
        st.markdown("**Energy service**")
        st.metric("Heat demand", f"{h['annual_heat_demand_MWh']:,.0f} MWh/year")
        if h["annual_cooling_demand_MWh"] > 0:
            st.metric("Cooling demand", f"{h['annual_cooling_demand_MWh']:,.0f} MWh/year")
        st.metric("Levelised energy service", f"£{h['levelised_energy_service_GBP_per_kWh']:.3f}/kWh")

    st.subheader("Source energy supplied")
    rows = []
    for duty, values in [("Heating", h["annual_heat_by_source_MWh"]), ("Cooling", h["annual_cooling_by_source_MWh"])]:
        rows.extend({"Duty": duty, "Source": source, "MWh/year": value} for source, value in values.items())
    source_df = pd.DataFrame(rows)
    if not source_df.empty:
        st.bar_chart(source_df.pivot_table(index="Source", columns="Duty", values="MWh/year", aggfunc="sum", fill_value=0))
        st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.subheader("Monthly demand profile")
    index = pd.DatetimeIndex(result["demand"]["datetime_index"])
    heat = pd.Series(result["demand"]["total_heat_kW"], index=index).groupby(index.month).sum() / 1000
    cool = pd.Series(result["demand"]["total_cooling_kW"], index=index).groupby(index.month).sum() / 1000
    monthly = pd.DataFrame({"Heating + DHW (MWh)": heat, "Cooling (MWh)": cool})
    monthly.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.bar_chart(monthly)

    st.download_button("Download scenario JSON", scenario_to_json_bytes(result["input"]), "scenario.json", "application/json")
    csv = pd.DataFrame([result_summary_row(result)]).to_csv(index=False).encode("utf-8")
    st.download_button("Download result summary CSV", csv, "scenario_result.csv", "text/csv")


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
        uploaded = st.file_uploader("Upload scenario JSON", type=["json"])
        if uploaded is not None and st.button("Load uploaded JSON", use_container_width=True):
            try:
                _clear_editor_widget_state()
                st.session_state.scenario = apply_defaults(json.load(uploaded))
                st.session_state.last_result = None
                st.rerun()
            except (json.JSONDecodeError, TypeError) as exc:
                st.error(f"Could not read JSON: {exc}")
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
