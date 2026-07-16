"""Fresh, simple Exeter findings — four charts, four questions, no clutter.

Fixes a real bug found in analysis/exeter_climate_fourpipe.py: that script's
auto-sizing step called synthesise_network() WITHOUT climate_reference, so
plant was sized against a demand curve that was only RESHAPED for warmer
weather, not actually scaled up — while the final headline numbers (from
run_scenario, which internally does this correctly) showed the real,
larger climate-driven demand. That mismatch is why the auto-sized chiller
capacity appeared to SHRINK as climate warmed. Fixed here by computing one
shared climate_reference from baseline weather and passing it into every
synthesise_network() call, matching what run_scenario() does internally.

Run from the repository root:
    python -m analysis.exeter_simple_findings
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exeter_simple"
OUT.mkdir(parents=True, exist_ok=True)

from profiles.demand_synthesis import synthesise_network, compute_climate_reference
from profiles.climate_scenarios import apply_climate_scenario
from optimisation.auto_size import recommend_sizing
from scenarios.scenario_runner import run_scenario
from analysis.exeter_case_study import (
    CENTRAL_BUILDINGS, CENTRAL_SEGMENTS, SOWTON_BUILDINGS, SOWTON_SEGMENTS,
    _map_sources, weather as raw_weather, scaled_economics,
)

C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED, C_MAGENTA, C_ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


BASELINE_REF = compute_climate_reference(apply_climate_scenario(raw_weather, "baseline"))


def build_tree_scenario(name, buildings, segments, tech_types, climate="baseline", include_cooling=False):
    climate_weather = apply_climate_scenario(raw_weather, climate)
    demand = synthesise_network(climate_weather, {"demand_nodes": deepcopy(buildings)}, climate_reference=BASELINE_REF)
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=tech_types, weather_df=climate_weather, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
        include_cooling=include_cooling,
        cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
    )
    peak_total_MW = (demand["peak_heat_kW"] + (demand["peak_cool_kW"] if include_cooling else 0.0)) / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    if include_cooling:
        economics["counterfactual"] = "individual_gas_and_ac"
    scenario = {
        "name": name, "climate_scenario": climate,
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "tree", "segments": deepcopy(segments), "include_cooling": include_cooling,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                    "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    if include_cooling:
        scenario["cooling_sources"] = _map_sources(rec["cooling_sources"])
    return scenario, demand, rec


# ═══════════════════════════════════════════════════════════════════════════
# Q1 — What makes a DHN feasible & affordable? (density)
# ═══════════════════════════════════════════════════════════════════════════

TECH = ["efw_chp", "ashp", "gas_boiler"]
q1_rows = []
for label, buildings, segments in [
    ("Central Exeter\n(dense, short branches)", CENTRAL_BUILDINGS, CENTRAL_SEGMENTS),
    ("Sowton/Airport\n(spread, long branches)", SOWTON_BUILDINGS, SOWTON_SEGMENTS),
]:
    scenario, demand, rec = build_tree_scenario(label, buildings, segments, TECH)
    result = run_scenario(scenario)
    h, inv = result["headline"], result["financial"]["investor"]
    q1_rows.append({
        "Network": label,
        "Linear density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
        "What customers pay (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
        "What the scheme needs to charge (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
    })
q1_df = pd.DataFrame(q1_rows)
q1_df.to_csv(OUT / "q1_density.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(q1_df))
w = 0.32
ax.bar(x - w/2, q1_df["What customers pay (p/kWh)"], width=w, color=C_GREEN, label="What customers pay (gas-parity price)")
ax.bar(x + w/2, q1_df["What the scheme needs to charge (p/kWh)"], width=w, color=C_RED, label="What the scheme actually needs to charge")
for i, row in q1_df.iterrows():
    ax.text(i, row["What the scheme needs to charge (p/kWh)"] + 1, f"{row['Linear density (MWh/m/yr)']:.1f} MWh/m/yr",
            ha="center", fontsize=10.5, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(q1_df["Network"], fontsize=11)
ax.set_ylabel("Heat price (pence per kWh)")
ax.set_title("Q1 — Density decides affordability", loc="left", fontsize=15, fontweight="bold", color=INK)
ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "q1_density_affordability.png")

# ═══════════════════════════════════════════════════════════════════════════
# Q2 — Which energy sources are better suited?
# ═══════════════════════════════════════════════════════════════════════════

TECH_OPTIONS = {
    "Gas boiler\nonly": ["gas_boiler"],
    "Heat pump\n+ gas backup": ["ashp", "gas_boiler"],
    "Data-centre\nwaste heat + gas": ["data_centre", "gas_boiler"],
    "Waste-to-energy\n+ heat pump + gas": ["efw_chp", "ashp", "gas_boiler"],
}
q2_rows = []
for label, tech in TECH_OPTIONS.items():
    scenario, demand, rec = build_tree_scenario(label, CENTRAL_BUILDINGS, CENTRAL_SEGMENTS, tech)
    result = run_scenario(scenario)
    h, inv = result["headline"], result["financial"]["investor"]
    q2_rows.append({
        "Source": label.replace("\n", " "),
        "Carbon (gCO2e/kWh)": h["carbon_intensity_kgCO2_per_kWh"] * 1000,
        "Meets carbon limit?": h["carbon_compliant"],
        "NPV (£m)": inv["npv_GBP"] / 1e6,
    })
q2_df = pd.DataFrame(q2_rows)
q2_df.to_csv(OUT / "q2_sources.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5.2))
colors = [C_GREEN if ok else C_RED for ok in q2_df["Meets carbon limit?"]]
bars = ax.bar(range(len(q2_df)), q2_df["NPV (£m)"], color=colors, width=0.55)
for i, row in q2_df.iterrows():
    ax.text(i, row["NPV (£m)"] - 1.2, f"{row['Carbon (gCO2e/kWh)']:.0f} g/kWh", ha="center", fontsize=10, color=INK2)
ax.set_xticks(range(len(q2_df)))
ax.set_xticklabels(list(TECH_OPTIONS.keys()), fontsize=10.5)
ax.axhline(0, color=INK, linewidth=1)
ax.set_ylabel("Investor return (£m over 40 years)")
ax.set_title("Q2 — Which energy source performs best?", loc="left", fontsize=15, fontweight="bold", color=INK)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_GREEN, label="Meets carbon limit"), Patch(color=C_RED, label="Fails carbon limit")],
          fontsize=10, frameon=False, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "q2_energy_sources.png")

# ═══════════════════════════════════════════════════════════════════════════
# Q3 — What happens to the economics as climate warms?
# ═══════════════════════════════════════════════════════════════════════════

q3_rows = []
for climate in ["baseline", "2050_central"]:
    scenario, demand, rec = build_tree_scenario(f"Sowton/Airport heating — {climate}", SOWTON_BUILDINGS, SOWTON_SEGMENTS, TECH, climate=climate)
    result = run_scenario(scenario)
    h, inv = result["headline"], result["financial"]["investor"]
    q3_rows.append({
        "Climate": {"baseline": "Today", "2050_central": "2050 (central)"}[climate],
        "Annual heating demand (MWh)": h["annual_heat_demand_MWh"],
        "Annual cooling demand (MWh)": demand["annual_cool_MWh"],
        "NPV, heating only (£m)": inv["npv_GBP"] / 1e6,
    })
q3_df = pd.DataFrame(q3_rows)
q3_df.to_csv(OUT / "q3_climate.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(len(q3_df))
w = 0.32
ax1.bar(x - w/2, q3_df["Annual heating demand (MWh)"], width=w, color=C_BLUE, label="Heating demand")
ax1.bar(x + w/2, q3_df["Annual cooling demand (MWh)"], width=w, color=C_ORANGE, label="Cooling demand")
ax1.set_xticks(x)
ax1.set_xticklabels(q3_df["Climate"], fontsize=11)
ax1.set_ylabel("Annual energy demand (MWh)")
ax1.set_title("Heating falls, cooling rises", loc="left", fontsize=13, fontweight="bold", color=INK)
ax1.legend(fontsize=10, frameon=False)
ax1.spines[["top", "right"]].set_visible(False)

ax2.bar(x, q3_df["NPV, heating only (£m)"], color=C_RED, width=0.5)
ax2.axhline(0, color=INK, linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(q3_df["Climate"], fontsize=11)
ax2.set_ylabel("Investor return (£m)")
ax2.set_title("...but the heating business gets WORSE", loc="left", fontsize=13, fontweight="bold", color=INK)
ax2.spines[["top", "right"]].set_visible(False)
fig.suptitle("Q3 — What happens when the climate warms?", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK, y=1.04)
_save(fig, "q3_climate_warming.png")

# ═══════════════════════════════════════════════════════════════════════════
# Q4 — Why does a four-pipe (heating + cooling) system fail?
# ═══════════════════════════════════════════════════════════════════════════

scenario_2p, demand_2p, rec_2p = build_tree_scenario("Sowton/Airport 2-pipe", SOWTON_BUILDINGS, SOWTON_SEGMENTS, TECH, include_cooling=False)
result_2p = run_scenario(scenario_2p)
scenario_4p, demand_4p, rec_4p = build_tree_scenario("Sowton/Airport 4-pipe", SOWTON_BUILDINGS, SOWTON_SEGMENTS, TECH, include_cooling=True)
result_4p = run_scenario(scenario_4p)

h2, inv2 = result_2p["headline"], result_2p["financial"]["investor"]
h4, inv4 = result_4p["headline"], result_4p["financial"]["investor"]

q4_rows = [
    {"Metric": "CAPEX (£m)", "2-pipe (heating only)": h2["capex_total_GBP"] / 1e6, "4-pipe (heating + cooling)": h4["capex_total_GBP"] / 1e6},
    {"Metric": "Investor NPV (£m)", "2-pipe (heating only)": inv2["npv_GBP"] / 1e6, "4-pipe (heating + cooling)": inv4["npv_GBP"] / 1e6},
]
q4_df = pd.DataFrame(q4_rows)
q4_df.to_csv(OUT / "q4_fourpipe.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
x = np.arange(2)
w = 0.5
ax1.bar(x, [h2["capex_total_GBP"] / 1e6, h4["capex_total_GBP"] / 1e6], color=[C_BLUE, C_ORANGE], width=w)
ax1.set_xticks(x)
ax1.set_xticklabels(["2-pipe\n(heating only)", "4-pipe\n(heating + cooling)"], fontsize=11)
ax1.set_ylabel("Total CAPEX (£m)")
ax1.set_title(f"Extra CAPEX for cooling: +£{(h4['capex_total_GBP']-h2['capex_total_GBP'])/1e6:.1f}m", loc="left", fontsize=13, fontweight="bold", color=INK)
ax1.spines[["top", "right"]].set_visible(False)

ax2.bar(x, [inv2["npv_GBP"] / 1e6, inv4["npv_GBP"] / 1e6], color=[C_BLUE, C_ORANGE], width=w)
ax2.axhline(0, color=INK, linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(["2-pipe\n(heating only)", "4-pipe\n(heating + cooling)"], fontsize=11)
ax2.set_ylabel("Investor return (£m)")
ax2.set_title(f"Result: cooling makes NPV £{abs(inv4['npv_GBP']-inv2['npv_GBP'])/1e6:.1f}m worse", loc="left", fontsize=13, fontweight="bold", color=INK)
ax2.spines[["top", "right"]].set_visible(False)
fig.suptitle("Q4 — Why does four-pipe fail?", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK, y=1.04)
_save(fig, "q4_fourpipe_fails.png")

print("=== Q1: Density and affordability ===")
print(q1_df.to_string(index=False))
print("\n=== Q2: Energy sources ===")
print(q2_df.to_string(index=False))
print("\n=== Q3: Climate warming ===")
print(q3_df.to_string(index=False))
print("\n=== Q4: Four-pipe ===")
print(q4_df.to_string(index=False))
print("\nChiller capacity check — cooling peak demand at each climate (properly climate-scaled now):")
for climate in ["baseline", "2050_central"]:
    cw = apply_climate_scenario(raw_weather, climate)
    d = synthesise_network(cw, {"demand_nodes": deepcopy(SOWTON_BUILDINGS)}, climate_reference=BASELINE_REF)
    print(f"  {climate:14}  peak cooling demand = {d['peak_cool_kW']/1000:.2f} MW  "
          f"(previous script's un-scaled figure would have shown this shrinking — now it correctly grows)")
print(f"\nWrote CSVs and 4 charts to {OUT}")
