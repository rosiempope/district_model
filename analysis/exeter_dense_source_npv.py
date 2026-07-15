"""Two simple charts comparing energy-source NPV for the DENSE Central
Exeter network only — the final 40-year return, and how it gets there
year by year.

Run from the repository root:
    python -m analysis.exeter_dense_source_npv
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

from profiles.demand_synthesis import synthesise_network
from optimisation.auto_size import recommend_sizing
from scenarios.scenario_runner import run_scenario
from analysis.exeter_case_study import (
    CENTRAL_BUILDINGS, CENTRAL_SEGMENTS, PRESET_FOR_TYPE, _map_sources, weather, scaled_economics,
)

C_BLUE, C_AQUA, C_VIOLET, C_RED = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
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


TECH_OPTIONS = {
    "Gas boiler only": (["gas_boiler"], C_RED),
    "Heat pump + gas backup": (["ashp", "gas_boiler"], C_BLUE),
    "Data-centre waste heat + gas": (["data_centre", "gas_boiler"], C_AQUA),
    "Waste-to-energy + heat pump + gas": (["efw_chp", "ashp", "gas_boiler"], C_VIOLET),
}


def build_scenario(label, tech_types):
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(CENTRAL_BUILDINGS)})
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=tech_types, weather_df=weather, network_flow_temp_C=70.0,
        n_buildings=len(CENTRAL_BUILDINGS), building_types=[b["type"] for b in CENTRAL_BUILDINGS],
    )
    economics, scale_factor = scaled_economics(demand["peak_heat_kW"] / 1000.0)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    return {
        "name": label, "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(CENTRAL_BUILDINGS)},
        "network": {"mode": "tree", "segments": deepcopy(CENTRAL_SEGMENTS), "include_cooling": False,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }


results = {}
rows = []
for label, (tech, color) in TECH_OPTIONS.items():
    scenario = build_scenario(label, tech)
    result = run_scenario(scenario)
    results[label] = result
    h, inv = result["headline"], result["financial"]["investor"]
    rows.append({
        "Source": label,
        "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 0),
        "Meets carbon limit?": h["carbon_compliant"],
        "Final NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
    })
df = pd.DataFrame(rows)
df.to_csv(OUT / "dense_source_npv.csv", index=False)
print(df.to_string(index=False))

# --- Chart 1: final NPV, bar ---
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = [c for _, c in TECH_OPTIONS.values()]
bars = ax.bar(range(len(df)), df["Final NPV (£m)"], color=colors, width=0.6)
for i, row in df.iterrows():
    va = "bottom" if row["Final NPV (£m)"] >= 0 else "top"
    offset = 1.0 if row["Final NPV (£m)"] >= 0 else -1.0
    ax.text(i, row["Final NPV (£m)"] + offset, f"£{row['Final NPV (£m)']:.1f}m", ha="center",
            va=va, fontsize=12, fontweight="bold", color=INK)
ax.axhline(0, color=INK, linewidth=1.1)
wrapped_labels = [label.replace(" + ", "\n+ ", 1) for label in TECH_OPTIONS]
ax.set_xticks(range(len(df)))
ax.set_xticklabels(wrapped_labels, fontsize=11)
ax.set_ylabel("Investor return over 40 years (£m)")
ax.set_title("Dense Exeter (town centre) — final return by energy source", loc="left", fontsize=15.5, fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "dense_npv_by_source.png")

# --- Chart 2: NPV trajectory over time, direct-labelled lines (no legend box) ---
fig, ax = plt.subplots(figsize=(9, 6))
for label, (tech, color) in TECH_OPTIONS.items():
    inv = results[label]["financial"]["investor"]
    years = inv["cashflow_years"]
    cum = np.array(inv["cumulative_discounted_GBP"]) / 1e6
    ax.plot(years, cum, color=color, linewidth=2.6, zorder=3)
    ax.text(years[-1] + 0.6, cum[-1], label, color=color, fontsize=11.5, fontweight="bold",
            va="center", ha="left")
ax.axhline(0, color=INK, linewidth=1.1)
ax.set_xlim(0, 46)
ax.set_xlabel("Project year")
ax.set_ylabel("Cumulative discounted cash position (£m)")
ax.set_title("Dense Exeter (town centre) — how each source's return builds over 40 years", loc="left", fontsize=14.5, fontweight="bold", color=INK)
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "dense_npv_over_time.png")

print(f"\nWrote 2 charts to {OUT}")
