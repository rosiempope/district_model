"""How much does climate change move the NPV needle, heating vs cooling?

    python -m analysis.climate_scenario_sweep

Three climate scenarios (UKCP18-based, see profiles/climate_scenarios.py) swept
across the three density archetypes used throughout this pack, run BOTH
heating-only (2-pipe) and heating+cooling (4-pipe), same EfW + ASHP + gas peak
heat stack, chiller auto-sized, GHNF 40% where carbon-eligible, gas-parity
billing.

  baseline      : London Heathrow representative year (2011-2025), no shift
  2050_central  : UKCP18 RCP4.5 central — +1.0C winter, +2.7C summer
  2050_high     : UKCP18 RCP8.5 high + urban heat island — +2.0C winter,
                  +4.0C summer, UHI peaking +2.5C in summer, 0 in winter

A warmer climate cuts heating demand (fewer HDD) and grows cooling demand
(more CDD) at the same time — this sweep is the only place in the pack that
prices BOTH movements together, on the SAME scheme, so it answers "does
climate change make the 4-pipe case look any better?" as well as "how much
does warming alone move heating-only NPV?".

Sources are re-auto-sized per climate (a warmer design day genuinely changes
the ASHP/EfW/gas-peak split), following the same pattern as
analysis/source_frontier.py and analysis/dalkia_screening_study.py.

Writes CSVs, PNGs and findings.md to output/climate_scenario_sweep/.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "climate_scenario_sweep"
OUT.mkdir(parents=True, exist_ok=True)

from optimisation.auto_size import recommend_sizing
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import compute_climate_reference, synthesise_network
from scenarios.fixed_cost_scaling import scaled_economics
from scenarios.scenario_runner import run_scenario
from analysis.archetypes import ARCHETYPES

# ── Palette (validated categorical set, see dataviz skill) ──────────────────
C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1. The three archetypes — same building mixes and route lengths used
#    throughout this pack (dalkia_screening_study, ghnf_affordability,
#    source_frontier), imported from analysis/archetypes.py so every study's
#    rows line up with the others.
# ═══════════════════════════════════════════════════════════════════════════

CLIMATE_SCENARIOS = ["baseline", "2050_central", "2050_high"]
CLIMATE_LABELS = {
    "baseline": "Baseline (2011-2025)",
    "2050_central": "2050 central (RCP4.5)",
    "2050_high": "2050 high + UHI (RCP8.5)",
}

PRESETS = {"ashp": "ealing_phase1", "gas_boiler": "ealing_phase1",
           "efw_chp": "newlincs_style", "air_cooled_chiller": "generic_2MW_bank"}

raw_weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(raw_weather) == 8760
raw_weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")
# One reference, computed from BASELINE weather, reused for every scenario —
# this is what makes annual totals genuinely shift with climate rather than
# only reshaping across the year (profiles/demand_synthesis.py docstring).
climate_reference = compute_climate_reference(apply_climate_scenario(raw_weather, "baseline"))


def _map(srcs):
    return [{"type": s["type"], "preset": PRESETS[s["type"]],
             "name": f"{s['type']} ({s['role']})", "capacity_MW": float(s["capacity_MW"]),
             **({"n_units": int(s["n_units"])} if "n_units" in s else {})}
            for s in srcs]


rows = []
for arch_label, cfg in ARCHETYPES.items():
    for climate in CLIMATE_SCENARIOS:
        weather = apply_climate_scenario(raw_weather, climate)
        demand = synthesise_network(
            weather, {"demand_nodes": deepcopy(cfg["buildings"])},
            climate_reference=climate_reference,
        )
        for include_cooling in (False, True):
            rec = recommend_sizing(
                demand_kW=demand["total_heat_kW"],
                peak_demand_kW=demand["peak_heat_kW"],
                technology_types=["efw_chp", "ashp", "gas_boiler"],
                weather_df=weather,
                network_flow_temp_C=70.0,
                n_buildings=len(cfg["buildings"]),
                building_types=[b["type"] for b in cfg["buildings"]],
                include_cooling=include_cooling,
                cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
                peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
            )
            peak_MW = demand["peak_heat_kW"] / 1000.0
            if include_cooling:
                peak_MW += demand["peak_cool_kW"] / 1000.0
            economics, scale = scaled_economics(peak_MW)
            economics["counterfactual"] = (
                "individual_gas_and_ac" if include_cooling else "individual_gas")
            economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
            scenario = {
                "name": f"{arch_label} — {climate} — {'4-pipe' if include_cooling else '2-pipe'}",
                "climate_scenario": climate,
                "demand": {"buildings": deepcopy(cfg["buildings"])},
                "network": {"mode": "generic_length", "length_m": float(cfg["route_m"]),
                            "include_cooling": include_cooling,
                            "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                            "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
                "sources": _map(rec["sources"]),
                "economics": economics,
            }
            if include_cooling:
                scenario["cooling_sources"] = _map(rec["cooling_sources"])
            result = run_scenario(scenario)
            h = result["headline"]
            inv = result["financial"]["investor"]
            grant = result.get("grant")
            rows.append({
                "Archetype": arch_label,
                "Climate": CLIMATE_LABELS[climate],
                "Network": "4-pipe (heat+cool)" if include_cooling else "2-pipe (heat only)",
                "Annual heat (GWh)": round((demand["annual_heat_MWh"]
                                            + demand["annual_dhw_MWh"]) / 1000.0, 2),
                "Annual cooling (GWh)": round(demand["annual_cool_MWh"] / 1000.0, 2),
                "Peak heat (MW)": round(demand["peak_heat_kW"] / 1000.0, 2),
                "Peak cooling (MW)": round(demand["peak_cool_kW"] / 1000.0, 2),
                "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
                "GHNF grant (£m)": round((grant["grant_GBP"] if grant else 0) / 1e6, 2),
                "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
                "Screening": result["screening"]["status"],
            })
            print(f"{arch_label} | {climate} | "
                  f"{'4-pipe' if include_cooling else '2-pipe'}: "
                  f"NPV £{inv['npv_GBP']/1e6:.2f}m")

df = pd.DataFrame(rows)
df.to_csv(OUT / "climate_scenario_sweep.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Incremental cooling NPV by climate — does warming make 4-pipe look better?
# ═══════════════════════════════════════════════════════════════════════════

pivot = df.pivot_table(index=["Archetype", "Climate"], columns="Network",
                       values="Investor NPV (£m)")
pivot["Incremental NPV of cooling (£m)"] = (
    pivot["4-pipe (heat+cool)"] - pivot["2-pipe (heat only)"])
incr_df = pivot.reset_index()
incr_df.to_csv(OUT / "incremental_cooling_by_climate.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Figures
# ═══════════════════════════════════════════════════════════════════════════

ARCH_COLOURS = {"Dense (town centre)": C_BLUE, "Middle (suburban mixed)": C_AQUA,
                "Scarce (low-density edge)": C_YELLOW}
CLIMATE_ORDER = [CLIMATE_LABELS[c] for c in CLIMATE_SCENARIOS]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=False)
for arch_label in ARCHETYPES:
    heat_only = df[(df["Archetype"] == arch_label)
                   & (df["Network"] == "2-pipe (heat only)")].set_index("Climate").loc[CLIMATE_ORDER]
    four_pipe = df[(df["Archetype"] == arch_label)
                   & (df["Network"] == "4-pipe (heat+cool)")].set_index("Climate").loc[CLIMATE_ORDER]
    axes[0].plot(CLIMATE_ORDER, heat_only["Investor NPV (£m)"], "-o",
                 color=ARCH_COLOURS[arch_label], lw=2, ms=6, label=arch_label)
    axes[1].plot(CLIMATE_ORDER, four_pipe["Investor NPV (£m)"], "-o",
                 color=ARCH_COLOURS[arch_label], lw=2, ms=6, label=arch_label)
for ax in axes:
    ax.axhline(0, color=INK, lw=1.2)
    ax.tick_params(axis="x", rotation=12)
axes[0].set_ylabel("Investor NPV (£m, 10.5% hurdle)")
axes[0].set_title("Heating only (2-pipe)", fontsize=11)
axes[1].set_title("Heating + cooling (4-pipe)", fontsize=11)
axes[0].legend(fontsize=8.5)
fig.suptitle("Investor NPV across climate scenarios — EfW + ASHP + gas peak, GHNF 40%, gas parity",
             fontsize=12.5)
_save(fig, "CS1_npv_by_climate.png")

fig, ax = plt.subplots(figsize=(10.5, 5.6))
for arch_label in ARCHETYPES:
    sub = incr_df[incr_df["Archetype"] == arch_label].set_index("Climate").loc[CLIMATE_ORDER]
    ax.plot(CLIMATE_ORDER, sub["Incremental NPV of cooling (£m)"], "-o",
            color=ARCH_COLOURS[arch_label], lw=2, ms=6, label=arch_label)
ax.axhline(0, color=INK, lw=1.2)
ax.tick_params(axis="x", rotation=12)
ax.set_ylabel("Incremental NPV of adding 4-pipe cooling (£m)")
ax.set_title("Does a warmer climate make 4-pipe cooling more attractive?\n"
             "Incremental NPV = NPV(4-pipe) - NPV(2-pipe), same scheme", fontsize=12)
ax.legend(fontsize=9)
_save(fig, "CS2_cooling_incremental_by_climate.png")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True)
for arch_label in ARCHETYPES:
    sub2 = df[(df["Archetype"] == arch_label)
             & (df["Network"] == "2-pipe (heat only)")].set_index("Climate").loc[CLIMATE_ORDER]
    axes[0].plot(CLIMATE_ORDER, sub2["Annual heat (GWh)"], "-o",
                 color=ARCH_COLOURS[arch_label], lw=2, ms=6, label=arch_label)
    sub4 = df[(df["Archetype"] == arch_label)
             & (df["Network"] == "4-pipe (heat+cool)")].set_index("Climate").loc[CLIMATE_ORDER]
    axes[1].plot(CLIMATE_ORDER, sub4["Annual cooling (GWh)"], "-o",
                 color=ARCH_COLOURS[arch_label], lw=2, ms=6, label=arch_label)
for ax in axes:
    ax.tick_params(axis="x", rotation=12)
axes[0].set_ylabel("Annual heat demand (GWh)")
axes[0].set_title("Heating demand falls as climate warms", fontsize=11)
axes[1].set_ylabel("Annual cooling demand (GWh)")
axes[1].set_title("Cooling demand rises as climate warms", fontsize=11)
axes[0].legend(fontsize=8.5)
fig.suptitle("What is actually moving: demand, not just price", fontsize=12.5)
_save(fig, "CS3_demand_by_climate.png")

# ═══════════════════════════════════════════════════════════════════════════
# findings.md
# ═══════════════════════════════════════════════════════════════════════════

lines = ["# Climate scenarios — heating and cooling investor NPV by archetype", "",
         "Generated by `python -m analysis.climate_scenario_sweep`. EfW + ASHP + gas peak, "
         "auto-sized per climate; GHNF 40% where carbon-eligible; gas-parity billing.", ""]

for arch_label in ARCHETYPES:
    two_base = df[(df["Archetype"] == arch_label) & (df["Climate"] == CLIMATE_LABELS["baseline"])
                 & (df["Network"] == "2-pipe (heat only)")]["Investor NPV (£m)"].iloc[0]
    two_high = df[(df["Archetype"] == arch_label) & (df["Climate"] == CLIMATE_LABELS["2050_high"])
                 & (df["Network"] == "2-pipe (heat only)")]["Investor NPV (£m)"].iloc[0]
    incr_base = incr_df[(incr_df["Archetype"] == arch_label)
                        & (incr_df["Climate"] == CLIMATE_LABELS["baseline"])][
                            "Incremental NPV of cooling (£m)"].iloc[0]
    incr_high = incr_df[(incr_df["Archetype"] == arch_label)
                        & (incr_df["Climate"] == CLIMATE_LABELS["2050_high"])][
                            "Incremental NPV of cooling (£m)"].iloc[0]
    lines.append(f"- **{arch_label}**: heat-only NPV moves £{two_base:.2f}m (baseline) → "
                f"£{two_high:.2f}m (2050 high); cooling's incremental NPV moves "
                f"£{incr_base:.2f}m → £{incr_high:.2f}m.")

lines += [
    "",
    "## What is driving it",
    "",
    "- A warmer climate cuts annual heat demand (fewer HDD) and grows annual cooling",
    "  demand (more CDD) on the SAME building stock — see CS3. Revenue is bill-parity",
    "  capped either way, so less heat delivered under gas parity does not obviously",
    "  help or hurt NPV on its own; the CAPEX/fixed-cost base is unchanged.",
    "- Carbon intensity moves with climate too, via the heat-source mix the",
    "  auto-sizer picks for a different design day — check the Carbon gate column",
    "  before reading NPV where the gate flips (breaks the GHNF grant).",
    "- 2050_high stacks a summer urban-heat-island offset on top of the RCP8.5 delta",
    "  (MODEL_SUMMARY §8), so its cooling growth is not simply proportional to the",
    "  2050_central case.",
    "",
    "## Full sweep",
    "",
    df.to_markdown(index=False),
    "",
    "## Incremental cooling NPV by climate",
    "",
    incr_df.round(2).to_markdown(index=False),
    "",
    "## Caveat",
    "",
    "- Annual physical performance is repeated across the 40-year horizon in every",
    "  case in this pack (MODEL_SUMMARY §12) — this study prices three FIXED climate",
    "  states, not a year-by-year warming trajectory over the scheme's real life.",
    "  A scheme commissioned today lives through baseline-like early years and",
    "  warmer late years; this sweep brackets that range, it does not average it.",
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 3 figures.")
