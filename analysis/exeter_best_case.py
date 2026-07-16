"""What is the BEST case for a 3rd-generation network over 40 years?

    python -m analysis.exeter_best_case

Runs against the live engine (scenarios.scenario_runner.run_scenario) — the same
entry point main.py and the Streamlit app use.

Everything is stacked in district heat's favour here, deliberately, because the
question is "can this work at all", not "is this typical":

  - REAL TREE TOPOLOGY. Exeter Central, per-branch lengths, per-segment pipe
    sizing, real delivered-temperature checking. Not an equivalent trunk.
  - 62/30 with instantaneous HIUs. Flow lowered from 70 to buy heat-pump COP;
    return at 30 against CP1 2020's <33C VWART best practice. Delivered
    temperature is gated, so any design that starves customers of hot water
    FAILS rather than quietly banking the COP.
  - INDIVIDUAL HEAT PUMPS as the counterfactual, with the BUS grant where
    eligible. This is the alternative that is actually legal long-term and the
    one heat network zoning is judged against. Comparing to a gas boiler answers
    a question policy has already closed.
  - ANCHOR LOADS added (see EXTENDED_BUILDINGS — ILLUSTRATIVE, not real Exeter
    data) to test whether scale rescues the fixed-cost problem.
  - GHNF grant applied wherever the carbon gate allows.
  - A source mix, including energy-from-waste.

Two ways to use an energy-from-waste plant, and they are NOT the same
----------------------------------------------------------------------
  STEAM EXTRACTION (efw_chp, ~90C): the plant gives up some electricity to
  export high-grade heat directly into the network. No booster needed. This is
  the strong card, and it is what the model's efw_chp presets represent.

  LOW-GRADE WASTE HEAT (~20-40C, needs a booster): what the DESNZ Birmingham
  report actually identified at Tyseley — Table 10 lists Tyseley ERF at 4,000
  kWp, Birmingham Biomass at 20,000 kWp and the SCC data centre at 6,000 kWp,
  ALL at 20-40C. That is not steam extraction; it is rejected heat that needs
  lifting to network temperature, so it carries a booster's capital and its
  electricity.

Both are modelled below. Reading the report's Tyseley opportunity as a 90C EfW
CHP export would overstate it by a wide margin, and that is exactly the mistake
this file exists to avoid.

Writes to output/exeter_best_case/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.exeter_case_study import CENTRAL_BUILDINGS, CENTRAL_SEGMENTS
from scenarios.fixed_cost_scaling import scaled_economics
from scenarios.scenario_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exeter_best_case"

C_BLUE, C_RED, C_GREEN, C_YELLOW, C_VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

FLOW_C, RETURN_C = 62.0, 30.0

# ── Added anchor loads — ILLUSTRATIVE, NOT REAL EXETER DATA ───────────────────
#
# The existing five zones come from the DESNZ Heat Network Zoning Pilot "City
# Typologies" map for Exeter. These four do NOT. They are plausible city-centre
# anchor archetypes at plausible scales, added to answer "does more anchor load
# rescue the fixed-cost problem" — a scale sensitivity, not a claim about Exeter.
# Branch lengths are chosen to keep them close to the energy centre, which is
# generous: a real anchor is where it is, not where the model wants it.
EXTENDED_BUILDINGS = CENTRAL_BUILDINGS + [
    {"name": "Acute hospital (main site)", "type": "hospital", "floor_area_m2": 45000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
    {"name": "City retail centre", "type": "retail", "floor_area_m2": 30000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.95},
    {"name": "Civic and commercial offices", "type": "office_ac", "floor_area_m2": 28000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.95},
    {"name": "Leisure centre and pool", "type": "hotel", "floor_area_m2": 9000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
]

EXTENDED_SEGMENTS = CENTRAL_SEGMENTS + [
    {"node_id": "N6", "parent_id": "N4", "length_m": 350.0, "building": "Acute hospital (main site)"},
    {"node_id": "N7", "parent_id": "N1", "length_m": 300.0, "building": "City retail centre"},
    {"node_id": "N8", "parent_id": "N1", "length_m": 450.0, "building": "Civic and commercial offices"},
    {"node_id": "N9", "parent_id": "N2", "length_m": 400.0, "building": "Leisure centre and pool"},
]


def _sources(mix: str, peak_MW: float, include_cooling: bool):
    """Plant stacks. Capacities are sized off the scenario's own peak, not guessed."""
    gas = {"type": "gas_boiler", "preset": "ealing_phase2", "name": "Gas peak/backup",
           "capacity_MW": round(peak_MW * 1.1, 2)}
    ashp = {"type": "ashp", "preset": "large_energy_centre", "name": "ASHP",
            "capacity_MW": round(peak_MW * 0.55, 2), "n_units": 6, "flow_temp_C": FLOW_C}

    if mix == "ASHP + gas peak":
        return [ashp, gas]

    if mix == "ASHP + WSHP + GSHP + gas peak":
        return [
            {**ashp, "capacity_MW": round(peak_MW * 0.30, 2), "n_units": 4},
            {"type": "wshp", "preset": "generic_river_5MW", "name": "River WSHP",
             "capacity_MW": round(peak_MW * 0.20, 2), "n_units": 2},
            {"type": "gshp", "preset": "generic_borehole_2MW", "name": "Borehole GSHP",
             "capacity_MW": round(peak_MW * 0.15, 2), "n_units": 2},
            gas,
        ]

    if mix == "EfW steam extraction + ASHP + gas peak":
        # High-grade heat straight into the network at ~90C. No booster.
        return [
            {"type": "efw_chp", "preset": "mid_scale_generic", "name": "EfW CHP heat export",
             "capacity_MW": round(peak_MW * 0.45, 2)},
            {**ashp, "capacity_MW": round(peak_MW * 0.30, 2), "n_units": 4},
            gas,
        ]

    if mix == "EfW steam extraction + ASHP (gas-free)":
        return [
            {"type": "efw_chp", "preset": "mid_scale_generic", "name": "EfW CHP heat export",
             "capacity_MW": round(peak_MW * 0.50, 2)},
            # ASHP carries the peak instead of a boiler. Sized on the design-day
            # derate, the same conversion auto_size applies.
            {**ashp, "capacity_MW": round(peak_MW * 0.85, 2), "n_units": 8},
        ]

    if mix == "Tyseley-style low-grade waste heat + booster + ASHP + gas peak":
        # The DESNZ report's ACTUAL Tyseley opportunity: 20-40C waste heat that
        # needs lifting. Modelled as a low-grade source + booster, NOT as a 90C
        # EfW export. DataCentre is used here as the model's generic low-grade
        # waste-heat source class — the physics (a fixed-temperature rejected
        # heat stream with an availability profile) is the same whether the heat
        # comes from a server hall, a biomass plant or an ERF.
        return [
            {"type": "data_centre", "preset": "gtr_southall_medium", "name": "Waste heat (20-40C, Tyseley-style)",
             "capacity_MW": round(peak_MW * 0.35, 2), "supply_temp_C": 30.0, "dispatch_direct": False},
            {"type": "booster_heat_pump", "preset": "generic_2MW", "name": "Booster heat pump",
             "capacity_MW": round(peak_MW * 0.40, 2), "n_units": 4, "depends_on": 0},
            {**ashp, "capacity_MW": round(peak_MW * 0.25, 2), "n_units": 4},
            gas,
        ]

    raise ValueError(f"unknown mix {mix!r}")


def build(name, buildings, segments, mix, include_cooling, peak_MW, grant=True):
    econ, scale = scaled_economics(peak_MW)
    # Individual heat pumps on BOTH sides — the legal alternative — so the
    # 2-pipe and 4-pipe columns are actually comparable.
    econ["counterfactual"] = "individual_ashp_and_ac" if include_cooling else "individual_ashp"
    if grant:
        econ["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    network = {
        "mode": "tree", "segments": copy.deepcopy(segments),
        "include_cooling": include_cooling,
        "heat_flow_temp_C": FLOW_C, "heat_return_temp_C": RETURN_C,
        "cool_flow_temp_C": 6.0, "cool_return_temp_C": 18.0,   # 12K dT — see findings
        "dhw_system": "instantaneous_hiu",
    }
    s = {
        "name": name, "climate_scenario": "baseline",
        "demand": {"buildings": copy.deepcopy(buildings)},
        "network": network,
        "sources": _sources(mix, peak_MW, include_cooling),
        "economics": econ,
        "screening": {"maximum_unmet_energy_fraction": 0.001,
                      "maximum_carbon_gCO2e_per_kWh": 100.0,
                      "require_n_minus_one": False},
    }
    if include_cooling:
        s["cooling_sources"] = [{
            "type": "air_cooled_chiller", "preset": "generic_2MW_bank", "name": "Central chillers",
            "capacity_MW": round(peak_MW * 0.5, 2), "n_units": 6, "chilled_water_temp_C": 6.0,
        }]
    return s


def _peak_MW(buildings, include_cooling):
    from profiles.demand_synthesis import synthesise_network
    from scenarios.scenario_runner import load_weather
    d = synthesise_network(load_weather(), {"demand_nodes": copy.deepcopy(buildings)})
    p = d["peak_heat_kW"] / 1000.0
    return p + (d["peak_cool_kW"] / 1000.0 if include_cooling else 0.0)


MIXES = [
    "ASHP + gas peak",
    "ASHP + WSHP + GSHP + gas peak",
    "Tyseley-style low-grade waste heat + booster + ASHP + gas peak",
    "EfW steam extraction + ASHP + gas peak",
    "EfW steam extraction + ASHP (gas-free)",
]


def run_matrix() -> pd.DataFrame:
    rows = []
    for demand_label, buildings, segments in (
        ("Base (5 zones)", CENTRAL_BUILDINGS, CENTRAL_SEGMENTS),
        ("+ anchor loads (9 zones)", EXTENDED_BUILDINGS, EXTENDED_SEGMENTS),
    ):
        for include_cooling in (False, True):
            peak = _peak_MW(buildings, include_cooling)
            for mix in MIXES:
                s = build(f"{demand_label} | {mix}", buildings, segments, mix,
                          include_cooling, peak)
                try:
                    r = run_scenario(s)
                except Exception as e:   # noqa: BLE001 — report, do not hide
                    rows.append({"Demand": demand_label,
                                 "System": "4-pipe" if include_cooling else "2-pipe",
                                 "Mix": mix, "Decision": f"ERROR: {type(e).__name__}"})
                    continue
                h, inv, fin = r["headline"], r["financial"]["investor"], r["financial"]
                by = h["annual_heat_by_source_MWh"]
                lc = sum(v for k, v in by.items() if "Gas" not in k) / max(sum(by.values()), 1e-9)
                rows.append({
                    "Demand": demand_label,
                    "System": "4-pipe" if include_cooling else "2-pipe",
                    "Mix": mix,
                    "Service GWh/yr": round((h["annual_heat_demand_MWh"] + h["annual_cooling_demand_MWh"]) / 1000, 1),
                    "Route (m)": h["network_total_length_m"],
                    "LHD (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
                    "Low-carbon heat (%)": round(lc * 100, 1),
                    "Delivered T (°C)": h["worst_case_delivered_temp_C"],
                    "Delivered gate": "PASS" if h["delivered_temp_compliant"] else "FAIL",
                    "Carbon (g/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                    "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
                    "GHNF (£m)": round((r.get("grant") or {}).get("grant_GBP", 0) / 1e6, 2),
                    "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 1),
                    "Req. tariff (p/kWh)": inv.get("required_heat_tariff_p_per_kWh_for_zero_NPV"),
                    "Fair tariff (p/kWh)": inv.get("equivalent_year1_heat_tariff_p_per_kWh"),
                    "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
                    "Whole-system NPV @3.5% (£m)": round(fin["npv_vs_counterfactual_GBP"] / 1e6, 2),
                    "Payback (yrs)": (round(fin["simple_payback_years"], 1)
                                      if fin.get("simple_payback_years") else None),
                    "Decision": r["screening"]["status"],
                })
    return pd.DataFrame(rows)


def fig_matrix(df: pd.DataFrame):
    d = df[df["Decision"].astype(str).str.startswith("ERROR") == False]  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for ax, sys in zip(axes, ("2-pipe", "4-pipe")):
        sub = d[d["System"] == sys]
        for i, dem in enumerate(sub["Demand"].unique()):
            s2 = sub[sub["Demand"] == dem]
            ax.barh([f"{m[:34]}" for m in s2["Mix"]],
                    s2["Whole-system NPV @3.5% (£m)"],
                    height=0.38, label=dem,
                    color=[C_GREEN, C_BLUE][i], alpha=0.9 if i == 0 else 0.65)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f"{sys} — whole-system NPV vs individual heat pumps",
                     loc="left", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Whole-system NPV @3.5% (£m)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].tick_params(labelsize=8)
    _save(fig, "BC1_best_case_matrix.png")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_matrix()
    df.to_csv(OUT / "best_case_matrix.csv", index=False)
    fig_matrix(df)

    ok = df[(df["Decision"] == "PASS") | (df["Decision"] == "CONDITIONAL PASS")]
    viable = df[df.get("Whole-system NPV @3.5% (£m)", pd.Series(dtype=float)) > 0]

    print("\n=== Best-case matrix: Exeter Central, real tree, 62/30, vs individual heat pumps ===")
    cols = ["Demand", "System", "Mix", "Service GWh/yr", "LHD (MWh/m/yr)", "Low-carbon heat (%)",
            "Delivered gate", "Carbon (g/kWh)", "Carbon gate", "GHNF (£m)", "CAPEX (£m)",
            "Req. tariff (p/kWh)", "Fair tariff (p/kWh)", "Investor NPV (£m)",
            "Whole-system NPV @3.5% (£m)", "Decision"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nScreening PASS/CONDITIONAL: {len(ok)} of {len(df)}")
    print(f"Whole-system NPV positive vs individual heat pumps: {len(viable)} of {len(df)}")
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
