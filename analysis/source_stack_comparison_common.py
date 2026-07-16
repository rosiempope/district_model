"""Shared logic for the source-stack comparison studies — how each heat-
source technology performs on a FIXED network, across duty (heating only
vs 4-pipe heating+cooling), linear density, network shape (branched tree
vs a single equivalent trunk), and climate scenario.

This module holds everything that's NETWORK-AGNOSTIC (tech-stack sizing,
scenario builders, the plotting helpers, the five-graph study runner).
Site-specific data (buildings, tree segments) lives in each site's own
thin driver script — see analysis/source_stack_comparison.py (Exeter
Central) and analysis/source_stack_comparison_ealing.py (Ealing Town
Centre) for the two current callers.

Three technology stacks, held constant across every figure below
------------------------------------------------------------------------
  "ASHP + gas backup"                       — ashp (baseload) + gas_boiler (peak)
  "EfW + ASHP (no gas)"                     — efw_chp (baseload) + ashp (peak) —
                                               deliberately gas-free; see
                                               _recommend() below for why this
                                               needs a small manual step.
  "Data centre + booster + electric backup" — data_centre (baseload, booster
                                               auto-attached by recommend_sizing)
                                               + electric_boiler (peak)

All revenue uses this project's existing gas-bill-parity / AC-bill-parity
mechanism (`counterfactual_bill_parity`, the schema default — see
economics/metrics.py's counterfactual_gas_boiler_dispatch /
counterfactual_individual_ac_dispatch) — i.e. customers are charged what
they'd pay for individual gas heating / individual AC cooling, never more.
GHNF grant (40% of eligible CAPEX) is applied throughout wherever the
scenario's own carbon gate allows it. Discount rate 10.5%, project
lifetime 40 years — both from scenarios/worked_scenarios.py's
COMMON_ECONOMICS, reused unchanged via analysis.exeter_case_study's
scaled_economics().

Why "EfW + ASHP (no gas)" needs a manual sizing step
------------------------------------------------------------------------
optimisation/auto_size.py classifies source types into BASELOAD_TYPES
({"ashp", "efw_chp", "data_centre"}) and PEAK_TYPES ({"gas_boiler",
"electric_boiler"}) — ASHP is never treated as a peak/backup candidate, so
calling recommend_sizing(technology_types=["efw_chp", "ashp"]) would
silently add a gas boiler anyway ("No peak/backup technology selected —
gas boiler added automatically"). To get a genuinely gas-free EfW+ASHP
stack (the deliberate point of comparison against option 1's gas backup),
_recommend() calls recommend_sizing() as normal, then converts that
auto-added gas-boiler peak entry into an ASHP peak entry of equivalent
design-day covering capacity — nameplate_MW = required_MW /
ASHP_DESIGN_DAY_DERATING, the exact same cold-weather-derating conversion
recommend_sizing() already applies to baseload ASHP sizing. This is a
sizing-time data transform, not a change to any model physics.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from profiles.demand_synthesis import synthesise_network, compute_climate_reference
from profiles.climate_scenarios import apply_climate_scenario
from optimisation.auto_size import recommend_sizing, ASHP_DESIGN_DAY_DERATING, _sensible_ashp_unit_size
from scenarios.scenario_runner import run_scenario
from analysis.exeter_case_study import PRESET_FOR_TYPE, _map_sources, weather, scaled_economics

# ── Palette — same validated set used throughout the Exeter/Dalkia scripts ──
C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED, C_MAGENTA, C_ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def _save(fig, filename, out_dir):
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Technology stacks
# ═══════════════════════════════════════════════════════════════════════════

TECH_LABELS = [
    "ASHP + gas backup",
    "EfW + ASHP (no gas)",
    "Data centre + booster + electric backup",
]
TECH_TYPES = {
    "ASHP + gas backup": ["ashp", "gas_boiler"],
    "EfW + ASHP (no gas)": ["efw_chp", "ashp"],  # post-processed, see below
    "Data centre + booster + electric backup": ["data_centre", "electric_boiler"],
}
TECH_COLOR = {
    "ASHP + gas backup": C_BLUE,
    "EfW + ASHP (no gas)": C_VIOLET,
    "Data centre + booster + electric backup": C_AQUA,
}
TECH_SHORT = {
    "ASHP + gas backup": "ASHP+gas",
    "EfW + ASHP (no gas)": "EfW+ASHP",
    "Data centre + booster + electric backup": "DC+booster+elec",
}


def _recommend(tech_label, **kwargs):
    """Dispatch to recommend_sizing(), with the EfW+ASHP no-gas conversion
    applied where needed. kwargs forwarded: demand_kW, peak_demand_kW,
    weather_df, n_buildings, building_types, include_cooling,
    cooling_demand_kW, peak_cooling_kW, network_flow_temp_C."""
    rec = recommend_sizing(technology_types=TECH_TYPES[tech_label], **kwargs)
    if tech_label == "EfW + ASHP (no gas)":
        new_sources = []
        for s in rec["sources"]:
            if s["type"] == "gas_boiler" and s["role"] == "peak":
                nameplate_MW = s["capacity_MW"] / ASHP_DESIGN_DAY_DERATING
                unit_MW = _sensible_ashp_unit_size(nameplate_MW)
                n_units = max(1, round(nameplate_MW / unit_MW))
                nameplate_MW = n_units * unit_MW
                new_sources.append({
                    "type": "ashp", "capacity_MW": round(nameplate_MW, 2), "n_units": n_units,
                    "role": "peak", "flow_temp_C": kwargs.get("network_flow_temp_C", 70.0),
                    "rationale": "Peak/backup ASHP (replaces the auto-sized gas boiler) — "
                                 "this stack deliberately excludes gas.",
                })
            else:
                new_sources.append(s)
        rec = dict(rec)
        rec["sources"] = new_sources
    return rec


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scenario builders (tree mode and generic_length/"trunk" mode)
# ═══════════════════════════════════════════════════════════════════════════

def build_tree_scenario(tech_label, buildings, segments, include_cooling=False,
                          climate_scenario="baseline", climate_reference=None,
                          weather_df=None):
    weather_df = weather if weather_df is None else weather_df
    demand = synthesise_network(weather_df, {"demand_nodes": deepcopy(buildings)}, climate_reference=climate_reference)
    rec = _recommend(
        tech_label,
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        weather_df=weather_df, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
        include_cooling=include_cooling,
        cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
    )
    peak_total_MW = (demand["peak_heat_kW"] + (demand["peak_cool_kW"] if include_cooling else 0.0)) / 1000.0
    economics, _scale = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    if include_cooling:
        economics["counterfactual"] = "individual_gas_and_ac"
    scenario = {
        "name": f"{tech_label} — tree", "climate_scenario": climate_scenario,
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


def build_generic_scenario(tech_label, buildings, route_m, include_cooling=False):
    """generic_length ("single equivalent trunk") mode — same demand, one
    representative pipe carrying the whole route's peak. This IS the
    codebase's existing single-trunk stand-in (see scenarios/screening.py's
    own "Equivalent-trunk route" flag on this mode)."""
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    rec = _recommend(
        tech_label,
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        weather_df=weather, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
        include_cooling=include_cooling,
        cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
    )
    peak_total_MW = (demand["peak_heat_kW"] + (demand["peak_cool_kW"] if include_cooling else 0.0)) / 1000.0
    economics, _scale = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    if include_cooling:
        economics["counterfactual"] = "individual_gas_and_ac"
    scenario = {
        "name": f"{tech_label} — trunk {route_m:.0f}m", "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "generic_length", "length_m": float(route_m), "include_cooling": include_cooling,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                    "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    if include_cooling:
        scenario["cooling_sources"] = _map_sources(rec["cooling_sources"])
    return scenario


def _extrapolated_payback_years(inv, discount_rate=0.105, max_years=250):
    """Discounted payback in years — if it already happens within the
    modelled project life, return that directly. If not, extrapolate by
    holding the LAST modelled year's net cash flow flat indefinitely and
    finding when the cumulative discounted position would turn positive.
    This is clearly an extrapolation beyond the 40-year modelled lifetime
    (flagged via the second return value), but turns "no payback" — a
    single flat outcome that can't be compared across technologies — into
    an actual, comparable number: HOW FAR beyond the project life each
    stack would need, which is real information a flat cap throws away.

    Returns (years, is_extrapolated) — years is None only if the last
    modelled year's net cash flow is itself negative (truly never pays
    back under any horizon at unchanged prices).
    """
    p = inv.get("discounted_payback_years")
    life = len(inv["net_cashflow_GBP"]) - 1
    if p is not None and p <= life:
        return float(p), False
    net = np.asarray(inv["net_cashflow_GBP"], dtype=float)
    last_net = net[-1]
    if last_net <= 0:
        return None, True
    balance = inv["cumulative_discounted_GBP"][-1]
    year = life
    while balance < 0 and year < max_years:
        year += 1
        balance += last_net / (1 + discount_rate) ** year
    return (float(year) if balance >= 0 else None), True


def _three_panel_bar(df, title, filename, out_dir, life_years=40):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
    x = np.arange(len(df))
    colors = [TECH_COLOR[t] for t in df["Technology"]]
    labels = [TECH_SHORT[t] for t in df["Technology"]]

    ax = axes[0]
    ax.bar(x, df["Investor NPV (£m)"], color=colors, width=0.6, zorder=3)
    ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Investor NPV (£m)")
    ax.set_title("Net present value", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.bar(x, df["Carbon intensity (gCO2e/kWh)"], color=colors, width=0.6, zorder=3)
    thresh = df["Carbon threshold (gCO2e/kWh)"].iloc[0]
    ax.axhline(thresh, color=C_ORANGE, linestyle="--", linewidth=1.5, zorder=2,
               label=f"Compliance threshold ({thresh:.0f} gCO2e/kWh)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Carbon intensity (gCO2e/kWh)")
    ax.set_title("Carbon intensity of heat delivered", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    payback = df["Discounted payback (yrs)"].tolist()
    extrapolated = df["Payback extrapolated beyond 40y?"].tolist()
    cap_years = max([life_years * 1.5] + [p for p in payback if p is not None]) * 1.12
    bar_vals = [p if p is not None else cap_years for p in payback]
    hatches = ["///" if extrapolated[i] and payback[i] is not None else
               ("xxx" if payback[i] is None else None) for i in range(len(payback))]
    bar_colors = [colors[i] if payback[i] is not None else GRID for i in range(len(payback))]
    bars = ax.bar(x, bar_vals, color=bar_colors, width=0.6, zorder=3, edgecolor=INK2, linewidth=0.6)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    for xi, p, ext in zip(x, payback, extrapolated):
        if p is None:
            label = "never pays back\n(flat extrapolation)"
        elif ext:
            label = f"~{p:.0f}y\n(extrapolated)"
        else:
            label = f"{p:.1f}y"
        ax.text(xi, bar_vals[x.tolist().index(xi)] + cap_years * 0.015, label,
                ha="center", va="bottom", fontsize=8.2, color=INK2)
    ax.axhline(life_years, color=INK, linestyle=":", linewidth=1.4, zorder=2, label=f"{life_years}-year project life")
    ax.set_ylim(0, cap_years * 1.20)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Discounted payback period (years)")
    ax.set_title("Discounted payback (hatched = beyond 40-yr project life)", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, y=1.02)
    _save(fig, filename, out_dir)


SWEEP_LENGTHS_M = [250, 400, 600, 900, 1300, 1800, 2500, 3500, 5000, 7000, 10000, 14000, 19000]
LENGTH_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
CLIMATE_SCENARIOS = ["baseline", "2050_central", "2050_high"]
CLIMATE_LABEL = {"baseline": "Baseline", "2050_central": "2050 central\n(RCP4.5)", "2050_high": "2050 high\n(RCP8.5+UHI)"}


def run_study(network_label, buildings, segments, out_dir):
    """Produce all 5 comparison figures (+ CSVs) for one network — same
    method used for every site this is run against. See module docstring
    for the 3 technology stacks and the shared economic assumptions.

    Parameters
    ----------
    network_label : short display name, e.g. "Central Exeter" or
                     "Ealing Town Centre" — used in titles/prints only.
    buildings       : list of building dicts (demand_synthesis format)
    segments        : list of {"node_id","parent_id","length_m","building"}
                       dicts — the real tree topology for this network.
    out_dir         : Path — CSVs and PNGs are written here (created if
                       it doesn't exist).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_total_length_m = sum(seg["length_m"] for seg in segments)
    print(f"{network_label} network: {len(buildings)} buildings, "
          f"{base_total_length_m:.0f} m total route.")

    # ── GRAPH 1 — heating only (2-pipe), baseline climate, gas-parity
    # revenue, GHNF applied. NPV / carbon intensity / payback. ──────────────
    g1_rows = []
    for tech in TECH_LABELS:
        scenario, demand, rec = build_tree_scenario(tech, buildings, segments, include_cooling=False)
        result = run_scenario(scenario)
        h, inv, grant = result["headline"], result["financial"]["investor"], result["grant"]
        g1_rows.append({
            "Technology": tech,
            "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
            "GHNF grant (£m)": round((grant["grant_GBP"] if grant else 0.0) / 1e6, 2),
            "Carbon intensity (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
            "Carbon threshold (gCO2e/kWh)": h["carbon_threshold_gCO2e_per_kWh"],
            "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
            "Discounted payback (yrs)": _extrapolated_payback_years(inv)[0],
            "Payback extrapolated beyond 40y?": _extrapolated_payback_years(inv)[1],
            "Screening decision": result["screening"]["status"],
        })
    g1_df = pd.DataFrame(g1_rows)
    g1_df.to_csv(out_dir / "fig1_heating_only_comparison.csv", index=False)
    print(f"\n=== Graph 1: heating-only (2-pipe) source-stack comparison, {network_label} ===")
    print(g1_df.to_string(index=False))
    _three_panel_bar(
        g1_df, f"Heating-only (2-pipe) — {network_label}, baseline climate, gas-parity revenue, GHNF applied",
        "fig1_heating_only_comparison.png", out_dir,
    )

    # ── GRAPH 2 — 4-pipe (heating + cooling), baseline climate, gas- &
    # AC-parity revenue, GHNF applied. ──────────────────────────────────────
    g2_rows = []
    for tech in TECH_LABELS:
        scenario, demand, rec = build_tree_scenario(tech, buildings, segments, include_cooling=True)
        result = run_scenario(scenario)
        h, inv, grant = result["headline"], result["financial"]["investor"], result["grant"]
        g2_rows.append({
            "Technology": tech,
            "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
            "GHNF grant (£m)": round((grant["grant_GBP"] if grant else 0.0) / 1e6, 2),
            "Carbon intensity (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
            "Carbon threshold (gCO2e/kWh)": h["carbon_threshold_gCO2e_per_kWh"],
            "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
            "Discounted payback (yrs)": _extrapolated_payback_years(inv)[0],
            "Payback extrapolated beyond 40y?": _extrapolated_payback_years(inv)[1],
            "Screening decision": result["screening"]["status"],
        })
    g2_df = pd.DataFrame(g2_rows)
    g2_df.to_csv(out_dir / "fig2_fourpipe_comparison.csv", index=False)
    print(f"\n=== Graph 2: 4-pipe (heating+cooling) source-stack comparison, {network_label} ===")
    print(g2_df.to_string(index=False))
    _three_panel_bar(
        g2_df, f"4-pipe heating + cooling — {network_label}, baseline climate, gas- & AC-parity revenue, GHNF applied",
        "fig2_fourpipe_comparison.png", out_dir,
    )
    print(f"\nWrote graphs 1-2 to {out_dir}")

    # ── GRAPH 3 — required break-even tariff vs achieved linear density,
    # heating-only vs heating+cooling. See module docstring on why this is
    # a curve, not an extrapolated "density needed" bar chart. ─────────────
    sweep_rows = []
    for tech in TECH_LABELS:
        for duty, include_cooling in [("Heating only", False), ("Heating + cooling", True)]:
            for length_m in SWEEP_LENGTHS_M:
                scenario = build_generic_scenario(tech, buildings, length_m, include_cooling=include_cooling)
                result = run_scenario(scenario)
                h, inv = result["headline"], result["financial"]["investor"]
                heat_d = h["linear_heat_density_MWh_per_m_year"] or 0.0
                cool_d = (h.get("linear_cooling_density_MWh_per_m_year") or 0.0) if include_cooling else 0.0
                sweep_rows.append({
                    "Technology": tech, "Duty": duty, "Route (m)": length_m,
                    "Density (MWh/m/yr)": heat_d + cool_d,
                    "Required break-even tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
                    "Modelled parity tariff (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
                })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "fig3_density_vs_required_tariff.csv", index=False)

    summary_rows = []
    for tech in TECH_LABELS:
        for duty in ["Heating only", "Heating + cooling"]:
            sub = sweep_df[(sweep_df["Technology"] == tech) & (sweep_df["Duty"] == duty)]
            summary_rows.append({
                "Technology": tech, "Duty": duty,
                "Best required tariff in sweep (p/kWh)": round(sub["Required break-even tariff (p/kWh)"].min(), 2),
                "...at density (MWh/m/yr)": round(
                    sub.loc[sub["Required break-even tariff (p/kWh)"].idxmin(), "Density (MWh/m/yr)"], 2,
                ),
                "Modelled parity tariff (p/kWh)": round(sub["Modelled parity tariff (p/kWh)"].mean(), 2),
                "Still x parity tariff, even at best density": round(
                    sub["Required break-even tariff (p/kWh)"].min() / sub["Modelled parity tariff (p/kWh)"].mean(), 1,
                ),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "fig3_density_breakeven_summary.csv", index=False)
    print(f"\n=== Graph 3: required break-even tariff vs achieved linear density, {network_label} ===")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, duty in zip(axes, ["Heating only", "Heating + cooling"]):
        for tech in TECH_LABELS:
            sub = sweep_df[(sweep_df["Technology"] == tech) & (sweep_df["Duty"] == duty)].sort_values("Density (MWh/m/yr)")
            ax.plot(sub["Density (MWh/m/yr)"], sub["Required break-even tariff (p/kWh)"],
                    color=TECH_COLOR[tech], linewidth=2.0, marker="o", markersize=4.2,
                    label=TECH_SHORT[tech], zorder=3)
        parity = sweep_df[sweep_df["Duty"] == duty]["Modelled parity tariff (p/kWh)"].mean()
        ax.axhline(parity, color=C_ORANGE, linestyle="--", linewidth=1.6, zorder=2,
                   label=f"Modelled parity tariff (~{parity:.1f} p/kWh)")
        for tech in TECH_LABELS:
            real_scenario, real_demand, real_rec = build_tree_scenario(
                tech, buildings, segments, include_cooling=(duty == "Heating + cooling"),
            )
            real_result = run_scenario(real_scenario)
            rh, rinv = real_result["headline"], real_result["financial"]["investor"]
            rd = (rh["linear_heat_density_MWh_per_m_year"] or 0.0) + (
                (rh.get("linear_cooling_density_MWh_per_m_year") or 0.0) if duty == "Heating + cooling" else 0.0
            )
            ax.scatter([rd], [rinv["required_heat_tariff_p_per_kWh_for_zero_NPV"]], marker="*", s=220,
                       color=TECH_COLOR[tech], edgecolor=INK, linewidth=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("Linear density (MWh / route metre / year, log scale)")
        ax.set_title(duty, loc="left", fontsize=12, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Required break-even heat tariff (p/kWh)")
    axes[1].legend(fontsize=8.2, frameon=True, facecolor="#fcfcfb", edgecolor="none", loc="upper right")
    fig.suptitle(
        f"What linear density would each stack need to break even? "
        f"(★ = the real {base_total_length_m:.0f}m {network_label} route)",
        x=0.01, ha="left", fontsize=13, color=INK, y=1.03,
    )
    _save(fig, "fig3_density_breakeven.png", out_dir)
    print(f"\nWrote graph 3 to {out_dir}")

    # ── GRAPH 4 — branched tree (scaled proportionally) vs a single
    # equivalent trunk of the same total length, as route length scales. ───
    def _scaled_segments(multiplier):
        return [dict(seg, length_m=seg["length_m"] * multiplier) for seg in segments]

    topology_rows = []
    for tech in TECH_LABELS:
        for mult in LENGTH_MULTIPLIERS:
            total_length_m = base_total_length_m * mult
            tree_scenario, _, _ = build_tree_scenario(
                tech, buildings, _scaled_segments(mult), include_cooling=False,
            )
            tree_result = run_scenario(tree_scenario)
            trunk_scenario = build_generic_scenario(tech, buildings, total_length_m, include_cooling=False)
            trunk_result = run_scenario(trunk_scenario)
            for shape, result in [("Tree (branched)", tree_result), ("Single trunk", trunk_result)]:
                h, inv = result["headline"], result["financial"]["investor"]
                topology_rows.append({
                    "Technology": tech, "Shape": shape, "Length multiplier": mult,
                    "Total route (m)": round(total_length_m, 0),
                    "Network CAPEX (£m)": round(h["capex_network_GBP"] / 1e6, 3),
                    "Total CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
                    "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
                })
    topology_df = pd.DataFrame(topology_rows)
    topology_df.to_csv(out_dir / "fig4_tree_vs_trunk.csv", index=False)
    print(f"\n=== Graph 4: branched tree vs single-trunk economics as route length scales, {network_label} ===")
    print(topology_df[topology_df["Length multiplier"] == 1.0].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for tech in TECH_LABELS:
        tree_sub = topology_df[(topology_df["Technology"] == tech) & (topology_df["Shape"] == "Tree (branched)")].sort_values("Total route (m)")
        trunk_sub = topology_df[(topology_df["Technology"] == tech) & (topology_df["Shape"] == "Single trunk")].sort_values("Total route (m)")
        axes[0].plot(tree_sub["Total route (m)"], tree_sub["Network CAPEX (£m)"], color=TECH_COLOR[tech],
                     linewidth=2.0, marker="o", markersize=4.2, linestyle="-", zorder=3)
        axes[0].plot(trunk_sub["Total route (m)"], trunk_sub["Network CAPEX (£m)"], color=TECH_COLOR[tech],
                     linewidth=2.0, marker="s", markersize=4.2, linestyle="--", zorder=3)
        axes[1].plot(tree_sub["Total route (m)"], tree_sub["Investor NPV (£m)"], color=TECH_COLOR[tech],
                     linewidth=2.0, marker="o", markersize=4.2, linestyle="-", zorder=3)
        axes[1].plot(trunk_sub["Total route (m)"], trunk_sub["Investor NPV (£m)"], color=TECH_COLOR[tech],
                     linewidth=2.0, marker="s", markersize=4.2, linestyle="--", zorder=3)
    axes[0].axvline(base_total_length_m, color=INK, linestyle=":", linewidth=1.2, zorder=2)
    axes[1].axvline(base_total_length_m, color=INK, linestyle=":", linewidth=1.2, zorder=2,
                    label=f"Real {network_label} route ({base_total_length_m:.0f} m)")
    axes[1].axhline(0, color=INK, linewidth=1.0, zorder=2)
    axes[0].set_xlabel("Total route length (m)"); axes[0].set_ylabel("Network (pipework) CAPEX (£m)")
    axes[0].set_title("Pipework cost: branching lets pipe taper, a trunk can't", loc="left", fontsize=11.5)
    axes[1].set_xlabel("Total route length (m)"); axes[1].set_ylabel("Investor NPV (£m)")
    axes[1].set_title("Net present value", loc="left", fontsize=11.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    handles = (
        [Line2D([0], [0], color=MUTED, linewidth=2.0, linestyle="-", marker="o", label="Tree (branched)"),
         Line2D([0], [0], color=MUTED, linewidth=2.0, linestyle="--", marker="s", label="Single trunk")]
        + [Line2D([0], [0], color=TECH_COLOR[t], linewidth=2.0, label=TECH_SHORT[t]) for t in TECH_LABELS]
        + [Line2D([0], [0], color=INK, linewidth=1.2, linestyle=":", label=f"Real route ({base_total_length_m:.0f} m)")]
    )
    axes[1].legend(handles=handles, fontsize=7.6, frameon=True, facecolor="#fcfcfb", edgecolor="none", loc="lower left")
    fig.suptitle(
        f"Same demand, same total route length — branched tree vs a single equivalent trunk "
        f"({network_label}, heating only, baseline climate)",
        x=0.01, ha="left", fontsize=13, color=INK, y=1.03,
    )
    _save(fig, "fig4_tree_vs_trunk.png", out_dir)
    print(f"\nWrote graph 4 to {out_dir}")

    # ── GRAPH 5 — climate change sensitivity: baseline / 2050_central /
    # 2050_high, 4-pipe, plant re-sized FRESH per climate scenario.
    # Under a big enough cooling-demand jump (2050_high can push cooling
    # demand up by an order of magnitude — see the Central Exeter run,
    # where it rose ~20x), the coincident cooling peak can exceed what a
    # single standard pipe can carry (DN600 is the largest in the
    # catalogue) — a real physical limit, not a bug. Handled the same way
    # analysis/exeter_climate_fourpipe.py already handles it: catch it,
    # record it plainly, and keep going rather than crash the whole study.
    # Demand figures (panel 1) are computed independently of technology
    # sizing/dispatch, so they're unaffected even when a given tech's
    # scenario run fails this way. ──────────────────────────────────────
    baseline_climate_ref = compute_climate_reference(apply_climate_scenario(weather, "baseline"))
    demand_rows = []
    climate_rows = []
    for climate_key in CLIMATE_SCENARIOS:
        climate_weather = apply_climate_scenario(weather, climate_key)
        climate_demand = synthesise_network(
            climate_weather, {"demand_nodes": deepcopy(buildings)}, climate_reference=baseline_climate_ref,
        )
        demand_rows.append({
            "Climate": climate_key,
            "Annual heat demand (MWh)": float(climate_demand["total_heat_kW"].sum() / 1000.0),
            "Annual cooling demand (MWh)": float(climate_demand["total_cooling_kW"].sum() / 1000.0),
        })
        for tech in TECH_LABELS:
            scenario, demand, rec = build_tree_scenario(
                tech, buildings, segments, include_cooling=True,
                climate_scenario=climate_key, climate_reference=baseline_climate_ref, weather_df=climate_weather,
            )
            try:
                result = run_scenario(scenario)
            except ValueError as exc:
                climate_rows.append({
                    "Climate": climate_key, "Technology": tech,
                    "CAPEX (£m)": None, "Carbon intensity (gCO2e/kWh)": None, "Investor NPV (£m)": None,
                    "Screening decision": "EXCEEDS PIPE CATALOG",
                })
                print(f"  {climate_key:14} {tech:42} EXCEEDS PIPE CATALOG — cooling peak too large for a "
                      f"single standard main under this climate ({exc})")
                continue
            h, inv = result["headline"], result["financial"]["investor"]
            climate_rows.append({
                "Climate": climate_key, "Technology": tech,
                "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
                "Carbon intensity (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
                "Screening decision": result["screening"]["status"],
            })
    demand_only_df = pd.DataFrame(demand_rows).set_index("Climate")
    climate_df = pd.DataFrame(climate_rows).merge(
        pd.DataFrame(demand_rows), on="Climate", how="left",
    )[["Climate", "Technology", "Annual heat demand (MWh)", "Annual cooling demand (MWh)",
       "CAPEX (£m)", "Carbon intensity (gCO2e/kWh)", "Investor NPV (£m)", "Screening decision"]]
    climate_df.to_csv(out_dir / "fig5_climate_sensitivity.csv", index=False)
    print(f"\n=== Graph 5: climate scenario sensitivity, {network_label} 4-pipe ===")
    print(climate_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    demand_df = demand_only_df.loc[CLIMATE_SCENARIOS]
    xw = np.arange(len(CLIMATE_SCENARIOS))
    bw = 0.35
    axes[0].bar(xw - bw / 2, demand_df["Annual heat demand (MWh)"], width=bw, color=C_RED, label="Heating", zorder=3)
    axes[0].bar(xw + bw / 2, demand_df["Annual cooling demand (MWh)"], width=bw, color=C_BLUE, label="Cooling", zorder=3)
    axes[0].set_xticks(xw); axes[0].set_xticklabels([CLIMATE_LABEL[c] for c in CLIMATE_SCENARIOS])
    axes[0].set_ylabel("Annual demand (MWh/yr)")
    axes[0].set_title("Demand shifts as climate warms", loc="left", fontsize=11.5)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    bw2 = 0.25
    for i, tech in enumerate(TECH_LABELS):
        sub = climate_df[climate_df["Technology"] == tech].set_index("Climate").loc[CLIMATE_SCENARIOS]
        offset = (i - 1) * bw2
        npv_vals = sub["Investor NPV (£m)"]
        axes[1].bar(xw + offset, npv_vals.fillna(0.0), width=bw2, color=TECH_COLOR[tech],
                    label=TECH_SHORT[tech], zorder=3)
        for xi, v in zip(xw + offset, npv_vals):
            if pd.isna(v):
                axes[1].text(xi, 0.0, "exceeds\npipe\ncatalog", ha="center", va="bottom",
                              fontsize=6.6, color=INK2, rotation=0)
    axes[1].axhline(0, color=INK, linewidth=1.0, zorder=2)
    axes[1].set_xticks(xw); axes[1].set_xticklabels([CLIMATE_LABEL[c] for c in CLIMATE_SCENARIOS])
    axes[1].set_ylabel("Investor NPV (£m)")
    axes[1].set_title("Network economics under each climate scenario", loc="left", fontsize=11.5)
    axes[1].legend(fontsize=8.2, frameon=False, loc="lower left")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Climate sensitivity — {network_label}, 4-pipe heating+cooling, plant freshly re-sized per scenario",
        x=0.01, ha="left", fontsize=13, color=INK, y=1.03,
    )
    _save(fig, "fig5_climate_sensitivity.png", out_dir)
    print(f"\nWrote graph 5 to {out_dir}")
    print(f"\nAll 5 graphs written to {out_dir}")

    return {"g1": g1_df, "g2": g2_df, "g3_summary": summary_df, "g4": topology_df, "g5": climate_df}
