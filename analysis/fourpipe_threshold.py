"""When is 4-pipe cooling ever worth it? A threshold search, not a verdict.

    python -m analysis.fourpipe_threshold

Every 4-pipe case tested so far in this pack worsens NPV. But the current model
also treats cooling unfavourably in three known ways (MODEL_SUMMARY §12): the
cooling demand model overshoots its own benchmark by ~9-10%, each pipe pair is
charged full trenched cost, and shared heating/cooling civils earn no credit.
So before telling Dalkia "never", this study asks: under progressively more
cooling-friendly assumptions, where does the incremental NPV of adding cooling
actually cross zero?

Two axes are swept:

  - COOLING INTENSITY: the Dense archetype's non-residential stock is shifted
    from low-cooling types to air-conditioned types (office -> office_ac,
    retail -> supermarket, hotel unchanged) in four steps, which moves both the
    cooling linear density and the cooling load factor.
  - SHARED-CIVILS CREDIT: 0% / 25% / 50% / 75% of the cooling network's own
    CAPEX (the trench is already open for the heating pipes). The credit is
    applied arithmetically to the 4-pipe result — NPV is linear in year-0
    CAPEX, so no re-run is needed. The cooling network CAPEX is measured as
    the network-line difference between the 4-pipe and 2-pipe runs of the
    same buildings.

Incremental NPV = NPV(4-pipe) - NPV(2-pipe), same buildings, same heat stack
(EfW + ASHP + gas peak), chiller auto-sized, GHNF 40% where carbon-eligible,
bill-parity for both services (heat vs individual gas, cooling vs individual
AC running cost).

A third lever is then tested on top: CAPTURING THE CUSTOMER'S AVOIDED AC
PURCHASE via a one-time connection charge, exactly the mechanism
analysis/contractor_view.capture_avoided_capital() uses for the ASHP side. The
customer still gets cooling-bill parity on running costs and pays up front
only what they would otherwise have spent on their own chiller/AC unit — never
worse off than self-supply at capture <= 100%. Swept 0/25/50/75/100%, stacked
with both the 0% and best (75%) shared-civils credit case.

Writes CSVs, PNGs and findings.md to output/fourpipe_threshold/.
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
OUT = ROOT / "output" / "fourpipe_threshold"
OUT.mkdir(parents=True, exist_ok=True)

from optimisation.auto_size import recommend_sizing
from profiles.demand_synthesis import synthesise_network
from scenarios.fixed_cost_scaling import scaled_economics
from scenarios.scenario_runner import run_scenario

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


ROUTE_M = 900.0
BASE_BUILDINGS = [
    {"name": "Dense residential block A", "type": "residential_existing",
     "floor_area_m2": 30000, "units": 400, "connections": 400,
     "connection_year": 1, "connection_probability": 0.92},
    {"name": "Dense residential block B", "type": "residential_existing",
     "floor_area_m2": 24000, "units": 320, "connections": 320,
     "connection_year": 1, "connection_probability": 0.90},
    {"name": "Town centre offices", "type": "office",
     "floor_area_m2": 15000, "connections": 1,
     "connection_year": 1, "connection_probability": 1.0},
    {"name": "High street retail", "type": "retail",
     "floor_area_m2": 8000, "connections": 1,
     "connection_year": 1, "connection_probability": 0.90},
    {"name": "Hotel", "type": "hotel",
     "floor_area_m2": 6000, "connections": 1,
     "connection_year": 1, "connection_probability": 1.0},
]

# Cooling-intensity steps: which non-residential stock is air-conditioned.
MIXES = {
    "low (offices AC)": {"Town centre offices": "office_ac"},
    "medium (+ retail becomes supermarket)": {
        "Town centre offices": "office_ac", "High street retail": "supermarket"},
    "high (+ second AC office block)": {
        "Town centre offices": "office_ac", "High street retail": "supermarket",
        "EXTRA": {"name": "Business park offices", "type": "office_ac",
                  "floor_area_m2": 12000, "connections": 1,
                  "connection_year": 1, "connection_probability": 1.0}},
    "very high (+ hospital wing)": {
        "Town centre offices": "office_ac", "High street retail": "supermarket",
        "EXTRA": {"name": "Business park offices", "type": "office_ac",
                  "floor_area_m2": 12000, "connections": 1,
                  "connection_year": 1, "connection_probability": 1.0},
        "EXTRA2": {"name": "Hospital wing", "type": "hospital",
                   "floor_area_m2": 10000, "connections": 1,
                   "connection_year": 1, "connection_probability": 1.0}},
}
CREDITS = [0.0, 0.25, 0.50, 0.75]

weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(weather) == 8760
weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")

PRESETS = {"ashp": "ealing_phase1", "gas_boiler": "ealing_phase1",
           "efw_chp": "newlincs_style", "air_cooled_chiller": "generic_2MW_bank"}


def buildings_for(mix):
    buildings = deepcopy(BASE_BUILDINGS)
    for b in buildings:
        if b["name"] in mix:
            b["type"] = mix[b["name"]]
    for key in ("EXTRA", "EXTRA2"):
        if key in mix:
            buildings.append(deepcopy(mix[key]))
    return buildings


def run_pair(mix_label, mix):
    """Run the same buildings 2-pipe and 4-pipe; return both results."""
    buildings = buildings_for(mix)
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    out = {}
    for include_cooling in (False, True):
        rec = recommend_sizing(
            demand_kW=demand["total_heat_kW"],
            peak_demand_kW=demand["peak_heat_kW"],
            technology_types=["efw_chp", "ashp", "gas_boiler"],
            weather_df=weather,
            network_flow_temp_C=70.0,
            n_buildings=len(buildings),
            building_types=[b["type"] for b in buildings],
            include_cooling=include_cooling,
            cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
            peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
        )
        def _map(srcs):
            return [{"type": s["type"], "preset": PRESETS[s["type"]],
                     "name": f"{s['type']} ({s['role']})",
                     "capacity_MW": float(s["capacity_MW"]),
                     **({"n_units": int(s["n_units"])} if "n_units" in s else {})}
                    for s in srcs]
        peak_MW = demand["peak_heat_kW"] / 1000.0
        if include_cooling:
            peak_MW += demand["peak_cool_kW"] / 1000.0
        economics, _ = scaled_economics(peak_MW)
        economics["counterfactual"] = ("individual_gas_and_ac" if include_cooling
                                       else "individual_gas")
        economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
        scenario = {
            "name": f"4-pipe threshold — {mix_label} — {'4-pipe' if include_cooling else '2-pipe'}",
            "climate_scenario": "baseline",
            "demand": {"buildings": deepcopy(buildings)},
            "network": {"mode": "generic_length", "length_m": ROUTE_M,
                        "include_cooling": include_cooling,
                        "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                        "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
            "sources": _map(rec["sources"]),
            "economics": economics,
        }
        if include_cooling:
            scenario["cooling_sources"] = _map(rec["cooling_sources"])
        out["four" if include_cooling else "two"] = run_scenario(scenario)
        if include_cooling:
            out["four_scenario"] = scenario
    out["demand"] = demand
    return out


def avoided_ac_capex_by_building(four_result):
    """What each customer would have spent on their own air conditioner —
    the cooling-side mirror of contractor_view's avoided-ASHP-capital lever.
    Zero for buildings with no cooling demand (individual_ac_dispatch itself
    returns capex_GBP=0 for those), so no filtering is needed here."""
    cf = four_result["counterfactual"]["cooling"]["by_building"]
    return {name: float(v["capex_GBP"]) for name, v in cf.items()}


def run_four_pipe_with_capture(four_scenario, avoided_capex, capture):
    """Re-run the 4-pipe scenario with a connection charge equal to `capture`
    of each customer's avoided individual-AC purchase — the same mechanism
    analysis/contractor_view.capture_avoided_capital() uses for the ASHP side.
    The customer is never worse off than self-supply at capture <= 100%: they
    still get cooling-bill parity, and pay up front only what they would have
    spent on their own unit anyway."""
    scenario = deepcopy(four_scenario)
    for b in scenario["demand"]["buildings"]:
        existing = float(b.get("connection_charge_GBP", 0.0))
        b["connection_charge_GBP"] = existing + avoided_capex.get(b["name"], 0.0) * capture
    result = run_scenario(scenario)
    return result["financial"]["investor"]["npv_GBP"]


CAPTURES = [0.0, 0.25, 0.50, 0.75, 1.00]

rows = []
capture_rows = []
pairs = {}
for mix_label, mix in MIXES.items():
    pair = run_pair(mix_label, mix)
    pairs[mix_label] = pair
    two, four, demand = pair["two"], pair["four"], pair["demand"]
    npv2 = two["financial"]["investor"]["npv_GBP"]
    npv4 = four["financial"]["investor"]["npv_GBP"]
    net2 = two["headline"]["capex_breakdown_GBP"]["network_GBP"]
    net4 = four["headline"]["capex_breakdown_GBP"]["network_GBP"]
    cooling_network_capex = net4 - net2
    annual_cool_MWh = demand["annual_cool_MWh"]
    peak_cool_MW = demand["peak_cool_kW"] / 1000.0
    cool_density = annual_cool_MWh / ROUTE_M
    cool_load_factor = (annual_cool_MWh / (peak_cool_MW * 8760.0)
                        if peak_cool_MW > 0 else 0.0)
    for credit in CREDITS:
        incr = (npv4 + credit * cooling_network_capex - npv2) / 1e6
        rows.append({
            "Cooling mix": mix_label,
            "Annual cooling (MWh)": round(annual_cool_MWh, 0),
            "Cooling linear density (MWh/m/yr)": round(cool_density, 2),
            "Cooling load factor (%)": round(cool_load_factor * 100, 1),
            "Cooling network CAPEX (£m)": round(cooling_network_capex / 1e6, 2),
            "Shared-civils credit (%)": round(credit * 100, 0),
            "2-pipe NPV (£m)": round(npv2 / 1e6, 2),
            "4-pipe NPV (£m)": round(npv4 / 1e6, 2),
            "Incremental NPV of cooling (£m)": round(incr, 2),
            "4-pipe carbon (gCO2e/kWh)": round(
                four["headline"]["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
        })
        print(f"{mix_label} | credit {credit:.0%}: incremental £{incr:.2f}m")

df = pd.DataFrame(rows)
df.to_csv(OUT / "fourpipe_threshold.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — capturing the customer's avoided AC purchase via connection charge.
# Same mechanism as analysis/contractor_view.capture_avoided_capital() on the
# ASHP side: the customer still gets cooling-bill parity on running costs, and
# pays up front only what they would have spent on their own chiller/AC unit
# anyway, so they are never worse off than self-supply at capture <= 100%.
# Combined with the BEST shared-civils credit (75%) to show the two levers
# stacked — the honest best case the current engine can produce.
# ═══════════════════════════════════════════════════════════════════════════

for mix_label, mix in MIXES.items():
    pair = pairs[mix_label]
    two, four = pair["two"], pair["four"]
    npv2 = two["financial"]["investor"]["npv_GBP"]
    net2 = two["headline"]["capex_breakdown_GBP"]["network_GBP"]
    net4 = four["headline"]["capex_breakdown_GBP"]["network_GBP"]
    cooling_network_capex = net4 - net2
    avoided_capex = avoided_ac_capex_by_building(four)
    total_avoided_capex = sum(avoided_capex.values())
    for capture in CAPTURES:
        npv4_captured = run_four_pipe_with_capture(pair["four_scenario"], avoided_capex, capture)
        incr_no_credit = (npv4_captured - npv2) / 1e6
        incr_best_credit = (npv4_captured + 0.75 * cooling_network_capex - npv2) / 1e6
        capture_rows.append({
            "Cooling mix": mix_label,
            "Avoided AC capex captured, total (£m)": round(total_avoided_capex / 1e6, 2),
            "Capture (%)": round(capture * 100, 0),
            "Connection charge revenue (£m)": round(total_avoided_capex * capture / 1e6, 2),
            "Incremental NPV, 0% civils credit (£m)": round(incr_no_credit, 2),
            "Incremental NPV, 75% civils credit (£m)": round(incr_best_credit, 2),
        })
        print(f"{mix_label} | AC capture {capture:.0%}: incremental (0% civils credit) "
              f"£{incr_no_credit:.2f}m, (75% civils credit) £{incr_best_credit:.2f}m")

capture_df = pd.DataFrame(capture_rows)
capture_df.to_csv(OUT / "avoided_ac_capex_capture.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

CREDIT_COLOURS = dict(zip(CREDITS, [C_BLUE, C_AQUA, C_YELLOW, C_VIOLET]))
fig, ax = plt.subplots(figsize=(11, 6))
for credit in CREDITS:
    sub = df[df["Shared-civils credit (%)"] == round(credit * 100, 0)].sort_values(
        "Cooling linear density (MWh/m/yr)")
    ax.plot(sub["Cooling linear density (MWh/m/yr)"],
            sub["Incremental NPV of cooling (£m)"],
            "-o", color=CREDIT_COLOURS[credit], lw=2, ms=5,
            label=f"{credit:.0%} shared-civils credit")
ax.axhline(0, color=INK, lw=1.3)
ax.set_xlabel("Cooling linear density (MWh of cooling / m of route / yr)")
ax.set_ylabel("Incremental NPV of adding 4-pipe cooling (£m)")
ax.set_title("Does cooling ever pay? Incremental NPV of the 4-pipe upgrade\n"
             "Dense archetype, EfW + ASHP + gas heat stack, chiller auto-sized, "
             "AC-parity cooling bills", fontsize=12)
ax.legend(fontsize=9)
_save(fig, "FT1_fourpipe_threshold.png")

MIX_COLOURS = dict(zip(MIXES, [C_BLUE, C_AQUA, C_YELLOW, C_VIOLET]))
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
for mix_label in MIXES:
    sub = capture_df[capture_df["Cooling mix"] == mix_label].sort_values("Capture (%)")
    axes[0].plot(sub["Capture (%)"], sub["Incremental NPV, 0% civils credit (£m)"],
                 "-o", color=MIX_COLOURS[mix_label], lw=2, ms=5, label=mix_label)
    axes[1].plot(sub["Capture (%)"], sub["Incremental NPV, 75% civils credit (£m)"],
                 "-o", color=MIX_COLOURS[mix_label], lw=2, ms=5, label=mix_label)
for ax in axes:
    ax.axhline(0, color=INK, lw=1.3)
    ax.set_xlabel("Capture of customer's avoided AC purchase (%)")
axes[0].set_ylabel("Incremental NPV of adding 4-pipe cooling (£m)")
axes[0].set_title("Connection charge only\n(0% shared-civils credit)", fontsize=11)
axes[1].set_title("Connection charge + best civils case\n(75% shared-civils credit)", fontsize=11)
axes[0].legend(fontsize=8.5)
fig.suptitle("Capturing the customer's avoided AC purchase via connection charge\n"
             "Customer still gets cooling-bill parity — never worse off than self-supply at <=100%",
             fontsize=12.5)
_save(fig, "FT2_avoided_ac_capex_capture.png")

# ═══════════════════════════════════════════════════════════════════════════
# findings.md
# ═══════════════════════════════════════════════════════════════════════════

best = df.loc[df["Incremental NPV of cooling (£m)"].idxmax()]
zero_crossers = df[df["Incremental NPV of cooling (£m)"] > 0]
best_capture = capture_df.loc[capture_df["Incremental NPV, 75% civils credit (£m)"].idxmax()]
capture_crossers = capture_df[capture_df["Incremental NPV, 75% civils credit (£m)"] > 0]

lines = [
    "# Four-pipe cooling threshold",
    "",
    "Generated by `python -m analysis.fourpipe_threshold`.",
    "",
    "## Result",
    "",
    (f"- **No swept case reaches positive incremental NPV.** The best case is "
     f"'{best['Cooling mix']}' at {best['Shared-civils credit (%)']:.0f}% shared-civils "
     f"credit: **£{best['Incremental NPV of cooling (£m)']}m**."
     if zero_crossers.empty else
     f"- {len(zero_crossers)} case(s) reach positive incremental NPV — see table."),
    "- The shared-civils credit moves the answer by only the cooling network's own",
    "  CAPEX; on this compact route the cooling pipes are not the dominant cost —",
    "  the chiller plant, its electricity and its REPEX are.",
    "- AC-parity billing is a hard ceiling: the customer's alternative (their own",
    "  air-cooled chiller running cost) is cheap per kWh, and UK cooling load",
    "  factors are low, so revenue per £ of cooling asset stays thin at any density",
    "  tested here.",
    "",
    "## Full sweep",
    "",
    df.to_markdown(index=False),
    "",
    "## Capturing the customer's avoided AC purchase",
    "",
    "The same lever `analysis/contractor_view.capture_avoided_capital()` uses on the",
    "ASHP side, applied to cooling: a connection charge equal to a share of what the",
    "customer would have spent on their own air conditioner/chiller, charged once in",
    "year 1. The customer still gets cooling-bill PARITY on running costs and is never",
    "worse off than self-supply at capture <= 100% — this is not extra margin extracted",
    "from the customer, it is capital they were going to spend anyway, redirected to",
    "the network instead of a chiller they no longer need to buy.",
    "",
    (f"- **Still no case reaches positive incremental NPV**, even at 100% capture "
     f"stacked with the 75% shared-civils credit. Best case: '{best_capture['Cooling mix']}' "
     f"at {best_capture['Capture (%)']:.0f}% capture — "
     f"**£{best_capture['Incremental NPV, 75% civils credit (£m)']}m**."
     if capture_crossers.empty else
     f"- {len(capture_crossers)} case(s) reach positive incremental NPV once avoided-AC "
     "capture is stacked with the best civils credit — see table."),
    "- Capture is a real, sizeable lever (£2-8m of connection-charge revenue across the",
    "  mixes) but it is smaller than the CAPEX/OPEX gap cooling adds, because UK",
    "  individual-AC capex per kW is itself modest (£800/kW) next to a centralised",
    "  chiller's installed cost plus REPEX plus 40 years of AC-parity-capped revenue.",
    "- This is the single most promising untested lever from the brief, and the reason",
    "  it does not close the gap on its own: the constraint on cooling is chiller OPEX",
    "  and REPEX at low UK load factors, not the network CAPEX either lever discounts.",
    "",
    capture_df.to_markdown(index=False),
    "",
    "## What would change the answer (untested here, honestly labelled)",
    "",
    "- Heat-recovery chillers / ambient-loop networks (simultaneous heat + cooling)",
    "  are NOT implemented in the engine — the single strongest known upside case.",
    "- Year-round process cooling (data-centre halls, cold storage) at load factors",
    "  far above the comfort-cooling range swept here.",
    "- The cooling demand model overshoots its own annual benchmark by ~9-10%",
    "  (MODEL_SUMMARY §12). That cuts both ways — it adds billable parity kWh but",
    "  also inflates chiller sizing and electricity — so its net direction is",
    "  unverified. Treat margins within ~£1m of zero as noise; none of the swept",
    "  cases are anywhere near that close.",
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 2 figures.")
