"""Source-technology frontier, density frontier and source break-even prices.

    python -m analysis.source_frontier

Three questions, one study:

  1. WHICH SOURCE STACK? — every stack the engine supports, run on the same
     customer base, plotted as investor NPV (after any GHNF the model's own
     gates allow) against carbon intensity, with the 100 gCO2e/kWh GHNF
     boundary marked. "Best source" has to mean best *conditional on being
     grant-eligible*, because the grant follows the carbon gate.
  2. WHAT LINEAR DENSITY IS "GOOD"? — the same stacks swept across route
     lengths on three density archetypes. The honest answer the engine keeps
     giving: density is necessary but nowhere near sufficient at these
     connection counts, so the deliverable is the shape of the curve, not a
     single magic number.
  3. WHAT WOULD DALKIA PAY FOR A SOURCE? — break-even conditions expressed as
     negotiation numbers: the maximum EfW heat-export price, and the maximum
     transmission-leg distance to the EfW plant, at which the EfW stack still
     beats the next-best self-build alternative (ASHP + gas peak). The
     comparison is stack-vs-stack, not stack-vs-zero: no stack reaches zero
     NPV, so "worth paying" means "still the least-bad carbon-compliant
     design".

GHNF: enabled at the 40% base rate on every run; the engine itself zeroes the
grant where the 100 gCO2e/kWh gate fails and applies the <50% and 4.5p/kWh
caps (economics/grant.py). Billing is gas parity throughout (the pack's
headline customer proposition).

Writes CSVs, PNGs and findings.md to output/source_frontier/.
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
OUT = ROOT / "output" / "source_frontier"
OUT.mkdir(parents=True, exist_ok=True)

from optimisation.auto_size import recommend_sizing
from profiles.demand_synthesis import synthesise_network
from scenarios.fixed_cost_scaling import scaled_economics
from scenarios.scenario_runner import run_scenario
# 4-way comparison: three archetypes + the real validated Ealing Phase 1 case
# (honesty notes in analysis/archetypes.py). The per-stack frontier (SF1) stays
# Dense-only for legibility; the density sweep (SF2) shows all four.
from analysis.archetypes import ARCHETYPES_WITH_EALING as ARCHETYPES

# ── Palette (validated categorical set, see dataviz skill) ──────────────────
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


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Shared cases — same archetypes as the screening and affordability studies
# ═══════════════════════════════════════════════════════════════════════════

PRESET_FOR_TYPE = {
    "ashp": "ealing_phase1",
    "gas_boiler": "ealing_phase1",
    "electric_boiler": "ealing_backup",
    "data_centre": "redwire_ealing",
    "booster_heat_pump": "generic_2MW",
    "efw_chp": "newlincs_style",
}

# Stack key -> (auto_size technology list, swap the sized ASHP for this type)
# WSHP/GSHP are not auto-sized (optimisation/auto_size.py sizes ashp/efw/dc as
# baseload), so both are sized exactly like the ASHP they replace: same total
# capacity against the same load-duration curve — the difference under test is
# the COP physics of a stable 11-12C source vs winter air, not plant sizing.
STACKS = {
    "Gas-only (reference)": (["gas_boiler"], None),
    "Electric boiler only": (["electric_boiler"], None),
    "ASHP + gas peak": (["ashp", "gas_boiler"], None),
    "ASHP + electric peak": (["ashp", "electric_boiler"], None),
    "WSHP + gas peak": (["ashp", "gas_boiler"], ("wshp", "generic_river_5MW")),
    "GSHP + gas peak": (["ashp", "gas_boiler"], ("gshp", "birmingham_aston_university")),
    "EfW + ASHP + gas peak": (["efw_chp", "ashp", "gas_boiler"], None),
    "DC heat + booster + gas peak": (["data_centre", "gas_boiler"], None),
}
STACK_COLOURS = dict(zip(STACKS, [C_BLUE, C_AQUA, C_YELLOW, C_GREEN,
                                  C_VIOLET, C_RED, C_MAGENTA, C_ORANGE]))

weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(weather) == 8760
weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")

demand_cache = {
    label: synthesise_network(weather, {"demand_nodes": deepcopy(cfg["buildings"])})
    for label, cfg in ARCHETYPES.items()
}


def build_scenario(arch_label, stack_label, route_m=None, efw_overrides=None):
    cfg = ARCHETYPES[arch_label]
    demand = demand_cache[arch_label]
    tech_types, swap = STACKS[stack_label]
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"],
        peak_demand_kW=demand["peak_heat_kW"],
        technology_types=tech_types,
        weather_df=weather,
        network_flow_temp_C=70.0,
        n_buildings=len(cfg["buildings"]),
        building_types=[b["type"] for b in cfg["buildings"]],
    )
    sources = []
    for s in rec["sources"]:
        stype, preset = s["type"], PRESET_FOR_TYPE.get(s["type"])
        if swap and stype == "ashp":
            stype, preset = swap
        m = {"type": stype, "preset": preset,
             "name": f"{stype} ({s['role']})", "capacity_MW": float(s["capacity_MW"])}
        if "n_units" in s:
            m["n_units"] = int(s["n_units"])
        if "depends_on" in s:
            m["depends_on"] = int(s["depends_on"])
        if "dispatch_direct" in s:
            m["dispatch_direct"] = bool(s["dispatch_direct"])
        if stype == "efw_chp" and efw_overrides:
            m.update(efw_overrides)
        sources.append(m)
    economics, scale = scaled_economics(demand["peak_heat_kW"] / 1000.0)
    economics["counterfactual"] = "individual_gas"
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    return {
        "name": f"{arch_label} — {stack_label}",
        "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(cfg["buildings"])},
        "network": {"mode": "generic_length",
                    "length_m": float(route_m if route_m is not None else cfg["route_m"]),
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
        "sources": sources,
        "economics": economics,
        "description": f"Fixed CAPEX/OPEX scaled by {scale:.3f}x the heat-peak reference.",
    }


def row_from(result, **extra):
    h = result["headline"]
    inv = result["financial"]["investor"]
    grant = result.get("grant")
    grant_GBP = grant["grant_GBP"] if grant else 0.0
    return {
        **extra,
        "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Service gate": "PASS" if h["service_compliant"] else "FAIL",
        "Gross CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
        "GHNF grant (£m)": round(grant_GBP / 1e6, 2),
        "Net CAPEX (£m)": round((h["capex_total_GBP"] - grant_GBP) / 1e6, 2),
        "Linear density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
        "Required tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
        "NPV after GHNF (£m)": round(inv["npv_GBP"] / 1e6, 2),
        "Screening": result["screening"]["status"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Part A — the source frontier (all stacks x all archetypes)
# ═══════════════════════════════════════════════════════════════════════════

frontier_rows = []
frontier_results = {}
for arch_label in ARCHETYPES:
    for stack_label in STACKS:
        result = run_scenario(build_scenario(arch_label, stack_label))
        frontier_results[(arch_label, stack_label)] = result
        frontier_rows.append(row_from(result, Archetype=arch_label, Stack=stack_label))
frontier_df = pd.DataFrame(frontier_rows)
frontier_df.to_csv(OUT / "source_frontier.csv", index=False)
print("\n=== Source frontier (gas parity, GHNF 40% where carbon-eligible) ===")
print(frontier_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(11.5, 7))
dense = frontier_df[frontier_df["Archetype"] == "Dense (town centre)"]
for _, r in dense.iterrows():
    size = 60 + r["Net CAPEX (£m)"] * 14
    edge = INK if r["Service gate"] == "PASS" else C_RED
    ax.scatter(r["Carbon (gCO2e/kWh)"], r["NPV after GHNF (£m)"], s=size,
               color=STACK_COLOURS[r["Stack"]], edgecolor=edge, linewidth=1.4, zorder=4)
    ax.annotate(r["Stack"], (r["Carbon (gCO2e/kWh)"], r["NPV after GHNF (£m)"]),
                textcoords="offset points", xytext=(9, 5), fontsize=8.6, color=INK2)
ax.axvline(100, color=C_RED, lw=1.6, ls="--", zorder=2)
ax.text(101, ax.get_ylim()[0] + 0.6, "GHNF carbon boundary\n(100 gCO2e/kWh)",
        color=C_RED, fontsize=9, va="bottom")
ax.set_xlabel("Carbon intensity (gCO2e/kWh of delivered service)")
ax.set_ylabel("Investor NPV after GHNF (£m, 10.5% hurdle)")
ax.set_title("Source frontier — Dense archetype, gas-parity billing\n"
             "Bubble size = net CAPEX. Right of the red line, GHNF pays £0.",
             fontsize=12)
_save(fig, "SF1_source_frontier.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Part B — the density frontier (route sweep per archetype, EfW stack)
# ═══════════════════════════════════════════════════════════════════════════

ROUTE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
density_rows = []
for arch_label, cfg in ARCHETYPES.items():
    for mult in ROUTE_MULTIPLIERS:
        route_m = cfg["route_m"] * mult
        result = run_scenario(build_scenario(arch_label, "EfW + ASHP + gas peak",
                                             route_m=route_m))
        density_rows.append(row_from(result, Archetype=arch_label,
                                     Stack="EfW + ASHP + gas peak",
                                     **{"Route (m)": route_m, "Route multiplier": mult}))
density_df = pd.DataFrame(density_rows)
density_df.to_csv(OUT / "density_frontier.csv", index=False)
print("\n=== Density frontier (EfW + ASHP + gas peak) ===")
print(density_df.to_string(index=False))

ARCH_COLOURS = {"Dense (town centre)": C_BLUE, "Middle (suburban mixed)": C_AQUA,
                "Scarce (low-density edge)": C_YELLOW, "Ealing Phase 1 (real)": C_VIOLET}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for arch_label in ARCHETYPES:
    sub = density_df[density_df["Archetype"] == arch_label].sort_values(
        "Linear density (MWh/m/yr)")
    axes[0].plot(sub["Linear density (MWh/m/yr)"], sub["NPV after GHNF (£m)"],
                 "-o", color=ARCH_COLOURS[arch_label], lw=2, ms=5, label=arch_label)
    axes[1].plot(sub["Linear density (MWh/m/yr)"], sub["Required tariff (p/kWh)"],
                 "-o", color=ARCH_COLOURS[arch_label], lw=2, ms=5, label=arch_label)
axes[0].axhline(0, color=INK, lw=1.2)
axes[0].set_xlabel("Linear heat density (MWh/m/yr)")
axes[0].set_ylabel("Investor NPV after GHNF (£m)")
axes[0].set_xscale("log")
axes[0].set_title("NPV vs density — flattens long before zero", fontsize=11)
axes[0].legend(fontsize=8.5)
gas_cap = 7.33
axes[1].axhline(gas_cap, color=INK, lw=1.4, ls="--",
                label=f"Ofgem gas cap ({gas_cap}p)")
axes[1].set_xlabel("Linear heat density (MWh/m/yr)")
axes[1].set_ylabel("Required break-even tariff (p/kWh)")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_title("Required tariff vs density — never reaches the cap", fontsize=11)
axes[1].legend(fontsize=8.5)
fig.suptitle("Linear density frontier — EfW + ASHP + gas peak, GHNF 40%, gas parity",
             fontsize=12.5)
_save(fig, "SF2_density_frontier.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Part C — source break-even conditions (negotiation numbers)
#    Reference alternative: ASHP + gas peak on the same archetype. "Max price"
#    and "max distance" are where the EfW stack's NPV falls to the ASHP
#    stack's — beyond that, self-build ASHP is simply the better design.
# ═══════════════════════════════════════════════════════════════════════════

REF_ARCH = "Dense (town centre)"
ashp_npv = frontier_results[(REF_ARCH, "ASHP + gas peak")]["financial"]["investor"]["npv_GBP"] / 1e6

EFW_PRICES = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 120.0]
price_rows = []
for price in EFW_PRICES:
    result = run_scenario(build_scenario(
        REF_ARCH, "EfW + ASHP + gas peak",
        efw_overrides={"heat_export_cost_GBP_per_MWh": price}))
    price_rows.append(row_from(result, **{"EfW heat price (£/MWh)": price}))
price_df = pd.DataFrame(price_rows)
price_df.to_csv(OUT / "efw_price_breakeven.csv", index=False)

# Extra trunk to reach the EfW plant, modelled as additional route length —
# the same trench/pipe cost basis and real thermal losses as the network.
EFW_DISTANCES = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000]
dist_rows = []
for extra in EFW_DISTANCES:
    result = run_scenario(build_scenario(
        REF_ARCH, "EfW + ASHP + gas peak",
        route_m=ARCHETYPES[REF_ARCH]["route_m"] + extra))
    dist_rows.append(row_from(result, **{"Transmission leg (m)": extra}))
dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(OUT / "efw_distance_breakeven.csv", index=False)


def crossing(xs, ys, target):
    """First x where ys (monotone-ish) crosses target, linear interpolation."""
    for i in range(1, len(xs)):
        lo, hi = ys[i - 1], ys[i]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            return xs[i - 1] + (target - lo) * (xs[i] - xs[i - 1]) / (hi - lo)
    return None


max_price = crossing(list(price_df["EfW heat price (£/MWh)"]),
                     list(price_df["NPV after GHNF (£m)"]), ashp_npv)
max_dist = crossing(list(dist_df["Transmission leg (m)"]),
                    list(dist_df["NPV after GHNF (£m)"]), ashp_npv)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
axes[0].plot(price_df["EfW heat price (£/MWh)"], price_df["NPV after GHNF (£m)"],
             "-o", color=C_MAGENTA, lw=2, ms=5, label="EfW + ASHP + gas peak")
axes[0].axhline(ashp_npv, color=C_BLUE, lw=1.6, ls="--",
                label=f"ASHP + gas peak alternative (£{ashp_npv:.1f}m)")
if max_price is not None:
    axes[0].axvline(max_price, color=INK, lw=1.2, ls=":")
    axes[0].annotate(f"max worth paying\n≈ £{max_price:.0f}/MWh",
                     (max_price, ashp_npv), textcoords="offset points",
                     xytext=(8, 12), fontsize=9, color=INK)
else:
    # No crossing: dispatch self-protects. Above the ASHP's own marginal heat
    # cost (~24p/kWh / COP ~2.9 ≈ £83/MWh) purchased EfW heat is dearer than
    # making it, so that is the economic ceiling for any negotiated price.
    ashp_marginal = 240.0 / 2.88
    axes[0].axvline(ashp_marginal, color=INK, lw=1.2, ls=":")
    axes[0].annotate("economic ceiling ≈ ASHP\nmarginal heat cost "
                     f"(£{ashp_marginal:.0f}/MWh)",
                     (ashp_marginal, ashp_npv), textcoords="offset points",
                     xytext=(-150, 14), fontsize=9, color=INK)
axes[0].set_xlabel("EfW heat-export price (£/MWh)")
axes[0].set_ylabel("Investor NPV after GHNF (£m)")
axes[0].set_title("What is EfW heat worth?", fontsize=11)
axes[0].legend(fontsize=8.5)
axes[1].plot(dist_df["Transmission leg (m)"], dist_df["NPV after GHNF (£m)"],
             "-o", color=C_MAGENTA, lw=2, ms=5, label="EfW + ASHP + gas peak")
axes[1].axhline(ashp_npv, color=C_BLUE, lw=1.6, ls="--",
                label=f"ASHP + gas peak alternative (£{ashp_npv:.1f}m)")
if max_dist is not None:
    axes[1].axvline(max_dist, color=INK, lw=1.2, ls=":")
    axes[1].annotate(f"max transmission leg\n≈ {max_dist:,.0f} m",
                     (max_dist, ashp_npv), textcoords="offset points",
                     xytext=(8, 12), fontsize=9, color=INK)
axes[1].set_xlabel("Extra trunk to reach the EfW plant (m)")
axes[1].set_title("How far away can the EfW plant be?", fontsize=11)
axes[1].legend(fontsize=8.5)
fig.suptitle("Source break-even conditions — Dense archetype, vs the self-build ASHP alternative",
             fontsize=12.5)
_save(fig, "SF3_source_breakeven.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. findings.md
# ═══════════════════════════════════════════════════════════════════════════

compliant = frontier_df[frontier_df["Carbon gate"] == "PASS"]
best_by_arch = compliant.loc[compliant.groupby("Archetype")["NPV after GHNF (£m)"].idxmax()]

lines = [
    "# Source frontier, density frontier and source break-even conditions",
    "",
    "Generated by `python -m analysis.source_frontier`. Gas-parity billing; GHNF at the",
    "40% base rate, zeroed automatically by the engine where the 100 gCO2e/kWh gate",
    "fails. WSHP/GSHP are sized identically to the ASHP they replace, so their rows",
    "isolate the COP physics of a stable water/ground source, not a sizing choice.",
    "",
    "## Best carbon-compliant stack by archetype (NPV after GHNF)",
    "",
    best_by_arch[["Archetype", "Stack", "Carbon (gCO2e/kWh)",
                  "NPV after GHNF (£m)", "Required tariff (p/kWh)"]].to_markdown(index=False),
    "",
    "## Negotiation numbers (Dense archetype, vs self-build ASHP + gas peak)",
    "",
    (f"- Maximum EfW heat-export price worth paying: **≈ £{max_price:.0f}/MWh** "
     f"(default assumption £8/MWh)." if max_price is not None else
     "- The EfW stack stays ahead of self-build ASHP across the whole £0-120/MWh sweep — "
     "dispatch self-protects by shifting baseload back to the stack's own ASHP once EfW "
     "heat costs more than ASHP marginal cost. The practical negotiation ceiling is that "
     "marginal heat cost, **≈ £83/MWh** (24p/kWh at COP ~2.9); every £10/MWh on the "
     "EfW price costs the owner **≈ £1.1m of NPV**, so cheap heat is still the point."),
    f"- Maximum EfW transmission leg: "
    f"**{'≈ %.0f m' % max_dist if max_dist is not None else 'not crossed in the swept range'}** "
    "of extra trunk before self-build ASHP wins.",
    "- Both numbers compare stack against stack, not against zero NPV — no stack reaches",
    "  zero. They answer 'is this source worth connecting to', not 'does this scheme fund'.",
    "",
    "## Full frontier",
    "",
    frontier_df.to_markdown(index=False),
    "",
    "## Density sweep (EfW + ASHP + gas peak)",
    "",
    density_df.to_markdown(index=False),
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nMax EfW price ≈ {max_price}, max distance ≈ {max_dist}")
print(f"Wrote {OUT}/findings.md and 3 figures.")
