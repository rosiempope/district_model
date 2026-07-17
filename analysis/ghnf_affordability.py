"""GHNF-supported affordability frontier: required tariff vs affordable tariff.

    python -m analysis.ghnf_affordability

The single most commercial question in the pack: is there ANY customer
proposition — priced at what the customer would otherwise pay — that covers the
owner's costs once the maximum realistic GHNF capital grant is taken? Three
things are put on one axis for every case:

  1. the AFFORDABLE tariff — what the customer's own alternative costs them
     (their modelled individual-gas bill, or their modelled individual-heat-pump
     bill with the BUS grant netted off, per economics/metrics.py);
  2. the REQUIRED tariff — the p/kWh that gives the owner exactly zero NPV at
     the 10.5% hurdle, net of the modelled GHNF grant;
  3. the GRANT DEPENDENCY — the same required tariff at 0%, 40% and the
     just-under-50% GHNF ceiling, plus the grant rate that would be needed for
     the affordable tariff to clear the hurdle (almost always impossible,
     which is the finding).

GHNF treatment follows the agent brief for the Dalkia pack: the grant is
enabled wherever the model's own pre-checks pass (carbon <= 100 gCO2e/kWh; the
strictly-below-50% intensity cap and the 4.5p/kWh x 15yr output cap are both
enforced in economics/grant.py), the headline uses the 40% base assumption, a
49.9999% ceiling case is shown alongside, and a shadow no-grant column makes
the grant dependency explicit. 49% is never treated as guaranteed.

Writes CSVs, PNGs and findings.md to output/ghnf_affordability/.
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
OUT = ROOT / "output" / "ghnf_affordability"
OUT.mkdir(parents=True, exist_ok=True)

from economics.cashflow import discount_factors
from optimisation.auto_size import recommend_sizing
from profiles.demand_synthesis import synthesise_network
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
# 1. Cases — the three density archetypes from analysis/dalkia_screening_study
#    (same building mixes and illustrative route lengths, so this study's
#    tariff findings line up row-for-row with the screening matrix), and the
#    two carbon-compliant stacks GHNF could actually fund.
# ═══════════════════════════════════════════════════════════════════════════

PRESET_FOR_TYPE = {
    "ashp": "ealing_phase1",
    "gas_boiler": "ealing_phase1",
    "efw_chp": "newlincs_style",
}

STACKS = {
    "ASHP + gas peak": ["ashp", "gas_boiler"],
    "EfW + ASHP + gas peak": ["efw_chp", "ashp", "gas_boiler"],
}

PARITIES = {
    "gas parity": "individual_gas",
    "heat-pump parity (BUS netted)": "individual_ashp",
}

# 0% is the shadow no-grant case; 40% is the headline base assumption; the
# ceiling is strictly below 50% (economics/grant.py clamps at 0.499999 anyway).
GRANT_RATES = [0.0, 0.40, 0.4999]

weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(weather) == 8760
weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")

demand_cache = {
    label: synthesise_network(weather, {"demand_nodes": deepcopy(cfg["buildings"])})
    for label, cfg in ARCHETYPES.items()
}


def build_scenario(arch_label, stack_label, counterfactual, grant_rate):
    cfg = ARCHETYPES[arch_label]
    demand = demand_cache[arch_label]
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"],
        peak_demand_kW=demand["peak_heat_kW"],
        technology_types=STACKS[stack_label],
        weather_df=weather,
        network_flow_temp_C=70.0,
        n_buildings=len(cfg["buildings"]),
        building_types=[b["type"] for b in cfg["buildings"]],
    )
    sources = []
    for s in rec["sources"]:
        m = {"type": s["type"], "preset": PRESET_FOR_TYPE[s["type"]],
             "name": f"{s['type']} ({s['role']})", "capacity_MW": float(s["capacity_MW"])}
        if "n_units" in s:
            m["n_units"] = int(s["n_units"])
        sources.append(m)
    economics, scale = scaled_economics(demand["peak_heat_kW"] / 1000.0)
    economics["counterfactual"] = counterfactual
    if grant_rate > 0:
        economics["ghnf_grant"] = {"enabled": True, "rate": grant_rate}
    return {
        "name": f"{arch_label} — {stack_label} — {counterfactual} — GHNF {grant_rate:.0%}",
        "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(cfg["buildings"])},
        "network": {"mode": "generic_length", "length_m": float(cfg["route_m"]),
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
        "sources": sources,
        "economics": economics,
        "description": f"Fixed CAPEX/OPEX scaled by {scale:.3f}x the heat-peak reference.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. The sweep
# ═══════════════════════════════════════════════════════════════════════════

rows = []
results = {}
for arch_label in ARCHETYPES:
    for stack_label in STACKS:
        for parity_label, counterfactual in PARITIES.items():
            for grant_rate in GRANT_RATES:
                scenario = build_scenario(arch_label, stack_label, counterfactual, grant_rate)
                result = run_scenario(scenario)
                results[(arch_label, stack_label, parity_label, grant_rate)] = result
                h = result["headline"]
                inv = result["financial"]["investor"]
                grant = result.get("grant")
                grant_GBP = grant["grant_GBP"] if grant else 0.0
                eligible = grant["eligible_capex_GBP"] if grant else 0.0
                output_cap = grant.get("output_based_cap_GBP") if grant else None
                # Which cap binds: the % of eligible CAPEX, or 4.5p/kWh x 15yr?
                if grant is None or grant_GBP <= 0:
                    binding = "no grant" if grant_rate == 0 else "carbon-ineligible"
                elif output_cap is not None and grant_GBP >= output_cap - 0.5:
                    binding = "output cap (4.5p/kWh x 15yr)"
                else:
                    binding = "percentage cap"
                affordable = inv["equivalent_year1_heat_tariff_p_per_kWh"]
                required = inv["required_heat_tariff_p_per_kWh_for_zero_NPV"]
                rows.append({
                    "Archetype": arch_label,
                    "Stack": stack_label,
                    "Customer proposition": parity_label,
                    "GHNF rate (%)": round(grant_rate * 100, 2),
                    "Gross CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
                    "GHNF-eligible CAPEX (£m)": round(eligible / 1e6, 2),
                    "Grant (£m)": round(grant_GBP / 1e6, 2),
                    "Binding cap": binding,
                    "Net CAPEX (£m)": round((h["capex_total_GBP"] - grant_GBP) / 1e6, 2),
                    "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                    "Affordable tariff (p/kWh)": affordable,
                    "Required tariff (p/kWh)": required,
                    "Tariff gap (p/kWh)": (round(required - affordable, 2)
                                           if None not in (required, affordable) else None),
                    "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
                    "Screening": result["screening"]["status"],
                })

df = pd.DataFrame(rows)
df.to_csv(OUT / "affordability_frontier.csv", index=False)
print("\n=== GHNF affordability frontier ===")
print(df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 3. Required grant rate for zero NPV — the honest "what would it take" line.
#    The GHNF grant is a year-0 inflow, so NPV is linear in the grant with a
#    coefficient of exactly 1: required_grant = grant_in_run - NPV.
# ═══════════════════════════════════════════════════════════════════════════

dependency_rows = []
for arch_label in ARCHETYPES:
    for stack_label in STACKS:
        for parity_label in PARITIES:
            r0 = results[(arch_label, stack_label, parity_label, 0.0)]
            r40 = results[(arch_label, stack_label, parity_label, 0.40)]
            npv0 = r0["financial"]["investor"]["npv_GBP"]
            grant40 = r40["grant"]["grant_GBP"] if r40.get("grant") else 0.0
            eligible = (r40["grant"]["eligible_capex_GBP"]
                        if r40.get("grant") and r40["grant"]["eligible_capex_GBP"] else None)
            required_grant = -npv0
            required_rate = (required_grant / eligible * 100) if eligible else None
            dependency_rows.append({
                "Archetype": arch_label,
                "Stack": stack_label,
                "Customer proposition": parity_label,
                "NPV with no grant (£m)": round(npv0 / 1e6, 2),
                "40% grant (£m)": round(grant40 / 1e6, 2),
                "Grant needed for zero NPV (£m)": round(required_grant / 1e6, 2),
                "Implied grant rate (% of eligible CAPEX)": (
                    round(required_rate, 0) if required_rate is not None else None),
                "Within the GHNF <50% cap?": (
                    "yes" if required_rate is not None and required_rate < 50 else "no"),
            })
dep_df = pd.DataFrame(dependency_rows)
dep_df.to_csv(OUT / "grant_dependency.csv", index=False)
print("\n=== Grant dependency: what grant rate would zero NPV actually take? ===")
print(dep_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Figure AF1 — required vs affordable tariff, by archetype
# ═══════════════════════════════════════════════════════════════════════════

GRANT_COLOURS = {0.0: C_YELLOW, 0.40: C_BLUE, 0.4999: C_AQUA}
GRANT_LABELS = {0.0: "no grant", 0.40: "GHNF 40% (base)", 0.4999: "GHNF ~50% (ceiling)"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
for ax, arch_label in zip(axes, ARCHETYPES):
    ypos, ylabels = [], []
    y = 0
    for stack_label in STACKS:
        for grant_rate in GRANT_RATES:
            sub = df[(df["Archetype"] == arch_label) & (df["Stack"] == stack_label)
                     & (df["Customer proposition"] == "gas parity")
                     & (df["GHNF rate (%)"] == round(grant_rate * 100, 2))]
            req = sub["Required tariff (p/kWh)"].iloc[0]
            ax.barh(y, req, height=0.62, color=GRANT_COLOURS[grant_rate], zorder=3)
            ypos.append(y)
            ylabels.append(f"{stack_label}\n{GRANT_LABELS[grant_rate]}" if grant_rate == 0.40
                           else GRANT_LABELS[grant_rate])
            y += 1
        y += 0.6
    gas_aff = df[(df["Archetype"] == arch_label)
                 & (df["Customer proposition"] == "gas parity")]["Affordable tariff (p/kWh)"].iloc[0]
    hp_aff = df[(df["Archetype"] == arch_label)
                & (df["Customer proposition"] == "heat-pump parity (BUS netted)")][
                    "Affordable tariff (p/kWh)"].iloc[0]
    ax.axvline(gas_aff, color=INK, lw=1.6, ls="--", zorder=4,
               label=f"affordable at gas parity ({gas_aff:.1f}p)")
    ax.axvline(hp_aff, color=C_VIOLET, lw=1.6, ls=":", zorder=4,
               label=f"affordable at HP parity ({hp_aff:.1f}p)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_title(arch_label, fontsize=11)
    ax.set_xlabel("Tariff (p/kWh)")
    ax.invert_yaxis()
fig.suptitle("Required break-even tariff (bars) vs what customers can afford (lines)\n"
             "Every bar to the right of its lines is an unfundable customer proposition",
             fontsize=12.5)
_save(fig, "AF1_required_vs_affordable.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Figure AF2 — the affordability-gap waterfall (Middle x EfW x GHNF 40%)
# ═══════════════════════════════════════════════════════════════════════════


def waterfall_components(result):
    """PV decomposition of the investor NPV from the annual cash-flow table."""
    inv = result["financial"]["investor"]
    table = inv["annual_table"]
    life = len(table) - 1
    rate = 0.105
    f = discount_factors(life, rate)
    pv = lambda key: float(sum(row[key] * f[i] for i, row in enumerate(table)))
    comps = {
        "Gross CAPEX": -pv("capex_GBP"),
        "GHNF grant": pv("grant_GBP"),
        "Customer revenue (PV)": pv("revenue_GBP"),
        "OPEX (PV)": -pv("opex_GBP"),
        "REPEX (PV)": -pv("repex_GBP"),
    }
    npv = inv["npv_GBP"]
    resid = npv - sum(comps.values())
    assert abs(resid) < 1000, f"waterfall does not reconcile: residual £{resid:,.0f}"
    return comps, npv


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, parity_label in zip(axes, PARITIES):
    result = results[("Middle (suburban mixed)", "EfW + ASHP + gas peak", parity_label, 0.40)]
    comps, npv = waterfall_components(result)
    labels = list(comps.keys()) + ["Residual NPV gap"]
    values = list(comps.values()) + [npv]
    cum = 0.0
    for i, (lab, val) in enumerate(zip(labels, values)):
        if lab == "Residual NPV gap":
            ax.bar(i, val / 1e6, bottom=0, color=C_RED if val < 0 else C_GREEN, zorder=3)
            ax.text(i, val / 2e6, f"£{val/1e6:,.1f}m", ha="center", va="center",
                    fontsize=9, color="#ffffff", fontweight="bold")
        else:
            colour = C_AQUA if val >= 0 else C_BLUE
            if lab == "GHNF grant":
                colour = C_GREEN
            ax.bar(i, val / 1e6, bottom=cum / 1e6, color=colour, zorder=3)
            ax.text(i, (cum + val / 2.0) / 1e6, f"{val/1e6:+,.1f}", ha="center",
                    va="center", fontsize=9, color="#ffffff", fontweight="bold")
            cum += val
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_title(parity_label, fontsize=11)
axes[0].set_ylabel("£m (PV at 10.5%)")
fig.suptitle("Where the owner's money goes — Middle archetype, EfW + ASHP + gas peak, GHNF 40%",
             fontsize=12.5)
_save(fig, "AF2_affordability_waterfall.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Figure AF3 — grant dependency (NPV vs grant rate)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
STACK_COLOURS = {"ASHP + gas peak": C_BLUE, "EfW + ASHP + gas peak": C_AQUA}
ARCH_STYLES = {"Dense (town centre)": "-", "Middle (suburban mixed)": "--",
               "Scarce (low-density edge)": ":"}
for ax, parity_label in zip(axes, PARITIES):
    for stack_label in STACKS:
        for arch_label in ARCHETYPES:
            xs = [r * 100 for r in GRANT_RATES]
            ys = [results[(arch_label, stack_label, parity_label, r)]
                  ["financial"]["investor"]["npv_GBP"] / 1e6 for r in GRANT_RATES]
            ax.plot(xs, ys, ARCH_STYLES[arch_label], color=STACK_COLOURS[stack_label],
                    lw=2, marker="o", ms=5,
                    label=f"{stack_label} — {arch_label.split(' (')[0]}")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlabel("GHNF grant rate (% of eligible CAPEX)")
    ax.set_title(parity_label, fontsize=11)
axes[0].set_ylabel("Investor NPV (£m, 10.5% hurdle)")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Grant dependency: even the ~50% GHNF ceiling does not lift owner NPV to zero",
             fontsize=12.5)
_save(fig, "AF3_grant_dependency.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7. findings.md
# ═══════════════════════════════════════════════════════════════════════════

base40 = df[(df["GHNF rate (%)"] == 40.0)]
best = base40.loc[base40["Investor NPV (£m)"].idxmax()]
worst_gap = base40.loc[base40["Tariff gap (p/kWh)"].idxmax()]
hp_rows = base40[base40["Customer proposition"] == "heat-pump parity (BUS netted)"]
gas_rows = base40[base40["Customer proposition"] == "gas parity"]

lines = [
    "# GHNF affordability frontier — required vs affordable tariff",
    "",
    "Generated by `python -m analysis.ghnf_affordability`. GHNF is applied wherever the",
    "model's own pre-checks pass (carbon <= 100 gCO2e/kWh; <50% intensity cap and the",
    "4.5p/kWh x 15yr output cap enforced in `economics/grant.py`). 40% is the headline",
    "base assumption; the ~50% ceiling is shown but never treated as guaranteed; a",
    "shadow no-grant column makes grant dependency explicit.",
    "",
    "## Headline",
    "",
    f"- Best case at GHNF 40%: **{best['Archetype']} / {best['Stack']} / "
    f"{best['Customer proposition']}** — NPV **£{best['Investor NPV (£m)']}m**, "
    f"required tariff {best['Required tariff (p/kWh)']}p vs affordable "
    f"{best['Affordable tariff (p/kWh)']}p.",
    f"- Widest tariff gap at 40%: {worst_gap['Archetype']} / {worst_gap['Stack']} "
    f"({worst_gap['Customer proposition']}) — gap {worst_gap['Tariff gap (p/kWh)']}p/kWh.",
    "- Heat-pump parity is the stronger customer proposition everywhere: the customer's",
    "  alternative is more expensive (even after BUS), so the affordable tariff is higher —",
    f"  mean affordable tariff {hp_rows['Affordable tariff (p/kWh)'].mean():.1f}p vs "
    f"{gas_rows['Affordable tariff (p/kWh)'].mean():.1f}p under gas parity.",
    "",
    "## Grant dependency",
    "",
    dep_df.to_markdown(index=False),
    "",
    "## Full frontier",
    "",
    df.to_markdown(index=False),
    "",
    "## Reading notes",
    "",
    "- The affordable tariff is the model's equivalent year-1 parity tariff: the",
    "  customer's own modelled alternative bill divided by their delivered kWh. It is a",
    "  whole-bill figure (fuel + standing charge + plant lifecycle), not a unit-rate cap.",
    "- The required tariff is net of the modelled grant (the engine subtracts the grant",
    "  series before dividing by discounted delivered heat).",
    "- The heat-pump-parity affordable tariff nets the £7,500 BUS grant off eligible",
    "  installations (45 kWth per-installation cap; per-building `bus_eligible: false`",
    "  override available for social housing, which BUS excludes).",
    "- Where the binding-cap column says 'output cap', the 4.5p/kWh x 15yr limit bit",
    "  before the percentage did — bigger grant rates cannot help those schemes.",
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 3 figures.")
