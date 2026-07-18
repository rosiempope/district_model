"""How much does domestic take-up risk move the owner's NPV?

    python -m analysis.connection_risk

The DESNZ zoning position mandates (or can mandate) connection for existing
COMMUNALLY-heated and qualifying NON-DOMESTIC buildings, but treats existing
INDIVIDUALLY-heated homes differently — their take-up is a genuine commercial
risk, not a policy certainty. GHNF guidance treats that risk as material and
expects heads of terms (and, before construction funding, binding supply
agreements) from key customers.

This study sweeps the RESIDENTIAL connection probability across the
downside / central / upside range in analysis/archetypes.py
(RESIDENTIAL_CONNECTION_SCENARIOS) while holding anchors at their base
probability, on all three archetypes, and reports owner NPV. It turns the
review comment "your residential probabilities are optimistic" into a number:
how much of the owner's position is riding on domestic take-up the owner does
not control.

Stack: EfW + ASHP + gas peak (the pack's carbon-compliant winner), auto-sized
per case; GHNF 40% where carbon-eligible; gas-parity billing.

Writes CSVs, a PNG and findings.md to output/connection_risk/.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "connection_risk"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.archetypes import ARCHETYPES, RESIDENTIAL_CONNECTION_SCENARIOS
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

RESIDENTIAL_TYPES = {"residential", "residential_existing"}
LEVELS = ["downside", "central", "upside"]
PRESETS = {"ashp": "ealing_phase1", "gas_boiler": "ealing_phase1",
           "efw_chp": "newlincs_style"}

weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(weather) == 8760
weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")


def buildings_at(arch_label, residential_prob):
    """Archetype buildings with every RESIDENTIAL block's connect probability
    set to `residential_prob`; anchors keep their own base probability."""
    buildings = deepcopy(ARCHETYPES[arch_label]["buildings"])
    for b in buildings:
        if b["type"] in RESIDENTIAL_TYPES:
            b["connection_probability"] = residential_prob
    return buildings


rows = []
for arch_label, cfg in ARCHETYPES.items():
    scen = RESIDENTIAL_CONNECTION_SCENARIOS[arch_label]
    for level in LEVELS:
        prob = scen[level]
        buildings = buildings_at(arch_label, prob)
        demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
        rec = recommend_sizing(
            demand_kW=demand["total_heat_kW"],
            peak_demand_kW=demand["peak_heat_kW"],
            technology_types=["efw_chp", "ashp", "gas_boiler"],
            weather_df=weather, network_flow_temp_C=70.0,
            n_buildings=len(buildings),
            building_types=[b["type"] for b in buildings],
        )
        sources = [{"type": s["type"], "preset": PRESETS[s["type"]],
                    "name": f"{s['type']} ({s['role']})",
                    "capacity_MW": float(s["capacity_MW"]),
                    **({"n_units": int(s["n_units"])} if "n_units" in s else {})}
                   for s in rec["sources"]]
        economics, _ = scaled_economics(demand["peak_heat_kW"] / 1000.0)
        economics["counterfactual"] = "individual_gas"
        economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
        result = run_scenario({
            "name": f"{arch_label} — residential take-up {level} ({prob:.0%})",
            "climate_scenario": "baseline",
            "demand": {"buildings": buildings},
            "network": {"mode": "generic_length", "length_m": float(cfg["route_m"]),
                        "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
            "sources": sources,
            "economics": economics,
        })
        inv = result["financial"]["investor"]
        rows.append({
            "Archetype": arch_label,
            "Take-up level": level,
            "Residential connect prob (%)": round(prob * 100, 0),
            "Annual heat (GWh)": round(
                (demand["annual_heat_MWh"] + demand["annual_dhw_MWh"]) / 1000, 2),
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
            "Required tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
            "Screening": result["screening"]["status"],
        })
        print(f"{arch_label} | {level} ({prob:.0%}): NPV £{inv['npv_GBP']/1e6:.2f}m")

df = pd.DataFrame(rows)
df.to_csv(OUT / "connection_risk.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Figure — NPV range across the take-up band, per archetype
# ═══════════════════════════════════════════════════════════════════════════

ARCH_COLOURS = {"Dense (town centre)": C_BLUE, "Middle (suburban mixed)": C_AQUA,
                "Scarce (low-density edge)": C_YELLOW}
fig, ax = plt.subplots(figsize=(10.5, 5.8))
for arch_label in ARCHETYPES:
    sub = df[df["Archetype"] == arch_label].set_index("Take-up level").loc[LEVELS]
    xs = sub["Residential connect prob (%)"]
    ax.plot(xs, sub["Investor NPV (£m)"], "-o", color=ARCH_COLOURS[arch_label],
            lw=2, ms=6, label=arch_label)
    # Mark the central point.
    cen = sub.loc["central"]
    ax.scatter(cen["Residential connect prob (%)"], cen["Investor NPV (£m)"],
               s=130, facecolor="white", edgecolor=ARCH_COLOURS[arch_label],
               linewidth=2.2, zorder=5)
ax.axhline(0, color=INK, lw=1.2)
ax.set_xlabel("Residential connection probability (%) — downside · central · upside")
ax.set_ylabel("Owner NPV after GHNF (£m, 10.5% hurdle)")
ax.set_title("Domestic take-up risk — owner NPV across the connection-probability band\n"
             "Anchors held at base; hollow marker = central case. EfW + ASHP + gas peak, GHNF 40%",
             fontsize=11.5)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "CR1_connection_risk.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# findings.md
# ═══════════════════════════════════════════════════════════════════════════

lines = ["# Connection (take-up) risk — how much rides on domestic customers", "",
         "Generated by `python -m analysis.connection_risk`. Residential connect "
         "probability swept across the downside/central/upside band from "
         "`analysis/archetypes.py`; anchors held at base; EfW + ASHP + gas peak, "
         "GHNF 40%, gas parity.", ""]
for arch_label in ARCHETYPES:
    sub = df[df["Archetype"] == arch_label].set_index("Take-up level")
    span = sub.loc["upside", "Investor NPV (£m)"] - sub.loc["downside", "Investor NPV (£m)"]
    scen = RESIDENTIAL_CONNECTION_SCENARIOS[arch_label]
    lines.append(
        f"- **{arch_label}**: owner NPV swings **£{span:.1f}m** across the take-up "
        f"band ({scen['downside']:.0%}→{scen['upside']:.0%}), from "
        f"£{sub.loc['downside','Investor NPV (£m)']}m (downside) to "
        f"£{sub.loc['upside','Investor NPV (£m)']}m (upside); central "
        f"£{sub.loc['central','Investor NPV (£m)']}m.")
lines += [
    "",
    "## Reading it",
    "",
    "- Every case stays negative across the whole band — take-up risk changes the",
    "  size of the loss, not the sign. Consistent with the rest of the pack: the",
    "  binding constraint is fixed cost per connection, not revenue volume.",
    "- The swing is widest where the scheme leans hardest on dispersed individual",
    "  homes (the ones DESNZ zoning does NOT mandate), which is exactly where the",
    "  owner has least control — reinforcing the anchor-led recommendation.",
    "- Central residential probabilities are 0.85 (dense communal), 0.60 (suburban",
    "  individual), 0.45/0.40 (dispersed individual). These are modelling ranges,",
    "  not published take-up statistics; anchors sit at 0.95 (1.00 only once",
    "  contracted, per GHNF heads-of-terms evidence expectations).",
    "",
    "## Full sweep",
    "",
    df.to_markdown(index=False),
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 1 figure.")
