"""Five ways Dalkia could sit on the same scheme, and what each one risks.

    python -m analysis.dalkia_roles

Extends analysis/contractor_view.py (same scheme, same declared scope split and
margins) from four *positions* to the five *commercial roles* Dalkia could
actually take, and prices the civils-overrun downside for each:

  R1  Designer only                 — fee on design scope, PI exposure, no capital
  R2  M&E/controls contractor       — margin on Dalkia's build scope, no civils
  R3  M&E + long-term O&M           — R2 plus the 40-year service-fee stream
  R4  Whole-scheme D&B incl civils  — margin on EVERYTHING, wears the trench
  R5  Owner/operator (equity)       — pays CAPEX, collects tariff, holds residual

The civils test is the one that matters: the DESNZ Birmingham report prices
routes 9-51% above this model's own SEAI-fitted cost curve, so overruns in that
band are the OBSERVED range, not a stress case. R2/R3 are structurally immune
(civils is someone else's contract); R4's margin is a thin cushion against a
big number; R5 absorbs whatever is left.

No loss probabilities are invented — the deliverable is the break-even overrun
per role and the packaged-vs-separate procurement comparison, both of which are
arithmetic on declared assumptions, not distributions we do not have.

Writes CSVs, PNGs and findings.md to output/dalkia_roles/.
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
OUT = ROOT / "output" / "dalkia_roles"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.contractor_view import (
    BUILD_YEARS, CONSTRUCTION_MARGIN, DALKIA_WACC, DESIGN_MARGIN, OM_MARGIN,
    _capex_items, _dense_scheme, dalkia_position, split_capex,
)
from economics.cashflow import discount_factors
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. The scheme — the same itemised Dense town-centre archetype used by
#    contractor_view's scope split (HP counterfactual).
# ═══════════════════════════════════════════════════════════════════════════

res = run_scenario(_dense_scheme())
split = split_capex(res)
pos = dalkia_position(res, split)
owner_npv = res["financial"]["investor"]["npv_GBP"]
life = int(res["financial"]["investor"]["life_years"])
dfs = discount_factors(life, DALKIA_WACC)
om_annuity = float(sum(dfs[1:]))

items = _capex_items(res)
total_capex = sum(items.values())
civils_capex = float(items.get("network_GBP", 0.0)) + float(items.get("energy_centre_building_GBP", 0.0))
design_turnover = pos["design_turnover_GBP"]
me_turnover = pos["construction_turnover_GBP"]
om_margin_yr = pos["om_margin_GBP_per_yr"]
om_margin_npv = om_margin_yr * om_annuity
# Whole-scheme D&B: everything buildable (all CAPEX except land), one main
# contract, construction margin on all of it.
db_turnover = total_capex - float(items.get("land_and_enabling_GBP", 0.0))

ROLES = {
    "R1 Designer only": {
        "turnover_GBP": design_turnover,
        "margin_GBP": design_turnover * DESIGN_MARGIN,
        "om_margin_npv_GBP": 0.0,
        "civils_exposed_GBP": 0.0,
        "exposure": "design PI, to ~12 yr latent defects",
    },
    "R2 M&E/controls contractor": {
        "turnover_GBP": me_turnover + design_turnover,
        "margin_GBP": me_turnover * CONSTRUCTION_MARGIN + design_turnover * DESIGN_MARGIN,
        "om_margin_npv_GBP": 0.0,
        "civils_exposed_GBP": 0.0,
        "exposure": f"{BUILD_YEARS} yr build + commissioning",
    },
    "R3 M&E + long-term O&M": {
        "turnover_GBP": me_turnover + design_turnover,
        "margin_GBP": me_turnover * CONSTRUCTION_MARGIN + design_turnover * DESIGN_MARGIN,
        "om_margin_npv_GBP": om_margin_npv,
        "civils_exposed_GBP": 0.0,
        "exposure": f"{BUILD_YEARS} yr build + {life} yr availability/performance",
    },
    "R4 Whole-scheme D&B (incl civils)": {
        "turnover_GBP": db_turnover,
        "margin_GBP": db_turnover * CONSTRUCTION_MARGIN,
        "om_margin_npv_GBP": 0.0,
        "civils_exposed_GBP": civils_capex,
        "exposure": f"{BUILD_YEARS} yr build incl trench + structure risk",
    },
    "R5 Owner/operator (equity)": {
        "turnover_GBP": 0.0,
        "margin_GBP": 0.0,
        "om_margin_npv_GBP": 0.0,
        "civils_exposed_GBP": civils_capex,
        "exposure": f"{life} yr residual — the whole business case",
    },
}

OVERRUNS = [0.0, 0.10, 0.20, 0.30, 0.51]

rows = []
for role, r in ROLES.items():
    base_npv = (r["margin_GBP"] + r["om_margin_npv_GBP"]
                if role != "R5 Owner/operator (equity)" else owner_npv)
    row = {
        "Role": role,
        "Turnover (£m)": round(r["turnover_GBP"] / 1e6, 2),
        "Base NPV (£m)": round(base_npv / 1e6, 2),
        "Civils exposure (£m)": round(r["civils_exposed_GBP"] / 1e6, 2),
        "Exposure": r["exposure"],
    }
    for o in OVERRUNS[1:]:
        hit = r["civils_exposed_GBP"] * o
        row[f"NPV at {o:.0%} civils overrun (£m)"] = round((base_npv - hit) / 1e6, 2)
    breakeven = (r["margin_GBP"] + r["om_margin_npv_GBP"]) / r["civils_exposed_GBP"] \
        if r["civils_exposed_GBP"] and role != "R5 Owner/operator (equity)" else None
    row["Break-even overrun (%)"] = round(breakeven * 100, 1) if breakeven is not None else None
    rows.append(row)

roles_df = pd.DataFrame(rows)
roles_df.to_csv(OUT / "five_roles.csv", index=False)
print("=== Five roles ===")
print(roles_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Figure DR1 — role NPV under civils overrun
# ═══════════════════════════════════════════════════════════════════════════

ROLE_COLOURS = dict(zip(ROLES, [C_YELLOW, C_BLUE, C_AQUA, C_VIOLET, C_RED]))
xs = [o * 100 for o in OVERRUNS]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
for role, r in ROLES.items():
    base = (r["margin_GBP"] + r["om_margin_npv_GBP"]
            if role != "R5 Owner/operator (equity)" else owner_npv)
    ys = [(base - r["civils_exposed_GBP"] * o) / 1e6 for o in OVERRUNS]
    target = axes[1] if role == "R5 Owner/operator (equity)" else axes[0]
    target.plot(xs, ys, "-o", color=ROLE_COLOURS[role], lw=2, ms=5, label=role)
axes[0].axhline(0, color=INK, lw=1.2)
axes[0].axvspan(9, 51, color=GRID, alpha=0.45, zorder=1)
axes[0].text(30, axes[0].get_ylim()[1] * 0.92, "observed overrun range\n(DESNZ Birmingham vs model curve)",
             ha="center", fontsize=8.5, color=MUTED)
axes[0].set_xlabel("Civils cost overrun (%)")
axes[0].set_ylabel("Dalkia NPV (£m, 8% WACC)")
axes[0].set_title("Contractor roles: only the civils-carrying contract can lose", fontsize=11)
axes[0].legend(fontsize=8.5)
axes[1].axvspan(9, 51, color=GRID, alpha=0.45, zorder=1)
axes[1].axhline(0, color=INK, lw=1.2)
axes[1].set_xlabel("Civils cost overrun (%)")
axes[1].set_ylabel("Owner NPV (£m, 10.5% hurdle)")
axes[1].set_title("The owner: starts deep underwater, overrun digs deeper", fontsize=11)
axes[1].legend(fontsize=8.5)
fig.suptitle("Who loses when the trench costs more — Dense town-centre archetype, HP counterfactual",
             fontsize=12.5)
_save(fig, "DR1_roles_under_overrun.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Packaged vs separate civils procurement — the owner's side of the choice
#    Packaged: Dalkia D&B prices civils with a risk premium (extra contingency
#    c on civils); the owner pays the premium ALWAYS, overrun capped at the
#    contract price. Separate: no premium; the owner pays the actual overrun.
# ═══════════════════════════════════════════════════════════════════════════

premiums = [0.05, 0.10, 0.15, 0.20]
proc_rows = []
for o in OVERRUNS:
    row = {"Actual civils overrun (%)": o * 100,
           "Separate: owner pays (£m)": round(civils_capex * o / 1e6, 2)}
    for c in premiums:
        row[f"Packaged @ +{c:.0%} premium (£m)"] = round(civils_capex * c / 1e6, 2)
    proc_rows.append(row)
proc_df = pd.DataFrame(proc_rows)
proc_df.to_csv(OUT / "packaged_vs_separate.csv", index=False)

fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.plot([o * 100 for o in OVERRUNS], [civils_capex * o / 1e6 for o in OVERRUNS],
        "-o", color=C_BLUE, lw=2.2, ms=5, label="separate civils package — owner pays actual overrun")
prem_colours = [C_AQUA, C_YELLOW, C_VIOLET, C_RED]
for c, colour in zip(premiums, prem_colours):
    ax.axhline(civils_capex * c / 1e6, color=colour, lw=1.8, ls="--",
               label=f"packaged D&B, +{c:.0%} civils risk premium (paid regardless)")
ax.axvspan(9, 51, color=GRID, alpha=0.45, zorder=1)
ax.set_xlabel("Actual civils overrun (%)")
ax.set_ylabel("Owner's extra cost vs base civils price (£m)")
ax.set_title("Packaged vs separate civils — the premium only pays off above its own size\n"
             f"Civils base £{civils_capex/1e6:.2f}m. Shaded band: observed overrun range.",
             fontsize=11.5)
ax.legend(fontsize=8.5)
_save(fig, "DR2_packaged_vs_separate.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. findings.md
# ═══════════════════════════════════════════════════════════════════════════

r4 = ROLES["R4 Whole-scheme D&B (incl civils)"]
r4_breakeven = r4["margin_GBP"] / r4["civils_exposed_GBP"] * 100
r3_npv = (ROLES["R3 M&E + long-term O&M"]["margin_GBP"]
          + ROLES["R3 M&E + long-term O&M"]["om_margin_npv_GBP"]) / 1e6

lines = [
    "# Five roles, one scheme: where Dalkia can sit safely",
    "",
    "Generated by `python -m analysis.dalkia_roles`. Same scheme, scope split and",
    "declared margins as `analysis/contractor_view.py` (see its findings for the",
    "honesty notes — every margin is an assumption to be replaced with real terms).",
    "",
    "## The five roles",
    "",
    roles_df.to_markdown(index=False),
    "",
    "## What it says",
    "",
    f"- **R2/R3 cannot lose money to a trench.** Civils sits in someone else's",
    f"  contract; Dalkia's downside is its own M&E performance. R3 is worth",
    f"  ~£{r3_npv:.2f}m NPV on declared margins.",
    f"- **R4's cushion is {r4_breakeven:.1f}% of civils.** The observed overrun range",
    "  on real schemes is 9-51%, so the whole-scheme margin is wiped out INSIDE the",
    "  observed range. Whole-scheme D&B including civils is a bet on ground",
    "  conditions Dalkia has not surveyed, paid at an M&E margin.",
    "- **R5 is not a contracting decision, it is an investment decision** — and every",
    "  scheme in this pack fails the 10.5% investor test before any overrun.",
    "- **Packaged civils only protects the owner above the premium's own size**: a",
    "  +10% priced premium costs the owner more than self-insuring any overrun",
    "  below 10%, and real premiums for unsurveyed urban trenching sit well above",
    "  the bottom of the observed band.",
    "",
    "## Recommendation shape",
    "",
    "Design + M&E + controls + long-term O&M (R3), with civils owner-procured as a",
    "separate package, unless survey-quality route information exists AND the",
    "civils risk is explicitly priced. That matches the contractor-view finding",
    "that Dalkia's margin survives schemes the owner cannot fund.",
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 2 figures.")
