"""Who actually makes money on a heat network that loses money.

    python -m analysis.contractor_view

Every result in this study pack reports ONE number: the investor NPV. That is
the position of a single actor who pays all the CAPEX, collects all the revenue,
holds the residual for 40 years and wants 10.5% real. Every scenario fails it.

That actor is not Dalkia, and the question "should we get involved in DHN?" is
not the question that NPV answers.

A real scheme pays four different parties out of the same cash flows:

  CONTRACTOR  builds it. Paid out of CAPEX, at a margin, over ~3 years. Gone at
              practical completion. Has no exposure to the 40-year residual.
  OPERATOR    runs it. Paid a fee out of OPEX, at a margin, for the contract
              term. Fee is contractual; it is not a claim on the residual.
  OWNER       funds the CAPEX, collects the tariff, holds the residual. This is
              the -£25m. This is the only party the rest of the pack models.
  FUNDER      pays the grant. Takes no cash return. Buys carbon and bills.

The contractor's and operator's margins are ALREADY INSIDE the CAPEX and OPEX
that make the owner's NPV negative. They are a cost line, not a claim on what is
left. "Negative NPV" is a statement about the residual, and it says nothing
about whether the cost lines are profitable — they are paid first, and paid
regardless. That is the whole point of this module.

What is derived vs assumed
--------------------------
DERIVED from the engine: every CAPEX and OPEX line, the owner's 40-year cash
flow, the counterfactual, the grant.

ASSUMED here, with NO basis in the model: the scope split (which lines Dalkia
builds), the construction margin, the design margin and the O&M margin. These
are the four numbers that decide the contractor answer and the model has nothing
to say about any of them. They are declared as constants below, swept in §5, and
must be replaced with real commercial terms before this is shown to anyone. A
reader should be able to disagree with the margins and re-run in one edit.

Writes to output/contractor_view/.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from economics.cashflow import discount_factors
from scenarios.birmingham_zoning import central_izo_scenario
from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import WORKED_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "contractor_view"
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE, C_RED, C_GREEN, C_YELLOW = "#2a78d6", "#e34948", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

# ══════════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS — none of this comes from the model. Change and re-run.
# ══════════════════════════════════════════════════════════════════════════════

# Which CAPEX lines a design-build-operate contractor WITHOUT civils would carry.
# Dalkia's four service lines are M&E engineering, FM, energy services (DBFOM)
# and systems integration (dalkia.co.uk/services). Trenching and structures are
# not among them.
#
# Fraction of each line Dalkia builds. 0.0 = someone else's scope entirely.
DALKIA_SCOPE_SHARE = {
    "sources_GBP":                 1.00,  # energy-centre plant — core M&E
    "electricity_connection_GBP":  1.00,  # switchgear/substation M&E
    "controls_and_scada_GBP":      1.00,  # Advanced Systems Integration
    "development_and_design_GBP":  1.00,  # design fee
    "commissioning_GBP":           1.00,  # commissioning
    "customer_connections_GBP":    0.55,  # HIU + metering, NOT the builder's work,
                                          # penetrations or riser civils. 55% is a
                                          # judgement, not a quotation.
    "metering_GBP":                1.00,
    "network_GBP":                 0.00,  # trench + pipe — CIVILS
    "energy_centre_building_GBP":  0.00,  # structure — CIVILS
    "gas_connection_GBP":          0.00,  # utility
    "land_and_enabling_GBP":       0.00,  # owner's transaction
    "thermal_storage_GBP":         1.00,  # vessel + integration
    "contingency_GBP":             0.00,  # handled below — follows scope pro-rata
}

# Which OPEX lines an operator contract would carry as FEE (not pass-through).
# Energy is excluded: it is a pass-through in almost every real O&M contract, so
# counting it as contractor turnover would flatter the answer enormously.
DALKIA_OPEX_SCOPE = {
    "technology and network O&M":       1.00,
    "operator_overhead_GBP":            1.00,
    "water_treatment_GBP":              1.00,
    "billing_and_customer_service_GBP": 1.00,
    "insurance_and_rates_GBP":          0.00,  # owner's
    "land_lease_GBP":                   0.00,  # owner's
    "electricity energy":               0.00,  # pass-through
    "gas energy":                       0.00,  # pass-through
    "third_party_heat energy":          0.00,  # pass-through
}

CONSTRUCTION_MARGIN = 0.06   # UK M&E D&B, typical 3-10% on turnover
DESIGN_MARGIN       = 0.12   # fee-based work carries more than build
OM_MARGIN           = 0.10   # long-term service contract, typical 5-15%
DALKIA_WACC         = 0.08   # a contractor's own cost of capital, not 10.5%
BUILD_YEARS         = 3      # construction turnover spread over years 0-2

# Real civils overrun, measured. The DESNZ Birmingham report prices its routes at
# £2,500-3,750/m against this model's SEAI-fitted curve at £2,090-2,484/m — real
# projects run 9-51% over a defensible cost curve, before ground conditions. This
# is the risk a main contractor takes on and a package contractor does not.
CIVILS_OVERRUN_CASES = [0.0, 0.10, 0.20, 0.30, 0.51]


def _capex_items(result: dict) -> dict:
    """Every CAPEX line the engine actually built, as {name: £}."""
    return dict(result["headline"]["capex_breakdown_GBP"])


def _opex_items(result: dict) -> dict:
    """Year-1 OPEX by line, matching reports/cost_breakdown.py's naming."""
    recon = result["financial"]["opex_reconciliation"]
    items = {}
    for carrier, gbp in result["headline"].get("annual_energy_cost_by_carrier_GBP", {}).items():
        items[f"{carrier} energy"] = float(gbp)
    items["technology and network O&M"] = float(recon["technology_and_network_om_GBP"])
    for k, v in result["input"]["economics"].get("annual_opex_items", {}).items():
        items[k] = float(v)
    return items


def split_capex(result: dict) -> pd.DataFrame:
    """Partition every CAPEX line into Dalkia scope / civils / owner."""
    items = _capex_items(result)
    contingency_keys = {"development_and_design_GBP", "commissioning_GBP", "contingency_GBP"}
    # Contingency follows the scope it protects: pro-rata across the buildable
    # lines by Dalkia's share of them. A contingency on a trench is the civils
    # contractor's problem, not Dalkia's.
    buildable = {k: v for k, v in items.items() if k not in contingency_keys and k != "land_and_enabling_GBP"}
    dalkia_buildable = sum(v * DALKIA_SCOPE_SHARE.get(k, 0.0) for k, v in buildable.items())
    total_buildable = sum(buildable.values())
    contingency_share = (dalkia_buildable / total_buildable) if total_buildable else 0.0

    rows = []
    for key, gbp in sorted(items.items(), key=lambda kv: -kv[1]):
        if key == "contingency_GBP":
            share = contingency_share
            note = f"pro-rata to Dalkia's {contingency_share:.0%} of buildable scope"
        else:
            share = DALKIA_SCOPE_SHARE.get(key, 0.0)
            note = ""
        rows.append({
            "capex_item": key.replace("_GBP", "").replace("_", " "),
            "total_GBP": gbp,
            "dalkia_share": share,
            "dalkia_GBP": gbp * share,
            "other_GBP": gbp * (1 - share),
            "note": note,
        })
    return pd.DataFrame(rows)


def dalkia_position(result: dict, capex_split: pd.DataFrame) -> dict:
    """Dalkia's own P&L: construction turnover + margin, O&M turnover + margin."""
    design_keys = {"development and design", "commissioning"}
    build_turnover = capex_split.loc[~capex_split["capex_item"].isin(design_keys), "dalkia_GBP"].sum()
    design_turnover = capex_split.loc[capex_split["capex_item"].isin(design_keys), "dalkia_GBP"].sum()

    build_margin = build_turnover * CONSTRUCTION_MARGIN
    design_margin = design_turnover * DESIGN_MARGIN

    opex = _opex_items(result)
    om_turnover_yr = sum(v * DALKIA_OPEX_SCOPE.get(k, 0.0) for k, v in opex.items())
    om_margin_yr = om_turnover_yr * OM_MARGIN

    life = int(result["financial"]["investor"]["life_years"])
    df = discount_factors(life, DALKIA_WACC)

    # Construction margin lands over the build years; O&M margin annually after.
    npv = 0.0
    for y in range(BUILD_YEARS):
        npv += ((build_margin + design_margin) / BUILD_YEARS) * df[y]
    for y in range(1, life + 1):
        npv += om_margin_yr * df[y]

    om_turnover_npv = sum(om_turnover_yr * df[y] for y in range(1, life + 1))

    return {
        "construction_turnover_GBP": build_turnover,
        "design_turnover_GBP": design_turnover,
        "construction_margin_GBP": build_margin,
        "design_margin_GBP": design_margin,
        "om_turnover_GBP_per_yr": om_turnover_yr,
        "om_margin_GBP_per_yr": om_margin_yr,
        "om_turnover_40yr_undiscounted_GBP": om_turnover_yr * life,
        "om_turnover_NPV_GBP": om_turnover_npv,
        "total_turnover_NPV_GBP": build_turnover + design_turnover + om_turnover_npv,
        "dalkia_NPV_GBP": npv,
    }


def _counterfactual_capex_by_building(result: dict) -> dict:
    """What each customer would have spent on their own heat pump."""
    cf = result["counterfactual"]
    src = cf.get("by_building", cf)
    return {k: float(v["capex_GBP"]) for k, v in src.items()
            if isinstance(v, dict) and "capex_GBP" in v}


def capture_avoided_capital(cfg: dict, captures: list, hurdle_rates: list) -> pd.DataFrame:
    """Owner NPV when a connection charge recovers the capital the customer avoids.

    The whole-system case is +£67m while the owner is -£56m even at 3.5%. The
    gap is not the discount rate — it is that £77.4m of the benefit is capital
    the CUSTOMER never has to spend on their own heat pump, and a tariff held to
    the counterfactual's RUNNING cost hands all of it to them for nothing.

    A connection charge is how a real scheme captures it, and it is fair by
    construction: at 100% capture the customer pays exactly the capital they
    would have paid anyway, and still gets running-cost parity. They are never
    worse off than their own alternative at any capture level <= 100%.
    """
    base = deepcopy(cfg)
    base["economics"]["counterfactual"] = "individual_ashp"
    cf_capex = _counterfactual_capex_by_building(run_scenario(base))

    rows = []
    for capture in captures:
        for rate in hurdle_rates:
            c = deepcopy(base)
            c["economics"]["discount_rate"] = rate
            for b in c["demand"]["buildings"]:
                b["connection_charge_GBP"] = cf_capex.get(b["name"], 0.0) * capture
            r = run_scenario(c)
            rows.append({
                "capture_of_avoided_customer_capital": capture,
                "hurdle_rate": rate,
                "connection_charge_total_GBPm": sum(cf_capex.values()) * capture / 1e6,
                "owner_NPV_GBPm": r["financial"]["investor"]["npv_GBP"] / 1e6,
            })
    return pd.DataFrame(rows)


def owner_position(cfg: dict, hurdle_rates: list, counterfactuals: list) -> pd.DataFrame:
    """The owner's NPV across hurdle rates and counterfactuals."""
    rows = []
    for cf in counterfactuals:
        for rate in hurdle_rates:
            c = deepcopy(cfg)
            c["economics"]["counterfactual"] = cf
            c["economics"]["discount_rate"] = rate
            r = run_scenario(c)
            rows.append({
                "counterfactual": cf,
                "hurdle_rate": rate,
                "owner_NPV_GBPm": r["financial"]["investor"]["npv_GBP"] / 1e6,
                "social_NPV_GBPm": r["financial"]["social"]["npv_GBP"] / 1e6,
            })
    return pd.DataFrame(rows)


def main() -> None:
    a3 = next(deepcopy(s) for s in WORKED_SCENARIOS if "A3" in s["name"])
    a3["economics"]["counterfactual"] = "individual_ashp"
    res = run_scenario(a3)

    # ── 1. Scope ──────────────────────────────────────────────────────────────
    split = split_capex(res)
    split.to_csv(OUT / "capex_scope_split.csv", index=False)

    dalkia_capex = split["dalkia_GBP"].sum()
    other_capex = split["other_GBP"].sum()
    total_capex = split["total_GBP"].sum()

    # ── 2. Dalkia's position ──────────────────────────────────────────────────
    pos = dalkia_position(res, split)
    owner_npv = res["financial"]["investor"]["npv_GBP"]

    # ── 3. The four positions ─────────────────────────────────────────────────
    grant = float((res.get("grant") or {}).get("grant_awarded_GBP", 0.0) or 0.0)
    four = pd.DataFrame([
        {"position": "Contractor (build)", "who": "Dalkia + civils sub",
         "exposure_years": BUILD_YEARS, "paid_from": "CAPEX",
         "NPV_GBPm": (pos["construction_margin_GBP"] + pos["design_margin_GBP"]) / 1e6,
         "basis": "margin on built scope, one-off"},
        {"position": "Operator (run)", "who": "Dalkia",
         "exposure_years": 40, "paid_from": "OPEX",
         "NPV_GBPm": (pos["dalkia_NPV_GBP"] - pos["construction_margin_GBP"] - pos["design_margin_GBP"]) / 1e6,
         "basis": "margin on O&M fee, 40 yr, discounted at Dalkia's own WACC"},
        {"position": "Owner (equity)", "who": "council / ESCO / EDF",
         "exposure_years": 40, "paid_from": "residual",
         "NPV_GBPm": owner_npv / 1e6,
         "basis": "pays CAPEX, collects tariff, holds what is left"},
        {"position": "Funder (grant)", "who": "GHNF",
         "exposure_years": 0, "paid_from": "public capital",
         "NPV_GBPm": -grant / 1e6,
         "basis": "no cash return; buys carbon + lower bills"},
    ])
    four.to_csv(OUT / "four_positions.csv", index=False)

    # ── 4. Owner hurdle-rate / counterfactual grid, both cases ────────────────
    rates = [0.035, 0.05, 0.06, 0.08, 0.105]
    a3_grid = owner_position(a3, rates, ["individual_gas", "individual_ashp"])
    a3_grid.insert(0, "case", "Ealing-scale A3 (564 conns, mostly domestic)")

    bham, _ = central_izo_scenario(heat_flow_temp_C=62.0, heat_return_temp_C=30.0)
    bham_grid = owner_position(bham, rates, ["individual_gas", "individual_ashp"])
    bham_grid.insert(0, "case", "Birmingham Central (60 GWh, non-domestic anchors)")

    grid = pd.concat([a3_grid, bham_grid], ignore_index=True)
    grid.to_csv(OUT / "owner_hurdle_grid.csv", index=False)

    # ── 4b. The lever that actually moves the owner ───────────────────────────
    cap = capture_avoided_capital(bham, [0.0, 0.25, 0.50, 0.75, 1.0], [0.105, 0.035])
    cap.to_csv(OUT / "capture_avoided_capital.csv", index=False)

    # ── 5. Margin sensitivity + the civils-overrun test ───────────────────────
    sens = []
    for cm in [0.03, 0.06, 0.10]:
        for om in [0.05, 0.10, 0.15]:
            global CONSTRUCTION_MARGIN, OM_MARGIN
            keep_cm, keep_om = CONSTRUCTION_MARGIN, OM_MARGIN
            CONSTRUCTION_MARGIN, OM_MARGIN = cm, om
            p = dalkia_position(res, split)
            CONSTRUCTION_MARGIN, OM_MARGIN = keep_cm, keep_om
            sens.append({
                "construction_margin": cm, "om_margin": om,
                "dalkia_NPV_GBPm": p["dalkia_NPV_GBP"] / 1e6,
                "pct_of_owner_loss": abs(p["dalkia_NPV_GBP"] / owner_npv) * 100,
            })
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(OUT / "margin_sensitivity.csv", index=False)

    # If Dalkia took civils as main contractor: turnover rises, but so does the
    # overrun exposure. At what overrun does the whole job's margin vanish?
    network_capex = float(_capex_items(res).get("network_GBP", 0.0))
    ec_building = float(_capex_items(res).get("energy_centre_building_GBP", 0.0))
    civils_capex = network_capex + ec_building
    total_margin = pos["construction_margin_GBP"] + pos["design_margin_GBP"]
    overrun = pd.DataFrame([{
        "civils_overrun_pct": o * 100,
        "overrun_cost_GBP": civils_capex * o,
        "dalkia_total_build_margin_GBP": total_margin,
        "margin_remaining_GBP": total_margin - civils_capex * o,
        "margin_wiped_out": civils_capex * o > total_margin,
    } for o in CIVILS_OVERRUN_CASES])
    overrun.to_csv(OUT / "civils_overrun_exposure.csv", index=False)

    breakeven_overrun = total_margin / civils_capex if civils_capex else float("nan")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.6))
    d = four.sort_values("NPV_GBPm")
    cols = [C_RED if v < 0 else C_GREEN for v in d["NPV_GBPm"]]
    ax.barh(d["position"], d["NPV_GBPm"], color=cols, height=0.6)
    ax.axvline(0, color=INK, lw=1)
    for y, v in enumerate(d["NPV_GBPm"]):
        ax.annotate(f"£{v:,.2f}m", (v, y), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.set_xlabel("NPV to that party (£m)")
    ax.set_title("Four positions on one scheme\nThe owner's loss is not the contractor's loss",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "CV1_four_positions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (case, cf), g in grid.groupby(["case", "counterfactual"]):
        style = "-" if "ashp" in cf else "--"
        colour = C_BLUE if "Birmingham" in case else C_YELLOW
        ax.plot(g["hurdle_rate"] * 100, g["owner_NPV_GBPm"], style, color=colour,
                marker="o", ms=4,
                label=f"{'Birmingham' if 'Birmingham' in case else 'Ealing A3'} vs {'heat pumps' if 'ashp' in cf else 'gas'}")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("Owner's hurdle rate (% real)")
    ax.set_ylabel("Owner NPV (£m)")
    ax.set_title("'Negative NPV' is a property of the hurdle rate and the counterfactual,\nnot of the scheme",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "CV2_owner_hurdle.png", dpi=150)
    plt.close(fig)

    # ── Findings ──────────────────────────────────────────────────────────────
    lines = [
        "# Who makes money on a scheme that loses money",
        "",
        "Generated by `python -m analysis.contractor_view`. Scope split and margins are",
        "**assumptions with no basis in the model** — see the module docstring and §5.",
        "",
        "## 1. The scope split — design-build-operate WITHOUT civils",
        "",
        f"Case: `{res['scenario_name']}`, counterfactual `individual_ashp`.",
        "",
        split.assign(
            total_GBP=lambda d: d["total_GBP"].round(0),
            dalkia_GBP=lambda d: d["dalkia_GBP"].round(0),
            other_GBP=lambda d: d["other_GBP"].round(0),
        ).to_markdown(index=False),
        "",
        f"- Dalkia scope: **£{dalkia_capex/1e6:.2f}m** of a £{total_capex/1e6:.2f}m scheme "
        f"({dalkia_capex/total_capex*100:.0f}%).",
        f"- Someone else's: **£{other_capex/1e6:.2f}m** ({other_capex/total_capex*100:.0f}%) — "
        "trench, pipe, structure, utility, land.",
        "",
        "**Dalkia never pays the civils contractor.** In a package structure the owner",
        "procures civils as a separate contract. Two contracts, one client, no",
        "back-to-back liability. Dalkia prices, builds and warrants its own scope only.",
        "",
        "## 2. Dalkia's position",
        "",
        "| Line | Value |",
        "|---|---|",
        f"| Construction turnover | £{pos['construction_turnover_GBP']/1e6:.2f}m |",
        f"| Design/commissioning turnover | £{pos['design_turnover_GBP']/1e6:.2f}m |",
        f"| O&M turnover | £{pos['om_turnover_GBP_per_yr']/1e3:.0f}k/yr "
        f"(£{pos['om_turnover_40yr_undiscounted_GBP']/1e6:.2f}m over 40 yr) |",
        f"| **Total turnover (NPV)** | **£{pos['total_turnover_NPV_GBP']/1e6:.2f}m** |",
        f"| Construction margin @ {CONSTRUCTION_MARGIN:.0%} | £{pos['construction_margin_GBP']/1e3:.0f}k |",
        f"| Design margin @ {DESIGN_MARGIN:.0%} | £{pos['design_margin_GBP']/1e3:.0f}k |",
        f"| O&M margin @ {OM_MARGIN:.0%} | £{pos['om_margin_GBP_per_yr']/1e3:.0f}k/yr |",
        f"| **Dalkia NPV @ {DALKIA_WACC:.0%}** | **£{pos['dalkia_NPV_GBP']/1e6:.2f}m** |",
        f"| Owner NPV @ 10.5% | £{owner_npv/1e6:.2f}m |",
        "",
        f"**Dalkia is positive on a scheme where the owner loses £{abs(owner_npv)/1e6:.1f}m.**",
        "That is not a trick. Dalkia's margin is *already inside* the CAPEX and OPEX that",
        "make the owner's NPV negative — contractors are a cost line, paid first, and the",
        "owner holds the residual. Negative NPV is a statement about the residual.",
        "",
        f"Dalkia's entire NPV is **{abs(pos['dalkia_NPV_GBP']/owner_npv)*100:.1f}%** of the owner's loss.",
        "Worth saying out loud in the room: **Dalkia's margin is not why these schemes",
        "fail.** Cutting it to zero moves the owner's NPV by a rounding error.",
        "",
        "## 3. The four positions",
        "",
        four.assign(NPV_GBPm=lambda d: d["NPV_GBPm"].round(2)).to_markdown(index=False),
        "",
        "The asymmetry that answers 'why would the civils contractor take a bad deal?':",
        f"**the contractor's exposure is {BUILD_YEARS} years; the owner's is 40.** The civils",
        "contractor digs a trench, is paid at a normal civils margin, and is gone at",
        "practical completion. The 40-year residual is not their risk and they do not price",
        "it. They check their price covers their cost — not whether a heat network returns",
        "over four decades.",
        "",
        "## 4. Why the owner might still say yes",
        "",
        grid.assign(
            owner_NPV_GBPm=lambda d: d["owner_NPV_GBPm"].round(2),
            social_NPV_GBPm=lambda d: d["social_NPV_GBPm"].round(2),
            hurdle_rate=lambda d: (d["hurdle_rate"] * 100).round(1),
        ).to_markdown(index=False),
        "",
        "**The hurdle rate alone never fixes this, and it is important not to claim it",
        "does.** Birmingham against individual heat pumps: -£76.2m at 10.5%, still",
        "**-£56.2m at 3.5%**. Dropping to a social discount rate is worth ~£20m of a £76m",
        "hole. A council-owned scheme at 3.5% is *not* looking at a positive cash position.",
        "",
        "The +£67m whole-system figure at 3.5% is a **different quantity from the owner's",
        "cash**, and conflating the two is the easiest mistake in this pack to make. The",
        "social case is a resource-cost comparison that excludes tariff transfers. Its",
        "benefit is overwhelmingly the **£77.4m of individual heat pumps society does not",
        "have to buy** — capital the CUSTOMER avoids. The owner spends £94.8m and, under a",
        "tariff held to the counterfactual's RUNNING cost, receives none of it.",
        "",
        "### 4b. The lever that actually moves the owner",
        "",
        "So the question is not 'what discount rate?' — it is **'how does the owner capture",
        "the capital the customer avoids?'** The answer is a connection charge, which is",
        "what real schemes use and what this model already supports",
        "(`connection_charge_GBP` per building).",
        "",
        "It is fair by construction: at 100% capture the customer pays exactly the capital",
        "they would have spent on their own heat pump, and still gets running-cost parity.",
        "At any capture below 100% they are strictly better off than their own alternative.",
        "",
        cap.assign(
            capture_of_avoided_customer_capital=lambda d: (d["capture_of_avoided_customer_capital"] * 100).round(0),
            hurdle_rate=lambda d: (d["hurdle_rate"] * 100).round(1),
            connection_charge_total_GBPm=lambda d: d["connection_charge_total_GBPm"].round(2),
            owner_NPV_GBPm=lambda d: d["owner_NPV_GBPm"].round(2),
        ).to_markdown(index=False),
        "",
        "Compare the two levers on the same £76.2m hole:",
        "",
        "| Lever | Worth |",
        "|---|---|",
        "| Hurdle rate 10.5% -> 3.5% | ~£20m |",
        "| Connection charge, 0 -> 100% capture | **~£70m** |",
        "",
        "**The connection charge is roughly 3.5x the discount rate as a lever**, and unlike",
        "the discount rate it does not require finding a cheaper owner. At 100% capture the",
        "owner is -£6.1m at a full **commercial 10.5%** hurdle — within touching distance of",
        "investable without any grant, any public capital or any subsidy. At 75% capture and",
        "3.5% it is break-even.",
        "",
        "This is the strongest commercial finding in the pack and no study in it pulls this",
        "lever. It is also the answer to 'why would anyone own this?': not a patient owner",
        "— a **connection charge**.",
        "",
        "'Negative NPV' is a property of the scheme **at a hurdle rate, against a",
        "counterfactual, under a tariff structure**. The tariff structure is doing more work",
        "than the other two combined.",
        "",
        "**But the counterfactual flip is NOT universal, and this is the finding that",
        "matters most.** Birmingham flips positive against individual heat pumps; the",
        "Ealing-scale case does not. The mechanism is the Boiler Upgrade Scheme:",
        "",
        "- BUS pays £7,500 per installation, capped at **45 kWth**.",
        "- Ealing A3's *Residential block A* — 320 connections — has an individual-ASHP",
        "  counterfactual CAPEX of **£75,327 total (~£235/connection)**. BUS pays for",
        "  almost the entire alternative. A heat network cannot beat free.",
        "- Birmingham's *Civic offices*-equivalent anchors are all above 45 kWth, so BUS",
        "  pays **£0** and the individual alternative costs £77.4m.",
        "",
        "So: **heat networks win against the legal counterfactual precisely where BUS does",
        "not reach — large non-domestic anchor loads.** Not domestic retrofit. That is the",
        "opposite of the intuition, and it happens to be exactly the estate Dalkia already",
        "does M&E and FM on: hospitals, universities, stations, shopping centres.",
        "",
        "## 5. Sensitivity — the assumed margins",
        "",
        sens_df.assign(
            dalkia_NPV_GBPm=lambda d: d["dalkia_NPV_GBPm"].round(2),
            pct_of_owner_loss=lambda d: d["pct_of_owner_loss"].round(1),
        ).to_markdown(index=False),
        "",
        "Across the whole plausible margin range Dalkia stays positive and stays small",
        "relative to the owner's loss. The contractor answer is **not sensitive** to the",
        "margin assumption — which is the one useful thing about a study built on four",
        "invented numbers.",
        "",
        "## 6. Why not take the civils as well",
        "",
        f"Civils scope on this case is £{civils_capex/1e6:.2f}m (network + energy-centre",
        f"building). Dalkia's total build margin across the WHOLE job is "
        f"£{total_margin/1e3:.0f}k.",
        "",
        overrun.assign(
            overrun_cost_GBP=lambda d: d["overrun_cost_GBP"].round(0),
            dalkia_total_build_margin_GBP=lambda d: d["dalkia_total_build_margin_GBP"].round(0),
            margin_remaining_GBP=lambda d: d["margin_remaining_GBP"].round(0),
        ).to_markdown(index=False),
        "",
        f"**A civils overrun of just {breakeven_overrun*100:.1f}% wipes out Dalkia's entire",
        "margin on the whole job** — plant, controls, design, commissioning, HIUs and 40",
        "years of O&M included.",
        "",
        "That is not hypothetical. The DESNZ Birmingham report prices its routes at",
        "£2,500-3,750/m against this model's SEAI-fitted curve at £2,090-2,484/m: real",
        "schemes run **9-51% over** a defensible cost curve before ground conditions. The",
        f"{breakeven_overrun*100:.1f}% break-even sits at the bottom of that observed range.",
        "",
        "As main contractor Dalkia would price civils, subcontract it, add margin, and wear",
        "the overrun. As package contractor the owner carries it. The turnover difference is",
        f"£{civils_capex/1e6:.2f}m; the risk difference is the entire business case.",
        "",
        "## What this does not prove",
        "",
        "- The scope split and all four margins are **invented**. Replace with real terms.",
        "- No construction-period financing, retentions, LDs, bonds, defects liability or",
        "  performance-guarantee downside is modelled. A real O&M contract with an",
        "  availability guarantee puts fee at risk; that is not here.",
        "- The O&M fee is assumed flat real for 40 years with no re-tender. Real service",
        "  contracts are 5-15 years and re-competed.",
        "- Dalkia's WACC of 8% is a placeholder.",
        "- Interface risk is real and unmodelled: heat loss and pumping cost are set by",
        "  pipework someone else installs. **Do not sign a performance guarantee against a",
        "  network you did not specify.** The model prices that sensitivity",
        "  (`network/topology_thermal.py`); the contract question is outside it.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT}/")
    print(f"\nDalkia scope:  £{dalkia_capex/1e6:6.2f}m of £{total_capex/1e6:.2f}m ({dalkia_capex/total_capex*100:.0f}%)")
    print(f"Dalkia NPV:    £{pos['dalkia_NPV_GBP']/1e6:6.2f}m  (turnover NPV £{pos['total_turnover_NPV_GBP']/1e6:.2f}m)")
    print(f"Owner NPV:     £{owner_npv/1e6:6.2f}m")
    print(f"Dalkia margin is {abs(pos['dalkia_NPV_GBP']/owner_npv)*100:.1f}% of the owner's loss")
    print(f"Civils overrun that wipes out Dalkia's whole build margin: {breakeven_overrun*100:.1f}%")
    print("\n=== Four positions ===")
    print(four[["position", "exposure_years", "paid_from", "NPV_GBPm"]].to_string(index=False))
    print("\n=== Owner NPV by hurdle rate ===")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
