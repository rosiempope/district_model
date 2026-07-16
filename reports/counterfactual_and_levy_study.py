"""Two questions the gas-boiler counterfactual cannot answer.

    python -m reports.counterfactual_and_levy_study

Runs against the live engine (scenarios.scenario_runner.run_scenario).

1. WHAT IS THE ALTERNATIVE?
   Every result in this study pack compares district heat against an individual
   GAS BOILER. That is the cheap alternative policy is removing. Heat network
   zoning designates zones where heat networks are "the lowest-cost solution for
   DECARBONISING heating" (Energy Act 2023 framework, per the DESNZ Birmingham
   report) — i.e. against the alternative that is actually legal long-term,
   which is a heat pump in every building.

   The model had counterfactual_individual_ashp_dispatch() written but not
   selectable, so it could not run the comparison the whole policy rests on.

2. WHAT IF ELECTRICITY GOT CHEAPER?
   Policy costs are ~18% of an electricity bill against ~8% of a gas bill, and
   Nesta estimated 82% of domestic levy revenue came from electricity under the
   Oct-Dec 2024 cap. That loads the levies onto the technology policy is trying
   to encourage. Government has begun unwinding it (75% of the Renewables
   Obligation moved to public funding in April 2026; ECO off bills from 1 July
   2026). This asks what a further shift would do.

   The answer is NOT the obvious one, and that is the point of running it.

Writes to output/counterfactual_and_levy/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from economics.CAPEX import BUS_MAX_CAPACITY_KWTH, bus_grant_GBP
from economics.tariffs import (
    OFGEM_ELECTRICITY_CAP_P_PER_KWH,
    OFGEM_GAS_CAP_P_PER_KWH,
    OFGEM_SPARK_GAP_RATIO,
    rebalanced_caps,
)
from scenarios.birmingham_zoning import central_izo_scenario
from scenarios.scenario_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "counterfactual_and_levy"

C_BLUE, C_RED, C_GREEN, C_YELLOW = "#2a78d6", "#e34948", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _base():
    s, _ = central_izo_scenario(heat_flow_temp_C=62.0, heat_return_temp_C=30.0)
    return s


def counterfactual_comparison() -> pd.DataFrame:
    rows = []
    for cf, label in (
        ("individual_gas", "Individual gas boilers (being phased out)"),
        ("individual_ashp", "Individual heat pumps (the legal alternative)"),
    ):
        s = copy.deepcopy(_base())
        s["economics"]["counterfactual"] = cf
        r = run_scenario(s)
        inv, fin, c = r["financial"]["investor"], r["financial"], r["counterfactual"]
        rows.append({
            "Counterfactual": label,
            "Alternative CAPEX (£m)": round(c["total_capex_GBP"] / 1e6, 1),
            "Alternative bill (£m/yr)": round(c["total_annual_opex_GBP"] / 1e6, 2),
            "District CAPEX (£m)": round(r["headline"]["capex_total_GBP"] / 1e6, 1),
            "District OPEX (£m/yr)": round(r["headline"]["annual_total_opex_GBP"] / 1e6, 2),
            "Incremental CAPEX (£m)": round(fin["incremental_capex_GBP"] / 1e6, 1),
            "Avoided cost (£m/yr)": round(fin["annual_avoided_cost_GBP"] / 1e6, 2),
            "Fair tariff at parity (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 1),
            "Whole-system NPV @3.5% (£m)": round(fin["npv_vs_counterfactual_GBP"] / 1e6, 1),
            "Whole-system payback (yrs)": (
                round(fin["simple_payback_years"], 1) if fin["simple_payback_years"] else None
            ),
        })
    return pd.DataFrame(rows)


def bus_eligibility_table() -> pd.DataFrame:
    """Where the £7,500 grant actually lands. The 45 kWth cap is the whole story."""
    cases = [
        ("Birmingham New Street Station", 15000.0, 1),
        ("Bullring Shopping Centre West", 7000.0, 1),
        ("Colmore Plaza (office)", 900.0, 1),
        ("Residential block, 50 flats", 400.0, 50),
        ("Residential block, 200 flats", 1600.0, 200),
        ("Single house", 8.0, 1),
    ]
    rows = []
    for name, peak, conns in cases:
        grant = bus_grant_GBP(peak, conns)
        rows.append({
            "Building": name,
            "Peak (kW)": peak,
            "Connections": conns,
            "kW per installation": round(peak / conns, 1),
            f"Within BUS {BUS_MAX_CAPACITY_KWTH:.0f} kWth cap": peak / conns <= BUS_MAX_CAPACITY_KWTH,
            "BUS grant (£)": round(grant, 0),
        })
    return pd.DataFrame(rows)


def levy_sensitivity() -> pd.DataFrame:
    """Move policy costs off electricity, and see who benefits.

    Both sides move: the district scheme's own heat pumps get cheaper to run,
    AND the individual-heat-pump alternative it is compared against gets cheaper
    to run. Whether district heat gains or loses depends on which side is MORE
    electrified — which, for a scheme leaning on a gas peak, is not the scheme.
    """
    rows = []
    for shift in (0.0, 0.25, 0.5, 0.75, 1.0):
        caps = rebalanced_caps(shift)
        s = copy.deepcopy(_base())
        s["economics"]["counterfactual"] = "individual_ashp"
        # BOTH sides must move, or the sensitivity reports half the effect and
        # flatters district heat. The scheme buys electricity commercially rather
        # than at the cap, but the same policy costs sit in both, so the same
        # proportional reduction is applied to each.
        elec_scale = caps["electricity_p_per_kWh"] / OFGEM_ELECTRICITY_CAP_P_PER_KWH
        for src in s["sources"]:
            if src["type"] in {"ashp", "wshp", "gshp"}:
                src["electricity_price_GBP_per_MWh"] = 240.0 * elec_scale
        # ...and the individual-heat-pump alternative it is compared against.
        s["economics"]["counterfactual_electricity_price_p_per_kWh"] = caps["electricity_p_per_kWh"]
        r = run_scenario(s)
        inv, fin = r["financial"]["investor"], r["financial"]
        rows.append({
            "Policy cost shifted off electricity (%)": int(shift * 100),
            "Electricity (p/kWh)": caps["electricity_p_per_kWh"],
            "Gas (p/kWh)": caps["gas_p_per_kWh"],
            "Spark gap": caps["spark_gap_ratio"],
            "District OPEX (£m/yr)": round(r["headline"]["annual_total_opex_GBP"] / 1e6, 2),
            "Individual-HP bill (£m/yr)": round(r["counterfactual"]["total_annual_opex_GBP"] / 1e6, 2),
            "Avoided cost (£m/yr)": round(fin["annual_avoided_cost_GBP"] / 1e6, 2),
            "Fair tariff (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 1),
            "Whole-system NPV @3.5% (£m)": round(fin["npv_vs_counterfactual_GBP"] / 1e6, 1),
        })
    return pd.DataFrame(rows)


def fig_counterfactual(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = ["vs individual\ngas boilers", "vs individual\nheat pumps"]
    vals = df["Whole-system NPV @3.5% (£m)"]
    ax.bar(labels, vals, color=[C_RED if v < 0 else C_GREEN for v in vals], width=0.5)
    ax.axhline(0, color=INK, lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + (4 if v > 0 else -8), f"£{v:+.0f}m", ha="center", fontweight="bold",
                color=C_GREEN if v > 0 else C_RED)
    ax.set_ylabel("Whole-system NPV @3.5% (£m)")
    ax.set_title("Birmingham Central: the answer depends entirely on the alternative",
                 loc="left", fontweight="bold")
    _save(fig, "CF1_counterfactual_flips_the_sign.png")


def fig_levy(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = df["Policy cost shifted off electricity (%)"]
    ax1.plot(x, df["District OPEX (£m/yr)"], "o-", color=C_BLUE, lw=2, label="District scheme OPEX")
    ax1.plot(x, df["Individual-HP bill (£m/yr)"], "o-", color=C_YELLOW, lw=2, label="Individual heat-pump bill")
    ax1.set_xlabel("Policy cost shifted off electricity (%)"); ax1.set_ylabel("£m/yr")
    ax1.set_title("Both sides get cheaper", loc="left", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)

    ax2.plot(x, df["Whole-system NPV @3.5% (£m)"], "o-", color=C_GREEN, lw=2)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_xlabel("Policy cost shifted off electricity (%)")
    ax2.set_ylabel("Whole-system NPV @3.5% (£m)")
    ax2.set_title("But the alternative gains more", loc="left", fontweight="bold")
    _save(fig, "CF2_levy_rebalancing.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cf = counterfactual_comparison()
    bus = bus_eligibility_table()
    levy = levy_sensitivity()

    for df, n in ((cf, "counterfactual_comparison"), (bus, "bus_eligibility"), (levy, "levy_sensitivity")):
        df.to_csv(OUT / f"{n}.csv", index=False)
    fig_counterfactual(cf)
    fig_levy(levy)

    lines = [
        "# The alternative, and the levy",
        "",
        "Site: Birmingham Central IZO anchor core (10 named buildings, 60.2 GWh, report costs,",
        "62/30). Run against the live engine.",
        "",
        "## 1. What is the alternative?",
        "",
        cf.to_markdown(index=False),
        "",
        "## 2. Where the £7,500 BUS grant actually lands",
        "",
        f"BUS caps at **{BUS_MAX_CAPACITY_KWTH:.0f} kWth per installation**. It transforms the",
        "individual-heat-pump case for a house and does nothing whatever for a shopping centre.",
        "",
        bus.to_markdown(index=False),
        "",
        "Birmingham Central is ~75% non-domestic by heat, so BUS returns **£0** across its anchor",
        "loads. A residential-led zone would look completely different — worth testing separately.",
        "",
        "## 3. What if the green levy came off electricity?",
        "",
        f"Today: electricity {OFGEM_ELECTRICITY_CAP_P_PER_KWH}p, gas {OFGEM_GAS_CAP_P_PER_KWH}p, "
        f"spark gap **{OFGEM_SPARK_GAP_RATIO:.2f}:1** at the Ofgem cap.",
        "",
        levy.to_markdown(index=False),
        "",
        "## Method and caveats",
        "",
        "- The levy shift is modelled revenue-neutral between the two fuels. That is the",
        "  CONSERVATIVE form: moving the cost to general taxation instead (as the April 2026",
        "  Renewables Obligation change actually did) cuts electricity without raising gas, which",
        "  is better for heat pumps than what is modelled here.",
        "- Policy-cost shares (18% electricity, 8% gas) are applied to the unit rate. Some policy",
        "  cost genuinely sits in the standing charge, so this is approximate. Treat the direction",
        "  and rough magnitude as the finding, not the decimals.",
        "- The individual-heat-pump counterfactual now prices electricity at the Ofgem cap, not at",
        "  the ~24p/kWh large-business rate the component default resolves to. That was a real bug",
        "  — the same one already found and fixed on the gas side — and is very likely why this",
        "  counterfactual was never wired up.",
        "- BUS is a customer-facing transfer, not a resource cost, so it is applied to the",
        "  customer/investor view and excluded from the whole-system social case — the same",
        "  treatment this project already gives the GHNF grant.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== 1. What is the alternative? ===")
    print(cf.to_string(index=False))
    print("\n=== 2. Where the BUS grant lands ===")
    print(bus.to_string(index=False))
    print("\n=== 3. Taking the levy off electricity ===")
    print(levy.to_string(index=False))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
