"""What must electricity cost for heat networks to beat gas?

    python -m reports.electricity_breakeven

Runs against the live engine (scenarios.scenario_runner.run_scenario).

The question
------------
Gas heat is cheap: 7.33 p/kWh of gas divided by a 90% boiler gives 8.14 p/kWh of
HEAT. Everything electric is priced against that. So the useful question is not
"is district heat viable" — it is "at what electricity price does anything
electric beat 8.14p?"

Two comparisons, and they give different answers
-------------------------------------------------
RUNNING COST ONLY (p/kWh of heat, fuel + O&M, no capital). This is the number
everyone quotes and it flatters heat pumps, because their whole disadvantage is
capital. Useful because it is the number a customer feels on a bill.

FULL LIFETIME COST (capital + running, levelised). This is what actually decides
anything. A gas boiler is £111/kW; an individual heat pump is £1,150/kW; a
district network is neither. Ignoring that is how people talk themselves into
schemes that do not work.

Both are reported. Lead with the second.

Three curves on one axis
-------------------------
  GAS BOILER          flat — it does not care what electricity costs
  INDIVIDUAL HEAT PUMP  electricity / COP — the alternative that is actually legal
  DISTRICT NETWORK     from the real model run at each electricity price

Where the district curve crosses gas is the PPA Dalkia would need to beat gas.
Where it crosses the individual heat pump is the price at which a network beats
the alternative it is really competing with.

Writes to output/electricity_breakeven/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from economics.tariffs import (
    OFGEM_ELECTRICITY_CAP_P_PER_KWH,
    OFGEM_GAS_CAP_P_PER_KWH,
)
from scenarios.scenario_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "electricity_breakeven"

C_BLUE, C_RED, C_GREEN, C_YELLOW, C_VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

# A condensing gas boiler at seasonal efficiency. The model's own part-load curve
# lands near this; 0.90 is used here so the headline number is reproducible by
# hand, which matters when someone in the room challenges it.
GAS_BOILER_SEASONAL_EFFICIENCY = 0.90
GAS_HEAT_COST_P_PER_KWH = OFGEM_GAS_CAP_P_PER_KWH / GAS_BOILER_SEASONAL_EFFICIENCY   # 8.14

ELEC_PRICES = [4.0, 8.0, 12.0, 16.0, 20.0, 22.0, 24.0, 26.11, 28.0, 30.0]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _scenario(elec_p_per_kWh: float, mix: str):
    """The Exeter best case, at a given electricity price."""
    from analysis.exeter_best_case import (
        EXTENDED_BUILDINGS, EXTENDED_SEGMENTS, _peak_MW, build,
    )
    peak = _peak_MW(EXTENDED_BUILDINGS, False)
    s = build(f"breakeven {mix} @ {elec_p_per_kWh}p", EXTENDED_BUILDINGS,
              EXTENDED_SEGMENTS, mix, False, peak)
    s = copy.deepcopy(s)
    for src in s["sources"]:
        if src["type"] in {"ashp", "wshp", "gshp", "booster_heat_pump", "electric_boiler"}:
            src["electricity_price_GBP_per_MWh"] = elec_p_per_kWh * 10.0
    s["economics"]["counterfactual_electricity_price_p_per_kWh"] = elec_p_per_kWh
    s["economics"]["parasitic_electricity_price_GBP_per_MWh"] = elec_p_per_kWh * 10.0
    return s


def sweep() -> pd.DataFrame:
    rows = []
    for mix in ("ASHP + gas peak", "EfW steam extraction + ASHP + gas peak"):
        for p in ELEC_PRICES:
            r = run_scenario(_scenario(p, mix))
            h, inv, cf = r["headline"], r["financial"]["investor"], r["counterfactual"]
            heat_kWh = h["annual_heat_demand_MWh"] * 1000.0

            # Running cost only — fuel + O&M + overhead, no capital.
            district_running = h["annual_total_opex_GBP"] / heat_kWh * 100.0
            # Individual heat pump running cost, from the model's own counterfactual.
            hp_running = cf["total_annual_fuel_electricity_GBP"] / heat_kWh * 100.0
            # Full lifetime cost of the district scheme = the tariff it must charge
            # to break even, which is exactly "capital + running, levelised".
            district_full = inv["required_heat_tariff_p_per_kWh_for_zero_NPV"]

            rows.append({
                "Plant mix": mix,
                "Electricity price (p/kWh)": p,
                "Gas boiler heat — running cost (p/kWh)": round(GAS_HEAT_COST_P_PER_KWH, 2),
                "Individual heat pump — running cost (p/kWh)": round(hp_running, 2),
                "District network — running cost (p/kWh)": round(district_running, 2),
                "District network — full cost inc. capital (p/kWh)": district_full,
                "District beats gas on running cost": district_running < GAS_HEAT_COST_P_PER_KWH,
                "District beats heat pump on running cost": district_running < hp_running,
                "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 1),
            })
    return pd.DataFrame(rows)


def _crossing(x, y, target):
    """Where curve y(x) crosses a constant target.

    Returns (price, verdict). A bare None conflates two opposite outcomes —
    "always cheaper than gas" and "never cheaper than gas" — which is exactly the
    kind of thing that gets misread off a slide, so the verdict is explicit.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
            t = (target - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i])), "crosses"
    if np.all(y < target):
        return None, "always cheaper than gas"
    return None, "never cheaper than gas at any swept price"


def breakeven_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mix in df["Plant mix"].unique():
        d = df[df["Plant mix"] == mix].sort_values("Electricity price (p/kWh)")
        x = d["Electricity price (p/kWh)"]
        vs_gas, vs_gas_v = _crossing(
            x, d["District network — running cost (p/kWh)"], GAS_HEAT_COST_P_PER_KWH)
        hp_vs_gas, hp_v = _crossing(
            x, d["Individual heat pump — running cost (p/kWh)"], GAS_HEAT_COST_P_PER_KWH)
        # The one that decides: does the FULL cost, capital included, ever beat gas?
        full_vs_gas, full_v = _crossing(
            x, d["District network — full cost inc. capital (p/kWh)"], GAS_HEAT_COST_P_PER_KWH)
        full = d["District network — full cost inc. capital (p/kWh)"]
        rows.append({
            "Plant mix": mix,
            "Today's electricity price (p/kWh)": OFGEM_ELECTRICITY_CAP_P_PER_KWH,
            "RUNNING cost: electricity price where district beats gas (p/kWh)": (
                round(vs_gas, 2) if vs_gas else vs_gas_v
            ),
            "Cut needed from today (%)": (
                round((1 - vs_gas / OFGEM_ELECTRICITY_CAP_P_PER_KWH) * 100, 1) if vs_gas else None
            ),
            "RUNNING cost: electricity price where an individual heat pump beats gas (p/kWh)": (
                round(hp_vs_gas, 2) if hp_vs_gas else hp_v
            ),
            "FULL cost inc. capital: electricity price where district beats gas (p/kWh)": (
                round(full_vs_gas, 2) if full_vs_gas else full_v
            ),
            "District FULL cost at FREE electricity (p/kWh)": round(float(full.min()), 2),
            "District FULL cost today (p/kWh)": round(
                float(d.loc[d["Electricity price (p/kWh)"] == OFGEM_ELECTRICITY_CAP_P_PER_KWH,
                            "District network — full cost inc. capital (p/kWh)"].iloc[0]), 2
            ),
        })
    return pd.DataFrame(rows)


def fig_breakeven(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharex=True)
    for ax, mix in zip(axes, df["Plant mix"].unique()):
        d = df[df["Plant mix"] == mix].sort_values("Electricity price (p/kWh)")
        x = d["Electricity price (p/kWh)"]
        ax.axhline(GAS_HEAT_COST_P_PER_KWH, color=C_RED, lw=2.2,
                   label=f"Gas boiler heat — {GAS_HEAT_COST_P_PER_KWH:.2f}p (flat)")
        ax.plot(x, d["Individual heat pump — running cost (p/kWh)"], "o-", color=C_YELLOW, lw=2,
                label="Individual heat pump — running cost")
        ax.plot(x, d["District network — running cost (p/kWh)"], "o-", color=C_BLUE, lw=2,
                label="District network — running cost")
        ax.plot(x, d["District network — full cost inc. capital (p/kWh)"], "s--", color=C_VIOLET,
                lw=1.6, alpha=0.85, label="District network — FULL cost inc. capital")
        ax.axvline(OFGEM_ELECTRICITY_CAP_P_PER_KWH, color=MUTED, ls=":", lw=1.4)
        ax.text(OFGEM_ELECTRICITY_CAP_P_PER_KWH - 0.4, ax.get_ylim()[1] * 0.92, "today\n26.11p",
                ha="right", fontsize=8.5, color=MUTED)
        ax.set_title(mix, loc="left", fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Electricity price (p/kWh)")
    axes[0].set_ylabel("Cost of 1 kWh of heat (p)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Everything electric is priced against 8.14p gas heat",
                 y=1.02, fontsize=11.5, fontweight="bold", x=0.01, ha="left")
    _save(fig, "BE1_electricity_breakeven.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = sweep()
    be = breakeven_table(df)
    df.to_csv(OUT / "electricity_sweep.csv", index=False)
    be.to_csv(OUT / "breakeven_prices.csv", index=False)
    fig_breakeven(df)

    lines = [
        "# What must electricity cost for heat networks to beat gas?",
        "",
        f"**A kWh of heat from a gas boiler costs {GAS_HEAT_COST_P_PER_KWH:.2f}p** "
        f"({OFGEM_GAS_CAP_P_PER_KWH}p gas at the Ofgem cap, divided by a "
        f"{GAS_BOILER_SEASONAL_EFFICIENCY:.0%} seasonal efficiency). Everything electric is",
        "priced against that number.",
        "",
        "Site: Exeter Central + anchor loads, real tree topology, 62/30, instantaneous HIUs.",
        "",
        "## The break-even prices",
        "",
        be.to_markdown(index=False),
        "",
        "## Full sweep",
        "",
        df.to_markdown(index=False),
        "",
        "## How to read this",
        "",
        "- **Running cost** is fuel + O&M + overhead, no capital. It is the number everyone",
        "  quotes and it FLATTERS heat pumps, whose entire disadvantage is capital.",
        "- **Full cost including capital** is the district scheme's break-even tariff — what it",
        "  must charge to return zero NPV. That is the number that decides anything.",
        "- The gap between the two lines is the capital. It does not move when electricity moves.",
        "",
        "## The caveat that matters",
        "",
        "A cheaper electricity price helps the individual heat pump too — it is 100% electric,",
        "while a district scheme with a gas peak is not. So closing the gap on RUNNING cost does",
        "not automatically make a network beat the alternative it is really competing against.",
        "See output/counterfactual_and_levy/ for that effect priced separately.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nA kWh of heat from a gas boiler = {GAS_HEAT_COST_P_PER_KWH:.2f}p "
          f"({OFGEM_GAS_CAP_P_PER_KWH}p gas / {GAS_BOILER_SEASONAL_EFFICIENCY:.0%})")
    print("\n=== Break-even electricity prices ===")
    print(be.to_string(index=False))
    print("\n=== Full sweep ===")
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
