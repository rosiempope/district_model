"""What do the design temperatures cost? A bounded sweep.

    python -m reports.temperature_sensitivity

Runs against the live engine (scenarios.scenario_runner.run_scenario) — the same
entry point main.py and the Streamlit app use.

Why this is bounded rather than a free sweep
---------------------------------------------
Temperature is one of the few levers an operator genuinely controls, so the
economic gradient is worth knowing. But a sweep with no boundary conditions
answers the wrong question: it would report that 45°C flow is free money — a
much better ASHP COP, lower losses, the same pipe — while quietly starving every
customer of hot water. The delivered-temperature gate is now wired into the
engine (see network/design_temperature_limits.py) precisely so this sweep cannot
do that.

The bounds are the standards, not the modeller's taste:

    flow    55-70°C   CP1 2020: 70°C is the MAXIMUM for new schemes; 55°C is the
                      permitted minimum. Anything outside is not a design CP1
                      sanctions, so there is no point pricing it.
    return  25-45°C   CP1 2020 best practice is a VWART below 33°C. The upper end
                      is where UK schemes actually sit today.
    floor   55 or 65°C depending on DHW system — instantaneous HIU vs stored
                      cylinder. This is the constraint that decides the answer.

Two levers, not one
-------------------
They are independent and it matters:

  FLOW TEMPERATURE moves COP and heat loss. It does NOT move pipe size — pipe
  size follows delta-T. So 55/25 and 70/40 size to the same DN. Dropping flow
  temperature at constant delta-T is close to free, and is bounded only by
  whether heat still arrives hot enough to make DHW.

  DELTA-T moves pipe size, pumping and hence CAPEX. A wider delta-T (lower
  return) shrinks the pipe. At zone scale it also decides whether the network is
  buildable with standard pipe at all: a single DN600 trunk carries ~80 MW at
  70/40 but ~110 MW at 70/30.

Site
----
Exeter Central (real tree topology from the DESNZ zoning map, 5 zones, 254
connections, 3,900 m). Tree mode is required: delivered temperature needs real
route lengths to propagate along, and generic_length has none.

Writes to output/temperature_sensitivity/.
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network.design_temperature_limits import (
    CIBSE_HEAT_PUMP_TARGET_FLOW_C,
    CIBSE_HEAT_PUMP_TARGET_RETURN_C,
    CP1_BEST_PRACTICE_VWART_C,
    CP1_MAX_FLOW_TEMP_NEW_SCHEME_C,
    CP1_MIN_PERMITTED_FLOW_TEMP_C,
    minimum_delivered_temp_C,
)
from network.pipe_catalog import size_pipe_for_peak
from scenarios.scenario_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "temperature_sensitivity"

C_BLUE, C_RED, C_GREEN, C_YELLOW = "#2a78d6", "#e34948", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

FLOW_TEMPS = [55.0, 60.0, 65.0, 70.0]
RETURN_TEMPS = [25.0, 30.0, 35.0, 40.0, 45.0]
DHW_SYSTEMS = ["instantaneous_hiu", "stored_cylinder"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _base_scenario():
    from analysis.exeter_case_study import (
        CENTRAL_BUILDINGS, CENTRAL_SEGMENTS, build_tree_scenario,
    )
    return build_tree_scenario(
        "Exeter Central — temperature sweep", CENTRAL_BUILDINGS, CENTRAL_SEGMENTS,
        ["ashp", "gas_boiler"],
    )


def sweep() -> pd.DataFrame:
    base = _base_scenario()
    rows = []
    for dhw in DHW_SYSTEMS:
        floor = minimum_delivered_temp_C(dhw)
        for flow in FLOW_TEMPS:
            for ret in RETURN_TEMPS:
                if flow - ret < 5.0:
                    continue   # schema requires a usable delta-T
                s = copy.deepcopy(base)
                s["network"].update({
                    "heat_flow_temp_C": flow, "heat_return_temp_C": ret, "dhw_system": dhw,
                })
                for src in s["sources"]:
                    if src["type"] == "ashp":
                        src["flow_temp_C"] = flow
                r = run_scenario(s)
                h, inv = r["headline"], r["financial"]["investor"]
                cops = [
                    float(np.asarray(src.cop_hourly).mean())
                    for src in r["heat_sources"] if hasattr(src, "cop_hourly")
                ]
                rows.append({
                    "dhw_system": dhw,
                    "flow_C": flow,
                    "return_C": ret,
                    "delta_T_K": flow - ret,
                    "delivered_floor_C": floor,
                    "worst_delivered_C": h["worst_case_delivered_temp_C"],
                    "delivered_margin_K": h["delivered_temp_margin_C"],
                    "delivered_compliant": h["delivered_temp_compliant"],
                    "mean_ASHP_COP": round(cops[0], 3) if cops else None,
                    "CAPEX_GBP": h["capex_total_GBP"],
                    "network_CAPEX_GBP": h["capex_network_GBP"],
                    "annual_OPEX_GBP": h["annual_total_opex_GBP"],
                    "heat_loss_pct": round(h["network_heat_loss_fraction"] * 100, 2),
                    "pumping_MWh": h["annual_pumping_electricity_MWh"],
                    "carbon_gCO2e_per_kWh": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                    "required_tariff_p_per_kWh": inv.get("required_heat_tariff_p_per_kWh_for_zero_NPV"),
                    "NPV_GBP": inv["npv_GBP"],
                    "meets_CP1_flow": CP1_MIN_PERMITTED_FLOW_TEMP_C <= flow <= CP1_MAX_FLOW_TEMP_NEW_SCHEME_C,
                    "meets_CP1_VWART": ret <= CP1_BEST_PRACTICE_VWART_C,
                    "decision": r["screening"]["status"],
                })
    return pd.DataFrame(rows)


def trunk_ceiling_table() -> pd.DataFrame:
    """The delta-T lever at zone scale: what peak fits the largest standard pipe.

    Not a sweep of the Exeter site — a property of the pipe catalog itself, and
    the constraint that stopped the Birmingham Central IZO being modellable as a
    single trunk at 70/40.
    """
    rows = []
    for flow, ret in [(70, 40), (70, 35), (70, 30), (70, 25), (60, 30), (60, 25), (55, 30)]:
        last_ok = None
        for MW in range(10, 201, 5):
            try:
                p = size_pipe_for_peak(MW * 1000.0, float(flow), float(ret))
                last_ok = (MW, p.DN)
            except ValueError:
                break
        rows.append({
            "flow/return": f"{flow}/{ret}",
            "delta_T_K": flow - ret,
            "max peak on one standard trunk (MW)": last_ok[0] if last_ok else None,
            "at": f"DN{last_ok[1]}" if last_ok else "—",
        })
    return pd.DataFrame(rows)


def fig_npv_grid(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, dhw in zip(axes, DHW_SYSTEMS):
        d = df[df["dhw_system"] == dhw]
        for ret in sorted(d["return_C"].unique()):
            sub = d[d["return_C"] == ret].sort_values("flow_C")
            ok = sub[sub["delivered_compliant"]]
            bad = sub[~sub["delivered_compliant"].astype(bool)]
            line, = ax.plot(ok["flow_C"], ok["NPV_GBP"] / 1e6, "o-", lw=1.8, label=f"return {ret:.0f}°C")
            ax.plot(bad["flow_C"], bad["NPV_GBP"] / 1e6, "x", ms=7, color=line.get_color(), alpha=0.55)
        ax.set_title(
            f"{dhw.replace('_', ' ')} — floor {minimum_delivered_temp_C(dhw):.0f}°C",
            loc="left", fontweight="bold", fontsize=11,
        )
        ax.set_xlabel("Network flow temperature (°C)")
    axes[0].set_ylabel("Investor NPV (£m)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "x = heat does not arrive hot enough to make DHW (would be an invalid design)",
        y=0.02, fontsize=9, color=MUTED,
    )
    _save(fig, "T1_npv_vs_flow_temp.png")


def fig_cop_and_capex(df: pd.DataFrame):
    d = df[df["dhw_system"] == "instantaneous_hiu"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    by_flow = d.groupby("flow_C")["mean_ASHP_COP"].mean()
    ax1.plot(by_flow.index, by_flow.values, "o-", color=C_BLUE, lw=2)
    ax1.axvline(CP1_MAX_FLOW_TEMP_NEW_SCHEME_C, ls="--", color=C_RED, lw=1.2)
    ax1.text(CP1_MAX_FLOW_TEMP_NEW_SCHEME_C - 0.4, by_flow.min(), "CP1 max\n(new schemes)",
             ha="right", fontsize=8.5, color=C_RED)
    ax1.axvline(CIBSE_HEAT_PUMP_TARGET_FLOW_C, ls="--", color=C_GREEN, lw=1.2)
    ax1.text(CIBSE_HEAT_PUMP_TARGET_FLOW_C + 0.3, by_flow.min(), "CIBSE 60/30\ntarget",
             fontsize=8.5, color=C_GREEN)
    ax1.set_xlabel("Network flow temperature (°C)"); ax1.set_ylabel("Mean annual ASHP COP")
    ax1.set_title("Flow temperature buys COP", loc="left", fontweight="bold")

    by_dt = d.groupby("delta_T_K")["network_CAPEX_GBP"].mean() / 1e6
    ax2.plot(by_dt.index, by_dt.values, "o-", color=C_YELLOW, lw=2)
    ax2.set_xlabel("Design ΔT (K)"); ax2.set_ylabel("Network CAPEX (£m)")
    ax2.set_title("ΔT buys pipe size — a separate lever", loc="left", fontweight="bold")
    _save(fig, "T2_cop_and_capex_levers.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = sweep()
    ceiling = trunk_ceiling_table()
    df.to_csv(OUT / "temperature_sweep.csv", index=False)
    ceiling.to_csv(OUT / "trunk_ceiling_by_delta_t.csv", index=False)
    fig_npv_grid(df)
    fig_cop_and_capex(df)

    inst = df[df["dhw_system"] == "instantaneous_hiu"]
    stored = df[df["dhw_system"] == "stored_cylinder"]
    best_inst = inst[inst["delivered_compliant"]].nlargest(1, "NPV_GBP")
    best_stored = stored[stored["delivered_compliant"]].nlargest(1, "NPV_GBP")
    base = df[(df["flow_C"] == 70) & (df["return_C"] == 40) & (df["dhw_system"] == "instantaneous_hiu")]

    lines = [
        "# What do the design temperatures cost?",
        "",
        "Generated by `python -m reports.temperature_sensitivity`, against the live engine.",
        "Site: Exeter Central, real tree topology, 254 connections, 3,900 m.",
        "",
        "## Bounds — the standards, not taste",
        "",
        f"- Flow **{CP1_MIN_PERMITTED_FLOW_TEMP_C:.0f}–{CP1_MAX_FLOW_TEMP_NEW_SCHEME_C:.0f}°C**: "
        "CP1 2020 sets 70°C as the maximum for new schemes and permits down to 55°C.",
        f"- Return: CP1 2020 best practice is a VWART below **{CP1_BEST_PRACTICE_VWART_C:.0f}°C**.",
        f"- CIBSE's heat-pump-led target is **{CIBSE_HEAT_PUMP_TARGET_FLOW_C:.0f}/"
        f"{CIBSE_HEAT_PUMP_TARGET_RETURN_C:.0f}**.",
        "- Delivered floor: **55°C** with instantaneous HIUs (50°C outlet, HSE 'low risk', "
        "+5K approach) or **65°C** with stored cylinders (60°C stored per HSG274, +5K coil).",
        "",
        "## Full sweep",
        "",
        df.to_markdown(index=False),
        "",
        "## The ΔT lever at zone scale",
        "",
        "What peak a single largest-standard-pipe trunk can carry. This is the constraint that",
        "stopped the Birmingham Central IZO (~99 MW) being modellable as one trunk at 70/40 —",
        "and note what widening ΔT does to it.",
        "",
        ceiling.to_markdown(index=False),
        "",
        "## Read-out",
        "",
    ]
    if len(base) and len(best_inst):
        b, bi = base.iloc[0], best_inst.iloc[0]
        lines += [
            f"- Today's design (70/40, instantaneous HIU): NPV £{b['NPV_GBP']/1e6:.2f}m, "
            f"mean ASHP COP {b['mean_ASHP_COP']:.2f}, delivered {b['worst_delivered_C']:.1f}°C "
            f"(**{b['delivered_margin_K']:+.1f}K** against its floor — unused margin).",
            f"- Best compliant, instantaneous HIU: **{bi['flow_C']:.0f}/{bi['return_C']:.0f}**, "
            f"NPV £{bi['NPV_GBP']/1e6:.2f}m, COP {bi['mean_ASHP_COP']:.2f} "
            f"(**{(bi['mean_ASHP_COP']/b['mean_ASHP_COP']-1)*100:+.1f}%** vs 70/40), "
            f"NPV delta **£{(bi['NPV_GBP']-b['NPV_GBP'])/1e6:+.2f}m**.",
        ]
    if len(best_stored):
        bs = best_stored.iloc[0]
        lines.append(
            f"- Best compliant with **stored cylinders**: {bs['flow_C']:.0f}/{bs['return_C']:.0f}, "
            f"NPV £{bs['NPV_GBP']/1e6:.2f}m. The cylinder's 65°C floor forecloses the low-temperature "
            "options entirely — the DHW system choice, not the network, is what gates the COP gain."
        )
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Temperature sweep (instantaneous HIU) ===")
    cols = ["flow_C", "return_C", "delta_T_K", "worst_delivered_C", "delivered_margin_K",
            "delivered_compliant", "mean_ASHP_COP", "CAPEX_GBP", "annual_OPEX_GBP",
            "heat_loss_pct", "carbon_gCO2e_per_kWh", "NPV_GBP"]
    print(inst[cols].to_string(index=False))
    print("\n=== Stored cylinder — the 65C floor forecloses low temperatures ===")
    print(stored[["flow_C", "return_C", "worst_delivered_C", "delivered_margin_K",
                  "delivered_compliant", "NPV_GBP"]].to_string(index=False))
    print("\n=== Max peak on one standard trunk, by delta-T ===")
    print(ceiling.to_string(index=False))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
