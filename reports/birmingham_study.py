"""Birmingham heat network zoning — run the real report through the model.

    python -m reports.birmingham_study

Runs against the live engine (scenarios.scenario_runner.run_scenario) — the same
entry point main.py and the Streamlit app use. Not a test fixture.

Source: DESNZ, "Heat Network Zoning: Zone Opportunity Report — Birmingham",
February 2025. See scenarios/birmingham_zoning.py for every figure quoted, with
its table number, and for an explicit list of what the report does NOT contain.

The three questions
-------------------
1. What does the report's own cost per metre look like against this model's
   SEAI-fitted pipe curve? The report gives four IZOs at four different £/m, in
   one city, in one year, from one methodology — which is about as clean a test
   of a screening cost curve as you will find.

2. What does reusing existing pipework do to the cost? The QE Hospital IZO
   "utilises the existing heat network pipes" and comes in at £1,750/m against
   Birmingham Central's £3,750/m. Both are in the report; the difference is real
   money.

3. Run at the report's real costs, is the Central IZO investable on gas-parity
   customer revenue? (Spoiler: the report never claims it is. It used "a proxy
   for economic viability", not a cash-flow model, and this is the first time its
   numbers meet one.)

Writes to output/birmingham/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from network.pipe_catalog import estimate_pipe_cost_GBP_per_m, size_pipe_for_peak
from scenarios.birmingham_zoning import (
    BDEC_GROWTH_SCENARIOS,
    CENTRAL_ASHP_TOTAL_MW,
    CENTRAL_HEAT_SOURCES,
    CENTRAL_TOTAL_IDENTIFIED_SUPPLY_MW,
    IZO_NETWORK,
    REPORT_CITATION,
    REPORT_DESIGN_LHD_TARGET_MWh_per_m,
    central_izo_scenario,
)
from scenarios.scenario_runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "birmingham"

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


# ═══════════════════════════════════════════════════════════════════════════
# 1. The report's own cost per metre, against this model's pipe curve
# ═══════════════════════════════════════════════════════════════════════════

def izo_cost_table() -> pd.DataFrame:
    rows = []
    for izo, m in IZO_NETWORK.items():
        length_m = m["length_km"] * 1000.0
        rows.append({
            "IZO": izo,
            "Route (km)": m["length_km"],
            "Report network cost (£m)": m["network_cost_GBP"] / 1e6,
            "Report £/m": round(m["network_cost_GBP"] / length_m, 0),
            "Report total CapEx (£m)": m["total_capex_GBP"] / 1e6,
            "Non-network CapEx (£m)": (m["total_capex_GBP"] - m["network_cost_GBP"]) / 1e6,
            "Network share of CapEx (%)": round(m["network_cost_GBP"] / m["total_capex_GBP"] * 100, 1),
            "Heat (GWh/yr)": m["annual_heat_GWh"],
            "Report LHD (MWh/m/yr)": m["linear_heat_density_MWh_per_m"],
            "Meets 4 MWh/m design target": m["linear_heat_density_MWh_per_m"] >= REPORT_DESIGN_LHD_TARGET_MWh_per_m,
            "Reuses existing pipework": bool(m.get("reuses_existing_pipework", False)),
            "Sources (report)": m["sources_described"],
        })
    return pd.DataFrame(rows)


def model_curve_comparison() -> pd.DataFrame:
    """What this model's pipe curve would have charged for the same routes.

    The model sizes a pipe from a peak duty, so each IZO's peak is estimated from
    its annual heat at a 0.30 district-heating load factor, then the trunk DN is
    sized at 70/40 and priced off the SEAI-fitted curve. This is a trunk-only
    figure — a real network is a mix of trunk and smaller branches, so it is a
    LOWER bound on what the curve implies for the whole route, which only makes
    the gap below more conservative, not less.
    """
    rows = []
    for izo, m in IZO_NETWORK.items():
        peak_MW = m["annual_heat_GWh"] * 1000.0 / 8760.0 / 0.30
        report_per_m = m["network_cost_GBP"] / (m["length_km"] * 1000.0)
        try:
            pipe = size_pipe_for_peak(peak_MW * 1000.0, 70.0, 40.0)
            dn_label = f"DN{pipe.DN}"
            model_per_m = estimate_pipe_cost_GBP_per_m(pipe.DN)
        except ValueError:
            # The duty exceeds DN600, the largest pipe in the catalog (Logstor's
            # published EN 13941 range). The catalog raises rather than
            # extrapolating into bespoke territory, which is correct behaviour and
            # is itself a finding: a zone-scale trunk at this peak needs parallel
            # mains or a larger bespoke product, and this screening tool should
            # hand off to a real quote rather than invent a number. Price the
            # route at the largest pipe that DOES exist, and flag it — that makes
            # the model figure a hard LOWER bound.
            dn_label = ">DN600 (exceeds catalog)"
            model_per_m = estimate_pipe_cost_GBP_per_m(600)
        rows.append({
            "IZO": izo,
            "Est. peak (MW)": round(peak_MW, 1),
            "Model trunk DN": dn_label,
            "Model £/m (SEAI curve)": round(model_per_m, 0),
            "Report £/m": round(report_per_m, 0),
            "Report ÷ model": round(report_per_m / model_per_m, 2),
            "Model would price route at (£m)": round(model_per_m * m["length_km"] * 1000.0 / 1e6, 1),
            "Report network cost (£m)": m["network_cost_GBP"] / 1e6,
            "Understatement (£m)": round(
                (report_per_m - model_per_m) * m["length_km"] * 1000.0 / 1e6, 1
            ),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Run the Central IZO at the report's real costs
# ═══════════════════════════════════════════════════════════════════════════

def run_central_cases() -> tuple[pd.DataFrame, dict]:
    cases, provenance = [], None

    scenario, provenance = central_izo_scenario()
    report_cost = run_scenario(scenario)
    cases.append(("Report network cost (£150m, 40km)", report_cost))

    model_scenario, _ = central_izo_scenario(
        name="Birmingham Central IZO (model pipe curve)", use_report_network_cost=False,
    )
    cases.append(("Model SEAI pipe curve", run_scenario(model_scenario)))

    # CIBSE's 60/30 heat-pump target, at the report's costs.
    lowtemp, _ = central_izo_scenario(
        name="Birmingham Central IZO (60/30, instantaneous HIU)",
        heat_flow_temp_C=60.0, heat_return_temp_C=30.0, dhw_system="instantaneous_hiu",
    )
    cases.append(("CIBSE 60/30 target, instantaneous HIU", run_scenario(lowtemp)))

    rows = []
    for label, r in cases:
        h, inv = r["headline"], r["financial"]["investor"]
        rows.append({
            "Case": label,
            "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 1),
            "of which network (£m)": round(h["capex_network_GBP"] / 1e6, 1),
            "Annual OPEX (£m)": round(h["annual_total_opex_GBP"] / 1e6, 2),
            "Heat delivered (GWh)": round(h["annual_heat_demand_MWh"] / 1000.0, 1),
            "Peak heat (MW)": h["peak_heat_MW"],
            "LHD (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
            "Loss (%)": round(h["network_heat_loss_fraction"] * 100, 1),
            "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
            "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
            "Unmet (MWh)": round(h["annual_unmet_demand_MWh"], 1),
            "Required tariff (p/kWh)": inv.get("required_heat_tariff_p_per_kWh_for_zero_NPV"),
            "Equivalent tariff (p/kWh)": inv.get("equivalent_year1_heat_tariff_p_per_kWh"),
            "NPV (£m)": round(inv["npv_GBP"] / 1e6, 1),
            "Decision": r["screening"]["status"],
        })
    return pd.DataFrame(rows), provenance


# ═══════════════════════════════════════════════════════════════════════════
# 3. Existing pipework
# ═══════════════════════════════════════════════════════════════════════════

def existing_pipework_analysis() -> pd.DataFrame:
    """What reusing existing pipe is worth, from the report's own two data points.

    QE Hospital reuses existing pipe and costs £1,750/m. Birmingham Central is new
    build through a congested city centre at £3,750/m. Applying the QE rate to the
    other three routes is a bounding illustration, not a proposal — you cannot
    reuse pipe that does not exist. It sizes the prize.
    """
    qe = IZO_NETWORK["Queen Elizabeth Hospital / University of Birmingham"]
    qe_rate = qe["network_cost_GBP"] / (qe["length_km"] * 1000.0)
    rows = []
    for izo, m in IZO_NETWORK.items():
        length_m = m["length_km"] * 1000.0
        actual = m["network_cost_GBP"]
        at_qe_rate = qe_rate * length_m
        rows.append({
            "IZO": izo,
            "Route (km)": m["length_km"],
            "Reuses existing pipework": bool(m.get("reuses_existing_pipework", False)),
            "Actual £/m (report)": round(actual / length_m, 0),
            "£/m if at QE reuse rate": round(qe_rate, 0),
            "Actual network cost (£m)": round(actual / 1e6, 1),
            "At QE reuse rate (£m)": round(at_qe_rate / 1e6, 1),
            "Notional saving (£m)": round((actual - at_qe_rate) / 1e6, 1),
            "Saving (% of network cost)": round((actual - at_qe_rate) / actual * 100, 1) if actual else 0.0,
        })
    return pd.DataFrame(rows)


def bdec_table() -> pd.DataFrame:
    rows = []
    for name, s in BDEC_GROWTH_SCENARIOS.items():
        rows.append({
            "BDEC growth scenario": name,
            "Annual demand (GWh)": s["annual_demand_GWh"],
            "CapEx (£m)": s["capex_GBP"] / 1e6,
            "£ per annual MWh": round(s["capex_GBP"] / (s["annual_demand_GWh"] * 1000.0), 0),
            "Heat sources": s["sources"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_cost_per_m(curve: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [i.replace(" / ", "\n").replace("Queen Elizabeth Hospital", "QE Hospital") for i in curve["IZO"]]
    x = range(len(curve))
    ax.bar([i - 0.2 for i in x], curve["Model £/m (SEAI curve)"], width=0.4,
           color=C_BLUE, label="Model — SEAI fitted curve")
    ax.bar([i + 0.2 for i in x], curve["Report £/m"], width=0.4,
           color=C_RED, label="Report — DESNZ Birmingham, Feb 2025")
    for i, (mv, rv) in enumerate(zip(curve["Model £/m (SEAI curve)"], curve["Report £/m"])):
        ax.text(i + 0.2, rv + 60, f"{rv/mv:.1f}x", ha="center", fontsize=9, color=C_RED, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Network cost (£ per metre)")
    ax.set_title("The model's pipe curve against four real Birmingham routes", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, "B1_cost_per_m_model_vs_report.png")


def fig_existing_pipework(ex: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    reuse = ex["Reuses existing pipework"]
    colors = [C_GREEN if r else C_RED for r in reuse]
    labels = [i.replace(" / ", "\n").replace("Queen Elizabeth Hospital", "QE Hospital") for i in ex["IZO"]]
    ax.bar(labels, ex["Actual £/m (report)"], color=colors)
    ax.axhline(ex["£/m if at QE reuse rate"].iloc[0], ls="--", color=C_GREEN, lw=1.4)
    ax.text(0.02, ex["£/m if at QE reuse rate"].iloc[0] + 90,
            "QE rate — reuses existing pipe (£1,750/m)", color=C_GREEN, fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_ylabel("Network cost (£ per metre)")
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("Existing pipework vs congested new build, from the report's own figures",
                 loc="left", fontweight="bold")
    _save(fig, "B2_existing_pipework.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    izo = izo_cost_table()
    curve = model_curve_comparison()
    cases, provenance = run_central_cases()
    existing = existing_pipework_analysis()
    bdec = bdec_table()

    for df, name in ((izo, "izo_costs"), (curve, "model_vs_report_pipe_cost"),
                     (cases, "central_izo_model_runs"), (existing, "existing_pipework"),
                     (bdec, "bdec_growth")):
        df.to_csv(OUT / f"{name}.csv", index=False)

    fig_cost_per_m(curve)
    fig_existing_pipework(existing)

    sources_df = pd.DataFrame(CENTRAL_HEAT_SOURCES)
    sources_df.to_csv(OUT / "central_heat_sources.csv", index=False)

    lines = [
        "# Birmingham heat network zoning — the real report, run through the model",
        "",
        f"Source: **{REPORT_CITATION}**.",
        "",
        "Run against the live engine (`scenarios.scenario_runner.run_scenario`) — the same",
        "entry point `main.py` and the Streamlit app use. Every cost below is the report's own",
        "figure; the model's pipe curve is used only for the comparison in section 2.",
        "",
        "## 1. The four IZOs, as reported",
        "",
        izo.to_markdown(index=False),
        "",
        f"The report designed the IZOs to a linear heat density target of "
        f"**{REPORT_DESIGN_LHD_TARGET_MWh_per_m} MWh/m/yr**, describing that as \"a relatively low proxy "
        "for economic viability with the heat network sector in England\".",
        "",
        "## 2. The report's cost per metre against this model's pipe curve",
        "",
        curve.to_markdown(index=False),
        "",
        "## 3. Birmingham Central IZO, run at the report's real costs",
        "",
        cases.to_markdown(index=False),
        "",
        "## 4. Existing pipework",
        "",
        existing.to_markdown(index=False),
        "",
        f"> {IZO_NETWORK['Queen Elizabeth Hospital / University of Birmingham']['existing_pipework_note']}",
        "",
        "## 5. The existing BDEC network's costed growth path (report Table 2)",
        "",
        bdec.to_markdown(index=False),
        "",
        "## 6. Birmingham Central heat sources (report Table 5)",
        "",
        sources_df.to_markdown(index=False),
        "",
        f"Report identifies ~{CENTRAL_TOTAL_IDENTIFIED_SUPPLY_MW} MWth total. This model can represent "
        f"**{CENTRAL_ASHP_TOTAL_MW:.1f} MW of ASHP only** — it has no water- or ground-source heat pump "
        "type, so the 5 MW river WSHP and 1.2+ MW of GSHP are omitted and gas peak covers the balance. "
        "Carbon and OPEX here are therefore **conservative**, and worse than the report's intent.",
        "",
        "## Provenance — which numbers are the report's and which are the model's",
        "",
        "| Input | Source |",
        "|---|---|",
    ]
    for k, v in provenance.items():
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "The report contains **no** design temperatures, tariffs, discount rate, or operating",
        "costs — it used \"a proxy for economic viability\", not a cash-flow model. Those inputs",
        "are this model's assumptions and the NPV figures inherit them. The report never claims",
        "these zones are investable on customer revenue, and nothing here should be read as",
        "contradicting it.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== 1. The four IZOs, as reported ===")
    print(izo.to_string(index=False))
    print("\n=== 2. Report £/m vs this model's SEAI pipe curve ===")
    print(curve.to_string(index=False))
    print("\n=== 3. Birmingham Central at the report's real costs ===")
    print(cases.to_string(index=False))
    print("\n=== 4. Existing pipework ===")
    print(existing.to_string(index=False))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
