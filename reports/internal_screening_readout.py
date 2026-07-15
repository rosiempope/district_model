"""Create an internal comparison pack from the version 2.4 screening sets."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import WORKED_SCENARIOS
from scenarios.feasibility_comparison import FEASIBILITY_SCENARIOS
from scenarios.data_centre_feasibility import DATA_CENTRE_SCENARIOS


SCENARIO_SETS = {
    "Worked technology option": WORKED_SCENARIOS,
    "Route/commercial feasibility": FEASIBILITY_SCENARIOS,
    "Data-centre waste heat": DATA_CENTRE_SCENARIOS,
}


def _row(group, result):
    h = result["headline"]
    inv = result["financial"]["investor"]
    screen = result["screening"]
    return {
        "Group": group,
        "Scenario": result["scenario_name"],
        "Decision": screen["status"],
        "Failed gates": "; ".join(screen["failed_gate_names"]),
        "Route (km)": h["network_total_length_m"] / 1000.0,
        "Heat density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
        "Heat demand (GWh/yr)": h["annual_heat_demand_MWh"] / 1000.0,
        "Cooling demand (GWh/yr)": h["annual_cooling_demand_MWh"] / 1000.0,
        "Gross CAPEX (£m)": h["capex_total_GBP"] / 1e6,
        "Net CAPEX after grant (£m)": h["effective_capex_GBP"] / 1e6,
        "Annual OPEX (£m)": h["annual_total_opex_GBP"] / 1e6,
        "Investor NPV (£m)": inv["npv_GBP"] / 1e6,
        "Investor IRR (%)": None if inv["irr"] is None else inv["irr"] * 100.0,
        "Hurdle rate (%)": result["screening"]["hurdle_rate"] * 100.0,
        "Break-even heat tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
        "Carbon (gCO2e/kWh)": h["carbon_intensity_kgCO2_per_kWh_service"] * 1000.0,
        "Unmet heat (MWh/yr)": h["annual_unmet_demand_MWh"],
        "N-1 heat margin (MW)": h["n_minus_one_heat_margin_MW"],
    }


def _write_charts(results, comparison, output):
    colors = {"PASS": "#2ca02c", "CONDITIONAL PASS": "#ffbf00", "FAIL": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in comparison.iterrows():
        ax.scatter(row["Carbon (gCO2e/kWh)"], row["Investor NPV (£m)"],
                   s=max(40, row["Gross CAPEX (£m)"] * 9),
                   c=colors.get(row["Decision"], "#777777"), alpha=0.8,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(row["Scenario"].split(" — ")[0].split(" - ")[0],
                    (row["Carbon (gCO2e/kWh)"], row["Investor NPV (£m)"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.axvline(100, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Operational carbon (gCO2e/kWh service)")
    ax.set_ylabel("Investor NPV (£m)")
    ax.set_title("Internal screen: investor value versus operational carbon")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "npv_vs_carbon.png", dpi=180)
    plt.close(fig)

    focus_groups = ["Route/commercial feasibility", "Data-centre waste heat"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, focus in zip(axes, focus_groups):
        for group, result in results:
            if group != focus:
                continue
            inv = result["financial"]["investor"]
            short_name = result["scenario_name"].split(" - ")[0]
            ax.plot(inv["cashflow_years"], [v / 1e6 for v in inv["cumulative_discounted_GBP"]],
                    label=short_name, linewidth=2)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xlabel("Project year")
        ax.set_title(focus)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Cumulative discounted investor cash position (£m)")
    fig.suptitle("Lifetime discounted cash position", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "lifetime_discounted_cash_position.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(directory="output/internal_screening"):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for group, scenarios in SCENARIO_SETS.items():
        results.extend((group, run_scenario(scenario)) for scenario in scenarios)
    comparison = pd.DataFrame([_row(group, result) for group, result in results])
    comparison.to_csv(output / "internal_scenario_comparison.csv", index=False)

    gate_rows = []
    lifetime_rows = []
    for group, result in results:
        for gate in result["screening"]["gates"]:
            gate_rows.append({"Group": group, "Scenario": result["scenario_name"], **gate})
        inv = result["financial"]["investor"]
        for year, discounted, undiscounted in zip(
            inv["cashflow_years"], inv["cumulative_discounted_GBP"], inv["cumulative_undiscounted_GBP"]
        ):
            lifetime_rows.append({
                "Group": group, "Scenario": result["scenario_name"], "Year": year,
                "Cumulative discounted (£)": discounted,
                "Cumulative undiscounted (£)": undiscounted,
            })
    pd.DataFrame(gate_rows).to_csv(output / "screening_gate_audit.csv", index=False)
    pd.DataFrame(lifetime_rows).to_csv(output / "lifetime_cash_positions.csv", index=False)
    _write_charts(results, comparison, output)

    progressing = comparison[comparison["Decision"].isin(["PASS", "CONDITIONAL PASS"])]
    lines = [
        "# Internal district heating and cooling screening readout", "",
        "## Headline", "",
        f"{len(progressing)} of {len(comparison)} legacy illustrative cases pass the selected mandatory gates after customer-bill parity is applied. "
        "Use the technology-frontier report for the current source-mix, demand and route comparison.",
        "",
        "The current examples show that source technology alone does not make a scheme feasible. Customer bills are now held to the same gas/AC counterfactual, so a higher district tariff cannot create a pass.",
        "",
        "## Cases that pass the selected gates", "",
    ]
    for row in progressing.to_dict("records"):
        lines.append(
            f"- **{row['Scenario']}** — {row['Decision']}; NPV £{row['Investor NPV (£m)']:.2f}m, "
            f"IRR {row['Investor IRR (%)']:.1f}%, carbon {row['Carbon (gCO2e/kWh)']:.1f} gCO2e/kWh, "
            f"heat density {row['Heat density (MWh/m/yr)']:.2f} MWh/m/yr."
        )
    lines.extend([
        "", "## What changes feasibility", "",
        "- **Route and density:** shorter routes improve NPV, but density alone cannot overcome an adverse source-energy margin under customer-bill parity.",
        "- **Service completeness:** the data-centre-only stress case leaves material unmet heat and fails irrespective of its commercial assumptions.",
        "- **Funding and connections:** grant and contributions help but do not make the legacy compact data-centre hybrid pass at fair customer bills.",
        "- **Carbon:** the gas reference and several generic hybrids fail the carbon screen even when they serve demand.",
        "- **Resilience:** all illustrative cases currently have a negative peak-hour N-1 margin. N-1 is informational by default; if it is made mandatory, plant configuration/backup must change.",
        "- **Cooling:** the generic four-pipe example fails economics and carbon. Cooling should be added only around a concentrated anchor and tested on its own tariff, route and counterfactual.",
        "", "## How to use this internally", "",
        "Present the decision table first, then the NPV/carbon chart and lifetime cash-position chart. Use the comparison CSV to filter failed gates. Do not present `CONDITIONAL PASS` as investment approval; it means the option is worth developing and evidencing.",
        "", "## Important limitation", "",
        "The N-1 result is a peak-capacity screen, the physical operating year is repeated through the project life, and generic routes/demand archetypes remain uncertain. Detailed GIS/hydraulics, outage duration, construction phasing, finance/tax and independent reconciliation remain required before external circulation.",
    ])
    (output / "internal_screening_readout.md").write_text("\n".join(lines), encoding="utf-8")
    return output, comparison


if __name__ == "__main__":
    directory, table = write_outputs()
    print(table.to_string(index=False))
    print(f"\nWrote {directory}")
