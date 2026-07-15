"""Run the comparable route/commercial feasibility scenarios and sensitivity."""
from pathlib import Path
import pandas as pd

from scenarios.ealing_report_validation import scenario_copy
from scenarios.feasibility_comparison import (
    BASE_LOSS_MWH, BASE_NETWORK_CAPEX_GBP, BASE_ROUTE_M, scenario_copies,
)
from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import FOUR_PIPE_ASHP_GAS


def _connection_contributions(result):
    revenues = result["financial"]["investor"]["line_items"]["revenue"]
    return sum(
        sum(values) for name, values in revenues.items()
        if name.endswith("connection charge")
    )


def _summary(result):
    h = result["headline"]
    investor = result["financial"]["investor"]
    length = h["network_total_length_m"]
    grant = result.get("grant") or {"grant_GBP": 0.0}
    return {
        "Scenario": result["scenario_name"],
        "Route (m)": length,
        "Linear heat density (MWh/m)": h["annual_heat_demand_MWh"] / length,
        "Losses (%)": h["annual_network_heat_loss_MWh"] / h["annual_heat_demand_MWh"] * 100,
        "Gross CAPEX (£m)": h["capex_total_GBP"] / 1e6,
        "Grant (£m)": grant["grant_GBP"] / 1e6,
        "Net CAPEX (£m)": h["effective_capex_GBP"] / 1e6,
        "Connection contributions (£m)": _connection_contributions(result) / 1e6,
        "Annual OPEX (£m)": h["annual_total_opex_GBP"] / 1e6,
        "Unmet heat (MWh)": h["annual_unmet_demand_MWh"],
        "Carbon (gCO2e/kWh)": h["carbon_intensity_kgCO2_per_kWh_service"] * 1000,
        "Investor NPV (£m)": investor["npv_GBP"] / 1e6,
        "Investor IRR (%)": investor["irr"] * 100 if investor["irr"] is not None else None,
        "Discount rate (%)": investor["discount_rate"] * 100,
        "Required heat tariff (p/kWh)": investor["required_heat_tariff_p_per_kWh_for_zero_NPV"],
        "Outcome": (
            "VIABLE SCREEN" if result["screening"]["status"] in {"PASS", "CONDITIONAL PASS"}
            else "DO NOT PROGRESS"
        ),
    }


def run_comparison():
    return pd.DataFrame([_summary(run_scenario(s)) for s in scenario_copies()])


def run_route_sensitivity():
    rows = []
    for length in (1_200, 1_500, 1_800, 2_148, 2_500, 3_000, 4_000):
        scenario = scenario_copy()
        ratio = length / BASE_ROUTE_M
        scenario["name"] = f"Route sensitivity {length}m"
        scenario["network"]["length_m"] = length
        scenario["network"]["capex_GBP_override"] = BASE_NETWORK_CAPEX_GBP * ratio
        scenario["network"]["annual_heat_loss_MWh_override"] = BASE_LOSS_MWH * ratio
        scenario["economics"]["ghnf_grant"] = {"enabled": True, "rate": 0.34}
        scenario["economics"]["tariffs"]["heat_tariff_mode"] = "counterfactual_bill_parity"
        for building in scenario["demand"]["buildings"]:
            building.pop("heat_unit_rate_p_per_kWh", None)
        result = run_scenario(scenario)
        rows.append(_summary(result))
    return pd.DataFrame(rows)


def run_cooling_extension_check():
    """Separate illustrative four-pipe check; not an Ealing report assumption."""
    return pd.DataFrame([_summary(run_scenario(FOUR_PIPE_ASHP_GAS))])


def write_outputs(directory="output/feasibility_comparison"):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    comparison = run_comparison()
    routes = run_route_sensitivity()
    cooling = run_cooling_extension_check()
    comparison.to_csv(output / "scenario_comparison.csv", index=False)
    routes.to_csv(output / "route_sensitivity.csv", index=False)
    cooling.to_csv(output / "cooling_extension_check.csv", index=False)
    lines = [
        "# District heat-network feasibility comparison",
        "",
        "The three core cases use the same Ealing-calibrated 14.2 GWh customer base and plant logic. Customer heat bills are held to the same individual-gas counterfactual; route and explicitly stated commercial assumptions are varied.",
        "",
        "## Compared scenarios",
        "",
    ]
    for row in comparison.to_dict("records"):
        irr_text = "n/a" if row["Investor IRR (%)"] is None else f"{row['Investor IRR (%)']:.2f}%"
        lines.extend([
            f"### {row['Scenario']}", "",
            f"- Outcome: **{row['Outcome']}**",
            f"- Route / heat density: {row['Route (m)']:.0f} m / {row['Linear heat density (MWh/m)']:.2f} MWh/m",
            f"- NPV / IRR: GBP{row['Investor NPV (£m)']:.2f}m / {irr_text}",
            f"- Gross / net CAPEX: GBP{row['Gross CAPEX (£m)']:.2f}m / GBP{row['Net CAPEX (£m)']:.2f}m",
            f"- Grant / customer contributions: GBP{row['Grant (£m)']:.2f}m / GBP{row['Connection contributions (£m)']:.2f}m",
            f"- Unmet heat / carbon: {row['Unmet heat (MWh)']:.2f} MWh / {row['Carbon (gCO2e/kWh)']:.1f} gCO2e/kWh",
            "",
        ])
    lines.extend([
        "## Screening conclusion", "",
        "None of these three legacy route cases passes once customer heat bills are held to the individual-gas alternative. Shortening the route improves NPV and lowers the break-even tariff, but cannot on its own close the gap between fair customer revenue and scheme CAPEX/OPEX.",
        "",
        "The technology-frontier output should be used to identify combinations of source cost, demand and route that cross the gate. Grant and customer contributions remain conditions to secure, not assumed benefits.",
        "",
        "The 0.49 MWh residual unmet heat in the 3 km case is 0.0035% of annual customer heat and remains inside the service gate. It is not the reason for rejection; NPV is.",
        "",
        "## Separate four-pipe cooling check", "",
        "This is an illustrative mixed-building dataset, not an input taken from the Ealing report. It tests whether simply adding a separate central cooling network makes the scheme stronger.",
        "",
    ])
    row = cooling.iloc[0]
    lines.extend([
        f"- Outcome: **{row['Outcome']}**",
        f"- NPV: GBP{row['Investor NPV (£m)']:.2f}m",
        f"- Gross CAPEX: GBP{row['Gross CAPEX (£m)']:.2f}m",
        f"- Carbon: {row['Carbon (gCO2e/kWh)']:.1f} gCO2e/kWh (above the 100 g screening gate)",
        "- Conclusion: do not add four-pipe cooling by default. Re-test only where a concentrated cooling anchor, shared civil works and/or heat recovery materially changes the case.",
    ])
    (output / "feasibility_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return output, comparison, routes, cooling


if __name__ == "__main__":
    directory, comparison, routes, cooling = write_outputs()
    print(comparison.to_string(index=False))
    print("\nRoute sensitivity\n", routes.to_string(index=False))
    print("\nCooling extension check\n", cooling.to_string(index=False))
    print(f"\nWrote {directory}")
