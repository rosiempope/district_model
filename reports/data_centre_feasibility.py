"""Run data-centre waste-heat cases, policy pre-checks and sensitivities."""
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scenarios.data_centre_feasibility import (
    DATA_CENTRE_SCENARIOS,
    LIFETIME_COMPARISON_SCENARIOS,
    data_centre_case,
)
from scenarios.scenario_runner import run_scenario


def _connection_contributions(result):
    revenues = result["financial"]["investor"]["line_items"]["revenue"]
    return sum(
        sum(values) for name, values in revenues.items()
        if name.endswith("connection charge")
    )


def _summary(result):
    h = result["headline"]
    investor = result["financial"]["investor"]
    grant = result.get("grant") or {"grant_GBP": 0.0, "output_based_cap_GBP": None}
    viable = (
        h["service_compliant"] and h["carbon_compliant"]
        and investor["npv_GBP"] >= 0
    )
    return {
        "Scenario": result["scenario_name"],
        "Route (m)": h["network_total_length_m"],
        "Heat demand (GWh)": h["annual_heat_demand_MWh"] / 1_000,
        "Gross CAPEX (£m)": h["capex_total_GBP"] / 1e6,
        "Grant (£m)": grant["grant_GBP"] / 1e6,
        "Connection contributions (£m)": _connection_contributions(result) / 1e6,
        "Annual OPEX (£m)": h["annual_total_opex_GBP"] / 1e6,
        "Unmet heat (MWh)": h["annual_unmet_demand_MWh"],
        "Carbon (gCO2e/kWh)": h["carbon_intensity_kgCO2_per_kWh_service"] * 1_000,
        "NPV (£m)": investor["npv_GBP"] / 1e6,
        "IRR (%)": investor["irr"] * 100 if investor["irr"] is not None else None,
        "Required tariff (p/kWh)": investor["required_heat_tariff_p_per_kWh_for_zero_NPV"],
        "Service gate": "PASS" if h["service_compliant"] else "FAIL",
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Outcome": "VIABLE SCREEN" if viable else "DO NOT PROGRESS",
    }


def run_comparison():
    return pd.DataFrame([
        _summary(run_scenario(deepcopy(scenario)))
        for scenario in DATA_CENTRE_SCENARIOS
    ])


def run_policy_prechecks():
    rows = []
    for scenario in DATA_CENTRE_SCENARIOS:
        result = run_scenario(deepcopy(scenario))
        h = result["headline"]
        grant = result.get("grant")
        grant_cfg = scenario["economics"].get("ghnf_grant", {})
        output_cap_pass = (
            not grant or grant.get("output_based_cap_GBP") is None
            or grant["grant_GBP"] <= grant["output_based_cap_GBP"] + 1
        )
        rows.append({
            "Scenario": scenario["name"],
            "Urban demand >= 2 GWh/year": h["annual_heat_demand_MWh"] >= 2_000,
            "Carbon <= 100 gCO2e/kWh": h["carbon_compliant"],
            "Service screen": h["service_compliant"],
            "Requested grant < 50%": (
                not grant_cfg.get("enabled") or grant_cfg.get("rate", 0) < 0.50
            ),
            "4.5p/kWh grant cap": output_cap_pass,
            "Investor NPV >= 0": result["financial"]["investor"]["npv_GBP"] >= 0,
            "Customer detriment gate": "REQUIRES CUSTOMER CLASSIFICATION / GHNF WORKBOOK",
            "GHNF social IRR >= 3.5%": "REQUIRES GHNF APPLICATION MODEL",
            "Jurisdiction / additionality": "MANUAL EVIDENCE REQUIRED",
        })
    return pd.DataFrame(rows)


SENSITIVITY_VALUES = {
    "Total network route (m)": [800, 1_000, 1_200, 1_500, 1_800, 2_200, 2_500, 2_750, 3_000],
    "Waste-heat fee (£/MWh)": [0, 5, 10, 15, 20, 40, 55, 60],
    "Source temperature (C)": [20, 25, 30, 35, 40, 45],
    "Source availability (%)": [60, 65, 75, 85, 90, 95, 97, 99],
    "GHNF eligible-cost rate (%)": [0, 5, 10, 20, 34, 40, 45, 49],
    "Connection contribution (£/kW)": [0, 300, 400, 450, 500, 600, 800, 1_000, 1_200],
    "Customer heat tariff (p/kWh)": [6, 6.3, 7, 8, 9, 9.56, 10, 11, 12],
    "Recoverable source heat (MW)": [1, 1.5, 2, 2.5, 3, 3.6, 4.5, 5.5, 7],
}


def _sensitivity_scenario(variable, value):
    kwargs = {}
    if variable == "Total network route (m)":
        kwargs["route_m"] = value
    elif variable == "Waste-heat fee (£/MWh)":
        kwargs["waste_heat_fee_GBP_per_MWh"] = value
    elif variable == "Source temperature (C)":
        kwargs["source_temperature_C"] = value
    elif variable == "Source availability (%)":
        kwargs["source_availability"] = value / 100
    elif variable == "GHNF eligible-cost rate (%)":
        kwargs["grant_rate"] = value / 100
    elif variable == "Connection contribution (£/kW)":
        kwargs["connection_contribution_GBP_per_kW"] = value
    elif variable == "Customer heat tariff (p/kWh)":
        kwargs["heat_tariff_p_per_kWh"] = value
        kwargs["heat_tariff_mode"] = "manual"
    elif variable == "Recoverable source heat (MW)":
        kwargs["recoverable_heat_MW"] = value
        # Co-size the booster rather than charging for deliberately unused plant.
        kwargs["booster_capacity_MW"] = value * 1.6
    return data_centre_case(f"Sensitivity: {variable} = {value}", **kwargs)


def run_sensitivities(variables=None):
    rows = []
    selected = SENSITIVITY_VALUES if variables is None else {
        key: SENSITIVITY_VALUES[key] for key in variables
    }
    for variable, values in selected.items():
        for value in values:
            result = run_scenario(_sensitivity_scenario(variable, value))
            summary = _summary(result)
            rows.append({
                "Variable": variable,
                "Value": value,
                "NPV (£m)": summary["NPV (£m)"],
                "IRR (%)": summary["IRR (%)"],
                "Unmet heat (MWh)": summary["Unmet heat (MWh)"],
                "Carbon (gCO2e/kWh)": summary["Carbon (gCO2e/kWh)"],
                "Required tariff (p/kWh)": summary["Required tariff (p/kWh)"],
                "Outcome": summary["Outcome"],
            })
    return pd.DataFrame(rows)


def _markdown_table(frame):
    """Small dependency-free Markdown table for the generated report."""
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def run_lifetime_comparison():
    rows = []
    for scenario in LIFETIME_COMPARISON_SCENARIOS:
        result = run_scenario(deepcopy(scenario))
        investor = result["financial"]["investor"]
        for year, discounted, undiscounted in zip(
            investor["cashflow_years"],
            investor["cumulative_discounted_GBP"],
            investor["cumulative_undiscounted_GBP"],
        ):
            rows.append({
                "Scenario": result["scenario_name"],
                "Year": year,
                "Cumulative discounted cash flow (£m)": discounted / 1e6,
                "Cumulative undiscounted cash flow (£m)": undiscounted / 1e6,
            })
    return pd.DataFrame(rows)


def plot_lifetime(lifetime, path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for name, group in lifetime.groupby("Scenario", sort=False):
        ax.plot(
            group["Year"], group["Cumulative discounted cash flow (£m)"],
            linewidth=2.2, label=name,
        )
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("Cumulative discounted project cash flow", loc="left", weight="bold")
    ax.set_xlabel("Operating year")
    ax.set_ylabel("Cumulative discounted cash flow (£m)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs(directory="output/data_centre_feasibility"):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    comparison = run_comparison()
    policy = run_policy_prechecks()
    sensitivities = run_sensitivities()
    lifetime = run_lifetime_comparison()
    comparison.to_csv(output / "data_centre_scenario_comparison.csv", index=False)
    policy.to_csv(output / "uk_support_policy_prechecks.csv", index=False)
    sensitivities.to_csv(output / "data_centre_sensitivities.csv", index=False)
    lifetime.to_csv(output / "lifetime_cashflows.csv", index=False)
    plot_lifetime(lifetime, output / "lifetime_cashflow_comparison.png")

    best = comparison.loc[comparison["NPV (£m)"].idxmax()]
    lines = [
        "# Data-centre waste-heat feasibility study", "",
        "## Result", "",
        "A data centre should be treated as a steady low-carbon baseload source, not the sole source of heat. Independent peak and reserve capacity remains necessary because heat-source and booster outages do not coincide neatly with customer demand.",
        "",
        f"The strongest tested data-centre case is **{best['Scenario']}**, with NPV GBP{best['NPV (£m)']:.2f}m, {best['Unmet heat (MWh)']:.2f} MWh unmet heat and {best['Carbon (gCO2e/kWh)']:.1f} gCO2e/kWh. It is not investable at gas-bill-parity customer revenue under the tested assumptions.",
        "", "## Core comparison", "",
        _markdown_table(comparison),
        "", "## What moves feasibility", "",
        "The one-at-a-time sensitivity is centred on the compact liquid-cooled hybrid. It is not a probability forecast.",
        "",
        "- Short total route and a heat-recovery energy centre close to the data centre.",
        "- Higher source temperature, ideally from liquid cooling, because it raises booster COP.",
        "- A low or zero waste-heat fee that recognises avoided data-centre cooling cost.",
        "- Enough recoverable heat for baseload, but not oversized recovery/booster plant that is rarely used.",
        "- Long-term heat availability and offtake contracts, plus independent reserve capacity.",
        "- Grant and customer connection funding. The compact case does not remain investable when both are removed.",
        "", "### Tested breakpoints for the compact hybrid", "",
        "These are interpolated screening breakpoints, not procurement limits: total route about 2.6 km; source temperature about 24C; waste-heat fee about GBP55/MWh; connection contribution about GBP420/kW when the high-grant assumption is retained; and heat tariff about 6.26p/kWh. Source availability below roughly 62% breaches the carbon gate even though backup preserves heat service.",
        "",
        "The highest model NPV occurs with only about 1 MW of 40C recovered baseload heat, but that sits at approximately 99 gCO2e/kWh and therefore has almost no carbon-gate headroom. The selected 2 MW case sacrifices some NPV for a much stronger carbon result.",
        "", "## UK support and policy position (checked 14 July 2026)", "",
        "- [GHNF Round 12](https://www.gov.uk/government/publications/green-heat-network-fund-ghnf-guidance-on-how-to-apply) is open to public, private and third-sector applicants in England and Wales until 25 September 2026.",
        "- [GHNF Round 12 guidance](https://assets.publishing.service.gov.uk/media/6a2927b6f553ec1112221871/GHNF_Guidance_for_Applicants_R12.pdf) sets the 2 GWh urban demand, 100 gCO2e/kWh, customer detriment, 3.5% social IRR, <50% eligible-cost and 4.5p/kWh-over-15-years gates.",
        "- [National Wealth Fund](https://www.nationalwealthfund.org.uk/news-and-publications/news/national-wealth-fund-backs-hull-city-centre-heat-network/) lending can sit alongside GHNF and local contributions; Hull combined a GBP15m GHNF grant, GBP1.5m local funding and a GBP27m NWF loan.",
        "- [Heat-network zoning](https://www.gov.uk/government/consultations/proposals-for-heat-network-zoning-2023/outcome/heat-network-zoning-consultation-2023-summary-of-government-response) is intended to improve demand and waste-heat-source certainty, but project-specific rights and duties still need confirmation.",
        "- [Ofgem consumer-protection regulation](https://www.ofgem.gov.uk/blog/heat-networks-regulation-now-live) has applied since 27 January 2026; fair pricing, billing and reliability must be designed into the commercial case.",
        "- The [OPDC data-centre network](https://www.london.gov.uk/who-we-are/city-halls-partners/old-oak-and-park-royal-development-corporation-opdc/opdc-media-centre/opdc-press-releases/opdc-awarded-ps36m-keep-thousands-homes-warm-waste-heat-data-centres-uk-first) demonstrates the funding stack: GBP36m GHNF support for a phased 95 GWh scheme, with separate development support.",
        "", "## Important limitations", "",
        "The model repeats one operating year over the 40-year cash flow. It does not yet simulate a data-centre tenant ramp-up, an expiring heat-offtake contract, debt service, tax or annual grid-carbon trajectories. GHNF social IRR and customer-detriment calculations must be completed in the official application workbook; the CSV only pre-checks the gates this model can evidence.",
    ]
    (output / "data_centre_feasibility.md").write_text("\n".join(lines), encoding="utf-8")
    return output, comparison, policy, sensitivities, lifetime


if __name__ == "__main__":
    directory, comparison, policy, sensitivities, lifetime = write_outputs()
    print(comparison.to_string(index=False))
    print(f"\nWrote {directory}")
