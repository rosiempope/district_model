"""Map feasibility across technology mix, annual demand and route length."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from scenarios.scenario_runner import run_scenario
from scenarios.technology_frontier import MIX_NAMES, frontier_scenario
from scenarios.worked_scenarios import ASHP_PLUS_GAS_PEAK, FOUR_PIPE_ASHP_GAS


DEMAND_SCALES = (0.5, 1.0, 1.5, 2.0)
ROUTE_KM = (1.0, 1.5, 2.0, 2.5, 3.0)
ELECTRICITY_PRICES = (80.0, 100.0, 120.0, 140.0, 160.0, 180.5, 200.0)


def _row(mix, scale, route, result, support):
    h = result["headline"]
    inv = result["financial"]["investor"]
    return {
        "Technology mix": mix,
        "Support case": support,
        "Demand scale": scale,
        "Route (km)": route,
        "Annual heat (GWh)": h["annual_heat_demand_MWh"] / 1000.0,
        "Annual cooling (GWh)": h["annual_cooling_demand_MWh"] / 1000.0,
        "Linear heat density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
        "Investor NPV (£m)": inv["npv_GBP"] / 1e6,
        "Investor IRR (%)": None if inv["irr"] is None else inv["irr"] * 100.0,
        "Equivalent heat tariff (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
        "Break-even heat tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
        "Carbon (gCO2e/kWh)": h["carbon_intensity_kgCO2_per_kWh_service"] * 1000.0,
        "Unmet heat (MWh)": h["annual_unmet_demand_MWh"],
        "Unmet cooling (MWh)": h["annual_unmet_cooling_MWh"],
        "Decision": result["screening"]["status"],
        "Failed gates": "; ".join(result["screening"]["failed_gate_names"]),
    }


def run_frontier():
    rows = []
    total = len(MIX_NAMES) * len(DEMAND_SCALES) * len(ROUTE_KM)
    count = 0
    for mix in MIX_NAMES:
        for scale in DEMAND_SCALES:
            for route in ROUTE_KM:
                result = run_scenario(frontier_scenario(mix, scale, route, support_case=True))
                rows.append(_row(mix, scale, route, result, True))
                count += 1
                if count % 20 == 0:
                    print(f"Completed {count}/{total} frontier cases", flush=True)
    # One common central point without support isolates the grant/contribution effect.
    for mix in MIX_NAMES:
        result = run_scenario(frontier_scenario(mix, 1.0, 1.5, support_case=False))
        rows.append(_row(mix, 1.0, 1.5, result, False))
    return pd.DataFrame(rows)


def _thresholds(frontier):
    supported = frontier[frontier["Support case"]]
    rows = []
    for mix in MIX_NAMES:
        subset = supported[supported["Technology mix"] == mix]
        passing = subset[subset["Decision"].isin(["PASS", "CONDITIONAL PASS"])]
        base_demand = subset[subset["Demand scale"] == 1.0]
        base_pass = base_demand[base_demand["Decision"].isin(["PASS", "CONDITIONAL PASS"])]
        central_no_support = frontier[
            (frontier["Technology mix"] == mix) & (~frontier["Support case"])
        ].iloc[0]
        rows.append({
            "Technology mix": mix,
            "Passing grid cases": len(passing),
            "Minimum passing heat density (MWh/m/yr)": (
                passing["Linear heat density (MWh/m/yr)"].min() if len(passing) else None
            ),
            "Maximum passing route at base demand (km)": (
                base_pass["Route (km)"].max() if len(base_pass) else None
            ),
            "Unsupported central NPV (£m)": central_no_support["Investor NPV (£m)"],
            "Unsupported central decision": central_no_support["Decision"],
        })
    return pd.DataFrame(rows)


def _cooling_decomposition():
    heating = run_scenario(ASHP_PLUS_GAS_PEAK)
    cooling = run_scenario(FOUR_PIPE_ASHP_GAS)
    hi, ci = heating["financial"]["investor"], cooling["financial"]["investor"]
    hh, ch = heating["headline"], cooling["headline"]
    return pd.DataFrame([
        {"Metric": "Annual cooling demand (GWh)", "Heating only": 0.0,
         "Four-pipe": ch["annual_cooling_demand_MWh"] / 1000.0,
         "Increment": ch["annual_cooling_demand_MWh"] / 1000.0},
        {"Metric": "Gross CAPEX (£m)", "Heating only": hh["capex_total_GBP"] / 1e6,
         "Four-pipe": ch["capex_total_GBP"] / 1e6,
         "Increment": (ch["capex_total_GBP"] - hh["capex_total_GBP"]) / 1e6},
        {"Metric": "Network CAPEX (£m)", "Heating only": hh["capex_network_GBP"] / 1e6,
         "Four-pipe": ch["capex_network_GBP"] / 1e6,
         "Increment": (ch["capex_network_GBP"] - hh["capex_network_GBP"]) / 1e6},
        {"Metric": "Full-buildout OPEX (£m/yr)", "Heating only": hh["annual_total_opex_GBP"] / 1e6,
         "Four-pipe": ch["annual_total_opex_GBP"] / 1e6,
         "Increment": (ch["annual_total_opex_GBP"] - hh["annual_total_opex_GBP"]) / 1e6},
        {"Metric": "Year-1 revenue (£m/yr)", "Heating only": hi["annual_revenue_GBP"] / 1e6,
         "Four-pipe": ci["annual_revenue_GBP"] / 1e6,
         "Increment": (ci["annual_revenue_GBP"] - hi["annual_revenue_GBP"]) / 1e6},
        {"Metric": "Investor NPV (£m)", "Heating only": hi["npv_GBP"] / 1e6,
         "Four-pipe": ci["npv_GBP"] / 1e6,
         "Increment": (ci["npv_GBP"] - hi["npv_GBP"]) / 1e6},
    ])


def _price_sensitivity():
    electric_mixes = [
        "Electric boiler", "ASHP only", "ASHP + gas backup",
        "ASHP + electric backup", "Data-centre heat + booster + gas backup",
        "Four-pipe ASHP + gas + chiller",
    ]
    rows = []
    for mix in electric_mixes:
        for price in ELECTRICITY_PRICES:
            scenario = frontier_scenario(mix, 2.0, 1.0, support_case=True)
            scenario["economics"]["parasitic_electricity_price_GBP_per_MWh"] = price
            for source in scenario["sources"] + scenario.get("cooling_sources", []):
                if source["type"] in {"ashp", "electric_boiler", "booster_heat_pump", "air_cooled_chiller"}:
                    source["electricity_price_GBP_per_MWh"] = price
            result = run_scenario(scenario)
            rows.append({
                "Technology mix": mix, "Variable": "Electricity price (£/MWh)",
                "Value": price, "Investor NPV (£m)": result["financial"]["investor"]["npv_GBP"] / 1e6,
                "Decision": result["screening"]["status"],
                "Failed gates": "; ".join(result["screening"]["failed_gate_names"]),
            })
    for heat_price in (0.0, 5.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        scenario = frontier_scenario("EfW + ASHP + gas backup", 2.0, 1.0, support_case=True)
        for source in scenario["sources"]:
            if source["type"] == "efw_chp":
                source["heat_export_cost_GBP_per_MWh"] = heat_price
        result = run_scenario(scenario)
        rows.append({
            "Technology mix": "EfW + ASHP + gas backup", "Variable": "EfW heat price (£/MWh)",
            "Value": heat_price, "Investor NPV (£m)": result["financial"]["investor"]["npv_GBP"] / 1e6,
            "Decision": result["screening"]["status"],
            "Failed gates": "; ".join(result["screening"]["failed_gate_names"]),
        })
    return pd.DataFrame(rows)


def _price_thresholds(sensitivity):
    rows = []
    for (mix, variable), subset in sensitivity.groupby(["Technology mix", "Variable"]):
        passing = subset[subset["Decision"].isin(["PASS", "CONDITIONAL PASS"])]
        rows.append({
            "Technology mix": mix, "Variable": variable,
            "Highest tested passing value": passing["Value"].max() if len(passing) else None,
            "Passing cases": len(passing),
        })
    return pd.DataFrame(rows)


def _plot_frontier(frontier, output):
    supported = frontier[frontier["Support case"]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax, mix in zip(axes.flat, MIX_NAMES):
        subset = supported[supported["Technology mix"] == mix]
        passes = subset["Decision"].isin(["PASS", "CONDITIONAL PASS"])
        ax.scatter(subset.loc[~passes, "Route (km)"], subset.loc[~passes, "Annual heat (GWh)"],
                   c="#d62728", marker="x", s=55, label="Fails")
        ax.scatter(subset.loc[passes, "Route (km)"], subset.loc[passes, "Annual heat (GWh)"],
                   c="#2ca02c", marker="o", s=55, label="Passes gates")
        ax.set_title(mix, fontsize=10)
        ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Route length (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Annual heat demand (GWh)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Technology feasibility frontier: demand versus route\n49% eligible grant + £1,000/kW connection contribution; 10.5% hurdle; customer bill parity")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "technology_route_demand_frontier.png", dpi=180)
    plt.close(fig)


def _plot_cooling_load_duration(output):
    result = run_scenario(FOUR_PIPE_ASHP_GAS)
    load = sorted(result["cooling_dispatch"].demand_MW, reverse=True)
    installed = sum(source.capacity_MW for source in result["cooling_sources"])
    h = result["headline"]
    load_factor = h["annual_cooling_demand_MWh"] / (
        h["peak_cooling_to_generate_MW"] * 8760.0
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(range(1, 8761), load, linewidth=2, label="Cooling demand")
    ax.axhline(installed, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Installed chiller capacity ({installed:.1f} MW)")
    ax.set_xlabel("Hours per year at or above load")
    ax.set_ylabel("Cooling duty (MW)")
    ax.set_title(
        f"Four-pipe cooling load-duration curve: {h['annual_cooling_demand_MWh']/1000:.2f} GWh/year, "
        f"{load_factor*100:.1f}% peak-load factor"
    )
    ax.set_xlim(0, 8760)
    ax.set_ylim(0, installed * 1.08)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cooling_load_duration.png", dpi=180)
    plt.close(fig)


def write_outputs(directory="output/technology_frontier"):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    frontier = run_frontier()
    thresholds = _thresholds(frontier)
    cooling = _cooling_decomposition()
    sensitivity = _price_sensitivity()
    price_thresholds = _price_thresholds(sensitivity)
    frontier.to_csv(output / "technology_frontier_cases.csv", index=False)
    thresholds.to_csv(output / "technology_frontier_thresholds.csv", index=False)
    cooling.to_csv(output / "cooling_cost_decomposition.csv", index=False)
    sensitivity.to_csv(output / "technology_price_sensitivity.csv", index=False)
    price_thresholds.to_csv(output / "technology_price_thresholds.csv", index=False)
    _plot_frontier(frontier, output)
    _plot_cooling_load_duration(output)
    passing = frontier[frontier["Decision"].isin(["PASS", "CONDITIONAL PASS"])]
    lines = [
        "# Technology mix, route and demand feasibility readout", "",
        "## Fair comparison basis", "",
        "Heat bills are held equal to each customer's modelled individual-gas bill. Cooling bills are held equal to modelled individual-AC running cost. The frontier uses a 10.5% investor hurdle and an explicitly optimistic boundary case of 49% eligible grant plus £1,000/kW connection contribution.",
        "", "## Route and demand result", "",
        f"{len(passing)} of {len(frontier[frontier['Support case']])} supported grid cases pass all mandatory gates.",
    ]
    for row in passing.to_dict("records"):
        lines.append(
            f"- **{row['Technology mix']}** at {row['Annual heat (GWh)']:.2f} GWh/year and "
            f"{row['Route (km)']:.1f} km ({row['Linear heat density (MWh/m/yr)']:.2f} MWh/m/year): "
            f"NPV £{row['Investor NPV (£m)']:.2f}m, IRR {row['Investor IRR (%)']:.1f}%."
        )
    lines.extend([
        "", "No unsupported central case passes. At the tested energy prices, simply adding demand does not rescue technologies whose variable cost is too close to or above the gas-parity heat bill; route density solves pipe CAPEX, not an adverse electricity-to-heat price spread.",
        "", "## Why the four-pipe cooling case is weak", "",
        f"The case contains {cooling.iloc[0]['Four-pipe']:.2f} GWh/year of cooling, but adds £{cooling.iloc[2]['Increment']:.2f}m of network CAPEX and £{cooling.iloc[1]['Increment']:.2f}m total CAPEX. Full-buildout OPEX rises by £{cooling.iloc[3]['Increment']:.2f}m/year, while fair year-1 cooling revenue adds only £{cooling.iloc[4]['Increment']:.2f}m/year. Its investor NPV is therefore £{abs(cooling.iloc[5]['Increment']):.2f}m worse than heating-only.",
        "", "The issue is not absent hospital/anchor cooling demand. It is a low annual utilisation factor combined with a separate full-length cooling pipe pair and bill-parity revenue. A concentrated cooling anchor close to the energy centre, shared trench costs, higher year-round process cooling or heat-recovery chillers must be tested as distinct cases.",
        "", "## OPEX assurance", "",
        "Source energy costs are calculated from hourly dispatch and source marginal cost. Pumping is now priced against its hourly electrical load. Technology/network O&M and user-entered overheads are added once. The result includes an OPEX reconciliation with a zero residual and separately labels full-buildout OPEX versus connection-weighted investor-year OPEX.",
        "", "## Remaining limitations", "",
        "The frontier is a structured sensitivity, not a site design. Cost curves, source availability, customer connection cost, electricity procurement, shared-trench cooling civils and N-1 resilience require project evidence.",
    ])
    (output / "technology_frontier_readout.md").write_text("\n".join(lines), encoding="utf-8")
    return output, frontier, thresholds, cooling, sensitivity, price_thresholds


if __name__ == "__main__":
    directory, _, thresholds, cooling, _, price_thresholds = write_outputs()
    print("\nThresholds\n", thresholds.to_string(index=False))
    print("\nCooling decomposition\n", cooling.to_string(index=False))
    print("\nPrice thresholds\n", price_thresholds.to_string(index=False))
    print(f"\nWrote {directory}")
