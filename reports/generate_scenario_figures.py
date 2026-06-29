"""Generate screening figures from the worked scenarios.

Run from the repository root:
    python reports/generate_scenario_figures.py

Outputs are written to outputs/figures and outputs/tables. These figures are
for the current HEATING business case. A 4-pipe scenario can be included for
network CAPEX sizing, but cooling plant dispatch/economics are not yet in the
universal runner; do not present a 4-pipe case as a full cooling business case
until that work is added.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import WORKED_SCENARIOS
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

def _financial_value(result: dict, key: str):
    return result.get("financial", {}).get(key)

def build_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        h = result["headline"]
        rows.append({
            "scenario": result["scenario_name"],
            "capex_GBP": h["capex_total_GBP"],
            "annual_opex_GBP": h["annual_total_opex_GBP"],
            "lcoh_GBP_per_kWh": h["lcoh_GBP_per_kWh"],
            "carbon_tCO2_per_year": h["annual_carbon_tCO2"],
            "carbon_kgCO2_per_kWh": h["carbon_intensity_kgCO2_per_kWh"],
            "annual_unmet_MWh": h["annual_unmet_demand_MWh"],
            "peak_unmet_MW": h["peak_unmet_MW"],
            "annual_unmet_cooling_MWh": h.get("annual_unmet_cooling_MWh", 0.0),
            "annual_cooling_demand_MWh": h.get("annual_cooling_demand_MWh", 0.0),
            "system_type": h.get("system_type", "2_pipe_heating"),
            "npv_vs_counterfactual_GBP": _financial_value(result, "npv_vs_counterfactual_GBP"),
            "simple_payback_years": _financial_value(result, "simple_payback_years"),
            "discounted_payback_years": _financial_value(result, "discounted_payback_years"),
        })
    return pd.DataFrame(rows)

def _save(fig, filename: str):
    fig.tight_layout()
    fig.savefig(OUT_FIG / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_financials(summary: pd.DataFrame) -> None:
    labels = summary["scenario"].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(summary))
    ax.bar(x, summary["capex_GBP"] / 1e6, label="CAPEX")
    ax.bar(x, summary["annual_opex_GBP"] / 1e6, label="Annual OPEX")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("£ million")
    ax.set_title("Installed CAPEX and annual operating cost")
    ax.legend()
    _save(fig, "01_capex_and_annual_opex.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    values = summary["npv_vs_counterfactual_GBP"].fillna(0) / 1e6
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("£ million, discounted")
    ax.set_title("NPV versus selected counterfactual")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "02_npv_vs_counterfactual.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    life = 25
    values = summary["discounted_payback_years"].copy()
    display = values.fillna(life + 1)
    bars = ax.bar(labels, display)
    ax.axhline(life, linestyle="--", linewidth=1, label=f"{life}-year appraisal life")
    ax.set_ylabel("Years")
    ax.set_title("Discounted payback versus selected counterfactual")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    for bar, original in zip(bars, values):
        label = "No payback" if pd.isna(original) else f"{original:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    _save(fig, "03_discounted_payback.png")

def plot_carbon_and_service(summary: pd.DataFrame) -> None:
    labels = summary["scenario"].tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, summary["carbon_kgCO2_per_kWh"] * 1000)
    ax.set_ylabel("gCO₂e/kWh heat")
    ax.set_title("Operational carbon intensity")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "04_carbon_intensity.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, summary["annual_unmet_MWh"])
    ax.set_ylabel("MWh/year")
    ax.set_title("Annual unmet heating demand")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "05_annual_unmet_heat.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    total_unmet = summary["annual_unmet_MWh"] + summary["annual_unmet_cooling_MWh"]
    ax.bar(labels, total_unmet)
    ax.set_ylabel("MWh/year")
    ax.set_title("Annual unmet heating and cooling demand")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "05b_annual_unmet_energy_service.png")

def plot_monthly_demand(reference_result: dict) -> None:
    demand = reference_result["demand"]
    index = pd.DatetimeIndex(demand["datetime_index"])
    heat = pd.Series(demand["total_heat_kW"], index=index)
    cool = pd.Series(demand["total_cooling_kW"], index=index)
    monthly = pd.DataFrame({
        "Heating + DHW MWh": heat.groupby(heat.index.month).sum() / 1000,
        "Cooling MWh": cool.groupby(cool.index.month).sum() / 1000,
    })
    monthly.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(11, 6))
    monthly.plot(kind="bar", ax=ax)
    ax.set_ylabel("MWh/month")
    ax.set_title("Underlying district demand profile")
    ax.legend()
    _save(fig, "06_monthly_heating_and_cooling_demand.png")

def plot_heat_mix(results: list[dict]) -> None:
    source_names = sorted({name for r in results for name in (r["headline"]["annual_heat_by_source_MWh"] | r["headline"].get("annual_cooling_by_source_MWh", {}))})
    data = pd.DataFrame(
        [{name: (r["headline"]["annual_heat_by_source_MWh"].get(name, 0.0) + r["headline"].get("annual_cooling_by_source_MWh", {}).get(name, 0.0)) for name in source_names}
         for r in results],
        index=[r["scenario_name"] for r in results],
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("MWh/year")
    ax.set_title("Annual heating and cooling supplied by source")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, "07_annual_energy_service_mix.png")

def main() -> None:
    results = [run_scenario(s) for s in WORKED_SCENARIOS]
    summary = build_summary(results)
    summary.to_csv(OUT_TAB / "scenario_summary.csv", index=False)
    plot_financials(summary)
    plot_carbon_and_service(summary)
    plot_monthly_demand(results[0])
    plot_heat_mix(results)
    print(summary.to_string(index=False))
    print(f"\nSaved figures to: {OUT_FIG}")
    print(f"Saved summary table to: {OUT_TAB / 'scenario_summary.csv'}")

if __name__ == "__main__":
    main()
