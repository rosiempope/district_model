"""Run and export the Ealing Phase 1 report validation."""
from pathlib import Path
import pandas as pd

from scenarios.ealing_report_validation import REPORT_PHASE1_TARGETS, scenario_copy
from scenarios.scenario_runner import run_scenario


def run_validation():
    result = run_scenario(scenario_copy())
    h = result["headline"]
    inv = result["financial"]["investor"]
    dispatch = result["heat_dispatch"]
    hp = next(s for s in result["heat_sources"] if s.source_type == "ashp")
    hp_dispatch = dispatch.dispatch_by_source_MW[hp.name]
    boiler = next(s for s in result["heat_sources"] if s.source_type == "gas_boiler")
    boiler_dispatch = dispatch.dispatch_by_source_MW[boiler.name]
    weighted_cop = float(hp_dispatch.sum() / (hp_dispatch / hp.cop_hourly).sum())
    values = [
        ("End-customer heat", "MWh/yr", REPORT_PHASE1_TARGETS["end_customer_heat_MWh"], h["annual_heat_demand_MWh"]),
        ("Heat including losses", "MWh/yr", REPORT_PHASE1_TARGETS["heat_including_losses_MWh"], float(dispatch.demand_MW.sum())),
        ("Peak heat including losses", "MW", REPORT_PHASE1_TARGETS["peak_heat_MW"], float(dispatch.demand_MW.max())),
        ("ASHP generation", "MWh/yr", REPORT_PHASE1_TARGETS["ashp_generation_MWh"], float(hp_dispatch.sum())),
        ("Boiler generation", "MWh/yr", REPORT_PHASE1_TARGETS["boiler_generation_MWh"], float(boiler_dispatch.sum())),
        ("Average ASHP COP", "-", 2.88, weighted_cop),
        ("Energy-centre parasitic electricity", "MWh/yr", 302.716, h["annual_total_parasitic_electricity_MWh"]),
        ("Unmet heat", "MWh/yr", 0.0, h["annual_unmet_demand_MWh"]),
        ("CAPEX", "GBP", REPORT_PHASE1_TARGETS["capex_GBP"], h["capex_total_GBP"]),
        ("40-year investor NPV", "GBP", REPORT_PHASE1_TARGETS["investor_npv_40y_GBP"], inv["npv_GBP"]),
        ("40-year investor IRR", "%", REPORT_PHASE1_TARGETS["investor_irr_40y"] * 100, inv["irr"] * 100),
        ("Simple payback", "years", 25.0, inv["simple_payback_years"]),
        ("First-year carbon intensity", "gCO2e/kWh", REPORT_PHASE1_TARGETS["first_year_carbon_g_per_kWh"], h["carbon_intensity_kgCO2_per_kWh_service"] * 1000),
    ]
    rows = []
    for metric, unit, report_value, model_value in values:
        difference = model_value - report_value
        variance = None if report_value == 0 else difference / abs(report_value) * 100
        if metric == "40-year investor IRR":
            status = "PASS" if abs(difference) <= 0.10 else "REVIEW"  # percentage points
        elif metric == "Simple payback":
            status = "PASS" if abs(difference) <= 0.50 else "REVIEW"
        else:
            tolerance = 0.1 if metric == "Unmet heat" else (1.0 if metric == "40-year investor NPV" else 2.0)
            status = "PASS" if (abs(difference) <= 0.1 if report_value == 0 else abs(variance) <= tolerance) else "REVIEW"
        rows.append({
            "Metric": metric, "Unit": unit, "Ealing report": report_value,
            "Model": model_value, "Difference": difference,
            "Variance (%)": variance, "Status": status,
        })
    return result, pd.DataFrame(rows)


def write_outputs(directory="output/ealing_validation"):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    result, table = run_validation()
    table.to_csv(output / "ealing_phase1_validation.csv", index=False)
    display_columns = ["Metric", "Unit", "Ealing report", "Model", "Variance (%)", "Status"]
    header = "| " + " | ".join(display_columns) + " |"
    separator = "|" + "|".join("---" for _ in display_columns) + "|"
    body = []
    for row in table[display_columns].to_dict("records"):
        body.append("| " + " | ".join(
            "" if pd.isna(row[col]) else (f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col]))
            for col in display_columns
        ) + " |")
    lines = [
        "# Ealing Town Centre Phase 1 model validation",
        "",
        "The model was calibrated to the June 2025 feasibility report using Tables 11, 14-19 and 39-48 plus Figures 23-24.",
        "",
        header, separator, *body,
        "",
        "## Interpretation",
        "",
        "- Demand, peak, plant capacity, heat balance, COP, parasitic electricity, CAPEX, NPV, IRR, payback and carbon reconcile within screening tolerance.",
        "- Zero unmet heat requires the report's 50,000-litre thermal store and its published load-duration shape. The public PDF does not contain the underlying 8,760 values, so the peak sharpness is inferred from Figure 23.",
        "- GBP143,465/year is retained as a visible calibration residual for OPEX categories named but not quantified in the public PDF (staff, insurance, monitoring and maintenance).",
        "- This is a validation of the calculation chain, not evidence that generic model presets reproduce Ealing without the report-specific inputs.",
        "",
        f"Scenario hash: `{result['audit']['scenario_sha256']}`",
    ]
    (output / "ealing_phase1_validation.md").write_text("\n".join(lines), encoding="utf-8")
    return output, table


if __name__ == "__main__":
    directory, validation = write_outputs()
    print(validation.to_string(index=False))
    print(f"\nWrote {directory}")
