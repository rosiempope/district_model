"""One Excel workbook of every model output, for the Dalkia deck.

    python -m reports.excel_export

Reads the CSVs the study scripts have already written to output/, relabels them
for a non-technical audience, and assembles one workbook. It does NOT re-run the
model — run the studies first (see the README tab for the order), so the
workbook can never silently disagree with the findings it came from.

Tabs are ordered as the argument, not as the folder listing:

    1. READ ME          what each tab is, and what was run to produce it
    2. THE ARGUMENT     the four numbers the whole thing turns on
    3. Where money goes cost decomposition by how each line scales
    4. Cost of heat     the electricity break-even against 8.14p gas
    5. Best case        20 Exeter cases — the one that works, and why
    6. Birmingham       the real DESNZ report, run through the model
    7. Temperature      what 62/30 vs 70/40 is worth
    8. The alternative  gas boilers vs heat pumps, and the levy
    9. ASSUMPTIONS      every number that could be argued with, and its source

Plain English, deliberately
----------------------------
"Social NPV" is replaced everywhere by "Value to the country". "Counterfactual"
becomes "The alternative". Nobody outside this repo should have to learn the
model's vocabulary to read its output.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WORKBOOK = OUT / "dalkia_model_outputs.xlsx"

# Column renames applied across every tab. The model's vocabulary is precise and
# unreadable; a board paper needs the second thing more than the first.
RENAMES = {
    "Whole-system NPV @3.5% (£m)": "Value to the country, 40yr (£m)",
    "Investor NPV (£m)": "Commercial return to Dalkia, 40yr (£m)",
    "npv_vs_counterfactual_GBP": "Value to the country (£)",
    "Counterfactual": "The alternative that gets built instead",
    "Alternative CAPEX (£m)": "Cost of the alternative — upfront (£m)",
    "Alternative bill (£m/yr)": "Cost of the alternative — running (£m/yr)",
    "Incremental CAPEX (£m)": "Extra upfront cost vs the alternative (£m)",
    "Avoided cost (£m/yr)": "Running cost saved each year (£m/yr)",
    "Whole-system payback (yrs)": "Years to pay back, country view",
    "Req. tariff (p/kWh)": "Price Dalkia must charge to break even (p/kWh)",
    "Fair tariff (p/kWh)": "Price the customer can fairly be charged (p/kWh)",
    "required_tariff_p_per_kWh": "Price Dalkia must charge to break even (p/kWh)",
    "LHD (MWh/m/yr)": "Heat density (MWh per metre of pipe per year)",
    "Low-carbon heat (%)": "Share of heat from low-carbon plant (%)",
    "Carbon (g/kWh)": "Carbon (g CO2 per kWh)",
    "Carbon (gCO2e/kWh)": "Carbon (g CO2 per kWh)",
    "Delivered gate": "Hot water hot enough? (gate)",
    "GHNF (£m)": "Government grant (£m)",
    "size_independent_p_per_kWh": "Fixed cost per kWh regardless of scheme size (p/kWh)",
    "size_independent_per_connection_GBP": "Fixed cost per connection (£)",
    "scaling_basis": "How this cost scales",
    "pct_of_capex": "Share of upfront cost (%)",
    "pct_of_opex": "Share of running cost (%)",
}

# (sheet name, source csv, one-line description for the README tab)
TABS = [
    ("Where money goes", "cost_breakdown/capex_breakdown.csv",
     "Every upfront cost line, tagged by how it scales with scheme size."),
    ("Where money goes (running)", "cost_breakdown/opex_breakdown.csv",
     "Every running cost line, tagged by how it scales."),
    ("Fixed cost trap", "cost_breakdown/fixed_cost_exposure.csv",
     "The cost that does NOT move with scheme size. 5.93 p/kWh against a 7.33p cap."),
    ("Cost of heat vs electricity", "electricity_breakeven/electricity_sweep.csv",
     "Cost of a kWh of heat as electricity price moves, against 8.14p gas."),
    ("Break-even prices", "electricity_breakeven/breakeven_prices.csv",
     "The electricity price at which each option beats gas. Read this one first."),
    ("Best case (20 runs)", "exeter_best_case/best_case_matrix.csv",
     "Exeter, real branched network, 62/30, vs heat pumps. 2-pipe and 4-pipe, five plant mixes."),
    ("Birmingham — the 4 zones", "birmingham/izo_costs.csv",
     "DESNZ Birmingham zoning report, Feb 2025, as published."),
    ("Birmingham — our pipe costs", "birmingham/model_vs_report_pipe_cost.csv",
     "The report's real £/m against this model's cost curve. The curve is 1.5x low in a city centre."),
    ("Birmingham — model runs", "birmingham/central_izo_model_runs.csv",
     "The Birmingham Central anchor core run at the report's own costs."),
    ("Existing pipework", "birmingham/existing_pipework.csv",
     "What reusing existing pipe is worth. £1,750/m vs £3,750/m, from the report's own two cases."),
    ("Temperature", "temperature_sensitivity/temperature_sweep.csv",
     "Flow/return sweep. What 62/30 buys over 70/40, and where hot water fails."),
    ("Trunk size limit", "temperature_sensitivity/trunk_ceiling_by_delta_t.csv",
     "Biggest scheme one standard pipe can serve, by design delta-T."),
    ("The alternative", "counterfactual_and_levy/counterfactual_comparison.csv",
     "Gas boilers vs heat pumps as the thing that gets built instead. This flips the sign."),
    ("Heat pump grant (BUS)", "counterfactual_and_levy/bus_eligibility.csv",
     "Where the £7,500 grant lands. It caps at 45kW, so it does nothing for big buildings."),
    ("Green levy", "counterfactual_and_levy/levy_sensitivity.csv",
     "What happens if policy costs move off electricity. Not what you would expect."),
]


def _argument_tab() -> pd.DataFrame:
    """The four numbers the whole deck turns on."""
    return pd.DataFrame([
        {"Step": "1. The problem",
         "Finding": "A heat network never beats gas. Not at any electricity price.",
         "Number": "Gas heat costs 8.14p/kWh. Our best scheme costs 21.5p/kWh all-in — "
                   "and 20.5p even if electricity were FREE.",
         "So what": "Stop defending the scheme against gas. The gap is capital, not energy. "
                    "Note the running cost already beats gas (5.3-6.6p) — it is the pipe and "
                    "plant that does not."},
        {"Step": "2. The reframe",
         "Finding": "Gas is not the alternative. Heat pumps are.",
         "Number": "Birmingham Central: -£85.5m against gas boilers, +£67.0m against heat "
                   "pumps. Same scheme, same costs.",
         "So what": "Heat network zoning designates zones where networks are 'the lowest-cost "
                    "solution for DECARBONISING heat' — not where they beat a gas boiler. "
                    "Against the alternative that is actually legal, the sign flips."},
        {"Step": "3. The gap",
         "Finding": "It works for the country and never for the investor.",
         "Number": "Exeter best case: +£13.0m to the country, -£32.7m to Dalkia. "
                   "20 of 20 cases fail the commercial test.",
         "So what": "The £77m of heat pumps nobody buys is real money saved — and Dalkia "
                    "cannot bank a penny of it. The customer captures it. That gap is what "
                    "grant and zoning exist to close. This is infrastructure, not an investment."},
        {"Step": "4. The actions",
         "Finding": "Scale is the only lever that crossed zero. Cooling never worked.",
         "Number": "Heat density 2.85 -> 6.28 turned -£8.5m into +£13.0m. Every 4-pipe case "
                   "was worse than its 2-pipe twin (best: +£13.0m -> -£36.4m).",
         "So what": "Hunt for: >30 GWh/yr, heat density >6 MWh/m, a high-grade heat source "
                    "(EfW steam beat every heat-pump-only mix), grant, and NO cooling. "
                    "Miss any one and it goes negative."},
    ])


def _assumptions_tab() -> pd.DataFrame:
    from economics.CAPEX import BUS_GRANT_GBP, BUS_MAX_CAPACITY_KWTH, INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW
    from economics.tariffs import (
        OFGEM_ELECTRICITY_CAP_P_PER_KWH, OFGEM_GAS_CAP_P_PER_KWH, OFGEM_GAS_CAP_REVIEW_PERIOD,
    )
    from network.design_temperature_limits import (
        CP1_BEST_PRACTICE_VWART_C, CP1_MAX_FLOW_TEMP_NEW_SCHEME_C, CP1_MIN_PERMITTED_FLOW_TEMP_C,
    )
    from components.peak_demand_option import CARBON_INTENSITY
    from components.water_ground_source_hp import DEFAULT_CARNOT_FRACTION

    rows = [
        ("Gas price (retail)", f"{OFGEM_GAS_CAP_P_PER_KWH} p/kWh",
         f"Ofgem price cap, {OFGEM_GAS_CAP_REVIEW_PERIOD}", "Reset quarterly"),
        ("Electricity price (retail)", f"{OFGEM_ELECTRICITY_CAP_P_PER_KWH} p/kWh",
         f"Ofgem price cap, {OFGEM_GAS_CAP_REVIEW_PERIOD}", "Reset quarterly"),
        ("Gas boiler efficiency", "90%", "Seasonal, condensing",
         "Sets the 8.14p gas-heat figure everything is compared against"),
        ("Discount rate — Dalkia", "10.5%", "BEIS cited 9-12% for UK heat network investors",
         "NPV can flip sign across that range"),
        ("Discount rate — country", "3.5%", "HM Treasury Green Book social rate", ""),
        ("Project life", "40 years", "Model default", ""),
        ("Individual gas boiler cost", f"£{INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW['gas_boiler']:.0f}/kW",
         "UK installer market review, 2025/26", ""),
        ("Individual heat pump cost", f"£{INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW['individual_ashp']:.0f}/kW",
         "UK installer market review, 2025/26, BEFORE grant",
         "LOAD-BEARING: the whole 'value to the country' result rests on this. Sensitivity-test it."),
        ("Heat pump grant (BUS)", f"£{BUS_GRANT_GBP:.0f}, max {BUS_MAX_CAPACITY_KWTH:.0f} kW/installation",
         "Ofgem; DESNZ confirmed to at least March 2028",
         "The 45kW cap means it does nothing for large non-domestic buildings"),
        ("Gas carbon", f"{CARBON_INTENSITY['gas']} kg CO2/kWh", "DESNZ 2026 GHG factors, gross CV", ""),
        ("Electricity carbon", f"{CARBON_INTENSITY['electric']} kg CO2/kWh",
         "DESNZ 2026 GHG factors, consumption basis", "Fell ~26% in one year; volatile"),
        ("Max flow temperature", f"{CP1_MAX_FLOW_TEMP_NEW_SCHEME_C:.0f} °C", "CIBSE CP1 2020, new schemes", ""),
        ("Min flow temperature", f"{CP1_MIN_PERMITTED_FLOW_TEMP_C:.0f} °C", "CIBSE CP1 2020 permitted", ""),
        ("Best-practice return temp", f"<{CP1_BEST_PRACTICE_VWART_C:.0f} °C", "CIBSE CP1 2020 VWART", ""),
        ("Hot water floor — instant HIU", "55 °C delivered",
         "50°C outlet (CIBSE GN 2021, HSE 'low risk') + 5K heat exchanger", ""),
        ("Hot water floor — cylinder", "65 °C delivered", "60°C stored (HSG274) + 5K coil",
         "Forecloses every low-temperature option"),
        ("Pipe cost", "£1,158/m at DN100, curve ^0.426", "Fitted to SEAI National Heat Study 2023",
         "PROVEN 1.5x LOW for congested city-centre routes — see Birmingham tab"),
        ("WSHP/GSHP efficiency", f"{DEFAULT_CARNOT_FRACTION:.0%} of Carnot",
         "Conservative vs Drammen (COP 3.05, implies 69%)", ""),
        ("Weather", "London Heathrow, 2011-2025 representative year",
         "EPW, 8,760 hours", "Not Exeter or Birmingham weather — a known simplification"),
    ]
    return pd.DataFrame(rows, columns=[
        "Assumption", "Value", "Source", "Health warning",
    ])


def _readme_tab(built: list[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {"Tab": "THE ARGUMENT", "What it shows": "The four numbers the whole deck turns on.",
         "Produced by": "This file"},
        {"Tab": "ASSUMPTIONS", "What it shows":
         "Every number that could be argued with, its source, and its health warning.",
         "Produced by": "This file, read live from the model's own constants"},
    ]
    produced_by = {
        "cost_breakdown": "python -m reports.cost_breakdown",
        "electricity_breakeven": "python -m reports.electricity_breakeven",
        "exeter_best_case": "python -m analysis.exeter_best_case",
        "birmingham": "python -m reports.birmingham_study",
        "temperature_sensitivity": "python -m reports.temperature_sensitivity",
        "counterfactual_and_levy": "python -m reports.counterfactual_and_levy_study",
    }
    for sheet, src in built:
        desc = next((d for s, c, d in TABS if s == sheet), "")
        rows.append({"Tab": sheet, "What it shows": desc,
                     "Produced by": produced_by.get(src.split("/")[0], "")})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheets: list[tuple[str, pd.DataFrame]] = []
    built: list[tuple[str, str]] = []
    missing: list[str] = []

    for sheet, csv, _ in TABS:
        path = OUT / csv
        if not path.exists():
            missing.append(csv)
            continue
        df = pd.read_csv(path).rename(columns=RENAMES)
        sheets.append((sheet, df))
        built.append((sheet, csv))

    if missing:
        print("MISSING — these studies have not been run, so their tabs are absent:")
        for m in missing:
            print(f"  {m}")
        print("  (see the README tab for the command that produces each)\n")

    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as xl:
        _readme_tab(built).to_excel(xl, sheet_name="READ ME", index=False)
        _argument_tab().to_excel(xl, sheet_name="THE ARGUMENT", index=False)
        for sheet, df in sheets:
            df.to_excel(xl, sheet_name=sheet[:31], index=False)
        _assumptions_tab().to_excel(xl, sheet_name="ASSUMPTIONS", index=False)

        # Readable column widths and frozen headers — a board paper, not a dump.
        for ws in xl.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 12), 62)
            for cell in ws[1]:
                cell.font = Font(bold=True)

    print(f"Wrote {WORKBOOK}")
    print(f"  {len(sheets) + 3} tabs: READ ME, THE ARGUMENT, "
          f"{len(sheets)} data tabs, ASSUMPTIONS")


if __name__ == "__main__":
    main()
