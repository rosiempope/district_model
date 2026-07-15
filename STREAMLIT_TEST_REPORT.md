# Streamlit application verification - version 2.7.1

Date: 14 July 2026

## Finding

The 1,001.9 MWh/year unmet-heat screenshot is not reproduced by a clean
version 2.7 UI round trip. Loading the Ealing template and running it through
the actual Streamlit application produces zero unmet heat. The screenshot is
therefore a modified or stale UI/session result, not the fixed report
validation case. The exact mutation cannot be reconstructed without the input
JSON from that run.

Version 2.7.1 clears all form widget state when a template is loaded and displays
an explicit error if a result carrying the Ealing validation name fails any of
the published calibration checks.

## End-to-end Ealing UI result

| Check | UI result | Target | Status |
|---|---:|---:|---|
| End-customer heat | 14,161.2 MWh/year | 14,161.194 MWh/year | PASS |
| Peak heat to generate | 7.190 MW | 7.190 MW | PASS |
| Unmet heat | 0.0 MWh/year | 0.0 MWh/year | PASS |
| Total CAPEX | £21,635,191 | £21,635,190 | PASS |
| Annual OPEX | £1,355,468/year | calibrated model value | PASS |
| 40-year investor NPV | -£2,249,115 | -£2,249,115 | PASS |

## UI workflows exercised

1. Start application, load Ealing Phase 1, render every form section and run.
2. Deliberately contaminate route, CAPEX, source-capacity and storage widgets;
   load Ealing and confirm every value resets to the template before running.
3. Load the ASHP plus gas worked scenario, auto-size, rerender and run; confirm
   no application exception and zero unmet heat.
4. Start a live Streamlit server and confirm both its health endpoint and root
   application page respond successfully.

## Automated result

The combined engineering, finance, feasibility and Streamlit suite contains
35 tests. All 35 passed. Streamlit UI tests use its `AppTest` application
runner and therefore exercise widget/session-state round trips rather than
calling only the underlying scenario runner.

All 21 selectable templates and all four comparison controls were also run as
a complete matrix. See `STREAMLIT_FULL_UI_VERIFICATION.md`.
