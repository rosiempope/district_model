# Full Streamlit scenario verification

Version: 2.7.1-streamlit-matrix-verified  
Date: 14 July 2026

## Scope

Every selectable sidebar template was loaded, rendered and run through
Streamlit's application test runner. Each UI result was then compared with a
fresh direct run of the source template.

The comparison covered annual heat and cooling demand, peak heat generation,
network heat loss, CAPEX, annual OPEX, operational carbon, unmet heat and
cooling, and investor NPV. All 21 UI results matched their direct-run values.
Every heat balance closed and every investor NPV equalled the final cumulative
discounted cash position.

## Template results

`FAIL` below is the model's commercial/carbon/service screening outcome; it is
not a software-test failure. Several scenarios are deliberately adverse stress
tests.

| Scenario | Decision | Unmet heat MWh/yr | Unmet cooling MWh/yr | UI test |
|---|---|---:|---:|---|
| Ealing report validation - Phase 1 | FAIL | 0.00 | 0.00 | PASS |
| F1 - Dense town-centre core / public appraisal | FAIL | 0.00 | 0.00 | PASS |
| F2 - Compact anchor cluster / private hurdle | FAIL | 0.00 | 0.00 | PASS |
| F3 - Extended lower-density route / reject | FAIL | 0.49 | 0.00 | PASS |
| DC1 - Data-centre-only service stress test | FAIL | 314.19 | 0.00 | PASS - intentional stress case |
| DC2 - Typical air-cooled data centre plus reserve | FAIL | 0.00 | 0.00 | PASS |
| DC3 - Compact liquid-cooled baseload hybrid | FAIL | 0.00 | 0.00 | PASS |
| DC4 - Same compact hybrid without support | FAIL | 0.00 | 0.00 | PASS |
| A1 - Gas district reference | FAIL | 0.00 | 0.00 | PASS |
| A3 - ASHP plus gas peak/backup | FAIL | 0.15 | 0.00 | PASS |
| A4 - Data-centre heat plus booster and backup | FAIL | 0.28 | 0.00 | PASS |
| A5 - EfW export plus ASHP and backup | FAIL | 0.00 | 0.00 | PASS |
| A6 - Four-pipe heating and cooling | FAIL | 0.15 | 0.00 | PASS |
| Gas boiler reference - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| Electric boiler - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| ASHP only - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| ASHP plus gas backup - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| ASHP plus electric backup - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| EfW plus ASHP plus gas backup - demand x2, route 1 km | CONDITIONAL PASS | 0.00 | 0.00 | PASS |
| Data-centre heat plus booster plus gas - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |
| Four-pipe ASHP plus gas plus chiller - demand x2, route 1 km | FAIL | 0.00 | 0.00 | PASS |

Ealing's technical calibration passes. Its overall screening decision is FAIL
because the report-calibrated investor NPV and IRR are below the selected
commercial requirements, not because of unmet heat.

## Comparison controls

| Comparison set | Expected results | Actual results | UI exceptions | Invariants |
|---|---:|---:|---:|---|
| Worked technology options | 5 | 5 | 0 | PASS |
| Representative technology mixes | 8 | 8 | 0 | PASS |
| Heat-network feasibility routes | 3 | 3 | 0 | PASS |
| Data-centre waste-heat cases | 4 | 4 | 0 | PASS |

The comparison table and comparison charts rendered successfully after each
bulk run.

## Defects found during the full matrix

The full equivalence check found that four worked templates omitted an explicit
unit count. The Streamlit editor used one unit instead of the selected preset's
unit count, changing annual OPEX even though total MW capacity was unchanged.
Version 2.7.1 corrects this and includes a regression test.

## Assurance boundary

This verifies the supplied templates, principal UI state transitions,
auto-sizing, result rendering, energy balances and financial reconciliation.
It cannot prove that every possible future combination of user inputs is
correct, nor does it replace project-specific demand, route, tariff, plant-cost
and engineering validation before an investment decision.
