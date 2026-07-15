# Changelog

## 2.7.1-streamlit-matrix-verified — 2026-07-14

- Ran all 21 selectable templates and all four comparison sets through the
  Streamlit application test runner with no UI exceptions.
- Corrected the source editor so a scenario that omits `n_units` inherits the
  selected preset's unit count instead of silently forcing one unit and
  changing availability and annual OPEX.
- Verified that every UI template result matches the direct scenario runner on
  demand, peak, network loss, CAPEX, OPEX, carbon, unmet service and investor
  NPV.

## 2.7.0-streamlit-verified — 2026-07-14

- Added end-to-end Streamlit application tests for the Ealing template,
  contaminated session-state reset and auto-size/run workflow.
- Changed template loading to clear all form widget state so stale CAPEX, OPEX,
  plant, storage or route values cannot overwrite a newly loaded scenario.
- Fixed auto-sizing metadata leaking into plant constructors and preserved
  scenario aggregate peak/load-shape calibration during sizing.
- Disabled auto-sizing while the scenario retains the exact Ealing validation
  name; users must rename it before changing the report's fixed design.
- Added visible Ealing report calibration gates for demand, peak, unmet heat,
  CAPEX and 40-year investor NPV.

## 2.6.0-flexible-demand-inputs — 2026-07-14

- Buildings can now run from either floor area/dwelling count or measured
  annual heat, cooling or DHW; measured annual heat alone no longer triggers a
  floor-area error.
- Added annual cooling and DHW override columns to the Streamlit building
  editor, with explicit per-service precedence and missing-service behaviour.
- Added validation and regression coverage for annual-energy-only Ealing inputs
  and mixed measured/archetype demand rows.

## 2.5.0-fair-comparison-frontier — 2026-07-14

- Added gas-counterfactual heat-bill parity and individual-AC cooling-bill
  parity so customer tariffs cannot be changed to manufacture feasibility.
- Corrected counterfactual gas standing charges from one per aggregate block to
  one per billed connection.
- Changed pumping OPEX from annual-average pricing to hourly load × hourly
  electricity price and added a zero-residual OPEX audit.
- Added eight representative source mixes across 160 demand/route cases, plus
  unsupported central cases and electricity/EfW heat-price sensitivities.
- Added a cooling cost decomposition showing demand, incremental network/CAPEX,
  OPEX, fair revenue and NPV impacts.

## 2.4.0-internal-screening — 2026-07-14

- Replaced conflicting UI feasibility rules with one auditable decision object.
- Made the investor hurdle, service tolerance, carbon limit, minimum NPV,
  optional N-1 test and optional tariff ceiling explicit scenario inputs.
- Added route heat density, loss share, CAPEX intensity, unmet percentages and
  peak-hour N-1 firm-capacity outputs.
- Expanded comparison exports with assumptions, investor economics and `N/A`
  preservation for unavailable counterfactual results.
- Added NPV/carbon and lifetime discounted cash-position comparison charts.
- Added selectable worked, feasibility-route and data-centre scenario sets.
- Added regression coverage for consistent gate thresholds and service status.

## 2.3.0-data-centre-feasibility — 2026-07-14

- Added data-centre-only, typical air-cooled hybrid, compact liquid-cooled
  hybrid and unsupported comparison scenarios.
- Added one-at-a-time sensitivities for route, source heat, source temperature,
  availability, waste-heat fee, grant, customer contribution and heat tariff.
- Added a 40-year cumulative discounted/undiscounted cash-flow export and chart.
- Added UK GHNF gate pre-checks while explicitly reserving customer detriment,
  social IRR, jurisdiction and additionality for the official application work.
- Exposed data-centre source temperature, availability, heat fee and recovery
  interface CAPEX in the application editor.
- Added five data-centre feasibility regression tests.

## 2.2.0-feasibility-scenarios — 2026-07-14

- Added three investor-facing, Ealing-calibrated route/commercial scenarios:
  dense public-appraisal core, compact private-hurdle cluster and extended-route
  rejection case.
- Added route-length / linear-heat-density sensitivity exports and an explicit
  viable-screen decision gate.
- Added a separate four-pipe cooling extension check so cooling is not assumed
  to improve the case without a suitable anchor load.
- Added grant, customer-contribution, required-tariff, service and carbon
  assumptions to the comparison outputs.
- Added all three feasibility scenarios to the application template selector.
- Expanded the automated regression suite from 16 to 19 passing tests.

## 2.0.0-screening — 2026-07-14

- Corrected total capacity semantics across UI, auto-sizing and components.
- Replaced flat NPV calculations with one auditable 40-year cash-flow engine.
- Added customer-specific heat/cooling tariffs, billed connection counts,
  connection phasing and probability.
- Added explicit project CAPEX, annual OPEX and technology REPEX lines.
- Integrated climate reference, pumping cost/carbon and data-centre booster
  source-energy constraints.
- Added discounted LCOH/LCO-service, required tariff, peak funding, service and
  carbon gates, GHNF output caps, scenario hashes and warnings.
- Replaced technology-only examples with a common-project option matrix.
- Added 12 regression tests covering the critical calculation invariants.

## 2.1.0-ealing-validation — 2026-07-14

- Added a report-calibrated Ealing Phase 1 scenario and validation export.
- Integrated the existing thermal-storage engine into the live runner and UI.
- Added peak-reserve and boiler-displacement storage strategies.
- Preserved preset unit counts when total capacity is overridden without an
  explicit unit count.
- Corrected IRR root selection for cash flows with scheduled REPEX.
- Added measured annual/peak demand calibration and published-load-duration
  shape controls.
- Added report/hydraulic overrides for network losses, parasitic electricity,
  carbon factors and network CAPEX.
- Added a strict regression invariant that adding a source cannot increase
  unmet demand for an otherwise unchanged case.
- Expanded the suite from 12 to 16 passing regression tests.
