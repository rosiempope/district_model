# Version 2.2 implementation summary

This update implements the calculation-integrity blockers and the core
screening-completeness recommendations from the July 2026 review.

## Material corrections

| Area | Version 2.0 behaviour |
|---|---|
| Capacity | `capacity_MW` is total installed capacity for every technology; per-unit capacity is derived once. |
| Cash flow | One annual table drives NPV, IRR, simple/discounted payback and both chart tracks. |
| OPEX | Electricity, gas, third-party heat, pumping, technical O&M and overheads are separate lines with isolated real changes. |
| Revenue | Heat and cooling tariffs are separate; billed connections, standing charges, connection years and probabilities are explicit. |
| REPEX | Technology replacement intervals and CAPEX fractions produce scheduled annual spend. |
| Climate | Every climate case is measured against a shared baseline climate reference. |
| Pumping | Heating/cooling pumping electricity contributes to cost and carbon. |
| Waste heat | Booster output is limited by hourly recoverable source heat and source outages; extracted source heat is costed. |
| Carbon | Correct kg/kWh units, visible 100 g/kWh gate and GHNF eligibility status. |
| Grant | One-time year-0 inflow, strictly below 50% intensity and capped at 4.5p/kWh over 15 years. |
| LCOH | Discounted project costs including OPEX/REPEX divided by discounted connected energy over 40 years. |
| Audit | Annual investor/whole-system CSVs, line-item tables, energy-balance residuals, version, timestamp, scenario hash and warnings. |

## Interface and scenario changes

- Added commercial CAPEX/OPEX/tariff inputs and customer connection phasing.
- Corrected auto-size capacity transfer, deterministic technology order,
  double-diversity error and network-loss allowance.
- Renamed the counterfactual tab to avoid claiming an incomplete social appraisal.
- Added required zero-NPV heat tariff, peak funding requirement, service/carbon
  gates and assurance warnings.
- Replaced technology-only examples with a common customer/route option matrix;
  the ASHP-only case is explicitly a service stress test.

## Verification

`python -m unittest discover -s tests -v` passes 19 tests covering capacity,
cash-flow reconciliation, carrier escalation, REPEX timing, grant cap, climate,
zero cooling, zero network, booster coupling, energy balance, carbon units and
worked-scenario execution.

The Ealing validation update adds a
direct Phase 1 reconciliation under `output/ealing_validation`. The calibrated
case reproduces zero unmet heat and the report's demand, peak, generation mix,
COP, parasitic electricity, CAPEX, NPV, IRR, payback and first-year carbon
within screening tolerance.

Version 2.2 adds the runnable dense-core, compact-private and extended-route
comparison under `output/feasibility_comparison`, plus a separate four-pipe
cooling check. Regression tests now lock the expected screening decisions and
the monotonic effect of route extension on NPV and required heat tariff.

## Release position

The model is materially safer and more auditable, but remains a screening tool.
The unresolved items in `MODEL_ASSURANCE.md`—particularly independent
reconciliation, project quotations, N-1 design, route/civils detail, long-term
physical trajectories, nominal/debt/tax modelling and customer detriment—must
be completed before investor circulation.
