# Model assurance and release limits

## Correctness controls implemented through version 2.7.1

- total/unit plant-capacity invariant;
- explicit hourly heat/cooling energy-balance residuals;
- NPV equals final cumulative discounted cash position;
- carrier-specific price escalation;
- scheduled replacement expenditure;
- zero-network cost/loss/pumping case;
- climate-demand monotonicity regression;
- floor-area and measured-annual-energy demand input regression;
- end-to-end Streamlit template-load, session-reset and auto-size regressions;
- source/booster energy and outage coupling;
- pumping cost/carbon inclusion;
- carbon-unit and GHNF grant-cap tests;
- service and carbon comparison gates;
- annual investor and whole-system audit exports.
- one shared screening decision across the UI, API result and CSV export;
- scenario-specific hurdle, service, carbon, NPV and optional tariff gates;
- peak-hour N-1 capacity margin, route heat density and loss/CAPEX intensities.
- per-connection gas standing charges in the customer counterfactual;
- gas-bill heat parity and individual-AC cooling-bill parity modes;
- hourly pumping-electricity pricing and an explicit OPEX reconciliation.

## Important remaining screening limitations

- Auto-sizing is still a transparent load-duration heuristic, not a constrained
  unit-combination optimiser. It includes a network-loss allowance and avoids
  double diversity. The N-1 output is a peak-capacity screen and does not prove
  outage duration, storage autonomy or network resilience.
- Generic-length network mode is an equivalent trunk. Tree mode still lacks
  GIS route surfaces, utility congestion, crossings and shared-trench four-pipe
  civils logic.
- Counterfactual CAPEX/O&M remains parametric rather than customer-by-customer
  contract data.
- Annual physical performance is repeated through the cash-flow horizon. A
  year-specific grid-carbon, climate, degradation and demand trajectory is not
  yet simulated.
- Construction-period CAPEX drawdown, debt/tax/accounting statements, bad debt,
  capacity charges and residual/decommissioning inputs are not complete.
- Customer detriment, nominal mode, monetised carbon/social benefits, tornado
  sensitivity and switching-value optimisation are not implemented.
- Technology and pipe cost presets require project-specific price-year updates,
  quotations and uncertainty ranges.

## Required before investor circulation

1. Replace every visible warning/default with evidence or an approved assumption.
2. Calibrate customer demand and route lengths against measured/GIS data.
3. Obtain utility, civils, energy-centre and customer-connection quotations.
4. Independently reconcile at least three cases in a separate spreadsheet.
5. Have a heat-network engineer review temperatures, hydraulics, losses,
   availability and resilience.
6. Have a project-finance modeller review tariffs, phasing, tax/funding, price
   curves, REPEX and discount-rate basis.
7. Freeze and sign the scenario hash, assumptions register and test output used
   in the investment paper.
