# Model assurance and release limits

## Correctness controls implemented in version 2.0

- total/unit plant-capacity invariant;
- explicit hourly heat/cooling energy-balance residuals;
- NPV equals final cumulative discounted cash position;
- carrier-specific price escalation;
- scheduled replacement expenditure;
- zero-network cost/loss/pumping case;
- climate-demand monotonicity regression;
- source/booster energy and outage coupling;
- pumping cost/carbon inclusion;
- carbon-unit and GHNF grant-cap tests;
- service and carbon comparison gates;
- annual investor and whole-system audit exports.

## Important remaining screening limitations

- Auto-sizing is still a transparent load-duration heuristic, not a constrained
  unit-combination optimiser. It includes a network-loss allowance and avoids
  double diversity, but does not prove N-1 resilience.
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
