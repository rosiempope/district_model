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
- hourly pumping-electricity pricing and an explicit OPEX reconciliation;
- design/commissioning/contingency applied to the whole delivered scope;
- fixed CAPEX/OPEX scaled to scheme peak capacity, with the factor recorded.

## Test coverage — what is and is not covered

142 tests, all passing (`python -m unittest discover -s tests`).

| Layer | Files | Coverage |
|---|---|---|
| Integration / regression | `test_regressions.py`, `test_feasibility_comparison.py`, `test_data_centre_feasibility.py`, `test_streamlit_app.py` | Scenario runner end-to-end, finance identities, gates, grant caps, UI round trip |
| Physics units | `test_pipe_catalog.py`, `test_demand_synthesis.py`, `test_cop_curves.py` | Hydraulics, pipe sizing, cost curve, EN 253 heat loss, demand synthesis, all three COP models |

**Still without unit coverage** — exercised only indirectly through `run_scenario()`:

- `optimisation/dispatch.py` — merit order, storage tiering, unmet-demand accounting;
- `network/topology_thermal.py` — the Shukhov formula, delivered-temperature and
  minimum-flow-temperature checks (integration-tested via network loss only);
- `network/network_pumping.py`;
- `components/thermal_storage.py`;
- `economics/tariffs.py` — the price shapes;
- `optimisation/auto_size.py` — the sizing heuristic.

Until version 2.7.1 there were **no** physics unit tests at all: every test ran
through `run_scenario()`, and 18 modules ended with a `__main__` block directing
you to test files that had never been written. Writing the first three
immediately surfaced two real defects (below). Assume the untested modules above
carry comparable risk until they are covered.

## Defects found and fixed since the last release

- **NaN cooling demand.** `_cooling_profile()` guarded on the scenario year's
  cooling degree-hours but divided by the *reference* year's. A baseline weather
  year that never reached `cool_base_C` produced a divide-by-zero, and the
  resulting NaN propagated silently through cooling demand, dispatch, OPEX and
  NPV rather than failing. Not triggered by the shipped London EPW, which has
  cooling degree-hours; a cooler-climate EPW would have hit it.
- **Contingency base.** Design (10%), commissioning (3%) and contingency (20%)
  were applied to plant and network only, exempting ~£9.4m of a ~£21m scheme —
  the energy centre, utility connections, controls, customer connections and
  metering. Now applied to the whole delivered scope except land.
- **Unreachable fixed-cost scaling.** `scaled_economics()` existed but was buried
  in a study script, so the archetype study ran on unscaled Ealing overheads —
  the exact caveat its own findings recorded.
- **Parallel financial stack.** `economics/metrics.py` carried a second, unused
  implementation of NPV/IRR/payback/LCOH on a 25-year flat-annuity basis,
  contradicting the live 40-year table. Removed.

## Important remaining screening limitations

- **The cooling model overshoots its own benchmark by ~9-10%.** Part 3's comfort
  floor is applied as `np.maximum()` on top of an already fully-allocated budget,
  so annual cooling lands above the CIBSE-style target rather than on it. The
  docstring claimed it summed "EXACTLY"; that claim was false and is corrected.
  Conservative in direction (overstates cooling), and far better than the ~47%
  over-allocation it replaced, but it is an overshoot, not an identity.
- Auto-sizing is still a transparent load-duration heuristic, not a constrained
  unit-combination optimiser. It includes a network-loss allowance and avoids
  double diversity. The N-1 output is a peak-capacity screen and does not prove
  outage duration, storage autonomy or network resilience.
- Generic-length network mode is an equivalent trunk. Its pumping model applies
  the design-point pressure gradient at every hour, so part-load pumping is
  overstated and peak understated (real pumping power scales with flow cubed).
  Tree mode still lacks GIS route surfaces, utility congestion, crossings and
  shared-trench four-pipe civils logic.
- Counterfactual CAPEX/O&M remains parametric rather than customer-by-customer
  contract data.
- Annual physical performance is repeated through the cash-flow horizon. A
  year-specific grid-carbon, climate, degradation and demand trajectory is not
  yet simulated. Given DESNZ's 2026 grid factor moved ~26% in a single update,
  this materially understates the carbon case for electrified options over 40
  years.
- Peak plant and pipe diameters are sized against the representative weather
  year. `profiles/TMY_weather_single_rep_year.py` states that a CIBSE Design
  Summer Year / Design Winter Year should be used for design sizing; that is not
  yet implemented.
- Fixed CAPEX/OPEX scaling is a ratio against peak thermal capacity with a 0.20
  floor. It is a screening approximation — a real project scopes these items from
  a drawing.
- Replacement expenditure covers generating plant only. Controls/SCADA carries no
  replacement across the 40-year horizon despite a shorter real service life.
- `billing_and_customer_service_GBP` is a flat annual figure and does not scale
  with connection count, which it plainly should.
- Construction-period CAPEX drawdown, debt/tax/accounting statements, bad debt,
  capacity charges and residual/decommissioning inputs are not complete.
- Customer detriment, nominal mode, monetised carbon/social benefits, tornado
  sensitivity and switching-value optimisation are not implemented.
- Technology and pipe cost presets require project-specific price-year updates,
  quotations and uncertainty ranges.

## Verification record

**Ealing Town Centre Phase 1** (`python -m reports.ealing_validation`) — 13 of 13
metrics PASS against the June 2025 SEL feasibility report: demand, peak, plant
capacity, heat balance, COP, parasitic electricity, CAPEX, NPV, IRR, payback and
carbon all reconcile within screening tolerance. Two caveats stand: zero unmet
heat requires the report's 50,000-litre store and its published load-duration
shape (the public PDF has no 8,760 values, so peak sharpness is inferred from
Figure 23), and £143,465/year is retained as a visible calibration residual for
OPEX categories the PDF names but does not quantify. This validates the
calculation chain; it is not evidence that generic presets reproduce Ealing
without the report-specific inputs.

The validation is structurally insulated from the contingency-base correction:
that scenario sets all three adder percentages to zero and uses the report's own
explicit CAPEX lines.

**Streamlit UI** (version 2.7.1-streamlit-matrix-verified, 14 July 2026) — every
selectable sidebar template was loaded, rendered and run through Streamlit's
application test runner, and each UI result compared against a fresh direct run
of the source template. The comparison covered annual heat and cooling demand,
peak heat generation, network heat loss, CAPEX, annual OPEX, operational carbon,
unmet heat and cooling, and investor NPV. All 21 UI results matched their
direct-run values. Version 2.7.1 clears all form widget state on template load.

A previously circulated screenshot showing 1,001.9 MWh/year unmet heat is **not**
reproduced by a clean round trip: loading the Ealing template through the
application produces zero unmet heat. That screenshot is a modified or stale
session result, not the fixed report validation case; the exact mutation cannot
be reconstructed without the input JSON from that run.

## Required before investor circulation

1. Replace every visible warning/default with evidence or an approved assumption.
2. Calibrate customer demand and route lengths against measured/GIS data.
3. Obtain utility, civils, energy-centre and customer-connection quotations.
4. Independently reconcile at least three cases in a separate spreadsheet.
5. Have a heat-network engineer review temperatures, hydraulics, losses,
   availability and resilience.
6. Have a project-finance modeller review tariffs, phasing, tax/funding, price
   curves, REPEX and discount-rate basis.
7. Extend unit coverage to dispatch, topology thermal and auto-sizing before
   relying on any single scenario's absolute numbers.
8. Freeze and sign the scenario hash, assumptions register and test output used
   in the investment paper.
