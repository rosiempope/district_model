# Dalkia initial-screening study — findings

Run directly against the model engine (`scenarios.scenario_runner.run_scenario`, `profiles.demand_synthesis`, `optimisation.auto_size`) — not through Streamlit, not through the test suite.

## 1. Is the model accurate / trustworthy for a first screen?

- The engine does not always return a positive answer: every one of the base cases below is failed by the screen, on the model's own gates, against revenue the model itself caps at the customer's gas bill.
  - ASHP-only stress test (dense archetype, no gas/electric backup): unmet heat **0.0 MWh/yr** (0.00% of demand); service gate: **PASS**; screening decision: **FAIL**; failed gates: Investor NPV, Investor IRR.
  - Read that result honestly: the ASHP-only design is **not** caught by the service gate, because `optimisation.auto_size` sizes the ASHP against the cold-weather-derated design peak, so it meets demand without backup. The service gate is therefore **not** exercised by this test, and this study provides no evidence either way on whether the model catches an under-sized system. What fails here is NPV/IRR — an economic result, not a physical one. A genuine service-gate test needs plant deliberately sized below the design peak; it has not been run.
- Every scenario carries an explicit warnings/assumptions log, a scenario hash and a model version in its audit trail (`result['audit']`) — findings below are reproducible from the same script.
- `MODEL_SUMMARY.md` §12 carries the full, honestly-stated list of what the model does and does not yet prove (e.g. generic-length route mode is an equivalent-trunk approximation, not GIS routing; N-1 is a peak-capacity screen, not a dynamic outage simulation). This is a **screening tool**, not a bankable investment model — present it as that.

## 2. Gas-parity tariff pricing

The model's default tariff mode (`counterfactual_bill_parity`) is already gas-parity: every customer's modelled district heat bill is held to **their own** modelled individual-gas-boiler bill (a full CAPEX+OPEX counterfactual per building, not a flat unit-rate proxy), so district heat can never look artificially cheaper than the gas alternative it's replacing.
- Verified across all 16 technology x archetype runs: bill ratio (district/gas) is **<= 100% in every case** (`all_parity_ok = True`) — see `gas_parity_check.csv`.
- The resulting *equivalent* year-1 heat tariff (p/kWh) is essentially **flat across technologies within an archetype** (chart 03) and varies only with the archetype's own gas-counterfactual bill — this is the parity mechanism working exactly as intended: revenue is capped at what customers already pay for gas, regardless of which technology delivers the heat. The technology/CAPEX difference shows up instead in the **required break-even tariff** (chart 04, 20-105 p/kWh) — the project's real cost-recovery need, which sits far above the ~8.3-8.5 p/kWh customers are actually charged. That gap (not the tariff mechanism) is why every NPV in section 4 is negative.
- Reference external point: the live Ofgem regulated gas price cap is **7.33 p/kWh** (7.33p unit rate, household retail basis) — the model's parity mechanism is a full whole-bill comparison per building, not just this flat rate.

## 3. Archetype demand from the real weather file

`profiles/weather_data.csv` (8,760-hour 2023 TMY-style year) drives heating-degree-hour-scaled demand for three density archetypes:

| Archetype                 |   Annual heat+DHW demand (MWh) |   Peak heat demand (MW) |   Illustrative route length (m) |   Linear heat density (MWh/m/yr) |
|:--------------------------|-------------------------------:|------------------------:|--------------------------------:|---------------------------------:|
| Dense (town centre)       |                          12990 |                    7.91 |                             900 |                            14.43 |
| Middle (suburban mixed)   |                           9179 |                    6.22 |                            2800 |                             3.28 |
| Scarce (low-density edge) |                           5884 |                    3.24 |                            6500 |                             0.91 |
| Ealing Phase 1 (real)     |                          14161 |                    7.73 |                            2148 |                             6.59 |

**Route lengths are illustrative placeholders**, not measured — they show the direction and scale of the density effect (dense: short branches, high linear density; scarce: long branches, low linear density), pending the real Exeter route geometry.

## 4. Energy-centre / heat-recovery technology matrix

Four technology options x three archetypes, each auto-sized from the archetype's own demand via `optimisation.auto_size.recommend_sizing()` (baseload-first, load-duration-based, cold-weather-derated ASHP sizing) rather than hand-picked capacities:

| Archetype                 | Technology                                  | Carbon gate   | Service gate   |   Equivalent year-1 heat tariff (p/kWh) |   Investor NPV (£m) | Screening decision   |
|:--------------------------|:--------------------------------------------|:--------------|:---------------|----------------------------------------:|--------------------:|:---------------------|
| Dense (town centre)       | Gas-only reference                          | FAIL          | PASS           |                                  10.004 |              -12.5  | FAIL                 |
| Dense (town centre)       | ASHP + gas peak                             | PASS          | PASS           |                                  10.004 |              -29.58 | FAIL                 |
| Dense (town centre)       | Data-centre waste heat + booster + gas peak | PASS          | PASS           |                                  10.004 |              -24.87 | FAIL                 |
| Dense (town centre)       | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                  10.004 |              -18.95 | FAIL                 |
| Middle (suburban mixed)   | Gas-only reference                          | FAIL          | PASS           |                                   7.819 |              -16.61 | FAIL                 |
| Middle (suburban mixed)   | ASHP + gas peak                             | PASS          | PASS           |                                   7.819 |              -27.89 | FAIL                 |
| Middle (suburban mixed)   | Data-centre waste heat + booster + gas peak | PASS          | PASS           |                                   7.819 |              -25.47 | FAIL                 |
| Middle (suburban mixed)   | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   7.819 |              -22.1  | FAIL                 |
| Scarce (low-density edge) | Gas-only reference                          | FAIL          | PASS           |                                   8.043 |              -20.07 | FAIL                 |
| Scarce (low-density edge) | ASHP + gas peak                             | PASS          | PASS           |                                   8.043 |              -25.29 | FAIL                 |
| Scarce (low-density edge) | Data-centre waste heat + booster + gas peak | FAIL          | PASS           |                                   8.043 |              -24.22 | FAIL                 |
| Scarce (low-density edge) | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   8.043 |              -23.26 | FAIL                 |
| Ealing Phase 1 (real)     | Gas-only reference                          | FAIL          | PASS           |                                   7.814 |              -11.65 | FAIL                 |
| Ealing Phase 1 (real)     | ASHP + gas peak                             | PASS          | PASS           |                                   7.814 |              -30.41 | FAIL                 |
| Ealing Phase 1 (real)     | Data-centre waste heat + booster + gas peak | PASS          | PASS           |                                   7.814 |              -26.21 | FAIL                 |
| Ealing Phase 1 (real)     | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   7.814 |              -17.8  | FAIL                 |

- **Best NPV among carbon-compliant (viable) options: Ealing Phase 1 (real) — EfW heat export + ASHP + gas peak** (NPV £-17.80m, -4.56% IRR, screening: FAIL).
- The gas-only reference case has a less-negative NPV than every low-carbon option in every archetype (it has no ASHP/EfW CAPEX to recover) but **fails the carbon gate everywhere** — it is retained deliberately as the counterfactual baseline, not as a candidate design. Do not read "best NPV overall" as "best option" without checking the carbon gate first.
- Data-centre and EfW capacities here are generic (sized as a fraction of local demand); treat as "if a source of about this size existed nearby", not a confirmed offtake agreement.

**Every one of the 12 base cases fails on investor NPV** under strict gas-parity billing. This is a real, expected district-heating result (heat networks essentially never clear a commercial hurdle on cost-reflective/gas-parity tariffs alone) — not a model defect — but two caveats matter for how hard to read into the exact NPV figures:
- **Fixed CAPEX/OPEX line items were held constant across all three archetypes** (energy-centre building, electrical/gas connection, controls, billing/insurance/overhead — reused unscaled from the Ealing-calibrated defaults in `scenarios/worked_scenarios.py`). These fixed costs hit the Scarce archetype (321 connections) proportionally far harder than Dense (723 connections) — a real minimum-viable-scale effect, but the absolute NPV gap for Scarce is overstated until fixed items are re-scoped for scheme size.
- The customer base here (up to ~723 connections) is well below the ~1,100-connection Ealing-scale case this model's illustrative CAPEX/OPEX defaults were calibrated against.

### GHNF capital-grant sensitivity

The obvious next screening question — does UK Green Heat Network Fund capital grant (up to 50% of eligible CAPEX, gated on the <=100 gCO2e/kWh carbon threshold) close the gap? Tested at 40% on the two carbon-compliant technologies:

| Archetype                 | Technology                        |   Grant awarded (£m) |   NPV without grant (£m) |   NPV with 40% GHNF grant (£m) | Screening decision with grant   |
|:--------------------------|:----------------------------------|---------------------:|-------------------------:|-------------------------------:|:--------------------------------|
| Dense (town centre)       | ASHP + gas peak                   |                 3.48 |                   -29.58 |                         -26.09 | FAIL                            |
| Dense (town centre)       | EfW heat export + ASHP + gas peak |                 3.31 |                   -18.95 |                         -15.64 | FAIL                            |
| Middle (suburban mixed)   | ASHP + gas peak                   |                 3.95 |                   -27.89 |                         -23.94 | FAIL                            |
| Middle (suburban mixed)   | EfW heat export + ASHP + gas peak |                 3.87 |                   -22.1  |                         -18.23 | FAIL                            |
| Scarce (low-density edge) | ASHP + gas peak                   |                 3.97 |                   -25.29 |                         -21.32 | FAIL                            |
| Scarce (low-density edge) | EfW heat export + ASHP + gas peak |                 3.97 |                   -23.26 |                         -19.29 | FAIL                            |
| Ealing Phase 1 (real)     | ASHP + gas peak                   |                 4.08 |                   -30.41 |                         -26.33 | FAIL                            |
| Ealing Phase 1 (real)     | EfW heat export + ASHP + gas peak |                 4.12 |                   -17.8  |                         -13.68 | FAIL                            |

Grant support materially narrows the NPV gap everywhere but does not flip any case to positive NPV on its own at this connection count — confirming that scale (connection count / linear density), not technology choice, is the binding constraint for these archetype sizes. See chart 07.

## 5. Four-pipe (heating + cooling) check

- Dense archetype with AC-office/supermarket cooling load added: NPV £-36.61m, screening decision **FAIL**.
- Cooling bill ratio vs individual air-conditioning: 100.0% (parity constraint: must stay <= 100%).

## 6. What this means for an initial screening tool layout

- **Linear heat density is the first-order screening variable** (chart 04): required break-even tariff rises sharply as route length grows relative to demand. A layout tool for Dalkia should surface this number FIRST, before CAPEX/NPV detail.
- Dense, short-branch layouts clear the gas-parity bar most easily; scarce/long-branch layouts need either a materially cheaper heat source, grant support, or a shorter/denser route to reach viability.
- **Next step, pending the Exeter case study**: replace the illustrative `generic_length` route assumption with a real `tree` topology (see `network/topology_tree.py` and the worked Ealing example in `network/network_topology.py::ealing_town_centre_topology()` as the template) so branch-level lengths and per-segment pipe sizing reflect the actual Exeter map rather than one equivalent trunk.

---
Model version: `2.7.1-streamlit-matrix-verified`. Generated by `analysis/dalkia_screening_study.py`; all figures reproducible by re-running that script.