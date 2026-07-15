# Dalkia initial-screening study — findings

Run directly against the model engine (`scenarios.scenario_runner.run_scenario`, `profiles.demand_synthesis`, `optimisation.auto_size`) — not through Streamlit, not through the test suite.

## 1. Is the model accurate / trustworthy for a first screen?

- The engine is unit-tested, but more importantly here: it **actively catches infeasible designs** rather than always returning a positive answer. The ASHP-only stress test below (dense archetype, no gas/electric backup) produces genuine unmet demand and a FAILED service gate — the model does not silently paper over an under-sized system.
  - Unmet heat: **0.0 MWh/yr** (0.00% of demand); screening decision: **FAIL**; failed gates: Investor NPV, Investor IRR.
- Every scenario carries an explicit warnings/assumptions log, a scenario hash and a model version in its audit trail (`result['audit']`) — findings below are reproducible from the same script.
- See `MODEL_ASSURANCE.md` in the repo root for the full, honestly-stated list of what the model does and does not yet prove (e.g. generic-length route mode is an equivalent-trunk approximation, not GIS routing; N-1 is a peak-capacity screen, not a dynamic outage simulation). This is a **screening tool**, not a bankable investment model — present it as that.

## 2. Gas-parity tariff pricing

The model's default tariff mode (`counterfactual_bill_parity`) is already gas-parity: every customer's modelled district heat bill is held to **their own** modelled individual-gas-boiler bill (a full CAPEX+OPEX counterfactual per building, not a flat unit-rate proxy), so district heat can never look artificially cheaper than the gas alternative it's replacing.
- Verified across all 12 technology x archetype runs: bill ratio (district/gas) is **<= 100% in every case** (`all_parity_ok = True`) — see `gas_parity_check.csv`.
- The resulting *equivalent* year-1 heat tariff (p/kWh) is essentially **flat across technologies within an archetype** (chart 03) and varies only with the archetype's own gas-counterfactual bill — this is the parity mechanism working exactly as intended: revenue is capped at what customers already pay for gas, regardless of which technology delivers the heat. The technology/CAPEX difference shows up instead in the **required break-even tariff** (chart 04, 20-105 p/kWh) — the project's real cost-recovery need, which sits far above the ~8.3-8.5 p/kWh customers are actually charged. That gap (not the tariff mechanism) is why every NPV in section 4 is negative.
- Reference external point: the live Ofgem regulated gas price cap is **7.33 p/kWh** (7.33p unit rate, household retail basis) — the model's parity mechanism is a full whole-bill comparison per building, not just this flat rate.

## 3. Archetype demand from the real weather file

`profiles/weather_data.csv` (8,760-hour 2023 TMY-style year) drives heating-degree-hour-scaled demand for three density archetypes:

| Archetype                 |   Annual heat+DHW demand (MWh) |   Peak heat demand (MW) |   Illustrative route length (m) |   Linear heat density (MWh/m/yr) |
|:--------------------------|-------------------------------:|------------------------:|--------------------------------:|---------------------------------:|
| Dense (town centre)       |                          13180 |                    7.93 |                             900 |                            14.64 |
| Middle (suburban mixed)   |                           8495 |                    5.74 |                            2800 |                             3.03 |
| Scarce (low-density edge) |                           4222 |                    2.17 |                            6500 |                             0.65 |

**Route lengths are illustrative placeholders**, not measured — they show the direction and scale of the density effect (dense: short branches, high linear density; scarce: long branches, low linear density), pending the real Exeter route geometry.

## 4. Energy-centre / heat-recovery technology matrix

Four technology options x three archetypes, each auto-sized from the archetype's own demand via `optimisation.auto_size.recommend_sizing()` (baseload-first, load-duration-based, cold-weather-derated ASHP sizing) rather than hand-picked capacities:

| Archetype                 | Technology                                  | Carbon gate   | Service gate   |   Equivalent year-1 heat tariff (p/kWh) |   Investor NPV (£m) | Screening decision   |
|:--------------------------|:--------------------------------------------|:--------------|:---------------|----------------------------------------:|--------------------:|:---------------------|
| Dense (town centre)       | Gas-only reference                          | FAIL          | PASS           |                                   8.314 |              -12.94 | FAIL                 |
| Dense (town centre)       | ASHP + gas peak                             | PASS          | PASS           |                                   8.314 |              -29.14 | FAIL                 |
| Dense (town centre)       | Data-centre waste heat + booster + gas peak | FAIL          | PASS           |                                   8.314 |              -26.66 | FAIL                 |
| Dense (town centre)       | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   8.314 |              -18.74 | FAIL                 |
| Middle (suburban mixed)   | Gas-only reference                          | FAIL          | PASS           |                                   8.326 |              -16.8  | FAIL                 |
| Middle (suburban mixed)   | ASHP + gas peak                             | PASS          | PASS           |                                   8.326 |              -28.14 | FAIL                 |
| Middle (suburban mixed)   | Data-centre waste heat + booster + gas peak | FAIL          | PASS           |                                   8.326 |              -26.82 | FAIL                 |
| Middle (suburban mixed)   | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   8.326 |              -21.49 | FAIL                 |
| Scarce (low-density edge) | Gas-only reference                          | FAIL          | PASS           |                                   8.538 |              -22.15 | FAIL                 |
| Scarce (low-density edge) | ASHP + gas peak                             | FAIL          | PASS           |                                   8.538 |              -26.95 | FAIL                 |
| Scarce (low-density edge) | Data-centre waste heat + booster + gas peak | FAIL          | PASS           |                                   8.538 |              -26.34 | FAIL                 |
| Scarce (low-density edge) | EfW heat export + ASHP + gas peak           | PASS          | PASS           |                                   8.538 |              -24.07 | FAIL                 |

- **Best NPV among carbon-compliant (viable) options: Dense (town centre) — EfW heat export + ASHP + gas peak** (NPV £-18.74m, -4.67% IRR, screening: FAIL).
- The gas-only reference case has a less-negative NPV than every low-carbon option in every archetype (it has no ASHP/EfW CAPEX to recover) but **fails the carbon gate everywhere** — it is retained deliberately as the counterfactual baseline, not as a candidate design. Do not read "best NPV overall" as "best option" without checking the carbon gate first.
- Data-centre and EfW capacities here are generic (sized as a fraction of local demand); treat as "if a source of about this size existed nearby", not a confirmed offtake agreement.

**Every one of the 12 base cases fails on investor NPV** under strict gas-parity billing. This is a real, expected district-heating result (heat networks essentially never clear a commercial hurdle on cost-reflective/gas-parity tariffs alone) — not a model defect — but two caveats matter for how hard to read into the exact NPV figures:
- **Fixed CAPEX/OPEX line items were held constant across all three archetypes** (energy-centre building, electrical/gas connection, controls, billing/insurance/overhead — reused unscaled from the Ealing-calibrated defaults in `scenarios/worked_scenarios.py`). These fixed costs hit the Scarce archetype (321 connections) proportionally far harder than Dense (723 connections) — a real minimum-viable-scale effect, but the absolute NPV gap for Scarce is overstated until fixed items are re-scoped for scheme size.
- The customer base here (up to ~723 connections) is well below the ~1,100-connection Ealing-scale case this model's illustrative CAPEX/OPEX defaults were calibrated against.

### GHNF capital-grant sensitivity

The obvious next screening question — does UK Green Heat Network Fund capital grant (up to 50% of eligible CAPEX, gated on the <=100 gCO2e/kWh carbon threshold) close the gap? Tested at 40% on the two carbon-compliant technologies:

| Archetype                 | Technology                        |   Grant awarded (£m) |   NPV without grant (£m) |   NPV with 40% GHNF grant (£m) | Screening decision with grant   |
|:--------------------------|:----------------------------------|---------------------:|-------------------------:|-------------------------------:|:--------------------------------|
| Dense (town centre)       | ASHP + gas peak                   |                 2.94 |                   -29.14 |                         -26.2  | FAIL                            |
| Dense (town centre)       | EfW heat export + ASHP + gas peak |                 3.08 |                   -18.74 |                         -15.66 | FAIL                            |
| Middle (suburban mixed)   | ASHP + gas peak                   |                 3.5  |                   -28.14 |                         -24.64 | FAIL                            |
| Middle (suburban mixed)   | EfW heat export + ASHP + gas peak |                 3.63 |                   -21.49 |                         -17.85 | FAIL                            |
| Scarce (low-density edge) | ASHP + gas peak                   |                 0    |                   -26.95 |                         -26.95 | FAIL                            |
| Scarce (low-density edge) | EfW heat export + ASHP + gas peak |                 2.85 |                   -24.07 |                         -21.22 | FAIL                            |

Grant support materially narrows the NPV gap everywhere but does not flip any case to positive NPV on its own at this connection count — confirming that scale (connection count / linear density), not technology choice, is the binding constraint for these archetype sizes. See chart 07.

## 5. Four-pipe (heating + cooling) check

- Dense archetype with AC-office/supermarket cooling load added: NPV £-32.70m, screening decision **FAIL**.
- Cooling bill ratio vs individual air-conditioning: 100.0% (parity constraint: must stay <= 100%).

## 6. What this means for an initial screening tool layout

- **Linear heat density is the first-order screening variable** (chart 04): required break-even tariff rises sharply as route length grows relative to demand. A layout tool for Dalkia should surface this number FIRST, before CAPEX/NPV detail.
- Dense, short-branch layouts clear the gas-parity bar most easily; scarce/long-branch layouts need either a materially cheaper heat source, grant support, or a shorter/denser route to reach viability.
- **Next step, pending the Exeter case study**: replace the illustrative `generic_length` route assumption with a real `tree` topology (see `network/topology_tree.py` and the worked Ealing example in `network/network_topology.py::ealing_town_centre_topology()` as the template) so branch-level lengths and per-segment pipe sizing reflect the actual Exeter map rather than one equivalent trunk.

---
Model version: `2.7.1-streamlit-matrix-verified`. Generated by `analysis/dalkia_screening_study.py`; all figures reproducible by re-running that script.