# District heating and cooling screening model

Version 2.7.1 is an hourly technical and commercial screening model for two-pipe
heating and four-pipe heating/cooling networks. It is intended to help an
operator compare a common customer/route case across design alternatives.

**`MODEL_SUMMARY.md` is the place to start** — what the model takes in, what it
produces, the thermodynamic and economic basis, the benchmarks and citations, the
case studies and their findings, and the limitations to state openly.

## Install and run

```bash
pip install -r requirements.txt

python main.py                          # the five worked scenarios, side by side
streamlit run app.py                    # the UI, on http://localhost:8501
python -m unittest discover -s tests    # 209 tests
```

## What's in the folder

| Path | What it is |
|---|---|
| `MODEL_SUMMARY.md` | Full capability summary — inputs, outputs, equations, benchmarks, case studies, limitations |
| `scenarios/` | The scenario layer: JSON-compatible scenario definitions, the schema, and `scenario_runner.run_scenario()` — the single entry point. Consumed by both the UI and the report runners |
| `profiles/` | Weather (London Heathrow TMYx 2011–2025), climate scenarios, 8,760-hour demand synthesis |
| `components/` | Heat and cooling sources: ASHP, chillers (air-cooled, water-cooled+tower, free-cooling glycol, absorption), booster heat pump, data centre, EfW CHP, boilers, thermal storage |
| `network/` | Pipe catalog (hydraulics, sizing, heat loss, cost), tree topology, thermal physics, pumping |
| `optimisation/` | Merit-order dispatch, auto-sizing, sizing helpers |
| `economics/` | CAPEX, O&M rates, tariffs, cash flow, GHNF grant, individual-system counterfactuals |
| `reports/` | Study runners that write to `output/` |
| `analysis/` | Site case studies (Exeter, Dalkia archetypes, source-stack comparisons) |
| `tests/` | 209 tests — integration through `run_scenario()`, plus physics units |
| `output/` | Generated figures and tables. Regenerable; only the `.md` findings are tracked in git |

## The studies

```bash
python -m reports.ealing_validation        # calibration against the published SEL report
python -m reports.cost_breakdown           # where the money goes, by scaling basis
python -m reports.feasibility_comparison   # dense core / compact private / extended route
python -m reports.data_centre_feasibility  # waste-heat cases, UK support pre-checks, 40-yr cash flow
python -m reports.technology_frontier      # route/demand frontier, price sensitivities
python -m reports.internal_screening_readout
python -m analysis.exeter_case_study       # real tree topology from the DESNZ zoning map
python -m analysis.dalkia_screening_study  # 4 technologies x 3 density archetypes
```

Each writes CSVs, figures and a `findings.md` to its own `output/` subdirectory.

`reports.cost_breakdown` is the one to read first: it tags every CAPEX and OPEX
line with how it scales, and reports the size-independent burden — 5.93 p/kWh
against a 7.33p Ofgem cap on the worked scenarios, which is the mechanism behind
most of the other findings.

## The UI

The UI controls building archetype and either floor area/dwellings or measured
annual heat/cooling/DHW (plus optional measured peak, billed connections,
connection year and probability); climate scenario; equivalent-trunk length or an
editable energy-centre/junction/customer tree; heating and cooling flow/return
temperatures; technology, preset, capacity and unit count; and separate
heat/cooling tariffs, project CAPEX, annual OPEX, real price paths,
project/social discount rates, grant and counterfactual.

Workflow: load a worked scenario or start blank → modify inputs in **Build
scenario** → **Validate and run scenario** → resolve the visible
assumptions/default warnings and any failed service or carbon gate → inspect the
investor and whole-system cash-flow audits in **Results** → **Add result to
comparison** → repeat for alternatives, then use **Compare scenarios** to graph a
metric and download the table.

The UI uploads and downloads a plain scenario JSON — the same input contract as
`scenarios.scenario_runner.run_scenario()`. It is a prototype client for a future
API implementation, not a second model. For a four-pipe scenario,
`network.include_cooling` must be `true`, at least one cooling source is
required, and the counterfactual is automatically set to `individual_gas_and_ac`.

The UI template **Ealing report validation - Phase 1** contains the published
report inputs and produces zero unmet heat.

## End-user workflow

1. Enter each customer/site using either floor area/dwellings for an archetype
   estimate or measured annual heat/cooling/DHW values. Measured annual heat is
   sufficient when floor area is unknown; services left blank without an area
   scale are treated as zero. Add billed connections, connection year and
   probability, and measured peak heat where available.
2. Use tree network mode for meaningful screening: add the energy-centre route,
   junctions, customer branches and lengths. Generic length is an uncertain
   equivalent-trunk approximation.
3. Choose allowed heat/cooling resources. `capacity_MW` always means total
   installed capacity; unit count is entered separately.
4. Auto-size as a starting design, then check unmet-energy/service gates and
   adjust for the required resilience standard.
5. Enter customer heat and cooling tariffs, project CAPEX additions, annual
   overheads, connection phasing and real price paths.
6. Review the decision outputs, warnings, carbon/service gates, CAPEX/OPEX
   breakdown, investor cash flow, whole-system comparison, required heat tariff
   and downloadable annual audit tables.

The internal decision applies the scenario's own unmet-energy, carbon, investor
NPV and IRR thresholds, with optional N-1 and maximum break-even tariff gates.

## Calculation basis

- 8,760-hour building demand, source availability, dispatch and network losses.
- Climate cases use one shared baseline degree-hour reference.
- Network pumping electricity contributes to OPEX and carbon.
- Data-centre source heat and booster output are coupled hourly.
- One years 0–40 cash-flow table drives NPV, IRR, payback and charts.
- Electricity, gas, third-party heat, tariffs and other OPEX have isolated real
  change rates.
- CAPEX, OPEX, grant and scheduled REPEX remain separate auditable line items.
- Design, commissioning and contingency apply to the whole delivered scope
  except land.
- Fixed CAPEX/OPEX items are scaled to scheme peak capacity, with the factor
  recorded in the scenario's audit hash.
- Discounted LCOH/LCO-service use discounted project costs and connected energy.
- GHNF screening applies a strictly-below-50% intensity limit, the thermal-output
  cap and a visible 100 gCO2e/kWh carbon gate.
- Heating can be held to each customer's modelled individual-gas bill and
  cooling to its individual-AC running cost, preventing tariff changes from
  creating artificial feasibility.
- Full-buildout OPEX reconciles source energy, hourly-priced pumping,
  technology/network O&M and additional overheads with a zero residual.

The scenario interface remains plain JSON-compatible data through
`scenarios.scenario_runner.run_scenario`.

## Assurance status

This is a screening model, not a bankable financial model. Every output carries
an assumption/default warning report, model version, run timestamp and scenario
hash. Before external investor use, replace illustrative presets with sourced
project assumptions and complete independent engineering and project-finance
reconciliation.

The Ealing Town Centre Phase 1 validation reconciles 13 of 13 metrics against the
published June 2025 SEL report. See `MODEL_SUMMARY.md` §12 for the full
limitations list and the pre-circulation checklist. The largest remaining gaps to
state openly: the cooling model overshoots its own benchmark by ~9-10%; annual
physical performance is repeated across all 40 years, with no year-by-year grid
carbon trajectory; four-pipe costs each duty at full trenched £/m, so shared
civils are not credited; and the ambient loop (5GDH) is not implemented.
