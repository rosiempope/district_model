# District heating and cooling screening model

Version 2.7.1 is an hourly technical and commercial screening model for two-pipe
heating and four-pipe heating/cooling networks. It is intended to help an
operator compare a common customer/route case across design alternatives.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the worked option matrix from the command line:

```bash
python main.py
```

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```

Re-run the Ealing Town Centre Phase 1 validation:

```bash
python -m reports.ealing_validation
```

Run the dense-core / compact-private / extended-route comparison:

```bash
python -m reports.feasibility_comparison
```

The command writes scenario, route-sensitivity and separate four-pipe cooling
checks to `output/feasibility_comparison/`.

Run the data-centre waste-heat study, UK support pre-checks, sensitivities and
40-year cash-flow comparison:

```bash
python -m reports.data_centre_feasibility
```

Run the representative technology-mix route/demand frontier, price
sensitivities, cooling decomposition and fair customer-bill comparison:

```bash
python -m reports.technology_frontier
```

The UI template **Ealing report validation - Phase 1** contains the same
published report inputs and produces zero unmet heat.

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
See `SCREENING_TOOL_GUIDE.md` for the review workflow and graph guide.

## Calculation basis

- 8,760-hour building demand, source availability, dispatch and network losses.
- Climate cases use one shared baseline degree-hour reference.
- Network pumping electricity contributes to OPEX and carbon.
- Data-centre source heat and booster output are coupled hourly.
- One years 0–40 cash-flow table drives NPV, IRR, payback and charts.
- Electricity, gas, third-party heat, tariffs and other OPEX have isolated real
  change rates.
- CAPEX, OPEX, grant and scheduled REPEX remain separate auditable line items.
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
reconciliation. See `MODEL_ASSURANCE.md` for the remaining limitations and
release checks.
