# District heating and cooling screening model

Version 2.0 is an hourly technical and commercial screening model for two-pipe
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

## End-user workflow

1. Enter each customer/site, floor area or dwellings, billed connections,
   connection year and probability. Replace archetype demand with measured
   annual heat/cooling overrides where available.
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

The scenario interface remains plain JSON-compatible data through
`scenarios.scenario_runner.run_scenario`.

## Assurance status

This is a screening model, not a bankable financial model. Every output carries
an assumption/default warning report, model version, run timestamp and scenario
hash. Before external investor use, replace illustrative presets with sourced
project assumptions and complete independent engineering and project-finance
reconciliation. See `MODEL_ASSURANCE.md` for the remaining limitations and
release checks.
