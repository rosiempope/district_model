# District energy screening UI

Run from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the local application. The default URL is normally
`http://localhost:8501`.

## What the UI controls

- Building archetype and either floor area/dwellings or measured annual
  heat/cooling/DHW, plus optional measured peak, billed connections,
  connection year and connection probability
- Climate scenario
- Equivalent-trunk length or an editable energy-centre/junction/customer tree
- Heating and cooling flow/return temperatures
- Heating and cooling technology, preset, capacity and unit count
- Separate heat/cooling tariffs, project CAPEX, annual OPEX, real price paths,
  project/social discount rates, grant and counterfactual

## Workflow

1. Load a worked scenario or start with a blank example.
2. Modify the inputs in **Build scenario**.
3. Select **Validate and run scenario**.
4. Resolve the visible assumptions/default warnings and any failed service or
   carbon gate.
5. Inspect the investor and whole-system cash-flow audits in **Results**.
6. Select **Add result to comparison**.
7. Repeat for alternative cases, then use **Compare scenarios** to choose the
   metric to graph and download the comparison table.

## JSON interface

The UI downloads and uploads a plain scenario JSON file. It is the same input
contract used by `scenarios.scenario_runner.run_scenario()`, so this UI is a
prototype client for a future FastAPI + React implementation rather than a
second model.

For a four-pipe scenario, `network.include_cooling` must be `true`, at least one
cooling source must be added, and the counterfactual is automatically set to
`individual_gas_and_ac`.

`capacity_MW` always means total installed capacity. The model derives per-unit
capacity from total capacity and unit count; do not pre-divide the value.
