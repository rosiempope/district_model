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

- Building archetype, floor area and dwelling count
- Climate scenario
- Network length and 2-pipe / 4-pipe selection
- Heating and cooling flow/return temperatures
- Heating and cooling technology, preset, capacity and unit count
- Project life, discount rate, O&M rate and counterfactual

## Workflow

1. Load a worked scenario or start with a blank example.
2. Modify the inputs in **Build scenario**.
3. Select **Validate and run scenario**.
4. Inspect hourly-model outputs in **Results**.
5. Select **Add result to comparison**.
6. Repeat for alternative cases, then use **Compare scenarios** to choose the
   metric to graph and download the comparison table.

## JSON interface

The UI downloads and uploads a plain scenario JSON file. It is the same input
contract used by `scenarios.scenario_runner.run_scenario()`, so this UI is a
prototype client for a future FastAPI + React implementation rather than a
second model.

For a four-pipe scenario, `network.include_cooling` must be `true`, at least one
cooling source must be added, and the counterfactual is automatically set to
`individual_gas_and_ac`.
