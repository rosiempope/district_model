
## Local screening UI

Install the dependencies and run the local UI from the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The UI supports editable 2-pipe heating and 4-pipe heating/cooling scenarios,
JSON upload/download, source and network inputs, scenario validation, live
results, and comparison charts. It uses the same plain scenario JSON contract
as `scenarios.scenario_runner.run_scenario`, so it can later be placed behind a
FastAPI service and a React frontend without changing the model interface.
