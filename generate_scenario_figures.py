import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenarios.scenario_schema import validate_scenario
from scenarios.worked_scenarios import GAS_ONLY
assert validate_scenario(GAS_ONLY) == []
bad = {"name":"", "demand":{"buildings":[]}, "sources":[]}
errors = validate_scenario(bad)
assert any("name" in e for e in errors) and any("buildings" in e for e in errors)
print("scenario schema tests passed")
