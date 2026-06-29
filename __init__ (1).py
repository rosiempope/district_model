import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import GAS_ONLY, DATACENTRE_PLUS_BOOSTER

r = run_scenario(GAS_ONLY)
h = r["headline"]
assert h["annual_unmet_demand_MWh"] == 0.0
assert h["annual_heat_to_generate_MWh"] > h["annual_heat_demand_MWh"]
assert r["financial"]["counterfactual"] == "individual_gas"
r2 = run_scenario(DATACENTRE_PLUS_BOOSTER)
assert r2["headline"]["annual_unmet_demand_MWh"] == 0.0
assert r2["headline"]["carbon_intensity_kgCO2_per_kWh"] < h["carbon_intensity_kgCO2_per_kWh"]
print("scenario runner tests passed")
