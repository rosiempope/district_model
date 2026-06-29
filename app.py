from scenarios.worked_scenarios import FOUR_PIPE_ASHP_GAS
from scenarios.scenario_runner import run_scenario


def test_four_pipe_scenario_runs_with_combined_counterfactual():
    result = run_scenario(FOUR_PIPE_ASHP_GAS)
    h = result["headline"]
    assert h["system_type"] == "4_pipe_heating_cooling"
    assert h["annual_cooling_demand_MWh"] > 0
    assert h["annual_network_cooling_gain_MWh"] > 0
    assert h["annual_unmet_cooling_MWh"] == 0
    assert result["financial"]["counterfactual"] == "individual_gas_and_ac"
