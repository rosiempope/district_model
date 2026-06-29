"""Run the five worked screening scenarios.

A future UI/API should submit the same plain dictionary shape consumed by
scenarios.scenario_runner.run_scenario().
"""
from scenarios.scenario_runner import run_scenario, comparison_table
from scenarios.worked_scenarios import WORKED_SCENARIOS

if __name__ == "__main__":
    print(comparison_table([run_scenario(s) for s in WORKED_SCENARIOS]).to_string(index=False))
