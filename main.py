'''

This is the main file that is run to calculate:

1. 


'''


def run_scenario(config: dict) -> dict:
    weather = load_weather(config["weather"])
    weather = apply_climate_scenario(weather, config["climate_scenario"])

    demand = synthesise_demand(
        weather_df=weather,
        buildings=config["buildings"],
    )

    sources = build_source_stack(
        weather_df=weather,
        source_configs=config["sources"],
    )

    dispatch = dispatch_heat(
        demand_MW=demand["heat_MW"],
        sources=sources,
    )

    network = size_heating_network(
        peak_heat_kW=dispatch["network_peak_heat_kW"],
        network_length_m=config["network"]["length_m"],
        flow_temp_C=config["network"]["flow_temp_C"],
        return_temp_C=config["network"]["return_temp_C"],
    )

    capex = calculate_capex(sources, network)
    opex = calculate_opex(dispatch, network)
    economics = calculate_metrics(capex, opex, dispatch)

    return {
        "weather": weather,
        "demand": demand,
        "dispatch": dispatch,
        "network": network,
        "capex": capex,
        "opex": opex,
        "economics": economics,
    }