'''

This is the main file that is run to calculate:

1. 


'''



#this is for the climate considaeation.
'''
for scenario_name in ['baseline', '2050_central', '2050_high']:
    w = apply_climate_scenario(weather_df, scenario_name)
    network = synthesise_network(w, scenario_config)
    results[scenario_name] = network
'''