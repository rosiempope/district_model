"""Five illustrative, editable scenario dictionaries for the UI/API layer.
They deliberately use plain JSON-compatible values only."""
from copy import deepcopy

BASE_BUILDINGS=[
    {"name":"Civic offices","type":"office","floor_area_m2":18000},
    {"name":"Residential block A","type":"residential_existing","floor_area_m2":24000},
    {"name":"Residential block B","type":"residential_existing","floor_area_m2":18000},
    {"name":"Retail centre","type":"retail","floor_area_m2":9000},
    {"name":"Hotel","type":"hotel","floor_area_m2":8500},
    {"name":"Health centre","type":"hospital","floor_area_m2":6000},
]

def base_scenario(name: str) -> dict:
    return {
        "name":name,
        "climate_scenario":"baseline",
        "demand":{"buildings":deepcopy(BASE_BUILDINGS)},
        "network":{"mode":"generic_length","length_m":3000.0,"include_cooling":False,
                   "heat_flow_temp_C":70.0,"heat_return_temp_C":40.0},
        "economics":{"project_lifetime_years":25,"discount_rate":0.105,"om_rate":0.01,
                     "counterfactual":"individual_gas"},
    }

GAS_ONLY=base_scenario("Gas-only district baseline")
GAS_ONLY["sources"]=[{"type":"gas_boiler","preset":"ealing_phase2","name":"Gas boiler","capacity_MW":10.0}]

ASHP_ONLY=base_scenario("ASHP-only district heat")
ASHP_ONLY["sources"]=[{"type":"ashp","preset":"ealing_phase2","name":"ASHP bank","capacity_MW":6.5,"flow_temp_C":70.0}]

ASHP_PLUS_GAS_PEAK=base_scenario("ASHP plus gas peak boiler")
ASHP_PLUS_GAS_PEAK["sources"]=[
    {"type":"ashp","preset":"ealing_phase1","name":"ASHP bank","capacity_MW":2.8,"flow_temp_C":70.0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":6.5},
]

DATACENTRE_PLUS_BOOSTER=base_scenario("Data-centre waste heat plus booster and gas peak")
DATACENTRE_PLUS_BOOSTER["sources"]=[
    {"type":"data_centre","preset":"redwire_ealing","name":"Data-centre waste heat","capacity_MW":3.6, "dispatch_direct":False},
    {"type":"booster_heat_pump","preset":"generic_2MW","name":"Booster heat pump","capacity_MW":3.0,"depends_on":0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":7.5},
]

EFW_PLUS_ASHP=base_scenario("EfW heat export plus ASHP and gas peak")
EFW_PLUS_ASHP["sources"]=[
    {"type":"efw_chp","preset":"newlincs_style","name":"EfW heat export","capacity_MW":3.0},
    {"type":"ashp","preset":"ealing_phase1","name":"ASHP bank","capacity_MW":2.8,"flow_temp_C":70.0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":5.5},
]

FOUR_PIPE_ASHP_GAS = base_scenario("4-pipe ASHP, gas peak and central cooling")
FOUR_PIPE_ASHP_GAS["network"].update({"include_cooling": True, "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0})
FOUR_PIPE_ASHP_GAS["economics"]["counterfactual"] = "individual_gas_and_ac"
FOUR_PIPE_ASHP_GAS["sources"] = [
    {"type":"ashp", "preset":"ealing_phase1", "name":"ASHP bank", "capacity_MW":2.8, "flow_temp_C":70.0},
    {"type":"gas_boiler", "preset":"ealing_phase1", "name":"Peak gas boiler", "capacity_MW":6.5},
]
FOUR_PIPE_ASHP_GAS["cooling_sources"] = [
    {"type":"air_cooled_chiller", "preset":"generic_2MW_bank", "name":"Central chiller bank", "capacity_MW":7.0, "n_units":4, "unit_capacity_MW":1.75, "chilled_water_temp_C":6.0},
]

WORKED_SCENARIOS=[GAS_ONLY,ASHP_ONLY,ASHP_PLUS_GAS_PEAK,DATACENTRE_PLUS_BOOSTER,EFW_PLUS_ASHP,FOUR_PIPE_ASHP_GAS]

if __name__ == "__main__":
    from scenarios.scenario_runner import run_scenario, comparison_table
    print(comparison_table([run_scenario(s) for s in WORKED_SCENARIOS]).to_string(index=False))
