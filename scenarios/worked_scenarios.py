"""Five illustrative, editable scenario dictionaries for the UI/API layer.
They deliberately use plain JSON-compatible values only."""
from copy import deepcopy

BASE_BUILDINGS=[
    {"name":"Civic offices","type":"office","floor_area_m2":18000,"connections":1,"connection_year":1,"connection_probability":1.0},
    {"name":"Residential block A","type":"residential_existing","floor_area_m2":24000,"units":320,"connections":320,"connection_year":2,"connection_probability":0.90},
    {"name":"Residential block B","type":"residential_existing","floor_area_m2":18000,"units":240,"connections":240,"connection_year":3,"connection_probability":0.85},
    {"name":"Retail centre","type":"retail","floor_area_m2":9000,"connections":1,"connection_year":1,"connection_probability":0.90},
    {"name":"Hotel","type":"hotel","floor_area_m2":8500,"connections":1,"connection_year":1,"connection_probability":1.0},
    {"name":"Health centre","type":"hospital","floor_area_m2":6000,"connections":1,"connection_year":1,"connection_probability":1.0},
]

COMMON_ECONOMICS = {
    "project_lifetime_years": 40,
    "discount_rate": 0.105,
    "social_discount_rate": 0.035,
    "financial_basis": "real",
    "price_year": 2026,
    "om_rate": 0.01,
    "counterfactual": "individual_gas",
    "price_changes": {
        "electricity_real_rate": 0.0, "gas_real_rate": 0.0,
        "third_party_heat_real_rate": 0.0, "heat_tariff_real_rate": 0.0,
        "cooling_tariff_real_rate": 0.0, "other_opex_real_rate": 0.0,
    },
    "tariffs": {
        "heat_unit_rate_p_per_kWh": 10.0,
        "cooling_unit_rate_p_per_kWh": 12.0,
        "standing_charge_GBP_per_connection_year": 150.0,
    },
    "capex_items": {
        "energy_centre_building_GBP": 2_000_000,
        "land_and_enabling_GBP": 500_000,
        "electricity_connection_GBP": 1_500_000,
        "gas_connection_GBP": 250_000,
        "controls_and_scada_GBP": 400_000,
        "customer_connection_GBP_per_connection": 8_000,
        "metering_GBP_per_connection": 500,
        "development_and_design_pct": 0.10,
        "commissioning_pct": 0.03,
        "contingency_pct": 0.20,
    },
    "annual_opex_items": {
        "billing_and_customer_service_GBP": 60_000,
        "insurance_and_rates_GBP": 120_000,
        "land_lease_GBP": 25_000,
        "water_treatment_GBP": 35_000,
        "operator_overhead_GBP": 100_000,
    },
    "replacement_overrides": {},
}

def base_scenario(name: str) -> dict:
    return {
        "name":name,
        "climate_scenario":"baseline",
        "demand":{"buildings":deepcopy(BASE_BUILDINGS)},
        "network":{"mode":"generic_length","length_m":3000.0,"include_cooling":False,
                   "heat_flow_temp_C":70.0,"heat_return_temp_C":40.0},
        "economics":deepcopy(COMMON_ECONOMICS),
    }

GAS_ONLY=base_scenario("A1 — Gas district reference (not low-carbon)")
GAS_ONLY["description"] = "Common-project technical/economic reference; expected to fail the 100 gCO2e/kWh carbon gate."
GAS_ONLY["sources"]=[{"type":"gas_boiler","preset":"ealing_phase2","name":"Gas boiler","capacity_MW":10.0}]

ASHP_ONLY=base_scenario("A2 — ASHP-only service stress test")
ASHP_ONLY["description"] = "Deliberately tests outage/design-day shortfall; retain only if the service gate passes after resizing or backup is added."
ASHP_ONLY["sources"]=[{"type":"ashp","preset":"ealing_phase2","name":"ASHP bank","capacity_MW":6.5,"flow_temp_C":70.0}]

ASHP_PLUS_GAS_PEAK=base_scenario("A3 — ASHP plus gas peak/backup")
ASHP_PLUS_GAS_PEAK["description"] = "Low-carbon primary plant with gas peak and backup on the common customer and route case."
ASHP_PLUS_GAS_PEAK["sources"]=[
    {"type":"ashp","preset":"ealing_phase1","name":"ASHP bank","capacity_MW":2.8,"flow_temp_C":70.0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":6.5},
]

DATACENTRE_PLUS_BOOSTER=base_scenario("A4 — Data-centre heat plus booster and backup")
DATACENTRE_PLUS_BOOSTER["description"] = "Waste-heat option with hourly source/booster energy coupling and gas backup."
DATACENTRE_PLUS_BOOSTER["sources"]=[
    {"type":"data_centre","preset":"redwire_ealing","name":"Data-centre waste heat","capacity_MW":3.6, "dispatch_direct":False},
    {"type":"booster_heat_pump","preset":"generic_2MW","name":"Booster heat pump","capacity_MW":3.0,"depends_on":0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":7.5},
]

EFW_PLUS_ASHP=base_scenario("A5 — EfW export plus ASHP and backup")
EFW_PLUS_ASHP["description"] = "Third-party heat hybrid; source availability and heat purchase cost are explicit dispatch inputs."
EFW_PLUS_ASHP["sources"]=[
    {"type":"efw_chp","preset":"newlincs_style","name":"EfW heat export","capacity_MW":3.0},
    {"type":"ashp","preset":"ealing_phase1","name":"ASHP bank","capacity_MW":2.8,"flow_temp_C":70.0},
    {"type":"gas_boiler","preset":"ealing_phase1","name":"Peak gas boiler","capacity_MW":5.5},
]

FOUR_PIPE_ASHP_GAS = base_scenario("A6 — Four-pipe heating and cooling")
FOUR_PIPE_ASHP_GAS["description"] = "Common-project heating case extended with central cooling and a separate cooling tariff/counterfactual."
FOUR_PIPE_ASHP_GAS["network"].update({"include_cooling": True, "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0})
FOUR_PIPE_ASHP_GAS["economics"]["counterfactual"] = "individual_gas_and_ac"
FOUR_PIPE_ASHP_GAS["sources"] = [
    {"type":"ashp", "preset":"ealing_phase1", "name":"ASHP bank", "capacity_MW":2.8, "flow_temp_C":70.0},
    {"type":"gas_boiler", "preset":"ealing_phase1", "name":"Peak gas boiler", "capacity_MW":6.5},
]
FOUR_PIPE_ASHP_GAS["cooling_sources"] = [
    {"type":"air_cooled_chiller", "preset":"generic_2MW_bank", "name":"Central chiller bank", "capacity_MW":7.0, "n_units":4, "chilled_water_temp_C":6.0},
]

WORKED_SCENARIOS=[GAS_ONLY,ASHP_ONLY,ASHP_PLUS_GAS_PEAK,DATACENTRE_PLUS_BOOSTER,EFW_PLUS_ASHP,FOUR_PIPE_ASHP_GAS]

if __name__ == "__main__":
    from scenarios.scenario_runner import run_scenario, comparison_table
    print(comparison_table([run_scenario(s) for s in WORKED_SCENARIOS]).to_string(index=False))
