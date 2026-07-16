"""Ealing Town Centre feasibility report Phase 1 validation case.

Source: Ealing Town Centre Heat Network Feasibility Report (June 2025),
Tables 11, 14-19 and 39-48. Rounded Table 11 customer energy is scaled to
the exact Table 16 Phase 1 end-customer total (14,161,194 kWh).
"""
from copy import deepcopy


_TABLE_11 = [
    ("CP House", "mixed_use", 2088, 1349, 291.30),
    ("Perceval House Car Park", "mixed_use", 772, 276, 200.20),
    ("Perceval House", "office", 1245, 964, 188.30),
    ("Ealing Town Hall", "hotel", 1136, 640, 142.60),
    ("Ealing Green College", "school", 380, 243, 49.90),
    ("Christ the Saviour CoE School", "school", 469, 489, 19.30),
    ("Sandringham Mews", "residential", 903, 242, 365.40),
    ("Christ the Saviour Church", "mixed_use", 156, 158, 33.80),
    ("The Arcadia Centre", "retail", 735, 431, 98.70),
    ("Marks and Spencer", "retail", 614, 355, 80.00),
    ("Christ the Saviour CoE School - Groves Site", "school", 132, 128, 26.00),
    ("Ealing Broadway", "retail", 4065, 2327, 495.80),
    ("Broadway Connection", "mixed_use", 1011, 415, 41.60),
    ("Bakers House and Wells House", "residential_existing", 663, 327, 87.00),
]
_ANNUAL_SCALE = 14_161.194 / sum(row[2] for row in _TABLE_11)


EALING_PHASE1_BUILDINGS = [
    {
        "name": name,
        "type": building_type,
        "annual_heat_kWh": annual_MWh * _ANNUAL_SCALE * 1000.0,
        "annual_dhw_kWh": 0.0,
        "annual_cool_kWh": 0.0,
        "peak_total_heat_kW": peak_kW,
        "connections": 1,
        "connection_year": 1,
        "connection_probability": 1.0,
        "heat_unit_rate_p_per_kWh": 9.56,
        "standing_charge_GBP_per_connection_year": fixed_GBP_day * 365.0,
        "connection_charge_GBP_per_kW": 600.0,
    }
    for name, building_type, annual_MWh, peak_kW, fixed_GBP_day in _TABLE_11
]


EALING_PHASE1_VALIDATION = {
    "name": "Ealing report validation - Phase 1",
    "description": "Direct calibration to the June 2025 feasibility report Phase 1 tables.",
    "climate_scenario": "baseline",
    "demand": {
        "buildings": EALING_PHASE1_BUILDINGS,
        # Table 14 peak includes the report losses; the runner adds the
        # 974.614 MWh loss as a flat 111.3 kW screening approximation.
        "aggregate_peak_heat_kW": 7_190.0 - 974_614.0 / 8_760.0,
        # Inferred from the published load-duration curve (Figure 23). The
        # underlying 8,760 report workbook was not supplied.
        "aggregate_load_shape_sharpness": 1.10,
    },
    "network": {
        "mode": "generic_length",
        "length_m": 2_148.0,
        "include_cooling": False,
        "heat_flow_temp_C": 70.0,
        "heat_return_temp_C": 40.0,
        # Table 16: 888,041 kWh network + 86,573 kWh building losses.
        "annual_heat_loss_MWh_override": 974.614,
        # Table 48: spine, additional network, feeds and substations.
        "capex_GBP_override": 10_461_831.0,
    },
    "sources": [
        {
            "type": "ashp", "preset": "ealing_phase1", "name": "Phase 1 ASHP bank",
            "capacity_MW": 2.8, "n_units": 4, "flow_temp_C": 70.0,
            "electricity_price_GBP_per_MWh": 180.5,
            # Table 48 heat pump + rooftop exchangers/civils + HP M&E.
            "capex_GBP_per_MW": 4_670_000.0 / 2.8,
            "availability_factor": 0.97,
            # The report quotes useful output capacity from its manufacturer
            # curves rather than EN14825 nameplate derating.
            "min_capacity_fraction": 1.0,
            "apply_defrost": False,
            # Calibrates the public weather curve to Figure 24's 2.88 average
            # CoP; the proprietary manufacturer performance table is absent.
            "cop_calibration_factor": 1.176,
        },
        {
            "type": "gas_boiler", "preset": "ealing_phase1", "name": "Peak and reserve boilers",
            "capacity_MW": 3.6, "eta_full_load": 0.90,
            "gas_price_GBP_per_MWh": 46.9,
            # Table 48 boiler plant plus flues.
            "capex_GBP_per_MW": 494_400.0 / 3.6,
        },
    ],
    "thermal_storage": {
        "enabled": True,
        "name": "Phase 1 thermal store",
        "volume_litres": 50_000.0,
        "delta_T_K": 45.0,
        "max_charge_MW": 2.0,
        "max_discharge_MW": 2.0,
        "round_trip_efficiency": 0.95,
        "initial_soc_fraction": 1.0,
        "dispatch_strategy": "peak_reserve",
        "capex_GBP": 165_000.0,
    },
    "economics": {
        "project_lifetime_years": 40,
        "discount_rate": 0.035,
        "social_discount_rate": 0.035,
        "financial_basis": "real",
        "base_year": 2030,
        "price_year": 2025,
        "om_rate": 0.01,
        "counterfactual": "individual_gas",
        "price_changes": {
            "electricity_real_rate": 0.0, "gas_real_rate": 0.0,
            "third_party_heat_real_rate": 0.0, "heat_tariff_real_rate": 0.0,
            "cooling_tariff_real_rate": 0.0, "other_opex_real_rate": 0.0,
        },
        "tariffs": {
            "heat_tariff_mode": "manual",
            "heat_unit_rate_p_per_kWh": 9.56,
            "cooling_unit_rate_p_per_kWh": 0.0,
            "standing_charge_GBP_per_connection_year": 0.0,
        },
        "capex_items": {
            # This scenario reproduces the published report's own CAPEX line by
            # line, and the report's total already contains its customer
            # connections. Pricing them again from DECC components would
            # double-count and break the validation — so the connection build-up
            # is switched off here rather than zeroed by accident.
            "connection_cost_mode": "flat_per_connection",
            "customer_connection_GBP_per_connection": 0.0,
            "metering_GBP_per_connection": 0.0,
            "further_project_development_GBP": 925_572.0,
            "contractor_preliminaries_and_design_GBP": 1_295_800.0,
            "construction_insurance_GBP": 61_088.0,
            "energy_centre_building_GBP": 2_070_000.0,
            "pressurisation_and_water_treatment_GBP": 60_500.0,
            "main_network_pumps_GBP": 146_300.0,
            "controls_and_scada_GBP": 378_000.0,
            "other_energy_centre_me_GBP": 558_000.0,
            "gas_connection_GBP": 253_000.0,
            "electricity_connection_GBP": 95_700.0,
            "development_and_design_pct": 0.0,
            "commissioning_pct": 0.0,
            "contingency_pct": 0.0,
        },
        "annual_opex_items": {
            # The public PDF describes these categories but does not disclose
            # the individual values. Zero keeps this discrepancy explicit.
            "billing_and_customer_service_GBP": 0.0,
            "insurance_and_rates_GBP": 0.0,
            "land_lease_GBP": 0.0,
            "water_treatment_GBP": 0.0,
            "operator_overhead_GBP": 0.0,
            # Calibration residual required to reproduce Table 17 after the
            # disclosed fuel, tariff, CAPEX and replacement inputs are used.
            # The PDF names staff, insurance, monitoring and maintenance but
            # does not publish their individual annual values.
            "report_undisclosed_fixed_opex_GBP": 143_465.0,
        },
        "replacement_overrides": {
            "ashp": {"interval_years": 20, "capex_fraction": 0.50},
            "gas_boiler": {"interval_years": 30, "capex_fraction": 1.00},
        },
        # Figure 24 total energy-centre parasitic electricity.
        "total_parasitic_electricity_MWh_override": 302.716,
        "parasitic_electricity_price_GBP_per_MWh": 180.5,
        # Table 44 IAG commercial factor for the 2030 start year and Table 45.
        "carbon_factors": {
            "electricity_kgCO2_per_kWh": 0.091,
            "gas_kgCO2_per_kWh": 0.1839,
        },
    },
}


REPORT_PHASE1_TARGETS = {
    "end_customer_heat_MWh": 14_161.194,
    "heat_including_losses_MWh": 15_135.808,
    "peak_heat_MW": 7.190,
    "ashp_capacity_MW": 2.8,
    "boiler_capacity_MW": 3.6,
    "ashp_generation_MWh": 13_474.122,
    "boiler_generation_MWh": 1_661.687,
    "capex_GBP": 21_635_190.0,
    "investor_irr_40y": 0.026,
    "investor_npv_40y_GBP": -2_249_115.0,
    "first_year_carbon_g_per_kWh": 56.0,
}


def scenario_copy():
    return deepcopy(EALING_PHASE1_VALIDATION)
