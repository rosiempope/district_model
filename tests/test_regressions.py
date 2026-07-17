import copy
import unittest

import numpy as np

from economics.cashflow import build_cashflow
from economics.grant import apply_ghnf_grant
from economics.metrics import (counterfactual_gas_boiler_dispatch,
                               counterfactual_individual_ashp_dispatch)
from economics.tariffs import OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY
from optimisation.auto_size import recommend_sizing
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import compute_climate_reference, synthesise_network
from scenarios.scenario_runner import build_heat_sources, load_weather, run_scenario
from scenarios.scenario_schema import validate_scenario
from scenarios.worked_scenarios import DATACENTRE_PLUS_BOOSTER, GAS_ONLY, WORKED_SCENARIOS
from scenarios.ealing_report_validation import scenario_copy as ealing_scenario


class CapexAdderBaseTests(unittest.TestCase):
    """The design/commissioning/contingency base.

    These adders used to apply to plant + network only, silently exempting the
    energy centre, utility connections, controls, customer connections and
    metering — the majority of the fixed scope. See EXCLUDED_FROM_CAPEX_ADDERS
    and the adder_base_capex note in scenario_runner.run_scenario().
    """

    def _scenario(self):
        scenario = copy.deepcopy(GAS_ONLY)
        scenario["economics"]["capex_items"] = {
            "energy_centre_building_GBP": 1_000_000.0,
            "land_and_enabling_GBP": 500_000.0,
            "electricity_connection_GBP": 0.0,
            "gas_connection_GBP": 0.0,
            "controls_and_scada_GBP": 0.0,
            # These tests isolate the ADDER BASE, so connections are priced on the
            # flat basis where the rate below is the only input. The
            # by_building_type build-up would add a DECC-derived cost that varies
            # with each building's peak and drown the thing being tested.
            "connection_cost_mode": "flat_per_connection",
            "customer_connection_GBP_per_connection": 0.0,
            "metering_GBP_per_connection": 0.0,
            "development_and_design_pct": 0.0,
            "commissioning_pct": 0.0,
            "contingency_pct": 0.10,
        }
        return scenario

    def test_contingency_covers_fixed_build_items_not_only_plant_and_network(self):
        breakdown = run_scenario(self._scenario())["headline"]["capex_breakdown_GBP"]
        plant_and_network = breakdown["sources_GBP"] + breakdown["network_GBP"]
        # The energy centre building must be inside the contingency base.
        self.assertAlmostEqual(
            breakdown["contingency_GBP"],
            0.10 * (plant_and_network + 1_000_000.0),
            delta=1.0,
        )
        self.assertGreater(breakdown["contingency_GBP"], 0.10 * plant_and_network)

    def test_land_is_excluded_from_the_adders(self):
        """Land is a transaction, not a designed/constructed scope — a tenfold
        land price must not move contingency by a penny."""
        breakdown = run_scenario(self._scenario())["headline"]["capex_breakdown_GBP"]
        richer = self._scenario()
        richer["economics"]["capex_items"]["land_and_enabling_GBP"] = 5_000_000.0
        richer_breakdown = run_scenario(richer)["headline"]["capex_breakdown_GBP"]
        self.assertAlmostEqual(
            richer_breakdown["contingency_GBP"], breakdown["contingency_GBP"], delta=1.0
        )

    def test_adders_scale_with_connection_count(self):
        """Customer connections are the biggest fixed line and the likeliest to
        overrun; contingency must track them."""
        scenario = self._scenario()
        scenario["economics"]["capex_items"]["customer_connection_GBP_per_connection"] = 8_000.0
        breakdown = run_scenario(scenario)["headline"]["capex_breakdown_GBP"]
        baseline = run_scenario(self._scenario())["headline"]["capex_breakdown_GBP"]
        self.assertAlmostEqual(
            breakdown["contingency_GBP"] - baseline["contingency_GBP"],
            0.10 * breakdown["customer_connections_GBP"],
            delta=1.0,
        )

    def test_capex_total_still_equals_sum_of_breakdown(self):
        result = run_scenario(self._scenario())["headline"]
        self.assertAlmostEqual(
            result["capex_total_GBP"], sum(result["capex_breakdown_GBP"].values()), delta=1.0
        )


class FinanceRegressionTests(unittest.TestCase):
    def test_npv_is_final_discounted_cash_position(self):
        result = build_cashflow(
            life_years=4, discount_rate=0.10,
            capex={"plant": 1000}, revenues={"sales": [0, 400, 420, 440, 460]},
            opex={"operations": [0, 100, 105, 110, 115]},
            repex={"replacement": [0, 0, 0, 50, 0]},
        )
        self.assertAlmostEqual(result["npv_GBP"], result["cumulative_discounted_GBP"][-1], places=2)
        self.assertEqual(result["annual_table"][0]["net_cashflow_GBP"], -1000)

    def test_carrier_escalation_is_not_applied_to_every_cost(self):
        scenario = copy.deepcopy(GAS_ONLY)
        scenario["economics"]["price_changes"] = {
            "gas_real_rate": 0.10, "electricity_real_rate": 0.0,
            "third_party_heat_real_rate": 0.0, "heat_tariff_real_rate": 0.0,
            "cooling_tariff_real_rate": 0.0, "other_opex_real_rate": 0.0,
        }
        for building in scenario["demand"]["buildings"]:
            building["connection_year"] = 1
            building["connection_probability"] = 1.0
        result = run_scenario(scenario)["financial"]["investor"]["line_items"]["opex"]
        gas = result["gas energy"]
        pumping = result["electricity energy"]
        self.assertAlmostEqual(gas[2] / gas[1], 1.10, places=6)
        self.assertAlmostEqual(pumping[2] / pumping[1], 1.00, places=6)

    def test_worked_scenario_cashflows_reconcile(self):
        for scenario in WORKED_SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                result = run_scenario(scenario)
                investor = result["financial"]["investor"]
                self.assertAlmostEqual(
                    investor["npv_GBP"], investor["cumulative_discounted_GBP"][-1], places=2
                )
                self.assertAlmostEqual(result["headline"]["heat_energy_balance_residual_MWh"], 0.0, places=5)

    def test_repex_appears_only_on_schedule(self):
        result = run_scenario(GAS_ONLY)["financial"]["investor"]["line_items"]["repex"]
        gas = result["Gas boiler"]
        self.assertEqual(gas[19], 0.0)
        self.assertGreater(gas[20], 0.0)
        self.assertEqual(gas[21], 0.0)

    def test_ealing_report_finance_and_irr_root(self):
        result = run_scenario(ealing_scenario())
        investor = result["financial"]["investor"]
        self.assertAlmostEqual(investor["npv_GBP"], -2_249_115, delta=100)
        self.assertAlmostEqual(investor["irr"], 0.026, delta=0.001)

    def test_ghnf_output_cap_is_enforced(self):
        result = apply_ghnf_grant(
            10_000_000, 5_000_000, 5_000_000, 0.49,
            annual_thermal_delivered_kWh=1_000_000,
        )
        self.assertEqual(result["grant_GBP"], 675_000)

    def test_screening_uses_scenario_hurdle_and_same_service_gate(self):
        scenario = copy.deepcopy(GAS_ONLY)
        scenario["economics"]["tariffs"]["heat_tariff_mode"] = "manual"
        scenario["economics"]["tariffs"]["heat_unit_rate_p_per_kWh"] = 30.0
        scenario["screening"] = {
            "maximum_unmet_energy_fraction": 0.001,
            "maximum_carbon_gCO2e_per_kWh": 250.0,
            "investor_hurdle_rate": 0.0,
            "minimum_investor_npv_GBP": -20_000_000,
            "require_n_minus_one": False,
            "maximum_required_heat_tariff_p_per_kWh": None,
        }
        result = run_scenario(scenario)
        gates = {gate["name"]: gate for gate in result["screening"]["gates"]}
        self.assertEqual(gates["Investor IRR"]["threshold"], 0.0)
        self.assertTrue(gates["Investor IRR"]["passed"])
        self.assertEqual(gates["Heat and cooling service"]["passed"], result["headline"]["service_compliant"])
        self.assertTrue(gates["Carbon intensity"]["passed"])
        self.assertFalse(gates["N-1 peak capacity"]["required"])

    def test_counterfactual_gas_standing_charge_is_per_connection(self):
        node = {
            "name": "Residential block", "connections": 10,
            "total_heat_kW": np.ones(8760) * 10.0,
        }
        ten = counterfactual_gas_boiler_dispatch(node)
        one = counterfactual_gas_boiler_dispatch({**node, "connections": 1})
        expected = 9 * OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY * 365 / 100
        # Test the standing charge itself, not the whole bill. The bill now has
        # TWO per-connection terms — the standing charge and the boiler lifecycle
        # (10 flats have 10 boilers to service and replace) — so differencing the
        # total would silently measure both and this test would be asserting that
        # the standing charge is the ONLY per-connection cost, which it is not.
        self.assertAlmostEqual(
            ten["annual_standing_charge_GBP"] - one["annual_standing_charge_GBP"],
            expected, delta=2.0,
        )

    def test_counterfactual_boiler_lifecycle_is_per_connection_and_optional(self):
        """A block of 10 flats has 10 boilers to service and replace, not one."""
        node = {
            "name": "Residential block", "connections": 10,
            "total_heat_kW": np.ones(8760) * 10.0,
        }
        ten = counterfactual_gas_boiler_dispatch(node)
        one = counterfactual_gas_boiler_dispatch({**node, "connections": 1})
        # delta accommodates rounding: the TOTAL is rounded to the pound, so
        # 10 x £416.67 = £4,167 rather than 10 x £417 = £4,170.
        self.assertAlmostEqual(
            ten["annual_boiler_lifecycle_GBP"],
            10 * one["annual_boiler_lifecycle_GBP"], delta=10.0,
        )
        # And it must be switchable off, because it moves the parity revenue and
        # the strict fuel-only comparison has to stay reproducible.
        off = counterfactual_gas_boiler_dispatch(node, include_boiler_lifecycle=False)
        self.assertEqual(off["annual_boiler_lifecycle_GBP"], 0.0)
        self.assertGreater(
            ten["annual_customer_bill_GBP"], off["annual_customer_bill_GBP"]
        )

    def test_boiler_lifecycle_hits_small_dwellings_hardest_per_kWh(self):
        """DECC's central point: the fixed cost of owning a boiler is spread over
        a small heat demand, so it dominates exactly the dwellings heat networks
        serve. DECC: 1.1 p/kWh for a large dwelling, 4.6 p/kWh for a small one."""
        big = counterfactual_gas_boiler_dispatch(
            {"name": "big", "connections": 1, "total_heat_kW": np.ones(8760) * 2.5}
        )
        small = counterfactual_gas_boiler_dispatch(
            {"name": "small", "connections": 1, "total_heat_kW": np.ones(8760) * 0.9}
        )
        big_p = big["annual_boiler_lifecycle_GBP"] / (2.5 * 8760) * 100
        small_p = small["annual_boiler_lifecycle_GBP"] / (0.9 * 8760) * 100
        self.assertGreater(small_p, big_p * 2.0)

    def test_bus_grant_respects_per_building_eligibility_flag(self):
        """Social housing and most new-build homes are excluded from BUS
        regardless of capacity, so a building marked bus_eligible: false must
        get no grant even when every installation is under the 45 kWth cap."""
        weather = load_weather()
        node = {
            "name": "Social housing block", "connections": 10,
            "total_heat_kW": np.ones(8760) * 20.0,   # 2 kW per installation
        }
        eligible = counterfactual_individual_ashp_dispatch(node, weather)
        excluded = counterfactual_individual_ashp_dispatch(
            {**node, "bus_eligible": False}, weather
        )
        self.assertGreater(eligible["bus_grant_GBP"], 0.0)
        self.assertTrue(eligible["bus_eligible"])
        self.assertEqual(excluded["bus_grant_GBP"], 0.0)
        self.assertFalse(excluded["bus_eligible"])
        self.assertAlmostEqual(
            excluded["capex_GBP"] - eligible["capex_GBP"],
            eligible["bus_grant_GBP"], delta=1.0,
        )

    def test_hp_lifecycle_and_bus_reach_the_parity_bill(self):
        """The DECC principle applied to the heat-pump side: the parity bill
        must include the heat pump's own service and (BUS-netted) replacement
        cost, exactly as the gas bill includes the boiler lifecycle. Without
        it, the customer's biggest avoided cost never touched the tariff."""
        weather = load_weather()
        node = {
            "name": "Residential block", "connections": 10,
            "total_heat_kW": np.ones(8760) * 20.0,
        }
        on = counterfactual_individual_ashp_dispatch(node, weather)
        off = counterfactual_individual_ashp_dispatch(
            node, weather, include_hp_lifecycle=False
        )
        self.assertEqual(off["annual_hp_lifecycle_GBP"], 0.0)
        self.assertGreater(
            on["annual_customer_bill_GBP"], off["annual_customer_bill_GBP"]
        )
        # BUS makes the customer's alternative cheaper, so the BUS-eligible
        # bill must be LOWER than the social-housing (excluded) bill, by the
        # grant spread over the heat pump's life.
        no_bus = counterfactual_individual_ashp_dispatch(
            {**node, "bus_eligible": False}, weather
        )
        self.assertLess(
            on["annual_customer_bill_GBP"], no_bus["annual_customer_bill_GBP"]
        )
        self.assertAlmostEqual(
            no_bus["annual_hp_lifecycle_GBP"] - on["annual_hp_lifecycle_GBP"],
            on["bus_grant_GBP"] / 20.0, delta=1.0,
        )

    def test_gas_bill_parity_uses_the_same_customer_counterfactual(self):
        result = run_scenario(GAS_ONLY)
        investor = result["financial"]["investor"]
        self.assertEqual(investor["heat_tariff_mode"], "counterfactual_bill_parity")
        self.assertAlmostEqual(investor["year1_customer_bill_ratio"], 1.0, places=8)
        gates = {gate["name"]: gate for gate in result["screening"]["gates"]}
        self.assertTrue(gates["Customer heat-bill parity"]["passed"])

    def test_opex_reconciliation_closes_without_double_counting(self):
        result = run_scenario(GAS_ONLY)
        audit = result["financial"]["opex_reconciliation"]
        self.assertAlmostEqual(audit["full_buildout_reconciliation_residual_GBP"], 0.0, places=6)
        self.assertAlmostEqual(
            audit["full_buildout_total_opex_GBP"],
            result["headline"]["annual_total_opex_GBP"],
            delta=1.0,
        )


class EngineeringRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_weather = load_weather()

    def test_capacity_mw_means_total_for_unit_assets(self):
        configs = [{
            "type": "ashp", "preset": "ealing_phase1", "name": "test",
            "capacity_MW": 3.0, "n_units": 6, "flow_temp_C": 70,
        }]
        source = build_heat_sources(configs, self.raw_weather, 70)[0]
        self.assertAlmostEqual(source.capacity_MW, 3.0)
        self.assertAlmostEqual(source.unit_capacity_MW, 0.5)

    def test_preset_unit_count_is_retained_when_only_total_capacity_is_overridden(self):
        configs = [{
            "type": "ashp", "preset": "ealing_phase1", "name": "test",
            "capacity_MW": 2.8, "flow_temp_C": 70,
        }]
        source = build_heat_sources(configs, self.raw_weather, 70)[0]
        self.assertEqual(source.n_units, 4)
        self.assertAlmostEqual(source.unit_capacity_MW, 0.7)

    def test_explicit_zero_cooling_is_respected(self):
        demand = synthesise_network(self.raw_weather, {"demand_nodes": [{
            "name": "No cooling", "type": "office", "floor_area_m2": 1000,
            "annual_cool_kWh": 0,
        }]})
        self.assertAlmostEqual(demand["annual_cool_MWh"], 0.0, places=8)

    def test_measured_annual_heat_runs_without_floor_area_or_units(self):
        demand = synthesise_network(self.raw_weather, {"demand_nodes": [{
            "name": "Metered anchor", "type": "hospital",
            "annual_heat_kWh": 2_500_000,
        }]})
        self.assertAlmostEqual(
            demand["annual_heat_MWh"] + demand["annual_dhw_MWh"],
            2_500.0, places=6,
        )
        self.assertAlmostEqual(demand["annual_cool_MWh"], 0.0, places=8)
        self.assertAlmostEqual(demand["annual_dhw_MWh"], 0.0, places=8)

    def test_floor_area_estimates_services_not_explicitly_overridden(self):
        demand = synthesise_network(self.raw_weather, {"demand_nodes": [{
            "name": "Part-metered office", "type": "office",
            "floor_area_m2": 1_000, "annual_heat_kWh": 50_000,
        }]})
        self.assertAlmostEqual(demand["annual_heat_MWh"], 50.0, places=6)
        self.assertGreater(demand["annual_cool_MWh"], 0.0)
        self.assertGreater(demand["annual_dhw_MWh"], 0.0)

    def test_validator_accepts_measured_heat_as_the_only_demand_scale(self):
        scenario = ealing_scenario()
        for building in scenario["demand"]["buildings"]:
            building.pop("floor_area_m2", None)
            building.pop("units", None)
            building.pop("annual_cool_kWh", None)
            building.pop("annual_dhw_kWh", None)
        self.assertEqual(validate_scenario(scenario), [])
        result = run_scenario(scenario)["headline"]
        self.assertAlmostEqual(result["annual_heat_demand_MWh"], 14_161.2, delta=0.2)

    def test_warmer_climate_reduces_annual_space_heat(self):
        baseline = apply_climate_scenario(self.raw_weather, "baseline")
        reference = compute_climate_reference(baseline)
        building = {"name": "Office", "type": "office", "floor_area_m2": 1000}
        today = synthesise_network(
            baseline, {"demand_nodes": [building]}, climate_reference=reference
        )
        future = synthesise_network(
            apply_climate_scenario(self.raw_weather, "2050_high"),
            {"demand_nodes": [building]}, climate_reference=reference,
        )
        self.assertLess(future["annual_heat_MWh"], today["annual_heat_MWh"])

    def test_auto_size_does_not_double_apply_diversity(self):
        demand = np.ones(8760) * 1000
        rec = recommend_sizing(
            demand, 1000, ["gas_boiler"], n_buildings=10,
            building_types=["office"] * 10, network_loss_margin=0.0,
        )
        self.assertEqual(rec["diversity_factor"], 1.0)
        self.assertEqual(rec["diversified_peak_kW"], 1000.0)

    def test_no_network_has_zero_network_cost_loss_and_pumping(self):
        scenario = copy.deepcopy(GAS_ONLY)
        scenario["network"] = {
            "mode": "none", "include_cooling": False,
            "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
        }
        result = run_scenario(scenario)["headline"]
        self.assertEqual(result["capex_network_GBP"], 0.0)
        self.assertEqual(result["annual_network_heat_loss_MWh"], 0.0)
        self.assertEqual(result["annual_pumping_electricity_MWh"], 0.0)

    def test_booster_is_limited_by_recoverable_source_heat(self):
        result = run_scenario(DATACENTRE_PLUS_BOOSTER)
        booster = next(s for s in result["heat_sources"] if s.source_type == "booster_heat_pump")
        source_fraction = 1.0 - 1.0 / booster.cop_hourly
        self.assertTrue(np.all(booster.supply_MW * source_fraction <= booster.source_heat_available_MW + 1e-9))
        self.assertTrue(np.all(booster.supply_MW[booster.source_heat_available_MW == 0] == 0))

    def test_carbon_unit_is_kg_per_kwh_not_kg_per_mwh(self):
        result = run_scenario(GAS_ONLY)["headline"]
        self.assertGreater(result["carbon_intensity_kgCO2_per_kWh_service"], 0.15)
        self.assertLess(result["carbon_intensity_kgCO2_per_kWh_service"], 0.30)

    def test_adding_a_source_cannot_increase_unmet_heat(self):
        baseline = copy.deepcopy(GAS_ONLY)
        base_unmet = run_scenario(baseline)["headline"]["annual_unmet_demand_MWh"]
        added = copy.deepcopy(baseline)
        added["sources"].insert(0, {
            "type": "ashp", "preset": "ealing_phase1", "name": "Additional ASHP",
            "capacity_MW": 2.8, "n_units": 4, "flow_temp_C": 70,
        })
        added_unmet = run_scenario(added)["headline"]["annual_unmet_demand_MWh"]
        self.assertLessEqual(added_unmet, base_unmet + 1e-9)

    def test_ealing_report_case_has_zero_unmet_heat(self):
        result = run_scenario(ealing_scenario())["headline"]
        self.assertEqual(result["annual_unmet_demand_MWh"], 0.0)
        self.assertEqual(result["peak_unmet_MW"], 0.0)


if __name__ == "__main__":
    unittest.main()
