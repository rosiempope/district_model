"""Component + integration tests for the three efficiency-upgrade cooling units.

These check the things that distinguish the new cooling sources from the
air-cooled baseline (chiller.py) and that the rest of the engine relies on:
  - each exposes the SAME public surface as AirCooledChiller, so dispatch,
    build_source_stack, aggregate_capex and the cashflow engine work unchanged;
  - water-cooled uses less electricity than air-cooled (higher wet-bulb COP);
  - free-cooling uses less electricity than air-cooled (compressor off on cold
    hours) but never more;
  - absorption is driven by HEAT, not grid electricity, and is correctly kept
    OUT of the runner's ELECTRIC_SOURCE_TYPES;
  - a full 4-pipe scenario runs end-to-end with each new cooling technology.
"""
import unittest

import numpy as np
import pandas as pd

from components.chiller import AirCooledChiller
from components.water_cooled_chiller import WaterCooledChiller
from components.free_cooling_chiller import FreeCoolingChiller
from components.absorption_chiller import AbsorptionChiller

N = 8760

# Public attributes every cooling source must expose for the engine to use it.
REQUIRED_SURFACE = [
    "capacity_MW", "supply_MW", "cop_hourly", "marginal_cost",
    "carbon_intensity_kgCO2_per_kWh", "electrical_demand_MW", "supply_temp_C",
    "source_type",
]


def _weather(mean_C=11.5, amplitude_C=12.0, rh_pct=75.0):
    """A simple seasonal dry-bulb sinusoid + flat humidity, enough to exercise
    the wet-bulb and free-cooling logic across a realistic UK range."""
    hours = np.arange(N)
    temp = mean_C - amplitude_C * np.cos(2 * np.pi * hours / N)
    return pd.DataFrame(
        {"temp_drybulb_C": temp, "rel_humidity_pct": np.full(N, rh_pct)},
        index=pd.date_range("2021-01-01", periods=N, freq="h"),
    )


class PublicSurfaceTests(unittest.TestCase):
    def test_all_cooling_sources_expose_the_air_cooled_surface(self):
        w = _weather()
        units = [
            AirCooledChiller.from_preset("generic_2MW_bank", w),
            WaterCooledChiller.from_preset("generic_2MW_bank", w),
            FreeCoolingChiller.from_preset("generic_2MW_bank", w),
            AbsorptionChiller.from_preset("generic_2MW_efw", w),
        ]
        for u in units:
            for attr in REQUIRED_SURFACE:
                self.assertTrue(hasattr(u, attr), f"{type(u).__name__} missing .{attr}")
            self.assertEqual(len(np.asarray(u.supply_MW)), N)
            self.assertEqual(len(np.asarray(u.marginal_cost)), N)
            # summary() and resize() must work (used across the pack)
            self.assertIn("source_type", u.summary())
            self.assertEqual(u.resize(n_units=1).n_units, 1)


class EfficiencyOrderingTests(unittest.TestCase):
    def setUp(self):
        self.w = _weather()
        self.ac = AirCooledChiller.from_preset("generic_2MW_bank", self.w)
        self.wc = WaterCooledChiller.from_preset("generic_2MW_bank", self.w)
        self.fc = FreeCoolingChiller.from_preset("generic_2MW_bank", self.w)

    def test_water_cooled_uses_less_electricity_than_air_cooled(self):
        self.assertLess(
            self.wc.electrical_demand_MW.sum(), self.ac.electrical_demand_MW.sum()
        )

    def test_free_cooling_uses_no_more_electricity_than_air_cooled(self):
        # Free cooling can only ever match or beat the mechanical chiller.
        self.assertLessEqual(
            self.fc.electrical_demand_MW.sum(), self.ac.electrical_demand_MW.sum() + 1e-6
        )

    def test_free_cooling_cop_never_below_mechanical(self):
        self.assertTrue(np.all(self.fc.cop_hourly >= self.fc._mech_cop - 1e-9))


class AbsorptionTests(unittest.TestCase):
    def setUp(self):
        self.w = _weather()
        self.ab = AbsorptionChiller.from_preset("generic_2MW_efw", self.w)

    def test_absorption_is_heat_driven_not_electric(self):
        # Driving heat should dwarf parasitic electricity.
        self.assertGreater(self.ab.heat_demand_MW.sum(), 10 * self.ab.electrical_demand_MW.sum())

    def test_absorption_excluded_from_electric_source_types(self):
        from scenarios.scenario_runner import ELECTRIC_SOURCE_TYPES
        self.assertNotIn("absorption_chiller", ELECTRIC_SOURCE_TYPES)
        self.assertIn("water_cooled_chiller", ELECTRIC_SOURCE_TYPES)
        self.assertIn("free_cooling_chiller", ELECTRIC_SOURCE_TYPES)

    def test_absorption_cost_scales_with_heat_price(self):
        cheap = AbsorptionChiller.from_preset("generic_2MW_efw", self.w, heat_price_GBP_per_MWh=5.0)
        dear = AbsorptionChiller.from_preset("generic_2MW_efw", self.w, heat_price_GBP_per_MWh=50.0)
        self.assertLess(cheap.marginal_cost.mean(), dear.marginal_cost.mean())


class FourPipeIntegrationTests(unittest.TestCase):
    """A full 4-pipe scenario must run end-to-end with each new cooling type."""

    def _scenario(self, ctype, preset):
        return {
            "name": f"test-{ctype}", "climate_scenario": "baseline",
            "demand": {"buildings": [
                {"name": "Office", "type": "office_ac", "floor_area_m2": 12000,
                 "connections": 1, "connection_year": 1, "connection_probability": 1.0},
                {"name": "Flats", "type": "residential_existing", "floor_area_m2": 15000,
                 "units": 200, "connections": 200, "connection_year": 1,
                 "connection_probability": 0.9},
            ]},
            "network": {"mode": "generic_length", "length_m": 800,
                        "include_cooling": True, "heat_flow_temp_C": 70.0,
                        "heat_return_temp_C": 40.0, "cool_flow_temp_C": 6.0,
                        "cool_return_temp_C": 12.0},
            "sources": [
                {"type": "efw_chp", "preset": "newlincs_style", "name": "EfW", "capacity_MW": 2.0},
                {"type": "ashp", "preset": "ealing_phase1", "name": "ASHP", "capacity_MW": 2.0},
                {"type": "gas_boiler", "preset": "ealing_phase1", "name": "Gas", "capacity_MW": 2.0},
            ],
            "cooling_sources": [
                {"type": ctype, "preset": preset, "name": "chiller",
                 "capacity_MW": 2.0, "chilled_water_temp_C": 6.0},
            ],
            "economics": {"counterfactual": "individual_gas_and_ac",
                          "project_lifetime_years": 40, "discount_rate": 0.035,
                          "ghnf_grant": {"enabled": True, "rate": 0.40}},
        }

    def test_each_cooling_technology_runs_and_gives_finite_npv_and_lcoe(self):
        from scenarios.scenario_runner import run_scenario
        for ctype, preset in [
            ("air_cooled_chiller", "generic_2MW_bank"),
            ("water_cooled_chiller", "generic_2MW_bank"),
            ("free_cooling_chiller", "generic_2MW_bank"),
            ("absorption_chiller", "generic_2MW_efw"),
        ]:
            r = run_scenario(self._scenario(ctype, preset))
            npv = r["financial"]["investor"]["npv_GBP"]
            lcoe = r["headline"]["levelised_energy_service_GBP_per_kWh"]
            self.assertTrue(np.isfinite(npv), f"{ctype}: NPV not finite")
            self.assertTrue(np.isfinite(lcoe) and lcoe > 0, f"{ctype}: LCOE not positive/finite")


if __name__ == "__main__":
    unittest.main()
