"""Unit tests for profiles/demand_synthesis.py.

Demand is the first number in the chain: get it wrong and every MW, every
tonne of CO2 and every pound downstream is wrong too. It had no unit coverage.

The cooling model is the highest-risk piece here. Its docstring records two real
bugs it was written to fix — a ~94x peak-to-mean ratio, and a ~47%
over-allocation from max()-ing two separately-normalised parts — so these tests
pin the properties those fixes were supposed to establish, including the one
that is still not fully true (see CoolingAnnualTotalTests).
"""
import unittest

import numpy as np
import pandas as pd

from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import (
    BUILDING_TYPES,
    _cooling_profile,
    _dhw_profile,
    _heating_profile,
    _make_occupancy,
    _resolve_annual_demands,
    compute_climate_reference,
    synthesise_building,
    synthesise_network,
)

N = 8760


def _weather(mean_C=11.5, amplitude_C=12.0):
    """A smooth synthetic year — deterministic, no file dependency.

    The default amplitude is chosen so the year genuinely crosses the 20C
    cooling-degree base (peaking ~23.5C, like a real London year), otherwise
    every cooling test would silently exercise only the zero-CDD branch.
    """
    hours = np.arange(N)
    temp = mean_C - amplitude_C * np.cos(2 * np.pi * (hours - 400) / N)
    return pd.DataFrame(
        {"temp_drybulb_C": temp},
        index=pd.date_range("2021-01-01", periods=N, freq="h"),
    )


class OccupancyTests(unittest.TestCase):
    def test_every_archetype_schedule_is_a_valid_fraction(self):
        for btype, spec in BUILDING_TYPES.items():
            occ = _make_occupancy(spec["occupancy"])
            self.assertEqual(len(occ), N, btype)
            self.assertTrue(np.all(occ >= 0.0) and np.all(occ <= 1.0), btype)

    def test_office_is_empty_at_night_and_busy_on_a_weekday_afternoon(self):
        occ = _make_occupancy("office")
        self.assertEqual(occ[3], 0.0)      # 03:00 Friday
        self.assertEqual(occ[14], 1.0)     # 14:00 Friday

    def test_school_is_empty_during_the_summer_holidays(self):
        occ = _make_occupancy("school")
        august_midday = 220 * 24 + 12
        self.assertEqual(occ[august_midday], 0.0)

    def test_always_on_is_always_on(self):
        self.assertTrue(np.all(_make_occupancy("always_on") == 1.0))


class HeatingProfileTests(unittest.TestCase):
    def test_annual_total_matches_the_benchmark_when_self_referenced(self):
        w = _weather()
        occ = _make_occupancy("office")
        p = _heating_profile(w["temp_drybulb_C"].values, 500_000.0, occ, 0.15)
        self.assertAlmostEqual(p.sum(), 500_000.0, delta=1.0)

    def test_no_heat_is_produced_above_the_base_temperature(self):
        """HDD-driven: an hour warmer than 15.5C contributes zero heating."""
        w = _weather()
        T = w["temp_drybulb_C"].values
        occ = _make_occupancy("office")
        p = _heating_profile(T, 500_000.0, occ, 0.15, heat_base_C=15.5)
        self.assertTrue(np.all(p[T >= 15.5] == 0.0))

    def test_base_load_keeps_fabric_loss_running_when_unoccupied(self):
        """An empty building in January still loses heat through its fabric."""
        w = _weather()
        T = w["temp_drybulb_C"].values
        occ = _make_occupancy("office")
        p = _heating_profile(T, 500_000.0, occ, base_load_frac=0.15)
        cold_and_empty = (T < 5.0) & (occ == 0.0)
        self.assertTrue(np.any(cold_and_empty))
        self.assertTrue(np.all(p[cold_and_empty] > 0.0))

    def test_a_milder_year_genuinely_uses_less_heat_not_the_same_reshuffled(self):
        """The whole point of the shared climate reference. Without it, a 2050
        scenario rescales back to the fixed CIBSE benchmark and shows identical
        annual heat — hiding the effect being studied."""
        baseline = _weather(mean_C=11.0)
        warmer = _weather(mean_C=14.0)
        ref = compute_climate_reference(baseline)
        occ = _make_occupancy("office")
        cold = _heating_profile(
            baseline["temp_drybulb_C"].values, 500_000.0, occ, 0.15,
            reference_annual_HDD_h=ref["annual_HDD_h"],
        )
        mild = _heating_profile(
            warmer["temp_drybulb_C"].values, 500_000.0, occ, 0.15,
            reference_annual_HDD_h=ref["annual_HDD_h"],
        )
        self.assertLess(mild.sum(), cold.sum() * 0.85)

    def test_baseline_against_its_own_reference_nets_out_to_a_factor_of_one(self):
        w = _weather()
        ref = compute_climate_reference(w)
        occ = _make_occupancy("office")
        p = _heating_profile(
            w["temp_drybulb_C"].values, 500_000.0, occ, 0.15,
            reference_annual_HDD_h=ref["annual_HDD_h"],
        )
        self.assertAlmostEqual(p.sum(), 500_000.0, delta=1.0)


class CoolingAnnualTotalTests(unittest.TestCase):
    """The cooling model's own docstring claims it sums 'EXACTLY' to the annual
    target. It does not, and these tests pin the real behaviour.

    Part 3 is applied as np.maximum(part_1 + part_2, comfort_floor). Wherever the
    comfort floor binds, it adds energy on top of the fully-allocated budget — so
    the annual total lands ABOVE target. On a real London office year the excess
    is ~9-10%. That is far better than the ~47% of the max()-of-two-normalised-
    parts version it replaced, but 'EXACTLY' is not accurate.

    These tests bound the overshoot rather than asserting equality, so the model
    is free to be corrected but cannot silently drift back toward 47%.
    """

    def test_cooling_total_is_at_least_the_target(self):
        w = _weather()
        occ = _make_occupancy("office")
        p = _cooling_profile(w["temp_drybulb_C"].values, 540_000.0, occ, 0.15)
        self.assertGreaterEqual(p.sum(), 540_000.0 * 0.999)

    def test_cooling_overshoot_stays_within_15_percent(self):
        """Documents the known Part-3 floor overshoot and caps it. If this fails
        upward, the comfort floor has started double-counting again."""
        w = _weather()
        occ = _make_occupancy("office")
        p = _cooling_profile(w["temp_drybulb_C"].values, 540_000.0, occ, 0.15)
        self.assertLess(
            p.sum() / 540_000.0, 1.15,
            "comfort floor is adding more energy than the documented margin",
        )

    def test_peak_to_mean_stays_realistic(self):
        """The internal-gains floor exists to kill the old ~94x peak-to-mean
        artefact that pure CDD scaling structurally produces."""
        w = _weather()
        occ = _make_occupancy("office")
        p = _cooling_profile(w["temp_drybulb_C"].values, 540_000.0, occ, 0.15)
        self.assertLess(p.max() / p.mean(), 40.0)

    def test_internal_gains_floor_runs_in_cool_weather_when_occupied(self):
        """Real commercial cooling is substantially internal-gains driven — it
        does not vanish just because it is mild outside."""
        w = _weather()
        T = w["temp_drybulb_C"].values
        occ = _make_occupancy("office")
        p = _cooling_profile(T, 540_000.0, occ, 0.15, internal_gains_fraction=0.65)
        mild_and_occupied = (T < 15.0) & (occ == 1.0)
        self.assertTrue(np.any(mild_and_occupied))
        self.assertTrue(np.all(p[mild_and_occupied] > 0.0))

    def test_a_higher_internal_gains_fraction_flattens_the_profile(self):
        w = _weather()
        occ = _make_occupancy("office")
        peaky = _cooling_profile(w["temp_drybulb_C"].values, 540_000.0, occ, 0.15,
                                 internal_gains_fraction=0.1)
        flat = _cooling_profile(w["temp_drybulb_C"].values, 540_000.0, occ, 0.15,
                                internal_gains_fraction=0.9)
        self.assertLess(flat.max() / flat.mean(), peaky.max() / peaky.mean())

    def test_zero_cooling_demand_gives_a_zero_profile(self):
        w = _weather()
        occ = _make_occupancy("office")
        p = _cooling_profile(w["temp_drybulb_C"].values, 0.0, occ, 0.15)
        self.assertTrue(np.all(p == 0.0))

    def test_a_cool_baseline_reference_does_not_produce_nan_cooling(self):
        """Regression: the CDD guard tested THIS year's cooling degree-hours but
        the ratio divided by the REFERENCE year's. A baseline weather year that
        never reaches cool_base_C (a cool-climate EPW, or a higher cool_base_C)
        gave reference_annual_CDD_h == 0, so any warmer scenario divided by zero
        and produced NaN — which then propagated silently through cooling
        demand, dispatch, OPEX and NPV rather than failing loudly."""
        cool_baseline = _weather(mean_C=8.0, amplitude_C=6.0)     # never exceeds 20C
        self.assertEqual(compute_climate_reference(cool_baseline)["annual_CDD_h"], 0.0)
        ref = compute_climate_reference(cool_baseline)
        warmer = _weather(mean_C=14.0, amplitude_C=12.0)          # genuinely has CDD
        occ = _make_occupancy("office")
        p = _cooling_profile(
            warmer["temp_drybulb_C"].values, 540_000.0, occ, 0.15,
            reference_annual_CDD_h=ref["annual_CDD_h"],
        )
        self.assertTrue(np.all(np.isfinite(p)), "cooling profile contains NaN/inf")
        self.assertGreater(p.sum(), 0.0)

    def test_a_hotter_year_needs_more_cooling(self):
        baseline = _weather(mean_C=11.0)
        hotter = _weather(mean_C=16.0)
        ref = compute_climate_reference(baseline)
        occ = _make_occupancy("office")
        cool_year = _cooling_profile(
            baseline["temp_drybulb_C"].values, 540_000.0, occ, 0.15,
            reference_annual_CDD_h=ref["annual_CDD_h"],
        )
        hot_year = _cooling_profile(
            hotter["temp_drybulb_C"].values, 540_000.0, occ, 0.15,
            reference_annual_CDD_h=ref["annual_CDD_h"],
        )
        self.assertGreater(hot_year.sum(), cool_year.sum())


class DHWProfileTests(unittest.TestCase):
    def test_annual_total_is_met_exactly(self):
        p = _dhw_profile(100_000.0)
        self.assertAlmostEqual(p.sum(), 100_000.0, delta=0.5)

    def test_dhw_is_not_weather_driven(self):
        """People shower regardless of the outside temperature."""
        self.assertTrue(np.allclose(_dhw_profile(100_000.0), _dhw_profile(100_000.0)))

    def test_winter_dhw_exceeds_summer_dhw(self):
        """Colder inlet water needs more energy to reach setpoint."""
        p = _dhw_profile(100_000.0)
        january = p[: 31 * 24].sum()
        july = p[181 * 24 : 212 * 24].sum()
        self.assertGreater(january, july)

    def test_overnight_base_load_never_reaches_zero(self):
        """Legionella cycling continues when nobody is awake."""
        p = _dhw_profile(100_000.0, occupancy=_make_occupancy("office"))
        self.assertTrue(np.all(p > 0.0))


class AnnualDemandResolverTests(unittest.TestCase):
    def test_floor_area_drives_the_archetype_estimate(self):
        heat, cool, dhw = _resolve_annual_demands(
            {"name": "B", "type": "office", "floor_area_m2": 1000.0}
        )
        self.assertAlmostEqual(heat, BUILDING_TYPES["office"]["heat_kWh_m2"] * 1000.0)
        self.assertAlmostEqual(cool, BUILDING_TYPES["office"]["cool_kWh_m2"] * 1000.0)
        self.assertAlmostEqual(dhw, BUILDING_TYPES["office"]["dhw_kWh_m2"] * 1000.0)

    def test_measured_values_win_over_the_archetype(self):
        heat, _, _ = _resolve_annual_demands(
            {"name": "B", "type": "office", "floor_area_m2": 1000.0,
             "annual_heat_kWh": 42_000.0}
        )
        self.assertEqual(heat, 42_000.0)

    def test_measured_heat_alone_is_enough_without_a_floor_area(self):
        """A heat-meter total should run a screen even when floor area is
        unknown — and must not invent an unmetered cooling load from nothing."""
        heat, cool, dhw = _resolve_annual_demands(
            {"name": "B", "type": "office", "annual_heat_kWh": 42_000.0}
        )
        self.assertEqual(heat, 42_000.0)
        self.assertEqual(cool, 0.0)
        self.assertEqual(dhw, 0.0)

    def test_explicit_zero_is_honoured_not_treated_as_missing(self):
        _, cool, _ = _resolve_annual_demands(
            {"name": "B", "type": "office", "floor_area_m2": 1000.0, "annual_cool_kWh": 0.0}
        )
        self.assertEqual(cool, 0.0)

    def test_dwelling_count_uses_the_75_m2_assumption(self):
        heat, _, _ = _resolve_annual_demands(
            {"name": "B", "type": "residential", "units": 10}
        )
        self.assertAlmostEqual(heat, BUILDING_TYPES["residential"]["heat_kWh_m2"] * 750.0)

    def test_no_scale_and_no_measurement_raises(self):
        with self.assertRaises(ValueError):
            _resolve_annual_demands({"name": "B", "type": "office"})

    def test_negative_annual_energy_raises(self):
        with self.assertRaises(ValueError):
            _resolve_annual_demands(
                {"name": "B", "type": "office", "annual_heat_kWh": -1.0}
            )

    def test_unknown_building_type_raises(self):
        with self.assertRaises(ValueError):
            _resolve_annual_demands({"name": "B", "type": "casino", "floor_area_m2": 100.0})


class PeakCalibrationTests(unittest.TestCase):
    def test_measured_peak_is_matched_while_annual_is_preserved(self):
        w = _weather()
        building = {"name": "B", "type": "office", "floor_area_m2": 5000.0,
                    "peak_total_heat_kW": 400.0}
        node = synthesise_building(w, building)
        self.assertAlmostEqual(node["total_heat_kW"].max(), 400.0, delta=1.0)
        expected_annual = (
            BUILDING_TYPES["office"]["heat_kWh_m2"] + BUILDING_TYPES["office"]["dhw_kWh_m2"]
        ) * 5000.0
        self.assertAlmostEqual(
            node["annual_heat_kWh"] + node["annual_dhw_kWh"], expected_annual, delta=5.0
        )

    def test_a_peak_below_the_annual_average_is_impossible_and_raises(self):
        w = _weather()
        with self.assertRaises(ValueError):
            synthesise_building(
                w, {"name": "B", "type": "office", "floor_area_m2": 5000.0,
                    "peak_total_heat_kW": 1.0}
            )


class NetworkAggregationTests(unittest.TestCase):
    def test_totals_are_the_sum_of_the_nodes(self):
        w = _weather()
        result = synthesise_network(w, {"demand_nodes": [
            {"name": "A", "type": "office", "floor_area_m2": 5000.0},
            {"name": "B", "type": "residential", "units": 50},
        ]})
        self.assertAlmostEqual(
            result["annual_heat_MWh"],
            sum(n["annual_heat_kWh"] for n in result["nodes"]) / 1000.0,
            places=6,
        )
        self.assertTrue(np.allclose(
            result["total_heat_kW"],
            sum(n["total_heat_kW"] for n in result["nodes"]),
        ))

    def test_network_peak_is_at_or_below_the_sum_of_individual_peaks(self):
        """Diversity: buildings do not all peak in the same hour."""
        w = _weather()
        result = synthesise_network(w, {"demand_nodes": [
            {"name": "A", "type": "office", "floor_area_m2": 5000.0},
            {"name": "B", "type": "residential", "units": 50},
            {"name": "C", "type": "school", "floor_area_m2": 3000.0},
        ]})
        sum_of_peaks = sum(float(n["total_heat_kW"].max()) for n in result["nodes"])
        self.assertLessEqual(result["peak_heat_kW"], sum_of_peaks + 1e-6)

    def test_empty_node_list_raises(self):
        with self.assertRaises(ValueError):
            synthesise_network(_weather(), {"demand_nodes": []})

    def test_wrong_length_weather_raises(self):
        short = _weather().iloc[:100]
        with self.assertRaises(ValueError):
            synthesise_building(short, {"name": "B", "type": "office", "floor_area_m2": 100.0})


class ClimateScenarioIntegrationTests(unittest.TestCase):
    def test_warmer_climate_reduces_heat_and_increases_cooling(self):
        w = _weather()
        ref = compute_climate_reference(apply_climate_scenario(w, "baseline"))
        demand = {"demand_nodes": [{"name": "A", "type": "office_ac", "floor_area_m2": 10_000.0}]}
        base = synthesise_network(apply_climate_scenario(w, "baseline"), demand,
                                  climate_reference=ref)
        high = synthesise_network(apply_climate_scenario(w, "2050_high"), demand,
                                  climate_reference=ref)
        self.assertLess(high["annual_heat_MWh"], base["annual_heat_MWh"])
        self.assertGreater(high["annual_cool_MWh"], base["annual_cool_MWh"])


if __name__ == "__main__":
    unittest.main()
