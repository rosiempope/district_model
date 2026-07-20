"""Unit tests for the three COP models: ASHP, air-cooled chiller, booster.

These curves set the electricity bill and therefore the carbon number and the
NPV of every electrified option in the study pack. They had no unit coverage.

Each model cites specific real-world anchors in its own docstring. The tests
below check the code against THOSE anchors — the Ruhnau regression coefficients,
REHVA's EER 4.0 at 35C ambient, the 2.5-3.5 measured range for deployed
data-centre heat recovery, the 2.2-2.7 UK ASHP field-trial band — rather than
against current output. A test that just records what the code does today cannot
fail when the code is wrong.
"""
import unittest

import numpy as np

from components.ASHP import RATING_POINT_TEMP_C, _ashp_cop_base, ashp_cop
from components.booster_heat_pump import (
    CARNOT_EFFICIENCY_FRACTION,
    COP_CEILING,
    COP_FLOOR,
    booster_cop,
)
from components.chiller import chiller_cop
from components.cooling_common import wet_bulb_temp_C
from components.water_cooled_chiller import water_cooled_chiller_cop
from components.free_cooling_chiller import (
    free_cooling_fraction, free_cooling_effective_cop,
)


class ASHPCopTests(unittest.TestCase):
    def test_base_regression_matches_ruhnau_coefficients_exactly(self):
        """COP = 6.08 - 0.09*dT + 0.0005*dT^2, dT = T_flow - T_ambient.
        Ruhnau et al. (2019), Scientific Data 6:189 — also PyPSA-Eur's default."""
        for t_ambient, t_flow in [(7.0, 55.0), (-5.0, 70.0), (15.0, 45.0)]:
            dT = t_flow - t_ambient
            expected = 6.08 - 0.09 * dT + 0.0005 * dT**2
            actual = _ashp_cop_base(np.array([t_ambient]), t_flow)[0]
            self.assertAlmostEqual(actual, expected, places=9)

    def test_cop_falls_as_lift_rises(self):
        cops = [ashp_cop(np.array([t]), 70.0)[0] for t in (15.0, 7.0, 0.0, -10.0)]
        for colder, warmer in zip(cops[1:], cops[:-1]):
            self.assertLess(colder, warmer)

    def test_higher_flow_temperature_costs_cop(self):
        low = ashp_cop(np.array([7.0]), 45.0)[0]
        high = ashp_cop(np.array([7.0]), 70.0)[0]
        self.assertGreater(low, high)

    def test_defrost_penalty_bites_in_the_icing_band(self):
        """0-5C is where coil frost forces defrost cycles. The penalty is the
        reason modelled COP matches UK field trials rather than lab curves."""
        with_defrost = ashp_cop(np.array([2.0]), 70.0, apply_defrost=True)[0]
        without = ashp_cop(np.array([2.0]), 70.0, apply_defrost=False)[0]
        self.assertLess(with_defrost, without)
        self.assertGreater(without / with_defrost, 1.05)   # ~10% derate claimed

    def test_defrost_penalty_is_absent_well_above_freezing(self):
        warm_on = ashp_cop(np.array([15.0]), 70.0, apply_defrost=True)[0]
        warm_off = ashp_cop(np.array([15.0]), 70.0, apply_defrost=False)[0]
        self.assertAlmostEqual(warm_on, warm_off, places=6)

    def test_cop_is_bounded(self):
        freezing = ashp_cop(np.array([-30.0]), 90.0)[0]
        mild = ashp_cop(np.array([20.0]), 30.0)[0]
        self.assertGreaterEqual(freezing, 1.2)   # never worse than resistive
        self.assertLessEqual(mild, 6.0)

    def test_cop_at_the_rating_point_is_plausible_for_a_70C_network(self):
        """A 70C LTHW network ASHP at the EN14825 7C rating point should land in
        the low 2s to low 3s. The Ealing report's own average was 2.88."""
        cop = ashp_cop(np.array([RATING_POINT_TEMP_C]), 70.0)[0]
        self.assertGreater(cop, 2.0)
        self.assertLess(cop, 3.5)

    def test_vectorises_over_a_full_year(self):
        temps = np.linspace(-10.0, 35.0, 8760)
        cops = ashp_cop(temps, 70.0)
        self.assertEqual(cops.shape, (8760,))
        self.assertTrue(np.all(np.isfinite(cops)))


class ChillerCopTests(unittest.TestCase):
    def test_rehva_anchor_eer_4_at_35C_ambient_and_7C_chilled(self):
        """The module's primary cited anchor: EER minimum 4.0 at ~35C ambient,
        the standard AHRI 550/590 rating condition (dT = 28C)."""
        cop = chiller_cop(np.array([35.0]), 7.0)[0]
        self.assertAlmostEqual(cop, 4.0, delta=0.15)

    def test_cold_weather_anchor_matches_the_measured_6_to_7_range(self):
        """Reported EER 6-7 (midpoint 6.75) during Nov-Mar at ~10C ambient."""
        cop = chiller_cop(np.array([10.0]), 7.0)[0]
        self.assertGreater(cop, 5.5)
        self.assertLess(cop, 7.5)

    def test_cop_falls_as_ambient_rises(self):
        cops = [chiller_cop(np.array([t]), 7.0)[0] for t in (10.0, 20.0, 30.0, 40.0)]
        for hotter, cooler in zip(cops[1:], cops[:-1]):
            self.assertLess(hotter, cooler)

    def test_colder_chilled_water_costs_cop(self):
        self.assertGreater(
            chiller_cop(np.array([30.0]), 7.0)[0], chiller_cop(np.array([30.0]), 4.0)[0]
        )

    def test_cop_is_bounded(self):
        self.assertGreaterEqual(chiller_cop(np.array([50.0]), 4.0)[0], 1.5)
        self.assertLessEqual(chiller_cop(np.array([-5.0]), 7.0)[0], 8.0)


class BoosterCopTests(unittest.TestCase):
    def test_carnot_fraction_formula_is_applied_as_documented(self):
        """COP = (T_sink / (T_sink - T_source)) * carnot_efficiency_fraction,
        in KELVIN."""
        source_C, sink_C = 30.0, 65.0
        t_sink_K = sink_C + 273.15
        t_source_K = source_C + 273.15
        expected = (t_sink_K / (t_sink_K - t_source_K)) * CARNOT_EFFICIENCY_FRACTION
        actual = booster_cop(np.array([source_C]), sink_C)[0]
        self.assertAlmostEqual(actual, expected, places=6)

    def test_fitted_against_real_deployed_systems_at_the_midpoint(self):
        """The fraction (0.244) was fitted so 37.5C source -> 65C sink gives
        COP 3.0, the midpoint of the real measured 2.5-3.5 range for deployed
        data-centre heat recovery."""
        cop = booster_cop(np.array([37.5]), 65.0)[0]
        self.assertAlmostEqual(cop, 3.0, delta=0.1)

    def test_warmer_waste_heat_gives_a_better_cop(self):
        """The whole economic case for liquid cooling: a higher source
        temperature is a smaller lift."""
        cool_source = booster_cop(np.array([28.0]), 70.0)[0]
        warm_source = booster_cop(np.array([35.0]), 70.0)[0]
        self.assertGreater(warm_source, cool_source)

    def test_higher_network_temperature_costs_cop(self):
        self.assertGreater(
            booster_cop(np.array([30.0]), 55.0)[0], booster_cop(np.array([30.0]), 80.0)[0]
        )

    def test_cop_is_bounded_at_both_ends(self):
        tiny_lift = booster_cop(np.array([64.0]), 65.0)[0]
        self.assertLessEqual(tiny_lift, COP_CEILING)
        huge_lift = booster_cop(np.array([-40.0]), 95.0)[0]
        self.assertGreaterEqual(huge_lift, COP_FLOOR)

    def test_source_hotter_than_sink_raises_rather_than_returning_nonsense(self):
        """No lift means no booster. Returning a negative or infinite COP here
        would silently corrupt the electricity bill, so this must raise."""
        with self.assertRaises(ValueError):
            booster_cop(np.array([80.0]), 65.0)
        with self.assertRaises(ValueError):
            booster_cop(np.array([65.0]), 65.0)

    def test_accepts_an_hourly_sink_temperature_array(self):
        source = np.full(8760, 30.0)
        sink = np.linspace(55.0, 75.0, 8760)
        cops = booster_cop(source, sink)
        self.assertEqual(cops.shape, (8760,))
        self.assertTrue(np.all(np.isfinite(cops)))


class WetBulbTests(unittest.TestCase):
    def test_wet_bulb_never_exceeds_dry_bulb(self):
        for T in (0.0, 10.0, 20.0, 35.0):
            for RH in (10.0, 50.0, 90.0):
                self.assertLessEqual(wet_bulb_temp_C(np.array([T]), np.array([RH]))[0], T + 1e-9)

    def test_wet_bulb_equals_dry_bulb_at_saturation(self):
        """At ~100% RH the air is saturated, so wet-bulb ≈ dry-bulb."""
        T = 20.0
        wb = wet_bulb_temp_C(np.array([T]), np.array([99.0]))[0]
        self.assertAlmostEqual(wb, T, delta=0.6)

    def test_wet_bulb_matches_known_psychrometric_point(self):
        """25C dry-bulb / 50% RH -> ~18.0C wet-bulb (standard psychrometric chart)."""
        wb = wet_bulb_temp_C(np.array([25.0]), np.array([50.0]))[0]
        self.assertAlmostEqual(wb, 18.0, delta=0.5)


class WaterCooledCopTests(unittest.TestCase):
    def test_design_anchor_cop_58_at_24C_wetbulb(self):
        """Module anchor 1: ~24C wet-bulb / 7C chilled -> COP ~5.8 (dT=17)."""
        self.assertAlmostEqual(water_cooled_chiller_cop(np.array([24.0]), 7.0)[0], 5.8, delta=0.15)

    def test_mild_anchor_cop_8_at_10C_wetbulb(self):
        """Module anchor 2: ~10C wet-bulb / 7C chilled -> COP ~8 (dT=3)."""
        self.assertAlmostEqual(water_cooled_chiller_cop(np.array([10.0]), 7.0)[0], 8.0, delta=0.2)

    def test_water_cooled_beats_air_cooled_at_the_same_conditions(self):
        """The whole point: rejecting to the wet-bulb (below dry-bulb) is a
        smaller lift than rejecting to the dry-bulb, so COP is higher. Compare a
        realistic warm-day pairing (30C dry-bulb, ~21C wet-bulb at 45% RH)."""
        dry = 30.0
        wet = wet_bulb_temp_C(np.array([dry]), np.array([45.0]))[0]
        self.assertLess(wet, dry)
        self.assertGreater(
            water_cooled_chiller_cop(np.array([wet]), 7.0)[0],
            chiller_cop(np.array([dry]), 7.0)[0],
        )


class FreeCoolingTests(unittest.TestCase):
    def test_full_free_cooling_when_cold(self):
        """Below (chilled supply - approach) the dry cooler carries the whole
        load on fans/pumps alone -> free fraction 1."""
        f = free_cooling_fraction(np.array([0.0]), cool_flow_temp_C=7.0, cool_return_temp_C=12.0)[0]
        self.assertEqual(f, 1.0)

    def test_no_free_cooling_when_warm(self):
        f = free_cooling_fraction(np.array([25.0]), cool_flow_temp_C=7.0, cool_return_temp_C=12.0)[0]
        self.assertEqual(f, 0.0)

    def test_effective_cop_spikes_on_cold_hours_and_matches_mechanical_when_hot(self):
        cop_cold, f_cold, _ = free_cooling_effective_cop(
            np.array([0.0]), 7.0, cool_flow_temp_C=7.0, cool_return_temp_C=12.0)
        cop_hot, f_hot, mech_hot = free_cooling_effective_cop(
            np.array([30.0]), 7.0, cool_flow_temp_C=7.0, cool_return_temp_C=12.0)
        self.assertEqual(f_cold[0], 1.0)
        self.assertGreater(cop_cold[0], 15.0)                 # free cooling: fans+pumps only
        self.assertEqual(f_hot[0], 0.0)
        self.assertAlmostEqual(cop_hot[0], mech_hot[0], places=6)  # no free benefit at peak


class CrossModelSanityTests(unittest.TestCase):
    def test_booster_beats_ashp_at_the_same_sink_because_its_source_is_warmer(self):
        """This is the entire premise of data-centre waste-heat recovery: a
        ~30C source is a much smaller lift than a 0C winter air source."""
        ashp_winter = ashp_cop(np.array([0.0]), 70.0)[0]
        booster = booster_cop(np.array([30.0]), 70.0)[0]
        self.assertGreater(booster, ashp_winter)


if __name__ == "__main__":
    unittest.main()
