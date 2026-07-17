"""Unit tests for network/topology_thermal.py.

This decides how much heat the network loses, what temperature arrives at each
customer, and therefore whether the delivered-temperature gate passes — the gate
that stops a scenario banking heat-pump COP by starving customers of hot water.
It had no unit coverage.

Tested against the physics the module cites, not against current output: the
Shukhov exponential, Busby's seasonal ground curve, and the sign behaviour that
makes a cooling duty GAIN heat rather than lose it.
"""
import unittest

import numpy as np

from network.network_topology import NetworkTopology
from network.pipe_catalog import heat_loss_coefficient_W_per_mK, water_properties
from network.topology_thermal import (
    DEFAULT_GROUND_TEMP_C,
    GROUND_TEMP_MEAN_C,
    GROUND_TEMP_SEASONAL_AMPLITUDE_C,
    MIN_DELIVERED_TEMP_C,
    seasonal_ground_temp_C,
    segment_outlet_temp_C,
)

N = 8760


def _tree(peaks=(500.0, 500.0), lengths=(200.0, 300.0)):
    """A small real topology: energy centre -> two branches with real demand."""
    topo = NetworkTopology(name="test")
    topo.add_node("EC", parent_id=None, length_m=0.0, building_name="Energy centre")
    for i, (peak, length) in enumerate(zip(peaks, lengths), start=1):
        topo.add_node(f"N{i}", parent_id="EC", length_m=length,
                      peak_kW=peak, building_name=f"Building {i}")
    topo.validate()
    return topo


class ShukhovTests(unittest.TestCase):
    """T_out = T_ground + (T_in - T_ground) * exp(-U*L / (cp*m_dot))"""

    def test_matches_the_closed_form_exactly(self):
        args = dict(inlet_temp_C=70.0, mass_flow_kg_s=5.0, length_m=250.0,
                    heat_loss_coefficient_W_per_mK=0.4, ground_temp_C=10.0,
                    cp_J_kgK=4186.0)
        expected = 10.0 + (70.0 - 10.0) * np.exp(-(0.4 * 250.0) / (4186.0 * 5.0))
        self.assertAlmostEqual(segment_outlet_temp_C(**args), expected, places=9)

    def test_a_hot_pipe_cools_toward_the_ground_but_never_past_it(self):
        out = segment_outlet_temp_C(70.0, 5.0, 250.0, 0.4, ground_temp_C=10.0)
        self.assertLess(out, 70.0)
        self.assertGreater(out, 10.0)

    def test_a_COLD_pipe_WARMS_toward_the_ground(self):
        """The cooling duty. Same formula, opposite-sign physical effect — chilled
        water gains heat from warmer ground. Getting this wrong once produced a
        negative 'heat loss' for a chiller network."""
        out = segment_outlet_temp_C(6.0, 5.0, 250.0, 0.4, ground_temp_C=12.0)
        self.assertGreater(out, 6.0)
        self.assertLess(out, 12.0)

    def test_a_longer_pipe_loses_more(self):
        short = segment_outlet_temp_C(70.0, 5.0, 100.0, 0.4, ground_temp_C=10.0)
        long = segment_outlet_temp_C(70.0, 5.0, 1000.0, 0.4, ground_temp_C=10.0)
        self.assertGreater(short, long)

    def test_a_faster_flow_loses_LESS_temperature(self):
        """More mass carries the same loss with a smaller temperature drop — which
        is why a trunk arrives hotter than a thin branch of the same length."""
        slow = segment_outlet_temp_C(70.0, 1.0, 500.0, 0.4, ground_temp_C=10.0)
        fast = segment_outlet_temp_C(70.0, 20.0, 500.0, 0.4, ground_temp_C=10.0)
        self.assertGreater(fast, slow)

    def test_zero_flow_raises_rather_than_dividing_by_zero(self):
        with self.assertRaises(ValueError):
            segment_outlet_temp_C(70.0, 0.0, 250.0, 0.4)

    def test_better_insulation_delivers_hotter(self):
        poor = segment_outlet_temp_C(70.0, 5.0, 500.0, 1.0, ground_temp_C=10.0)
        good = segment_outlet_temp_C(70.0, 5.0, 500.0, 0.2, ground_temp_C=10.0)
        self.assertGreater(good, poor)


class SeasonalGroundTests(unittest.TestCase):
    """Busby (2015), 106 UK Met Office stations, ~1 m burial depth."""

    def test_annual_mean_matches_the_sourced_figure(self):
        temps = seasonal_ground_temp_C(np.arange(N))
        self.assertAlmostEqual(float(temps.mean()), GROUND_TEMP_MEAN_C, places=2)

    def test_swing_matches_the_sourced_amplitude(self):
        temps = seasonal_ground_temp_C(np.arange(N))
        self.assertAlmostEqual(
            float(temps.max() - temps.min()), 2 * GROUND_TEMP_SEASONAL_AMPLITUDE_C, places=2
        )

    def test_ground_is_coldest_in_winter_and_warmest_in_summer(self):
        """A fixed annual average understates winter loss — the coldest season,
        when demand and flow temperature are both highest. That is the wrong
        direction for a conservative assessment, which is why this curve exists."""
        temps = seasonal_ground_temp_C(np.arange(N))
        january = temps[: 31 * 24].mean()
        july = temps[181 * 24 : 212 * 24].mean()
        self.assertLess(january, july)

    def test_ground_lags_air_by_about_a_month(self):
        """At 1 m depth the ground peaks ~1 month after the air. The module's air
        convention peaks at hour 4200; ground must peak later."""
        temps = seasonal_ground_temp_C(np.arange(N))
        peak_hour = int(np.argmax(temps))
        self.assertGreater(peak_hour, 4200)
        self.assertLess(peak_hour - 4200, 24 * 45)   # ~1 month, not a season

    def test_the_scalar_default_agrees_with_the_seasonal_mean(self):
        """These are two figures for the same thing; if they drift apart, a
        spot-check and an hourly run silently disagree."""
        self.assertAlmostEqual(DEFAULT_GROUND_TEMP_C, GROUND_TEMP_MEAN_C, places=6)


class DeliveredTemperatureTests(unittest.TestCase):
    def test_delivered_is_below_the_source_and_falls_with_distance(self):
        topo = _tree(peaks=(500.0, 500.0), lengths=(200.0, 2000.0))
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        near = topo.delivered_temperature_C("N1", sized, source_flow_temp_C=70.0)
        far = topo.delivered_temperature_C("N2", sized, source_flow_temp_C=70.0)
        self.assertLess(near, 70.0)
        self.assertLess(far, near)

    def test_a_hotter_source_delivers_hotter_everywhere(self):
        """Monotonic in source temperature — this is what makes the binary search
        in minimum_safe_flow_temp_C() valid."""
        topo = _tree()
        last = -273.0
        for flow in (55.0, 60.0, 65.0, 70.0, 80.0):
            sized = topo.size_all_segments(flow, 40.0, duty="heat")
            d = topo.delivered_temperature_C("N1", sized, source_flow_temp_C=flow)
            self.assertGreater(d, last)
            last = d

    def test_asking_for_a_segment_with_no_pipe_for_this_duty_raises(self):
        """A heating-only branch has no entry in a cooling-duty sizing. That must
        be a clear error, not a KeyError."""
        topo = _tree()
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        with self.assertRaises(KeyError):
            topo.delivered_temperature_C("N1", {}, source_flow_temp_C=70.0)
        self.assertIsNotNone(sized)


class ComplianceGateTests(unittest.TestCase):
    def test_a_short_route_at_70C_passes_the_floor(self):
        topo = _tree(lengths=(200.0, 300.0))
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        chk = topo.check_minimum_delivered_temperature(sized, source_flow_temp_C=70.0)
        self.assertTrue(chk["all_compliant"])
        self.assertEqual(chk["buildings_checked"], 2)
        self.assertGreater(chk["worst_case_delivered_temp_C"], MIN_DELIVERED_TEMP_C)

    def test_a_low_source_temperature_FAILS_the_floor(self):
        """The gate must actually bite, or it is decoration."""
        topo = _tree()
        sized = topo.size_all_segments(45.0, 30.0, duty="heat")
        chk = topo.check_minimum_delivered_temperature(sized, source_flow_temp_C=45.0)
        self.assertFalse(chk["all_compliant"])
        self.assertLess(chk["margin_C"] if "margin_C" in chk else 0.0, 1e9)
        worst = chk["by_building"][chk["worst_case_building"]]
        self.assertLess(worst["margin_C"], 0.0)

    def test_the_worst_case_is_the_coldest_building(self):
        topo = _tree(peaks=(500.0, 500.0), lengths=(200.0, 3000.0))
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        chk = topo.check_minimum_delivered_temperature(sized, source_flow_temp_C=70.0)
        temps = {k: v["delivered_temp_C"] for k, v in chk["by_building"].items()}
        self.assertEqual(chk["worst_case_building"], min(temps, key=temps.get))

    def test_checking_nothing_returns_None_not_True(self):
        """all([]) is True. A topology with no building peaks must NOT report a
        pass having checked nobody — that is a vacuous truth and exactly the trap
        a compliance gate must never fall into."""
        topo = NetworkTopology(name="empty")
        topo.add_node("EC", parent_id=None, length_m=0.0, building_name="Energy centre")
        topo.add_node("J1", parent_id="EC", length_m=100.0, peak_kW=0.0)
        topo.validate()
        chk = topo.check_minimum_delivered_temperature({}, source_flow_temp_C=70.0)
        self.assertIsNone(chk["all_compliant"])
        self.assertEqual(chk["buildings_checked"], 0)


class NetworkHeatLossTests(unittest.TestCase):
    def test_loss_is_reported_as_a_positive_magnitude_for_BOTH_duties(self):
        """abs() is required, not cosmetic: for cooling T_in < T_out, so an
        unsigned subtraction would report a negative 'heat loss'."""
        topo = _tree()
        heat = topo.size_all_segments(70.0, 40.0, duty="heat")
        self.assertGreater(topo.network_heat_loss_kW(heat, 70.0)["total_kW"], 0.0)

        cool_topo = NetworkTopology(name="cool")
        cool_topo.add_node("EC", parent_id=None, length_m=0.0, building_name="Energy centre")
        cool_topo.add_node("N1", parent_id="EC", length_m=300.0,
                           peak_cool_kW=500.0, building_name="B1")
        cool_topo.validate()
        cool = cool_topo.size_all_segments(6.0, 12.0, duty="cool")
        self.assertGreater(cool_topo.network_heat_loss_kW(cool, 6.0)["total_kW"], 0.0)

    def test_hourly_loss_sums_to_the_reported_annual_total(self):
        topo = _tree()
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        r = topo.network_heat_loss_kW_hourly(sized, 70.0)
        self.assertAlmostEqual(
            float(r["total_kW_hourly"].sum()) / 1000.0, r["annual_total_MWh"], places=6
        )
        self.assertEqual(len(r["total_kW_hourly"]), N)

    def test_hourly_loss_is_higher_in_winter_than_summer(self):
        """Ground temperature is the only thing varying, so the seasonal curve
        must show up in the loss."""
        topo = _tree()
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        h = topo.network_heat_loss_kW_hourly(sized, 70.0)["total_kW_hourly"]
        self.assertGreater(h[: 31 * 24].mean(), h[181 * 24 : 212 * 24].mean())

    def test_per_segment_losses_sum_to_the_total(self):
        topo = _tree()
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        r = topo.network_heat_loss_kW(sized, 70.0)
        self.assertAlmostEqual(sum(r["by_segment_kW"].values()), r["total_kW"], places=9)

    def test_wrong_length_ground_temp_array_raises(self):
        topo = _tree()
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        with self.assertRaises(ValueError):
            topo.network_heat_loss_kW_hourly(sized, 70.0, ground_temp_C_hourly=np.ones(100))


class MinimumSafeFlowTempTests(unittest.TestCase):
    def test_it_finds_a_flow_temp_that_just_meets_the_floor(self):
        topo = _tree(lengths=(200.0, 500.0))
        safe = topo.minimum_safe_flow_temp_C(return_temp_C=40.0)
        sized = topo.size_all_segments(safe, 40.0, duty="heat")
        chk = topo.check_minimum_delivered_temperature(sized, source_flow_temp_C=safe)
        self.assertTrue(chk["all_compliant"])
        # And just below it must fail — i.e. it is genuinely the crossover.
        sized_low = topo.size_all_segments(safe - 1.0, 40.0, duty="heat")
        chk_low = topo.check_minimum_delivered_temperature(
            sized_low, source_flow_temp_C=safe - 1.0
        )
        self.assertFalse(chk_low["all_compliant"])

    def test_a_longer_network_needs_a_hotter_source(self):
        short = _tree(lengths=(200.0, 300.0)).minimum_safe_flow_temp_C(return_temp_C=40.0)
        long = _tree(lengths=(200.0, 6000.0)).minimum_safe_flow_temp_C(return_temp_C=40.0)
        self.assertGreater(long, short)

    def test_a_topology_with_no_demand_raises_rather_than_answering(self):
        """'The lowest safe flow temp for a network serving nobody' has no
        meaningful answer, and search_low_C would look like a real, very
        attractive result."""
        topo = NetworkTopology(name="empty")
        topo.add_node("EC", parent_id=None, length_m=0.0, building_name="Energy centre")
        topo.add_node("J1", parent_id="EC", length_m=100.0, peak_kW=0.0)
        topo.validate()
        with self.assertRaises(ValueError):
            topo.minimum_safe_flow_temp_C(return_temp_C=40.0)


class ConsistencyTests(unittest.TestCase):
    def test_loss_and_delivered_temperature_use_the_same_physics(self):
        """A single segment's reported loss must equal m_dot * cp * dT computed
        from the delivered temperature it reports. If these drift apart, the
        energy balance and the compliance gate are telling different stories."""
        topo = _tree(peaks=(500.0,), lengths=(400.0,))
        sized = topo.size_all_segments(70.0, 40.0, duty="heat")
        delivered = topo.delivered_temperature_C("N1", sized, source_flow_temp_C=70.0)
        seg = sized["N1"]
        props = water_properties(70.0)
        expected_kW = seg.mass_flow_kg_s * props["cp_J_kgK"] * (70.0 - delivered) / 1000.0
        reported = topo.network_heat_loss_kW(sized, 70.0)["by_segment_kW"]["N1"]
        self.assertAlmostEqual(reported, expected_kW, delta=0.01)
        self.assertGreater(
            heat_loss_coefficient_W_per_mK(seg.pipe.DN, seg.pipe.construction), 0.0
        )


if __name__ == "__main__":
    unittest.main()
