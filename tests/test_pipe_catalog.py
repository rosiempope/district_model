"""Unit tests for network/pipe_catalog.py — hydraulics, sizing, heat loss, cost.

This module had no unit coverage: every test in the suite was integration-level
through run_scenario(), which cannot tell you whether the Darcy-Weisbach
implementation is right, only that the whole chain produces a number.

Tests are written against the module's own CITED anchors wherever one exists
(SEAI cost curve, EN 253 Table 7 casing data, Logstor's twin-pipe range,
standard water property tables) rather than against whatever the code currently
returns, so they can actually catch a regression rather than enshrine one.
"""
import unittest

from network.pipe_catalog import (
    PIPE_COST_GBP_PER_M_AT_REF_DN,
    PIPE_COST_REFERENCE_DN,
    STANDARD_DN_SERIES,
    TWIN_PIPE_MAX_DN,
    casing_to_pipe_ratio_at_dn,
    darcy_friction_factor,
    estimate_pipe_cost_GBP_per_m,
    heat_loss_coefficient_W_per_mK,
    pressure_gradient_Pa_per_m,
    select_pipe,
    size_pipe_for_peak,
    water_properties,
)


class WaterPropertyTests(unittest.TestCase):
    def test_density_and_cp_match_standard_table_at_known_points(self):
        # Standard saturated-liquid water at 1 atm.
        self.assertAlmostEqual(water_properties(20.0)["density_kg_m3"], 998.2, places=1)
        self.assertAlmostEqual(water_properties(80.0)["density_kg_m3"], 971.6, places=1)
        self.assertAlmostEqual(water_properties(20.0)["cp_J_kgK"], 4182.0, places=0)

    def test_viscosity_falls_roughly_fourfold_between_a_chilled_and_a_hot_loop(self):
        """The module's stated reason for temperature-dependent properties: a
        cold loop and a hot loop need genuinely different pipe sizes for the
        same duty, because viscosity changes ~4x across the range."""
        cold = water_properties(8.0)["viscosity_Pa_s"]
        hot = water_properties(80.0)["viscosity_Pa_s"]
        self.assertGreater(cold / hot, 3.0)
        self.assertLess(cold / hot, 5.0)

    def test_out_of_range_temperatures_clamp_rather_than_extrapolate(self):
        self.assertEqual(water_properties(-20.0), water_properties(0.0))
        self.assertEqual(water_properties(150.0), water_properties(100.0))


class FrictionFactorTests(unittest.TestCase):
    def test_laminar_branch_uses_64_over_re(self):
        self.assertAlmostEqual(darcy_friction_factor(1000.0, 0.001), 64.0 / 1000.0, places=6)

    def test_turbulent_branch_is_in_the_physically_sane_band(self):
        f = darcy_friction_factor(1e5, 0.1 / 1000.0 / 0.1)
        self.assertGreater(f, 0.01)
        self.assertLess(f, 0.05)

    def test_rougher_pipe_has_more_friction(self):
        smooth = darcy_friction_factor(1e5, 1e-5)
        rough = darcy_friction_factor(1e5, 1e-2)
        self.assertGreater(rough, smooth)

    def test_zero_flow_does_not_divide_by_zero(self):
        self.assertTrue(darcy_friction_factor(0.0, 0.001) > 0)


class PipeSelectionTests(unittest.TestCase):
    def test_dual_criterion_is_actually_enforced(self):
        """The selected DN must satisfy BOTH velocity and pressure gradient —
        this is the module's headline sizing claim."""
        for peak_kW in (200.0, 1000.0, 7200.0):
            pipe = size_pipe_for_peak(peak_kW, flow_temp_C=70.0, return_temp_C=40.0)
            self.assertLessEqual(pipe.velocity_ms, 2.5, f"velocity at {peak_kW} kW")
            self.assertLessEqual(pipe.pressure_gradient_Pa_per_m, 150.0, f"gradient at {peak_kW} kW")

    def test_selects_the_SMALLEST_dn_that_satisfies_both(self):
        """Not just any compliant DN — the smallest, or the cost curve is
        meaningless."""
        pipe = size_pipe_for_peak(1000.0, flow_temp_C=70.0, return_temp_C=40.0)
        smaller = [dn for dn, _, _ in STANDARD_DN_SERIES if dn < pipe.DN]
        for dn in smaller:
            inner_m = next(i for d, i, _ in STANDARD_DN_SERIES if d == dn) / 1000.0
            props = water_properties(70.0)
            flow = (1000.0 * 1000.0) / (props["cp_J_kgK"] * 30.0) / props["density_kg_m3"]
            dp, v, _ = pressure_gradient_Pa_per_m(
                flow, inner_m, props["density_kg_m3"], props["viscosity_Pa_s"]
            )
            self.assertTrue(
                v > 2.5 or dp > 150.0,
                f"DN{dn} would also have passed — selection is not picking the smallest",
            )

    def test_a_cold_loop_needs_a_bigger_pipe_than_a_hot_loop_for_the_same_kW(self):
        """Cooling runs a much smaller design delta-T, forcing higher mass flow.
        This is the reason the catalog is temperature-aware at all."""
        hot = size_pipe_for_peak(2000.0, flow_temp_C=70.0, return_temp_C=40.0)   # dT 30
        cold = size_pipe_for_peak(2000.0, flow_temp_C=6.0, return_temp_C=12.0)   # dT 6
        self.assertGreater(cold.DN, hot.DN)

    def test_bigger_duty_never_gets_a_smaller_pipe(self):
        last = 0
        for peak_kW in (100, 500, 1000, 3000, 7000, 15000):
            dn = size_pipe_for_peak(float(peak_kW), 70.0, 40.0).DN
            self.assertGreaterEqual(dn, last)
            last = dn

    def test_duty_beyond_the_largest_dn_raises_rather_than_silently_extrapolating(self):
        with self.assertRaises(ValueError):
            size_pipe_for_peak(500_000.0, flow_temp_C=70.0, return_temp_C=40.0)

    def test_near_zero_delta_t_raises(self):
        with self.assertRaises(ValueError):
            size_pipe_for_peak(1000.0, flow_temp_C=70.0, return_temp_C=70.0)

    def test_delta_t_sign_does_not_matter_only_magnitude(self):
        a = size_pipe_for_peak(1000.0, flow_temp_C=40.0, return_temp_C=70.0)
        b = size_pipe_for_peak(1000.0, flow_temp_C=40.0, return_temp_C=10.0)
        self.assertEqual(a.DN, b.DN)

    def test_low_velocity_is_flagged_but_does_not_block_selection(self):
        pipe = select_pipe(flow_m3_s=0.0001, fluid_temp_C=70.0)
        self.assertTrue(pipe.below_min_velocity)
        self.assertGreater(pipe.DN, 0)


class TwinPipeTests(unittest.TestCase):
    def test_twin_is_rejected_above_the_real_logstor_product_range(self):
        """There is no commercial twin-pipe product above DN200; the module
        promises a loud error rather than a plausible number for a product that
        does not exist."""
        too_big = next(dn for dn, _, _ in STANDARD_DN_SERIES if dn > TWIN_PIPE_MAX_DN)
        with self.assertRaises(ValueError):
            estimate_pipe_cost_GBP_per_m(too_big, construction="twin")
        with self.assertRaises(ValueError):
            heat_loss_coefficient_W_per_mK(too_big, construction="twin")

    def test_twin_is_allowed_at_and_below_dn200(self):
        self.assertGreater(estimate_pipe_cost_GBP_per_m(TWIN_PIPE_MAX_DN, construction="twin"), 0)

    def test_twin_costs_more_and_loses_less_than_single(self):
        single_cost = estimate_pipe_cost_GBP_per_m(100, "single")
        twin_cost = estimate_pipe_cost_GBP_per_m(100, "twin")
        self.assertGreater(twin_cost, single_cost)
        self.assertLess(
            heat_loss_coefficient_W_per_mK(100, "twin"),
            heat_loss_coefficient_W_per_mK(100, "single"),
        )

    def test_invalid_construction_raises(self):
        with self.assertRaises(ValueError):
            estimate_pipe_cost_GBP_per_m(100, construction="triple")


class CostCurveTests(unittest.TestCase):
    def test_reference_point_matches_the_seai_fitted_anchor(self):
        self.assertAlmostEqual(
            estimate_pipe_cost_GBP_per_m(PIPE_COST_REFERENCE_DN),
            PIPE_COST_GBP_PER_M_AT_REF_DN,
            places=6,
        )

    def test_cost_rises_with_diameter_but_sublinearly(self):
        """The fitted exponent is 0.426 — flatter than linear, because trenching
        dominates and is largely diameter-independent. If cost ever scales
        linearly or faster with DN, the curve has been broken."""
        c100 = estimate_pipe_cost_GBP_per_m(100)
        c400 = estimate_pipe_cost_GBP_per_m(400)
        self.assertGreater(c400, c100)
        self.assertLess(c400 / c100, 4.0, "cost must not scale linearly with DN")
        self.assertGreater(c400 / c100, 1.5)

    def test_non_positive_dn_raises(self):
        with self.assertRaises(ValueError):
            estimate_pipe_cost_GBP_per_m(0)


class HeatLossTests(unittest.TestCase):
    def test_casing_ratio_is_exact_at_real_table7_points(self):
        """DN20-DN500 interpolates real EN 253 Table 7 data and must be exact at
        the measured points, not merely close."""
        self.assertAlmostEqual(casing_to_pipe_ratio_at_dn(20), 90.0 / 26.9, places=6)
        self.assertAlmostEqual(casing_to_pipe_ratio_at_dn(100), 200.0 / 114.3, places=6)
        self.assertAlmostEqual(casing_to_pipe_ratio_at_dn(500), 630.0 / 508.0, places=6)

    def test_insulation_gets_proportionally_thinner_as_dn_rises(self):
        self.assertGreater(casing_to_pipe_ratio_at_dn(20), casing_to_pipe_ratio_at_dn(500))

    def test_dn600_is_extrapolated_but_stays_physically_sane(self):
        """DN600 is beyond Table 7's ceiling and uses the power-law fit."""
        ratio = casing_to_pipe_ratio_at_dn(600)
        self.assertGreater(ratio, 1.0, "casing must be larger than the pipe")
        self.assertLess(ratio, casing_to_pipe_ratio_at_dn(500))

    def test_loss_coefficient_covers_both_supply_and_return(self):
        """The returned coefficient is per metre of RUN and already includes
        both pipes — callers rely on this and would otherwise double-count."""
        from network.pipe_catalog import DEFAULT_INSULATION_K_W_MK
        import numpy as np

        pipe = next(p for p in STANDARD_DN_SERIES if p[0] == 100)
        d_outer = pipe[2] / 1000.0
        d_casing = d_outer * casing_to_pipe_ratio_at_dn(100)
        r_per_m = np.log(d_casing / d_outer) / (2 * np.pi * DEFAULT_INSULATION_K_W_MK)
        one_pipe = 1.0 / r_per_m
        self.assertAlmostEqual(
            heat_loss_coefficient_W_per_mK(100, "single"), 2.0 * one_pipe, places=6
        )

    def test_bigger_pipes_lose_more_heat_per_metre(self):
        self.assertGreater(
            heat_loss_coefficient_W_per_mK(400), heat_loss_coefficient_W_per_mK(50)
        )

    def test_unknown_insulation_series_raises(self):
        with self.assertRaises(ValueError):
            heat_loss_coefficient_W_per_mK(100, insulation_series="series_2")

    def test_unknown_dn_raises(self):
        with self.assertRaises(ValueError):
            heat_loss_coefficient_W_per_mK(77)


if __name__ == "__main__":
    unittest.main()
