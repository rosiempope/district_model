"""Unit tests for optimisation/auto_size.py.

Auto-sizing decides how many MW of each technology a scenario buys, so it sets
plant CAPEX, the low-carbon share of heat, and whether the service gate passes
at all. Every study that does not hand-pick capacities goes through it. It had no
unit coverage.

The rules under test are the ones a reader would otherwise have to take on trust
from the module docstring — baseload-first, cold-weather derating, no double
diversity, a network-loss margin, and a gas boiler added automatically when no
peak plant was asked for.
"""
import unittest

import numpy as np
import pandas as pd

from optimisation.auto_size import (
    ASHP_DESIGN_DAY_DERATING,
    DEFAULT_BASELOAD_FRACTION,
    DIVERSITY_FACTORS,
    _round_capacity,
    _sensible_ashp_unit_size,
    recommend_sizing,
)

N = 8760


def _weather(mean_C=11.5, amplitude_C=12.0):
    hours = np.arange(N)
    temp = mean_C - amplitude_C * np.cos(2 * np.pi * (hours - 400) / N)
    return pd.DataFrame(
        {"temp_drybulb_C": temp},
        index=pd.date_range("2021-01-01", periods=N, freq="h"),
    )


def _peaky_profile(peak_kW=5000.0, base_kW=500.0):
    """A realistic-shaped heating year: high in winter, low in summer."""
    hours = np.arange(N)
    shape = 0.5 * (1 + np.cos(2 * np.pi * hours / N))   # 1 at hour 0, 0 mid-year
    return base_kW + (peak_kW - base_kW) * shape


def _by_type(rec, t):
    return next((s for s in rec["sources"] if s["type"] == t), None)


class BaseloadFirstTests(unittest.TestCase):
    def test_efw_baseload_is_capped_at_50pct_of_peak_and_rounded_up(self):
        """EfW is a flat baseload machine: the module caps it at 50% of the
        design peak even when a higher baseload fraction is asked for, and rounds
        up to a 0.5 MW step so the UI does not show a spuriously precise number.
        Both rules are easy to miss by reading the headline baseload_fraction."""
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["efw_chp", "gas_boiler"], weather_df=_weather(),
            baseload_fraction=0.80,   # deliberately above the cap
        )
        efw = _by_type(rec, "efw_chp")
        self.assertIsNotNone(efw)
        self.assertEqual(efw["role"], "baseload")
        capped = rec["diversified_peak_kW"] / 1000.0 * 0.50
        self.assertAlmostEqual(efw["capacity_MW"], _round_capacity(capped, 0.5), delta=1e-6)
        # The cap must actually bite: 0.80 must not buy 80%.
        self.assertLess(efw["capacity_MW"],
                        rec["diversified_peak_kW"] / 1000.0 * 0.80)

    def test_peak_plant_covers_the_rest(self):
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["efw_chp", "gas_boiler"], weather_df=_weather(),
        )
        gas = _by_type(rec, "gas_boiler")
        self.assertIsNotNone(gas)
        self.assertEqual(gas["role"], "peak")
        total = sum(s["capacity_MW"] for s in rec["sources"])
        self.assertGreaterEqual(total + 1e-6, rec["diversified_peak_kW"] / 1000.0)

    def test_a_bigger_baseload_fraction_buys_more_baseload_and_less_peak(self):
        """Uses ASHP, not EfW — EfW's 50% cap would mask the effect above 0.50."""
        demand = _peaky_profile()
        kw = dict(demand_kW=demand, peak_demand_kW=demand.max(),
                  technology_types=["ashp", "gas_boiler"], weather_df=_weather())
        low = recommend_sizing(baseload_fraction=0.30, **kw)
        high = recommend_sizing(baseload_fraction=0.70, **kw)
        self.assertGreater(_by_type(high, "ashp")["capacity_MW"],
                           _by_type(low, "ashp")["capacity_MW"])
        self.assertLess(_by_type(high, "gas_boiler")["capacity_MW"],
                        _by_type(low, "gas_boiler")["capacity_MW"])


class ASHPDeratingTests(unittest.TestCase):
    def test_ashp_nameplate_is_uprated_for_the_design_day(self):
        """An ASHP's nameplate is quoted at the 7C rating point but it only
        delivers ~65% of that on a -5C design day. Sizing to nameplate would
        leave the scheme short exactly when it is coldest."""
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
        )
        ashp = _by_type(rec, "ashp")
        required_MW = rec["diversified_peak_kW"] / 1000.0 * DEFAULT_BASELOAD_FRACTION
        nameplate_MW = required_MW / ASHP_DESIGN_DAY_DERATING
        # Then rounded to a whole number of sensibly-sized units — you cannot buy
        # 4.85 MW of heat pump, you buy 7 x 0.7 MW.
        unit = _sensible_ashp_unit_size(nameplate_MW)
        expected = max(1, round(nameplate_MW / unit)) * unit
        self.assertAlmostEqual(ashp["capacity_MW"], round(expected, 2), delta=1e-6)
        self.assertEqual(ashp["n_units"], max(1, round(nameplate_MW / unit)))
        # i.e. genuinely bigger than the un-derated figure
        self.assertGreater(ashp["capacity_MW"], required_MW)

    def test_ashp_is_bought_in_whole_units_of_a_real_size(self):
        """You cannot buy 4.85 MW of heat pump. Capacity must be a whole number
        of units, each one of the sizes the module offers."""
        valid_unit_sizes = {0.1, 0.5, 0.7, 1.0, 2.0}
        for peak in (400.0, 1500.0, 5000.0, 20000.0):
            d = _peaky_profile(peak_kW=peak, base_kW=peak * 0.1)
            rec = recommend_sizing(
                demand_kW=d, peak_demand_kW=d.max(),
                technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
            )
            ashp = _by_type(rec, "ashp")
            self.assertGreaterEqual(ashp["n_units"], 1)
            unit_MW = ashp["capacity_MW"] / ashp["n_units"]
            self.assertTrue(
                any(abs(unit_MW - u) < 1e-6 for u in valid_unit_sizes),
                f"peak {peak} kW gave a {unit_MW:.4f} MW unit, which is not a size "
                f"the module offers ({sorted(valid_unit_sizes)})",
            )

    def test_an_ashp_is_sized_bigger_than_an_efw_for_the_same_duty(self):
        demand = _peaky_profile()
        kw = dict(demand_kW=demand, peak_demand_kW=demand.max(), weather_df=_weather())
        ashp = recommend_sizing(technology_types=["ashp", "gas_boiler"], **kw)
        efw = recommend_sizing(technology_types=["efw_chp", "gas_boiler"], **kw)
        self.assertGreater(_by_type(ashp, "ashp")["capacity_MW"],
                           _by_type(efw, "efw_chp")["capacity_MW"])


class DiversityTests(unittest.TestCase):
    def test_an_hourly_aggregate_profile_gets_NO_second_diversity_factor(self):
        """The module's own fix: an hourly aggregate already contains the
        coincidence between buildings. Applying a diversity factor on top of it
        undersized plant.

        NOTE the building mix. With the default n_buildings=1 the chosen
        diversity factor is 1.00 anyway, so this test would pass even if
        peak_is_coincident were ignored entirely — it could not fail, and a
        mutation proved it. A real multi-building mix is required for the
        assertion to mean anything.

        The result's "diversity_factor" field reports what was APPLIED, not what
        was chosen, so it is 1.0 here by design and cannot be used to show the
        mix picked something lower. The paired non-coincident run below does that.
        """
        demand = _peaky_profile()
        mix = dict(n_buildings=6,
                   building_types=["office", "residential", "retail",
                                   "office", "residential", "hotel"])
        kw = dict(demand_kW=demand, peak_demand_kW=demand.max(),
                  technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
                  network_loss_margin=0.0, **mix)

        # Same mix, told the peak is an arithmetic SUM: diversity must apply.
        summed = recommend_sizing(peak_is_coincident=False, **kw)
        self.assertAlmostEqual(summed["diversity_factor"],
                               DIVERSITY_FACTORS["mixed_use"], places=6)
        self.assertAlmostEqual(
            summed["diversified_peak_kW"],
            demand.max() * DIVERSITY_FACTORS["mixed_use"], delta=1.0,
        )

        # Same mix, told the peak is already coincident: it must NOT.
        coincident = recommend_sizing(peak_is_coincident=True, **kw)
        self.assertAlmostEqual(coincident["diversified_peak_kW"], demand.max(), delta=1.0)
        self.assertGreater(coincident["diversified_peak_kW"], summed["diversified_peak_kW"])

    def test_diversity_applies_only_to_an_arithmetic_sum_of_peaks(self):
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
            peak_is_coincident=False, diversity_factor=0.85, network_loss_margin=0.0,
        )
        self.assertAlmostEqual(rec["diversified_peak_kW"], demand.max() * 0.85, delta=1.0)

    def test_diversity_is_chosen_from_the_building_mix(self):
        demand = _peaky_profile()
        kw = dict(demand_kW=demand, peak_demand_kW=demand.max(),
                  technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
                  peak_is_coincident=False, network_loss_margin=0.0)
        resi = recommend_sizing(
            n_buildings=5, building_types=["residential"] * 5, **kw)
        single = recommend_sizing(n_buildings=1, building_types=["office"], **kw)
        self.assertAlmostEqual(resi["diversity_factor"],
                               DIVERSITY_FACTORS["residential_only"], places=6)
        # A single building cannot diversify against itself.
        self.assertAlmostEqual(single["diversity_factor"],
                               DIVERSITY_FACTORS["single_building"], places=6)


class NetworkLossMarginTests(unittest.TestCase):
    def test_plant_is_sized_for_the_network_loss_on_top_of_demand(self):
        """The network itself consumes heat. Sizing to building demand alone
        leaves nothing to carry it."""
        demand = _peaky_profile()
        kw = dict(demand_kW=demand, peak_demand_kW=demand.max(),
                  technology_types=["ashp", "gas_boiler"], weather_df=_weather())
        none = recommend_sizing(network_loss_margin=0.0, **kw)
        with_loss = recommend_sizing(network_loss_margin=0.10, **kw)
        self.assertAlmostEqual(
            with_loss["diversified_peak_kW"], none["diversified_peak_kW"] * 1.10, delta=1.0
        )


class PeakPlantTests(unittest.TestCase):
    def test_a_gas_boiler_is_added_when_no_peak_plant_was_requested(self):
        """Asking for baseload only leaves the design-day peak uncovered. The
        module adds a boiler and SAYS so rather than silently under-sizing."""
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp"], weather_df=_weather(),
        )
        self.assertIsNotNone(_by_type(rec, "gas_boiler"))
        self.assertTrue(
            any("gas boiler" in n.lower() for n in rec["sizing_notes"]),
            f"the automatic addition must be stated; got {rec['sizing_notes']}",
        )

    def test_ashp_is_never_treated_as_peak_plant(self):
        """BASELOAD_TYPES/PEAK_TYPES are disjoint — an ASHP is never a backup
        candidate. This is why a genuinely gas-free stack needs a manual step
        (see analysis/source_stack_comparison_common.py)."""
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp"], weather_df=_weather(),
        )
        self.assertEqual(_by_type(rec, "ashp")["role"], "baseload")


class SizingNotesTests(unittest.TestCase):
    def test_every_recommendation_explains_itself(self):
        demand = _peaky_profile()
        rec = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
        )
        self.assertTrue(rec["sizing_notes"])
        for s in rec["sources"]:
            self.assertTrue(s.get("rationale"), f"{s['type']} has no rationale")


class CoolingTests(unittest.TestCase):
    def test_cooling_is_sized_only_when_asked_for(self):
        demand = _peaky_profile()
        cooling = _peaky_profile(peak_kW=2000.0, base_kW=100.0)[::-1]
        without = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
        )
        with_c = recommend_sizing(
            demand_kW=demand, peak_demand_kW=demand.max(),
            technology_types=["ashp", "gas_boiler"], weather_df=_weather(),
            include_cooling=True, cooling_demand_kW=cooling,
            peak_cooling_kW=cooling.max(),
        )
        self.assertFalse(without.get("cooling_sources"))
        self.assertTrue(with_c["cooling_sources"])
        self.assertGreater(with_c["cooling_sources"][0]["capacity_MW"], 0.0)


class MonotonicityTests(unittest.TestCase):
    def test_more_peak_demand_never_buys_less_plant(self):
        w = _weather()
        last = 0.0
        for peak in (2000.0, 4000.0, 8000.0, 16000.0):
            d = _peaky_profile(peak_kW=peak, base_kW=peak * 0.1)
            rec = recommend_sizing(
                demand_kW=d, peak_demand_kW=d.max(),
                technology_types=["ashp", "gas_boiler"], weather_df=w,
            )
            total = sum(s["capacity_MW"] for s in rec["sources"])
            self.assertGreaterEqual(total, last)
            last = total


if __name__ == "__main__":
    unittest.main()
