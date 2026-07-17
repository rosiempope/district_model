"""Unit tests for optimisation/dispatch.py.

Dispatch decides how much each source runs in each of 8,760 hours, so it sets
every OPEX figure, every carbon figure and every unmet-energy gate in the model.
It had no unit coverage — every test reached it through run_scenario(), which can
tell you the chain produces a number but not whether the merit order is right.

These use tiny synthetic sources with hand-checkable costs, so a failure points
at the dispatch logic rather than at demand synthesis or a component's physics.
"""
import unittest

import numpy as np

from components.thermal_storage import ThermalStorage
from optimisation.dispatch import run_dispatch

N = 8760


class _Source:
    """Minimal stand-in exposing exactly the contract dispatch relies on.

    Deliberately not a real component: this isolates the merit-order logic from
    ASHP/boiler physics, so these tests fail for dispatch reasons only.
    """

    def __init__(self, name, capacity_MW, marginal_cost, source_type="ashp",
                 carbon=0.05, available=None):
        self.name = name
        self.capacity_MW = float(capacity_MW)
        self.source_type = source_type
        self.n_units = 1
        self.capex_GBP_per_MW = 1e6
        self.supply_MW = (
            np.full(N, float(capacity_MW)) if available is None
            else np.asarray(available, dtype=float)
        )
        self.marginal_cost = np.full(N, float(marginal_cost))
        self.carbon_intensity_kgCO2_per_kWh = np.full(N, float(carbon))
        self.supply_temp_C = np.full(N, 70.0)


def _Store(capacity_MWh=10.0, rate_MW=5.0, eff=1.0):
    """The REAL ThermalStorage, not a fake.

    A fake would have to reimplement charge/discharge/standing-loss/reset, and
    then these tests would be checking the fake. Storage is small enough to use
    directly, and using it means the dispatch/storage INTERFACE is under test
    too — which is where a break would actually happen.
    """
    return ThermalStorage(
        name="Store", capacity_MWh=capacity_MWh,
        max_charge_MW=rate_MW, max_discharge_MW=rate_MW,
        round_trip_efficiency=eff, standing_loss_pct_per_hour=0.0,
        initial_soc_fraction=0.5, delta_T_K=30.0,
    )


class EnergyBalanceTests(unittest.TestCase):
    """The invariant everything else rests on: energy in equals energy out."""

    def test_supply_plus_unmet_equals_demand_every_hour(self):
        demand = np.random.default_rng(0).uniform(0, 8, N)
        r = run_dispatch(demand * 1000.0, [_Source("A", 5.0, 40.0)], duty="heat")
        total = sum(r.dispatch_by_source_MW.values()) + r.unmet_demand_MW
        self.assertTrue(np.allclose(total, r.demand_MW, atol=1e-9))

    def test_no_source_ever_exceeds_its_capacity(self):
        demand = np.full(N, 20.0) * 1000.0
        a, b = _Source("A", 5.0, 40.0), _Source("B", 3.0, 50.0)
        r = run_dispatch(demand, [a, b], duty="heat")
        self.assertTrue(np.all(r.dispatch_by_source_MW["A"] <= 5.0 + 1e-9))
        self.assertTrue(np.all(r.dispatch_by_source_MW["B"] <= 3.0 + 1e-9))

    def test_nothing_dispatches_below_zero(self):
        demand = np.zeros(N)
        r = run_dispatch(demand, [_Source("A", 5.0, 40.0)], duty="heat")
        for v in r.dispatch_by_source_MW.values():
            self.assertTrue(np.all(v >= -1e-9))
        self.assertTrue(np.all(r.unmet_demand_MW >= -1e-9))


class MeritOrderTests(unittest.TestCase):
    def test_cheapest_primary_source_runs_first(self):
        demand = np.full(N, 4.0) * 1000.0
        cheap = _Source("cheap", 5.0, 20.0)
        dear = _Source("dear", 5.0, 80.0)
        r = run_dispatch(demand, [dear, cheap], duty="heat")   # order must not matter
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["cheap"], 4.0))
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["dear"], 0.0))

    def test_merit_order_follows_HOURLY_cost_not_an_annual_average(self):
        """The whole point of the tariff shape: a source that is cheap overnight
        and dear at 6pm must be dispatched accordingly, hour by hour."""
        demand = np.full(N, 3.0) * 1000.0
        a = _Source("A", 5.0, 0.0)
        b = _Source("B", 5.0, 0.0)
        # A is cheaper for the first half of the year, B for the second.
        a.marginal_cost = np.concatenate([np.full(N // 2, 10.0), np.full(N - N // 2, 90.0)])
        b.marginal_cost = np.concatenate([np.full(N // 2, 90.0), np.full(N - N // 2, 10.0)])
        r = run_dispatch(demand, [a, b], duty="heat")
        self.assertAlmostEqual(r.dispatch_by_source_MW["A"][:N // 2].mean(), 3.0, places=6)
        self.assertAlmostEqual(r.dispatch_by_source_MW["A"][N // 2:].mean(), 0.0, places=6)
        self.assertAlmostEqual(r.dispatch_by_source_MW["B"][N // 2:].mean(), 3.0, places=6)

    def test_boilers_are_tiered_last_by_TYPE_even_when_cheaper(self):
        """A gas boiler that undercuts the heat pump must STILL run last. Real
        network controls do not swap to backup plant for a few pence of
        arbitrage; backup exists for reliability, not economics."""
        demand = np.full(N, 4.0) * 1000.0
        hp = _Source("heat pump", 5.0, 90.0, source_type="ashp")
        boiler = _Source("boiler", 5.0, 10.0, source_type="gas_boiler")   # far cheaper
        r = run_dispatch(demand, [hp, boiler], duty="heat")
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["heat pump"], 4.0))
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["boiler"], 0.0))

    def test_boiler_picks_up_only_the_shortfall(self):
        demand = np.full(N, 7.0) * 1000.0
        hp = _Source("heat pump", 5.0, 30.0, source_type="ashp")
        boiler = _Source("boiler", 10.0, 60.0, source_type="gas_boiler")
        r = run_dispatch(demand, [hp, boiler], duty="heat")
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["heat pump"], 5.0))
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["boiler"], 2.0))
        self.assertTrue(np.allclose(r.unmet_demand_MW, 0.0))

    def test_cheapest_boiler_runs_first_among_boilers(self):
        demand = np.full(N, 6.0) * 1000.0
        cheap = _Source("cheap boiler", 4.0, 30.0, source_type="gas_boiler")
        dear = _Source("dear boiler", 4.0, 70.0, source_type="electric_boiler")
        r = run_dispatch(demand, [dear, cheap], duty="heat")
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["cheap boiler"], 4.0))
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["dear boiler"], 2.0))


class UnmetDemandTests(unittest.TestCase):
    def test_shortfall_is_reported_not_hidden(self):
        """A persistently nonzero figure is a sizing red flag. It must surface,
        not be silently absorbed."""
        demand = np.full(N, 10.0) * 1000.0
        r = run_dispatch(demand, [_Source("A", 4.0, 40.0)], duty="heat")
        self.assertTrue(np.allclose(r.unmet_demand_MW, 6.0))
        self.assertAlmostEqual(r.summary()["annual_unmet_demand_MWh"], 6.0 * N, places=3)
        self.assertAlmostEqual(r.summary()["peak_unmet_MW"], 6.0, places=6)

    def test_adequately_sized_plant_leaves_no_shortfall(self):
        demand = np.full(N, 3.0) * 1000.0
        r = run_dispatch(demand, [_Source("A", 5.0, 40.0)], duty="heat")
        self.assertEqual(r.summary()["annual_unmet_demand_MWh"], 0.0)

    def test_an_unavailable_source_creates_real_unmet_demand(self):
        """Outages must bite. A source that is out cannot serve."""
        demand = np.full(N, 4.0) * 1000.0
        avail = np.full(N, 5.0)
        avail[:100] = 0.0
        r = run_dispatch(demand, [_Source("A", 5.0, 40.0, available=avail)], duty="heat")
        self.assertTrue(np.allclose(r.unmet_demand_MW[:100], 4.0))
        self.assertTrue(np.allclose(r.unmet_demand_MW[100:], 0.0))


class StorageTests(unittest.TestCase):
    def test_storage_discharges_before_a_boiler_fires(self):
        """Stored energy was already paid for at the price it was charged at, so
        using it is free at the point of use — cheaper than any boiler."""
        demand = np.full(N, 6.0) * 1000.0
        hp = _Source("heat pump", 5.0, 30.0, source_type="ashp")
        boiler = _Source("boiler", 10.0, 60.0, source_type="gas_boiler")
        r = run_dispatch(demand, [hp, boiler], storage=_Store(), duty="heat")
        # The store starts half full, so it must cover some shortfall before the
        # boiler does any work at all.
        self.assertGreater(r.storage_discharge_MW.sum(), 0.0)
        self.assertLess(
            r.dispatch_by_source_MW["boiler"].sum(),
            (6.0 - 5.0) * N,   # what the boiler would have burned unaided
        )

    def test_storage_state_of_charge_stays_within_its_capacity(self):
        demand = np.concatenate([np.full(N // 2, 2.0), np.full(N - N // 2, 8.0)]) * 1000.0
        store = _Store(capacity_MWh=10.0, rate_MW=5.0)
        r = run_dispatch(demand, [_Source("A", 5.0, 30.0)], storage=store, duty="heat")
        self.assertTrue(np.all(r.storage_soc_MWh >= -1e-9))
        self.assertTrue(np.all(r.storage_soc_MWh <= store.capacity_MWh + 1e-9))

    def test_storage_never_charges_and_discharges_in_the_same_hour(self):
        demand = np.random.default_rng(1).uniform(0, 8, N) * 1000.0
        r = run_dispatch(demand, [_Source("A", 5.0, 30.0)], storage=_Store(), duty="heat")
        self.assertTrue(np.all(r.storage_charge_MW * r.storage_discharge_MW < 1e-9))

    def test_energy_balance_holds_with_storage(self):
        demand = np.random.default_rng(2).uniform(0, 8, N) * 1000.0
        r = run_dispatch(demand, [_Source("A", 5.0, 30.0)], storage=_Store(), duty="heat")
        total = (
            sum(r.dispatch_by_source_MW.values())
            + r.storage_discharge_MW
            - r.storage_charge_MW
            + r.unmet_demand_MW
        )
        self.assertTrue(np.allclose(total, r.demand_MW, atol=1e-6))


class SummaryTests(unittest.TestCase):
    def test_opex_is_dispatch_times_that_hours_marginal_cost(self):
        demand = np.full(N, 2.0) * 1000.0
        r = run_dispatch(demand, [_Source("A", 5.0, 25.0)], duty="heat")
        # 2 MW x 8760 h x £25/MWh
        self.assertAlmostEqual(
            r.summary()["total_annual_opex_GBP"], 2.0 * N * 25.0, delta=1.0
        )

    def test_annual_MWh_by_source_sums_to_annual_demand(self):
        demand = np.full(N, 7.0) * 1000.0
        hp = _Source("hp", 5.0, 30.0, source_type="ashp")
        boiler = _Source("boiler", 5.0, 60.0, source_type="gas_boiler")
        s = run_dispatch(demand, [hp, boiler], duty="heat").summary()
        self.assertAlmostEqual(
            sum(s["annual_MWh_by_source"].values()), s["annual_demand_MWh"], places=3
        )


class DutyTests(unittest.TestCase):
    def test_cooling_duty_dispatches_the_same_way(self):
        """The merit-order engine does not care which commodity it moves."""
        demand = np.full(N, 4.0) * 1000.0
        cheap = _Source("cheap chiller", 5.0, 20.0, source_type="air_cooled_chiller")
        dear = _Source("dear chiller", 5.0, 80.0, source_type="air_cooled_chiller")
        r = run_dispatch(demand, [dear, cheap], duty="cool")
        self.assertEqual(r.duty, "cool")
        self.assertTrue(np.allclose(r.dispatch_by_source_MW["cheap chiller"], 4.0))


class InputValidationTests(unittest.TestCase):
    def test_wrong_length_demand_raises(self):
        with self.assertRaises(ValueError):
            run_dispatch(np.ones(100), [_Source("A", 5.0, 40.0)], duty="heat")

    def test_no_sources_raises(self):
        with self.assertRaises(ValueError):
            run_dispatch(np.ones(N) * 1000.0, [], duty="heat")


if __name__ == "__main__":
    unittest.main()
