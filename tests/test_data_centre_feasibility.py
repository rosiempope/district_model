"""Regression checks for the data-centre investor cases."""
import unittest

from reports.data_centre_feasibility import (
    run_comparison, run_lifetime_comparison, run_sensitivities,
)


class DataCentreFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = run_comparison().set_index("Scenario")

    def test_data_centre_only_fails_service(self):
        row = self.comparison.loc["DC1 - Data-centre-only service stress test"]
        self.assertEqual(row["Service gate"], "FAIL")
        self.assertGreater(row["Unmet heat (MWh)"], 100)

    def test_compact_liquid_cooled_hybrid_is_not_made_viable_by_overcharging(self):
        row = self.comparison.loc["DC3 - Compact liquid-cooled baseload hybrid"]
        self.assertEqual(row["Outcome"], "DO NOT PROGRESS")
        self.assertEqual(row["Unmet heat (MWh)"], 0)
        self.assertLess(row["Carbon (gCO2e/kWh)"], 100)
        self.assertLess(row["NPV (£m)"], 0)

    def test_same_engineering_without_support_is_not_viable(self):
        row = self.comparison.loc[
            "DC4 - Same compact hybrid without grant or contributions"
        ]
        self.assertEqual(row["Outcome"], "DO NOT PROGRESS")
        self.assertLess(row["NPV (£m)"], 0)

    def test_longer_route_reduces_npv(self):
        sensitivity = run_sensitivities(["Total network route (m)"])
        route = sensitivity[
            sensitivity["Variable"] == "Total network route (m)"
        ].sort_values("Value")
        self.assertTrue(route["NPV (£m)"].is_monotonic_decreasing)

    def test_lifetime_export_has_years_zero_to_forty(self):
        lifetime = run_lifetime_comparison()
        self.assertEqual(lifetime["Year"].min(), 0)
        self.assertEqual(lifetime["Year"].max(), 40)
        self.assertEqual(lifetime.groupby("Scenario").size().nunique(), 1)


if __name__ == "__main__":
    unittest.main()
