"""End-to-end Streamlit UI regressions.

These tests exercise the real application script and widget/session-state
round trip. They complement the calculation-layer tests in test_regressions.
"""
import unittest
import copy

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - requirements.txt installs Streamlit
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class StreamlitApplicationTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file("app.py", default_timeout=120).run()
        self.assert_no_exceptions()

    def by_label(self, elements, label, occurrence=0):
        matches = [element for element in elements if element.label == label]
        self.assertGreater(len(matches), occurrence, f"No widget labelled {label!r}")
        return matches[occurrence]

    def assert_no_exceptions(self):
        self.assertEqual([item.message for item in self.app.exception], [])

    def load_template(self, name):
        selector = self.by_label(self.app.selectbox, "Load example")
        selector.set_value(name)
        self.by_label(self.app.button, "Load selected scenario").click().run()
        self.assert_no_exceptions()

    def run_scenario(self):
        self.by_label(self.app.button, "Validate and run scenario").click().run()
        self.assert_no_exceptions()
        self.assertIsNotNone(self.app.session_state["last_result"])
        return self.app.session_state["last_result"]

    def test_clean_ealing_ui_round_trip_matches_report_calibration(self):
        self.load_template("Ealing report validation - Phase 1")
        self.assertTrue(self.by_label(self.app.button, "Auto-size from demand").disabled)
        result = self.run_scenario()
        headline = result["headline"]
        investor = result["financial"]["investor"]
        self.assertEqual(headline["annual_unmet_demand_MWh"], 0.0)
        self.assertAlmostEqual(headline["annual_heat_demand_MWh"], 14_161.2, delta=0.2)
        self.assertAlmostEqual(headline["capex_total_GBP"], 21_635_190, delta=100)
        self.assertAlmostEqual(headline["annual_total_opex_GBP"], 1_355_468, delta=2)
        self.assertAlmostEqual(investor["npv_GBP"], -2_249_115, delta=150)

    def test_template_load_clears_contaminated_widget_state(self):
        # Deliberately contaminate explicit and implicit widgets before loading
        # Ealing. The newly loaded template must remain authoritative.
        self.by_label(self.app.number_input, "Energy-centre building (£)").set_value(9_000_000)
        self.by_label(self.app.number_input, "Total network length (m)").set_value(9_000)
        self.by_label(self.app.number_input, "Total capacity (MW)", 0).set_value(0.5)
        self.by_label(self.app.toggle, "Include thermal storage").set_value(False)
        self.app.run()

        self.load_template("Ealing report validation - Phase 1")
        self.assertEqual(self.by_label(self.app.number_input, "Energy-centre building (£)").value, 2_070_000)
        self.assertEqual(self.by_label(self.app.number_input, "Total network length (m)").value, 2_148)
        self.assertEqual(self.by_label(self.app.number_input, "Total capacity (MW)", 0).value, 2.8)
        self.assertTrue(self.by_label(self.app.toggle, "Include thermal storage").value)
        self.assertEqual(self.run_scenario()["headline"]["annual_unmet_demand_MWh"], 0.0)

    def test_auto_size_round_trip_runs_and_meets_heat_service(self):
        self.load_template("A3 — ASHP plus gas peak/backup")
        auto_size = self.by_label(self.app.button, "Auto-size from demand")
        self.assertFalse(auto_size.disabled)
        auto_size.click().run()
        self.assert_no_exceptions()

        result = self.run_scenario()
        headline = result["headline"]
        self.assertEqual(headline["annual_unmet_demand_MWh"], 0.0)
        self.assertGreaterEqual(
            headline["peak_available_heat_capacity_MW"],
            headline["peak_heat_to_generate_MW"],
        )

    def test_unedited_worked_template_preserves_preset_unit_count_and_opex(self):
        from scenarios.scenario_runner import run_scenario
        from scenarios.worked_scenarios import ASHP_PLUS_GAS_PEAK

        self.load_template("A3 — ASHP plus gas peak/backup")
        ui_result = self.run_scenario()
        direct_result = run_scenario(copy.deepcopy(ASHP_PLUS_GAS_PEAK))
        self.assertEqual(ui_result["input"]["sources"][0]["n_units"], 4)
        self.assertAlmostEqual(
            ui_result["headline"]["annual_total_opex_GBP"],
            direct_result["headline"]["annual_total_opex_GBP"],
            delta=1.0,
        )


if __name__ == "__main__":
    unittest.main()
