# Ealing Town Centre Phase 1 model validation

The model was calibrated to the June 2025 feasibility report using Tables 11, 14-19 and 39-48 plus Figures 23-24.

| Metric | Unit | Ealing report | Model | Variance (%) | Status |
|---|---|---|---|---|---|
| End-customer heat | MWh/yr | 14161.194 | 14161.200 | 0.000 | PASS |
| Heat including losses | MWh/yr | 15135.808 | 15135.808 | 0.000 | PASS |
| Peak heat including losses | MW | 7.190 | 7.190 | 0.000 | PASS |
| ASHP generation | MWh/yr | 13474.122 | 13483.827 | 0.072 | PASS |
| Boiler generation | MWh/yr | 1661.687 | 1668.374 | 0.402 | PASS |
| Average ASHP COP | - | 2.880 | 2.880 | 0.000 | PASS |
| Energy-centre parasitic electricity | MWh/yr | 302.716 | 302.700 | -0.005 | PASS |
| Unmet heat | MWh/yr | 0.000 | 0.000 |  | PASS |
| CAPEX | GBP | 21635190.000 | 21635191.000 | 0.000 | PASS |
| 40-year investor NPV | GBP | -2249115.000 | -2249123.780 | -0.000 | PASS |
| 40-year investor IRR | % | 2.600 | 2.539 | -2.363 | PASS |
| Simple payback | years | 25.000 | 24.561 | -1.755 | PASS |
| First-year carbon intensity | gCO2e/kWh | 56.000 | 55.500 | -0.893 | PASS |

## Interpretation

- Demand, peak, plant capacity, heat balance, COP, parasitic electricity, CAPEX, NPV, IRR, payback and carbon reconcile within screening tolerance.
- Zero unmet heat requires the report's 50,000-litre thermal store and its published load-duration shape. The public PDF does not contain the underlying 8,760 values, so the peak sharpness is inferred from Figure 23.
- GBP143,465/year is retained as a visible calibration residual for OPEX categories named but not quantified in the public PDF (staff, insurance, monitoring and maintenance).
- This is a validation of the calculation chain, not evidence that generic model presets reproduce Ealing without the report-specific inputs.

Scenario hash: `26a63aa7fbbbd5265bcf9ed8206d9f0638dd807e364162af2e9b01a7ff9a1bc3`