# Birmingham heat network zoning — the real report, run through the model

Source: **DESNZ, Heat Network Zoning: Zone Opportunity Report — Birmingham, February 2025**.

Run against the live engine (`scenarios.scenario_runner.run_scenario`) — the same
entry point `main.py` and the Streamlit app use. Every cost below is the report's own
figure; the model's pipe curve is used only for the comparison in section 2.

## 1. The four IZOs, as reported

| IZO                                                 |   Route (km) |   Report network cost (£m) |   Report £/m |   Report total CapEx (£m) |   Non-network CapEx (£m) |   Network share of CapEx (%) |   Heat (GWh/yr) |   Report LHD (MWh/m/yr) | Meets 4 MWh/m design target   | Reuses existing pipework   | Sources (report)   |
|:----------------------------------------------------|-------------:|---------------------------:|-------------:|--------------------------:|-------------------------:|-----------------------------:|----------------:|------------------------:|:------------------------------|:---------------------------|:-------------------|
| Birmingham Central                                  |           40 |                        150 |         3750 |                       325 |                      175 |                         46.2 |             260 |                     6.7 | True                          | False                      | ASHPs and WSHPs    |
| Tyseley Central                                     |           25 |                         75 |         3000 |                       150 |                       75 |                         50   |              73 |                     2.9 | False                         | False                      | ERF and Biomass    |
| North East                                          |           40 |                        100 |         2500 |                       200 |                      100 |                         50   |             150 |                     3.3 | False                         | False                      | Minworth STW       |
| Queen Elizabeth Hospital / University of Birmingham |            4 |                          7 |         1750 |                        50 |                       43 |                         14   |             100 |                    21.5 | True                          | True                       | ASHPs              |

The report designed the IZOs to a linear heat density target of **4.0 MWh/m/yr**, describing that as "a relatively low proxy for economic viability with the heat network sector in England".

## 2. The report's cost per metre against this model's pipe curve

| IZO                                                 |   Est. peak (MW) | Model trunk DN           |   Model £/m (SEAI curve) |   Report £/m |   Report ÷ model |   Model would price route at (£m) |   Report network cost (£m) |   Understatement (£m) |
|:----------------------------------------------------|-----------------:|:-------------------------|-------------------------:|-------------:|-----------------:|----------------------------------:|---------------------------:|----------------------:|
| Birmingham Central                                  |             98.9 | >DN600 (exceeds catalog) |                     2484 |         3750 |             1.51 |                              99.4 |                        150 |                  50.6 |
| Tyseley Central                                     |             27.8 | DN400                    |                     2090 |         3000 |             1.44 |                              52.3 |                         75 |                  22.7 |
| North East                                          |             57.1 | DN500                    |                     2299 |         2500 |             1.09 |                              91.9 |                        100 |                   8.1 |
| Queen Elizabeth Hospital / University of Birmingham |             38.1 | DN500                    |                     2299 |         1750 |             0.76 |                               9.2 |                          7 |                  -2.2 |

## 3. Birmingham Central IZO, run at the report's real costs

| Case                                  |   Low-carbon share of heat (%) |   CAPEX (£m) |   of which network (£m) |   Annual OPEX (£m) |   Heat delivered (GWh) |   Peak heat (MW) |   LHD (MWh/m/yr) |   Loss (%) |   Carbon (gCO2e/kWh) | Carbon gate   |   Unmet (MWh) |   Required tariff (p/kWh) |   Equivalent tariff (p/kWh) |   NPV (£m) | Decision   |
|:--------------------------------------|-------------------------------:|-------------:|------------------------:|-------------------:|-----------------------:|-----------------:|-----------------:|-----------:|---------------------:|:--------------|--------------:|--------------------------:|----------------------------:|-----------:|:-----------|
| Report network cost (£150m, 40km)     |                           40.8 |         94.8 |                    34.7 |               4.83 |                   60.2 |           60.462 |              6.5 |       16.3 |                160.1 | FAIL          |             0 |                    25.187 |                       7.737 |      -98.2 | FAIL       |
| Model SEAI pipe curve                 |                           40.8 |         83.1 |                    23   |               4.71 |                   60.2 |           60.462 |              6.5 |       16.3 |                160.1 | FAIL          |             0 |                    22.909 |                       7.737 |      -85.4 | FAIL       |
| 62/30, instantaneous HIU (proposed)   |                           40.1 |         94.8 |                    34.7 |               4.39 |                   60.2 |           60.462 |              6.5 |        6.7 |                147   | FAIL          |             0 |                    24.465 |                       7.737 |      -94.2 | FAIL       |
| CIBSE 60/30 target, instantaneous HIU |                           40.6 |         94.8 |                    34.7 |               4.37 |                   60.2 |           60.462 |              6.5 |       12.6 |                152.2 | FAIL          |             0 |                    24.423 |                       7.737 |      -93.9 | FAIL       |

> **Do not read the `Loss (%)` column across temperature cases.** These runs use
> `generic_length` mode, which models the whole 9.3 km route as ONE equivalent trunk
> carrying full peak flow. Pipe size is a discrete series, so a small ΔT change can tip
> the trunk across a DN boundary and halve the modelled loss — 70/40 and 60/30 (both
> ΔT=30) size to DN600 and lose ~8-10 GWh, while 62/30 (ΔT=32) drops to DN500 and loses
> ~4 GWh. That is a step in the catalog, not a thermal result. A real branched network
> carries full peak only in its first metres; the DESNZ report's own Figure 6 shows a
> branched route. Tree mode would resolve this, and cannot be built here — see below.

**Why this study is on the weaker network mode.** The report gives a total route length
(40 km) and a routed map (Figure 6), but not the per-segment geometry needed for tree
mode. Appendix 2 is explicit: *"GIS outputs are not being published alongside the report
as they are subject to change"*, and the data room "will remain restricted to DESNZ and
the local authority". So branch-level sizing is not reproducible from the public
document. The Exeter case study (`analysis/exeter_case_study.py`) IS in tree mode with
real per-branch lengths, and is the right reference for any conclusion that depends on
network losses or branch sizing.

## 4. Existing pipework

| IZO                                                 |   Route (km) | Reuses existing pipework   |   Actual £/m (report) |   £/m if at QE reuse rate |   Actual network cost (£m) |   At QE reuse rate (£m) |   Notional saving (£m) |   Saving (% of network cost) |
|:----------------------------------------------------|-------------:|:---------------------------|----------------------:|--------------------------:|---------------------------:|------------------------:|-----------------------:|-----------------------------:|
| Birmingham Central                                  |           40 | False                      |                  3750 |                      1750 |                        150 |                    70   |                   80   |                         53.3 |
| Tyseley Central                                     |           25 | False                      |                  3000 |                      1750 |                         75 |                    43.8 |                   31.2 |                         41.7 |
| North East                                          |           40 | False                      |                  2500 |                      1750 |                        100 |                    70   |                   30   |                         30   |
| Queen Elizabeth Hospital / University of Birmingham |            4 | True                       |                  1750 |                      1750 |                          7 |                     7   |                    0   |                          0   |

> The proposed IZO distribution network utilises the existing heat network pipes serving the Queen Elizabeth Hospital Heritage heat network and the University of Birmingham's low temperature hot water network branch.

## 5. The existing BDEC network's costed growth path (report Table 2)

| BDEC growth scenario   |   Annual demand (GWh) |   CapEx (£m) |   £ per annual MWh | Heat sources                  |
|:-----------------------|----------------------:|-------------:|-------------------:|:------------------------------|
| Low                    |                    74 |           16 |                216 | Data Centre and ASHPs         |
| Medium                 |                   176 |           62 |                352 | WSHP (Edgbaston Reservoir)    |
| High                   |                   661 |          545 |                825 | WSHPs (reservoir, river, STW) |

## 6. Birmingham Central heat sources (report Table 5)

| name                                | type   |   capacity_kWp |   source_temp_C | ec_ref   |
|:------------------------------------|:-------|---------------:|----------------:|:---------|
| WSHP River Rea                      | WSHP   |           5000 |              10 | E6       |
| GSHP Birmingham Children's Hospital | GSHP   |            nan |              12 | E1       |
| ASHP Aston energy centre            | ASHP   |           2200 |               8 | E2       |
| GSHP Aston University               | GSHP   |           1200 |              12 | E3       |
| ASHP Bullring                       | ASHP   |           2800 |               8 | E4       |
| ASHP Birmingham New Street Station  | ASHP   |           2800 |               8 | E5       |
| ASHP Smithfield energy centre       | ASHP   |           7000 |               8 | E13      |

Report identifies ~23.0 MWth total. This model represents **21.0 MW** of it: 14.8 MW of ASHP, 5.0 MW of river WSHP and 1.2 MW of GSHP, each on its own source temperature and COP curve rather than substituted for an ASHP. The balance is the Birmingham Children's Hospital GSHP, for which Table 5 gives no capacity, plus gas peak. Carbon and OPEX therefore remain slightly **conservative** against the report's intent, but no longer materially so.

## Provenance — which numbers are the report's and which are the model's

| Input | Source |
|---|---|
| `network_capex_GBP` | 150000000.0 |
| `network_capex_source` | DESNZ, Heat Network Zoning: Zone Opportunity Report — Birmingham, February 2025, Tables 3 and 7 |
| `non_network_capex_GBP` | 175000000.0 |
| `non_network_capex_source` | Report total CapEx minus report network cost; not broken down in the report |
| `fixed_opex_scale_factor` | 0.2 |
| `fixed_opex_source` | MODEL ASSUMPTION — the report gives no operating costs |
| `tariff_source` | MODEL ASSUMPTION — the report gives no tariffs |
| `discount_rate_source` | MODEL ASSUMPTION — the report gives no discount rate |
| `anchor_load_share_of_zone_heat` | 0.2315 |
| `route_m_prorated` | 9262.0 |
| `report_network_cost_per_m` | 3750.0 |

The report contains **no** design temperatures, tariffs, discount rate, or operating
costs — it used "a proxy for economic viability", not a cash-flow model. Those inputs
are this model's assumptions and the NPV figures inherit them. The report never claims
these zones are investable on customer revenue, and nothing here should be read as
contradicting it.