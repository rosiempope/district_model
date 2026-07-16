# The alternative, and the levy

Site: Birmingham Central IZO anchor core (10 named buildings, 60.2 GWh, report costs,
62/30). Run against the live engine.

## 1. What is the alternative?

| Counterfactual                                |   Alternative CAPEX (£m) |   Alternative bill (£m/yr) |   District CAPEX (£m) |   District OPEX (£m/yr) |   Incremental CAPEX (£m) |   Avoided cost (£m/yr) |   Fair tariff at parity (p/kWh) |   Investor NPV (£m) |   Whole-system NPV @3.5% (£m) |   Whole-system payback (yrs) |
|:----------------------------------------------|-------------------------:|---------------------------:|----------------------:|------------------------:|-------------------------:|-----------------------:|--------------------------------:|--------------------:|------------------------------:|-----------------------------:|
| Individual gas boilers (being phased out)     |                      7.5 |                       4.73 |                  94.8 |                    4.39 |                     87.3 |                   0.33 |                           7.73  |               -94.2 |                         -85.5 |                        nan   |
| Individual heat pumps (the legal alternative) |                     77.4 |                       7.35 |                  94.8 |                    4.39 |                     17.4 |                   2.96 |                          10.929 |               -76.2 |                          67   |                          5.9 |

## 2. Where the £7,500 BUS grant actually lands

BUS caps at **45 kWth per installation**. It transforms the
individual-heat-pump case for a house and does nothing whatever for a shopping centre.

| Building                      |   Peak (kW) |   Connections |   kW per installation | Within BUS 45 kWth cap   |   BUS grant (£) |
|:------------------------------|------------:|--------------:|----------------------:|:-------------------------|----------------:|
| Birmingham New Street Station |       15000 |             1 |                 15000 | False                    |         0       |
| Bullring Shopping Centre West |        7000 |             1 |                  7000 | False                    |         0       |
| Colmore Plaza (office)        |         900 |             1 |                   900 | False                    |         0       |
| Residential block, 50 flats   |         400 |            50 |                     8 | True                     |    375000       |
| Residential block, 200 flats  |        1600 |           200 |                     8 | True                     |         1.5e+06 |
| Single house                  |           8 |             1 |                     8 | True                     |      7500       |

Birmingham Central is ~75% non-domestic by heat, so BUS returns **£0** across its anchor
loads. A residential-led zone would look completely different — worth testing separately.

## 3. What if the green levy came off electricity?

Today: electricity 26.11p, gas 7.33p, spark gap **3.56:1** at the Ofgem cap.

|   Policy cost shifted off electricity (%) |   Electricity (p/kWh) |   Gas (p/kWh) |   Spark gap |   District OPEX (£m/yr) |   Individual-HP bill (£m/yr) |   Avoided cost (£m/yr) |   Fair tariff (p/kWh) |   Investor NPV (£m) |   Whole-system NPV @3.5% (£m) |
|------------------------------------------:|----------------------:|--------------:|------------:|------------------------:|-----------------------------:|-----------------------:|----------------------:|--------------------:|------------------------------:|
|                                         0 |                26.11  |         7.33  |       3.562 |                    4.36 |                         7.35 |                   2.99 |                10.929 |               -75.9 |                          67.7 |
|                                        25 |                24.935 |         7.606 |       3.278 |                    4.26 |                         7.06 |                   2.8  |                10.437 |               -77.7 |                          63.5 |
|                                        50 |                23.76  |         7.882 |       3.015 |                    4.16 |                         6.76 |                   2.6  |                 9.946 |               -79.5 |                          59.3 |
|                                        75 |                22.585 |         8.158 |       2.769 |                    4.06 |                         6.47 |                   2.4  |                 9.454 |               -81.4 |                          55.1 |
|                                       100 |                21.41  |         8.433 |       2.539 |                    3.96 |                         6.17 |                   2.21 |                 8.962 |               -83.2 |                          50.9 |

## Method and caveats

- The levy shift is modelled revenue-neutral between the two fuels. That is the
  CONSERVATIVE form: moving the cost to general taxation instead (as the April 2026
  Renewables Obligation change actually did) cuts electricity without raising gas, which
  is better for heat pumps than what is modelled here.
- Policy-cost shares (18% electricity, 8% gas) are applied to the unit rate. Some policy
  cost genuinely sits in the standing charge, so this is approximate. Treat the direction
  and rough magnitude as the finding, not the decimals.
- The individual-heat-pump counterfactual now prices electricity at the Ofgem cap, not at
  the ~24p/kWh large-business rate the component default resolves to. That was a real bug
  — the same one already found and fixed on the gas side — and is very likely why this
  counterfactual was never wired up.
- BUS is a customer-facing transfer, not a resource cost, so it is applied to the
  customer/investor view and excluded from the whole-system social case — the same
  treatment this project already gives the GHNF grant.