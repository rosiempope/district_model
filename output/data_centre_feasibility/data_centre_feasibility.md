# Data-centre waste-heat feasibility study

## Result

A data centre should be treated as a steady low-carbon baseload source, not the sole source of heat. Independent peak and reserve capacity remains necessary because heat-source and booster outages do not coincide neatly with customer demand.

The strongest tested data-centre case is **DC3 - Compact liquid-cooled baseload hybrid**, with NPV GBP-5.26m, 0.00 MWh unmet heat and 53.6 gCO2e/kWh. It is not investable at gas-bill-parity customer revenue under the tested assumptions.

## Core comparison

| Scenario | Route (m) | Heat demand (GWh) | Gross CAPEX (£m) | Grant (£m) | Connection contributions (£m) | Annual OPEX (£m) | Unmet heat (MWh) | Carbon (gCO2e/kWh) | NPV (£m) | IRR (%) | Required tariff (p/kWh) | Service gate | Carbon gate | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DC1 - Data-centre-only service stress test | 1200.00 | 14.16 | 18.89 | 6.31 | 8.34 | 1.31 | 314.19 | 35.10 | -7.89 | None | 13.72 | FAIL | PASS | DO NOT PROGRESS |
| DC2 - Typical air-cooled data centre plus reserve | 1800.00 | 14.16 | 21.55 | 7.61 | 8.34 | 1.71 | 0.00 | 55.50 | -12.94 | None | 17.53 | PASS | PASS | DO NOT PROGRESS |
| DC3 - Compact liquid-cooled baseload hybrid | 1200.00 | 14.16 | 15.73 | 4.76 | 8.34 | 1.25 | 0.00 | 53.60 | -5.26 | None | 11.73 | PASS | PASS | DO NOT PROGRESS |
| DC4 - Same compact hybrid without grant or contributions | 1200.00 | 14.16 | 15.73 | 0.00 | 0.00 | 1.25 | 0.00 | 53.60 | -17.57 | None | 21.03 | PASS | PASS | DO NOT PROGRESS |

## What moves feasibility

The one-at-a-time sensitivity is centred on the compact liquid-cooled hybrid. It is not a probability forecast.

- Short total route and a heat-recovery energy centre close to the data centre.
- Higher source temperature, ideally from liquid cooling, because it raises booster COP.
- A low or zero waste-heat fee that recognises avoided data-centre cooling cost.
- Enough recoverable heat for baseload, but not oversized recovery/booster plant that is rarely used.
- Long-term heat availability and offtake contracts, plus independent reserve capacity.
- Grant and customer connection funding. The compact case does not remain investable when both are removed.

### Tested breakpoints for the compact hybrid

These are interpolated screening breakpoints, not procurement limits: total route about 2.6 km; source temperature about 24C; waste-heat fee about GBP55/MWh; connection contribution about GBP420/kW when the high-grant assumption is retained; and heat tariff about 6.26p/kWh. Source availability below roughly 62% breaches the carbon gate even though backup preserves heat service.

The highest model NPV occurs with only about 1 MW of 40C recovered baseload heat, but that sits at approximately 99 gCO2e/kWh and therefore has almost no carbon-gate headroom. The selected 2 MW case sacrifices some NPV for a much stronger carbon result.

## UK support and policy position (checked 14 July 2026)

- [GHNF Round 12](https://www.gov.uk/government/publications/green-heat-network-fund-ghnf-guidance-on-how-to-apply) is open to public, private and third-sector applicants in England and Wales until 25 September 2026.
- [GHNF Round 12 guidance](https://assets.publishing.service.gov.uk/media/6a2927b6f553ec1112221871/GHNF_Guidance_for_Applicants_R12.pdf) sets the 2 GWh urban demand, 100 gCO2e/kWh, customer detriment, 3.5% social IRR, <50% eligible-cost and 4.5p/kWh-over-15-years gates.
- [National Wealth Fund](https://www.nationalwealthfund.org.uk/news-and-publications/news/national-wealth-fund-backs-hull-city-centre-heat-network/) lending can sit alongside GHNF and local contributions; Hull combined a GBP15m GHNF grant, GBP1.5m local funding and a GBP27m NWF loan.
- [Heat-network zoning](https://www.gov.uk/government/consultations/proposals-for-heat-network-zoning-2023/outcome/heat-network-zoning-consultation-2023-summary-of-government-response) is intended to improve demand and waste-heat-source certainty, but project-specific rights and duties still need confirmation.
- [Ofgem consumer-protection regulation](https://www.ofgem.gov.uk/blog/heat-networks-regulation-now-live) has applied since 27 January 2026; fair pricing, billing and reliability must be designed into the commercial case.
- The [OPDC data-centre network](https://www.london.gov.uk/who-we-are/city-halls-partners/old-oak-and-park-royal-development-corporation-opdc/opdc-media-centre/opdc-press-releases/opdc-awarded-ps36m-keep-thousands-homes-warm-waste-heat-data-centres-uk-first) demonstrates the funding stack: GBP36m GHNF support for a phased 95 GWh scheme, with separate development support.

## Important limitations

The model repeats one operating year over the 40-year cash flow. It does not yet simulate a data-centre tenant ramp-up, an expiring heat-offtake contract, debt service, tax or annual grid-carbon trajectories. GHNF social IRR and customer-detriment calculations must be completed in the official application workbook; the CSV only pre-checks the gates this model can evidence.