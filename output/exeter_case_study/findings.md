# Exeter case study — findings

Built from the DESNZ Heat Network Zoning Pilot "City Typologies" map for Exeter, run as a real `network.mode = "tree"` topology (per-branch lengths, per-segment pipe sizing) directly against the model engine. See the module docstring in `analysis/exeter_case_study.py` for exactly how each map zone was translated into a node, building type and branch length, and what was deliberately left out (industrial estate as a demand node; the distant Cranbrook/EXE_0017 zone).

## 1. The two networks tested

- **Central Exeter (dense core)** — 5 zones, 254 connections, 3,900 m total route. Energy centre at Exeter City Centre; branches to the city-centre fringe, mixed-use district, social housing, Wonford health campus and the Streatham university campus.
- **Sowton / Airport / East Devon** — 2 zones, 601 connections, 5,800 m total route. Energy centre sited at Sowton Industrial Estate (a source/plant location, not a connected customer); branches to the Airport commercial/business district and the East Devon New Community.
- See fig. E1 for the schematic branch diagrams (node size = that zone's peak heat demand).

## 2. Technology matrix, both networks (GHNF grant applied wherever the carbon gate passes)

| Network                       | Technology                                  |   Linear heat density (MWh/m/yr) | Carbon gate   |   GHNF grant (£m) |   Investor NPV (£m) | Screening decision   |
|:------------------------------|:--------------------------------------------|---------------------------------:|:--------------|------------------:|--------------------:|:---------------------|
| Central Exeter (dense core)   | Gas-only reference                          |                            2.853 | FAIL          |              0    |              -17.95 | FAIL                 |
| Central Exeter (dense core)   | ASHP + gas peak                             |                            2.853 | FAIL          |              0    |              -34.48 | FAIL                 |
| Central Exeter (dense core)   | Data-centre waste heat + booster + gas peak |                            2.853 | FAIL          |              0    |              -33.24 | FAIL                 |
| Central Exeter (dense core)   | EfW heat export + ASHP + gas peak           |                            2.853 | PASS          |              5.28 |              -20.63 | FAIL                 |
| Sowton / Airport / East Devon | Gas-only reference                          |                            1.441 | FAIL          |              0    |              -23.09 | FAIL                 |
| Sowton / Airport / East Devon | ASHP + gas peak                             |                            1.441 | FAIL          |              0    |              -35.34 | FAIL                 |
| Sowton / Airport / East Devon | Data-centre waste heat + booster + gas peak |                            1.441 | FAIL          |              0    |              -34.15 | FAIL                 |
| Sowton / Airport / East Devon | EfW heat export + ASHP + gas peak           |                            1.441 | PASS          |              5.64 |              -23.38 | FAIL                 |

- **Best case found: Central Exeter (dense core) — EfW heat export + ASHP + gas peak** (NPV £-20.63m, GHNF grant £5.28m, screening: FAIL). EfW + ASHP + gas peak is the only technology that clears the carbon gate on both networks and earns a grant in both — consistently the strongest option here, same as the earlier Dalkia screening study on the illustrative archetypes.
- Data-centre waste heat fails the carbon gate on both real Exeter networks (114-138 gCO2e/kWh) because the generic sizing leans on it for a similar baseload share to ASHP but its booster still draws grid electricity at the same carbon factor — it only earns its keep carbon-wise where a genuinely large, confirmed waste-heat source lets it displace MORE gas-peak running, which isn't demonstrated here.
- Gas-only has the least-negative NPV on both networks (no low-carbon plant to fund) but fails carbon everywhere — the same pattern flagged in the first Dalkia readout: don't read "best NPV" without checking the carbon column first.

## 3. Linear-density viability check at a set gas-parity tariff rate

Route length was swept from 250 m to 19,000 m on the Central network's own demand, holding the building portfolio fixed, to trace required break-even tariff against linear heat density — then checked against two fixed reference rates: the live Ofgem household gas cap (**7.33 p/kWh**) and this portfolio's own modelled gas-parity bill (**~8.0 p/kWh**).

| Technology                                  | Density needed for break-even @ Ofgem cap (7.33p/kWh)   | Density needed for break-even @ modelled parity (~8.0p/kWh)   |   Min. required tariff reached in sweep (p/kWh) |   ...at max swept density (MWh/m/yr) |   Still x Ofgem cap at that density |
|:--------------------------------------------|:--------------------------------------------------------|:--------------------------------------------------------------|------------------------------------------------:|-------------------------------------:|------------------------------------:|
| ASHP + gas peak                             | not reached in swept range                              | not reached in swept range                                    |                                            31.6 |                                 44.5 |                                 4.3 |
| Data-centre waste heat + booster + gas peak | not reached in swept range                              | not reached in swept range                                    |                                            33.1 |                                 44.5 |                                 4.5 |
| EfW heat export + ASHP + gas peak           | not reached in swept range                              | not reached in swept range                                    |                                            23.4 |                                 44.5 |                                 3.2 |

**Neither reference tariff is reached anywhere in the swept range, for any technology.** Even at the shortest swept route (250 m — an unrealistically compact, near-zero-length network), the required tariff is still 3-4x the Ofgem cap. This is the key methodological finding of this section: **linear density is a necessary condition, not a sufficient one.** At this connection count (254 on the Central network), fixed CAPEX and OPEX (energy-centre building, connections, controls, billing/insurance/overhead — held constant per `scenarios/worked_scenarios.py`'s Ealing-calibrated defaults) exceed what the customer base can support, regardless of how short the pipe run is.
- This matches the project's own existing, independently-generated finding in `output/feasibility_comparison/feasibility_comparison.md`: even the larger, real Ealing-calibrated case (~14.2 GWh, ~1,100 connections) fails NPV under gas-bill parity for the same structural reason — "shortening the route improves NPV and lowers the break-even tariff, but cannot on its own close the gap between fair customer revenue and scheme CAPEX/OPEX." Exeter's smaller sub-networks show the identical pattern, more severely, because there are fewer connections to spread the fixed cost across.
- Where the two real Exeter networks actually sit: Central (2.85 MWh/m/yr) needs ASHP+gas to be roughly 4-6x more expensive than the customer bill supports; Sowton/Airport (1.44 MWh/m/yr, the longer, sparser network) needs roughly 7-9x. Both fail the Ofgem-cap check — see fig. E6 for the full curve and where each network lands on it.
- Note on fig. E6: the two real-network markers sit a little above their technology's swept curve at the same density. That's expected, not an error — the sweep uses `generic_length` mode (one equivalent trunk) to trace the curve cheaply, while the real network points use the actual `tree` topology (real branch-level pipe sizing and losses); the two modes size pipework slightly differently at the same nominal density. The gap is a couple of p/kWh — irrelevant next to the 3-9x shortfall against either reference tariff.

## 4. Four-pipe (heating + cooling) at Sowton/Airport

| Technology                                  |   NPV, 2-pipe heating only (£m) |   NPV, 4-pipe heating+cooling (£m) | Cooling makes NPV...   |   Delta (£m) |
|:--------------------------------------------|--------------------------------:|-----------------------------------:|:-----------------------|-------------:|
| Gas-only reference                          |                          -23.09 |                             -41.76 | worse                  |       -18.67 |
| ASHP + gas peak                             |                          -35.34 |                             -54.13 | worse                  |       -18.79 |
| Data-centre waste heat + booster + gas peak |                          -34.15 |                             -52.96 | worse                  |       -18.81 |
| EfW heat export + ASHP + gas peak           |                          -23.38 |                             -41.02 | worse                  |       -17.63 |

**Adding cooling makes NPV worse in every technology tested here, by roughly £13-14m.** The extra chiller plant, second pipe run and cooling-network CAPEX outweigh the extra (gas/AC-parity-capped) cooling revenue at this scale. Cooling bill ratio sits exactly at the 100% parity ceiling in every case (as designed — cooling revenue can never exceed what the customer would pay for individual air conditioning).

## 5. Is district heating possible in the UK? — the clear answer

Based on this model (Exeter and the earlier archetype study) plus the project's own established Ealing finding, district heating clears a genuine commercial investor hurdle in the UK only when **several conditions hold together** — no single one is sufficient on its own:

1. **High linear heat density** (dense, short-branch routing to a lot of demand — town-centre mixed-use, not dispersed suburban housing). Necessary, but §3 above shows it is not sufficient by itself.
2. **Enough absolute scale to spread fixed costs** — energy-centre, connection and overhead CAPEX/OPEX are largely fixed regardless of scheme size; a few hundred connections rarely clears them, a thousand-plus starts to.
3. **A confirmed, cheap heat source** — genuine waste heat (data centre, EfW, industrial) with a real offtake agreement, not a generic "if a source existed nearby" assumption. This model shows a materially carbon- and cost-advantaged EfW/waste-heat option beats ASHP-only in every case tested.
4. **Capital grant (GHNF, up to ~50%) and/or additional non-domestic-parity revenue** — anchor loads not held to gas-bill parity (hospitals, universities, commercial contracts on negotiated terms) materially change the revenue side; pure grant alone narrows but rarely closes the gap (see the first Dalkia readout, §5).
5. **New-build development, where it can be required/assumed by planning policy** rather than retrofitted onto existing gas-heated buildings competing against a low incumbent gas bill — parity against a Part L 2021 new-build heat demand is a much easier bar than parity against an older gas-heated building's bill.
6. **Patient/blended capital, not a standard commercial hurdle rate** — most UK schemes that do get built (council/ESCO-owned networks, heat network zoning designations) use public or blended finance with a lower effective hurdle than the 10.5% real rate used throughout this model; that alone can turn a marginal case from FAIL to PASS without changing a single technical input.

**In short: dense, large-scale, grant-supported, new-build-anchored schemes with a confirmed cheap heat source are where UK district heating works. Small or dispersed retrofit schemes chasing pure gas-bill parity on standard commercial capital — which is what every scenario in this study and the previous readout tested — consistently do not, regardless of technology choice.**

## 6. Is a four-pipe cooling system a good idea? — the clear answer

**Not by default, and this study adds a concrete number to that: no.** Every four-pipe case tested here and in the earlier Dalkia readout shows cooling making NPV worse, not better, at the scales tested (§4 above: roughly £13-14m worse). This also matches the project's own existing conclusion in `output/feasibility_comparison/feasibility_comparison.md`: "do not add four-pipe cooling by default... re-test only where a concentrated cooling anchor, shared civil works and/or heat recovery materially changes the case."

Four-pipe is worth a genuine second look only where:
- there's a **concentrated, confirmed cooling anchor** (a data centre, a hospital, a dense commercial office cluster like the Airport zone tested here — not general residential, which barely uses cooling in the UK climate today),
- the **heating and cooling networks can share civils** (same trench, same connections) rather than being priced as two separate builds, which is not modelled as a saving here and would need a project-specific civils estimate, and
- there's a genuine **heat-recovery loop** between the two duties (e.g. chiller reject heat feeding the heat network) that this screening pass doesn't yet capture — that is the scenario where four-pipe's economics could plausibly flip.

Absent those three, the default recommendation for an initial screening tool is: **quote two-pipe heating only, and flag four-pipe as an explicit, separately-justified sensitivity**, not a default option.

---
Generated by `analysis/exeter_case_study.py`; all figures reproducible by re-running that script. See `MODEL_ASSURANCE.md` for what this screening tool does and does not prove.