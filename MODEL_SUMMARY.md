# District Heating & Cooling Screening Model — Capability Summary

**Version 2.7.1** · ~19,000 lines Python · 212 tests passing · Prepared for Dalkia review

---

## 1. What it is, in one paragraph

An **hourly technical and commercial screening model** for 2-pipe heating and 4-pipe
heating/cooling district networks. It takes a set of customer buildings, a pipe route, a
choice of heat/cooling sources and a set of commercial assumptions, and simulates all
**8,760 hours of a year** — demand, plant dispatch, network losses, pumping — then turns
that into a **40-year investor cash flow** with a pass/fail screening decision against
explicit gates (service, carbon, NPV, IRR, N-1, tariff).

**What it is not:** a bankable financial model. It is deliberately positioned as a
*screening* tool — every run carries a warnings log, model version, run timestamp and a
SHA-256 scenario hash for auditability. §12 below states the remaining limitations
honestly and lists what must happen before investor circulation.

**The single most important design decision:** customer revenue is capped at what the
customer would otherwise pay. Heat is billed at the customer's own modelled
individual-gas bill; cooling at their modelled individual-AC running cost. This makes it
*structurally impossible* to manufacture feasibility by raising the tariff.

---

## 2. Input data

### Per customer/building
| Input | Notes |
|---|---|
| Building type | 11 archetypes (office, office_ac, residential, residential_existing, hospital, retail, supermarket, hotel, school, mixed_use, data_centre) |
| Floor area (m²) **or** dwelling count | 75 m²/dwelling assumption if units given |
| **or** measured annual heat/cooling/DHW (kWh) | Per-service precedence — a heat-meter total alone is enough to run a screen |
| Measured peak heat (kW) | Optional; reshapes the profile to hit annual **and** peak exactly |
| Connections, connection year, connection probability | Drives phased build-out and risk-weighting in the cash flow |
| Per-building tariff overrides, connection charges | Optional |

### Network
- **Mode:** `none` / `generic_length` (equivalent single trunk) / `tree` (real branched topology)
- Tree mode: energy-centre root + junctions + customer branches with individual lengths
- Design temperatures: heat flow/return (default 70/40 °C), cooling flow/return (default 6/12 °C)

### Sources
8 heat source types (`ashp`, `wshp`, `gshp`, `gas_boiler`, `electric_boiler`, `data_centre`,
`booster_heat_pump`, `efw_chp`) + 1 cooling type (`air_cooled_chiller`), each with named
presets. WSHP/GSHP run on their own source temperature and COP curve, not as a substituted
ASHP — a 10 °C river or 12 °C ground source is a smaller, far more stable lift than UK winter
air. Total installed capacity and unit count entered separately. Optional thermal storage.

### Economics
40-year life, 10.5% investor discount rate, 3.5% social discount rate, base/price year 2026,
tariff mode, CAPEX additions, annual overheads, per-carrier real price escalation, GHNF grant,
scheduled replacement (REPEX) overrides.

### Weather
`profiles/weather_data.csv` — 8,760-hour representative year, **London Heathrow**, built from
2011–2025 EPW data. Annual mean 11.98 °C, 2,389 HDD, 68 CDD, range −8.3 to +32.0 °C.
Climate scenario applied on top: `baseline` / `2050_central` / `2050_high`.

---

## 3. Output data

| Category | Outputs |
|---|---|
| **Demand** | Annual + peak heat/cooling/DHW, per building and network-wide; 8,760-hour profiles |
| **Energy balance** | Heat/cooling to generate, network losses, dispatch by source, unmet energy, explicit balance residuals (verified to 1e-9) |
| **Network** | Total route length, per-segment DN, per-segment CAPEX, linear heat density (MWh/m/yr), loss fraction, £/m |
| **Parasitics** | Pumping electricity (MWh), hourly-priced pumping cost, additional auxiliary |
| **Carbon** | Annual tCO₂, intensity (gCO₂e/kWh of service), carbon gate pass/fail |
| **CAPEX** | Total + full breakdown (sources, network, storage, building, connections, metering, design, commissioning, contingency) |
| **OPEX** | Energy by carrier, per-technology O&M, overheads, **reconciled to a zero residual** |
| **Financial** | 40-year investor cash-flow table, NPV, IRR, simple + discounted payback, peak funding requirement, discounted LCOH / levelised cost of service |
| **Commercial** | Required break-even heat tariff for zero NPV, equivalent year-1 tariff, customer bill ratio vs counterfactual |
| **Whole-system** | Social NPV vs individual-gas(+AC) counterfactual at 3.5% |
| **Resilience** | N-1 firm capacity at peak, margin, compliance |
| **Decision** | One screening verdict + per-gate audit, shared identically by UI, API result and CSV export |
| **Cost decomposition** | Every CAPEX and OPEX line tagged by scaling basis (fixed / per-connection / plant / network / % adder), plus the size-independent burden per connection and in p/kWh |
| **Audit** | Model version, UTC timestamp, scenario SHA-256, warnings/assumptions log |

---

## 4. Thermodynamic basis

### Demand synthesis (`profiles/demand_synthesis.py`)
- **Heating:** annual benchmark distributed by **heating degree-hours** (base 15.5 °C, UK standard),
  modulated by a per-archetype occupancy schedule with a base-load floor (fabric loss continues
  when unoccupied). Degree-day *normalisation* against a shared baseline reference means a milder
  climate year genuinely uses **less** heat — not the same total reshuffled.
- **Cooling:** a **three-part model** — (1) an internal-gains floor driven by occupancy, not weather
  (65% of annual budget for offices; 85% supermarket; justified against MIT/ScienceDirect
  8–17 W/m² and 20–30 W/m² UK figures); (2) CDD-scaled weather-driven share (base 20 °C) taking the
  remainder; (3) a comfort-urgency ramp (22 → 26 °C) acting as an hourly **floor**, not an additive
  term. This fixed an earlier ~94× peak-to-mean artefact and a ~47% over-allocation from a
  `max()`-of-two-normalised-parts approach.
- **DHW:** not weather-driven. Sinusoidal seasonal shape (±12%, peaking mid-winter for cold inlet
  water) × diurnal 07:00/19:00 peaks + overnight legionella base.

### Heat pump / chiller COP
| Component | Method |
|---|---|
| **ASHP** | `COP = 6.08 − 0.09·ΔT + 0.0005·ΔT²` — **Ruhnau et al. (2019)**, *Scientific Data* 6:189, the regression used in PyPSA-Eur. Fitted to real manufacturer/field data, not theoretical Carnot. Plus a **defrost penalty** in the 0–5 °C icing band (~10% derate), which is what brings modelled COP into line with real UK trial data (2.2–2.7 annual). Bounded 1.2–6.0. |
| **Air-cooled chiller** | `COP = 7.1136 − 0.1224·ΔT + 0.0004·ΔT²`, fitted to two real anchors (REHVA: EER 4.0 at 35 °C ambient / AHRI 550/590; measured 6–7 in Nov–Mar operation) + a measured gradient. Bounded 1.5–8.0. Hot-day capacity derate applied. |
| **Booster heat pump** | **Carnot-efficiency-fraction** method: `COP = T_sink/(T_sink−T_source) × η`, with η = 0.244 fitted to *deployed* data-centre heat-recovery systems (real measured COP 2.5–3.5 at 37.5 °C source / 65 °C sink). Deliberately more conservative than the 45–65% textbook range and than a 4.92 lab result. |

### Network thermal (`network/topology_thermal.py`)
- **Shukhov formula** — the standard closed-form exponential temperature decay along a buried pipe:
  `T_out = T_ground + (T_in − T_ground)·exp(−U·L / (cp·ṁ))`.
  Applied **per segment**, propagating each segment's real inlet temperature from its parent — not one
  mean temperature over the whole route. Handles cooling correctly by sign (chilled water *warms*
  toward the ground). Vectorised across all 8,760 hours.
- **Ground temperature is seasonal, not fixed** — mean 11.5 °C, amplitude ±5.15 °C, **730-hour (1 month)
  phase lag** behind air temperature at 1 m burial depth. Source: **Busby (2015)**, UK shallow ground
  temperatures (106 Met Office stations), cross-checked against Thames Valley regional data. A fixed
  annual average would understate winter loss — the wrong direction for a conservative case.
- **Compliance checks:** minimum 60 °C delivered at every customer (CIBSE/HSE legionella basis, applied
  uniformly because HIU losses put tap temperature below the network-side figure); maximum delivered
  chilled water at the chiller's own design supply temperature. Plus a **binary search for the lowest
  safe flow temperature** on the real network geometry.

### Dispatch (`optimisation/dispatch.py`)
Hour-by-hour **tiered merit order**: primary sources cheapest-first on *actual hourly* marginal cost
(so the electricity tariff shape genuinely drives dispatch) → thermal storage discharge → backup
boilers cheapest-first. Boilers are tiered last **by type, not price** — real network controls don't
swap to backup plant for a few pence of arbitrage. Spare primary capacity charges storage. Anything
left is genuine unmet demand, which the model reports rather than hides.

### Auto-sizing (`optimisation/auto_size.py`)
Load-duration-based, baseload-first. Diversity factor 0.85 (CIBSE/CHDU mixed-use feasibility standard)
applied to coincident peak. **ASHP capacity derated for cold weather** — nameplate at 7 °C delivers
~65% at −5 °C design day. Includes a network-loss allowance and avoids double diversity.

---

## 5. Pipework — sizing, thermal and cost

### Sizing (`network/pipe_catalog.py`)
- **Dual criterion**, standard industry practice: smallest standard DN satisfying **both** velocity
  0.3–2.5 m/s **and** pressure gradient ≤ 150 Pa/m. Velocity governs trunks; pressure gradient governs
  branches.
- **Darcy-Weisbach** pressure drop with the **Swamee-Jain** explicit approximation to Colebrook-White
  (±1% across the turbulent range, no iteration). Laminar fallback below Re 2300.
- **Temperature-dependent water properties** interpolated from a standard NIST-consistent table.
  This matters: viscosity changes ~4× between an 8 °C chilled loop and an 80 °C heating loop, so the
  same kW duty needs a genuinely different pipe size on the cold loop.
- **DN20–DN600** standard series (matches Logstor's EN 13941 published range). Twin-pipe construction
  restricted to ≤ DN200 — it isn't a real commercial product above that, so requesting it raises a
  clear error rather than returning a plausible number for a product that doesn't exist.
- Tree mode sizes **every segment individually** on its own accumulated downstream peak, including DHW.

### Heat loss
Analytical cylindrical thermal resistance of the insulation layer:
`R = ln(D_casing/D_pipe)/(2πk)`, PUR foam k = 0.0245 W/m·K. Casing diameters from **Table 7, District
Heating Manual for London** (EN 253 basis) — real data, interpolated DN20–DN500, power-law extrapolated
(R²=0.92) only for DN600. Twin-pipe combined loss = 0.55 × 2× single.

> A note worth making to the room: an earlier version carried three invented "insulation series". EN 253
> is a *construction* standard, not a multi-grade classification — "Series 1/2/3" is a manufacturer
> product-line name. The fabricated tiers were removed in favour of one real, sourced profile.

### Cost
**Power-law curve fitted to real published data**, not assumed:
`£/m = 1,158 × (DN/100)^0.426`

- **Source:** SEAI *National Heat Study*, Appendix B, Table 4 (Element Energy/Ricardo for SEAI, 2023) —
  2-pipe, inner-city, 2020 prices, DN20–DN600. Their own source is the Scottish Building Standards
  Agency (2009), SEAI-inflated to 2020.
- **Method:** EUR→GBP at the report's own €1.12/£1, UK CPI uplift ~24% (2020→2026), log-log regression
  across all 19 points. **R² = 0.84.**
- **Honest caveat carried in the code:** real data is near-flat DN20–DN65 (a fixed trenching/mobilisation
  floor a single power law can't capture), so the curve likely **underestimates the smallest branches**.
  Most reliable DN80–DN600, where the mains actually sit.
- The exponent 0.426 is *flatter* than the chemical-engineering six-tenths rule — correctly so, because
  buried-pipe cost is dominated by trenching/civils largely independent of diameter, not by equipment
  surface-area economics.

### Pumping
Hydraulic pumping electricity from real per-segment pressure drops and mass flows, priced at the
**hourly** electricity tariff, and carried into both OPEX and carbon.

---

## 6. Economic basis

### Tariffs (`economics/tariffs.py`)
| Item | Value | Source |
|---|---|---|
| **Electricity** | 24.0 p/kWh central (range 21–27), 146 p/day standing | Large-business negotiated range, 2026. Explicit `negotiated_discount_pct` parameter so an EDF-relationship rate is stated, not hidden in the central number |
| Electricity **shape** | 16:00–19:00 peak +15p, 23:00–06:00 overnight −8p, winter +10% | Calibrated to public Octopus Agile data. The shape is what makes overnight running an actual dispatch lever |
| **Gas — DESNZ central** | 85 p/therm | DESNZ *Fossil Fuel Price Assumptions 2025* (Jan 2026), Table 1, Assumption B, 2026, real 2024 prices |
| **Gas — current actual** | 120 p/therm | UK NBP wholesale, ~10 July 2026 (Catalyst Commercial), reflecting Middle East supply risk + reduced Norwegian flows |
| **Ofgem gas price cap** | 7.33 p/kWh + 29.04 p/day | Official cap, 1 Jul–30 Sep 2026. This is the **retail** rate — the right basis for what a customer would otherwise pay, as distinct from the wholesale figures above |

Gas is modelled flat intraday (no equivalent evening spike); the level, not shape, is what matters.
Each carrier has an **isolated real escalation rate**.

### Carbon factors (`components/peak_demand_option.py`)
| Carrier | kgCO₂e/kWh | Basis |
|---|---|---|
| Natural gas | 0.1823 | DESNZ 2026 GHG Conversion Factors, gross CV |
| Grid electricity | 0.1440 | DESNZ 2026, **consumption basis** = generation 0.13096 + T&D 0.01299. Replaced a stale 0.207; DESNZ's 2026 update was ~26% lower, part real decarbonisation, part a data-lag methodology change |
| EfW CHP heat | 0.0580 | BRE Technical Note (SAP 2012), calibrated to SELCHP. **Not zero** — extracting heat reduces electricity exported, a real opportunity cost |
| Data-centre waste heat | 0.0 | A **genuine** zero, not a placeholder — IT load is fixed by computing demand; the heat is rejected to atmosphere either way, so no displaced-generation term applies |

### CAPEX
Aggregated from each component's own installed cost, plus user-entered building, land, utility
connection, controls, per-connection customer connection/metering, and % design/commissioning/contingency.
**Individual-system counterfactual costs** held separately and deliberately: gas boiler £111/kW,
individual ASHP £1,150/kW, individual AC £800/kW. The £7,500 BUS grant is netted off the
customer-facing ASHP counterfactual where eligible (45 kWth per-installation cap, per-building
`bus_eligible: false` override for the social-housing exclusion) and excluded from the
whole-system social case — the same transfer-vs-resource-cost treatment as GHNF. Both
counterfactual bills carry the DECC-mandated heat-source lifecycle: boiler service/repairs +
£4,000 replacement every 15 years on the gas side; £150/yr service + BUS-netted replacement
over 20 years on the heat-pump side. The point being that going individual avoids all network CAPEX but pays *more* per kW for
plant — that asymmetry is exactly what the comparison exists to surface.

### O&M (`economics/om_rates.py`)
Per-technology annual % of that asset's own CAPEX, replacing a flat 1%: ASHP/chiller/booster **2.5%**,
gas boiler 1.25%, electric boiler 0.9%, EfW CHP 3.5%, DC heat exchanger 0.5%, **pipework 1.0%**.

> Worth flagging as a credibility point: this module openly records that an earlier version cited
> "BSRIA BG 44/2023" — **a document that does not exist**. Rates that couldn't be independently verified
> are now labelled as benchmark midpoints or engineering judgement, not attributed to unverifiable
> citations. Pipework's 1.0% is the one rate with a directly confirmed live citation (CHDU/DECC).

### Cash flow & metrics
- Explicit **years 0–40** table drives NPV, IRR, payback and every chart — no silent divergence between
  KPI and graph.
- NPV verified to equal the final cumulative discounted position (regression-tested).
- Scheduled **REPEX**: ASHP/booster/chiller 15 yr @ 60% of CAPEX; boilers 20 yr @ 50%; DC/EfW 25 yr @ 50%.
- **Discounted LCOH / levelised cost of service** = discounted project costs ÷ discounted *connected*
  customer energy. Grant excluded — it transfers who pays, not resource cost.
- **Social/whole-system NPV** at 3.5% vs the individual counterfactual, excluding grant and tariff transfers.

### GHNF grant (`economics/grant.py`)
Modelled as a year-0 capital inflow (not an OPEX reduction). Default 40% (scheme allows up to 50%;
awards typically 30–50%). Enforces a **strictly-below-50%** intensity limit, the **4.5 p/kWh over 15 years**
output-based cap, and a visible **100 gCO₂e/kWh** carbon gate. Funding drawable through FY2029-30.

---

## 7. Benchmarks used

| Domain | Benchmark |
|---|---|
| Heating EUI | CIBSE TM46 / Arup–Carbon Trust CIBSE benchmarks; Part L 2021 for new-build residential. Space-heat/DHW split 75/25 residential, 85/15 non-residential per **CIBSE Guide G** |
| Cooling EUI | SEL in-house benchmarks (per Ealing report), CIBSE Guide F. Hospital 55, supermarket 100, retail 60, office 30 / office_ac 50 kWh/m²/yr |
| DHW EUI | CIBSE Guide G, EST data |
| Degree days | HDD base 15.5 °C (UK standard); CDD base 20 °C |
| Diversity | 0.85 mixed-use (CIBSE/CHDU feasibility standard) |
| Pipe hydraulics | EN 253 / EN 13941; Logstor published ranges |
| Pipe cost | SEAI National Heat Study Appendix B Table 4 |
| Boiler part-load | **DIN 4702-8**: `η_seasonal = 0.81·η_30% + 0.19·η_100%` |
| Chiller rating | BS EN 14511 / AHRI 550/590 |
| Ground temp | Busby (2015) |
| Climate | **UKCP18** RCP4.5 (2050 central) and RCP8.5 (2050 high) |
| Carbon | DESNZ 2026 GHG Conversion Factors |
| Prices | DESNZ Fossil Fuel Price Assumptions 2025; Ofgem cap; Octopus Agile |

---

## 8. Weather and climate scenarios

**Baseline:** London Heathrow representative year (2011–2025). Being a representative year it averages
out extremes — conservative for *revenue*, but the code explicitly notes that a **CIBSE Design Summer
Year / Design Winter Year should be used for pipe diameter and peak design** (not yet implemented — a
known gap worth stating).

**2050_central** (UKCP18 RCP4.5, 50th pct): +1.0 °C winter, +2.7 °C summer, +1.4–2.2 °C shoulder.
**2050_high** (UKCP18 RCP8.5, 50th pct): +2.0 °C winter, +4.0 °C summer, **plus** a seasonally-weighted
urban heat island offset peaking at +2.5 °C in summer, tapering to 0 in winter.

> A good slide: the UHI offset was originally applied **flat** year-round, which double-counted winter
> warming and produced a **58.6% HDD reduction by 2050** — worse than Staffell et al.'s ~42% for a *full
> century* of RCP8.5. Physically implausible. Tapering UHI to zero in winter brings it to **39%**,
> correctly below the century benchmark. 2050_central's ~21% sits sensibly below the RCP4.5 24% figure.
> This is the kind of self-checking-against-literature the model does throughout.

---

## 9. Validation — Ealing Town Centre Phase 1

Calibrated against the **June 2025 SEL feasibility report** (Tables 11, 14–19, 39–48; Figures 23–24).
Reproducible via `python -m reports.ealing_validation`.

| Metric | Report | Model | Variance |
|---|---|---|---|
| End-customer heat (MWh/yr) | 14,161.19 | 14,161.20 | 0.000% |
| Heat incl. losses (MWh/yr) | 15,135.81 | 15,135.81 | 0.000% |
| Peak heat incl. losses (MW) | 7.190 | 7.190 | 0.000% |
| ASHP generation (MWh/yr) | 13,474.12 | 13,483.83 | 0.072% |
| Boiler generation (MWh/yr) | 1,661.69 | 1,668.37 | 0.402% |
| Average ASHP COP | 2.880 | 2.880 | 0.000% |
| Parasitic electricity (MWh/yr) | 302.72 | 302.70 | −0.005% |
| Unmet heat (MWh/yr) | 0.000 | 0.000 | — |
| CAPEX (£) | 21,635,190 | 21,635,191 | 0.000% |
| 40-yr investor NPV (£) | −2,249,115 | −2,249,124 | −0.000% |
| 40-yr investor IRR (%) | 2.600 | 2.539 | −2.363% |
| Simple payback (yrs) | 25.0 | 24.56 | −1.755% |
| Year-1 carbon (gCO₂e/kWh) | 56.0 | 55.5 | −0.893% |

**All PASS.** Two honest caveats retained in the output: (1) zero unmet heat requires the report's
50,000-litre thermal store and its published load-duration shape — the public PDF has no 8,760 values,
so peak sharpness is inferred from Figure 23; (2) **£143,465/yr is retained as a visible calibration
residual** for OPEX categories the public PDF names but doesn't quantify.

> Be precise in the room about what this proves: it validates **the calculation chain end-to-end**. It is
> not evidence that generic presets reproduce Ealing without the report-specific inputs.

---

## 10. Case studies completed

| Study | Command | What it tests |
|---|---|---|
| **Ealing Phase 1 validation** | `python -m reports.ealing_validation` | Calibration against a published report |
| **Feasibility comparison** | `python -m reports.feasibility_comparison` | Dense core vs compact private cluster vs extended route, on one 14.2 GWh customer base; route sensitivity; separate 4-pipe cooling check |
| **Data-centre waste heat** | `python -m reports.data_centre_feasibility` | 4 DC scenarios, UK support pre-checks, one-at-a-time sensitivities, 40-yr cash-flow comparison |
| **Technology frontier** | `python -m reports.technology_frontier` | Route/demand frontier, price sensitivities, cooling cost decomposition, fair customer-bill comparison |
| **Cost decomposition** | `python -m reports.cost_breakdown` | Every CAPEX/OPEX line by scenario, tagged with how it scales; unit costs; size-independent exposure |
| **Dalkia screening study** | `analysis/dalkia_screening_study.py` | 4 technologies × 3 density archetypes (dense/middle/scarce), auto-sized; GHNF grant sensitivity; gas-parity verification |
| **Exeter case study** | `analysis/exeter_case_study.py` | **Real tree topology** from the DESNZ Heat Network Zoning Pilot "City Typologies" map — Central Exeter (5 zones, 254 connections, 3,900 m) and Sowton/Airport (2 zones, 601 connections, 5,800 m); linear-density sweep 250 m–19,000 m; 2-pipe vs 4-pipe |
| **Source-stack comparison** | `analysis/source_stack_comparison{,_ealing}.py` | 3 tech stacks × duty × density × tree-vs-trunk × climate, on two fixed real networks (Exeter Central + Ealing) |
| **GHNF affordability frontier** | `analysis/ghnf_affordability.py` | Required vs affordable tariff under gas AND heat-pump parity; GHNF 0/40/~50%; binding-cap audit; affordability waterfall; grant dependency |
| **Source & density frontier** | `analysis/source_frontier.py` | 8 source stacks × 3 archetypes on NPV-vs-carbon axes with the GHNF boundary; route-length density sweep; EfW price and distance break-evens |
| **Anchor/BUS customer-mix sweep** | `analysis/anchor_bus_sweep.py` | Anchor heat share 0–95% × {BUS, no BUS, social housing, gas parity}; what BUS costs the owner under HP-parity billing |
| **Dalkia roles & civils risk** | `analysis/dalkia_roles.py` | Five commercial roles under 0–51% civils overruns; break-even overrun; packaged-vs-separate civils procurement |
| **Four-pipe threshold** | `analysis/fourpipe_threshold.py` | Incremental NPV of adding cooling vs cooling density, at 0/25/50/75% shared-civils credit; plus capturing the customer's avoided AC-purchase capex via connection charge, 0-100% |
| **Climate scenario sweep** | `analysis/climate_scenario_sweep.py` | Heating and cooling investor NPV across baseline/2050 central/2050 high, 3 archetypes, 2-pipe vs 4-pipe |
| **Connection (take-up) risk** | `analysis/connection_risk.py` | Owner NPV across the downside/central/upside residential connection-probability band, 4 cases; anchors held at base |
| **Sensitivity matrix** | `analysis/sensitivity_matrix.py` | 288-combination feasibility factorial (source × case × proposition × grant × capture) with PASS/FAIL on investor NPV, contractor/operator/owner NPV, and a physical-lever tier (heating/cooling dT, 2v4-pipe, climate, electricity price, discount rate) |
| **Archetype reference table** | `analysis/archetype_reference_table.py` | Presentation-ready tables of the three density archetypes' composition and connection assumptions, plus a density calibration against validated Ealing Phase 1 (canonical definitions in `analysis/archetypes.py`) |

> The three density archetypes were revised after external review to be defensible England/UK cases:
> dwelling floor areas vary by settlement type (English Housing Survey 2023-24), building typing was
> corrected (health centre → `mixed_use`, village hall → `school`), and connection probabilities are
> policy-aware — anchors 0.95, dense communal blocks 0.85, existing individually-heated homes on a
> 0.60/0.45/0.40 central band, reflecting that DESNZ zoning can mandate communal and qualifying
> non-domestic buildings but treats individual residential differently. The Dense archetype is
> calibrated to the validated Ealing Phase 1 density envelope. Connection *costs* were already priced
> per building type (`economics/connection_costs.py`), so this is a demand/take-up realism revision.
> The validated **Ealing Phase 1 mix is also run as a fourth comparison case** through the same
> screening pipeline (affordability, source/density frontier, climate, take-up risk) — it is the
> least-bad case in the pack (real anchor-led data), carried building-level; its pipeline NPV is a
> like-for-like screening figure, distinct from the bespoke validated feasibility result (§9).

### Headline findings

1. **Size-independent cost alone consumes ~81% of the customer's bill.** The decomposition puts
   **5.93 p/kWh** — £13,880 per connection — on cost that does not move with scheme size, against a
   7.33p Ofgem cap and a ~8.3p modelled gas-parity bill. That is before a single kWh is generated, a
   metre of pipe is laid, or a connection is made. This is the mechanism behind every other finding
   here, and the strongest single number in the pack.
2. **Every base case fails the investor NPV gate under strict gas-parity billing.** This is a real,
   expected district-heating result — not a model defect. Heat networks essentially never clear a
   commercial hurdle on gas-parity tariffs alone.
3. **The gap is structural, not a tariff-mechanism problem.** Customers are charged ~8.3–8.5 p/kWh
   (their own gas bill). Required break-even tariff is **20–105 p/kWh**. Gas-parity billing works
   exactly as designed — it just reveals the size of the hole.
4. **Linear density is a necessary but not sufficient condition.** The Exeter sweep is the cleanest
   result in the pack: even at a 250 m route (an unrealistically compact network), required tariff is
   still **3–4× the Ofgem cap**. At 254 connections, fixed CAPEX/OPEX exceeds what the customer base can
   support *regardless of how short the pipe run is*. Independently reproduced on the larger Ealing-scale
   case (~1,100 connections).
4. **EfW heat export + ASHP + gas peak is consistently the strongest carbon-compliant option** across
   both the illustrative archetypes and the real Exeter networks.
5. **Gas-only always has the least-negative NPV and always fails the carbon gate.** It is retained as the
   counterfactual baseline, not a candidate. Never read "best NPV" without the carbon column.
6. **Data-centre waste heat fails the carbon gate on both real Exeter networks** (114–138 gCO₂e/kWh)
   because generic sizing leans on it for a similar baseload share to ASHP while its booster still draws
   grid electricity. It earns its keep only where a genuinely large confirmed source displaces *more*
   gas-peak running.
7. **Cooling makes NPV worse in every tested case** at Sowton/Airport (−£12m to −£19m delta).
8. **The model fails designs on economics, including our own** — every base case above is failed by the
   screen on its own gates. Do **not** claim it is demonstrated to catch under-sizing: the ASHP-only
   stress test in `analysis/dalkia_screening_study.py` returns 0 MWh unmet and a service-gate PASS,
   because auto-sizing sizes the ASHP against the cold-weather-derated design peak, so it meets demand
   without backup. The service gate is untested by that run.

---

## 11. Correctness controls in place

Total/unit capacity invariant · explicit hourly energy-balance residuals · NPV = final cumulative
discounted position · carrier-specific escalation · scheduled REPEX · zero-network case · climate-demand
monotonicity · floor-area and measured-energy input regressions · end-to-end Streamlit template-load /
session-reset / auto-size regressions · source/booster energy and outage coupling · pumping cost and
carbon inclusion · carbon-unit and GHNF cap tests · service and carbon comparison gates · one shared
screening decision across UI/API/CSV · per-connection gas standing charges in the counterfactual ·
gas-bill and AC-bill parity modes · hourly pumping-electricity pricing · explicit OPEX reconciliation to
a zero residual · design/commissioning/contingency applied to the whole delivered scope · fixed
CAPEX/OPEX scaled to scheme peak with the factor recorded in the audit hash.

**212 tests, all passing** — integration/regression through `run_scenario()`, plus unit tests for pipe
hydraulics/sizing/cost, demand synthesis and all three COP curves, written against the cited sources
(Ruhnau's coefficients, REHVA's EER 4.0 at 35 °C, the SEAI cost curve, EN 253 Table 7).

### What review found and fixed

Be ready for "what did the review turn up" — the honest answer is stronger than a clean one:

- **NaN cooling demand.** The CDD branch guarded on the scenario year's degree-hours but divided by the
  *reference* year's. A cool-climate weather file produced a divide-by-zero whose NaN propagated silently
  into NPV. Not triggered by the London EPW; found the day physics unit tests were first written.
- **Contingency base.** Design/commissioning/contingency applied to plant and network only, exempting
  ~£9.4m of a ~£21m scheme — including the £4.5m of customer connections, the line likeliest to overrun.
  Now covers the whole delivered scope except land. CAPEX +14%.
- **Unreachable fixed-cost scaling.** The helper that scales overheads to scheme size existed but was
  buried in a study script, so the archetype study ran on unscaled Ealing overheads — the exact caveat
  its own findings recorded. Scarce archetype improves £22.15m → £17.73m once applied.
- **A parallel financial stack.** `metrics.py` carried an unused second NPV/IRR/payback/LCOH
  implementation on a 25-year flat-annuity basis, contradicting the live 40-year table. Removed.
- **Double-counted climate warming in cooling peak sizing.** The comfort-urgency floor
  (`_cooling_profile()` Part 3) multiplied its peak by a second, independent climate ratio
  (hours crossing the comfort threshold vs a reference year) on top of the climate response
  Part 2 already applies to the annual total via the CDD ratio. On real weather data this
  inflated a single office building's peak cooling demand ~10.8x from baseline to 2050-high
  RCP8.5 — for a ~4°C summer / 2°C winter shift — and pushed a full archetype's chiller/network
  capacity past the pipe catalogue's largest standard main, crashing sizing outright. Found
  while building the climate-scenario sweep for the Dalkia pack. Fixed by anchoring the floor
  to the already-climate-responsive `base_total.max()` alone; regression-tested against real
  weather (`tests/test_regressions.py::test_cooling_peak_does_not_double_count_climate_warming`).
  A no-op at baseline climate — the Ealing validation (§9) still reconciles 13/13 at ~0% variance.

---

## 12. Known limitations — state these openly

- **The cooling model overshoots its own benchmark by ~9–10%.** Part 3's comfort floor is applied as a
  `max()` on top of an already fully-allocated budget. The docstring claimed it summed "EXACTLY"; that
  was false and is corrected. Conservative in direction, and far better than the ~47% over-allocation it
  replaced — but an overshoot, not an identity.
- **Physics unit coverage is now broad but not complete.** Pipe hydraulics/sizing/cost, demand
  synthesis, all three COP curves, dispatch, topology thermal (Shukhov) and auto-sizing have unit
  tests. **Pumping, thermal storage and tariff shapes are still exercised only through
  `run_scenario()`.** Every module that has gained unit tests so far surfaced at least one real
  defect on the way in; assume the three untested ones carry comparable risk.
- Auto-sizing is a transparent load-duration heuristic, **not a constrained unit-combination optimiser**.
- **N-1 is a peak-capacity screen only** — it does not prove outage duration, storage autonomy or network
  resilience.
- Tree mode lacks **GIS route surfaces, utility congestion, crossings and shared-trench 4-pipe civils**.
  Generic-length mode is an equivalent trunk with high CAPEX/pumping uncertainty.
- Counterfactual CAPEX/O&M is parametric, not customer-by-customer contract data.
- **Annual physical performance is repeated across the 40-year horizon** — no year-by-year grid-carbon,
  climate, degradation or demand trajectory yet. Given DESNZ's grid factor just moved ~26% in one update,
  this materially understates the carbon case for electrified options over 40 years.
- No construction-period CAPEX drawdown, debt/tax/accounting statements, bad debt, capacity charges or
  residual/decommissioning inputs.
- No customer detriment, nominal mode, monetised carbon/social benefits, tornado sensitivity or
  switching-value optimisation.
- **CIBSE DSY/DWY design weather not yet implemented** — peak sizing currently uses the representative year.
- Fixed CAPEX/OPEX items are now scaled to scheme peak capacity (with a 0.20 floor) rather than held flat,
  but that is a **ratio, not a scoping exercise** — a real project sizes these from a drawing.
- **REPEX covers generating plant only.** Controls/SCADA carries no replacement across 40 years despite a
  shorter real life. `billing_and_customer_service` is a flat annual figure that plainly should scale with
  connection count.
- **Generic-mode pumping applies the design-point pressure gradient at every hour**, so part-load pumping is
  overstated and peak understated (real pumping power scales with flow cubed).
- Technology and pipe cost presets need project-specific price-year updates, quotations and uncertainty
  ranges.

### Required before investor circulation
1. Replace every visible warning/default with evidence or an approved assumption.
2. Calibrate demand and route lengths against measured/GIS data.
3. Obtain utility, civils, energy-centre and customer-connection quotations.
4. Independently reconcile at least three cases in a separate spreadsheet.
5. Heat-network engineer review of temperatures, hydraulics, losses, availability, resilience.
6. Project-finance review of tariffs, phasing, tax/funding, price curves, REPEX, discount-rate basis.
7. Freeze and sign the scenario hash, assumptions register and test output used in the investment paper.

---

## 13. Status

| Component | Status |
|---|---|
| Core engine (`scenarios.scenario_runner.run_scenario`) | Complete, JSON-in/JSON-out |
| Demand, weather, climate, dispatch, sizing | Complete |
| Network — tree + generic-length modes | Complete |
| Economics, tariffs, grant, cash flow, screening | Complete |
| Test suite (212 tests) | Passing — integration + physics units |
| Physics unit coverage | Broad: pipe/demand/COP/dispatch/Shukhov/auto-size covered; pumping, storage, tariff shapes not yet |
| Validation vs published report | Passing, 13/13 metrics |
| 8 case studies with figures + CSV exports | Complete |
| **Streamlit UI (`app.py`, 1,110 lines)** | **Runs; 21/21 templates match direct runs. Polish next sprint** |

The scenario interface is plain JSON-compatible data through `scenarios.scenario_runner.run_scenario`,
so the UI is a presentation layer over an engine that already runs headless from the command line. Every
case study in section 10 was generated **directly against the engine**, not through the UI.

---

## Suggested narrative for the deck

0. **Lead with the cost decomposition.** 5.93 p/kWh of size-independent cost against a 7.33p cap —
   81% of the bill gone before anything is built. Everything else follows from that one number.
1. **We built an auditable screening engine, not a spreadsheet.** 8,760-hour physics, 40-year finance,
   one shared decision, hashed and versioned.
2. **Every number is sourced or flagged.** Where a citation couldn't be verified, we say so in the code
   (the BSRIA and pipe-cost YAML examples are worth being proud of, not hiding).
3. **It validates against a real published report to ~0%.**
4. **It refuses to flatter itself.** Gas-parity billing makes artificial feasibility structurally
   impossible; the model fails designs, including our own.
5. **The finding: density alone doesn't save these schemes at this connection count.** Fixed cost per
   connection is the binding constraint. That reframes the commercial question from "find a denser route"
   to "find scale, grant, or a genuinely cheap heat source."
6. **EfW + ASHP is the strongest carbon-compliant stack tested.**
7. **Next:** UI, DSY/DWY design weather, year-by-year grid carbon, scheme-scaled fixed costs.
