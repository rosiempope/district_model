# District Heating & Cooling Techno-Economic Model

## What this is

A modular Python techno-economic feasibility model for UK district heating
(and, where relevant, cooling) networks, built as a pilot tool for Dalkia.
It estimates the technical and financial feasibility of connecting a mix of
heat sources (energy-from-waste CHP, large-scale ASHP, data centre waste
heat, gas/electric peak boilers) to a network of buildings with realistic
UK demand profiles, and produces the cost/revenue outputs (LCOH, NPV, IRR)
needed to judge whether a scheme stacks up.

It is **not** a detailed engineering design tool. It's a scoping and
decision-support model — the kind of thing used to answer "is this worth
taking to full feasibility study" rather than "here are the final pipe
specifications." Where the model simplifies reality, that's noted in the
relevant file's docstring.

## Why it exists

The team has been asked to investigate large-scale heat pumps integrated with
energy-from-waste potential for a district network. Building a pilot model
(rather than relying on spreadsheets or one-off consultant reports) lets us:

- Test different source combinations and scales quickly
- See where network CAPEX (the dominant cost driver) actually bites
- Give a defensible, source-cited answer rather than a guess
- Leave something the team can extend for the next site, not a one-off

## The plan, in order

The model is being built in five stages. Stages are listed in the order
they need to be built, because each depends on the one before it.

1. **Weather & climate data** — done. Parses a real 20-year TMYx weather
   file for London Heathrow into a clean hourly dataset, with optional
   2050 climate-change scenarios layered on top.
2. **Demand profiles** — done. Turns annual benchmark energy use (CIBSE-
   sourced) into realistic hour-by-hour heating, cooling, and hot water
   demand for any mix of buildings (offices, homes, hospitals, retail,
   schools, hotels).
3. **Heat sources** — done. Four source technologies, each built so you
   can resize or swap them without touching any other code: data centre
   waste heat, large-scale air source heat pumps, energy-from-waste CHP,
   and gas/electric peak boilers.
4. **Dispatch & network sizing** — in progress. Decides, hour by hour,
   which source supplies how much heat (cheapest first), and sizes the
   pipe network and its cost accordingly. **Heating-only first** (3rd/4th
   generation network); a heating-and-cooling (4-pipe) version comes
   after that works.
5. **Economics** — not yet started. Turns the dispatch and network outputs
   into LCOH, NPV, IRR, and sensitivity analysis — the numbers Dalkia
   actually needs to make a decision.

### A deliberate scope decision worth knowing about

We considered modelling an "ambient loop" (5th generation) network — a
single shared low-temperature loop with decentralised heat pumps at each
building, which can elegantly balance heating and cooling demand against
each other (e.g. an office's daytime cooling reject helping heat homes in
the evening). Real UK examples exist (GreenSCIES in Islington, BEN at
LSBU) and it's a genuinely promising direction.

We're **not** building that for this pilot. It needs a fundamentally
different network model (bi-directional flow, no fixed supply/return),
which is too large a lift for the timeline. Instead: build the
conventional heating network properly first, let the model's own
economics tell us whether cooling is worth adding via a 4-pipe extension,
and document the ambient loop as a recommended follow-on phase rather than
trying to build two network topologies at once.

## Repository structure

```
district_model/
│
├── components/              Heat sources and demand modelling
│   ├── demand_synthesis.py
│   ├── source.py             (data centre)
│   ├── ashp_source.py
│   ├── efw_chp_source.py
│   ├── peak_options.py       (gas + electric boilers)
│   ├── large_scale_heat_pumps.py   [legacy/superseded — see note below]
│   ├── network.py            [not yet built]
│   └── storage.py            [not yet built]
│
├── data/
│   ├── costs/                 Pipe/plant CAPEX benchmark data
│   └── profiles/               Weather files, parsed weather_data.csv
│
├── economics/                  [not yet built]
│   ├── CAPEX.py
│   ├── OPEX.py
│   └── metrics.py              LCOH / NPV / IRR
│
├── optimisation/
│   ├── dispatch.py             [being built next]
│   └── network_layout.py       [not yet built]
│
├── scenarios/                   YAML configs — one per scenario tested
│
├── utilities/
│   └── TMY_weather_single_rep_year.py   EPW parser
│
├── climate_scenarios.py          2050 climate delta application
├── main.py                        [to be built — single entry point]
└── README.md                      this file
```

## What each file does

### `utilities/TMY_weather_single_rep_year.py`
Parses the EPW weather file (London Heathrow, TMYx 2011–2025) into a clean
8,760-hour DataFrame: temperature, humidity, solar, wind, plus derived
heating/cooling degree-hours. Outputs `weather_data.csv` and a monthly
summary. This is a single representative year built from the *most
typical* months across 2011–2025 — it is **not** an average, and it
deliberately excludes extreme years, so it's good for revenue/economics
but not for peak/design sizing (see the note at the top of the file).

### `climate_scenarios.py`
Takes the parsed weather data and shifts it to represent future climate
scenarios — `baseline` (no change), `2050_central` (UKCP18 RCP4.5,
+2.7°C summer), `2050_high` (UKCP18 RCP8.5 + urban heat island, +4.0°C
summer). Used to stress-test the model against a hotter future without
needing a separate weather file for each scenario.

### `components/demand_synthesis.py`
Converts annual CIBSE benchmark energy intensities (kWh/m²/yr, by building
type) into realistic 8,760-hour heating, cooling, and domestic hot water
profiles. Heating uses degree-day scaling; cooling uses a two-part model
(degree-day scaling **plus** a comfort-urgency ramp that kicks in above
22°C, so the model shows realistic demand spikes in heatwaves even where
no air conditioning exists today — important for climate-change
scenarios). DHW follows a seasonal/diurnal pattern, not weather. Building
types and benchmarks are sourced from CIBSE TM46/Guide G and cross-checked
against the Ealing Town Centre feasibility report.

### `components/source.py`
Models data centre waste heat as a source: low-grade heat (25–35°C) at
near-constant output, governed by an availability factor (planned +
unplanned outages). Comes with presets built from real figures in the
Ealing/Southall feasibility report (Redwire, GTR, CyrusOne data centres at
several offtake scenarios) or fully custom sizing.

### `components/ashp_source.py`
Models a large-scale air source heat pump array. COP varies hour-by-hour
with ambient temperature using the Ruhnau et al. (2019) regression — the
same curve used in PyPSA-Eur — plus a defrost-cycle penalty and a capacity
derating at low temperatures, both calibrated against real UK field trial
data. Scale by changing `n_units` and `unit_capacity_MW`; comes with a
preset matching the Ealing report's 2.8 MW Phase 1 ASHP bank.

### `components/efw_chp_source.py`
Models an energy-from-waste CHP plant: high-grade heat (90°C+) direct from
steam turbine extraction, near-baseload availability with one long annual
maintenance outage (not dispersed short outages like a data centre).
Calibrated against three real UK reference plants (Sheffield ERF, SELCHP,
Newlincs) — you can size it by waste throughput, electrical capacity, or
heat capacity, and it infers whichever you didn't specify.

### `components/peak_options.py`
Gas and electric boilers for peak/backup duty — high marginal cost, fully
dispatchable, sit at the top of the merit order. Models part-load
efficiency properly (condensing boilers get *more* efficient at partial
load, not less — a real and slightly counter-intuitive effect) rather
than inventing a fake size-based efficiency curve, since the evidence
shows boiler efficiency doesn't meaningfully change with nameplate size.

### `components/large_scale_heat_pumps.py`
Legacy file from early architecture planning — **superseded by
`ashp_source.py`**. Kept for reference; not used in the current model.
Worth deleting once everyone's confirmed nothing still imports it.

### `optimisation/dispatch.py` *(in progress)*
Will take the demand profiles and the source stack and decide, for every
one of the 8,760 hours, how much each source contributes — cheapest first
(merit order), using `build_source_stack()`. Heating-only version first;
a heating-and-cooling version will follow as a separate file so the two
can be switched between cleanly.

### `optimisation/network_layout.py` *(not yet built)*
Will size and cost the pipe network itself — diameters, routing, and
£/m CAPEX — using peak flow figures from dispatch and benchmark costs
from the Ealing report. This is expected to be the single largest CAPEX
line item, consistent with every real feasibility study we've reviewed.

### `economics/CAPEX.py`, `OPEX.py`, `metrics.py` *(not yet built)*
Will combine source CAPEX, network CAPEX, and dispatch-derived running
costs into LCOH, NPV, IRR, and sensitivity outputs — the actual numbers
Dalkia needs to see.

### `components/network.py`, `components/storage.py` *(not yet built)*
Placeholder files for future network topology and thermal storage
modelling.

### `main.py` *(to be built)*
Single entry point that loads a scenario config, builds the weather data
and climate scenario, builds the demand and source stacks, runs dispatch,
costs the network, and produces the economics outputs — end to end, for
one command. Will also be where any UI/API layer for exploring results
gets hung, once the core model is working.

## A note on data provenance

Every benchmark number in this model is sourced from somewhere — mostly
CIBSE guidance, the Ealing Town Centre Heat Network Feasibility Report, or
named real UK plants (Sheffield ERF, SELCHP, Newlincs). Where a figure is
an engineering estimate rather than a directly-citable source (e.g. the
fraction of heating load that persists overnight in an empty building),
the relevant file's docstring says so explicitly. If you're adding a new
number, please reference where it came from in the same way — it's what
makes this defensible to Dalkia rather than a black box.

## Running it

Each component file has a self-test block (`if __name__ == "__main__":`)
that runs sanity checks and prints a summary table — useful for confirming
a module works in isolation before wiring it into anything else. Run any
file directly to see its self-test:

```bash
python3 components/source.py
python3 components/ashp_source.py
python3 components/efw_chp_source.py
python3 components/peak_options.py
python3 components/demand_synthesis.py
```

Once `main.py` exists, the full model will run end-to-end from a single
scenario config file in `scenarios/`.
