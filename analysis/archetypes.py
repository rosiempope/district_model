"""The three density archetypes used throughout the Dalkia analysis pack.

Single source of truth — imported by dalkia_screening_study.py,
ghnf_affordability.py, source_frontier.py, climate_scenario_sweep.py,
fourpipe_threshold.py, connection_risk.py and archetype_reference_table.py, so
every study's rows line up and nothing can drift.

Grounding and realism (revised after external review)
-----------------------------------------------------
The Dense archetype is CALIBRATED against the validated Ealing Town Centre
Phase 1 case (scenarios/ealing_report_validation.py, 13/13 metrics matching the
June 2025 SEL feasibility report — MODEL_SUMMARY §9): a real, published,
anchor-supported town-centre scheme at ~14 GWh/yr. Dense is not scaled from
Ealing (its mix is deliberately more residential-led, to test that harder
case), but its density, demand intensity and demand-per-connection sit in the
same envelope as the real scheme — see analysis/archetype_reference_table.py.
Middle and Scarce are PURPOSE-BUILT lower-density settlement types (suburban
estate, dispersed edge), not scaled-down copies of Ealing, because the building
mix genuinely differs between a town centre, a suburb and a rural edge.

Route lengths are ILLUSTRATIVE placeholders reflecting typical relative spacing
(dense/middle/scarce), not measured from a real map. They exist to show how
linear heat density moves the economics; the real Exeter tree-topology case
studies (analysis/exeter_*.py) use measured segment lengths. As a sanity check,
the three land at ~14.6 / ~3.0 / ~0.65 MWh/m/yr — bracketing DESNZ's zoning-
pilot thresholds of 4 MWh/m/yr (initial target) and ~8 MWh/m/yr (generally more
attractive): one clearly-viable density, one marginal, one clearly sub-viable.

Dwelling floor areas (English Housing Survey 2023-24)
-----------------------------------------------------
EHS reports a mean ~96 m2 across all English homes (social rent ~66, private
rent ~75, owner-occupied ~110). Uniform 75 m2 is right for dense flats but low
for suburban and rural houses, so dwelling size now varies by settlement type:
dense flats 75, suburban houses 90, dispersed/rural houses 105 m2.

Connection probabilities (policy-aware)
---------------------------------------
Under the English heat-network zoning position, existing COMMUNALLY-heated
buildings and qualifying NON-DOMESTIC buildings (proposed threshold >100
MWh/yr) may be REQUIRED to connect, while existing residential buildings with
INDIVIDUAL heating are treated differently and are not mandated the same way
(DESNZ heat-network zoning consultation response). GHNF application guidance
treats connection risk as material and expects heads of terms — and, before
construction funding, binding supply agreements — from key customers. So:

  - Anchors (school, office, hotel, retail, health): 0.95 base. 1.00 only for a
    contracted/required customer; not assumed here pre-heads-of-terms.
  - Dense communal residential blocks: 0.85 — credible as a single, largely
    correlated block decision (one owner / one agreement / in-zone), NOT as 400
    independent household choices.
  - Existing individually-heated suburban / dispersed homes: genuinely
    uncertain — modelled at a CENTRAL value here, with downside/upside carried
    in RESIDENTIAL_CONNECTION_SCENARIOS for analysis/connection_risk.py.

connection_probability scales a building's whole probability-weighted
contribution linearly in the engine, i.e. it already behaves as one correlated
per-building decision (the block's connect probability), not 400 independent
draws. Connection years are staggered (domestic roll-out lags the anchors)
rather than everything landing in year 1, which understated delivery risk.
"""
from __future__ import annotations

# Central connection probabilities, by role. See module docstring for basis.
ANCHOR_CONNECT = 0.95          # contracted/required anchors reach 1.00; 0.95 base
DENSE_COMMUNAL_RES_CONNECT = 0.85
SUBURBAN_INDIVIDUAL_RES_CONNECT = 0.60
SPARSE_INDIVIDUAL_RES_CONNECT_A = 0.45
SPARSE_INDIVIDUAL_RES_CONNECT_B = 0.40

# Downside / central / upside take-up for EXISTING individually-heated homes,
# used by analysis/connection_risk.py. Modelling ranges, not published national
# take-up statistics; the central values match the inline archetype figures.
RESIDENTIAL_CONNECTION_SCENARIOS = {
    "Dense (town centre)":       {"downside": 0.70, "central": 0.85, "upside": 0.95},
    "Middle (suburban mixed)":   {"downside": 0.40, "central": 0.60, "upside": 0.85},
    "Scarce (low-density edge)": {"downside": 0.25, "central": 0.45, "upside": 0.75},
}

ARCHETYPES = {
    "Dense (town centre)": {
        "buildings": [
            # Communal/in-zone residential blocks — one largely correlated block
            # decision, phased over the first two years.
            {"name": "Dense residential block A", "type": "residential_existing",
             "floor_area_m2": 30000, "units": 400, "connections": 400,      # 75 m2/dwelling
             "connection_year": 1, "connection_probability": DENSE_COMMUNAL_RES_CONNECT},
            {"name": "Dense residential block B", "type": "residential_existing",
             "floor_area_m2": 24000, "units": 320, "connections": 320,      # 75 m2/dwelling
             "connection_year": 2, "connection_probability": DENSE_COMMUNAL_RES_CONNECT},
            {"name": "Town centre offices", "type": "office",
             "floor_area_m2": 15000, "connections": 1,
             "connection_year": 1, "connection_probability": ANCHOR_CONNECT},
            # Renamed from "High street retail": 8,000 m2 is a department-store /
            # town-centre retail complex, not one high-street unit. Trimmed to a
            # still-substantial town-centre anchor.
            {"name": "Department store / retail complex", "type": "retail",
             "floor_area_m2": 6000, "connections": 1,
             "connection_year": 1, "connection_probability": 0.90},
            {"name": "Hotel", "type": "hotel",
             "floor_area_m2": 6000, "connections": 1,
             "connection_year": 1, "connection_probability": ANCHOR_CONNECT},
        ],
        "route_m": 900,
        "note": "Tight urban grid; short closely-packed connections. Calibrated "
                "to the validated Ealing Phase 1 density envelope.",
    },
    "Middle (suburban mixed)": {
        "buildings": [
            # Existing individually-heated suburban houses — genuinely uncertain
            # take-up, roll-out lags the anchors.
            {"name": "Suburban residential estate", "type": "residential_existing",
             "floor_area_m2": 43200, "units": 480, "connections": 480,      # 90 m2/dwelling
             "connection_year": 2, "connection_probability": SUBURBAN_INDIVIDUAL_RES_CONNECT},
            {"name": "Secondary school", "type": "school",
             "floor_area_m2": 9000, "connections": 1,
             "connection_year": 1, "connection_probability": ANCHOR_CONNECT},
            {"name": "District retail parade", "type": "retail",
             "floor_area_m2": 4000, "connections": 1,
             "connection_year": 1, "connection_probability": 0.90},
            # Retyped from "hospital": a health centre/clinic has daytime, mixed
            # occupancy with real DHW — closer to mixed_use than a 24/7 acute
            # hospital's load profile (agent review point).
            {"name": "Health centre (clinic)", "type": "mixed_use",
             "floor_area_m2": 3000, "connections": 1,
             "connection_year": 1, "connection_probability": ANCHOR_CONNECT},
        ],
        "route_m": 2800,
        "note": "Estate-scale spacing; moderate branch lengths; existing "
                "individually-heated homes on uncertain take-up.",
    },
    "Scarce (low-density edge)": {
        "buildings": [
            {"name": "Dispersed housing cluster A", "type": "residential_existing",
             "floor_area_m2": 21000, "units": 200, "connections": 200,      # 105 m2/dwelling
             "connection_year": 2, "connection_probability": SPARSE_INDIVIDUAL_RES_CONNECT_A},
            {"name": "Dispersed housing cluster B", "type": "residential_existing",
             "floor_area_m2": 12600, "units": 120, "connections": 120,      # 105 m2/dwelling
             "connection_year": 4, "connection_probability": SPARSE_INDIVIDUAL_RES_CONNECT_B},
            # Retyped from "retail": a village/community hall has intermittent
            # heating and a poor load factor — the school profile (low base load,
            # daytime-intermittent) is the closest available proxy (agent point).
            {"name": "Village hall (community use)", "type": "school",
             "floor_area_m2": 1500, "connections": 1,
             "connection_year": 1, "connection_probability": 0.85},
        ],
        "route_m": 6500,
        "note": "Spread housing clusters; long branch runs to reach few "
                "connections; dispersed individual homes on low take-up.",
    },
}

# ── Ealing Phase 1 as a fourth, REAL comparison case ─────────────────────────
# The validated Ealing Town Centre Phase 1 building mix (SEL feasibility report,
# June 2025 — scenarios/ealing_report_validation.py, 13/13 metrics at ~0%),
# with the report's bespoke tariff and connection-charge overrides STRIPPED so
# it runs through the SAME standard screening pipeline as the three archetypes
# (auto-sized stack, generic scaled economics, gas parity, GHNF) and is directly
# comparable on the same axes.
#
# Two honesty notes travel with this case, and belong on any slide that shows it:
#   1. Ealing is modelled BUILDING-LEVEL — connections = 1 per building, including
#      its two residential blocks — matching the validation and the DESNZ national
#      opportunity assessment, NOT dwelling-level like the archetypes. Part of its
#      very low per-connection fixed-cost burden is that aggregation, not only its
#      anchor-heavy mix.
#   2. The NPV this pipeline produces is NOT the bespoke validated feasibility
#      result (-£2.25m on report-specific CAPEX, tariffs and sources — MODEL_SUMMARY
#      §9). It is "Ealing's real demand under our standard screening assumptions",
#      for like-for-like comparison with the archetypes only.
from scenarios.ealing_report_validation import (  # noqa: E402
    EALING_PHASE1_BUILDINGS as _EALING_RAW,
)

_EALING_KEEP = {"name", "type", "annual_heat_kWh", "annual_dhw_kWh",
                "annual_cool_kWh", "peak_total_heat_kW", "connections",
                "connection_year", "connection_probability"}

EALING_PHASE1 = {
    "buildings": [{k: v for k, v in b.items() if k in _EALING_KEEP}
                  for b in _EALING_RAW],
    "route_m": 2148.0,
    "note": "Real validated anchor-led town-centre scheme (SEL June 2025), "
            "building-level connections, standard screening pipeline.",
    "is_real": True,
}

# For the heating/affordability/source/climate comparison studies that benefit
# from the 4-way view. Cooling-specific studies (four-pipe) and the anchor-share
# sweep keep the three matched archetypes — Ealing carries no cooling and is not
# a constructed anchor-fraction zone.
ARCHETYPES_WITH_EALING = {**ARCHETYPES, "Ealing Phase 1 (real)": EALING_PHASE1}

# Take-up band for analysis/connection_risk.py. Ealing is a committed scheme with
# only two (building-level) residential blocks, so its band is deliberately tight
# — it barely moves, which is itself the point: anchor-led schemes carry little
# domestic take-up risk.
RESIDENTIAL_CONNECTION_SCENARIOS["Ealing Phase 1 (real)"] = {
    "downside": 0.80, "central": 0.90, "upside": 1.00,
}
