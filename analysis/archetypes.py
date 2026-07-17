"""The three density archetypes used throughout the Dalkia analysis pack.

Single source of truth — imported by dalkia_screening_study.py,
ghnf_affordability.py, source_frontier.py, climate_scenario_sweep.py and
analysis/archetype_reference_table.py, so every study's rows line up and the
reference table can never drift from what was actually run.

Route lengths are ILLUSTRATIVE placeholders reflecting typical relative
spacing (dense/middle/scarce), not measured from a real map — see
dalkia_screening_study.py's module docstring. They exist to show how linear
heat density moves the economics; the real Exeter tree-topology case studies
(analysis/exeter_*.py) replace this generic_length placeholder with measured
segment lengths.
"""
from __future__ import annotations

ARCHETYPES = {
    "Dense (town centre)": {
        "buildings": [
            {"name": "Dense residential block A", "type": "residential_existing",
             "floor_area_m2": 30000, "units": 400, "connections": 400,
             "connection_year": 1, "connection_probability": 0.92},
            {"name": "Dense residential block B", "type": "residential_existing",
             "floor_area_m2": 24000, "units": 320, "connections": 320,
             "connection_year": 1, "connection_probability": 0.90},
            {"name": "Town centre offices", "type": "office",
             "floor_area_m2": 15000, "connections": 1,
             "connection_year": 1, "connection_probability": 1.0},
            {"name": "High street retail", "type": "retail",
             "floor_area_m2": 8000, "connections": 1,
             "connection_year": 1, "connection_probability": 0.90},
            {"name": "Hotel", "type": "hotel",
             "floor_area_m2": 6000, "connections": 1,
             "connection_year": 1, "connection_probability": 1.0},
        ],
        "route_m": 900,
        "note": "Tight urban grid; short closely-packed connections.",
    },
    "Middle (suburban mixed)": {
        "buildings": [
            {"name": "Suburban residential estate", "type": "residential_existing",
             "floor_area_m2": 36000, "units": 480, "connections": 480,
             "connection_year": 1, "connection_probability": 0.85},
            {"name": "Secondary school", "type": "school",
             "floor_area_m2": 9000, "connections": 1,
             "connection_year": 1, "connection_probability": 1.0},
            {"name": "District retail parade", "type": "retail",
             "floor_area_m2": 4000, "connections": 1,
             "connection_year": 1, "connection_probability": 0.85},
            {"name": "Health centre", "type": "hospital",
             "floor_area_m2": 3000, "connections": 1,
             "connection_year": 1, "connection_probability": 1.0},
        ],
        "route_m": 2800,
        "note": "Estate-scale spacing; moderate branch lengths.",
    },
    "Scarce (low-density edge)": {
        "buildings": [
            {"name": "Dispersed housing cluster A", "type": "residential_existing",
             "floor_area_m2": 15000, "units": 200, "connections": 200,
             "connection_year": 1, "connection_probability": 0.75},
            {"name": "Dispersed housing cluster B", "type": "residential_existing",
             "floor_area_m2": 9000, "units": 120, "connections": 120,
             "connection_year": 2, "connection_probability": 0.70},
            {"name": "Village hall / community retail", "type": "retail",
             "floor_area_m2": 1500, "connections": 1,
             "connection_year": 1, "connection_probability": 0.80},
        ],
        "route_m": 6500,
        "note": "Spread housing clusters; long branch runs to reach few connections.",
    },
}
