"""Source-stack comparison — Ealing Town Centre network.

Same method and same three technology stacks as
analysis/source_stack_comparison.py (the Central Exeter driver) — see
analysis/source_stack_comparison_common.py for the full methodology
docstring — but run against the REAL Ealing Town Centre tree topology
and buildings this project already has calibrated from the actual Ealing
Town Centre Heat Network Feasibility Report (SEL, 2025), rather than the
Exeter DESNZ-typology-map estimate.

Where this network's data comes from
------------------------------------------------------------------------
- Topology: EALING_SEGMENTS below is transcribed directly from
  network/network_topology.py's own EALING_SEGMENTS + BUILDING_NAME_MAP
  (16 nodes: 6 real named buildings + 10 pipe-corridor junctions with no
  demand of their own), converted from that module's
  (node_id, parent_id, length_m, is_building) tuple format into this
  project's {"node_id","parent_id","length_m","building"} segment-dict
  format (the format scenarios/scenario_runner.py's tree-mode builder
  expects). Total route length sums to 3,165 m — reproduces the report's
  own published Phase 2 cumulative network length exactly (see that
  module's docstring for the full Figure-18 route-tracing/calibration
  note).
- Buildings: EALING_BUILDINGS below matches the real building mix in
  figures/generate_demand_figures.py's EALING_SCENARIO (floor areas /
  unit counts as published), extended with connections/connection_year/
  connection_probability fields (this project's standard convention for
  a tree-mode scenario — see analysis/exeter_case_study.py's
  CENTRAL_BUILDINGS for the same pattern) since the source list didn't
  need those for demand-figure generation but a full economics run does.
- Technology presets: PRESET_FOR_TYPE (reused from analysis.exeter_case_study)
  maps ashp/gas_boiler -> "ealing_phase1", electric_boiler -> "ealing_backup",
  data_centre -> "redwire_ealing" — these presets are THE real Ealing report
  figures already (2.8 MW ASHP, 3.6 MW peak/reserve boiler, Redwire DC 3.6 MW
  waste heat), so this network uses its own source presets even more
  directly than the Exeter driver does.

Run from the repository root:
    python -m analysis.source_stack_comparison_ealing

Outputs CSVs and PNGs to output/source_stack_comparison_ealing/.
"""
from __future__ import annotations

from pathlib import Path

from network.network_topology import EALING_SEGMENTS as _RAW_EALING_SEGMENTS, BUILDING_NAME_MAP
from analysis.source_stack_comparison_common import run_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "source_stack_comparison_ealing"

# ── Real Ealing Town Centre building mix ────────────────────────────────────
# Matches figures/generate_demand_figures.py's EALING_SCENARIO exactly
# (floor areas / unit counts), extended with connections/connection_year/
# connection_probability for a full tree-mode economics run — see module
# docstring.
EALING_BUILDINGS = [
    {"name": "Perceval House", "type": "office", "floor_area_m2": 8500,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
    {"name": "High Street Retail", "type": "retail", "floor_area_m2": 3000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.90},
    {"name": "Ealing Hospital Wing", "type": "hospital", "floor_area_m2": 12000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
    {"name": "Dickens Yard Ph1", "type": "residential", "units": 350,
     "connections": 350, "connection_year": 1, "connection_probability": 0.85},
    {"name": "Broadway Hotel", "type": "hotel", "floor_area_m2": 5000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.90},
    {"name": "Ellen Wilkinson Sch", "type": "school", "floor_area_m2": 6000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
]

# ── Real Ealing Town Centre tree topology ───────────────────────────────────
# Transcribed from network/network_topology.py's EALING_SEGMENTS +
# BUILDING_NAME_MAP (see module docstring) into this project's
# {"node_id","parent_id","length_m","building"} segment-dict format.
EALING_SEGMENTS = [
    {"node_id": node_id, "parent_id": parent_id, "length_m": length_m,
     "building": BUILDING_NAME_MAP.get(node_id)}
    for node_id, parent_id, length_m, _is_building in _RAW_EALING_SEGMENTS
]

if __name__ == "__main__":
    run_study("Ealing Town Centre", EALING_BUILDINGS, EALING_SEGMENTS, OUT)
