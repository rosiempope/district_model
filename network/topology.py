"""
topology.py
======================
A GENERIC, parameterised tree topology for a district heating (or
cooling) network — nodes connected by real pipe segments with real
lengths, rather than network.py's single "one representative trunk"
simplification.

Why this exists, and how it relates to network.py
-----------------------------------------------------
network.py's own docstring is explicit that it models the network as
ONE representative trunk pipe per duty, "not a real routed multi-segment
topology (no node positions, no branching, no per-segment peak
diversity)" — a reasonable first-pass simplification, but pipework is
usually THE dominant cost and feasibility lever for a real DHN (see
network.py's own citation: SEAI/CHDU put pipework at roughly a third to
over half of total scheme cost). A single fictional trunk can't answer
"does the routing/topology choice change feasibility" — which is often
the actual question a real feasibility study needs to answer.

This module is the missing piece: a real TREE structure (energy centre
at the root, branches fanning out to buildings, junction nodes where
branches split) with REAL per-segment lengths and REAL per-segment peak
flow (the sum of everything downstream of that segment, not the whole
network's peak applied uniformly everywhere — which is what made the
old single-trunk model unable to size branch pipework correctly).

This file's role after the three-way split
----------------------------------------------
This file used to contain ALL of: tree mechanics, pipe sizing, and
thermal physics, in one ~2000-line file with one NetworkTopology
dataclass carrying every method. That's been split into three focused
files, each handling one genuinely separate concern:

  - network/topology_tree.py    -- TopologyNode + TopologyTreeMixin
                                    (add_node, children_of, descendants_of,
                                    segment_peak_kW, total_length_m,
                                    total_peak_kW, leaf_nodes,
                                    path_to_root, validate, summary)
  - network/topology_sizing.py  -- SegmentPipeResult + TopologySizingMixin
                                    (size_all_segments, total_capex_GBP,
                                    dn_by_segment)
  - network/topology_thermal.py -- seasonal_ground_temp_C(),
                                    segment_outlet_temp_C() (Shukhov
                                    formula), TopologyThermalMixin
                                    (delivered_temperature_C,
                                    check_minimum/maximum_delivered_temperature,
                                    minimum_safe_flow_temp_C,
                                    network_heat_loss_kW[_hourly])

THIS file is now just the assembly point: NetworkTopology inherits from
all three mixins (so every existing `topo.size_all_segments(...)`,
`topo.delivered_temperature_C(...)` etc. call site elsewhere in the
project — dispatch.py, ASHP.py — keeps working completely unchanged),
plus the Ealing Town Centre worked-example data and factory function,
which stay here since they're "real worked example data", not engine
logic.

All re-exports below (NetworkTopology, TopologyNode, SegmentPipeResult,
segment_outlet_temp_C, seasonal_ground_temp_C, the constants) mean any
existing `from network.network_topology import X` elsewhere in the
project continues to resolve exactly as before — nothing downstream
needed to change because of this split.

Deliberately generic, not hardcoded to Ealing
-----------------------------------------------
The topology engine (TopologyNode, NetworkTopology, the segment-sizing
logic) knows NOTHING about specific buildings or place names — it only
knows node_id, parent_id, length_m, and peak_kW/peak_cool_kW. The Ealing
worked example (see ealing_town_centre_topology() below) is built by
feeding real, named buildings into this generic engine via a separate
MAPPING dict (NODE_ID -> building name) — exactly so a different
project area can reuse the exact same engine by supplying a different
mapping and a different set of (length_m, peak) inputs, without
touching the engine's actual logic.

Tree structure and why peak flow has to accumulate
--------------------------------------------------------
Every node except the root (the energy centre) has exactly one parent
(this IS a tree, not a general graph — no loops, matching how most real
DH branch networks are actually built, ring networks being the
deliberately-excluded exception). A node's OWN peak_kW is the demand of
the building living there (0 for a pure junction/branch-point node that
has no building of its own). The peak flow that the pipe SEGMENT FROM
a node's parent TO that node must carry is the SUM of that node's own
peak_kW plus the peak_kW of every node further downstream (every node
for which this one is an ancestor) — this is what makes branch pipes
correctly come out smaller than the trunk near the energy centre, which
a single representative trunk model structurally cannot do.

This uses each node's OWN peak, summed, NOT a coincident/diversified
peak recalculated per segment — i.e. it assumes every connected
building's individual peak happens to occur at the same hour, which
overstates the true coincident peak somewhat (real diversity factors
mean buildings don't all peak simultaneously). This is the standard,
conservative simplification used at feasibility stage (sizing for the
worst case, not the average case) — a true diversified per-segment peak
needs the full 8760-hour profile per node, which IS available from
demand_synthesis.py's per-node hourly arrays if a future iteration wants
to do that hour-by-hour instead of by simple summation. Flagged here
as a real, known limitation, not hidden.

Source of the real Ealing route geometry
--------------------------------------------
The Ealing worked example's segment lengths are calibrated against the
real route shown in Figure 18, "Ealing Town Centre district heating
network" (Ealing Town Centre Heat Network Feasibility Report, SEL,
2025, p.46) — traced by hand against that figure's own printed scale
bar (0/250/500m), then uniformly rescaled so the traced total matches
the report's own published total network length (Phase 2 cumulative:
3,165m, per the report's summary table). This means individual segment
lengths are a calibrated ESTIMATE (hand-traced from a small map image,
not a precision GIS digitisation) — the relative proportions and the
overall total are grounded in the real published figures, but treat
any single segment's exact length as indicative, not survey-grade.

Usage
-----
    from network.network_topology import NetworkTopology, TopologyNode

    # Build a generic topology from scratch
    topo = NetworkTopology(name="My Network")
    topo.add_node("EC", parent_id=None, length_m=0, peak_kW=0)         # energy centre / root
    topo.add_node("N1", parent_id="EC", length_m=120, peak_kW=450)     # building A
    topo.add_node("N2", parent_id="N1", length_m=80,  peak_kW=200)     # building B, downstream of A

    print(topo.segment_peak_kW("N1"))   # 450 + 200 = 650 kW (A's own + everything behind it)
    print(topo.total_length_m())        # 200 m

    # Or use the real Ealing worked example directly
    from network.network_topology import ealing_town_centre_topology
    topo = ealing_town_centre_topology()

Self-test
---------
Moved to tests/test_network_topology.py — run that file directly for
the full demonstration/verification suite this file used to carry
inline (see this project's file-restructuring decision: self-tests
were taking up 30%+ of several large files and are now split out
project-wide into tests/, one file per module).
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipe_catalog import PipeSpec  # re-exported below for backward compatibility

from topology_tree import TopologyNode, TopologyTreeMixin
from topology_sizing import SegmentPipeResult, TopologySizingMixin
from topology_thermal import (
    TopologyThermalMixin,
    seasonal_ground_temp_C,
    segment_outlet_temp_C,
    DEFAULT_GROUND_TEMP_C,
    GROUND_TEMP_MEAN_C,
    GROUND_TEMP_SEASONAL_AMPLITUDE_C,
    GROUND_TEMP_PHASE_LAG_HOURS,
    MIN_DELIVERED_TEMP_C,
)


# ── NetworkTopology — the assembly point ────────────────────────────────────────

@dataclass
class NetworkTopology(TopologyTreeMixin, TopologySizingMixin, TopologyThermalMixin):
    """
    A full tree topology: one root (energy centre, parent_id=None) plus
    any number of downstream nodes, each with exactly one parent.

    Building this incrementally via add_node() validates structure as
    you go (catches duplicate IDs, missing parents, multiple roots)
    rather than letting a malformed tree silently produce wrong segment
    peaks later.

    This class itself holds NO method bodies (other than __repr__) —
    every real method comes from the three mixins above. See this
    file's module docstring for what each mixin covers. This is the
    ONLY @dataclass in the group (the mixins are plain classes, not
    dataclasses themselves) specifically to avoid Python's multiple-
    dataclass-inheritance field-ordering pitfalls — all real fields
    (name, nodes, _root_id, _children) live here, once.
    """
    name: str
    nodes: dict = field(default_factory=dict)   # node_id -> TopologyNode
    _root_id: Optional[str] = field(default=None, repr=False)
    _children: dict = field(default_factory=dict, repr=False)  # parent_id -> [child_ids]

    def __repr__(self):
        return (
            f"NetworkTopology('{self.name}', {len(self.nodes)} nodes, "
            f"{self.total_length_m():.0f}m total, {self.total_peak_kW():.0f}kW peak)"
        )


# ── Ealing Town Centre worked example ───────────────────────────────────────────
#
# Real route geometry, calibrated against Figure 18 of the Ealing Town
# Centre Heat Network Feasibility Report (see module docstring for the
# full sourcing/calibration note). This is ONE way to populate the
# generic engine above — swap in a different (length_m, peak_kW, name)
# table for a different project area; nothing else in this file changes.
#
# Node IDs are generic ("N1", "N2", ...) by design (see module
# docstring) — real building names are attached via the separate
# BUILDING_NAME_MAP below, which is the ONLY place a specific project's
# building names appear. To reuse this engine for a different site,
# replace EALING_SEGMENTS and BUILDING_NAME_MAP; TopologyNode/
# NetworkTopology themselves need no changes.

# (node_id, parent_id, length_m, is_building) — calibrated to the real
# Figure 18 route trace, rescaled so the total matches the report's own
# published Phase 2 cumulative network length (3,165m).
EALING_SEGMENTS = [
    # Western branch (Phase 3 area near Drayton Green / building 32-39 cluster)
    ("N1",  "EC",  334.0, False),   # junction, west corridor start
    ("N2",  "N1",  609.0, False),   # junction, continuing east along the rail corridor
    ("N3",  "N2",  142.0, False),   # spur point up toward N4
    ("N4",  "N3",  283.0, True),    # building leaf (north spur, e.g. buildings 41/42 area)
    ("N5",  "N3",  252.0, False),   # main corridor continuing east
    ("N6",  "N5",  100.0, True),    # building leaf (Phase 1 core cluster -- mapped to Perceval House)
    ("N7",  "N6",  150.0, False),   # junction, entering dense Phase 1 building cluster
    ("N8",  "N7",  190.0, True),    # building leaf (e.g. buildings 3/4/52 cluster)
    ("EC2", "N8",  127.0, True),    # building leaf (NE terminus, buildings 9/13 area)
    # Southward branch (toward Phase 2/3 area near Walpole Park and buildings 54/55)
    ("N9",  "N7",  127.0, True),    # building leaf (buildings 7/12 area)
    ("N10", "N9",  145.0, True),    # building leaf (buildings 11/14 area)
    ("N11", "N10", 223.0, True),    # building leaf (building 5 area)
    ("N12", "N11", 100.0, False),   # junction near building 25/26 (Walpole Park edge)
    ("N13", "N12", 145.0, False),   # junction, turning south past the park
    ("N14", "N13", 127.0, True),    # building leaf (approaching building 54)
    ("N15", "N14", 111.0, True),    # building leaf (building 54/55 area, southern terminus)
]

# The ONLY place real building names appear — maps specific generic
# node_ids onto the named buildings already used elsewhere in this
# project (demand_synthesis.py's self-test scenario). A different
# project area would replace this dict (and EALING_SEGMENTS above)
# without touching anything else in this file.
BUILDING_NAME_MAP = {
    "N6":  "Perceval House",
    "N4":  "Ellen Wilkinson Sch",
    "N8":  "High Street Retail",
    "EC2": "Ealing Hospital Wing",
    "N9":  "Broadway Hotel",
    "N15": "Dickens Yard Ph1",
}


def ealing_town_centre_topology(
    peak_kW_by_building: Optional[dict] = None,
    peak_cool_kW_by_building: Optional[dict] = None,
) -> NetworkTopology:
    """
    Build the real (calibrated) Ealing Town Centre worked-example
    topology. See module docstring for the route's sourcing/calibration
    note, and EALING_SEGMENTS/BUILDING_NAME_MAP above for the actual
    data.

    Parameters
    ----------
    peak_kW_by_building : optional {building_name: peak_kW} dict —
                  HEATING peak. E.g. built from demand_synthesis.py's
                  synthesise_network() output (each node's
                  "peak_heat_kW"). If omitted, building leaf nodes
                  default to peak_kW=0 (useful for just inspecting the
                  route geometry/lengths without needing a real demand
                  run first).
    peak_cool_kW_by_building : optional {building_name: peak_kW} dict —
                  COOLING peak, independent of the heating dict above.
                  E.g. built from synthesise_network()'s "peak_cool_kW".
                  If omitted, defaults to 0 for every building (i.e.
                  heating-only, the original behaviour before cooling
                  support existed in this module).

    Returns
    -------
    NetworkTopology, fully built and validated.
    """
    peak_kW_by_building = peak_kW_by_building or {}
    peak_cool_kW_by_building = peak_cool_kW_by_building or {}

    topo = NetworkTopology(name="Ealing Town Centre (worked example)")
    topo.add_node("EC", parent_id=None, length_m=0.0, peak_kW=0.0,
                   building_name="Energy Centre")

    for node_id, parent_id, length_m, is_building in EALING_SEGMENTS:
        building_name = BUILDING_NAME_MAP.get(node_id) if is_building else None
        peak_kW = peak_kW_by_building.get(building_name, 0.0) if building_name else 0.0
        peak_cool_kW = peak_cool_kW_by_building.get(building_name, 0.0) if building_name else 0.0
        topo.add_node(
            node_id, parent_id=parent_id, length_m=length_m,
            peak_kW=peak_kW, peak_cool_kW=peak_cool_kW, building_name=building_name,
        )

    topo.validate()
    return topo


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_network_topology.py "
        "(see this file's module docstring for why) -- run:\n"
        "    python3 tests/test_network_topology.py\n"
    )
