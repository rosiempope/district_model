"""
network_topology.py
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

Deliberately generic, not hardcoded to Ealing
-----------------------------------------------
The topology engine here (TopologyNode, NetworkTopology, the segment-
sizing logic) knows NOTHING about specific buildings or place names —
it only knows node_id, parent_id, length_m, and peak_kW. The Ealing
worked example (see ealing_town_centre_topology() at the bottom of this
file) is built by feeding real, named buildings into this generic
engine via a separate MAPPING dict (NODE_ID -> building name) — exactly
so a different project area can reuse the exact same engine by
supplying a different mapping and a different set of (length_m, peak)
inputs, without touching this file's actual logic.

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
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# This file lives in network/, same as pipe_catalog.py — import it by
# bare module name (not "network.pipe_catalog") to sidestep the same
# package-name collision network.py itself works around (running this
# file directly as a script makes Python treat network/ as the script's
# home directory, which collides with absolute-importing a "network"
# package). See network.py's own docstring for the full explanation.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipe_catalog import (
    size_pipe_for_peak, PipeSpec, water_properties, heat_loss_coefficient_W_per_mK,
)


# ── Constants ──────────────────────────────────────────────────────────────────

# Typical UK undisturbed ground temperature at pipe-laying depth (~1m,
# the standard burial depth for pre-insulated DH pipe) — kept as a
# simple scalar default for callers that don't need the seasonal detail
# below; same default as network.py's DEFAULT_GROUND_TEMP_C, duplicated
# here (rather than imported) so this module stays independently usable
# without requiring network.py.
DEFAULT_GROUND_TEMP_C = 10.0

# Seasonal ground temperature model — real data, not a fixed annual
# average. A fixed ground temp UNDERSTATES winter heat loss (the coldest
# season, when demand and flow temp are also both at their highest,
# compounding the effect) and OVERSTATES summer heat loss -- the wrong
# direction for a conservative feasibility assessment, since it's
# exactly the high-demand winter months where understating losses
# matters most.
#
# Source: Busby (2015), "UK shallow ground temperatures for ground
# coupled heat exchangers" (analysis of 106 UK Met Office soil
# temperature stations, peer-reviewed). At 1m depth: mean annual
# temperature 12.7C in southern England down to 8.8C in northern
# Scotland, with seasonal RANGE 10.3C (south) to 7.9C (north).
# Cross-checked against Possible's "Ground and water heat resource"
# guide, which independently cites 11.5C as the Thames Valley regional
# average at 1m -- used here as London/Ealing's mean, sitting between
# Busby's southern-England figure and the wider UK range.
#
# Phase lag: ground temperature lags air temperature by roughly 1 month
# at 1m depth (consistently reported across multiple independent
# studies at this depth — e.g. ~1 month at 120cm in Greek monitoring
# data, "a month or two" at 1m per practitioner consensus). This module's
# existing air-temp convention peaks at hour 4200 (~day 175, late June);
# ground temp is modelled peaking ~1 month (730 hours) later.
GROUND_TEMP_MEAN_C = 11.5            # Thames Valley / London average at 1m depth
GROUND_TEMP_SEASONAL_AMPLITUDE_C = 5.15  # half of Busby's 10.3C southern-England seasonal range
GROUND_TEMP_PHASE_LAG_HOURS = 730     # ~1 month lag behind air temperature at 1m depth


def seasonal_ground_temp_C(hour_of_year: np.ndarray, n_hours: int = 8760) -> np.ndarray:
    """
    Real, seasonally-varying UK ground temperature at ~1m depth (typical
    DH pipe burial depth) — see the GROUND_TEMP_* constants above for
    full sourcing. Sinusoidal, same mathematical form already used
    elsewhere in this project for air temperature, but with its own
    REAL mean/amplitude/phase (ground temp is NOT just a damped copy of
    whatever air temp profile happens to be in a given weather file —
    it's a real, independently-measured seasonal signal).

    Deliberately a function of hour_of_year alone, not of the air
    temperature array — ground temperature at 1m depth is governed by
    the LONG-TERM seasonal average (which Busby's real station data
    captures directly), not by short-term swings in any particular
    weather year's daily air temperature, which would otherwise leak
    unrealistic day-to-day noise into a signal that's physically very
    smooth at this depth.

    Parameters
    ----------
    hour_of_year   : array of hour-of-year values (0 to n_hours-1) — NOT
                  necessarily a full 8760-length array; can be any subset
                  (e.g. for a single segment's calculation at a specific hour)
    n_hours        : hours in a full year (8760, matching this project's
                  convention throughout)

    Returns
    -------
    np.ndarray, same shape as hour_of_year, of ground temperatures (°C).
    """
    h = np.asarray(hour_of_year, dtype=float)
    return GROUND_TEMP_MEAN_C + GROUND_TEMP_SEASONAL_AMPLITUDE_C * np.cos(
        2 * np.pi * (h - 4200 - GROUND_TEMP_PHASE_LAG_HOURS) / n_hours
    )


# Minimum temperature that must arrive at the customer (building) end of
# the network — NOT the source/design flow temperature. Set per the
# user's own stated requirement: even though some guidance allows 50°C
# where no DHW cylinder is present, HIU pressure-drop/heat-exchanger
# resistance means the actual temperature reaching taps is meaningfully
# below the network-side figure in practice, so 60°C is used as the
# uniform minimum regardless of cylinder presence -- this is the
# standard headline figure cited for Legionella control (CIBSE/HSE
# guidance: hot water should be stored/distributed at >=60°C, with
# return/outlet temperatures not allowed to drift into the 20-45°C
# bacterial growth range for extended periods).
MIN_DELIVERED_TEMP_C = 60.0


# ── Per-segment temperature drop (Shukhov formula) ──────────────────────────────

def segment_outlet_temp_C(
    inlet_temp_C: float,
    mass_flow_kg_s: float,
    length_m: float,
    heat_loss_coefficient_W_per_mK: float,
    ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    cp_J_kgK: float = 4186.0,
) -> float:
    """
    Temperature arriving at the FAR end of one pipe segment, given the
    temperature entering it — the Shukhov formula (exponential
    temperature decay along a pipe due to heat loss to the surroundings),
    standard in the district heating literature for steady-state pipe
    flow:

        T_out = T_ground + (T_in - T_ground) * exp(-loss_coeff * L / (cp * m_dot))

    Reference: the Shukhov formula is the standard closed-form solution
    used throughout DH network modelling literature for this exact
    problem — see e.g. "A Two-Stage Polynomial Approach to Stochastic
    Optimization of District Heating Networks" (arxiv.org/pdf/1807.06266,
    Appendix D) and "Optimization of District Heating Network Parameters
    in Steady-State Operation" (arxiv.org/pdf/2404.18868, eq. 6-7), both
    citing the same exponential-decay derivation from a 1st-order linear
    ODE (dT/dx = -loss_coeff/(cp*m_dot) * (T(x) - T_ground)).

    Parameters
    ----------
    inlet_temp_C        : temperature entering this segment (°C) — either
                  the source's flow_temp_C (for the FIRST segment out of
                  the energy centre) or the previous segment's
                  segment_outlet_temp_C() (for every subsequent segment
                  along a path)
    mass_flow_kg_s       : mass flow rate through THIS segment (kg/s) —
                  use the segment's OWN accumulated peak (segment_peak_kW())
                  via the same mass-flow formula pipe_catalog.py's
                  size_pipe_for_peak() already uses, for consistency
                  between sizing and heat-loss physics
    length_m             : this segment's length (m)
    heat_loss_coefficient_W_per_mK : this segment's own sized pipe's
                  loss coefficient (W/m.K, combined supply+return — see
                  pipe_catalog.py's heat_loss_coefficient_W_per_mK()).
                  NOTE: this combined coefficient already accounts for
                  BOTH supply and return pipes together — using it here
                  for a one-way temperature-drop calculation is a
                  reasonable approximation for a feasibility-stage model
                  (splitting the supply-only vs return-only coefficient
                  would need the casing geometry to be modelled
                  separately for each, which pipe_catalog.py doesn't
                  currently do), not an exact physical equivalence.
    ground_temp_C        : surrounding ground temperature (°C)
    cp_J_kgK             : water specific heat (J/kg.K) — use
                  water_properties(inlet_temp_C)["cp_J_kgK"] for a
                  temperature-correct value rather than the fixed
                  default, which is only a reasonable approximation in
                  the typical DH temperature range.

    Returns
    -------
    Outlet temperature (°C) arriving at the far end of this segment.
    """
    if mass_flow_kg_s <= 0:
        raise ValueError(
            f"mass_flow_kg_s must be positive; got {mass_flow_kg_s}. A segment "
            f"with zero peak (e.g. an unused junction) has no flow to carry heat "
            f"loss physics — check for a building with zero demand."
        )
    exponent = -(heat_loss_coefficient_W_per_mK * length_m) / (cp_J_kgK * mass_flow_kg_s)
    return ground_temp_C + (inlet_temp_C - ground_temp_C) * _exp(exponent)


def _exp(x: float) -> float:
    """Thin wrapper so this module doesn't need a top-level numpy import
    just for one exponential — keeps this file's only hard dependency
    as pipe_catalog.py, consistent with its 'generic, swappable' design goal."""
    import math
    return math.exp(x)


# ── Core data structure ──────────────────────────────────────────────────────────

@dataclass
class SegmentPipeResult:
    """
    The sized pipe for ONE segment, plus the inputs that produced it —
    what size_all_segments() returns per node_id. Keeps the PipeSpec
    (DN, velocity, heat loss coefficient, cost) bundled with which
    segment it belongs to and what peak/length drove that sizing, so
    downstream code (CAPEX totals, heat-loss-along-the-route
    calculations) doesn't need to re-derive any of it.
    """
    node_id: str
    length_m: float
    peak_kW: float
    pipe: PipeSpec
    mass_flow_kg_s: float

    @property
    def capex_GBP(self) -> float:
        """Installed cost for just this segment (£/m x this segment's length)."""
        return self.pipe.cost_GBP_per_m * self.length_m

    def __repr__(self):
        return (
            f"SegmentPipeResult({self.node_id}, DN{self.pipe.DN}, "
            f"{self.length_m:.0f}m, peak={self.peak_kW:.0f}kW, "
            f"capex=£{self.capex_GBP:,.0f})"
        )


@dataclass
class TopologyNode:
    """
    One node in the tree. The pipe SEGMENT this node sits at the end of
    is the edge FROM parent_id TO this node — length_m and the resulting
    sized pipe both describe THAT segment, not the node itself.

    node_id     : unique identifier within a NetworkTopology
    parent_id   : the node this one connects back toward (None ONLY for
                  the root / energy centre)
    length_m    : length of the pipe segment from parent to this node
                  (0.0 for the root, which has no incoming segment)
    peak_kW     : THIS node's own peak heat (or cooling) demand — 0 for
                  a pure junction/branch-point with no building attached
    building_name : optional human-readable label (e.g. a real building
                  name) — purely cosmetic/reporting, the engine itself
                  never reads this field for any calculation
    """
    node_id: str
    parent_id: Optional[str]
    length_m: float
    peak_kW: float = 0.0
    building_name: Optional[str] = None

    def __repr__(self):
        label = f" ({self.building_name})" if self.building_name else ""
        return (
            f"TopologyNode({self.node_id}{label}, parent={self.parent_id}, "
            f"length={self.length_m:.0f}m, peak={self.peak_kW:.0f}kW)"
        )


@dataclass
class NetworkTopology:
    """
    A full tree topology: one root (energy centre, parent_id=None) plus
    any number of downstream nodes, each with exactly one parent.

    Building this incrementally via add_node() validates structure as
    you go (catches duplicate IDs, missing parents, multiple roots)
    rather than letting a malformed tree silently produce wrong segment
    peaks later.
    """
    name: str
    nodes: dict = field(default_factory=dict)   # node_id -> TopologyNode
    _root_id: Optional[str] = field(default=None, repr=False)
    _children: dict = field(default_factory=dict, repr=False)  # parent_id -> [child_ids]

    def add_node(
        self,
        node_id: str,
        parent_id: Optional[str],
        length_m: float,
        peak_kW: float = 0.0,
        building_name: Optional[str] = None,
    ) -> None:
        """
        Add one node to the tree. parent_id must already exist in the
        tree (or be None, for the single root node).
        """
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node_id '{node_id}' — node IDs must be unique.")

        if parent_id is None:
            if self._root_id is not None:
                raise ValueError(
                    f"A root node ('{self._root_id}') already exists — a tree can only "
                    f"have ONE root (the energy centre). Did you mean to set parent_id "
                    f"to an existing node instead of None?"
                )
            if length_m != 0:
                raise ValueError(
                    f"Root node '{node_id}' (parent_id=None) should have length_m=0 — "
                    f"the root has no incoming pipe segment of its own."
                )
            self._root_id = node_id
        else:
            if parent_id not in self.nodes:
                raise ValueError(
                    f"parent_id '{parent_id}' for node '{node_id}' doesn't exist yet — "
                    f"add parent nodes before their children."
                )
            if length_m <= 0:
                raise ValueError(
                    f"Node '{node_id}' has length_m={length_m} — every non-root segment "
                    f"must have a positive length."
                )

        self.nodes[node_id] = TopologyNode(
            node_id=node_id, parent_id=parent_id, length_m=length_m,
            peak_kW=peak_kW, building_name=building_name,
        )
        self._children.setdefault(parent_id, []).append(node_id)

    @property
    def root_id(self) -> str:
        if self._root_id is None:
            raise ValueError(f"Topology '{self.name}' has no root node yet — call add_node() with parent_id=None first.")
        return self._root_id

    def children_of(self, node_id: str) -> list:
        """Direct children of a node (empty list for leaf nodes)."""
        return self._children.get(node_id, [])

    def descendants_of(self, node_id: str) -> list:
        """
        ALL nodes downstream of node_id (children, grandchildren, etc.),
        not including node_id itself. This is what segment_peak_kW()
        sums over.
        """
        result = []
        stack = list(self.children_of(node_id))
        while stack:
            current = stack.pop()
            result.append(current)
            stack.extend(self.children_of(current))
        return result

    def segment_peak_kW(self, node_id: str) -> float:
        """
        Peak flow (kW) the pipe segment FROM node_id's PARENT TO node_id
        must carry — this node's own peak_kW plus every downstream
        node's peak_kW (see module docstring for the diversified-peak
        caveat: this is a simple sum, not an hour-by-hour coincident
        peak, which is the standard conservative feasibility-stage
        simplification).
        """
        node = self.nodes[node_id]
        downstream_total = sum(self.nodes[d].peak_kW for d in self.descendants_of(node_id))
        return node.peak_kW + downstream_total

    def total_length_m(self) -> float:
        """Sum of every segment's length — the real total network route length."""
        return sum(n.length_m for n in self.nodes.values())

    def total_peak_kW(self) -> float:
        """Sum of every node's OWN peak — i.e. the energy centre's required peak output."""
        return sum(n.peak_kW for n in self.nodes.values())

    def leaf_nodes(self) -> list:
        """Nodes with no children — i.e. actual end connections, not junctions."""
        return [nid for nid in self.nodes if nid not in self._children or not self._children[nid]]

    def path_to_root(self, node_id: str) -> list:
        """
        Every node on the path from node_id back to the root, INCLUDING
        both endpoints — e.g. for delivered-temperature calculations
        that need to know every segment heat is lost across, from a
        given building back to the energy centre.
        """
        path = [node_id]
        current = node_id
        while self.nodes[current].parent_id is not None:
            current = self.nodes[current].parent_id
            path.append(current)
        return path

    def validate(self) -> None:
        """
        Check the tree is well-formed: exactly one root, every node
        reachable from the root (no orphaned subtrees from a typo'd
        parent_id), no cycles. add_node() already prevents most of
        these as you build, but this is a final whole-tree check —
        useful after building a topology from external data (e.g. a
        CSV of segments) rather than incrementally via add_node().
        """
        if self._root_id is None:
            raise ValueError(f"Topology '{self.name}' has no root node.")

        reachable = {self._root_id}
        stack = [self._root_id]
        while stack:
            current = stack.pop()
            for child in self.children_of(current):
                if child in reachable:
                    raise ValueError(
                        f"Cycle detected: node '{child}' is reachable from the root "
                        f"via more than one path — this must be a tree, not a general graph."
                    )
                reachable.add(child)
                stack.append(child)

        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            raise ValueError(
                f"Topology '{self.name}' has {len(unreachable)} node(s) not reachable "
                f"from the root: {sorted(unreachable)} — check for a typo'd parent_id."
            )

    def summary(self) -> dict:
        """High-level stats — total length, total peak, node/leaf counts."""
        return {
            "name": self.name,
            "total_length_m": round(self.total_length_m(), 1),
            "total_peak_kW": round(self.total_peak_kW(), 1),
            "n_nodes": len(self.nodes),
            "n_leaf_nodes": len(self.leaf_nodes()),
            "max_segment_peak_kW": round(
                max(self.segment_peak_kW(nid) for nid in self.nodes if nid != self.root_id), 1
            ) if len(self.nodes) > 1 else 0.0,
        }

    # ── Per-segment pipe sizing ──────────────────────────────────────────────

    def size_all_segments(
        self,
        flow_temp_C: float,
        return_temp_C: float,
        **pipe_kwargs,
    ) -> dict:
        """
        Size EVERY segment's own pipe, using THAT segment's own
        accumulated peak (segment_peak_kW()) and length — not one
        trunk size applied uniformly. This is the actual point of
        having a real topology instead of network.py's single
        representative trunk: a branch carrying 250kW to one building
        correctly comes out as a much smaller (and cheaper) pipe than
        the segment near the energy centre carrying the full
        accumulated network peak.

        The root node (energy centre) itself has no incoming segment
        (length_m=0) and is skipped — there's nothing to size for it.

        Parameters
        ----------
        flow_temp_C, return_temp_C : design supply/return temperatures
                  (°C). Currently ONE fixed pair applied to every
                  segment — this is where weather-compensated flow
                  temperature will plug in later (varying flow_temp_C
                  hour-by-hour rather than a single fixed design value),
                  without needing to change this method's structure.
        **pipe_kwargs : passed through to size_pipe_for_peak() for
                  EVERY segment (construction, insulation_series,
                  max_velocity_ms, etc.) — i.e. currently one shared
                  pipe spec choice across the whole network. A future
                  refinement could vary construction (e.g. twin pipe on
                  smaller branches, single on the trunk) per segment;
                  not built here since nothing in this project currently
                  needs that distinction made automatically.

        Returns
        -------
        dict: node_id -> SegmentPipeResult, for every non-root node.
        """
        results = {}
        delta_T_K = abs(flow_temp_C - return_temp_C)
        for node_id, node in self.nodes.items():
            if node_id == self.root_id:
                continue   # root has no incoming segment to size
            peak_kW = self.segment_peak_kW(node_id)
            pipe = size_pipe_for_peak(
                peak_heat_kW=peak_kW,
                flow_temp_C=flow_temp_C,
                return_temp_C=return_temp_C,
                **pipe_kwargs,
            )
            # Same mass-flow formula pipe_catalog.py's size_pipe_for_peak()
            # uses internally (Q = m_dot * cp * delta_T) -- kept consistent
            # here so the Shukhov temperature-drop calculation downstream
            # uses the SAME flow rate that was used to size the pipe in
            # the first place, not a separately-derived (and potentially
            # inconsistent) value.
            props = water_properties(flow_temp_C)
            mass_flow_kg_s = (peak_kW * 1000.0) / (props["cp_J_kgK"] * delta_T_K)
            results[node_id] = SegmentPipeResult(
                node_id=node_id, length_m=node.length_m, peak_kW=peak_kW,
                pipe=pipe, mass_flow_kg_s=mass_flow_kg_s,
            )
        return results

    @staticmethod
    def total_capex_GBP(sized_segments: dict) -> float:
        """Sum of every segment's own installed cost — the real network CAPEX,
        built up from actual per-segment pipe sizes rather than one trunk
        DN's £/m applied to the whole route length."""
        return sum(s.capex_GBP for s in sized_segments.values())

    @staticmethod
    def dn_by_segment(sized_segments: dict) -> dict:
        """Convenience: {node_id: DN} — useful for a quick look at how
        pipe size tapers down from the energy centre to the leaves."""
        return {nid: s.pipe.DN for nid, s in sized_segments.items()}

    # ── Delivered temperature and network-wide heat loss ─────────────────────

    def delivered_temperature_C(
        self,
        node_id: str,
        sized_segments: dict,
        source_flow_temp_C: float,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    ) -> float:
        """
        The actual temperature ARRIVING at node_id, after real heat loss
        across every segment on the path from the energy centre to it —
        NOT the source's design flow temperature, which is what
        network.py's single-trunk model implicitly assumed customers
        always received.

        Walks path_to_root(node_id) from the ENERGY CENTRE END inward
        (i.e. root-to-node order, since each segment's outlet becomes
        the next segment's inlet), applying segment_outlet_temp_C()
        (the Shukhov formula) once per segment using that segment's own
        sized pipe (loss coefficient) and own mass flow rate.

        Parameters
        ----------
        node_id             : the building/node to compute delivered
                  temperature for
        sized_segments       : the dict returned by size_all_segments()
                  — must have been built with the SAME flow/return
                  temperatures as source_flow_temp_C implies, or the
                  mass flow rates won't be consistent with the inlet
                  temperature being propagated here
        source_flow_temp_C   : temperature LEAVING the energy centre
                  (i.e. the first segment's inlet temperature) — this is
                  where weather-compensated flow temperature plugs in:
                  pass a different value per hour to see how delivered
                  temperature varies through the year, rather than only
                  checking the fixed design point
        ground_temp_C        : surrounding ground temperature (°C)

        Returns
        -------
        Temperature (°C) actually arriving at node_id.
        """
        path_from_ec = list(reversed(self.path_to_root(node_id)))  # [EC, ..., node_id]
        current_temp = source_flow_temp_C
        for seg_node_id in path_from_ec[1:]:   # skip EC itself -- no incoming segment
            seg = sized_segments[seg_node_id]
            loss_coeff = heat_loss_coefficient_W_per_mK(
                seg.pipe.DN, seg.pipe.construction,
            )
            props = water_properties(current_temp)
            current_temp = segment_outlet_temp_C(
                inlet_temp_C=current_temp,
                mass_flow_kg_s=seg.mass_flow_kg_s,
                length_m=seg.length_m,
                heat_loss_coefficient_W_per_mK=loss_coeff,
                ground_temp_C=ground_temp_C,
                cp_J_kgK=props["cp_J_kgK"],
            )
        return current_temp

    def check_minimum_delivered_temperature(
        self,
        sized_segments: dict,
        source_flow_temp_C: float,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
        min_temp_C: float = MIN_DELIVERED_TEMP_C,
    ) -> dict:
        """
        Compliance check: does every building actually connected to this
        network (every node with a building_name) receive at least
        min_temp_C, after real route heat loss — not just "is the
        source's design flow temperature high enough", which says
        nothing about what arrives at the far end of a long branch.

        Returns
        -------
        dict, keyed by building_name -> {
            "node_id", "delivered_temp_C", "compliant" (bool),
            "margin_C" (positive = above the minimum, negative = below it)
        }
        Plus a top-level "all_compliant" bool and "worst_case_building"
        (the building with the lowest delivered temperature — usually,
        but not necessarily, the one furthest from the energy centre by
        route length, since loss also depends on pipe size/flow rate
        along the way, not distance alone).
        """
        results = {}
        for node_id, node in self.nodes.items():
            if node.building_name is None or node_id == self.root_id:
                continue
            delivered = self.delivered_temperature_C(
                node_id, sized_segments, source_flow_temp_C, ground_temp_C,
            )
            margin = delivered - min_temp_C
            results[node.building_name] = {
                "node_id": node_id,
                "delivered_temp_C": round(delivered, 2),
                "compliant": bool(margin >= 0),
                "margin_C": round(margin, 2),
            }

        all_compliant = all(r["compliant"] for r in results.values()) if results else True
        worst_case = min(results, key=lambda k: results[k]["delivered_temp_C"]) if results else None

        return {
            "by_building": results,
            "all_compliant": all_compliant,
            "worst_case_building": worst_case,
            "worst_case_delivered_temp_C": results[worst_case]["delivered_temp_C"] if worst_case else None,
            "min_required_temp_C": min_temp_C,
        }

    def minimum_safe_flow_temp_C(
        self,
        return_temp_C: float,
        min_temp_C: float = MIN_DELIVERED_TEMP_C,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
        search_low_C: float = 35.0,
        search_high_C: float = 95.0,
        tolerance_C: float = 0.05,
        **pipe_kwargs,
    ) -> float:
        """
        The REAL question behind "how do we ensure 60°C is always met":
        not "is THIS flow temperature safe", but "what's the LOWEST flow
        temperature that's STILL safe for every building on THIS real
        network" — found by binary search rather than guessed at, using
        the network's own real route lengths and real pipe sizing.

        This is what a weather-compensation curve's mild-end floor
        SHOULD be set to (with a margin for safety/control tolerance on
        top — see this method's docstring note on margin, below) — not
        an arbitrary round number like "60°C" picked to match the
        regulatory minimum itself, which ignores that real transit
        losses eat into that minimum before it ever reaches a building.

        Binary search works because delivered temperature is monotonic
        in source flow temperature for a fixed topology/return temp (a
        hotter source always produces a hotter — or equal — delivered
        temperature everywhere, never colder) — so there's exactly one
        crossover point, not multiple solutions to choose between.

        Parameters
        ----------
        return_temp_C        : design return temperature (°C) — needed
                  to size each segment's pipe/mass flow at each trial
                  flow temp during the search (sizing depends on BOTH
                  flow and return temp via delta_T)
        min_temp_C           : the delivered-temperature floor to solve
                  for (default: MIN_DELIVERED_TEMP_C, 60°C)
        ground_temp_C        : surrounding ground temperature (°C)
        search_low_C, search_high_C : bounds for the binary search —
                  widen these if the true answer might sit outside the
                  default 35-95°C range for an unusual network
        tolerance_C           : stop the search once the bracket is
                  narrower than this (°C) — 0.05°C is far tighter than
                  any real control system can hold anyway, so this is
                  about search precision, not physical precision
        **pipe_kwargs         : passed through to size_all_segments()
                  (construction, insulation_series, etc.)

        Returns
        -------
        The minimum source flow temperature (°C) at which every
        connected building's delivered temperature is >= min_temp_C.

        NOTE on margin: this returns the EXACT crossover point — at
        precisely this flow temp, the worst-case building is AT
        min_temp_C, with zero margin. A real control system can't hold
        flow temp with zero error, and this calculation doesn't include
        weather/demand variation beyond what's already baked into the
        topology's peak figures. Add a real margin (a few °C, chosen
        based on the control system's actual achievable tolerance) on
        top of this result before using it as an actual compensation-
        curve floor — this function answers "what's the physical limit",
        not "what's a safe operating setpoint".
        """
        lo, hi = search_low_C, search_high_C

        def is_safe(flow_temp_C: float) -> bool:
            sized = self.size_all_segments(
                flow_temp_C=flow_temp_C, return_temp_C=return_temp_C, **pipe_kwargs,
            )
            compliance = self.check_minimum_delivered_temperature(
                sized, source_flow_temp_C=flow_temp_C,
                ground_temp_C=ground_temp_C, min_temp_C=min_temp_C,
            )
            return compliance["all_compliant"]

        if not is_safe(hi):
            raise ValueError(
                f"Even the upper search bound ({hi}°C) doesn't reach {min_temp_C}°C "
                f"delivered everywhere on this network — check the network's route "
                f"lengths/pipe sizing, or widen search_high_C."
            )
        if is_safe(lo):
            # Even the lowest bound is already safe -- no need to search,
            # return it directly rather than searching needlessly
            return lo

        while hi - lo > tolerance_C:
            mid = (lo + hi) / 2.0
            if is_safe(mid):
                hi = mid
            else:
                lo = mid

        return round(hi, 2)

    def network_heat_loss_kW(
        self,
        sized_segments: dict,
        source_flow_temp_C: float,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    ) -> dict:
        """
        Total heat ACTUALLY lost across the whole network, segment by
        segment, using each segment's REAL inlet temperature (i.e.
        accounting for the fact that downstream segments already start
        from a slightly-cooled inlet, not the source's full design flow
        temperature) — not network.py's old approach of one mean-
        temperature figure applied to the whole route length at once.

        This is the number that needs to be ADDED ON TOP of building
        demand when sizing/dispatching sources — the network itself
        consumes real heat just transporting demand around, and nothing
        upstream of this function currently feeds that back into
        anything (a real, separate gap from the per-segment physics
        itself — see this method's caller for where that connection
        actually gets made).

        Returns
        -------
        dict: by_segment (node_id -> kW lost in that segment) and
        total_kW (sum across the whole network).
        """
        by_segment = {}
        # Process nodes in an order where every node's parent has
        # already been processed (BFS from the root), so each segment's
        # inlet temperature is always its parent's already-computed
        # outlet temperature, not the raw source temperature.
        outlet_temp = {self.root_id: source_flow_temp_C}
        queue = list(self.children_of(self.root_id))
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]
            seg = sized_segments[node_id]
            inlet_temp = outlet_temp[node.parent_id]
            loss_coeff = heat_loss_coefficient_W_per_mK(seg.pipe.DN, seg.pipe.construction)
            props = water_properties(inlet_temp)
            out_temp = segment_outlet_temp_C(
                inlet_temp_C=inlet_temp,
                mass_flow_kg_s=seg.mass_flow_kg_s,
                length_m=seg.length_m,
                heat_loss_coefficient_W_per_mK=loss_coeff,
                ground_temp_C=ground_temp_C,
                cp_J_kgK=props["cp_J_kgK"],
            )
            outlet_temp[node_id] = out_temp
            # Heat lost in THIS segment (kW) = mass_flow * cp * (T_in - T_out)
            loss_kW = seg.mass_flow_kg_s * props["cp_J_kgK"] * (inlet_temp - out_temp) / 1000.0
            by_segment[node_id] = loss_kW
            queue.extend(self.children_of(node_id))

        return {
            "by_segment_kW": by_segment,
            "total_kW": sum(by_segment.values()),
        }

    def network_heat_loss_kW_hourly(
        self,
        sized_segments: dict,
        source_flow_temp_C: float,
        ground_temp_C_hourly: Optional[np.ndarray] = None,
        n_hours: int = 8760,
    ) -> dict:
        """
        Full-year HOURLY network heat loss — the real number to add on
        top of hourly building demand before dispatch, rather than one
        annual/design-point figure applied uniformly.

        Source flow temperature is held FIXED across the year (a
        deliberate simplification — see this project's recent design
        discussion: this is a feasibility-stage model, and stacking
        weather-compensated flow temperature on top of the topology/
        carbon/heat-loss work was judged to be adding an operational-
        efficiency variable that obscured the core economic question,
        which should be assessed conservatively). What DOES vary hourly
        here is GROUND TEMPERATURE — see seasonal_ground_temp_C() above
        — since holding ground temp fixed at one annual average
        understates real winter heat loss (when ground is genuinely
        colder than the annual mean) and overstates summer loss, the
        wrong direction for a conservative assessment.

        Because flow temperature and pipe sizing are both fixed across
        the year, only the ground-temp term in the Shukhov formula
        actually changes hour to hour — this is exploited here to
        compute all 8760 hours via a single vectorised pass through the
        network's segments (one Shukhov calculation per segment, across
        all hours at once) rather than looping the scalar
        network_heat_loss_kW() 8760 times, which would needlessly repeat
        identical pipe-property lookups every single hour for no benefit.

        Parameters
        ----------
        sized_segments         : from size_all_segments() — built ONCE
                  at the fixed source_flow_temp_C, reused for every hour
        source_flow_temp_C      : the FIXED source flow temperature (°C)
                  — same value for all 8760 hours (see note above)
        ground_temp_C_hourly     : optional (n_hours,) array of ground
                  temperatures. If None (default), uses
                  seasonal_ground_temp_C() — the real, sourced UK
                  seasonal curve (see module constants).
        n_hours                  : hours in a year (8760, project convention)

        Returns
        -------
        dict: {
            "total_kW_hourly": (n_hours,) array, network heat loss each hour,
            "by_segment_kW_hourly": {node_id: (n_hours,) array},
            "annual_total_MWh": total annual heat loss in MWh
        }
        """
        if ground_temp_C_hourly is None:
            ground_temp_C_hourly = seasonal_ground_temp_C(np.arange(n_hours), n_hours=n_hours)
        else:
            ground_temp_C_hourly = np.asarray(ground_temp_C_hourly, dtype=float)
            if len(ground_temp_C_hourly) != n_hours:
                raise ValueError(
                    f"ground_temp_C_hourly must have {n_hours} entries; got "
                    f"{len(ground_temp_C_hourly)}."
                )

        # Fixed flow temp -> water properties at the source don't vary by
        # hour either, computed once rather than recomputed every hour
        props = water_properties(source_flow_temp_C)
        cp = props["cp_J_kgK"]

        by_segment_hourly = {}
        # outlet_temp_hourly[node_id] is an (n_hours,) array — the
        # temperature arriving at node_id, every hour. Root's "outlet"
        # is simply the fixed source flow temp, broadcast across all hours.
        outlet_temp_hourly = {self.root_id: np.full(n_hours, source_flow_temp_C)}

        queue = list(self.children_of(self.root_id))
        while queue:
            node_id = queue.pop(0)
            node = self.nodes[node_id]
            seg = sized_segments[node_id]
            inlet_temp_hourly = outlet_temp_hourly[node.parent_id]
            loss_coeff = heat_loss_coefficient_W_per_mK(seg.pipe.DN, seg.pipe.construction)

            # Shukhov formula, vectorised across all n_hours at once --
            # ground_temp_C_hourly and inlet_temp_hourly are both
            # (n_hours,) arrays; mass_flow_kg_s/length_m/loss_coeff are
            # scalars (fixed sizing), so this is a single elementwise
            # numpy expression covering the whole year in one pass.
            exponent = -(loss_coeff * seg.length_m) / (cp * seg.mass_flow_kg_s)
            out_temp_hourly = ground_temp_C_hourly + (inlet_temp_hourly - ground_temp_C_hourly) * np.exp(exponent)

            outlet_temp_hourly[node_id] = out_temp_hourly
            loss_kW_hourly = seg.mass_flow_kg_s * cp * (inlet_temp_hourly - out_temp_hourly) / 1000.0
            by_segment_hourly[node_id] = loss_kW_hourly
            queue.extend(self.children_of(node_id))

        total_kW_hourly = sum(by_segment_hourly.values())
        annual_total_MWh = float(total_kW_hourly.sum()) / 1000.0   # kWh -> MWh (each hour's kW = that hour's kWh)

        return {
            "total_kW_hourly": total_kW_hourly,
            "by_segment_kW_hourly": by_segment_hourly,
            "annual_total_MWh": annual_total_MWh,
        }

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
) -> NetworkTopology:
    """
    Build the real (calibrated) Ealing Town Centre worked-example
    topology. See module docstring for the route's sourcing/calibration
    note, and EALING_SEGMENTS/BUILDING_NAME_MAP above for the actual
    data.

    Parameters
    ----------
    peak_kW_by_building : optional {building_name: peak_kW} dict — e.g.
                  built from demand_synthesis.py's synthesise_network()
                  output (each node's "peak_heat_kW"). If omitted,
                  building leaf nodes default to peak_kW=0 (useful for
                  just inspecting the route geometry/lengths without
                  needing a real demand run first).

    Returns
    -------
    NetworkTopology, fully built and validated.
    """
    peak_kW_by_building = peak_kW_by_building or {}

    topo = NetworkTopology(name="Ealing Town Centre (worked example)")
    topo.add_node("EC", parent_id=None, length_m=0.0, peak_kW=0.0,
                   building_name="Energy Centre")

    for node_id, parent_id, length_m, is_building in EALING_SEGMENTS:
        building_name = BUILDING_NAME_MAP.get(node_id) if is_building else None
        peak_kW = peak_kW_by_building.get(building_name, 0.0) if building_name else 0.0
        topo.add_node(
            node_id, parent_id=parent_id, length_m=length_m,
            peak_kW=peak_kW, building_name=building_name,
        )

    topo.validate()
    return topo


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  network_topology.py — self-test")
    print("=" * 70)

    # --- Minimal generic example, built from scratch ---
    print("\n  Minimal generic tree (3 nodes, no real-world data):")
    simple = NetworkTopology(name="Simple test tree")
    simple.add_node("EC", parent_id=None, length_m=0.0, peak_kW=0.0)
    simple.add_node("A", parent_id="EC", length_m=100.0, peak_kW=500.0)
    simple.add_node("B", parent_id="A",  length_m=80.0,  peak_kW=200.0)
    simple.add_node("C", parent_id="A",  length_m=60.0,  peak_kW=150.0)
    simple.validate()
    print(f"    {simple}")
    print(f"    segment_peak_kW('A'): {simple.segment_peak_kW('A')} kW "
          f"(expect 500+200+150=850, A's own + B + C downstream)")
    print(f"    segment_peak_kW('B'): {simple.segment_peak_kW('B')} kW (expect 200, leaf node)")
    print(f"    total_length_m: {simple.total_length_m()} m (expect 100+80+60=240)")
    print(f"    total_peak_kW: {simple.total_peak_kW()} kW (expect 500+200+150=850)")
    print(f"    leaf_nodes: {simple.leaf_nodes()} (expect ['B', 'C'])")
    print(f"    path_to_root('B'): {simple.path_to_root('B')} (expect ['B', 'A', 'EC'])")

    # --- Error handling checks ---
    print("\n  Error handling — malformed trees should fail loudly:")
    try:
        bad = NetworkTopology(name="Duplicate ID test")
        bad.add_node("EC", parent_id=None, length_m=0.0)
        bad.add_node("EC", parent_id=None, length_m=0.0)
        print("    ✗ FAIL: should have raised on duplicate node_id")
    except ValueError as e:
        print(f"    ✓ Correctly raised (duplicate ID): {str(e)[:70]}...")

    try:
        bad2 = NetworkTopology(name="Missing parent test")
        bad2.add_node("EC", parent_id=None, length_m=0.0)
        bad2.add_node("A", parent_id="DOES_NOT_EXIST", length_m=50.0)
        print("    ✗ FAIL: should have raised on missing parent")
    except ValueError as e:
        print(f"    ✓ Correctly raised (missing parent): {str(e)[:70]}...")

    try:
        bad3 = NetworkTopology(name="Double root test")
        bad3.add_node("EC1", parent_id=None, length_m=0.0)
        bad3.add_node("EC2", parent_id=None, length_m=0.0)
        print("    ✗ FAIL: should have raised on second root")
    except ValueError as e:
        print(f"    ✓ Correctly raised (double root): {str(e)[:70]}...")

    # --- Ealing worked example, geometry only (no demand data) ---
    print("\n  Ealing Town Centre worked example (route geometry only):")
    ealing_geom = ealing_town_centre_topology()
    for k, v in ealing_geom.summary().items():
        print(f"    {k}: {v}")
    print(f"\n  Report's published Phase 2 cumulative network length: 3,165 m")
    print(f"  This topology's total_length_m: {ealing_geom.total_length_m():.0f} m")

    # --- Ealing worked example, WITH real demand data from demand_synthesis.py ---
    print("\n  Ealing Town Centre worked example, WITH real per-building peaks:")
    sys.path.insert(0, str(_PROJECT_ROOT))
    import numpy as np
    import pandas as pd
    from profiles.demand_synthesis import synthesise_network

    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5 + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    weather_df = pd.DataFrame({"temp_drybulb_C": T})
    scenario = {
        "demand_nodes": [
            {"name": "Perceval House",       "type": "office",      "floor_area_m2": 8500},
            {"name": "High Street Retail",   "type": "retail",      "floor_area_m2": 3000},
            {"name": "Ealing Hospital Wing", "type": "hospital",    "floor_area_m2": 12000},
            {"name": "Dickens Yard Ph1",     "type": "residential", "units": 350},
            {"name": "Broadway Hotel",       "type": "hotel",       "floor_area_m2": 5000},
            {"name": "Ellen Wilkinson Sch",  "type": "school",      "floor_area_m2": 6000},
        ]
    }
    demand_result = synthesise_network(weather_df, scenario)
    peak_by_building = {
        n["name"]: n["peak_heat_kW"] for n in demand_result["nodes"]
    }
    print(f"    Per-building peak_heat_kW from demand_synthesis.py: {peak_by_building}")

    ealing_real = ealing_town_centre_topology(peak_kW_by_building=peak_by_building)
    for k, v in ealing_real.summary().items():
        print(f"    {k}: {v}")

    print(f"\n  Segment peaks along the route (kW each segment must carry):")
    for node_id, parent_id, length_m, is_building in EALING_SEGMENTS:
        seg_peak = ealing_real.segment_peak_kW(node_id)
        label = ealing_real.nodes[node_id].building_name or "(junction)"
        print(f"    {node_id:<5} (from {parent_id:<5}, {length_m:>5.0f}m): "
              f"{seg_peak:>7.1f} kW  {label}")

    # --- NEW: per-segment pipe sizing, the actual point of this module ---
    print(f"\n  Per-segment pipe sizing (real DN tapering down from the energy")
    print(f"  centre to individual buildings — this is what a single-trunk")
    print(f"  model CANNOT do):")
    sized = ealing_real.size_all_segments(flow_temp_C=70.0, return_temp_C=40.0)
    print(f"  {'Segment':<6} {'Length':>8} {'Peak kW':>9} {'DN':>5} {'v m/s':>7} {'£/m':>8} {'Segment CAPEX':>15}")
    for node_id, parent_id, length_m, is_building in EALING_SEGMENTS:
        s = sized[node_id]
        print(f"  {node_id:<6} {s.length_m:>7.0f}m {s.peak_kW:>9.1f} {s.pipe.DN:>5} "
              f"{s.pipe.velocity_ms:>7.2f} £{s.pipe.cost_GBP_per_m:>6.0f} £{s.capex_GBP:>14,.0f}")

    real_topology_capex = NetworkTopology.total_capex_GBP(sized)
    print(f"\n  Real per-segment-sized total CAPEX: £{real_topology_capex:,.0f}")

    # --- Compare against the OLD single-trunk approach on the same data,
    #     to show the real difference this makes ---
    from pipe_catalog import size_pipe_for_peak as _size_for_peak
    single_trunk_pipe = _size_for_peak(
        peak_heat_kW=ealing_real.total_peak_kW(),
        flow_temp_C=70.0, return_temp_C=40.0,
    )
    single_trunk_capex = single_trunk_pipe.cost_GBP_per_m * ealing_real.total_length_m()
    print(f"  OLD single-trunk approach (one DN{single_trunk_pipe.DN} pipe for the WHOLE "
          f"{ealing_real.total_length_m():.0f}m route): £{single_trunk_capex:,.0f}")
    print(f"  Difference: £{single_trunk_capex - real_topology_capex:,.0f} "
          f"({(single_trunk_capex/real_topology_capex - 1)*100:+.0f}% vs real per-segment sizing)")
    print(f"  -> the single-trunk model sizes EVERY branch as if it carried the FULL network")
    print(f"     peak — overstating cost on every branch that isn't the segment nearest the")
    print(f"     energy centre, which is most of the network by length.")

    # --- DN tapering check ---
    dn_map = NetworkTopology.dn_by_segment(sized)
    print(f"\n  DN by segment (should taper down moving away from the energy centre):")
    print(f"    Segment nearest EC (N1): DN{dn_map['N1']}")
    print(f"    A leaf segment (N15):    DN{dn_map['N15']}")

    # --- NEW: delivered temperature at each building, after real route heat loss ---
    print(f"\n  Delivered temperature at each building (source flow = 70.0°C,")
    print(f"  after real per-segment heat loss along the route to get there):")
    compliance = ealing_real.check_minimum_delivered_temperature(
        sized_segments=sized, source_flow_temp_C=70.0,
    )
    for building, r in compliance["by_building"].items():
        status = "✓ compliant" if r["compliant"] else "✗ BELOW MINIMUM"
        print(f"    {building:<20} ({r['node_id']:<5}): {r['delivered_temp_C']:>6.2f}°C  "
              f"margin={r['margin_C']:>+6.2f}°C  {status}")
    print(f"\n  All buildings compliant (>= {MIN_DELIVERED_TEMP_C}°C)? {compliance['all_compliant']}")
    print(f"  Worst case: {compliance['worst_case_building']} at "
          f"{compliance['worst_case_delivered_temp_C']}°C")

    # --- NEW: total network heat loss, the real number that must be
    #     added on top of building demand ---
    print(f"\n  Network-wide heat loss (the real extra demand the energy centre")
    print(f"  must cover beyond what buildings actually request):")
    loss_result = ealing_real.network_heat_loss_kW(
        sized_segments=sized, source_flow_temp_C=70.0,
    )
    print(f"    Total network heat loss: {loss_result['total_kW']:.1f} kW")
    print(f"    Building demand (peak):  {ealing_real.total_peak_kW():.1f} kW")
    print(f"    Network loss as % of building demand: "
          f"{loss_result['total_kW']/ealing_real.total_peak_kW()*100:.1f}%")
    print(f"    -> sources must actually supply "
          f"{ealing_real.total_peak_kW() + loss_result['total_kW']:.1f} kW at peak, "
          f"not just the {ealing_real.total_peak_kW():.1f} kW buildings request.")

    # --- NEW: full-year hourly heat loss with REAL seasonal ground temp ---
    print(f"\n  Seasonal ground temperature model (real UK data — Busby 2015,")
    print(f"  106 Met Office soil stations — not a fixed annual placeholder):")
    ground_hourly = seasonal_ground_temp_C(np.arange(8760))
    print(f"    Mean: {ground_hourly.mean():.2f}°C, "
          f"Jan mean: {ground_hourly[:744].mean():.2f}°C, "
          f"Jul mean: {ground_hourly[4344:5088].mean():.2f}°C")
    print(f"    (Ground genuinely colder in winter, warmer in summer, lagged ~1 month behind air temp)")

    print(f"\n  Full-year HOURLY network heat loss (fixed 70°C source flow, real")
    print(f"  seasonal ground temp — the actual number to feed into dispatch):")
    hourly_loss = ealing_real.network_heat_loss_kW_hourly(
        sized_segments=sized, source_flow_temp_C=70.0,
    )
    print(f"    Annual total: {hourly_loss['annual_total_MWh']:.1f} MWh/year")
    print(f"    Jan mean hourly loss: {hourly_loss['total_kW_hourly'][:744].mean():.1f} kW")
    print(f"    Jul mean hourly loss: {hourly_loss['total_kW_hourly'][4344:5088].mean():.1f} kW")
    print(f"    (Winter loss higher than summer -- colder ground AND the model now reflects it)")

    # --- Show what happens at a LOWER (weather-compensated-style) flow temp ---
    print(f"\n  Same network at a LOWER flow temperature (50°C, e.g. a mild-weather")
    print(f"  weather-compensated value) — delivered temp drops further, may breach minimum:")
    sized_50 = ealing_real.size_all_segments(flow_temp_C=50.0, return_temp_C=30.0)
    compliance_50 = ealing_real.check_minimum_delivered_temperature(
        sized_segments=sized_50, source_flow_temp_C=50.0,
    )
    for building, r in compliance_50["by_building"].items():
        status = "✓ compliant" if r["compliant"] else "✗ BELOW MINIMUM"
        print(f"    {building:<20}: {r['delivered_temp_C']:>6.2f}°C  {status}")
    print(f"  All compliant at 50°C source flow? {compliance_50['all_compliant']}")

    # --- NEW: solve for the REAL minimum safe flow temperature ---
    print(f"\n  Solving for the ACTUAL minimum safe flow temperature on this real")
    print(f"  network (return=40°C, min delivered=60°C) — not a guessed round number:")
    min_safe_flow = ealing_real.minimum_safe_flow_temp_C(return_temp_C=40.0, min_temp_C=60.0)
    print(f"    Minimum safe source flow temp: {min_safe_flow:.2f}°C")
    print(f"    (i.e. a 60.0°C compensation-curve floor is NOT quite safe for this")
    print(f"    network's real route lengths — the true physical floor is "
          f"{min_safe_flow:.2f}°C,")
    print(f"    eaten into by real transit losses on the longest branch)")

    # Confirm: at exactly this flow temp, compliance should hold (with ~0 margin)
    sized_at_floor = ealing_real.size_all_segments(flow_temp_C=min_safe_flow, return_temp_C=40.0)
    compliance_at_floor = ealing_real.check_minimum_delivered_temperature(
        sized_segments=sized_at_floor, source_flow_temp_C=min_safe_flow, min_temp_C=60.0,
    )
    print(f"\n    Confirming at {min_safe_flow:.2f}°C flow: all compliant? "
          f"{compliance_at_floor['all_compliant']}, "
          f"worst case {compliance_at_floor['worst_case_delivered_temp_C']}°C "
          f"(should be very close to 60.0°C, ~0 margin)")

    # And confirm just BELOW the floor breaches compliance
    sized_below_floor = ealing_real.size_all_segments(flow_temp_C=min_safe_flow - 1.0, return_temp_C=40.0)
    compliance_below_floor = ealing_real.check_minimum_delivered_temperature(
        sized_segments=sized_below_floor, source_flow_temp_C=min_safe_flow - 1.0, min_temp_C=60.0,
    )
    print(f"    Confirming at {min_safe_flow - 1.0:.2f}°C flow (1°C below): all compliant? "
          f"{compliance_below_floor['all_compliant']} (should be False)")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert simple.segment_peak_kW("A") == 850.0, "Segment peak should sum node + all descendants"
    assert simple.segment_peak_kW("B") == 200.0, "Leaf segment peak should equal just its own peak"
    assert simple.total_length_m() == 240.0, "Total length should sum all segments"
    assert simple.total_peak_kW() == 850.0, "Total peak should sum all node peaks"
    assert set(simple.leaf_nodes()) == {"B", "C"}, "Leaf nodes should be exactly B and C"
    assert simple.path_to_root("B") == ["B", "A", "EC"], "Path to root should list every ancestor in order"

    ealing_geom.validate()  # should not raise
    assert abs(ealing_geom.total_length_m() - 3165.0) < 1.0, \
        "Ealing worked example should match the report's published total length (3,165m) after calibration"
    assert ealing_geom.nodes["EC"].parent_id is None, "Energy centre should be the root (no parent)"
    # Building nodes can sit mid-branch (more network beyond them) as well
    # as at true leaves -- check the right thing: every mapped building
    # name actually exists somewhere in the tree, not that it's a leaf
    nodes_with_buildings = [n for n in ealing_geom.nodes.values() if n.building_name and n.node_id != "EC"]
    assert len(nodes_with_buildings) == len(BUILDING_NAME_MAP), \
        f"Expected {len(BUILDING_NAME_MAP)} building-mapped nodes, found {len(nodes_with_buildings)}"

    # The segment closest to the energy centre should carry the LARGEST
    # peak (everything downstream funnels through it) -- this is the
    # exact thing a single-trunk model can't capture: branch pipes near
    # the edges should carry much less than the trunk near the energy centre
    root_child = ealing_real.children_of(ealing_real.root_id)[0]
    leaf_example = ealing_real.leaf_nodes()[0]
    assert ealing_real.segment_peak_kW(root_child) >= ealing_real.segment_peak_kW(leaf_example), \
        "Segment nearest the energy centre should carry at least as much peak as a leaf segment"

    for name in BUILDING_NAME_MAP.values():
        assert name in peak_by_building, f"Building '{name}' should have real demand data available"

    # New per-segment sizing assertions
    assert len(sized) == len(ealing_real.nodes) - 1, \
        "size_all_segments() should size every node except the root"
    assert all(isinstance(s.pipe, PipeSpec) for s in sized.values()), \
        "Every sized segment should hold a real PipeSpec"
    assert dn_map["N1"] >= dn_map["N15"], \
        "DN should taper down (or stay equal) moving from the energy centre toward a leaf, never increase"
    assert real_topology_capex > 0, "Real per-segment CAPEX should be positive"
    assert single_trunk_capex > real_topology_capex, \
        "Single-trunk approach should OVERSTATE cost vs real per-segment sizing " \
        "(every branch gets oversized to the full network peak instead of its own actual peak)"
    # Total length sized should match the topology's own total (every segment accounted for)
    total_sized_length = sum(s.length_m for s in sized.values())
    assert abs(total_sized_length - ealing_real.total_length_m()) < 0.01, \
        "Sum of sized segment lengths should exactly match the topology's total_length_m()"

    # New delivered-temperature and heat-loss assertions
    assert compliance["all_compliant"], \
        f"At 70°C source flow, the real Ealing network should comfortably clear " \
        f"the {MIN_DELIVERED_TEMP_C}°C minimum -- got worst case " \
        f"{compliance['worst_case_delivered_temp_C']}°C"
    for building, r in compliance["by_building"].items():
        assert r["delivered_temp_C"] < 70.0, \
            f"{building}: delivered temperature should be LOWER than the source flow " \
            f"temp (70°C) -- heat loss can only cool the water down, never heat it up"
        assert r["delivered_temp_C"] > DEFAULT_GROUND_TEMP_C, \
            f"{building}: delivered temperature should stay well above ground temperature " \
            f"at this scale of network (sanity bound, not a tight check)"
    # A building further down a longer path should generally show MORE
    # cumulative loss than one close to the energy centre -- check this
    # holds for the clearest case (Dickens Yard, the longest path, vs
    # Perceval House, a short path)
    assert (compliance["by_building"]["Dickens Yard Ph1"]["delivered_temp_C"]
            < compliance["by_building"]["Perceval House"]["delivered_temp_C"]), \
        "The building at the end of the longest path should show more cumulative heat " \
        "loss (lower delivered temp) than one on a short path near the energy centre"

    assert loss_result["total_kW"] > 0, "Network heat loss should be positive (real, not zero)"
    assert loss_result["total_kW"] < ealing_real.total_peak_kW(), \
        "Network heat loss should be a real but modest addition on top of building demand, " \
        "not larger than the demand itself (would indicate a sizing/physics error)"
    assert set(loss_result["by_segment_kW"].keys()) == set(sized.keys()), \
        "network_heat_loss_kW() should cover exactly the same segments as size_all_segments()"

    # New seasonal ground temp + hourly heat loss assertions
    assert abs(ground_hourly.mean() - GROUND_TEMP_MEAN_C) < 0.01, \
        "Seasonal ground temp's annual mean should exactly equal the cited GROUND_TEMP_MEAN_C constant"
    assert ground_hourly[:744].mean() < ground_hourly.mean(), \
        "January ground temp should be BELOW the annual mean (winter trough)"
    assert ground_hourly[4344:5088].mean() > ground_hourly.mean(), \
        "July ground temp should be ABOVE the annual mean (summer peak)"
    assert ground_hourly.min() > 0, \
        "Ground temp should stay above freezing at this depth/latitude for a sane UK climate (sanity bound)"
    assert len(hourly_loss["total_kW_hourly"]) == 8760, \
        "Hourly heat loss array should have exactly 8760 entries"
    assert hourly_loss["total_kW_hourly"][:744].mean() > hourly_loss["total_kW_hourly"][4344:5088].mean(), \
        "Mean hourly heat loss should be HIGHER in January than July -- colder ground, more loss"
    assert all(v >= 0 for v in hourly_loss["total_kW_hourly"]), \
        "Heat loss should never be negative at any hour (source flow is always hotter than ground at this scale)"
    assert set(hourly_loss["by_segment_kW_hourly"].keys()) == set(sized.keys()), \
        "network_heat_loss_kW_hourly() should cover exactly the same segments as size_all_segments()"
    # Cross-check: hourly method's annual total should be in the same
    # ballpark as the old fixed-ground-temp single-point calc (not
    # identical, since the real annual mean ground temp differs from
    # the old 10°C placeholder -- see the design discussion captured in
    # network_heat_loss_kW_hourly()'s docstring)
    old_style_annual_MWh = loss_result["total_kW"] * 8760 / 1000.0
    assert abs(hourly_loss["annual_total_MWh"] - old_style_annual_MWh) / old_style_annual_MWh < 0.10, \
        "Hourly seasonal-ground-temp annual total should be reasonably close to (within 10% of) " \
        "the old fixed-design-point estimate -- a big divergence would suggest a units/logic error, " \
        "not just the expected small difference from a more realistic ground temp mean"

    # The critical real-world finding: lower (weather-compensated-style)
    # flow temperature should show WORSE (lower) delivered temperatures
    # than the full design flow temperature -- confirms the physics
    # responds in the right direction, and that this is a genuine,
    # demonstrable trade-off worth flagging, not a hypothetical one
    assert not compliance_50["all_compliant"], \
        "At a meaningfully lower flow temperature (50°C), this network's real route " \
        "lengths should breach the 60°C minimum -- if this assertion ever starts " \
        "failing, it likely means the network got shorter/the physics changed, not " \
        "that the underlying tension stopped being real"
    for building in compliance["by_building"]:
        assert (compliance_50["by_building"][building]["delivered_temp_C"]
                < compliance["by_building"][building]["delivered_temp_C"]), \
            f"{building}: lower source flow temp should produce a lower delivered " \
            f"temp than the higher design flow temp, for the same building"

    # New minimum_safe_flow_temp_C() assertions
    assert 55.0 < min_safe_flow < 65.0, \
        f"Minimum safe flow temp ({min_safe_flow}°C) should be a modest amount above " \
        f"the 60°C target itself (a small transit-loss margin, not a huge one) for " \
        f"this network's real scale -- if this assertion fails, check whether the " \
        f"network's route lengths or pipe sizing changed significantly"
    assert compliance_at_floor["all_compliant"], \
        "At the solved minimum safe flow temp, the network should be exactly compliant"
    assert abs(compliance_at_floor["worst_case_delivered_temp_C"] - 60.0) < 0.2, \
        "At the solved floor, the worst-case building's delivered temp should sit " \
        "very close to the target minimum (near-zero margin by construction)"
    assert not compliance_below_floor["all_compliant"], \
        "Just 1°C below the solved floor should breach compliance -- confirms the " \
        "binary search converged on the real crossover point, not an overly safe one"

    print("  ✓ Generic tree correctly sums descendant peaks for each segment")
    print("  ✓ Leaf segment peak equals just its own node's peak (no descendants)")
    print("  ✓ Total length and total peak sum correctly across the whole tree")
    print("  ✓ path_to_root() returns the full ancestor chain in order")
    print("  ✓ Malformed trees (duplicate ID, missing parent, double root) all correctly rejected")
    print("  ✓ Ealing worked example validates as a proper tree (no cycles, fully reachable)")
    print("  ✓ Ealing worked example's total length matches the report's published 3,165m")
    print("  ✓ Segment nearest the energy centre carries the accumulated peak, exactly as a real")
    print("    branching network should -- this is what the old single-trunk model couldn't do")
    print("  ✓ Every non-root segment correctly sized with its own real PipeSpec")
    print("  ✓ DN correctly tapers down (never increases) moving away from the energy centre")
    print("  ✓ Single-trunk approach overstates CAPEX vs real per-segment sizing, as expected")
    print("  ✓ Sum of all sized segment lengths exactly matches the topology's total length")
    print("  ✓ Delivered temperature always cools down from source flow, never exceeds it")
    print("  ✓ A building on a longer path shows more cumulative heat loss than one on a short path")
    print("  ✓ Network heat loss is real and positive, but a modest fraction of building demand")
    print("  ✓ Lower (weather-compensated-style) flow temp correctly breaches the 60°C minimum --")
    print("    a genuine, demonstrated trade-off, not a hypothetical one")
    print("  ✓ minimum_safe_flow_temp_C() correctly solves for the real physical floor,")
    print("    confirmed exactly compliant at the floor and non-compliant 1°C below it")
    print("  ✓ Seasonal ground temperature model matches its cited real UK data exactly,")
    print("    with winter correctly colder and summer correctly warmer than the annual mean")
    print("  ✓ Hourly network heat loss correctly higher in winter than summer, never negative,")
    print("    and broadly consistent with the old fixed-ground-temp estimate")
    print()