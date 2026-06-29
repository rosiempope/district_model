"""
topology_sizing.py
======================
Per-segment pipe sizing for NetworkTopology — calls into pipe_catalog.py
to turn each segment's accumulated peak (from topology_tree.py's
segment_peak_kW()) into a real, properly-sized PipeSpec. Split out of
network_topology.py — see that file's docstring and topology_tree.py's
docstring for the full rationale behind the three-way split.

This file defines:
  - SegmentPipeResult: the sized pipe for one segment, plus the inputs
    that produced it (length, peak, mass flow)
  - TopologySizingMixin: size_all_segments(), total_capex_GBP(),
    dn_by_segment()

NetworkTopology (in network_topology.py) inherits from this mixin
alongside TopologyTreeMixin and TopologyThermalMixin. This mixin
assumes `self.nodes`, `self.root_id`, and `self.segment_peak_kW()`
(from TopologyTreeMixin) exist on whatever class includes it.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# This file lives in network/, same as pipe_catalog.py — import it by
# bare module name (not "network.pipe_catalog") to sidestep the same
# package-name collision network.py itself works around (running a
# sibling file directly as a script makes Python treat network/ as the
# script's home directory, which collides with absolute-importing a
# "network" package). See network.py's own docstring for the full
# explanation.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipe_catalog import size_pipe_for_peak, PipeSpec, water_properties


# ── Sized-segment result ─────────────────────────────────────────────────────────

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


# ── Pipe sizing mixin ─────────────────────────────────────────────────────────────

class TopologySizingMixin:
    """
    Per-segment pipe sizing methods — see module docstring for what's
    assumed to already exist on the including class (self.nodes,
    self.root_id, self.segment_peak_kW()).
    """

    def size_all_segments(
        self,
        flow_temp_C: float,
        return_temp_C: float,
        duty: str = "heat",
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

        Heating and cooling are sized INDEPENDENTLY via separate calls
        (duty="heat" or duty="cool") — they're physically separate pipe
        pairs sharing only the route length, mirroring network.py's
        existing 4-pipe concept (size_4pipe_network()) extended to a
        real branching topology. Call this twice (once per duty) for a
        4-pipe network; once for a heating-only or cooling-only one.

        Parameters
        ----------
        flow_temp_C, return_temp_C : design supply/return temperatures
                  (°C) for THIS duty. For heating: e.g. 70/40°C (this
                  project's real Ealing design value). For cooling: e.g.
                  6/12°C (the standard BS EN 14511 rating condition,
                  already used elsewhere in this project — see
                  pipe_catalog.py's and network.py's cooling examples).
        duty : "heat" (default) or "cool" — selects which accumulated
                  peak (segment_peak_kW(..., duty=duty)) drives sizing.
        **pipe_kwargs : passed through to size_pipe_for_peak() for
                  EVERY segment (construction, insulation_series,
                  max_velocity_ms, etc.) — i.e. currently one shared
                  pipe spec choice across the whole network for THIS
                  duty. A future refinement could vary construction
                  (e.g. twin pipe on smaller branches, single on the
                  trunk) per segment; not built here since nothing in
                  this project currently needs that distinction made
                  automatically.

        Returns
        -------
        dict: node_id -> SegmentPipeResult, for every non-root node
              that has nonzero peak for this duty somewhere on its
              subtree (a node with segment_peak_kW(duty=duty) == 0 is
              skipped — there's no real pipe to size for a duty nothing
              downstream of it ever uses, e.g. a heating-only branch
              when duty="cool").
        """
        if duty not in ("heat", "cool"):
            raise ValueError(f"duty must be 'heat' or 'cool'; got '{duty}'.")

        results = {}
        delta_T_K = abs(flow_temp_C - return_temp_C)
        for node_id, node in self.nodes.items():
            if node_id == self.root_id:
                continue   # root has no incoming segment to size
            peak_kW = self.segment_peak_kW(node_id, duty=duty)
            if peak_kW <= 0:
                # No demand for this duty anywhere on this segment's
                # subtree -- skip rather than sizing a real pipe for
                # zero flow (size_pipe_for_peak() would either error or
                # return a nonsensical minimum-DN pipe for genuinely
                # zero peak, neither of which represents anything real)
                continue
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
