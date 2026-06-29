"""
topology_tree.py
======================
Pure tree-graph mechanics for NetworkTopology — no pipe physics, no
thermal calculations. Split out of network_topology.py (which had grown
to ~2000 lines covering three genuinely separate concerns: tree
mechanics, pipe sizing, and thermal physics) so each concern lives in
its own focused file.

This file defines:
  - TopologyNode: one node's data (length, peaks, building name)
  - TopologyTreeMixin: add_node(), children_of(), descendants_of(),
    segment_peak_kW(), total_length_m(), total_peak_kW(), leaf_nodes(),
    path_to_root(), validate(), summary()

NetworkTopology (in network_topology.py) inherits from this mixin
alongside TopologySizingMixin and TopologyThermalMixin — see that
file's docstring for the full assembly. This mixin assumes `self.nodes`
(dict), `self.name` (str), `self._root_id` (str or None), and
`self._children` (dict) exist on whatever class includes it — those
fields are defined on NetworkTopology itself (the only actual
@dataclass in this group), not duplicated here, to avoid the field-
ordering pitfalls of multiple dataclass inheritance in Python.

This file has NO dependency on pipe_catalog.py or any thermal/Shukhov
code — it only ever touches node_id, parent_id, length_m, peak_kW,
peak_cool_kW. That's deliberate: tree-walking logic should never need
to know what a pipe or a temperature is.
"""

from dataclasses import dataclass
from typing import Optional


# ── Core node data structure ─────────────────────────────────────────────────────

@dataclass
class TopologyNode:
    """
    One node in the tree. The pipe SEGMENT this node sits at the end of
    is the edge FROM parent_id TO this node — length_m and the resulting
    sized pipe(s) both describe THAT segment, not the node itself.

    A node can carry HEATING and/or COOLING demand simultaneously (e.g.
    a hospital with a heated ward and a cooled server room in the same
    building) — peak_kW and peak_cool_kW are independent, both default
    to 0.0 for a pure junction node with no building of its own. Heating
    and cooling pipes physically run the SAME route (same length_m) as
    separate pipe pairs within it — mirrors network.py's existing 4-pipe
    concept (size_4pipe_network()), just extended to a real branching
    topology instead of one representative trunk.

    node_id     : unique identifier within a NetworkTopology
    parent_id   : the node this one connects back toward (None ONLY for
                  the root / energy centre)
    length_m    : length of the pipe segment from parent to this node
                  (0.0 for the root, which has no incoming segment) —
                  ONE length serves both heating and cooling duties,
                  since they share the same physical route
    peak_kW     : THIS node's own peak HEATING demand — 0 for a pure
                  junction/branch-point with no heating load attached
    peak_cool_kW : THIS node's own peak COOLING demand — 0 for a node
                  with no cooling load attached. Independent of peak_kW.
    building_name : optional human-readable label (e.g. a real building
                  name) — purely cosmetic/reporting, the engine itself
                  never reads this field for any calculation
    """
    node_id: str
    parent_id: Optional[str]
    length_m: float
    peak_kW: float = 0.0
    peak_cool_kW: float = 0.0
    building_name: Optional[str] = None

    def __repr__(self):
        label = f" ({self.building_name})" if self.building_name else ""
        return (
            f"TopologyNode({self.node_id}{label}, parent={self.parent_id}, "
            f"length={self.length_m:.0f}m, peak={self.peak_kW:.0f}kW heat, "
            f"{self.peak_cool_kW:.0f}kW cool)"
        )


# ── Tree mechanics mixin ─────────────────────────────────────────────────────────

class TopologyTreeMixin:
    """
    Pure tree-graph methods — see module docstring for what's assumed
    to already exist on the including class (self.nodes, self.name,
    self._root_id, self._children).
    """

    def add_node(
        self,
        node_id: str,
        parent_id: Optional[str],
        length_m: float,
        peak_kW: float = 0.0,
        peak_cool_kW: float = 0.0,
        building_name: Optional[str] = None,
    ) -> None:
        """
        Add one node to the tree. parent_id must already exist in the
        tree (or be None, for the single root node).

        peak_kW and peak_cool_kW are independent — a node can carry
        heating demand, cooling demand, both, or neither (a pure
        junction). See TopologyNode's docstring for the full rationale.
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
            peak_kW=peak_kW, peak_cool_kW=peak_cool_kW, building_name=building_name,
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

    def segment_peak_kW(self, node_id: str, duty: str = "heat") -> float:
        """
        Peak flow (kW) the pipe segment FROM node_id's PARENT TO node_id
        must carry — this node's own peak plus every downstream node's
        peak, for the given duty (see module docstring for the
        diversified-peak caveat: this is a simple sum, not an hour-by-
        hour coincident peak, which is the standard conservative
        feasibility-stage simplification).

        Parameters
        ----------
        duty : "heat" (default, uses peak_kW) or "cool" (uses
               peak_cool_kW). Heating and cooling are independent sums
               over the SAME tree structure — a node with both heating
               and cooling demand contributes to both totals
               independently, since they're physically separate pipe
               pairs sharing only the route.
        """
        if duty not in ("heat", "cool"):
            raise ValueError(f"duty must be 'heat' or 'cool'; got '{duty}'.")
        attr = "peak_kW" if duty == "heat" else "peak_cool_kW"

        node = self.nodes[node_id]
        downstream_total = sum(getattr(self.nodes[d], attr) for d in self.descendants_of(node_id))
        return getattr(node, attr) + downstream_total

    def total_length_m(self) -> float:
        """Sum of every segment's length — the real total network route length."""
        return sum(n.length_m for n in self.nodes.values())

    def total_peak_kW(self, duty: str = "heat") -> float:
        """Sum of every node's OWN peak for the given duty — i.e. the
        energy centre's required peak output for that duty.
        duty: "heat" (default) or "cool" — see segment_peak_kW()."""
        if duty not in ("heat", "cool"):
            raise ValueError(f"duty must be 'heat' or 'cool'; got '{duty}'.")
        attr = "peak_kW" if duty == "heat" else "peak_cool_kW"
        return sum(getattr(n, attr) for n in self.nodes.values())

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
        """High-level stats — total length, total peak (both duties), node/leaf counts."""
        has_cooling = self.total_peak_kW(duty="cool") > 0
        result = {
            "name": self.name,
            "total_length_m": round(self.total_length_m(), 1),
            "total_peak_kW": round(self.total_peak_kW(duty="heat"), 1),
            "n_nodes": len(self.nodes),
            "n_leaf_nodes": len(self.leaf_nodes()),
            "max_segment_peak_kW": round(
                max(self.segment_peak_kW(nid, duty="heat") for nid in self.nodes if nid != self.root_id), 1
            ) if len(self.nodes) > 1 else 0.0,
        }
        if has_cooling:
            result.update({
                "total_peak_cool_kW": round(self.total_peak_kW(duty="cool"), 1),
                "max_segment_peak_cool_kW": round(
                    max(self.segment_peak_kW(nid, duty="cool") for nid in self.nodes if nid != self.root_id), 1
                ) if len(self.nodes) > 1 else 0.0,
            })
        return result
