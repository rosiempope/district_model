"""
topology_thermal.py
======================
Thermal physics for NetworkTopology: the Shukhov formula (per-segment
temperature drop/gain), the real seasonal UK ground temperature model,
delivered-temperature calculations, heating/cooling compliance checks,
and network-wide heat loss (annual and hourly). Split out of
network_topology.py — see that file's docstring and topology_tree.py's
docstring for the full rationale behind the three-way split.

This file defines:
  - seasonal_ground_temp_C(): real UK seasonal ground temperature curve
  - segment_outlet_temp_C(): the Shukhov formula
  - TopologyThermalMixin: delivered_temperature_C(),
    check_minimum_delivered_temperature(),
    check_maximum_delivered_temperature(), minimum_safe_flow_temp_C(),
    network_heat_loss_kW(), network_heat_loss_kW_hourly()

NetworkTopology (in network_topology.py) inherits from this mixin
alongside TopologyTreeMixin and TopologySizingMixin. This mixin assumes
`self.nodes`, `self.root_id`, `self.children_of()`,
`self.path_to_root()` (from TopologyTreeMixin), and `self.size_all_segments()`
(from TopologySizingMixin) exist on whatever class includes it.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipe_catalog import water_properties, heat_loss_coefficient_W_per_mK


# ── Constants ──────────────────────────────────────────────────────────────────

# Typical UK undisturbed ground temperature at pipe-laying depth (~1m,
# the standard burial depth for pre-insulated DH pipe) — kept as a
# simple scalar default for callers that don't need the seasonal detail
# below (e.g. single-point spot-check methods like delivered_temperature_C()).
# Set to 11.5°C to MATCH GROUND_TEMP_MEAN_C below (the real Thames Valley/
# London annual average — see that constant's sourcing note), so the
# single-point default and the seasonal model's mean always agree rather
# than silently disagreeing by a degree and a half. Same value as
# network.py's DEFAULT_GROUND_TEMP_C, duplicated here (rather than
# imported) so this module stays independently usable without requiring
# network.py.
DEFAULT_GROUND_TEMP_C = 11.5

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

# NOTE on the COOLING-duty equivalent of MIN_DELIVERED_TEMP_C: unlike
# heating, there is no separate universal constant here. The maximum
# acceptable delivered chilled water temperature IS the chiller's own
# design supply temperature (e.g. 7°C, the standard BS EN 14511 rating
# condition already used elsewhere in this project) — that's the
# temperature the downstream fan coil units / cooling coils were
# actually SIZED against (see check_maximum_delivered_temperature()'s
# docstring for the real engineering basis: fan coil unit heat/mass
# transfer margins are calculated assuming chilled water arrives AT
# the design supply temperature, not some tolerance band above it).
# Any warming in transit beyond that design value is a real capacity
# shortfall, not something to budget a separate allowance for — so
# check_maximum_delivered_temperature() takes the design chilled water
# temperature itself as its threshold parameter, with no separate
# named constant analogous to MIN_DELIVERED_TEMP_C above.


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
    """Thin wrapper so this module's only hard import is pipe_catalog.py
    (numpy is also imported, for the vectorised hourly methods below,
    but this scalar helper avoids depending on it for one exponential)."""
    import math
    return math.exp(x)


# ── Thermal physics mixin ─────────────────────────────────────────────────────────

class TopologyThermalMixin:
    """
    Delivered temperature, compliance checks, and network heat loss —
    see module docstring for what's assumed to already exist on the
    including class (self.nodes, self.root_id, self.children_of(),
    self.path_to_root(), self.size_all_segments()).
    """

    def delivered_temperature_C(
        self,
        node_id: str,
        sized_segments: dict,
        source_flow_temp_C: float,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    ) -> float:
        """
        The actual temperature ARRIVING at node_id, after real heat loss
        (or, for a cooling duty, heat GAIN — the Shukhov formula handles
        both correctly by sign, see segment_outlet_temp_C()) across every
        segment on the path from the energy centre to it — NOT the
        source's design flow temperature, which is what network.py's
        single-trunk model implicitly assumed customers always received.

        Walks path_to_root(node_id) from the ENERGY CENTRE END inward
        (i.e. root-to-node order, since each segment's outlet becomes
        the next segment's inlet), applying segment_outlet_temp_C()
        (the Shukhov formula) once per segment using that segment's own
        sized pipe (loss coefficient) and own mass flow rate.

        Parameters
        ----------
        node_id             : the building/node to compute delivered
                  temperature for
        sized_segments       : the dict returned by size_all_segments(),
                  for ONE specific duty ("heat" or "cool") — must have
                  been built with the SAME duty and flow/return
                  temperatures as source_flow_temp_C implies, or the
                  mass flow rates won't be consistent with the inlet
                  temperature being propagated here. node_id must
                  actually carry nonzero peak for that SAME duty
                  somewhere on its path to the root — size_all_segments()
                  skips segments with zero peak for a given duty (e.g. a
                  heating-only branch has no entry when sized_segments
                  was built with duty="cool"), and calling this method
                  for such a node/duty combination will raise a clear
                  error rather than crash on a confusing KeyError.
        source_flow_temp_C   : temperature LEAVING the energy centre for
                  THIS duty (i.e. the first segment's inlet temperature)
                  — this is where weather-compensated flow temperature
                  plugs in: pass a different value per hour to see how
                  delivered temperature varies through the year, rather
                  than only checking the fixed design point
        ground_temp_C        : surrounding ground temperature (°C)

        Returns
        -------
        Temperature (°C) actually arriving at node_id.
        """
        path_from_ec = list(reversed(self.path_to_root(node_id)))  # [EC, ..., node_id]
        current_temp = source_flow_temp_C
        for seg_node_id in path_from_ec[1:]:   # skip EC itself -- no incoming segment
            if seg_node_id not in sized_segments:
                raise KeyError(
                    f"Segment '{seg_node_id}' (on the path to '{node_id}') has no "
                    f"entry in sized_segments — this means size_all_segments() found "
                    f"zero peak for this duty on that segment's subtree (see that "
                    f"method's docstring on skipping zero-peak segments). Check that "
                    f"node_id actually has nonzero peak_kW/peak_cool_kW for the SAME "
                    f"duty sized_segments was built with."
                )
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
        HEATING compliance check: does every building actually connected
        to this network's HEATING duty (every node with a building_name
        AND nonzero peak_kW) receive at least min_temp_C, after real
        route heat loss — not just "is the source's design flow
        temperature high enough", which says nothing about what arrives
        at the far end of a long branch.

        For the cooling-duty MIRROR of this check (does chilled water
        arrive cool ENOUGH, i.e. a MAXIMUM rather than a minimum), see
        check_maximum_delivered_temperature() below — kept as a SEPARATE
        method rather than a directional flag on this one, since
        "minimum" and "maximum" compliance are different enough
        concepts (different real-world consequence — Legionella risk
        for heating vs degraded cooling capacity for cooling) that
        conflating them into one method with a confusing parameter
        risks a caller silently checking the wrong direction.

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
            # Only buildings with actual HEATING demand -- a building
            # with cooling-only demand (peak_kW == 0) has no entry in a
            # heating-duty sized_segments and shouldn't be checked here
            if node.building_name is None or node_id == self.root_id or node.peak_kW <= 0:
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

    def check_maximum_delivered_temperature(
        self,
        sized_segments: dict,
        source_flow_temp_C: float,
        design_chilled_water_temp_C: float,
        ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    ) -> dict:
        """
        COOLING compliance check, the MIRROR of
        check_minimum_delivered_temperature(): does every building
        actually connected to this network's COOLING duty (every node
        with a building_name AND nonzero peak_cool_kW) receive chilled
        water at or BELOW the chiller's own design supply temperature,
        after real route heat GAIN (the Shukhov formula pulls a cold
        pipe's temperature UP toward the warmer surrounding ground —
        see segment_outlet_temp_C() — the cooling-duty equivalent of
        heating's heat LOSS, same formula, opposite-sign physical effect).

        UNLIKE heating's MIN_DELIVERED_TEMP_C, there is no separate
        universal constant for this threshold. The threshold IS the
        chiller's own design chilled water supply temperature (e.g.
        7°C, the standard BS EN 14511 rating condition) — that's the
        exact temperature downstream fan coil units / cooling coils
        were SIZED against (real engineering basis: a typical FCU
        design assumes a ~5°C heat-transfer + ~5°C mass-transfer margin
        AT a specific supply temperature — e.g. ScienceDirect's chilled-
        water-temperature-difference study gives 7°C supply as the
        ceiling for a 25°C/60%RH comfort cooling design). Any warming in
        transit ABOVE that design value is a real capacity shortfall,
        not something with its own separate tolerance band to budget —
        so this method takes design_chilled_water_temp_C as a REQUIRED
        parameter (no default), forcing the caller to state explicitly
        what the chiller was actually designed to deliver, rather than
        silently assuming some generic "safe" number.

        Real-world consequence if this fails: chilled water arriving
        too warm can't provide the cooling capacity a building's
        cooling coils were sized for, even though the chiller itself is
        producing water at the correct design temperature — a real,
        analogous problem to heating's Legionella risk, but a capacity/
        comfort failure rather than a health and safety one.

        Parameters
        ----------
        sized_segments                : from size_all_segments(duty="cool")
        source_flow_temp_C              : temperature LEAVING the chiller
                  (should normally equal design_chilled_water_temp_C,
                  unless deliberately testing an off-design scenario)
        design_chilled_water_temp_C      : the chiller's own design supply
                  temperature (°C) — e.g. AirCooledChiller's
                  chilled_water_temp_C attribute. REQUIRED, no default
                  (see note above on why).
        ground_temp_C                    : surrounding ground temperature (°C)

        Returns
        -------
        dict, keyed by building_name -> {
            "node_id", "delivered_temp_C", "compliant" (bool),
            "margin_C" (positive = BELOW the design temp with room to
            spare, negative = above it, i.e. too warm)
        }
        Plus a top-level "all_compliant" bool and "worst_case_building"
        (the building with the HIGHEST delivered temperature this time
        — the mirror of heating's "lowest" worst case, since for
        cooling, higher delivered temp is the bad direction).
        """
        results = {}
        for node_id, node in self.nodes.items():
            # Only buildings with actual COOLING demand
            if node.building_name is None or node_id == self.root_id or node.peak_cool_kW <= 0:
                continue
            delivered = self.delivered_temperature_C(
                node_id, sized_segments, source_flow_temp_C, ground_temp_C,
            )
            margin = design_chilled_water_temp_C - delivered   # NOTE: reversed vs the heating check
            results[node.building_name] = {
                "node_id": node_id,
                "delivered_temp_C": round(delivered, 2),
                "compliant": bool(margin >= 0),
                "margin_C": round(margin, 2),
            }

        all_compliant = all(r["compliant"] for r in results.values()) if results else True
        # Worst case for cooling is the HIGHEST delivered temp (too warm),
        # the mirror of heating's lowest-delivered-temp worst case
        worst_case = max(results, key=lambda k: results[k]["delivered_temp_C"]) if results else None

        return {
            "by_building": results,
            "all_compliant": all_compliant,
            "worst_case_building": worst_case,
            "worst_case_delivered_temp_C": results[worst_case]["delivered_temp_C"] if worst_case else None,
            "max_allowed_temp_C": design_chilled_water_temp_C,
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
        HEATING-specific: the REAL question behind "how do we ensure
        60°C is always met": not "is THIS flow temperature safe", but
        "what's the LOWEST flow temperature that's STILL safe for every
        building on THIS real network" — found by binary search rather
        than guessed at, using the network's own real route lengths and
        real pipe sizing.

        This method is intentionally HEATING-only (always uses
        duty="heat" internally) — there is no meaningful cooling mirror
        of "search for the lowest safe flow temp": a chiller's design
        supply temperature is normally FIXED at its rating point (e.g.
        7°C), not something operationally lowered for efficiency the
        way heating flow temp is raised/lowered with weather
        compensation. The cooling-side equivalent question — "does this
        network's real heat GAIN ever push delivered temperature above
        the chiller's fixed design value" — is answered directly by
        check_maximum_delivered_temperature() instead, given the
        chiller's actual (fixed) supply temperature; there's no
        "solve for the right temperature" search to run for cooling the
        way there is for heating's variable, compensable flow temp.

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
                flow_temp_C=flow_temp_C, return_temp_C=return_temp_C, duty="heat", **pipe_kwargs,
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
        Total heat ACTUALLY lost (or, for a cooling-duty sized_segments,
        GAINED — same Shukhov formula, opposite-sign physical effect,
        see segment_outlet_temp_C()) across the whole network, segment
        by segment, using each segment's REAL inlet temperature (i.e.
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

        Works for EITHER duty — pass a sized_segments dict built with
        duty="heat" or duty="cool" (see size_all_segments()). Segments
        with zero peak for that duty (skipped by size_all_segments(),
        and therefore absent from sized_segments) are correctly
        excluded from BOTH the loss calculation AND from propagating an
        inlet temperature onward to their own children — i.e. a whole
        heating-only sub-branch is properly skipped in its entirety
        when sized_segments was built with duty="cool", not just the
        first segment of it.

        Returns
        -------
        dict: by_segment (node_id -> kW transferred in that segment,
        ALWAYS A POSITIVE MAGNITUDE regardless of duty -- see the abs()
        note in this method's implementation for why; whether that
        magnitude represents a loss or a gain is a property of the
        DUTY, not this return value's sign) and total_kW (sum across
        the whole network, for whichever segments actually exist in
        sized_segments).
        """
        by_segment = {}
        # Process nodes in an order where every node's parent has
        # already been processed (BFS from the root), so each segment's
        # inlet temperature is always its parent's already-computed
        # outlet temperature, not the raw source temperature. Nodes
        # absent from sized_segments (zero peak for this duty) are
        # skipped, and their ENTIRE subtree is skipped too (queue.extend
        # is only reached for nodes that were actually processed) —
        # correct, since a node with no entry has no real pipe and
        # therefore no real inlet temperature to hand down to children.
        outlet_temp = {self.root_id: source_flow_temp_C}
        queue = list(self.children_of(self.root_id))
        while queue:
            node_id = queue.pop(0)
            if node_id not in sized_segments:
                continue   # zero peak for this duty -- no real pipe, skip (and don't extend queue with its children)
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
            # Heat TRANSFERRED in THIS segment (kW) = mass_flow * cp * |T_in - T_out|.
            # abs() is REQUIRED, not cosmetic: for heating, T_in > T_out (the pipe
            # cools toward ground), so (T_in - T_out) is naturally positive. For
            # cooling, T_in < T_out (the pipe WARMS toward the warmer ground -- see
            # segment_outlet_temp_C()'s docstring on this being the same Shukhov
            # formula, opposite-sign physical effect), so (T_in - T_out) would be
            # NEGATIVE without abs() -- this was a real bug, caught when cooling-
            # duty dispatch was first wired up and produced a negative "heat loss"
            # for a chiller network, which is physically nonsensical (this function
            # reports a magnitude of energy transferred; whether that's a loss or a
            # gain is a property of the DUTY, established elsewhere, e.g. by which
            # direction check_minimum/maximum_delivered_temperature() checks against
            # -- not something this aggregate total should encode via its sign).
            loss_kW = seg.mass_flow_kg_s * props["cp_J_kgK"] * abs(inlet_temp - out_temp) / 1000.0
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

        Works for EITHER duty — pass a sized_segments dict built with
        duty="heat" or duty="cool" (see size_all_segments()). For
        cooling, this returns hourly heat GAIN (chilled water warming
        in transit toward the warmer ground), not loss — same formula,
        opposite-sign physical effect, see network_heat_loss_kW()'s own
        note on this. Segments absent from sized_segments (zero peak
        for that duty) and their entire subtree are correctly excluded.

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
            "total_kW_hourly": (n_hours,) array, ALWAYS A POSITIVE
                MAGNITUDE of heat transferred each hour, regardless of
                duty (see network_heat_loss_kW()'s equivalent note),
            "by_segment_kW_hourly": {node_id: (n_hours,) array},
            "annual_total_MWh": total annual magnitude in MWh
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
            if node_id not in sized_segments:
                continue   # zero peak for this duty -- no real pipe, skip (and don't extend queue with its children)
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
            # Heat TRANSFERRED, not signed loss -- see network_heat_loss_kW()'s
            # equivalent comment for the full explanation of why abs() is
            # required here, not optional: without it, a cooling-duty network
            # (where the pipe WARMS toward the ground, T_in < T_out) would
            # report a physically nonsensical negative "heat loss" total.
            loss_kW_hourly = seg.mass_flow_kg_s * cp * np.abs(inlet_temp_hourly - out_temp_hourly) / 1000.0
            by_segment_hourly[node_id] = loss_kW_hourly
            queue.extend(self.children_of(node_id))

        total_kW_hourly = sum(by_segment_hourly.values())
        annual_total_MWh = float(total_kW_hourly.sum()) / 1000.0   # kWh -> MWh (each hour's kW = that hour's kWh)

        return {
            "total_kW_hourly": total_kW_hourly,
            "by_segment_kW_hourly": by_segment_hourly,
            "annual_total_MWh": annual_total_MWh,
        }
