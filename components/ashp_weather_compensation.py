"""
ashp_weather_compensation.py
======================
Weather-compensated flow temperature for ASHP — a real, tested, but
CURRENTLY DORMANT feature in this project's live feasibility pipeline.
Split out of ASHP.py (which had grown to ~1,280 lines, with this
feature alone accounting for ~200 of them, including a cross-module
dependency on network_topology.py that sat slightly awkwardly inside a
"component" file) as part of a project-wide restructuring — see
ASHP.py's own module docstring for the full file-split rationale.

STATUS: built and tested, but NOT currently active in this project's
live feasibility pipeline (ASHPArray's enable_weather_compensation
defaults to False everywhere, and nothing in dispatch.py/
network_topology.py's real integration calls it). This was a deliberate
project decision: weather compensation is an OPERATIONAL EFFICIENCY
lever, and stacking it on top of the topology/carbon/heat-loss
feasibility work was judged to add a variable that obscured the core
economic question, which should be assessed conservatively (one fixed
design flow temperature, the real cited Ealing value, all year) rather
than with an optimistic average. See network_topology.py's
network_heat_loss_kW_hourly() docstring for the same decision stated
from the network-physics side. This module is kept (not deleted) since
it's real, tested, and may be useful for a genuinely separate
operational-efficiency study later — but it is NOT the assumption this
project's actual dispatch/topology results are currently built on.

Real UK heat networks vary flow temperature with outdoor temperature
rather than holding one fixed peak value year-round -- see e.g. the
Ealing report's own note that "the real network is actually variable,
65-70C seasonally". Lower flow temp on mild days means: less heat lost
from the pipes (loss is driven by pipe-to-ground delta T), and a
smaller COP penalty for the ASHP (smaller lift from ambient air to flow
temp).

Standard convention (linear two-point heating curve), confirmed across
the DH/heat-pump literature -- e.g. nPro's heating-curve documentation
(citing Ruhnau, Hirth & Praktiknjo 2019, the SAME paper ASHP.py's COP
regression is already built on) and a 2025 ScienceDirect review of
heating-curve optimisation, both describing exactly this two-anchor-
point linear form: a high flow temp at the coldest design outdoor
temperature, falling linearly to a low "foot-point" flow temp at a mild
outdoor temperature where heating need is small.

Return temperature: held CONSTANT across the same range, rather than
sliding independently -- this matches real published weather-
compensation schedules (e.g. 70/40C in winter design conditions down to
55/40C in mild end-of-season conditions: flow drops, return stays put).
Return temperature is set by how much heat the building actually
extracts before sending water back (radiator/HIU return
characteristics), which doesn't move as freely as the source's flow
setpoint -- see CIBSE Journal's "The perfect return" for why chasing a
lower return temp is a SEPARATE design problem (flushing bypasses, HIU
sizing, control strategy) from simply commanding a lower flow temp.

COLD END kept at 70°C -- the SAME real, cited Ealing design value used
everywhere else in this project (ASHP_PRESETS, dispatch.py's
network_flow_temp_C, network_topology.py's self-test, pipe_catalog.py's
examples). An earlier version of this module raised this to 80°C to
make room for a compensation BAND down to a 70°C floor -- that's been
reverted: since compensation is currently inactive (see STATUS note
above), there's no live reason for this constant to disagree with the
project's single converged-on design temperature.

MILD-END FLOOR also kept at 70°C, for the same reason -- this makes the
default curve FLAT (cold end = mild end), which is intentional while
the feature is dormant: enabling it with NO other changes should not
silently produce a different flow temperature than the rest of the
project assumes. A real compensation band requires deliberately
choosing different cold/mild values via ASHPArray's constructor
(compensation_mild_temp_C override) -- see
check_compensation_floor_against_network() below for how to verify any
chosen floor is actually safe for a specific real network before using
it, and tests/test_ashp.py for a worked example using
compensation_mild_temp_C=62.0 (verified safe for the real Ealing
network without changing the 70°C design value at all).

Usage
-----
    from components.ashp_weather_compensation import (
        weather_compensated_flow_temp_C, check_compensation_floor_against_network,
    )

    # Direct curve evaluation
    flow_temps = weather_compensated_flow_temp_C(ambient_temp_array)

    # Cross-check a candidate floor against a real network topology
    result = check_compensation_floor_against_network(my_topology)
    print(result["recommendation"])

ASHPArray (in ASHP.py) imports weather_compensated_flow_temp_C from
this module to actually compute its hourly flow_temp_C array when
enable_weather_compensation=True — see that file for the integration.
"""

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

COMPENSATION_FLOW_TEMP_AT_COLD_C = 70.0    # = the project's single real design value (Ealing report)
COMPENSATION_FLOW_TEMP_AT_MILD_C = 70.0    # = same value -- flat by default; see module docstring
COMPENSATION_COLD_ANCHOR_AMBIENT_C = -10.0  # outdoor temp at which flow = the cold value
COMPENSATION_MILD_ANCHOR_AMBIENT_C = 15.0   # outdoor temp at/above which flow = the mild value
COMPENSATION_RETURN_TEMP_C = 40.0           # held constant across the whole range


# ── Weather compensation curve ──────────────────────────────────────────────────

def weather_compensated_flow_temp_C(
    T_ambient_C: np.ndarray,
    cold_anchor_ambient_C: float = COMPENSATION_COLD_ANCHOR_AMBIENT_C,
    mild_anchor_ambient_C: float = COMPENSATION_MILD_ANCHOR_AMBIENT_C,
    flow_temp_at_cold_C: float = COMPENSATION_FLOW_TEMP_AT_COLD_C,
    flow_temp_at_mild_C: float = COMPENSATION_FLOW_TEMP_AT_MILD_C,
) -> np.ndarray:
    """
    Hourly weather-compensated flow temperature (°C) — linear between two
    anchor points, the standard heating-curve convention (see module
    note above).

    At or below cold_anchor_ambient_C: flow_temp_at_cold_C (the peak
    design value, used for pipe/plant sizing — unchanged from before).
    At or above mild_anchor_ambient_C: flow_temp_at_mild_C (the floor
    value — heating is barely needed, but the network still needs SOME
    minimum useful temperature for the residual demand, e.g. DHW reheat).
    Between them: straight-line interpolation.

    Parameters
    ----------
    T_ambient_C   : hourly outdoor air temperature array (°C)
    cold_anchor_ambient_C, mild_anchor_ambient_C : the two outdoor-temp
                  anchor points (°C)
    flow_temp_at_cold_C, flow_temp_at_mild_C : the flow temps AT those
                  two anchor points (°C)

    Returns
    -------
    np.ndarray, same length as T_ambient_C, of hourly flow temperatures.
    """
    T = np.asarray(T_ambient_C, dtype=float)
    # np.interp expects the x-coordinates in increasing order; ambient
    # temp increases from cold to mild, but flow temp DECREASES over
    # that same range, so this is an inverse (downward-sloping) linear
    # interpolation -- np.interp handles that correctly as long as the
    # xp array (ambient anchors) is increasing, regardless of whether fp
    # (flow temps) is increasing or decreasing.
    return np.interp(
        T,
        [cold_anchor_ambient_C, mild_anchor_ambient_C],
        [flow_temp_at_cold_C, flow_temp_at_mild_C],
    )


# ── Cross-check against a real network topology ────────────────────────────────

def check_compensation_floor_against_network(
    network_topology,
    return_temp_C: float = COMPENSATION_RETURN_TEMP_C,
    proposed_mild_floor_C: float = COMPENSATION_FLOW_TEMP_AT_MILD_C,
    min_delivered_temp_C: float = 60.0,
) -> dict:
    """
    Verify that a PROPOSED compensation mild-end floor (default: this
    module's COMPENSATION_FLOW_TEMP_AT_MILD_C, 70°C) is actually safe for
    a SPECIFIC real network topology — i.e. closes the loop between this
    module's curve and network_topology.py's real route-length physics,
    rather than trusting a default that was only verified against one
    worked example (Ealing).

    This module's 70°C default was verified against the real Ealing
    worked example (see the constants block above) and found to leave a
    genuine 8.52°C margin there — but Ealing's specific route lengths and
    pipe sizing won't be true of every network. A longer or less-
    insulated network could need a HIGHER floor than 70°C to stay safe;
    a shorter or better-insulated one could safely go LOWER, leaving
    efficiency on the table if the 70°C default is used unquestioned.

    Parameters
    ----------
    network_topology       : a NetworkTopology instance (see
                  network.network_topology) — typically already populated
                  with real per-building peak demand. Duck-typed, not
                  imported directly — this module only ever calls
                  network_topology.minimum_safe_flow_temp_C(), so any
                  object with that method works, avoiding a hard import-
                  time dependency on network_topology.py.
    return_temp_C           : the compensation curve's return temperature
                  (°C) — held constant, see module note above
    proposed_mild_floor_C    : the mild-end floor to check (°C). Defaults
                  to this module's own constant, so calling this with no
                  override checks "is OUR default actually safe for this
                  network" — but any other candidate floor can be passed
                  to check alternatives.
    min_delivered_temp_C     : the regulatory/safety minimum (°C) — see
                  network_topology.py's MIN_DELIVERED_TEMP_C for the
                  Legionella-control basis of the standard 60°C value

    Returns
    -------
    dict: {
        "proposed_floor_safe": bool,
        "proposed_floor_C": the floor that was checked,
        "actual_minimum_safe_flow_temp_C": the network's own calculated
            physical floor (from network_topology's own solver),
        "margin_C": proposed_floor_C - actual_minimum_safe_flow_temp_C
            (positive = the proposed floor has real headroom; negative
            = the proposed floor is UNSAFE for this specific network),
        "recommendation": a plain-English verdict
    }
    """
    actual_floor = network_topology.minimum_safe_flow_temp_C(
        return_temp_C=return_temp_C, min_temp_C=min_delivered_temp_C,
    )
    margin = proposed_mild_floor_C - actual_floor
    safe = margin >= 0

    if safe:
        recommendation = (
            f"Safe — {proposed_mild_floor_C}°C leaves a {margin:.2f}°C margin above "
            f"this network's real physical floor ({actual_floor:.2f}°C)."
        )
    else:
        recommendation = (
            f"NOT SAFE for this network — {proposed_mild_floor_C}°C is "
            f"{abs(margin):.2f}°C BELOW this network's real physical floor "
            f"({actual_floor:.2f}°C). Raise the mild-end floor to at least "
            f"{actual_floor:.2f}°C (plus a real margin) before using it for this network."
        )

    return {
        "proposed_floor_safe": bool(safe),
        "proposed_floor_C": proposed_mild_floor_C,
        "actual_minimum_safe_flow_temp_C": actual_floor,
        "margin_C": round(margin, 2),
        "recommendation": recommendation,
    }
