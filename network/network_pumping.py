"""
network_pumping.py
======================
Real pumping power for a district network — a genuine, previously-
unmodelled piece of physics. Until now, this project tracked pressure
drop per pipe segment (pipe_catalog.py's pressure_gradient_Pa_per_m)
but never converted that into an actual pump electrical load or cost —
a real gap a project review correctly identified.

Methodology
--------------
Standard hydraulic power formula (see e.g. EngineeringToolbox's Pump
Power Calculator, and Wikipedia's "Specific Pump Power" article, both
citing the same standard relation):

    P_hydraulic = Q * dP            (Q = volumetric flow m3/s, dP = total head loss Pa)
    P_electrical = P_hydraulic / eta_pump

This module computes dP as the CRITICAL PATH pressure drop — the
accumulated pressure drop along the route from the energy centre to
the building with the WORST (highest) pressure drop, doubled for the
round trip (supply AND return legs both have to be pushed by the same
pump) — NOT the sum of every segment's pressure drop across the WHOLE
network. Most branches are in PARALLEL, not in series with each other;
summing every segment regardless of branching would overstate the
real required pump head by a large margin. The pump only has to
overcome whichever SINGLE path has the most resistance; every other,
shorter/lower-resistance branch is automatically satisfied once the
critical path's head is met (real network hydraulics: parallel branches
share the same available head budget, they don't each need their own
full head independently).

Pump efficiency
------------------
DEFAULT_PUMP_EFFICIENCY = 0.75 — real sourcing: large centrifugal pumps
typically achieve 70-93% hydraulic efficiency (Pumps & Systems
industry trade publication, "How to Define & Measure Centrifugal Pump
Efficiency"), combined with a real motor efficiency factor for the
genuine "wire-to-water" figure that determines actual electricity
drawn (the same source explicitly warns against averaging pump and
motor efficiency rather than multiplying them) — 0.75 is a reasonable,
real, citable midpoint for a large district heating circulator
specifically (consistent with the "70% for large pumps" figure
combined with a high-efficiency modern motor), not an invented round
number.

This is deliberately a SINGLE flat efficiency for the whole pumping
system, not a detailed variable-speed-drive part-load efficiency curve
— a real, flagged simplification appropriate for a feasibility-stage
screening tool (see this project's broader simplification policy).

Usage
-----
    from network.network_pumping import critical_path_pressure_drop_Pa, pumping_power_MW

    dP_Pa = critical_path_pressure_drop_Pa(topology, sized_segments)
    pump_MW = pumping_power_MW(volumetric_flow_m3_s, dP_Pa)
"""

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Real sourcing — see module docstring for the full citation.
DEFAULT_PUMP_EFFICIENCY = 0.75


def critical_path_pressure_drop_Pa(topology, sized_segments: dict) -> dict:
    """
    Find the WORST-CASE (highest accumulated pressure drop) path from
    the energy centre to any connected building, and return its total
    round-trip pressure drop — the real figure that sizes the pump,
    NOT the sum of every segment in the whole network (see module
    docstring for why that would be wrong: most branches are in
    parallel, not in series).

    Parameters
    ----------
    topology         : a NetworkTopology instance
    sized_segments     : the dict from topology.size_all_segments(duty=...)
                — must be for ONE specific duty; segments absent from
                this dict (zero peak for that duty) are correctly
                skipped when walking each building's path

    Returns
    -------
    dict: {
        "worst_case_building_node_id": the node_id with the highest
            accumulated one-way pressure drop,
        "one_way_pressure_drop_Pa": that path's one-way (supply-only)
            accumulated pressure drop,
        "round_trip_pressure_drop_Pa": doubled, for the real pump duty
            (the pump has to push water out AND back),
    }
    """
    worst_drop_Pa = 0.0
    worst_node_id = None

    for node_id, node in topology.nodes.items():
        if node.building_name is None or node_id == topology.root_id:
            continue
        # Only buildings with a real path entirely covered by sized_segments
        # for this duty (mirrors the same skip-logic used elsewhere for
        # zero-peak segments, e.g. check_minimum_delivered_temperature())
        path = topology.path_to_root(node_id)[:-1]   # exclude the root itself
        if not all(p in sized_segments for p in path):
            continue

        accumulated_Pa = sum(
            sized_segments[p].pipe.pressure_gradient_Pa_per_m * sized_segments[p].length_m
            for p in path
        )
        if accumulated_Pa > worst_drop_Pa:
            worst_drop_Pa = accumulated_Pa
            worst_node_id = node_id

    return {
        "worst_case_building_node_id": worst_node_id,
        "one_way_pressure_drop_Pa": worst_drop_Pa,
        "round_trip_pressure_drop_Pa": worst_drop_Pa * 2.0,
    }


def pumping_power_MW(
    volumetric_flow_m3_s: float,
    pressure_drop_Pa: float,
    pump_efficiency: float = DEFAULT_PUMP_EFFICIENCY,
) -> float:
    """
    Real pump electrical power (MW) — see module docstring for the
    standard hydraulic power formula and the real efficiency sourcing.

        P_hydraulic_W = Q_m3_s * dP_Pa
        P_electrical_W = P_hydraulic_W / pump_efficiency

    Parameters
    ----------
    volumetric_flow_m3_s : the flow rate the pump must drive (m3/s) —
                  typically the energy centre's own total mass flow,
                  converted to volumetric flow at the source temperature
    pressure_drop_Pa      : total pressure drop the pump must overcome
                  (Pa) — use critical_path_pressure_drop_Pa()'s
                  "round_trip_pressure_drop_Pa" for the real network figure
    pump_efficiency        : wire-to-water efficiency (0-1) — default
                  0.75, see module docstring

    Returns
    -------
    Pump electrical power (MW).
    """
    P_hydraulic_W = volumetric_flow_m3_s * pressure_drop_Pa
    P_electrical_W = P_hydraulic_W / pump_efficiency
    return P_electrical_W / 1e6


def annual_pumping_electricity_MWh(
    topology,
    sized_segments: dict,
    mass_flow_kg_s_hourly: np.ndarray,
    density_kg_m3: float = 977.7,
    pump_efficiency: float = DEFAULT_PUMP_EFFICIENCY,
) -> dict:
    """
    Full-year hourly pumping electricity, using the SAME fixed critical-
    path pressure drop every hour (the network's own hydraulic
    resistance doesn't change hour to hour — pipe sizes/route are
    fixed) but the REAL hourly mass flow (which DOES vary with demand,
    since less heat demanded means less flow needed) — i.e. pumping
    power scales with actual hourly flow, not a single annual-average
    figure.

    Parameters
    ----------
    topology                  : a NetworkTopology instance
    sized_segments              : from topology.size_all_segments(duty=...)
    mass_flow_kg_s_hourly         : (8760,) array — the energy centre's
                  own total hourly mass flow (kg/s). For a fixed flow
                  temp/return temp design (this project's standard
                  simplification), this scales directly with hourly
                  heat demand: m_dot = Q_kW*1000 / (cp * delta_T_K).
    density_kg_m3                 : water density at the relevant
                  temperature (default: ~70°C water, matching this
                  project's real Ealing design flow temp)
    pump_efficiency                : see pumping_power_MW()

    Returns
    -------
    dict: {
        "hourly_pumping_MW": (8760,) array,
        "annual_pumping_MWh": total annual pumping electricity (MWh),
        "critical_path_info": the dict from critical_path_pressure_drop_Pa()
    }
    """
    critical_path = critical_path_pressure_drop_Pa(topology, sized_segments)
    dP_Pa = critical_path["round_trip_pressure_drop_Pa"]

    volumetric_flow_hourly_m3_s = mass_flow_kg_s_hourly / density_kg_m3
    P_hydraulic_W_hourly = volumetric_flow_hourly_m3_s * dP_Pa
    P_electrical_MW_hourly = (P_hydraulic_W_hourly / pump_efficiency) / 1e6

    return {
        "hourly_pumping_MW": P_electrical_MW_hourly,
        "annual_pumping_MWh": float(P_electrical_MW_hourly.sum()),
        "critical_path_info": critical_path,
    }
