"""Find the scale at which the Sowton/Airport (Exeter) four-pipe scheme
would reach positive investor NPV, holding heat and cooling tariffs at
STRICT parity throughout (heat = each building's own modelled individual-
gas-boiler bill; cooling = each building's own modelled individual-AC
bill, at the model's default commercial electricity tariff — see
economics/metrics.py::counterfactual_individual_ac_dispatch).

Direct follow-up to a real gap in the two prior analyses: the fixed CAPEX
items (energy-centre building, land, electricity/gas connection, controls)
and fixed OPEX overheads (billing, insurance, land lease, water treatment,
operator overhead) were reused UNCHANGED from scenarios/worked_scenarios.py's
COMMON_ECONOMICS — a set of figures calibrated for one specific reference
scale (the Ealing-style BASE_BUILDINGS case: 564 connections, 8.58 MW peak
heat) — across every archetype and Exeter network regardless of that
network's own actual size. That was flagged as a caveat in both prior
readouts but not corrected. This script corrects it: fixed CAPEX/OPEX
items are scaled by this scenario's own peak thermal capacity relative to
that same reference, so a bigger scheme properly gets a bigger energy
centre/connection/overhead cost, not the same flat number a much smaller
scheme was charged.

Run from the repository root:
    python -m analysis.exeter_breakeven_scale
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exeter_case_study"
OUT.mkdir(parents=True, exist_ok=True)

from profiles.demand_synthesis import synthesise_network
from optimisation.auto_size import recommend_sizing
from scenarios.scenario_runner import run_scenario
from analysis.exeter_case_study import (
    SOWTON_BUILDINGS, SOWTON_SEGMENTS, _map_sources, weather, scaled_economics as _scaled_economics_base,
)


def scaled_economics(peak_total_MW: float) -> tuple[dict, float]:
    """Thin wrapper on analysis.exeter_case_study.scaled_economics() — this
    script always wants GHNF grant enabled and the combined gas+AC
    counterfactual, so apply those two on top of the shared base rather
    than duplicating the fixed-cost scaling logic itself here."""
    econ, scale = _scaled_economics_base(peak_total_MW)
    econ["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    econ["counterfactual"] = "individual_gas_and_ac"
    return econ, scale


def scale_buildings(buildings, multiplier):
    scaled = deepcopy(buildings)
    for b in scaled:
        if b.get("floor_area_m2"):
            b["floor_area_m2"] = b["floor_area_m2"] * multiplier
        if b.get("units"):
            b["units"] = int(round(b["units"] * multiplier))
            b["connections"] = b["units"]
    return scaled


def build_scaled_scenario(name, buildings, segments, tech_types=("efw_chp", "ashp", "gas_boiler")):
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=list(tech_types), weather_df=weather, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
        include_cooling=True, cooling_demand_kW=demand["total_cooling_kW"],
        peak_cooling_kW=demand["peak_cool_kW"],
    )
    peak_total_MW = (demand["peak_heat_kW"] + demand["peak_cool_kW"]) / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    scenario = {
        "name": name, "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "tree", "segments": deepcopy(segments), "include_cooling": True,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                    "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
        "sources": _map_sources(rec["sources"]),
        "cooling_sources": _map_sources(rec["cooling_sources"]),
        "economics": economics,
    }
    return scenario, scale_factor, demand


# ═══════════════════════════════════════════════════════════════════════════
# Sweep A: "balanced growth" — scale both buildings (commercial + new
# community) by the same multiplier, same footprint/route (densification,
# not sprawl — same 2,600 m / 3,200 m branch lengths throughout).
# ═══════════════════════════════════════════════════════════════════════════

MULTIPLIERS = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140]

rows_balanced = []
for m in MULTIPLIERS:
    buildings = scale_buildings(SOWTON_BUILDINGS, m)
    scenario, scale_factor, demand = build_scaled_scenario(
        f"Sowton/Airport x{m} (balanced)", buildings, SOWTON_SEGMENTS,
    )
    try:
        result = run_scenario(scenario)
    except ValueError as exc:
        # A real physical limit surfaced by the model's own pipe catalog —
        # a single branch segment can't carry this much peak load with any
        # standard pipe size (would need parallel mains, not modelled here).
        # Record and move on rather than crash the whole sweep.
        connections = sum(b.get("connections", 1) for b in buildings)
        rows_balanced.append({
            "Multiplier": m, "Connections": connections,
            "Annual heat+cool (GWh)": None, "Fixed-cost scale factor": round(scale_factor, 2),
            "CAPEX (£m)": None, "Carbon gate": None, "Investor NPV (£m)": None,
            "Investor IRR (%)": None, "Screening decision": f"EXCEEDS PIPE CATALOG: {exc}",
        })
        print(f"[balanced] x{m:>4}  conn={connections:>5}  EXCEEDS PIPE CATALOG (single branch too large "
              f"for any standard DN — would need parallel mains)")
        continue
    h, inv = result["headline"], result["financial"]["investor"]
    connections = sum(b.get("connections", 1) for b in buildings)
    rows_balanced.append({
        "Multiplier": m, "Connections": connections,
        "Annual heat+cool (GWh)": round((h["annual_heat_demand_MWh"] + h["annual_cooling_demand_MWh"]) / 1000, 1),
        "Fixed-cost scale factor": round(scale_factor, 2),
        "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
        "Investor IRR (%)": round(inv["irr"] * 100, 2) if inv["irr"] is not None else None,
        "Screening decision": result["screening"]["status"],
    })
    print(f"[balanced] x{m:>4}  conn={connections:>5}  NPV=£{inv['npv_GBP']/1e6:8.2f}m  "
          f"carbon={'PASS' if h['carbon_compliant'] else 'FAIL':4}  {result['screening']['status']}")

balanced_df = pd.DataFrame(rows_balanced)
balanced_df.to_csv(OUT / "breakeven_scale_balanced.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Sweep B: "cooling-heavy growth" — scale the commercial/office_ac building
# much harder than the residential one, to test whether the AC-parity
# counterfactual (individual AC CAPEX + commercial electricity, both
# genuinely expensive) gives cooling more margin than the gas-parity
# counterfactual (individual gas boiler + cheap retail gas) gives heating.
# ═══════════════════════════════════════════════════════════════════════════

rows_cooling_heavy = []
for m in MULTIPLIERS:
    buildings = deepcopy(SOWTON_BUILDINGS)
    for b in buildings:
        if b["type"] == "office_ac":
            b["floor_area_m2"] = b["floor_area_m2"] * m
        else:
            b["floor_area_m2"] = b.get("floor_area_m2", 0)
            if b.get("units"):
                b["units"] = int(round(b["units"] * max(1.0, m / 4)))  # community still grows, slower
                b["connections"] = b["units"]
    scenario, scale_factor, demand = build_scaled_scenario(
        f"Sowton/Airport x{m} (cooling-heavy)", buildings, SOWTON_SEGMENTS,
    )
    try:
        result = run_scenario(scenario)
    except ValueError as exc:
        connections = sum(b.get("connections", 1) for b in buildings)
        rows_cooling_heavy.append({
            "Multiplier": m, "Connections": connections,
            "Annual heat+cool (GWh)": None, "Annual cooling share (%)": None,
            "Fixed-cost scale factor": round(scale_factor, 2), "CAPEX (£m)": None,
            "Carbon gate": None, "Investor NPV (£m)": None, "Investor IRR (%)": None,
            "Screening decision": f"EXCEEDS PIPE CATALOG: {exc}",
        })
        print(f"[coolheavy] x{m:>4}  conn={connections:>5}  EXCEEDS PIPE CATALOG (single branch too large "
              f"for any standard DN — would need parallel mains)")
        continue
    h, inv = result["headline"], result["financial"]["investor"]
    connections = sum(b.get("connections", 1) for b in buildings)
    rows_cooling_heavy.append({
        "Multiplier": m, "Connections": connections,
        "Annual heat+cool (GWh)": round((h["annual_heat_demand_MWh"] + h["annual_cooling_demand_MWh"]) / 1000, 1),
        "Annual cooling share (%)": round(h["annual_cooling_demand_MWh"] / max(h["annual_heat_demand_MWh"] + h["annual_cooling_demand_MWh"], 1e-9) * 100, 1),
        "Fixed-cost scale factor": round(scale_factor, 2),
        "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
        "Investor IRR (%)": round(inv["irr"] * 100, 2) if inv["irr"] is not None else None,
        "Screening decision": result["screening"]["status"],
    })
    print(f"[coolheavy] x{m:>4}  conn={connections:>5}  cool%={rows_cooling_heavy[-1]['Annual cooling share (%)']:>5.1f}  "
          f"NPV=£{inv['npv_GBP']/1e6:8.2f}m  {result['screening']['status']}")

cooling_heavy_df = pd.DataFrame(rows_cooling_heavy)
cooling_heavy_df.to_csv(OUT / "breakeven_scale_cooling_heavy.csv", index=False)

print("\n=== Balanced growth sweep ===")
print(balanced_df.to_string(index=False))
print("\n=== Cooling-heavy growth sweep ===")
print(cooling_heavy_df.to_string(index=False))

first_positive_balanced = balanced_df[balanced_df["Investor NPV (£m)"] > 0]
first_positive_coolheavy = cooling_heavy_df[cooling_heavy_df["Investor NPV (£m)"] > 0]
print(f"\nBalanced sweep: first positive NPV at multiplier "
      f"{first_positive_balanced['Multiplier'].min() if len(first_positive_balanced) else 'NONE FOUND in range tested'}")
print(f"Cooling-heavy sweep: first positive NPV at multiplier "
      f"{first_positive_coolheavy['Multiplier'].min() if len(first_positive_coolheavy) else 'NONE FOUND in range tested'}")

# ═══════════════════════════════════════════════════════════════════════════
# Sweep C: "replicated dense neighbourhoods" — the balanced/cooling-heavy
# sweeps above concentrate ALL growth onto the SAME 1-2 branches, which
# (a) doesn't get real network-CAPEX economies of scale (pipe size keeps
# growing with demand on that one branch, not spreading fixed pipe cost
# over more customers on a shared trunk) and (b) hits a genuine pipe-
# catalog physical limit. This sweep tests a more realistic "how UK
# networks actually grow" shape instead: N replicas of the SHORT-BRANCH
# Central Exeter pattern (5 short branches, 400-1400 m each) radiating
# from ONE shared energy centre — many dense short branches, not one
# giant overloaded one.
# ═══════════════════════════════════════════════════════════════════════════
from analysis.exeter_case_study import CENTRAL_BUILDINGS, CENTRAL_SEGMENTS

def replicate_central(n_replicas):
    buildings, segments = [], []
    for i in range(n_replicas):
        for b in deepcopy(CENTRAL_BUILDINGS):
            b["name"] = f"{b['name']} ({i+1})"
            buildings.append(b)
        for seg in deepcopy(CENTRAL_SEGMENTS):
            seg["node_id"] = f"{seg['node_id']}_{i+1}"
            seg["parent_id"] = "EC" if seg["parent_id"] == "EC" else f"{seg['parent_id']}_{i+1}"
            seg["building"] = f"{seg['building']} ({i+1})"
            segments.append(seg)
    return buildings, segments

REPLICA_COUNTS = [1, 2, 4, 6, 8, 12, 16, 20]
rows_replicated = []
for n in REPLICA_COUNTS:
    buildings, segments = replicate_central(n)
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=["efw_chp", "ashp", "gas_boiler"], weather_df=weather,
        network_flow_temp_C=70.0, n_buildings=len(buildings),
        building_types=[b["type"] for b in buildings],
    )
    peak_total_MW = demand["peak_heat_kW"] / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    economics["counterfactual"] = "individual_gas"
    scenario = {
        "name": f"Replicated central x{n}", "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "tree", "segments": deepcopy(segments), "include_cooling": False,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    try:
        result = run_scenario(scenario)
    except ValueError as exc:
        rows_replicated.append({"Replicas": n, "Connections": sum(b.get("connections", 1) for b in buildings),
                                 "Investor NPV (£m)": None, "Screening decision": f"EXCEEDS PIPE CATALOG: {exc}"})
        print(f"[replicated] x{n:>3}  EXCEEDS PIPE CATALOG")
        continue
    h, inv = result["headline"], result["financial"]["investor"]
    connections = sum(b.get("connections", 1) for b in buildings)
    rows_replicated.append({
        "Replicas": n, "Connections": connections,
        "Total route (m)": h["network_total_length_m"],
        "Linear heat density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
        "Fixed-cost scale factor": round(scale_factor, 2),
        "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
        "Investor IRR (%)": round(inv["irr"] * 100, 2) if inv["irr"] is not None else None,
        "Screening decision": result["screening"]["status"],
    })
    print(f"[replicated] x{n:>3}  conn={connections:>5}  NPV=£{inv['npv_GBP']/1e6:8.2f}m  {result['screening']['status']}")

replicated_df = pd.DataFrame(rows_replicated)
replicated_df.to_csv(OUT / "breakeven_scale_replicated.csv", index=False)
print("\n=== Replicated dense-neighbourhood sweep (heating only, strict gas parity) ===")
print(replicated_df.to_string(index=False))
first_positive_rep = replicated_df[replicated_df["Investor NPV (£m)"].notna() & (replicated_df["Investor NPV (£m)"] > 0)]
print(f"\nReplicated sweep: first positive NPV at replica count "
      f"{first_positive_rep['Replicas'].min() if len(first_positive_rep) else 'NONE FOUND in range tested'}")
