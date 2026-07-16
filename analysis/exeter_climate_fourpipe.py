"""Does four-pipe (heating+cooling) at Sowton/Airport look better under a
warmer future climate? Heating demand falls and cooling demand rises under
climate change — that could narrow or reverse the "cooling makes NPV worse"
finding from the baseline-climate four-pipe test.

Tests THREE climate scenarios (baseline / 2050_central / 2050_high — see
profiles/climate_scenarios.py for the real UKCP18-based deltas), and for
each one, sizes the plant FRESH against that climate's own weather (a
genuinely redesigned system for that future, not baseline plant just
operated under different weather) — since the real question is "would a
NEW four-pipe scheme make sense if designed for a higher climate
prediction", not "does old infrastructure survive future weather".

Run from the repository root:
    python -m analysis.exeter_climate_fourpipe
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exeter_case_study"
OUT.mkdir(parents=True, exist_ok=True)

from profiles.demand_synthesis import synthesise_network, compute_climate_reference
from profiles.climate_scenarios import apply_climate_scenario
from optimisation.auto_size import recommend_sizing
from scenarios.scenario_runner import run_scenario
from analysis.exeter_case_study import (
    SOWTON_BUILDINGS, SOWTON_SEGMENTS, _map_sources, weather as raw_weather, scaled_economics,
)

TECH_TYPES = ["efw_chp", "ashp", "gas_boiler"]
CLIMATE_SCENARIOS = ["baseline", "2050_central", "2050_high"]
# Real climate uplift requires a SHARED baseline reference passed into every
# synthesise_network() call — without it, each call self-references and the
# result is only RESHAPED for warmer weather, not actually scaled up (see
# analysis/exeter_simple_findings.py's module docstring for the bug this
# caused when it was missing here).
BASELINE_REF = compute_climate_reference(apply_climate_scenario(raw_weather, "baseline"))


def build_climate_scenario(climate_key, include_cooling):
    """Size and build a scenario for Sowton/Airport, FRESH against this
    climate scenario's own weather — plant capacities respond to the
    climate-shifted peak, not baseline peaks reused unchanged."""
    climate_weather = apply_climate_scenario(raw_weather, climate_key)
    demand = synthesise_network(climate_weather, {"demand_nodes": deepcopy(SOWTON_BUILDINGS)}, climate_reference=BASELINE_REF)
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=TECH_TYPES, weather_df=climate_weather, network_flow_temp_C=70.0,
        n_buildings=len(SOWTON_BUILDINGS), building_types=[b["type"] for b in SOWTON_BUILDINGS],
        include_cooling=include_cooling,
        cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
    )
    peak_total_MW = (demand["peak_heat_kW"] + (demand["peak_cool_kW"] if include_cooling else 0.0)) / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    if include_cooling:
        economics["counterfactual"] = "individual_gas_and_ac"
    scenario = {
        "name": f"Sowton/Airport {'4-pipe' if include_cooling else '2-pipe'} — {climate_key}",
        "climate_scenario": climate_key,
        "demand": {"buildings": deepcopy(SOWTON_BUILDINGS)},
        "network": {"mode": "tree", "segments": deepcopy(SOWTON_SEGMENTS), "include_cooling": include_cooling,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                    "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    if include_cooling:
        scenario["cooling_sources"] = _map_sources(rec["cooling_sources"])
    return scenario, demand, rec


rows = []
for climate_key in CLIMATE_SCENARIOS:
    for include_cooling in (False, True):
        scenario, demand, rec = build_climate_scenario(climate_key, include_cooling)
        try:
            result = run_scenario(scenario)
        except ValueError as exc:
            rows.append({
                "Climate": climate_key, "System": "4-pipe" if include_cooling else "2-pipe",
                "Annual heat demand (MWh)": None, "Annual cooling demand (MWh)": None,
                "Peak heat plant (MW)": None, "Peak cooling plant (MW)": None,
                "CAPEX (£m)": None, "Carbon gate": None, "Investor NPV (£m)": None,
                "Screening decision": f"EXCEEDS PIPE CATALOG: {exc}",
            })
            print(f"{climate_key:14} {'4-pipe' if include_cooling else '2-pipe':7} "
                  f"EXCEEDS PIPE CATALOG on the Airport branch — cooling peak too large for a single "
                  f"standard main under this climate; would need parallel cooling mains")
            continue
        h, inv = result["headline"], result["financial"]["investor"]
        cooling_peak_MW = sum(s["capacity_MW"] for s in rec.get("cooling_sources", []))
        heat_peak_MW = sum(s["capacity_MW"] for s in rec["sources"])
        rows.append({
            "Climate": climate_key,
            "System": "4-pipe" if include_cooling else "2-pipe",
            "Annual heat demand (MWh)": round(h["annual_heat_demand_MWh"], 0),
            "Annual cooling demand (MWh)": round(h["annual_cooling_demand_MWh"], 0),
            "Peak heat plant (MW)": round(heat_peak_MW, 2),
            "Peak cooling plant (MW)": round(cooling_peak_MW, 2),
            "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
            "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
            "Screening decision": result["screening"]["status"],
        })
        print(f"{climate_key:14} {'4-pipe' if include_cooling else '2-pipe':7} "
              f"heat={h['annual_heat_demand_MWh']:9.0f} MWh  cool={h['annual_cooling_demand_MWh']:8.0f} MWh  "
              f"NPV=£{inv['npv_GBP']/1e6:8.2f}m  {result['screening']['status']}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "climate_fourpipe_comparison.csv", index=False)

print("\n=== Full comparison ===")
print(df.to_string(index=False))

print("\n=== Does cooling help more as climate warms? (4-pipe NPV minus 2-pipe NPV, same climate) ===")
for climate_key in CLIMATE_SCENARIOS:
    two = df[(df["Climate"] == climate_key) & (df["System"] == "2-pipe")]["Investor NPV (£m)"].iloc[0]
    four_rows = df[(df["Climate"] == climate_key) & (df["System"] == "4-pipe")]
    four = four_rows["Investor NPV (£m)"].iloc[0]
    if pd.isna(two) or pd.isna(four):
        print(f"  {climate_key:14}  one or both cases exceeded the pipe catalog — see Screening decision column")
        continue
    print(f"  {climate_key:14}  2-pipe NPV=£{two:8.2f}m  4-pipe NPV=£{four:8.2f}m  "
          f"delta=£{four-two:7.2f}m  {'cooling helps more' if (four-two) > -12.80 else 'no material change / still worse'}")
