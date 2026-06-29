"""
test_network_pumping.py
======================
Self-test / demonstration suite for network.network_pumping
(critical_path_pressure_drop_Pa, pumping_power_MW,
annual_pumping_electricity_MWh) — the real pumping power physics that
was completely unmodelled in this project until a project review
correctly flagged the gap.

Run directly: python3 tests/test_network_pumping.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from network.network_pumping import (
    DEFAULT_PUMP_EFFICIENCY, critical_path_pressure_drop_Pa,
    pumping_power_MW, annual_pumping_electricity_MWh,
)
from network.network_topology import ealing_town_centre_topology
from network.pipe_catalog import water_properties
from profiles.demand_synthesis import synthesise_network


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  network_pumping.py — self-test")
    print("=" * 70)

    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
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
    peak_by_building = {n["name"]: n["peak_heat_kW"] for n in demand_result["nodes"]}
    topo = ealing_town_centre_topology(peak_kW_by_building=peak_by_building)
    sized = topo.size_all_segments(flow_temp_C=70.0, return_temp_C=40.0)

    # --- Critical path pressure drop ---
    print("\n  Critical path pressure drop (the real figure that sizes the pump --")
    print("  NOT the sum of every segment, since most branches are parallel):")
    critical_path = critical_path_pressure_drop_Pa(topo, sized)
    print(f"    Worst-case building: {critical_path['worst_case_building_node_id']}")
    print(f"    One-way pressure drop: {critical_path['one_way_pressure_drop_Pa']/1000:.1f} kPa "
          f"({critical_path['one_way_pressure_drop_Pa']/1e5:.2f} bar)")
    print(f"    Round-trip pressure drop: {critical_path['round_trip_pressure_drop_Pa']/1000:.1f} kPa "
          f"({critical_path['round_trip_pressure_drop_Pa']/1e5:.2f} bar)")

    # --- Sanity check: every OTHER building's path drop should be <= the worst case ---
    print(f"\n  Confirming the worst case really is the worst (checking every building):")
    all_drops = {}
    for node_id, node in topo.nodes.items():
        if node.building_name is None or node_id == topo.root_id:
            continue
        path = topo.path_to_root(node_id)[:-1]
        if not all(p in sized for p in path):
            continue
        drop_Pa = sum(sized[p].pipe.pressure_gradient_Pa_per_m * sized[p].length_m for p in path)
        all_drops[node_id] = drop_Pa
        print(f"    {node_id} ({node.building_name}): {drop_Pa/1000:.1f} kPa one-way")

    # --- Pumping power at design peak ---
    print(f"\n  Pumping power at design peak flow:")
    peak_demand_kW = demand_result["total_heat_kW"].max()
    delta_T_K = 30.0
    cp = water_properties(70.0)["cp_J_kgK"]
    density = water_properties(70.0)["density_kg_m3"]
    peak_mass_flow_kg_s = (peak_demand_kW * 1000.0) / (cp * delta_T_K)
    peak_volumetric_flow_m3_s = peak_mass_flow_kg_s / density
    peak_pump_MW = pumping_power_MW(
        peak_volumetric_flow_m3_s, critical_path["round_trip_pressure_drop_Pa"],
    )
    print(f"    Peak heat demand: {peak_demand_kW/1000:.2f} MW")
    print(f"    Peak mass flow: {peak_mass_flow_kg_s:.1f} kg/s")
    print(f"    Peak pumping power: {peak_pump_MW*1000:.1f} kW "
          f"({peak_pump_MW/(peak_demand_kW/1000)*100:.2f}% of peak heat demand)")

    # --- Full year hourly pumping electricity ---
    print(f"\n  Full-year hourly pumping electricity (real hourly mass flow, fixed network hydraulics):")
    demand_kW = demand_result["total_heat_kW"]
    mass_flow_kg_s_hourly = (demand_kW * 1000.0) / (cp * delta_T_K)
    pumping_result = annual_pumping_electricity_MWh(
        topo, sized, mass_flow_kg_s_hourly, density_kg_m3=density,
    )
    annual_heat_MWh = demand_kW.sum() / 1000.0
    print(f"    Annual heat demand: {annual_heat_MWh:,.0f} MWh")
    print(f"    Annual pumping electricity: {pumping_result['annual_pumping_MWh']:,.1f} MWh")
    print(f"    Pumping as % of annual heat demand: "
          f"{pumping_result['annual_pumping_MWh']/annual_heat_MWh*100:.3f}%")

    # --- Sensitivity: lower pump efficiency should mean MORE electricity ---
    print(f"\n  Sensitivity: pump efficiency (default {DEFAULT_PUMP_EFFICIENCY:.0%}):")
    for eff in [0.60, 0.75, 0.90]:
        r = annual_pumping_electricity_MWh(
            topo, sized, mass_flow_kg_s_hourly, density_kg_m3=density, pump_efficiency=eff,
        )
        print(f"    At {eff:.0%} efficiency: {r['annual_pumping_MWh']:,.1f} MWh/year")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert critical_path["round_trip_pressure_drop_Pa"] == critical_path["one_way_pressure_drop_Pa"] * 2.0, \
        "Round-trip pressure drop should be exactly double the one-way figure"
    assert critical_path["worst_case_building_node_id"] is not None, \
        "Should find a real worst-case building on this real Ealing topology"
    assert all(
        drop <= critical_path["one_way_pressure_drop_Pa"] + 1e-6 for drop in all_drops.values()
    ), "The identified worst-case building should genuinely have the highest (or tied-highest) " \
       "one-way pressure drop of every building checked"
    assert 0 < critical_path["round_trip_pressure_drop_Pa"] / 1e5 < 20, \
        "Round-trip pressure drop should be a physically realistic few bar, not a sign/units error"

    assert peak_pump_MW > 0, "Peak pumping power should be positive"
    assert peak_pump_MW < peak_demand_kW / 1000.0, \
        "Pumping power should be a real but SMALL fraction of peak heat demand " \
        "(a value comparable to or exceeding heat demand would indicate a units/physics error)"

    assert pumping_result["annual_pumping_MWh"] > 0, "Annual pumping electricity should be positive"
    assert pumping_result["annual_pumping_MWh"] < annual_heat_MWh * 0.05, \
        "Pumping electricity should be a real but modest fraction (well under 5%) of annual heat " \
        "demand for a well-designed network -- consistent with general DH engineering practice"

    eff_60 = annual_pumping_electricity_MWh(topo, sized, mass_flow_kg_s_hourly, density_kg_m3=density, pump_efficiency=0.60)
    eff_90 = annual_pumping_electricity_MWh(topo, sized, mass_flow_kg_s_hourly, density_kg_m3=density, pump_efficiency=0.90)
    assert eff_60["annual_pumping_MWh"] > eff_90["annual_pumping_MWh"], \
        "LOWER pump efficiency should mean MORE electricity consumed for the same hydraulic work " \
        "(efficiency is in the denominator of the electrical power formula)"

    assert np.array_equal(
        pumping_result["hourly_pumping_MW"][demand_kW == 0],
        np.zeros(int((demand_kW == 0).sum())),
    ) if (demand_kW == 0).any() else True, \
        "Hours with zero heat demand should show zero pumping power (zero flow, zero hydraulic work)"

    print("  ✓ Round-trip pressure drop is exactly double the one-way figure")
    print("  ✓ The identified worst-case building genuinely has the highest pressure drop of any")
    print("    building checked -- confirms the critical-path search is correct, not arbitrary")
    print("  ✓ Round-trip pressure drop is a physically realistic few bar for this real network")
    print("  ✓ Pumping power is a real but small fraction of heat demand, both at peak and annually")
    print("  ✓ Lower pump efficiency correctly means MORE electricity for the same hydraulic work")
    print("  ✓ Zero heat demand correctly produces zero pumping power")
    print()
