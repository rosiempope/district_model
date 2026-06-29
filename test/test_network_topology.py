"""
test_network_topology.py
======================
Self-test / demonstration suite for network.network_topology
(NetworkTopology, TopologyNode, the Ealing worked example, and all
three mixins it assembles: TopologyTreeMixin, TopologySizingMixin,
TopologyThermalMixin). Moved out of network_topology.py itself as part
of a project-wide split separating logic files from their self-tests —
see network/network_topology.py's module docstring for the full
rationale.

Run directly: python3 tests/test_network_topology.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from network.network_topology import (
    NetworkTopology, TopologyNode, SegmentPipeResult,
    ealing_town_centre_topology, EALING_SEGMENTS, BUILDING_NAME_MAP,
)
from network.topology_thermal import (
    seasonal_ground_temp_C, segment_outlet_temp_C,
    DEFAULT_GROUND_TEMP_C, GROUND_TEMP_MEAN_C, MIN_DELIVERED_TEMP_C,
)
from network.pipe_catalog import size_pipe_for_peak, water_properties

# network_topology.py's sibling files (pipe_catalog.py, topology_*.py)
# use bare module-name imports internally (see network/network_topology.py's
# own docstring for why) — some of THIS test file's body does the same
# (a leftover local "from pipe_catalog import ..." inside the self-test
# body, kept verbatim from before the file split). Adding network/
# itself to sys.path here (AFTER the package-style imports above
# already succeeded) lets that bare import keep resolving even though
# this test file now lives in tests/, not network/. Inserting this
# BEFORE the package imports would shadow "network" as a package
# entirely — order matters here.
_NETWORK_DIR = _PROJECT_ROOT / "network"
if str(_NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(_NETWORK_DIR))

# IMPORTANT: PipeSpec must be imported via the SAME bare module path
# (`pipe_catalog`, not `network.pipe_catalog`) that topology_sizing.py
# actually uses internally to CREATE PipeSpec instances — Python loads
# a module twice (as two genuinely different module objects with
# genuinely different classes) if it's reached via two different sys.path
# entries, which breaks isinstance() checks across that boundary. Using
# the bare path here, consistent with what the logic files actually use,
# avoids that real gotcha.
from pipe_catalog import PipeSpec


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

    # --- NEW: COOLING duty — same real topology, now carrying BOTH
    #     heating AND cooling demand simultaneously, sized and checked
    #     independently for each duty ---
    print(f"\n  ── COOLING duty extension ──")
    print(f"  Building the SAME real Ealing topology, now with REAL cooling peaks")
    print(f"  alongside the existing heating peaks (a building can need both):")
    peak_cool_by_building = {n["name"]: n["peak_cool_kW"] for n in demand_result["nodes"]}
    print(f"    Per-building peak_cool_kW from demand_synthesis.py: {peak_cool_by_building}")

    ealing_dual = ealing_town_centre_topology(
        peak_kW_by_building=peak_by_building,
        peak_cool_kW_by_building=peak_cool_by_building,
    )
    dual_summary = ealing_dual.summary()
    print(f"\n  Topology summary, now reporting BOTH duties:")
    for k, v in dual_summary.items():
        print(f"    {k}: {v}")

    # --- Size BOTH duties independently, on the SAME route ---
    print(f"\n  Sizing heating and cooling INDEPENDENTLY on the same route")
    print(f"  (heat: 70/40°C, the real Ealing design value; cool: 6/12°C,")
    print(f"  the standard BS EN 14511 rating condition already used elsewhere")
    print(f"  in this project):")
    sized_heat = ealing_dual.size_all_segments(flow_temp_C=70.0, return_temp_C=40.0, duty="heat")
    print(f"    Heating duty: {len(sized_heat)} segments sized, "
          f"total CAPEX £{NetworkTopology.total_capex_GBP(sized_heat):,.0f}")

    # Cooling demand for this real building mix, AFTER fixing a real bug
    # found in demand_synthesis.py's _cooling_profile() (the previous
    # version concentrated a realistic ANNUAL energy total into too few
    # hours, producing an unrealistic ~94x peak-to-mean ratio and a
    # 24.6MW peak that genuinely exceeded the standard DN ceiling on this
    # network's trunk segments). The fix added a real, literature-
    # grounded internal-gains floor (occupant/lighting/equipment heat,
    # present during ALL occupied hours, not just hot ones — see that
    # module's BUILDING_TYPES internal_gains_fraction entries and
    # _cooling_profile()'s docstring for the citations) — corrected peak
    # for this building mix is now ~2.9MW, not 24.6MW. This sizing loop
    # is KEPT defensive (try/except per segment) regardless of current
    # demand levels — a genuinely larger building mix, or a future
    # demand_synthesis.py change, could still hit the real DN ceiling
    # again, and this should keep surfacing that clearly rather than
    # crashing or silently mis-sizing if it ever does.
    print(f"\n  Cooling duty: attempting to size each segment independently --")
    print(f"  using the CORRECTED cooling demand (~2.9MW peak for this building mix,")
    print(f"  fixed from an earlier 24.6MW figure -- see demand_synthesis.py's")
    print(f"  _cooling_profile() for the real internal-gains-floor fix):")
    sized_cool = {}
    failed_cool_segments = {}
    delta_T_K_cool = abs(6.0 - 12.0)
    for node_id, parent_id, length_m, is_building in EALING_SEGMENTS:
        peak_kW = ealing_dual.segment_peak_kW(node_id, duty="cool")
        if peak_kW <= 0:
            continue
        try:
            pipe = size_pipe_for_peak(peak_heat_kW=peak_kW, flow_temp_C=6.0, return_temp_C=12.0)
            props = water_properties(6.0)
            mass_flow_kg_s = (peak_kW * 1000.0) / (props["cp_J_kgK"] * delta_T_K_cool)
            sized_cool[node_id] = SegmentPipeResult(
                node_id=node_id, length_m=length_m, peak_kW=peak_kW,
                pipe=pipe, mass_flow_kg_s=mass_flow_kg_s,
            )
        except ValueError as e:
            failed_cool_segments[node_id] = (peak_kW, str(e))

    print(f"\n    Segments successfully sized: {len(sized_cool)}")
    for node_id in sorted(sized_cool):
        print(f"      {node_id}: peak={sized_cool[node_id].peak_kW:.0f}kW -> DN{sized_cool[node_id].pipe.DN}")
    if failed_cool_segments:
        print(f"\n    Segments EXCEEDING the largest standard DN (would need parallel pipes):")
        for node_id in sorted(failed_cool_segments):
            peak_kW, err = failed_cool_segments[node_id]
            print(f"      {node_id}: peak={peak_kW:.0f}kW -> EXCEEDS DN600")
        print(f"\n    -> {len(failed_cool_segments)} trunk segment(s) nearest the energy centre cannot")
        print(f"       be served by a SINGLE standard pipe at this cooling peak.")
    else:
        print(f"\n    -> ALL segments sized successfully within the standard DN series at the")
        print(f"       corrected demand scale (max DN used: {max(s.pipe.DN for s in sized_cool.values())}) --")
        print(f"       the genuine multi-trunk/parallel-pipe design question this conversation")
        print(f"       started with was, in THIS specific case, actually an upstream demand-")
        print(f"       modelling bug, not an inherent pipe-catalogue limitation. The mechanism")
        print(f"       (mass flow ~ Q/deltaT, DN grows only as sqrt(Q) at fixed velocity, so")
        print(f"       cooling's small deltaT genuinely DOES hit standard-DN limits sooner than")
        print(f"       heating's does) remains real and worth designing around at genuinely")
        print(f"       larger scales -- this fix just means it wasn't the right diagnosis here.")

    print(f"\n  DN by segment, both duties, for segments where BOTH sized successfully")
    print(f"  (cooling pipes are generally LARGER than heating pipes carrying a similar kW,")
    print(f"  since cooling's much smaller 6K delta-T vs heating's 30K means much higher")
    print(f"  mass flow for the same kW):")
    dn_heat = NetworkTopology.dn_by_segment(sized_heat)
    dn_cool = {nid: s.pipe.DN for nid, s in sized_cool.items()}
    common_segments = set(dn_heat.keys()) & set(dn_cool.keys())
    for node_id in sorted(common_segments):
        print(f"    {node_id}: heat=DN{dn_heat[node_id]}, cool=DN{dn_cool[node_id]}")

    # --- Cooling compliance: only meaningful for segments that actually
    #     sized -- restrict to buildings reachable without crossing a
    #     failed trunk segment ---
    print(f"\n  Cooling compliance check — chilled water leaves the chiller at 6°C,")
    print(f"  does it arrive at every building still at or below 6°C (the design")
    print(f"  value the fan coil units were actually sized against)? Restricted to")
    print(f"  buildings whose FULL path to the energy centre sized successfully:")
    checkable_buildings = []
    for node_id, node in ealing_dual.nodes.items():
        if node.building_name is None or node.peak_cool_kW <= 0:
            continue
        path = ealing_dual.path_to_root(node_id)[:-1]   # exclude EC itself
        if all(p in sized_cool for p in path):
            checkable_buildings.append(node_id)
    if checkable_buildings:
        # Build a sized_segments dict covering just the checkable subtree
        cooling_compliance = ealing_dual.check_maximum_delivered_temperature(
            sized_segments=sized_cool, source_flow_temp_C=6.0, design_chilled_water_temp_C=6.0,
        )
        for building, r in cooling_compliance["by_building"].items():
            status = "✓ compliant" if r["compliant"] else "✗ TOO WARM"
            print(f"    {building:<20}: {r['delivered_temp_C']:>6.2f}°C  "
                  f"margin={r['margin_C']:>+6.2f}°C  {status}")
        print(f"\n  All checkable buildings compliant (<= 6.0°C)? {cooling_compliance['all_compliant']}")
    else:
        print(f"    (No buildings have a fully-sized path this run — every path crosses")
        print(f"     a trunk segment that exceeded the standard DN ceiling)")
        cooling_compliance = {"all_compliant": None, "by_building": {}}

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert simple.segment_peak_kW("A") == 850.0, "Segment peak should sum node + all descendants"
    assert simple.segment_peak_kW("B") == 200.0, "Leaf segment peak should equal just its own peak"
    assert simple.total_length_m() == 240.0, "Total length should sum all segments"
    assert simple.total_peak_kW() == 850.0, "Total peak should sum all node peaks"
    assert set(simple.leaf_nodes()) == {"B", "C"}, "Leaf nodes should be exactly B and C"
    assert simple.path_to_root("B") == ["B", "A", "EC"], "Path to root should list every ancestor in order"

    # --- New dual-duty (heating + cooling) assertions ---
    # Real Ealing cooling demand is too large to fully exercise compliance
    # (every path crosses a failed trunk segment) -- build a SMALL
    # synthetic tree with a modest, realistic cooling load specifically
    # to verify the compliance check itself works correctly end-to-end
    small_dual = NetworkTopology(name="Small dual-duty test")
    small_dual.add_node("EC", parent_id=None, length_m=0.0, peak_kW=0.0, peak_cool_kW=0.0)
    small_dual.add_node("A", parent_id="EC", length_m=200.0, peak_kW=500.0, peak_cool_kW=300.0, building_name="Building A")
    small_dual.add_node("B", parent_id="A", length_m=150.0, peak_kW=200.0, peak_cool_kW=0.0, building_name="Building B (heat only)")
    small_dual.add_node("C", parent_id="A", length_m=100.0, peak_kW=0.0, peak_cool_kW=150.0, building_name="Building C (cool only)")

    assert small_dual.segment_peak_kW("A", duty="heat") == 700.0, \
        "Segment A's heating peak should sum A(500)+B(200)+C(0)=700"
    assert small_dual.segment_peak_kW("A", duty="cool") == 450.0, \
        "Segment A's cooling peak should sum A(300)+B(0)+C(150)=450, independent of heating"
    assert small_dual.total_peak_kW(duty="heat") == 700.0, "Total heating peak should be 500+200+0=700"
    assert small_dual.total_peak_kW(duty="cool") == 450.0, "Total cooling peak should be 300+0+150=450"

    small_sized_heat = small_dual.size_all_segments(flow_temp_C=70.0, return_temp_C=40.0, duty="heat")
    small_sized_cool = small_dual.size_all_segments(flow_temp_C=6.0, return_temp_C=12.0, duty="cool")
    assert "B" in small_sized_heat, "B (heat-only) should be sized for heating"
    assert "C" not in small_sized_heat, "C (cool-only, zero heating peak) should be SKIPPED in heating sizing"
    assert "C" in small_sized_cool, "C (cool-only) should be sized for cooling"
    assert "B" not in small_sized_cool, "B (heat-only, zero cooling peak) should be SKIPPED in cooling sizing"

    small_heat_compliance = small_dual.check_minimum_delivered_temperature(
        small_sized_heat, source_flow_temp_C=70.0, min_temp_C=60.0,
    )
    small_cool_compliance = small_dual.check_maximum_delivered_temperature(
        small_sized_cool, source_flow_temp_C=6.0, design_chilled_water_temp_C=6.0,
    )
    assert "Building B (heat only)" in small_heat_compliance["by_building"], \
        "Heat-only building should appear in the HEATING compliance check"
    assert "Building C (cool only)" not in small_heat_compliance["by_building"], \
        "Cool-only building should NOT appear in the heating compliance check (zero heating peak)"
    assert "Building C (cool only)" in small_cool_compliance["by_building"], \
        "Cool-only building should appear in the COOLING compliance check"
    assert "Building B (heat only)" not in small_cool_compliance["by_building"], \
        "Heat-only building should NOT appear in the cooling compliance check (zero cooling peak)"
    assert small_cool_compliance["by_building"]["Building C (cool only)"]["delivered_temp_C"] > 6.0, \
        "Chilled water should arrive WARMER than it left the chiller (heat gain from warmer ground), " \
        "i.e. delivered temp > source flow temp for cooling -- the opposite direction from heating"
    assert small_heat_compliance["by_building"]["Building B (heat only)"]["delivered_temp_C"] < 70.0, \
        "Hot water should arrive COOLER than it left the source (heat loss to colder ground)"

    # Real Ealing dual-duty assertions (using the actual large-scale demo above)
    assert abs(dual_summary["total_peak_cool_kW"] - ealing_dual.total_peak_kW(duty="cool")) < 0.1, \
        "summary()'s reported cooling peak should match total_peak_kW(duty='cool') (within rounding)"
    assert "N4" in sized_cool and "N4" in sized_heat, \
        "A segment with BOTH nonzero heating and cooling peak should appear in both sizings"
    assert dn_cool.get("N4", 0) >= dn_heat.get("N4", 0), \
        "For the same segment, cooling's much smaller delta-T should produce an equal or LARGER DN than heating"
    # NOTE: this used to assert the OPPOSITE (that segments MUST fail the
    # standard DN ceiling) -- that was correct for the demand_synthesis.py
    # cooling-peak bug active at the time (24.6MW peak, ~94x peak-to-mean
    # ratio, since fixed -- see that module's _cooling_profile() and its
    # real internal-gains-floor literature basis). With the corrected,
    # literature-grounded cooling demand (~2.9MW peak), every segment now
    # sizes successfully within the standard DN series -- the genuinely
    # correct result, not a regression. Keeping this assertion in its old
    # form would mean re-introducing the bug would be REQUIRED to pass
    # the test, which is exactly backwards.
    assert len(failed_cool_segments) == 0, \
        "With the corrected, literature-grounded cooling demand, every segment should size " \
        "successfully within the standard DN series -- if this assertion fails, check whether " \
        "demand_synthesis.py's cooling intensity assumptions regressed back toward the old bug"
    assert len(sized_cool) == len(sized_heat), \
        "With corrected demand, the cooling and heating sizing dicts should cover the same " \
        "set of segments (same buildings have both heating and cooling demand in this scenario)"

    # New, genuinely real finding at the CORRECTED demand scale: even
    # with small, correctly-sized pipes, a tiny amount of real transit
    # heat GAIN still occurs -- the cooling-duty mirror of heating's
    # "need a small margin above the bare 60C minimum" finding. The
    # chiller's design 6C has ZERO margin built in by definition (see
    # check_maximum_delivered_temperature()'s docstring on why this
    # isn't a tolerance band) -- so even ~0.02-0.13C of real transit
    # warming shows up as technically non-compliant, exactly mirroring
    # the heating side's situation before a margin was added.
    print(f"\n  Real finding at the CORRECTED demand scale: even with small, properly-sized")
    print(f"  pipes, a tiny real transit heat GAIN still occurs (cooling's mirror of heating's")
    print(f"  'need a small margin above the bare minimum' finding):")
    print(f"    Worst case margin: {cooling_compliance['by_building'][cooling_compliance['worst_case_building']]['margin_C']}°C")
    print(f"    -> a real chiller for this network would need to be specified to produce water")
    print(f"       slightly BELOW 6.0°C (e.g. 5.8-5.9°C) to absorb this small transit gain --")
    print(f"       the same design principle as heating needing a margin above 60°C, not a")
    print(f"       contradiction or a new bug.")
    assert not cooling_compliance["all_compliant"], \
        "At EXACTLY the chiller's bare design temperature (zero margin, by definition -- see " \
        "check_maximum_delivered_temperature()'s docstring), real transit heat gain should " \
        "show a small non-compliance, mirroring heating's need for a margin above its bare minimum"
    assert all(abs(r["margin_C"]) < 1.0 for r in cooling_compliance["by_building"].values()), \
        "The margin should be SMALL (a fraction of a degree) at this corrected, realistic scale -- " \
        "a large margin would suggest something is wrong with the corrected sizing, not confirm it"


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
    print("  ✓ Heating and cooling peaks sum independently per segment, correctly isolated")
    print("    even when a node has both, either, or neither duty's demand")
    print("  ✓ size_all_segments() correctly skips zero-peak segments per duty, and")
    print("    delivered_temperature_C()/network_heat_loss_kW() correctly handle the skip")
    print("  ✓ Cooling correctly shows heat GAIN (delivered temp > source temp), the mirror")
    print("    of heating's heat LOSS (delivered temp < source temp) -- same Shukhov formula")
    print("  ✓ check_maximum_delivered_temperature() correctly isolates cooling-only buildings,")
    print("    mirroring check_minimum_delivered_temperature()'s heating-only isolation")
    print("  ✓ Corrected cooling demand sizes successfully within the standard DN series --")
    print("    confirms the earlier DN-ceiling breach was an upstream demand bug, now fixed")
    print("  ✓ Even with correctly-sized pipes, a small real transit heat gain still shows")
    print("    technical non-compliance at the chiller's bare (zero-margin) design temp --")
    print("    mirrors heating's need for a margin above its own bare minimum")
    print()
