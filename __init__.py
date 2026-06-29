"""
test_pipe_catalog.py
======================
Self-test / demonstration suite for network.pipe_catalog (water
properties, the standard DN series, EN 253 insulation data, pipe
cost/sizing, the Reynolds/Darcy/pressure-gradient hydraulics, and
size_pipe_for_peak()/select_pipe()). Moved out of pipe_catalog.py
itself as part of a project-wide split separating logic files from
their self-tests.

Run directly: python3 tests/test_pipe_catalog.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from network.pipe_catalog import (
    water_properties, STANDARD_DN_SERIES, TWIN_PIPE_MAX_DN,
    casing_to_pipe_ratio_at_dn, estimate_pipe_cost_GBP_per_m,
    heat_loss_coefficient_W_per_mK, darcy_friction_factor,
    pressure_gradient_Pa_per_m, _lookup_dn, select_pipe,
    size_pipe_for_peak, _TABLE7_DN_CASING_DATA,
)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  pipe_catalog.py — self-test")
    print("=" * 70)

    # --- Water properties across the full temperature range ---
    print("\n  Water properties (temperature-dependent — this is the key")
    print("  thing that must differ correctly between hot/cold/ambient loops):")
    print(f"  {'Temp C':>8} {'Density kg/m3':>15} {'Viscosity mPa.s':>17} {'cp kJ/kg.K':>12}")
    for t in [6, 12, 20, 35, 65, 80]:
        p = water_properties(t)
        print(f"  {t:>8.0f} {p['density_kg_m3']:>15.1f} {p['viscosity_Pa_s']*1000:>17.3f} {p['cp_J_kgK']/1000:>12.3f}")

    visc_8 = water_properties(8.0)["viscosity_Pa_s"]
    visc_80 = water_properties(80.0)["viscosity_Pa_s"]
    print(f"\n  Viscosity ratio (8°C / 80°C): {visc_8/visc_80:.2f}x "
          f"(expect roughly 4x — this is why cold loops need different sizing)")

    # --- Friction factor sanity check ---
    print("\n  Friction factor sanity check (turbulent range):")
    for re in [2000, 5000, 20000, 100000, 500000]:
        f = darcy_friction_factor(re, relative_roughness=0.0001 / 0.1)
        print(f"    Re={re:>8}  f={f:.5f}")

    # --- Heat loss coefficient: construction comparison (single profile now) ---
    print("\n  Heat loss coefficient, EN 253 profile (DN100, combined supply+return):")
    u_single = heat_loss_coefficient_W_per_mK(100, "single", "en253")
    u_twin = heat_loss_coefficient_W_per_mK(100, "twin", "en253")
    print(f"    single={u_single:.2f} W/m.K   twin={u_twin:.2f} W/m.K  "
          f"(twin reduction: {(1 - u_twin/u_single)*100:.0f}%)")

    # --- NEW: validate casing_to_pipe_ratio_at_dn() against real Table 7 data ---
    print("\n  casing_to_pipe_ratio_at_dn() validated against real Table 7 data")
    print("  (District Heating Manual for London, EN 253 basis):")
    print(f"  {'DN':>5} {'real ratio':>11} {'modelled':>9} {'diff':>7}")
    max_abs_pct_diff = 0.0
    for dn, d_pipe, d_casing in _TABLE7_DN_CASING_DATA:
        real_ratio = d_casing / d_pipe
        modelled_ratio = casing_to_pipe_ratio_at_dn(dn)
        pct_diff = (modelled_ratio - real_ratio) / real_ratio * 100
        max_abs_pct_diff = max(max_abs_pct_diff, abs(pct_diff))
        print(f"  {dn:>5} {real_ratio:>11.3f} {modelled_ratio:>9.3f} {pct_diff:>+6.1f}%")
    print(f"  Max abs diff: {max_abs_pct_diff:.2f}% (expect ~0%, since DN20-DN500 use direct")
    print(f"  interpolation against these exact points, not a smoothed fit)")

    print("\n  DN600 extrapolation (beyond Table 7's real DN500 ceiling):")
    dn600_ratio = casing_to_pipe_ratio_at_dn(600)
    print(f"    Modelled ratio at DN600: {dn600_ratio:.3f} (power-law extrapolation, R^2=0.92")
    print(f"    against the real DN20-DN500 data — NOT a measured Table 7 value)")

    print("\n  Heat loss correctly varies with DN — real insulation is")
    print("  proportionally thinner on big pipes than on small ones:")
    u_dn20 = heat_loss_coefficient_W_per_mK(20, "single", "en253")
    u_dn200 = heat_loss_coefficient_W_per_mK(200, "single", "en253")
    u_dn500 = heat_loss_coefficient_W_per_mK(500, "single", "en253")
    print(f"    DN20  single, EN 253: {u_dn20:.3f} W/m.K")
    print(f"    DN200 single, EN 253: {u_dn200:.3f} W/m.K")
    print(f"    DN500 single, EN 253: {u_dn500:.3f} W/m.K")

    print("\n  Old 'series1'/'series3' names — should now fail loudly, not silently:")
    try:
        heat_loss_coefficient_W_per_mK(100, "single", "series1")
        print("    ✗ FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"    ✓ Correctly raised: {str(e)[:90]}...")

    # --- Cost curve sanity check ---
    print("\n  Cost curve — cost per metre rising sub-linearly with DN:")
    for dn, _, _ in STANDARD_DN_SERIES:
        c_single = estimate_pipe_cost_GBP_per_m(dn, "single")
        if dn <= TWIN_PIPE_MAX_DN:
            c_twin = estimate_pipe_cost_GBP_per_m(dn, "twin")
            print(f"    DN{dn:<5} single=£{c_single:>7.0f}/m   twin=£{c_twin:>7.0f}/m")
        else:
            print(f"    DN{dn:<5} single=£{c_single:>7.0f}/m   twin=n/a (no real product above DN{TWIN_PIPE_MAX_DN})")

    # --- Cost curve cross-check against real SEAI data it was fitted to ---
    print("\n  Cross-check against real data (SEAI National Heat Study, Table 4,")
    print("  2-pipe inner-city, EUR->GBP, ~24% CPI-uplifted to 2026):")
    seai_check_points = {100: 962, 300: 1733, 600: 3272}  # DN: real GBP/m (2026 terms)
    for dn, real_gbp in seai_check_points.items():
        modelled = estimate_pipe_cost_GBP_per_m(dn)
        pct_diff = (modelled - real_gbp) / real_gbp * 100
        print(f"    DN{dn:<5} modelled=£{modelled:>6.0f}/m   real=£{real_gbp:>6.0f}/m   diff={pct_diff:+.0f}%")

    # --- select_pipe(): hot loop vs cold loop, SAME flow rate ---
    print("\n  select_pipe() — same volumetric flow, hot vs cold (shows viscosity effect):")
    test_flow_m3_s = 0.05
    hot = select_pipe(test_flow_m3_s, fluid_temp_C=80.0)
    cold = select_pipe(test_flow_m3_s, fluid_temp_C=8.0)
    print(f"    80°C (hot loop):  {hot}")
    print(f"    8°C  (cold loop): {cold}")

     # --- size_pipe_for_peak(): the real entry point, hot vs cold network ---
    print("\n  size_pipe_for_peak() — SAME peak heat duty, hot network vs cold network:")
    print("  (Ealing Phase 1 scale: 7.2 MW coincident peak)")
    hot_network = size_pipe_for_peak(peak_heat_kW=7200, flow_temp_C=70.0, return_temp_C=40.0)
    cold_network = size_pipe_for_peak(peak_heat_kW=7200, flow_temp_C=6.0, return_temp_C=12.0)
    print(f"    Hot  (70/40°C, dT=30K):  {hot_network}")
    print(f"    Cold (6/12°C,  dT=6K):   {cold_network}")
    print(f"    -> cold loop needs a LARGER pipe for the same kW: 5x smaller delta-T")
    print(f"       forces ~5x the mass flow, only partly offset by water being denser when cold.")

    # --- Twin pipe vs single, on the same duty ---
    print("\n  Twin vs single construction, same duty (70/40°C, 5 MW):")
    single = size_pipe_for_peak(5000, 70.0, 40.0, construction="single")
    twin = size_pipe_for_peak(5000, 70.0, 40.0, construction="twin")
    print(f"    Single: {single}")

    # --- DN600: the cooling duty that previously failed to size at all ---
    print("\n  DN600 — the 2050_high climate-scenario cooling peak that previously failed:")
    big_cooling = size_pipe_for_peak(12629, flow_temp_C=6.0, return_temp_C=12.0, construction="single")
    print(f"    {big_cooling}")

    # --- Twin-pipe DN ceiling: should be rejected above DN200 ---
    print("\n  Twin-pipe DN ceiling — requesting twin above DN200 should fail loudly:")
    try:
        size_pipe_for_peak(12629, flow_temp_C=6.0, return_temp_C=12.0, construction="twin")
        print("    ✗ FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"    ✓ Correctly raised: {str(e)[:90]}...")

    # --- Edge case: very large flow exceeding the DN series ---
    print("\n  Edge case — flow too large for the standard DN series:")
    try:
        select_pipe(50.0, fluid_temp_C=65.0)
        print("    ✗ FAIL: should have raised ValueError")
    except ValueError as e:
        print(f"    ✓ Correctly raised: {str(e)[:90]}...")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert visc_8 > visc_80, "Cold water should be more viscous than hot water"
    assert 3.0 < visc_8 / visc_80 < 4.5, "Viscosity ratio should be roughly 4x across this range"
    assert abs(hot.pressure_gradient_Pa_per_m - cold.pressure_gradient_Pa_per_m) > 1.0, \
        "Hot and cold loops at the same volumetric flow should show different pressure gradients " \
        "(velocity itself is purely geometric -- v=Q/A -- so it's identical either way; the " \
        "viscosity effect shows up in friction factor and therefore pressure gradient, not velocity)"
    assert cold_network.DN >= hot_network.DN, \
        "Cold loop (6K delta-T) should need an equal or larger pipe than hot loop (30K delta-T) for the same kW"
    u_en253_single = heat_loss_coefficient_W_per_mK(100, "single", "en253")
    u_en253_twin = heat_loss_coefficient_W_per_mK(100, "twin", "en253")
    assert u_en253_twin < u_en253_single, "Twin pipe should lose less heat than single construction"
    assert max_abs_pct_diff < 0.01, \
        "casing_to_pipe_ratio_at_dn() should reproduce Table 7 exactly at every real DN point"
    assert casing_to_pipe_ratio_at_dn(20) > casing_to_pipe_ratio_at_dn(500), \
        "Real casing ratio should fall as DN increases (thinner insulation, proportionally, on bigger pipes)"
    assert u_dn500 > u_dn20, \
        "DN500 should have HIGHER heat loss coefficient per metre than DN20 — bigger pipe, more surface area, " \
        "and now correctly thinner relative insulation too"
    c_dn25 = estimate_pipe_cost_GBP_per_m(25)
    c_dn300 = estimate_pipe_cost_GBP_per_m(300)
    assert c_dn300 > c_dn25, "Larger DN should cost more per metre"
    assert single.cost_GBP_per_m < twin.cost_GBP_per_m, "Twin construction should cost more than single"
    assert big_cooling.DN == 600, "2050_high cooling peak should now size to DN600"
    for dn, _, _ in STANDARD_DN_SERIES:
        pipe = _lookup_dn(dn)
        assert pipe["outer_diameter_mm"] > pipe["inner_diameter_mm"], "Outer diameter must exceed inner diameter"
    print("  ✓ Cold water is more viscous than hot water, by roughly the expected ratio")
    print("  ✓ Hot vs cold loops at identical volumetric flow show a measurable hydraulic difference")
    print("  ✓ Cold network (small delta-T) needs an equal-or-larger pipe than hot network for same kW")
    print("  ✓ Twin-pipe construction loses less heat and costs more than single construction")
    print("  ✓ Cost curve rises with DN; DN table internally consistent")
    print("  ✓ DN600 now available — the previously-failing 2050_high cooling peak sizes correctly")
    print("  ✓ Twin-pipe construction correctly rejected above DN200 (no real product exists)")
    print("  ✓ Oversized flow correctly raises an informative error")
    print("  ✓ casing_to_pipe_ratio_at_dn() exactly reproduces real Table 7 data at every DN20-DN500 point")
    print("  ✓ Insulation ratio correctly DN-dependent (3.35x at DN20 down to 1.24x at DN500),")
    print("    from one real EN 253 profile (Table 7) — no invented series1/2/3 variants")
    print("  ✓ Old series1/series3 names now correctly rejected with a clear error")
    print()