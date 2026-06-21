"""
pipe_catalog.py
================
Standard pipe series, hydraulics, heat-loss, and cost lookups for the
district energy network. This is the lowest-level physics module that
topology.py and hydraulics.py will both build on.

Deliberately generation-agnostic
-----------------------------------
This module doesn't know or care whether the fluid is an 80°C network
supply pipe, an 8°C chilled water pipe, or a 20°C ambient loop pipe.
Diameter selection, heat-loss coefficients, and cost are all functions of
(DN, insulation series, fluid temperature, flow rate) — nothing here is
hardcoded to "hot network" assumptions. This is what lets the same catalog
serve:
    Gen 3  — a single 2-pipe heating loop (this is what we're building first)
    4-pipe — a hot loop + a cold loop sharing the same trench/route
    Gen 5  — a single 2-pipe ambient loop, stepped up/down by decentralised
              building-level heat pumps
without rewriting any of this module when those scenarios arrive.

Why water properties matter here specifically
-------------------------------------------------
Density changes only a few percent across the 0-100°C range relevant to
district energy, but dynamic viscosity changes by roughly 4x between an
8°C chilled loop and an 80°C heating loop. That changes Reynolds number,
friction factor, and therefore pressure drop and required diameter for
the SAME kW duty — a cold loop carrying the same heat flow as a hot loop
needs a meaningfully different pipe size, both because of viscosity and
because cooling networks typically run a much smaller design ΔT (forcing
higher mass flow for the same kW). Hardcoding water properties at one
reference temperature would make gen-3 results look fine while quietly
producing wrong pipe sizes for the cold/ambient loops down the line — so
properties are temperature-dependent from the start.

Pipe sizing methodology (dual criterion — standard industry practice)
--------------------------------------------------------------------------
For a given coincident peak flow, the smallest standard DN is selected
that satisfies BOTH:
    1. Velocity within an allowable band (default 0.3-2.5 m/s) — below
       the lower bound risks sediment deposition, above the upper bound
       risks noise and erosion-corrosion.
    2. Pressure gradient below a target (default 100-150 Pa/m, published
       guidance ranges roughly 30-350 Pa/m depending on source/country).
In transmission/trunk mains velocity tends to govern; in distribution
branches the pressure gradient usually governs. Both are checked
simultaneously here, as is standard practice.

Heat loss methodology
------------------------
Computed analytically from the cylindrical thermal resistance of the
insulation layer (PUR foam, k ~= 0.023-0.026 W/m.K typical for modern
pre-insulated pipe), following the standard logarithmic-resistance model
for concentric cylinders. Steel pipe wall resistance and soil resistance
are neglected — a standard simplification that's valid for well-insulated
pipe, used throughout the academic DH literature at feasibility stage.
This is a GENERIC approximation suitable for screening-level work — for
a procurement-grade design, replace the insulation-thickness assumptions
here with the specific manufacturer's catalogue (e.g. Logstor, Brugg,
Isoplus all publish per-DN, per-series heat-loss tables).

Cost methodology
-------------------
£/m installed cost is dominated by trenching, not pipe material (pipe
network typically accounts for 40-60% of total DH investment cost, and
within that, civils/trenching usually dominates pipe material). Modelled
as a power-law cost curve — cost per metre rises with DN but sub-linearly,
mirroring the same pattern used in thermal_storage.py's CAPEX curve.
Calibrated so a ~DN100 distribution pipe lands close to the £1,200/m
figure already used in your scenario YAML (Community Heat Development
Unit-style UK benchmark) — treat the curve as a sensitivity input, not a
quoted price; get a real trenching quote before using this for investment
decisions.

Usage
-----
    from network.pipe_catalog import size_pipe_for_peak, water_properties

    # Gen 3 hot loop: 7.2 MW peak (Ealing Phase 1 figure), 65/35 deg C
    pipe = size_pipe_for_peak(
        peak_heat_kW=7200, flow_temp_C=65.0, return_temp_C=35.0,
    )
    print(pipe.DN, pipe.velocity_ms, pipe.cost_GBP_per_m)

    # Same duty, but as a chilled loop instead — notice the different DN
    pipe_cold = size_pipe_for_peak(
        peak_heat_kW=7200, flow_temp_C=6.0, return_temp_C=12.0,
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

GRAVITY = 9.81

# Steel pipe roughness (mm). New commercial steel ~0.045mm; aged/used pipe
# can be higher. 0.1mm is a reasonable mid-life default for a feasibility
# model — override per-segment if you have better information.
DEFAULT_ROUGHNESS_MM = 0.1

# PUR (polyurethane) foam insulation thermal conductivity (W/m.K) at typical
# mean operating temperature. Range 0.023-0.026 is standard for modern
# pre-insulated pipe; 0.0245 is the midpoint default.
DEFAULT_INSULATION_K_W_MK = 0.0245


# ── Water properties — temperature dependent ───────────────────────────────────
# Standard saturated-liquid water property table at 1 atm (values consistent
# with standard steam/water property references, e.g. NIST/Perry's). Linearly
# interpolated for any temperature in range. This is the piece that makes the
# catalog correctly distinguish a hot loop from a cold/ambient one.

_WATER_TEMP_C = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_WATER_DENSITY_KG_M3 = np.array(
    [999.8, 999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.7, 971.6, 965.3, 958.4]
)
_WATER_VISCOSITY_MPA_S = np.array(
    [1.792, 1.307, 1.002, 0.798, 0.653, 0.547, 0.467, 0.404, 0.355, 0.315, 0.282]
)
_WATER_CP_KJ_KGK = np.array(
    [4.217, 4.192, 4.182, 4.178, 4.179, 4.181, 4.185, 4.190, 4.196, 4.205, 4.216]
)


def water_properties(temp_C: float) -> dict:
    """
    Return density (kg/m3), dynamic viscosity (Pa.s), and specific heat
    (J/kg.K) of water at the given temperature, via linear interpolation
    of a standard property table (valid 0-100 deg C at ~1 atm).

    This is what makes pipe sizing correctly differ between a hot network
    loop and a cold/ambient one — viscosity alone changes by roughly 3x
    across that range, directly affecting Reynolds number and friction
    factor for the same flow rate.
    """
    t = float(np.clip(temp_C, 0.0, 100.0))
    rho = float(np.interp(t, _WATER_TEMP_C, _WATER_DENSITY_KG_M3))
    mu = float(np.interp(t, _WATER_TEMP_C, _WATER_VISCOSITY_MPA_S)) * 1e-3  # mPa.s -> Pa.s
    cp = float(np.interp(t, _WATER_TEMP_C, _WATER_CP_KJ_KGK)) * 1000.0     # kJ/kg.K -> J/kg.K
    return {"density_kg_m3": rho, "viscosity_Pa_s": mu, "cp_J_kgK": cp}


# ── Standard DN pipe series ────────────────────────────────────────────────────
# Generic standard-wall steel pipe series typical of EN253-compliant
# pre-insulated district heating pipe. These are representative values for
# feasibility-stage screening — cross-check against the specific
# manufacturer's datasheet (Logstor, Brugg, Isoplus etc.) before using for
# a procurement-grade design. Inner/outer diameters in mm.

STANDARD_DN_SERIES = [
    # (DN, inner_diameter_mm, outer_diameter_mm)
    (20,  21.7,  26.9),
    (25,  27.3,  33.7),
    (32,  36.0,  42.4),
    (40,  41.8,  48.3),
    (50,  53.0,  60.3),
    (65,  68.8,  76.1),
    (80,  80.8,  88.9),
    (100, 105.3, 114.3),
    (125, 130.0, 139.7),
    (150, 154.1, 168.3),
    (200, 202.7, 219.1),
    (250, 254.5, 273.0),
    (300, 304.9, 323.9),
    (350, 333.4, 355.6),
    (400, 387.4, 406.4),
    (500, 488.0, 508.0),
]


# ── Insulation series presets ──────────────────────────────────────────────────
# casing_to_pipe_ratio = outer casing diameter / pipe outer diameter.
# Higher ratio = thicker insulation = lower heat loss, higher cost.
# Generic approximation of typical Series 1/2/3 pre-insulated pipe — swap
# in manufacturer-specific casing diameters for a procurement-grade design.

INSULATION_SERIES_PRESETS = {
    "series1": {
        "description":         "Standard insulation (thinnest, cheapest)",
        "casing_to_pipe_ratio": 1.45,
        "insulation_k_W_mK":    DEFAULT_INSULATION_K_W_MK,
    },
    "series2": {
        "description":         "Reinforced insulation (typical UK DH default)",
        "casing_to_pipe_ratio": 1.65,
        "insulation_k_W_mK":    DEFAULT_INSULATION_K_W_MK,
    },
    "series3": {
        "description":         "Thick insulation (long transmission mains, high-loss-sensitivity routes)",
        "casing_to_pipe_ratio": 1.85,
        "insulation_k_W_mK":    DEFAULT_INSULATION_K_W_MK,
    },
}

DEFAULT_INSULATION_SERIES = "series2"

# Twin-pipe (flow + return sharing one casing) loses meaningfully less heat
# than two separate single pre-insulated pipes of the same series, at a
# capital cost premium. Empirical approximation from published thermal/
# economic comparisons — not a substitute for a manufacturer twin-pipe
# datasheet at design stage.
TWIN_PIPE_LOSS_REDUCTION_FACTOR = 0.55   # combined twin loss = 0.55 x (2 x single-pipe loss)
TWIN_PIPE_COST_PREMIUM_FACTOR   = 1.15   # +15% capex vs single construction


# ── Pipe cost curve ─────────────────────────────────────────────────────────────
# Power-law cost curve (£/m installed, dominated by trenching) — same
# pattern as thermal_storage.py's CAPEX curve. Calibrated so a ~DN100
# distribution pipe lands close to the £1,200/m figure already used in
# your scenario YAML (Community Heat Development Unit-style UK benchmark).
# Sensitivity input, not a quoted price.

PIPE_COST_REFERENCE_DN        = 100
PIPE_COST_GBP_PER_M_AT_REF_DN = 1200.0
PIPE_COST_SCALE_EXPONENT      = 0.55   # cost/m rises sub-linearly with DN


def estimate_pipe_cost_GBP_per_m(
    DN: int,
    construction: str = "single",
) -> float:
    """
    Estimate installed cost (£/m) for a given DN, via the power-law cost
    curve. construction='twin' applies the twin-pipe cost premium.
    """
    cost = PIPE_COST_GBP_PER_M_AT_REF_DN * (
        DN / PIPE_COST_REFERENCE_DN
    ) ** PIPE_COST_SCALE_EXPONENT

    if construction == "twin":
        cost *= TWIN_PIPE_COST_PREMIUM_FACTOR
    elif construction != "single":
        raise ValueError(f"construction must be 'single' or 'twin'; got '{construction}'")

    return cost


# ── Heat loss coefficient ───────────────────────────────────────────────────────

def heat_loss_coefficient_W_per_mK(
    DN: int,
    construction: str = "single",
    insulation_series: str = DEFAULT_INSULATION_SERIES,
) -> float:
    """
    Combined heat-loss coefficient (W per metre of pipe RUN, per Kelvin of
    temperature difference to the surrounding ground) for a full 2-pipe
    segment — i.e. ALREADY includes both supply and return. Multiply by
    segment length and (pipe_temp - ground_temp) to get total heat loss
    in watts; the sign of that temperature difference handles cold/ambient
    loops correctly (heat flows IN from the ground when the pipe is colder
    than the ground, which is a genuine loss for a cooling network).

    Computed from the standard logarithmic thermal resistance of a
    cylindrical insulation layer:
        R_per_m = ln(D_casing / D_pipe_outer) / (2 * pi * k_insulation)
        U_per_m = 1 / R_per_m
    Steel pipe wall and soil resistance are neglected (standard
    simplification, valid for well-insulated pipe — see module docstring).

    Parameters
    ----------
    DN                 : nominal pipe size (must be in STANDARD_DN_SERIES)
    construction        : 'single' (two separate pre-insulated pipes) or
                          'twin' (shared casing — lower combined loss)
    insulation_series   : one of INSULATION_SERIES_PRESETS
    """
    pipe = _lookup_dn(DN)
    series = INSULATION_SERIES_PRESETS[insulation_series]

    D_pipe_outer_m = pipe["outer_diameter_mm"] / 1000.0
    D_casing_m = D_pipe_outer_m * series["casing_to_pipe_ratio"]
    k_ins = series["insulation_k_W_mK"]

    R_per_m = np.log(D_casing_m / D_pipe_outer_m) / (2 * np.pi * k_ins)
    U_single_per_m = 1.0 / R_per_m   # W/m.K, ONE pipe

    if construction == "single":
        # Two separate pipes (supply + return) -> simple sum
        return 2.0 * U_single_per_m
    elif construction == "twin":
        # Shared casing -> meaningfully lower combined loss than 2x single
        return 2.0 * U_single_per_m * TWIN_PIPE_LOSS_REDUCTION_FACTOR
    else:
        raise ValueError(f"construction must be 'single' or 'twin'; got '{construction}'")


# ── Hydraulics: friction factor, velocity, pressure gradient ───────────────────

def reynolds_number(
    velocity_ms: float,
    inner_diameter_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
) -> float:
    """Re = rho * v * D / mu"""
    return density_kg_m3 * velocity_ms * inner_diameter_m / viscosity_Pa_s


def darcy_friction_factor(
    reynolds: float,
    relative_roughness: float,
) -> float:
    """
    Darcy friction factor via the Swamee-Jain explicit approximation to the
    Colebrook-White equation — accurate to within ~1% across the turbulent
    range relevant here (Re > 4000), without needing an iterative solve.
    Falls back to the laminar formula (f = 64/Re) below Re=2300, which
    should essentially never bind in a real network but is included so the
    function doesn't misbehave if called with an unrealistically low flow.

    Parameters
    ----------
    reynolds            : Reynolds number (dimensionless)
    relative_roughness  : pipe roughness / inner diameter (dimensionless)
    """
    re = max(float(reynolds), 1.0)   # guard against divide-by-zero at zero flow

    if re < 2300:
        return 64.0 / re

    # Swamee-Jain explicit approximation
    term = relative_roughness / 3.7 + 5.74 / re ** 0.9
    f = 0.25 / (np.log10(term) ** 2)
    return f


def pressure_gradient_Pa_per_m(
    flow_m3_s: float,
    inner_diameter_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    roughness_mm: float = DEFAULT_ROUGHNESS_MM,
) -> tuple[float, float, float]:
    """
    Darcy-Weisbach pressure gradient for a given volumetric flow rate and
    pipe diameter. Returns (pressure_gradient_Pa_per_m, velocity_ms,
    reynolds_number) so callers get the intermediate values for free.
    """
    area_m2 = np.pi * (inner_diameter_m ** 2) / 4.0
    velocity_ms = flow_m3_s / area_m2

    re = reynolds_number(velocity_ms, inner_diameter_m, density_kg_m3, viscosity_Pa_s)
    rel_roughness = (roughness_mm / 1000.0) / inner_diameter_m
    f = darcy_friction_factor(re, rel_roughness)

    dp_dl = f * (density_kg_m3 * velocity_ms ** 2) / (2.0 * inner_diameter_m)
    return dp_dl, velocity_ms, re


# ── DN lookup helper ────────────────────────────────────────────────────────────

def _lookup_dn(DN: int) -> dict:
    for dn, inner_mm, outer_mm in STANDARD_DN_SERIES:
        if dn == DN:
            return {"DN": dn, "inner_diameter_mm": inner_mm, "outer_diameter_mm": outer_mm}
    raise ValueError(f"DN{DN} not found in STANDARD_DN_SERIES.")


# ── PipeSpec — the result of sizing a pipe ─────────────────────────────────────

@dataclass
class PipeSpec:
    """
    A fully-specified, sized pipe segment — what select_pipe() and
    size_pipe_for_peak() return, and what topology.py will attach to
    graph edges.
    """
    DN: int
    inner_diameter_m: float
    outer_diameter_m: float
    velocity_ms: float
    pressure_gradient_Pa_per_m: float
    reynolds_number: float
    friction_factor: float
    heat_loss_coefficient_W_per_mK: float   # combined supply+return, see above
    cost_GBP_per_m: float
    construction: str
    insulation_series: str
    fluid_temp_C: float
    below_min_velocity: bool   # informational flag, not a hard reject

    def __repr__(self):
        return (
            f"PipeSpec(DN{self.DN}, v={self.velocity_ms:.2f} m/s, "
            f"dp/L={self.pressure_gradient_Pa_per_m:.0f} Pa/m, "
            f"loss={self.heat_loss_coefficient_W_per_mK:.2f} W/m.K, "
            f"£{self.cost_GBP_per_m:.0f}/m)"
        )


# ── Pipe selection — the dual-criterion sizing decision ────────────────────────

def select_pipe(
    flow_m3_s: float,
    fluid_temp_C: float,
    max_velocity_ms: float = 2.5,
    min_velocity_ms: float = 0.3,
    max_pressure_gradient_Pa_per_m: float = 150.0,
    roughness_mm: float = DEFAULT_ROUGHNESS_MM,
    construction: str = "single",
    insulation_series: str = DEFAULT_INSULATION_SERIES,
    dn_series: Optional[list] = None,
) -> PipeSpec:
    """
    Select the smallest standard DN that satisfies BOTH the velocity limit
    and the pressure-gradient limit simultaneously (standard dual-criterion
    sizing method). Returns a fully-specified PipeSpec including heat loss
    and cost for the chosen DN.

    A resulting velocity below min_velocity_ms is flagged via
    below_min_velocity=True but does NOT block selection — minimum
    velocity matters most at part-load operation (sediment risk over
    time), which is a dispatch-time check, not a design-sizing one. At
    design (peak) conditions it's rare to bind anyway.

    Parameters
    ----------
    flow_m3_s     : design (coincident peak) volumetric flow rate (m3/s)
    fluid_temp_C  : representative fluid temperature, for property lookup
                    (use the supply temp — properties don't vary enough
                    across a typical network ΔT to matter for sizing)
    dn_series     : override the standard DN table (defaults to
                    STANDARD_DN_SERIES) — this is the hook for swapping in
                    a manufacturer-specific catalogue later
    """
    props = water_properties(fluid_temp_C)
    series = dn_series if dn_series is not None else STANDARD_DN_SERIES

    for DN, inner_mm, outer_mm in series:
        inner_m = inner_mm / 1000.0
        dp_dl, velocity, re = pressure_gradient_Pa_per_m(
            flow_m3_s, inner_m, props["density_kg_m3"], props["viscosity_Pa_s"], roughness_mm,
        )

        if velocity <= max_velocity_ms and dp_dl <= max_pressure_gradient_Pa_per_m:
            rel_roughness = (roughness_mm / 1000.0) / inner_m
            f = darcy_friction_factor(re, rel_roughness)
            return PipeSpec(
                DN=DN,
                inner_diameter_m=inner_m,
                outer_diameter_m=outer_mm / 1000.0,
                velocity_ms=velocity,
                pressure_gradient_Pa_per_m=dp_dl,
                reynolds_number=re,
                friction_factor=f,
                heat_loss_coefficient_W_per_mK=heat_loss_coefficient_W_per_mK(
                    DN, construction, insulation_series
                ),
                cost_GBP_per_m=estimate_pipe_cost_GBP_per_m(DN, construction),
                construction=construction,
                insulation_series=insulation_series,
                fluid_temp_C=fluid_temp_C,
                below_min_velocity=velocity < min_velocity_ms,
            )

    raise ValueError(
        f"No standard DN satisfies both criteria for flow={flow_m3_s:.4f} m3/s "
        f"at {fluid_temp_C}°C (max_velocity={max_velocity_ms} m/s, "
        f"max_pressure_gradient={max_pressure_gradient_Pa_per_m} Pa/m). "
        f"Largest available is DN{series[-1][0]} — consider a parallel pipe "
        f"or extending the DN series."
    )


def size_pipe_for_peak(
    peak_heat_kW: float,
    flow_temp_C: float,
    return_temp_C: float,
    **kwargs,
) -> PipeSpec:
    """
    The main entry point topology.py / hydraulics.py will call per segment:
    convert a peak heat duty (kW) directly to a sized pipe, handling the
    kW -> mass flow -> volumetric flow conversion internally with
    temperature-correct water properties.

    Q = m_dot * cp * delta_T  =>  m_dot = Q / (cp * delta_T)

    Works identically for a hot loop (flow_temp_C > return_temp_C, heat
    flowing OUT to the network) or a cold loop (flow_temp_C < return_temp_C,
    heat flowing IN from the network) — delta_T is taken as an absolute
    value, since the pipe doesn't care about sign, only magnitude of flow.

    Parameters
    ----------
    peak_heat_kW    : coincident peak heat (or cooling) duty for this
                      segment (kW) — see network.py for how this is derived
                      from downstream demand profiles
    flow_temp_C      : design supply temperature (°C)
    return_temp_C     : design return temperature (°C)
    **kwargs         : passed through to select_pipe() (max_velocity_ms,
                       construction, insulation_series, etc.)
    """
    delta_T_K = abs(flow_temp_C - return_temp_C)
    if delta_T_K < 0.1:
        raise ValueError(
            f"flow_temp_C and return_temp_C are within 0.1K of each other "
            f"({flow_temp_C} vs {return_temp_C}) — delta_T too small to size "
            f"a sensible flow rate. Check your design temperatures."
        )

    # Use the supply temperature for fluid properties (see select_pipe docstring)
    props = water_properties(flow_temp_C)
    mass_flow_kg_s = (peak_heat_kW * 1000.0) / (props["cp_J_kgK"] * delta_T_K)
    volumetric_flow_m3_s = mass_flow_kg_s / props["density_kg_m3"]

    return select_pipe(volumetric_flow_m3_s, flow_temp_C, **kwargs)


# ── Self-test ──────────────────────────────────────────────────────────────────

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

    # --- Heat loss coefficients: series + construction comparison ---
    print("\n  Heat loss coefficient by series and construction (DN100, combined supply+return):")
    for series_key in INSULATION_SERIES_PRESETS:
        u_single = heat_loss_coefficient_W_per_mK(100, "single", series_key)
        u_twin = heat_loss_coefficient_W_per_mK(100, "twin", series_key)
        print(f"    {series_key:<10} single={u_single:.2f} W/m.K   twin={u_twin:.2f} W/m.K  "
              f"(twin reduction: {(1 - u_twin/u_single)*100:.0f}%)")

    # --- Cost curve sanity check ---
    print("\n  Cost curve — cost per metre rising sub-linearly with DN:")
    for dn, _, _ in STANDARD_DN_SERIES:
        c_single = estimate_pipe_cost_GBP_per_m(dn, "single")
        c_twin = estimate_pipe_cost_GBP_per_m(dn, "twin")
        print(f"    DN{dn:<5} single=£{c_single:>7.0f}/m   twin=£{c_twin:>7.0f}/m")

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
    hot_network = size_pipe_for_peak(peak_heat_kW=7200, flow_temp_C=65.0, return_temp_C=35.0)
    cold_network = size_pipe_for_peak(peak_heat_kW=7200, flow_temp_C=6.0, return_temp_C=12.0)
    print(f"    Hot  (65/35°C, dT=30K):  {hot_network}")
    print(f"    Cold (6/12°C,  dT=6K):   {cold_network}")
    print(f"    -> cold loop needs a LARGER pipe for the same kW: 5x smaller delta-T")
    print(f"       forces ~5x the mass flow, only partly offset by water being denser when cold.")

    # --- Twin pipe vs single, on the same duty ---
    print("\n  Twin vs single construction, same duty (65/35°C, 5 MW):")
    single = size_pipe_for_peak(5000, 65.0, 35.0, construction="single")
    twin = size_pipe_for_peak(5000, 65.0, 35.0, construction="twin")
    print(f"    Single: {single}")
    print(f"    Twin:   {twin}")

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
    u_s2_single = heat_loss_coefficient_W_per_mK(100, "single", "series2")
    u_s2_twin = heat_loss_coefficient_W_per_mK(100, "twin", "series2")
    assert u_s2_twin < u_s2_single, "Twin pipe should lose less heat than single construction"
    c_dn25 = estimate_pipe_cost_GBP_per_m(25)
    c_dn300 = estimate_pipe_cost_GBP_per_m(300)
    assert c_dn300 > c_dn25, "Larger DN should cost more per metre"
    assert single.cost_GBP_per_m < twin.cost_GBP_per_m, "Twin construction should cost more than single"
    for dn, _, _ in STANDARD_DN_SERIES:
        pipe = _lookup_dn(dn)
        assert pipe["outer_diameter_mm"] > pipe["inner_diameter_mm"], "Outer diameter must exceed inner diameter"
    print("  ✓ Cold water is more viscous than hot water, by roughly the expected ratio")
    print("  ✓ Hot vs cold loops at identical volumetric flow show a measurable hydraulic difference")
    print("  ✓ Cold network (small delta-T) needs an equal-or-larger pipe than hot network for same kW")
    print("  ✓ Twin-pipe construction loses less heat and costs more than single construction")
    print("  ✓ Cost curve rises with DN; DN table internally consistent")
    print("  ✓ Oversized flow correctly raises an informative error")
    print()