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
£/m installed cost is dominated by trenching, not pipe material. Published
figures on pipework's share of total DH investment cost vary by source —
the Community Heat Development Unit and the Irish National Heat Study
(SEAI/Element Energy, 2023) both put it at roughly a third of total scheme
cost, while some sources cite up to 40-60% depending on network density
and ground conditions. Treat "pipework is a major, often dominant, CAPEX
line item" as the defensible claim — not a single precise percentage.

Modelled as a power-law cost curve, FITTED to real published data (SEAI
National Heat Study Appendix B, Table 4 — see PIPE_COST_GBP_PER_M_AT_REF_DN
below for the full calibration note), not assumed. The SEAI study
independently observed the same qualitative shape in their own raw data:
cost rises more slowly with diameter than people expect, because pipe
LENGTH (driven by network layout, not diameter) is usually the more
significant cost driver — consistent with, and now the actual basis for,
this curve. Treat it as a sensitivity input, not a quoted price; get a
real trenching quote before using this for investment decisions.

DN range and construction limits — grounded in real manufacturer data
---------------------------------------------------------------------------
The standard DN series goes up to DN600, matching Logstor's published
Design Manual range (valid under EN 13941 up to DN600 — beyond that,
larger products exist but move into bespoke/contact-the-manufacturer
territory, which is exactly where this screening tool should hand off to
a real quote rather than silently extrapolate).

TWIN construction is only offered up to DN200 (219.1mm outer diameter) —
this matches Logstor's actual published TwinPipe product range (26.9mm to
219.1mm OD). There is no real commercial twin-pipe product at larger
diameters: past a certain size, housing two large bores in one casing
stops being a practical product, so manufacturers only sell single
construction for trunk-main-scale pipes. Requesting twin construction
above DN200 raises a clear error rather than silently returning a
plausible-looking number for a product that doesn't actually exist.

Usage
-----
    from network.pipe_catalog import size_pipe_for_peak, water_properties

    # Gen 3 hot loop: 7.2 MW peak (Ealing Phase 1 figure), 70/40 deg C
    # (real network's peak design point -- it's actually variable
    # temperature, 65-70C flow seasonally, per the Ealing report; this
    # fixed-temperature model uses the peak figure throughout)
    pipe = size_pipe_for_peak(
        peak_heat_kW=7200, flow_temp_C=70.0, return_temp_C=40.0,
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
    # DN600 — added to match Logstor's published Design Manual range
    # (valid up to and including DN600 under EN 13941). Outer diameter
    # 610mm is the standard DIN/ISO value for this size; inner diameter
    # is an approximation following the same wall-thickness fraction as
    # the DN400/DN500 rows above (~4%), since an exact manufacturer
    # datasheet figure wasn't to hand. Verify against a current Logstor/
    # Isoplus/Brugg DN600 datasheet before using this row for anything
    # beyond feasibility-stage screening.
    (600, 585.0, 610.0),
]

# TwinPipe construction is only a real commercial product up to DN200
# (219.1mm outer diameter) — matching Logstor's published TwinPipe range
# (26.9mm to 219.1mm OD). There is no real twin-pipe product at larger
# diameters; past this size, housing two large bores in one casing stops
# being practical, so manufacturers only offer single construction for
# trunk-main-scale pipes. See module docstring.
TWIN_PIPE_MAX_DN = 200


# ── Insulation profile — single EN 253 standard, not three invented "series" ───
#
# EN 253 is a CONSTRUCTION STANDARD (steel service pipe + PUR foam +
# HDPE casing), not a multi-grade classification — it defines one
# insulation profile, not "series 1/2/3" tiers. "Series 1/2/3" is a
# manufacturer (e.g. Logstor) PRODUCT-LINE naming convention, and no
# independent series-specific dimension data exists in this project.
# Carrying three fabricated "series" (one real + two scaled guesses) was
# solving a problem this feasibility-stage model doesn't actually have.
# So: ONE real, DN-dependent casing profile, taken directly from Table 7,
# "District Heating Manual for London" (Recommended dimensions for
# casing pipe outside diameter and wall thickness for bonded steel
# service pipe system according to EN 253, p.39). No series, no scaling.
#
# If a genuine "what if we spec thicker insulation on this route"
# sensitivity case is wanted later, that's a real, separate question —
# reintroduce a scale factor deliberately for that purpose then, with
# its own justification, rather than carrying unused machinery now.

# (DN, steel_pipe_OD_mm, casing_pipe_OD_mm) — covers DN20 to DN500.
# DN600 is NOT in this table (the manual's Table 7 stops at DN500); see
# casing_to_pipe_ratio_at_dn() for how that's handled.
_TABLE7_DN_CASING_DATA = [
    (20,  26.9,  90.0),
    (25,  33.7,  90.0),
    (32,  42.4,  110.0),
    (40,  48.3,  110.0),
    (50,  60.3,  125.0),
    (65,  76.1,  140.0),
    (80,  88.9,  160.0),
    (100, 114.3, 200.0),
    (125, 139.7, 225.0),
    (150, 168.3, 250.0),
    (200, 219.1, 315.0),
    (250, 273.0, 400.0),
    (300, 323.9, 450.0),
    (350, 355.6, 500.0),
    (400, 406.4, 520.0),
    (450, 457.0, 560.0),
    (500, 508.0, 630.0),
]
_TABLE7_DN = np.array([r[0] for r in _TABLE7_DN_CASING_DATA], dtype=float)
_TABLE7_RATIO = np.array(
    [r[2] / r[1] for r in _TABLE7_DN_CASING_DATA], dtype=float
)  # casing_OD / pipe_OD at each real DN — falls from 3.35 (DN20) to 1.24 (DN500)

# Power-law fit to the real ratio data above, used ONLY to extrapolate
# beyond DN500 (i.e. for DN600, which is in STANDARD_DN_SERIES but not in
# Table 7). Within DN20-DN500, casing_to_pipe_ratio_at_dn() uses direct
# interpolation against the real points instead (exact at every real DN,
# not just close). Fit: ratio = a * DN^b, log-log least squares.
#   a=6.32, b=-0.271, R^2=0.92 across all 17 real Table 7 points.
# Worst single-point fit error is at DN20 (-16%) — small pipes have a
# fixed minimum practical casing size (90mm, shared by DN20 and DN25
# alike) that a smooth power law can't fully capture; the fit is
# reliable from DN32 upward (typically within ~10%, often under 5%).
_RATIO_FIT_A = 6.3191
_RATIO_FIT_B = -0.2709


def casing_to_pipe_ratio_at_dn(DN: int) -> float:
    """
    Real, DN-dependent casing-to-pipe ratio, from Table 7 of the District
    Heating Manual for London (EN 253 basis).

    For DN20-DN500 (covered by the real table): linear interpolation
    between the actual measured points — exact at every standard DN that
    has a real data point, not just close.

    For DN600 (one row beyond the table's DN500 ceiling): extrapolated
    via the power-law fit to the same real data (a*DN^b, R^2=0.92). This
    is a genuine extrapolation, not a measured value — flagged here so
    it isn't mistaken for Table 7 data it isn't.

    Insulation gets proportionally THINNER relative to pipe size as DN
    increases (3.35x at DN20 down to 1.24x at DN500) — a real effect, not
    a modelling artifact.
    """
    if DN <= _TABLE7_DN.max():
        return float(np.interp(DN, _TABLE7_DN, _TABLE7_RATIO))
    else:
        # Extrapolation beyond the real table (currently only DN600)
        return float(_RATIO_FIT_A * DN ** _RATIO_FIT_B)


# Single insulation profile — EN 253 only. Kept as a dict (rather than a
# bare constant) purely so heat_loss_coefficient_W_per_mK()'s
# insulation_series= parameter still has somewhere to look up
# insulation_k_W_mK without changing its call signature.
DEFAULT_INSULATION_SERIES = "en253"
INSULATION_SERIES_PRESETS = {
    "en253": {
        "description":      "EN 253 standard pre-insulated pipe (PUR foam, HDPE casing) — "
                             "Table 7, District Heating Manual for London",
        "insulation_k_W_mK": DEFAULT_INSULATION_K_W_MK,
    },
}

# Twin-pipe (flow + return sharing one casing) loses meaningfully less heat
# than two separate single pre-insulated pipes of the same series, at a
# capital cost premium. Empirical approximation from published thermal/
# economic comparisons — not a substitute for a manufacturer twin-pipe
# datasheet at design stage.
TWIN_PIPE_LOSS_REDUCTION_FACTOR = 0.55   # combined twin loss = 0.55 x (2 x single-pipe loss)
TWIN_PIPE_COST_PREMIUM_FACTOR   = 1.15   # +15% capex vs single construction


def _validate_construction(DN: int, construction: str) -> None:
    """
    Shared guard for every function that accepts a construction= argument.
    Raises a clear, loud error rather than silently returning a
    plausible-looking number for a combination that doesn't exist as a
    real product — see TWIN_PIPE_MAX_DN and module docstring.
    """
    if construction not in ("single", "twin"):
        raise ValueError(f"construction must be 'single' or 'twin'; got '{construction}'")
    if construction == "twin" and DN > TWIN_PIPE_MAX_DN:
        raise ValueError(
            f"construction='twin' requested at DN{DN}, but twin-pipe is not a "
            f"real commercial product above DN{TWIN_PIPE_MAX_DN} (matches "
            f"Logstor's published TwinPipe range, 26.9-219.1mm OD). Use "
            f"construction='single' for DN{DN} — large trunk mains are single "
            f"pipes in practice. See module docstring."
        )


# ── Pipe cost curve ─────────────────────────────────────────────────────────────
# Power-law cost curve (£/m installed, dominated by trenching).
#
# Calibration source and method
# ------------------------------
# Fitted to real published data: SEAI National Heat Study, Appendix B
# "District Heating and Cooling: Spatial Analysis of Infrastructure Costs
# and Potential in Ireland" (Element Energy/Ricardo Energy & Environment
# for SEAI, 2023), Table 4 — 2-pipe, inner-city, 2020 prices, DN20-DN600.
# That table's own source is the Scottish Building Standards Agency
# (2009) "Heating Supply Options for New Development", inflated by SEAI
# to 2020 prices.
#
# Method: EUR->GBP at the report's own stated rate (€1.12 per £1), then a
# simple UK CPI uplift (~24% cumulative, 2020->2026) to bring to present-
# day nominal terms, then a log-log linear regression (power law fit)
# across all 19 DN20-DN600 data points. Fit quality: R^2 = 0.84 — solid,
# but NOT perfect: the real data is close to FLAT from DN20 to DN65
# (suggesting a fixed trenching/mobilisation cost floor for small branch
# pipes that a single smooth power law doesn't fully capture), then rises
# more steeply above that. This curve is most reliable in the DN80-DN600
# range, where this project's actual mains will mostly sit, and likely
# UNDERESTIMATES the smallest branch sizes (DN20-65) slightly.
#
# An earlier version of this module claimed the DN100 anchor point was
# "already used in your scenario YAML" — that claim was checked and found
# to be false; no such YAML exists anywhere in this repository. That
# specific number turned out to be close to what this real, properly
# cited dataset predicts anyway (£1,158/m vs the old £1,200/m), but
# that's a coincidence worth knowing about, not a justification — the
# numbers below are now traceable to an actual source rather than
# inherited from an unverified claim.
PIPE_COST_REFERENCE_DN        = 100
PIPE_COST_GBP_PER_M_AT_REF_DN = 1158.0   # fitted, see calibration note above
PIPE_COST_SCALE_EXPONENT      = 0.426    # fitted; NOT the chemical-engineering
                                          # "six-tenths rule" (Sinnott, 2005) —
                                          # that rule relates cost to CAPACITY
                                          # (cost ~ capacity^0.6), and capacity
                                          # scales as DN^2 for a fixed velocity,
                                          # so naively applying it to diameter
                                          # directly would predict cost ~ DN^1.2
                                          # (rising FASTER than linear) — the
                                          # opposite of what real pipe-cost data
                                          # actually shows. Buried-pipe cost is
                                          # dominated by trenching/civils
                                          # overhead that's largely independent
                                          # of diameter, not by the surface-area
                                          # economics the six-tenths rule
                                          # captures for fabricated equipment —
                                          # which is why the real exponent here
                                          # (0.43) is flatter than 0.6, not
                                          # steeper. See module docstring.


def estimate_pipe_cost_GBP_per_m(
    DN: int,
    construction: str = "single",
) -> float:
    """
    Estimate installed cost (£/m) for a given DN, via the power-law cost
    curve. construction='twin' applies the twin-pipe cost premium.
    """
    if DN <= 0:
        raise ValueError(f"DN must be positive; got {DN}.")
    _validate_construction(DN, construction)

    cost = PIPE_COST_GBP_PER_M_AT_REF_DN * (
        DN / PIPE_COST_REFERENCE_DN
    ) ** PIPE_COST_SCALE_EXPONENT

    if construction == "twin":
        cost *= TWIN_PIPE_COST_PREMIUM_FACTOR

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

    Casing diameter is taken from the real, DN-dependent EN 253 Table 7
    data (District Heating Manual for London) — see
    casing_to_pipe_ratio_at_dn(). There is currently only one insulation
    profile (EN 253); see module note above casing_to_pipe_ratio_at_dn()
    for why "series 1/2/3" was removed rather than kept as guessed
    variants of a standard that doesn't define them.

    Parameters
    ----------
    DN                 : nominal pipe size (must be in STANDARD_DN_SERIES)
    construction        : 'single' (two separate pre-insulated pipes) or
                          'twin' (shared casing — lower combined loss)
    insulation_series   : currently only "en253" is valid (kept as a
                          parameter for interface stability / future use,
                          not because alternatives exist yet)
    """
    pipe = _lookup_dn(DN)
    if insulation_series not in INSULATION_SERIES_PRESETS:
        raise ValueError(
            f"Unknown insulation_series '{insulation_series}'. Only "
            f"{list(INSULATION_SERIES_PRESETS.keys())} is currently "
            f"defined — EN 253 doesn't specify multiple insulation grades; "
            f"see module note above casing_to_pipe_ratio_at_dn()."
        )
    series = INSULATION_SERIES_PRESETS[insulation_series]
    _validate_construction(DN, construction)

    D_pipe_outer_m = pipe["outer_diameter_mm"] / 1000.0
    D_casing_m = D_pipe_outer_m * casing_to_pipe_ratio_at_dn(DN)
    if D_casing_m <= D_pipe_outer_m:
        raise ValueError(
            f"Computed casing diameter <= pipe outer diameter for DN{DN} "
            f"— check casing_to_pipe_ratio_at_dn(); this would otherwise "
            f"silently produce a nonsensical (negative or infinite) "
            f"heat-loss coefficient."
        )
    k_ins = series["insulation_k_W_mK"]

    R_per_m = np.log(D_casing_m / D_pipe_outer_m) / (2 * np.pi * k_ins)
    U_single_per_m = 1.0 / R_per_m   # W/m.K, ONE pipe

    if construction == "single":
        # Two separate pipes (supply + return) -> simple sum
        return 2.0 * U_single_per_m
    else:
        # construction == "twin" — already validated above. Shared casing
        # -> meaningfully lower combined loss than 2x single
        return 2.0 * U_single_per_m * TWIN_PIPE_LOSS_REDUCTION_FACTOR


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


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_pipe_catalog.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_pipe_catalog.py\n"
    )
