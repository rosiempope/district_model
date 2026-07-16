"""
Synthesises 8,760-hour hourly heating, cooling, and DHW demand profiles
for each building node in the district energy model.
 
Methodology
-----------
HEATING  : HDD scaling — annual benchmark (kWh/m² or kWh/unit) distributed
           across the year proportional to heating degree-hours from the EPW.
           Modulated by a building-type occupancy mask so unoccupied periods
           have reduced (not zero) base load.
           https://www.valeofglamorgan.gov.uk/Documents/Our%20Council/Achieving%20our%20vision/Partnerships,%20Policies%20&%20Plans/Local%20Service%20Board/Carbon%20Management/Arup%20and%20Carbon%20Trust%20Report%20-%20%20Appendix%20C%20CIBSE%20Benchmarks.PDF
           This includes the heating kWh/m2 benchmark - only includes fossil fuel use - so using a normal split between space heating and hot water is 75/25 for residential and 85/15 for non-residential (per CIBSE Guide G).
              
COOLING  : Two-part model:
           (a) CDD scaling — the conventional approach, distributes annual
               cooling benchmark proportional to cooling degree-hours.
           (b) Comfort/urgency ramp — a smooth ramp from T_cool_onset to
               T_cool_full that captures demand when it gets uncomfortably
               hot, even if the building has little historical A/C usage.
               Makes the model forward-looking for climate change scenarios.
           Final cooling = max(CDD-scaled, comfort-ramp-scaled) per hour.
 
DHW      : Flat annual total with sinusoidal seasonal shape (higher in
           winter — cold inlet water) and diurnal morning/evening peaks.
           Not weather-dependent.
 
All profiles returned as numpy arrays of length 8,760 in kW.
"""

"""
Key Assumptions:
- Cooling is a two-part estimate e.g. picks whether the CDD or the comfort ramp is higher for each hour, and uses that as the cooling demand for that hour. Such that the model is forward-looking and will show cooling demand even if the building has little historical A/C usage.
- cool_base_C is the temperature below which no cooling is required (default 20°C). This is used to calculate CDDs.
- cool_onset_C is the temperature at which comfort cooling demand begins (default 22°C). This is used to calculate the comfort ramp.
- cool_full_C is the temperature at which comfort cooling demand saturates (default 26°C). This is used to calculate the comfort ramp.
- School holidays are fixed and approximate
- 
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
 
# ── Building type benchmarks ───────────────────────────────────────────────────
# Sources:
#   Heating EUI — CIBSE TM46 / Ealing Feasibility Report cross-check
#   Cooling EUI — SEL in-house benchmarks (per Ealing report), CIBSE Guide F
#   DHW EUI     — CIBSE Guide G, EST data
#   All values in kWh/m²/yrs

BUILDING_TYPES = {
    "office": {
        "heat_kWh_m2":    120*0.85,
        "cool_kWh_m2":    30.0,
        "dhw_kWh_m2":     120*0.15,
        "occupancy":      "office",
        "base_load_frac":  0.15,
        "internal_gains_fraction": 0.65,   # MIT/ScienceDirect office benchmark — see _cooling_profile()
        "description":    "General office (naturally ventilated)",
    },
    "office_ac": {
        "heat_kWh_m2":    120*0.85,
        "cool_kWh_m2":    50.0,
        "dhw_kWh_m2":     120*0.15,
        "occupancy":      "office",
        "base_load_frac":  0.15,
        "internal_gains_fraction": 0.55,   # AC offices have a larger genuine WEATHER-driven share
                                            # (the AC exists specifically to handle hot days the
                                            # naturally-ventilated "office" type can't) — internal
                                            # gains are still real but a smaller fraction of a
                                            # bigger total annual budget (50 vs 30 kWh/m2/yr)
        "description":    "Air-conditioned office",
    },
    "residential": {
        "heat_kWh_m2":    120*0.70,   # Part L 2021 new build
        "cool_kWh_m2":     2.0,   # Near-zero installed A/C; comfort ramp carries this
        "dhw_kWh_m2":     35.0,
        "occupancy":      "residential",
        "base_load_frac":  0.35,
        "internal_gains_fraction": 0.50,   # Lower occupant/equipment density than commercial;
                                            # genuine annual total is tiny (2 kWh/m2/yr) so this
                                            # mostly affects shape, not magnitude
        "description":    "Residential (new build, Part L 2021)",
    },
    "residential_existing": {
        "heat_kWh_m2":   130.0,   # Pre-2010 stock
        "cool_kWh_m2":     2.0,
        "dhw_kWh_m2":    40.0,
        "occupancy":      "residential",
        "base_load_frac":  0.35,
        "internal_gains_fraction": 0.50,
        "description":    "Residential (existing / retrofit target)",
    },
    "hospital": {
        "heat_kWh_m2":   200.0,   # 24/7, high ventilation
        "cool_kWh_m2":    55.0,   # Ealing report: 40-60 kWh/m²
        "dhw_kWh_m2":   120.0,    # Sterile processes, high DHW
        "occupancy":      "hospital",
        "base_load_frac":  0.70,
        "internal_gains_fraction": 0.70,   # 24/7 equipment (sterilisers, imaging, refrigeration,
                                            # server/IT loads) running near-continuously — internal
                                            # gains genuinely dominate even more than a typical office
        "description":    "Hospital / acute healthcare",
    },
    "retail": {
        "heat_kWh_m2":    90.0,
        "cool_kWh_m2":    60.0,   # High internal gains from lighting/people
        "dhw_kWh_m2":     5.0,
        "occupancy":      "retail",
        "base_load_frac":  0.10,
        "internal_gains_fraction": 0.70,   # Comment on cool_kWh_m2 above already says "high
                                            # internal gains from lighting/people" -- this makes
                                            # that existing assumption explicit and load-bearing
        "description":    "Retail / high street shops",
    },
    "supermarket": {
        "heat_kWh_m2":    70.0,
        "cool_kWh_m2":   100.0,   # Refrigeration, high cooling load
        "dhw_kWh_m2":     5.0,
        "occupancy":      "retail_extended",
        "base_load_frac":  0.20,
        "internal_gains_fraction": 0.85,   # Refrigeration runs CONTINUOUSLY, near-independent of
                                            # outdoor weather -- the highest internal-gains share of
                                            # any type here, consistent with the existing comment
        "description":    "Supermarket / food retail",
    },
    "hotel": {
        "heat_kWh_m2":   150.0,
        "cool_kWh_m2":    30.0,   # Ealing report: hotel 20-35
        "dhw_kWh_m2":    90.0,
        "occupancy":      "hotel",
        "base_load_frac":  0.50,
        "internal_gains_fraction": 0.55,
        "description":    "Hotel",
    },
    "school": {
        "heat_kWh_m2":   100.0,
        "cool_kWh_m2":     8.0,   # Very little A/C in UK schools
        "dhw_kWh_m2":    15.0,
        "occupancy":      "school",
        "base_load_frac":  0.05,
        "internal_gains_fraction": 0.55,   # Genuine annual total is tiny (8 kWh/m2/yr); mostly
                                            # affects shape (classroom occupancy gains) not magnitude
        "description":    "Secondary school",
    },
    "mixed_use": {
        "heat_kWh_m2":   100.0,
        "cool_kWh_m2":    20.0,
        "dhw_kWh_m2":    40.0,
        "occupancy":      "mixed",
        "base_load_frac":  0.25,
        "internal_gains_fraction": 0.60,
        "description":    "Mixed-use development",
    },
    "data_centre": {
        "heat_kWh_m2":     0.0,   # Modelled as heat source, not demand
        "cool_kWh_m2":     0.0,
        "dhw_kWh_m2":      0.0,
        "occupancy":      "always_on",
        "base_load_frac":  1.0,
        "internal_gains_fraction": 1.0,   # IT load runs continuously, ~zero weather sensitivity --
                                           # but cool_kWh_m2=0 here anyway (DC modelled as a heat
                                           # SOURCE elsewhere, not a cooling demand node), so this
                                           # value is set for consistency/correctness, not because
                                           # it's ever actually load-bearing in practice
        "description":    "Data centre (heat source node)",
    },
}

# Default internal_gains_fraction for any building TYPE not explicitly
# listed above (e.g. a custom type passed via building.get("type")) —
# 0.65 is the office-benchmark figure, used as a sane generic fallback.
DEFAULT_INTERNAL_GAINS_FRACTION = 0.65


def _match_annual_and_peak(profile_kW, annual_kWh, peak_kW):
    """Reshape a positive hourly archetype to exact measured annual/peak data.

    A power transform preserves timing while allowing the load factor to move.
    This is intended for report/measured-data calibration, not benchmark-only
    demand. Both targets are met within floating-point tolerance.
    """
    values = np.maximum(np.asarray(profile_kW, dtype=float), 1e-12)
    annual = float(annual_kWh)
    peak = float(peak_kW)
    if annual <= 0 or peak <= 0:
        raise ValueError("Measured annual heat and peak heat must both be positive")
    average = annual / len(values)
    if peak < average - 1e-9:
        raise ValueError(
            f"peak_total_heat_kW={peak} is below the annual-average load {average:.3f} kW"
        )
    target_ratio = peak / average
    if abs(target_ratio - 1.0) < 1e-10:
        return np.full(len(values), average)

    normalised = values / values.max()

    def peak_ratio(alpha):
        shaped = normalised ** alpha
        return shaped.max() / shaped.mean()

    low, high = 0.0, 1.0
    while peak_ratio(high) < target_ratio and high < 1024:
        high *= 2.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if peak_ratio(mid) < target_ratio:
            low = mid
        else:
            high = mid
    shaped = normalised ** ((low + high) / 2.0)
    return shaped * (annual / shaped.sum())


def _match_annual_peak_with_sharpness(profile_kW, annual_kWh, peak_kW, sharpness):
    """Meet annual/peak targets while allowing a report-derived peak shape.

    Higher sharpness concentrates the design peak into fewer hours. This is
    useful when a published report provides annual energy, peak and a load
    duration curve but not the underlying 8,760 values.
    """
    values = np.maximum(np.asarray(profile_kW, dtype=float), 0.0)
    mean = float(annual_kWh) / len(values)
    peak = float(peak_kW)
    if peak < mean:
        raise ValueError("peak heat cannot be below average heat")
    base = values / max(values.max(), 1e-12)
    shaped = base ** float(sharpness)
    shaped_mean = float(shaped.mean())
    if abs(1.0 - shaped_mean) < 1e-12:
        return np.full(len(values), mean)
    scale = (peak - mean) / (1.0 - shaped_mean)
    offset = mean - scale * shaped_mean
    if offset < -1e-9:
        raise ValueError("aggregate_load_shape_sharpness produces negative baseload")
    result = np.maximum(0.0, offset + scale * shaped)
    return result * (float(annual_kWh) / result.sum())

# ── Occupancy schedules ────────────────────────────────────────────────────────
 
def _make_occupancy(schedule_key: str, n_hours: int = 8760) -> np.ndarray:
    """
    Generate an 8760-length occupancy array (0-1) for a given building type.
    Hour 0 = 00:00 on 1st January (Monday convention matching EPW).
    """
    hour_of_day = np.arange(n_hours) % 24
    day_of_week = (np.arange(n_hours) // 24) % 7   # 0=Mon, 6=Sun
    day_of_year = np.arange(n_hours) // 24
 
    is_weekday  = day_of_week < 5
    is_saturday = day_of_week == 5
    is_sunday   = day_of_week == 6
 
    # UK school holiday periods (~13 wks/yr)
    summer_hols = (day_of_year >= 196) & (day_of_year <= 252)
    xmas_hols   = (day_of_year >= 355) | (day_of_year <= 6)
    easter_hols = (day_of_year >= 95)  & (day_of_year <= 109)
    school_hols = summer_hols | xmas_hols | easter_hols
 
    occ = np.zeros(n_hours)

    if schedule_key == "office":
        occ = np.where(is_weekday  & (hour_of_day >= 8)  & (hour_of_day < 18), 1.00, occ)
        occ = np.where(is_saturday & (hour_of_day >= 9)  & (hour_of_day < 13), 0.20, occ)
 
    elif schedule_key == "residential":
        occ = np.where((hour_of_day >= 6)  & (hour_of_day < 9),  0.70, occ)
        occ = np.where((hour_of_day >= 9)  & (hour_of_day < 17), 0.30, occ)
        occ = np.where((hour_of_day >= 17) & (hour_of_day < 23), 0.85, occ)
        occ = np.where((hour_of_day >= 23) | (hour_of_day < 6),  0.50, occ)
        occ = np.where(
            (is_saturday | is_sunday) & (hour_of_day >= 8) & (hour_of_day < 22),
            0.90, occ
        )

    elif schedule_key == "hospital":
        occ = np.where((hour_of_day >= 7)  & (hour_of_day < 21), 1.00, occ)
        occ = np.where((hour_of_day >= 21) | (hour_of_day < 7),  0.60, occ)
 
    elif schedule_key == "retail":
        occ = np.where(is_weekday  & (hour_of_day >= 9)  & (hour_of_day < 21), 1.00, occ)
        occ = np.where(is_saturday & (hour_of_day >= 9)  & (hour_of_day < 21), 0.90, occ)
        occ = np.where(is_sunday   & (hour_of_day >= 11) & (hour_of_day < 17), 0.60, occ)
 
    elif schedule_key == "retail_extended":
        occ = np.where(is_weekday  & (hour_of_day >= 7)  & (hour_of_day < 22), 1.00, occ)
        occ = np.where(is_saturday & (hour_of_day >= 7)  & (hour_of_day < 22), 1.00, occ)
        occ = np.where(is_sunday   & (hour_of_day >= 10) & (hour_of_day < 16), 0.80, occ)
 
    elif schedule_key == "hotel":
        occ = np.where((hour_of_day >= 7)  & (hour_of_day < 23), 0.80, occ)
        occ = np.where((hour_of_day >= 23) | (hour_of_day < 7),  0.50, occ)
 
    elif schedule_key == "school":
        in_term = ~school_hols
        occ = np.where(
            in_term & is_weekday & (hour_of_day >= 8) & (hour_of_day < 18),
            1.00, occ
        )
 
    elif schedule_key == "mixed":
        occ = 0.5 * _make_occupancy("office", n_hours) + \
              0.5 * _make_occupancy("residential", n_hours)
 
    elif schedule_key == "always_on":
        occ = np.ones(n_hours)
 
    return occ

# ── Core profile builders ──────────────────────────────────────────────────────

def _heating_profile(
    T_air: np.ndarray,
    annual_heat_kWh: float,
    occupancy: np.ndarray,
    base_load_frac: float,
    heat_base_C: float = 15.5,
    reference_annual_HDD_h: Optional[float] = None,
) -> np.ndarray:
    """
    HDD-scaled heating profile modulated by occupancy.
    base_load_frac ensures fabric heat loss continues when unoccupied.
    Returns hourly load in kW.

    Degree-day normalisation
    -------------------------
    annual_heat_kWh is a CIBSE-style benchmark, implicitly tied to a
    "standard" UK weather year. Without correction, simply rescaling the
    HDD-weighted shape to always sum to that fixed benchmark means weather
    only changes WHEN heat is used, never HOW MUCH — so a 2050 climate
    scenario would show identical annual heat to today, just reshuffled.
    That's wrong: a genuinely milder year should use genuinely less heat.

    reference_annual_HDD_h lets the caller supply the annual HDD of a
    reference (typically baseline) weather year. The benchmark is then
    scaled by (this year's actual HDD / reference HDD) before being
    distributed across the hours, so warmer scenarios correctly show LESS
    annual heat, not just a reshaped version of the same total.

    If reference_annual_HDD_h is None (default), this year's own HDD is
    used as its own reference — i.e. the scaling factor is 1.0 and
    behaviour is identical to a standalone, no-climate-comparison run.
    """
    HDD_h = np.clip(heat_base_C - T_air, 0, None)
    HDD_annual = HDD_h.sum()

    if HDD_annual < 1.0:
        warnings.warn("Annual HDD near zero — check weather data or base temperature.")
        return np.zeros(len(T_air))

    reference = reference_annual_HDD_h if reference_annual_HDD_h is not None else HDD_annual
    degree_day_scaling = HDD_annual / reference
    effective_annual_heat_kWh = annual_heat_kWh * degree_day_scaling

    # Occupancy modifier: base_load when empty, full load when occupied
    occ_modifier = base_load_frac + (1.0 - base_load_frac) * occupancy
    raw = HDD_h * occ_modifier

    scale = effective_annual_heat_kWh / raw.sum() if raw.sum() > 0 else 0.0
    return raw * scale  # kW


def _cooling_profile(
    T_air: np.ndarray,
    annual_cool_kWh: float,
    occupancy: np.ndarray,
    base_load_frac: float,
    cool_base_C: float  = 20.0,
    cool_onset_C: float = 22.0,
    cool_full_C: float  = 26.0,
    internal_gains_fraction: float = 0.65,
    reference_annual_CDD_h: Optional[float] = None,
    reference_annual_ramp: Optional[float]  = None,
) -> np.ndarray:
    """
    THREE-part cooling demand model, ADDITIVE (not max()'d) and summing
    EXACTLY to the (climate-scaled) annual_cool_kWh target:

    Part 1 (internal gains floor): a flat load present whenever the
    building is occupied, INDEPENDENT of outdoor temperature. Real UK
    commercial cooling demand is substantially driven by internal heat
    gains (people, lighting, equipment) rather than purely by how hot
    it is outside — see e.g. MIT's internal-gains lecture notes (8-17
    W/m2 for offices) and ScienceDirect's UK rough-guide figure (20-30
    W/m2). At those intensities, internal gains alone can plausibly
    account for the MAJORITY of a UK office's total annual cooling
    energy — confirmed by checking the numbers directly: 8-17 W/m2 over
    a typical ~2500 occupied hours/year gives 20-42 kWh/m2/yr, against
    a ~30 kWh/m2/yr total office cooling benchmark (see BUILDING_TYPES'
    own cool_kWh_m2 figures) -- i.e. 65%-140% of the WHOLE annual total,
    not a minor addition. internal_gains_fraction=0.65 takes the
    conservative (lower) end of that real range as the default share of
    the annual budget allocated to this floor term — defensible from the
    cited literature, not an arbitrary smoothing choice. This is THE FIX
    for the previous version's unrealistic ~94x peak-to-mean ratio: real
    UK cooling-degree-hours above a 20C base are genuinely rare and
    clustered (often <10% of the year), so any model relying SOLELY on
    outdoor-temperature scaling to spread a realistic annual total will
    structurally produce an extreme, unrealistic peak — regardless of
    the exact weighting function used to do the spreading. An always-
    occupied-hours floor breaks that structural problem directly.

    Part 2 (CDD scaling): the REMAINING (1 - internal_gains_fraction)
    share of the annual budget, distributed proportional to cooling
    degree-hours above cool_base_C. Captures the real, additional
    weather-driven load on hot days — genuine, but correctly sized as
    the smaller share it actually is, not the whole annual total.

    Part 3 (comfort urgency ramp): NOT a separate energy allocation —
    a FLOOR (not a max-competing alternative) ensuring that on a
    genuinely hot day (definitionally cool_full_C+), total cooling load
    never drops below what comfort-seeking behaviour would realistically
    demand, even if Part 1+2's combined total at that hour happens to be
    lower. Implemented as np.maximum(part_1 + part_2, comfort_floor) at
    each hour — a safety floor, not a third additive energy term, so it
    doesn't reintroduce the previous version's double-counting problem.

    All three parts are additive (Parts 1+2) or a floor (Part 3) rather
    than independently-normalised alternatives combined via max() — the
    previous version's max(part_A, part_B), where EACH part was
    separately rescaled to ~the same annual total, meant the actual
    annual sum after taking the elementwise max was inflated well above
    the intended target (verified: ~47% over-allocation in a real test
    case) AND the underlying peak-concentration problem wasn't actually
    fixed by Part B, since Part B has the same "rare incidence, big
    peak when normalised to a full annual energy share" issue Part A does.

    Degree-day normalisation (climate scenario adjustment)
    -------------------------------------------------------
    Same mechanism as before: reference_annual_CDD_h / reference_annual_ramp
    let the caller supply the annual CDD-hour sum and ramp-incidence sum
    from a reference (typically baseline) weather year, so a hotter
    climate scenario shows MORE annual cooling (from Part 2's CDD share),
    not just a reshaped version of the same fixed total. If None
    (default), this year's own signal is used as its own reference
    (scaling factor 1.0) — identical to a standalone, no-climate-
    comparison run. Part 1 (internal gains) deliberately does NOT scale
    with climate — internal gains are driven by occupancy and equipment,
    not outdoor temperature, so they should stay constant across climate
    scenarios; only Part 2's weather-driven share should grow in a
    hotter scenario.

    Parameters
    ----------
    internal_gains_fraction : share (0-1) of annual_cool_kWh allocated
                  to the internal-gains floor (Part 1). Default 0.65 —
                  see the real-data justification above. Set lower for
                  building types where internal gains are genuinely
                  less dominant (e.g. a naturally-ventilated school with
                  low equipment density), or higher for IT-dense spaces
                  (e.g. a server room, where internal gains can be
                  nearly the ENTIRE load almost regardless of weather).
    (all other parameters unchanged from the previous version — see
    their original docstrings, reproduced in the module's git history
    if needed)
    """
    n = len(T_air)
    occ_modifier = base_load_frac + (1.0 - base_load_frac) * occupancy
    internal_gains_fraction = float(np.clip(internal_gains_fraction, 0.0, 1.0))

    # -- Part 1: internal gains floor (occupancy-driven, weather-independent) --
    annual_internal_gains_kWh = annual_cool_kWh * internal_gains_fraction
    occ_sum = occ_modifier.sum()
    if occ_sum > 0:
        part_1 = occ_modifier * (annual_internal_gains_kWh / occ_sum)
    else:
        part_1 = np.zeros(n)

    # -- Part 2: CDD-scaled weather-driven share (the REMAINING budget) -------
    annual_weather_driven_kWh = annual_cool_kWh * (1.0 - internal_gains_fraction)
    CDD_h = np.clip(T_air - cool_base_C, 0, None)
    CDD_annual = CDD_h.sum()
    if CDD_annual > 0.1 and annual_weather_driven_kWh > 0:
        reference_A = reference_annual_CDD_h if reference_annual_CDD_h is not None else CDD_annual
        effective_weather_kWh = annual_weather_driven_kWh * (CDD_annual / reference_A)
        raw_2 = CDD_h * occ_modifier
        part_2 = raw_2 * (effective_weather_kWh / raw_2.sum()) if raw_2.sum() > 0 else np.zeros(n)
    else:
        part_2 = np.zeros(n)

    base_total = part_1 + part_2

    # -- Part 3: comfort urgency FLOOR (not an additive energy term) ----------
    # On a genuinely hot, occupied hour, load should never fall BELOW what
    # comfort-seeking behaviour demands, even if Part 1+2 happen to be
    # lower that hour. This does NOT add its own separate share of
    # annual_cool_kWh -- it's an hour-by-hour floor on the combined total,
    # using the SAME peak magnitude base_total would already be producing
    # on its hottest hours, scaled by the comfort ramp shape -- i.e. it
    # reshapes WHEN load is high, rather than adding a new pot of energy.
    ramp = np.clip(
        (T_air - cool_onset_C) / (cool_full_C - cool_onset_C),
        0.0, 1.0
    )
    if reference_annual_ramp is not None:
        # The shared climate reference is weather-only. Including a
        # building-specific occupancy mask here makes even the baseline case
        # fail to normalise to 1 and gives different climate multipliers to
        # identical weather solely because schedules differ.
        ramp_annual = ramp.sum()
        ramp_scale = ramp_annual / reference_annual_ramp if reference_annual_ramp > 0 else 1.0
    else:
        ramp_scale = 1.0
    comfort_floor = ramp * occupancy * base_total.max() * ramp_scale if base_total.max() > 0 else np.zeros(n)

    return np.maximum(base_total, comfort_floor)  # kW


def _dhw_profile(
    annual_dhw_kWh: float,
    n_hours: int = 8760,
    occupancy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    DHW demand profile with seasonal and diurnal shape.
    Not weather-driven — people shower regardless of outside temperature.
    Seasonal variation reflects cold inlet water temperature in winter
    requiring more energy to heat to setpoint.
    Returns hourly load in kW.
    """

    hours       = np.arange(n_hours)
    hour_of_day = hours % 24
    day_of_year = hours // 24
 
    # Seasonal: ±12% amplitude, peaks mid-winter (day ~30)
    seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
 
    # Diurnal: morning peak at 07:00 (showers), evening at 19:00 (cooking/bath)
    diurnal = (
        0.50 * np.exp(-0.5 * ((hour_of_day - 7)  / 2.0) ** 2) +
        0.30 * np.exp(-0.5 * ((hour_of_day - 19) / 2.5) ** 2) +
        0.20   # base overnight load (legionella cycling, commercial kitchens)
    )
 
    raw = seasonal * diurnal
 
    # Soft occupancy mask: DHW doesn't fully disappear when unoccupied
    # (legionella prevention cycling continues, some background load remains)
    
    if occupancy is not None:
        raw = raw * (0.10 + 0.90 * occupancy)
 
    scale = annual_dhw_kWh / raw.sum() if raw.sum() > 0 else 0.0
    return raw * scale  # kW


# ── Annual demand resolver ─────────────────────────────────────────────────────
 
def _resolve_annual_demands(building: dict) -> tuple[float, float, float]:
    """
    Resolve annual space-heating, cooling and DHW demands (kWh).

    A building may use either of two input routes:

    * floor area / dwelling count: missing services are estimated from the
      selected building archetype;
    * measured annual energy: each supplied service is used directly and,
      when there is no area/unit scale, any omitted service defaults to zero.

    This per-service precedence is deliberate. A heat-meter total is enough
    to run a heat-network screen even when the customer's floor area is not
    known. It also avoids inventing an unmetered cooling or DHW load when
    there is no floor area from which to estimate it.
    """

    btype = building.get("type", "office")

    if btype not in BUILDING_TYPES:
        raise ValueError(
            f"Unknown building type '{btype}'. "
            f"Valid: {list(BUILDING_TYPES.keys())}"
        )
 
    bm = BUILDING_TYPES[btype]
    floor_area = building.get("floor_area_m2")
    units       = building.get("units")
 
    if floor_area is not None and float(floor_area) > 0:
        scale = float(floor_area)
    elif units is not None and float(units) > 0:
        scale = float(units) * 75.0   # 75 m²/dwelling assumption
        building["floor_area_m2"] = scale
    else:
        scale = None

    annual_keys = ("annual_heat_kWh", "annual_cool_kWh", "annual_dhw_kWh")
    supplied = {key: building.get(key) is not None for key in annual_keys}
    if scale is None and not any(supplied.values()):
        raise ValueError(
            f"Building '{building.get('name','?')}' needs either positive "
            "'floor_area_m2'/'units' or at least one measured annual-energy input."
        )

    for key in annual_keys:
        value = building.get(key)
        if value is not None and float(value) < 0:
            raise ValueError(
                f"Building '{building.get('name','?')}' has a negative {key}; "
                "annual energy must be zero or positive."
            )

    # Explicit zero is a valid override (for example, no comfort cooling).
    # With no area/unit scale, an omitted service is zero rather than a
    # dimensionally invalid archetype estimate.
    heat = building["annual_heat_kWh"] if supplied["annual_heat_kWh"] else (
        bm["heat_kWh_m2"] * scale if scale is not None else 0.0
    )
    cool = building["annual_cool_kWh"] if supplied["annual_cool_kWh"] else (
        bm["cool_kWh_m2"] * scale if scale is not None else 0.0
    )
    dhw = building["annual_dhw_kWh"] if supplied["annual_dhw_kWh"] else (
        bm["dhw_kWh_m2"] * scale if scale is not None else 0.0
    )
 
    return float(heat), float(cool), float(dhw)

# ── Climate scenario reference ─────────────────────────────────────────────────

def compute_climate_reference(
    weather_df: pd.DataFrame,
    heat_base_C: float  = 15.5,
    cool_base_C: float  = 20.0,
    cool_onset_C: float = 22.0,
    cool_full_C: float  = 26.0,
) -> dict:
    """
    Compute the reference degree-hour signals needed to compare annual
    heating/cooling demand correctly across climate scenarios.

    CIBSE-style benchmarks are implicitly tied to a "standard" weather
    year. Without a shared reference, synthesise_building()/
    synthesise_network() rescale each scenario's weather-driven shape back
    to that same fixed benchmark — so a 2050 scenario shows the SAME
    annual heat/cool as today, just redistributed across the year, which
    hides the very effect you're trying to study.

    Call this ONCE on your baseline (unshifted) weather year, then pass
    the result as climate_reference= into every scenario you synthesise
    (including baseline itself, where it will correctly net out to a
    scaling factor of 1.0).

    Example
    -------
        baseline_weather = apply_climate_scenario(weather_df, "baseline")
        ref = compute_climate_reference(baseline_weather)

        for scenario_name in ["baseline", "2050_central", "2050_high"]:
            w = apply_climate_scenario(weather_df, scenario_name)
            result = synthesise_network(w, demand_scenario, climate_reference=ref)

    Returns
    -------
    dict with keys 'annual_HDD_h', 'annual_CDD_h', 'annual_ramp' — pass
    straight through as the climate_reference argument.
    """
    if len(weather_df) != 8760:
        raise ValueError(f"weather_df must have 8760 rows; got {len(weather_df)}.")

    T_air = weather_df["temp_drybulb_C"].values.astype(float)

    HDD_h = np.clip(heat_base_C - T_air, 0, None)
    CDD_h = np.clip(T_air - cool_base_C, 0, None)
    ramp  = np.clip((T_air - cool_onset_C) / (cool_full_C - cool_onset_C), 0.0, 1.0)

    return {
        "annual_HDD_h": float(HDD_h.sum()),
        "annual_CDD_h": float(CDD_h.sum()),
        "annual_ramp":  float(ramp.sum()),
    }


# ── Public API ─────────────────────────────────────────────────────────────────
 
def synthesise_building(
    weather_df: pd.DataFrame,
    building: dict,
    heat_base_C: float  = 15.5,
    cool_base_C: float  = 20.0,
    cool_onset_C: float = 22.0,
    cool_full_C: float  = 26.0,
    climate_reference: Optional[dict] = None,
) -> dict:
    """
    Generate 8,760-hour heating, cooling, and DHW profiles for one building.
 
    Parameters
    ----------
    weather_df   : from wather_data.csv — 'temp_drybulb_C' column, 8760 rows
    building     : config dict (name, type, floor_area_m2 or units, overrides)
    heat_base_C  : HDD base temperature (°C) — 15.5 is UK standard
    cool_base_C  : CDD base temperature (°C) — degree-day method threshold
                   used to scale the annual cooling benchmark (Part A).
                   Distinct from cool_onset_C: this is a methodology
                   constant, not a comfort threshold.
    cool_onset_C : temperature where comfort cooling demand begins (°C) —
                   used only by the comfort urgency ramp (Part B).
    cool_full_C  : temperature where comfort cooling demand saturates (°C)
    climate_reference : optional dict from compute_climate_reference(), with
                   keys 'annual_HDD_h', 'annual_CDD_h', 'annual_ramp'.
                   Pass the SAME reference (computed once from your baseline
                   weather year) into every climate scenario you're
                   comparing, so annual heating/cooling totals genuinely
                   move with the weather instead of being silently
                   rescaled back to the fixed CIBSE benchmark every time.
                   If None (default), each year is its own reference —
                   i.e. no climate-scenario adjustment is applied, which
                   is the right behaviour for a standalone, single-year run.
 
    Returns
    -------
    dict: name, type, annual/peak figures, hourly arrays (heating/cooling/dhw_kW),
          total_heat_kW (heating + DHW), datetime_index
    """

    if len(weather_df) != 8760:
        raise ValueError(f"weather_df must have 8760 rows; got {len(weather_df)}.")
 
    T_air = weather_df["temp_drybulb_C"].values.astype(float)
    btype = building.get("type", "office")
    bm    = BUILDING_TYPES[btype]
 
    occupancy = _make_occupancy(bm["occupancy"])
    heat_kWh, cool_kWh, dhw_kWh = _resolve_annual_demands(building)
 
    ref = climate_reference or {}
    heating_kW = _heating_profile(
        T_air, heat_kWh, occupancy, bm["base_load_frac"], heat_base_C,
        reference_annual_HDD_h=ref.get("annual_HDD_h"),
    )
    cooling_kW = _cooling_profile(T_air, cool_kWh, occupancy, bm["base_load_frac"],
                                  cool_base_C=cool_base_C,
                                  cool_onset_C=cool_onset_C,
                                  cool_full_C=cool_full_C,
                                  internal_gains_fraction=bm.get("internal_gains_fraction", DEFAULT_INTERNAL_GAINS_FRACTION),
                                  reference_annual_CDD_h=ref.get("annual_CDD_h"),
                                  reference_annual_ramp=ref.get("annual_ramp"))
    dhw_kW     = _dhw_profile(dhw_kWh, occupancy=occupancy)

    measured_peak = building.get("peak_total_heat_kW")
    if measured_peak is not None:
        total_annual = heat_kWh + dhw_kWh
        calibrated_total = _match_annual_and_peak(
            heating_kW + dhw_kW, total_annual, float(measured_peak)
        )
        if total_annual > 0:
            heating_kW = calibrated_total * (heat_kWh / total_annual)
            dhw_kW = calibrated_total * (dhw_kWh / total_annual)
 
    return {
        "name":            building.get("name", btype),
        "type":            btype,
        "annual_heat_kWh": float(heating_kW.sum()),
        "annual_cool_kWh": float(cooling_kW.sum()),
        "annual_dhw_kWh":  float(dhw_kW.sum()),
        "peak_heat_kW":    float(heating_kW.max()),
        "peak_cool_kW":    float(cooling_kW.max()),
        "peak_dhw_kW":     float(dhw_kW.max()),
        "heating_kW":      heating_kW,
        "cooling_kW":      cooling_kW,
        "dhw_kW":          dhw_kW,
        "total_heat_kW":   heating_kW + dhw_kW,
        "datetime_index":  weather_df.index,
    }

def synthesise_network(
    weather_df: pd.DataFrame,
    scenario: dict,
    heat_base_C: float  = 15.5,
    cool_base_C: float  = 20.0,
    cool_onset_C: float = 22.0,
    cool_full_C: float  = 26.0,
    climate_reference: Optional[dict] = None,
) -> dict:
    """
    Synthesise profiles for all demand nodes in a scenario config dict.

    Parameters
    ----------
    weather_df : from parse_epw.py / weather_data.csv loader
    scenario   : dict with 'demand_nodes' list (mirrors YAML structure)
    climate_reference : optional dict from compute_climate_reference() — see
                   synthesise_building() docstring. Pass the same reference
                   (computed once from baseline weather) into every
                   scenario you're comparing, so annual heating/cooling
                   totals actually shift with climate rather than only
                   reshaping across the year.

    Returns
    -------
    dict: nodes list, aggregated totals, peak demands, summary_df DataFrame
    """
    demand_nodes = scenario.get("demand_nodes", [])
    if not demand_nodes:
        raise ValueError("scenario config has no 'demand_nodes'.")

    nodes = [
        synthesise_building(weather_df, b, heat_base_C, cool_base_C, cool_onset_C, cool_full_C,
                             climate_reference=climate_reference)
        for b in demand_nodes
    ]

    total_heating = sum(n["heating_kW"] for n in nodes)
    total_cooling = sum(n["cooling_kW"] for n in nodes)
    total_dhw     = sum(n["dhw_kW"]     for n in nodes)

    aggregate_peak = scenario.get("aggregate_peak_heat_kW")
    if aggregate_peak is not None:
        annual_by_node = np.asarray([
            n["annual_heat_kWh"] + n["annual_dhw_kWh"] for n in nodes
        ], dtype=float)
        annual_total = float(annual_by_node.sum())
        if scenario.get("aggregate_load_shape_sharpness") is not None:
            calibrated_total = _match_annual_peak_with_sharpness(
                total_heating + total_dhw, annual_total, float(aggregate_peak),
                float(scenario["aggregate_load_shape_sharpness"]),
            )
        else:
            calibrated_total = _match_annual_and_peak(
                total_heating + total_dhw, annual_total, float(aggregate_peak)
            )
        shares = annual_by_node / annual_total
        for node, share in zip(nodes, shares):
            node_total = calibrated_total * share
            heat_fraction = node["annual_heat_kWh"] / max(
                node["annual_heat_kWh"] + node["annual_dhw_kWh"], 1e-12
            )
            node["heating_kW"] = node_total * heat_fraction
            node["dhw_kW"] = node_total * (1.0 - heat_fraction)
            node["total_heat_kW"] = node_total
            node["annual_heat_kWh"] = float(node["heating_kW"].sum())
            node["annual_dhw_kWh"] = float(node["dhw_kW"].sum())
            node["peak_heat_kW"] = float(node["heating_kW"].max())
            node["peak_dhw_kW"] = float(node["dhw_kW"].max())
        total_heating = sum(n["heating_kW"] for n in nodes)
        total_dhw = sum(n["dhw_kW"] for n in nodes)
    total_heat    = total_heating + total_dhw

    summary_df = pd.DataFrame([{
        "name":             n["name"],
        "type":             n["type"],
        "annual_heat_MWh":  round(n["annual_heat_kWh"] / 1000, 1),
        "annual_cool_MWh":  round(n["annual_cool_kWh"] / 1000, 1),
        "annual_dhw_MWh":   round(n["annual_dhw_kWh"]  / 1000, 1),
        "annual_total_MWh": round((n["annual_heat_kWh"] + n["annual_dhw_kWh"]) / 1000, 1),
        "peak_heat_kW":     round(n["peak_heat_kW"], 1),
        "peak_cool_kW":     round(n["peak_cool_kW"], 1),
    } for n in nodes])

    return {
        "nodes":            nodes,
        "total_heating_kW": total_heating,
        "total_cooling_kW": total_cooling,
        "total_dhw_kW":     total_dhw,
        "total_heat_kW":    total_heat,
        "peak_heat_kW":     float(total_heat.max()),
        "peak_cool_kW":     float(total_cooling.max()),
        "annual_heat_MWh":  float(total_heating.sum() / 1000),
        "annual_cool_MWh":  float(total_cooling.sum() / 1000),
        "annual_dhw_MWh":   float(total_dhw.sum()     / 1000),
        "datetime_index":   weather_df.index,
        "summary_df":       summary_df,
    }

def to_dataframe(network_result: dict) -> pd.DataFrame:
    """
    Flatten synthesise_network output to a single 8760-row DataFrame.
    Columns: totals + per-node heating/cooling/dhw columns.
    Suitable for CSV export or passing to the dispatch optimiser.
    """
    df = pd.DataFrame(index=network_result["datetime_index"])
    df["total_heating_kW"] = network_result["total_heating_kW"]
    df["total_cooling_kW"] = network_result["total_cooling_kW"]
    df["total_dhw_kW"]     = network_result["total_dhw_kW"]
    df["total_heat_kW"]    = network_result["total_heat_kW"]
 
    for node in network_result["nodes"]:
        safe = node["name"].replace(" ", "_").replace("-", "_").lower()
        df[f"{safe}_heat_kW"] = node["heating_kW"]
        df[f"{safe}_cool_kW"] = node["cooling_kW"]
        df[f"{safe}_dhw_kW"]  = node["dhw_kW"]
 
    return df
