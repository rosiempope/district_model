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
        "description":    "General office (naturally ventilated)",
    },
    "office_ac": {
        "heat_kWh_m2":    120*0.85,
        "cool_kWh_m2":    50.0,
        "dhw_kWh_m2":     120*0.15,
        "occupancy":      "office",
        "base_load_frac":  0.15,
        "description":    "Air-conditioned office",
    },
    "residential": {
        "heat_kWh_m2":    120*0.70,   # Part L 2021 new build
        "cool_kWh_m2":     2.0,   # Near-zero installed A/C; comfort ramp carries this
        "dhw_kWh_m2":     35.0,
        "occupancy":      "residential",
        "base_load_frac":  0.35,
        "description":    "Residential (new build, Part L 2021)",
    },
    "residential_existing": {
        "heat_kWh_m2":   130.0,   # Pre-2010 stock
        "cool_kWh_m2":     2.0,
        "dhw_kWh_m2":    40.0,
        "occupancy":      "residential",
        "base_load_frac":  0.35,
        "description":    "Residential (existing / retrofit target)",
    },
    "hospital": {
        "heat_kWh_m2":   200.0,   # 24/7, high ventilation
        "cool_kWh_m2":    55.0,   # Ealing report: 40-60 kWh/m²
        "dhw_kWh_m2":   120.0,    # Sterile processes, high DHW
        "occupancy":      "hospital",
        "base_load_frac":  0.70,
        "description":    "Hospital / acute healthcare",
    },
    "retail": {
        "heat_kWh_m2":    90.0,
        "cool_kWh_m2":    60.0,   # High internal gains from lighting/people
        "dhw_kWh_m2":     5.0,
        "occupancy":      "retail",
        "base_load_frac":  0.10,
        "description":    "Retail / high street shops",
    },
    "supermarket": {
        "heat_kWh_m2":    70.0,
        "cool_kWh_m2":   100.0,   # Refrigeration, high cooling load
        "dhw_kWh_m2":     5.0,
        "occupancy":      "retail_extended",
        "base_load_frac":  0.20,
        "description":    "Supermarket / food retail",
    },
    "hotel": {
        "heat_kWh_m2":   150.0,
        "cool_kWh_m2":    30.0,   # Ealing report: hotel 20-35
        "dhw_kWh_m2":    90.0,
        "occupancy":      "hotel",
        "base_load_frac":  0.50,
        "description":    "Hotel",
    },
    "school": {
        "heat_kWh_m2":   100.0,
        "cool_kWh_m2":     8.0,   # Very little A/C in UK schools
        "dhw_kWh_m2":    15.0,
        "occupancy":      "school",
        "base_load_frac":  0.05,
        "description":    "Secondary school",
    },
    "mixed_use": {
        "heat_kWh_m2":   100.0,
        "cool_kWh_m2":    20.0,
        "dhw_kWh_m2":    40.0,
        "occupancy":      "mixed",
        "base_load_frac":  0.25,
        "description":    "Mixed-use development",
    },
    "data_centre": {
        "heat_kWh_m2":     0.0,   # Modelled as heat source, not demand
        "cool_kWh_m2":     0.0,
        "dhw_kWh_m2":      0.0,
        "occupancy":      "always_on",
        "base_load_frac":  1.0,
        "description":    "Data centre (heat source node)",
    },
}

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
    reference_annual_CDD_h: Optional[float] = None,
    reference_annual_ramp: Optional[float]  = None,
) -> np.ndarray:
    """
    Two-part cooling demand model — returns element-wise maximum:

    Part A (CDD scaling): distributes annual_cool_kWh proportional to
    cooling degree-hours. Captures installed A/C load.

    Part B (comfort urgency ramp): smooth linear ramp 0→1 between
    cool_onset_C and cool_full_C. Captures forward-looking demand —
    people WILL seek cooling when it gets to 26°C+ even without
    existing A/C infrastructure. Only active during occupied hours.

    Using max() means hot spells always show realistic demand peaks
    even if the annual CDD total is low due to no A/C infrastructure.

    Degree-day normalisation
    -------------------------
    Same issue as _heating_profile, but on the cooling side it's worse:
    Part B is specifically meant to be the climate-change-sensitive
    signal, but without this correction BOTH parts get rescaled to the
    same fixed annual_cool_kWh regardless of how much hotter the weather
    actually is — so the comfort ramp reshapes cooling demand across the
    year without ever increasing the annual total, silently defeating its
    own stated purpose.

    reference_annual_CDD_h / reference_annual_ramp let the caller supply
    the annual CDD-hour sum and ramp sum from a reference (typically
    baseline) weather year. Each part is then independently scaled by
    (this year's actual signal / reference signal) before being
    distributed, so a hotter scenario shows MORE annual cooling from
    both parts, not just a reshaped version of the same fixed total.

    If either reference is None (default), this year's own signal is used
    as its own reference (scaling factor 1.0) — identical to a
    standalone, no-climate-comparison run.
    """
    n = len(T_air)
    occ_modifier = base_load_frac + (1.0 - base_load_frac) * occupancy

    # -- Part A: CDD scaling --------------------------------------------------
    CDD_h = np.clip(T_air - cool_base_C, 0, None)
    CDD_annual = CDD_h.sum()
    if CDD_annual > 0.1:
        reference_A = reference_annual_CDD_h if reference_annual_CDD_h is not None else CDD_annual
        effective_annual_cool_kWh_A = annual_cool_kWh * (CDD_annual / reference_A)
        raw_A = CDD_h * occ_modifier
        part_A = raw_A * (effective_annual_cool_kWh_A / raw_A.sum())
    else:
        # Too few hours above base — CDD gives near-zero; Part B takes over
        part_A = np.zeros(n)

    # -- Part B: Comfort urgency ramp -----------------------------------------
    ramp = np.clip(
        (T_air - cool_onset_C) / (cool_full_C - cool_onset_C),
        0.0, 1.0
    )
    raw_B = ramp * occupancy
    ramp_annual = raw_B.sum()
    if ramp_annual > 0:
        reference_B = reference_annual_ramp if reference_annual_ramp is not None else ramp_annual
        effective_annual_cool_kWh_B = annual_cool_kWh * (ramp_annual / reference_B)
        part_B = raw_B * (effective_annual_cool_kWh_B / ramp_annual)
    else:
        part_B = np.zeros(n)

    # Final: whichever method gives higher load wins at each hour
    return np.maximum(part_A, part_B)  # kW


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
        Resolve annual heating, cooling, DHW demands (kWh).
        Uses explicit config values if provided, otherwise scales benchmarks
        by floor_area_m2 or units (assuming 75 m²/dwelling).
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
 
    if floor_area and float(floor_area) > 0:
        scale = float(floor_area)
    elif units and float(units) > 0:
        scale = float(units) * 75.0   # 75 m²/dwelling assumption
        building["floor_area_m2"] = scale
    else:
        raise ValueError(
            f"Building '{building.get('name','?')}' needs "
            f"'floor_area_m2' or 'units'."
        )
 
    heat = building.get("annual_heat_kWh") or (bm["heat_kWh_m2"] * scale)
    cool = building.get("annual_cool_kWh") or (bm["cool_kWh_m2"] * scale)
    dhw  = building.get("annual_dhw_kWh")  or (bm["dhw_kWh_m2"]  * scale)
 
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
                                  reference_annual_CDD_h=ref.get("annual_CDD_h"),
                                  reference_annual_ramp=ref.get("annual_ramp"))
    dhw_kW     = _dhw_profile(dhw_kWh, occupancy=occupancy)
 
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

# ── Self-test ──────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  demand_synthesis.py — self-test (synthetic weather)")
    print("="*65)
 
    np.random.seed(42)
    hours = np.arange(8760)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    dates      = pd.date_range("2023-01-01", periods=8760, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    scenario = {
        "demand_nodes": [
            {"name": "Perceval House",       "type": "office",              "floor_area_m2": 8500},
            {"name": "High Street Retail",   "type": "retail",              "floor_area_m2": 3000},
            {"name": "Ealing Hospital Wing", "type": "hospital",            "floor_area_m2": 12000},
            {"name": "Dickens Yard Ph1",     "type": "residential",         "units": 350},
            {"name": "Broadway Hotel",       "type": "hotel",               "floor_area_m2": 5000},
            {"name": "Ellen Wilkinson Sch",  "type": "school",              "floor_area_m2": 6000},
        ]
    }

    network = synthesise_network(weather_df, scenario)
 
    print("\n  Per-building summary:")
    print(network["summary_df"].to_string(index=False))
 
    hh = network["total_heat_kW"]
    cc = network["total_cooling_kW"]
    jan_heat = hh[:744].mean()
    jul_heat = hh[4344:5088].mean()
    jul_cool = cc[4344:5088].mean()
    jan_cool = cc[:744].mean()

    print(f"\n  Network totals:")
    print(f"    Annual space heat : {network['annual_heat_MWh']:>8.0f} MWh")
    print(f"    Annual DHW        : {network['annual_dhw_MWh']:>8.0f} MWh")
    print(f"    Annual cooling    : {network['annual_cool_MWh']:>8.0f} MWh")
    print(f"    Peak heat demand  : {network['peak_heat_kW']:>8.1f} kW")
    print(f"    Peak cooling      : {network['peak_cool_kW']:>8.1f} kW")
    print(f"    Cool:Heat ratio   : {network['annual_cool_MWh']/(network['annual_heat_MWh']+network['annual_dhw_MWh']):.2f}  (expect ~0.05-0.15 for UK)")

    print(f"\n  Seasonal sanity:")
    print(f"    Jan heat: {jan_heat:.0f} kW  |  Jul heat: {jul_heat:.0f} kW  → {'✓ winter peak' if jan_heat > jul_heat else '✗ FAIL'}")
    print(f"    Jan cool: {jan_cool:.1f} kW  |  Jul cool: {jul_cool:.1f} kW  → {'✓ summer peak' if jul_cool > jan_cool else '✗ FAIL'}")
    print(f"    Zero cooling hours: {(cc == 0).sum()} / 8760  (expect majority)")
    print()