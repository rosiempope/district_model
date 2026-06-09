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
import numpy as np
import pandas as pd
from typing import Optional
import warnings
 
# ── Building type benchmarks ───────────────────────────────────────────────────
# Sources:
#   Heating EUI — CIBSE TM46 / Ealing Feasibility Report cross-check
#   Cooling EUI — SEL in-house benchmarks (per Ealing report), CIBSE Guide F
#   DHW EUI     — CIBSE Guide G, EST data
#   All values in kWh/m²/yr

BUILDING_TYPES = {
    "office": {
        "heat_kWh_m2":    120*0.85,
        "cool_kWh_m2":    30.0,
        "dhw_kWh_m2":     "heat_kWh_m2" * 0.15,
        "occupancy":      "office",
        "base_load_frac":  0.15,
        "description":    "General office (naturally ventilated)",
    },
    "office_ac": {
        "heat_kWh_m2":    120*0.85,
        "cool_kWh_m2":    50.0,
        "dhw_kWh_m2":     "heat_kWh_m2" * 0.15,
        "occupancy":      "office",
        "base_load_frac":  0.15,
        "description":    "Air-conditioned office",
    },
    "residential": {
        "heat_kWh_m2":    80.0,   # Part L 2021 new build
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