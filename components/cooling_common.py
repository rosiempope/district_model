"""
cooling_common.py
=================
Shared physics helpers for the cooling-source family, so the three
efficiency-upgrade chillers (water_cooled_chiller.py, free_cooling_chiller.py,
absorption_chiller.py) and the baseline air-cooled chiller.py don't each grow
their own slightly-drifting copy of the same real-world relationships.

Two things live here that more than one cooling component needs:

1. WET-BULB temperature (wet_bulb_temp_C)
   Air-cooled units reject condenser heat to the DRY-bulb ambient; a cooling
   tower rejects to something close to the WET-bulb, which is always at or
   below dry-bulb and, on a warm humid UK afternoon, several degrees below it.
   That lower heat-rejection temperature is the entire physical reason a
   water-cooled (or absorption) chiller runs at a higher COP than an air-cooled
   one — so any honest model of those units has to work in wet-bulb, not
   dry-bulb. The project's weather file carries dry-bulb + relative humidity
   (profiles/weather_data.csv), so wet-bulb is derived, not measured.

2. COOLING-TOWER WATER OPEX (cooling_tower_water_cost_GBP_per_MWh_cooling)
   The flip side of the tower's efficiency win: an open evaporative tower
   CONSUMES water (evaporation + drift + blowdown) and needs treatment
   chemicals. That is a genuine running cost the air-cooled chiller simply does
   not have, and leaving it out would flatter every water-cooled/absorption
   case. Returned as a £/MWh_cooling adder so it folds straight into a
   component's marginal_cost, exactly like its electricity cost does.
"""

import numpy as np

N_HOURS = 8760

# Cooling-tower "approach" — how far above the ambient WET-bulb the tower can
# actually bring the condenser water. A real open tower gets to within ~3-5°C
# of wet-bulb (ASHRAE / CTI design practice; 5°F ≈ 2.8°C is a common tight
# design, 5°C a comfortable one). 4°C is a reasonable mid-range default and is
# used as the anchor point for the water-cooled COP curve, so it is NOT applied
# a second time on top of that curve (that would double-count) — see
# water_cooled_chiller.py's COP note.
COOLING_TOWER_APPROACH_C = 4.0

# Open-tower make-up water per kWh of heat REJECTED. Evaporating enough water
# to reject 1 kWh takes ~1.5 L (latent heat of vaporisation ≈ 2.4 MJ/kg ⇒
# 3.6 MJ / 2.4 ≈ 1.5 kg); drift and blowdown (to stop the circulating water
# concentrating up as it evaporates) add roughly a third on top, giving ~2 L/kWh
# of make-up. This is the standard cooling-tower rule of thumb (~1.8-2.2 L per
# kWh rejected; e.g. BSRIA / SPX Cooling Technologies tower guidance).
TOWER_MAKEUP_L_PER_KWH_REJECTED = 2.0

# Delivered cost of that make-up water + treatment chemicals + effluent, per m3.
# UK non-domestic mains + trade-effluent + dosing sits around £2-3/m3 all-in;
# £2.50/m3 is a mid figure. (Water is a small fraction of a water-cooled
# chiller's running cost next to its electricity, but a real and non-zero one.)
TOWER_WATER_COST_GBP_PER_M3 = 2.50


def wet_bulb_temp_C(
    T_drybulb_C: np.ndarray,
    rel_humidity_pct: np.ndarray,
) -> np.ndarray:
    """
    Ambient WET-bulb temperature (°C) from dry-bulb (°C) and relative
    humidity (%), via the Stull (2011) single-equation empirical fit.

    Reference: Stull, R. (2011), "Wet-Bulb Temperature from Relative Humidity
    and Air Temperature", Journal of Applied Meteorology and Climatology 50,
    2267-2269 (doi:10.1175/JAMC-D-11-0143.1). Fitted against a full
    psychrometric calculation and accurate to ~±0.3°C over the range that
    matters here (RH 5-99%, dry-bulb roughly -20..+50°C at sea-level pressure),
    which comfortably covers a UK cooling season. This is the same "fit to a
    real reference, don't hand-build a curve" philosophy the COP models in this
    project already follow (ASHP's Ruhnau regression, chiller's anchored fit).

    Note: the fit assumes near-sea-level pressure. The weather file also carries
    dew point and station pressure, so a full iterative psychrometric solve is
    possible later if a site well above sea level is ever modelled; for UK
    lowland district schemes the Stull approximation is well within the noise of
    everything else in a feasibility-stage model.
    """
    T = np.asarray(T_drybulb_C, dtype=float)
    # Clamp RH into the fit's validated band so a stray 0% or >100% reading
    # (real EPW files occasionally saturate to 100) can't push the empirical
    # arctangents somewhere the fit was never tested.
    RH = np.clip(np.asarray(rel_humidity_pct, dtype=float), 5.0, 99.0)

    Tw = (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )
    # Physical guard: wet-bulb can never exceed dry-bulb. The fit respects this
    # everywhere in its validated range, but clip anyway so the invariant holds
    # by construction for any input.
    return np.minimum(Tw, T)


def cooling_tower_water_cost_GBP_per_MWh_cooling(
    cop_hourly: np.ndarray,
    makeup_L_per_kWh_rejected: float = TOWER_MAKEUP_L_PER_KWH_REJECTED,
    water_cost_GBP_per_m3: float = TOWER_WATER_COST_GBP_PER_M3,
) -> np.ndarray:
    """
    Make-up water + treatment cost of an evaporative cooling tower, expressed
    per MWh of COOLING delivered so it can be added directly onto a component's
    electricity/COP marginal cost.

    For every 1 unit of cooling the tower must reject the cooling PLUS the work
    (or driving heat) that produced it: heat_rejected = cooling × (1 + 1/COP).
    So cop_hourly here is the ratio of cooling delivered to the energy input
    that ends up as extra rejected heat — the ELECTRICAL COP for a mechanical
    water-cooled chiller, or the THERMAL COP for an absorption chiller (whose
    driving heat is also dumped through the tower). Passing the right COP is the
    caller's job; the arithmetic is identical either way.

    Unit check (the /1000 L->m3 and ×1000 kWh->MWh cancel):
        £/MWh_cooling = makeup_L_per_kWh_rejected
                        × (1 + 1/COP)              # heat rejected per unit cooling
                        × water_cost_GBP_per_m3
    """
    cop = np.asarray(cop_hourly, dtype=float)
    heat_rejected_ratio = 1.0 + 1.0 / cop
    return makeup_L_per_kWh_rejected * heat_rejected_ratio * water_cost_GBP_per_m3
