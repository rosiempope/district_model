"""
water_cooled_chiller.py
=======================
Water-cooled chiller + open evaporative cooling tower — the canonical
efficiency upgrade over the air-cooled chiller in components/chiller.py.

Same vapour-compression machine; the ONLY thing that changes is where the
condenser dumps its heat. An air-cooled chiller rejects into the DRY-bulb
ambient; a water-cooled chiller rejects into condenser water that a cooling
tower holds close to the ambient WET-bulb (always at/below dry-bulb, and
several degrees below it on a warm humid day). A lower heat-rejection
temperature means a smaller lift for the compressor, which means a higher COP —
industry sources put water-cooled COP at ~4-6 against ~2.5-3.5 for an
equivalent air-cooled unit (20-40% less electricity for the same cooling; see
e.g. ARANER, EEA Consulting, ChillerOne chiller-comparison guides). This module
mirrors chiller.py's structure exactly (same COP-curve SHAPE, same per-unit
outage model) with the heat-rejection side moved from dry-bulb to wet-bulb, and
adds the real running cost that buys that efficiency: cooling-tower water.

COP methodology
---------------
COP = a + b*dT + c*dT^2          where dT = T_wetbulb - T_chilled_water

Same functional form as chiller.py, but dT is the lift against the WET-bulb
(the temperature the tower rejects to), not the dry-bulb. The tower approach
(~4°C, cooling_common.COOLING_TOWER_APPROACH_C) and the condenser approach are
BAKED INTO the anchor points below — exactly as chiller.py bakes the air-side
refrigerant-to-air approach into its dry-bulb anchors — so they are NOT applied
a second time on top of the curve (that would double-count the approach).

Quadratic fitted to two real anchor points + one real measured slope, the same
philosophy as chiller.py:

  - Anchor 1 (design / AHRI 550/590 condition): a water-cooled chiller rated at
    the standard 6.7°C leaving chilled water / 29.4°C (85°F) entering condenser
    water reports a full-load COP of ~5.8 (mid of the widely-quoted 5.5-6.5 for
    a good large water-cooled unit). 29.4°C condenser water corresponds to a
    ~24°C wet-bulb plus the tower + condenser approach, so at T_chilled=7°C,
    dT = 24-7 = 17°C, COP = 5.8.
  - Anchor 2 (mild / low condenser water): the same class of unit at a ~10°C
    wet-bulb (cool, humid UK shoulder-season) reaches COP ~8 — condenser water
    drops with wet-bulb, the lift collapses, and centrifugal part-load
    efficiency is at its best. dT = 10-7 = 3°C, COP = 8.0.
  - Slope constraint at anchor 1: the SAME real floating-head-pressure rule
    chiller.py cites (ChillerOne, "~2.5% compressor power per °C of condensing
    temperature") gives d(COP)/d(dT) at dT=17 = -0.025 x 5.8.

Solved coefficients (see the module's self-fit, reproducible):
    COP = 8.5157 - 0.17449*dT + 0.000867*dT^2

Capacity derating
-----------------
A water-cooled unit barely derates across a UK climate: the tower is sized to
the site's design wet-bulb, and UK wet-bulbs rarely exceed ~22°C, so the
condenser stays near its rated condition almost all year. Modelled as a gentle
derate only ABOVE a high wet-bulb design point (mirror of chiller.py's
_capacity_derate_hot, but keyed on wet-bulb and much shallower) — it essentially
never bites in the UK, which is the honest answer, not a modelling shortcut.

Water OPEX
----------
The tower's efficiency comes at the cost of evaporating water. That make-up +
treatment cost is added to the electricity marginal cost via
cooling_common.cooling_tower_water_cost_GBP_per_MWh_cooling() — a real running
cost the air-cooled chiller does not have. See that helper for sourcing.

CAPEX methodology
-----------------
£150,000/MW — a modest premium over chiller.py's air-cooled £130,000/MW. The
water-cooled chiller barrel itself is actually CHEAPER per ton than air-cooled
(~$300-400/ton vs air-cooled's ~$600/ton installed), but it needs a cooling
tower, condenser-water pumps, condenser-water pipework and a water-treatment
skid on top, which industry cost guides put at roughly $150-250/ton of added
plant. Netting the cheaper barrel against the added balance-of-plant lands a
large fully-installed water-cooled system a little ABOVE the air-cooled figure
rather than below it (cooling-tower-cost guides, thecoolingco.com / LNEYA 2025).
£150,000/MW is a deliberately conservative mid estimate — the point of this
module is the OPEX efficiency win, and it should not be flattered by also
assuming the plant is cheaper to build.

Usage mirrors chiller.py's AirCooledChiller exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from economics.tariffs import resolve_electricity_price, ElectricityTariff
from components.peak_demand_option import CARBON_INTENSITY
# Reuse ASHP.py's real, tested per-unit outage model directly — identical
# reasoning to chiller.py (staggered one-unit-at-a-time servicing is the same
# real O&M logic regardless of heating/cooling/rejection medium).
from components.ASHP import _ashp_unit_outage_profile
from components.cooling_common import (
    N_HOURS, wet_bulb_temp_C, cooling_tower_water_cost_GBP_per_MWh_cooling,
)


# ── Constants ──────────────────────────────────────────────────────────────────

# Real COP curve, fitted to two real anchor points + one real measured slope
# (see module docstring). COP = A + B*dT + C*dT^2, dT = T_wetbulb - T_chilled.
COP_FIT_A = 8.51566
COP_FIT_B = -0.17449
COP_FIT_C = 0.000867

# Design wet-bulb above which the tower starts to lose grip on the condenser
# water (a UK cooling tower is typically sized to a ~20-22°C design wet-bulb).
RATING_WETBULB_C = 22.0


# ── Generic presets ──────────────────────────────────────────────────────────
# Generic, round-number presets — no project-specific water-cooled unit exists
# in this project's source documents yet (same status note as chiller.py).

WATER_COOLED_CHILLER_PRESETS = {
    "generic_500kW": {
        "description":          "Generic 500kW water-cooled chiller + tower (single unit)",
        "n_units":               1,
        "unit_capacity_MW":      0.5,
        "chilled_water_temp_C":  7.0,
        "max_wetbulb_design_C":  28.0,
        "reference":            "Generic — no project-specific water-cooled data sourced yet",
    },
    "generic_2MW_bank": {
        "description":          "Generic 2MW water-cooled chiller bank + tower (4x500kW)",
        "n_units":               4,
        "unit_capacity_MW":      0.5,
        "chilled_water_temp_C":  7.0,
        "max_wetbulb_design_C":  28.0,
        "reference":            "Generic — no project-specific water-cooled data sourced yet",
    },
}


# ── COP model ──────────────────────────────────────────────────────────────────

def water_cooled_chiller_cop(
    T_wetbulb_C: np.ndarray,
    T_chilled_water_C: float,
    cop_floor: float = 2.0,
    cop_ceiling: float = 9.0,
) -> np.ndarray:
    """
    Water-cooled chiller COP at every hour — see module docstring for the full
    real-data-anchored fitting methodology. Note the lift is against the
    WET-bulb (what the tower rejects to), NOT the dry-bulb.

    cop_ceiling is higher than chiller.py's (9.0 vs 8.0) because a water-cooled
    unit genuinely reaches higher part-load COPs at low condenser water — anchor
    2 alone is COP 8.0 — so an 8.0 cap would clip real, achievable performance.
    """
    T = np.asarray(T_wetbulb_C, dtype=float)
    dT = T - T_chilled_water_C
    cop = COP_FIT_A + COP_FIT_B * dT + COP_FIT_C * dT ** 2
    return np.clip(cop, cop_floor, cop_ceiling)


# ── Capacity derating (very shallow, HIGH wet-bulb) ────────────────────────────

def _capacity_derate_wetbulb(
    T_wetbulb_C: np.ndarray,
    rating_wetbulb_C: float = RATING_WETBULB_C,
    max_wetbulb_C: float = 28.0,
    min_capacity_fraction: float = 0.92,
) -> np.ndarray:
    """
    Mirror of chiller.py's _capacity_derate_hot(), but keyed on WET-bulb and
    much shallower (min fraction 0.92 vs the air-cooled 0.80): the tower holds
    the condenser near its rated condition until the wet-bulb climbs above the
    design value, which in a UK climate is a handful of hours per year at most.
    Linear from 100% at rating_wetbulb_C to min_capacity_fraction at
    max_wetbulb_C, held flat beyond.
    """
    if max_wetbulb_C <= rating_wetbulb_C:
        raise ValueError(
            f"max_wetbulb_design_C ({max_wetbulb_C}) must exceed the rating "
            f"wet-bulb ({rating_wetbulb_C}) — otherwise the derate range is "
            f"zero or negative (the same guard chiller.py carries)."
        )
    T = np.asarray(T_wetbulb_C, dtype=float)
    derate_progress = np.clip((T - rating_wetbulb_C) / (max_wetbulb_C - rating_wetbulb_C), 0, 1)
    return 1.0 - (1.0 - min_capacity_fraction) * derate_progress


# ── WaterCooledChiller class ─────────────────────────────────────────────────

class WaterCooledChiller:
    """
    A generalised array of N identical water-cooled chiller units, each on a
    shared open evaporative cooling tower. Structurally parallel to
    AirCooledChiller (see chiller.py) — same public surface, same modular
    n_units x unit_capacity_MW scaling — with heat rejection moved to wet-bulb
    and a real tower water OPEX added.

    Parameters mirror AirCooledChiller's, with two differences:
      - max_wetbulb_design_C replaces max_ambient_design_C (rejection is to the
        wet-bulb, so the design ceiling is a wet-bulb, not a dry-bulb).
      - the weather_df must ALSO carry 'rel_humidity_pct' (needed to derive
        wet-bulb) as well as 'temp_drybulb_C'.
    """

    source_type = "water_cooled_chiller"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        chilled_water_temp_C: float          = 7.0,
        weather_df: Optional[pd.DataFrame]    = None,
        max_wetbulb_design_C: float           = 28.0,
        min_capacity_fraction: float          = 0.92,
        electricity_price_GBP_per_MWh         = None,
        capex_GBP_per_MW: float                = 150_000.0,
        availability_factor: float             = 0.97,
        seed: int                              = 11,
        reference: str                         = "",
    ):
        if weather_df is None:
            raise ValueError(
                "WaterCooledChiller requires weather_df (must have "
                "'temp_drybulb_C' AND 'rel_humidity_pct' columns, 8760 rows) — "
                "wet-bulb heat rejection is humidity-dependent."
            )
        if len(weather_df) != N_HOURS:
            raise ValueError(f"weather_df must have {N_HOURS} rows; got {len(weather_df)}.")
        if "rel_humidity_pct" not in weather_df.columns:
            raise ValueError(
                "WaterCooledChiller needs a 'rel_humidity_pct' column to derive "
                "wet-bulb temperature (the whole point of a cooling tower is that "
                "it rejects to the wet-bulb, not the dry-bulb)."
            )
        if max_wetbulb_design_C <= RATING_WETBULB_C:
            raise ValueError(
                f"max_wetbulb_design_C ({max_wetbulb_design_C}) must exceed the "
                f"rating wet-bulb ({RATING_WETBULB_C}°C)."
            )

        self.name                    = name
        self.n_units                  = int(n_units)
        self.unit_capacity_MW         = float(unit_capacity_MW)
        self.capacity_MW              = self.n_units * self.unit_capacity_MW
        self.chilled_water_temp_C      = float(chilled_water_temp_C)
        self.max_wetbulb_design_C      = float(max_wetbulb_design_C)
        self.min_capacity_fraction     = float(min_capacity_fraction)
        self.capex_GBP_per_MW          = float(capex_GBP_per_MW)
        self.availability_factor       = float(availability_factor)
        self.seed                      = int(seed)
        self.reference                 = reference

        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        RH = weather_df["rel_humidity_pct"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air
        self._rel_humidity_pct = RH          # kept so resize() can rebuild weather_df exactly
        self.wetbulb_temp_C = wet_bulb_temp_C(T_air, RH)

        # COP at every hour — lift against the WET-bulb
        self.cop_hourly = water_cooled_chiller_cop(self.wetbulb_temp_C, self.chilled_water_temp_C)

        # Capacity derating — shallow, wet-bulb-driven
        self._capacity_fraction = _capacity_derate_wetbulb(
            self.wetbulb_temp_C,
            rating_wetbulb_C=RATING_WETBULB_C,
            max_wetbulb_C=self.max_wetbulb_design_C,
            min_capacity_fraction=self.min_capacity_fraction,
        )

        # Units available each hour — reuse ASHP's staggered outage model
        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = (
            self.units_available / self.n_units if self.n_units > 0 else np.ones(N_HOURS)
        )

        # Available cooling supply each hour (MW)
        self.supply_MW = (
            self.capacity_MW * self._capacity_fraction * self._unit_availability_fraction
        )
        self.supply_temp_C = np.full(N_HOURS, self.chilled_water_temp_C)

        # Electricity price and marginal cost
        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)
        # Tower make-up water + treatment, per MWh cooling (uses the ELECTRICAL
        # COP: the tower rejects the cooling plus the compressor work).
        self.water_cost_GBP_per_MWh = cooling_tower_water_cost_GBP_per_MWh_cooling(self.cop_hourly)
        # Marginal cost of cooling = electricity/COP + tower water. This is what
        # the merit-order dispatcher compares against other cooling sources.
        self.marginal_cost = self._elec_price / self.cop_hourly + self.water_cost_GBP_per_MWh

        # Carbon per unit cooling — grid electricity / COP, identical basis to
        # chiller.py (the tower fans' small draw is already inside the rated COP).
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly

        # Electrical demand IF running at full available supply (MW_elec)
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly

    # ── constructors / helpers — identical contract to AirCooledChiller ────────

    @classmethod
    def from_preset(cls, preset_key: str, weather_df: pd.DataFrame, **overrides) -> "WaterCooledChiller":
        if preset_key not in WATER_COOLED_CHILLER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(WATER_COOLED_CHILLER_PRESETS.keys())}"
            )
        params = WATER_COOLED_CHILLER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(cls, config: dict, weather_df: pd.DataFrame) -> "WaterCooledChiller":
        params = config.copy()
        tariff_block = params.pop("electricity_tariff", None)
        if tariff_block is not None:
            params["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_block)
        return cls(weather_df=weather_df, **params)

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        return WaterCooledChiller(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            chilled_water_temp_C=self.chilled_water_temp_C,
            weather_df=pd.DataFrame({
                "temp_drybulb_C": self.ambient_temp_C,
                "rel_humidity_pct": self._rel_humidity_pct,
            }),
            max_wetbulb_design_C=self.max_wetbulb_design_C,
            min_capacity_fraction=self.min_capacity_fraction,
            electricity_price_GBP_per_MWh=self._elec_price,
            capex_GBP_per_MW=self.capex_GBP_per_MW,
            availability_factor=self.availability_factor,
            seed=self.seed,
            reference=self.reference,
        )

    def summary(self) -> dict:
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "total_capacity_MW":          round(self.capacity_MW, 2),
            "chilled_water_temp_C":       self.chilled_water_temp_C,
            "mean_wetbulb_C":             round(float(self.wetbulb_temp_C.mean()), 2),
            "cop_mean":                   round(float(self.cop_hourly.mean()), 2),
            "cop_min":                    round(float(self.cop_hourly.min()), 2),
            "cop_max":                    round(float(self.cop_hourly.max()), 2),
            "annual_cooling_available_MWh": round(float(self.supply_MW.sum()), 0),
            "annual_electrical_demand_MWh": round(float(self.electrical_demand_MW.sum()), 0),
            "seasonal_avg_cop":           round(
                float(self.supply_MW.sum() / self.electrical_demand_MW.sum()), 2
            ),
            "mean_water_cost_GBP_per_MWh": round(float(self.water_cost_GBP_per_MWh.mean()), 2),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "max_wetbulb_design_C":       self.max_wetbulb_design_C,
            "availability_factor":        self.availability_factor,
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"WaterCooledChiller(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"T_chw={self.chilled_water_temp_C:.0f}°C, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )
