"""
free_cooling_chiller.py
=======================
Free-cooling ("dry-cooler" / glycol economiser) chiller — an air-cooled chiller
with an added glycol free-cooling coil, the arrangement the user ran in the
coffee factory (a glycol loop cooled by a dry air-cooler, with a chiller behind
it for the warm hours).

The physics that makes it worth modelling separately from chiller.py: when the
ambient air is cold enough, the glycol loop can be cooled all the way to the
chilled-water setpoint by the dry cooler ALONE — the compressor switches off and
the only electricity drawn is the dry-cooler fans and the glycol pump. That is
an effective COP of ~15-25 instead of ~4-6, for as many hours of the year as
the weather allows. In between, the dry cooler PRE-cools the return glycol and
the chiller only has to make up the last few degrees ("partial free cooling").
Above a threshold ambient the coil does nothing and the unit runs as a plain
air-cooled chiller.

This matters a lot in a UK climate, where a large share of comfort-cooling
demand is driven by internal gains (people, lighting, IT) and therefore occurs
on mild or cold days when free cooling is fully available — see
profiles/demand_synthesis.py's internal-gains cooling split. The dispatch engine
picks this unit up cheaply on exactly those hours because its marginal cost
collapses when the compressor is off.

Free-cooling availability (per hour, from the dry-bulb)
------------------------------------------------------
A dry cooler can pull the glycol down to about ambient + a small approach
(~3°C). The share of the cooling LOAD it can carry is the share of the
return->supply temperature drop it can achieve on its own:

    f = (no_free_ambient - T_ambient) / (no_free_ambient - full_free_ambient)   (clipped 0..1)

  full_free_ambient = cool_flow_temp - dry_cooler_approach   (dry cooler alone
                       reaches the chilled setpoint; compressor OFF; f = 1)
  no_free_ambient   = cool_return_temp - dry_cooler_approach  (dry cooler can no
                       longer help at all; f = 0)

Real-world basis for these thresholds: free cooling begins once ambient is ~1°C
below the return temperature and reaches 100% once ambient is ~5-9°F (3-5°C)
below the chilled setpoint (Daikin Applied "Is Free Cooling Right for My
Application"; Chiller & Cooling Best Practices free-cooling fundamentals). The
approach-shifted return/supply thresholds above reproduce exactly that band.

Effective COP (energy-correct blend)
------------------------------------
Fraction f of the cooling is delivered by free cooling at free_cooling_cop;
the remaining (1-f) by the mechanical chiller at its air-cooled COP (reusing
chiller.py's real chiller_cop() directly — physically this IS an air-cooled
chiller when the compressor runs). The electricity per unit cooling is the
load-weighted sum of the two inverse COPs, so:

    effective_COP = 1 / ( f/free_cooling_cop + (1-f)/mechanical_COP )

free_cooling_cop = 20: dry-cooler fans + glycol pump draw ~4-6% of the cooling
they move (a ~0.05 kW/kW parasitic), i.e. COP ~17-25; 20 is a mid value.

No tower, no water: a dry cooler is a sealed glycol-to-air coil, so unlike
water_cooled_chiller.py there is NO evaporative water OPEX — its only running
cost is electricity, which is exactly why it is cheap on cold hours.

CAPEX: £150,000/MW — an air-cooled chiller (£130k/MW in chiller.py) plus the
free-cooling coil, glycol charge and changeover valves, which manufacturers
price at roughly a 10-20% premium on the base unit.

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
from components.ASHP import _ashp_unit_outage_profile
# The mechanical half of this unit IS an air-cooled chiller — reuse chiller.py's
# real, anchored COP curve and its high-ambient capacity derate directly rather
# than maintaining a second copy.
from components.chiller import (
    chiller_cop, _capacity_derate_hot, RATING_POINT_AMBIENT_C,
)
from components.cooling_common import N_HOURS


# ── Free-cooling model ─────────────────────────────────────────────────────────

def free_cooling_fraction(
    T_ambient_C: np.ndarray,
    cool_flow_temp_C: float,
    cool_return_temp_C: float,
    dry_cooler_approach_C: float = 3.0,
) -> np.ndarray:
    """Share of the cooling load the dry cooler can carry each hour (0..1).
    See module docstring for the derivation and the real threshold sourcing."""
    full_free_ambient = cool_flow_temp_C - dry_cooler_approach_C
    no_free_ambient = cool_return_temp_C - dry_cooler_approach_C
    if no_free_ambient <= full_free_ambient:
        raise ValueError(
            "cool_return_temp_C must exceed cool_flow_temp_C — otherwise the "
            "free-cooling band collapses (return must be warmer than supply)."
        )
    T = np.asarray(T_ambient_C, dtype=float)
    return np.clip((no_free_ambient - T) / (no_free_ambient - full_free_ambient), 0.0, 1.0)


def free_cooling_effective_cop(
    T_ambient_C: np.ndarray,
    T_chilled_water_C: float,
    cool_flow_temp_C: float,
    cool_return_temp_C: float,
    dry_cooler_approach_C: float = 3.0,
    free_cooling_cop: float = 20.0,
) -> tuple:
    """
    Returns (effective_cop, free_fraction, mechanical_cop) at every hour.
    effective_cop blends the free-cooling and mechanical COPs by cooling load
    (energy-correct harmonic blend — see module docstring).
    """
    f = free_cooling_fraction(T_ambient_C, cool_flow_temp_C, cool_return_temp_C, dry_cooler_approach_C)
    mech_cop = chiller_cop(T_ambient_C, T_chilled_water_C)
    inv = f / free_cooling_cop + (1.0 - f) / mech_cop
    effective = 1.0 / inv
    return effective, f, mech_cop


# ── Generic presets ──────────────────────────────────────────────────────────

FREE_COOLING_CHILLER_PRESETS = {
    "generic_500kW": {
        "description":          "Generic 500kW free-cooling (dry-cooler/glycol) chiller",
        "n_units":               1,
        "unit_capacity_MW":      0.5,
        "chilled_water_temp_C":  7.0,
        "cool_return_temp_C":    12.0,
        "max_ambient_design_C":  40.0,
        "reference":            "Generic — no project-specific free-cooling data sourced yet",
    },
    "generic_2MW_bank": {
        "description":          "Generic 2MW free-cooling chiller bank (4x500kW)",
        "n_units":               4,
        "unit_capacity_MW":      0.5,
        "chilled_water_temp_C":  7.0,
        "cool_return_temp_C":    12.0,
        "max_ambient_design_C":  40.0,
        "reference":            "Generic — no project-specific free-cooling data sourced yet",
    },
}


# ── FreeCoolingChiller class ──────────────────────────────────────────────────

class FreeCoolingChiller:
    """
    A generalised array of N identical free-cooling (dry-cooler/glycol) chiller
    units. Same public surface and n_units x unit_capacity_MW scaling as
    AirCooledChiller (chiller.py); the difference is the hourly COP, which
    reflects free-cooling availability from the ambient dry-bulb.

    Extra parameters over AirCooledChiller:
      cool_return_temp_C     : chilled-water RETURN temperature (°C) — sets, with
                               the supply temp and the dry-cooler approach, how
                               warm the ambient can be and still give free cooling.
      dry_cooler_approach_C  : how close to ambient the dry cooler brings the
                               glycol (~3°C).
      free_cooling_cop       : effective COP when the compressor is off (fans +
                               glycol pump only), default 20.
    """

    source_type = "free_cooling_chiller"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        chilled_water_temp_C: float          = 7.0,
        cool_return_temp_C: float             = 12.0,
        dry_cooler_approach_C: float          = 3.0,
        free_cooling_cop: float               = 20.0,
        weather_df: Optional[pd.DataFrame]    = None,
        max_ambient_design_C: float           = 40.0,
        min_capacity_fraction: float          = 0.80,
        electricity_price_GBP_per_MWh         = None,
        capex_GBP_per_MW: float                = 150_000.0,
        availability_factor: float             = 0.97,
        seed: int                              = 11,
        reference: str                         = "",
    ):
        if weather_df is None:
            raise ValueError(
                "FreeCoolingChiller requires weather_df (must have "
                "'temp_drybulb_C' column, 8760 rows) — free-cooling availability "
                "is weather-dependent."
            )
        if len(weather_df) != N_HOURS:
            raise ValueError(f"weather_df must have {N_HOURS} rows; got {len(weather_df)}.")

        self.name                    = name
        self.n_units                  = int(n_units)
        self.unit_capacity_MW         = float(unit_capacity_MW)
        self.capacity_MW              = self.n_units * self.unit_capacity_MW
        self.chilled_water_temp_C      = float(chilled_water_temp_C)
        self.cool_return_temp_C        = float(cool_return_temp_C)
        self.dry_cooler_approach_C     = float(dry_cooler_approach_C)
        self.free_cooling_cop          = float(free_cooling_cop)
        self.max_ambient_design_C      = float(max_ambient_design_C)
        self.min_capacity_fraction     = float(min_capacity_fraction)
        self.capex_GBP_per_MW          = float(capex_GBP_per_MW)
        self.availability_factor       = float(availability_factor)
        self.seed                      = int(seed)
        self.reference                 = reference

        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air

        # Effective (blended) COP, and the free-cooling fraction, each hour
        self.cop_hourly, self.free_fraction, self._mech_cop = free_cooling_effective_cop(
            T_air, self.chilled_water_temp_C,
            cool_flow_temp_C=self.chilled_water_temp_C,
            cool_return_temp_C=self.cool_return_temp_C,
            dry_cooler_approach_C=self.dry_cooler_approach_C,
            free_cooling_cop=self.free_cooling_cop,
        )

        # Capacity derate — the mechanical chiller's high-ambient derate (from
        # chiller.py). Free cooling only ever happens on COLD hours, where the
        # derate is 1.0, so it correctly leaves free-cooling capacity untouched.
        self._capacity_fraction = _capacity_derate_hot(
            T_air, rating_point_C=RATING_POINT_AMBIENT_C,
            max_ambient_C=self.max_ambient_design_C,
            min_capacity_fraction=self.min_capacity_fraction,
        )

        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = (
            self.units_available / self.n_units if self.n_units > 0 else np.ones(N_HOURS)
        )

        self.supply_MW = (
            self.capacity_MW * self._capacity_fraction * self._unit_availability_fraction
        )
        self.supply_temp_C = np.full(N_HOURS, self.chilled_water_temp_C)

        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)
        # No water OPEX — a dry cooler does not evaporate water.
        self.marginal_cost = self._elec_price / self.cop_hourly
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly

    # ── constructors / helpers ─────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset_key: str, weather_df: pd.DataFrame, **overrides) -> "FreeCoolingChiller":
        if preset_key not in FREE_COOLING_CHILLER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(FREE_COOLING_CHILLER_PRESETS.keys())}"
            )
        params = FREE_COOLING_CHILLER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(cls, config: dict, weather_df: pd.DataFrame) -> "FreeCoolingChiller":
        params = config.copy()
        tariff_block = params.pop("electricity_tariff", None)
        if tariff_block is not None:
            params["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_block)
        return cls(weather_df=weather_df, **params)

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        return FreeCoolingChiller(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            chilled_water_temp_C=self.chilled_water_temp_C,
            cool_return_temp_C=self.cool_return_temp_C,
            dry_cooler_approach_C=self.dry_cooler_approach_C,
            free_cooling_cop=self.free_cooling_cop,
            weather_df=pd.DataFrame({"temp_drybulb_C": self.ambient_temp_C}),
            max_ambient_design_C=self.max_ambient_design_C,
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
            "cool_return_temp_C":         self.cool_return_temp_C,
            "free_cooling_cop":           self.free_cooling_cop,
            "hours_with_free_cooling":    int((self.free_fraction > 0).sum()),
            "hours_full_free_cooling":    int((self.free_fraction >= 0.999).sum()),
            "mean_free_fraction":         round(float(self.free_fraction.mean()), 3),
            "cop_mean":                   round(float(self.cop_hourly.mean()), 2),
            "cop_min":                    round(float(self.cop_hourly.min()), 2),
            "cop_max":                    round(float(self.cop_hourly.max()), 2),
            "annual_cooling_available_MWh": round(float(self.supply_MW.sum()), 0),
            "annual_electrical_demand_MWh": round(float(self.electrical_demand_MW.sum()), 0),
            "seasonal_avg_cop":           round(
                float(self.supply_MW.sum() / self.electrical_demand_MW.sum()), 2
            ),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "max_ambient_design_C":       self.max_ambient_design_C,
            "availability_factor":        self.availability_factor,
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"FreeCoolingChiller(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"T_chw={self.chilled_water_temp_C:.0f}°C, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )
