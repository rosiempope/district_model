"""
absorption_chiller.py
=====================
Absorption chiller — cooling driven by HEAT, not electricity. A fundamentally
different machine from the vapour-compression chillers (chiller.py,
water_cooled_chiller.py, free_cooling_chiller.py): instead of a compressor
lifting refrigerant with shaft power, an absorption cycle uses a
lithium-bromide/water (or water/ammonia) pair driven by a hot-water or steam
GENERATOR, with only small pumps and cooling-tower fans drawing electricity.

Why it belongs in a district-energy model: this project's heat stack includes an
Energy-from-Waste plant (components/EfW.py) whose heat export costs only ~£8/MWh
and which, in summer, has heat it would otherwise reject. An absorption chiller
turns that near-free waste heat into cooling, drawing almost no electricity — so
its economics live or die on the price of the driving heat, not the grid. That
is the whole point of including it, and the single most important assumption in
the module (heat_price_GBP_per_MWh) is exposed as a parameter for exactly that
reason.

Performance
-----------
thermal_cop (default 0.7): cooling delivered per unit of DRIVING HEAT. 0.7 is
the standard figure for a single-effect LiBr absorption chiller driven by ~90°C
hot water (a good match for an EfW/CHP hot-water circuit); single-effect COP maxes
around 0.85, double-effect (steam-driven) reaches 1.1-1.3 but needs higher-grade
heat than a hot-water district circuit provides. Single-effect is the honest
default here. (Sources: ScienceDirect "effect absorption chiller" overview;
Chiller & Cooling Best Practices "Busting Four Myths About Absorption Cooling".)

electric_parasitic_cop (default 25): cooling delivered per unit of ELECTRICITY
(solution/refrigerant pumps + cooling-tower fans). Absorption machines draw only
~3-5% of their cooling output in electricity, i.e. an electrical COP of ~20-30 —
an order of magnitude less than a vapour-compression chiller.

Heat rejection & water OPEX
---------------------------
An absorption chiller rejects MORE heat per unit cooling than a compression
chiller — it must dump the cooling PLUS all the driving heat — so it is almost
always paired with an evaporative cooling tower (wet-bulb rejection). That tower
carries the same make-up water + treatment OPEX as water_cooled_chiller.py, here
computed against the THERMAL COP (0.7), because the tower rejects cooling +
driving heat = cooling x (1 + 1/thermal_cop). See
cooling_common.cooling_tower_water_cost_GBP_per_MWh_cooling().

Marginal cost of cooling
------------------------
    marginal_cost = heat_price / thermal_cop          # driving heat
                    + elec_price / electric_parasitic_cop   # pumps + fans
                    + tower_water_cost                # make-up water + treatment
The heat term dominates and scales inversely with thermal_cop, so a cheap
(waste) heat source is what makes absorption competitive; at a market heat price
it is not.

Carbon
------
Deliberately NOT in the project's ELECTRIC_SOURCE_TYPES: its primary input is
heat, not grid electricity. Its carbon is the driving-heat carbon
(heat_carbon_intensity_kgCO2_per_kWh, default 0 — EfW heat that would otherwise
be dumped is treated as carbon-free at the margin, consistent with how this
project treats EfW/DC waste heat elsewhere) plus the tiny parasitic-electric
term. NOTE: the network-level cooling carbon roll-up in scenario_runner only
attributes carbon to electric and gas sources, so an absorption chiller's small
parasitic-electric carbon is not currently counted there — a known, documented
simplification, immaterial next to a waste-heat-driven unit's ~0 direct carbon.

CAPEX methodology
-----------------
£220,000/MW — absorption chillers cost materially more per MW of cooling than
electric chillers (the generator/absorber vessels are large, and a cooling tower
+ condenser pumps come on top). Industry comparisons put absorption capital at
roughly 1.5-2x an equivalent electric chiller; £220k/MW is ~1.5x this project's
air-cooled £130k/MW, a deliberately mid-conservative figure — absorption's case
is an OPEX (cheap heat) case, and should not be flattered on CAPEX.

Usage mirrors chiller.py's AirCooledChiller (same public surface), with the
heat-side parameters added.
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
from components.cooling_common import (
    N_HOURS, wet_bulb_temp_C, cooling_tower_water_cost_GBP_per_MWh_cooling,
)
# Reuse water_cooled_chiller's wet-bulb capacity derate + its rating wet-bulb:
# an absorption chiller shares the same cooling-tower heat-rejection side, so it
# derates with wet-bulb on the same (shallow, UK-rarely-binding) basis.
from components.water_cooled_chiller import _capacity_derate_wetbulb, RATING_WETBULB_C


# ── Generic presets ──────────────────────────────────────────────────────────

ABSORPTION_CHILLER_PRESETS = {
    "generic_1MW_efw": {
        "description":          "Generic 1MW single-effect absorption chiller on EfW waste heat",
        "n_units":               1,
        "unit_capacity_MW":      1.0,
        "chilled_water_temp_C":  7.0,
        "thermal_cop":           0.70,
        "heat_price_GBP_per_MWh": 12.0,   # cheap EfW waste heat (EfW export ≈ £8/MWh)
        "max_wetbulb_design_C":  28.0,
        "reference":            "Generic single-effect LiBr on EfW hot water — no project-specific unit sourced yet",
    },
    "generic_2MW_efw": {
        "description":          "Generic 2MW absorption chiller bank on EfW waste heat (2x1MW)",
        "n_units":               2,
        "unit_capacity_MW":      1.0,
        "chilled_water_temp_C":  7.0,
        "thermal_cop":           0.70,
        "heat_price_GBP_per_MWh": 12.0,
        "max_wetbulb_design_C":  28.0,
        "reference":            "Generic single-effect LiBr on EfW hot water — no project-specific unit sourced yet",
    },
}


# ── AbsorptionChiller class ───────────────────────────────────────────────────

class AbsorptionChiller:
    """
    A generalised array of N identical single-effect absorption chiller units on
    a shared cooling tower, driven by hot water (e.g. from the EfW/CHP). Same
    public surface and n_units x unit_capacity_MW scaling as AirCooledChiller.

    Heat-side parameters (the ones that make it an absorption chiller):
      thermal_cop              : cooling delivered per unit DRIVING HEAT (~0.7)
      heat_price_GBP_per_MWh   : price of the driving heat — THE key assumption
      electric_parasitic_cop   : cooling per unit ELECTRICITY (pumps + fans, ~25)
      heat_carbon_intensity_kgCO2_per_kWh : carbon of the driving heat
                                 (default 0 — waste heat treated carbon-free at
                                 the margin, as elsewhere in this project)
    """

    source_type = "absorption_chiller"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        chilled_water_temp_C: float          = 7.0,
        thermal_cop: float                    = 0.70,
        heat_price_GBP_per_MWh: float          = 12.0,
        electric_parasitic_cop: float          = 25.0,
        heat_carbon_intensity_kgCO2_per_kWh: float = 0.0,
        weather_df: Optional[pd.DataFrame]    = None,
        max_wetbulb_design_C: float           = 28.0,
        min_capacity_fraction: float          = 0.90,
        electricity_price_GBP_per_MWh         = None,
        capex_GBP_per_MW: float                = 220_000.0,
        availability_factor: float             = 0.95,
        seed: int                              = 11,
        reference: str                         = "",
    ):
        if weather_df is None:
            raise ValueError(
                "AbsorptionChiller requires weather_df (must have 'temp_drybulb_C' "
                "AND 'rel_humidity_pct' columns, 8760 rows) — its cooling tower "
                "rejects to the wet-bulb."
            )
        if len(weather_df) != N_HOURS:
            raise ValueError(f"weather_df must have {N_HOURS} rows; got {len(weather_df)}.")
        if "rel_humidity_pct" not in weather_df.columns:
            raise ValueError(
                "AbsorptionChiller needs a 'rel_humidity_pct' column to derive "
                "wet-bulb temperature for its cooling tower."
            )
        if not (0.0 < thermal_cop <= 1.5):
            raise ValueError("thermal_cop must be in (0, 1.5]; single-effect ≈0.7, double-effect ≈1.2.")

        self.name                    = name
        self.n_units                  = int(n_units)
        self.unit_capacity_MW         = float(unit_capacity_MW)
        self.capacity_MW              = self.n_units * self.unit_capacity_MW
        self.chilled_water_temp_C      = float(chilled_water_temp_C)
        self.thermal_cop               = float(thermal_cop)
        self.heat_price_GBP_per_MWh     = float(heat_price_GBP_per_MWh)
        self.electric_parasitic_cop     = float(electric_parasitic_cop)
        self.heat_carbon_intensity_kgCO2_per_kWh = float(heat_carbon_intensity_kgCO2_per_kWh)
        self.max_wetbulb_design_C       = float(max_wetbulb_design_C)
        self.min_capacity_fraction      = float(min_capacity_fraction)
        self.capex_GBP_per_MW           = float(capex_GBP_per_MW)
        self.availability_factor        = float(availability_factor)
        self.seed                       = int(seed)
        self.reference                  = reference

        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        RH = weather_df["rel_humidity_pct"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air
        self._rel_humidity_pct = RH
        self.wetbulb_temp_C = wet_bulb_temp_C(T_air, RH)

        # THERMAL COP is essentially flat for a single-effect machine over the
        # normal operating range — expose it as the hourly "cop_hourly" so the
        # component's public surface matches the other chillers.
        self.cop_hourly = np.full(N_HOURS, self.thermal_cop)

        # Capacity derate — shallow, wet-bulb-driven (shared with water-cooled)
        self._capacity_fraction = _capacity_derate_wetbulb(
            self.wetbulb_temp_C, rating_wetbulb_C=RATING_WETBULB_C,
            max_wetbulb_C=self.max_wetbulb_design_C,
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

        # Driving-HEAT demand to deliver the available cooling (MW_thermal)
        self.heat_demand_MW = self.supply_MW / self.thermal_cop
        # Parasitic ELECTRICITY demand (MW_elec) — pumps + tower fans only
        self.electrical_demand_MW = self.supply_MW / self.electric_parasitic_cop

        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)
        # Tower make-up water uses the THERMAL COP (rejects cooling + driving heat)
        self.water_cost_GBP_per_MWh = cooling_tower_water_cost_GBP_per_MWh_cooling(
            self.cop_hourly
        )
        # Marginal cost of cooling — driving heat dominates, plus parasitic
        # electricity, plus tower water. See module docstring for the formula.
        self.marginal_cost = (
            self.heat_price_GBP_per_MWh / self.thermal_cop
            + self._elec_price / self.electric_parasitic_cop
            + self.water_cost_GBP_per_MWh
        )

        # Carbon per unit cooling — driving-heat carbon (≈0 for waste heat) plus
        # the small parasitic-electric term. See module docstring's carbon note.
        self.carbon_intensity_kgCO2_per_kWh = (
            self.heat_carbon_intensity_kgCO2_per_kWh / self.thermal_cop
            + CARBON_INTENSITY["electric"] / self.electric_parasitic_cop
        )

    # ── constructors / helpers ─────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset_key: str, weather_df: pd.DataFrame, **overrides) -> "AbsorptionChiller":
        if preset_key not in ABSORPTION_CHILLER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(ABSORPTION_CHILLER_PRESETS.keys())}"
            )
        params = ABSORPTION_CHILLER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(cls, config: dict, weather_df: pd.DataFrame) -> "AbsorptionChiller":
        params = config.copy()
        tariff_block = params.pop("electricity_tariff", None)
        if tariff_block is not None:
            params["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_block)
        return cls(weather_df=weather_df, **params)

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        return AbsorptionChiller(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            chilled_water_temp_C=self.chilled_water_temp_C,
            thermal_cop=self.thermal_cop,
            heat_price_GBP_per_MWh=self.heat_price_GBP_per_MWh,
            electric_parasitic_cop=self.electric_parasitic_cop,
            heat_carbon_intensity_kgCO2_per_kWh=self.heat_carbon_intensity_kgCO2_per_kWh,
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
            "thermal_cop":                self.thermal_cop,
            "electric_parasitic_cop":     self.electric_parasitic_cop,
            "heat_price_GBP_per_MWh":     self.heat_price_GBP_per_MWh,
            "annual_cooling_available_MWh": round(float(self.supply_MW.sum()), 0),
            "annual_driving_heat_MWh":    round(float(self.heat_demand_MW.sum()), 0),
            "annual_parasitic_elec_MWh":  round(float(self.electrical_demand_MW.sum()), 0),
            "mean_water_cost_GBP_per_MWh": round(float(self.water_cost_GBP_per_MWh.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "mean_carbon_kgCO2_per_kWh":  round(float(np.mean(self.carbon_intensity_kgCO2_per_kWh)), 4),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "availability_factor":        self.availability_factor,
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"AbsorptionChiller(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"thermal COP={self.thermal_cop:.2f}, heat £{self.heat_price_GBP_per_MWh:.0f}/MWh, "
            f"marg £{self.marginal_cost.mean():.0f}/MWh)"
        )
