"""
chiller.py
==============
Air-Cooled Chiller cooling source model for the district energy system.

Mechanically, a chiller IS an ASHP, reversed: same vapour-compression
cycle, same compressor/condenser/evaporator hardware family — the only
real difference is which side of the cycle does useful work. An ASHP
extracts heat from cold ambient air and rejects it (at a lift) into a
hot flow circuit; a chiller extracts heat from a warm chilled-water
circuit and rejects it (at a lift) into hot ambient air. This module
deliberately mirrors ASHP.py's structure (same COP-curve shape, same
capacity-derate shape, same per-unit outage model) with the physics
correctly reversed, rather than building an unrelated model from
scratch — see each function's docstring for exactly what's mirrored
and what's genuinely different.

COP methodology
----------------
COP = a + b*dT + c*dT^2          where dT = T_ambient - T_chilled_water
(NOTE the sign: for ASHP, dT = T_flow - T_ambient, i.e. ambient is the
SOURCE being extracted from; for a chiller, ambient is the SINK being
rejected into, so dT is ambient MINUS the cold side, not the other way
round. Both forms have the same physical meaning: dT is the "lift" the
compressor has to work against, and COP falls as dT grows, in both cases.)

Quadratic fitted to two REAL data points plus a real linear sensitivity
rule, the same fitting philosophy as ASHP.py's Ruhnau regression (fit to
real data, not constructed from a theoretical Carnot curve or guessed
at from a single anchor point):

  - Anchor 1 (full load / hot ambient): a real, named 680kW air-cooled
    chiller (R-134A, 12C inlet / 7C outlet chilled water) reported in
    REHVA Journal "Improved energy efficiency of air cooled chillers" —
    EER minimum of 4.0 at ~35C ambient (the standard AHRI 550/590 /
    BS EN 14511 rating condition, also independently confirmed by CIBSE
    Journal Module 132). At T_chilled=7C, dT = 35-7 = 28C, COP=4.0.
  - Anchor 2 (part load / cold ambient): the SAME real chiller's
    reported EER of 6-7 (midpoint 6.75) during Nov-Mar operation at
    ~60% load. A representative cold-ambient point of 10C gives
    dT = 10-7 = 3C, COP=6.75.
  - Slope constraint at anchor 1: a real, independently-cited linear
    sensitivity rule (ChillerOne, "floating head pressure control" —
    "every 1C decrease in condensing temperature reduces compressor
    power consumption by ~2-3%") gives d(COP)/d(dT) at dT=28 via
    d(COP)/COP = -d(Power)/Power at fixed cooling output, using the
    2.5% midpoint of the cited 2-3% range.

This gives a real-data-anchored curve across the whole operating range,
rather than one anchor point and a borrowed curve SHAPE (the honest
limitation flagged in this project's first attempt at a chiller curve,
before this module existed) — still a constructed fit, not an
independently published manufacturer curve, but now anchored at TWO
real points plus a real measured slope, not one point plus an assumed shape.

Two additional real-world corrections, mirroring ASHP.py's structure:

1. NO DEFROST PENALTY — defrost is specifically an extract-heat-FROM-
   cold-air problem (ice forms on the outdoor coil when pulling heat out
   of cold, humid air). A chiller's outdoor coil is REJECTING heat INTO
   ambient air, never extracting it — there's no equivalent icing
   mechanism. This is a genuine physical difference from ASHP, not an
   oversight.

2. CAPACITY DERATING AT HIGH AMBIENT (not low) — chiller thermal output
   falls as ambient temperature RISES (the mirror image of ASHP's
   low-ambient derate). Real sourcing: multiple manufacturer/industry
   sources (LNEYA, ATC, VRCoolerTech — see _capacity_derate_hot()
   docstring) agree standard air-cooled units see meaningful capacity
   loss starting around 40C ambient, with units typically specified to
   a rated ambient ceiling comfortably above the worst-case site
   temperature. Modelled as a linear derate above 35C (the AHRI rating
   point, where capacity is still 100%) up to a design ceiling.

CAPEX methodology
------------------
£130,000/MW — real-sourcing note: the previous £100,000/MW figure was
derived from a 2012 Trane EQUIPMENT price list (~$450/ton, consistent
with Enersion's independently cited "~$450/ton above 150-tons" figure)
with no inflation or installation markup applied — that number is now
stale. This revised figure uses two current (2025), fully-installed,
large-chiller worked examples from thecoolingco.com (a commercial
chiller cost-analysis site publishing itemised quote breakdowns): a
200-ton air-cooled package at $110,000-125,000 installed (midpoint
$117,500 = $587.5/ton) and a 1,000-ton high-efficiency unit at
$590,000-640,000 installed (midpoint $615,000 = $615/ton). Averaging
those two large-scale, fully-installed data points gives ≈$601/ton =
$170.9/kW (at the standard 3.517 kW/ton refrigeration factor),
converted at ≈0.7457 USD/GBP (mid-2026) = ≈£127,400/MW, rounded to
£130,000/MW. Still notably CHEAPER per MW than ASHP's £770,000/MW —
which makes physical sense: a chiller is mechanically simpler (no
cold-climate-rated compressor, no defrost cycle/controls, lower maximum
pressure ratio than an ASHP designed for sub-zero heating duty) — but
the gap is now honestly narrower than the old (uninflated,
equipment-only) figures implied.

Generalised array design
-------------------------
Mirrors ASHPArray exactly: AirCooledChiller represents n_units x
unit_capacity_MW, with the same staggered per-unit outage model
(see ASHP.py's _ashp_unit_outage_profile() — reused here directly,
not reimplemented, since the real maintenance-scheduling logic is
identical regardless of whether the unit heats or cools).

Usage
-----
    from components.chiller import AirCooledChiller

    chiller = AirCooledChiller.from_preset("generic_500kW", weather_df)
    print(chiller.capacity_MW)            # cooling capacity, MW
    print(chiller.cop_hourly[:24])        # First day's COP profile
    print(chiller.supply_MW[:24])         # First day's available cooling output
    print(chiller.electrical_demand_MW[:24])  # Electricity consumed to deliver that cooling
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

# Reuse ASHP.py's real, tested per-unit outage model directly — the
# real-world maintenance-scheduling logic (units serviced one at a time,
# staggered across the year) is identical regardless of whether the unit
# heats or cools; reimplementing it here would just be a second, possibly-
# drifting copy of the same real logic.
from components.ASHP import _ashp_unit_outage_profile


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# Standard AHRI 550/590 / BS EN 14511 air-cooled chiller rating condition
# — the ambient temperature at which manufacturers quote "rated capacity"
# and "rated COP/EER" on datasheets. Mirrors ASHP.py's RATING_POINT_TEMP_C
# (7°C for ASHP, the EN14825 standard) — same CONCEPT, different real
# standard and different real number, since heating and cooling rating
# conventions are genuinely different standards.
RATING_POINT_AMBIENT_C = 35.0

# Real COP curve, fitted to two real anchor points + one real measured
# slope (see module docstring for full sourcing):
#   COP = COP_FIT_A + COP_FIT_B*dT + COP_FIT_C*dT^2
#   where dT = T_ambient - T_chilled_water_C
COP_FIT_A = 7.1136
COP_FIT_B = -0.1224
COP_FIT_C = 0.000400


# ── Generic presets ──────────────────────────────────────────────────────────
# No real, named, project-specific chiller exists yet in this project's
# source documents (unlike ASHP/EfW/DataCentre, which all trace back to
# the real Ealing report) — these are generic, round-number presets for
# getting started, not claimed to be site-specific. Replace with real
# project figures once a cooling demand/site assessment exists.

CHILLER_PRESETS = {
    "generic_500kW": {
        "description":       "Generic 500kW air-cooled chiller (single unit)",
        "n_units":            1,
        "unit_capacity_MW":   0.5,
        "chilled_water_temp_C": 7.0,   # standard BS EN 14511 rating condition
        "max_ambient_design_C": 40.0,
        "reference":         "Generic — no project-specific chiller data sourced yet",
    },
    "generic_2MW_bank": {
        "description":       "Generic 2MW air-cooled chiller bank (4x500kW)",
        "n_units":            4,
        "unit_capacity_MW":   0.5,
        "chilled_water_temp_C": 7.0,
        "max_ambient_design_C": 40.0,
        "reference":         "Generic — no project-specific chiller data sourced yet",
    },
    "low_temp_4C": {
        "description":       "Lower chilled water temperature variant (4°C — e.g. for a tighter dehumidification duty)",
        "n_units":            1,
        "unit_capacity_MW":   0.5,
        "chilled_water_temp_C": 4.0,
        "max_ambient_design_C": 40.0,
        "reference":         "Generic — no project-specific chiller data sourced yet",
    },
}


# ── COP model ──────────────────────────────────────────────────────────────────

def chiller_cop(
    T_ambient_C: np.ndarray,
    T_chilled_water_C: float,
    cop_floor: float = 1.5,
    cop_ceiling: float = 8.0,
) -> np.ndarray:
    """
    Air-cooled chiller COP at every hour — see module docstring for the
    full real-data-anchored fitting methodology.

    Parameters
    ----------
    T_ambient_C        : hourly ambient air temperature array (°C) — the
                  SINK the chiller rejects heat into (mirrors ASHP's
                  T_ambient_C, which is the SOURCE it extracts from)
    T_chilled_water_C   : chilled water supply temperature (°C) — assumed
                  constant (real networks may reset this with load, but
                  this is the same fixed-design-point simplification
                  used throughout this project, e.g. ASHP's fixed flow
                  temp — see ASHP.py's own note on why operational
                  compensation is currently out of scope for this
                  feasibility-stage model)
    cop_floor          : minimum physically realistic COP (very hot
                  ambient, e.g. >45C, where a real unit would more likely
                  trip on high-pressure protection than keep running at
                  a very low but stable COP — this floor is a modelling
                  safety net, not a claim that operation stays stable
                  down to this value)
    cop_ceiling        : maximum COP cap (prevents unrealistic values at
                  very small dT, e.g. a mild night with high chilled
                  water temp)

    Returns
    -------
    np.ndarray of hourly COP values, same length as T_ambient_C
    """
    T = np.asarray(T_ambient_C, dtype=float)
    dT = T - T_chilled_water_C   # NOTE: reversed vs ASHP's dT = T_flow - T_ambient
    cop = COP_FIT_A + COP_FIT_B * dT + COP_FIT_C * dT ** 2
    return np.clip(cop, cop_floor, cop_ceiling)


# ── Capacity derating (HIGH ambient, mirror of ASHP's LOW-ambient derate) ──────

def _capacity_derate_hot(
    T_ambient_C: np.ndarray,
    rating_point_C: float = RATING_POINT_AMBIENT_C,
    max_ambient_C: float = 40.0,
    min_capacity_fraction: float = 0.80,
) -> np.ndarray:
    """
    Chiller thermal (cooling) output capacity falls at HIGH ambient
    temperature — the mirror image of ASHP's _capacity_derate(), which
    falls at LOW ambient. Real sourcing: multiple industry/manufacturer
    sources (LNEYA "How Ambient Temperature Affects Chiller
    Performance"; ATC "How Does Ambient Temperature Affect Chiller
    Performance?") agree that standard air-cooled units see meaningful
    capacity loss starting somewhere above ~40°C ambient, and that units
    should be specified with a rated ambient ceiling comfortably above
    the worst-case site temperature — i.e. derating is a real, expected
    mechanism for operation BEYOND a unit's design ceiling, not normal
    behaviour within it.

    Modelled as a linear interpolation:
      - At rating_point_C (35°C, the AHRI/BS EN 14511 standard) and
        below: 100% capacity
      - At max_ambient_C (the design ceiling): min_capacity_fraction of
        rated capacity
      - Linear between those two points
      - Above max_ambient_C: held at min_capacity_fraction (mirrors
        ASHP's behaviour below ITS min_ambient_C — most real units
        maintain SOME output at the extreme rather than cutting out
        entirely, though a real high-pressure trip is also a genuine
        possibility this simplified model doesn't capture)

    Parameters
    ----------
    min_capacity_fraction : fraction of rated capacity retained at the
                  design ambient ceiling. 0.80 is a reasonable default —
                  somewhat less severe than ASHP's 0.65 low-ambient
                  derate, since a chiller's design ceiling is normally
                  chosen with real headroom above typical climate
                  extremes (see the "specify comfortably above worst-
                  case" sourcing note above), whereas ASHP cold-climate
                  operation is more often pushed close to its real limit
                  by UK winter design conditions.
    """
    if max_ambient_C <= rating_point_C:
        raise ValueError(
            f"max_ambient_design_C ({max_ambient_C}) must be greater than "
            f"the rating point ({rating_point_C}) — otherwise the derate "
            f"range is zero or negative, which would silently divide by "
            f"zero below. This exact bug occurred once already in this "
            f"project's first chiller attempt — guarded against here "
            f"explicitly so it can't happen silently again."
        )

    T = np.asarray(T_ambient_C, dtype=float)

    # Below rating point: full capacity
    frac = np.ones_like(T)

    # Linear derate zone (mirrors ASHP's derate zone, but ABOVE the
    # rating point instead of below it)
    in_derate_zone = T > rating_point_C
    derate_range = max_ambient_C - rating_point_C
    derate_progress = np.clip(
        (T - rating_point_C) / derate_range, 0, 1
    )
    derated_frac = 1.0 - (1.0 - min_capacity_fraction) * derate_progress

    frac = np.where(in_derate_zone, derated_frac, frac)

    return frac


# ── AirCooledChiller class ───────────────────────────────────────────────────

class AirCooledChiller:
    """
    A generalised array of N identical air-cooled chiller units —
    structurally parallel to ASHPArray (see module docstring for the
    full "chiller IS an ASHP, reversed" framing).

    Scale the system by changing n_units and/or unit_capacity_MW — same
    modular philosophy as ASHPArray and DataCentre.

    Parameters
    ----------
    name                   : descriptive name for reporting
    n_units                 : number of identical chiller units in the array
    unit_capacity_MW        : rated cooling output per unit at the AHRI/
                  BS EN 14511 standard rating point (35°C ambient) (MW)
    chilled_water_temp_C     : chilled water SUPPLY temperature (°C) —
                  the cold side the chiller maintains. Fixed (not load-
                  reset) — see chiller_cop()'s own note on why.
                  Typical UK comfort-cooling network: 6-7°C
    weather_df               : EPW weather DataFrame (must have
                  'temp_drybulb_C' column, 8760 rows) — chiller output IS
                  weather-dependent (mirrors ASHPArray's requirement)
    max_ambient_design_C     : design ambient ceiling for capacity
                  derating (°C) — see _capacity_derate_hot() above
    min_capacity_fraction    : fraction of rated capacity at
                  max_ambient_design_C — see _capacity_derate_hot() above
    electricity_price_GBP_per_MWh : accepts None (default realistic
                  tariff shape), an ElectricityTariff object, a flat
                  scalar override, or an 8760-length array — identical
                  contract to ASHPArray, via the same
                  economics.tariffs.resolve_electricity_price()
    capex_GBP_per_MW         : capital cost per MW installed. Default
                  £130,000/MW — see module docstring for the real,
                  current (2025) fully-installed large-chiller sourcing.
                  Notably much cheaper per MW than ASHPArray's
                  £770,000/MW default — a real difference, not an
                  inconsistency (see module docstring for why).
    availability_factor     : fleet-average fraction of time each UNIT is
                  available (not in maintenance). Default 0.97, same as
                  ASHPArray — reuses _ashp_unit_outage_profile() directly,
                  same staggered one-unit-at-a-time real O&M logic.
    seed                     : random seed for the outage schedule
    """

    source_type = "air_cooled_chiller"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        chilled_water_temp_C: float          = 7.0,
        weather_df: Optional[pd.DataFrame]    = None,
        max_ambient_design_C: float           = 40.0,
        min_capacity_fraction: float          = 0.80,
        electricity_price_GBP_per_MWh         = None,
        capex_GBP_per_MW: float                = 130_000.0,
        availability_factor: float             = 0.97,
        seed: int                              = 11,
        reference: str                         = "",
    ):
        if weather_df is None:
            raise ValueError(
                "AirCooledChiller requires weather_df (must have "
                "'temp_drybulb_C' column, 8760 rows) — chiller output is "
                "weather-dependent, mirroring ASHPArray's requirement."
            )
        if len(weather_df) != N_HOURS:
            raise ValueError(
                f"weather_df must have {N_HOURS} rows; got {len(weather_df)}."
            )
        if max_ambient_design_C <= RATING_POINT_AMBIENT_C:
            raise ValueError(
                f"max_ambient_design_C ({max_ambient_design_C}) must exceed "
                f"the rating point ({RATING_POINT_AMBIENT_C}°C) — see "
                f"_capacity_derate_hot()'s guard for why this matters."
            )

        self.name                    = name
        self.n_units                  = int(n_units)
        self.unit_capacity_MW         = float(unit_capacity_MW)
        self.capacity_MW              = self.n_units * self.unit_capacity_MW
        self.chilled_water_temp_C      = float(chilled_water_temp_C)
        self.max_ambient_design_C      = float(max_ambient_design_C)
        self.min_capacity_fraction     = float(min_capacity_fraction)
        self.capex_GBP_per_MW          = float(capex_GBP_per_MW)
        self.availability_factor       = float(availability_factor)
        self.seed                      = int(seed)
        self.reference                 = reference

        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air

        # COP at every hour
        self.cop_hourly = chiller_cop(T_air, self.chilled_water_temp_C)

        # Capacity derating at every hour — HIGH-ambient derate, mirror
        # of ASHP's low-ambient derate
        self._capacity_fraction = _capacity_derate_hot(
            T_air,
            rating_point_C=RATING_POINT_AMBIENT_C,
            max_ambient_C=self.max_ambient_design_C,
            min_capacity_fraction=self.min_capacity_fraction,
        )

        # Units available at each hour — maintenance-driven, reusing
        # ASHP.py's real per-unit outage model directly (see module
        # docstring: the real O&M logic is identical regardless of
        # heating/cooling duty)
        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = (
            self.units_available / self.n_units if self.n_units > 0 else np.ones(N_HOURS)
        )

        # Available cooling supply at each hour (MW)
        self.supply_MW = (
            self.capacity_MW * self._capacity_fraction * self._unit_availability_fraction
        )

        # Supply temperature is the chilled water temperature the
        # chiller maintains — constant (fixed-design-point simplification,
        # see chilled_water_temp_C docstring note above)
        self.supply_temp_C = np.full(N_HOURS, self.chilled_water_temp_C)

        # Electricity price — None / Tariff / scalar / array, all resolved
        # to a clean 8760 £/MWh array, identical contract to ASHPArray
        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)

        # Marginal cost of cooling delivered (£/MWh_cooling) = elec_price / COP
        self.marginal_cost = self._elec_price / self.cop_hourly

        # Carbon intensity per unit COOLING delivered (kgCO2e/kWh_cooling)
        # = grid carbon intensity / COP — identical formula and identical
        # sourcing to ASHP's carbon_intensity_kgCO2_per_kWh (the
        # electricity is the SAME grid electricity either way; only what
        # it's used for differs). Varies hourly because COP varies with
        # ambient temperature, same mechanism as ASHP, opposite direction:
        # a HOT day with poor chiller COP is BOTH more expensive AND more
        # carbon-intensive per unit cooling, same root cause as ASHP's
        # cold-day equivalent.
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly

        # Electrical demand IF running at full available supply (MW_elec)
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        weather_df: pd.DataFrame,
        **overrides,
    ) -> "AirCooledChiller":
        """
        Construct an AirCooledChiller from a named preset (see
        CHILLER_PRESETS dict). Mirrors ASHPArray.from_preset() exactly.

        Example
        -------
            chiller = AirCooledChiller.from_preset("generic_500kW", weather_df)
            chiller = AirCooledChiller.from_preset(
                "generic_500kW", weather_df, chilled_water_temp_C=6.0)  # override
        """
        if preset_key not in CHILLER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(CHILLER_PRESETS.keys())}"
            )

        params = CHILLER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(
        cls,
        config: dict,
        weather_df: pd.DataFrame,
    ) -> "AirCooledChiller":
        """
        Construct an AirCooledChiller from a plain dict (e.g. parsed from
        YAML/JSON config) — mirrors ASHPArray.from_config() exactly,
        including the nested electricity_tariff block support.

        Example
        -------
            config = {
                "name": "Town centre chiller bank",
                "n_units": 4,
                "unit_capacity_MW": 0.5,
                "chilled_water_temp_C": 7.0,
                "electricity_tariff": {"negotiated_discount_pct": 15.0},
            }
            chiller = AirCooledChiller.from_config(config, weather_df)
        """
        params = config.copy()
        tariff_block = params.pop("electricity_tariff", None)
        if tariff_block is not None:
            params["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_block)
        return cls(weather_df=weather_df, **params)

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        """
        Return a NEW AirCooledChiller with a different scale, reusing
        all other parameters from this instance. Does not mutate self.
        Mirrors ASHPArray.resize() exactly.

        Example
        -------
            chiller_small = AirCooledChiller.from_preset("generic_500kW", weather_df)
            chiller_big   = chiller_small.resize(n_units=8)
        """
        return AirCooledChiller(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            chilled_water_temp_C=self.chilled_water_temp_C,
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
        """Return key parameters and performance stats as a dict. Mirrors ASHPArray.summary()."""
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "total_capacity_MW":          round(self.capacity_MW, 2),
            "chilled_water_temp_C":       self.chilled_water_temp_C,
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
            "annual_outage_hours_per_unit": int(round((1.0 - self.availability_factor) * N_HOURS)),
            "min_units_available":        int(self.units_available.min()),
            "hours_below_full_fleet":     int((self.units_available < self.n_units).sum()),
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"AirCooledChiller(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"T_chw={self.chilled_water_temp_C:.0f}°C, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )
