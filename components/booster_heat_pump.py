"""
booster_heat_pump.py
======================
Water-to-water booster heat pump — lifts a stable, relatively warm
liquid source (e.g. DataCentre's 28-35C cooling loop) up to district
heating network temperature (e.g. 70C). This is a GENUINELY DIFFERENT
machine from ASHPArray, not the same model with a relabelled source —
see the design discussion this module came from for why ASHPArray's
air-source physics would be wrong in both directions if reused here.

Verifying "water-to-water" is the right choice for THIS project
--------------------------------------------------------------------
This matters: >80% of DCs globally use air-side cooling (CRAC/CRAH
units), not direct liquid/immersion cooling (Liu, Chen & Wei et al.,
"Data centre waste heat recovery" review, ScienceDirect 2025, citing a
market-share figure for air-side cooling — see that review's Table 2
for the full DC cooling-system taxonomy). Critically, that doesn't mean
the recoverable waste heat is AIR — air-side cooling still rejects heat
into a WATER (or refrigerant) loop somewhere in the chain before that
loop rejects to outdoor air; raw exhaust air is impractical to pipe for
heat recovery (low energy density). The same review's Table 2 gives
real temperature BANDS for each tap point in that chain:
  - CRAC "return warm water"      : 15-20°C  (lowest grade, biggest volume)
  - Air-to-liquid HE "return warm water" : 20-30°C
  - CRAC/CRAH "condenser coolant" : 40-50°C  (higher grade, smaller volume)
  - Liquid-side "return warm water" : 50-60°C (highest grade, smallest market share)

This project's actual DataCentre presets (see datacentre_source.py)
specify supply_temp_C in the 28-35°C range, with that module's own
comment confirming "Typical range: 0.05-0.15 [temperature sensitivity]
for AIR-COOLED systems" — i.e. this project's DC sources are ALREADY
modelling the air-to-liquid heat exchanger water loop (the 20-30°C
band above), not raw exhaust air, and not the higher-grade condenser-
coolant or liquid-cooling-system options. The cited Ealing report
source (datacentre_source.py's "Redwire DC, Ealing town centre" preset)
explicitly describes this as "cooling provision of up to 3.6 MW",
consistent with an offtake from existing DC cooling infrastructure
water loops, not a different, higher-temperature tap point. This
confirms water-to-water (not water-to-air, and not a higher-grade
booster sized for 40-60°C condenser coolant) is the right model for
THIS project's specific DC presets — but if a future project's DC
preset instead represents condenser-coolant or liquid-cooling-system
offtake (genuinely possible per the review's taxonomy above, and
worth checking explicitly for any NEW DC preset added later, not
assumed), the booster's real achievable COP would be meaningfully
BETTER than this module's curve predicts, since the source temperature
and lift would both be more favourable.

Why this isn't ASHPArray
--------------------------
ASHPArray's COP model (ashp_cop(), see ASHP.py) is built entirely
around extracting heat from OUTDOOR AIR: it includes a defrost penalty
(ice forming on the outdoor coil when pulling heat out of cold, humid
air — a genuine air-side-only mechanism) and a low-ambient capacity
derate (driven by UK winter design conditions). Neither applies here:

  - NO DEFROST PENALTY: the source is liquid water at 28-35C, never
    near freezing, never humid air passing over a cold coil. There is
    no icing mechanism on a water-to-water exchanger.
  - NO LOW-SOURCE-TEMP CAPACITY DERATE: DataCentre's cooling loop
    temperature is set by the DC's OWN cooling system design, not by
    UK weather — it doesn't have a "winter design minimum" the way
    outdoor air does. The source is stable by design (see
    datacentre_source.py's own supply_temp_C, which varies only
    mildly with weather, nowhere near as much as outdoor air does).
  - MUCH BETTER COP AT A SMALL LIFT, BUT NOT GUARANTEED AT A LARGE ONE:
    water-to-water heat exchange has no fan power overhead and a better
    heat transfer coefficient than air-to-refrigerant, and there's no
    frost-cycling efficiency loss. This module's fitted curve confirms
    that advantage clearly at SMALL lifts (e.g. <15K) — but at this
    project's actual operating point (a DC source around 28-35C lifted
    to a 70C network, a ~35-42K lift), this module's specific curve
    (anchored on a single real-world deployment data point at a 27.5K
    lift) does NOT outperform ASHP's real Ruhnau regression at the same
    lift. This is an honest finding about the limits of a single-anchor
    fit, not a claim that water-to-water is universally better — see
    tests/test_booster_heat_pump.py's controlled comparison for the
    actual numbers, and treat this curve's large-lift behaviour as a
    genuine, flagged uncertainty rather than a settled physical fact.

COP methodology
----------------
Uses the Carnot-efficiency-fraction method — a standard, general
heat-pump sizing approach (see e.g. 2G Energy's "Temperature options
for large heat pumps": COP_actual = COP_Carnot x carnot_efficiency,
where COP_Carnot = T_sink_K / (T_sink_K - T_source_K), and
carnot_efficiency is "in practice... between 45 and 65%" for well-
optimized industrial heat pumps).

This module's carnot_efficiency_fraction default (0.244) is fitted to
REAL-WORLD DEPLOYED-SYSTEM data, not best-case lab results — see
Velasolaris' "Data Center Heat Reuse" review of operational data
centre-to-district-heating projects: for a traditional 60-70C district
heating network supplied from typical 30-45C DC exit temperatures,
real measured COP is 2.5-3.5. Fitted at the midpoint (37.5C source,
65C sink, COP=3.0), this gives a genuinely real-world-grounded
efficiency fraction, deliberately more conservative than the 45-65%
figure quoted for "well-optimized industrial" units, and MUCH more
conservative than the 4.92 COP achieved in a peer-reviewed lab
prototype with an internal heat exchanger (Wang et al. 2024, ScienceDirect/
ORNL, "District heating utilizing waste heat of a data center:
High-temperature heat pumps") — that lab figure was checked against
this module's fitted curve and found NOT to fit well (this curve
underpredicts it by ~25%), which is itself informative: lab-prototype
performance with optimized heat exchangers is a genuinely different,
better regime than typical deployed commercial equipment, and a
feasibility-stage model should anchor on the latter, not the former.

No capacity derating is modelled — see "NO LOW-SOURCE-TEMP CAPACITY
DERATE" above. No outage model beyond the standard staggered per-unit
maintenance reused from ASHP.py (the same real O&M logic applies
regardless of which heat pump physics a unit uses).

Usage
-----
    from components.booster_heat_pump import BoosterHeatPump

    booster = BoosterHeatPump.from_preset("generic_2MW", source_temp_C_hourly=dc.supply_temp_C)
    print(booster.cop_hourly[:24])
    print(booster.electrical_demand_MW[:24])
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
# real-world maintenance-scheduling logic (units serviced one at a
# time, staggered across the year) is identical regardless of which
# heat pump physics a unit uses.
from components.ASHP import _ashp_unit_outage_profile


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# Carnot-efficiency-fraction, fitted to real-world DEPLOYED-system data
# (not best-case lab results) — see module docstring for the full
# sourcing note and the lab-prototype comparison that was checked and
# deliberately NOT used as the anchor.
CARNOT_EFFICIENCY_FRACTION = 0.244

COP_FLOOR = 1.5    # physically-sane lower bound (very large lift, e.g. fault condition)
COP_CEILING = 8.0  # physically-sane upper bound (very small lift)


# ── Generic presets ──────────────────────────────────────────────────────────
# No real, named, project-specific booster heat pump exists yet in this
# project's source documents (this component doesn't trace back to the
# Ealing report the way ASHP/EfW/DataCentre do) — these are generic,
# round-number presets sized to plausibly match DataCentre's own preset
# capacities, not claimed to be site-specific.

BOOSTER_PRESETS = {
    "generic_2MW": {
        "description":       "Generic 2MW water-to-water booster heat pump",
        "n_units":            2,
        "unit_capacity_MW":   1.0,
        "reference":         "Generic — no project-specific booster data sourced yet",
    },
    "generic_5MW": {
        "description":       "Generic 5MW water-to-water booster bank (sized for a larger DC source)",
        "n_units":            4,
        "unit_capacity_MW":   1.25,
        "reference":         "Generic — no project-specific booster data sourced yet",
    },
}


# ── COP model ──────────────────────────────────────────────────────────────────

def booster_cop(
    T_source_C: np.ndarray,
    T_sink_C,
    carnot_efficiency_fraction: float = CARNOT_EFFICIENCY_FRACTION,
    cop_floor: float = COP_FLOOR,
    cop_ceiling: float = COP_CEILING,
) -> np.ndarray:
    """
    Water-to-water booster heat pump COP — Carnot-efficiency-fraction
    method (see module docstring for the full real-data sourcing note).

        COP_Carnot = T_sink_K / (T_sink_K - T_source_K)
        COP_actual = COP_Carnot * carnot_efficiency_fraction

    Parameters
    ----------
    T_source_C   : hourly source (e.g. DC cooling loop) temperature
                  array (°C) — typically near-constant, see module
                  docstring on why this source doesn't have ASHP's
                  weather-driven variability
    T_sink_C     : sink (network flow) temperature (°C) — scalar (fixed
                  design temp) or an (N_HOURS,) array (e.g. if the
                  network side is itself weather-compensated; this
                  module doesn't assume either)
    carnot_efficiency_fraction : real-world fraction of theoretical
                  Carnot COP actually achieved — see module docstring
    cop_floor, cop_ceiling : physically-sane bounds, same purpose as
                  ASHP.py's/chiller.py's equivalent clips

    Returns
    -------
    np.ndarray of hourly COP values.
    """
    T_source_K = np.asarray(T_source_C, dtype=float) + 273.15
    T_sink_K = np.asarray(T_sink_C, dtype=float) + 273.15
    lift_K = T_sink_K - T_source_K
    if np.any(lift_K <= 0):
        raise ValueError(
            "Sink temperature must exceed source temperature everywhere (a booster "
            "heat pump only makes sense for a genuine lift) — check T_source_C/T_sink_C."
        )
    carnot_cop = T_sink_K / lift_K
    cop = carnot_cop * carnot_efficiency_fraction
    return np.clip(cop, cop_floor, cop_ceiling)


# ── BoosterHeatPump class ────────────────────────────────────────────────────

class BoosterHeatPump:
    """
    A generalised array of N identical water-to-water booster heat pump
    units — structurally parallel to ASHPArray and AirCooledChiller (see
    those modules for the same n_units x unit_capacity_MW scaling
    pattern), but with genuinely different physics (see module
    docstring's "Why this isn't ASHPArray" section).

    Parameters
    ----------
    name                   : descriptive name for reporting
    n_units                 : number of identical booster units
    unit_capacity_MW        : rated heat OUTPUT (sink side) per unit (MW)
    source_temp_C_hourly     : (N_HOURS,) array of source (e.g. DC loop)
                  temperature — pass e.g. a DataCentre instance's
                  supply_temp_C attribute directly
    sink_temp_C              : network flow temperature (°C) — scalar
                  (fixed design temp, this project's standard
                  simplification — see ASHP.py's own note on why
                  operational compensation is out of scope) or an
                  (N_HOURS,) array
    carnot_efficiency_fraction : real-world Carnot efficiency fraction
                  — see booster_cop() and the module docstring
    electricity_price_GBP_per_MWh : accepts None (default realistic
                  tariff), an ElectricityTariff object, a flat scalar,
                  or an 8760-length array — identical contract to
                  ASHPArray/AirCooledChiller
    capex_GBP_per_MW         : capital cost per MW installed. Kept at
                  £600,000/MW — originally borrowed from ASHPArray's
                  old default as an unresearched placeholder, but an
                  independent check now shows this actually sits
                  squarely inside the real published range for
                  water-to-water / excess-heat-source large heat
                  pumps: Vannoni et al. (2023), "Large size heat pumps
                  advanced cost functions..." (Energy 284, 129204)
                  gives water-sourced ≈€779/kW (≈£660/kW) and
                  excess-heat-sourced ≈€689/kW (≈£586/kW) — i.e.
                  £586,000-660,000/MW, bracketing this module's
                  £600,000/MW closely. Deliberately NOT changed to
                  match ASHPArray's revised £770,000/MW (that figure
                  is specifically for AIR-sourced units, a different,
                  costlier category — see Vannoni et al.'s own
                  source-type breakdown) — this value now has its own
                  independent real-data support, not a borrowed
                  number.
    availability_factor     : fleet-average fraction of time each unit
                  is available — default 0.97, same as ASHPArray,
                  reusing the same real per-unit outage model directly.
    seed                     : random seed for the outage schedule
    """

    source_type = "booster_heat_pump"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        source_temp_C_hourly: np.ndarray,
        sink_temp_C,
        source_heat_available_MW: Optional[np.ndarray] = None,
        source_heat_cost_GBP_per_MWh: float = 0.0,
        carnot_efficiency_fraction: float = CARNOT_EFFICIENCY_FRACTION,
        electricity_price_GBP_per_MWh = None,
        capex_GBP_per_MW: float = 600_000.0,
        availability_factor: float = 0.97,
        seed: int = 13,
        reference: str = "",
    ):
        source_temp_C_hourly = np.asarray(source_temp_C_hourly, dtype=float)
        if len(source_temp_C_hourly) != N_HOURS:
            raise ValueError(
                f"source_temp_C_hourly must have {N_HOURS} entries; got "
                f"{len(source_temp_C_hourly)}."
            )

        self.name = name
        self.n_units = int(n_units)
        self.unit_capacity_MW = float(unit_capacity_MW)
        self.capacity_MW = self.n_units * self.unit_capacity_MW
        self.source_temp_C = source_temp_C_hourly
        if source_heat_available_MW is None:
            self.source_heat_available_MW = np.full(N_HOURS, np.inf)
        else:
            source_available = np.asarray(source_heat_available_MW, dtype=float)
            if len(source_available) != N_HOURS:
                raise ValueError(
                    f"source_heat_available_MW must have {N_HOURS} entries; got "
                    f"{len(source_available)}."
                )
            if np.any(source_available < 0):
                raise ValueError("source_heat_available_MW cannot contain negative values.")
            self.source_heat_available_MW = source_available
        self.source_heat_cost_GBP_per_MWh = float(source_heat_cost_GBP_per_MWh)
        self.sink_temp_C = np.broadcast_to(sink_temp_C, N_HOURS).astype(float).copy()
        self.carnot_efficiency_fraction = float(carnot_efficiency_fraction)
        self.capex_GBP_per_MW = float(capex_GBP_per_MW)
        self.availability_factor = float(availability_factor)
        self.seed = int(seed)
        self.reference = reference

        # COP at every hour
        self.cop_hourly = booster_cop(
            self.source_temp_C, self.sink_temp_C,
            carnot_efficiency_fraction=self.carnot_efficiency_fraction,
        )

        # Units available at each hour — maintenance-driven, reusing
        # ASHP.py's real per-unit outage model directly (see module
        # docstring: the real O&M logic is identical regardless of
        # which heat pump physics a unit uses). NO weather-driven
        # capacity derate (see module docstring) — supply_MW only
        # varies with the outage schedule, not with source temperature.
        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = (
            self.units_available / self.n_units if self.n_units > 0 else np.ones(N_HOURS)
        )

        # Sink-side output is constrained by both booster availability and
        # recoverable source heat. Q_source = Q_sink * (1 - 1/COP).
        plant_available_MW = self.capacity_MW * self._unit_availability_fraction
        source_fraction = np.maximum(1.0 - 1.0 / self.cop_hourly, 1e-9)
        source_limited_sink_MW = self.source_heat_available_MW / source_fraction
        self.supply_MW = np.minimum(plant_available_MW, source_limited_sink_MW)

        # Supply temperature delivered to the network
        self.supply_temp_C = self.sink_temp_C.copy()

        # Electricity price — identical contract to ASHPArray/AirCooledChiller
        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)

        # Compressor electricity plus contracted low-grade source heat.
        self.marginal_cost = (
            self._elec_price / self.cop_hourly
            + self.source_heat_cost_GBP_per_MWh * source_fraction
        )

        # Carbon intensity per unit heat delivered (kgCO2e/kWh_heat) =
        # grid carbon intensity / COP — identical formula/sourcing to
        # ASHPArray and AirCooledChiller (same grid electricity, just a
        # different end use)
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly

        # Electrical demand IF running at full available supply (MW_elec)
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        source_temp_C_hourly: np.ndarray,
        sink_temp_C=70.0,
        **overrides,
    ) -> "BoosterHeatPump":
        """
        Construct a BoosterHeatPump from a named preset (see
        BOOSTER_PRESETS dict). Mirrors ASHPArray.from_preset()/
        AirCooledChiller.from_preset().

        Example
        -------
            booster = BoosterHeatPump.from_preset(
                "generic_2MW", source_temp_C_hourly=dc.supply_temp_C, sink_temp_C=70.0,
            )
        """
        if preset_key not in BOOSTER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. Available: {list(BOOSTER_PRESETS.keys())}"
            )
        params = BOOSTER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(
            source_temp_C_hourly=source_temp_C_hourly, sink_temp_C=sink_temp_C, **params,
        )

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        """Return a NEW BoosterHeatPump with a different scale, reusing all
        other parameters from this instance. Does not mutate self."""
        return BoosterHeatPump(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            source_temp_C_hourly=self.source_temp_C,
            sink_temp_C=self.sink_temp_C,
            source_heat_available_MW=self.source_heat_available_MW,
            source_heat_cost_GBP_per_MWh=self.source_heat_cost_GBP_per_MWh,
            carnot_efficiency_fraction=self.carnot_efficiency_fraction,
            electricity_price_GBP_per_MWh=self._elec_price,
            capex_GBP_per_MW=self.capex_GBP_per_MW,
            availability_factor=self.availability_factor,
            seed=self.seed,
            reference=self.reference,
        )

    def summary(self) -> dict:
        """Return key parameters and performance stats as a dict."""
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "total_capacity_MW":          round(self.capacity_MW, 2),
            "mean_source_temp_C":         round(float(self.source_temp_C.mean()), 1),
            "mean_sink_temp_C":           round(float(self.sink_temp_C.mean()), 1),
            "cop_mean":                   round(float(self.cop_hourly.mean()), 2),
            "cop_min":                    round(float(self.cop_hourly.min()), 2),
            "cop_max":                    round(float(self.cop_hourly.max()), 2),
            "annual_heat_available_MWh":  round(float(self.supply_MW.sum()), 0),
            "annual_electrical_demand_MWh": round(float(self.electrical_demand_MW.sum()), 0),
            "seasonal_avg_cop":           round(
                float(self.supply_MW.sum() / self.electrical_demand_MW.sum()), 2
            ),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "source_heat_cost_GBP_per_MWh": self.source_heat_cost_GBP_per_MWh,
            "annual_source_heat_used_at_full_output_MWh": round(
                float((self.supply_MW * (1.0 - 1.0 / self.cop_hourly)).sum()), 0
            ),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "availability_factor":        self.availability_factor,
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"BoosterHeatPump(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"source={self.source_temp_C.mean():.0f}°C -> sink={self.sink_temp_C.mean():.0f}°C, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )
