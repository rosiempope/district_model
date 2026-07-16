"""

Heat source models for the district energy system.

Each source is a class that produces an 8,760-hour supply array (MW available)
and associated metadata. All sources share the same interface so the dispatch
optimiser in dispatch.py can treat them identically regardless of technology.

Supported source types
----------------------
DataCentre       — Waste heat offtake from a data centre cooling system.
                   Modelled as a near-constant baseload with availability factor
                   and optional seasonal variation in supply temperature.
                   Based on Ealing/Southall feasibility study parameters:
                     • Redwire DC (Ealing town centre): up to 3.6 MW
                     • GTR (Southall): 260 MW electrical → 52–182 MW heat offtake
                     • CyrusOne (Southall): 108 MW electrical → 22–76 MW heat offtake
                   Reference: Ealing Town Centre Heat Network Feasibility Report
                              (SEL, 2503-SEL-RP-001-V02), Appendix 9

Public interface
----------------
    source.supply_MW      np.ndarray (8760,) — available heat at each hour (MW)
    source.supply_temp_C  np.ndarray (8760,) — supply temperature at each hour (°C)
    source.marginal_cost  np.ndarray (8760,) — £/MWh at each hour
    source.capacity_MW    float              — nameplate/rated capacity (MW)
    source.name           str
    source.source_type    str
    source.summary()      dict               — key parameters for reporting

Usage
-----
    from source import DataCentre, PeakBoiler
    from parse_epw import parse_epw

    _, weather_df = parse_epw("data/profiles/GBR_ENG_London-Heathrow.epw")

    # Redwire data centre (Ealing town centre)
    redwire = DataCentre.from_preset("redwire_ealing", weather_df)

    # Southall GTR at 50% heat offtake
    gtr = DataCentre.from_preset("gtr_southall_medium", weather_df)

    # Or build a custom source
    custom_dc = DataCentre(
        name="My Data Centre",
        it_load_MW=100.0,
        heat_offtake_fraction=0.5,
        supply_temp_C=30.0,
        availability_factor=0.95,
        weather_df=weather_df,
        waste_heat_cost_GBP_per_MWh=8.0,
    )

    # Peak backup boiler
    boiler = PeakBoiler(
        name="Gas peak boiler",
        capacity_MW=3.6,
        fuel="gas",
        efficiency=0.92,
        gas_price_GBP_per_MWh=45.0,
    )
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings

# Make sure the project root is on sys.path — same pattern as the other
# source modules, lets `from components.peak_demand_option import ...`
# resolve regardless of how/where this file is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the SAME carbon intensity figures used by the other source
# classes, rather than maintaining a second, possibly-drifting copy. See
# peak_demand_option.py's CARBON_INTENSITY dict for sourcing notes.
from components.peak_demand_option import CARBON_INTENSITY


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# Data centre industry rule of thumb:
# Heat available (MW) ≈ IT load (MW) × (PUE - 1) / PUE
# For a modern hyperscale DC with PUE ~1.3:
#   Heat available ≈ IT load × 0.23
# But heat *offtake* depends on what fraction the network can capture.
# The Ealing report uses 20% / 50% / 70% of electrical capacity as scenarios.
# We model this directly as heat_offtake_fraction × it_load_MW.

# Southall data centre parameters from Ealing feasibility study (Appendix 9)
# Reference: 2503-SEL-RP-001-V02, Table 1 & Table 2
DC_PRESETS = {
    # Redwire — small DC in Ealing town centre, near Perceval House
    # Source: Ealing report section 4, p.28: "cooling provision of up to 3.6 MW"
    "redwire_ealing": {
        "description":            "Redwire DC, Ealing town centre",
        "it_load_MW":             14.4,    # Estimated from 3.6 MW cooling at ~25% utilisation
        "heat_offtake_MW":        3.6,     # Directly from report: "up to 3.6 MW"
        "heat_offtake_fraction":  None,    # Override: use heat_offtake_MW directly
        "supply_temp_low_C":      25.0,    # Lower end: similar to ASHP source temp
        "supply_temp_high_C":     35.0,    # Upper end per Ealing report section 3.5
        "supply_temp_C":          30.0,    # Central estimate (midpoint 25–35°C)
        "availability_factor":    0.95,    # 95% uptime typical for colo DC
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Ealing feasibility report (SEL 2025), p.28",
    },

    # GTR data centre, Southall — International Trading Estate
    # Source: Appendix 9, Table 1: 260 MW electrical
    "gtr_southall_low": {
        "description":            "GTR DC Southall — 20% heat offtake",
        "it_load_MW":             260.0,
        "heat_offtake_fraction":  0.20,    # Low scenario
        "supply_temp_C":          28.0,
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: GTR 260 MW, 20% offtake = 52 MW",
    },
    "gtr_southall_medium": {
        "description":            "GTR DC Southall — 50% heat offtake",
        "it_load_MW":             260.0,
        "heat_offtake_fraction":  0.50,    # Medium scenario
        "supply_temp_C":          30.0,    # Higher temp = higher COP on heat pump
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: GTR 260 MW, 50% offtake = 130 MW",
    },
    "gtr_southall_high": {
        "description":            "GTR DC Southall — 70% heat offtake",
        "it_load_MW":             260.0,
        "heat_offtake_fraction":  0.70,    # High scenario
        "supply_temp_C":          35.0,    # Best case: higher temp offtake
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: GTR 260 MW, 70% offtake = 182 MW",
    },

    # CyrusOne data centre, Southall — former Honeymonster factory site
    # Source: Appendix 9, Table 1: 108 MW electrical
    "cyrusone_southall_low": {
        "description":            "CyrusOne DC Southall — 20% heat offtake",
        "it_load_MW":             108.0,
        "heat_offtake_fraction":  0.20,
        "supply_temp_C":          28.0,
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: CyrusOne 108 MW, 20% = 22 MW",
    },
    "cyrusone_southall_medium": {
        "description":            "CyrusOne DC Southall — 50% heat offtake",
        "it_load_MW":             108.0,
        "heat_offtake_fraction":  0.50,
        "supply_temp_C":          30.0,
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: CyrusOne 108 MW, 50% = 54 MW",
    },
    "cyrusone_southall_high": {
        "description":            "CyrusOne DC Southall — 70% heat offtake",
        "it_load_MW":             108.0,
        "heat_offtake_fraction":  0.70,
        "supply_temp_C":          35.0,
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: CyrusOne 108 MW, 70% = 76 MW",
    },

    # Combined Southall (both DCs) — useful for borough-wide scenario
    "southall_combined_medium": {
        "description":            "GTR + CyrusOne combined — 50% offtake",
        "it_load_MW":             368.0,   # 260 + 108 MW total electrical
        "heat_offtake_fraction":  0.50,
        "supply_temp_C":          30.0,
        "availability_factor":    0.95,
        "waste_heat_cost_GBP_per_MWh": 5.0,
        "reference":              "Appendix 9, Table 1: Total 368 MW, 50% = 184 MW",
    },
}


# ── Availability profile ───────────────────────────────────────────────────────

def _availability_profile(
    availability_factor: float,
    n_hours: int = N_HOURS,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate an 8760-length binary availability array for a data centre.

    Data centres don't fail gradually — they have planned maintenance windows
    and rare unplanned outages. This models that as:
      - Planned maintenance: two 48-hour windows per year (realistic for DC)
      - Random unplanned: remaining unavailability spread as short outages

    Returns an array of 0.0 (unavailable) or 1.0 (available).
    The mean of the array will approximately equal availability_factor.
    """
    rng  = np.random.default_rng(seed)
    avail = np.ones(n_hours)

    # Annual unavailable hours from availability factor
    unavail_hours = int(round((1.0 - availability_factor) * n_hours))

    # Planned maintenance: two blocks of equal size
    # Schedule them in spring and autumn (avoid peak winter demand)
    planned_hours = min(unavail_hours // 2, 48)
    if planned_hours > 0:
        # Spring: around hour 2160 (late March), Autumn: around hour 6552 (late Sep)
        for start in [2160, 6552]:
            actual_start = rng.integers(start - 48, start + 48)
            end = min(actual_start + planned_hours, n_hours)
            avail[actual_start:end] = 0.0

    # Remaining as short random outages (2-6 hours each)
    remaining = unavail_hours - 2 * planned_hours
    while remaining > 0:
        outage_len = min(rng.integers(2, 7), remaining)
        start = rng.integers(0, n_hours - outage_len)
        avail[start:start + outage_len] = 0.0
        remaining -= outage_len

    return avail


# ── Data centre supply temperature ────────────────────────────────────────────

def _dc_supply_temperature(
    supply_temp_C: float,
    weather_df: Optional[pd.DataFrame] = None,
    temp_sensitivity: float = 0.0,
) -> np.ndarray:
    """
    Generate hourly supply temperature array for a data centre heat source.

    Data centre waste heat temperature depends on the cooling system design:
      - Chiller/cooling tower systems: relatively stable year-round (~25-35°C)
      - Air-cooled systems: slightly higher in summer (ambient affects efficiency)
      - Liquid-cooled (immersion/direct): higher and more stable (~40-45°C)

    The Ealing report (section 3.5) states heat offtake temperatures of 25-35°C
    for typical cooling systems. We model this as:
      - A constant base temperature (supply_temp_C)
      - Optional small seasonal variation driven by ambient temperature
        (temp_sensitivity controls amplitude, default 0 = perfectly constant)

    Parameters
    ----------
    supply_temp_C    : nominal supply temperature (°C)
    weather_df       : EPW weather DataFrame — needed if temp_sensitivity > 0
    temp_sensitivity : °C change in supply temp per °C change in ambient
                       Typical range: 0.05–0.15 for air-cooled systems
                       0.0 for liquid-cooled / chiller systems (default)
    """
    if temp_sensitivity == 0.0 or weather_df is None:
        return np.full(N_HOURS, supply_temp_C)

    T_air  = weather_df["temp_drybulb_C"].values[:N_HOURS]
    T_mean = T_air.mean()

    # Supply temp rises slightly when ambient is hot (worse cooling efficiency)
    supply = supply_temp_C + temp_sensitivity * (T_air - T_mean)

    # Clamp to physically reasonable bounds
    return np.clip(supply, supply_temp_C - 5.0, supply_temp_C + 10.0)


# ── DataCentre class ───────────────────────────────────────────────────────────

class DataCentre:
    """
    Waste heat source from a data centre cooling system.

    The heat available is modelled as:
        supply_MW(t) = heat_offtake_MW × availability(t)

    Where:
        heat_offtake_MW = it_load_MW × heat_offtake_fraction
                          (or set directly via heat_offtake_MW parameter)

    This is before the heat pump — the source delivers low-grade heat at
    supply_temp_C which a heat pump then lifts to network temperature.
    The heat pump model lives in large_scale_heat_pumps.py.

    The marginal cost of waste heat is very low (the data centre would
    otherwise reject it to atmosphere), but not zero — there are pumping
    costs, heat exchanger maintenance, and a negotiated offtake charge.

    Parameters
    ----------
    name                       : descriptive name for reporting
    it_load_MW                 : data centre IT electrical load (MW)
    heat_offtake_fraction      : fraction of IT load recoverable as heat (0–1)
                                 e.g. 0.50 = 50% offtake (medium scenario)
    heat_offtake_MW            : direct override — if set, ignores fraction
    supply_temp_C              : nominal waste heat supply temperature (°C)
                                 Ealing report: 25–35°C typical range
    supply_temp_high_C         : upper bound on supply temp (for sensitivity)
    supply_temp_low_C          : lower bound on supply temp (for sensitivity)
    availability_factor        : fraction of hours the source is available (0–1)
                                 0.95 = 95% uptime = ~438 hours downtime/year
    waste_heat_cost_GBP_per_MWh: float for negotiated charge for waste heat offtake (£/MWh)
                                 Very low — DC benefits from reduced cooling load
    weather_df                 : EPW weather DataFrame (for temp sensitivity)
    temp_sensitivity           : how much supply temp varies with ambient (°C/°C)
    seed                       : random seed for availability profile
    capex_GBP_per_MW           : capital cost per MW of heat OFFTAKE capacity
                                 (this.capacity_MW) — the DC-side heat
                                 recovery equipment ONLY (heat exchangers,
                                 piping, controls, metering), NOT the data
                                 centre itself (which exists regardless of
                                 district heating) and NOT the booster heat
                                 pump that lifts this heat to network
                                 temperature (see components/
                                 booster_heat_pump.py, costed separately).
                                 Default £462,000/MW — real sourcing: a
                                 2026 industry cost review
                                 (moduledge.com/blog/data-center-waste-
                                 heat-recovery) gives EUR400,000-700,000/MW
                                 for "data center side" heat recovery
                                 infrastructure (heat exchangers, piping,
                                 controls, metering) — midpoint
                                 EUR550,000/MW, converted at ~0.84 EUR/GBP.
                                 Deliberately a DIFFERENT figure from
                                 ASHPArray's £770,000/MW or
                                 BoosterHeatPump's £600,000/MW — this is
                                 genuinely cheaper, lower-grade equipment
                                 (a heat exchanger tapping an existing
                                 water loop, not a full compression cycle).
    """

    source_type = "data_centre"

    def __init__(
        self,
        name: str,
        it_load_MW: float,
        heat_offtake_fraction: float          = 0.50,
        heat_offtake_MW: Optional[float]      = None,
        supply_temp_C: float                  = 30.0,
        supply_temp_high_C: Optional[float]   = None,
        supply_temp_low_C: Optional[float]    = None,
        availability_factor: float            = 0.95,
        waste_heat_cost_GBP_per_MWh: float    = 5.0,
        weather_df: Optional[pd.DataFrame]    = None,
        temp_sensitivity: float               = 0.0,
        seed: int                             = 42,
        capex_GBP_per_MW: float                = 462_000.0,
        reference: str                        = "",
    ):
        self.name                       = name
        self.it_load_MW                 = float(it_load_MW)
        self.heat_offtake_fraction      = float(heat_offtake_fraction)
        self.supply_temp_nominal_C      = float(supply_temp_C)
        self.supply_temp_high_C         = supply_temp_high_C or supply_temp_C + 5.0
        self.supply_temp_low_C          = supply_temp_low_C  or supply_temp_C - 5.0
        self.availability_factor        = float(availability_factor)
        self.waste_heat_cost_GBP_per_MWh = float(waste_heat_cost_GBP_per_MWh)
        self.capex_GBP_per_MW            = float(capex_GBP_per_MW)
        self.reference                  = reference

        # Resolve rated capacity
        if heat_offtake_MW is not None:
            self.capacity_MW = float(heat_offtake_MW)
        else:
            self.capacity_MW = self.it_load_MW * self.heat_offtake_fraction

        # Build hourly arrays
        self._avail    = _availability_profile(availability_factor, seed=seed)
        self._supply_T = _dc_supply_temperature(supply_temp_C, weather_df, temp_sensitivity)

        # Supply available at each hour (MW) — before heat pump
        self.supply_MW = self.capacity_MW * self._avail

        # Supply temperature at each hour (°C)
        self.supply_temp_C = self._supply_T

        # Marginal cost at each hour (£/MWh)
        # Constant — DC waste heat price is typically a fixed negotiated rate
        self.marginal_cost = np.full(N_HOURS, waste_heat_cost_GBP_per_MWh)

        # Carbon intensity per unit heat delivered (kgCO2e/kWh_heat).
        # Genuinely zero, not "we don't have a number so we used 0" — see
        # CARBON_INTENSITY["dc_waste_heat"] in peak_demand_option.py for
        # the full mechanism note. Unlike EfW, capturing this heat doesn't
        # reduce any other useful output: IT load (and therefore cooling/
        # heat-rejection load) is fixed by computing demand regardless of
        # whether district heating draws off the waste heat or not.
        self.carbon_intensity_kgCO2_per_kWh = np.full(N_HOURS, CARBON_INTENSITY["dc_waste_heat"])

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        weather_df: Optional[pd.DataFrame] = None,
        **overrides,
    ) -> "DataCentre":
        """
        Construct a DataCentre from a named preset (see DC_PRESETS dict).

        Parameters
        ----------
        preset_key : one of the keys in DC_PRESETS (e.g. 'redwire_ealing')
        weather_df : EPW weather DataFrame
        **overrides: any DataCentre parameter to override from the preset

        Example
        -------
            dc = DataCentre.from_preset("gtr_southall_medium", weather_df,
                                        waste_heat_cost_GBP_per_MWh=8.0)
        """
        if preset_key not in DC_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(DC_PRESETS.keys())}"
            )

        params = DC_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.pop("supply_temp_low_C",  None)  # handled by constructor defaults
        params.pop("supply_temp_high_C", None)

        # heat_offtake_MW override (Redwire uses this instead of fraction)
        if params.get("heat_offtake_fraction") is None:
            params.pop("heat_offtake_fraction", None)

        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(
        cls,
        config: dict,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> "DataCentre":
        """
        Construct a DataCentre from a YAML/dict config block.

        Expected keys (mirrors scenarios/opdc_phase1.yaml):
            name, it_load_MW, heat_offtake_fraction (or heat_offtake_MW),
            supply_temp_C, availability_factor, waste_heat_cost_GBP_per_MWh

        Example YAML block
        ------------------
            heat_sources:
              - type: data_centre
                name: "GTR Southall (medium offtake)"
                it_load_MW: 260
                heat_offtake_fraction: 0.5
                supply_temp_C: 30.0
                availability_factor: 0.95
                waste_heat_cost_GBP_per_MWh: 5.0
        """
        cfg = {k: v for k, v in config.items() if k != "type"}
        return cls(weather_df=weather_df, **cfg)

    def summary(self) -> dict:
        """Return key parameters as a dict for reporting / logging."""
        avail_hours = int(self._avail.sum())
        return {
            "name":                      self.name,
            "source_type":               self.source_type,
            "it_load_MW":                self.it_load_MW,
            "heat_offtake_fraction":     self.heat_offtake_fraction,
            "capacity_MW":               round(self.capacity_MW, 2),
            "supply_temp_nominal_C":     self.supply_temp_nominal_C,
            "supply_temp_range_C":       f"{self.supply_temp_low_C}–{self.supply_temp_high_C}",
            "availability_factor":       self.availability_factor,
            "available_hours_per_year":  avail_hours,
            "annual_heat_available_MWh": round(self.supply_MW.sum(), 0),
            "marginal_cost_GBP_per_MWh": self.waste_heat_cost_GBP_per_MWh,
            "capex_GBP_per_MW":          self.capex_GBP_per_MW,
            "estimated_capex_GBP":       round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "reference":                 self.reference,
        }

    def __repr__(self):
        return (
            f"DataCentre(name='{self.name}', capacity={self.capacity_MW:.1f} MW, "
            f"T_supply={self.supply_temp_nominal_C}°C, "
            f"availability={self.availability_factor:.0%})"
        )


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_datacentre_source.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_datacentre_source.py\n"
    )
