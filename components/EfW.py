"""
==================
Energy-from-Waste (EfW) Combined Heat and Power plant model — waste
incineration with steam turbine extraction supplying both electricity
and high-temperature heat to a district network.
 
This is intentionally a SIMPLIFIED model — not a full thermodynamic
steam cycle simulation. It captures the realistic behaviour that matters
for a feasibility-stage techno-economic model: high, stable supply
temperature (so no heat pump boost needed, unlike DC waste heat or ASHP),
near-baseload availability, and a heat:power trade-off that real plants
must navigate.
 
Why EfW CHP is different from your other sources
--------------------------------------------------
DataCentre   : low-grade heat (25-35°C), needs a heat pump to lift to
               network temperature, near-constant supply.
ASHPArray    : weather-dependent capacity and COP, electrically driven.
EfW CHP      : HIGH-grade heat (90-120°C+) extracted directly from the
               steam turbine — can usually feed an LTHW network directly
               with no boosting required. Baseload by design (waste must
               be processed continuously), with one long annual planned
               outage rather than dispersed short outages.
 
Real UK reference plants used to calibrate this model
--------------------------------------------------------
SELCHP (Bermondsey, London)
    Capacity: up to 420,000 tonnes/yr MSW, up to 35 MWe electricity-only mode
    Reference: en.wikipedia.org/wiki/SELCHP
 
Sheffield ERF (Veolia)
    Permitted capacity: 245,000 tonnes/yr
    CHP output: up to 21 MWe exported to grid + up to 45 MWth heat to the
    Sheffield District Energy Network (140+ buildings, ~3,000 homes)
    Steam conditions: combustion >850°C, superheated steam at 400°C
    Reference: wikiwaste.org.uk/index.php/Sheffield_ERF,
               en.wikipedia.org/wiki/Sheffield_EfW_Plant
 
Newlincs (smaller reference plant)
    56,000 tonnes/yr waste → 3 MWe + 3 MWth, ~8,000 operating hours/year
    (91% availability) — a useful SMALL-scale reference point
    Reference: arxiv.org/pdf/1404.3167 (Humber region economic model)
 
Heat:power ratio
------------------
Sheffield ERF's ratio of ~45 MWth : 21 MWe ≈ 2.1:1 (heat:power) is used as
the default when only one of the two figures is known. This is broadly
consistent with the published EfW CHP literature on backpressure/extraction
turbine operation — converting from condensing (electricity-only) mode to
heat extraction sacrifices some electrical output but extracts much more
total useful energy. The model lets you set whichever figures you actually
know (tonnes/yr, MWe, MWth) and infers the others using the heat:power ratio
and standard EfW conversion factors as a fallback.
 
Supply temperature
---------------------
Modelled as a high, near-constant temperature (default 90°C) reflecting
typical backpressure/extraction steam turbine operation for district
heating — see search-verified literature on steam turbine extraction
temperatures for DH (90-120°C range typical, e.g. Finnish/European CHP
turbine optimisation literature). This is deliberately simplified —
real plants vary extraction pressure/temperature with seasonal demand,
but for a feasibility-stage model a constant high temperature is a
reasonable and conservative assumption.
 
Usage
-----
    from efw_chp_source import EfWChp
 
    # From a preset (Sheffield ERF — useful as a 'large plant' reference)
    efw = EfWChp.from_preset("sheffield_erf_style", weather_df=None)
 
    # From a preset (Newlincs — useful as a 'small plant' reference)
    efw = EfWChp.from_preset("newlincs_style", weather_df=None)
 
    # Custom sizing — specify whichever you know
    efw = EfWChp(
        name="Local EfW CHP",
        waste_throughput_tonnes_per_year=150_000,
        heat_capacity_MW=25.0,        # if known directly
        electrical_capacity_MW=12.0,  # if known directly
        availability_factor=0.88,
    )

"""
 
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional

# Make sure the project root is on sys.path, same pattern as ASHP.py and
# peak_demand_option.py — lets `from components.peak_demand_option import
# ...` resolve regardless of how/where this file is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the SAME carbon intensity figures used by the other source
# classes (BEIS/DESNZ 2024 conversion factors + BRE EfW heat factor)
# rather than maintaining a second, possibly-drifting copy. See
# peak_demand_option.py's CARBON_INTENSITY dict for sourcing notes.
from components.peak_demand_option import CARBON_INTENSITY
 
 
# ── Constants ──────────────────────────────────────────────────────────────────
 
N_HOURS = 8760
 
# Typical EfW conversion factor: net electricity per tonne of MSW incinerated
# Source: Veolia plant data (~420,000 t/yr → 29 MWe net, ~8000 op hours/yr)
# 29 MW x 8000 h = 232,000 MWh / 420,000 t = 0.552 MWh/t ≈ 552 kWh/t
# This is broadly consistent with the 519 kWh/t figure reported for a
# similar Veolia electricity-only plant.
ELEC_KWH_PER_TONNE_ELECTRICITY_ONLY = 535.0   # Midpoint of cited range (519-552)
 
# When a plant switches from condensing (electricity-only) to backpressure/
# extraction (CHP) mode, electrical output typically falls somewhat but
# total useful energy (heat + power) rises substantially. This factor
# captures that trade-off in a simple way.
CHP_ELECTRICAL_DERATING = 0.75   # CHP-mode electricity ≈ 75% of electricity-only mode
 
# Default heat:power ratio when only one output is specified
# Calibrated against Sheffield ERF: 45 MWth : 21 MWe ≈ 2.14:1
DEFAULT_HEAT_TO_POWER_RATIO = 2.1
 
 
# ── EfW CHP presets — real UK reference plants ────────────────────────────────
 
EFW_PRESETS = {
    "sheffield_erf_style": {
        "description":                  "Large EfW CHP (Sheffield ERF scale)",
        "waste_throughput_tonnes_per_year": 245_000,
        "electrical_capacity_MW":           21.0,
        "heat_capacity_MW":                 45.0,
        "supply_temp_C":                    90.0,
        "availability_factor":              0.90,
        "reference":     "Sheffield ERF (Veolia): 245kt/yr, 21 MWe + 45 MWth, "
                          "wikiwaste.org.uk/index.php/Sheffield_ERF",
    },
    "newlincs_style": {
        "description":                  "Small EfW CHP (Newlincs scale)",
        "waste_throughput_tonnes_per_year": 56_000,
        "electrical_capacity_MW":           3.0,
        "heat_capacity_MW":                 3.0,
        "supply_temp_C":                    90.0,
        "availability_factor":              0.91,   # 8000 op hrs / 8760 ≈ 91%
        "reference":     "Newlincs reference plant: 56kt/yr, 3 MWe + 3 MWth, "
                          "8000 op hrs/yr, arxiv.org/pdf/1404.3167",
    },
    "selchp_style": {
        "description":                  "Large EfW (SELCHP scale, electricity-biased)",
        "waste_throughput_tonnes_per_year": 420_000,
        "electrical_capacity_MW":           35.0,   # Electricity-only mode max
        "heat_capacity_MW":                 None,   # Inferred via heat:power ratio
        "supply_temp_C":                    90.0,
        "availability_factor":              0.88,
        "reference":     "SELCHP (Bermondsey): 420kt/yr, up to 35 MWe, "
                          "en.wikipedia.org/wiki/SELCHP — heat capacity not "
                          "publicly specified, inferred from heat:power ratio",
    },
    "mid_scale_generic": {
        "description":                  "Mid-scale generic EfW CHP",
        "waste_throughput_tonnes_per_year": 150_000,
        "electrical_capacity_MW":           13.0,
        "heat_capacity_MW":                 27.0,
        "supply_temp_C":                    90.0,
        "availability_factor":              0.89,
        "reference":     "Generic interpolation between Newlincs and Sheffield scales",
    },
}
 
 
# ── Availability profile ───────────────────────────────────────────────────────
 
def _efw_availability_profile(
    availability_factor: float,
    n_hours: int = N_HOURS,
    seed: int = 10,
) -> np.ndarray:
    """
    EfW plants run baseload by design — waste must be processed continuously,
    so unlike a data centre's dispersed short outages, EfW plants typically
    have ONE long planned annual maintenance shutdown (boiler/turbine
    inspection, refractory repairs) lasting roughly 2-5 weeks, scheduled in
    a shoulder season to minimise lost heat revenue during peak winter
    demand.
 
    Returns an 8760-length array of 0.0 (unavailable) or 1.0 (available).
    """
    rng = np.random.default_rng(seed)
    avail = np.ones(n_hours)
 
    unavail_hours = int(round((1.0 - availability_factor) * n_hours))
    if unavail_hours <= 0:
        return avail
 
    # Schedule the single outage in spring (around hour 2000-2800, i.e.
    # March-April) — avoids both winter heat peak and is before summer
    start = int(rng.integers(2000, 2800))
    end = min(start + unavail_hours, n_hours)
    avail[start:end] = 0.0
 
    # If the outage would overrun the array (very low availability edge case),
    # wrap remaining hours to the start
    overrun = unavail_hours - (end - start)
    if overrun > 0:
        avail[:overrun] = 0.0
 
    return avail
 
 
# ── EfWChp class ────────────────────────────────────────────────────────────────
 
class EfWChp:
    """
    Energy-from-Waste CHP plant — simplified model for feasibility-stage
    techno-economic analysis.
 
    Supplies HIGH temperature heat (default 90°C) directly usable by an
    LTHW network without a boosting heat pump, at near-baseload
    availability (single annual planned outage).
 
    You can specify the plant size via ANY of: waste throughput (t/yr),
    electrical capacity (MW), or heat capacity (MW) — the model will
    infer the others using standard EfW conversion factors and the
    heat:power ratio where needed. If you provide more than one, the
    explicitly provided values are used as-is (no re-inference).
 
    Parameters
    ----------
    name                              : descriptive name for reporting
    waste_throughput_tonnes_per_year  : annual waste processed (tonnes)
    electrical_capacity_MW            : net electrical output capacity (MW)
                                         If None, inferred from waste
                                         throughput and heat:power ratio.
    heat_capacity_MW                  : heat export capacity (MW)
                                         If None, inferred from electrical
                                         capacity x heat_to_power_ratio.
    heat_to_power_ratio               : MWth per MWe in CHP mode. Default
                                         2.1 (Sheffield ERF-calibrated).
    supply_temp_C                     : heat export temperature (°C).
                                         Default 90°C — typical backpressure/
                                         extraction turbine DH supply temp.
    availability_factor               : fraction of year operational (0-1).
                                         UK EfW plants typically 0.85-0.91.
    heat_export_cost_GBP_per_MWh      : marginal cost of heat from this
                                         source (£/MWh). Very low — EfW
                                         heat is largely a by-product once
                                         the plant exists for waste disposal
                                         and electricity generation; the
                                         waste gate fee is the primary
                                         revenue driver, not heat price.
    seed                               : random seed for the outage schedule
    capex_GBP_per_MW                   : capital cost per MW of heat
                                         EXPORT capacity (heat_capacity_MW)
                                         -- the "inside the gate" RETROFIT
                                         equipment needed to extract heat
                                         from an EXISTING EfW plant (NOT
                                         the cost of building the plant
                                         itself, which exists regardless
                                         of district heating, driven by
                                         waste disposal need not heat
                                         demand). Default GBP775,000/MW --
                                         real sourcing: turbine retrofitting
                                         (GBP400k-700k/MWth) plus heat
                                         exchangers and pumps (GBP150k-300k/MWth)
                                         to move heat from the steam cycle
                                         to the DH water loop -- midpoint
                                         of the combined GBP550k-1,000k/MWth
                                         range. Deliberately EXCLUDES flue
                                         gas condensation (an optional
                                         capacity-boosting enhancement, not
                                         required for basic heat export) and
                                         excludes outside-the-gate costs
                                         (pipework, energy centre, consumer
                                         substations), already modelled
                                         elsewhere in this project.
    """
 
    source_type = "efw_chp"
 
    def __init__(
        self,
        name: str,
        waste_throughput_tonnes_per_year: Optional[float] = None,
        electrical_capacity_MW: Optional[float]           = None,
        heat_capacity_MW: Optional[float]                 = None,
        heat_to_power_ratio: float                        = DEFAULT_HEAT_TO_POWER_RATIO,
        supply_temp_C: float                               = 90.0,
        availability_factor: float                        = 0.89,
        heat_export_cost_GBP_per_MWh: float               = 8.0,
        seed: int                                          = 10,
        capex_GBP_per_MW: float                            = 775_000.0,
        reference: str                                     = "",
    ):
        self.name                 = name
        self.heat_to_power_ratio  = float(heat_to_power_ratio)
        self.supply_temp_nominal_C = float(supply_temp_C)
        self.availability_factor  = float(availability_factor)
        self.heat_export_cost_GBP_per_MWh = float(heat_export_cost_GBP_per_MWh)
        self.capex_GBP_per_MW      = float(capex_GBP_per_MW)
        self.reference            = reference
 
        # --- Resolve capacities, filling in gaps from whatever IS known ---
        self.waste_throughput_tonnes_per_year = waste_throughput_tonnes_per_year
 
        if electrical_capacity_MW is None and waste_throughput_tonnes_per_year is not None:
            # Infer electrical-only-equivalent capacity from waste throughput,
            # then derate for CHP mode
            assumed_op_hours = N_HOURS * availability_factor
            elec_only_MWh = (
                waste_throughput_tonnes_per_year
                * ELEC_KWH_PER_TONNE_ELECTRICITY_ONLY / 1000
            )
            elec_only_MW = elec_only_MWh / assumed_op_hours
            electrical_capacity_MW = elec_only_MW * CHP_ELECTRICAL_DERATING
 
        if electrical_capacity_MW is None:
            raise ValueError(
                "Must provide either waste_throughput_tonnes_per_year or "
                "electrical_capacity_MW (or both) to size the plant."
            )
 
        self.electrical_capacity_MW = float(electrical_capacity_MW)
 
        if heat_capacity_MW is None:
            heat_capacity_MW = self.electrical_capacity_MW * self.heat_to_power_ratio
 
        self.capacity_MW = float(heat_capacity_MW)   # 'capacity_MW' = HEAT capacity,
                                                       # matching the shared source interface
 
        # --- Build hourly arrays ---
        self._avail = _efw_availability_profile(availability_factor, seed=seed)
 
        self.supply_MW     = self.capacity_MW * self._avail
        self.supply_temp_C = np.full(N_HOURS, self.supply_temp_nominal_C)
 
        # Marginal cost: low and constant — EfW heat is largely a
        # by-product revenue stream once the plant exists for waste
        # disposal duty, not a cost-driven dispatch decision like a boiler
        self.marginal_cost = np.full(N_HOURS, self.heat_export_cost_GBP_per_MWh)

        # Carbon intensity per unit heat delivered (kgCO2e/kWh_heat).
        # NOT zero, despite being "waste heat" — extracting heat from the
        # steam cycle reduces electricity sent to the grid, so EfW heat
        # carries a real opportunity-cost carbon factor (displaced grid
        # generation), not a direct combustion factor on the heat itself.
        # See CARBON_INTENSITY["efw_heat"] in peak_demand_option.py for
        # the full BRE/SAP 2012 sourcing note (calibrated against the
        # real SELCHP plant). Constant across all 8760 hours — EfW runs
        # baseload by design (see module docstring), no part-load
        # variation modelled here unlike the boilers.
        self.carbon_intensity_kgCO2_per_kWh = np.full(N_HOURS, CARBON_INTENSITY["efw_heat"])
 
        # Electricity also produced alongside heat (informational —
        # useful for revenue-side reporting even though this model focuses
        # on heat supply for the dispatch optimiser)
        self.electrical_output_MW = self.electrical_capacity_MW * self._avail
 
    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        weather_df: Optional[pd.DataFrame] = None,   # accepted for interface
        **overrides,                                  # consistency, unused
    ) -> "EfWChp":
        """
        Construct an EfWChp from a named preset (see EFW_PRESETS dict).
 
        weather_df is accepted but not required — EfW CHP supply is not
        weather-dependent in this model (unlike ASHP). It's accepted purely
        so this class can be called identically to DataCentre/ASHPArray in
        a loop that builds all sources for a scenario.
 
        Example
        -------
            efw = EfWChp.from_preset("sheffield_erf_style")
            efw = EfWChp.from_preset("newlincs_style", availability_factor=0.93)
        """
        if preset_key not in EFW_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(EFW_PRESETS.keys())}"
            )
 
        params = EFW_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(**params)
 
    @classmethod
    def from_config(
        cls,
        config: dict,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> "EfWChp":
        """
        Construct an EfWChp from a YAML/dict config block.
 
        Example YAML block
        -------------------
            heat_sources:
              - type: efw_chp
                name: "Local EfW CHP"
                waste_throughput_tonnes_per_year: 150000
                heat_capacity_MW: 27.0
                electrical_capacity_MW: 13.0
                supply_temp_C: 90.0
                availability_factor: 0.89
        """
        cfg = {k: v for k, v in config.items() if k != "type"}
        return cls(**cfg)
 
    def summary(self) -> dict:
        """Return key parameters as a dict for reporting / logging."""
        return {
            "name":                        self.name,
            "source_type":                 self.source_type,
            "waste_throughput_tonnes_per_year": self.waste_throughput_tonnes_per_year,
            "electrical_capacity_MW":      round(self.electrical_capacity_MW, 2),
            "heat_capacity_MW":            round(self.capacity_MW, 2),
            "heat_to_power_ratio":         round(self.heat_to_power_ratio, 2),
            "supply_temp_C":               self.supply_temp_nominal_C,
            "availability_factor":         self.availability_factor,
            "annual_heat_available_MWh":   round(float(self.supply_MW.sum()), 0),
            "annual_electricity_MWh":      round(float(self.electrical_output_MW.sum()), 0),
            "marginal_cost_GBP_per_MWh":   self.heat_export_cost_GBP_per_MWh,
            "capex_GBP_per_MW":            self.capex_GBP_per_MW,
            "estimated_capex_GBP":         round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "reference":                   self.reference,
        }
 
    def __repr__(self):
        return (
            f"EfWChp(name='{self.name}', heat_capacity={self.capacity_MW:.1f} MW, "
            f"elec_capacity={self.electrical_capacity_MW:.1f} MW, "
            f"T_supply={self.supply_temp_nominal_C}°C, "
            f"availability={self.availability_factor:.0%})"
        )
 
