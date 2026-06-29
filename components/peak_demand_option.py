"""
peak_demand_option.py
================
Peak and backup heat sources for the district energy system — gas boilers
and electric boilers. These sit at the top of the merit order: high
marginal cost, fully dispatchable, called on only when lower-cost sources
(data centre waste heat, ASHP) can't meet demand.

A note on "efficiency vs scale"
--------------------------------------------------------------------
Research into boiler efficiency data found NO meaningful efficiency
improvement with nameplate size for gas boilers — a study of 41 large-scale
units (58 kW–2,900 kW) found efficiency data essentially absent as a
function of size, while domestic units across 18-42 kW sit in a narrow
89-89.5% band regardless of size. The dominant technology effect is
condensing vs non-condensing, not bigger-is-better.

What DOES matter — and what we model here — is PART-LOAD efficiency.
Per DIN 4702-8 (the standard seasonal efficiency methodology):
    seasonal_efficiency = 0.81 * eta_30pct_load + 0.19 * eta_100pct_load
Condensing boilers are typically a few percentage points MORE efficient
at low part-load (lower flue gas temperatures favour condensing operation),
which is the opposite of what people often assume ("smaller load = worse
efficiency"). This module models that load-dependent curve explicitly,
rather than inventing a fake size-based efficiency curve that isn't
supported by the evidence.

Scale still matters economically (£/MW falls with size) — but that belongs
in economics/CAPEX.py, not here. This module only models technical
performance (efficiency, output, marginal cost).

Gas and electricity pricing
-----------------------------
Both GasBoiler and ElectricBoiler accept gas_price_GBP_per_MWh /
electricity_price_GBP_per_MWh as one of FOUR input types, resolved via
economics.tariffs:
    None                              -> realistic default tariff
                                         (GasBoiler: ~£25/MWh DESNZ central;
                                          ElectricBoiler: ~£240/MWh central
                                          commercial electricity)
    GasTariff / ElectricityTariff      -> a specific tariff scenario
    float / int                        -> flat scalar override
    8760-length array                  -> a fully custom hourly price series
This replaces the old flat placeholder defaults (£45/MWh gas, £120/MWh
electricity) so the cost comparison between gas and electric backup options
reflects realistic pricing by default, not an arbitrary placeholder.

Supported source types
-----------------------
GasBoiler        — Natural gas-fired boiler. Condensing or non-condensing.
                   Part-load efficiency curve per DIN 4702-8 methodology.
                   Typical UK commercial: 90-92% full-load condensing.

ElectricBoiler   — Resistive or electrode electric boiler. Near-constant
                   efficiency regardless of load (no combustion losses to
                   worry about) — typically 98-99.5%.
                   Useful as a clean peak/backup option, or where gas
                   connection isn't available/desired.

Both classes share the same interface as DataCentre and ASHPArray:
    supply_MW, supply_temp_C, marginal_cost, capacity_MW, summary()

Usage
-----
    from peak_demand_option import GasBoiler, ElectricBoiler
    from economics.tariffs import GasTariff, ElectricityTariff

    # From a preset (Ealing report Phase 1/2 figures) — uses the realistic
    # default tariff automatically
    gas_boiler = GasBoiler.from_preset("ealing_phase1")

    # With a specific gas price scenario
    gas_boiler = GasBoiler.from_preset(
        "ealing_phase1", gas_price_GBP_per_MWh=GasTariff.from_scenario("current_actual")
    )

    # Custom sizing
    gas_boiler = GasBoiler(
        name="Town centre peak boiler",
        capacity_MW=3.6,
        condensing=True,
    )

    elec_boiler = ElectricBoiler(
        name="Electric backup boiler",
        capacity_MW=1.0,
    )

    # Both need an hourly demand/dispatch signal externally — they don't
    # have a 'supply available' weather dependency like ASHP/DataCentre.
    # supply_MW here represents the MAXIMUM available (= capacity at all
    # hours, since boilers have no weather/availability constraint).
    # The dispatch optimiser decides how much of that to actually use.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional

# Make sure the project root (one level up from this file's own folder,
# i.e. district_model/) is on sys.path — regardless of where this script
# is launched from or how (absolute path, relative path, -m, or imported
# by another module). This is what lets `from economics.tariffs import
# ...` resolve whether you run this file directly for a quick self-test
# or as part of the full pipeline via main.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# resolve_gas_price() / resolve_electricity_price() turn None / Tariff /
# scalar / array into a clean 8760 £/MWh series — see economics/tariffs.py.
from economics.tariffs import (
    resolve_gas_price,
    resolve_electricity_price,
    GasTariff,
    ElectricityTariff,
)


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# Carbon intensities (kgCO2e/kWh HEAT delivered, i.e. already reflects
# each source's own efficiency/COP where that's a fixed property of the
# fuel pathway — gas and electric here are PER UNIT FUEL/ELECTRICITY
# input, not per unit heat output, since boiler/heater efficiency varies
# by load and is applied separately in marginal_cost — see each class's
# _recompute_marginal_cost()/carbon attribute for how it's actually used.
CARBON_INTENSITY = {
    "gas":      0.183,   # Natural gas, gross CV basis (Scope 1). BEIS/DESNZ 2024.
    "electric": 0.207,   # UK grid average 2024 (will fall over time — see note)

    # EfW (Energy from Waste) CHP heat — kgCO2e per kWh of HEAT delivered
    # (not per unit fuel/waste input; this is already a final heat-output
    # factor, unlike gas/electric above).
    # Source: BRE Technical Note "Modelling Energy from Waste Facilities"
    # (SAP 2012 methodology), calibrated against the real SELCHP plant
    # (the same reference plant family cited in EfW.py's own docstring).
    # NOT zero, despite being a "waste heat byproduct": extracting heat
    # from the steam cycle means LESS electricity is generated and sent
    # to the grid, so EfW heat carries a real opportunity-cost carbon
    # factor (lost low-carbon-displacing grid electricity), not a direct
    # combustion factor. BRE's worked calculation: 0.0580 kgCO2/kWh heat
    # (heat-displaced-electricity term 0.0503 + backup-boiler-on-the-3%-
    # shortfall term 0.0076). This factor does NOT vary with ambient
    # conditions (EfW runs baseload year-round), unlike ASHP below.
    "efw_heat": 0.0580,

    # DataCentre waste heat — kgCO2e per kWh of heat delivered.
    # Set to 0.0, and this IS a genuine zero, not a simplification like
    # "we don't have a number so we used 0" — the mechanism is different
    # from EfW above: a data centre's IT load (and therefore its cooling/
    # heat-rejection load) is fixed by computing demand, NOT by whether
    # district heating captures the waste heat or not. Capturing it
    # doesn't reduce any other useful output (unlike EfW, where capturing
    # heat measurably reduces electricity sent to the grid) — it's heat
    # that would otherwise be rejected to atmosphere via dry air coolers
    # either way. No displaced-generation term applies, so the BRE-style
    # calculation that gives EfW its non-zero factor doesn't apply here.
    "dc_waste_heat": 0.0,
}

# DIN 4702-8 part-load weighting (matches docstring above)
PART_LOAD_REFERENCE = 0.30   # The "30% load" reference point in the standard


# ── Boiler presets — Ealing report figures ────────────────────────────────────
# Source: Ealing Town Centre Heat Network Feasibility Report (SEL, 2025), Table 1
# "Phase 1: 3.6 MW peak/reserve boiler alongside 2.8 MW ASHP"
# "Phase 2: 5.8 MW peak/reserve boiler"
# None of these presets hard-code a fuel price — they all rely on the class
# default (the realistic tariff) unless you override it.

GAS_BOILER_PRESETS = {
    "ealing_phase1": {
        "description":          "Ealing Town Centre Phase 1 peak/reserve gas boiler",
        "capacity_MW":            3.6,
        "condensing":             True,
        "eta_full_load":          0.92,
        "reference":             "Ealing report Table 1: 3.6 MW peak/reserve boiler",
    },
    "ealing_phase2": {
        "description":          "Ealing Town Centre Phase 2 peak/reserve gas boiler",
        "capacity_MW":            5.8,
        "condensing":             True,
        "eta_full_load":          0.92,
        "reference":             "Ealing report Table 1: 5.8 MW peak/reserve boiler",
    },
    "small_commercial": {
        "description":          "Small commercial condensing gas boiler",
        "capacity_MW":            0.5,
        "condensing":             True,
        "eta_full_load":          0.90,
        "reference":             "Generic small commercial sizing",
    },
    "non_condensing_legacy": {
        "description":          "Legacy non-condensing boiler (existing plant)",
        "capacity_MW":            2.0,
        "condensing":             False,
        "eta_full_load":          0.78,
        "reference":             "Typical pre-2005 non-condensing commercial boiler",
    },
}

ELECTRIC_BOILER_PRESETS = {
    "ealing_backup": {
        "description":          "Electric backup boiler (zero-gas-connection option)",
        "capacity_MW":            3.6,    # Mirrors gas boiler capacity for direct comparison
        "efficiency":             0.99,
        "reference":             "Sized to match Ealing gas boiler preset for comparison",
    },
    "small_electric": {
        "description":          "Small electric boiler (top-up duty)",
        "capacity_MW":            0.3,
        "efficiency":             0.98,
        "reference":             "Generic small electric boiler",
    },
}


# ── Part-load efficiency model (gas boilers) ──────────────────────────────────

def gas_boiler_part_load_efficiency(
    load_fraction: np.ndarray,
    eta_full_load: float,
    condensing: bool = True,
    min_turndown: float = 0.20,
) -> np.ndarray:
    """
    Part-load efficiency curve following DIN 4702-8 seasonal efficiency
    methodology. Condensing boilers gain efficiency at low load (lower flue
    gas temperatures improve condensing heat recovery); non-condensing
    boilers are roughly flat or slightly worse at low load (standing losses
    become proportionally larger).

    Parameters
    ----------
    load_fraction  : array of hourly load as a fraction of rated capacity (0-1)
    eta_full_load   : efficiency at 100% load (from datasheet/preset)
    condensing      : if True, efficiency RISES at lower load (condensing
                       benefit); if False, efficiency FALLS slightly at low
                       load (standing loss penalty dominates)
    min_turndown    : minimum load fraction the boiler can operate at before
                       cycling on/off (below this, treat as the min_turndown
                       efficiency — real boilers don't run stably below ~15-20%)

    Returns
    -------
    np.ndarray of efficiency values, same shape as load_fraction
    """
    load = np.clip(np.asarray(load_fraction, dtype=float), min_turndown, 1.0)

    if condensing:
        # Efficiency RISES towards low load — condensing benefit dominates
        eta_30pct = min(eta_full_load + 0.03, 0.98)
    else:
        # Efficiency FALLS slightly towards low load — standing losses dominate
        eta_30pct = max(eta_full_load - 0.04, 0.50)

    # Linear interpolation between the 30%-load point and the 100%-load point
    # (matches the two-point structure of the DIN 4702-8 standard)
    eta = eta_30pct + (eta_full_load - eta_30pct) * (
        (load - PART_LOAD_REFERENCE) / (1.0 - PART_LOAD_REFERENCE)
    )

    return np.clip(eta, 0.45, 0.99)


# ── Shared capacity resolver (GasBoiler + ElectricBoiler) ──────────────────────

def _resolve_capacity(
    capacity_MW: Optional[float],
    n_units: Optional[int],
    unit_capacity_MW: Optional[float],
) -> tuple:
    """
    Resolve a boiler's scale from EITHER capacity_MW directly OR
    (n_units AND unit_capacity_MW) — see GasBoiler/ElectricBoiler
    docstrings for why both forms exist (matches ASHPArray's pattern,
    needed for optimisation/sizing.py's discrete-unit capacity sweeps).

    Returns (capacity_MW, n_units, unit_capacity_MW) — always all three,
    regardless of which form was given, so summary()/resize() have
    consistent data either way.
    """
    n_pair_given = (n_units is not None) or (unit_capacity_MW is not None)

    if n_pair_given:
        if n_units is None or unit_capacity_MW is None:
            raise ValueError(
                "Must provide BOTH n_units and unit_capacity_MW together "
                "(or neither, and use capacity_MW directly)."
            )
        if capacity_MW is not None:
            raise ValueError(
                "Provide EITHER capacity_MW OR (n_units and unit_capacity_MW), not both."
            )
        n_units = int(n_units)
        unit_capacity_MW = float(unit_capacity_MW)
        return n_units * unit_capacity_MW, n_units, unit_capacity_MW

    if capacity_MW is None:
        raise ValueError(
            "Must provide either capacity_MW, or both n_units and unit_capacity_MW."
        )
    capacity_MW = float(capacity_MW)
    # Legacy/simple path: treat as "one unit of this size" so summary()
    # and resize() still have something sensible to report and scale from.
    return capacity_MW, 1, capacity_MW


# ── GasBoiler class ────────────────────────────────────────────────────────────

class GasBoiler:
    """
    Natural gas-fired peak/backup boiler.

    Always-available (no weather dependency) but high marginal cost — sits
    at the top of the dispatch merit order, called on only to cover peaks
    that cheaper sources (DC waste heat, ASHP) can't meet.

    supply_MW represents the MAXIMUM available capacity at every hour
    (boilers don't have a weather-driven availability constraint like
    ASHP or a planned-outage profile like DataCentre). The actual amount
    used is a dispatch decision made elsewhere.

    Efficiency varies with PART LOAD, not nameplate size (see module
    docstring) — pass an hourly load_fraction array via set_load_profile()
    once dispatch has run, or use the rated capacity_MW assumption for an
    initial/standalone sizing pass (defaults to full-load efficiency).

    Sizing — two equivalent ways to specify scale
    ---------------------------------------------------
    1. capacity_MW directly — a single design figure (e.g. "3.6 MW",
       matching how the Ealing report quotes it). Simple, and fine when
       you're not trying to model discrete real units.
    2. n_units + unit_capacity_MW — matches ASHPArray's pattern, and
       matches real practice: energy centres commonly install several
       smaller boilers in parallel rather than one giant unit, for
       redundancy and part-load turn-down flexibility. Use this when you
       want to sweep "how many units of THIS size do I need" via
       optimisation/sizing.py, the same way you would for ASHP.
    Provide EITHER capacity_MW OR (n_units AND unit_capacity_MW), not
    both. If you provide capacity_MW alone, n_units defaults to 1 and
    unit_capacity_MW defaults to the full capacity_MW (i.e. "one unit of
    that size") — so summary()/resize() behave consistently either way.

    Parameters
    ----------
    name                    : descriptive name for reporting
    capacity_MW             : rated thermal output (MW) — see sizing note above
    n_units                 : number of identical boiler units — see sizing note above
    unit_capacity_MW        : rated output per unit (MW) — see sizing note above
    condensing              : True for condensing technology (UK standard
                               since 2005), False for legacy non-condensing
    eta_full_load           : efficiency at 100% load (datasheet value)
    gas_price_GBP_per_MWh   : accepts None (default realistic tariff —
                               DESNZ central case, ~£25/MWh), a GasTariff
                               object (e.g. the more conservative
                               'current_actual' scenario, or escalated to
                               a future year), a flat scalar override, or
                               an 8760-length array. See economics/tariffs.py.
    carbon_price_GBP_per_tonne : carbon price applied to CO2e cost (£/tCO2e)
                               Default 0.0 — set explicitly if you want
                               carbon costs included in marginal cost
    """

    source_type = "gas_boiler"

    def __init__(
        self,
        name: str,
        capacity_MW: Optional[float] = None,
        n_units: Optional[int]       = None,
        unit_capacity_MW: Optional[float] = None,
        condensing: bool                = True,
        eta_full_load: float            = 0.92,
        gas_price_GBP_per_MWh           = None,
        carbon_price_GBP_per_tonne: float = 0.0,
        reference: str                  = "",
    ):
        self.name           = name
        self.capacity_MW, self.n_units, self.unit_capacity_MW = _resolve_capacity(
            capacity_MW, n_units, unit_capacity_MW
        )
        self.condensing     = condensing
        self.eta_full_load  = float(eta_full_load)
        self.carbon_price_GBP_per_tonne = float(carbon_price_GBP_per_tonne)
        self.reference      = reference

        # Gas price — None / Tariff / scalar / array, all resolved to a
        # clean 8760 £/MWh array. Default (None) now pulls in the DESNZ
        # central scenario rather than a flat placeholder.
        self._gas_price = resolve_gas_price(gas_price_GBP_per_MWh)

        # Default: assume full-load operation until a real load profile is set
        self._load_fraction = np.ones(N_HOURS)
        self.efficiency_hourly = gas_boiler_part_load_efficiency(
            self._load_fraction, self.eta_full_load, condensing
        )

        # Always fully available — no weather/outage dependency
        self.supply_MW     = np.full(N_HOURS, self.capacity_MW)
        self.supply_temp_C = np.full(N_HOURS, 90.0)  # Can reach full network temp

        self._recompute_marginal_cost()

    def set_load_profile(self, load_fraction: np.ndarray):
        """
        Update the boiler's operating load profile (0-1, fraction of
        capacity_MW) after a dispatch run, recalculating part-load
        efficiency and marginal cost accordingly.

        Parameters
        ----------
        load_fraction : np.ndarray (8760,) — actual dispatched load as a
                        fraction of capacity_MW at each hour
        """
        load_fraction = np.asarray(load_fraction, dtype=float)
        if len(load_fraction) != N_HOURS:
            raise ValueError(f"load_fraction must have {N_HOURS} elements.")

        self._load_fraction = load_fraction
        self.efficiency_hourly = gas_boiler_part_load_efficiency(
            load_fraction, self.eta_full_load, self.condensing
        )
        self._recompute_marginal_cost()

    def _recompute_marginal_cost(self):
        """Recalculate £/MWh_heat from current efficiency_hourly."""
        fuel_cost = self._gas_price / self.efficiency_hourly
        carbon_cost = (
            CARBON_INTENSITY["gas"] / self.efficiency_hourly
            * self.carbon_price_GBP_per_tonne
        )
        self.marginal_cost = fuel_cost + carbon_cost

        # Carbon intensity PER UNIT HEAT delivered (kgCO2e/kWh_heat), not
        # per unit gas burned — divides through by efficiency, same as the
        # carbon_cost term above, so a boiler running at poor part-load
        # efficiency correctly shows HIGHER carbon per unit heat delivered.
        # Used by dispatch.py's network-wide carbon compliance check
        # (London Heat Network Manual Table 8: max 0.216 kgCO2e/kWh).
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["gas"] / self.efficiency_hourly

    @classmethod
    def from_preset(cls, preset_key: str, **overrides) -> "GasBoiler":
        """
        Construct a GasBoiler from a named preset (see GAS_BOILER_PRESETS).

        Example
        -------
            boiler = GasBoiler.from_preset("ealing_phase1")
            boiler = GasBoiler.from_preset(
                "ealing_phase1",
                gas_price_GBP_per_MWh=GasTariff.from_scenario("current_actual"),
            )
        """
        if preset_key not in GAS_BOILER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(GAS_BOILER_PRESETS.keys())}"
            )
        params = GAS_BOILER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_config(cls, config: dict) -> "GasBoiler":
        """
        Construct a GasBoiler from a YAML/dict config block.

        Gas pricing in config — three ways to specify it
        ------------------------------------------------------
        1. Omit it entirely -> realistic default tariff (DESNZ central)
        2. A flat number -> gas_price_GBP_per_MWh: 30.0
        3. A scenario name -> builds a GasTariff for you:
               gas_tariff_scenario: current_actual
           This is the form a future scenario-menu UI would write.

        Example YAML block
        -------------------
            peak_sources:
              - type: gas_boiler
                name: "Peak/reserve gas boiler"
                capacity_MW: 3.6
                condensing: true
                eta_full_load: 0.92
                gas_tariff_scenario: desnz_central
        """
        cfg = {k: v for k, v in config.items() if k != "type"}

        # Named scenario string -> build a GasTariff object
        if "gas_tariff_scenario" in cfg:
            scenario_key = cfg.pop("gas_tariff_scenario")
            cfg["gas_price_GBP_per_MWh"] = GasTariff.from_scenario(scenario_key)

        return cls(**cfg)

    def resize(
        self,
        n_units: Optional[int] = None,
        unit_capacity_MW: Optional[float] = None,
    ) -> "GasBoiler":
        """
        Return a NEW GasBoiler at a different scale, reusing all other
        parameters (tariff, efficiency, condensing, etc.) from this
        instance. Does not mutate self. Mirrors ASHPArray.resize() —
        this is the hook optimisation/sizing.py expects for a
        "how many units do I need" capacity sweep.

        Example
        -------
            boiler_small = GasBoiler.from_preset("ealing_phase1")
            boiler_big   = boiler_small.resize(n_units=3)   # 3x the array
        """
        return GasBoiler(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            condensing=self.condensing,
            eta_full_load=self.eta_full_load,
            gas_price_GBP_per_MWh=self._gas_price,
            carbon_price_GBP_per_tonne=self.carbon_price_GBP_per_tonne,
            reference=self.reference,
        )

    def summary(self) -> dict:
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "capacity_MW":                self.capacity_MW,
            "condensing":                 self.condensing,
            "eta_full_load":              self.eta_full_load,
            "eta_at_current_load":        round(float(self.efficiency_hourly.mean()), 3),
            "mean_gas_price_GBP_per_MWh": round(float(self._gas_price.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"GasBoiler(name='{self.name}', {self.n_units}x{self.unit_capacity_MW:.2f}MW "
            f"= {self.capacity_MW:.1f} MW, condensing={self.condensing}, "
            f"η_full={self.eta_full_load:.0%})"
        )


# ── ElectricBoiler class ───────────────────────────────────────────────────────

class ElectricBoiler:
    """
    Electric resistive or electrode boiler — peak/backup duty, or a
    gas-free option where no gas connection exists/is wanted.

    Efficiency is near-constant regardless of load (no combustion process,
    so no flue losses or part-load combustion effects) — this is a genuine
    technical difference from gas boilers, not a simplification.

    Sizing — two equivalent ways to specify scale
    ---------------------------------------------------
    Same pattern as GasBoiler (see its docstring for the full rationale):
    either capacity_MW directly, or n_units + unit_capacity_MW for
    discrete real-unit sizing (matches ASHPArray, needed for
    optimisation/sizing.py capacity sweeps). Provide one or the other,
    not both.

    Parameters
    ----------
    name                          : descriptive name for reporting
    capacity_MW                   : rated thermal output (MW) — see sizing note above
    n_units                       : number of identical units — see sizing note above
    unit_capacity_MW              : rated output per unit (MW) — see sizing note above
    efficiency                    : conversion efficiency (0-1). Typical
                                     UK commercial electric boilers: 0.98-0.995
    electricity_price_GBP_per_MWh : accepts None (default realistic tariff —
                                     ~£240/MWh central commercial case), an
                                     ElectricityTariff object, a flat scalar
                                     override, or an 8760-length array.
                                     See economics/tariffs.py.
    carbon_price_GBP_per_tonne    : carbon price applied to CO2e cost
    """

    source_type = "electric_boiler"

    def __init__(
        self,
        name: str,
        capacity_MW: Optional[float] = None,
        n_units: Optional[int]       = None,
        unit_capacity_MW: Optional[float] = None,
        efficiency: float                       = 0.99,
        electricity_price_GBP_per_MWh           = None,
        carbon_price_GBP_per_tonne: float       = 0.0,
        reference: str                          = "",
    ):
        self.name        = name
        self.capacity_MW, self.n_units, self.unit_capacity_MW = _resolve_capacity(
            capacity_MW, n_units, unit_capacity_MW
        )
        self.efficiency  = float(efficiency)
        self.carbon_price_GBP_per_tonne = float(carbon_price_GBP_per_tonne)
        self.reference   = reference

        # Electricity price — None / Tariff / scalar / array, all resolved
        # to a clean 8760 £/MWh array. Default (None) now pulls in the
        # realistic central commercial tariff shape rather than a flat
        # placeholder.
        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)

        # Always fully available, constant efficiency regardless of load
        self.supply_MW     = np.full(N_HOURS, self.capacity_MW)
        self.supply_temp_C = np.full(N_HOURS, 90.0)
        self.efficiency_hourly = np.full(N_HOURS, self.efficiency)

        fuel_cost = self._elec_price / self.efficiency
        carbon_cost = (
            CARBON_INTENSITY["electric"] / self.efficiency
            * self.carbon_price_GBP_per_tonne
        )
        self.marginal_cost = fuel_cost + carbon_cost

        # Carbon intensity per unit heat delivered (kgCO2e/kWh_heat).
        # Constant across all 8760 hours — efficiency doesn't vary with
        # load (see class docstring) and this model uses a fixed annual
        # grid average rather than a time-varying grid carbon signal (a
        # known simplification — see CARBON_INTENSITY note in this
        # module). Used by dispatch.py's network-wide carbon compliance
        # check (London Heat Network Manual Table 8: max 0.216 kgCO2e/kWh).
        self.carbon_intensity_kgCO2_per_kWh = np.full(
            N_HOURS, CARBON_INTENSITY["electric"] / self.efficiency
        )

    @classmethod
    def from_preset(cls, preset_key: str, **overrides) -> "ElectricBoiler":
        """Construct from a named preset (see ELECTRIC_BOILER_PRESETS)."""
        if preset_key not in ELECTRIC_BOILER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(ELECTRIC_BOILER_PRESETS.keys())}"
            )
        params = ELECTRIC_BOILER_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_config(cls, config: dict) -> "ElectricBoiler":
        """
        Construct from a YAML/dict config block.

        Electricity pricing in config — three ways to specify it
        ------------------------------------------------------------
        1. Omit it entirely -> realistic default tariff (recommended default)
        2. A flat number -> electricity_price_GBP_per_MWh: 220.0
        3. A nested tariff block -> builds an ElectricityTariff for you:
               electricity_tariff:
                 negotiated_discount_pct: 10.0

        Example YAML block
        -------------------
            peak_sources:
              - type: electric_boiler
                name: "Electric backup boiler"
                capacity_MW: 1.0
                efficiency: 0.99
                electricity_tariff:
                  negotiated_discount_pct: 10.0
        """
        cfg = {k: v for k, v in config.items() if k != "type"}

        if "electricity_tariff" in cfg:
            tariff_kwargs = cfg.pop("electricity_tariff")
            cfg["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_kwargs)

        return cls(**cfg)

    def resize(
        self,
        n_units: Optional[int] = None,
        unit_capacity_MW: Optional[float] = None,
    ) -> "ElectricBoiler":
        """
        Return a NEW ElectricBoiler at a different scale, reusing all
        other parameters from this instance. Does not mutate self.
        Mirrors ASHPArray.resize() / GasBoiler.resize().
        """
        return ElectricBoiler(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            efficiency=self.efficiency,
            electricity_price_GBP_per_MWh=self._elec_price,
            carbon_price_GBP_per_tonne=self.carbon_price_GBP_per_tonne,
            reference=self.reference,
        )

    def summary(self) -> dict:
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "capacity_MW":                self.capacity_MW,
            "efficiency":                 self.efficiency,
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"ElectricBoiler(name='{self.name}', {self.n_units}x{self.unit_capacity_MW:.2f}MW "
            f"= {self.capacity_MW:.1f} MW, η={self.efficiency:.1%})"
        )


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_peak_demand_option.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_peak_demand_option.py\n"
    )
