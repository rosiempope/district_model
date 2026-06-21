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

# Carbon intensities (kgCO2e/kWh fuel, BEIS/DESNZ 2024 conversion factors)
CARBON_INTENSITY = {
    "gas":      0.183,   # Natural gas, gross CV basis (Scope 1)
    "electric": 0.207,   # UK grid average 2024 (will fall over time — see note)
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

    Parameters
    ----------
    name                    : descriptive name for reporting
    capacity_MW             : rated thermal output (MW)
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
        capacity_MW: float,
        condensing: bool                = True,
        eta_full_load: float            = 0.92,
        gas_price_GBP_per_MWh           = None,
        carbon_price_GBP_per_tonne: float = 0.0,
        reference: str                  = "",
    ):
        self.name           = name
        self.capacity_MW    = float(capacity_MW)
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

    def summary(self) -> dict:
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
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
            f"GasBoiler(name='{self.name}', capacity={self.capacity_MW:.1f} MW, "
            f"condensing={self.condensing}, η_full={self.eta_full_load:.0%})"
        )


# ── ElectricBoiler class ───────────────────────────────────────────────────────

class ElectricBoiler:
    """
    Electric resistive or electrode boiler — peak/backup duty, or a
    gas-free option where no gas connection exists/is wanted.

    Efficiency is near-constant regardless of load (no combustion process,
    so no flue losses or part-load combustion effects) — this is a genuine
    technical difference from gas boilers, not a simplification.

    Parameters
    ----------
    name                          : descriptive name for reporting
    capacity_MW                   : rated thermal output (MW)
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
        capacity_MW: float,
        efficiency: float                       = 0.99,
        electricity_price_GBP_per_MWh           = None,
        carbon_price_GBP_per_tonne: float       = 0.0,
        reference: str                          = "",
    ):
        self.name        = name
        self.capacity_MW = float(capacity_MW)
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

    def summary(self) -> dict:
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "capacity_MW":                self.capacity_MW,
            "efficiency":                 self.efficiency,
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"ElectricBoiler(name='{self.name}', capacity={self.capacity_MW:.1f} MW, "
            f"η={self.efficiency:.1%})"
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  peak_demand_option.py — self-test")
    print("="*70)

    # Test part-load efficiency curve directly
    print("\n  Part-load efficiency curve (condensing, eta_full=0.92):")
    test_loads = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    eta_cond = gas_boiler_part_load_efficiency(test_loads, 0.92, condensing=True)
    eta_noncond = gas_boiler_part_load_efficiency(test_loads, 0.78, condensing=False)
    print(f"  {'Load':>6} {'Condensing':>12} {'Non-condensing':>16}")
    for l, ec, en in zip(test_loads, eta_cond, eta_noncond):
        print(f"  {l:>6.1f} {ec:>12.3f} {en:>16.3f}")

    # Test all gas boiler presets — now with the realistic default gas tariff
    print("\n  All gas boiler presets (default DESNZ central gas tariff):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Gas £/MWh':>11} {'Marg. cost £/MWh':>17} {'Condensing':>11}")
    print("  " + "-"*82)
    for key in GAS_BOILER_PRESETS:
        b = GasBoiler.from_preset(key)
        s = b.summary()
        print(f"  {key:<25} {s['capacity_MW']:>12.1f} {s['mean_gas_price_GBP_per_MWh']:>11.2f} "
              f"{s['mean_marginal_cost_GBP_per_MWh']:>17.2f} {str(s['condensing']):>11}")

    # Test all electric boiler presets — now with the realistic default electricity tariff
    print("\n  All electric boiler presets (default central commercial electricity tariff):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Elec £/MWh':>11} {'Marg. cost £/MWh':>17} {'Efficiency':>11}")
    print("  " + "-"*82)
    for key in ELECTRIC_BOILER_PRESETS:
        b = ElectricBoiler.from_preset(key)
        s = b.summary()
        print(f"  {key:<25} {s['capacity_MW']:>12.1f} {s['mean_electricity_price_GBP_per_MWh']:>11.2f} "
              f"{s['mean_marginal_cost_GBP_per_MWh']:>17.2f} {s['efficiency']:>11.1%}")

    # Detailed test: Ealing Phase 1 gas boiler
    print("\n  Ealing Phase 1 gas boiler (detailed, default tariff):")
    ealing_gas = GasBoiler.from_preset("ealing_phase1")
    for k, v in ealing_gas.summary().items():
        print(f"    {k:<36} {v}")

    # --- NEW: tariff integration tests for both boiler types ---
    print("\n  Gas tariff integration — comparing all four accepted price input types:")
    gas_default = GasBoiler.from_preset("ealing_phase1")
    gas_scenario = GasBoiler.from_preset(
        "ealing_phase1", gas_price_GBP_per_MWh=GasTariff.from_scenario("current_actual")
    )
    gas_flat = GasBoiler.from_preset("ealing_phase1", gas_price_GBP_per_MWh=45.0)
    gas_array = GasBoiler.from_preset("ealing_phase1", gas_price_GBP_per_MWh=np.full(N_HOURS, 60.0))
    print(f"    Default (None, DESNZ central) -> £{gas_default._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    GasTariff (current_actual)    -> £{gas_scenario._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_scenario.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Flat scalar override (£45)    -> £{gas_flat._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_flat.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Raw array override (£60)      -> £{gas_array._gas_price.mean():.2f}/MWh, "
          f"marginal £{gas_array.marginal_cost.mean():.2f}/MWh heat")

    print("\n  Electric boiler tariff integration:")
    elec_default = ElectricBoiler.from_preset("ealing_backup")
    elec_discounted = ElectricBoiler.from_preset(
        "ealing_backup", electricity_price_GBP_per_MWh=ElectricityTariff(negotiated_discount_pct=10.0)
    )
    print(f"    Default (None)        -> £{elec_default._elec_price.mean():.2f}/MWh, "
          f"marginal £{elec_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    10% discounted tariff -> £{elec_discounted._elec_price.mean():.2f}/MWh, "
          f"marginal £{elec_discounted.marginal_cost.mean():.2f}/MWh heat")

    # from_config tests
    print("\n  from_config() — gas boiler with named scenario:")
    gas_cfg = GasBoiler.from_config({
        "type": "gas_boiler", "name": "Peak boiler (config)", "capacity_MW": 3.6,
        "gas_tariff_scenario": "current_actual",
    })
    print(f"    {gas_cfg}  ->  £{gas_cfg._gas_price.mean():.2f}/MWh (expect ~£34.46, current_actual)")

    print("\n  from_config() — electric boiler with nested tariff block:")
    elec_cfg = ElectricBoiler.from_config({
        "type": "electric_boiler", "name": "Electric backup (config)", "capacity_MW": 1.0,
        "electricity_tariff": {"negotiated_discount_pct": 20.0},
    })
    print(f"    {elec_cfg}  ->  £{elec_cfg._elec_price.mean():.2f}/MWh (expect 20% below £240 central)")

    # Test set_load_profile — simulate a realistic part-load dispatch pattern
    print("\n  Testing set_load_profile() — simulated winter-peaking dispatch:")
    hours = np.arange(N_HOURS)
    # Boiler runs harder in winter (more peak shaving needed), idles in summer
    simulated_load = 0.15 + 0.55 * np.clip(
        np.cos(2 * np.pi * (hours - 0) / 8760), 0, 1
    )
    ealing_gas.set_load_profile(simulated_load)
    print(f"    Mean load fraction:        {simulated_load.mean():.2f}")
    print(f"    Mean efficiency (updated): {ealing_gas.efficiency_hourly.mean():.3f}")
    print(f"    Mean marginal cost:        £{ealing_gas.marginal_cost.mean():.2f}/MWh")

    # Compare gas vs electric boiler cost at realistic default prices
    print("\n  Cost comparison — Gas vs Electric boiler (same 3.6 MW capacity, REALISTIC default tariffs):")
    gas = GasBoiler.from_preset("ealing_phase1")
    elec = ElectricBoiler.from_preset("ealing_backup")
    print(f"    Gas boiler:      £{gas.marginal_cost.mean():.2f}/MWh heat  "
          f"(gas @ £{gas._gas_price.mean():.0f}/MWh DESNZ central, η={gas.eta_full_load:.0%})")
    print(f"    Electric boiler: £{elec.marginal_cost.mean():.2f}/MWh heat "
          f"(elec @ £{elec._elec_price.mean():.0f}/MWh central commercial, η={elec.efficiency:.0%})")
    print(f"    → Electric is {elec.marginal_cost.mean()/gas.marginal_cost.mean():.1f}x more expensive per MWh heat at these realistic prices")

    # Sanity checks
    print("\n  Sanity checks:")
    assert len(gas.supply_MW)      == N_HOURS, "GasBoiler supply_MW wrong length"
    assert len(gas.marginal_cost)  == N_HOURS, "GasBoiler marginal_cost wrong length"
    assert len(elec.supply_MW)     == N_HOURS, "ElectricBoiler supply_MW wrong length"
    assert gas.supply_MW.max()  <= gas.capacity_MW + 0.001, "Gas supply exceeds capacity"
    assert elec.supply_MW.max() <= elec.capacity_MW + 0.001, "Electric supply exceeds capacity"
    assert eta_cond[0] > eta_cond[-1], "Condensing boiler should be MORE efficient at low load"
    assert eta_noncond[0] < eta_noncond[-1], "Non-condensing boiler should be LESS efficient at low load"

    # New tariff-integration assertions
    assert abs(gas_default._gas_price.mean() - 24.57) < 0.5, \
        "Default gas price should be the DESNZ central tariff (~£24.57/MWh), not the old £45 placeholder"
    assert abs(elec_default._elec_price.mean() - 240.0) < 0.5, \
        "Default electricity price should be the realistic ~£240/MWh tariff, not the old £120 placeholder"
    assert gas_scenario._gas_price.mean() > gas_default._gas_price.mean(), \
        "current_actual gas scenario should be pricier than desnz_central"
    assert abs(gas_flat._gas_price.mean() - 45.0) < 0.01, "Flat scalar override should be respected exactly"
    assert elec_discounted._elec_price.mean() < elec_default._elec_price.mean(), \
        "Discounted electricity tariff should be cheaper than the undiscounted default"
    assert abs(gas_cfg._gas_price.mean() - 34.46) < 0.5, \
        "from_config gas_tariff_scenario should resolve to the named GasTariff scenario"
    assert abs(elec_cfg._elec_price.mean() - 240.0 * 0.80) < 1.0, \
        "from_config nested electricity_tariff block should apply the 20% discount correctly"

    print("  ✓ All array shapes correct (8760 hours)")
    print("  ✓ Supply never exceeds nameplate capacity")
    print("  ✓ Condensing boiler gains efficiency at low load (as expected)")
    print("  ✓ Non-condensing boiler loses efficiency at low load (as expected)")
    print("  ✓ Default gas price now uses DESNZ central tariff (~£25/MWh), not old £45 placeholder")
    print("  ✓ Default electricity price now uses realistic tariff (~£240/MWh), not old £120 placeholder")
    print("  ✓ GasTariff/ElectricityTariff objects, flat scalars, and raw arrays all behave correctly")
    print("  ✓ from_config() named scenario and nested tariff block both resolve correctly")
    print()