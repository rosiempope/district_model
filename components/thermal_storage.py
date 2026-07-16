"""
thermal_storage.py
===================
Hot water thermal storage for the district energy system — models a
buffer/storage tank that can charge from surplus supply and discharge to
cover shortfalls, sitting between the source stack and the demand profile
in the dispatch loop.
 
Two distinct roles, both handled by this one class (just at different
scales — see presets)
-----------------------------------------------------------------------
1. OPERATIONAL BUFFERING — small (hundreds to low thousands of litres),
   exists to prevent heat pumps and boilers short-cycling, not to shift
   meaningful energy across the day. Sized using industry rules of thumb:
   25-50 litres per kW of heat pump/boiler capacity. Cheap, low-stakes,
   doesn't materially change network economics.
 
2. STRATEGIC / DIURNAL STORAGE — large (tens to thousands of m³), sized
   to actually shift hours of baseload surplus (EfW, data centre) into
   periods of higher demand, and to let ASHP/boiler charge preferentially
   during cheap electricity periods. This is the one that can genuinely
   change network feasibility — it lets you avoid oversizing expensive
   peaking plant to cover short demand spikes.
 
Capacity is left as a free, sweepable parameter (capacity_MWh) rather than
hard-coded, because the RIGHT size depends on the actual supply/demand
mismatch curve that comes out of dispatch — you can't size this properly
in a vacuum. Run dispatch with a few different capacities and compare.
 
Costing
-------
Thermal storage tank capex (£/kWh or £/m³) falls as the store gets bigger
— costs are dominated by fixed elements (controls, foundations, pumps)
that don't scale linearly with tank volume. This is modelled as a
power-law cost curve, fitted (not just "calibrated to roughly match")
against four REAL data points from DECC/Delta-EE, "Evidence Gathering:
Thermal Energy Storage (TES) Technologies" (2016, assets.publishing.
service.gov.uk/government/uploads/system/uploads/attachment_data/file/
545249/DELTA_EE_DECC_TES_Final__1_.pdf), Table 10 — specifically its
TTES (Tank Thermal Energy Storage) figures, i.e. hot-water TANKS, which
is genuinely the same technology this class models:
    domestic tanks:  ~£3,400/m³  (<0.5 m³, ≈£73/kWh)
    300 m³ tank:       £360/m³  (medium DH-scale,  ≈£7.7/kWh)
    4,300 m³ tank:      £114/m³  (large DH-scale,   ≈£2.5/kWh)
    12,000 m³ tank:      £91/m³  (very large DH-scale, ≈£2.0/kWh)
(£/kWh conversions use the same 46.5 kWh/m³ at 40K delta-T used
throughout this module — see mwh_to_m3()/m3_to_mwh() below.) A log-log
least-squares fit across these four real points gives a reference cost
of ≈£18,050/MWh at a 1 MWh reference scale and a scale exponent of
≈-0.357 — reproducing all four real points to within ~10%.

This REPLACES an earlier version of this curve, which blended in a
€0.4-0.6/kWh "Danish large-scale tank storage" figure as its large-scale
anchor. That figure was actually for PTES (Pit Thermal Energy Storage —
a lined earth pit, e.g. Marstal/Dronninglund), a genuinely CHEAPER,
DIFFERENT technology from TTES — the same DECC report states explicitly
that "PTES costs are lower than tank costs." Anchoring a tank-storage
curve on pit-storage data overstated how cheap this class's own
technology (hot water TANKS) gets at large scale. The fit above uses
only real TTES data points, so it should track this module's own
physics correctly at every scale, not just at the two ends.
 
Physics
-------
Energy stored: Q = V x rho x cp x dT
For water: rho ≈ 1000 kg/m3, cp ≈ 4.186 kJ/kg.K = 1.163 Wh/kg.K
So 1 m³ of water with a 40°C usable delta-T stores:
    1000 kg x 1.163 Wh/kg.K x 40 K = 46,520 Wh ≈ 46.5 kWh
 
Round-trip losses are split evenly across charge and discharge (each
applies sqrt(round_trip_efficiency)) rather than all on one side — this
is a common simplification and doesn't materially change results at the
efficiencies involved (94-97% round trip is typical for a well-insulated
hot water tank; this is NOT a battery, losses are much smaller).
 
Standing losses (heat loss to surroundings while idle) are modelled as a
small fractional loss per hour, consistent with the requirement that
well-insulated systems lose roughly 1-2% of stored heat per 24 hours.
 
Usage
-----
    from thermal_storage import ThermalStorage
 
    # Operational buffer (rule-of-thumb sized against a heat pump)
    buffer = ThermalStorage.from_buffer_rule(
        name="ASHP buffer vessel",
        connected_capacity_MW=2.8,         # the ASHP array it buffers
        litres_per_kW=40,
    )
 
    # Strategic storage — capacity is a free parameter, sweep it
    store = ThermalStorage(
        name="Strategic diurnal store",
        capacity_MWh=20.0,
        max_charge_MW=5.0,
        max_discharge_MW=5.0,
    )
 
    # Each dispatch hour:
    unmet_surplus_MW, shortfall_MW = store.step(net_surplus_MW)
    # net_surplus_MW > 0  → supply exceeds demand, store tries to absorb it
    # net_surplus_MW < 0  → demand exceeds supply, store tries to cover it
"""
 
import numpy as np
import pandas as pd
from typing import Optional
 
 
# ── Constants ──────────────────────────────────────────────────────────────────
 
N_HOURS = 8760
 
WATER_DENSITY_KG_M3   = 1000.0
WATER_SPECIFIC_HEAT_WH_KGK = 1.163   # Wh per kg per Kelvin
 
# Cost curve — log-log least-squares fit to four REAL DECC/Delta-EE (2016)
# TTES (tank) data points spanning domestic buffer tanks to 12,000 m³
# district-heating-scale tanks — see module docstring "Costing" section
# for the full real sourcing, the four data points, and why this
# replaced an earlier curve that incorrectly anchored on PTES (pit
# storage) data, a cheaper and genuinely different technology.
# Treat as a sensitivity input, not a quoted price — get a real quote
# before using this for an investment decision.
STORAGE_COST_REFERENCE_MWH   = 1.0      # Reference scale point (MWh)
STORAGE_COST_GBP_PER_MWH_AT_REF = 18_050.0   # Cost at the reference scale
STORAGE_COST_SCALE_EXPONENT  = -0.357   # Cost per MWh falls as scale grows
 
 
# ── Storage cost model ─────────────────────────────────────────────────────────
 
def estimate_storage_capex(capacity_MWh: float) -> float:
    """
    Estimate total CAPEX (£) for a thermal storage tank of given capacity,
    using a power-law cost curve (cost per MWh falls as scale increases).
 
    This is a ROUGH ESTIMATE for feasibility-stage screening — get a real
    quote before using this for an investment decision. Fitted to four
    real DECC/Delta-EE (2016) TTES (tank storage) data points spanning
    domestic buffer tanks to 12,000 m³ district-heating-scale tanks —
    see module docstring "Costing" section — but the exponent and
    reference cost remain sensitivity inputs, not certainties.
    """
    if capacity_MWh <= 0:
        return 0.0
    cost_per_MWh = STORAGE_COST_GBP_PER_MWH_AT_REF * (
        capacity_MWh / STORAGE_COST_REFERENCE_MWH
    ) ** STORAGE_COST_SCALE_EXPONENT
    return cost_per_MWh * capacity_MWh
 
 
def mwh_to_m3(energy_MWh: float, delta_T_K: float = 40.0) -> float:
    """
    Convert a thermal energy quantity (MWh) to the equivalent hot water
    tank volume (m³) for a given usable temperature differential.
 
    Default delta_T of 40K is typical for a UK LTHW network buffer
    (e.g. 80°C charged / 40°C discharged-to floor).
    """
    energy_Wh = energy_MWh * 1_000_000
    mass_kg = energy_Wh / (WATER_SPECIFIC_HEAT_WH_KGK * delta_T_K)
    volume_m3 = mass_kg / WATER_DENSITY_KG_M3
    return volume_m3
 
 
def m3_to_mwh(volume_m3: float, delta_T_K: float = 40.0) -> float:
    """Inverse of mwh_to_m3 — tank volume (m³) to stored energy (MWh)."""
    mass_kg = volume_m3 * WATER_DENSITY_KG_M3
    energy_Wh = mass_kg * WATER_SPECIFIC_HEAT_WH_KGK * delta_T_K
    return energy_Wh / 1_000_000
 
 
# ── ThermalStorage class ───────────────────────────────────────────────────────
 
class ThermalStorage:
    """
    Hot water thermal storage tank with charge/discharge dynamics for use
    in the dispatch loop.
 
    Call .step(net_surplus_MW) once per dispatch hour. Positive surplus
    (supply > demand) charges the store; negative surplus (demand > supply)
    discharges it. Returns (unmet_surplus_MW, shortfall_MW) — both are
    normally 0, but become non-zero if the store is full (can't absorb
    more surplus, which is then curtailed/wasted) or empty (can't cover
    the full shortfall, which becomes genuine unmet demand).
 
    State of charge (soc_MWh) is tracked internally and clipped to
    [0, capacity_MWh] at every step to avoid floating-point drift.
 
    Parameters
    ----------
    name                       : descriptive name for reporting
    capacity_MWh               : usable thermal storage capacity (MWh).
                                  Free parameter — size this by sweeping
                                  values against real dispatch curves,
                                  not by guessing up front.
    max_charge_MW              : maximum charging rate (MW)
    max_discharge_MW           : maximum discharging rate (MW)
    round_trip_efficiency      : fraction of energy recovered after a full
                                  charge/discharge cycle (0-1). Default 0.95
                                  — well-insulated hot water tanks have low
                                  losses compared to e.g. batteries.
    standing_loss_pct_per_hour : fractional heat loss per hour while idle
                                  (self-discharge). Default ~0.0008/hour
                                  ≈ 2% per 24 hours, consistent with a
                                  well-insulated large tank.
    initial_soc_fraction       : starting state of charge as a fraction of
                                  capacity (0-1). Default 0.5 (half full).
    delta_T_K                  : usable temperature differential (K), used
                                  only for volume reporting via summary().
    capex_GBP                  : override the cost-curve estimate with a
                                  real quote, if you have one. If None,
                                  uses estimate_storage_capex().
    """
 
    def __init__(
        self,
        name: str,
        capacity_MWh: float,
        max_charge_MW: float,
        max_discharge_MW: float,
        round_trip_efficiency: float       = 0.95,
        standing_loss_pct_per_hour: float  = 0.0008,
        initial_soc_fraction: float        = 0.5,
        delta_T_K: float                    = 40.0,
        capex_GBP: Optional[float]          = None,
        dispatch_strategy: str              = "displace_boiler",
    ):
        self.name                       = name
        self.capacity_MWh               = float(capacity_MWh)
        self.max_charge_MW              = float(max_charge_MW)
        self.max_discharge_MW           = float(max_discharge_MW)
        self.round_trip_efficiency      = float(round_trip_efficiency)
        self.standing_loss_pct_per_hour = float(standing_loss_pct_per_hour)
        self.delta_T_K                  = float(delta_T_K)
        if dispatch_strategy not in {"displace_boiler", "peak_reserve"}:
            raise ValueError("dispatch_strategy must be 'displace_boiler' or 'peak_reserve'")
        self.dispatch_strategy = dispatch_strategy
 
        self.soc_MWh = self.capacity_MWh * float(initial_soc_fraction)
 
        self.capex_GBP = (
            float(capex_GBP) if capex_GBP is not None
            else estimate_storage_capex(self.capacity_MWh)
        )
 
        # History tracking — populated as .step() is called
        self.soc_history          = []
        self.charge_history_MW    = []
        self.discharge_history_MW = []
        self.unmet_surplus_history_MW = []
        self.shortfall_history_MW     = []
 
    def step(self, net_surplus_MW: float, dt_hours: float = 1.0) -> tuple[float, float]:
        """
        Advance the storage by one timestep.
 
        Parameters
        ----------
        net_surplus_MW : supply minus demand for this hour (MW).
                         Positive = surplus available to charge the store.
                         Negative = shortfall the store is asked to cover.
        dt_hours       : timestep length in hours (default 1.0 for hourly
                         dispatch; pass 0.5 for half-hourly data etc.)
 
        Returns
        -------
        (unmet_surplus_MW, shortfall_MW)
            unmet_surplus_MW : surplus that could NOT be stored (tank full
                                or charge rate exceeded) — this is curtailed/
                                wasted energy, not an error.
            shortfall_MW     : demand that could NOT be covered by the store
                                (tank empty or discharge rate exceeded) —
                                this is genuine unmet demand that the next
                                source in the merit order (or nothing) must
                                cover.
        """
        if net_surplus_MW > 0:
            # Charging
            max_possible_MW = min(net_surplus_MW, self.max_charge_MW)
            headroom_MWh = max(0.0, self.capacity_MWh - self.soc_MWh)
            max_by_headroom_MW = headroom_MWh / dt_hours
            actual_charge_MW = min(max_possible_MW, max_by_headroom_MW)
 
            energy_stored_MWh = (
                actual_charge_MW * dt_hours * np.sqrt(self.round_trip_efficiency)
            )
            self.soc_MWh = min(self.capacity_MWh, self.soc_MWh + energy_stored_MWh)
 
            unmet_surplus_MW = net_surplus_MW - actual_charge_MW
            shortfall_MW = 0.0
            discharge_MW = 0.0
            charge_MW = actual_charge_MW
 
        else:
            # Discharging (or idle if exactly zero)
            requested_MW = -net_surplus_MW
            max_possible_MW = min(requested_MW, self.max_discharge_MW)
            max_by_soc_MW = max(0.0, self.soc_MWh) / dt_hours
            actual_discharge_MW = min(max_possible_MW, max_by_soc_MW)
 
            energy_removed_MWh = (
                actual_discharge_MW * dt_hours / np.sqrt(self.round_trip_efficiency)
            )
            self.soc_MWh = max(0.0, self.soc_MWh - energy_removed_MWh)
 
            shortfall_MW = requested_MW - actual_discharge_MW
            unmet_surplus_MW = 0.0
            charge_MW = 0.0
            discharge_MW = actual_discharge_MW
 
        # Standing loss applied after charge/discharge, then re-clip for safety
        self.soc_MWh *= (1.0 - self.standing_loss_pct_per_hour * dt_hours)
        self.soc_MWh = max(0.0, min(self.capacity_MWh, self.soc_MWh))
 
        self.soc_history.append(self.soc_MWh)
        self.charge_history_MW.append(charge_MW)
        self.discharge_history_MW.append(discharge_MW)
        self.unmet_surplus_history_MW.append(unmet_surplus_MW)
        self.shortfall_history_MW.append(shortfall_MW)
 
        return unmet_surplus_MW, shortfall_MW
 
    def run_series(self, net_surplus_MW_series: np.ndarray, dt_hours: float = 1.0) -> dict:
        """
        Run the storage across a full hourly series in one call (convenience
        wrapper around repeated .step() calls — useful for testing the
        storage in isolation before wiring it into the full dispatch loop).
 
        Parameters
        ----------
        net_surplus_MW_series : np.ndarray of hourly net surplus/shortfall (MW)
 
        Returns
        -------
        dict with soc_MWh, unmet_surplus_MW, shortfall_MW arrays (same
        length as input), plus summary stats.
        """
        for val in net_surplus_MW_series:
            self.step(val, dt_hours)
 
        return {
            "soc_MWh":              np.array(self.soc_history),
            "charge_MW":            np.array(self.charge_history_MW),
            "discharge_MW":         np.array(self.discharge_history_MW),
            "unmet_surplus_MW":     np.array(self.unmet_surplus_history_MW),
            "shortfall_MW":         np.array(self.shortfall_history_MW),
        }
 
    def reset(self, initial_soc_fraction: float = 0.5):
        """Reset the storage to a fresh state (e.g. before a new scenario run)."""
        self.soc_MWh = self.capacity_MWh * initial_soc_fraction
        self.soc_history          = []
        self.charge_history_MW    = []
        self.discharge_history_MW = []
        self.unmet_surplus_history_MW = []
        self.shortfall_history_MW     = []
 
    @classmethod
    def from_buffer_rule(
        cls,
        name: str,
        connected_capacity_MW: float,
        litres_per_kW: float = 40.0,
        delta_T_K: float = 40.0,
        max_charge_fraction: float = 1.0,
        max_discharge_fraction: float = 1.0,
        **kwargs,
    ) -> "ThermalStorage":
        """
        Build a small OPERATIONAL buffer tank using the standard industry
        rule of thumb: litres per kW of connected plant capacity.
 
        Typical values (see module docstring for sources):
            ASHP:            40-50 litres/kW
            Defrost cycling: 25 litres/kW (BS EN 14511 guidance)
            Biomass boiler:  50-100 litres/kW
 
        Parameters
        ----------
        connected_capacity_MW : capacity of the plant this buffer serves (MW)
        litres_per_kW          : sizing rule of thumb (default 40, ASHP-typical)
        delta_T_K               : usable temperature differential for volume
                                   → energy conversion (default 40K)
        max_charge_fraction,
        max_discharge_fraction : charge/discharge rate as a fraction of
                                  connected_capacity_MW (default both 1.0,
                                  i.e. the buffer can charge/discharge at
                                  the full rate of the plant it serves)
 
        Example
        -------
            buffer = ThermalStorage.from_buffer_rule(
                "ASHP buffer vessel", connected_capacity_MW=2.8
            )
        """
        connected_capacity_kW = connected_capacity_MW * 1000
        volume_litres = connected_capacity_kW * litres_per_kW
        volume_m3 = volume_litres / 1000
        capacity_MWh = m3_to_mwh(volume_m3, delta_T_K)
 
        return cls(
            name=name,
            capacity_MWh=capacity_MWh,
            max_charge_MW=connected_capacity_MW * max_charge_fraction,
            max_discharge_MW=connected_capacity_MW * max_discharge_fraction,
            delta_T_K=delta_T_K,
            **kwargs,
        )
 
    def summary(self) -> dict:
        """Return key parameters and (if run) performance stats as a dict."""
        volume_m3 = mwh_to_m3(self.capacity_MWh, self.delta_T_K)
 
        result = {
            "name":                  self.name,
            "capacity_MWh":          round(self.capacity_MWh, 3),
            "capacity_m3":           round(volume_m3, 1),
            "max_charge_MW":         self.max_charge_MW,
            "max_discharge_MW":      self.max_discharge_MW,
            "round_trip_efficiency": self.round_trip_efficiency,
            "estimated_capex_GBP":   round(self.capex_GBP, 0),
        }
 
        if self.soc_history:
            unmet = np.array(self.unmet_surplus_history_MW)
            short = np.array(self.shortfall_history_MW)
            result.update({
                "hours_run":               len(self.soc_history),
                "mean_soc_fraction":       round(float(np.mean(self.soc_history)) / self.capacity_MWh, 2) if self.capacity_MWh > 0 else 0,
                "annual_curtailed_MWh":     round(float(unmet.sum()), 1),
                "annual_unmet_demand_MWh":  round(float(short.sum()), 1),
                "annual_throughput_MWh":   round(float(np.sum(self.discharge_history_MW)), 1),
            })
 
        return result
 
    def __repr__(self):
        return (
            f"ThermalStorage(name='{self.name}', capacity={self.capacity_MWh:.2f} MWh, "
            f"max_charge={self.max_charge_MW:.1f} MW, max_discharge={self.max_discharge_MW:.1f} MW)"
        )
 

if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_thermal_storage.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_thermal_storage.py\n"
    )
