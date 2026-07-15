"""
dispatch.py
============
Hour-by-hour merit-order dispatch across the heat source stack (plus
optional thermal storage), for the district energy system.

Heating AND cooling, one engine
---------------------------------
This module handles BOTH duties via run_dispatch(...,duty="heat") /
run_dispatch(...,duty="cool") (and the same parameter on
run_n1_stress_test()) — the merit-order algorithm itself doesn't care
which commodity it's moving, only supply_MW/marginal_cost/
carbon_intensity_kgCO2_per_kWh, which every source type exposes
identically regardless of duty (ASHPArray/EfWChp/DataCentre/
BoosterHeatPump for heating; AirCooledChiller for cooling). duty only
controls (a) which network_topology methods get called (heat loss vs.
heat GAIN have the same Shukhov-formula physics, opposite sign — see
network_topology.py), and (b) gates check_carbon_compliance() to
heating only, since the London Heat Network Manual's 0.216 kgCO2e/kWh
threshold has no cited cooling equivalent in this project's sources.
The rest of this docstring uses "heat sources" as the generic term
throughout, applying equally to cooling unless stated otherwise.

What this answers
------------------
Given a demand profile and a stack of heat sources (+ optional thermal
storage), decide HOW MUCH each source delivers at every one of the 8760
hours, cheapest-first. This is what turns a pile of source/demand/storage
objects into real annual OPEX, source utilisation, and a check on whether
your backup plant is actually sized big enough.


Dispatch logic — tiered, not pure cost-ranking
--------------------------------------------------
1. PRIMARY sources (everything except gas_boiler/electric_boiler) are
   dispatched cheapest-first by ACTUAL hourly marginal cost (not a static
   annual average — this is what lets the tariff shape in
   economics/tariffs.py actually matter: an ASHP that's expensive at 6pm
   and cheap at 3am gets used accordingly).
2. Any shortfall remaining after primary sources is covered by thermal
   storage discharge NEXT, ahead of backup boilers — stored energy was
   already paid for at the (cheap) price it was charged at, so using it
   is "free" at the point of use, cheaper than firing a boiler regardless
   of the boiler's instantaneous tariff.
3. Whatever shortfall remains after storage is covered by backup boilers
   (gas_boiler / electric_boiler), cheapest-first by hourly marginal cost.
   Boilers are deliberately tiered LAST by source TYPE, not pure cost
   ranking — even though an electric boiler occasionally undercuts a
   frozen-COP ASHP on a bitter night, real heat network controls don't
   swap to backup plant for a few pence of arbitrage; backup plant exists
   for reliability and peak-shaving, not economic optimisation. This
   matches the role peak_demand_option.py's own docstring describes for
   these sources.
4. If primary sources fully meet demand with capacity to spare, that
   spare capacity (the cheapest source with headroom, never a more
   expensive UNTOUCHED one) is offered to thermal storage to charge, up
   to the storage's own rate/headroom limits.
5. If, after all of the above, demand still isn't fully met, that's
   genuine unmet demand — should be ~zero with adequately-sized backup
   plant; a persistently nonzero figure is a sizing red flag, not
   something this module tries to paper over.

This is a single-pass heuristic, not a true rolling optimisation (it does
not look ahead to decide whether to hold back cheap capacity now in
favour of a bigger price spike six hours later). That's a deliberate
scope decision for a feasibility-stage screening tool — a full optimal-
control storage dispatch is a worthwhile Phase 2 upgrade, not something
this pilot needs to get a credible LCOH/NPV/IRR answer.

Boiler part-load efficiency — single-pass correction
---------------------------------------------------------
GasBoiler/ElectricBoiler's marginal_cost depends on their OWN realised
load fraction (see peak_demand_option.py's set_load_profile()) — but the
dispatch decision is what determines that load fraction, a circularity.
This module resolves it with one pass: dispatch assuming full-load
efficiency (the boiler classes' own default), then calls
set_load_profile() on each boiler with its REALISED hourly load,
mutating it in place so its reported efficiency/cost/summary() reflects
what actually happened. Dispatch itself is NOT re-run with the corrected
costs — in practice boilers run in genuinely peaky, low-hour-count
conditions where a few % efficiency swing essentially never flips the
merit order, so a second pass buys negligible accuracy for real added
complexity. Call run_dispatch() again afterwards (the boiler objects now
carry corrected costs) if you want to check that assumption holds for
your specific scenario.

Usage
-----
    from optimisation.dispatch import run_dispatch

    result = run_dispatch(
        demand_kW=network_result["total_heat_kW"],   # from demand_synthesis.py
        sources=[dc, efw, ashp, gas_boiler, elec_boiler],
        storage=my_thermal_storage,   # optional, pass None to skip
    )

    print(result.summary())
    df = result.to_dataframe()
"""

import sys
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.thermal_storage import ThermalStorage


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# Sources of these types are tiered LAST in dispatch (peak/backup duty),
# regardless of their instantaneous cost ranking — see module docstring.
BOILER_SOURCE_TYPES = {"gas_boiler", "electric_boiler"}

_EPS = 1e-9   # floating-point tolerance for "is this essentially zero"

# London Heat Network Manual, Table 8 "DH Manual Design Standards",
# row 9 "Carbon intensity of heat supply": "Maximum 0.216 kgCO2e/kWh
# (on a Gross CV basis for Scope 1 emissions)". Source cited in the
# manual: DECC/Defra (www.defra.gov.uk/environment/economy/business-
# efficiency/reporting/). This is a BLENDED ANNUAL figure across the
# whole network's heat supply, not a per-source or per-hour limit — see
# check_carbon_compliance() below for how it's actually evaluated.
LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH = 0.216


# ── DispatchResult ──────────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """
    Full hourly dispatch output, plus the source objects used (kept around
    so summary() can look up each source's marginal_cost for OPEX
    reporting and set_load_profile() has already been called on any
    boilers — see module docstring).
    """
    demand_MW:              np.ndarray
    dispatch_by_source_MW:  dict            # name -> np.ndarray(8760,)
    storage_charge_MW:      np.ndarray
    storage_discharge_MW:   np.ndarray
    storage_soc_MWh:        np.ndarray
    unmet_demand_MW:        np.ndarray
    curtailed_surplus_MW:   np.ndarray      # spare capacity storage couldn't absorb (full/rate-limited)
    sources:                list
    has_storage:            bool
    building_demand_MW:     Optional[np.ndarray] = None   # demand_MW BEFORE network heat loss was added (None if no network_topology was used)
    network_heat_loss_MW:   Optional[np.ndarray] = None   # the hourly network loss that was added on top of building_demand_MW (None if no network_topology was used)
    duty:                   str = "heat"   # "heat" or "cool" — what this result was actually dispatched for; see run_dispatch()'s duty parameter

    def summary(self) -> dict:
        """Annual energy, cost, and utilisation breakdown — the headline numbers."""
        demand_total = float(self.demand_MW.sum())
        unmet_total  = float(self.unmet_demand_MW.sum())

        by_source_MWh = {name: float(arr.sum()) for name, arr in self.dispatch_by_source_MW.items()}
        by_source_cost_GBP = {
            s.name: float((self.dispatch_by_source_MW[s.name] * s.marginal_cost).sum())
            for s in self.sources
        }
        total_opex_GBP = sum(by_source_cost_GBP.values())

        # Carbon: sum(source output x that source's own hourly carbon
        # intensity), same pattern as the cost calculation above. Storage
        # charge/discharge isn't carbon-tagged separately — the energy's
        # carbon content is attributed at the point it was GENERATED
        # (i.e. whichever source's dispatched total it's already folded
        # into), not re-attributed at the point of storage discharge.
        # This avoids double-counting: dispatch_by_source_MW for the
        # charging source already includes the extra MW sent to storage,
        # so multiplying by that source's own carbon intensity captures
        # it exactly once. See check_carbon_compliance() for the London
        # Heat Network Manual threshold check built on this same figure.
        by_source_carbon_kgCO2 = {
            s.name: float((self.dispatch_by_source_MW[s.name] * s.carbon_intensity_kgCO2_per_kWh * 1000.0).sum())
            for s in self.sources
            if hasattr(s, "carbon_intensity_kgCO2_per_kWh")
        }
        total_carbon_kgCO2 = sum(by_source_carbon_kgCO2.values())
        # Blended intensity per kWh of HEAT DELIVERED TO DEMAND (not per
        # kWh generated) — matches how the London Heat Network Manual
        # states its threshold ("carbon intensity of heat supply").
        # demand_total is in MWh; x1000 converts to kWh for the per-kWh figure.
        blended_carbon_kgCO2_per_kWh = (
            total_carbon_kgCO2 / (demand_total * 1000.0) if demand_total > 0 else 0.0
        )

        result = {
            "annual_demand_MWh":         round(demand_total, 0),
            "annual_unmet_demand_MWh":   round(unmet_total, 2),
            "pct_demand_unmet":          round(unmet_total / demand_total * 100, 3) if demand_total > 0 else 0.0,
            "annual_MWh_by_source":      {k: round(v, 0) for k, v in by_source_MWh.items()},
            "pct_demand_by_source":      {
                k: round(v / demand_total * 100, 1) for k, v in by_source_MWh.items()
            } if demand_total > 0 else {},
            "annual_cost_GBP_by_source": {k: round(v, 0) for k, v in by_source_cost_GBP.items()},
            "total_annual_opex_GBP":     round(total_opex_GBP, 0),
            "peak_demand_MW":            round(float(self.demand_MW.max()), 2),
            "peak_unmet_MW":             round(float(self.unmet_demand_MW.max()), 3),
            "annual_carbon_tCO2_by_source": {
                k: round(v / 1000.0, 2) for k, v in by_source_carbon_kgCO2.items()
            },
            "total_annual_carbon_tCO2": round(total_carbon_kgCO2 / 1000.0, 1),
            "blended_carbon_intensity_kgCO2_per_kWh": round(blended_carbon_kgCO2_per_kWh, 4),
            "london_carbon_compliant": bool(
                blended_carbon_kgCO2_per_kWh <= LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH
            ),
        }

        if self.has_storage:
            result.update({
                "storage_annual_charge_MWh":    round(float(self.storage_charge_MW.sum()), 1),
                "storage_annual_discharge_MWh": round(float(self.storage_discharge_MW.sum()), 1),
                "storage_annual_curtailed_MWh": round(float(self.curtailed_surplus_MW.sum()), 1),
                "storage_mean_soc_MWh":         round(float(self.storage_soc_MWh.mean()), 2),
            })

        if self.network_heat_loss_MW is not None:
            building_demand_total = float(self.building_demand_MW.sum())
            network_loss_total = float(self.network_heat_loss_MW.sum())
            result.update({
                "annual_building_demand_MWh": round(building_demand_total, 0),
                "annual_network_heat_loss_MWh": round(network_loss_total, 0),
                "network_loss_pct_of_building_demand": round(
                    network_loss_total / building_demand_total * 100, 2
                ) if building_demand_total > 0 else 0.0,
            })

        return result

    def check_carbon_compliance(self) -> dict:
        """
        Dedicated carbon compliance check against the London Heat Network
        Manual's Table 8 threshold (max 0.216 kgCO2e/kWh, blended annual,
        Gross CV basis Scope 1). Pulls the same figures summary() already
        computes, presented as a standalone pass/fail report — useful
        when carbon compliance is the specific question being asked,
        rather than scrolling to find it inside the full summary dict.

        HEATING-ONLY: raises if self.duty != "heat". The 0.216
        kgCO2e/kWh threshold is a specific HEATING regulatory figure
        (London Heat Network Manual Table 8, "Carbon intensity of heat
        supply") — there is no cited cooling-carbon-intensity equivalent
        in this project's source documents, so calling this on a
        cooling-duty result would silently produce a meaningless
        compliance verdict against the wrong regulation. If a real
        cooling carbon threshold is identified later, this gating can be
        relaxed with a genuinely sourced cooling-specific limit — not by
        reusing the heating figure.

        Returns
        -------
        dict with: blended_carbon_intensity_kgCO2_per_kWh, threshold,
        compliant (bool), margin_kgCO2_per_kWh (positive = under the
        limit, negative = over it), and a breakdown of which sources are
        driving the total — useful for seeing AT A GLANCE whether it's
        boiler reliance, ASHP volume, or something else pushing the
        blended figure toward (or over) the line.
        """
        if self.duty != "heat":
            raise ValueError(
                f"check_carbon_compliance() is heating-only (London Heat Network Manual's "
                f"0.216 kgCO2e/kWh threshold has no cited cooling equivalent in this "
                f"project's source documents) — this result was dispatched for "
                f"duty='{self.duty}'. See this method's docstring."
            )
        s = self.summary()
        margin = LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH - s["blended_carbon_intensity_kgCO2_per_kWh"]
        return {
            "blended_carbon_intensity_kgCO2_per_kWh": s["blended_carbon_intensity_kgCO2_per_kWh"],
            "threshold_kgCO2_per_kWh": LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH,
            "compliant": s["london_carbon_compliant"],
            "margin_kgCO2_per_kWh": round(margin, 4),
            "total_annual_carbon_tCO2": s["total_annual_carbon_tCO2"],
            "annual_carbon_tCO2_by_source": s["annual_carbon_tCO2_by_source"],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the full hourly result to a DataFrame for export/plotting."""
        df = pd.DataFrame({
            "demand_MW":            self.demand_MW,
            "unmet_demand_MW":      self.unmet_demand_MW,
            "storage_charge_MW":    self.storage_charge_MW,
            "storage_discharge_MW": self.storage_discharge_MW,
            "storage_soc_MWh":      self.storage_soc_MWh,
            "curtailed_surplus_MW": self.curtailed_surplus_MW,
        })
        for name, arr in self.dispatch_by_source_MW.items():
            safe = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").lower()
            df[f"dispatch_{safe}_MW"] = arr
        return df


# ── Core dispatch loop ──────────────────────────────────────────────────────────

def run_dispatch(
    demand_kW: np.ndarray,
    sources: list,
    storage: Optional[ThermalStorage] = None,
    storage_initial_soc_fraction: float = 0.5,
    network_topology=None,
    network_flow_temp_C: float = 70.0,
    network_return_temp_C: float = 40.0,
    network_sized_segments: Optional[dict] = None,
    duty: str = "heat",
) -> DispatchResult:
    """
    Run the full 8760-hour merit-order dispatch. See module docstring for
    the tiering logic (primary sources -> storage -> boilers -> unmet).

    The merit-order ALGORITHM itself is duty-agnostic — sort sources
    cheapest-first, fill demand, spill surplus to storage, fall back to
    boilers, track unmet demand. Nothing about that logic cares whether
    the commodity is heat or cooling; it only ever looks at
    supply_MW/marginal_cost/carbon_intensity_kgCO2_per_kWh, all of which
    every source type (heating or cooling) already exposes identically.
    duty exists ONLY to (a) select the right network_topology methods
    (size_all_segments/network_heat_loss_kW_hourly both take their own
    duty parameter — see network_topology.py) when network_topology is
    provided, and (b) gate check_carbon_compliance() to heating only,
    since the London Heat Network Manual's 0.216 kgCO2e/kWh threshold
    is a HEATING-specific regulatory figure with no cited cooling
    equivalent — see that method's own docstring.

    This is a deliberate design choice over building a separate
    cooling_dispatch.py file: duplicating the merit-order algorithm
    across two files would mean keeping two copies in sync forever (the
    same maintenance risk this project already hit once with duplicated
    constants drifting apart — see the project's file-restructuring
    work). One parameterised engine avoids that.

    Parameters
    ----------
    demand_kW   : 8760-length hourly demand (kW) for THIS duty — e.g.
                  network_result["total_heat_kW"] for duty="heat", or
                  network_result["total_cooling_kW"] for duty="cool"
                  (both from demand_synthesis.py). Converted to MW
                  internally; everything else in this module (and the
                  rest of the codebase) works in MW.
                  This is BUILDING demand only -- if network_topology is
                  provided, real network heat loss/gain is added ON TOP
                  of this automatically (see network_topology below);
                  don't pre-add a loss estimate to demand_kW yourself if
                  you're also passing network_topology, or it will be
                  double-counted.
    sources     : list of source objects for THIS duty. For duty="heat":
                  DataCentre, ASHPArray, EfWChp, BoosterHeatPump,
                  GasBoiler, ElectricBoiler. For duty="cool":
                  AirCooledChiller, or anything else sharing the same
                  interface (.name, .source_type, .supply_MW,
                  .capacity_MW, .marginal_cost, all length-8760 except
                  capacity_MW). This function does NOT check that every
                  source in the list actually matches the requested
                  duty — passing a heating source into a duty="cool" run
                  (or vice versa) will not raise an error, since the
                  merit-order algorithm itself doesn't need to know;
                  it's the CALLER's responsibility to pass the right
                  source list for the duty being dispatched. Names
                  should be unique — they're used as dict keys for
                  reporting.
    storage     : an optional ThermalStorage instance. Reset to a fresh
                  state at the start of this call (see
                  storage_initial_soc_fraction) so re-running dispatch on
                  the same storage object gives reproducible results.
    storage_initial_soc_fraction : starting state of charge (0-1) the
                  storage is reset to before dispatch begins. Ignored if
                  storage is None.
    network_topology : optional network.network_topology.NetworkTopology
                  instance — if provided, REAL hourly network heat loss
                  (or, for duty="cool", heat GAIN — see
                  network_heat_loss_kW_hourly()'s own note on this, the
                  same Shukhov formula handles both correctly by sign)
                  is computed for THIS duty and ADDED to demand_kW
                  before dispatch runs, so sources are sized and
                  dispatched against what they ACTUALLY need to supply
                  (building demand + real transport losses/gains), not
                  just building demand alone. If None (default),
                  dispatch runs exactly as before -- building demand
                  only, no network loss feedback.
    network_flow_temp_C, network_return_temp_C : the FIXED source flow/
                  return temperatures (°C) used for network sizing and
                  heat-loss calculation for THIS duty. For duty="heat",
                  defaults match this project's real Ealing design value
                  (70/40°C). For duty="cool", pass the chiller's actual
                  design temperatures (e.g. 6/12°C) explicitly — the
                  70/40°C defaults are NOT appropriate for cooling and
                  are not auto-switched based on duty, to avoid a
                  surprising silent default; always pass these
                  explicitly for duty="cool".
    network_sized_segments : optional pre-computed result from
                  network_topology.size_all_segments(duty=duty) — pass
                  this if you've already sized the network (e.g. to
                  also report CAPEX) to avoid sizing it twice. If None
                  and network_topology is provided, sizing is done
                  internally using network_flow_temp_C/
                  network_return_temp_C AND duty.
    duty        : "heat" (default) or "cool" — see the module-level
                  note above on what this actually controls.

    Returns
    -------
    DispatchResult. If network_topology was provided, also carries
    .network_heat_loss_MW (the hourly array that was added to demand)
    and .building_demand_MW (the ORIGINAL demand_kW, before loss was
    added) so the loss contribution can be inspected/reported separately
    from total demand.
    """
    demand_MW = np.asarray(demand_kW, dtype=float) / 1000.0
    if len(demand_MW) != N_HOURS:
        raise ValueError(f"demand_kW must have {N_HOURS} elements; got {len(demand_MW)}.")
    if not sources:
        raise ValueError("run_dispatch requires at least one source.")
    if duty not in ("heat", "cool"):
        raise ValueError(f"duty must be 'heat' or 'cool'; got '{duty}'.")

    building_demand_MW = demand_MW.copy()
    network_heat_loss_MW = None

    if network_topology is not None:
        sized = network_sized_segments
        if sized is None:
            sized = network_topology.size_all_segments(
                flow_temp_C=network_flow_temp_C, return_temp_C=network_return_temp_C, duty=duty,
            )
        loss_result = network_topology.network_heat_loss_kW_hourly(
            sized_segments=sized, source_flow_temp_C=network_flow_temp_C,
        )
        network_heat_loss_MW = loss_result["total_kW_hourly"] / 1000.0
        demand_MW = demand_MW + network_heat_loss_MW

    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(
            f"Source names must be unique (used as dispatch_by_source_MW dict "
            f"keys for reporting). Got: {names}"
        )

    if storage is not None:
        storage.reset(initial_soc_fraction=storage_initial_soc_fraction)

    primary_sources = [s for s in sources if s.source_type not in BOILER_SOURCE_TYPES]
    boiler_sources   = [s for s in sources if s.source_type in BOILER_SOURCE_TYPES]


    dispatch_by_source = {s.name: np.zeros(N_HOURS) for s in sources}
    storage_charge      = np.zeros(N_HOURS)
    storage_discharge   = np.zeros(N_HOURS)
    storage_soc          = np.zeros(N_HOURS)
    unmet                = np.zeros(N_HOURS)
    curtailed            = np.zeros(N_HOURS)

    for t in range(N_HOURS):
        remaining = demand_MW[t]
        marginal_source = None   # (source, spare_MW) — cheapest primary source with headroom

        # --- Tier 1: primary sources, cheapest-first by THIS HOUR's actual cost ---
        for s in sorted(primary_sources, key=lambda s: s.marginal_cost[t]):
            avail = s.supply_MW[t]
            take = min(remaining, avail)
            dispatch_by_source[s.name][t] = take
            remaining -= take
            if remaining <= _EPS:
                marginal_source = (s, avail - take)
                break
            # else: this source is fully exhausted with demand still
            # outstanding — move on to the next (more expensive) one

        if remaining > _EPS:
            peak_reserve = storage is not None and getattr(
                storage, "dispatch_strategy", "displace_boiler"
            ) == "peak_reserve"
            # Storage can either displace boiler fuel or be held back solely
            # for capacity shortfalls. The latter matches small buffer/peak
            # stores such as the Ealing concept design.
            if storage is not None and not peak_reserve:
                requested = remaining
                _, shortfall = storage.step(-requested)
                storage_discharge[t] = requested - shortfall
                storage_soc[t] = storage.soc_MWh
                remaining = shortfall

            # --- Tier 3: boilers, cheapest-first, cover whatever's left ---
            for s in sorted(boiler_sources, key=lambda s: s.marginal_cost[t]):
                if remaining <= _EPS:
                    break
                avail = s.supply_MW[t]
                take = min(remaining, avail)
                dispatch_by_source[s.name][t] = take
                remaining -= take

            if storage is not None and peak_reserve and remaining > _EPS:
                requested = remaining
                _, shortfall = storage.step(-requested)
                storage_discharge[t] = requested - shortfall
                storage_soc[t] = storage.soc_MWh
                remaining = shortfall

            if remaining > _EPS:
                unmet[t] = remaining   # genuine unmet demand — sizing red flag

        else:
            # Demand fully met by primary sources — offer the CHEAPEST
            # source's spare capacity (never a pricier untouched one) to
            # charge storage.
            if storage is not None:
                if marginal_source is not None and marginal_source[1] > _EPS:
                    src, spare_MW = marginal_source
                    unmet_surplus, _ = storage.step(spare_MW)
                    actual_charge = spare_MW - unmet_surplus
                    dispatch_by_source[src.name][t] += actual_charge
                    storage_charge[t] = actual_charge
                    curtailed[t] = unmet_surplus
                storage_soc[t] = storage.soc_MWh

    # --- Post-hoc boiler part-load correction (single pass, see docstring) ---
    for s in boiler_sources:
        if hasattr(s, "set_load_profile"):
            load_fraction = np.clip(dispatch_by_source[s.name] / s.capacity_MW, 0.0, 1.0)
            s.set_load_profile(load_fraction)

    return DispatchResult(
        demand_MW=demand_MW,
        dispatch_by_source_MW=dispatch_by_source,
        storage_charge_MW=storage_charge,
        storage_discharge_MW=storage_discharge,
        storage_soc_MWh=storage_soc,
        unmet_demand_MW=unmet,
        curtailed_surplus_MW=curtailed,
        sources=sources,
        has_storage=(storage is not None),
        building_demand_MW=building_demand_MW,
        network_heat_loss_MW=network_heat_loss_MW,
        duty=duty,
    )


# ── N-1 outage stress test ───────────────────────────────────────────────────────

def run_n1_stress_test(
    demand_kW: np.ndarray,
    sources: list,
    storage: Optional[ThermalStorage] = None,
    storage_initial_soc_fraction: float = 0.5,
    outage_window_hours: Optional[tuple] = None,
    duty: str = "heat",
) -> dict:
    """
    Worst-case single-source-loss ("N-1") stress test: for EACH primary
    source in turn, simulate it being COMPLETELY unavailable (zero supply)
    and re-run dispatch with everything else unchanged, to answer the
    question storage's "maintenance/outage backup" role is actually
    about: if this source goes down hard, does the REST of the stack
    (other sources + storage + boilers) still meet demand, or does
    genuine unmet demand appear?

    This is deliberately a different, harsher test than the normal
    weather/maintenance variation already baked into each source's own
    supply_MW (e.g. ASHP's per-unit outage model, EfW's annual planned
    shutdown, DataCentre's dispersed outages) — those represent NORMAL
    operation including realistic, survivable maintenance. This function
    asks "what if a source is unavailable for reasons beyond its own
    normal profile" — a transformer fault, a burst pipe at the energy
    centre, an extended unplanned shutdown — i.e. genuine N-1
    contingency planning, the standard utility-sector concept of "can the
    system survive losing its single largest/most-relied-on asset."

    Parameters
    ----------
    demand_kW            : same as run_dispatch()
    sources               : same as run_dispatch() — every PRIMARY source
                            (not boilers) gets tested in turn; boilers are
                            deliberately excluded from the "what if this
                            goes down" test since they're already the
                            backup layer, not something else backs THEM up
    storage               : optional ThermalStorage — tested fresh (reset)
                            for every source's stress-test run, same as a
                            normal run_dispatch() call
    storage_initial_soc_fraction : same as run_dispatch()
    outage_window_hours   : optional (start_hour, end_hour) tuple to limit
                            the outage to a specific window instead of the
                            full year — e.g. test losing a source ONLY
                            during winter peak (the worst-case timing),
                            rather than an unrealistic full-year loss.
                            If None (default), the source is zeroed for
                            ALL 8760 hours — the maximum-severity case,
                            useful for an at-a-glance worst case but not
                            necessarily the most realistic single
                            scenario; use the window form for a more
                            targeted "what if this fails during the cold
                            snap" question.
    duty                   : "heat" (default) or "cool" — passed straight
                            through to the internal run_dispatch() calls;
                            no other behaviour in this function depends
                            on it (BOILER_SOURCE_TYPES filtering already
                            works correctly for any duty's source types).

    Returns
    -------
    dict, keyed by the name of the source being tested as "down" -> {
        "unmet_demand_MWh": annual unmet demand with this source down,
        "pct_demand_unmet": as a % of annual demand,
        "peak_unmet_MW": worst single-hour shortfall,
        "storage_helped": bool — did storage discharge at all during
                           this source's simulated outage,
        "survives_without_unmet": bool — True if the rest of the stack
                           fully covered the gap (genuine pass/fail for
                           "can the network survive losing this source")
    }
    Boilers are not included as test subjects (see above), but ARE
    included in the remaining stack for every other source's test, since
    backup boilers stepping up is exactly the behaviour being checked.

    Every source and the storage object are DEEP-COPIED for each
    individual test, so neither this function's own repeated calls nor
    the caller's original objects are left with mutated state afterward
    (GasBoiler/ElectricBoiler's marginal_cost gets mutated in place by
    run_dispatch()'s part-load correction — see that function's
    docstring — so a shallow copy alone wouldn't be enough to isolate
    one test's run from the next).
    """
    primary_sources = [s for s in sources if s.source_type not in BOILER_SOURCE_TYPES]
    results = {}

    for failed_source in primary_sources:
        # Deep-copy EVERY source for this test, not just the failed one.
        # Boilers (GasBoiler/ElectricBoiler) get set_load_profile() called
        # on them by run_dispatch() itself (see that function's docstring
        # on the single-pass part-load correction) — that MUTATES their
        # marginal_cost in place. Without a fresh copy each iteration,
        # boiler state from one source's stress test would leak into the
        # next, and into whatever the caller does with the original
        # `sources` list afterward. deepcopy is used (not copy.copy) for
        # boilers specifically because their internal arrays
        # (efficiency_hourly, marginal_cost) need to be independent
        # copies, not shared references, for the mutation isolation to
        # actually work.
        sources_for_this_test = [copy.deepcopy(s) for s in sources]
        # Find the deep-copied equivalent of failed_source by name and
        # zero its supply for the outage window
        for s in sources_for_this_test:
            if s.name == failed_source.name:
                if outage_window_hours is None:
                    s.supply_MW[:] = 0.0
                else:
                    start, end = outage_window_hours
                    s.supply_MW[start:end] = 0.0
                # electrical_demand_MW (ASHP) or similar derived arrays
                # aren't recalculated here -- they're not read by
                # run_dispatch(), which only ever looks at supply_MW,
                # marginal_cost, and carbon_intensity_kgCO2_per_kWh.
                break

        stress_storage = copy.deepcopy(storage) if storage is not None else None

        result = run_dispatch(
            demand_kW, sources_for_this_test, storage=stress_storage,
            storage_initial_soc_fraction=storage_initial_soc_fraction, duty=duty,
        )

        unmet_total = float(result.unmet_demand_MW.sum())
        demand_total = float(result.demand_MW.sum())

        results[failed_source.name] = {
            "unmet_demand_MWh": round(unmet_total, 2),
            "pct_demand_unmet": round(unmet_total / demand_total * 100, 3) if demand_total > 0 else 0.0,
            "peak_unmet_MW": round(float(result.unmet_demand_MW.max()), 3),
            "storage_helped": bool(result.storage_discharge_MW.sum() > _EPS) if storage is not None else None,
            "survives_without_unmet": bool(unmet_total <= _EPS),
        }

    return results


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_dispatch.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_dispatch.py\n"
    )
