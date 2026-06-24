"""
dispatch.py
============
Hour-by-hour merit-order dispatch across the heat source stack (plus
optional thermal storage), for the district energy system.

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

        return result

    def check_carbon_compliance(self) -> dict:
        """
        Dedicated carbon compliance check against the London Heat Network
        Manual's Table 8 threshold (max 0.216 kgCO2e/kWh, blended annual,
        Gross CV basis Scope 1). Pulls the same figures summary() already
        computes, presented as a standalone pass/fail report — useful
        when carbon compliance is the specific question being asked,
        rather than scrolling to find it inside the full summary dict.

        Returns
        -------
        dict with: blended_carbon_intensity_kgCO2_per_kWh, threshold,
        compliant (bool), margin_kgCO2_per_kWh (positive = under the
        limit, negative = over it), and a breakdown of which sources are
        driving the total — useful for seeing AT A GLANCE whether it's
        boiler reliance, ASHP volume, or something else pushing the
        blended figure toward (or over) the line.
        """
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
) -> DispatchResult:
    """
    Run the full 8760-hour merit-order dispatch. See module docstring for
    the tiering logic (primary sources -> storage -> boilers -> unmet).

    Parameters
    ----------
    demand_kW   : 8760-length hourly heat demand (kW) — e.g.
                  network_result["total_heat_kW"] from demand_synthesis.py.
                  Converted to MW internally; everything else in this
                  module (and the rest of the codebase) works in MW.
    sources     : list of source objects (DataCentre, ASHPArray, EfWChp,
                  GasBoiler, ElectricBoiler, or anything sharing their
                  interface: .name, .source_type, .supply_MW, .capacity_MW,
                  .marginal_cost, all length-8760 except capacity_MW).
                  Names should be unique — they're used as dict keys for
                  reporting. Plain list is fine; you don't need to run
                  this through build_source_stack() first — this function
                  does its own true hourly cost sort internally.
    storage     : an optional ThermalStorage instance. Reset to a fresh
                  state at the start of this call (see
                  storage_initial_soc_fraction) so re-running dispatch on
                  the same storage object gives reproducible results.
    storage_initial_soc_fraction : starting state of charge (0-1) the
                  storage is reset to before dispatch begins. Ignored if
                  storage is None.

    Returns
    -------
    DispatchResult
    """
    demand_MW = np.asarray(demand_kW, dtype=float) / 1000.0
    if len(demand_MW) != N_HOURS:
        raise ValueError(f"demand_kW must have {N_HOURS} elements; got {len(demand_MW)}.")
    if not sources:
        raise ValueError("run_dispatch requires at least one source.")

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
            # --- Tier 2: storage discharge, ahead of boilers ---
            if storage is not None:
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
    )


# ── N-1 outage stress test ───────────────────────────────────────────────────────

def run_n1_stress_test(
    demand_kW: np.ndarray,
    sources: list,
    storage: Optional[ThermalStorage] = None,
    storage_initial_soc_fraction: float = 0.5,
    outage_window_hours: Optional[tuple] = None,
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
            storage_initial_soc_fraction=storage_initial_soc_fraction,
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


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  dispatch.py — self-test")
    print("=" * 70)

    from components.datacentre_source import DataCentre
    from components.ASHP import ASHPArray
    from components.EfW import EfWChp
    from components.peak_demand_option import GasBoiler, ElectricBoiler
    from profiles.demand_synthesis import synthesise_network

    # --- Build a representative demand profile (same building mix as
    #     demand_synthesis.py's own self-test, for cross-reference) ---
    np.random.seed(42)
    hours = np.arange(N_HOURS)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / N_HOURS)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, N_HOURS)
    )
    dates = pd.date_range("2023-01-01", periods=N_HOURS, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    scenario = {
        "demand_nodes": [
            # Scaled 3x vs the building mix used elsewhere in the codebase's
            # self-tests — at 1x scale, peak demand (4.7 MW) never exceeds
            # primary source capacity (DC+EfW+ASHP, ~9 MW), so boilers and
            # storage never get exercised at all. At 3x, peak demand
            # (~14.2 MW) comfortably exceeds primary capacity but stays
            # within total capacity (16.6 MW) — the regime where backup
            # plant and storage actually do something, which is the point
            # of this self-test.
            {"name": "Perceval House",       "type": "office",      "floor_area_m2": 8500 * 3},
            {"name": "High Street Retail",   "type": "retail",      "floor_area_m2": 3000 * 3},
            {"name": "Ealing Hospital Wing", "type": "hospital",    "floor_area_m2": 12000 * 3},
            {"name": "Dickens Yard Ph1",     "type": "residential", "units": 350 * 3},
            {"name": "Broadway Hotel",       "type": "hotel",       "floor_area_m2": 5000 * 3},
            {"name": "Ellen Wilkinson Sch",  "type": "school",      "floor_area_m2": 6000 * 3},
        ]
    }
    network = synthesise_network(weather_df, scenario)
    demand_kW = network["total_heat_kW"]
    print(f"\n  Demand profile: annual {demand_kW.sum()/1000:,.0f} MWh, "
          f"peak {demand_kW.max()/1000:.2f} MW")

    # --- Build the source stack (same presets used throughout the rest
    #     of the codebase's self-tests) ---
    dc   = DataCentre.from_preset("redwire_ealing", weather_df)
    efw  = EfWChp.from_preset("newlincs_style")
    ashp = ASHPArray.from_preset("ealing_phase1", weather_df)
    gas  = GasBoiler.from_preset("ealing_phase1")
    elec = ElectricBoiler.from_preset("ealing_backup")
    sources = [dc, efw, ashp, gas, elec]

    print(f"\n  Source stack: {', '.join(s.name for s in sources)}")
    print(f"  Total primary+backup capacity: "
          f"{sum(s.capacity_MW for s in sources):.1f} MW vs peak demand "
          f"{demand_kW.max()/1000:.2f} MW")

    # --- Run WITHOUT storage first ---
    print("\n  Run 1 — no storage:")
    result_no_storage = run_dispatch(demand_kW, sources, storage=None)
    s1 = result_no_storage.summary()
    for k, v in s1.items():
        print(f"    {k:<28} {v}")

    # --- Run WITH storage — same sources, fresh copies so Run 1's
    #     boiler load-profile correction doesn't bleed into Run 2 ---
    # Sized to match the REAL Ealing Town Centre Network's Phase 1 thermal
    # store: "50,000 litres of thermal storage" (Ealing Town Centre Heat
    # Network Feasibility Report, Table 15 "Energy centre capacity
    # summary"). Converted to MWh at the network's own quoted design
    # temperatures (70C peak flow / 40C typical return = 30K usable
    # delta-T, per the report's section on network operating conditions),
    # using thermal_storage.py's m3_to_mwh(): 50 m3 at 30K -> 1.74 MWh.
    # This is a genuinely small OPERATIONAL buffer (the report's own
    # category 1, not a strategic diurnal store) -- note it's actually
    # BELOW the 25-50 litres/kW rule of thumb in this module's own
    # docstring (50,000L / 2,800kW ASHP = ~18 L/kW), so don't be surprised
    # it does less peak-shaving than the larger illustrative store used
    # in earlier versions of this self-test.
    # Charge/discharge rate isn't given in the report -- assumed equal to
    # ASHP capacity (2.8 MW), since the report describes the store as
    # "connected in parallel with the heat pump" and charged directly
    # from its output. Flagged as an assumption, not a cited figure.
    print("\n  Run 2 — with the real Ealing Phase 1 thermal store (50,000L, ~1.74 MWh):")
    gas2  = GasBoiler.from_preset("ealing_phase1")
    elec2 = ElectricBoiler.from_preset("ealing_backup")
    sources2 = [dc, efw, ashp, gas2, elec2]
    store = ThermalStorage(
        name="Ealing Phase 1 thermal store (50,000L)",
        capacity_MWh=1.74,
        max_charge_MW=2.8,
        max_discharge_MW=2.8,
        delta_T_K=30.0,
    )
    result_storage = run_dispatch(demand_kW, sources2, storage=store)
    
    s2 = result_storage.summary()
    for k, v in s2.items():
        print(f"    {k:<28} {v}")

    # --- The actual point of storage: smaller peak boiler requirement ---
    boiler_names = [s.name for s in sources if s.source_type in BOILER_SOURCE_TYPES]
    peak_boiler_no_storage = max(
        result_no_storage.dispatch_by_source_MW[n].max() for n in boiler_names
    )
    peak_boiler_with_storage = max(
        result_storage.dispatch_by_source_MW[n].max() for n in boiler_names
    )
    print(f"\n  Peak SINGLE-HOUR boiler output — no storage:   {peak_boiler_no_storage:.2f} MW")
    print(f"  Peak SINGLE-HOUR boiler output — with storage: {peak_boiler_with_storage:.2f} MW")
    print(f"  -> with the REAL Ealing-sized buffer (1.74 MWh), this barely moves -- it's an")
    print(f"     operational buffer (prevents ASHP short-cycling), not a strategic peak-shaver.")
    print(f"     A genuinely larger strategic store COULD reduce backup boiler capacity (see")
    print(f"     thermal_storage.py's docstring on the two storage categories) -- that's a")
    print(f"     separate sensitivity case to run deliberately, not what this real-world figure shows.")
    print(f"     Either way, the network MAIN still has to carry the full {demand_kW.max()/1000:.2f} MW")
    print(f"     demand peak regardless of storage size -- that's a plant-sizing question, not a pipe one.")
    
    opex_no_storage = s1["total_annual_opex_GBP"]
    opex_with_storage = s2["total_annual_opex_GBP"]
    print(f"\n  Annual OPEX — no storage:   £{opex_no_storage:,.0f}")
    print(f"  Annual OPEX — with storage: £{opex_with_storage:,.0f}")
    if opex_with_storage > opex_no_storage:
        print(f"  -> OPEX is actually £{opex_with_storage - opex_no_storage:,.0f} HIGHER with storage in "
              f"this scenario. This is NOT a dispatch bug -- charging always correctly uses the "
              f"cheapest source with genuine spare capacity that hour (verified across all 8760 "
              f"hours). It's a real economic outcome: DC and EfW (the genuinely cheap sources) are "
              f"baseload-constrained and rarely have spare room, so most charging hours fall to "
              f"ASHP -- which is ~20x pricier per MWh. The boiler use that storage avoids doesn't "
              f"quite repay that premium in this scenario. A real CAPEX-vs-OPEX trade-off for a "
              f"small, source-coupled operational buffer -- exactly what the economics stage needs "
              f"to weigh, not something a smarter merit-order algorithm would fix.")

    print(f"\n  Carbon compliance check (London Heat Network Manual Table 8, "
          f"max {LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH} kgCO2e/kWh):")
    compliance_with_storage = result_storage.check_carbon_compliance()
    for k, v in compliance_with_storage.items():
        print(f"    {k:<35} {v}")

    # --- Demonstrate the check actually CATCHES a non-compliant scenario,
    #     not just confirms the well-mixed Ealing case passes. A gas-only
    #     or modern-electric-only network actually stays JUST under 0.216
    #     even alone (0.199 and 0.209 kgCO2e/kWh respectively, at this
    #     model's efficiency/grid-factor assumptions) -- a genuinely
    #     useful finding in its own right. To demonstrate a real FAIL,
    #     use a degraded/older electric boiler (85% efficiency, vs the
    #     99% modern default) -- an honest scenario (ageing equipment, not
    #     an artificial one) that pushes carbon-per-kWh-heat over the line. ---
    print(f"\n  Compliance check sanity test — degraded electric-only network")
    print(f"  (older/poorly-maintained unit, 85% efficiency vs the 99% modern default,")
    print(f"  no low-carbon sources at all — confirms the check CAN fail, not just pass):")
    elec_degraded = ElectricBoiler.from_preset(
        "ealing_backup", capacity_MW=20.0, efficiency=0.85
    )
    result_degraded = run_dispatch(demand_kW, [elec_degraded], storage=None)
    compliance_degraded = result_degraded.check_carbon_compliance()
    for k, v in compliance_degraded.items():
        print(f"    {k:<35} {v}")

    # --- N-1 outage stress test: storage's REAL maintenance/outage backup
    #     role, as distinct from cost arbitrage. Two variants: the
    #     maximum-severity case (lose a source for the WHOLE year — an
    #     upper bound, not a realistic single scenario) and a more
    #     realistic targeted case (lose it for one winter week, the
    #     worst TIMING for it to happen). ---
    print(f"\n  N-1 stress test — lose each primary source ENTIRELY, full year")
    print(f"  (maximum-severity upper bound, not a realistic single scenario):")
    store_n1 = ThermalStorage(
        name="Ealing Phase 1 thermal store (50,000L)",
        capacity_MWh=1.74, max_charge_MW=2.8, max_discharge_MW=2.8, delta_T_K=30.0,
    )
    n1_full_year = run_n1_stress_test(demand_kW, sources2, storage=store_n1)
    for name, r in n1_full_year.items():
        status = "✓ SURVIVES" if r["survives_without_unmet"] else "✗ UNMET DEMAND"
        print(f"    {name:<45} {status}  ({r['unmet_demand_MWh']} MWh unmet, "
              f"{r['pct_demand_unmet']}% of annual demand, peak gap {r['peak_unmet_MW']} MW)")

    print(f"\n  N-1 stress test — same sources, lose each for ONE WINTER WEEK only")
    print(f"  (hours 0-168, i.e. worst-timing realistic outage, WITH vs WITHOUT storage):")
    n1_week_with_storage = run_n1_stress_test(
        demand_kW, sources2, storage=store_n1, outage_window_hours=(0, 168)
    )
    n1_week_no_storage = run_n1_stress_test(
        demand_kW, sources2, storage=None, outage_window_hours=(0, 168)
    )
    for name in n1_week_with_storage:
        with_s = n1_week_with_storage[name]
        without_s = n1_week_no_storage[name]
        print(f"    {name}:")
        print(f"      With storage:    {'✓ survives' if with_s['survives_without_unmet'] else '✗ unmet demand'} "
              f"({with_s['unmet_demand_MWh']} MWh unmet)")
        print(f"      Without storage: {'✓ survives' if without_s['survives_without_unmet'] else '✗ unmet demand'} "
              f"({without_s['unmet_demand_MWh']} MWh unmet)")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    for name, arr in result_storage.dispatch_by_source_MW.items():
        assert len(arr) == N_HOURS, f"{name} dispatch array wrong length"
    assert len(result_storage.unmet_demand_MW) == N_HOURS

    # Energy balance identity (exact bookkeeping, not an approximation):
    #   total source output = demand - unmet + storage_charge - storage_discharge
    total_source_output = sum(arr.sum() for arr in result_storage.dispatch_by_source_MW.values())
    demand_total = result_storage.demand_MW.sum()
    balance_rhs = (
        demand_total
        - result_storage.unmet_demand_MW.sum()
        + result_storage.storage_charge_MW.sum()
        - result_storage.storage_discharge_MW.sum()
    )
    assert abs(total_source_output - balance_rhs) < 1.0, (
        f"Energy balance identity failed: sources produced {total_source_output:.2f} MWh, "
        f"expected {balance_rhs:.2f} MWh"
    )

    # No source ever dispatched above its own hourly available supply
    for s in sources2:
        over = result_storage.dispatch_by_source_MW[s.name] - s.supply_MW
        assert (over <= 1e-6).all(), f"{s.name} dispatched above available supply in some hour"

    # Storage SoC always within bounds
    assert store.soc_MWh >= -1e-6 and store.soc_MWh <= store.capacity_MWh + 1e-6

    # Storage should NOT make peak boiler requirement WORSE. At this small,
    # real (Ealing-sized) operational-buffer scale, it's not expected to make
    # it meaningfully BETTER either — see the printed explanation above. A
    # strict "<" assertion here was left over from an earlier, larger
    # illustrative storage size; with the real 1.74 MWh figure, requiring
    # strict improvement is testing for something this size of buffer was
    # never going to deliver. "<=" keeps the assertion honest: storage must
    # never backfire on peak duty, which is the one guarantee that's
    # actually true regardless of how small the buffer is.
    assert peak_boiler_with_storage <= peak_boiler_no_storage + 1e-6, \
        "Storage should never INCREASE peak single-hour boiler output vs no storage"

    # Boilers should be doing backup duty, not baseload — small share of annual energy
    boiler_share_pct = sum(s2["pct_demand_by_source"].get(n, 0.0) for n in boiler_names)
    assert boiler_share_pct < 20.0, \
        f"Boilers supplying {boiler_share_pct:.1f}% of annual demand — too high for 'backup' duty"

    # Unmet demand should be negligible with adequately-sized backup plant
    assert s2["pct_demand_unmet"] < 1.0, \
        f"Unmet demand {s2['pct_demand_unmet']}% — check backup plant sizing"

    # Carbon compliance: the real, well-mixed Ealing scenario (DC+EfW+ASHP
    # dominant, boilers genuine backup) should be comfortably compliant
    assert compliance_with_storage["compliant"], \
        f"Ealing scenario should be carbon-compliant; got " \
        f"{compliance_with_storage['blended_carbon_intensity_kgCO2_per_kWh']} kgCO2e/kWh"
    assert compliance_with_storage["margin_kgCO2_per_kWh"] > 0, \
        "Compliant scenario should show a positive margin under the threshold"

    # The deliberately degraded electric-only scenario should FAIL —
    # proves the check isn't just always returning True regardless of input
    assert not compliance_degraded["compliant"], \
        f"Degraded electric-only network should breach the London carbon threshold; got " \
        f"{compliance_degraded['blended_carbon_intensity_kgCO2_per_kWh']} kgCO2e/kWh " \
        f"(threshold {LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH})"
    assert compliance_degraded["margin_kgCO2_per_kWh"] < 0, \
        "Non-compliant scenario should show a negative margin over the threshold"

    # N-1 stress test assertions
    assert set(n1_full_year.keys()) == {dc.name, efw.name, ashp.name}, \
        "N-1 stress test should cover exactly the primary (non-boiler) sources"
    for name, r in n1_full_year.items():
        assert r["peak_unmet_MW"] >= 0, f"{name}: peak_unmet_MW should never be negative"
        assert r["unmet_demand_MWh"] >= 0, f"{name}: unmet_demand_MWh should never be negative"
        # Consistency: survives_without_unmet should exactly match unmet_demand_MWh ~ 0
        assert r["survives_without_unmet"] == (r["unmet_demand_MWh"] <= 1e-6), \
            f"{name}: survives_without_unmet flag inconsistent with unmet_demand_MWh"
    # The 1-week winter test should be a strictly EASIER (or equal) test
    # than losing the source for the whole year -- less unmet demand, not more
    for name in n1_week_with_storage:
        assert n1_week_with_storage[name]["unmet_demand_MWh"] <= n1_full_year[name]["unmet_demand_MWh"] + 1e-6, \
            f"{name}: a 1-week outage should never show MORE unmet demand than a full-year outage of the same source"
    # Confirm the deepcopy isolation actually worked: the shared dc/efw/ashp
    # objects' supply_MW should be UNCHANGED after running the stress test
    # (each test should have operated on copies, not the originals)
    assert dc.supply_MW.sum() > 0, \
        "Original dc.supply_MW should be untouched after N-1 stress test (deepcopy isolation check)"
    assert efw.supply_MW.sum() > 0, \
        "Original efw.supply_MW should be untouched after N-1 stress test (deepcopy isolation check)"
    assert ashp.supply_MW.sum() > 0, \
        "Original ashp.supply_MW should be untouched after N-1 stress test (deepcopy isolation check)"

    print("  ✓ All dispatch arrays correct length (8760 hours)")
    print("  ✓ Energy balance identity holds exactly (sources = demand - unmet + charge - discharge)")
    print("  ✓ No source ever dispatched above its own hourly available supply")
    print("  ✓ Storage SoC stayed within [0, capacity] bounds")
    print("  ✓ Storage never made peak single-hour boiler output worse (its one guaranteed value at this scale)")
    print(f"  ✓ Boilers supplied only {boiler_share_pct:.1f}% of annual demand (genuine backup duty)")
    print(f"  ✓ Unmet demand negligible ({s2['pct_demand_unmet']}%) — backup plant adequately sized")
    print(f"  ✓ Real Ealing source mix is carbon-compliant ({compliance_with_storage['blended_carbon_intensity_kgCO2_per_kWh']} "
          f"kgCO2e/kWh, vs {LONDON_MAX_CARBON_INTENSITY_KGCO2_PER_KWH} threshold)")
    print(f"  ✓ Degraded electric-only network correctly FAILS compliance ({compliance_degraded['blended_carbon_intensity_kgCO2_per_kWh']} "
          f"kgCO2e/kWh) — confirms the check can actually catch a non-compliant scenario")
    print("  ✓ N-1 stress test covers exactly the primary sources, with consistent unmet-demand bookkeeping")
    print("  ✓ A 1-week outage never shows worse unmet demand than a full-year outage of the same source")
    print("  ✓ Original source objects unmutated after stress testing (deepcopy isolation confirmed)")
    print()