"""
auto_size.py
=============
Given a demand profile and a user's chosen technology *types* (not
capacities), recommend the MW split across those technologies. This is
the missing "model tells you what you need" step between demand
synthesis and dispatch.

Design principles
-----------------
1. BASELOAD-FIRST: lowest-marginal-cost source gets sized to cover the
   bulk of the annual load (typically 60-80% of peak). Peak sources
   cover the remainder.

2. The split is driven by a LOAD-DURATION approach — the demand sorted
   in descending order shows how many hours per year each MW of capacity
   is actually needed. Baseload plant that runs >4000 h/yr is economic;
   peak plant running <500 h/yr should be as cheap-per-MW as possible
   (gas boiler).

3. A DIVERSITY FACTOR is applied to the coincident peak before sizing —
   individual building peaks rarely coincide exactly. The default 0.85
   is the standard CIBSE/CHDU feasibility-stage assumption for a mixed-
   use scheme (offices + residential + retail — different occupancy
   patterns). A residential-only scheme would use ~0.90; a single
   building gets 1.0.

4. For ASHP, capacity must be DERATED for cold-weather performance —
   the nameplate MW at 7°C rating point delivers only ~65% at -5°C
   design day. The sizing accounts for this: if the derated winter
   output is less than the baseload allocation, more units are needed.

Usage
-----
    from optimisation.auto_size import recommend_sizing

    rec = recommend_sizing(
        demand_kW=network_result["total_heat_kW"],
        peak_demand_kW=network_result["peak_heat_kW"],
        technology_types=["ashp", "gas_boiler"],
        weather_df=weather,
        network_flow_temp_C=70.0,
    )
    print(rec["sources"])
    # [{"type":"ashp", "capacity_MW":3.2, "role":"baseload", ...},
    #  {"type":"gas_boiler", "capacity_MW":5.8, "role":"peak", ...}]
"""

import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Diversity factors ────────────────────────────────────────────────
# Source: CIBSE Guide C (2007), Table 2.1 — "Diversity factors for
# central plant sizing", confirmed by CHDU guidance on heat network
# feasibility studies. These are DEMAND-SIDE diversity factors (how
# much the coincident peak of the whole estate is below the arithmetic
# sum of individual building peaks), NOT supply redundancy factors.

DIVERSITY_FACTORS = {
    "mixed_use":        0.85,   # offices + residential + retail etc
    "residential_only": 0.90,   # residential blocks with similar profiles
    "commercial_only":  0.80,   # offices + retail (more synchronised peaks)
    "single_building":  1.00,   # no diversity possible
}

DEFAULT_DIVERSITY_FACTOR = 0.85

# ── Source classification ────────────────────────────────────────────
# Which source types are baseload candidates (run cheaply many hours)
# vs peak/backup (cheap CAPEX, expensive fuel, few hours).
BASELOAD_TYPES = {"ashp", "efw_chp", "data_centre"}
PEAK_TYPES = {"gas_boiler", "electric_boiler"}
# booster_heat_pump is not independently sized — it follows a DC source

# Default baseload fraction of diversified peak: how much of the peak
# the baseload source(s) should cover. 0.60 = 60% of peak → baseload
# plant runs at high utilisation; the remaining 40% is peak plant that
# runs only in the coldest hours.
DEFAULT_BASELOAD_FRACTION = 0.60

# ASHP cold-weather derating: at design-day ambient (-5°C typically),
# an ASHP's real thermal output is only this fraction of its nameplate
# capacity at the EN14825 rating point (7°C). Source: manufacturer
# performance tables + components/ASHP.py's _capacity_derate().
ASHP_DESIGN_DAY_DERATING = 0.65


def _classify_sources(technology_types):
    """Split user-requested types into baseload vs peak lists."""
    baseload = [t for t in technology_types if t in BASELOAD_TYPES]
    peak = [t for t in technology_types if t in PEAK_TYPES]
    other = [t for t in technology_types if t not in BASELOAD_TYPES and t not in PEAK_TYPES]
    return baseload, peak, other


def _load_duration_baseload_hours(demand_kW, baseload_capacity_kW):
    """How many hours per year the baseload plant runs at or above its
    rated capacity — the 'base' of the load-duration curve."""
    sorted_demand = np.sort(demand_kW)[::-1]
    above = np.sum(sorted_demand >= baseload_capacity_kW)
    return int(above)


def recommend_sizing(
    demand_kW,
    peak_demand_kW,
    technology_types,
    weather_df=None,
    network_flow_temp_C=70.0,
    diversity_factor=None,
    baseload_fraction=DEFAULT_BASELOAD_FRACTION,
    include_cooling=False,
    cooling_demand_kW=None,
    peak_cooling_kW=0.0,
    n_buildings=1,
    building_types=None,
    peak_is_coincident=True,
    network_loss_margin=0.05,
    resilience_mode="largest_ashp_unit_out",
):
    """
    Recommend MW capacities for each requested technology type.

    Parameters
    ----------
    demand_kW          : 8760-hour total heat demand (kW), including DHW
    peak_demand_kW     : arithmetic peak of demand_kW (kW)
    technology_types   : list of source type strings the user wants
                         (e.g. ["ashp", "gas_boiler"])
    weather_df         : required if "ashp" is in the list (for COP)
    network_flow_temp_C : network design flow temperature
    diversity_factor   : override; if None, auto-selected from building_types
    baseload_fraction  : fraction of diversified peak for baseload (0-1)
    include_cooling    : if True, also size cooling sources
    cooling_demand_kW  : 8760-hour cooling demand (kW)
    peak_cooling_kW    : arithmetic peak cooling demand
    n_buildings        : number of connected buildings
    building_types     : list of building type strings (for diversity selection)

    Returns
    -------
    dict: {
        "diversity_factor": float used,
        "diversified_peak_kW": peak after diversity,
        "sources": [{"type", "capacity_MW", "role", "rationale", ...}],
        "cooling_sources": [...] if include_cooling,
        "sizing_notes": [str] — human-readable explanation of the split,
    }
    """
    demand_kW = np.asarray(demand_kW, dtype=float)

    # Auto-select diversity factor from building mix
    if diversity_factor is None:
        diversity_factor = _auto_diversity(building_types, n_buildings)

    # An hourly aggregate profile already contains the coincidence/diversity
    # between buildings. Applying a second diversity factor undersized plant.
    # Keep the option only for callers supplying an arithmetic sum of peaks.
    diversity_applied = 1.0 if peak_is_coincident else diversity_factor
    diversified_peak_kW = peak_demand_kW * diversity_applied * (1.0 + network_loss_margin)
    diversified_peak_MW = diversified_peak_kW / 1000.0

    baseload_types, peak_types, other_types = _classify_sources(technology_types)
    notes = []
    sources = []

    notes.append(
        f"Arithmetic peak: {peak_demand_kW/1000:.2f} MW. "
        + ("Hourly aggregate is already coincident; no second diversity factor. "
           if peak_is_coincident else f"Diversity factor: {diversity_factor:.2f}. ")
        + f"Network-loss margin: {network_loss_margin*100:.0f}% → design peak: "
        f"{diversified_peak_MW:.2f} MW."
    )

    # --- Baseload sizing ---
    if baseload_types:
        baseload_target_MW = diversified_peak_MW * baseload_fraction
        n_baseload = len(baseload_types)
        # Split evenly if multiple baseload types; in practice usually 1
        per_baseload_MW = baseload_target_MW / n_baseload

        for btype in baseload_types:
            capacity_MW = per_baseload_MW

            if btype == "ashp":
                # ASHP needs oversizing to account for cold-weather derating
                nameplate_MW = capacity_MW / ASHP_DESIGN_DAY_DERATING
                capacity_MW = nameplate_MW
                # Round to sensible unit sizes
                unit_size_MW = _sensible_ashp_unit_size(capacity_MW)
                n_units = max(1, round(capacity_MW / unit_size_MW))
                capacity_MW = n_units * unit_size_MW
                hours = _load_duration_baseload_hours(demand_kW, capacity_MW * ASHP_DESIGN_DAY_DERATING * 1000)
                notes.append(
                    f"ASHP: {capacity_MW:.2f} MW nameplate ({n_units} × {unit_size_MW:.1f} MW) "
                    f"→ {capacity_MW * ASHP_DESIGN_DAY_DERATING:.2f} MW at design day. "
                    f"Runs ≥{hours} hours/year at full output."
                )
                sources.append({
                    "type": "ashp",
                    "capacity_MW": round(capacity_MW, 2),
                    "n_units": n_units,
                    "role": "baseload",
                    "flow_temp_C": network_flow_temp_C,
                    "rationale": f"Baseload ASHP — {baseload_fraction*100:.0f}% of diversified peak, "
                                 f"derated for cold weather",
                })

            elif btype == "efw_chp":
                # EfW runs flat baseload; typically sized at 30-50% of peak
                efw_fraction = min(baseload_fraction, 0.50)
                capacity_MW = diversified_peak_MW * efw_fraction
                capacity_MW = _round_capacity(capacity_MW, step=0.5)
                hours = _load_duration_baseload_hours(demand_kW, capacity_MW * 1000)
                notes.append(
                    f"EfW CHP: {capacity_MW:.1f} MW — flat baseload at "
                    f"{efw_fraction*100:.0f}% of diversified peak. "
                    f"Runs ≥{hours} hours/year at rated output."
                )
                sources.append({
                    "type": "efw_chp",
                    "capacity_MW": round(capacity_MW, 2),
                    "role": "baseload",
                    "rationale": f"EfW baseload — {efw_fraction*100:.0f}% of diversified peak",
                })

            elif btype == "data_centre":
                # DC waste heat is supply-constrained, not demand-sized
                # Default to a sensible fraction; user can override
                dc_fraction = min(baseload_fraction, 0.40)
                capacity_MW = diversified_peak_MW * dc_fraction
                capacity_MW = _round_capacity(capacity_MW, step=0.5)
                notes.append(
                    f"Data-centre waste heat: {capacity_MW:.1f} MW — "
                    f"subject to actual DC availability. A booster heat pump "
                    f"is needed to lift ~30°C waste heat to {network_flow_temp_C:.0f}°C."
                )
                sources.append({
                    "type": "data_centre",
                    "capacity_MW": round(capacity_MW, 2),
                    "role": "baseload",
                    "dispatch_direct": False,
                    "rationale": f"DC waste heat — {dc_fraction*100:.0f}% of diversified peak",
                })
                # Auto-add a booster
                booster_MW = capacity_MW * 0.85  # booster COP ~4 → MW out ≈ MW in × (COP/(COP-1))
                booster_MW = _round_capacity(booster_MW, step=0.5)
                sources.append({
                    "type": "booster_heat_pump",
                    "capacity_MW": round(booster_MW, 2),
                    "depends_on": len(sources) - 1,
                    "role": "baseload",
                    "rationale": "Booster HP lifts DC waste heat to network temperature",
                })
                notes.append(f"Booster heat pump: {booster_MW:.1f} MW paired with DC source.")

            else:
                capacity_MW = _round_capacity(capacity_MW, step=0.5)
                sources.append({
                    "type": btype,
                    "capacity_MW": round(capacity_MW, 2),
                    "role": "baseload",
                    "rationale": f"Baseload — {baseload_fraction*100:.0f}% of diversified peak",
                })

    # --- Peak/backup sizing ---
    # Peak plant covers everything baseload can't reach at the design peak
    total_baseload_MW = sum(s["capacity_MW"] for s in sources if s["role"] == "baseload")
    # For ASHP, the peak-day real output is derated
    ashp_sources = [s for s in sources if s["type"] == "ashp"]
    ashp_derated_MW = sum(s["capacity_MW"] * ASHP_DESIGN_DAY_DERATING for s in ashp_sources)
    largest_ashp_unit_derated_MW = 0.0
    if resilience_mode == "largest_ashp_unit_out" and ashp_sources:
        largest_ashp_unit_derated_MW = max(
            (s["capacity_MW"] / max(1, s.get("n_units", 1))) * ASHP_DESIGN_DAY_DERATING
            for s in ashp_sources
        )
    non_ashp_baseload = sum(s["capacity_MW"] for s in sources
                           if s["role"] == "baseload" and s["type"] != "ashp"
                           and s["type"] != "booster_heat_pump")
    effective_baseload_MW = max(0.0, ashp_derated_MW - largest_ashp_unit_derated_MW) + non_ashp_baseload
    if largest_ashp_unit_derated_MW:
        notes.append(
            f"Resilience allowance: peak/backup sizing assumes the largest ASHP unit "
            f"({largest_ashp_unit_derated_MW:.2f} MW design-day output) is unavailable."
        )

    peak_shortfall_MW = max(0, diversified_peak_MW - effective_baseload_MW)
    # Add 10% margin for peak plant (real reserve / N+1 style)
    peak_with_margin_MW = peak_shortfall_MW * 1.10

    if not peak_types and peak_shortfall_MW > 0.1:
        # User didn't select a peak type — add gas boiler as default
        peak_types = ["gas_boiler"]
        notes.append("No peak/backup technology selected — gas boiler added automatically.")

    if peak_types and peak_with_margin_MW > 0.05:
        per_peak_MW = peak_with_margin_MW / len(peak_types)
        for ptype in peak_types:
            cap = _round_capacity(per_peak_MW, step=0.5)
            notes.append(
                f"Peak {ptype.replace('_',' ')}: {cap:.1f} MW — covers the "
                f"{(1-baseload_fraction)*100:.0f}% of diversified peak above "
                f"baseload + 10% reserve margin."
            )
            sources.append({
                "type": ptype,
                "capacity_MW": round(cap, 2),
                "role": "peak",
                "rationale": f"Peak/backup — covers shortfall above baseload output at design day",
            })

    total_installed_MW = sum(s["capacity_MW"] for s in sources)
    notes.append(
        f"Total installed heating capacity: {total_installed_MW:.2f} MW "
        f"(ratio to diversified peak: {total_installed_MW/diversified_peak_MW:.2f}x)."
    )

    # --- Cooling sizing ---
    cooling_sources = []
    if include_cooling and peak_cooling_kW > 0:
        cool_peak_MW = peak_cooling_kW * diversity_applied * (1.0 + network_loss_margin) / 1000.0
        # Round up to sensible chiller bank
        unit_size = 0.5 if cool_peak_MW < 3 else 1.0 if cool_peak_MW < 8 else 2.0
        n_units = max(1, int(np.ceil(cool_peak_MW / unit_size)))
        cool_cap = n_units * unit_size
        notes.append(
            f"Cooling: {cool_cap:.1f} MW ({n_units} × {unit_size:.1f} MW chillers) "
            f"for {cool_peak_MW:.2f} MW diversified cooling peak."
        )
        cooling_sources.append({
            "type": "air_cooled_chiller",
            "capacity_MW": round(cool_cap, 2),
            "n_units": n_units,
            "role": "baseload",
            "rationale": f"Central chiller bank for {cool_peak_MW:.2f} MW diversified cooling peak",
        })

    return {
        "diversity_factor": diversity_applied,
        "diversified_peak_kW": round(diversified_peak_kW, 1),
        "diversified_peak_MW": round(diversified_peak_MW, 3),
        "sources": sources,
        "cooling_sources": cooling_sources,
        "sizing_notes": notes,
    }


def _auto_diversity(building_types, n_buildings):
    """Pick a diversity factor from the building type mix."""
    if n_buildings <= 1:
        return DIVERSITY_FACTORS["single_building"]
    if not building_types:
        return DEFAULT_DIVERSITY_FACTOR
    types = set(building_types)
    residential_types = {"residential", "residential_existing"}
    commercial_types = {"office", "office_ac", "retail", "supermarket"}
    is_all_resi = types.issubset(residential_types)
    is_all_comm = types.issubset(commercial_types)
    if is_all_resi:
        return DIVERSITY_FACTORS["residential_only"]
    if is_all_comm:
        return DIVERSITY_FACTORS["commercial_only"]
    return DIVERSITY_FACTORS["mixed_use"]


def _sensible_ashp_unit_size(total_MW):
    """Pick a unit size that gives a reasonable number of units."""
    if total_MW <= 0.5:
        return 0.1    # small commercial
    if total_MW <= 2.0:
        return 0.5
    if total_MW <= 5.0:
        return 0.7
    if total_MW <= 10.0:
        return 1.0
    return 2.0


def _round_capacity(MW, step=0.5):
    """Round up to the nearest step (avoids oddly precise values in UI)."""
    return max(step, np.ceil(MW / step) * step)
