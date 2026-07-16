"""
CAPEX.py
==============
Whole-scheme CAPEX aggregation — collects the installed costs every
source/network/storage component already knows about itself, into one
real total broken down by category. This file adds NO new cost
physics; every number it touches is already computed elsewhere
(source.capex_GBP_per_MW x source.capacity_MW, NetworkTopology's own
total_capex_GBP(), ThermalStorage's estimate_storage_capex()) — this
module's only job is to SUM those into a whole-scheme figure, since
nothing in the project currently does that.

Why this was empty until now
------------------------------
Every component has always reported its OWN installed cost (see e.g.
ASHPArray.summary()'s "estimated_capex_GBP", NetworkTopology's
total_capex_GBP(sized_segments)) — but nothing combined them. A real
feasibility report needs ONE number (with a real breakdown), not six
scattered ones a reader has to manually add up.

Individual-system counterfactual CAPEX
-----------------------------------------
This module also holds INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW — real UK
domestic-scale £/kW figures (genuinely different from this project's
utility-scale plant figures, e.g. ASHPArray's £770,000/MW = £770/kW
vs. an individual domestic ASHP's much higher £1,150/kW — individual-
scale equipment costs MORE per kW than centralised plant, a real
result this project's network-vs-individual comparison is built to
surface, not assume). See each constant's own docstring note for the
real source figures these were derived from.

Usage
-----
    from economics.CAPEX import aggregate_capex

    result = aggregate_capex(
        sources=[dc, efw, ashp, gas, elec],
        network_topology=ealing_topo,
        sized_segments=sized_heat,
        storage=thermal_store,
    )
    print(result["grand_total_GBP"])
    print(result["by_category"])
"""

import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Individual ("decentralised") system CAPEX, £/kW ────────────────────────────
#
# Real UK domestic/small-commercial-scale installed costs, deliberately
# kept SEPARATE from this project's utility-scale plant figures (e.g.
# ASHPArray's capex_GBP_per_MW=770,000 = £770/kW) — these are genuinely
# different cost regimes, not the same number at a different scale.
# Sourced from multiple 2025/2026 UK installer/market-review sources
# (Heatable, MyJobQuote, Less, MoneyWeek, EnergyPages, BestBuilders —
# cross-checked for consistency, not a single source):
#
#   GAS_BOILER: ~£2,500-£3,500 installed for a typical 24-30kW domestic
#   combi/system boiler -> midpoint ~£3,000 / ~27kW = ~£111/kW
#
#   INDIVIDUAL_ASHP: ~£8,000-£15,000 installed BEFORE the UK Boiler
#   Upgrade Scheme (BUS) grant, for a typical ~10kW domestic air source
#   heat pump -> midpoint ~£11,500 / 10kW = ~£1,150/kW. Deliberately
#   BEFORE grant -- the BUS £7,500 grant is a POLICY SUBSIDY, not a true
#   underlying cost; this module reports the real cost, not what a
#   subsidised customer happens to pay. If a "what does the customer
#   actually pay after grant" view is ever needed, subtract the grant
#   separately rather than baking it into this constant.
#
#   INDIVIDUAL_AC: ~£1,500-£2,500 installed for a typical 2.5kW domestic
#   split-system air conditioner -> midpoint ~£2,000 / 2.5kW = ~£800/kW
#
# All three are SUPPLY-ONLY domestic-scale figures (one boiler/heat pump/
# AC unit per building, no network of any kind) -- the whole point of
# this project's network-vs-individual comparison is that going
# individual avoids ALL network CAPEX (pipework, energy centre) but
# pays MORE per kW for the generating equipment itself, since utility-
# scale plant benefits from real economies of scale individual
# equipment doesn't get.
INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW = {
    "gas_boiler":      111.0,
    "individual_ashp": 1150.0,
    "individual_ac":   800.0,
}


def aggregate_capex(
    sources: Optional[list] = None,
    network_topology=None,
    sized_segments: Optional[dict] = None,
    storage=None,
) -> dict:
    """
    Sum every component's already-known installed cost into one
    whole-scheme CAPEX total, broken down by category.

    This is a pure aggregator — it never invents a cost; it only reads
    .capex_GBP_per_MW / .capacity_MW off each source (the same pattern
    every source class in this project already follows — see e.g.
    ASHPArray's own constructor/summary()), and calls
    network_topology.total_capex_GBP(sized_segments) /
    storage.estimate_storage_capex-style methods directly.

    Parameters
    ----------
    sources           : list of source objects (ASHPArray, EfWChp,
                  DataCentre, BoosterHeatPump, AirCooledChiller,
                  GasBoiler, ElectricBoiler, or anything else exposing
                  .capacity_MW and .capex_GBP_per_MW). Optional — pass
                  None or [] if you only want network/storage CAPEX.
    network_topology  : optional NetworkTopology instance
    sized_segments     : REQUIRED if network_topology is provided — the
                  dict from network_topology.size_all_segments(duty=...)
                  (one call's worth, for ONE duty — call this function
                  twice, once per duty, for a 4-pipe network's full
                  network CAPEX, and sum the two "network_GBP" figures
                  yourself if you want a combined heating+cooling
                  network total)
    storage           : optional ThermalStorage instance

    Returns
    -------
    dict: {
        "by_category": {"sources_GBP": ..., "network_GBP": ..., "storage_GBP": ...},
        "by_source": {source.name: capex_GBP, ...},
        "grand_total_GBP": sum of all categories,
    }
    """
    sources = sources or []

    by_source = {}
    for s in sources:
        if not hasattr(s, "capacity_MW") or not hasattr(s, "capex_GBP_per_MW"):
            raise ValueError(
                f"Source '{getattr(s, 'name', s)}' is missing .capacity_MW or "
                f".capex_GBP_per_MW — every source class in this project exposes "
                f"both (see e.g. ASHPArray's constructor); check this object's type."
            )
        by_source[s.name] = s.capacity_MW * s.capex_GBP_per_MW

    sources_total = sum(by_source.values())

    network_total = 0.0
    if network_topology is not None:
        if sized_segments is None:
            raise ValueError(
                "network_topology was provided but sized_segments was not — "
                "call network_topology.size_all_segments(flow_temp_C=..., "
                "return_temp_C=..., duty=...) first and pass the result here."
            )
        network_total = network_topology.total_capex_GBP(sized_segments)

    storage_total = 0.0
    if storage is not None:
        if not hasattr(storage, "capacity_MWh"):
            raise ValueError(
                "storage object is missing .capacity_MWh — check this is a "
                "real ThermalStorage instance."
            )
        storage_total = float(storage.capex_GBP)

    grand_total = sources_total + network_total + storage_total

    return {
        "by_category": {
            "sources_GBP": round(sources_total, 0),
            "network_GBP": round(network_total, 0),
            "storage_GBP": round(storage_total, 0),
        },
        "by_source": {k: round(v, 0) for k, v in by_source.items()},
        "grand_total_GBP": round(grand_total, 0),
    }


def individual_system_capex_GBP(peak_kW: float, system_type: str) -> float:
    """
    Real UK domestic-scale installed cost for ONE individual system
    (gas boiler, ASHP, or AC unit) sized to peak_kW — see
    INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW above for the real sourcing note.

    Parameters
    ----------
    peak_kW      : the building's own peak demand (kW) this individual
                  system would need to be sized to cover
    system_type   : "gas_boiler", "individual_ashp", or "individual_ac"

    Returns
    -------
    Installed cost (£) for that one building's individual system.
    """
    if system_type not in INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW:
        raise ValueError(
            f"Unknown system_type '{system_type}'. "
            f"Available: {list(INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW.keys())}"
        )
    return peak_kW * INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW[system_type]
