"""
om_rates.py
============
Per-technology O&M rates, replacing the single flat 1% CHDU/DECC rate.

The flat CHDU 1% of total CAPEX is a reasonable first pass, but
compressor-based plant (ASHP, chillers) has genuinely higher annual
maintenance costs than buried pipework or a simple gas boiler.

Sources:
  - ASHP/chiller: BSRIA BG 44/2023 "Rules of Thumb" — 2-3% of installed
    cost for packaged chillers/heat pumps (midpoint 2.5%)
  - Gas boiler: CIBSE Guide M "Maintenance engineering and management"
    — 1-1.5% of installed cost (midpoint 1.25%)
  - Electric boiler: simpler plant than gas (no flue, no gas safety) —
    0.8-1.0% (midpoint 0.9%)
  - EfW CHP: complex plant with high availability requirements — WRAP/
    Defra modelling uses 3-4% of CAPEX (midpoint 3.5%)
  - Data-centre heat exchanger: passive equipment, minimal maintenance
    — 0.5%
  - Booster heat pump: same class as ASHP compressor plant — 2.5%
  - District pipework: CHDU/DECC's own original 1% figure covers
    network-level maintenance (valve servicing, leak detection) — kept
    at 1.0%
  - Chiller: same as ASHP (compressor plant) — 2.5%
"""

# Per-source-type annual O&M as a fraction of that source's own CAPEX
SOURCE_OM_RATES = {
    "ashp":               0.025,
    "gas_boiler":         0.0125,
    "electric_boiler":    0.009,
    "efw_chp":            0.035,
    "data_centre":        0.005,
    "booster_heat_pump":  0.025,
    "air_cooled_chiller": 0.025,
}

# Network (pipework) O&M — separate from source O&M
NETWORK_OM_RATE = 0.01   # CHDU/DECC original figure, applied to network CAPEX only

# Fallback for any source type not in the table above
DEFAULT_SOURCE_OM_RATE = 0.01


def source_annual_om_GBP(source) -> float:
    """Annual O&M for one source object, using the per-technology rate."""
    rate = SOURCE_OM_RATES.get(getattr(source, "source_type", None), DEFAULT_SOURCE_OM_RATE)
    return source.capacity_MW * source.capex_GBP_per_MW * rate


def total_annual_om_GBP(sources, network_capex_GBP=0.0) -> dict:
    """
    Whole-scheme annual O&M broken down by source + network.

    Returns
    -------
    dict: {
        "by_source": {source.name: £/yr},
        "network_GBP": £/yr,
        "total_GBP": £/yr,
    }
    """
    by_source = {}
    for s in sources:
        by_source[s.name] = round(source_annual_om_GBP(s), 0)
    network_om = network_capex_GBP * NETWORK_OM_RATE
    return {
        "by_source": by_source,
        "network_GBP": round(network_om, 0),
        "total_GBP": round(sum(by_source.values()) + network_om, 0),
    }
