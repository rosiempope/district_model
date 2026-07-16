"""
om_rates.py
============
Per-technology O&M rates, replacing the single flat 1% CHDU/DECC rate.

The flat CHDU 1% of total CAPEX is a reasonable first pass, but
compressor-based plant (ASHP, chillers) has genuinely higher annual
maintenance costs than buried pipework or a simple gas boiler.

Sourcing note (real citations, corrected)
------------------------------------------
An earlier version of this module cited "BSRIA BG 44/2023 'Rules of
Thumb'" for the ASHP/chiller rate — that document does not exist. BSRIA
DID retire its old "Rules of Thumb" guide (BG9/2011) in February 2024,
replacing it with a four-part BG84-87/2024 series (BG84 Space & Weight,
BG85 Mechanical Criteria, BG86 Electrical Criteria, BG87 Useful
Information — bsria.com), of which BG85/2024 is the real, current
document covering heating/cooling plant. However, BSRIA guides are a
paid product and their exact maintenance-cost percentages are not
independently checkable from public sources, so this module does NOT
cite a specific figure from within it. Where a specific document's
number couldn't be independently confirmed, the rate below is instead
cross-referenced against multiple accessible published sources and
kept as an honest industry-benchmark range/midpoint, not attributed to
a single unverifiable citation:

  - ASHP/chiller (2.5%): no single public authoritative UK source
    pins this down precisely (BSRIA BG85/2024 likely does, but is
    paywalled). Cross-referenced against multiple accessible
    compressor-plant maintenance benchmarks (industrial HVAC O&M cost
    literature; chiller total-cost-of-ownership guides) that
    consistently land in a 2-3% of installed-cost/year range for
    packaged chillers/heat pumps — 2.5% is the midpoint of that range.
  - Gas boiler (1.25%): CIBSE Guide M10 "Costs" (2023, cibse.org) is
    the real, current UK reference for building-services maintenance
    costing, but its exact published percentage is behind CIBSE's
    Knowledge Portal paywall. Cross-referenced against accessible
    boiler lifecycle-cost literature (US EPA industrial boiler O&M
    studies cite 1-3% of capital cost/year; several boiler lifecycle
    assessments use ~1% as a simplifying assumption) — 1-1.5% (kept at
    1.25% midpoint) is a defensible, slightly-conservative figure
    within that accessible range for a well-maintained commercial
    condensing gas boiler specifically.
  - Electric boiler (0.9%): no independent %-of-CAPEX source located.
    Kept as an engineering-judgment estimate below the gas boiler rate
    — electric boilers have no flue, no combustion products, and no
    statutory gas-safety inspection burden, so a somewhat lower rate
    than gas is a reasonable simplification, not a researched figure.
  - EfW CHP (3.5%): the "WRAP/Defra modelling" citation in an earlier
    version of this module could not be located and has been removed.
    No single authoritative UK EfW-specific O&M-as-%-of-CAPEX source
    was found publicly. Kept at 3.5% as a reasonable industry-benchmark
    midpoint for complex CHP plant with high availability requirements,
    broadly consistent with general waste-to-energy techno-economic
    literature (which typically cites O&M in the low-to-mid single
    digits of CAPEX/year) — flagged here as an estimate, not a
    precisely sourced figure, pending a real project-specific quote.
  - Data-centre heat exchanger (0.5%): no specific source; reasoned
    estimate (passive equipment — heat exchangers, piping, controls —
    genuinely needs less maintenance than rotating/compressor plant).
  - Booster heat pump (2.5%): same class of plant and same sourcing
    basis as ASHP/chiller above (compressor-based, water-to-water
    rather than air-to-water, but the same real maintenance drivers).
  - District pipework (1.0%): CHDU/DECC's own figure, independently
    confirmed verbatim on communityheat.org.uk/techno-economic-model/
    financial-modelling/ — this is the one rate in this module with a
    directly confirmed live citation.

Counterfactual O&M — the one flat rate that survives
-----------------------------------------------------
The per-technology rates above apply to SCHEME plant. The individual-system
counterfactual (one domestic boiler/ASHP/AC per building) keeps the original
single CHDU/DECC rate instead: "The CHDU heat network financial model assumes
that the annual O&M costs of the Energy Centre are 1% of its CAPEX, as suggested
by the DECC report on cost characteristics of UK heat networks."
(communityheat.org.uk/techno-economic-model/financial-modelling/). Applying a
utility-scale per-technology split to a domestic boiler would be reading more
into that figure than it supports, so the counterfactual stays on the flat rate
it was actually sourced for. See INDIVIDUAL_SYSTEM_OM_RATE below.
"""

# Per-source-type annual O&M as a fraction of that source's own CAPEX
SOURCE_OM_RATES = {
    "ashp":               0.025,
    # Same class of compressor plant as ASHP, same sourcing basis (water/ground
    # to water rather than air to water, but the same real maintenance drivers).
    "wshp":               0.025,
    "gshp":               0.025,
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

# Flat CHDU/DECC rate, used for the INDIVIDUAL-SYSTEM counterfactual only —
# see the module docstring's "Counterfactual O&M" note. Previously lived in
# economics/OPEX.py, which had shrunk to this constant plus one function and
# whose docstring still claimed a per-technology split "doesn't exist" long
# after this module implemented exactly that.
INDIVIDUAL_SYSTEM_OM_RATE = 0.01


def annual_om_cost_GBP(capex_GBP: float, om_rate: float = INDIVIDUAL_SYSTEM_OM_RATE) -> float:
    """Annual O&M (£/yr) as a flat fraction of CAPEX.

    Used for the individual-system counterfactual, where the flat CHDU/DECC
    rate is the right basis. Scheme plant uses the per-technology rates via
    total_annual_om_GBP() instead.
    """
    return capex_GBP * om_rate


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
