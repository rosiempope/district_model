"""What it actually costs to connect a building, by what kind of building it is.

The problem this replaces
-------------------------
The model charged `customer_connection_GBP_per_connection` — a single flat
figure, £8,000 in the worked scenarios — to every connection, plus £500 for a
meter. With the design/commissioning/contingency adders on top that is
(8,000 + 500) x 1.33 = £11,305 per connection, applied identically to:

  - a 45 m2 flat taking an 8 kW HIU, and
  - Birmingham New Street Station taking a 15 MW substation.

Those are not the same purchase. A dwelling connection is a unit cost; a
non-domestic connection is a CAPACITY cost. Charging the station £11,305 and a
tower of 320 flats £3.6m is wrong in both directions at once, and on a
residential-heavy scheme the per-connection line is the second largest item in
the whole CAPEX — £4.79m of a £21m scheme on the worked A3 case.

The evidence, and how thin it is
---------------------------------
DECC, "Assessment of the Costs, Performance, and Characteristics of UK Heat
Networks" (Crown copyright 2015), the capital-costs table:

    Domestic HIU, per dwelling        £1,075   (range £738-£1,326,  n=3)
    Heat meter, per dwelling            £579   (range £491-£668,    n=2)
    Heat meter, per building (bulk)   £2,878   (n=1)
    Heat meter, per building (non-bulk) £3,343 (range £551-£6,136,  n=2)
    Substation, per kW capacity          £32   (range £15-£40,      n=2 bulk)
                                         £35   (range £16-£53,      n=3 non-bulk)
    Internal (secondary) pipework       £169/m (range £94-£244,     n=2)
    Buried main network                 £984/m (range £422-£1,472,  n=4)

  assets.publishing.service.gov.uk/media/5a802b44e5274a2e8ab4e95d/heat_networks.pdf

TWO THINGS ABOUT THIS TABLE MATTER MORE THAN THE NUMBERS IN IT.

1. "All capital costs are presented as 2013/14 prices." These are twelve years
   old. They are inflated below, and the uplift is the single largest source of
   error in this module.

2. The sample is SEVEN SCHEMES, and most individual figures rest on n=1 to n=4.
   The substation figure spans £15-£53/kW — a 3.5x range. The report's own
   conclusion is that "individual dwelling connections in the form of HIUs and
   heat meters appear to dominate the capital cost of domestic schemes", and
   that connection costs "range significantly from £25/MWh for bulk schemes to
   £624/MWh for non-bulk" — a 25x spread. This is the right shape to build an
   assumption around. It is not a price list.

Why the DECC figures alone do not reach £8,000
-----------------------------------------------
HIU + meter, inflated, is about £2,300 per dwelling. The £8,000 in the worked
scenarios is not that number and was never meant to be: a real dwelling
connection also buys the spur off the main, internal secondary pipework,
controls, installation labour and making good. DECC's own £169/m internal
pipework, inflated, is ~£237/m — so 15 m inside a flat is another ~£3,500 on its
own. Build those up and £8,000 for an existing dwelling is defensible.

So the flat £8,000 is not wrong because it is too big. It is wrong because it is
applied to a shopping centre.
"""
from __future__ import annotations

# ── Inflation ────────────────────────────────────────────────────────────────
#
# DECC's costs are 2013/14. This uplift carries them to 2026 and is the biggest
# single uncertainty here — bigger than the sample-size problem, because it is a
# multiplier on everything.
#
# Derivation, consistent with the basis pipe_catalog.py already uses (~24% CPI,
# 2020->2026): UK CPI ran roughly +13% from 2013/14 to 2020, then +24% to 2026,
# giving 1.13 x 1.24 = 1.40.
#
# HEALTH WARNING: this is CPI. Construction cost inflation (BCIS) ran materially
# hotter than CPI over the same period, particularly 2021-23. So 1.40 is
# CONSERVATIVE — it probably understates the real 2026 cost of these items. It
# is used because CPI is checkable and BCIS is behind a paywall; replace it with
# a real construction index before any of this reaches a quotation.
DECC_2013_14_TO_2026_UPLIFT = 1.40
DECC_PRICE_YEAR = "2013/14"
DECC_CITATION = (
    "DECC, Assessment of the Costs, Performance, and Characteristics of UK Heat "
    "Networks (2015), capital costs table. 7 schemes; most figures n=1 to n=4."
)

# ── DECC figures, as published (2013/14) ─────────────────────────────────────
DECC_HIU_PER_DWELLING_GBP = 1_075.0
DECC_HEAT_METER_PER_DWELLING_GBP = 579.0
DECC_HEAT_METER_PER_BUILDING_GBP = 2_878.0      # bulk, n=1
DECC_SUBSTATION_GBP_PER_KW = 32.0               # bulk, n=2 (range £15-£40)
DECC_INTERNAL_PIPEWORK_GBP_PER_M = 169.0        # n=2 (range £94-£244)


def _inflate(v: float) -> float:
    return v * DECC_2013_14_TO_2026_UPLIFT


# 2026 equivalents.
HIU_PER_DWELLING_GBP = _inflate(DECC_HIU_PER_DWELLING_GBP)                    # ~£1,505
HEAT_METER_PER_DWELLING_GBP = _inflate(DECC_HEAT_METER_PER_DWELLING_GBP)      # ~£811
HEAT_METER_PER_BUILDING_GBP = _inflate(DECC_HEAT_METER_PER_BUILDING_GBP)      # ~£4,029
SUBSTATION_GBP_PER_KW = _inflate(DECC_SUBSTATION_GBP_PER_KW)                  # ~£44.8
INTERNAL_PIPEWORK_GBP_PER_M = _inflate(DECC_INTERNAL_PIPEWORK_GBP_PER_M)      # ~£237

# Internal secondary pipework run inside a dwelling, from the riser to the HIU.
# NOT a DECC figure — an engineering assumption, and load-bearing: at ~£237/m it
# is the largest single component after the HIU itself.
#
# A flat taps a communal riser a few metres away. The RISER is shared across
# every dwelling above and below it, so charging each flat a full riser's length
# would count the same pipe forty times — these figures are the lateral run plus
# a per-dwelling share of the riser, not a whole vertical distribution system. A
# house, with no riser to share, needs materially more.
INTERNAL_PIPEWORK_M_PER_DWELLING = {
    "low": 4.0,      # new-build flat, riser on the landing
    "base": 8.0,     # typical existing flat: lateral run + share of the riser
    "high": 18.0,    # house, or an awkward existing retrofit
}

# What DECC's table does NOT price.
#
# This matters: DECC collected the ACTUAL capital costs of real schemes, so
# £1,075/HIU and £579/meter are already INSTALLED costs, not supply-only. Putting
# a full installation uplift on top of them would double-count the fitting labour
# twice over. So this fraction covers only the genuinely unpriced scope: the spur
# from the main to the building, controls, commissioning inside the property, and
# making good.
#
# That is where an existing building gets expensive — DESNZ's zoning consultation
# notes connecting existing buildings "can be considerably more expensive than
# new-build" — but it is a smaller multiplier than a naive read of the table
# suggests.
INSTALLATION_AND_MAKING_GOOD_FRACTION = {
    "low": 0.20,     # new build, clear access, done alongside other trades
    "base": 0.40,    # existing dwelling, occupied, normal difficulty
    "high": 0.80,    # hard retrofit: excavation, listed fabric, heavy reinstatement
}

SENSITIVITY_CASES = ("low", "base", "high")

RESIDENTIAL_TYPES = {"residential", "residential_existing"}


def dwelling_connection_GBP(case: str = "base") -> dict:
    """Cost of connecting ONE dwelling, built up from components.

    Returned as a breakdown rather than a single number, because the components
    behave differently — the HIU is a commodity, the pipework is a site problem,
    and the making-good is a negotiation.
    """
    if case not in SENSITIVITY_CASES:
        raise ValueError(f"case must be one of {SENSITIVITY_CASES}; got {case!r}")
    hiu = HIU_PER_DWELLING_GBP
    meter = HEAT_METER_PER_DWELLING_GBP
    pipework = INTERNAL_PIPEWORK_GBP_PER_M * INTERNAL_PIPEWORK_M_PER_DWELLING[case]
    hardware = hiu + meter + pipework
    install = hardware * INSTALLATION_AND_MAKING_GOOD_FRACTION[case]
    return {
        "case": case,
        "hiu_GBP": round(hiu, 0),
        "heat_meter_GBP": round(meter, 0),
        "internal_pipework_GBP": round(pipework, 0),
        "internal_pipework_m": INTERNAL_PIPEWORK_M_PER_DWELLING[case],
        "installation_and_making_good_GBP": round(install, 0),
        "total_GBP": round(hardware + install, 0),
    }


def non_domestic_connection_GBP(peak_kW: float, case: str = "base") -> dict:
    """Cost of connecting ONE non-domestic building — a CAPACITY cost.

    A shopping centre does not take an HIU. It takes a substation sized to its
    load, a bulk meter, and civils. Charging it a flat per-dwelling figure is the
    error this function exists to remove: at DECC's ~£45/kW, Birmingham New
    Street Station's 15 MW substation is ~£672,000, not £8,000.
    """
    if case not in SENSITIVITY_CASES:
        raise ValueError(f"case must be one of {SENSITIVITY_CASES}; got {case!r}")
    if peak_kW <= 0:
        raise ValueError(f"peak_kW must be positive; got {peak_kW}")
    # DECC's own range is £15-£53/kW across bulk and non-bulk. Rather than invent
    # a spread, the low/high cases use the real observed bounds, inflated.
    rate = {
        "low": _inflate(15.0),
        "base": SUBSTATION_GBP_PER_KW,
        "high": _inflate(53.0),
    }[case]
    substation = rate * float(peak_kW)
    meter = HEAT_METER_PER_BUILDING_GBP
    hardware = substation + meter
    install = hardware * INSTALLATION_AND_MAKING_GOOD_FRACTION[case]
    return {
        "case": case,
        "peak_kW": round(float(peak_kW), 1),
        "substation_GBP_per_kW": round(rate, 1),
        "substation_GBP": round(substation, 0),
        "bulk_meter_GBP": round(meter, 0),
        "installation_and_making_good_GBP": round(install, 0),
        "total_GBP": round(hardware + install, 0),
    }


def building_connection_GBP(building_type: str, peak_kW: float, connections: int,
                            case: str = "base") -> dict:
    """Connection cost for one building, priced by what it actually is.

    Residential is priced per dwelling; everything else on its peak capacity.
    """
    if building_type in RESIDENTIAL_TYPES and connections > 1:
        per = dwelling_connection_GBP(case)
        return {
            "basis": "per dwelling",
            "connections": connections,
            "per_connection_GBP": per["total_GBP"],
            "total_GBP": round(per["total_GBP"] * connections, 0),
            "breakdown": per,
        }
    nd = non_domestic_connection_GBP(peak_kW, case)
    return {
        "basis": "per kW of peak capacity",
        "connections": connections,
        "per_connection_GBP": nd["total_GBP"],
        "total_GBP": nd["total_GBP"],
        "breakdown": nd,
    }
