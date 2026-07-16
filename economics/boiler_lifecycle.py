"""What a gas boiler ACTUALLY costs its owner, beyond the gas bill.

Why this exists
---------------
The counterfactual bill this model charges customers against was fuel + the
Ofgem gas standing charge, and nothing else. A real household with a gas boiler
also pays to service it, repair it, and replace it every 12-15 years. A customer
who joins a heat network stops paying all of that.

Leaving it out understates what the customer's alternative genuinely costs, which
directly understates the revenue a heat network can fairly charge without leaving
anyone worse off. DECC said so explicitly, in the same study this project already
uses for connection costs:

    "Whilst many customers may only consider the fuel or energy cost
     (predominantly gas or electricity), the full counterfactual cost should
     include the purchase, replacement, and costs of operation of the heat
     source."

    "The analysis suggests that in low heat demand dwellings, the counterfactual
     prices can be relatively high once boiler maintenance and replacement is
     included, and current heat sales prices from heat networks may be
     under-[valuing heat]"

    DECC, Assessment of the Costs, Performance, and Characteristics of UK Heat
    Networks (2015), pp.33-35.
    assets.publishing.service.gov.uk/media/5a802b44e5274a2e8ab4e95d/heat_networks.pdf

DECC's own method: "a 30 year lifecycle, with a 15-year boiler replacement", a
£2,500 boiler capital cost (from the Green Deal Final Stage Impact Assessment),
and operation costs "typically £150 - £200 per year, including servicing and
repairs".

Their result is the part that matters for a heat network. Boiler costs add about
1.1 p/kWh for a large dwelling, but 4.6 p/kWh for a small efficient one — and:

    "Heat network schemes are generally deployed in areas of high density
     consisting typically of smaller high density houses or more usually flats.
     Therefore the counterfactual heat cost appropriate for heat network schemes
     is likely to be at the higher end of the range, typically above 7.2 p/kWh."

That is the crux. The dwellings a heat network actually serves are exactly the
ones where boiler lifecycle cost is LARGEST per kWh, because the fixed cost of
owning a boiler is spread over a small heat demand.

Today's figures, not DECC's inflated
-------------------------------------
DECC's numbers are 2013/14. Rather than inflate them, these are current UK market
figures, which is more defensible and happens to reconcile with DECC anyway
(their implied ~£342/yr at 2014 prices x ~1.40 CPI = ~£479, against ~£417 below —
same order, and the direct figures win).

    Annual service      £80-140, average ~£100
    Boiler lifespan     10-15 years
    Full replacement    ~£4,000 installed (range £1,500-£5,000)

    checkatrade.com/blog/cost-guides/boiler-service-cost/
    myjobquote.co.uk/costs/cost-to-get-your-boiler-serviced
    heatable.co.uk/boiler-advice/boiler-service-costs

What this is NOT
----------------
It is not a claim that every customer would replace their boiler on schedule, or
that they consciously price it. Many do not. It is a claim about the real
resource cost of the alternative, which is what a fair tariff comparison needs —
and it is what DECC, the only UK government study of actual heat-network costs,
says the comparison should contain.

It is applied per CONNECTION, so a block of 50 flats has 50 boilers to service
and replace, not one.
"""
from __future__ import annotations

# Annual service by a Gas Safe engineer. Manufacturers require it to keep a
# 5-12 year warranty valid, so this is not optional in practice.
BOILER_ANNUAL_SERVICE_GBP = 100.0

# Repairs between services. DECC's £150-200/yr covers "servicing and repairs"
# together; with a ~£100 service that leaves ~£50-100 of repairs. The lower end
# is used — conservative, i.e. it understates the alternative's cost.
BOILER_ANNUAL_REPAIRS_GBP = 50.0

# Installed replacement cost. £4,000 is the current UK average for a full
# replacement; the range is wide (£1,500-£5,000) and depends on whether flues,
# pipework or a system change are involved.
BOILER_REPLACEMENT_GBP = 4_000.0

# Replacement interval. Real lifespan is 10-15 years. 15 is used — DECC's own
# assumption, and the CONSERVATIVE end here: a longer life spreads the
# replacement cost thinner and so charges the alternative less.
BOILER_LIFE_YEARS = 15.0

DECC_CITATION = (
    "DECC, Assessment of the Costs, Performance, and Characteristics of UK Heat "
    "Networks (2015), pp.33-35: the full counterfactual cost 'should include the "
    "purchase, replacement, and costs of operation of the heat source'."
)


def annualised_replacement_GBP() -> float:
    """Straight-line annualised boiler replacement, per boiler per year."""
    return BOILER_REPLACEMENT_GBP / BOILER_LIFE_YEARS


def boiler_lifecycle_GBP_per_year(connections: int = 1) -> float:
    """Everything a boiler owner pays that is NOT on the gas bill.

    Per connection, because a block of 50 flats has 50 boilers.
    """
    per_boiler = (
        BOILER_ANNUAL_SERVICE_GBP
        + BOILER_ANNUAL_REPAIRS_GBP
        + annualised_replacement_GBP()
    )
    return per_boiler * max(0, int(connections))


def breakdown(connections: int = 1) -> dict:
    n = max(0, int(connections))
    return {
        "connections": n,
        "annual_service_GBP": round(BOILER_ANNUAL_SERVICE_GBP * n, 0),
        "annual_repairs_GBP": round(BOILER_ANNUAL_REPAIRS_GBP * n, 0),
        "annualised_replacement_GBP": round(annualised_replacement_GBP() * n, 0),
        "total_GBP_per_year": round(boiler_lifecycle_GBP_per_year(n), 0),
        "per_boiler_GBP_per_year": round(boiler_lifecycle_GBP_per_year(1), 0),
        "replacement_basis": (
            f"£{BOILER_REPLACEMENT_GBP:,.0f} every {BOILER_LIFE_YEARS:.0f} years"
        ),
        "citation": DECC_CITATION,
    }
