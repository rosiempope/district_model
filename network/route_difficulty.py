"""What kind of ground the pipe goes in — and what that does to the cost.

The problem
-----------
pipe_catalog.estimate_pipe_cost_GBP_per_m() returns ONE cost for a given
diameter. But a metre of DN500 under a six-lane junction in a city centre and a
metre of DN500 across a greenfield development site are not the same purchase,
and the difference is larger than the difference between two pipe sizes.

The evidence
------------
The DESNZ Birmingham zoning report (Feb 2025) gives four routes, in one city, in
one year, from one methodology — which is about as clean a calibration set as
UK public data offers. Against this model's own SEAI-fitted curve at each
route's sized diameter:

    Birmingham Central   £3,750/m   1.51x    rail, tram, canal, multi-lane crossings
    Tyseley Central      £3,000/m   1.44x    industrial estate
    North East           £2,500/m   1.09x    industrial, "land around the perimeters
                                              of buildings which could be used to
                                              route pipework"
    QE Hospital / Uni    £1,750/m   0.76x    campus, REUSES existing pipe

The report says why Central is dearest, in its own constraints section: the route
crosses Birmingham New Street Station, an underground rail line, a tram line,
three multi-lane highway junctions and two canals, and "it is anticipated that
there will be significant congestion of existing services beneath many of the
roadways and footpaths".

And why QE is cheapest: it "utilises the existing heat network pipes serving the
Queen Elizabeth Hospital Heritage heat network and the University of Birmingham's
low temperature hot water network branch".

Cross-check against DECC (2015), buried main network, n=4 schemes:
£984/m average, range £422-£1,472, at 2013/14 prices — x1.40 CPI gives
~£1,378/m average, ~£591-£2,061 range in 2026 terms. The bottom of that range is
the easy-ground case and is well below anything Birmingham offers, which is why
the greenfield multiplier below is NOT calibrated from Birmingham.

Why a multiplier and not a table of £/m
----------------------------------------
Diameter still matters — a DN600 costs more than a DN200 on the same street. So
route difficulty scales the SEAI curve rather than replacing it, which keeps both
effects and keeps the curve's own DN20-DN600 fit doing its job.

The SEAI curve's own basis is "2-pipe, inner-city, 2020 prices" — Irish
inner-city. Birmingham Central sits 1.51x above it, which is the honest reading:
SEAI's inner-city is not a major English city centre with rail and canal
crossings. So the curve is the SUBURBAN case here, not the urban one.

HEALTH WARNING
--------------
Four data points, one city, one methodology, and the ratios depend on the
diameter this model happens to size for each route — which depends in turn on a
load-factor assumption. Treat these as the right ORDER, calibrated to real UK
figures, not as quotations. GREENFIELD in particular is NOT calibrated from
Birmingham (which has no greenfield route) and rests on DECC's low range.
"""
from __future__ import annotations

ROUTE_DIFFICULTY = {
    # Major English city centre. Rail/tram/canal crossings, multi-lane highways,
    # heavy congestion of existing buried services. Calibrated to Birmingham
    # Central: £3,750/m against a £2,484/m curve = 1.51x.
    "urban_congested": 1.50,

    # Ordinary streets, some crossings, ordinary utility congestion. This is what
    # the SEAI curve itself represents ("2-pipe, inner-city" — Irish inner-city,
    # which is not a Birmingham city centre). Birmingham North East, an industrial
    # zone, lands at 1.09x.
    "suburban": 1.00,

    # Industrial or commercial estate with perimeter land available to route
    # through, avoiding the highway. Between Birmingham's North East (1.09x) and
    # Tyseley (1.44x); Tyseley is dearer than its ground suggests because it also
    # carries a 4.5 km strategic heat main.
    "industrial_estate": 1.20,

    # Greenfield or a new development site — open ground, no existing services to
    # dodge, and trenching can share other infrastructure works. NOT calibrated
    # from Birmingham, which has no greenfield route. Rests on the bottom of
    # DECC's £422-£1,472/m buried-main range (2013/14), which inflates to
    # ~£591/m against a suburban curve near £2,300/m at DN500.
    "greenfield": 0.60,

    # Reusing existing pipe, or a campus with private land and no highway works.
    # Calibrated to the QE Hospital / University of Birmingham IZO: £1,750/m
    # against a £2,299/m curve = 0.76x.
    "existing_pipe_reuse": 0.75,
}

DEFAULT_ROUTE_DIFFICULTY = "suburban"

_BASIS = {
    "urban_congested": (
        "Birmingham Central IZO, £3,750/m: crosses New Street Station, an underground "
        "rail line, a tram line, three multi-lane junctions and two canals; report notes "
        "'significant congestion of existing services beneath many of the roadways'"
    ),
    "suburban": (
        "The SEAI curve's own basis (2-pipe, inner-city, 2020 prices). Birmingham "
        "North East, an industrial zone, sits at 1.09x"
    ),
    "industrial_estate": (
        "Between Birmingham North East (1.09x) and Tyseley (1.44x); perimeter land "
        "available to route through rather than the highway"
    ),
    "greenfield": (
        "NOT from Birmingham — no greenfield route exists there. Bottom of DECC's "
        "£422-£1,472/m buried-main range (2013/14, n=4), inflated"
    ),
    "existing_pipe_reuse": (
        "QE Hospital / University of Birmingham IZO, £1,750/m: 'utilises the existing "
        "heat network pipes serving the QE Hospital Heritage heat network and the "
        "University of Birmingham's low temperature hot water network branch'"
    ),
}


def route_difficulty_factor(route_type: str = DEFAULT_ROUTE_DIFFICULTY) -> float:
    """Cost multiplier on the SEAI pipe curve, by what the ground is like."""
    if route_type not in ROUTE_DIFFICULTY:
        raise ValueError(
            f"route_type must be one of {sorted(ROUTE_DIFFICULTY)}; got {route_type!r}. "
            "This is not a detail — a city-centre route costs 2.5x a greenfield one "
            "for the same pipe."
        )
    return ROUTE_DIFFICULTY[route_type]


def route_difficulty_basis(route_type: str = DEFAULT_ROUTE_DIFFICULTY) -> str:
    """Citable one-liner for why that factor applies. For audit output."""
    if route_type not in ROUTE_DIFFICULTY:
        raise ValueError(f"route_type must be one of {sorted(ROUTE_DIFFICULTY)}; got {route_type!r}")
    return _BASIS[route_type]
