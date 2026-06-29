"""
OPEX.py
==============
Whole-scheme annual OPEX — REAL fuel/electricity cost (already computed
correctly by dispatch.py's DispatchResult.summary(), from actual hourly
dispatch, not an assumed average) PLUS operations & maintenance (O&M)
cost, which doesn't exist ANYWHERE in this project until this module —
nothing currently charges for servicing a boiler, an ASHP compressor, a
chiller, or the network's own pipework.

Why this file is thin
------------------------
The genuinely hard part of OPEX — what does each source actually cost
to RUN, hour by hour, given real tariffs and real part-load efficiency
— is already done correctly by dispatch.py. This module does NOT
recompute that; it reads dispatch_result.summary()["total_annual_opex_GBP"]
directly. This file's only real job is the one piece that was missing:
O&M cost, which is a function of CAPEX, not of dispatch.

O&M cost methodology
-----------------------
annual_om_GBP = capex_GBP x om_rate

Real sourcing: the Community Heat Development Unit (CHDU) — the SAME
body already cited elsewhere in this project for network/pipework cost
benchmarks (see network.py, pipe_catalog.py) — states in their public
financial-modelling methodology: "The CHDU heat network financial model
assumes that the annual O&M costs of the Energy Centre are 1% of its
CAPEX, as suggested by the DECC report on cost characteristics of UK
heat networks." (communityheat.org.uk/techno-economic-model/financial-
modelling/). DEFAULT_OM_RATE = 0.01 uses this real, cited figure
directly — not an invented round number.

This is deliberately a SINGLE flat rate applied to whole-scheme CAPEX,
not a per-technology breakdown (a real refinement would give ASHPs/
chillers a different O&M rate than pipework, since compressor
maintenance and buried-pipe maintenance are genuinely different cost
regimes) — the CHDU figure itself is quoted as a single Energy Centre-
wide rate, so using it the same way here matches its actual sourced
scope rather than inventing a more granular split with no real backing.

Usage
-----
    from economics.OPEX import annual_om_cost_GBP, total_annual_opex_GBP

    om_cost = annual_om_cost_GBP(capex_GBP=11_721_302)
    total_opex = total_annual_opex_GBP(dispatch_result, capex_GBP=11_721_302)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Real, cited CHDU/DECC figure — see module docstring for the full sourcing note.
DEFAULT_OM_RATE = 0.01


def annual_om_cost_GBP(capex_GBP: float, om_rate: float = DEFAULT_OM_RATE) -> float:
    """
    Annual operations & maintenance cost (£/year), as a fraction of
    whole-scheme CAPEX — see module docstring for the real CHDU/DECC
    sourcing of the default 1% rate.

    Parameters
    ----------
    capex_GBP : whole-scheme CAPEX (£) — e.g. from
                economics.CAPEX.aggregate_capex()["grand_total_GBP"]
    om_rate    : annual O&M cost as a fraction of CAPEX (default 0.01,
                the real cited CHDU/DECC figure)

    Returns
    -------
    Annual O&M cost (£/year).
    """
    return capex_GBP * om_rate


def total_annual_opex_GBP(
    dispatch_result,
    capex_GBP: float,
    om_rate: float = DEFAULT_OM_RATE,
) -> dict:
    """
    Combine REAL dispatch fuel/electricity OPEX (already correctly
    computed by dispatch.py from actual hourly dispatch) with O&M cost
    (a function of CAPEX, computed here) into one whole-scheme annual
    OPEX figure.

    Parameters
    ----------
    dispatch_result : a DispatchResult instance (from run_dispatch()) —
                this function calls .summary() on it directly, so the
                fuel/electricity OPEX is exactly what actually happened
                in that dispatch run, not a separately-assumed average.
    capex_GBP        : whole-scheme CAPEX (£) — e.g. from
                economics.CAPEX.aggregate_capex()["grand_total_GBP"]
    om_rate           : annual O&M rate as a fraction of CAPEX (default
                0.01, the real cited CHDU/DECC figure)

    Returns
    -------
    dict: {
        "fuel_electricity_GBP": real dispatch OPEX (£/year),
        "om_GBP": O&M cost (£/year),
        "total_GBP": sum of both (£/year),
    }
    """
    s = dispatch_result.summary()
    fuel_electricity_GBP = s["total_annual_opex_GBP"]
    om_GBP = annual_om_cost_GBP(capex_GBP, om_rate)
    return {
        "fuel_electricity_GBP": round(fuel_electricity_GBP, 0),
        "om_GBP": round(om_GBP, 0),
        "total_GBP": round(fuel_electricity_GBP + om_GBP, 0),
    }


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "\nThis file's self-test lives in tests/test_economics.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_economics.py\n"
    )
