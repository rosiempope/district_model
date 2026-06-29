"""
metrics.py
==============
The actual economic EVALUATION layer: given a real CAPEX total (from
economics.CAPEX) and real OPEX (from economics.OPEX, built on a real
dispatch_result), answer the questions a feasibility report's executive
summary actually leads with — simple payback, discounted payback, NPV,
and Levelised Cost of Heat (LCOH).

This also builds the INDIVIDUAL-SYSTEM counterfactuals (gas boiler,
individual ASHP, individual AC) that the project's core question needs:
is a shared district network actually better than every building going
it alone? NPV needs a real "avoided cost" cash flow — what the customer
would have paid WITHOUT the network — and that requires a real
counterfactual baseline, not a guessed proxy price.

Counterfactual design — deliberately minimal, not a second feasibility study
--------------------------------------------------------------------------------
Each counterfactual is ONE source per building, sized exactly to that
building's own peak, with NO network (no pipework, no shared energy
centre — that's the entire point: going individual avoids ALL network
CAPEX but pays domestic-scale £/kW for the generating equipment, which
is genuinely more expensive per kW than centralised plant — see
economics/CAPEX.py's INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW for the real
sourcing). No storage, no backup redundancy (a single domestic boiler
doesn't have a second boiler behind it), no carbon compliance check
(the London Heat Network Manual's threshold is a NETWORK-level
regulatory figure with no individual-building equivalent). This is
intentionally a fair, simple comparison, not a fully-detailed parallel
study — see this project's design discussion for why that's the right
scope given realistic time constraints.

Real per-building dispatch IS used (not a flat assumed efficiency) —
each building's own real hourly demand profile is dispatched against
its own single source, reusing run_dispatch() (trivial with one
source: everything goes to that source, capped at its capacity), so
genuine part-load gas boiler efficiency and genuine weather-driven ASHP/
chiller COP are correctly reflected, not approximated away.

LCOH and NPV methodology
---------------------------
LCOH (Levelised Cost of Heat) — matches the UK government's own cited
definition exactly (gov.uk heat networks delivery unit guidance):
"the undiscounted whole lifetime cost (CAPEX + electricity running
costs + maintenance costs + replacement costs over the lifetime)
divided by total energy demand over the lifetime (£/kWh)". UNDISCOUNTED
by design — LCOH compares technologies on lifetime cost-per-unit-heat,
not on money-today investment value (that's what NPV is for).

NPV (Net Present Value) — standard discounted cash flow:
    NPV = -CAPEX + sum_{t=1}^{n} CashFlow_t / (1+r)^t
where CashFlow_t is the AVOIDED COST that year (counterfactual OPEX
minus the network's own actual OPEX) and r is the discount rate. Real
sourcing for the default discount rate: BEIS's 2017 heat networks
investment conference cites a 9-12% cost of capital range for UK heat
network investors — DEFAULT_DISCOUNT_RATE uses the midpoint (10.5%),
but r should be treated as a real SWEPT parameter, not a fixed point
estimate, since NPV can genuinely flip sign across that cited range for
a long-lived, CAPEX-heavy asset like a heat network.

Discounted payback — the same discounted cash flow series NPV uses,
but reporting WHEN the cumulative total turns positive, rather than its
final value. Genuinely different from simple payback (CAPEX / flat
annual saving), which implicitly treats every year's saving as worth
the same as today's — a real distortion for a 25+ year asset.

Project lifetime — CHDU/DECC's own cited assumption: 25 years for
energy centre component replacement, with a 20-30 year practical range,
inside an overall ~50-year heat network project appraisal horizon (same
source already cited in economics/OPEX.py). DEFAULT_PROJECT_LIFETIME_YEARS
uses 25 — the energy-centre-component figure, since that's the more
conservative (shorter) of the two, and the more directly comparable to
an individual system's own real lifetime (domestic boilers/ASHPs/AC
units don't last 50 years either).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from economics.CAPEX import INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW, individual_system_capex_GBP
from economics.OPEX import annual_om_cost_GBP, DEFAULT_OM_RATE
from optimisation.dispatch import run_dispatch
from components.peak_demand_option import GasBoiler
from components.ASHP import ASHPArray
from components.chiller import AirCooledChiller


# ── Real, cited defaults ────────────────────────────────────────────────────────

# BEIS 2017 heat networks investment conference cost-of-capital range
# for UK heat network investors: 9-12%. Midpoint used as the default
# single-value parameter, but r should be SWEPT across this real range
# for any genuine investment-decision use, not trusted as a point
# estimate — NPV for a CAPEX-heavy, long-lived asset like a heat
# network can genuinely flip sign across this range.
DEFAULT_DISCOUNT_RATE = 0.105
DISCOUNT_RATE_RANGE = (0.09, 0.12)

# CHDU/DECC's own cited heat network component replacement assumption
# (25 years, 20-30 year practical range) — see module docstring.
DEFAULT_PROJECT_LIFETIME_YEARS = 25


# ── Individual-system counterfactuals ───────────────────────────────────────────

def counterfactual_gas_boiler_dispatch(node: dict) -> dict:
    """
    ONE gas boiler, sized exactly to this building's own peak heating
    demand, dispatched against this building's own real hourly heat
    profile. No network, no backup redundancy — see module docstring
    for the full "deliberately minimal" rationale.

    Parameters
    ----------
    node : one building's node dict from demand_synthesis.py's
           synthesise_network() "nodes" list — must have "peak_heat_kW"
           and "total_heat_kW" (the real hourly array)

    Returns
    -------
    dict: {"capex_GBP", "dispatch_result", "annual_opex_GBP"}
    """
    peak_MW = node["peak_heat_kW"] / 1000.0
    boiler = GasBoiler(
        name=f"{node['name']} individual gas boiler",
        capacity_MW=peak_MW,
        capex_GBP_per_MW=INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW["gas_boiler"] * 1000.0,
    )
    result = run_dispatch(node["total_heat_kW"], [boiler], storage=None, duty="heat")
    capex_GBP = individual_system_capex_GBP(node["peak_heat_kW"], "gas_boiler")
    return {
        "capex_GBP": capex_GBP,
        "dispatch_result": result,
        "annual_opex_GBP": result.summary()["total_annual_opex_GBP"],
    }


def counterfactual_individual_ashp_dispatch(node: dict, weather_df) -> dict:
    """
    ONE air source heat pump, sized exactly to this building's own peak
    heating demand, dispatched against this building's own real hourly
    heat profile. No network, no backup redundancy. Uses the SAME real
    ASHP COP physics (ashp_cop(), Ruhnau et al. regression) as the
    centralised ASHPArray elsewhere in this project — only the SCALE
    and CAPEX figure differ, not the underlying physics, since a small
    domestic ASHP and a large centralised one share the same
    fundamental vapour-compression cycle.

    Parameters
    ----------
    node        : one building's node dict — must have "peak_heat_kW"
                  and "total_heat_kW"
    weather_df  : EPW weather DataFrame (ASHP output is weather-dependent)

    Returns
    -------
    dict: {"capex_GBP", "dispatch_result", "annual_opex_GBP"}
    """
    peak_MW = node["peak_heat_kW"] / 1000.0
    ashp = ASHPArray(
        name=f"{node['name']} individual ASHP",
        n_units=1,
        unit_capacity_MW=peak_MW,
        weather_df=weather_df,
        capex_GBP_per_MW=INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW["individual_ashp"] * 1000.0,
    )
    result = run_dispatch(node["total_heat_kW"], [ashp], storage=None, duty="heat")
    capex_GBP = individual_system_capex_GBP(node["peak_heat_kW"], "individual_ashp")
    return {
        "capex_GBP": capex_GBP,
        "dispatch_result": result,
        "annual_opex_GBP": result.summary()["total_annual_opex_GBP"],
    }


def counterfactual_individual_ac_dispatch(node: dict, weather_df) -> dict:
    """
    ONE air conditioning unit, sized exactly to this building's own
    peak COOLING demand, dispatched against this building's own real
    hourly cooling profile. No network, no backup redundancy. Uses the
    SAME real chiller COP physics (chiller_cop(), the real-data-anchored
    curve from components/chiller.py) as the centralised AirCooledChiller
    elsewhere in this project — domestic split-system AC and a large
    centralised chiller are both vapour-compression machines rejecting
    heat to ambient air, same fundamental physics at different scale.

    Parameters
    ----------
    node        : one building's node dict — must have "peak_cool_kW"
                  and "cooling_kW"
    weather_df  : EPW weather DataFrame (chiller output is weather-dependent)

    Returns
    -------
    dict: {"capex_GBP", "dispatch_result", "annual_opex_GBP"}
    """
    peak_MW = node["peak_cool_kW"] / 1000.0
    ac = AirCooledChiller(
        name=f"{node['name']} individual AC",
        n_units=1,
        unit_capacity_MW=peak_MW,
        weather_df=weather_df,
        capex_GBP_per_MW=INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW["individual_ac"] * 1000.0,
    )
    result = run_dispatch(node["cooling_kW"], [ac], storage=None, duty="cool")
    capex_GBP = individual_system_capex_GBP(node["peak_cool_kW"], "individual_ac")
    return {
        "capex_GBP": capex_GBP,
        "dispatch_result": result,
        "annual_opex_GBP": result.summary()["total_annual_opex_GBP"],
    }


def aggregate_counterfactual(
    nodes: list,
    counterfactual_fn,
    weather_df=None,
    om_rate: float = DEFAULT_OM_RATE,
) -> dict:
    """
    Run a counterfactual dispatch for EVERY building and sum into one
    whole-scheme total — the "everyone goes individual" baseline.
    Built as a sum over per-building results so a future per-building
    breakdown (rather than just the whole-scheme total) is a small
    extension, not a redesign — see this project's design discussion.

    Parameters
    ----------
    nodes              : the "nodes" list from demand_synthesis.py's
                  synthesise_network() output
    counterfactual_fn    : one of counterfactual_gas_boiler_dispatch,
                  counterfactual_individual_ashp_dispatch,
                  counterfactual_individual_ac_dispatch
    weather_df           : required for the ASHP/AC counterfactuals
                  (ignored by the gas boiler one, which doesn't need it
                  — passed via **kwargs internally, see below)
    om_rate              : O&M rate applied to each building's own
                  individual-system CAPEX (same real CHDU 1% default as
                  the centralised case — see economics/OPEX.py)

    Returns
    -------
    dict: {
        "total_capex_GBP", "total_annual_fuel_electricity_GBP",
        "total_annual_om_GBP", "total_annual_opex_GBP",
        "by_building": {building_name: {...per-building results...}}
    }
    """
    by_building = {}
    for node in nodes:
        if weather_df is not None:
            result = counterfactual_fn(node, weather_df)
        else:
            result = counterfactual_fn(node)
        om_GBP = annual_om_cost_GBP(result["capex_GBP"], om_rate)
        by_building[node["name"]] = {
            "capex_GBP": round(result["capex_GBP"], 0),
            "annual_fuel_electricity_GBP": round(result["annual_opex_GBP"], 0),
            "annual_om_GBP": round(om_GBP, 0),
            "annual_opex_GBP": round(result["annual_opex_GBP"] + om_GBP, 0),
        }

    total_capex = sum(b["capex_GBP"] for b in by_building.values())
    total_fuel_elec = sum(b["annual_fuel_electricity_GBP"] for b in by_building.values())
    total_om = sum(b["annual_om_GBP"] for b in by_building.values())

    return {
        "total_capex_GBP": round(total_capex, 0),
        "total_annual_fuel_electricity_GBP": round(total_fuel_elec, 0),
        "total_annual_om_GBP": round(total_om, 0),
        "total_annual_opex_GBP": round(total_fuel_elec + total_om, 0),
        "by_building": by_building,
    }


# ── Financial metrics ────────────────────────────────────────────────────────────

def simple_payback_years(capex_GBP: float, annual_avoided_cost_GBP: float) -> Optional[float]:
    """
    Payback = CAPEX / annual saving. No discounting, no lifetime cap —
    a crude but immediately intuitive number. Returns None (rather than
    raising or returning infinity) if annual_avoided_cost_GBP <= 0,
    since the project would never pay back at all in that case — a
    real, reportable outcome, not an error.
    """
    if annual_avoided_cost_GBP <= 0:
        return None
    return capex_GBP / annual_avoided_cost_GBP


def discounted_cash_flow_series(
    annual_avoided_cost_GBP: float,
    project_lifetime_years: int = DEFAULT_PROJECT_LIFETIME_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> np.ndarray:
    """
    The discounted annual cash flow for every year of the project —
    the shared building block both NPV and discounted payback use.

    Assumes a FLAT annual avoided cost every year (a real, deliberate
    simplification for a feasibility-stage model — see module docstring
    on why this is "reflective, not detailed": genuine year-by-year
    demand growth or major gas/electricity price shocks aren't modelled
    here; this answers "is the project worth it under today's prices
    held flat", not a full 25-year energy market forecast).

    Parameters
    ----------
    annual_avoided_cost_GBP : counterfactual OPEX minus the scheme's own
                  actual OPEX, held flat across the project lifetime
    project_lifetime_years   : default 25 (CHDU/DECC's own cited
                  energy-centre-component replacement assumption)
    discount_rate             : default 10.5% (midpoint of BEIS's cited
                  9-12% UK heat network cost-of-capital range)

    Returns
    -------
    np.ndarray, length project_lifetime_years, of discounted cash flow
    for years 1 through project_lifetime_years.
    """
    years = np.arange(1, project_lifetime_years + 1)
    return annual_avoided_cost_GBP / (1 + discount_rate) ** years


def npv(
    capex_GBP: float,
    annual_avoided_cost_GBP: float,
    project_lifetime_years: int = DEFAULT_PROJECT_LIFETIME_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """
    Net Present Value — see module docstring for the full formula and
    real discount-rate sourcing.

        NPV = -CAPEX + sum_{t=1}^{n} CashFlow_t / (1+r)^t

    A positive NPV means the project is worth MORE than it costs, in
    today's-money terms, at this discount rate. Treat discount_rate as
    a real parameter to SWEEP across the cited 9-12% range
    (DISCOUNT_RATE_RANGE), not a single trusted point estimate — NPV
    for a long-lived, CAPEX-heavy asset can genuinely flip sign across
    that range.
    """
    cash_flows = discounted_cash_flow_series(
        annual_avoided_cost_GBP, project_lifetime_years, discount_rate,
    )
    return -capex_GBP + float(cash_flows.sum())


def discounted_payback_years(
    capex_GBP: float,
    annual_avoided_cost_GBP: float,
    project_lifetime_years: int = DEFAULT_PROJECT_LIFETIME_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> Optional[float]:
    """
    Years until the CUMULATIVE DISCOUNTED cash flow first exceeds CAPEX
    — the same discounted cash flow series npv() uses, reporting WHEN
    it turns positive rather than its final value. Genuinely different
    from simple_payback_years(), which implicitly treats every year's
    saving as worth today's money — a real distortion for a 25+ year
    asset (see module docstring).

    Returns the FRACTIONAL year (linear interpolation within the year
    payback actually occurs), not just a whole-year count, for a more
    precise comparison against simple_payback_years().

    Returns None if payback never occurs within project_lifetime_years
    (a real, reportable outcome — the project's discounted return never
    catches up to its CAPEX within its own assumed life).
    """
    cash_flows = discounted_cash_flow_series(
        annual_avoided_cost_GBP, project_lifetime_years, discount_rate,
    )
    cumulative = np.cumsum(cash_flows)
    payback_year_indices = np.where(cumulative >= capex_GBP)[0]
    if len(payback_year_indices) == 0:
        return None

    payback_year_idx = payback_year_indices[0]   # 0-indexed -> year (payback_year_idx+1)
    if payback_year_idx == 0:
        prior_cumulative = 0.0
    else:
        prior_cumulative = cumulative[payback_year_idx - 1]

    # Linear interpolation within the payback year for a fractional result
    shortfall_at_start_of_year = capex_GBP - prior_cumulative
    this_year_cash_flow = cash_flows[payback_year_idx]
    fraction_into_year = shortfall_at_start_of_year / this_year_cash_flow if this_year_cash_flow > 0 else 0.0

    return payback_year_idx + fraction_into_year


def levelised_cost_of_heat_GBP_per_kWh(
    capex_GBP: float,
    annual_opex_GBP: float,
    annual_heat_delivered_kWh: float,
    project_lifetime_years: int = DEFAULT_PROJECT_LIFETIME_YEARS,
    replacement_costs_GBP: float = 0.0,
) -> float:
    """
    Levelised Cost of Heat (£/kWh) — matches the UK government's own
    cited definition exactly (see module docstring): the UNDISCOUNTED
    whole-lifetime cost divided by total lifetime energy delivered.

        LCOH = (CAPEX + sum(annual OPEX) + replacement costs) / sum(annual heat delivered)

    Deliberately UNDISCOUNTED — LCOH compares technologies on lifetime
    cost-per-unit-heat, not money-today investment value (that's NPV's
    job). Assumes FLAT annual OPEX and FLAT annual heat delivered across
    the project lifetime (same "reflective, not detailed" simplification
    as discounted_cash_flow_series() — see that function's docstring).

    Parameters
    ----------
    capex_GBP                  : whole-scheme (or counterfactual) CAPEX (£)
    annual_opex_GBP              : one year's real OPEX (£/year) — held
                  flat across the project lifetime
    annual_heat_delivered_kWh    : one year's real heat delivered (kWh/year)
                  — held flat across the project lifetime
    project_lifetime_years        : default 25 (CHDU/DECC's cited figure)
    replacement_costs_GBP         : optional lump-sum replacement cost
                  total over the project lifetime (£) — e.g. a
                  compressor replacement at year 15. Default 0.0: this
                  project does not currently model COMPONENT-LEVEL
                  replacement schedules (a real, flagged simplification
                  — a genuine refinement would need a real lifetime
                  assumption PER component type, not one number for
                  everything, which hasn't been researched yet). Pass
                  a real figure here if/when that's built.

    Returns
    -------
    LCOH (£/kWh).
    """
    if annual_heat_delivered_kWh <= 0:
        raise ValueError(
            "annual_heat_delivered_kWh must be positive — LCOH is undefined "
            "for zero heat delivered."
        )
    total_lifetime_cost = (
        capex_GBP
        + annual_opex_GBP * project_lifetime_years
        + replacement_costs_GBP
    )
    total_lifetime_heat_kWh = annual_heat_delivered_kWh * project_lifetime_years
    return total_lifetime_cost / total_lifetime_heat_kWh


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "\nThis file's self-test lives in tests/test_economics.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_economics.py\n"
    )
