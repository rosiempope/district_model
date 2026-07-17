"""
metrics.py
==============
The INDIVIDUAL-SYSTEM counterfactuals: what would each building have paid
without the network? This is the baseline the whole model is built to test
against — is a shared district network actually better than every building
going it alone — and it is what the gas-bill-parity revenue mechanism charges
customers against, so it has to be a real modelled bill, not a guessed proxy
price.

Where the financial metrics live (NOT here)
--------------------------------------------
NPV, IRR, payback and levelised cost are computed in economics/cashflow.py,
from one explicit years 0..N table, and assembled in
scenarios/scenario_runner.py. They are NOT computed here.

This module used to carry a second, parallel implementation of all of them —
npv(), irr(), simple_payback_years(), discounted_payback_years(),
discounted_cash_flow_series(), levelised_cost_of_heat_GBP_per_kWh() and
annual_revenue_GBP(). Nothing called any of it. Worse, it disagreed with the
live implementation: a 25-year default lifetime against the engine's 40, and a
flat-annuity cash flow against the engine's real year-by-year table with REPEX,
phasing and connection weighting. Anyone reading this file would reasonably have
concluded that was how the model computed NPV. It wasn't. Removed rather than
maintained as a trap — see git history if the flat-annuity form is ever wanted
for a quick sanity check.

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
"""

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from economics.CAPEX import (
    INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW, bus_grant_GBP, individual_system_capex_GBP,
)
from economics.boiler_lifecycle import boiler_lifecycle_GBP_per_year
from economics.om_rates import annual_om_cost_GBP, INDIVIDUAL_SYSTEM_OM_RATE
from economics.tariffs import (
    OFGEM_ELECTRICITY_CAP_P_PER_KWH, OFGEM_ELECTRICITY_CAP_STANDING_CHARGE_P_PER_DAY,
    OFGEM_GAS_CAP_P_PER_KWH, OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY,
)
from optimisation.dispatch import run_dispatch
from components.peak_demand_option import GasBoiler
from components.ASHP import ASHPArray
from components.chiller import AirCooledChiller


# ── Individual-system counterfactuals ───────────────────────────────────────────

def counterfactual_gas_boiler_dispatch(node: dict, include_boiler_lifecycle: bool = True) -> dict:
    """
    ONE gas boiler, sized exactly to this building's own peak heating
    demand, dispatched against this building's own real hourly heat
    profile. No network, no backup redundancy — see module docstring
    for the full "deliberately minimal" rationale.

    PRICING BASIS — this is a CUSTOMER counterfactual, not the scheme's
    own fuel purchasing: a real household with its own gas boiler pays
    the REGULATED RETAIL rate (the Ofgem price cap), not the wholesale
    fuel price a large scheme might negotiate. Earlier versions of this
    function left gas_price_GBP_per_MWh at its default, which resolves
    to GAS_PRICE_SCENARIOS["desnz_central"] (a WHOLESALE projection,
    ~£24.6/MWh) — roughly a third of the real Ofgem retail cap
    (OFGEM_GAS_CAP_P_PER_KWH = 7.33p/kWh = £73.30/MWh). That mismatch
    made every district-heating option being compared against this
    counterfactual look financially worse than it really would for an
    actual customer, since the "what would I have paid otherwise"
    baseline was priced about 3x too cheaply. Fixed here by passing the
    real retail unit rate explicitly, and by adding the real Ofgem
    standing charge (a genuine, separate component of what a household
    actually pays, not captured by a per-MWh dispatch cost at all) on
    top of the dispatch-based fuel cost.

    Parameters
    ----------
    node : one building's node dict from demand_synthesis.py's
           synthesise_network() "nodes" list — must have "peak_heat_kW"
           and "total_heat_kW" (the real hourly array)

    Returns
    -------
    dict: {"capex_GBP", "dispatch_result", "annual_opex_GBP"} — where
    annual_opex_GBP now correctly includes BOTH the real retail-priced
    unit-rate fuel cost AND the real standing charge for one connection.
    """
    # SIZING FIX: node["peak_heat_kW"] is the peak of the SPACE-HEATING
    # array only (see demand_synthesis.synthesise_building()), but this
    # boiler is dispatched against node["total_heat_kW"] = heating + DHW.
    # Sizing to the heating-only peak left the boiler 7-22% undersized on
    # the worked buildings, silently producing unmet demand INSIDE the
    # counterfactual (which under-costs it, since unmet heat burns no
    # fuel). A real individual boiler is sized to the building's full
    # coincident peak including hot water — use the true peak of the
    # actual array being dispatched.
    true_peak_kW = float(np.asarray(node["total_heat_kW"]).max())
    peak_MW = true_peak_kW / 1000.0
    retail_gas_price_GBP_per_MWh = OFGEM_GAS_CAP_P_PER_KWH * 10.0   # p/kWh -> £/MWh
    boiler = GasBoiler(
        name=f"{node['name']} individual gas boiler",
        capacity_MW=peak_MW,
        capex_GBP_per_MW=INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW["gas_boiler"] * 1000.0,
        gas_price_GBP_per_MWh=retail_gas_price_GBP_per_MWh,
    )
    result = run_dispatch(node["total_heat_kW"], [boiler], storage=None, duty="heat")
    capex_GBP = individual_system_capex_GBP(true_peak_kW, "gas_boiler")
    connections = max(1, int(node.get("connections", 1)))
    standing_charge_GBP = (
        OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY * 365.0 / 100.0 * connections
    )
    fuel_opex_GBP = result.summary()["total_annual_opex_GBP"]
    # What the boiler costs its owner BEYOND the gas bill: servicing, repairs and
    # replacement every 15 years. DECC is explicit that the counterfactual "should
    # include the purchase, replacement, and costs of operation of the heat
    # source", and warns that omitting it means heat networks "may be
    # under-[valuing heat]". Leaving it out was understating what the customer's
    # own alternative genuinely costs them. See economics/boiler_lifecycle.py.
    lifecycle_GBP = (
        boiler_lifecycle_GBP_per_year(connections) if include_boiler_lifecycle else 0.0
    )
    bill = fuel_opex_GBP + standing_charge_GBP + lifecycle_GBP
    return {
        "capex_GBP": capex_GBP,
        "dispatch_result": result,
        "annual_opex_GBP": round(bill, 0),
        "annual_customer_bill_GBP": round(bill, 0),
        "annual_fuel_GBP": round(fuel_opex_GBP, 0),
        "annual_standing_charge_GBP": round(standing_charge_GBP, 0),
        "annual_boiler_lifecycle_GBP": round(lifecycle_GBP, 0),
        "connections": connections,
    }


# What a heat pump costs its owner beyond the electricity bill — the exact
# mirror of economics/boiler_lifecycle.py, and required by the same DECC
# principle: the counterfactual "should include the purchase, replacement, and
# costs of operation of the heat source". Without this term the heat-pump
# parity bill was fuel + standing charge only, so the customer's single
# largest avoided cost (the heat pump itself) — and therefore the BUS grant —
# never reached the tariff comparison at all.
#
#   Annual service   £150 — UK ASHP service quotes run £150-300/yr
#                    (checkatrade.com/blog/cost-guides/air-source-heat-pump-
#                    maintenance-cost/); the LOW end is used, which understates
#                    the alternative's cost, i.e. conservative for district heat.
#   Life             20 years — the LONG end of the usual 15-20 quoted for
#                    ASHPs, again the conservative direction (spreads the
#                    replacement cost thinner).
#   Replacement      the building's own modelled install cost, net of BUS where
#                    eligible — assuming BUS (or a successor) recurs at every
#                    replacement is generous to the individual-HP case, i.e.
#                    conservative for district heat.
HP_ANNUAL_SERVICE_GBP = 150.0
HP_LIFE_YEARS = 20.0


def counterfactual_individual_ashp_dispatch(
    node: dict, weather_df, apply_bus_grant: bool = True,
    electricity_price_p_per_kWh: float | None = None,
    include_hp_lifecycle: bool = True,
) -> dict:
    """
    ONE air source heat pump, sized exactly to this building's own peak
    heating demand, dispatched against this building's own real hourly
    heat profile. No network, no backup redundancy. Uses the SAME real
    ASHP COP physics (ashp_cop(), Ruhnau et al. regression) as the
    centralised ASHPArray elsewhere in this project — only the SCALE
    and CAPEX figure differ, not the underlying physics, since a small
    domestic ASHP and a large centralised one share the same
    fundamental vapour-compression cycle.

    PRICING BASIS — the same fix the gas counterfactual already carries. This
    function previously left electricity_price_GBP_per_MWh at its default, which
    resolves to ElectricityTariff()'s ~24 p/kWh LARGE-BUSINESS negotiated rate.
    A household running its own heat pump pays the Ofgem cap (26.11 p/kWh), plus
    a real standing charge that no per-MWh dispatch cost captures at all. Both
    are now passed explicitly. This was the identical bug found and fixed on the
    gas side, and is very likely why this counterfactual was never wired up.

    BUS GRANT — apply_bus_grant=True nets the £7,500 Boiler Upgrade Scheme grant
    off the customer's capital cost, where eligible. Eligibility is enforced, not
    assumed: BUS caps at 45 kWth per installation, so it transforms the sums for
    a house and does nothing at all for a shopping centre. See
    economics/CAPEX.py's bus_grant_GBP().

    This is a CUSTOMER-FACING view. BUS is a transfer, not a resource cost, so it
    belongs in the investor/customer comparison and NOT in the whole-system
    social case — exactly the treatment this project already gives GHNF.

    Parameters
    ----------
    node             : one building's node dict — needs "peak_heat_kW",
                       "total_heat_kW" and optionally "connections"
    weather_df       : EPW weather DataFrame (ASHP output is weather-dependent)
    apply_bus_grant  : net the BUS grant off capex where eligible
    electricity_price_p_per_kWh : override the retail rate — this is the hook
                       for a levy-rebalancing sensitivity

    Returns
    -------
    dict: {"capex_GBP", "dispatch_result", "annual_opex_GBP",
           "annual_customer_bill_GBP", "bus_grant_GBP", "bus_eligible", ...}
    """
    true_peak_kW = float(np.asarray(node["total_heat_kW"]).max())
    peak_MW = true_peak_kW / 1000.0
    retail_elec_p = (
        OFGEM_ELECTRICITY_CAP_P_PER_KWH if electricity_price_p_per_kWh is None
        else float(electricity_price_p_per_kWh)
    )
    ashp = ASHPArray(
        name=f"{node['name']} individual ASHP",
        n_units=1,
        unit_capacity_MW=peak_MW,
        weather_df=weather_df,
        capex_GBP_per_MW=INDIVIDUAL_SYSTEM_CAPEX_GBP_PER_KW["individual_ashp"] * 1000.0,
        electricity_price_GBP_per_MWh=retail_elec_p * 10.0,   # p/kWh -> £/MWh
    )
    result = run_dispatch(node["total_heat_kW"], [ashp], storage=None, duty="heat")
    gross_capex_GBP = individual_system_capex_GBP(true_peak_kW, "individual_ashp")

    connections = max(1, int(node.get("connections", 1)))
    # Per-building eligibility override: BUS excludes social housing and most
    # new-build homes regardless of capacity, so a building can be marked
    # bus_eligible: false in the scenario. Default True preserves the existing
    # capacity-cap-only behaviour (generous to the individual-HP case).
    building_eligible = bool(node.get("bus_eligible", True))
    grant = (
        bus_grant_GBP(true_peak_kW, connections)
        if (apply_bus_grant and building_eligible) else 0.0
    )
    grant = min(grant, gross_capex_GBP)   # a grant cannot exceed the thing it buys

    standing_charge_GBP = (
        OFGEM_ELECTRICITY_CAP_STANDING_CHARGE_P_PER_DAY * 365.0 / 100.0 * connections
    )
    fuel_opex_GBP = result.summary()["total_annual_opex_GBP"]
    # Service + annuitised replacement (net of BUS), mirroring the boiler
    # lifecycle on the gas side. See HP_ANNUAL_SERVICE_GBP note above.
    lifecycle_GBP = (
        (gross_capex_GBP - grant) / HP_LIFE_YEARS
        + HP_ANNUAL_SERVICE_GBP * connections
    ) if include_hp_lifecycle else 0.0
    bill = fuel_opex_GBP + standing_charge_GBP + lifecycle_GBP
    return {
        "capex_GBP": gross_capex_GBP - grant,
        "gross_capex_GBP": gross_capex_GBP,
        "bus_grant_GBP": round(grant, 0),
        "bus_eligible": grant > 0,
        "dispatch_result": result,
        "annual_opex_GBP": round(bill, 0),
        "annual_customer_bill_GBP": round(bill, 0),
        "annual_fuel_GBP": round(fuel_opex_GBP, 0),
        "annual_standing_charge_GBP": round(standing_charge_GBP, 0),
        "annual_hp_lifecycle_GBP": round(lifecycle_GBP, 0),
        "connections": connections,
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
    # Buildings with genuinely zero cooling demand (e.g. a data_centre
    # archetype whose cooling is modelled elsewhere) would otherwise get
    # a 0-capacity chiller here — dispatch then divides by capacity_MW=0
    # (RuntimeWarning) and reports a meaningless load fraction. Skip
    # them cleanly: no cooling demand -> no individual AC to buy or run.
    if node["peak_cool_kW"] <= 0:
        return {"capex_GBP": 0.0, "dispatch_result": None, "annual_opex_GBP": 0.0}

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
    om_rate: float = INDIVIDUAL_SYSTEM_OM_RATE,
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
                  individual-system CAPEX (the flat CHDU/DECC 1% figure —
                  see economics/om_rates.py's INDIVIDUAL_SYSTEM_OM_RATE)

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
            "annual_customer_bill_GBP": round(
                result.get("annual_customer_bill_GBP", result["annual_opex_GBP"]), 0
            ),
            "connections": int(result.get("connections", node.get("connections", 1))),
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
