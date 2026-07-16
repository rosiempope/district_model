"""
==========
Electricity and gas pricing for the district energy dispatch model.

Provides realistic hourly price SHAPES (diurnal/seasonal patterns) scaled
to a chosen annual average level, plus future-year escalation scenarios.
This feeds directly into ASHPArray.marginal_cost, GasBoiler.marginal_cost,
and ElectricBoiler.marginal_cost via the resolve_electricity_price() and
resolve_gas_price() helpers at the bottom of this module — those components
now default to a realistic tariff shape rather than a flat placeholder.

Why the shape matters for dispatch
------------------------------------
ASHP marginal cost = electricity_price / COP. If electricity price is flat,
dispatch has no incentive to prefer running the ASHP overnight vs during
the evening peak — but in reality, running electrically-driven plant
during cheap overnight hours and leaning on storage/baseload sources
during the expensive evening peak is a real, available economic lever.
A flat price hides this lever entirely.

Electricity price shape
-------------------------
Calibrated against publicly available Octopus Agile half-hourly data
(the UK's most transparent retail time-of-use tariff, and a reasonable
public proxy for wholesale-driven shape even though Dalkia's actual
commercial contract will differ in level):
  - Evening peak (16:00-19:00): consistently the most expensive window,
    driven by network/balancing costs layered on top of wholesale price,
    not just wholesale price itself.
  - Overnight (23:00-06:00): consistently cheapest, when demand is low
    and (increasingly) wind generation is high.
  - Winter vs summer: winter average prices run higher than summer,
    reflecting higher overall system demand.
Reference: Octopus Energy Agile pricing explainer and public Agile rate
history (octopus.energy/blog/agile-pricing-explained,
energy-stats.uk/octopus-agile-tariff-pricing).

Electricity price LEVEL
--------------------------
Large commercial/industrial users in the UK are typically quoted bespoke,
negotiated rates rather than a published tariff — but the publicly known
range for large business electricity gives a sensible central estimate
and bounds:
    Large business electricity: 21.0 - 27.0 p/kWh, ~146p/day standing charge
    (typical range for negotiated bespoke commercial contracts, 2026)
This module uses 24.0 p/kWh (range midpoint) as the central case, with an
explicit negotiated_discount_pct parameter — if Dalkia's relationship with
a specific supplier (e.g. EDF) secures a better-than-market rate, set that
discount explicitly rather than guessing a different central number. This
keeps the assumption visible and auditable rather than baked silently into
a single hard-coded price.

Gas price scenarios
----------------------
Gas is much flatter intraday than electricity (no equivalent evening spike)
so the LEVEL matters far more than the shape here. Two scenarios are
provided as a deliberate sensitivity pair:
    'desnz_central'    : 85 p/therm — DESNZ's central long-term projection
                          for 2026. Source: DESNZ "Fossil Fuel Price
                          Assumptions 2025" (published January 2026,
                          assets.publishing.service.gov.uk/media/
                          696939b3448fedc1eb424870/fossil-fuel-price-
                          assumptions-2025.pdf), Table 1, "Assumption B"
                          (central case), 2026, real 2024 prices. This
                          REPLACES a previous 72p/therm figure, which was
                          the equivalent central projection from the
                          PREVIOUS (2024) edition of this same DESNZ
                          publication, not the current one.
    'current_actual'    : 120 p/therm — a live, dated wholesale-market
                          reading, higher than the DESNZ central case.
                          Source: UK NBP day-ahead/Winter-2026 wholesale
                          gas price, ≈118-132 p/therm during early-to-mid
                          July 2026 (Catalyst Commercial "UK Energy Market
                          Report", 10 July 2026), driven by Middle East
                          supply-risk premium, reduced Norwegian flows,
                          and tight European storage. This REPLACES a
                          previous 101p/therm figure which, by the time
                          of this update, had fallen BELOW the live
                          market price it was meant to represent as a
                          "conservative" (high) case — a reminder that a
                          scenario anchored to "actual observed prices"
                          needs re-dating periodically, not treated as a
                          fixed constant.
Reference: DESNZ Fossil Fuel Price Assumptions 2025 (Jan 2026); live NBP
wholesale gas market reporting, July 2026.
1 therm = 29.3071 kWh (standard UK gas conversion factor).

Future escalation
--------------------
Electricity and gas prices are NOT held flat in real terms — both have
real-terms escalation assumptions for multi-year financial modelling
(25-40 year network lifetimes need this). Escalation rates here are
simple, configurable real-terms annual percentages — NOT a market
forecast, just a sensible default to test sensitivity against. Override
with your own assumptions if Dalkia's finance team has a house view.

Usage
-----
    from tariffs import ElectricityTariff, GasTariff

    # Electricity — central commercial case
    elec = ElectricityTariff(annual_avg_p_per_kWh=24.0)
    print(elec.price_GBP_per_MWh[:24])      # First day's hourly prices

    # Electricity — with a negotiated EDF-style discount
    elec_edf = ElectricityTariff(annual_avg_p_per_kWh=24.0, negotiated_discount_pct=10.0)

    # Gas — sensitivity pair
    gas_central = GasTariff.from_scenario("desnz_central")
    gas_conservative = GasTariff.from_scenario("current_actual")

    # Future year escalation
    elec_2035 = elec.escalate_to_year(2035)

Component integration — resolve_electricity_price() / resolve_gas_price()
---------------------------------------------------------------------------
ASHP.py, peak_demand_option.py (GasBoiler, ElectricBoiler) call these two
helpers to turn whatever price input they were given — None, a Tariff
object, a flat scalar override, or a raw 8760 array — into a clean 8760
£/MWh array. This is what makes tariff scenarios swappable later from a
scenario config / UI menu without touching the component classes:

    from economics.tariffs import resolve_electricity_price, GasTariff

    # All four of these work identically wherever a component accepts
    # an `electricity_price_GBP_per_MWh` parameter:
    resolve_electricity_price(None)                          # realistic default
    resolve_electricity_price(ElectricityTariff(negotiated_discount_pct=10))
    resolve_electricity_price(150.0)                         # flat override
    resolve_electricity_price(some_8760_array)                # explicit array
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760
THERM_TO_KWH = 29.3071   # Standard UK gas conversion factor

BASE_YEAR = 2026   # The year this module's central-case prices represent


# ── Electricity price level — large commercial range ──────────────────────────
# Source: publicly reported large business electricity tariff range, 2026
ELEC_PRICE_RANGE_P_PER_KWH = {
    "low":     21.0,
    "central": 24.0,   # Midpoint — used as default
    "high":    27.0,
}
ELEC_STANDING_CHARGE_P_PER_DAY = 146.0


# ── Gas price scenarios — sensitivity pair ─────────────────────────────────────
# Source: DESNZ long-term gas price projections vs observed actual prices
GAS_PRICE_SCENARIOS = {
    "desnz_central": {
        "description":   "DESNZ central long-term gas price projection",
        "price_p_per_therm": 85.0,
        "reference":     "DESNZ Fossil Fuel Price Assumptions 2025 (Jan 2026), "
                          "Table 1, Assumption B, 2026",
    },
    "current_actual": {
        "description":   "Conservative case based on recent actual market prices",
        "price_p_per_therm": 120.0,
        "reference":     "UK NBP wholesale gas, ~10 July 2026 (Catalyst Commercial "
                          "UK Energy Market Report) — higher than DESNZ central",
    },
}

# Default gas scenario used when resolve_gas_price() is given None
DEFAULT_GAS_SCENARIO = "desnz_central"

# ── Ofgem energy price cap — the CUSTOMER-FACING retail tariff ─────────────────
#
# Genuinely DIFFERENT from GAS_PRICE_SCENARIOS above: those are WHOLESALE
# gas prices (what THIS SCHEME pays to buy fuel) — this is the REGULATED
# RETAIL rate a real household customer pays for gas on a standard
# variable tariff, already inclusive of network costs, policy costs,
# supplier operating margin, and VAT (Ofgem's own definition: "the
# maximum amount your supplier can charge for a unit of energy and
# standing charge together"). This is the right basis for this project's
# revenue mechanism: a district heating scheme's customer should
# reasonably be charged what they'd otherwise pay a retail gas supplier
# for the equivalent heat, NOT what the scheme itself pays for wholesale
# fuel input (which is a cost figure, not a price a real customer ever sees).
#
# Real sourcing: Ofgem's official price cap announcement for 1 July to
# 30 September 2026 (ofgem.gov.uk/news/changes-energy-price-cap-between-
# 1-july-and-30-september-2026, published 27 May 2026) — the live cap as
# of this project's current date. Reviewed and reset by Ofgem every 3
# months; OFGEM_GAS_CAP_REVIEW_DATE records which period these figures
# apply to, so a future update is a clear, dated, intentional change,
# not silent drift.
OFGEM_GAS_CAP_P_PER_KWH = 7.33
OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY = 29.04
OFGEM_GAS_CAP_REVIEW_PERIOD = "1 July 2026 - 30 September 2026"


# ── Default escalation rates ────────────────────────────────────────────────────
# Simple real-terms annual escalation — a sensitivity input, NOT a forecast.
# Set explicitly if Dalkia's finance team has a house view to use instead.
DEFAULT_ELEC_ESCALATION_PCT_PER_YEAR = 1.5
DEFAULT_GAS_ESCALATION_PCT_PER_YEAR  = 1.0


# ── Electricity price shape ────────────────────────────────────────────────────

def _build_electricity_shape(
    annual_avg_p_per_kWh: float,
    peak_premium_p: float        = 15.0,
    peak_start_hour: int          = 16,
    peak_end_hour: int            = 19,
    overnight_discount_p: float  = 8.0,
    overnight_start_hour: int     = 23,
    overnight_end_hour: int       = 6,
    winter_premium_pct: float    = 0.10,
    n_hours: int                  = N_HOURS,
) -> np.ndarray:
    """
    Build an 8760-hour electricity price array (p/kWh) with a realistic
    diurnal and seasonal shape, scaled to match the given annual average.

    Shape characteristics (calibrated against public Octopus Agile data):
      - 16:00-19:00 (peak_start_hour to peak_end_hour): consistently most
        expensive window, +peak_premium_p above the unshaped baseline
      - 23:00-06:00 (overnight_start_hour to overnight_end_hour): cheapest
        window, -overnight_discount_p below baseline
      - Winter (Dec-Feb) running ~winter_premium_pct above the annual mean,
        summer correspondingly below — a simple cosine seasonal pattern
        peaking at the winter solstice (day 0 / day 365)

    The peak_premium_p and overnight_discount_p are expressed relative to
    the annual_avg_p_per_kWh you pass in, so the shape scales sensibly
    whether you're modelling a 15p/kWh or 30p/kWh average contract.
    """
    hours = np.arange(n_hours)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    shape = np.ones(n_hours)

    is_peak = (hour_of_day >= peak_start_hour) & (hour_of_day < peak_end_hour)
    shape = np.where(is_peak, shape + peak_premium_p / annual_avg_p_per_kWh, shape)

    if overnight_start_hour > overnight_end_hour:
        # Wraps midnight (e.g. 23:00 to 06:00)
        is_overnight = (hour_of_day >= overnight_start_hour) | (hour_of_day < overnight_end_hour)
    else:
        is_overnight = (hour_of_day >= overnight_start_hour) & (hour_of_day < overnight_end_hour)
    shape = np.where(is_overnight, shape - overnight_discount_p / annual_avg_p_per_kWh, shape)

    # Seasonal: peaks at winter solstice (day ~0/365), troughs at summer solstice (day ~182)
    seasonal_factor = 1.0 + winter_premium_pct * np.cos(2 * np.pi * day_of_year / 365)
    shape = shape * seasonal_factor

    # Floor to avoid zero/negative prices in this simplified model
    # (real wholesale prices CAN go negative — see Agile 'plunge pricing' —
    # but for a commercial consumer's all-in rate this is a reasonable floor)
    shape = np.clip(shape, 0.1, None)

    scale = annual_avg_p_per_kWh / shape.mean()
    return shape * scale


# ── ElectricityTariff class ────────────────────────────────────────────────────

class ElectricityTariff:
    """
    Hourly electricity price profile for a large commercial/industrial
    consumer (e.g. a district energy centre), with realistic diurnal and
    seasonal shape.

    Parameters
    ----------
    annual_avg_p_per_kWh    : target annual average price (p/kWh) BEFORE
                               any negotiated discount. Default 24.0 —
                               midpoint of the publicly reported large
                               business electricity range (21.0-27.0 p/kWh).
    negotiated_discount_pct : explicit discount (%) applied on top of the
                               public-range central case, representing a
                               bespoke negotiated rate (e.g. via an existing
                               supplier relationship). Default 0.0 — set
                               this explicitly rather than changing
                               annual_avg_p_per_kWh directly, so the
                               assumption stays visible and auditable.
    peak_premium_p, overnight_discount_p, winter_premium_pct
                            : shape parameters, see _build_electricity_shape()
    standing_charge_p_per_day : daily standing charge (p/day). Reported
                               separately from the hourly array since it's
                               a fixed cost, not a per-MWh marginal cost —
                               include it in OPEX/economics calculations,
                               not in dispatch marginal cost comparisons.
    year                    : the year this tariff represents (for escalation)
    """

    def __init__(
        self,
        annual_avg_p_per_kWh: float        = ELEC_PRICE_RANGE_P_PER_KWH["central"],
        negotiated_discount_pct: float     = 0.0,
        peak_premium_p: float               = 15.0,
        overnight_discount_p: float        = 8.0,
        winter_premium_pct: float          = 0.10,
        standing_charge_p_per_day: float   = ELEC_STANDING_CHARGE_P_PER_DAY,
        year: int                           = BASE_YEAR,
    ):
        self.annual_avg_p_per_kWh_public = float(annual_avg_p_per_kWh)
        self.negotiated_discount_pct     = float(negotiated_discount_pct)
        self.standing_charge_p_per_day   = float(standing_charge_p_per_day)
        self.year                        = int(year)

        # Apply negotiated discount to get the effective average used for the shape
        self.annual_avg_p_per_kWh_effective = (
            self.annual_avg_p_per_kWh_public * (1 - self.negotiated_discount_pct / 100)
        )

        price_p_per_kWh = _build_electricity_shape(
            annual_avg_p_per_kWh=self.annual_avg_p_per_kWh_effective,
            peak_premium_p=peak_premium_p,
            overnight_discount_p=overnight_discount_p,
            winter_premium_pct=winter_premium_pct,
        )

        self.price_p_per_kWh   = price_p_per_kWh
        self.price_GBP_per_MWh = price_p_per_kWh * 10   # p/kWh -> £/MWh

    def escalate_to_year(
        self,
        target_year: int,
        escalation_pct_per_year: float = DEFAULT_ELEC_ESCALATION_PCT_PER_YEAR,
    ) -> "ElectricityTariff":
        """
        Return a NEW ElectricityTariff representing a future year, with the
        annual average escalated at a simple compound real-terms rate.
        Shape (diurnal/seasonal pattern) is held constant — only the level
        changes. Does not mutate self.

        Parameters
        ----------
        target_year             : the year to escalate to
        escalation_pct_per_year : real-terms annual escalation (%). Default
                                   is a sensitivity assumption, not a forecast
                                   — override with a house view if available.
        """
        years_ahead = target_year - self.year
        factor = (1 + escalation_pct_per_year / 100) ** years_ahead

        return ElectricityTariff(
            annual_avg_p_per_kWh=self.annual_avg_p_per_kWh_public * factor,
            negotiated_discount_pct=self.negotiated_discount_pct,
            standing_charge_p_per_day=self.standing_charge_p_per_day * factor,
            year=target_year,
        )

    def summary(self) -> dict:
        hour_of_day = np.arange(N_HOURS) % 24
        peak_mean = self.price_p_per_kWh[(hour_of_day >= 16) & (hour_of_day < 19)].mean()
        overnight_mean = self.price_p_per_kWh[(hour_of_day >= 23) | (hour_of_day < 6)].mean()

        return {
            "year":                          self.year,
            "annual_avg_p_per_kWh_public":   round(self.annual_avg_p_per_kWh_public, 2),
            "negotiated_discount_pct":       self.negotiated_discount_pct,
            "annual_avg_p_per_kWh_effective": round(self.annual_avg_p_per_kWh_effective, 2),
            "actual_mean_p_per_kWh":         round(float(self.price_p_per_kWh.mean()), 2),
            "min_p_per_kWh":                 round(float(self.price_p_per_kWh.min()), 2),
            "max_p_per_kWh":                 round(float(self.price_p_per_kWh.max()), 2),
            "evening_peak_mean_p_per_kWh":   round(float(peak_mean), 2),
            "overnight_mean_p_per_kWh":      round(float(overnight_mean), 2),
            "peak_to_overnight_ratio":       round(float(peak_mean / overnight_mean), 2),
            "standing_charge_GBP_per_year":  round(self.standing_charge_p_per_day * 365 / 100, 0),
        }

    def __repr__(self):
        return (
            f"ElectricityTariff(year={self.year}, "
            f"avg={self.annual_avg_p_per_kWh_effective:.1f}p/kWh, "
            f"discount={self.negotiated_discount_pct:.0f}%)"
        )


# ── GasTariff class ────────────────────────────────────────────────────────────

class GasTariff:
    """
    Gas price for the district energy model. Flat (no diurnal shape) since
    gas prices don't show the strong intraday pattern electricity does —
    the LEVEL is what matters, and is provided as a deliberate sensitivity
    pair (DESNZ central vs a more conservative current-actual case).

    Parameters
    ----------
    price_p_per_therm : gas price in p/therm (standard UK gas unit)
    year                : the year this tariff represents (for escalation)
    """

    def __init__(
        self,
        price_p_per_therm: float,
        year: int = BASE_YEAR,
        scenario_name: str = "custom",
        reference: str = "",
    ):
        self.price_p_per_therm = float(price_p_per_therm)
        self.year               = int(year)
        self.scenario_name      = scenario_name
        self.reference          = reference

        price_p_per_kWh = self.price_p_per_therm / THERM_TO_KWH
        price_GBP_per_MWh = price_p_per_kWh * 10

        # Flat across all hours — gas doesn't have electricity's diurnal shape
        self.price_p_per_kWh   = np.full(N_HOURS, price_p_per_kWh)
        self.price_GBP_per_MWh = np.full(N_HOURS, price_GBP_per_MWh)

    @classmethod
    def from_scenario(cls, scenario_key: str, **overrides) -> "GasTariff":
        """
        Construct from a named scenario (see GAS_PRICE_SCENARIOS).

        Example
        -------
            gas_central = GasTariff.from_scenario("desnz_central")
            gas_conservative = GasTariff.from_scenario("current_actual")
        """
        if scenario_key not in GAS_PRICE_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_key}'. "
                f"Available: {list(GAS_PRICE_SCENARIOS.keys())}"
            )
        params = GAS_PRICE_SCENARIOS[scenario_key].copy()
        params.pop("description")
        params["scenario_name"] = scenario_key
        params.update(overrides)
        return cls(**params)

    def escalate_to_year(
        self,
        target_year: int,
        escalation_pct_per_year: float = DEFAULT_GAS_ESCALATION_PCT_PER_YEAR,
    ) -> "GasTariff":
        """
        Return a NEW GasTariff representing a future year, escalated at a
        simple compound real-terms rate. Does not mutate self.
        """
        years_ahead = target_year - self.year
        factor = (1 + escalation_pct_per_year / 100) ** years_ahead

        return GasTariff(
            price_p_per_therm=self.price_p_per_therm * factor,
            year=target_year,
            scenario_name=self.scenario_name,
            reference=self.reference,
        )

    def summary(self) -> dict:
        return {
            "year":               self.year,
            "scenario_name":      self.scenario_name,
            "price_p_per_therm":  round(self.price_p_per_therm, 1),
            "price_p_per_kWh":    round(float(self.price_p_per_kWh[0]), 3),
            "price_GBP_per_MWh":  round(float(self.price_GBP_per_MWh[0]), 2),
            "reference":          self.reference,
        }

    def __repr__(self):
        return (
            f"GasTariff(year={self.year}, scenario='{self.scenario_name}', "
            f"{self.price_p_per_therm:.0f}p/therm = "
            f"£{self.price_GBP_per_MWh[0]:.2f}/MWh)"
        )


# ── Component-integration helpers ───────────────────────────────────────────────
# These are what ASHP.py / peak_demand_option.py call so that "what price do
# I use" becomes a single, swappable decision point rather than something
# hard-coded inside every component. Designed with the future scenario-menu
# in mind: a YAML/UI selection can resolve to a Tariff object, a flat number,
# or nothing (→ sensible default) and every component handles all three
# identically.

PriceInput = Union[None, "ElectricityTariff", "GasTariff", float, int, np.ndarray, list]


def resolve_electricity_price(
    price: PriceInput,
    n_hours: int = N_HOURS,
) -> np.ndarray:
    """
    Normalise any accepted electricity price input into an 8760-length
    £/MWh array.

    Accepts
    -------
    None               -> default ElectricityTariff() central case (~24p/kWh)
    ElectricityTariff  -> uses its .price_GBP_per_MWh hourly shape
    float / int        -> flat array at that £/MWh value (explicit override —
                           use this if you deliberately want to strip out the
                           diurnal/seasonal shape, e.g. for a quick sanity check)
    array-like (8760)   -> used directly (e.g. a custom or escalated series)
    """
    if price is None:
        return ElectricityTariff().price_GBP_per_MWh
    if isinstance(price, ElectricityTariff):
        return price.price_GBP_per_MWh
    if np.isscalar(price):
        return np.full(n_hours, float(price))
    arr = np.asarray(price, dtype=float)
    if len(arr) != n_hours:
        raise ValueError(
            f"electricity_price_GBP_per_MWh array must have {n_hours} "
            f"elements; got {len(arr)}."
        )
    return arr


def resolve_gas_price(
    price: PriceInput,
    n_hours: int = N_HOURS,
) -> np.ndarray:
    """
    Normalise any accepted gas price input into an 8760-length £/MWh array.

    Accepts
    -------
    None         -> default GasTariff.from_scenario('desnz_central')
    GasTariff    -> uses its .price_GBP_per_MWh (flat, but kept as an array
                    for interface consistency and future-proofing against
                    e.g. seasonal gas pricing)
    float / int   -> flat array at that £/MWh value (explicit override)
    array-like (8760) -> used directly
    """
    if price is None:
        return GasTariff.from_scenario(DEFAULT_GAS_SCENARIO).price_GBP_per_MWh
    if isinstance(price, GasTariff):
        return price.price_GBP_per_MWh
    if np.isscalar(price):
        return np.full(n_hours, float(price))
    arr = np.asarray(price, dtype=float)
    if len(arr) != n_hours:
        raise ValueError(
            f"gas_price_GBP_per_MWh array must have {n_hours} "
            f"elements; got {len(arr)}."
        )
    return arr


def customer_heat_revenue_GBP(
    annual_heat_delivered_kWh: float,
    n_connected_buildings: int,
    unit_rate_p_per_kWh: float = OFGEM_GAS_CAP_P_PER_KWH,
    standing_charge_p_per_day: float = OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY,
) -> dict:
    """
    Real customer-facing revenue for heat delivered, using the Ofgem
    price cap as the basis — "charge what a household would otherwise
    be charged for the equivalent gas heating" (see OFGEM_GAS_CAP_*
    constants above for the real, dated sourcing). This is NOT a
    proxy or an invented number — it's the actual regulated rate a
    real customer pays today, used directly as this project's revenue
    mechanism, per this project's own explicit design decision.

    Two real components of a household energy bill, both included:
      1. Unit rate x energy delivered (the variable component)
      2. Standing charge x days x number of CONNECTED BUILDINGS (the
         fixed component — every real customer pays a standing charge
         regardless of how much they use, and a district scheme bills
         EACH connected building separately, so this scales with
         n_connected_buildings, not just total energy)

    Parameters
    ----------
    annual_heat_delivered_kWh : total heat delivered to customers over
                  the year (kWh) — e.g. dispatch_result.summary()'s
                  "annual_demand_MWh" * 1000, or the building-level
                  total_heat_kW.sum() for a per-building view
    n_connected_buildings        : number of buildings billed (each pays
                  its own standing charge — a real district scheme
                  bills per connection, not once for the whole network)
    unit_rate_p_per_kWh           : default OFGEM_GAS_CAP_P_PER_KWH (the
                  live cap as of this project's current date — see that
                  constant's docstring note for the review period this
                  applies to)
    standing_charge_p_per_day      : default OFGEM_GAS_CAP_STANDING_CHARGE_P_PER_DAY

    Returns
    -------
    dict: {
        "unit_rate_revenue_GBP", "standing_charge_revenue_GBP", "total_revenue_GBP"
    }
    """
    unit_rate_revenue_GBP = annual_heat_delivered_kWh * unit_rate_p_per_kWh / 100.0
    standing_charge_revenue_GBP = (
        standing_charge_p_per_day * 365.0 * n_connected_buildings / 100.0
    )
    return {
        "unit_rate_revenue_GBP": round(unit_rate_revenue_GBP, 0),
        "standing_charge_revenue_GBP": round(standing_charge_revenue_GBP, 0),
        "total_revenue_GBP": round(unit_rate_revenue_GBP + standing_charge_revenue_GBP, 0),
    }


if __name__ == "__main__":
    print(
        "\nThis file's self-test has moved to tests/test_tariffs.py "
        "(see this project's file-restructuring decision) -- run:\n"
        "    python3 tests/test_tariffs.py\n"
    )
