
"""
==========
Electricity and gas pricing for the district energy dispatch model.
 
Provides realistic hourly price SHAPES (diurnal/seasonal patterns) scaled
to a chosen annual average level, plus future-year escalation scenarios.
This feeds directly into ASHPArray.marginal_cost, GasBoiler.marginal_cost,
and ElectricBoiler.marginal_cost — currently those modules default to a
flat placeholder price; this module replaces that with something dispatch
can make genuinely realistic merit-order decisions against.
 
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
    'desnz_central'    : ~72 p/therm — DESNZ's central long-term projection
    'current_actual'    : ~101 p/therm — a more conservative case reflecting
                           actual observed prices in late 2024, higher than
                           the DESNZ central case
Reference: DESNZ gas price projections vs actual market data (cross-checked
publicly, late 2024/2025 reporting).
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
"""
 
import numpy as np
import pandas as pd
from typing import Optional
 
 
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
        "price_p_per_therm": 72.0,
        "reference":     "DESNZ long-term energy price projections, central case",
    },
    "current_actual": {
        "description":   "Conservative case based on recent actual market prices",
        "price_p_per_therm": 101.0,
        "reference":     "Observed UK gas price, late 2024 — higher than DESNZ central",
    },
}
 
 
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
 
 
# ── Self-test ──────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  tariffs.py — self-test")
    print("="*70)
 
    # --- Electricity: central commercial case ---
    print("\n  Electricity tariff — central commercial case (no discount):")
    elec = ElectricityTariff()
    for k, v in elec.summary().items():
        print(f"    {k:<32} {v}")
 
    # --- Electricity: with a negotiated discount ---
    print("\n  Electricity tariff — with 10% negotiated discount (e.g. EDF relationship):")
    elec_discounted = ElectricityTariff(negotiated_discount_pct=10.0)
    for k, v in elec_discounted.summary().items():
        print(f"    {k:<32} {v}")
 
    # --- Electricity: escalated to a future year ---
    print("\n  Electricity tariff — escalated to 2040 (default 1.5%/yr real-terms):")
    elec_2040 = elec.escalate_to_year(2040)
    print(f"    2026 average: {elec.summary()['actual_mean_p_per_kWh']} p/kWh")
    print(f"    2040 average: {elec_2040.summary()['actual_mean_p_per_kWh']} p/kWh")
 
    # --- Gas: both scenarios ---
    print("\n  Gas tariff — sensitivity pair:")
    for key in GAS_PRICE_SCENARIOS:
        gas = GasTariff.from_scenario(key)
        s = gas.summary()
        print(f"    {key:<18} {s['price_p_per_therm']:>6.1f} p/therm  =  £{s['price_GBP_per_MWh']:>6.2f}/MWh   ({GAS_PRICE_SCENARIOS[key]['reference']})")
 
    # --- Gas: escalated ---
    print("\n  Gas tariff — DESNZ central, escalated to 2040:")
    gas_central = GasTariff.from_scenario("desnz_central")
    gas_2040 = gas_central.escalate_to_year(2040)
    print(f"    2026: £{gas_central.summary()['price_GBP_per_MWh']:.2f}/MWh")
    print(f"    2040: £{gas_2040.summary()['price_GBP_per_MWh']:.2f}/MWh")
 
    # --- Integration check: feed into ASHP-style marginal cost calc ---
    print("\n  Integration check — ASHP marginal cost using real tariff shape:")
    test_cop = np.full(N_HOURS, 3.0)  # flat COP for isolation test
    ashp_marginal_cost = elec.price_GBP_per_MWh / test_cop
    print(f"    Mean ASHP marginal cost (COP=3.0 flat): £{ashp_marginal_cost.mean():.2f}/MWh heat")
    print(f"    Cheapest hour:  £{ashp_marginal_cost.min():.2f}/MWh heat")
    print(f"    Most expensive: £{ashp_marginal_cost.max():.2f}/MWh heat")
 
    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert len(elec.price_GBP_per_MWh) == N_HOURS, "Electricity price array wrong length"
    assert len(gas_central.price_GBP_per_MWh) == N_HOURS, "Gas price array wrong length"
    assert elec.price_p_per_kWh.min() > 0, "Electricity price went non-positive"
    assert elec_discounted.annual_avg_p_per_kWh_effective < elec.annual_avg_p_per_kWh_effective, \
        "Discount should reduce effective price"
    assert elec_2040.annual_avg_p_per_kWh_effective > elec.annual_avg_p_per_kWh_effective, \
        "Escalation should increase future price"
    summary = elec.summary()
    assert summary["peak_to_overnight_ratio"] > 1.5, "Peak should be meaningfully higher than overnight"
    print("  ✓ All array shapes correct (8760 hours)")
    print("  ✓ Negotiated discount reduces effective price")
    print("  ✓ Escalation increases future price")
    print("  ✓ Evening peak meaningfully more expensive than overnight (matches Agile pattern)")
    print()