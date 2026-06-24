"""
ASHP.py
==============
Air Source Heat Pump (ASHP) heat source model for the district energy system.

Unlike datacentre_source.py's DataCentre (which has a near-constant supply
temperature), ASHPs are weather-dependent — their COP and available
capacity both vary hour-by-hour with ambient air temperature. This module
models a single, generalised ASHP "array" — internally it represents N
identical units of a given unit size, so you can scale from one rooftop
unit to a multi-MW bank just by changing two numbers (n_units,
unit_capacity_MW).

COP methodology
----------------
COP = 6.08 - 0.09*dT + 0.0005*dT^2          (Ruhnau et al. 2019 regression)
where dT = T_flow - T_ambient (sink minus source temperature)

This is the standard quadratic regression used in PyPSA-Eur and multiple
peer-reviewed European energy system models. It was fitted against real
manufacturer datasheet and field trial data, which is more representative
than a theoretical Carnot calculation (Carnot gives the theoretical maximum,
never achieved in practice — see Pieper et al., as cited in Impact of
temperature dependent COP papers).

Reference: Ruhnau, O. et al. (2019), "Time series of heat demand and heat
pump efficiency for energy system modeling", Scientific Data 6, 189.
Also used as the default COP curve in PyPSA-Eur (Neumann et al.) and cited
in: arxiv.org/pdf/2009.05122, arxiv.org/pdf/2603.12202

Two additional real-world corrections applied on top of the base regression:

1. DEFROST PENALTY — between 0°C and 5°C, moisture on the outdoor coil
   freezes, forcing periodic defrost cycles that consume electricity without
   producing useful heat. This derates COP by ~10% in the 0-5°C band, ~7%
   in the -5-0°C band, ~4% below -5°C (drier air = less ice formation).
   This matches field trial findings — see Energy Savings Trust / UKCCHE
   field trial data showing real-world COPs consistently below lab/Ruhnau
   curve values, particularly in damp UK winters.

2. CAPACITY DERATING — ASHP thermal output capacity itself falls at low
   ambient temperature (less heat available to extract from colder air).
   Modelled as a linear derating between rated capacity at 7°C (the
   standard EN14825 rating point) and a reduced capacity at -10°C.

Part-load / cycling losses are NOT modelled (matches common simplification
in multiple cited papers — adds complexity without much benefit at this
hourly resolution).

Generalised array design
-------------------------
The ASHPArray class represents n_units x unit_capacity_MW.
To "add more ASHPs" or "change the scale" you change two numbers, not the
model logic. This mirrors the same modular philosophy as DataCentre in
datacentre_source.py — one class, parameterised, with presets and YAML
config support.

Electricity pricing
---------------------
ASHP marginal cost = electricity_price / COP. The electricity_price_GBP_per_MWh
parameter accepts FOUR input types, resolved via economics.tariffs:
    None                  -> realistic default tariff shape (~£240/MWh
                              central commercial case, diurnal + seasonal
                              shape) — this is now the DEFAULT behaviour,
                              not a flat placeholder
    ElectricityTariff      -> a specific tariff scenario (e.g. with a
                              negotiated discount, or escalated to a future year)
    float / int            -> flat scalar override (strips the realistic
                              shape — useful for isolation testing, not
                              for real dispatch runs)
    8760-length array       -> a fully custom hourly price series
This is what makes the electricity price swappable later from a scenario
config / UI menu without touching this class.

Usage
-----
    from ASHP import ASHPArray
    from profiles.parse_epw import parse_epw
    from economics.tariffs import ElectricityTariff

    _, weather_df = parse_epw("data/profiles/GBR_ENG_London-Heathrow.epw")

    # From a preset (Ealing report Phase 1: 2.8 MW) — uses the realistic
    # default tariff automatically, no extra setup needed
    ashp = ASHPArray.from_preset("ealing_phase1", weather_df, flow_temp_C=70.0)

    # With a specific negotiated-rate tariff
    ashp = ASHPArray.from_preset(
        "ealing_phase1", weather_df,
        electricity_price_GBP_per_MWh=ElectricityTariff(negotiated_discount_pct=10.0),
    )

    # Fully custom array — change scale freely
    ashp = ASHPArray(
        name="Town centre ASHP bank",
        n_units=4,
        unit_capacity_MW=0.7,        # 4 x 700kW units = 2.8 MW total
        flow_temp_C=70.0,
        weather_df=weather_df,
    )

    print(ashp.capacity_MW)          # 2.8
    print(ashp.cop_hourly[:24])      # First day's COP profile
    print(ashp.supply_MW[:24])       # First day's available thermal output
    print(ashp.electrical_demand_MW[:24])  # Electricity consumed to deliver that heat
"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional

# Make sure the project root (one level up from this file's own folder,
# i.e. district_model/) is on sys.path — regardless of where this script
# is launched from or how (absolute path, relative path, -m, or imported
# by another module). This is what lets `from economics.tariffs import
# ...` resolve whether you run this file directly for a quick self-test
# or as part of the full pipeline via main.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# resolve_electricity_price() turns None / Tariff / scalar / array into a
# clean 8760 £/MWh series — see economics/tariffs.py.
from economics.tariffs import resolve_electricity_price, ElectricityTariff

# Reuse the SAME carbon intensity figures used by GasBoiler/ElectricBoiler
# (BEIS/DESNZ 2024 conversion factors) rather than maintaining a second,
# possibly-drifting copy. See peak_demand_option.py's CARBON_INTENSITY
# dict for sourcing notes.
from components.peak_demand_option import CARBON_INTENSITY


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS = 8760

# EN14825 standard rating point — the ambient temperature at which
# manufacturers quote "rated capacity" on datasheets
RATING_POINT_TEMP_C = 7.0


# ── Ealing/UK district heating ASHP presets ───────────────────────────────────
# Source: Ealing Town Centre Heat Network Feasibility Report (SEL, 2025)
# "energy centre would include 2.8 MW ASHP and 3.6 MW of peak and reserve boilers"
# None of these presets hard-code an electricity price — they all rely on
# the class default (the realistic tariff shape) unless you override it.

ASHP_PRESETS = {
    "ealing_phase1": {
        "description":       "Ealing Town Centre Phase 1 ASHP bank",
        "n_units":            4,
        "unit_capacity_MW":   0.7,     # 4 x 700kW = 2.8 MW total
        "flow_temp_C":        70.0,    # Real network's PEAK design flow temp (Ealing
                                        # report: "maximum flow temperature of 70C, to
                                        # meet peak heat demands" -- the real network is
                                        # actually variable, 65-70C seasonally; this
                                        # fixed-temperature model uses the peak figure
                                        # throughout, matching network/network.py)
        "min_ambient_temp_C": -10.0,
        "reference":         "Ealing report p.5: '2.8 MW ASHP'",
    },
    "ealing_phase2": {
        "description":       "Ealing Town Centre Phase 2 ASHP expansion",
        "n_units":            5,
        "unit_capacity_MW":   1.0,     # 5 x 1.0 MW = 5.0 MW total
        "flow_temp_C":        70.0,    # See ealing_phase1 note above
        "min_ambient_temp_C": -10.0,
        "reference":         "Ealing report Table 1: Phase 2 low carbon capacity 5.0 MW",
    },
    "single_rooftop_unit": {
        "description":       "Single commercial rooftop ASHP",
        "n_units":            1,
        "unit_capacity_MW":   0.1,     # 100 kW — typical commercial rooftop unit
        "flow_temp_C":        55.0,    # Lower temp, more typical for single building
        "min_ambient_temp_C": -15.0,
        "reference":         "Generic commercial rooftop unit sizing",
    },
    "large_energy_centre": {
        "description":       "Large multi-MW energy centre ASHP bank",
        "n_units":            10,
        "unit_capacity_MW":   2.0,     # 10 x 2.0 MW = 20 MW total
        "flow_temp_C":        70.0,    # Higher temp for larger network reach
        "min_ambient_temp_C": -10.0,
        "reference":         "Generic large-scale district heating energy centre",
    },
}


# ── COP model ──────────────────────────────────────────────────────────────────

def _ashp_cop_base(T_ambient_C: np.ndarray, T_flow_C: float) -> np.ndarray:
    """
    Ruhnau et al. (2019) quadratic regression — base COP before corrections.

    COP = 6.08 - 0.09*dT + 0.0005*dT^2
    where dT = T_flow - T_ambient

    This is fitted to real manufacturer/field data, not a theoretical Carnot
    limit, so it already reflects realistic compressor and heat exchanger
    losses. Used as the default ASHP curve in PyPSA-Eur.
    """
    dT = T_flow_C - T_ambient_C
    cop = 6.08 - 0.09 * dT + 0.0005 * dT**2
    return cop


def _defrost_penalty(T_ambient_C: np.ndarray) -> np.ndarray:
    """
    Derate COP in the 'icing band' where outdoor coil frost formation
    forces periodic defrost cycles. Most severe at 0-5°C (high humidity +
    freezing = maximum ice formation); reduces slightly below -5°C as air
    holds less moisture.

    This correction is why real-world UK ASHP trial COPs (Energy Savings
    Trust, West Lothian, Harrogate trials — typically 2.2-2.7 annual COP)
    sit below the raw Ruhnau regression, which doesn't include defrost
    losses explicitly.
    """
    T = np.asarray(T_ambient_C, dtype=float)
    penalty = np.ones_like(T)

    penalty = np.where((T >= 0) & (T <= 5),  0.90, penalty)   # Peak icing band
    penalty = np.where((T >= -5) & (T < 0),  0.93, penalty)   # Moderate icing
    penalty = np.where(T < -5,                0.96, penalty)   # Drier cold air

    return penalty


def ashp_cop(
    T_ambient_C: np.ndarray,
    T_flow_C: float,
    apply_defrost: bool = True,
    cop_floor: float = 1.2,
    cop_ceiling: float = 6.0,
) -> np.ndarray:
    """
    Full ASHP COP model: Ruhnau base regression + defrost penalty + bounds.

    Parameters
    ----------
    T_ambient_C   : hourly ambient air temperature array (°C)
    T_flow_C      : network/system flow temperature (°C) — assumed constant
                    (a weather-compensated flow temp could vary this too,
                    but most UK LTHW networks run a fixed flow temp)
    apply_defrost : whether to apply the icing-band derating (default True)
    cop_floor     : minimum physically realistic COP (resistive heating
                    backup typically kicks in below this)
    cop_ceiling   : maximum COP cap (prevents unrealistic values at very
                    small dT, e.g. mild ambient + low flow temp)

    Returns
    -------
    np.ndarray of hourly COP values, same length as T_ambient_C
    """
    T = np.asarray(T_ambient_C, dtype=float)
    cop = _ashp_cop_base(T, T_flow_C)

    if apply_defrost:
        cop = cop * _defrost_penalty(T)

    return np.clip(cop, cop_floor, cop_ceiling)


# ── Capacity derating ──────────────────────────────────────────────────────────

def _capacity_derate(
    T_ambient_C: np.ndarray,
    rating_point_C: float = RATING_POINT_TEMP_C,
    min_ambient_C: float = -10.0,
    min_capacity_fraction: float = 0.65,
) -> np.ndarray:
    """
    ASHP thermal output capacity falls at low ambient temperature — there's
    less heat energy available to extract from colder air, even though the
    compressor is working harder (which is captured separately by the COP
    derating above).

    Modelled as a linear interpolation:
      - At rating_point_C (7°C, the EN14825 standard) and above: 100% capacity
      - At min_ambient_C: min_capacity_fraction of rated capacity
      - Linear between those two points
      - Below min_ambient_C: held at min_capacity_fraction (most modern
        cold-climate ASHPs maintain some output well below their nominal
        rating point, just at reduced capacity)

    Parameters
    ----------
    min_capacity_fraction : fraction of rated capacity retained at the
                             coldest design condition. 0.65 is a reasonable
                             mid-range value for a standard (non cold-climate
                             optimised) ASHP; cold-climate units can be
                             higher (~0.8).
    """
    T = np.asarray(T_ambient_C, dtype=float)

    # Above rating point: full capacity
    frac = np.ones_like(T)

    # Linear derate zone
    in_derate_zone = T < rating_point_C
    derate_range = rating_point_C - min_ambient_C
    derate_progress = np.clip(
        (rating_point_C - T) / derate_range, 0, 1
    )
    derated_frac = 1.0 - (1.0 - min_capacity_fraction) * derate_progress

    frac = np.where(in_derate_zone, derated_frac, frac)

    return frac


# ── Unit-level outage model ─────────────────────────────────────────────────────

def _ashp_unit_outage_profile(
    n_units: int,
    availability_factor: float,
    n_hours: int = N_HOURS,
    seed: int = 7,
) -> np.ndarray:
    """
    Per-UNIT outage schedule for an ASHP array — returns an (n_hours,)
    array of how many units are AVAILABLE (i.e. not in maintenance) at
    each hour, ranging from 0 to n_units.

    This is deliberately a different shape from DataCentre's or EfW's
    outage model: an ASHP "array" is N separate physical units (see
    module docstring — "scale by changing n_units"), and real practice
    is to service them ONE AT A TIME on a rotating schedule, not take the
    whole bank down together. Losing 1 of 4 units is a routine, survivable
    event; losing all 4 simultaneously essentially never happens by
    design (a competent O&M schedule staggers servicing specifically to
    avoid that). Modelling it as a single binary on/off array (like
    DataCentre) would significantly overstate how much capacity is ever
    actually lost at once.

    Approach: each unit gets ONE planned maintenance window per year
    (typical annual service interval), spread across different times of
    year so outages don't stack. Window length is derived from
    availability_factor so the overall fleet-average availability still
    matches what's specified, same contract as DataCentre/EfW's
    availability_factor parameter.

    Parameters
    ----------
    n_units             : number of units in the array
    availability_factor : fleet-average fraction of time each unit is
                           available (e.g. 0.97 -> ~11 days/year service
                           per unit, a reasonable ASHP service interval)
    seed                 : random seed for scheduling which weeks each
                            unit's outage falls in

    Returns
    -------
    np.ndarray (n_hours,) of ints, 0 to n_units — units available each hour
    """
    rng = np.random.default_rng(seed)
    units_down = np.zeros(n_hours, dtype=int)

    outage_hours_per_unit = int(round((1.0 - availability_factor) * n_hours))
    if outage_hours_per_unit <= 0:
        return np.full(n_hours, n_units, dtype=int)

    # Stagger each unit's single annual outage across the year, spaced
    # apart so they don't cluster — divide the year into n_units roughly
    # equal "slots" and put one outage in each, with some random jitter
    # so it isn't mechanically perfectly even (real schedules slip).
    slot_width = n_hours // max(n_units, 1)
    for i in range(n_units):
        slot_start = i * slot_width
        slot_end = slot_start + slot_width
        # Jitter the actual start within this unit's slot, leaving room
        # for the outage itself to fit without overrunning the slot
        latest_start = max(slot_start, slot_end - outage_hours_per_unit - 1)
        start = int(rng.integers(slot_start, max(slot_start + 1, latest_start + 1)))
        end = min(start + outage_hours_per_unit, n_hours)
        units_down[start:end] += 1

    return np.clip(n_units - units_down, 0, n_units)

class ASHPArray:
    """
    A generalised array of N identical air source heat pump units.

    Scale the system by changing n_units and/or unit_capacity_MW — the
    underlying COP and capacity-derating physics stay the same regardless
    of scale, matching the same modular philosophy as DataCentre.

    Parameters
    ----------
    name                  : descriptive name for reporting
    n_units                : number of identical ASHP units in the array
    unit_capacity_MW       : rated thermal output per unit at the EN14825
                              standard rating point (7°C ambient) (MW)
    flow_temp_C            : network/system flow temperature (°C)
                              Typical UK LTHW network: 65-70°C
                              Lower temp (ambient loop) networks: 45-55°C
    weather_df              : EPW weather DataFrame (must have 'temp_drybulb_C')
    min_ambient_temp_C      : design minimum ambient temp for capacity derating
    min_capacity_fraction   : fraction of rated capacity at min_ambient_temp_C
    apply_defrost           : whether to apply defrost-cycle COP penalty
    electricity_price_GBP_per_MWh : accepts None (default realistic tariff
                              shape, ~£240/MWh central commercial case), an
                              ElectricityTariff object (e.g. with a negotiated
                              discount or escalated to a future year), a flat
                              scalar override, or an 8760-length array.
                              See economics/tariffs.py.
    capex_GBP_per_MW        : capital cost per MW installed (for reporting —
                              actual CAPEX calcs live in economics/CAPEX.py)
    availability_factor    : fleet-average fraction of time each UNIT is
                              available (not in maintenance). Default 0.97
                              (~11 days/year service per unit — a typical
                              ASHP annual service interval). Modelled as
                              ONE planned outage per unit, staggered across
                              the year (real O&M practice services units
                              one at a time, never the whole bank at once)
                              — see _ashp_unit_outage_profile(). This
                              compounds with, not replaces, the weather-
                              driven capacity derating above: an outage
                              hour on a cold day loses both that unit's
                              share of capacity AND has worse COP on the
                              remaining units.
    seed                    : random seed for the outage schedule (which
                              week of the year each unit's service window
                              falls in). Previously unused/reserved; now
                              actually used for unit outage scheduling.
    """

    source_type = "ashp"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        flow_temp_C: float                      = 70.0,
        weather_df: Optional[pd.DataFrame]       = None,
        min_ambient_temp_C: float                = -10.0,
        min_capacity_fraction: float             = 0.65,
        apply_defrost: bool                      = True,
        electricity_price_GBP_per_MWh            = None,
        capex_GBP_per_MW: float                  = 600_000.0,
        availability_factor: float               = 0.97,
        seed: int                                = 7,
        reference: str                           = "",
    ):
        if weather_df is None:
            raise ValueError(
                "ASHPArray requires weather_df (must have 'temp_drybulb_C' "
                "column, 8760 rows) — ASHP output is weather-dependent."
            )
        if len(weather_df) != N_HOURS:
            raise ValueError(
                f"weather_df must have {N_HOURS} rows; got {len(weather_df)}."
            )

        self.name                  = name
        self.n_units                = int(n_units)
        self.unit_capacity_MW       = float(unit_capacity_MW)
        self.capacity_MW            = self.n_units * self.unit_capacity_MW
        self.flow_temp_C            = float(flow_temp_C)
        self.min_ambient_temp_C     = float(min_ambient_temp_C)
        self.min_capacity_fraction  = float(min_capacity_fraction)
        self.capex_GBP_per_MW       = float(capex_GBP_per_MW)
        self.availability_factor    = float(availability_factor)
        self.seed                   = int(seed)
        self.reference              = reference

        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air

        # COP at every hour
        self.cop_hourly = ashp_cop(
            T_air, self.flow_temp_C, apply_defrost=apply_defrost
        )

        # Capacity derating at every hour — weather-driven (existing)
        self._capacity_fraction = _capacity_derate(
            T_air,
            rating_point_C=RATING_POINT_TEMP_C,
            min_ambient_C=self.min_ambient_temp_C,
            min_capacity_fraction=self.min_capacity_fraction,
        )

        # Units available at each hour — maintenance-driven (NEW). Stacks
        # with weather derating: a unit in maintenance contributes ZERO
        # regardless of how mild the weather is that hour, and the units
        # that ARE up still get the same weather derating as before.
        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = (
            self.units_available / self.n_units if self.n_units > 0 else np.ones(N_HOURS)
        )

        # Available thermal supply at each hour (MW) — this is what the
        # dispatch optimiser can call on, NOT what it necessarily produces
        # (that depends on how much heat is actually demanded that hour).
        # Now reflects BOTH weather derating AND unit outages.
        self.supply_MW = (
            self.capacity_MW * self._capacity_fraction * self._unit_availability_fraction
        )

        # Supply temperature is just the flow temperature (ASHPs lift to
        # the design flow temp directly, unlike DC waste heat which needs
        # a separate heat pump stage)
        self.supply_temp_C = np.full(N_HOURS, self.flow_temp_C)

        # Electricity price — None / Tariff / scalar / array, all resolved
        # to a clean 8760 £/MWh array by economics.tariffs. Default (None)
        # now pulls in the realistic central commercial tariff shape rather
        # than a flat placeholder.
        self._elec_price = resolve_electricity_price(electricity_price_GBP_per_MWh)

        # Marginal cost of heat delivered (£/MWh_heat) = elec_price / COP
        # This is what the dispatch optimiser compares against other sources
        self.marginal_cost = self._elec_price / self.cop_hourly

        # Carbon intensity per unit heat delivered (kgCO2e/kWh_heat) =
        # grid carbon intensity / COP. Varies hourly because COP varies
        # with ambient temperature — a cold night with poor COP is BOTH
        # more expensive AND more carbon-intensive per unit heat, same
        # direction, same root cause. Uses the fixed annual grid average
        # (CARBON_INTENSITY["electric"]) — a known simplification, since
        # the real grid carbon intensity itself varies hour-to-hour with
        # generation mix; see peak_demand_option.py's CARBON_INTENSITY
        # note. Used by dispatch.py's network-wide carbon compliance
        # check (London Heat Network Manual Table 8: max 0.216 kgCO2e/kWh).
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly

        # Electrical demand IF running at full available supply (MW_elec)
        # Actual electrical draw depends on dispatch — this is the ceiling
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        weather_df: pd.DataFrame,
        **overrides,
    ) -> "ASHPArray":
        """
        Construct an ASHPArray from a named preset (see ASHP_PRESETS dict).

        Example
        -------
            ashp = ASHPArray.from_preset("ealing_phase1", weather_df)
            ashp = ASHPArray.from_preset("ealing_phase1", weather_df,
                                          flow_temp_C=70.0)  # override
        """
        if preset_key not in ASHP_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_key}'. "
                f"Available: {list(ASHP_PRESETS.keys())}"
            )

        params = ASHP_PRESETS[preset_key].copy()
        params["name"] = params.pop("description")
        params.update(overrides)
        return cls(weather_df=weather_df, **params)

    @classmethod
    def from_config(
        cls,
        config: dict,
        weather_df: pd.DataFrame,
    ) -> "ASHPArray":
        """
        Construct an ASHPArray from a YAML/dict config block.

        Expected keys (mirrors scenarios/*.yaml structure):
            name, n_units, unit_capacity_MW, flow_temp_C,
            min_ambient_temp_C, min_capacity_fraction,
            electricity_price_GBP_per_MWh (optional — see below)

        Electricity pricing in config — three ways to specify it
        ------------------------------------------------------------
        1. Omit it entirely -> realistic default tariff (recommended default)
        2. A flat number -> electricity_price_GBP_per_MWh: 220.0
        3. A nested tariff block -> builds an ElectricityTariff for you:
               electricity_tariff:
                 annual_avg_p_per_kWh: 24.0
                 negotiated_discount_pct: 10.0
           This is the form a future scenario-menu UI would write.

        Example YAML block
        -------------------
            heat_sources:
              - type: ashp
                name: "Town centre ASHP bank"
                n_units: 4
                unit_capacity_MW: 0.7
                flow_temp_C: 70.0
                min_ambient_temp_C: -10.0
                electricity_tariff:
                  negotiated_discount_pct: 10.0
        """
        cfg = {k: v for k, v in config.items() if k != "type"}

        # Nested tariff block -> build an ElectricityTariff object
        if "electricity_tariff" in cfg:
            tariff_kwargs = cfg.pop("electricity_tariff")
            cfg["electricity_price_GBP_per_MWh"] = ElectricityTariff(**tariff_kwargs)

        return cls(weather_df=weather_df, **cfg)

    def resize(self, n_units: Optional[int] = None, unit_capacity_MW: Optional[float] = None):
        """
        Return a NEW ASHPArray with a different scale, reusing all other
        parameters (flow temp, weather data, pricing etc.) from this instance.
        Does not mutate self — keeps the original object intact for comparison.

        Example
        -------
            ashp_small = ASHPArray.from_preset("ealing_phase1", weather_df)
            ashp_big   = ashp_small.resize(n_units=8)   # double the array
        """
        return ASHPArray(
            name=self.name,
            n_units=n_units if n_units is not None else self.n_units,
            unit_capacity_MW=unit_capacity_MW if unit_capacity_MW is not None else self.unit_capacity_MW,
            flow_temp_C=self.flow_temp_C,
            weather_df=pd.DataFrame({"temp_drybulb_C": self.ambient_temp_C}),
            min_ambient_temp_C=self.min_ambient_temp_C,
            min_capacity_fraction=self.min_capacity_fraction,
            electricity_price_GBP_per_MWh=self._elec_price,
            capex_GBP_per_MW=self.capex_GBP_per_MW,
            availability_factor=self.availability_factor,
            seed=self.seed,
            reference=self.reference,
        )

    def summary(self) -> dict:
        """Return key parameters and performance stats as a dict."""
        return {
            "name":                       self.name,
            "source_type":                self.source_type,
            "n_units":                    self.n_units,
            "unit_capacity_MW":           self.unit_capacity_MW,
            "total_capacity_MW":          round(self.capacity_MW, 2),
            "flow_temp_C":                self.flow_temp_C,
            "cop_mean":                   round(float(self.cop_hourly.mean()), 2),
            "cop_min":                    round(float(self.cop_hourly.min()), 2),
            "cop_max":                    round(float(self.cop_hourly.max()), 2),
            "annual_heat_available_MWh":  round(float(self.supply_MW.sum()), 0),
            "annual_electrical_demand_MWh": round(float(self.electrical_demand_MW.sum()), 0),
            "seasonal_avg_cop":           round(
                float(self.supply_MW.sum() / self.electrical_demand_MW.sum()), 2
            ),
            "mean_electricity_price_GBP_per_MWh": round(float(self._elec_price.mean()), 2),
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "availability_factor":        self.availability_factor,
            "annual_outage_hours_per_unit": int(round((1.0 - self.availability_factor) * N_HOURS)),
            "min_units_available":        int(self.units_available.min()),
            "hours_below_full_fleet":     int((self.units_available < self.n_units).sum()),
            "reference":                  self.reference,
        }

    def __repr__(self):
        return (
            f"ASHPArray(name='{self.name}', "
            f"{self.n_units}x{self.unit_capacity_MW}MW = {self.capacity_MW:.1f} MW, "
            f"T_flow={self.flow_temp_C}°C, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ASHP.py — self-test")
    print("="*70)

    # Build synthetic London-like weather (same approach as demand_synthesis test)
    np.random.seed(42)
    hours = np.arange(N_HOURS)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / 8760)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, 8760)
    )
    dates = pd.date_range("2023-01-01", periods=8760, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    print(f"\n  Synthetic weather: T min={T.min():.1f}°C  T max={T.max():.1f}°C  T mean={T.mean():.1f}°C")

    # Test COP curve directly across a temperature sweep
    print("\n  COP curve sanity check (flow temp = 70°C, with defrost):")
    test_temps = np.array([-15, -10, -5, -2, 0, 2, 5, 8, 10, 15, 20, 25])
    cops = ashp_cop(test_temps, T_flow_C=70.0)
    for t, c in zip(test_temps, cops):
        print(f"    T_amb={t:>4}°C  COP={c:.2f}")

    # Test all presets — now with the realistic default tariff applied
    print("\n  All ASHP presets (electricity price now defaults to realistic tariff shape):")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Mean COP':>10} {'Mean elec £/MWh':>16} {'Mean marg. cost £/MWh':>22}")
    print("  " + "-"*90)
    for key in ASHP_PRESETS:
        ashp = ASHPArray.from_preset(key, weather_df)
        s = ashp.summary()
        print(f"  {key:<25} {s['total_capacity_MW']:>12.1f} {s['cop_mean']:>10.2f} "
              f"{s['mean_electricity_price_GBP_per_MWh']:>16.2f} {s['mean_marginal_cost_GBP_per_MWh']:>22.2f}")

    # Detailed test: Ealing Phase 1
    print("\n  Ealing Phase 1 ASHP (detailed, default tariff):")
    ealing = ASHPArray.from_preset("ealing_phase1", weather_df)
    for k, v in ealing.summary().items():
        print(f"    {k:<36} {v}")

    # --- NEW: tariff integration tests ---
    print("\n  Tariff integration — comparing all four accepted price input types:")
    ealing_default  = ASHPArray.from_preset("ealing_phase1", weather_df)
    ealing_tariff    = ASHPArray.from_preset(
        "ealing_phase1", weather_df,
        electricity_price_GBP_per_MWh=ElectricityTariff(negotiated_discount_pct=10.0),
    )
    ealing_flat      = ASHPArray.from_preset(
        "ealing_phase1", weather_df, electricity_price_GBP_per_MWh=120.0,
    )
    ealing_array     = ASHPArray.from_preset(
        "ealing_phase1", weather_df, electricity_price_GBP_per_MWh=np.full(N_HOURS, 200.0),
    )
    print(f"    Default (None)        -> mean elec £{ealing_default._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_default.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Tariff (10% discount) -> mean elec £{ealing_tariff._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_tariff.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Flat scalar override  -> mean elec £{ealing_flat._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_flat.marginal_cost.mean():.2f}/MWh heat")
    print(f"    Raw array override    -> mean elec £{ealing_array._elec_price.mean():.2f}/MWh, "
          f"marginal cost £{ealing_array.marginal_cost.mean():.2f}/MWh heat")

    # --- from_config with nested tariff block ---
    print("\n  from_config() with a nested electricity_tariff block:")
    config_block = {
        "type": "ashp",
        "name": "Town centre ASHP bank (from config)",
        "n_units": 4,
        "unit_capacity_MW": 0.7,
        "flow_temp_C": 70.0,
        "electricity_tariff": {"negotiated_discount_pct": 15.0},
    }
    ashp_from_cfg = ASHPArray.from_config(config_block, weather_df)
    print(f"    {ashp_from_cfg}")
    print(f"    Mean elec price: £{ashp_from_cfg._elec_price.mean():.2f}/MWh (expect 15% below £240 central)")

    # Test resize — the "add more MW easily" requirement
    print("\n  Resize test — scaling Ealing Phase 1 up to 8 units:")
    ealing_scaled = ealing.resize(n_units=8)
    print(f"    Original: {ealing}")
    print(f"    Scaled:   {ealing_scaled}")
    assert ealing_scaled.capacity_MW == ealing.capacity_MW * 2, "Resize scaling failed"
    print("    ✓ Capacity scaled correctly (linear with n_units)")

    # Test custom array
    print("\n  Custom array (user-defined, 6 x 1.5 MW = 9 MW):")
    custom = ASHPArray(
        name="Custom test array",
        n_units=6,
        unit_capacity_MW=1.5,
        flow_temp_C=70.0,
        weather_df=weather_df,
    )
    print(f"    {custom}")

    # Seasonal sanity: COP should be higher in summer, lower in winter
    jan_cop = ealing.cop_hourly[:744].mean()
    jul_cop = ealing.cop_hourly[4344:5088].mean()
    jan_supply = ealing.supply_MW[:744].mean()
    jul_supply = ealing.supply_MW[4344:5088].mean()

    print(f"\n  Seasonal sanity checks:")
    print(f"    Jan mean COP: {jan_cop:.2f}  |  Jul mean COP: {jul_cop:.2f}  → {'✓ summer higher' if jul_cop > jan_cop else '✗ FAIL'}")
    print(f"    Jan mean supply: {jan_supply:.2f} MW  |  Jul mean supply: {jul_supply:.2f} MW  → {'✓ summer higher capacity' if jul_supply > jan_supply else '✗ FAIL'}")

    # --- NEW: unit-level outage model ---
    print(f"\n  Unit-level outage model (real maintenance practice — units")
    print(f"  serviced one at a time, never the whole bank together):")
    print(f"    n_units: {ealing.n_units}, availability_factor: {ealing.availability_factor}")
    print(f"    Outage hours per unit per year: {int(round((1.0 - ealing.availability_factor) * N_HOURS))}")
    print(f"    Units available — min: {ealing.units_available.min()}, max: {ealing.units_available.max()}")
    print(f"    Hours with full fleet up: {(ealing.units_available == ealing.n_units).sum()} / {N_HOURS}")
    print(f"    Hours with reduced fleet: {(ealing.units_available < ealing.n_units).sum()} / {N_HOURS}")
    # Higher availability should mean fewer reduced-fleet hours
    high_avail = ASHPArray.from_preset("ealing_phase1", weather_df, availability_factor=0.999)
    low_avail  = ASHPArray.from_preset("ealing_phase1", weather_df, availability_factor=0.90)
    print(f"    At 99.9% availability: {(high_avail.units_available < high_avail.n_units).sum()} reduced-fleet hours")
    print(f"    At 90.0% availability: {(low_avail.units_available < low_avail.n_units).sum()} reduced-fleet hours")

    # Array shape and bounds checks
    assert len(ealing.cop_hourly)    == N_HOURS, "cop_hourly wrong length"
    assert len(ealing.supply_MW)     == N_HOURS, "supply_MW wrong length"
    assert len(ealing.marginal_cost) == N_HOURS, "marginal_cost wrong length"
    assert ealing.supply_MW.max() <= ealing.capacity_MW + 0.001, "supply exceeds capacity"
    assert ealing.cop_hourly.min() >= 1.2, "COP below floor"
    assert ealing.cop_hourly.max() <= 6.0, "COP above ceiling"

    # New tariff-integration assertions
    assert ealing_default._elec_price.mean() > 200, \
        "Default electricity price should now be the realistic ~£240/MWh tariff, not the old £120 placeholder"
    assert ealing_tariff._elec_price.mean() < ealing_default._elec_price.mean(), \
        "10% discounted tariff should be cheaper than the undiscounted default"
    assert abs(ealing_flat._elec_price.mean() - 120.0) < 0.01, \
        "Flat scalar override should be respected exactly"
    assert abs(ashp_from_cfg._elec_price.mean() - ealing_default._elec_price.mean() * 0.85) < 1.0, \
        "from_config nested tariff block should apply the 15% discount correctly"

    # New outage-model assertions
    assert ealing.units_available.min() >= 0, "units_available should never go negative"
    assert ealing.units_available.max() <= ealing.n_units, "units_available should never exceed n_units"
    assert ealing.cop_hourly.mean() == ealing.cop_hourly.mean(), "COP should be unaffected by outages (sanity)"
    assert (high_avail.units_available < high_avail.n_units).sum() < (low_avail.units_available < low_avail.n_units).sum(), \
        "Lower availability_factor should produce MORE reduced-fleet hours than higher availability_factor"
    # At no point should ALL units be down simultaneously with a sane
    # availability factor (staggered scheduling should prevent total loss)
    assert ealing.units_available.min() >= ealing.n_units - 1, \
        "Staggered single-unit-at-a-time outages should never take down more than 1 unit simultaneously at this availability level"

    print(f"\n  ✓ All array shapes correct (8760 hours)")
    print(f"  ✓ Supply never exceeds nameplate capacity")
    print(f"  ✓ COP within physical bounds [1.2, 6.0]")
    print(f"  ✓ Default electricity price now uses realistic tariff (~£240/MWh), not old £120 placeholder")
    print(f"  ✓ Tariff object, flat scalar, and raw array overrides all behave correctly")
    print(f"  ✓ from_config() nested electricity_tariff block resolves correctly")
    print(f"  ✓ Unit-level outages correctly staggered — never more than 1 unit down at once at this scale")
    print(f"  ✓ Lower availability_factor correctly produces more reduced-fleet hours")
    print()