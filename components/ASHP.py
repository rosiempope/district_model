"""
ashp_source.py
==============
Air Source Heat Pump (ASHP) heat source model for the district energy system.
 
Unlike source.py's DataCentre (which has a near-constant supply temperature),
ASHPs are weather-dependent — their COP and available capacity both vary
hour-by-hour with ambient air temperature. This module models a single,
generalised ASHP "array" — internally it represents N identical units of a
given unit size, so you can scale from one rooftop unit to a multi-MW bank
just by changing two numbers (n_units, unit_capacity_MW).
 
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
source.py — one class, parameterised, with presets and YAML config support.
 
Usage
-----
    from ashp_source import ASHPArray
    from parse_epw import parse_epw
 
    _, weather_df = parse_epw("data/profiles/GBR_ENG_London-Heathrow.epw")
 
    # From a preset (Ealing report Phase 1: 2.8 MW)
    ashp = ASHPArray.from_preset("ealing_phase1", weather_df, flow_temp_C=65.0)
 
    # Fully custom array — change scale freely
    ashp = ASHPArray(
        name="Town centre ASHP bank",
        n_units=4,
        unit_capacity_MW=0.7,        # 4 x 700kW units = 2.8 MW total
        flow_temp_C=65.0,
        weather_df=weather_df,
    )
 
    print(ashp.capacity_MW)          # 2.8
    print(ashp.cop_hourly[:24])      # First day's COP profile
    print(ashp.supply_MW[:24])       # First day's available thermal output
    print(ashp.electrical_demand_MW[:24])  # Electricity consumed to deliver that heat
"""

 
import numpy as np
import pandas as pd
from typing import Optional
 
 
# ── Constants ──────────────────────────────────────────────────────────────────
 
N_HOURS = 8760
 
# EN14825 standard rating point — the ambient temperature at which
# manufacturers quote "rated capacity" on datasheets
RATING_POINT_TEMP_C = 7.0
 
 
# ── Ealing/UK district heating ASHP presets ───────────────────────────────────
# Source: Ealing Town Centre Heat Network Feasibility Report (SEL, 2025)
# "energy centre would include 2.8 MW ASHP and 3.6 MW of peak and reserve boilers"
 
ASHP_PRESETS = {
    "ealing_phase1": {
        "description":       "Ealing Town Centre Phase 1 ASHP bank",
        "n_units":            4,
        "unit_capacity_MW":   0.7,     # 4 x 700kW = 2.8 MW total
        "flow_temp_C":        65.0,    # LTHW network, per Ealing report
        "min_ambient_temp_C": -10.0,
        "reference":         "Ealing report p.5: '2.8 MW ASHP'",
    },
    "ealing_phase2": {
        "description":       "Ealing Town Centre Phase 2 ASHP expansion",
        "n_units":            5,
        "unit_capacity_MW":   1.0,     # 5 x 1.0 MW = 5.0 MW total
        "flow_temp_C":        65.0,
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
 
 
# ── ASHPArray class ────────────────────────────────────────────────────────────
 
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
    electricity_price_GBP_per_MWh : either a constant float or an 8760-length
                              array for time-varying electricity pricing
    capex_GBP_per_MW        : capital cost per MW installed (for reporting —
                              actual CAPEX calcs live in economics/CAPEX.py)
    seed                    : unused currently (kept for interface consistency
                              with DataCentre — ASHPs have no random outages
                              modelled here, but reserved for future use)
    """
 
    source_type = "ashp"
 
    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        flow_temp_C: float                      = 65.0,
        weather_df: Optional[pd.DataFrame]       = None,
        min_ambient_temp_C: float                = -10.0,
        min_capacity_fraction: float             = 0.65,
        apply_defrost: bool                      = True,
        electricity_price_GBP_per_MWh            = 120.0,
        capex_GBP_per_MW: float                  = 600_000.0,
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
        self.reference              = reference
 
        T_air = weather_df["temp_drybulb_C"].values[:N_HOURS].astype(float)
        self.ambient_temp_C = T_air
 
        # COP at every hour
        self.cop_hourly = ashp_cop(
            T_air, self.flow_temp_C, apply_defrost=apply_defrost
        )
 
        # Capacity derating at every hour
        self._capacity_fraction = _capacity_derate(
            T_air,
            rating_point_C=RATING_POINT_TEMP_C,
            min_ambient_C=self.min_ambient_temp_C,
            min_capacity_fraction=self.min_capacity_fraction,
        )
 
        # Available thermal supply at each hour (MW) — this is what the
        # dispatch optimiser can call on, NOT what it necessarily produces
        # (that depends on how much heat is actually demanded that hour)
        self.supply_MW = self.capacity_MW * self._capacity_fraction
 
        # Supply temperature is just the flow temperature (ASHPs lift to
        # the design flow temp directly, unlike DC waste heat which needs
        # a separate heat pump stage)
        self.supply_temp_C = np.full(N_HOURS, self.flow_temp_C)
 
        # Electricity price — accept scalar or array
        if np.isscalar(electricity_price_GBP_per_MWh):
            self._elec_price = np.full(N_HOURS, float(electricity_price_GBP_per_MWh))
        else:
            elec_arr = np.asarray(electricity_price_GBP_per_MWh, dtype=float)
            if len(elec_arr) != N_HOURS:
                raise ValueError(
                    f"electricity_price_GBP_per_MWh array must have {N_HOURS} "
                    f"elements; got {len(elec_arr)}."
                )
            self._elec_price = elec_arr
 
        # Marginal cost of heat delivered (£/MWh_heat) = elec_price / COP
        # This is what the dispatch optimiser compares against other sources
        self.marginal_cost = self._elec_price / self.cop_hourly
 
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
            min_ambient_temp_C, min_capacity_fraction, electricity_price_GBP_per_MWh
 
        Example YAML block
        -------------------
            heat_sources:
              - type: ashp
                name: "Town centre ASHP bank"
                n_units: 4
                unit_capacity_MW: 0.7
                flow_temp_C: 65.0
                min_ambient_temp_C: -10.0
                electricity_price_GBP_per_MWh: 120.0
        """
        cfg = {k: v for k, v in config.items() if k != "type"}
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
            "mean_marginal_cost_GBP_per_MWh": round(float(self.marginal_cost.mean()), 2),
            "estimated_capex_GBP":        round(self.capacity_MW * self.capex_GBP_per_MW, 0),
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
    print("  ashp_source.py — self-test")
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
    print("\n  COP curve sanity check (flow temp = 65°C, with defrost):")
    test_temps = np.array([-15, -10, -5, -2, 0, 2, 5, 8, 10, 15, 20, 25])
    cops = ashp_cop(test_temps, T_flow_C=65.0)
    for t, c in zip(test_temps, cops):
        print(f"    T_amb={t:>4}°C  COP={c:.2f}")
 
    # Test all presets
    print("\n  All ASHP presets:")
    print(f"  {'Preset':<25} {'Capacity MW':>12} {'Mean COP':>10} {'Annual MWh':>12}")
    print("  " + "-"*62)
    for key in ASHP_PRESETS:
        ashp = ASHPArray.from_preset(key, weather_df)
        s = ashp.summary()
        print(f"  {key:<25} {s['total_capacity_MW']:>12.1f} {s['cop_mean']:>10.2f} {s['annual_heat_available_MWh']:>12.0f}")
 
    # Detailed test: Ealing Phase 1
    print("\n  Ealing Phase 1 ASHP (detailed):")
    ealing = ASHPArray.from_preset("ealing_phase1", weather_df)
    for k, v in ealing.summary().items():
        print(f"    {k:<36} {v}")
 
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
 
    # Array shape and bounds checks
    assert len(ealing.cop_hourly)    == N_HOURS, "cop_hourly wrong length"
    assert len(ealing.supply_MW)     == N_HOURS, "supply_MW wrong length"
    assert len(ealing.marginal_cost) == N_HOURS, "marginal_cost wrong length"
    assert ealing.supply_MW.max() <= ealing.capacity_MW + 0.001, "supply exceeds capacity"
    assert ealing.cop_hourly.min() >= 1.2, "COP below floor"
    assert ealing.cop_hourly.max() <= 6.0, "COP above ceiling"
    print(f"\n  ✓ All array shapes correct (8760 hours)")
    print(f"  ✓ Supply never exceeds nameplate capacity")
    print(f"  ✓ COP within physical bounds [1.2, 6.0]")
    print()