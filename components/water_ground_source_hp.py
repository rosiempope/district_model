"""Water-source and ground-source heat pumps.

Why these are not ASHPs with a different number
------------------------------------------------
An ASHP's source is the outdoor air, so its COP collapses exactly when demand
peaks: the coldest hour of the year is both the hour you need most heat and the
hour the source is worst. A river sits near 5-8C on that same night, and a
borehole sits at the local annual mean — around 12C — all year, every year.

That difference is not a level shift you can approximate by nudging an ASHP's
COP. It changes the SHAPE of the year: a WSHP/GSHP delivers its best relative
advantage precisely at the winter peak, which is where the gas backup would
otherwise run and where the carbon actually sits. Modelling Birmingham's 5 MW
river WSHP as an ASHP would understate its COP and overstate its carbon at the
only hours that matter.

COP methodology — same Carnot-fraction method as the booster
-------------------------------------------------------------
    COP_carnot = T_sink_K / (T_sink_K - T_source_K)
    COP        = COP_carnot * carnot_efficiency_fraction

DEFAULT_CARNOT_FRACTION = 0.50, the lower-middle of the 45-65% range generally
cited for well-designed large heat pumps. Deliberately conservative against the
best real reference available:

  Drammen, Norway — the reference case for high-temperature district-heating
  heat pumps. Three two-stage ammonia units in series, 14 MW combined, supplying
  85% of the city's hot water at 90C from ~8C seawater, achieving a measured
  COP of 3.05.
      neatpumps.com/case-studies/drammen-district-heating-heatpumps/
      en.wikipedia.org/wiki/Drammen_Heat_Pump

  Back-calculated: Carnot COP at 8C -> 90C is 363.15/82 = 4.43, so Drammen's
  3.05 implies a Carnot fraction of 0.69. That is exceptional, and earned by a
  three-stage SERIES arrangement specifically chosen to split a very large lift.
  A single-stage machine will not reach it. 0.50 is used here so a generic
  scenario does not quietly assume Drammen-class engineering; raise it
  explicitly, with a quotation, if a project genuinely proposes a staged system.

Cross-check at 0.50: a river WSHP at 10C source into a 70C network gives
COP 2.86, against ~2.53 for the same-sink ASHP on this project's weather year
(+13%). That is the right order — better, but not free — and the annual mean
understates the real advantage, which shows up on cold days.

Source temperature is the whole point, so it is modelled, not assumed
---------------------------------------------------------------------
GSHP: a borehole below roughly 15m sits at the local annual mean ground
temperature with no meaningful seasonal swing — this is the well-established
basis for closed-loop GSHP design and is why a borehole is a better source than
a 1m-deep pipe trench. Modelled as constant. NOTE this is deliberately NOT
topology_thermal.seasonal_ground_temp_C(), which is the 1m burial-depth curve
used for pipe heat loss and does swing +/-5.15K seasonally. Different depth,
different physics.

WSHP: a river does follow the seasons, damped and lagged against air. Modelled
as a sinusoid about the stated annual mean, with a default +/-5K amplitude and a
~3-week lag, floored at MIN_RIVER_TEMP_C. This is a shape assumption, not a
measurement — replace it with real gauged river data before using this for a
project decision, since the winter minimum is what sets the design-day COP.

What is NOT modelled
--------------------
Abstraction limits. A real river WSHP needs an environmental permit that caps
how far it may cool the river (typically a few K) and how much it may abstract.
That constrains CAPACITY, not COP, so it belongs on capacity_MW — which the
Birmingham scenarios take from the report's own figures. Open-loop borehole
schemes carry equivalent groundwater abstraction limits. Neither is enforced
here; both are real consenting risks.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.ASHP import _ashp_unit_outage_profile
from components.peak_demand_option import CARBON_INTENSITY
from economics.tariffs import resolve_electricity_price

N_HOURS = 8760

# See module docstring. 0.50 = lower-middle of the 45-65% band for well-designed
# large heat pumps; Drammen's real 3.05 COP implies 0.69 with a 3-stage series
# arrangement, which a generic scenario should not assume.
DEFAULT_CARNOT_FRACTION = 0.50

COP_FLOOR = 1.5
COP_CEILING = 8.0

# A river evaporator must not freeze. UK rivers rarely fall below ~3-4C; this
# floor stops a synthetic cold snap producing a physically impossible source.
MIN_RIVER_TEMP_C = 2.0

# River seasonal shape — see module docstring: an assumption, not gauged data.
RIVER_SEASONAL_AMPLITUDE_C = 5.0
RIVER_PHASE_LAG_HOURS = 500          # ~3 weeks behind air temperature
_AIR_PEAK_HOUR = 4200                # this project's convention (late June)


def wshp_gshp_cop(
    T_source_C: np.ndarray,
    T_sink_C,
    carnot_efficiency_fraction: float = DEFAULT_CARNOT_FRACTION,
    cop_floor: float = COP_FLOOR,
    cop_ceiling: float = COP_CEILING,
) -> np.ndarray:
    """Water/ground-source heat pump COP, Carnot-fraction method."""
    source_K = np.asarray(T_source_C, dtype=float) + 273.15
    sink_K = np.broadcast_to(T_sink_C, np.shape(source_K)).astype(float) + 273.15
    lift_K = sink_K - source_K
    if np.any(lift_K <= 0):
        raise ValueError(
            "Sink temperature must exceed source temperature everywhere — a heat pump "
            "only makes sense for a genuine lift. Check T_source_C/T_sink_C."
        )
    return np.clip(sink_K / lift_K * float(carnot_efficiency_fraction), cop_floor, cop_ceiling)


def river_source_temp_C(
    annual_mean_C: float,
    amplitude_C: float = RIVER_SEASONAL_AMPLITUDE_C,
    lag_hours: int = RIVER_PHASE_LAG_HOURS,
    n_hours: int = N_HOURS,
) -> np.ndarray:
    """Seasonal river temperature — damped and lagged against air.

    A shape assumption (see module docstring), not gauged data. The winter
    minimum is what sets the design-day COP, so replace this with real river
    data before a project decision.
    """
    h = np.arange(n_hours, dtype=float)
    temps = annual_mean_C + amplitude_C * np.cos(
        2 * np.pi * (h - _AIR_PEAK_HOUR - lag_hours) / n_hours
    )
    return np.maximum(temps, MIN_RIVER_TEMP_C)


WSHP_PRESETS = {
    # Birmingham Central, report Table 5: "WSHP River Rea, 5,000 kWp, 10 C".
    "birmingham_river_rea": {
        "description": "River Rea water-source heat pump (DESNZ Birmingham zoning report, Table 5)",
        "n_units": 2,
        "unit_capacity_MW": 2.5,          # 5,000 kWp total
        "source_annual_mean_temp_C": 10.0,
        "reference": "DESNZ Heat Network Zoning: Zone Opportunity Report — Birmingham, Feb 2025, Table 5",
    },
    "generic_river_5MW": {
        "description": "Generic UK river-source heat pump",
        "n_units": 2,
        "unit_capacity_MW": 2.5,
        "source_annual_mean_temp_C": 11.0,
        "reference": "Generic — replace with gauged river data",
    },
}

GSHP_PRESETS = {
    # Birmingham Central, report Table 5: "GSHP Aston University, 1,200 kWp, 12 C".
    "birmingham_aston_university": {
        "description": "Aston University ground-source heat pump (DESNZ Birmingham zoning report, Table 5)",
        "n_units": 2,
        "unit_capacity_MW": 0.6,          # 1,200 kWp total
        "source_temp_C": 12.0,
        "reference": "DESNZ Heat Network Zoning: Zone Opportunity Report — Birmingham, Feb 2025, Table 5",
    },
    # Report Table 5 lists this GSHP with capacity "Unknown". Left out of the
    # scenarios rather than guessed; the preset exists so a real figure can be
    # dropped in when one is available.
    "birmingham_childrens_hospital": {
        "description": "Birmingham Children's Hospital GSHP — capacity NOT stated in the report",
        "n_units": 1,
        "unit_capacity_MW": 1.0,          # PLACEHOLDER — report says "Unknown"
        "source_temp_C": 12.0,
        "reference": "DESNZ Birmingham zoning report, Table 5 — capacity Unknown, this is a placeholder",
    },
    "generic_borehole_2MW": {
        "description": "Generic UK closed-loop borehole ground-source heat pump",
        "n_units": 2,
        "unit_capacity_MW": 1.0,
        "source_temp_C": 11.5,
        "reference": "Generic — UK annual mean ground temperature",
    },
}


class _WaterGroundSourceHeatPump:
    """Shared implementation. Use WaterSourceHeatPump / GroundSourceHeatPump."""

    source_type = "wshp"

    def __init__(
        self,
        name: str,
        n_units: int,
        unit_capacity_MW: float,
        source_temp_C_hourly: np.ndarray,
        sink_temp_C,
        carnot_efficiency_fraction: float = DEFAULT_CARNOT_FRACTION,
        electricity_price_GBP_per_MWh=None,
        capex_GBP_per_MW: float = 800_000.0,
        availability_factor: float = 0.97,
        seed: int = 17,
        reference: str = "",
    ):
        source_temp_C_hourly = np.asarray(source_temp_C_hourly, dtype=float)
        if len(source_temp_C_hourly) != N_HOURS:
            raise ValueError(
                f"source_temp_C_hourly must have {N_HOURS} entries; got {len(source_temp_C_hourly)}."
            )
        if n_units <= 0:
            raise ValueError(f"n_units must be positive; got {n_units}")
        if unit_capacity_MW <= 0:
            raise ValueError(f"unit_capacity_MW must be positive; got {unit_capacity_MW}")

        self.name = name
        self.n_units = int(n_units)
        self.unit_capacity_MW = float(unit_capacity_MW)
        self.capacity_MW = self.n_units * self.unit_capacity_MW
        self.source_temp_C = source_temp_C_hourly
        self.sink_temp_C = np.broadcast_to(sink_temp_C, N_HOURS).astype(float).copy()
        self.carnot_efficiency_fraction = float(carnot_efficiency_fraction)
        self.capex_GBP_per_MW = float(capex_GBP_per_MW)
        self.availability_factor = float(availability_factor)
        self.seed = int(seed)
        self.reference = reference

        self.cop_hourly = wshp_gshp_cop(
            self.source_temp_C, self.sink_temp_C,
            carnot_efficiency_fraction=self.carnot_efficiency_fraction,
        )

        # Maintenance outages, reusing ASHP.py's real per-unit model. There is NO
        # weather-driven capacity derate: unlike an ASHP, the source does not
        # weaken on a cold day — which is the entire point of these machines.
        self.units_available = _ashp_unit_outage_profile(
            self.n_units, self.availability_factor, seed=self.seed
        )
        self._unit_availability_fraction = self.units_available / self.n_units
        self.supply_MW = self.capacity_MW * self._unit_availability_fraction
        self.supply_temp_C = self.sink_temp_C.copy()

        price = resolve_electricity_price(electricity_price_GBP_per_MWh)
        self.electricity_price_GBP_per_MWh = price
        self.marginal_cost = price / self.cop_hourly
        self.electrical_demand_MW = self.supply_MW / self.cop_hourly
        self.carbon_intensity_kgCO2_per_kWh = CARBON_INTENSITY["electric"] / self.cop_hourly

    def summary(self) -> dict:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "capacity_MW": round(self.capacity_MW, 3),
            "n_units": self.n_units,
            "mean_source_temp_C": round(float(self.source_temp_C.mean()), 2),
            "min_source_temp_C": round(float(self.source_temp_C.min()), 2),
            "mean_COP": round(float(self.cop_hourly.mean()), 3),
            "min_COP": round(float(self.cop_hourly.min()), 3),
            "carnot_efficiency_fraction": self.carnot_efficiency_fraction,
            "estimated_capex_GBP": round(self.capacity_MW * self.capex_GBP_per_MW, 0),
            "reference": self.reference,
        }

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.name}, {self.capacity_MW:.2f} MW, "
            f"mean COP={self.cop_hourly.mean():.2f})"
        )


class WaterSourceHeatPump(_WaterGroundSourceHeatPump):
    """River/sea/reservoir-source heat pump. Source follows the seasons, damped."""

    source_type = "wshp"

    @classmethod
    def from_preset(cls, preset_key: str, sink_temp_C, weather_df=None, **overrides):
        if preset_key not in WSHP_PRESETS:
            raise ValueError(
                f"Unknown WSHP preset '{preset_key}'. Available: {list(WSHP_PRESETS.keys())}"
            )
        params = dict(WSHP_PRESETS[preset_key])
        params.pop("description", None)
        params.setdefault("name", preset_key.replace("_", " ").title())
        params.update(overrides)
        mean_C = float(params.pop("source_annual_mean_temp_C"))
        amplitude = float(params.pop("source_amplitude_C", RIVER_SEASONAL_AMPLITUDE_C))
        source = params.pop(
            "source_temp_C_hourly", river_source_temp_C(mean_C, amplitude_C=amplitude)
        )
        return cls(source_temp_C_hourly=source, sink_temp_C=sink_temp_C, **params)


class GroundSourceHeatPump(_WaterGroundSourceHeatPump):
    """Closed-loop borehole heat pump. Source is constant at the annual mean."""

    source_type = "gshp"

    @classmethod
    def from_preset(cls, preset_key: str, sink_temp_C, weather_df=None, **overrides):
        if preset_key not in GSHP_PRESETS:
            raise ValueError(
                f"Unknown GSHP preset '{preset_key}'. Available: {list(GSHP_PRESETS.keys())}"
            )
        params = dict(GSHP_PRESETS[preset_key])
        params.pop("description", None)
        params.setdefault("name", preset_key.replace("_", " ").title())
        params.update(overrides)
        # Constant: a borehole below ~15m does not swing seasonally. This is NOT
        # topology_thermal.seasonal_ground_temp_C() (the 1m pipe-burial curve).
        source_C = float(params.pop("source_temp_C"))
        source = params.pop("source_temp_C_hourly", np.full(N_HOURS, source_C))
        return cls(source_temp_C_hourly=source, sink_temp_C=sink_temp_C, **params)
