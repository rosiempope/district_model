"""
network.py
============
"these defaults assume cooling-capable building-side emitters (fan coils/chilled ceilings); connecting radiator-only buildings would need either retrofit (uncosted) or a much higher network temperature (~16-18°C) with reduced capacity."

Sizes and costs the district network itself, for each topology variant
this project is working through in stages:

    1. 2-pipe, heating only       -- size_heating_network()
    2. 4-pipe, heating + cooling  -- size_4pipe_network()
    3. Ambient loop (5GDH)        -- NOT YET IMPLEMENTED, see note below

Uses peak duty figures from profiles/demand_synthesis.py and the pipe
sizing/cost physics from network/pipe_catalog.py — this file is the
piece that was missing to turn those two into an actual network CAPEX
number for a given scenario.

Deliberately simplified: this models the network as ONE representative
trunk pipe per duty (heating, and optionally cooling), carrying the
full site peak over an assumed total network length — not a real
routed multi-segment topology (no node positions, no branching, no
per-segment peak diversity). That's the right level of fidelity for
feasibility-stage scenario comparison (heating-only vs 4-pipe,
sensitivity to network length/linear heat density) — a real routed
topology with actual node positions and branch-level sizing is a
bigger, separate piece of future work, not something this stage needs.

Why ambient loop isn't here yet
-----------------------------------
A 2-pipe or 4-pipe network has a fixed flow direction (energy centre ->
buildings) and fixed design temperatures, which is what
size_pipe_for_peak() assumes. An ambient loop is fundamentally
different: flow direction and even which buildings are net heat
sources vs sinks can change hour-by-hour (a building rejecting heat one
hour might be drawing it the next), with building-level heat pumps
doing the actual temperature lift/drop. Sizing that needs a genuinely
different model — a time-varying bidirectional flow allocation, not a
single fixed-direction trunk pipe — which is exactly why this was
deliberately scoped out earlier in this project rather than bolted on
here as a wrong simplification. See ambient_loop_NOT_IMPLEMENTED() below
for a clear placeholder marking where that work will go.

Usage
-----
    from network.network import size_heating_network, size_4pipe_network

    # Stage 1: 2-pipe heating only
    heating_only = size_heating_network(
        peak_heat_kW=net["peak_heat_kW"],
        network_length_m=3000,
        flow_temp_C=70.0, return_temp_C=40.0,
    )
    print(heating_only.summary())

    # Stage 2: 4-pipe heating + cooling, same route length
    combined = size_4pipe_network(
        peak_heat_kW=net["peak_heat_kW"], peak_cool_kW=net["peak_cool_kW"],
        network_length_m=3000,
        heat_flow_temp_C=70.0, heat_return_temp_C=40.0,
        cool_flow_temp_C=6.0,  cool_return_temp_C=12.0,
    )
    print(combined.summary())

    # Direct heating-only vs 4-pipe CAPEX comparison
    print(f"4-pipe adds £{combined.total_capex_GBP - heating_only.total_capex_GBP:,.0f}/m... "
          f"wait, that's total, not per-m -- see total_capex_GBP on each NetworkScenario")
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd


def _find_project_root(start: Path) -> Path:
    """
    Walk upward from this file's location until we find a directory that
    looks like the project root (has both 'profiles' and 'components'
    subfolders) — works regardless of where this file ends up living.
    """
    current = start
    for _ in range(5):
        if (current / "profiles").is_dir() and (current / "components").is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        f"Could not find the district_model project root starting from {start}."
    )


_PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import pipe_catalog directly (NOT via "network.pipe_catalog") -- this
# file lives in a folder also called "network", so running it directly
# as a script (python3 network/network.py) makes Python treat network/
# as the script's home directory, which collides with absolute-importing
# a "network" package. Adding this file's own directory to sys.path and
# importing the sibling module by its bare name sidesteps that collision
# entirely, and works the same whether this file is run directly or
# imported normally as network.network from elsewhere.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pipe_catalog import size_pipe_for_peak, PipeSpec


# ── Constants ──────────────────────────────────────────────────────────────────

N_HOURS_PER_YEAR = 8760

# Typical UK undisturbed ground temperature at pipe-laying depth — used
# for the heat-loss calculation (loss is driven by pipe-to-ground delta
# T, not pipe-to-air). Set to 11.5°C, the real Thames Valley/London
# annual average at ~1m depth (Busby 2015, 106 UK Met Office soil
# stations — see network_topology.py's GROUND_TEMP_MEAN_C for the full
# sourcing note and the real SEASONAL curve this module's single fixed
# value is a simplification of). Matches network_topology.py's
# DEFAULT_GROUND_TEMP_C exactly, so both modules agree on the same
# real figure rather than two different placeholder numbers.
DEFAULT_GROUND_TEMP_C = 11.5


# ── Result types ─────────────────────────────────────────────────────────────────

@dataclass
class NetworkCostResult:
    """Cost and loss result for ONE pipe duty (e.g. just heating, or just cooling)."""
    duty_name: str
    pipe: PipeSpec
    network_length_m: float
    total_capex_GBP: float
    annual_heat_loss_MWh: float

    def __repr__(self):
        return (
            f"NetworkCostResult({self.duty_name}, DN{self.pipe.DN}, "
            f"length={self.network_length_m:.0f}m, "
            f"capex=£{self.total_capex_GBP:,.0f}, "
            f"loss={self.annual_heat_loss_MWh:.0f} MWh/yr)"
        )


@dataclass
class NetworkScenario:
    """
    A full network scenario — one or more pipe duties (e.g. heating
    alone for a 2-pipe network, or heating+cooling together for 4-pipe)
    sharing the same route/length. This is what makes the heating-only
    vs 4-pipe comparison a direct one: build both as a NetworkScenario,
    compare .total_capex_GBP.
    """
    name: str
    duties: list   # list[NetworkCostResult]

    @property
    def total_capex_GBP(self) -> float:
        return sum(d.total_capex_GBP for d in self.duties)

    @property
    def total_annual_heat_loss_MWh(self) -> float:
        return sum(d.annual_heat_loss_MWh for d in self.duties)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "total_capex_GBP": round(self.total_capex_GBP, 0),
            "total_annual_heat_loss_MWh": round(self.total_annual_heat_loss_MWh, 1),
            "by_duty": {
                d.duty_name: {
                    "DN": d.pipe.DN,
                    "construction": d.pipe.construction,
                    "capex_GBP": round(d.total_capex_GBP, 0),
                    "heat_loss_MWh": round(d.annual_heat_loss_MWh, 1),
                }
                for d in self.duties
            },
        }

    def __repr__(self):
        return (
            f"NetworkScenario('{self.name}', capex=£{self.total_capex_GBP:,.0f}, "
            f"duties={[d.duty_name for d in self.duties]})"
        )


# ── Core sizing ──────────────────────────────────────────────────────────────────

def _heat_loss_MWh_per_year(
    pipe: PipeSpec, network_length_m: float, mean_pipe_temp_C: float, ground_temp_C: float,
) -> float:
    """
    Annual heat loss (MWh): loss coefficient (W/m.K, ALREADY includes
    both supply+return — see pipe_catalog.py) x length x mean pipe-to-
    ground delta T x hours/year, converted W -> MWh. The sign of
    (mean_pipe_temp_C - ground_temp_C) handles cold loops correctly —
    a chilled pipe colder than the ground genuinely gains heat from it,
    which is a real loss for a cooling network too.
    """
    loss_W = pipe.heat_loss_coefficient_W_per_mK * network_length_m * (mean_pipe_temp_C - ground_temp_C)
    return loss_W * N_HOURS_PER_YEAR / 1e6


def size_pipe_duty(
    peak_kW: float,
    network_length_m: float,
    flow_temp_C: float,
    return_temp_C: float,
    duty_name: str = "duty",
    ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    **pipe_kwargs,
) -> NetworkCostResult:
    """
    Size and cost ONE pipe duty as a single representative trunk
    carrying the full peak over network_length_m. See module docstring
    for the single-trunk simplification this relies on.

    **pipe_kwargs passed through to size_pipe_for_peak() (construction,
    insulation_series, max_velocity_ms, etc.)
    """
    pipe = size_pipe_for_peak(peak_kW, flow_temp_C, return_temp_C, **pipe_kwargs)
    total_capex_GBP = pipe.cost_GBP_per_m * network_length_m
    mean_pipe_temp_C = (flow_temp_C + return_temp_C) / 2.0
    annual_heat_loss_MWh = _heat_loss_MWh_per_year(pipe, network_length_m, mean_pipe_temp_C, ground_temp_C)

    return NetworkCostResult(
        duty_name=duty_name,
        pipe=pipe,
        network_length_m=network_length_m,
        total_capex_GBP=total_capex_GBP,
        annual_heat_loss_MWh=annual_heat_loss_MWh,
    )


# ── Stage 1: 2-pipe, heating only ───────────────────────────────────────────────

def size_heating_network(
    peak_heat_kW: float,
    network_length_m: float,
    flow_temp_C: float = 70.0,
    return_temp_C: float = 40.0,
    ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    name: str = "Heating-only (2-pipe)",
    **pipe_kwargs,
) -> NetworkScenario:
    """Size a 2-pipe heating-only network: one duty, single representative trunk."""
    heating = size_pipe_duty(
        peak_heat_kW, network_length_m, flow_temp_C, return_temp_C,
        duty_name="heating", ground_temp_C=ground_temp_C, **pipe_kwargs,
    )
    return NetworkScenario(name=name, duties=[heating])


# ── Stage 2: 4-pipe, heating + cooling ──────────────────────────────────────────

def size_4pipe_network(
    peak_heat_kW: float,
    peak_cool_kW: float,
    network_length_m: float,
    heat_flow_temp_C: float = 70.0,
    heat_return_temp_C: float = 40.0,
    cool_flow_temp_C: float = 6.0,
    cool_return_temp_C: float = 12.0,
    ground_temp_C: float = DEFAULT_GROUND_TEMP_C,
    name: str = "Heating + cooling (4-pipe)",
    heat_pipe_kwargs: Optional[dict] = None,
    cool_pipe_kwargs: Optional[dict] = None,
) -> NetworkScenario:
    """
    Size a 4-pipe heating+cooling network — TWO independent duties
    (heating flow+return, cooling flow+return) sharing the same route
    length, each sized to its own peak and design temperatures.

    heat_pipe_kwargs / cool_pipe_kwargs let you set construction
    ('single' or 'twin') and insulation_series independently per duty —
    e.g. twin for the (typically smaller) heating main, single for the
    (typically larger) cooling main, matching the real DN200 twin-pipe
    ceiling already enforced in pipe_catalog.py.
    """
    heat_pipe_kwargs = heat_pipe_kwargs or {}
    cool_pipe_kwargs = cool_pipe_kwargs or {}

    heating = size_pipe_duty(
        peak_heat_kW, network_length_m, heat_flow_temp_C, heat_return_temp_C,
        duty_name="heating", ground_temp_C=ground_temp_C, **heat_pipe_kwargs,
    )
    cooling = size_pipe_duty(
        peak_cool_kW, network_length_m, cool_flow_temp_C, cool_return_temp_C,
        duty_name="cooling", ground_temp_C=ground_temp_C, **cool_pipe_kwargs,
    )
    return NetworkScenario(name=name, duties=[heating, cooling])


# ── Stage 3: ambient loop — deliberately not implemented yet ───────────────────

def ambient_loop_NOT_IMPLEMENTED(*args, **kwargs):
    """
    Placeholder marking where ambient loop (5GDH) network sizing will
    go. Not implemented: an ambient loop needs a time-varying
    bidirectional flow allocation (buildings can be net heat sources OR
    sinks, hour by hour, with their own heat pumps doing the temperature
    lift/drop) — fundamentally different from the fixed-direction trunk
    pipe model size_pipe_for_peak() assumes. Building this properly means
    designing that bidirectional allocation first, not reusing
    size_pipe_duty() with a different temperature pair. See module
    docstring.
    """
    raise NotImplementedError(
        "Ambient loop (5GDH) network sizing needs a different model "
        "(time-varying bidirectional flow, not a fixed-direction trunk "
        "pipe) — deliberately not built yet. See this function's docstring."
    )


# ── Convenience: build straight from a demand_synthesis.py result ──────────────

def size_network_from_demand(
    network_result: dict,
    network_length_m: float,
    include_cooling: bool = False,
    **kwargs,
) -> NetworkScenario:
    """
    Build a NetworkScenario directly from a demand_synthesis.py
    synthesise_network() result, instead of pulling out peak_heat_kW /
    peak_cool_kW yourself first.
    """
    if include_cooling:
        return size_4pipe_network(
            peak_heat_kW=network_result["peak_heat_kW"],
            peak_cool_kW=network_result["peak_cool_kW"],
            network_length_m=network_length_m,
            **kwargs,
        )
    return size_heating_network(
        peak_heat_kW=network_result["peak_heat_kW"],
        network_length_m=network_length_m,
        **kwargs,
    )


# ── Sensitivity sweep: network length / linear heat density ────────────────────

def network_length_sweep(
    peak_heat_kW: float,
    annual_heat_MWh: float,
    length_values_m: list,
    flow_temp_C: float = 65.0,
    return_temp_C: float = 35.0,
    **pipe_kwargs,
) -> pd.DataFrame:
    """
    Sweep network length (equivalently, linear heat density =
    annual_heat_MWh / length) holding peak demand fixed, and report
    network CAPEX and a simple CAPEX-only £/MWh figure at each length.

    NOT a full LCOH — that needs OPEX and discounting too, which belongs
    in economics/metrics.py once built. This is just the network-CAPEX
    half of the heat-density-vs-economics question: how sharply network
    cost per unit of delivered heat rises as heat density falls.
    """
    rows = []
    for length_m in length_values_m:
        scenario = size_heating_network(peak_heat_kW, length_m, flow_temp_C, return_temp_C, **pipe_kwargs)
        heat_density = annual_heat_MWh / length_m if length_m > 0 else float("inf")
        capex_per_MWh = scenario.total_capex_GBP / annual_heat_MWh if annual_heat_MWh > 0 else float("inf")
        rows.append({
            "network_length_m": length_m,
            "linear_heat_density_MWh_per_m": round(heat_density, 3),
            "network_capex_GBP": round(scenario.total_capex_GBP, 0),
            "network_capex_GBP_per_MWh_annual": round(capex_per_MWh, 1),
            "DN": scenario.duties[0].pipe.DN,
        })
    return pd.DataFrame(rows)
