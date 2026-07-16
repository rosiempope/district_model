"""Scale the size-independent CAPEX/OPEX items to a scheme's actual size.

The problem this exists to solve
---------------------------------
COMMON_ECONOMICS carries five fixed CAPEX items (energy-centre building, land
and enabling, electricity connection, gas connection, controls/SCADA) and five
fixed annual OPEX overheads (billing, insurance/rates, land lease, water
treatment, operator overhead). They are flat £ figures calibrated against ONE
reference scale: the Ealing-style BASE_BUILDINGS set (564 connections, ~8.6 MW
peak heat).

Reused unscaled, they charge a three-building scheme exactly what they charge a
thousand-connection one. reports/cost_breakdown.py puts a number on what that
means: ~£4.65m of CAPEX plus ~£340k/yr of overhead, ~5.9 p/kWh of size-
independent burden against a ~7.3-8.3 p/kWh customer bill. Get the scale wrong
and you are not modelling a scheme, you are modelling Ealing's overheads bolted
onto someone else's demand.

Why this module exists rather than a function inside a study script
--------------------------------------------------------------------
This logic was written inside analysis/exeter_case_study.py, under a comment
calling it "the single source of truth every Exeter script should import this
from". It was — by the Exeter scripts. Nothing else could reach it without
importing a 723-line study module (and its matplotlib setup, and its figure
output directory), so the Dalkia archetype study didn't, and ran on flat
Ealing-calibrated fixed costs instead. output/dalkia_screening/findings.md
flags the consequence in its own words: the fixed items "hit the Scarce
archetype (321 connections) proportionally far harder than Dense (723
connections) — a real minimum-viable-scale effect, but the absolute NPV gap
for Scarce is overstated until fixed items are re-scoped for scheme size."

Moving it here makes it reachable from any scenario builder without dragging a
study's plotting stack along with it.

Scaling basis
-------------
Scaled by PEAK THERMAL CAPACITY, not annual energy: energy-centre footprint,
utility connection capacity and controls scope are driven by peak plant size,
not by annual throughput. A MIN_SCALE_FACTOR floor applies because even a small
energy centre needs some irreducible land, enabling and controls — the
relationship is sublinear at the bottom end, not proportional to zero.

This remains a screening approximation. A real project scopes these items from
a drawing, not a ratio.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pandas as pd

from profiles.demand_synthesis import synthesise_network
from scenarios.worked_scenarios import BASE_BUILDINGS, COMMON_ECONOMICS

ROOT = Path(__file__).resolve().parents[1]

# The five flat CAPEX items and five flat OPEX overheads in COMMON_ECONOMICS.
# Everything else in capex_items either scales already (per-connection lines) or
# is a percentage adder derived from the rest.
FIXED_CAPEX_KEYS = [
    "energy_centre_building_GBP", "land_and_enabling_GBP",
    "electricity_connection_GBP", "gas_connection_GBP", "controls_and_scada_GBP",
]
FIXED_OPEX_KEYS = [
    "billing_and_customer_service_GBP", "insurance_and_rates_GBP",
    "land_lease_GBP", "water_treatment_GBP", "operator_overhead_GBP",
]

# Floor: even a small energy centre needs some minimum land/enabling/controls.
MIN_SCALE_FACTOR = 0.20

# Economies of scale.
#
# This scaled LINEARLY with peak MW, which means a scheme twice the size paid
# exactly twice the energy-centre cost — i.e. no economies of scale at all. That
# is wrong, and it is why adding customers to this model barely improved the loss
# per connection: every new customer brought very nearly its own full share of
# the "fixed" cost with it. An outside review flagged exactly this, and it was
# right.
#
# Real plant and buildings scale sublinearly. The classic basis is the
# six-tenths rule (Sinnott, Chemical Engineering Design): cost ~ capacity^0.6 for
# fabricated equipment. An energy centre is not pure equipment — it is a
# building, a grid connection, a control system and plant together, some of which
# scale better than others and some barely at all.
#
# 0.7 is used: between the 0.6 of pure equipment and the 1.0 of no economies at
# all, and deliberately toward the conservative end. At 0.7 a scheme at twice the
# reference peak pays 2^0.7 = 1.62x, not 2x — a 19% saving per MW. That is the
# right order against DESNZ's HNIP analysis, which expects economies of scale
# from larger networks.
#
# HEALTH WARNING: 0.7 is an engineering convention, not a measured figure for UK
# heat-network energy centres, and no public dataset was found that pins one
# down. It is a genuine judgement call and it moves NPV. Set
# FIXED_COST_SCALE_EXPONENT = 1.0 to recover the previous linear behaviour and
# see what it is worth.
FIXED_COST_SCALE_EXPONENT = 0.70


@lru_cache(maxsize=1)
def reference_peak_MW() -> float:
    """Peak heat of the BASE_BUILDINGS set the flat figures were calibrated to.

    Computed from the real demand model rather than hard-coded, so it tracks any
    change to BASE_BUILDINGS instead of silently going stale. Cached: this runs
    a full 8,760-hour synthesis and the answer never changes within a process.
    """
    weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
    if len(weather) != 8760:
        raise ValueError(f"weather_data.csv must have 8760 rows; got {len(weather)}")
    weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(BASE_BUILDINGS)})
    return demand["peak_heat_kW"] / 1000.0


def scaled_economics(peak_total_MW: float, base_economics: dict | None = None) -> tuple[dict, float]:
    """COMMON_ECONOMICS with the fixed CAPEX/OPEX items scaled to this scheme.

    Parameters
    ----------
    peak_total_MW  : this scenario's own peak thermal capacity — heat, or
                     heat + cooling for a 4-pipe scenario.
    base_economics : economics dict to scale (defaults to COMMON_ECONOMICS).

    Returns
    -------
    (economics_dict, scale_factor). The scale factor is worth carrying into
    reporting: "this scenario's fixed costs were scaled by 1.14x the reference"
    is an auditable statement; a silently rescaled number is not.
    """
    if peak_total_MW <= 0:
        raise ValueError(f"peak_total_MW must be positive; got {peak_total_MW}")
    base = base_economics if base_economics is not None else COMMON_ECONOMICS
    # Sublinear: (this scheme / reference) ^ 0.7, not ^ 1.0. See
    # FIXED_COST_SCALE_EXPONENT — linear scaling meant no economies of scale, so
    # every extra customer carried its own full share of the "fixed" cost.
    ratio = peak_total_MW / reference_peak_MW()
    scale = max(MIN_SCALE_FACTOR, ratio ** FIXED_COST_SCALE_EXPONENT)
    econ = deepcopy(base)
    for key in FIXED_CAPEX_KEYS:
        if key in econ.get("capex_items", {}):
            econ["capex_items"][key] = round(base["capex_items"][key] * scale, 0)
    for key in FIXED_OPEX_KEYS:
        if key in econ.get("annual_opex_items", {}):
            econ["annual_opex_items"][key] = round(base["annual_opex_items"][key] * scale, 0)
    return econ, scale
