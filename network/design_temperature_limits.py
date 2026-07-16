"""Cited boundary conditions for heat-network design temperatures.

Why this module exists
----------------------
`topology_thermal.py` carried a single constant, `MIN_DELIVERED_TEMP_C = 60.0`,
with a comment that fused two different rules together:

    "even though some guidance allows 50C where no DHW cylinder is present, HIU
     pressure-drop/heat-exchanger resistance means the actual temperature
     reaching taps is meaningfully below the network-side figure in practice, so
     60C is used as the uniform minimum regardless of cylinder presence -- this
     is the standard headline figure cited for Legionella control (CIBSE/HSE
     guidance: hot water should be stored/distributed at >=60C)"

Two separate things are being run together there. The **60C figure is the
STORAGE rule** (HSG274: a cylinder must be stored at >=60C, because legionella
needs stagnant bulk volume to proliferate). The **heat-exchanger approach
argument is the real reason** a network-side minimum exists at all. They land
near the same number, so the model got a defensible answer via the wrong rule --
and, critically, a rule that does not move when the system type does. An
instantaneous HIU and a stored cylinder are different systems with different
floors, and the old constant treated them identically.

The constraint is a chain, not a number
----------------------------------------
    tap outlet  ->  DHW heat exchanger approach  ->  delivered at building
                                                 ->  route heat loss  ->  source flow

This module encodes each link with its own citation, so a sensitivity sweep can
be bounded by what the standards actually permit rather than by a single
inherited constant.

Sources
-------
CIBSE/ADE CP1 Heat Networks: Code of Practice for the UK (2020)
  - Maximum flow temperature 70C on NEW schemes; the Code's direction of travel
    is explicitly downward, toward 4th-generation networks.
  - Permits network flow temperatures as low as 55C.
  - Best practice for indirect HIUs: tested volume-weighted average return
    temperature (VWART) below 33C, per the BESA UK HIU Test Regime.
  cibsejournal.com/technical/ten-big-changes-in-cp1-heat-network-code-of-practice/

CIBSE Guidance Note, August 2021: "Domestic hot-water temperatures from
instantaneous heat interface units"
  - Recommends a 50C target for DHW generated at the HIU.
  - The HSE confirmed that HIUs delivering instantaneous hot water are 'low
    risk' systems, required to deliver a minimum of 50C at the outlet, or
    immediately upstream of any TMV.
  - Its stated purpose is to enable heat networks to be designed at 60C rather
    than 80C or 90C, where instantaneous HIUs are used.
  cibse.org/knowledge-research/knowledge-portal/guidance-note-domestic-hot-water-
    temperatures-from-instantaneous-heat-interface-units/

HSE HSG274 Part 2, "Control of legionella bacteria in hot and cold water systems"
  - Stored hot water at >=60C; distributed so it reaches outlets at >=50C
    (55C in healthcare premises).
  - Stored systems additionally require the store to reach 60C for one hour
    daily. This requirement does not apply to instantaneous systems with no
    stored volume.
  hse.gov.uk/pUbns/priced/hsg274part2.pdf

CIBSE, "Integrating heat pumps in heat networks" (2020) -- the document is
titled 60/30, which is the flow/return target for a heat-pump-led network.
  cibse.org/media/ktdjv2qb/integrating-heat-pumps-in-heat-networks-02-11-2020.pdf

Third-generation context (ScienceDirect, "Third Generation District Heating"):
typical 3rd-gen networks run 70-130C flow / 40-70C return. Design radiator
temperatures are 70/40 in Denmark and Finland, 80/60 in Germany, and 82/70 in
the UK. A 70/40 network is therefore textbook 3rd-generation and matches
Danish/Finnish practice -- but it sits at CP1's ceiling, not below it.

What is NOT sourced here
-------------------------
DHW_HX_APPROACH_K is an engineering estimate, not a cited figure. A typical
brazed-plate DHW heat exchanger in an HIU needs a few kelvin of approach to
transfer its design duty; 5K is a common rule-of-thumb design allowance.
Replace it with the actual HIU manufacturer's datasheet before this is used for
anything beyond screening -- it directly sets the delivered-temperature floor,
so it is load-bearing, and a real HIU's approach varies with duty and flow.
"""
from __future__ import annotations

# ── CP1 2020 network flow-temperature envelope ────────────────────────────────
CP1_MAX_FLOW_TEMP_NEW_SCHEME_C = 70.0
CP1_MIN_PERMITTED_FLOW_TEMP_C = 55.0

# ── CP1 2020 return-temperature best practice ─────────────────────────────────
# Volume-weighted average return temperature, per the BESA UK HIU Test Regime.
# CP1 2020 put most of its new emphasis here: a low return temperature cuts peak
# flow rate, which cuts pipe size, heat loss and pumping energy together.
CP1_BEST_PRACTICE_VWART_C = 33.0

# ── CIBSE heat-pump-led target ────────────────────────────────────────────────
CIBSE_HEAT_PUMP_TARGET_FLOW_C = 60.0
CIBSE_HEAT_PUMP_TARGET_RETURN_C = 30.0

# ── DHW outlet requirements ───────────────────────────────────────────────────
# Instantaneous HIU: HSE 'low risk', 50C at the outlet / upstream of any TMV.
DHW_OUTLET_MIN_INSTANTANEOUS_C = 50.0
# Stored: HSG274 storage temperature.
DHW_STORED_MIN_C = 60.0
# Healthcare distribution figure, retained for reference.
DHW_DISTRIBUTION_MIN_HEALTHCARE_C = 55.0

# ── Heat-exchanger approach (ESTIMATE — see module docstring) ─────────────────
DHW_HX_APPROACH_K = 5.0

# ── Derived delivered-temperature floors ──────────────────────────────────────
# The temperature that must ARRIVE at the building, i.e. at the HIU's primary
# inlet, after real route heat loss.
#
# Instantaneous: 50C outlet + 5K approach = 55C delivered. This lands exactly on
# CP1's own permitted minimum flow temperature of 55C, which is a useful
# consistency check on both figures rather than a coincidence -- CP1's floor is
# set by the same DHW physics.
#
# Stored: 60C stored + 5K coil approach = 65C delivered. Note this is HIGHER
# than the old uniform 60C constant. The single inherited figure was therefore
# wrong in BOTH directions: too conservative for instantaneous systems (costing
# real ASHP COP for no legionella benefit) and too permissive for stored ones
# (a 60C network cannot charge a 60C cylinder through a heat exchanger; that
# needs supplementary input, which the CIBSE guidance note says explicitly).
MIN_DELIVERED_TEMP_INSTANTANEOUS_C = DHW_OUTLET_MIN_INSTANTANEOUS_C + DHW_HX_APPROACH_K   # 55.0
MIN_DELIVERED_TEMP_STORED_C = DHW_STORED_MIN_C + DHW_HX_APPROACH_K                        # 65.0

# Space heating only, no DHW on the network at all. CP1's permitted floor
# governs; there is no legionella pathway without domestic hot water.
MIN_DELIVERED_TEMP_SPACE_HEATING_ONLY_C = CP1_MIN_PERMITTED_FLOW_TEMP_C                   # 55.0

DHW_SYSTEM_TYPES = {"instantaneous_hiu", "stored_cylinder", "space_heating_only"}

_FLOOR_BY_TYPE = {
    "instantaneous_hiu": MIN_DELIVERED_TEMP_INSTANTANEOUS_C,
    "stored_cylinder": MIN_DELIVERED_TEMP_STORED_C,
    "space_heating_only": MIN_DELIVERED_TEMP_SPACE_HEATING_ONLY_C,
}

_BASIS_BY_TYPE = {
    "instantaneous_hiu": (
        "50C at the outlet (CIBSE GN 2021; HSE 'low risk', no stored volume) "
        f"+ {DHW_HX_APPROACH_K:.0f}K HIU heat-exchanger approach"
    ),
    "stored_cylinder": (
        "60C stored (HSG274 Part 2, plus the daily 60C/1h disinfection) "
        f"+ {DHW_HX_APPROACH_K:.0f}K cylinder coil approach"
    ),
    "space_heating_only": (
        "no DHW on the network, so no legionella pathway; CP1 2020's permitted "
        "minimum network flow temperature governs"
    ),
}


def minimum_delivered_temp_C(dhw_system: str = "instantaneous_hiu") -> float:
    """The temperature that must arrive at a building, by DHW system type."""
    if dhw_system not in DHW_SYSTEM_TYPES:
        raise ValueError(
            f"dhw_system must be one of {sorted(DHW_SYSTEM_TYPES)}; got {dhw_system!r}. "
            "This choice sets the delivered-temperature floor and therefore the "
            "achievable flow temperature and heat-pump COP — it is not a detail."
        )
    return _FLOOR_BY_TYPE[dhw_system]


def delivered_temp_basis(dhw_system: str = "instantaneous_hiu") -> str:
    """One-line, citable statement of WHY that floor applies. For audit output."""
    if dhw_system not in DHW_SYSTEM_TYPES:
        raise ValueError(f"dhw_system must be one of {sorted(DHW_SYSTEM_TYPES)}; got {dhw_system!r}")
    return _BASIS_BY_TYPE[dhw_system]


def check_flow_temp_against_cp1(flow_temp_C: float) -> dict:
    """Where a proposed flow temperature sits inside CP1 2020's envelope."""
    return {
        "flow_temp_C": float(flow_temp_C),
        "cp1_min_C": CP1_MIN_PERMITTED_FLOW_TEMP_C,
        "cp1_max_new_scheme_C": CP1_MAX_FLOW_TEMP_NEW_SCHEME_C,
        "within_cp1_envelope": (
            CP1_MIN_PERMITTED_FLOW_TEMP_C <= float(flow_temp_C) <= CP1_MAX_FLOW_TEMP_NEW_SCHEME_C
        ),
        "at_cp1_ceiling": float(flow_temp_C) >= CP1_MAX_FLOW_TEMP_NEW_SCHEME_C,
    }


def check_return_temp_against_cp1(return_temp_C: float) -> dict:
    """Where a proposed return temperature sits against CP1's VWART best practice."""
    return {
        "return_temp_C": float(return_temp_C),
        "cp1_best_practice_vwart_C": CP1_BEST_PRACTICE_VWART_C,
        "meets_best_practice": float(return_temp_C) <= CP1_BEST_PRACTICE_VWART_C,
        "excess_over_best_practice_K": max(0.0, float(return_temp_C) - CP1_BEST_PRACTICE_VWART_C),
    }
