"""Consistent, auditable pass/fail logic for early-stage screening.

The engine calculates facts; this module compares those facts with the
scenario's explicitly recorded screening criteria.  Keeping the decision in a
pure function prevents the UI and exported tables from applying different
thresholds to the same model run.
"""
from __future__ import annotations

from typing import Any


def _gate(name: str, passed: bool | None, actual: Any, threshold: Any,
          unit: str, message: str, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "threshold": threshold,
        "unit": unit,
        "required": required,
        "message": message,
    }


def evaluate_screening(result: dict[str, Any]) -> dict[str, Any]:
    """Return one decision used by the UI, CSV exports and API consumers."""
    cfg = result["input"]
    h = result["headline"]
    inv = result.get("financial", {}).get("investor", {})
    criteria = cfg.get("screening", {})
    hurdle = float(criteria.get("investor_hurdle_rate", cfg["economics"]["discount_rate"]))
    min_npv = float(criteria.get("minimum_investor_npv_GBP", 0.0))
    require_n1 = bool(criteria.get("require_n_minus_one", False))
    max_tariff = criteria.get("maximum_required_heat_tariff_p_per_kWh")

    irr = inv.get("irr")
    npv = inv.get("npv_GBP")
    req_tariff = inv.get("required_heat_tariff_p_per_kWh_for_zero_NPV")
    gates = [
        _gate(
            "Heat and cooling service", bool(h.get("service_compliant")),
            max(float(h.get("unmet_heat_fraction", 0.0)), float(h.get("unmet_cooling_fraction", 0.0))) * 100.0,
            float(criteria.get("maximum_unmet_energy_fraction", 0.001)) * 100.0, "% of annual energy",
            "Annual unmet heat and cooling must remain within the stated tolerance.",
        ),
        _gate(
            "Carbon intensity", bool(h.get("carbon_compliant")),
            float(h["carbon_intensity_kgCO2_per_kWh_service"]) * 1000.0,
            float(criteria.get("maximum_carbon_gCO2e_per_kWh", 100.0)), "gCO2e/kWh",
            "Operational heat/cooling service carbon must remain below the screening limit.",
        ),
        _gate(
            "Investor NPV", None if npv is None else float(npv) >= min_npv,
            npv, min_npv, "GBP",
            "Investor NPV includes scheme revenues, OPEX, REPEX, CAPEX and modelled grant.",
        ),
        _gate(
            "Investor IRR", None if irr is None else float(irr) >= hurdle,
            irr, hurdle, "fraction",
            "IRR is tested against the scenario hurdle rate, not a hard-coded UI value.",
        ),
        _gate(
            "Customer heat-bill parity", inv.get("customer_bill_compliant"),
            None if inv.get("year1_customer_bill_ratio") is None else inv["year1_customer_bill_ratio"] * 100.0,
            100.0, "% of individual-gas bill",
            "The district heat bill must not exceed the modelled individual-gas customer bill.",
            required=cfg["economics"].get("counterfactual") in {"individual_gas", "individual_gas_and_ac"},
        ),
        _gate(
            "Customer cooling-bill parity", inv.get("cooling_bill_compliant"),
            None if inv.get("year1_cooling_bill_ratio") is None else inv["year1_cooling_bill_ratio"] * 100.0,
            100.0, "% of individual-AC bill",
            "The district cooling bill must not exceed the modelled individual-AC customer bill.",
            required=bool(cfg["network"].get("include_cooling")),
        ),
        _gate(
            "N-1 peak capacity", bool(h.get("n_minus_one_compliant")) if require_n1 else h.get("n_minus_one_compliant"),
            h.get("n_minus_one_heat_margin_MW"), 0.0, "MW margin",
            "Peak-hour available capacity after the largest credible source/unit outage.",
            required=require_n1,
        ),
        # Delivered temperature. `passed` is None in generic_length mode (no route
        # to propagate a temperature along) and the gate is not required there —
        # "not assessed" must never read as "passed".
        _gate(
            "Delivered temperature", h.get("delivered_temp_compliant"),
            h.get("worst_case_delivered_temp_C"), h.get("minimum_delivered_temp_C"), "°C at the building",
            h.get("delivered_temp_basis") or "Heat must arrive hot enough to make domestic hot water.",
            required=cfg["network"].get("mode") == "tree",
        ),
    ]
    if max_tariff is not None:
        gates.append(_gate(
            "Required heat tariff", None if req_tariff is None else float(req_tariff) <= float(max_tariff),
            req_tariff, float(max_tariff), "p/kWh",
            "Break-even heat tariff must be no higher than the user-set commercial ceiling.",
        ))

    required_gates = [g for g in gates if g["required"]]
    failed = [g for g in required_gates if g["passed"] is not True]

    evidence_flags = []
    if cfg["network"].get("mode") == "generic_length":
        evidence_flags.append("Equivalent-trunk route; replace with a tree/GIS route before external use.")
    if any(b.get("annual_heat_kWh") is None for b in cfg["demand"]["buildings"]):
        evidence_flags.append("One or more heat demands are archetype-derived rather than measured.")
    if cfg["network"].get("include_cooling") and any(
        b.get("annual_cool_kWh") is None for b in cfg["demand"]["buildings"]
    ):
        evidence_flags.append("One or more cooling demands are archetype-derived rather than measured.")
    if not require_n1:
        evidence_flags.append("N-1 is reported but is not a mandatory gate in this scenario.")
    if cfg["network"].get("mode") != "tree":
        evidence_flags.append(
            "Delivered temperature is not assessed in generic-length mode; a real route is "
            "needed to know whether heat arrives hot enough to make domestic hot water."
        )

    if failed:
        status = "FAIL"
        summary = "Fails one or more mandatory technical, carbon or investor gates."
    elif evidence_flags:
        status = "CONDITIONAL PASS"
        summary = "Passes the selected gates, subject to the listed evidence and assurance actions."
    else:
        status = "PASS"
        summary = "Passes all selected internal screening gates."

    return {
        "status": status,
        "summary": summary,
        "hurdle_rate": hurdle,
        "gates": gates,
        "failed_gate_names": [g["name"] for g in failed],
        "evidence_flags": evidence_flags,
        "scope": "Internal early-stage screen; not an investment-grade approval.",
    }
