"""Where the money actually goes — CAPEX and OPEX decomposition by scenario.

The engine already computes a full CAPEX breakdown (`headline["capex_breakdown_GBP"]`)
and per-line OPEX series (`financial["investor"]["line_items"]["opex"]`), but nothing
presented them. This report does, and adds the thing the raw numbers don't show on
their own: **how each line scales**.

Why the scaling basis matters more than the totals
---------------------------------------------------
The screening studies keep landing on the same finding — schemes fail on fixed cost per
connection, not on route length. That finding is only legible if you can see which lines
are size-independent. So every line here is tagged:

    fixed          — same £ whether the scheme has 3 connections or 3,000
    per-connection  — scales with connection count
    plant           — scales with installed MW
    network         — scales with route length and peak
    % adder        — a percentage of some other subtotal

and the report reports the **size-independent burden**: the discounted lifetime cost that
does not move with scheme size, expressed per connection and per MWh. That number is the
one that decides these schemes.

Run
---
    python -m reports.cost_breakdown

Writes to output/cost_breakdown/:
    capex_breakdown.csv        every CAPEX line, every scenario, with scaling basis
    opex_breakdown.csv         every OPEX line, every scenario, with scaling basis
    unit_costs.csv             £/connection, £/MWh, £/m by scenario
    fixed_cost_exposure.csv    size-independent burden by scenario
    cost_breakdown.md          readable summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from economics.cashflow import discount_factors
from scenarios.scenario_runner import run_scenario
from scenarios.worked_scenarios import WORKED_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "cost_breakdown"

# How each CAPEX line scales with scheme size. Keys match the runner's
# project_capex_items. Anything not listed is reported as "unclassified" rather
# than silently assumed fixed.
CAPEX_BASIS = {
    "sources_GBP":                 ("plant",          "Installed MW of generating plant"),
    "network_GBP":                 ("network",        "Route length x sized pipe diameter"),
    "thermal_storage_GBP":         ("plant",          "Store capacity (MWh)"),
    "customer_connections_GBP":    ("per-connection", "Connection count x £/connection"),
    "metering_GBP":                ("per-connection", "Connection count x £/meter"),
    "energy_centre_building_GBP":  ("fixed",          "Entered as a flat £ — does not scale"),
    "land_and_enabling_GBP":       ("fixed",          "Entered as a flat £ — does not scale"),
    "electricity_connection_GBP":  ("fixed",          "Entered as a flat £ — does not scale"),
    "gas_connection_GBP":          ("fixed",          "Entered as a flat £ — does not scale"),
    "controls_and_scada_GBP":      ("fixed",          "Entered as a flat £ — does not scale"),
    "development_and_design_GBP":  ("% adder",        "% of plant + network ONLY (see note)"),
    "commissioning_GBP":           ("% adder",        "% of plant + network ONLY (see note)"),
    "contingency_GBP":             ("% adder",        "% of plant + network ONLY (see note)"),
}

# How each OPEX line scales. Energy lines are named "<carrier> energy" by the runner.
OPEX_BASIS = {
    "electricity energy":            ("variable", "Hourly dispatch x hourly tariff"),
    "gas energy":                    ("variable", "Hourly dispatch x gas price"),
    "third_party_heat energy":       ("variable", "Hourly dispatch x purchase price"),
    "technology and network O&M":    ("plant",    "Per-technology % of that asset's own CAPEX"),
    "billing_and_customer_service_GBP": ("fixed", "Flat £/yr — arguably should be per-connection"),
    "insurance_and_rates_GBP":       ("fixed",    "Flat £/yr — does not scale"),
    "land_lease_GBP":                ("fixed",    "Flat £/yr — does not scale"),
    "water_treatment_GBP":           ("fixed",    "Flat £/yr — does not scale"),
    "operator_overhead_GBP":         ("fixed",    "Flat £/yr — does not scale"),
}

SIZE_INDEPENDENT_CAPEX = {"fixed"}
SIZE_INDEPENDENT_OPEX = {"fixed"}


def _connections(scenario: dict) -> int:
    total = 0
    for b in scenario["demand"]["buildings"]:
        explicit = b.get("connections")
        if explicit is not None:
            total += max(0, int(explicit))
        elif b.get("type") in {"residential", "residential_existing"} and b.get("units"):
            total += max(1, int(b["units"]))
        else:
            total += 1
    return total


def capex_rows(result: dict) -> list[dict]:
    h = result["headline"]
    name = result["scenario_name"]
    total = float(h["capex_total_GBP"])
    rows = []
    for item, value in h["capex_breakdown_GBP"].items():
        basis, note = CAPEX_BASIS.get(item, ("unclassified", "Not classified by this report"))
        rows.append({
            "scenario": name,
            "capex_item": item.replace("_GBP", "").replace("_", " "),
            "GBP": round(float(value), 0),
            "pct_of_capex": round(float(value) / total * 100.0, 2) if total else 0.0,
            "scaling_basis": basis,
            "note": note,
        })
    return sorted(rows, key=lambda r: -r["GBP"])


def opex_rows(result: dict) -> list[dict]:
    name = result["scenario_name"]
    items = result["financial"]["investor"]["line_items"]["opex"]
    # Year 1 is the first operating year; use it as the representative year.
    year1 = {k: float(v[1]) for k, v in items.items()}
    total = sum(year1.values())
    rows = []
    for item, value in year1.items():
        basis, note = OPEX_BASIS.get(item, ("unclassified", "Not classified by this report"))
        rows.append({
            "scenario": name,
            "opex_item": item.replace("_GBP", "").replace("_", " "),
            "GBP_per_year": round(value, 0),
            "pct_of_opex": round(value / total * 100.0, 2) if total else 0.0,
            "scaling_basis": basis,
            "note": note,
        })
    return sorted(rows, key=lambda r: -r["GBP_per_year"])


def unit_costs(result: dict, connections: int) -> dict:
    h = result["headline"]
    inv = result["financial"]["investor"]
    heat_MWh = float(h["annual_heat_demand_MWh"]) + float(h["annual_cooling_demand_MWh"])
    route_m = float(h.get("network_total_length_m") or 0.0)
    capex = float(h["capex_total_GBP"])
    opex = float(h["annual_total_opex_GBP"])
    return {
        "scenario": result["scenario_name"],
        "connections": connections,
        "annual_service_MWh": round(heat_MWh, 0),
        "route_m": round(route_m, 0),
        "capex_total_GBP": round(capex, 0),
        "capex_per_connection_GBP": round(capex / connections, 0) if connections else None,
        "capex_per_MWh_yr_GBP": round(capex / heat_MWh, 0) if heat_MWh else None,
        "capex_per_route_m_GBP": round(capex / route_m, 0) if route_m else None,
        "opex_total_GBP_per_yr": round(opex, 0),
        "opex_per_connection_GBP_yr": round(opex / connections, 0) if connections else None,
        "required_tariff_p_per_kWh": inv.get("required_heat_tariff_p_per_kWh_for_zero_NPV"),
        "equivalent_tariff_p_per_kWh": inv.get("equivalent_year1_heat_tariff_p_per_kWh"),
        "npv_GBP": inv.get("npv_GBP"),
    }


def fixed_cost_exposure(result: dict, connections: int) -> dict:
    """The discounted lifetime cost that does NOT move with scheme size.

    This is the number behind every "fails on fixed cost, not density" finding in
    the screening studies. Expressed per connection and per MWh so it can be read
    directly against the ~7-8 p/kWh a customer can actually be charged.
    """
    h = result["headline"]
    cfg = result["input"]
    inv = result["financial"]["investor"]
    life = int(cfg["economics"]["project_lifetime_years"])
    rate = float(cfg["economics"]["discount_rate"])
    factors = discount_factors(life, rate)

    fixed_capex = sum(
        float(v) for k, v in h["capex_breakdown_GBP"].items()
        if CAPEX_BASIS.get(k, ("unclassified", ""))[0] in SIZE_INDEPENDENT_CAPEX
    )
    opex_items = inv["line_items"]["opex"]
    fixed_opex_series = np.zeros(life + 1)
    for k, v in opex_items.items():
        if OPEX_BASIS.get(k, ("unclassified", ""))[0] in SIZE_INDEPENDENT_OPEX:
            fixed_opex_series += np.asarray(v, dtype=float)
    fixed_opex_discounted = float((fixed_opex_series * factors).sum())
    total_size_independent = fixed_capex + fixed_opex_discounted

    # Discounted connected heat, the same denominator the required-tariff calc uses.
    heat_MWh = float(h["annual_heat_demand_MWh"]) + float(h["annual_cooling_demand_MWh"])
    discounted_MWh = float((np.full(life + 1, heat_MWh) * factors).sum()) - heat_MWh  # exclude year 0

    return {
        "scenario": result["scenario_name"],
        "connections": connections,
        "size_independent_capex_GBP": round(fixed_capex, 0),
        "size_independent_opex_discounted_GBP": round(fixed_opex_discounted, 0),
        "size_independent_lifetime_GBP": round(total_size_independent, 0),
        "pct_of_gross_capex": round(fixed_capex / float(h["capex_total_GBP"]) * 100.0, 1),
        "size_independent_per_connection_GBP": round(total_size_independent / connections, 0) if connections else None,
        "size_independent_p_per_kWh": round(
            total_size_independent / (discounted_MWh * 1000.0) * 100.0, 2
        ) if discounted_MWh > 0 else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    capex_all, opex_all, units_all, fixed_all = [], [], [], []
    for scenario in WORKED_SCENARIOS:
        result = run_scenario(scenario)
        conns = _connections(scenario)
        capex_all += capex_rows(result)
        opex_all += opex_rows(result)
        units_all.append(unit_costs(result, conns))
        fixed_all.append(fixed_cost_exposure(result, conns))

    capex_df = pd.DataFrame(capex_all)
    opex_df = pd.DataFrame(opex_all)
    units_df = pd.DataFrame(units_all)
    fixed_df = pd.DataFrame(fixed_all)

    capex_df.to_csv(OUT / "capex_breakdown.csv", index=False)
    opex_df.to_csv(OUT / "opex_breakdown.csv", index=False)
    units_df.to_csv(OUT / "unit_costs.csv", index=False)
    fixed_df.to_csv(OUT / "fixed_cost_exposure.csv", index=False)

    by_basis_capex = (
        capex_df.groupby(["scenario", "scaling_basis"])["GBP"].sum().unstack(fill_value=0.0)
    )
    by_basis_opex = (
        opex_df.groupby(["scenario", "scaling_basis"])["GBP_per_year"].sum().unstack(fill_value=0.0)
    )

    lines = [
        "# Where the money goes — CAPEX and OPEX decomposition",
        "",
        "Generated by `python -m reports.cost_breakdown`. Every line is tagged with how it",
        "scales, because the scaling basis — not the total — is what decides these schemes.",
        "",
        "## 1. CAPEX by scenario and line item",
        "",
        capex_df.to_markdown(index=False),
        "",
        "## 2. CAPEX grouped by scaling basis (£)",
        "",
        by_basis_capex.round(0).to_markdown(),
        "",
        "## 3. OPEX by scenario and line item (year 1, £/yr)",
        "",
        opex_df.to_markdown(index=False),
        "",
        "## 4. OPEX grouped by scaling basis (£/yr)",
        "",
        by_basis_opex.round(0).to_markdown(),
        "",
        "## 5. Unit costs",
        "",
        units_df.to_markdown(index=False),
        "",
        "## 6. Size-independent cost exposure",
        "",
        "Discounted lifetime cost that does **not** move with scheme size. Read the",
        "`size_independent_p_per_kWh` column against the ~7.33 p/kWh Ofgem cap and the",
        "~8.3 p/kWh modelled gas-parity bill: this is what every connection must carry",
        "*before* a single kWh of heat is generated, a metre of pipe is laid, or a",
        "connection is made.",
        "",
        fixed_df.to_markdown(index=False),
        "",
        "## Notes on method",
        "",
        "- **The % adders (design, commissioning, contingency) are applied to plant +",
        "  network only.** They exclude the energy-centre building, land, utility",
        "  connections, controls/SCADA, customer connections and metering. On the worked",
        "  scenarios that leaves roughly £9m of real cost carrying zero contingency. This",
        "  is the engine's current behaviour (`scenario_runner.py`, `base_capex`), reported",
        "  here rather than corrected, so the effect is visible before it is decided on.",
        "- Fixed CAPEX/OPEX line items in `scenarios/worked_scenarios.py` are Ealing-calibrated",
        "  (~1,100-connection scheme) and are **not rescaled** for smaller schemes. They are",
        "  entered as flat £ values, so any scenario with fewer connections carries the same",
        "  absolute burden. This overstates the NPV gap for small schemes.",
        "- OPEX is year 1. Energy lines are connection-weighted by the phased build-out;",
        "  fixed O&M and overhead remain payable in full from year 1.",
    ]
    (OUT / "cost_breakdown.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT}/")
    print("\n=== CAPEX by scaling basis (£) ===")
    print(by_basis_capex.round(0).to_string())
    print("\n=== OPEX by scaling basis (£/yr) ===")
    print(by_basis_opex.round(0).to_string())
    print("\n=== Size-independent exposure ===")
    print(fixed_df.to_string(index=False))


if __name__ == "__main__":
    main()
