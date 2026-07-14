"""Single-source-of-truth annual cash-flow calculations.

All financial metrics and chart series are derived from the same year-by-year
table.  Costs are supplied as positive numbers; the engine applies their cash
flow signs.  Year 0 is included explicitly.
"""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np


def _as_series(value, life_years: int, *, start_year: int = 1,
               escalation: float = 0.0) -> np.ndarray:
    """Resolve a scalar, sequence, or component config to years 0..life."""
    n = int(life_years) + 1
    if isinstance(value, Mapping):
        base = float(value.get("base_GBP", value.get("value", 0.0)))
        start = int(value.get("start_year", start_year))
        end = int(value.get("end_year", life_years))
        esc = float(value.get("escalation_rate", escalation))
        arr = np.zeros(n)
        for year in range(max(0, start), min(life_years, end) + 1):
            arr[year] = base * (1.0 + esc) ** max(0, year - start)
        return arr
    if np.isscalar(value):
        arr = np.zeros(n)
        years = np.arange(start_year, n)
        arr[start_year:] = float(value) * (1.0 + escalation) ** (years - start_year)
        return arr
    arr = np.asarray(value, dtype=float)
    if len(arr) != n:
        raise ValueError(f"Cash-flow series must contain {n} values (years 0..{life_years}); got {len(arr)}")
    return arr.copy()


def discount_factors(life_years: int, discount_rate: float) -> np.ndarray:
    years = np.arange(int(life_years) + 1)
    return 1.0 / (1.0 + float(discount_rate)) ** years


def npv_from_cashflows(net_cashflow: Sequence[float], discount_rate: float) -> float:
    values = np.asarray(net_cashflow, dtype=float)
    return float((values * discount_factors(len(values) - 1, discount_rate)).sum())


def irr_from_cashflows(net_cashflow: Sequence[float]) -> float | None:
    """Return the first conventional IRR root, or None if no bracket exists."""
    values = np.asarray(net_cashflow, dtype=float)
    if not (np.any(values < 0) and np.any(values > 0)):
        return None

    def value(rate):
        return npv_from_cashflows(values, rate)

    # Dense log-style search gives a safe bracket without assuming a flat
    # annuity. Multiple-sign-change cash flows remain flagged by the caller.
    grid = np.concatenate([
        np.linspace(-0.95, 0.0, 96, endpoint=False),
        np.linspace(0.0, 1.0, 201),
        np.linspace(1.05, 5.0, 80),
    ])
    vals = [value(r) for r in grid]
    for lo, hi, vlo, vhi in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if vlo == 0:
            return float(lo)
        if vlo * vhi > 0:
            continue
        for _ in range(100):
            mid = (lo + hi) / 2.0
            vmid = value(mid)
            if abs(vmid) < 0.01 or hi - lo < 1e-9:
                return float(mid)
            if vlo * vmid <= 0:
                hi, vhi = mid, vmid
            else:
                lo, vlo = mid, vmid
        return float((lo + hi) / 2.0)
    return None


def payback_from_cashflows(net_cashflow: Sequence[float], discount_rate: float = 0.0) -> float | None:
    values = np.asarray(net_cashflow, dtype=float)
    discounted = values * discount_factors(len(values) - 1, discount_rate)
    cumulative = np.cumsum(discounted)
    crossings = np.where(cumulative >= 0)[0]
    crossings = crossings[crossings > 0]
    if not len(crossings):
        return None
    year = int(crossings[0])
    previous = cumulative[year - 1]
    flow = discounted[year]
    fraction = (-previous / flow) if flow > 0 else 0.0
    return float(year - 1 + fraction)


def build_cashflow(
    *,
    life_years: int,
    discount_rate: float,
    capex: Mapping[str, object],
    revenues: Mapping[str, object],
    opex: Mapping[str, object],
    repex: Mapping[str, object] | None = None,
    grants: Mapping[str, object] | None = None,
    residual_values: Mapping[str, object] | None = None,
) -> dict:
    """Build the auditable table used by NPV, IRR, payback and charts."""
    life = int(life_years)
    if life <= 0:
        raise ValueError("life_years must be a positive integer")
    if discount_rate < 0:
        raise ValueError("discount_rate must be zero or positive")

    def resolve(items, default_start=1, one_time=False):
        result = {}
        for name, value in (items or {}).items():
            if one_time and np.isscalar(value):
                arr = np.zeros(life + 1)
                arr[default_start] = float(value)
            else:
                arr = _as_series(value, life, start_year=default_start)
            result[name] = arr
        return result

    capex_s = resolve(capex, default_start=0, one_time=True)
    revenue_s = resolve(revenues)
    opex_s = resolve(opex)
    repex_s = resolve(repex or {}, default_start=0)
    grant_s = resolve(grants or {}, default_start=0, one_time=True)
    residual_s = resolve(residual_values or {}, default_start=life, one_time=True)

    zeros = np.zeros(life + 1)
    total_capex = sum(capex_s.values(), zeros.copy())
    total_revenue = sum(revenue_s.values(), zeros.copy())
    total_opex = sum(opex_s.values(), zeros.copy())
    total_repex = sum(repex_s.values(), zeros.copy())
    total_grant = sum(grant_s.values(), zeros.copy())
    total_residual = sum(residual_s.values(), zeros.copy())
    net = total_revenue + total_grant + total_residual - total_capex - total_opex - total_repex
    factors = discount_factors(life, discount_rate)
    discounted = net * factors
    cumulative_undiscounted = np.cumsum(net)
    cumulative_discounted = np.cumsum(discounted)

    rows = []
    for year in range(life + 1):
        row = {
            "year": year,
            "capex_GBP": round(float(total_capex[year]), 2),
            "repex_GBP": round(float(total_repex[year]), 2),
            "opex_GBP": round(float(total_opex[year]), 2),
            "revenue_GBP": round(float(total_revenue[year]), 2),
            "grant_GBP": round(float(total_grant[year]), 2),
            "residual_value_GBP": round(float(total_residual[year]), 2),
            "net_cashflow_GBP": round(float(net[year]), 2),
            "discount_factor": float(factors[year]),
            "discounted_net_cashflow_GBP": round(float(discounted[year]), 2),
            "cumulative_undiscounted_GBP": round(float(cumulative_undiscounted[year]), 2),
            "cumulative_discounted_GBP": round(float(cumulative_discounted[year]), 2),
        }
        for prefix, group in (("capex", capex_s), ("revenue", revenue_s),
                              ("opex", opex_s), ("repex", repex_s)):
            for name, values in group.items():
                row[f"{prefix}:{name}"] = round(float(values[year]), 2)
        rows.append(row)

    irr_value = irr_from_cashflows(net)
    sign_changes = int(np.sum(np.signbit(net[1:]) != np.signbit(net[:-1])))
    return {
        "life_years": life,
        "discount_rate": float(discount_rate),
        "npv_GBP": round(float(cumulative_discounted[-1]), 2),
        "irr": None if irr_value is None else round(float(irr_value), 8),
        "simple_payback_years": payback_from_cashflows(net, 0.0),
        "discounted_payback_years": payback_from_cashflows(net, discount_rate),
        "cashflow_years": list(range(life + 1)),
        "net_cashflow_GBP": [round(float(x), 2) for x in net],
        "cumulative_undiscounted_GBP": [round(float(x), 2) for x in cumulative_undiscounted],
        "cumulative_discounted_GBP": [round(float(x), 2) for x in cumulative_discounted],
        "annual_table": rows,
        "line_items": {
            "capex": {k: v.tolist() for k, v in capex_s.items()},
            "revenue": {k: v.tolist() for k, v in revenue_s.items()},
            "opex": {k: v.tolist() for k, v in opex_s.items()},
            "repex": {k: v.tolist() for k, v in repex_s.items()},
        },
        "warnings": (["Cash flow has multiple sign changes; IRR may be non-unique."]
                     if sign_changes > 1 else []),
    }


def discounted_levelised_cost_GBP_per_kWh(
    *, costs_GBP: Sequence[float], delivered_kWh: Sequence[float],
    discount_rate: float,
) -> float:
    costs = np.asarray(costs_GBP, dtype=float)
    energy = np.asarray(delivered_kWh, dtype=float)
    if len(costs) != len(energy):
        raise ValueError("Cost and energy series must have the same length")
    factors = discount_factors(len(costs) - 1, discount_rate)
    denominator = float((energy * factors).sum())
    if denominator <= 0:
        raise ValueError("Discounted delivered energy must be positive")
    return float((costs * factors).sum() / denominator)
