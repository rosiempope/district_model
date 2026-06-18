

# ── Network-level source aggregator ───────────────────────────────────────────

def build_source_stack(sources: list) -> dict:
    """
    Aggregate multiple sources into a single supply stack for the dispatcher.

    Sources are stacked in merit order (cheapest first) automatically.
    The dispatcher should call on them in this order at each hour.

    Returns
    -------
    dict with keys:
        sources_ordered   : list of sources sorted by mean marginal cost
        total_capacity_MW : sum of all source capacities
        total_supply_MW   : np.ndarray (8760,) — sum of all supply arrays
        min_cost_MW       : np.ndarray (8760,) — cheapest source capacity per hour
        stack_df          : pd.DataFrame — one row per source, summary stats
    """
    if not sources:
        raise ValueError("No sources provided to build_source_stack.")

    # Sort by mean marginal cost — cheapest (DC waste heat) first
    ordered = sorted(sources, key=lambda s: s.marginal_cost.mean())

    total_supply = sum(s.supply_MW for s in ordered)

    stack_rows = []
    for s in ordered:
        row = {
            "name":              s.name,
            "type":              s.source_type,
            "capacity_MW":       round(s.capacity_MW, 2),
            "annual_MWh":        round(s.supply_MW.sum(), 0),
            "mean_cost_GBP_MWh": round(float(s.marginal_cost.mean()), 2),
        }
        stack_rows.append(row)

    return {
        "sources_ordered":    ordered,
        "total_capacity_MW":  sum(s.capacity_MW for s in ordered),
        "total_supply_MW":    total_supply,
        "stack_df":           pd.DataFrame(stack_rows),
    }
