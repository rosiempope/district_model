### this is to allow for different climate scenarios to be used in the model, and to allow for different climate scenarios to be assessed to understand future trends due to global warming
"""
climate_scenarios.py
=====================
Applies climate-change deltas to a TMY weather DataFrame, to allow the
model to be stress-tested against future UK climate scenarios without
needing a separate weather file for each one.

Key assumptions
---------------
- Deltas are applied to dry-bulb temperature only — no change to
  humidity or solar weather variables.
- Seasonal deltas are based on UKCP18 projections for the UK.
- Deltas are applied to the TMY weather data, which is a historical
  representative-year dataset — the result is a stress-tested version of
  that historical year, not an independently modelled future weather year.

Urban heat island (UHI) — why it's seasonally weighted, not flat
-------------------------------------------------------------------
UHI intensity is a well-established warm-season/nighttime effect (less
vegetation and more heat-retentive urban surfaces matter most when
there's solar heat to absorb and release) — it is NOT a flat, year-round
offset. Applying it as a flat addition in every month (including deep
winter) double-counts warming on top of the already-elevated winter delta
and pushes the high scenario well past published literature for heating
degree-day decline.

Sanity check against the literature: a 2019 European study of HDD trends
under climate change (Staffell et al., arxiv.org/pdf/1907.04067) found
heating degree-days falling by ~42% under RCP8.5 by 2100 (full century of
warming) and ~24% under RCP4.5 by 2100. A 2050 (mid-century) UK estimate
should sit BELOW the 2100 figures for the same pathway. Before this fix,
this module's '2050_high' scenario (RCP8.5-equivalent) showed a 58.6%
HDD reduction by 2050 — already past the full-century RCP8.5 figure,
which isn't physically sensible. Tapering the UHI offset to zero in
winter and full strength in summer (mirroring the same seasonal shape
already used for the climate deltas themselves) brings this down to a
39% reduction — appropriately below the 42% full-century benchmark.
'2050_central' was unaffected by this issue (its UHI offset is 0°C), and
its ~21% HDD reduction by 2050 already sits sensibly below the 24%
full-century RCP4.5 figure.

This is a calibration choice, not a precise forecast — treat the UHI
magnitude (2.5°C peak, summer-weighted) as a sensitivity input you can
revisit, not gospel.
"""
import numpy as np
import pandas as pd

N_HOURS = 8760

# ── Seasonal climate deltas (°C) — UKCP18-based ──────────────────────────────
DELTAS = {
    'baseline': {m: 0.0 for m in range(1, 13)},
    '2050_central': {
        12: 1.0, 1: 1.0, 2: 1.0,   # winter
        3: 1.4,  4: 1.8, 5: 2.2,   # spring shoulder
        6: 2.7,  7: 2.7, 8: 2.7,   # summer
        9: 2.2,  10: 1.8, 11: 1.4,  # autumn shoulder
    },
    '2050_high': {
        12: 2.0, 1: 2.0, 2: 2.0,
        3: 2.5,  4: 3.0, 5: 3.5,
        6: 4.0,  7: 4.0, 8: 4.0,
        9: 3.5,  10: 3.0, 11: 2.5,
    },
}

# ── Scenario metadata — for reporting/labelling, not used in the calc ───────
SCENARIOS = {
    "baseline": {
        "period":      "TMY as-is",
        "pathway":     "n/a",
        "percentile":  "n/a",
        "location":    "Site-specific UKCP18 grid cell",
        "uhi_peak_C":  0.0,
    },
    "2050_central": {
        "period":      "2041-2060",
        "pathway":     "RCP4.5",
        "percentile":  "50th",
        "location":    "Site-specific UKCP18 grid cell",
        "uhi_peak_C":  0.0,
    },
    "2050_high": {
        "period":      "2041-2060",
        "pathway":     "RCP8.5",
        "percentile":  "50th",
        "location":    "Site-specific UKCP18 grid cell",
        "uhi_peak_C":  2.5,
    },
}


def _seasonal_uhi_offset(deltas: dict, uhi_peak_C: float) -> dict:
    """
    Build a month -> UHI offset (°C) mapping that tapers from 0 at the
    month with the SMALLEST climate delta (deep winter) up to uhi_peak_C
    at the month with the LARGEST climate delta (peak summer), following
    the same seasonal shape already used for the climate deltas
    themselves. This is what keeps UHI a warm-season effect rather than a
    flat year-round addition — see module docstring.

    Returns all-zero if uhi_peak_C is 0 (e.g. 'baseline', '2050_central').
    """
    if uhi_peak_C == 0.0:
        return {m: 0.0 for m in deltas}

    lo, hi = min(deltas.values()), max(deltas.values())
    if hi == lo:
        # No seasonal variation in the deltas to taper against (e.g. a
        # flat scenario) — fall back to applying UHI flat across months.
        return {m: uhi_peak_C for m in deltas}

    return {m: uhi_peak_C * (d - lo) / (hi - lo) for m, d in deltas.items()}


def apply_climate_scenario(weather_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Apply a climate delta (+ seasonally-weighted UHI offset where
    applicable) to a TMY weather DataFrame. Returns a modified copy —
    never mutates the original.

    Scenarios
    ---------
    'baseline'     : no change (TMY as-is)
    '2050_central' : UKCP18 RCP4.5 central estimate
                     +2.7°C summer (Jun-Aug), +1.0°C winter (Dec-Feb),
                     +1.8°C shoulder seasons — linear interpolation.
                     No UHI offset applied (uhi_peak_C = 0.0).
    '2050_high'    : UKCP18 RCP8.5 high emissions + urban heat island
                     +4.0°C summer, +2.0°C winter, +3.0°C shoulder,
                     PLUS a seasonally-weighted UHI offset peaking at
                     +2.5°C in summer and tapering to 0°C in winter
                     (see module docstring for why this is tapered
                     rather than flat).
    """
    if scenario not in DELTAS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available: {list(DELTAS.keys())}"
        )

    df = weather_df.copy()
    month = df.index.month

    deltas = DELTAS[scenario]
    uhi_peak_C = SCENARIOS.get(scenario, {}).get("uhi_peak_C", 0.0)
    uhi_by_month = _seasonal_uhi_offset(deltas, uhi_peak_C)

    delta = month.map(deltas)
    uhi = month.map(uhi_by_month)

    df['temp_drybulb_C'] = df['temp_drybulb_C'] + delta + uhi

    return df


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  climate_scenarios.py — self-test")
    print("=" * 70)

    # Synthetic London-like weather (same approach as other modules' self-tests)
    np.random.seed(42)
    hours = np.arange(N_HOURS)
    T = (
        11.5
        + 8.0 * np.cos(2 * np.pi * (hours - 4200) / N_HOURS)
        + 3.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24)
        + np.random.normal(0, 1.5, N_HOURS)
    )
    dates = pd.date_range("2023-01-01", periods=N_HOURS, freq="h")
    weather_df = pd.DataFrame({"temp_drybulb_C": T}, index=dates)

    heat_base_C = 15.5
    baseline_HDD = np.clip(heat_base_C - T, 0, None).sum()

    print(f"\n  Baseline annual HDD_h: {baseline_HDD:,.0f}")
    print(f"\n  {'Scenario':<15} {'Jan mean C':>11} {'Jul mean C':>11} "
          f"{'Annual HDD_h':>13} {'HDD reduction':>14}")
    print("  " + "-" * 70)

    results = {}
    for s in DELTAS:
        w = apply_climate_scenario(weather_df, s)
        T_s = w["temp_drybulb_C"].values
        jan_mean = T_s[:744].mean()
        jul_mean = T_s[4344:5088].mean()
        HDD_s = np.clip(heat_base_C - T_s, 0, None).sum()
        reduction_pct = (1 - HDD_s / baseline_HDD) * 100
        results[s] = HDD_s
        print(f"  {s:<15} {jan_mean:>11.2f} {jul_mean:>11.2f} "
              f"{HDD_s:>13,.0f} {reduction_pct:>13.1f}%")

    print("\n  Literature cross-check (Staffell et al. 2019, European HDD trends):")
    print("    RCP4.5 by 2100: ~24% reduction   |   RCP8.5 by 2100: ~42% reduction")
    print("    A 2050 (mid-century) estimate should sit BELOW its own pathway's")
    print("    2100 figure — check this holds for both scenarios above.")

    # --- Sanity checks ---
    print("\n  Sanity checks:")
    assert results["baseline"] == baseline_HDD, "Baseline should leave HDD unchanged"
    assert results["2050_central"] < baseline_HDD, "2050_central should reduce HDD"
    assert results["2050_high"] < results["2050_central"], \
        "2050_high should reduce HDD more than 2050_central"

    central_reduction = (1 - results["2050_central"] / baseline_HDD) * 100
    high_reduction     = (1 - results["2050_high"] / baseline_HDD) * 100
    assert central_reduction < 24.0, \
        f"2050_central ({central_reduction:.1f}%) should sit below the RCP4.5-by-2100 figure (~24%)"
    assert high_reduction < 42.0, \
        f"2050_high ({high_reduction:.1f}%) should sit below the RCP8.5-by-2100 figure (~42%)"

    # UHI should taper to ~0 in deep winter and peak in summer for 2050_high
    uhi_high = _seasonal_uhi_offset(DELTAS["2050_high"], 2.5)
    assert uhi_high[1] == 0.0 and uhi_high[12] == 0.0, "UHI should be zero in midwinter"
    assert uhi_high[7] == 2.5, "UHI should peak in midsummer"

    print("  ✓ Baseline scenario leaves weather unchanged")
    print("  ✓ Both scenarios reduce annual HDD, with 2050_high reducing more")
    print("  ✓ Both scenarios sit BELOW their pathway's full-century literature figure")
    print("  ✓ UHI offset correctly tapers to zero in winter, peaks in summer")
    print()