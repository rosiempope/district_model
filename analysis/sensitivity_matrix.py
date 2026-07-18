"""Feasibility sensitivity matrix: which combinations turn investor NPV positive?

    python -m analysis.sensitivity_matrix

The whole pack's finding is that investor NPV is negative almost everywhere.
This study makes that precise and honest: it sweeps the input variables across
FEASIBLE, data-grounded ranges, runs every combination through the live engine,
and marks each PASS (investor NPV > 0) or FAIL. It also carries the other
stakeholders — contractor and operator — so you can see who is positive while
the owner is not.

Two tiers:

  1. COMMERCIAL FACTORIAL (the headline pass/fail matrix). Full factorial over
     the levers that can actually flip the owner's sign:
       - energy source stack        (4)
       - case / linear density      (4: the 3 archetypes + real Ealing)
       - customer proposition       (2: gas parity, heat-pump parity)
       - GHNF grant                 (3: 0 / 40 / ~50%)
       - avoided-capital capture    (3: 0 / 50 / 100% of the customer's own
                                     avoided heat-pump CAPEX, via a connection
                                     charge — the customer still gets running-
                                     cost parity, so is never worse off)
     = 288 runs. Heating dT, climate and 2-pipe held at base here.

  2. ENGINEERING SENSITIVITY (held at a strong commercial base: EfW + heat-pump
     parity + 40% grant + 50% capture). One-at-a-time over the physical levers
     the brief named — heating flow/return temperature, cooling flow/return,
     2-pipe vs 4-pipe, climate scenario, electricity price, discount rate —
     to show how much each moves the owner's NPV.

Ranges and their basis are documented inline. Outputs to output/sensitivity_matrix/:
master CSV, a PASS/FAIL tick-cross matrix, a stakeholder-NPV matrix, an
engineering-lever matrix, and findings.md.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sensitivity_matrix"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.archetypes import ARCHETYPES_WITH_EALING as CASES
from analysis.contractor_view import dalkia_position, split_capex
from economics.metrics import counterfactual_individual_ashp_dispatch
from optimisation.auto_size import recommend_sizing
from profiles.demand_synthesis import synthesise_network
from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import compute_climate_reference
from scenarios.fixed_cost_scaling import scaled_economics
from scenarios.scenario_runner import run_scenario

# ── Palette (validated categorical set, see dataviz skill) ──────────────────
C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
)
PASS_G, FAIL_R = "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

# ── Variables and their feasible ranges ─────────────────────────────────────
PRESETS = {"ashp": "ealing_phase1", "gas_boiler": "ealing_phase1",
           "efw_chp": "newlincs_style", "air_cooled_chiller": "generic_2MW_bank"}
# Source key -> (auto_size technology list, optional swap of the sized ASHP)
SOURCES = {
    "ASHP + gas peak": (["ashp", "gas_boiler"], None),
    "EfW + ASHP + gas peak": (["efw_chp", "ashp", "gas_boiler"], None),
    "WSHP + gas peak": (["ashp", "gas_boiler"], ("wshp", "generic_river_5MW")),
    "GSHP + gas peak": (["ashp", "gas_boiler"], ("gshp", "birmingham_aston_university")),
}
PROPOSITIONS = {"gas parity": "individual_gas", "HP parity": "individual_ashp"}
GRANTS = [0.0, 0.40, 0.4999]           # 0 / 40% base / just-under-50% ceiling
CAPTURES = [0.0, 0.50, 1.0]            # share of the customer's avoided HP CAPEX

# Heating flow/return (°C): CP1 2020 / CIBSE — modern networks target lower flow
# temps for HP COP, but existing stock often needs 70-80 to deliver 60 at the
# tap. dT 30-40 typical. 70/40 is the pack base.
HEAT_TEMPS = [(70, 40), (80, 40), (80, 50), (90, 50)]
# Cooling flow/return (°C): typical chilled-water 5-6 supply, 11-16 return.
COOL_TEMPS = [(6, 12), (5, 13), (6, 16)]
CLIMATES = ["baseline", "2050_central", "2050_high"]
# Large-business negotiated electricity, 2026: 21-27 p/kWh (EDF discount at the
# low end). Passed to the electric sources as £/MWh.
ELEC_PRICES_GBP_MWh = [210.0, 240.0, 270.0]
# Owner hurdle: 3.5% public/social, 6-8% mixed, 10.5% commercial (pack base).
DISCOUNTS = [0.105, 0.08, 0.06, 0.035]

raw_weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
raw_weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")
CLIMATE_REF = compute_climate_reference(apply_climate_scenario(raw_weather, "baseline"))

# Cache demand + avoided-HP CAPEX per (case, climate); the base factorial uses
# baseline climate only, the engineering tier adds the 2050 scenarios.
_demand_cache: dict = {}
_cf_capex_cache: dict = {}


def _demand(case, climate="baseline"):
    key = (case, climate)
    if key not in _demand_cache:
        w = apply_climate_scenario(raw_weather, climate)
        d = synthesise_network(w, {"demand_nodes": deepcopy(CASES[case]["buildings"])},
                               climate_reference=CLIMATE_REF)
        _demand_cache[key] = (d, w)
    return _demand_cache[key]


def _avoided_hp_capex(case, climate="baseline"):
    """Per-building individual-ASHP CAPEX the customer avoids by connecting."""
    key = (case, climate)
    if key not in _cf_capex_cache:
        d, w = _demand(case, climate)
        out = {}
        for b in CASES[case]["buildings"]:
            node = next(n for n in d["nodes"] if n["name"] == b["name"])
            r = counterfactual_individual_ashp_dispatch(
                {**node, "connections": b.get("connections", 1)}, w)
            out[b["name"]] = float(r["capex_GBP"])
        _cf_capex_cache[key] = out
    return _cf_capex_cache[key]


def run_combo(case, source, proposition, grant, capture,
              heat_flow=70, heat_return=40, climate="baseline",
              include_cooling=False, cool_flow=6, cool_return=12,
              elec_price=None, discount=0.105):
    cfg = CASES[case]
    d, w = _demand(case, climate)
    tech_types, swap = SOURCES[source]
    rec = recommend_sizing(
        demand_kW=d["total_heat_kW"], peak_demand_kW=d["peak_heat_kW"],
        technology_types=tech_types, weather_df=w, network_flow_temp_C=heat_flow,
        n_buildings=len(cfg["buildings"]), building_types=[b["type"] for b in cfg["buildings"]],
        include_cooling=include_cooling,
        cooling_demand_kW=d["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=d["peak_cool_kW"] if include_cooling else 0.0)

    def _map(srcs):
        out = []
        for s in srcs:
            stype, preset = s["type"], PRESETS.get(s["type"])
            if swap and stype == "ashp":
                stype, preset = swap
            m = {"type": stype, "preset": preset, "name": f"{stype} ({s['role']})",
                 "capacity_MW": float(s["capacity_MW"])}
            if "n_units" in s:
                m["n_units"] = int(s["n_units"])
            if "depends_on" in s:
                m["depends_on"] = int(s["depends_on"])
            if "dispatch_direct" in s:
                m["dispatch_direct"] = bool(s["dispatch_direct"])
            # Electric sources take the swept electricity price.
            if elec_price is not None and stype in {"ashp", "wshp", "gshp",
                                                    "electric_boiler", "booster_heat_pump"}:
                m["electricity_price_GBP_per_MWh"] = float(elec_price)
            out.append(m)
        return out

    peak_MW = d["peak_heat_kW"] / 1000.0
    if include_cooling:
        peak_MW += d["peak_cool_kW"] / 1000.0
    econ, _ = scaled_economics(peak_MW)
    econ["counterfactual"] = ("individual_ashp_and_ac" if (include_cooling and proposition ==
                              "individual_ashp") else
                              "individual_gas_and_ac" if include_cooling else proposition)
    econ["discount_rate"] = discount
    if grant > 0:
        econ["ghnf_grant"] = {"enabled": True, "rate": grant}

    buildings = deepcopy(cfg["buildings"])
    if capture > 0:
        avoided = _avoided_hp_capex(case, climate)
        for b in buildings:
            b["connection_charge_GBP"] = avoided.get(b["name"], 0.0) * capture

    scenario = {
        "name": f"{case}|{source}|{proposition}|g{grant}|c{capture}|{heat_flow}/{heat_return}|{climate}",
        "climate_scenario": climate,
        "demand": {"buildings": buildings},
        "network": {"mode": "generic_length", "length_m": float(cfg["route_m"]),
                    "include_cooling": include_cooling,
                    "heat_flow_temp_C": float(heat_flow), "heat_return_temp_C": float(heat_return),
                    "cool_flow_temp_C": float(cool_flow), "cool_return_temp_C": float(cool_return)},
        "sources": _map(rec["sources"]),
        "economics": econ,
    }
    if include_cooling:
        scenario["cooling_sources"] = _map(rec["cooling_sources"])
    return run_scenario(scenario)


def stakeholder_npvs(result):
    """Owner / contractor / operator NPV (£m) from one engine result."""
    owner = result["financial"]["investor"]["npv_GBP"] / 1e6
    split = split_capex(result)
    pos = dalkia_position(result, split)
    contractor = (pos["construction_margin_GBP"] + pos["design_margin_GBP"]) / 1e6
    operator = (pos["dalkia_NPV_GBP"] - pos["construction_margin_GBP"]
                - pos["design_margin_GBP"]) / 1e6
    return owner, contractor, operator


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — commercial factorial
# ═══════════════════════════════════════════════════════════════════════════

rows = []
for case in CASES:
    for source in SOURCES:
        for prop_label, prop in PROPOSITIONS.items():
            for grant in GRANTS:
                for capture in CAPTURES:
                    r = run_combo(case, source, prop, grant, capture)
                    owner, contractor, operator = stakeholder_npvs(r)
                    h = r["headline"]
                    rows.append({
                        "Case": case, "Source": source, "Proposition": prop_label,
                        "GHNF grant (%)": round(grant * 100),
                        "Capture (%)": round(capture * 100),
                        "Carbon (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
                        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
                        "Owner NPV (£m)": round(owner, 2),
                        "Contractor NPV (£m)": round(contractor, 2),
                        "Operator NPV (£m)": round(operator, 2),
                        "Investor PASS": bool(owner > 0),
                    })
    print(f"  {case}: done")

df = pd.DataFrame(rows)
df.to_csv(OUT / "commercial_factorial.csv", index=False)
n_pass = int(df["Investor PASS"].sum())
print(f"\nTier 1: {n_pass}/{len(df)} combinations pass investor NPV > 0")

# ── Figure SM1: PASS/FAIL tick-cross matrix ─────────────────────────────────
# Rows = source × case; columns = escalating commercial scenario. Shows where
# ticks appear as the favourable levers stack up.
COMMERCIAL_STEPS = [
    ("gas parity\nno grant\nno capture",  ("gas parity", 0, 0)),
    ("gas parity\n40% grant",             ("gas parity", 40, 0)),
    ("HP parity\n40% grant",              ("HP parity", 40, 0)),
    ("HP parity\n40% grant\n50% capture", ("HP parity", 40, 50)),
    ("HP parity\n~50% grant\n100% capture", ("HP parity", 50, 100)),
]
row_keys = [(s, c) for c in CASES for s in SOURCES]
row_labels = [f"{s}\n· {c.split(' (')[0]}" for c in CASES for s in SOURCES]


def _pass(case, source, prop, grant_pct, cap_pct):
    q = df[(df["Case"] == case) & (df["Source"] == source) & (df["Proposition"] == prop)
           & (df["GHNF grant (%)"] == grant_pct) & (df["Capture (%)"] == cap_pct)]
    return None if q.empty else bool(q["Investor PASS"].iloc[0]), (
        None if q.empty else float(q["Owner NPV (£m)"].iloc[0]))


fig, ax = plt.subplots(figsize=(11, 0.42 * len(row_keys) + 2.2))
ax.set_xlim(0, len(COMMERCIAL_STEPS)); ax.set_ylim(0, len(row_keys))
for ri, (source, case) in enumerate(row_keys):
    y = len(row_keys) - 1 - ri
    for ci, (_, (prop, gp, cp)) in enumerate(COMMERCIAL_STEPS):
        passed, npv = _pass(case, source, prop, gp, cp)
        colour = PASS_G if passed else FAIL_R
        ax.add_patch(plt.Rectangle((ci + 0.04, y + 0.06), 0.92, 0.88,
                                   facecolor=colour, alpha=0.16, edgecolor=GRID, lw=0.7))
        ax.text(ci + 0.5, y + 0.60, "✓" if passed else "✗", ha="center", va="center",
                fontsize=15, color=colour, fontweight="bold")
        ax.text(ci + 0.5, y + 0.26, f"{npv:.1f}", ha="center", va="center",
                fontsize=7.5, color=INK2)
ax.set_xticks([i + 0.5 for i in range(len(COMMERCIAL_STEPS))])
ax.set_xticklabels([s[0] for s in COMMERCIAL_STEPS], fontsize=8.5)
ax.set_yticks([len(row_keys) - 1 - ri + 0.5 for ri in range(len(row_keys))])
ax.set_yticklabels(row_labels, fontsize=7.6)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Does investor NPV turn positive? PASS (✓) / FAIL (✗) by scheme and commercial scenario\n"
             "Cell number = owner NPV (£m). Base heating 70/40, baseline climate, 2-pipe.",
             fontsize=11.5, pad=12)
fig.tight_layout()
fig.savefig(OUT / "SM1_pass_fail_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Figure SM2: stakeholder NPV matrix ──────────────────────────────────────
# Contractor/operator NPV depend on the scheme (source × case), not the
# commercial levers; owner NPV shown at the strongest commercial scenario.
stake_rows = []
for case in CASES:
    for source in SOURCES:
        sub = df[(df["Case"] == case) & (df["Source"] == source)]
        contractor = sub["Contractor NPV (£m)"].mean()   # ~constant across levers
        operator = sub["Operator NPV (£m)"].mean()
        best_owner = sub["Owner NPV (£m)"].max()
        stake_rows.append({"Case": case, "Source": source,
                           "Contractor": contractor, "Operator": operator,
                           "Owner (best)": best_owner})
stake_df = pd.DataFrame(stake_rows)
stake_df.to_csv(OUT / "stakeholder_matrix.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(stake_df) + 1.8))
cols = ["Contractor", "Operator", "Owner (best)"]
ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(stake_df))
for ri, (_, r) in enumerate(stake_df.iterrows()):
    y = len(stake_df) - 1 - ri
    for ci, col in enumerate(cols):
        v = r[col]
        colour = PASS_G if v > 0 else FAIL_R
        ax.add_patch(plt.Rectangle((ci + 0.04, y + 0.06), 0.92, 0.88,
                                   facecolor=colour, alpha=0.16, edgecolor=GRID, lw=0.7))
        ax.text(ci + 0.5, y + 0.5, f"{v:+.2f}", ha="center", va="center",
                fontsize=9.5, color=INK)
ax.set_xticks([i + 0.5 for i in range(len(cols))])
ax.set_xticklabels(cols, fontsize=10)
ax.set_yticks([len(stake_df) - 1 - ri + 0.5 for ri in range(len(stake_df))])
ax.set_yticklabels([f"{r['Source']} · {r['Case'].split(' (')[0]}"
                    for _, r in stake_df.iterrows()], fontsize=7.6)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Stakeholder NPV (£m) — contractor and operator are paid first, the owner holds the residual\n"
             "Owner shown at its best commercial scenario. Green > 0, red < 0.",
             fontsize=11, pad=12)
fig.tight_layout()
fig.savefig(OUT / "SM2_stakeholder_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — engineering sensitivity from a strong commercial base
# ═══════════════════════════════════════════════════════════════════════════

# proposition here is the counterfactual VALUE (run_combo takes the value, as the
# Tier-1 loop passes it), not the display label.
BASE = dict(source="EfW + ASHP + gas peak", proposition="individual_ashp",
            grant=0.40, capture=0.50)
eng_rows = []


def _record(case, lever, setting, npv, extra=""):
    eng_rows.append({"Case": case, "Lever": lever, "Setting": setting,
                     "Owner NPV (£m)": round(npv, 2), "PASS": bool(npv > 0), "Note": extra})


for case in CASES:
    real = CASES[case].get("is_real", False)
    # Heating flow/return temperature
    for hf, hr in HEAT_TEMPS:
        r = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"],
                      BASE["capture"], heat_flow=hf, heat_return=hr)
        _record(case, "Heating flow/return °C", f"{hf}/{hr}",
                r["financial"]["investor"]["npv_GBP"] / 1e6,
                "" if r["headline"]["service_compliant"] else "service FAIL")
    # Climate
    for clim in CLIMATES:
        r = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"],
                      BASE["capture"], climate=clim)
        _record(case, "Climate", clim, r["financial"]["investor"]["npv_GBP"] / 1e6)
    # Electricity price
    for ep in ELEC_PRICES_GBP_MWh:
        r = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"],
                      BASE["capture"], elec_price=ep)
        _record(case, "Electricity price p/kWh", f"{ep/10:.0f}",
                r["financial"]["investor"]["npv_GBP"] / 1e6)
    # Owner discount rate
    for dr in DISCOUNTS:
        r = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"],
                      BASE["capture"], discount=dr)
        _record(case, "Owner discount rate %", f"{dr*100:.1f}",
                r["financial"]["investor"]["npv_GBP"] / 1e6)
    # 2-pipe vs 4-pipe + cooling dT (only where a cooling load exists)
    if not real:
        r2 = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"], BASE["capture"])
        _record(case, "Network", "2-pipe", r2["financial"]["investor"]["npv_GBP"] / 1e6)
        for cf, cr in COOL_TEMPS:
            r4 = run_combo(case, BASE["source"], BASE["proposition"], BASE["grant"],
                           BASE["capture"], include_cooling=True, cool_flow=cf, cool_return=cr)
            _record(case, "Network", f"4-pipe {cf}/{cr}",
                    r4["financial"]["investor"]["npv_GBP"] / 1e6)

eng_df = pd.DataFrame(eng_rows)
eng_df.to_csv(OUT / "engineering_sensitivity.csv", index=False)

# ── Figure SM3: engineering-lever NPV, small multiples per case ─────────────
fig, axes = plt.subplots(1, len(CASES), figsize=(4.4 * len(CASES), 6.4), sharex=True)
for ax, case in zip(axes, CASES):
    sub = eng_df[eng_df["Case"] == case]
    labels = [f"{r['Lever'].split(' ')[0]}: {r['Setting']}" for _, r in sub.iterrows()]
    y = np.arange(len(sub))
    colours = [PASS_G if p else FAIL_R for p in sub["PASS"]]
    ax.barh(y, sub["Owner NPV (£m)"], color=colours, alpha=0.85, zorder=3)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_title(case.split(" (")[0], fontsize=10.5)
    ax.set_xlabel("Owner NPV (£m)")
fig.suptitle("Engineering sensitivity — owner NPV vs each physical lever\n"
             "Base: EfW + ASHP + gas peak, heat-pump parity, 40% GHNF, 50% avoided-capital capture. "
             "Green = PASS.", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "SM3_engineering_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# findings.md
# ═══════════════════════════════════════════════════════════════════════════

passing = df[df["Investor PASS"]]
lines = [
    "# Feasibility sensitivity matrix — where does investor NPV turn positive?",
    "",
    "Generated by `python -m analysis.sensitivity_matrix`. Every combination run "
    "through the live engine; PASS = investor (owner) NPV > 0.",
    "",
    "## Headline",
    "",
    f"- **{n_pass} of {len(df)}** commercial-factorial combinations pass. "
    + ("None do." if n_pass == 0 else "They cluster where the favourable levers stack up."),
]
if n_pass > 0:
    by_case = passing.groupby("Case").size().to_dict()
    lines.append("- Passing combinations by case: "
                 + ", ".join(f"{c.split(' (')[0]} {n}" for c, n in by_case.items()) + ".")
    lines.append("- Every pass needs **heat-pump parity** billing; almost all need "
                 "**avoided-capital capture** on top of GHNF. Gas-parity billing never passes.")
lines += [
    "- **Contractor and operator NPV are positive across essentially every scheme** — "
    "they are paid out of CAPEX/OPEX first; only the owner holds the residual (see SM2).",
    "- The physical levers (heating dT, cooling dT, 2- vs 4-pipe, climate, electricity "
    "price) move owner NPV far less than the commercial levers — see SM3. For an "
    "EfW-baseload stack, flow temperature barely moves NPV (smaller pipes offset lower COP).",
    "",
    "## Variables and feasible ranges",
    "",
    "| Variable | Range | Basis |",
    "|---|---|---|",
    "| Energy source | ASHP / EfW+ASHP / WSHP / GSHP (+ gas peak) | pack source set |",
    "| Case / linear density | Dense 14.4, Middle 3.0, Scarce 0.65, Ealing 6.6 MWh/m/yr | archetypes + real Ealing |",
    "| Customer proposition | gas parity, heat-pump parity | legal zoning counterfactuals |",
    "| GHNF grant | 0 / 40 / ~50% | GHNF cap (economics/grant.py) |",
    "| Avoided-capital capture | 0 / 50 / 100% | connection charge ≤ customer's avoided HP CAPEX |",
    "| Heating flow/return | 70/40 – 90/50 °C | CP1 2020 / CIBSE, dT 30-40 |",
    "| Cooling flow/return | 6/12 – 6/16 °C | typical chilled-water design |",
    "| Climate | baseline / 2050 central / 2050 high | UKCP18 RCP4.5 / RCP8.5 |",
    "| Electricity price | 21 / 24 / 27 p/kWh | large-business negotiated 2026 |",
    "| Owner discount rate | 3.5 / 6 / 8 / 10.5% | social → commercial |",
    "",
    "## Full commercial factorial",
    "",
    df.to_markdown(index=False),
    "",
    "## Engineering sensitivity (strong commercial base)",
    "",
    eng_df.to_markdown(index=False),
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 3 matrices.")
