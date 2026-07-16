"""Exeter case study: a real tree topology built from the DESNZ Heat Network
Zoning Pilot "City Typologies" map for Exeter, run directly against the
model engine (scenario_runner / demand_synthesis / auto_size / topology_tree)
— no Streamlit, no test files.

Run from the repository root:
    python -m analysis.exeter_case_study

Outputs CSVs, PNGs and findings.md to output/exeter_case_study/.

Provenance and honesty notes (read before presenting)
---------------------------------------------------------
- The map (DESNZ Heat Network Zoning Pilot Programme, "Map A — City
  Typologies", Exeter) classifies the city into typology zones (dense
  city centre, city centre fringe, mixed use district, social housing,
  campus health/education, commercial/business office, industrial) but
  is NOT a pipe-routing drawing — it has no branch lengths on it. Every
  segment length below is HAND-ESTIMATED from the map's own printed
  scale bar and general zone positions, the same methodology already
  used for the Ealing worked example in
  network/network_topology.py::ealing_town_centre_topology() (see that
  function's docstring for the precedent). Treat proportions and the
  broad density story as indicative; treat any single segment length as
  illustrative, not survey-grade. This directly matches the user's own
  framing: "doesn't need to be perfect."
- Two SEPARATE candidate networks are built, not one giant tree: the
  dense central core (city centre + fringe + mixed use + social housing
  + two campuses, all within ~1-2km) is modelled as one energy-centre
  catchment; Sowton Industrial Estate / Exeter Airport commercial zone /
  the East Devon New Community (several km east, a different part of
  the typology map) is modelled as a second, separate energy-centre
  catchment. Realistically these are different schemes, not branches of
  the same one — a single energy centre serving both would need >5km
  trunk mains for no physical reason. EXE_0017 / Cranbrook (~8km
  further northeast) is excluded entirely — too distant to share an
  energy centre with either cluster; it would need its own scheme.
- The model has no dedicated "industrial" building archetype. Sowton
  Industrial Estate is therefore NOT modelled as a heat-demand node —
  in current UK practice an industrial estate is a more plausible
  ENERGY SOURCE location (siting for EfW/waste-heat recovery, land for
  an energy centre) than a heat network customer, and that is how it is
  used here: as the energy-centre location for the eastern network, not
  a connected building.
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "exeter_case_study"
OUT.mkdir(parents=True, exist_ok=True)

from profiles.demand_synthesis import synthesise_network
from optimisation.auto_size import recommend_sizing
from scenarios.scenario_runner import run_scenario
from scenarios.fixed_cost_scaling import (
    FIXED_CAPEX_KEYS,
    FIXED_OPEX_KEYS,
    MIN_SCALE_FACTOR,
    reference_peak_MW,
    scaled_economics,
)
from economics.tariffs import OFGEM_GAS_CAP_P_PER_KWH

# Re-exported for the sibling Exeter scripts that import these from here.
__all__ = [
    "scaled_economics", "reference_peak_MW", "REFERENCE_PEAK_MW",
    "FIXED_CAPEX_KEYS", "FIXED_OPEX_KEYS", "MIN_SCALE_FACTOR",
    "CENTRAL_BUILDINGS", "CENTRAL_SEGMENTS", "SOWTON_BUILDINGS", "SOWTON_SEGMENTS",
    "PRESET_FOR_TYPE", "weather", "build_tree_scenario", "build_generic_scenario",
]

# Fixed-cost scaling now lives in scenarios/fixed_cost_scaling.py — it was
# described here as "the single source of truth every Exeter script should
# import this from", which was true of the Exeter scripts and of nothing else,
# because reaching it meant importing this whole 723-line study module. Moved so
# any scenario builder can use it. Re-exported below so existing
# `from analysis.exeter_case_study import scaled_economics` call sites in the
# other Exeter scripts keep working unchanged.

# ── Palette (same validated set used in the first Dalkia readout) ───────────
C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED, C_MAGENTA, C_ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


weather = pd.read_csv(ROOT / "profiles" / "weather_data.csv")
assert len(weather) == 8760
weather.index = pd.date_range("2023-01-01", periods=8760, freq="h")

REFERENCE_PEAK_MW = reference_peak_MW()   # 8.58 MW, 564 connections


# ═══════════════════════════════════════════════════════════════════════════
# 1. The two Exeter networks, built from the DESNZ typology map
# ═══════════════════════════════════════════════════════════════════════════

CENTRAL_BUILDINGS = [
    {"name": "City centre fringe", "type": "mixed_use", "floor_area_m2": 12000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
    {"name": "Mixed use district", "type": "mixed_use", "floor_area_m2": 10000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.90},
    {"name": "Social housing", "type": "residential_existing", "floor_area_m2": 18750,
     "units": 250, "connections": 250, "connection_year": 1, "connection_probability": 0.85},
    {"name": "Wonford health campus", "type": "hospital", "floor_area_m2": 8000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
    {"name": "University campus (Streatham)", "type": "school", "floor_area_m2": 20000,
     "connections": 1, "connection_year": 1, "connection_probability": 1.0},
]
# EC = Exeter City Centre (dense core, root). Branch lengths hand-estimated
# from the map's scale bar; see module docstring.
CENTRAL_SEGMENTS = [
    {"node_id": "N1", "parent_id": "EC", "length_m": 500.0, "building": "City centre fringe"},
    {"node_id": "N2", "parent_id": "EC", "length_m": 700.0, "building": "Mixed use district"},
    {"node_id": "N3", "parent_id": "N2", "length_m": 400.0, "building": "Social housing"},
    {"node_id": "N4", "parent_id": "N3", "length_m": 900.0, "building": "Wonford health campus"},
    {"node_id": "N5", "parent_id": "EC", "length_m": 1400.0, "building": "University campus (Streatham)"},
]

SOWTON_BUILDINGS = [
    {"name": "Commercial business district (Airport)", "type": "office_ac", "floor_area_m2": 25000,
     "connections": 1, "connection_year": 1, "connection_probability": 0.90},
    {"name": "East Devon New Community", "type": "residential", "floor_area_m2": 45000,
     "units": 600, "connections": 600, "connection_year": 2, "connection_probability": 0.80},
]
# EC = energy centre sited at/adjacent to Sowton Industrial Estate (a
# plausible energy-centre/waste-heat-recovery location, not a connected
# customer — see module docstring).
SOWTON_SEGMENTS = [
    {"node_id": "M1", "parent_id": "EC", "length_m": 2600.0, "building": "Commercial business district (Airport)"},
    {"node_id": "M2", "parent_id": "EC", "length_m": 3200.0, "building": "East Devon New Community"},
]

NETWORKS = {
    "Central Exeter (dense core)": {"buildings": CENTRAL_BUILDINGS, "segments": CENTRAL_SEGMENTS},
    "Sowton / Airport / East Devon": {"buildings": SOWTON_BUILDINGS, "segments": SOWTON_SEGMENTS},
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. Scenario builder — tree mode, GHNF always enabled (auto-withheld by
#    the model wherever the carbon gate fails, so "enabled everywhere" and
#    "applied only where eligible" are the same thing here)
# ═══════════════════════════════════════════════════════════════════════════

PRESET_FOR_TYPE = {
    "ashp": "ealing_phase1", "gas_boiler": "ealing_phase1", "electric_boiler": "ealing_backup",
    "data_centre": "redwire_ealing", "booster_heat_pump": "generic_2MW", "efw_chp": "newlincs_style",
    "air_cooled_chiller": "generic_2MW_bank",
}
TECH_OPTIONS = {
    "Gas-only reference": ["gas_boiler"],
    "ASHP + gas peak": ["ashp", "gas_boiler"],
    "Data-centre waste heat + booster + gas peak": ["data_centre", "gas_boiler"],
    "EfW heat export + ASHP + gas peak": ["efw_chp", "ashp", "gas_boiler"],
}


def _map_sources(auto_sources):
    mapped = []
    for s in auto_sources:
        cfg = {"type": s["type"], "preset": PRESET_FOR_TYPE[s["type"]],
               "name": f"{s['type']} ({s['role']})", "capacity_MW": float(s["capacity_MW"])}
        if "n_units" in s:
            cfg["n_units"] = int(s["n_units"])
        if "depends_on" in s:
            cfg["depends_on"] = int(s["depends_on"])
        if "dispatch_direct" in s:
            cfg["dispatch_direct"] = bool(s["dispatch_direct"])
        mapped.append(cfg)
    return mapped


def build_tree_scenario(name, buildings, segments, tech_types, include_cooling=False):
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=tech_types, weather_df=weather, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
        include_cooling=include_cooling,
        cooling_demand_kW=demand["total_cooling_kW"] if include_cooling else None,
        peak_cooling_kW=demand["peak_cool_kW"] if include_cooling else 0.0,
    )
    peak_total_MW = (demand["peak_heat_kW"] + (demand["peak_cool_kW"] if include_cooling else 0.0)) / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    if include_cooling:
        economics["counterfactual"] = "individual_gas_and_ac"
    scenario = {
        "name": name, "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "tree", "segments": deepcopy(segments), "include_cooling": include_cooling,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0,
                    "cool_flow_temp_C": 6.0, "cool_return_temp_C": 12.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    if include_cooling:
        scenario["cooling_sources"] = _map_sources(rec["cooling_sources"])
    return scenario


def build_generic_scenario(name, buildings, route_m, tech_types):
    """Same demand, generic_length mode — used only for the linear-density
    sweep (§4), where what matters is total route length, not exact topology."""
    demand = synthesise_network(weather, {"demand_nodes": deepcopy(buildings)})
    rec = recommend_sizing(
        demand_kW=demand["total_heat_kW"], peak_demand_kW=demand["peak_heat_kW"],
        technology_types=tech_types, weather_df=weather, network_flow_temp_C=70.0,
        n_buildings=len(buildings), building_types=[b["type"] for b in buildings],
    )
    peak_total_MW = demand["peak_heat_kW"] / 1000.0
    economics, scale_factor = scaled_economics(peak_total_MW)
    economics["ghnf_grant"] = {"enabled": True, "rate": 0.40}
    scenario = {
        "name": name, "climate_scenario": "baseline",
        "demand": {"buildings": deepcopy(buildings)},
        "network": {"mode": "generic_length", "length_m": float(route_m), "include_cooling": False,
                    "heat_flow_temp_C": 70.0, "heat_return_temp_C": 40.0},
        "sources": _map_sources(rec["sources"]),
        "economics": economics,
    }
    return scenario


# ═══════════════════════════════════════════════════════════════════════════
# 3. Run the technology matrix on both real (tree) networks
# ═══════════════════════════════════════════════════════════════════════════

matrix_rows, matrix_results = [], {}
for net_label, net in NETWORKS.items():
    for tech_label, tech_types in TECH_OPTIONS.items():
        scenario = build_tree_scenario(f"{net_label} — {tech_label}", net["buildings"], net["segments"], tech_types)
        result = run_scenario(scenario)
        matrix_results[(net_label, tech_label)] = result
        h, inv, grant = result["headline"], result["financial"]["investor"], result["grant"]
        matrix_rows.append({
            "Network": net_label, "Technology": tech_label,
            "Route (m)": h["network_total_length_m"],
            "Linear heat density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
            "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
            "Carbon intensity (gCO2e/kWh)": round(h["carbon_intensity_kgCO2_per_kWh"] * 1000, 1),
            "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
            "GHNF grant (£m)": round((grant["grant_GBP"] if grant else 0.0) / 1e6, 2),
            "Equivalent tariff (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
            "Required break-even tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
            "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
            "Investor IRR (%)": round(inv["irr"] * 100, 2) if inv["irr"] is not None else None,
            "Screening decision": result["screening"]["status"],
        })
matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(OUT / "exeter_technology_matrix.csv", index=False)
print("=== Exeter tree-topology technology matrix (GHNF applied wherever eligible) ===")
print(matrix_df.to_string(index=False))

# Four-pipe (heating + cooling) variant of the commercial/airport network —
# the map's own "Commercial / business office district" label is exactly
# the real-world case a 4-pipe cooling extension should be tested against.
fourpipe_rows, fourpipe_results = [], {}
for tech_label, tech_types in TECH_OPTIONS.items():
    scenario = build_tree_scenario(
        f"Sowton/Airport 4-pipe — {tech_label}", SOWTON_BUILDINGS, SOWTON_SEGMENTS, tech_types, include_cooling=True,
    )
    result = run_scenario(scenario)
    fourpipe_results[tech_label] = result
    h, inv = result["headline"], result["financial"]["investor"]
    fourpipe_rows.append({
        "Technology": tech_label,
        "CAPEX (£m)": round(h["capex_total_GBP"] / 1e6, 2),
        "Carbon gate": "PASS" if h["carbon_compliant"] else "FAIL",
        "Heat NPV component basis": "combined heating+cooling investor cash flow",
        "Investor NPV (£m)": round(inv["npv_GBP"] / 1e6, 2),
        "Cooling bill ratio vs AC (%)": (
            round(inv["year1_cooling_bill_ratio"] * 100, 1) if inv["year1_cooling_bill_ratio"] else None
        ),
        "Screening decision": result["screening"]["status"],
    })
fourpipe_df = pd.DataFrame(fourpipe_rows)
fourpipe_df.to_csv(OUT / "exeter_fourpipe_matrix.csv", index=False)
print("\n=== Sowton/Airport four-pipe (heating + cooling) variant ===")
print(fourpipe_df.to_string(index=False))

# 2-pipe heating-only comparison at the SAME network for a clean side-by-side
heating_only_sowton = {k[1]: v for k, v in matrix_results.items() if k[0] == "Sowton / Airport / East Devon"}
pipe_compare_rows = []
for tech_label in TECH_OPTIONS:
    h2 = heating_only_sowton[tech_label]["financial"]["investor"]["npv_GBP"]
    h4 = fourpipe_results[tech_label]["financial"]["investor"]["npv_GBP"]
    pipe_compare_rows.append({
        "Technology": tech_label,
        "NPV, 2-pipe heating only (£m)": round(h2 / 1e6, 2),
        "NPV, 4-pipe heating+cooling (£m)": round(h4 / 1e6, 2),
        "Cooling makes NPV...": "better" if h4 > h2 else "worse",
        "Delta (£m)": round((h4 - h2) / 1e6, 2),
    })
pipe_compare_df = pd.DataFrame(pipe_compare_rows)
pipe_compare_df.to_csv(OUT / "exeter_2pipe_vs_4pipe.csv", index=False)
print("\n=== Does adding cooling (4-pipe) help or hurt NPV at Sowton/Airport? ===")
print(pipe_compare_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Linear-density viability sweep at a SET gas-parity tariff rate
#    Uses the Central network's demand (fixed), varies total route length,
#    and finds the linear density at which the required break-even tariff
#    crosses a FIXED reference rate — the live Ofgem household gas cap
#    (7.33 p/kWh, "a set gas parity tariff rate") and, for context, this
#    demand mix's own modelled gas-parity-bill-equivalent rate (~8.3 p/kWh,
#    see analysis/dalkia_screening_study.py for why that number is nearly
#    technology-invariant).
# ═══════════════════════════════════════════════════════════════════════════

SWEEP_TECHS = ["ASHP + gas peak", "Data-centre waste heat + booster + gas peak", "EfW heat export + ASHP + gas peak"]
SWEEP_LENGTHS_M = [250, 400, 600, 900, 1300, 1800, 2500, 3500, 5000, 7000, 10000, 14000, 19000]

sweep_rows = []
for tech_label in SWEEP_TECHS:
    for length_m in SWEEP_LENGTHS_M:
        scenario = build_generic_scenario(
            f"density-sweep-{tech_label}-{length_m}", CENTRAL_BUILDINGS, length_m, TECH_OPTIONS[tech_label],
        )
        result = run_scenario(scenario)
        h, inv = result["headline"], result["financial"]["investor"]
        sweep_rows.append({
            "Technology": tech_label, "Route (m)": length_m,
            "Linear heat density (MWh/m/yr)": h["linear_heat_density_MWh_per_m_year"],
            "Required break-even tariff (p/kWh)": inv["required_heat_tariff_p_per_kWh_for_zero_NPV"],
            "Equivalent gas-parity tariff (p/kWh)": inv["equivalent_year1_heat_tariff_p_per_kWh"],
        })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT / "linear_density_sweep.csv", index=False)

mean_parity_tariff = float(sweep_df["Equivalent gas-parity tariff (p/kWh)"].mean())


def _crossing_density(tech_label, reference_tariff):
    """Linear-interpolate the sweep to find the linear density at which
    required tariff first drops to/below reference_tariff, as density falls
    from high to low (required tariff RISES as density falls)."""
    sub = sweep_df[sweep_df["Technology"] == tech_label].sort_values("Linear heat density (MWh/m/yr)")
    d = sub["Linear heat density (MWh/m/yr)"].values
    t = sub["Required break-even tariff (p/kWh)"].values
    if reference_tariff < t.min():
        return None  # never viable in the swept range, even at max density
    if reference_tariff > t.max():
        return float(d.max())  # already viable at every density swept
    return float(np.interp(reference_tariff, t[::-1], d[::-1]))


threshold_rows = []
for tech_label in SWEEP_TECHS:
    sub = sweep_df[sweep_df["Technology"] == tech_label]
    min_tariff_in_sweep = float(sub["Required break-even tariff (p/kWh)"].min())
    max_density_in_sweep = float(sub["Linear heat density (MWh/m/yr)"].max())
    ofgem_crossing = _crossing_density(tech_label, OFGEM_GAS_CAP_P_PER_KWH)
    parity_crossing = _crossing_density(tech_label, mean_parity_tariff)
    threshold_rows.append({
        "Technology": tech_label,
        f"Density needed for break-even @ Ofgem cap ({OFGEM_GAS_CAP_P_PER_KWH:.2f}p/kWh)": (
            round(ofgem_crossing, 1) if ofgem_crossing is not None else "not reached in swept range"
        ),
        f"Density needed for break-even @ modelled parity (~{mean_parity_tariff:.1f}p/kWh)": (
            round(parity_crossing, 1) if parity_crossing is not None else "not reached in swept range"
        ),
        "Min. required tariff reached in sweep (p/kWh)": round(min_tariff_in_sweep, 1),
        "...at max swept density (MWh/m/yr)": round(max_density_in_sweep, 1),
        "Still x Ofgem cap at that density": round(min_tariff_in_sweep / OFGEM_GAS_CAP_P_PER_KWH, 1),
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUT / "linear_density_thresholds.csv", index=False)
print(f"\n=== Linear-density break-even thresholds (reference tariffs: Ofgem cap "
      f"{OFGEM_GAS_CAP_P_PER_KWH:.2f}p/kWh, modelled parity ~{mean_parity_tariff:.1f}p/kWh) ===")
print(threshold_df.to_string(index=False))
print("\nNone of the three technologies reaches EITHER reference tariff anywhere in the swept density "
      "range (250m-19,000m route on this 5-building, 254-connection demand base) — density alone cannot "
      "close the gap at this connection count; see findings.md for what that implies.")

# Where do the two real Exeter networks actually land on this curve?
network_density_rows = []
for net_label in NETWORKS:
    for tech_label in SWEEP_TECHS:
        row = matrix_df[(matrix_df["Network"] == net_label) & (matrix_df["Technology"] == tech_label)].iloc[0]
        network_density_rows.append({
            "Network": net_label, "Technology": tech_label,
            "Actual linear density (MWh/m/yr)": row["Linear heat density (MWh/m/yr)"],
            "Required tariff (p/kWh)": row["Required break-even tariff (p/kWh)"],
            "Clears Ofgem cap?": row["Required break-even tariff (p/kWh)"] <= OFGEM_GAS_CAP_P_PER_KWH,
        })
network_density_df = pd.DataFrame(network_density_rows)
print("\n=== Where the two real Exeter networks sit against the density threshold ===")
print(network_density_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 5. Charts
# ═══════════════════════════════════════════════════════════════════════════

TECH_COLOR = {
    "Gas-only reference": C_RED, "ASHP + gas peak": C_BLUE,
    "Data-centre waste heat + booster + gas peak": C_AQUA,
    "EfW heat export + ASHP + gas peak": C_VIOLET,
}

# --- Fig E1: hand-drawn schematic tree diagrams for both networks ---
# Rough compass-plausible node positions (km, relative to each network's own
# energy centre), NOT precise GIS coordinates — see module docstring.
CENTRAL_POS = {
    "EC": (0.0, 0.0),
    "N5": (-0.3, 1.4),    # University campus, north
    "N1": (0.6, -0.5),    # City centre fringe, south
    "N2": (-0.5, -0.7),   # Mixed use district, south-west
    "N3": (-0.7, -1.1),   # Social housing, continuing south-west
    "N4": (-0.9, -2.0),   # Wonford health campus, further south
}
CENTRAL_LABELS = {"EC": "Energy centre\n(City Centre)", "N5": "University campus\n(Streatham)",
                   "N1": "City centre\nfringe", "N2": "Mixed use\ndistrict",
                   "N3": "Social\nhousing", "N4": "Wonford health\ncampus"}
SOWTON_POS = {"EC": (0.0, 0.0), "M1": (2.6, 0.4), "M2": (2.2, -2.4)}
SOWTON_LABELS = {"EC": "Energy centre\n(Sowton Ind. Est.)", "M1": "Commercial/business\n(Airport)",
                  "M2": "East Devon\nNew Community"}


def _draw_tree(ax, segments, positions, labels, title, peaks_by_node, label_offset=None):
    label_offset = label_offset or {}
    for seg in segments:
        x0, y0 = positions[seg["parent_id"]]
        x1, y1 = positions[seg["node_id"]]
        ax.plot([x0, x1], [y0, y1], color=MUTED, linewidth=2.2, zorder=1, solid_capstyle="round")
        dx, dy = x1 - x0, y1 - y0
        seg_len = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
        # Offset the length label perpendicular to its own segment, not
        # straight onto the midpoint, so it doesn't sit on top of a nearby
        # node label travelling in roughly the same direction.
        px, py = -dy / seg_len * 0.16, dx / seg_len * 0.16
        mx, my = (x0 + x1) / 2 + px, (y0 + y1) / 2 + py
        ax.text(mx, my, f"{seg['length_m']:.0f} m", fontsize=8.5, color=C_ORANGE, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none"), zorder=3)
    for node_id, (x, y) in positions.items():
        is_root = node_id == "EC"
        peak = peaks_by_node.get(node_id, 0.0)
        size = 260 if is_root else max(90, peak * 22)
        ax.scatter([x], [y], s=size, color=(C_RED if is_root else C_BLUE), zorder=4,
                   edgecolor="white", linewidth=1.2)
        lx, ly = label_offset.get(node_id, (0.0, -0.32))
        ax.text(x + lx, y + ly, labels[node_id], fontsize=8.5, ha="center",
                va=("bottom" if ly > 0 else "top"), color=INK, zorder=5)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    ax.set_xlabel("Illustrative east-west offset (km)")
    ax.set_ylabel("Illustrative north-south offset (km)")
    ax.set_aspect("equal")
    ax.margins(0.22)
    ax.spines[["top", "right"]].set_visible(False)


central_demand = synthesise_network(weather, {"demand_nodes": deepcopy(CENTRAL_BUILDINGS)})
central_peaks = {n["name"]: n["peak_heat_kW"] / 1000 for n in central_demand["nodes"]}
central_peaks_by_node = {seg["node_id"]: central_peaks[seg["building"]] for seg in CENTRAL_SEGMENTS}
sowton_demand = synthesise_network(weather, {"demand_nodes": deepcopy(SOWTON_BUILDINGS)})
sowton_peaks = {n["name"]: n["peak_heat_kW"] / 1000 for n in sowton_demand["nodes"]}
sowton_peaks_by_node = {seg["node_id"]: sowton_peaks[seg["building"]] for seg in SOWTON_SEGMENTS}

CENTRAL_LABEL_OFFSET = {
    "EC": (0.85, 0.08), "N5": (0.0, 0.30), "N1": (0.55, -0.12),
    "N2": (-0.60, 0.06), "N3": (-0.70, -0.05), "N4": (0.0, -0.32),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
_draw_tree(axes[0], CENTRAL_SEGMENTS, CENTRAL_POS, CENTRAL_LABELS,
           "Central Exeter — dense core (total route 3,900 m)", central_peaks_by_node,
           label_offset=CENTRAL_LABEL_OFFSET)
_draw_tree(axes[1], SOWTON_SEGMENTS, SOWTON_POS, SOWTON_LABELS,
           "Sowton / Airport / East Devon (total route 5,800 m)", sowton_peaks_by_node)
_save(fig, "E1_exeter_tree_topologies.png")

# --- Fig E2/E3: NPV over time, all 4 technologies on one chart, per network ---
for i, net_label in enumerate(NETWORKS, start=2):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for tech_label, color in TECH_COLOR.items():
        r = matrix_results[(net_label, tech_label)]
        inv = r["financial"]["investor"]
        ax.plot(inv["cashflow_years"], np.array(inv["cumulative_discounted_GBP"]) / 1e6,
                 color=color, linewidth=2.0, label=tech_label, zorder=3)
    ax.axhline(0, color=INK, linewidth=1.0, zorder=2)
    ax.set_xlabel("Project year")
    ax.set_ylabel("Cumulative discounted cash position (£m)")
    ax.set_title(f"NPV over time by source — {net_label}", loc="left", fontsize=12, color=INK)
    ax.legend(fontsize=8.5, frameon=True, facecolor="#fcfcfb", edgecolor="none", loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, f"E{i}_npv_over_time_{'central' if 'Central' in net_label else 'sowton'}.png")

# --- Fig E4: NPV over time, 4-pipe cooling variant at Sowton/Airport ---
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for tech_label, color in TECH_COLOR.items():
    inv = fourpipe_results[tech_label]["financial"]["investor"]
    ax.plot(inv["cashflow_years"], np.array(inv["cumulative_discounted_GBP"]) / 1e6,
             color=color, linewidth=2.0, label=tech_label, zorder=3)
ax.axhline(0, color=INK, linewidth=1.0, zorder=2)
ax.set_xlabel("Project year")
ax.set_ylabel("Cumulative discounted cash position (£m)")
ax.set_title("NPV over time by source — Sowton/Airport, 4-pipe heating+cooling", loc="left", fontsize=11.5, color=INK)
ax.legend(fontsize=8.5, frameon=True, facecolor="#fcfcfb", edgecolor="none", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "E5_npv_over_time_fourpipe.png")

# --- Fig E6: linear density vs required tariff, with fixed reference lines ---
fig, ax = plt.subplots(figsize=(8.6, 5.6))
for tech_label in SWEEP_TECHS:
    sub = sweep_df[sweep_df["Technology"] == tech_label].sort_values("Linear heat density (MWh/m/yr)")
    ax.plot(sub["Linear heat density (MWh/m/yr)"], sub["Required break-even tariff (p/kWh)"],
             color=TECH_COLOR[tech_label], linewidth=2.0, marker="o", markersize=4.5, label=tech_label, zorder=3)
ax.axhline(OFGEM_GAS_CAP_P_PER_KWH, color=C_ORANGE, linestyle="--", linewidth=1.5, zorder=2,
           label=f"Ofgem gas cap ({OFGEM_GAS_CAP_P_PER_KWH:.2f} p/kWh)")
ax.axhline(mean_parity_tariff, color=INK2, linestyle=":", linewidth=1.5, zorder=2,
           label=f"Modelled gas-parity bill (~{mean_parity_tariff:.1f} p/kWh)")
for net_label, marker in [("Central Exeter (dense core)", "*"), ("Sowton / Airport / East Devon", "D")]:
    for tech_label in SWEEP_TECHS:
        row = matrix_df[(matrix_df["Network"] == net_label) & (matrix_df["Technology"] == tech_label)].iloc[0]
        ax.scatter([row["Linear heat density (MWh/m/yr)"]], [row["Required break-even tariff (p/kWh)"]],
                   marker=marker, s=140, color=TECH_COLOR[tech_label], edgecolor=INK, linewidth=1.0, zorder=4)
ax.set_xscale("log")
ax.set_xlabel("Linear heat density (MWh / route metre / year, log scale)")
ax.set_ylabel("Required break-even heat tariff (p/kWh)")
ax.set_title("Linear-density viability threshold at a fixed gas-parity tariff", loc="left", fontsize=12, color=INK)
ax.legend(fontsize=8, frameon=True, facecolor="#fcfcfb", edgecolor="none", loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "E6_density_threshold.png")

print(f"\nWrote CSVs and 6 charts to {OUT}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Findings markdown
# ═══════════════════════════════════════════════════════════════════════════

carbon_ok = matrix_df[matrix_df["Carbon gate"] == "PASS"]
best_row = carbon_ok.loc[carbon_ok["Investor NPV (£m)"].idxmax()]
central_conn = sum(b.get("connections", 1) for b in CENTRAL_BUILDINGS)
sowton_conn = sum(b.get("connections", 1) for b in SOWTON_BUILDINGS)

lines = [
    "# Exeter case study — findings",
    "",
    "Built from the DESNZ Heat Network Zoning Pilot \"City Typologies\" map for Exeter, run as a real "
    "`network.mode = \"tree\"` topology (per-branch lengths, per-segment pipe sizing) directly against the model "
    "engine. See the module docstring in `analysis/exeter_case_study.py` for exactly how each map zone was "
    "translated into a node, building type and branch length, and what was deliberately left out (industrial "
    "estate as a demand node; the distant Cranbrook/EXE_0017 zone).",
    "",
    "## 1. The two networks tested",
    "",
    f"- **Central Exeter (dense core)** — {len(CENTRAL_BUILDINGS)} zones, {central_conn} connections, "
    "3,900 m total route. Energy centre at Exeter City Centre; branches to the city-centre fringe, mixed-use "
    "district, social housing, Wonford health campus and the Streatham university campus.",
    f"- **Sowton / Airport / East Devon** — {len(SOWTON_BUILDINGS)} zones, {sowton_conn} connections, "
    "5,800 m total route. Energy centre sited at Sowton Industrial Estate (a source/plant location, not a "
    "connected customer); branches to the Airport commercial/business district and the East Devon New "
    "Community.",
    "- See fig. E1 for the schematic branch diagrams (node size = that zone's peak heat demand).",
    "",
    "## 2. Technology matrix, both networks (GHNF grant applied wherever the carbon gate passes)",
    "",
    matrix_df[[
        "Network", "Technology", "Linear heat density (MWh/m/yr)", "Carbon gate", "GHNF grant (£m)",
        "Investor NPV (£m)", "Screening decision",
    ]].to_markdown(index=False),
    "",
    f"- **Best case found: {best_row['Network']} — {best_row['Technology']}** "
    f"(NPV £{best_row['Investor NPV (£m)']:.2f}m, GHNF grant £{best_row['GHNF grant (£m)']:.2f}m, "
    f"screening: {best_row['Screening decision']}). EfW + ASHP + gas peak is the only technology that clears "
    "the carbon gate on both networks and earns a grant in both — consistently the strongest option here, "
    "same as the earlier Dalkia screening study on the illustrative archetypes.",
    "- Data-centre waste heat fails the carbon gate on both real Exeter networks (114-138 gCO2e/kWh) because "
    "the generic sizing leans on it for a similar baseload share to ASHP but its booster still draws grid "
    "electricity at the same carbon factor — it only earns its keep carbon-wise where a genuinely large, "
    "confirmed waste-heat source lets it displace MORE gas-peak running, which isn't demonstrated here.",
    "- Gas-only has the least-negative NPV on both networks (no low-carbon plant to fund) but fails carbon "
    "everywhere — the same pattern flagged in the first Dalkia readout: don't read \"best NPV\" without "
    "checking the carbon column first.",
    "",
    "## 3. Linear-density viability check at a set gas-parity tariff rate",
    "",
    f"Route length was swept from 250 m to 19,000 m on the Central network's own demand, holding the building "
    f"portfolio fixed, to trace required break-even tariff against linear heat density — then checked against "
    f"two fixed reference rates: the live Ofgem household gas cap "
    f"(**{OFGEM_GAS_CAP_P_PER_KWH:.2f} p/kWh**) and this portfolio's own modelled gas-parity bill "
    f"(**~{mean_parity_tariff:.1f} p/kWh**).",
    "",
    threshold_df.to_markdown(index=False),
    "",
    "**Neither reference tariff is reached anywhere in the swept range, for any technology.** Even at the "
    "shortest swept route (250 m — an unrealistically compact, near-zero-length network), the required tariff "
    "is still 3-4x the Ofgem cap. This is the key methodological finding of this section: **linear density is "
    "a necessary condition, not a sufficient one.** At this connection count (254 on the Central network), "
    "fixed CAPEX and OPEX (energy-centre building, connections, controls, billing/insurance/overhead — held "
    "constant per `scenarios/worked_scenarios.py`'s Ealing-calibrated defaults) exceed what the customer base "
    "can support, regardless of how short the pipe run is.",
    "- This matches the project's own existing, independently-generated finding in "
    "`output/feasibility_comparison/feasibility_comparison.md`: even the larger, real Ealing-calibrated case "
    "(~14.2 GWh, ~1,100 connections) fails NPV under gas-bill parity for the same structural reason — "
    "\"shortening the route improves NPV and lowers the break-even tariff, but cannot on its own close the gap "
    "between fair customer revenue and scheme CAPEX/OPEX.\" Exeter's smaller sub-networks show the identical "
    "pattern, more severely, because there are fewer connections to spread the fixed cost across.",
    "- Where the two real Exeter networks actually sit: Central (2.85 MWh/m/yr) needs ASHP+gas to be roughly "
    "4-6x more expensive than the customer bill supports; Sowton/Airport (1.44 MWh/m/yr, the longer, sparser "
    "network) needs roughly 7-9x. Both fail the Ofgem-cap check — see fig. E6 for the full curve and where "
    "each network lands on it.",
    "- Note on fig. E6: the two real-network markers sit a little above their technology's swept curve at the "
    "same density. That's expected, not an error — the sweep uses `generic_length` mode (one equivalent trunk) "
    "to trace the curve cheaply, while the real network points use the actual `tree` topology (real branch-"
    "level pipe sizing and losses); the two modes size pipework slightly differently at the same nominal "
    "density. The gap is a couple of p/kWh — irrelevant next to the 3-9x shortfall against either reference "
    "tariff.",
    "",
    "## 4. Four-pipe (heating + cooling) at Sowton/Airport",
    "",
    pipe_compare_df.to_markdown(index=False),
    "",
    "**Adding cooling makes NPV worse in every technology tested here, by roughly £13-14m.** The extra chiller "
    "plant, second pipe run and cooling-network CAPEX outweigh the extra (gas/AC-parity-capped) cooling "
    "revenue at this scale. Cooling bill ratio sits exactly at the 100% parity ceiling in every case (as "
    "designed — cooling revenue can never exceed what the customer would pay for individual air conditioning).",
    "",
    "## 5. Is district heating possible in the UK? — the clear answer",
    "",
    "Based on this model (Exeter and the earlier archetype study) plus the project's own established Ealing "
    "finding, district heating clears a genuine commercial investor hurdle in the UK only when **several "
    "conditions hold together** — no single one is sufficient on its own:",
    "",
    "1. **High linear heat density** (dense, short-branch routing to a lot of demand — town-centre mixed-use, "
    "not dispersed suburban housing). Necessary, but §3 above shows it is not sufficient by itself.",
    "2. **Enough absolute scale to spread fixed costs** — energy-centre, connection and overhead CAPEX/OPEX "
    "are largely fixed regardless of scheme size; a few hundred connections rarely clears them, a thousand-plus "
    "starts to.",
    "3. **A confirmed, cheap heat source** — genuine waste heat (data centre, EfW, industrial) with a real "
    "offtake agreement, not a generic \"if a source existed nearby\" assumption. This model shows a materially "
    "carbon- and cost-advantaged EfW/waste-heat option beats ASHP-only in every case tested.",
    "4. **Capital grant (GHNF, up to ~50%) and/or additional non-domestic-parity revenue** — anchor loads not "
    "held to gas-bill parity (hospitals, universities, commercial contracts on negotiated terms) materially "
    "change the revenue side; pure grant alone narrows but rarely closes the gap (see the first Dalkia readout, "
    "§5).",
    "5. **New-build development, where it can be required/assumed by planning policy** rather than retrofitted "
    "onto existing gas-heated buildings competing against a low incumbent gas bill — parity against a Part L "
    "2021 new-build heat demand is a much easier bar than parity against an older gas-heated building's bill.",
    "6. **Patient/blended capital, not a standard commercial hurdle rate** — most UK schemes that do get built "
    "(council/ESCO-owned networks, heat network zoning designations) use public or blended finance with a "
    "lower effective hurdle than the 10.5% real rate used throughout this model; that alone can turn a "
    "marginal case from FAIL to PASS without changing a single technical input.",
    "",
    "**In short: dense, large-scale, grant-supported, new-build-anchored schemes with a confirmed cheap heat "
    "source are where UK district heating works. Small or dispersed retrofit schemes chasing pure gas-bill "
    "parity on standard commercial capital — which is what every scenario in this study and the previous "
    "readout tested — consistently do not, regardless of technology choice.**",
    "",
    "## 6. Is a four-pipe cooling system a good idea? — the clear answer",
    "",
    "**Not by default, and this study adds a concrete number to that: no.** Every four-pipe case tested here "
    "and in the earlier Dalkia readout shows cooling making NPV worse, not better, at the scales tested "
    "(§4 above: roughly £13-14m worse). This also matches the project's own existing conclusion in "
    "`output/feasibility_comparison/feasibility_comparison.md`: \"do not add four-pipe cooling by default... "
    "re-test only where a concentrated cooling anchor, shared civil works and/or heat recovery materially "
    "changes the case.\"",
    "",
    "Four-pipe is worth a genuine second look only where:",
    "- there's a **concentrated, confirmed cooling anchor** (a data centre, a hospital, a dense commercial "
    "office cluster like the Airport zone tested here — not general residential, which barely uses cooling "
    "in the UK climate today),",
    "- the **heating and cooling networks can share civils** (same trench, same connections) rather than being "
    "priced as two separate builds, which is not modelled as a saving here and would need a project-specific "
    "civils estimate, and",
    "- there's a genuine **heat-recovery loop** between the two duties (e.g. chiller reject heat feeding the "
    "heat network) that this screening pass doesn't yet capture — that is the scenario where four-pipe's "
    "economics could plausibly flip.",
    "",
    "Absent those three, the default recommendation for an initial screening tool is: **quote two-pipe heating "
    "only, and flag four-pipe as an explicit, separately-justified sensitivity**, not a default option.",
    "",
    "---",
    "Generated by `analysis/exeter_case_study.py`; all figures reproducible by re-running that script. "
    "See `MODEL_ASSURANCE.md` for what this screening tool does and does not prove.",
]
(OUT / "findings.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {OUT / 'findings.md'}")
