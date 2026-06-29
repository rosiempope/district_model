"""
generate_demand_figures.py
=============================
Generates a single summary figure (weather + demand) for team
presentations, covering:

  1. Monthly temperature across the three climate scenarios
  2. Monthly heating demand across the three climate scenarios
  3. Monthly cooling demand across the three climate scenarios
  4. Building-by-building annual demand breakdown (baseline year)
  5. A typical winter week's hourly demand shape
  6. A typical summer week's hourly demand shape

Uses ONLY validated, tested pipeline code (climate_scenarios.py,
demand_synthesis.py) against the real London weather_data.csv and the
real Ealing town-centre building mix already used throughout this
project's presets — nothing in this figure is placeholder/illustrative
data. Heating uses heating_kW (space heating only, NOT including DHW) —
DHW doesn't respond to climate at all, so mixing it in would dilute the
"heating drops as it warms" story this figure exists to show.

Usage
-----
    python3 generate_demand_figures.py

Output: figures/weather_demand_summary.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _find_project_root(start: Path) -> Path:
    """
    Walk upward from this file's location until we find a directory that
    looks like the project root (has both 'profiles' and 'components'
    subfolders) — works whether this script lives at the repo root or
    inside a subfolder like figures/.
    """
    current = start
    for _ in range(5):
        if (current / "profiles").is_dir() and (current / "components").is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        f"Could not find the district_model project root (looking for "
        f"'profiles/' and 'components/' folders) starting from {start}. "
        f"Make sure this script lives somewhere inside the district_model repo."
    )

_PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from profiles.climate_scenarios import apply_climate_scenario
from profiles.demand_synthesis import synthesise_network, compute_climate_reference


# ── Styling ──────────────────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "baseline":     "#5B7A8C",   # neutral steel blue
    "2050_central": "#E08E45",   # amber
    "2050_high":    "#C0392B",   # red
}
SCENARIO_LABELS = {
    "baseline":     "Baseline (today)",
    "2050_central": "2050 central (RCP4.5)",
    "2050_high":    "2050 high (RCP8.5)",
}
HEAT_COLOR = "#C0392B"
COOL_COLOR = "#2E86AB"
DHW_COLOR  = "#8D99A6"

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Real Ealing town-centre building mix — the same one used throughout
# this project's presets and self-tests.
EALING_SCENARIO = {
    "demand_nodes": [
        {"name": "Perceval House",       "type": "office",      "floor_area_m2": 8500},
        {"name": "High Street Retail",   "type": "retail",      "floor_area_m2": 3000},
        {"name": "Ealing Hospital Wing", "type": "hospital",    "floor_area_m2": 12000},
        {"name": "Dickens Yard Ph1",     "type": "residential", "units": 350},
        {"name": "Broadway Hotel",       "type": "hotel",       "floor_area_m2": 5000},
        {"name": "Ellen Wilkinson Sch",  "type": "school",      "floor_area_m2": 6000},
    ]
}


# ── Data prep ──────────────────────────────────────────────────────────────────

def load_weather() -> pd.DataFrame:
    weather_df = pd.read_csv(_PROJECT_ROOT / "profiles" / "weather_data.csv")
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    return weather_df.set_index("datetime")


def build_scenario_data(weather_df: pd.DataFrame) -> dict:
    """
    Run all three climate scenarios through demand_synthesis, sharing ONE
    climate_reference (computed from baseline) so heating/cooling totals
    genuinely shift with climate rather than just reshaping across the
    year — see demand_synthesis.py's compute_climate_reference() docstring.
    """
    baseline_weather = apply_climate_scenario(weather_df, "baseline")
    ref = compute_climate_reference(baseline_weather)

    data = {}
    for scenario_name in SCENARIO_LABELS:
        w = apply_climate_scenario(weather_df, scenario_name)
        net = synthesise_network(w, EALING_SCENARIO, climate_reference=ref)
        data[scenario_name] = {"weather": w, "net": net}
    return data


# ── Panel: monthly temperature, 3 scenarios ───────────────────────────────────

def plot_monthly_temperature(ax, scenario_data):
    for scenario_name, d in scenario_data.items():
        monthly = d["weather"]["temp_drybulb_C"].groupby(d["weather"].index.month).mean()
        ax.plot(range(1, 13), monthly.values, marker="o", markersize=4, linewidth=2,
                color=SCENARIO_COLORS[scenario_name], label=SCENARIO_LABELS[scenario_name])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Mean air temperature (°C)")
    ax.set_title("Monthly temperature — climate scenarios", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


# ── Panel: monthly heating/cooling demand, 3 scenarios ────────────────────────

def plot_monthly_demand(ax, scenario_data, key: str, title: str, ylabel: str):
    """key: 'total_heating_kW' or 'total_cooling_kW' from synthesise_network()."""
    width = 0.25
    x = np.arange(12)
    for i, (scenario_name, d) in enumerate(scenario_data.items()):
        series = pd.Series(d["net"][key], index=d["net"]["datetime_index"])
        monthly_MWh = series.groupby(series.index.month).sum() / 1000.0
        ax.bar(x + (i - 1) * width, monthly_MWh.reindex(range(1, 13), fill_value=0).values,
               width=width, color=SCENARIO_COLORS[scenario_name], label=SCENARIO_LABELS[scenario_name])
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


# ── Panel: building-by-building annual breakdown (stacked) ────────────────────

def plot_building_breakdown(ax, baseline_net):
    nodes = baseline_net["nodes"]
    names = [n["name"] for n in nodes]
    heat = np.array([n["annual_heat_kWh"] for n in nodes]) / 1000.0
    dhw  = np.array([n["annual_dhw_kWh"]  for n in nodes]) / 1000.0
    cool = np.array([n["annual_cool_kWh"] for n in nodes]) / 1000.0

    order = np.argsort(-(heat + dhw + cool))
    names = [names[i] for i in order]
    heat, dhw, cool = heat[order], dhw[order], cool[order]

    x = np.arange(len(names))
    ax.bar(x, heat, color=HEAT_COLOR, label="Space heating")
    ax.bar(x, dhw, bottom=heat, color=DHW_COLOR, label="DHW")
    ax.bar(x, cool, bottom=heat + dhw, color=COOL_COLOR, label="Cooling")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Annual demand (MWh)")
    ax.set_title("Demand by building (baseline year)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)


# ── Panel: typical week zoom ───────────────────────────────────────────────────

def plot_week_zoom(ax, net, start_date: str, title: str, color: str):
    idx = net["datetime_index"]
    start = pd.Timestamp(start_date)
    mask = (idx >= start) & (idx < start + pd.Timedelta(days=7))
    total_kW = (net["total_heating_kW"] + net["total_dhw_kW"] + net["total_cooling_kW"])[mask]
    times = idx[mask]

    ax.plot(times, total_kW / 1000.0, color=color, linewidth=1.5)
    ax.fill_between(times, 0, total_kW / 1000.0, color=color, alpha=0.15)
    ax.set_ylabel("Total demand (MW)")
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    ax.xaxis.set_major_locator(mdates.DayLocator())


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading weather data and running demand synthesis across 3 climate scenarios...")
    weather_df = load_weather()
    scenario_data = build_scenario_data(weather_df)
    baseline_net = scenario_data["baseline"]["net"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle("Ealing Town Centre District Energy — Weather & Demand Overview",
                 fontsize=14, fontweight="bold", y=0.995)

    plot_monthly_temperature(axes[0, 0], scenario_data)
    plot_monthly_demand(axes[0, 1], scenario_data, "total_heating_kW",
                         "Monthly heating demand — climate scenarios", "Heating demand (MWh)")
    plot_monthly_demand(axes[1, 0], scenario_data, "total_cooling_kW",
                         "Monthly cooling demand — climate scenarios", "Cooling demand (MWh)")
    plot_building_breakdown(axes[1, 1], baseline_net)
    plot_week_zoom(axes[2, 0], baseline_net, "2023-01-16", "Typical winter week (16-22 Jan)", HEAT_COLOR)
    plot_week_zoom(axes[2, 1], baseline_net, "2023-07-17", "Typical summer week (17-23 Jul)", COOL_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_dir = _PROJECT_ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "weather_demand_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()