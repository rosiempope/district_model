"""Presentation-ready reference table: what is actually in each archetype.

    python -m analysis.archetype_reference_table

Every Dalkia-pack study (screening matrix, GHNF affordability, source
frontier, climate sweep) runs the SAME three density archetypes, imported
from analysis/archetypes.py. This script makes that composition legible for
the deck: which buildings, what type, what floor area, how many connections,
when they connect, how likely they are to connect, and what route length the
archetype is tested against.

Writes CSVs and a presentation-ready PNG table to
output/archetype_reference_table/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "archetype_reference_table"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.archetypes import ARCHETYPES
from profiles.demand_synthesis import BUILDING_TYPES

# ── Palette (validated categorical set, see dataviz skill) ──────────────────
C_BLUE, C_AQUA, C_YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ARCH_TINT = {
    "Dense (town centre)": "#eaf1fb",
    "Middle (suburban mixed)": "#e7f8f1",
    "Scarce (low-density edge)": "#fdf3e0",
}
ARCH_ACCENT = {
    "Dense (town centre)": C_BLUE,
    "Middle (suburban mixed)": C_AQUA,
    "Scarce (low-density edge)": C_YELLOW,
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-building detail
# ═══════════════════════════════════════════════════════════════════════════

rows = []
for arch_label, cfg in ARCHETYPES.items():
    for b in cfg["buildings"]:
        btype = b["type"]
        bench = BUILDING_TYPES[btype]
        rows.append({
            "Archetype": arch_label,
            "Building": b["name"],
            "Type": btype,
            "Type description": bench["description"],
            "Floor area (m²)": b.get("floor_area_m2"),
            "Dwelling units": b.get("units"),
            "Connections": b.get("connections", 1),
            "Connection year": b.get("connection_year", 1),
            "Connection probability": b.get("connection_probability", 1.0),
            "Heat benchmark (kWh/m²/yr)": round(bench["heat_kWh_m2"], 1),
            "Cool benchmark (kWh/m²/yr)": round(bench["cool_kWh_m2"], 1),
        })
detail_df = pd.DataFrame(rows)
detail_df.to_csv(OUT / "archetype_buildings.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Archetype-level summary
# ═══════════════════════════════════════════════════════════════════════════

summary_rows = []
for arch_label, cfg in ARCHETYPES.items():
    buildings = cfg["buildings"]
    total_area = sum(b.get("floor_area_m2", 0) for b in buildings)
    total_conn = sum(b.get("connections", 1) for b in buildings)
    total_units = sum(b.get("units", 0) for b in buildings)
    anchor_buildings = [b for b in buildings if b["type"] not in
                        {"residential", "residential_existing"}]
    summary_rows.append({
        "Archetype": arch_label,
        "Buildings": len(buildings),
        "Anchor (non-residential) buildings": len(anchor_buildings),
        "Total floor area (m²)": total_area,
        "Residential dwelling units": total_units,
        "Total connections": total_conn,
        "Route length (m)": cfg["route_m"],
        "Floor area per metre of route (m²/m)": round(total_area / cfg["route_m"], 1),
        "Note": cfg.get("note", ""),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "archetype_summary.csv", index=False)

print("=== Archetype summary ===")
print(summary_df.to_string(index=False))
print("\n=== Per-building detail ===")
print(detail_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 3. Presentation-ready PNG table — one figure, grouped by archetype
# ═══════════════════════════════════════════════════════════════════════════

COLS = ["Building", "Type", "Floor area (m²)", "Dwelling\nunits",
        "Connections", "Conn.\nyear", "Conn.\nprob."]

cell_rows, cell_colours, row_labels = [], [], []
for arch_label in ARCHETYPES:
    cfg = ARCHETYPES[arch_label]
    accent = ARCH_ACCENT[arch_label]
    tint = ARCH_TINT[arch_label]
    header = [arch_label, "", "", "", "", "", ""]
    cell_rows.append(header)
    cell_colours.append([accent] * len(COLS))
    row_labels.append("header")
    for b in cfg["buildings"]:
        cell_rows.append([
            b["name"], b["type"],
            f"{b.get('floor_area_m2', 0):,.0f}" if b.get("floor_area_m2") else "—",
            f"{b.get('units', '')}" if b.get("units") else "—",
            f"{b.get('connections', 1):,}",
            f"{b.get('connection_year', 1)}",
            f"{b.get('connection_probability', 1.0):.0%}",
        ])
        cell_colours.append([tint] * len(COLS))
        row_labels.append("data")

COL_WIDTHS = [0.24, 0.15, 0.13, 0.10, 0.11, 0.09, 0.09]  # sums to 0.91, leaves margin

fig_height = 0.38 * len(cell_rows) + 0.6
fig, ax = plt.subplots(figsize=(13.5, fig_height))
ax.axis("off")
ax.set_title("Archetype reference — building composition and connection assumptions",
             fontsize=13, pad=14)
table = ax.table(cellText=cell_rows, cellColours=cell_colours,
                 colLabels=COLS, colWidths=COL_WIDTHS, cellLoc="left", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 1.65)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.6)
    cell.PAD = 0.02
    if row == 0:
        cell.set_text_props(weight="bold", color="white", ha="left")
        cell.set_facecolor(INK)
        continue
    data_idx = row - 1
    if row_labels[data_idx] == "header":
        cell.set_text_props(weight="bold", color="white", ha="left")
        if col > 0:
            cell.get_text().set_text("")
    else:
        cell.set_text_props(color=INK, ha="left" if col < 2 else "center")

fig.savefig(OUT / "AR1_archetype_buildings_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 4. Summary PNG table (short — the one to actually put on a slide)
# ═══════════════════════════════════════════════════════════════════════════

SUM_COLS = ["Archetype", "Buildings", "Anchor\nbuildings", "Total floor\narea (m²)",
            "Dwelling\nunits", "Connections", "Route\n(m)", "Floor area /\nroute m"]
sum_rows, sum_colours = [], []
for r in summary_rows:
    sum_rows.append([
        r["Archetype"], f"{r['Buildings']}", f"{r['Anchor (non-residential) buildings']}",
        f"{r['Total floor area (m²)']:,.0f}", f"{r['Residential dwelling units']:,}",
        f"{r['Total connections']:,}", f"{r['Route length (m)']:,.0f}",
        f"{r['Floor area per metre of route (m²/m)']:,.1f}",
    ])
    sum_colours.append([ARCH_TINT[r["Archetype"]]] * len(SUM_COLS))

SUM_COL_WIDTHS = [0.25, 0.09, 0.10, 0.14, 0.10, 0.12, 0.09, 0.12]  # sums to ~1.01

fig, ax = plt.subplots(figsize=(13.5, 2.6))
ax.axis("off")
ax.set_title("Archetype summary — density and scale", fontsize=13, pad=14)
table = ax.table(cellText=sum_rows, cellColours=sum_colours,
                 colLabels=SUM_COLS, colWidths=SUM_COL_WIDTHS, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 2.0)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_linewidth(0.6)
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor(INK)
    else:
        cell.set_text_props(color=INK)
        if col == 0:
            cell.set_text_props(weight="bold", color=INK, ha="left")
fig.savefig(OUT / "AR2_archetype_summary_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# findings.md
# ═══════════════════════════════════════════════════════════════════════════

lines = [
    "# Archetype reference — what is actually in each scheme",
    "",
    "Generated by `python -m analysis.archetype_reference_table`. Definitions are the",
    "canonical `ARCHETYPES` dict in `analysis/archetypes.py`, imported unchanged by",
    "every study in this pack (screening matrix, GHNF affordability, source frontier,",
    "climate sweep) — this table can never drift from what was actually run.",
    "",
    "Route lengths are illustrative placeholders reflecting typical relative spacing",
    "(dense/middle/scarce), not measured from a real map. The Exeter case studies",
    "(`analysis/exeter_*.py`) use real measured tree-topology route segments instead.",
    "",
    "## Archetype summary",
    "",
    summary_df.to_markdown(index=False),
    "",
    "## Per-building detail",
    "",
    detail_df.to_markdown(index=False),
]
(OUT / "findings.md").write_text("\n".join(lines))
print(f"\nWrote {OUT}/findings.md and 2 table figures.")
