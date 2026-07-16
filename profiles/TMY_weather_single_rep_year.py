#This file contains the 8760 hourly temperature values for London Heathrow based on a representative single year from 2011 to 2025.
#What this means is that it is more of a average and will ignore the extreme weather events. Like 2022 heat wave & extreme colds. This is useful for conservative revenue estimates
#But for pipe diameter and peak load on the network we must use the extreme ends for design. So a seperate data set: 'Design Summer Year (DSY) or Design Winter Year — CIBSE
#will be used for this.

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

#Parse the EPW file for London, UK

# The source EPW ships alongside this script, so the weather pipeline is
# reproducible without hunting for the input. This constant was previously
# commented out ("INCLUDE THE FILE HERE (single use)"), which left main()
# referencing an undefined name — i.e. this script could not run at all, and
# profiles/weather_data.csv could not be regenerated from its source. Pass a
# different EPW as argv[1] to rebuild for another site.
DEFAULT_EPW_PATH = str(
    Path(__file__).resolve().parent
    / "GBR_ENG_London-Heathrow.Intl.AP.037720_TMYx.2011-2025.epw"
)

# HDD/CDD base temperatures (degrees C)
HEATING_BASE_TEMP = 18.0   # UK standard
COOLING_BASE_TEMP = 21.0   # Reasonable for UK commercial

# Output file paths — written next to this script, i.e. profiles/, which is
# where scenario_runner.load_weather() reads weather_data.csv from.
OUTPUT_HOURLY_CSV  = str(Path(__file__).resolve().parent / "weather_data.csv")
OUTPUT_SUMMARY_CSV = str(Path(__file__).resolve().parent / "weather_summary.csv")

EPW_COLUMNS = {
    # (column_index, output_name, description, unit)
    "year":         (0,  "year",         "Year",                          "-"),
    "month":        (1,  "month",        "Month (1-12)",                  "-"),
    "day":          (2,  "day",          "Day of month",                  "-"),
    "hour":         (3,  "hour",         "Hour of day (0-23)",            "-"),
    # -- Temperature
    "temp_drybulb": (6,  "temp_drybulb_C",  "Dry bulb temperature",      "°C"),
    "temp_dewpoint":(7,  "temp_dewpoint_C", "Dew point temperature",      "°C"),
    # -- Humidity
    "rel_humidity": (8,  "rel_humidity_pct","Relative humidity",          "%"),
    # -- Pressure
    "pressure":     (9,  "pressure_Pa",  "Atmospheric pressure",          "Pa"),
    # -- Solar irradiance
    "ghi":          (13, "ghi_Wh_m2",   "Global horizontal irradiance",  "Wh/m²"),
    "dni":          (14, "dni_Wh_m2",   "Direct normal irradiance",      "Wh/m²"),
    "dhi":          (15, "dhi_Wh_m2",   "Diffuse horizontal irradiance", "Wh/m²"),
    # -- Wind
    "wind_dir":     (20, "wind_dir_deg", "Wind direction (0=N, 90=E)",   "°"),
    "wind_speed":   (21, "wind_speed_ms","Wind speed",                    "m/s"),
    # -- Sky cover
    "sky_cover":    (22, "sky_cover_tenths",  "Total sky cover",         "/10"),
    "sky_opaque":   (23, "sky_opaque_tenths", "Opaque sky cover",        "/10"),
}

def parse_epw(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    Parse an EPW file and return metadata dict and hourly DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EPW file not found: {filepath}")
 
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Extract metadata from header lines
    loc_fields = lines[0].strip().split(",")
    metadata = {
        "city":       loc_fields[1].strip() if len(loc_fields) > 1 else "Unknown",
        "state":      loc_fields[2].strip() if len(loc_fields) > 2 else "",
        "country":    loc_fields[3].strip() if len(loc_fields) > 3 else "Unknown",
        "source":     loc_fields[4].strip() if len(loc_fields) > 4 else "",
        "wmo_number": loc_fields[5].strip() if len(loc_fields) > 5 else "",
        "latitude":   float(loc_fields[6]) if len(loc_fields) > 6 else None,
        "longitude":  float(loc_fields[7]) if len(loc_fields) > 7 else None,
        "timezone":   float(loc_fields[8]) if len(loc_fields) > 8 else None,
        "elevation_m":float(loc_fields[9]) if len(loc_fields) > 9 else None,
    }
 
    # Try to extract TMY period from typical periods header (line 5, index 4)
    period_line = lines[4].strip() if len(lines) > 4 else ""
    metadata["tmy_period_raw"] = period_line
 
    # ── Parse data rows (lines 8 onwards) ─────────────────────────────────────
    data_lines = lines[8:]  # First 8 lines are header
 
    rows = []
    for i, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue
        cols = line.split(",")
        if len(cols) < 24:
            continue  # Skip malformed rows
 
        try:
            row = {}
            for key, (col_idx, out_name, _, _) in EPW_COLUMNS.items():
                raw = cols[col_idx].strip()
                # EPW uses 9999/99999 as missing data flags — replace with NaN
                val = float(raw)
                if val in (9999.0, 99999.0, 999.0):
                    val = np.nan
                row[out_name] = val
 
            # EPW hours run 1-24; convert to 0-23
            row["hour"] = int(row["hour"]) - 1
 
            rows.append(row)
        except (ValueError, IndexError):
            continue  # Skip any unparseable rows
 
    df = pd.DataFrame(rows)
 
    if len(df) == 0:
        raise ValueError("No valid data rows found in EPW file.")
 
    # ── Derived columns ────────────────────────────────────────────────────────
 
    # Datetime index (use a generic non-leap year — TMY has no fixed year)
    # Month/day/hour are real; year is set to 2023 for indexing convenience
    df["datetime"] = pd.to_datetime({
        "year":  2023,
        "month": df["month"].astype(int),
        "day":   df["day"].astype(int),
        "hour":  df["hour"].astype(int),
    })
    df = df.set_index("datetime")
 
    # Day of year (1-365)
    df["day_of_year"] = df.index.dayofyear
 
    # Heating and cooling degree hours → degree days
    # Clipped at 0 so negative differences don't contribute
    df["HDD_h"] = (HEATING_BASE_TEMP - df["temp_drybulb_C"]).clip(lower=0) / 24
    df["CDD_h"] = (df["temp_drybulb_C"] - COOLING_BASE_TEMP).clip(lower=0) / 24
 
    # Pressure in kPa (more intuitive alongside temperature)
    df["pressure_kPa"] = df["pressure_Pa"] / 1000
 
    # Simple sky clearness index (0=overcast, 1=clear)
    df["sky_clearness"] = 1 - df["sky_cover_tenths"] / 10
 
    return metadata, df
 
 
# ── Summary statistics ─────────────────────────────────────────────────────────
 
def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to monthly statistics.
    Returns a DataFrame with one row per month.
    """
    month_names = {
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
    }
 
    summary_rows = []
    for mo in range(1, 13):
        m = df[df["month"] == mo]
        row = {
            "month":            mo,
            "month_name":       month_names[mo],
            "temp_mean_C":      round(m["temp_drybulb_C"].mean(), 2),
            "temp_min_C":       round(m["temp_drybulb_C"].min(), 2),
            "temp_max_C":       round(m["temp_drybulb_C"].max(), 2),
            "dewpoint_mean_C":  round(m["temp_dewpoint_C"].mean(), 2),
            "rh_mean_pct":      round(m["rel_humidity_pct"].mean(), 1),
            "pressure_mean_kPa":round(m["pressure_kPa"].mean(), 2),
            "GHI_total_kWh_m2": round(m["ghi_Wh_m2"].sum() / 1000, 1),
            "DNI_total_kWh_m2": round(m["dni_Wh_m2"].sum() / 1000, 1),
            "DHI_total_kWh_m2": round(m["dhi_Wh_m2"].sum() / 1000, 1),
            "wind_speed_mean_ms":round(m["wind_speed_ms"].mean(), 2),
            "sky_cover_mean":   round(m["sky_cover_tenths"].mean(), 1),
            "HDD_total":        round(m["HDD_h"].sum(), 1),
            "CDD_total":        round(m["CDD_h"].sum(), 1),
        }
        summary_rows.append(row)
 
    summary = pd.DataFrame(summary_rows)
 
    # Totals row
    totals = {
        "month": 0,
        "month_name": "ANNUAL",
        "temp_mean_C":       round(df["temp_drybulb_C"].mean(), 2),
        "temp_min_C":        round(df["temp_drybulb_C"].min(), 2),
        "temp_max_C":        round(df["temp_drybulb_C"].max(), 2),
        "dewpoint_mean_C":   round(df["temp_dewpoint_C"].mean(), 2),
        "rh_mean_pct":       round(df["rel_humidity_pct"].mean(), 1),
        "pressure_mean_kPa": round(df["pressure_kPa"].mean(), 2),
        "GHI_total_kWh_m2":  round(df["ghi_Wh_m2"].sum() / 1000, 1),
        "DNI_total_kWh_m2":  round(df["dni_Wh_m2"].sum() / 1000, 1),
        "DHI_total_kWh_m2":  round(df["dhi_Wh_m2"].sum() / 1000, 1),
        "wind_speed_mean_ms":round(df["wind_speed_ms"].mean(), 2),
        "sky_cover_mean":    round(df["sky_cover_tenths"].mean(), 1),
        "HDD_total":         round(df["HDD_h"].sum(), 1),
        "CDD_total":         round(df["CDD_h"].sum(), 1),
    }
    summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)
    return summary
 
 
# ── Pretty printer ─────────────────────────────────────────────────────────────
 
def print_metadata(metadata: dict):
    print("\n" + "="*60)
    print("  EPW FILE METADATA")
    print("="*60)
    print(f"  Location  : {metadata['city']}, {metadata['country']}")
    print(f"  WMO No.   : {metadata['wmo_number']}")
    print(f"  Latitude  : {metadata['latitude']}°")
    print(f"  Longitude : {metadata['longitude']}°")
    print(f"  Elevation : {metadata['elevation_m']} m")
    print(f"  Timezone  : UTC{metadata['timezone']:+.1f}")
    print(f"  Source    : {metadata['source']}")
    print("="*60)
 
 
def print_summary(df: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "="*60)
    print("  HOURLY DATA OVERVIEW")
    print("="*60)
    print(f"  Rows loaded       : {len(df):,}  (expected 8,760)")
    print(f"  Columns extracted : {len(df.columns)}")
    print(f"  Date range        : {df.index[0].strftime('%d %b')} → "
          f"{df.index[-1].strftime('%d %b %Y')}")
    print(f"  Missing values    : {df.isnull().sum().sum()}")
 
    annual = summary[summary["month_name"] == "ANNUAL"].iloc[0]
    print(f"\n  Annual HDD (base {HEATING_BASE_TEMP}°C) : {annual['HDD_total']:.0f} °C·days")
    print(f"  Annual CDD (base {COOLING_BASE_TEMP}°C) : {annual['CDD_total']:.0f} °C·days")
    print(f"  Mean dry bulb temp          : {annual['temp_mean_C']:.1f} °C")
    print(f"  Min / Max temp              : {annual['temp_min_C']:.1f} / {annual['temp_max_C']:.1f} °C")
    print(f"  Annual GHI                  : {annual['GHI_total_kWh_m2']:.0f} kWh/m²")
    print(f"  Mean wind speed             : {annual['wind_speed_mean_ms']:.1f} m/s")
    print("="*60)
 
    print("\n  MONTHLY SUMMARY")
    print("-"*60)
    display_cols = ["month_name","temp_mean_C","temp_min_C","temp_max_C",
                    "rh_mean_pct","HDD_total","CDD_total","GHI_total_kWh_m2"]
    print(summary[display_cols].to_string(index=False))
    print("="*60 + "\n")
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    epw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EPW_PATH
 
    print(f"\nParsing: {epw_path}")
    metadata, df = parse_epw(epw_path)
 
    summary = build_monthly_summary(df)
 
    print_metadata(metadata)
    print_summary(df, summary)
 
    # Save outputs
    df.to_csv(OUTPUT_HOURLY_CSV)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
 
    print(f"  Saved hourly data  → {OUTPUT_HOURLY_CSV}")
    print(f"  Saved monthly summary → {OUTPUT_SUMMARY_CSV}\n")
 
    return df, summary, metadata
 
 
if __name__ == "__main__":
    df, summary, metadata = main() 
