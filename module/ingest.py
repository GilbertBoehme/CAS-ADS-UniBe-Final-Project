"""Data ingestion module: read Open-Meteo CSVs, clean columns, add time features."""

import os
import re
import pandas as pd


def _clean_column(col: str) -> str:
    """Normalize a raw Open-Meteo column name to a clean snake_case identifier."""
    col = col.lower().strip()

    # explicit mappings
    col = col.replace("temperature_2m (°c)", "temperature_2m_c")
    col = col.replace("apparent_temperature (°c)", "apparent_temperature_c")
    col = col.replace("wind_direction_10m (°)", "wind_direction_10m_deg")
    col = col.replace("weather_code (wmo code)", "weather_code")
    col = col.replace("visibility (undefined)", "visibility")

    # generic cleanup
    col = col.replace(" ", "_")
    col = col.replace("(", "").replace(")", "")
    col = col.replace("%", "pct")
    col = col.replace("km/h", "kmh")

    col = re.sub(r"[^a-z0-9_]", "", col)
    return col



def ingest(data_dir: str) -> pd.DataFrame:
    """Read all Open-Meteo CSVs from *data_dir*, clean columns, add time features."""
    frames = []

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(data_dir, file)
        print(f"Processing: {file}")

        df = pd.read_csv(file_path, skiprows=3, header=0, encoding="utf-8")
        df.columns = [_clean_column(c) for c in df.columns]

        location = file.replace("_openmeteo.csv", "")
        df["location"] = location

        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear
        df["hour"] = df["time"].dt.hour

        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # Forward-fill NaNs per location (physically plausible for time series)
    # then back-fill any leading NaNs at the start of each location's series
    numeric_cols = df_all.select_dtypes(include="number").columns.tolist()
    df_all = df_all.sort_values(["location", "time"]).reset_index(drop=True)
    df_all[numeric_cols] = df_all.groupby("location")[numeric_cols].transform(
        lambda s: s.ffill().bfill()
    )
    n_remaining = df_all[numeric_cols].isna().sum().sum()
    print(f"NaNs remaining after forward/back-fill: {n_remaining}")

    print(f"\nCombined shape: {df_all.shape}")
    return df_all
