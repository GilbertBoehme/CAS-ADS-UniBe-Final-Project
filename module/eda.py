"""Exploratory Data Analysis module: hazard labelling, feature analysis, visualizations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf

from config.constants import (
    FUTURE_HOURS, GAP, HAZARD_NAMES, EDA_FEATS as FEATS, MAIN_VARS,
)


# Hazard label construction
def get_hazard_categories(code: int) -> tuple:
    """Map WMO weather code to binary (rain, snow) hazard indicators."""
    rain = 1 if code in [63, 65, 66, 67, 82] else 0
    snow = 1 if code in [75, 86] else 0
    return rain, snow


def build_hazard_labels(df: pd.DataFrame, future_hours: int = FUTURE_HOURS, gap: int = GAP) -> pd.DataFrame:
    """Add hazard_now_* and hazard_future_* columns; trim incomplete future windows.

    Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()

    df[["hazard_now_rain", "hazard_now_snow"]] = df["weather_code"].apply(
        lambda c: pd.Series(get_hazard_categories(c))
    )

    for hazard in HAZARD_NAMES:
        df[f"hazard_future_{hazard}"] = (
            df.groupby("location")[f"hazard_now_{hazard}"]
            .shift(-gap)
            .rolling(future_hours, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .fillna(0)
            .astype(int)
        )

    # Remove rows where future window can't be fully constructed
    df_clean = df.groupby("location", group_keys=False).apply(
        lambda g: g.iloc[:-(future_hours + gap)] if len(g) > (future_hours + gap) else g
    ).reset_index(drop=True)

    return df_clean


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------
def plot_class_distribution(df_loc: pd.DataFrame) -> None:
    """Plot class counts, hourly and monthly hazard rates."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, hazard in enumerate(HAZARD_NAMES):
        target = f"hazard_future_{hazard}"
        counts = df_loc[target].value_counts()
        ax = axes[idx, 0]
        ax.bar(counts.index.astype(str), counts.values, color=["steelblue", "darkorange"])
        ax.set_title(f"{hazard.upper()} \u2013 Count")
        ax.set_ylabel("Samples")

        ax = axes[idx, 1]
        df_loc.groupby("hour")[target].mean().plot(ax=ax, marker="o")
        ax.set_title(f"{hazard.upper()} \u2013 Hazard rate by hour")
        ax.set_ylabel("Probability")

        ax = axes[idx, 2]
        df_loc.groupby("month")[target].mean().plot(ax=ax, marker="o")
        ax.set_title(f"{hazard.upper()} \u2013 Hazard rate by month")
    plt.tight_layout()
    plt.show()


def plot_event_durations(df_loc: pd.DataFrame) -> None:
    """Histogram of consecutive-hour event durations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, hazard in enumerate(HAZARD_NAMES):
        s = df_loc[f"hazard_future_{hazard}"]
        groups = (s != s.shift()).cumsum()
        durations = s.groupby(groups).agg(["size", "first"])
        durations = durations[durations["first"] == 1]["size"]
        axes[idx].hist(durations, bins=30, edgecolor="k")
        axes[idx].set_title(f"{hazard.upper()} \u2013 Duration (hours)")
        axes[idx].set_xlabel("Hours")
    plt.tight_layout()
    plt.show()


def print_hazard_overlap(df_loc: pd.DataFrame) -> None:
    """Print rain/snow overlap counts."""
    both = (df_loc["hazard_future_rain"] & df_loc["hazard_future_snow"]).sum()
    rain_only = (df_loc["hazard_future_rain"] & ~df_loc["hazard_future_snow"]).sum()
    snow_only = (~df_loc["hazard_future_rain"] & df_loc["hazard_future_snow"]).sum()
    print(f"Overlap: Both={both}, Rain only={rain_only}, Snow only={snow_only}")


def plot_boxplots(df_loc: pd.DataFrame) -> None:
    """Boxplots of main features split by hazard class."""
    fig, axes = plt.subplots(len(MAIN_VARS), 2, figsize=(14, 2.5 * len(MAIN_VARS)))
    for i, var in enumerate(MAIN_VARS):
        for j, hazard in enumerate(HAZARD_NAMES):
            sns.boxplot(x=f"hazard_future_{hazard}", y=var, data=df_loc, ax=axes[i][j])
            axes[i][j].set_title(f"{var} vs {hazard}")
    plt.tight_layout()
    plt.show()


def plot_scatter_hazard_status(df_loc: pd.DataFrame, n_sample: int = 2000) -> None:
    """Scatter: precipitation vs wind gusts colored by hazard status."""
    df_sample = df_loc.sample(min(n_sample, len(df_loc)))

    def status(row):
        r, s = row["hazard_future_rain"], row["hazard_future_snow"]
        if r and s:
            return "Both"
        elif r:
            return "Rain only"
        elif s:
            return "Snow only"
        return "None"

    df_sample["hazard_status"] = df_sample.apply(status, axis=1)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_sample, x="precipitation_mm", y="wind_gusts_10m_kmh",
        hue="hazard_status", style="hazard_status", alpha=0.6,
    )
    plt.title("Precipitation vs Wind gusts \u2013 hazard status")
    plt.show()


def plot_correlations(df_loc: pd.DataFrame) -> None:
    """Bar chart of feature correlations with each target."""
    valid_feats = [f for f in FEATS if f in df_loc.columns
                   and df_loc[f].notna().any() and df_loc[f].std() > 0]
    corr_rain = df_loc[valid_feats].corrwith(df_loc["hazard_future_rain"]).sort_values(ascending=False)
    corr_snow = df_loc[valid_feats].corrwith(df_loc["hazard_future_snow"]).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    corr_rain.plot.bar(ax=axes[0], color="steelblue")
    axes[0].set_title("Correlation with future RAIN")
    axes[0].set_ylabel("Correlation")
    corr_snow.plot.bar(ax=axes[1], color="darkorange")
    axes[1].set_title("Correlation with future SNOW")
    plt.tight_layout()
    plt.show()


def plot_pacf_hazards(df_loc: pd.DataFrame, lags: int = 48) -> None:
    """Partial autocorrelation of hazard targets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for idx, hazard in enumerate(HAZARD_NAMES):
        plot_pacf(df_loc[f"hazard_future_{hazard}"].astype(float), lags=lags, ax=axes[idx], title=f"PACF of {hazard}")
    plt.tight_layout()
    plt.show()


def plot_cross_correlation(df_loc: pd.DataFrame, max_lag: int = 24) -> None:
    """Cross-correlation heatmaps for each hazard target."""
    lags = np.arange(-max_lag, max_lag + 1)

    # Filter out features that are entirely NaN or constant
    valid_feats = [f for f in FEATS if f in df_loc.columns
                   and df_loc[f].notna().any() and df_loc[f].std() > 0]

    for hazard in HAZARD_NAMES:
        target = f"hazard_future_{hazard}"
        corr_mat = np.zeros((len(valid_feats), len(lags)))
        for i, ft in enumerate(valid_feats):
            x = df_loc[ft].values
            y = df_loc[target].values
            vals = []
            for lag in lags:
                if lag < 0:
                    vals.append(np.corrcoef(x[:lag], y[-lag:])[0, 1])
                elif lag > 0:
                    vals.append(np.corrcoef(x[lag:], y[:-lag])[0, 1])
                else:
                    vals.append(np.corrcoef(x, y)[0, 1])
            corr_mat[i, :] = vals

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            corr_mat, xticklabels=lags, yticklabels=valid_feats, cmap="RdBu", center=0,
            cbar_kws={"label": "Cross-correlation"},
        )
        plt.title(f"Cross-correlation with {hazard.upper()} (negative lag = feature leads)")
        plt.xlabel("Lag (hours)")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()


def print_data_quality(df_loc: pd.DataFrame) -> pd.DataFrame:
    """Print missing values and return descriptive statistics."""
    missing = df_loc[FEATS].isna().sum()
    missing_pct = (missing / len(df_loc)) * 100
    print("Missing values (%) per feature:")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    desc = df_loc[FEATS].describe().T
    print("\nFeature statistics:")
    print(desc[["min", "max", "mean", "std"]])
    return desc


def plot_multi_location(df_clean: pd.DataFrame) -> None:
    """Hazard rate distribution across locations and top features per station."""
    target_cols = [f"hazard_future_{h}" for h in HAZARD_NAMES]

    if df_clean["location"].nunique() <= 1:
        print("Only one location \u2013 skipping multi-location comparison.")
        return

    loc_rates = df_clean.groupby("location")[target_cols].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, hazard in enumerate(HAZARD_NAMES):
        ax = axes[idx]
        loc_rates[f"hazard_future_{hazard}"].hist(bins=20, edgecolor="k", ax=ax)
        ax.set_title(f"{hazard.upper()} rate across locations")
        ax.set_xlabel("Hazard rate")
    plt.tight_layout()
    plt.show()

    print("\nTop 3 features per location (first 5 stations):")
    for loc_name in df_clean["location"].unique()[:5]:
        d = df_clean[df_clean["location"] == loc_name]
        print(f"\n{loc_name}:")
        for hazard in HAZARD_NAMES:
            corrs = d[FEATS].corrwith(d[f"hazard_future_{hazard}"]).sort_values(ascending=False)
            print(f"  {hazard}: {corrs.head(3).to_dict()}")


# Orchestrator
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full EDA pipeline. Returns the cleaned DataFrame with hazard labels."""
    plt.rcParams["figure.figsize"] = (12, 6)
    sns.set_style("whitegrid")

    print("=" * 60)
    print("Building hazard labels...")
    print("=" * 60)
    df_clean = build_hazard_labels(df)
    print(f"Clean dataset: {df_clean.shape}")

    # Single-location detailed EDA
    loc = df_clean["location"].iloc[0]
    df_loc = df_clean[df_clean["location"] == loc].copy().sort_values("time")
    print(f"\nLocation for detailed EDA: {loc}")

    print("\n--- Class distribution & temporal rates ---")
    plot_class_distribution(df_loc)

    print("\n--- Event durations ---")
    plot_event_durations(df_loc)

    print("\n--- Hazard overlap ---")
    print_hazard_overlap(df_loc)

    print("\n--- Feature boxplots ---")
    plot_boxplots(df_loc)

    print("\n--- Scatter: precipitation vs wind gusts ---")
    plot_scatter_hazard_status(df_loc)

    print("\n--- Correlations ---")
    plot_correlations(df_loc)

    print("\n--- PACF ---")
    plot_pacf_hazards(df_loc)

    print("\n--- Cross-correlation heatmaps ---")
    plot_cross_correlation(df_loc)

    print("\n--- Data quality ---")
    print_data_quality(df_loc)

    print("\n--- Multi-location comparison ---")
    plot_multi_location(df_clean)

    return df_clean
