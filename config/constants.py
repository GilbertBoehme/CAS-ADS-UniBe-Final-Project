"""Central configuration constants for the CAS Weather Hazard Prediction pipeline."""

# Reproducibility
SEED = 42

# Temporal parameters
FUTURE_HOURS = 6        # Future window to check for hazard occurrence
GAP = 24                # Lead time: predict hazard 24h ahead
LOOKBACK = 24           # Sequence length (hours) for LSTM input

# Training parameters
BATCH_SIZE = 64
EPOCHS = 25
PATIENCE = 5

# Tuned hyperparameters (from 20-trial random search, best val macro F1=0.4945)
TUNED_HIDDEN_SIZE = 64
TUNED_NUM_LAYERS = 1
TUNED_DROPOUT = 0.40
TUNED_LR = 0.0005
TUNED_BATCH_SIZE = 32

# Hazard definitions
HAZARD_NAMES = ["rain", "snow"]

WEATHER_CODES = {
    # Clear / partly cloudy / overcast
    0: ("Clear sky", 0),
    1: ("Mainly clear", 0),
    2: ("Partly cloudy", 0),
    3: ("Overcast", 0),
    # Fog
    45: ("Fog", 1),
    48: ("Depositing rime fog", 1),
    # Drizzle (low hazard)
    51: ("Light drizzle", 0),
    53: ("Moderate drizzle", 0),
    55: ("Dense drizzle", 0),
    # Freezing drizzle (hazardous)
    56: ("Light freezing drizzle", 1),
    57: ("Dense freezing drizzle", 1),
    # Rain
    61: ("Slight rain", 0),
    63: ("Moderate rain", 1),
    65: ("Heavy rain", 1),
    # Freezing rain
    66: ("Light freezing rain", 1),
    67: ("Heavy freezing rain", 1),
    # Snow fall
    71: ("Slight snow fall", 0),
    73: ("Moderate snow fall", 0),
    75: ("Heavy snow fall", 1),
    # Snow grains
    77: ("Snow grains", 0),
    # Rain showers
    80: ("Slight rain shower", 0),
    81: ("Moderate rain shower", 0),
    82: ("Violent rain shower", 1),
    # Snow showers
    85: ("Slight snow shower", 0),
    86: ("Heavy snow shower", 1),
    # Thunderstorm
    95: ("Thunderstorm (slight/moderate)", 1),
    # Thunderstorm with hail
    96: ("Thunderstorm with slight hail", 1),
    99: ("Thunderstorm with heavy hail", 1),
}

# Feature lists
# Meteorological features (shared base, before time encoding)
METEO_FEATS = [
    "temperature_2m_c", "apparent_temperature_c", "relative_humidity_2m_pct",
    "precipitation_mm", "rain_mm", "snowfall_cm", "cloud_cover_pct",
    "pressure_msl_hpa", "wind_speed_10m_kmh", "wind_gusts_10m_kmh",
    "wind_direction_10m_deg",
]

# EDA features (raw temporal columns, pre-cyclical encoding)
EDA_FEATS = METEO_FEATS + ["month", "day_of_year", "hour"]

# Model features (after cyclical encoding of time variables)
MODEL_FEATS = METEO_FEATS + [
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "hour_sin", "hour_cos",
]

# Key variables for EDA boxplots and visual comparisons
MAIN_VARS = [
    "precipitation_mm", "temperature_2m_c", "wind_speed_10m_kmh",
    "pressure_msl_hpa", "relative_humidity_2m_pct", "cloud_cover_pct",
]