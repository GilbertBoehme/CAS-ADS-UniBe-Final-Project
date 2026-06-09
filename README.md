# Forecasting Natural Hazards in Graubuenden

CAS Applied Data Science final project, University of Bern (2026).

Multi-label binary classification predicting heavy rain and heavy snow events 19-24 hours in advance using hourly meteorological reanalysis data.

## Study Area

Single grid point in the canton of Graubunden, Switzerland. Bounding box: 46.46-46.72 N, 9.28-9.46 E at approx. 1500m elevation. This covers the Viamala/Hinterrhein region, including Thusis, Andeer, Zillis, Sufers, Splugen, and Nufenen.

## Data

- Source: Open-Meteo Historical Weather API (ERA5 reanalysis)
- Period: 1 January 1970 to 1 June 2026 (494,544 hourly observations)
- 27 raw meteorological variables per timestep
- Targets: WMO codes 63/65/66/67/82 (rain), 75/86 (snow)
- Class imbalance: rain 1.6%, snow 3.1% positive rate

## Method

- 24h lookback window, 6h prediction window after 24h gap
- Feature engineering: cyclical time encoding (6 features), rolling stats for 5 variables (15 features), 11 base meteo features = 32 total
- Baseline: Logistic Regression on last timestep only
- Challenger: single-layer LSTM (hidden=64, dropout=0.40, lr=0.0005, batch=32), architecture from 20-trial random search
- BCE loss with positive class weights (~63 for rain, ~30 for snow)
- Threshold optimisation per class on validation set (max F1)
- Train: through 2020 | Val: 2021-2022 | Test: 2023-2026

## Results (test set)

| Hazard | Model | AUC | F1 |
| --- | --- | --- | --- |
| Rain | Logistic Regression | 0.918 | 0.193 |
| Rain | LSTM | 0.935 | 0.283 |
| Snow | Logistic Regression | 0.810 | 0.159 |
| Snow | LSTM | 0.848 | 0.197 |

LSTM macro AUC: 0.892 vs LR baseline: 0.864.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline via the `task/run` notebook (Jupyter or Databricks).

## Dependencies

numpy, pandas, torch, scikit-learn, matplotlib, seaborn, statsmodels
