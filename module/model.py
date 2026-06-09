"""Model module: Baseline (Logistic Regression) + Challenger (LSTM)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, f1_score,
)
import joblib

from config.constants import (
    SEED, FUTURE_HOURS, GAP, LOOKBACK, BATCH_SIZE, EPOCHS, PATIENCE,
    HAZARD_NAMES, MODEL_FEATS as FEATS,
    TUNED_HIDDEN_SIZE, TUNED_NUM_LAYERS, TUNED_DROPOUT, TUNED_LR, TUNED_BATCH_SIZE,
)
from module.model_plots import (
    plot_loss_curves,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_threshold_analysis,
    plot_cv_macro_auc,
)

# Reproducibility
def _set_seed(seed: int = SEED) -> None:
    """Reset all random states for full reproducibility across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Loss functions
class FocalLoss(nn.Module):
    """Binary Focal Loss for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from raw logits."""
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # alpha weighting: alpha for positives, (1 - alpha) for negatives
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # BCE component (numerically stable via logsigmoid)
        bce = -targets * F.logsigmoid(logits) - (1 - targets) * F.logsigmoid(-logits)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Model architecture
class MultiHazardLSTM(nn.Module):
    """Multi-label LSTM for rain/snow hazard prediction."""

    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# Feature engineering
def _get_hazard_categories(code: int) -> tuple[int, int]:
    rain = 1 if code in [63, 65, 66, 67, 82] else 0
    snow = 1 if code in [75, 86] else 0
    return rain, snow


def _encode_cyclical(df: pd.DataFrame, col: str, period: int) -> None:
    rad = 2 * np.pi * df[col] / period
    df[f"{col}_sin"] = np.sin(rad)
    df[f"{col}_cos"] = np.cos(rad)
    df.drop(columns=col, inplace=True)


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add hazard labels, cyclical time encoding; return cleaned DataFrame for modelling."""
    df = df.copy()

    df[["hazard_now_rain", "hazard_now_snow"]] = df["weather_code"].apply(
        lambda c: pd.Series(_get_hazard_categories(c))
    )

    for hazard in HAZARD_NAMES:
        df[f"hazard_future_{hazard}"] = (
            df.groupby("location")[f"hazard_now_{hazard}"]
            .shift(-GAP)
            .rolling(FUTURE_HOURS, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .fillna(0)
            .astype(int)
        )

    df_clean = df.groupby("location", group_keys=False).apply(
        lambda g: g.iloc[:-(FUTURE_HOURS + GAP)] if len(g) > (FUTURE_HOURS + GAP) else g
    ).reset_index(drop=True)

    _encode_cyclical(df_clean, "month", 12)
    _encode_cyclical(df_clean, "day_of_year", 365)
    _encode_cyclical(df_clean, "hour", 24)

    return df_clean


# Feature engineering: rolling statistics for improved F1
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features that capture temporal trends."""
    df = df.copy()
    trend_vars = ["pressure_msl_hpa", "relative_humidity_2m_pct",
                  "cloud_cover_pct", "precipitation_mm", "wind_gusts_10m_kmh"]

    for col in trend_vars:
        if col not in df.columns:
            continue
        # 6h and 12h rolling mean
        df[f"{col}_roll6"] = df.groupby("location")[col].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
        df[f"{col}_roll12"] = df.groupby("location")[col].transform(
            lambda s: s.rolling(12, min_periods=1).mean()
        )
        # 6h change (tendency)
        df[f"{col}_diff6"] = df.groupby("location")[col].transform(
            lambda s: s.diff(6)
        )
    # Fill any NaN introduced by diff/rolling
    df = df.fillna(0)
    return df


# Sequence creation
def make_sequences(df: pd.DataFrame, lookback: int = LOOKBACK, stride: int = 1,
                   feats: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create LSTM input sequences grouped by location."""
    if feats is None:
        feats = FEATS
    target_cols = [f"hazard_future_{h}" for h in HAZARD_NAMES]
    X, y = [], []
    for _, grp in df.groupby("location"):
        grp = grp.sort_values("time")
        data = grp[feats].values
        targs = grp[target_cols].values
        for i in range(0, len(data) - lookback, stride):
            X.append(data[i:i + lookback])
            y.append(targs[i + lookback - 1])
    return np.array(X), np.array(y)


# Training loop
def _train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    input_dim: int | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    loss_fn: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    save_path: str = "best_model.pth",
) -> tuple[nn.Module, list[float], list[float]]:
    """Train LSTM and return (model, train_losses, val_losses).

    Parameters
    ----------
    loss_fn : str
        'bce' for BCEWithLogitsLoss (with pos_weight), or
        'focal' for FocalLoss (alpha/gamma controlled separately).
    focal_alpha : float
        Focal Loss alpha (positive-class weight). Only used when loss_fn='focal'.
    focal_gamma : float
        Focal Loss gamma (focusing strength). Only used when loss_fn='focal'.
    """
    if input_dim is None:
        input_dim = len(FEATS)

    model = MultiHazardLSTM(
        input_dim=input_dim, num_classes=len(HAZARD_NAMES),
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
    ).to(device)

    if loss_fn == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"  Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Using BCEWithLogitsLoss (pos_weight={pos_weight.cpu().numpy()})")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("  Early stopping triggered")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, train_losses, val_losses


# Threshold optimization
def _optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on binary predictions."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def cross_validate(df_clean: pd.DataFrame, device: torch.device, n_splits: int = 4,
                   feats: list[str] | None = None, batch_size: int = BATCH_SIZE,
                   hidden_size: int = TUNED_HIDDEN_SIZE, num_layers: int = TUNED_NUM_LAYERS,
                   dropout: float = TUNED_DROPOUT, lr: float = TUNED_LR) -> list[dict]:
    """Sliding-window time-series CV with minimum 2-year training window."""
    _set_seed()
    cv_df = df_clean[df_clean["time"] <= "2020-12-31 23:00:00"]
    min_train_years = 2
    time_min = cv_df["time"].min()
    time_max = cv_df["time"].max()
    total_range = time_max - time_min
    min_train_delta = pd.Timedelta(days=min_train_years * 365)

    # Create n_splits validation windows after the minimum training period
    usable_range = total_range - min_train_delta
    val_duration = usable_range / n_splits

    cv_results = []

    for fold in range(n_splits):
        val_start = time_min + min_train_delta + fold * val_duration
        val_end = val_start + val_duration

        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"  Train: {time_min.date()} to {val_start.date()}")
        print(f"  Val:   {val_start.date()} to {val_end.date()}")
        print(f"{'=' * 60}")

        train_fold_df = cv_df[cv_df["time"] < val_start].copy()
        val_fold_df = cv_df[(cv_df["time"] >= val_start) & (cv_df["time"] < val_end)].copy()

        if len(train_fold_df) == 0 or len(val_fold_df) == 0:
            print("  Skipping fold \u2013 insufficient data.")
            continue

        X_train_f, y_train_f = make_sequences(train_fold_df, stride=6, feats=feats)
        X_val_f, y_val_f = make_sequences(val_fold_df, stride=1, feats=feats)

        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f.reshape(-1, X_train_f.shape[-1])).reshape(X_train_f.shape)
        X_val_f = scaler.transform(X_val_f.reshape(-1, X_val_f.shape[-1])).reshape(X_val_f.shape)

        X_train_t = torch.tensor(X_train_f, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_f, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_f, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_f, dtype=torch.float32)

        n_pos = y_train_t.sum(dim=0)
        n_neg = (y_train_t == 0).sum(dim=0)
        pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        input_dim = X_train_f.shape[-1]
        model, _, _ = _train_model(
            train_loader, val_loader, device, pos_weight,
            input_dim=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, lr=lr,
            save_path=f"best_model_fold{fold}.pth",
        )

        model.eval()
        with torch.no_grad():
            probs_val = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy()

        # Optimize thresholds per fold
        fold_metrics = {}
        for i, name in enumerate(HAZARD_NAMES):
            thresh = _optimize_threshold(y_val_f[:, i], probs_val[:, i])
            bin_val = (probs_val[:, i] >= thresh).astype(int)
            if len(np.unique(y_val_f[:, i])) > 1:
                fold_metrics[f"{name}_AUC"] = roc_auc_score(y_val_f[:, i], probs_val[:, i])
            else:
                fold_metrics[f"{name}_AUC"] = 0.5
            fold_metrics[f"{name}_F1"] = f1_score(y_val_f[:, i], bin_val, zero_division=0)
            fold_metrics[f"{name}_thresh"] = thresh

        fold_metrics["macro_AUC"] = roc_auc_score(y_val_f, probs_val, average="macro", multi_class="ovr")
        cv_results.append(fold_metrics)
        print(f"  Fold {fold + 1} macro AUC: {fold_metrics['macro_AUC']:.4f}")

    print("\n====== CV Summary (LSTM) ======")
    avg_macro = np.mean([r["macro_AUC"] for r in cv_results])
    std_macro = np.std([r["macro_AUC"] for r in cv_results])
    print(f"Average macro AUC = {avg_macro:.4f} \u00b1 {std_macro:.4f}")
    for name in HAZARD_NAMES:
        aucs = [r[f"{name}_AUC"] for r in cv_results]
        f1s = [r[f"{name}_F1"] for r in cv_results]
        print(f"{name} AUC: {np.mean(aucs):.4f} \u00b1 {np.std(aucs):.4f}, F1: {np.mean(f1s):.4f}")

    plot_cv_macro_auc(cv_results)
    return cv_results


# Final training + test evaluation
def tune_lstm_hyperparameters(
    df_clean: pd.DataFrame,
    device: torch.device = None,
    n_trials: int = 20,
    max_epochs: int = 20,
    patience: int = 4,
    stride: int = 6,
    loss_fn: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    feats: list[str] | None = None,
) -> dict:
    """Random search over LSTM hyperparameters to maximize validation F1.

    Searches over: hidden_size, num_layers, dropout, learning_rate, batch_size.
    Uses a single train/val split (train \u2264Sep 2022, val Oct\u2013Dec 2022).

    Parameters
    ----------
    df_clean : pd.DataFrame
        Output of prepare_model_data() \u2014 must have hazard labels and cyclical features.
    device : torch.device, optional
        Defaults to GPU if available.
    n_trials : int
        Number of random hyperparameter configurations to try.
    max_epochs : int
        Maximum training epochs per trial.
    patience : int
        Early stopping patience.
    stride : int
        Sequence stride for training data (lower = more data, slower).
    loss_fn : str
        'bce' or 'focal'. Loss function to use during tuning.
    focal_alpha, focal_gamma : float
        Focal Loss parameters (only used when loss_fn='focal').
    feats : list[str] or None
        Feature columns. Defaults to FEATS.

    Returns
    -------
    dict with keys: 'best_params', 'best_f1', 'best_auc', 'all_trials'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feats is None:
        feats = FEATS

    input_dim = len(feats)
    target_cols = [f"hazard_future_{h}" for h in HAZARD_NAMES]

    # Data split
    train_df = df_clean[df_clean["time"] <= "2022-09-30 23:00:00"]
    val_df = df_clean[
        (df_clean["time"] > "2022-09-30 23:00:00")
        & (df_clean["time"] <= "2022-12-31 23:00:00")
    ]

    # Sequences
    print(f"Creating sequences (stride={stride})...")
    X_train, y_train = make_sequences(train_df, stride=stride, feats=feats)
    X_val, y_val = make_sequences(val_df, stride=1, feats=feats)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, input_dim)).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val.reshape(-1, input_dim)).reshape(X_val.shape)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # pos_weight from training labels
    n_pos = y_train.sum(axis=0)
    n_neg = len(y_train) - n_pos
    pw = torch.FloatTensor(n_neg / (n_pos + 1e-8)).to(device)

    # Hyperparameter search space
    param_grid = {
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "lr": [0.0005, 0.001, 0.002, 0.005],
        "batch_size": [32, 64, 128, 256],
    }

    np.random.seed(SEED)
    all_trials = []

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: {n_trials} trials (loss={loss_fn})")
    print(f"{'='*60}")

    best_f1, best_params, best_auc = 0.0, {}, 0.0

    for trial in range(n_trials):
        # Sample random params
        params = {k: np.random.choice(v) for k, v in param_grid.items()}
        params["hidden_size"] = int(params["hidden_size"])
        params["num_layers"] = int(params["num_layers"])
        params["batch_size"] = int(params["batch_size"])

        print(f"\n  Trial {trial+1}/{n_trials}: "
              f"h={params['hidden_size']}, L={params['num_layers']}, "
              f"do={params['dropout']:.2f}, lr={params['lr']:.4f}, bs={params['batch_size']}")

        # Build model
        model = MultiHazardLSTM(
            input_dim=input_dim, num_classes=len(HAZARD_NAMES),
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        ).to(device)

        if loss_fn == "focal":
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))

        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(SEED))

        # Training loop
        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t.to(device))
                val_loss = criterion(val_logits, y_val_t.to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Evaluate best model
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy()

        # Per-class threshold + F1
        trial_f1s, trial_aucs = [], []
        for i, name in enumerate(HAZARD_NAMES):
            thresh = _optimize_threshold(y_val[:, i], val_probs[:, i])
            binary = (val_probs[:, i] >= thresh).astype(int)
            f1_val = f1_score(y_val[:, i], binary, zero_division=0)
            auc_val = roc_auc_score(y_val[:, i], val_probs[:, i]) if len(np.unique(y_val[:, i])) > 1 else 0.5
            trial_f1s.append(f1_val)
            trial_aucs.append(auc_val)

        macro_f1 = np.mean(trial_f1s)
        macro_auc = np.mean(trial_aucs)
        params["macro_f1"] = macro_f1
        params["macro_auc"] = macro_auc
        params["epochs_run"] = epoch + 1
        all_trials.append(params)

        print(f"    -> macro F1={macro_f1:.4f}, macro AUC={macro_auc:.4f} "
              f"(stopped at epoch {epoch+1})")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_auc = macro_auc
            best_params = params.copy()

    # Summary
    print(f"\n{'='*60}")
    print(f"BEST HYPERPARAMETERS (macro F1 = {best_f1:.4f}, AUC = {best_auc:.4f}):")
    print(f"  hidden_size: {best_params['hidden_size']}")
    print(f"  num_layers:  {best_params['num_layers']}")
    print(f"  dropout:     {best_params['dropout']:.2f}")
    print(f"  lr:          {best_params['lr']:.4f}")
    print(f"  batch_size:  {best_params['batch_size']}")
    print(f"{'='*60}")

    # Print all trials ranked
    all_trials_sorted = sorted(all_trials, key=lambda x: x["macro_f1"], reverse=True)
    print(f"\n  All trials (ranked by macro F1):")
    for i, t in enumerate(all_trials_sorted):
        print(f"    {i+1:2d}. F1={t['macro_f1']:.4f} AUC={t['macro_auc']:.4f} | "
              f"h={t['hidden_size']}, L={t['num_layers']}, "
              f"do={t['dropout']:.2f}, lr={t['lr']:.4f}, bs={t['batch_size']}")

    return {
        "best_params": best_params,
        "best_f1": best_f1,
        "best_auc": best_auc,
        "all_trials": all_trials_sorted,
    }


# Orchestrator: train_pipeline
def train_pipeline(df: pd.DataFrame) -> dict:
    """Train all models, return artifacts for test_pipeline().

    Steps:
      1. Prepare data (hazard labels, cyclical encoding, rolling features)
      2. Create sequences (stride=6 train, stride=1 val/test)
      3. Train Logistic Regression baseline
      4. Cross-validate LSTM
      5. Train final LSTM, optimize thresholds on validation set

    Outputs: training loss curves (no validation plots).
    Returns artifacts dict needed by test_pipeline().
    """
    _set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nPreparing model data...")
    df_clean = prepare_model_data(df)
    df_clean = add_rolling_features(df_clean)

    rolling_cols = [c for c in df_clean.columns
                    if any(c.endswith(s) for s in ["_roll6", "_roll12", "_diff6"])]
    feats_ext = FEATS + rolling_cols
    feats_ext = [f for f in feats_ext if f in df_clean.columns]
    print(f"Features: {len(feats_ext)} ({len(FEATS)} base + {len(rolling_cols)} rolling)")
    print(f"Tuned config: hidden={TUNED_HIDDEN_SIZE}, layers={TUNED_NUM_LAYERS}, "
          f"dropout={TUNED_DROPOUT:.2f}, lr={TUNED_LR:.4f}, batch_size={TUNED_BATCH_SIZE}")

    train_df = df_clean[df_clean["time"] <= "2020-12-31 23:00:00"]
    val_df = df_clean[(df_clean["time"] > "2020-12-31 23:00:00") & (df_clean["time"] <= "2022-12-31 23:00:00")]
    test_df = df_clean[df_clean["time"] > "2022-12-31 23:00:00"]

    print(f"\nTrain: {len(train_df):,} rows | Val: {len(val_df):,} rows | Test: {len(test_df):,} rows")

    X_train_seq, y_train = make_sequences(train_df, stride=6, feats=feats_ext)
    X_val_seq, y_val = make_sequences(val_df, stride=1, feats=feats_ext)
    X_test_seq, y_test = make_sequences(test_df, stride=1, feats=feats_ext)

    print(f"Sequences — Train: {X_train_seq.shape[0]:,} | Val: {X_val_seq.shape[0]:,} | Test: {X_test_seq.shape[0]:,}")

    scaler_bl = StandardScaler()
    X_train_bl = scaler_bl.fit_transform(X_train_seq[:, -1, :])
    X_val_bl = scaler_bl.transform(X_val_seq[:, -1, :])

    print("\n" + "=" * 60)
    print("BASELINE: Logistic Regression")
    print("=" * 60)

    lr_models = {}
    lr_thresholds = {}

    for i, name in enumerate(HAZARD_NAMES):
        print(f"\n  Training LR for: {name}")
        lr = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=SEED,
        )
        lr.fit(X_train_bl, y_train[:, i])
        lr_models[name] = lr

        val_probs = lr.predict_proba(X_val_bl)[:, 1]
        thresh = _optimize_threshold(y_val[:, i], val_probs)
        lr_thresholds[name] = thresh
        val_binary = (val_probs >= thresh).astype(int)

        auc_val = roc_auc_score(y_val[:, i], val_probs) if len(np.unique(y_val[:, i])) > 1 else 0.5
        f1_val = f1_score(y_val[:, i], val_binary, zero_division=0)
        print(f"    Threshold: {thresh:.2f} | Val AUC: {auc_val:.4f} | Val F1: {f1_val:.4f}")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (Tuned LSTM)")
    print("=" * 60)
    cv_results = cross_validate(
        df_clean, device, feats=feats_ext, batch_size=TUNED_BATCH_SIZE,
        hidden_size=TUNED_HIDDEN_SIZE, num_layers=TUNED_NUM_LAYERS,
        dropout=TUNED_DROPOUT, lr=TUNED_LR,
    )

    print("\n" + "=" * 60)
    print("FINAL LSTM: Training")
    print("=" * 60)

    input_dim = len(feats_ext)
    scaler_lstm = StandardScaler()
    X_train_s = scaler_lstm.fit_transform(X_train_seq.reshape(-1, input_dim)).reshape(X_train_seq.shape)
    X_val_s = scaler_lstm.transform(X_val_seq.reshape(-1, input_dim)).reshape(X_val_seq.shape)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n_pos = y_train_t.sum(dim=0)
    n_neg = (y_train_t == 0).sum(dim=0)
    pos_weight = (n_neg / (n_pos + 1e-8)).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=TUNED_BATCH_SIZE,
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=TUNED_BATCH_SIZE, shuffle=False,
    )

    model, train_losses, val_losses = _train_model(
        train_loader, val_loader, device, pos_weight,
        input_dim=input_dim, hidden_size=TUNED_HIDDEN_SIZE,
        num_layers=TUNED_NUM_LAYERS, dropout=TUNED_DROPOUT, lr=TUNED_LR,
        save_path="best_multihazard_model_final.pth",
    )
    plot_loss_curves(train_losses, val_losses, title="LSTM — Training Loss Curves")

    # Threshold optimization on validation set
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for Xb, _ in DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=TUNED_BATCH_SIZE):
            val_probs_list.append(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
    val_probs_lstm = np.concatenate(val_probs_list, axis=0)

    lstm_thresholds = {}
    for i, name in enumerate(HAZARD_NAMES):
        thresh = _optimize_threshold(y_val[:, i], val_probs_lstm[:, i])
        lstm_thresholds[name] = thresh
        val_binary = (val_probs_lstm[:, i] >= thresh).astype(int)
        auc_val = roc_auc_score(y_val[:, i], val_probs_lstm[:, i]) if len(np.unique(y_val[:, i])) > 1 else 0.5
        f1_val = f1_score(y_val[:, i], val_binary, zero_division=0)
        print(f"  {name}: threshold={thresh:.2f} | Val AUC={auc_val:.4f} | Val F1={f1_val:.4f}")

    # Save artifacts
    joblib.dump(scaler_lstm, "scaler_multihazard.pkl")
    joblib.dump(lstm_thresholds, "thresholds_multihazard.pkl")

    print("\n Training complete. Call test_pipeline(artifacts) to evaluate on test set.")

    return {
        "model": model,
        "device": device,
        "lr_models": lr_models,
        "lr_thresholds": lr_thresholds,
        "lstm_thresholds": lstm_thresholds,
        "scaler_bl": scaler_bl,
        "scaler_lstm": scaler_lstm,
        "X_test_seq": X_test_seq,
        "y_test": y_test,
        "feats_ext": feats_ext,
        "cv_results": cv_results,
    }


# Orchestrator: test_pipeline
def test_pipeline(artifacts: dict) -> dict:
    """Evaluate trained models on held-out test set (2023+).

    Outputs: confusion matrices, ROC curves, PR curves, threshold analysis,
    and comparison table for both LR and LSTM.
    """
    model = artifacts["model"]
    device = artifacts["device"]
    lr_models = artifacts["lr_models"]
    lr_thresholds = artifacts["lr_thresholds"]
    lstm_thresholds = artifacts["lstm_thresholds"]
    scaler_bl = artifacts["scaler_bl"]
    scaler_lstm = artifacts["scaler_lstm"]
    X_test_seq = artifacts["X_test_seq"]
    y_test = artifacts["y_test"]
    feats_ext = artifacts["feats_ext"]

    print("=" * 60)
    print(f"TEST EVALUATION — {X_test_seq.shape[0]:,} sequences (2023+)")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("BASELINE: Logistic Regression (Test Set)")
    print("-" * 60)

    X_test_bl = scaler_bl.transform(X_test_seq[:, -1, :])
    all_test_probs_lr = np.zeros((len(y_test), len(HAZARD_NAMES)))
    baseline_results = {}

    for i, name in enumerate(HAZARD_NAMES):
        test_probs = lr_models[name].predict_proba(X_test_bl)[:, 1]
        all_test_probs_lr[:, i] = test_probs
        thresh = lr_thresholds[name]
        test_binary = (test_probs >= thresh).astype(int)

        auc_val = roc_auc_score(y_test[:, i], test_probs) if len(np.unique(y_test[:, i])) > 1 else 0.5
        acc = accuracy_score(y_test[:, i], test_binary)
        f1 = f1_score(y_test[:, i], test_binary, zero_division=0)

        baseline_results[f"{name}_AUC"] = auc_val
        baseline_results[f"{name}_acc"] = acc
        baseline_results[f"{name}_F1"] = f1
        baseline_results[f"{name}_thresh"] = thresh
        print(f"\n  {name.upper()}: Threshold={thresh:.2f} | AUC={auc_val:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
        print(f"    Confusion Matrix:\n{confusion_matrix(y_test[:, i], test_binary)}")

    macro_auc_bl = roc_auc_score(y_test, all_test_probs_lr, average="macro", multi_class="ovr")
    baseline_results["macro_AUC"] = macro_auc_bl
    print(f"\n  Macro AUC: {macro_auc_bl:.4f}")

    # LR test plots
    all_test_binary_lr = np.zeros_like(all_test_probs_lr, dtype=int)
    for i, name in enumerate(HAZARD_NAMES):
        all_test_binary_lr[:, i] = (all_test_probs_lr[:, i] >= lr_thresholds[name]).astype(int)
    plot_roc_curves(y_test, all_test_probs_lr, [f"{n} (LR test)" for n in HAZARD_NAMES])
    plot_precision_recall_curves(y_test, all_test_probs_lr, [f"{n} (LR test)" for n in HAZARD_NAMES])
    plot_confusion_matrices(y_test, all_test_binary_lr, [f"{n} (LR test)" for n in HAZARD_NAMES])

    print("\n" + "-" * 60)
    print("CHALLENGER: Tuned LSTM (Test Set)")
    print("-" * 60)

    input_dim = len(feats_ext)
    X_test_s = scaler_lstm.transform(X_test_seq.reshape(-1, input_dim)).reshape(X_test_seq.shape)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model.eval()
    test_probs_list, test_trues_list = [], []
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=TUNED_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for Xb, yb in test_loader:
            test_probs_list.append(torch.sigmoid(model(Xb.to(device))).cpu().numpy())
            test_trues_list.append(yb.numpy())
    test_probs_lstm = np.concatenate(test_probs_list, axis=0)
    test_trues = np.concatenate(test_trues_list, axis=0)

    test_binary_lstm = np.zeros_like(test_probs_lstm, dtype=int)
    lstm_results = {"model": "LSTM", "loss_fn": "bce"}

    for i, name in enumerate(HAZARD_NAMES):
        thresh = lstm_thresholds[name]
        test_binary_lstm[:, i] = (test_probs_lstm[:, i] >= thresh).astype(int)

        auc_val = roc_auc_score(test_trues[:, i], test_probs_lstm[:, i]) if len(np.unique(test_trues[:, i])) > 1 else 0.5
        acc = accuracy_score(test_trues[:, i], test_binary_lstm[:, i])
        f1 = f1_score(test_trues[:, i], test_binary_lstm[:, i], zero_division=0)

        lstm_results[f"{name}_AUC"] = auc_val
        lstm_results[f"{name}_acc"] = acc
        lstm_results[f"{name}_F1"] = f1
        lstm_results[f"{name}_thresh"] = thresh
        print(f"\n  {name.upper()}: Threshold={thresh:.2f} | AUC={auc_val:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
        print(f"    Confusion Matrix:\n{confusion_matrix(test_trues[:, i], test_binary_lstm[:, i])}")

    macro_auc_lstm = roc_auc_score(test_trues, test_probs_lstm, average="macro", multi_class="ovr")
    lstm_results["macro_AUC"] = macro_auc_lstm
    print(f"\n  Macro AUC: {macro_auc_lstm:.4f}")
    print("\n  Classification report:")
    print(classification_report(test_trues, test_binary_lstm, target_names=HAZARD_NAMES, zero_division=0))

    plot_roc_curves(test_trues, test_probs_lstm, [f"{n} (LSTM test)" for n in HAZARD_NAMES])
    plot_precision_recall_curves(test_trues, test_probs_lstm, [f"{n} (LSTM test)" for n in HAZARD_NAMES])
    plot_confusion_matrices(test_trues, test_binary_lstm, [f"{n} (LSTM test)" for n in HAZARD_NAMES])
    plot_threshold_analysis(test_trues, test_probs_lstm, HAZARD_NAMES)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON: Baseline vs Tuned LSTM (Test Set)")
    print("=" * 60)
    print(f"{'Metric':<20} {'LR Baseline':>12} {'Tuned LSTM':>12} {'Delta':>10}")
    print("-" * 56)
    for name in HAZARD_NAMES:
        bl_auc = baseline_results[f"{name}_AUC"]
        lstm_auc = lstm_results[f"{name}_AUC"]
        bl_f1 = baseline_results[f"{name}_F1"]
        lstm_f1 = lstm_results[f"{name}_F1"]
        print(f"{name}_AUC{'':<11} {bl_auc:>12.4f} {lstm_auc:>12.4f} {lstm_auc - bl_auc:>+10.4f}")
        print(f"{name}_F1{'':<12} {bl_f1:>12.4f} {lstm_f1:>12.4f} {lstm_f1 - bl_f1:>+10.4f}")
    print(f"{'macro_AUC':<20} {macro_auc_bl:>12.4f} {macro_auc_lstm:>12.4f} {macro_auc_lstm - macro_auc_bl:>+10.4f}")

    return {
        "baseline": baseline_results,
        "lstm": lstm_results,
    }

def run_model(df: pd.DataFrame) -> dict:
    """Full pipeline: train + test. Backward-compatible wrapper."""
    artifacts = train_pipeline(df)
    results = test_pipeline(artifacts)
    results["cv_results"] = artifacts["cv_results"]
    return results
