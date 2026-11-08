"""
City Router: two specialist LightGBM models (one per city) wrapped in a single
object with the same .predict(X) interface as final_pipeline.joblib.

Design
------
- CityRouter.predict(X) reads X['city'] and routes each row to the right model.
- Each specialist is trained ONLY on that city's rows — no cross-city dilution.
- 'city' column is used for routing only, not as a model input feature.
- Saved as models/city_router.joblib — drop-in replacement for final_pipeline.joblib.

Why this helps
--------------
The combined model's pm25_now dominates at 86% importance partly because
Karachi (low variance) and Lahore (high variance) are blended. Each specialist
can learn city-specific weather-PM2.5 relationships without being drowned out
by the other city's distribution.

Run:  python -m src.train_city_router
Out:  models/city_router.joblib
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from src import config
from src.features import FEATURE_COLS, TARGET
from src.metrics import (
    calculate_metrics,
    print_metrics,
)

MODELS_DIR = config.ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# City-specific feature set: drop 'city' since each specialist sees only one city
# (a constant column gives the model no information)
CITY_FEATURE_COLS = [c for c in FEATURE_COLS if c != "city"]

# LightGBM hyperparam grid (same as train.py for fair comparison)
PARAM_GRID = {
    "model__n_estimators":  [100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__num_leaves":    [31, 50],
    "model__max_depth":     [10, 20, -1],
}


# ---------------------------------------------------------------------------
# CityRouter — the single object that the app / API will load
# ---------------------------------------------------------------------------

class CityRouter:
    """
    Routes each prediction row to the correct city-specialist model.

    Parameters
    ----------
    models : dict
        {'karachi': fitted_pipeline, 'lahore': fitted_pipeline}

    Usage
    -----
    >>> router = CityRouter(models)
    >>> y_pred = router.predict(X)   # X must include a 'city' column
    >>> joblib.dump(router, 'models/city_router.joblib')
    """

    def __init__(self, models: dict):
        self.models_ = models   # persisted by joblib automatically

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Route each row to its city specialist and return predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Must include a 'city' column for routing.
            All columns in CITY_FEATURE_COLS must be present.

        Returns
        -------
        np.ndarray of float
            PM2.5 predictions in µg/m³.
        """
        X = X.reset_index(drop=True)
        result = np.full(len(X), np.nan)

        for city, model in self.models_.items():
            mask = X["city"] == city
            if mask.any():
                result[mask.values] = model.predict(X.loc[mask, CITY_FEATURE_COLS])

        # catch any rows whose city wasn't in training
        if np.isnan(result).any():
            unknown = X.loc[np.isnan(result), "city"].unique().tolist()
            raise ValueError(
                f"CityRouter has no specialist for: {unknown}. "
                f"Known cities: {list(self.models_.keys())}"
            )
        return result


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_lgbm_pipeline() -> Pipeline:
    """StandardScaler + LightGBM — no OHE needed (city is gone)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LGBMRegressor(random_state=42, verbose=-1)),
    ])


def tune_city_model(
    city: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Tune LightGBM for one city with TimeSeriesSplit CV."""
    print(f"\n{'='*70}")
    print(f"  Tuning LightGBM specialist — {city.upper()}")
    print(f"  Training rows : {len(X_train)}")
    print(f"{'='*70}")

    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(
        _build_lgbm_pipeline(),
        PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"\n  Best params  : {gs.best_params_}")
    print(f"  Best CV MAE  : {-gs.best_score_:.2f} µg/m³")
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("City Router Training")
    print("=" * 70)
    print(f"\nCity feature cols ({len(CITY_FEATURE_COLS)}): {CITY_FEATURE_COLS}")

    # --- load splits ---
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(config.PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(config.PROCESSED_DIR / "test.csv")

    print(f"\nTrain : {len(train)} rows | Val : {len(val)} rows | Test : {len(test)} rows")

    trained_models = {}

    for city in ["karachi", "lahore"]:
        # ---- filter to this city only ----
        tr = train[train["city"] == city].sort_values("datetime_pkt")
        va = val[val["city"] == city].sort_values("datetime_pkt")
        te = test[test["city"] == city].sort_values("datetime_pkt")

        X_tr, y_tr = tr[CITY_FEATURE_COLS], tr[TARGET]
        X_va, y_va = va[CITY_FEATURE_COLS], va[TARGET]
        X_te, y_te = te[CITY_FEATURE_COLS], te[TARGET]

        # ---- tune on train ----
        model = tune_city_model(city, X_tr, y_tr)

        # ---- validate ----
        val_metrics = calculate_metrics(y_va, model.predict(X_va))
        print_metrics(val_metrics, f"{city.title()} Specialist (Validation)")

        # ---- retrain on train + val before final test ----
        print(f"\n  Retraining {city} specialist on train+val ({len(X_tr)+len(X_va)} rows)...")
        model.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))

        # ---- test ----
        test_metrics = calculate_metrics(y_te, model.predict(X_te))
        print_metrics(test_metrics, f"{city.title()} Specialist (Test)")

        trained_models[city] = model

    # --- assemble router ---
    router = CityRouter(trained_models)

    # --- overall test metrics via router ---
    y_test_full  = test[TARGET]
    y_pred_router = router.predict(test[FEATURE_COLS])

    overall = calculate_metrics(y_test_full, y_pred_router)

    # --- load old combined model for comparison ---
    old_model    = joblib.load(MODELS_DIR / "final_pipeline.joblib")
    old_pred     = old_model.predict(test[FEATURE_COLS])
    old_metrics  = calculate_metrics(y_test_full, old_pred)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON — CityRouter vs Combined Model (Test Set)")
    print("=" * 70)
    print(f"\n{'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'AQI Acc':>10}")
    print("-" * 68)
    print(
        f"{'CityRouter (new)':<28} "
        f"{overall['mae']:>8.2f} {overall['rmse']:>8.2f} "
        f"{overall['r2']:>8.4f} {overall['aqi_accuracy']:>9.1%}"
    )
    print(
        f"{'Combined model (old)':<28} "
        f"{old_metrics['mae']:>8.2f} {old_metrics['rmse']:>8.2f} "
        f"{old_metrics['r2']:>8.4f} {old_metrics['aqi_accuracy']:>9.1%}"
    )
    print("-" * 68)

    # --- save ---
    out_path = MODELS_DIR / "city_router.joblib"
    joblib.dump(router, out_path)
    print(f"\n✓ CityRouter saved → {out_path}")
    print("=" * 70)

