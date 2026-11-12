"""Train the final deployable ensemble and record honest test metrics.

Run:  python -m src.train_final

The earlier stage scripts save every tuned candidate separately. This script
builds the exact winning architecture used by the report (Ridge + Random Forest
+ LightGBM, weighted 0.2 / 0.3 / 0.5), retrains it on train + validation, and
evaluates the untouched test split once. The resulting pipeline is what the
Streamlit app, FastAPI service, and Hugging Face Space load.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src import config
from src.aqi import get_health_advice
from src.evaluate import apply_buffer_zones
from src.features import FEATURE_COLS, NUMERIC_COLS, TARGET
from src.metrics import (
    calculate_metrics,
    calculate_per_city_metrics,
    print_metrics,
    print_per_city_metrics,
)
from src.pipeline import build_pipeline
from src.postprocess import apply_clearing_blend
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


MODELS_DIR = config.ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_pipeline.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"


def build_final_ensemble():
    """Return the documented winning ensemble (unfitted, raw-feature input)."""
    ridge = build_pipeline(Ridge(alpha=10.0), NUMERIC_COLS)
    rf = build_pipeline(
        RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        NUMERIC_COLS,
    )
    lgbm = build_pipeline(
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=45,
            random_state=42,
            verbose=-1,
        ),
        NUMERIC_COLS,
    )
    return VotingRegressor(
        estimators=[("ridge", ridge), ("rf", rf), ("lgbm", lgbm)],
        weights=[0.2, 0.3, 0.5],
    )


def load_split(name):
    df = pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", parse_dates=["datetime_pkt"])
    return df


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    combined = pd.concat([train, val], ignore_index=True)
    X_fit = combined[FEATURE_COLS]
    y_fit = combined[TARGET]

    print("=" * 70)
    print("Training final deployable ensemble")
    print("=" * 70)
    print(f"Train+Val rows: {len(X_fit):,} | Features: {len(FEATURE_COLS)}")
    print(f"Test rows (held out): {len(test):,}")

    model = build_final_ensemble()
    model.fit(X_fit, y_fit)

    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]
    y_pred = model.predict(X_test)

    raw_metrics = calculate_metrics(y_test, y_pred)
    buffered = apply_buffer_zones(y_pred)
    buffered_metrics = calculate_metrics(y_test, buffered)
    postprocessed = apply_clearing_blend(y_pred, X_test)
    post_metrics = calculate_metrics(y_test, postprocessed)

    print_metrics(raw_metrics, "Ensemble (Test, raw)")
    print_metrics(post_metrics, "Ensemble (Test, clearing blend)")
    print_metrics(buffered_metrics, "Ensemble (Test, conservative buffers)")

    scored = test.copy()
    scored["prediction"] = y_pred
    scored["postprocessed_prediction"] = postprocessed
    city_metrics = calculate_per_city_metrics(scored, TARGET, "prediction")
    print_per_city_metrics(city_metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "VotingRegressor(ridge=0.2, rf=0.3, lgbm=0.5)",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "fit_rows": int(len(combined)),
        "test_rows": int(len(test)),
        "test_metrics_raw": {k: float(v) for k, v in raw_metrics.items()},
        "test_metrics_clearing_blend": {k: float(v) for k, v in post_metrics.items()},
        "test_metrics_buffered": {k: float(v) for k, v in buffered_metrics.items()},
        "per_city_test_metrics": {
            city: {k: float(v) for k, v in metrics.items()}
            for city, metrics in city_metrics.items()
        },
        "aqi_reference": get_health_advice(50.0)["category"],
        "package_python": "3.14.7",
        "artifact": str(MODEL_PATH.relative_to(config.ROOT)),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved pipeline -> {MODEL_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
