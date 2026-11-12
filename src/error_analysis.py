"""Task 13: error analysis on the untouched test split.

Run:
    python -m src.error_analysis

Outputs:
    reports/error_analysis.md
    reports/error_analysis_summary.json
    reports/error_by_month.csv
    reports/error_by_city.csv
    reports/error_by_hour.csv
    reports/error_by_pm_bin.csv
    reports/error_by_regime.csv
    reports/worst_predictions.csv
    reports/worst_days.csv
    reports/figures/error_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.aqi import BREAKPOINTS, get_aqi_category_info
from src.features import FEATURE_COLS, TARGET
from src.metrics import calculate_metrics, calculate_per_city_metrics
from src.predict import load_model

REPORTS = config.ROOT / "reports"
FIG_DIR = REPORTS / "figures"
CATEGORY_ORDER = [b[4] for b in BREAKPOINTS]


def _categories(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.array([get_aqi_category_info(float(v))["category"] for v in values])


def _safe_metrics(y_true, y_pred) -> dict[str, float]:
    metrics = calculate_metrics(y_true, y_pred)
    return {k: float(v) for k, v in metrics.items()}


def _group_report(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(by, dropna=False):
        metrics = _safe_metrics(part[TARGET], part["prediction"])
        rows.append(
            {
                by: key,
                "rows": len(part),
                **metrics,
                "bias": float((part[TARGET] - part["prediction"]).mean()),
                "underprediction_rate": float((part[TARGET] > part["prediction"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _regime_frame(df: pd.DataFrame) -> pd.DataFrame:
    spike_threshold = float(df[TARGET].quantile(0.95))
    df = df.copy()
    df["is_spike"] = df[TARGET] >= spike_threshold
    df["is_onset"] = (df[TARGET] >= 125) & ((df[TARGET] - df["pm25_now"]) >= 50)
    df["is_recovery"] = (df["pm25_now"] - df[TARGET]) >= 50
    df["is_stable"] = ~(df["is_spike"] | df["is_onset"] | df["is_recovery"])

    parts = []
    for name in ["stable", "spike", "onset", "recovery"]:
        part = df[df[f"is_{name}"]]
        if part.empty:
            continue
        metrics = _safe_metrics(part[TARGET], part["prediction"])
        parts.append(
            {
                "regime": name,
                "rows": len(part),
                "share": len(part) / len(df),
                **metrics,
                "bias": float((part[TARGET] - part["prediction"]).mean()),
                "underprediction_rate": float(
                    (part[TARGET] > part["prediction"]).mean()
                ),
            }
        )
    return pd.DataFrame(parts), spike_threshold


def _confusion(df: pd.DataFrame) -> pd.DataFrame:
    true_cat = _categories(df[TARGET])
    pred_cat = _categories(df["prediction"])
    table = pd.crosstab(
        pd.Categorical(true_cat, categories=CATEGORY_ORDER),
        pd.Categorical(pred_cat, categories=CATEGORY_ORDER),
        dropna=False,
    )
    return table.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)


def _plot_residuals(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for city, part in df.groupby("city"):
        residual = part[TARGET] - part["prediction"]
        ax.hist(residual, bins=60, alpha=0.55, label=city.title())
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Test-set residuals (actual - predicted)")
    ax.set_xlabel("Residual (ug/m3)")
    ax.set_ylabel("Hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_residuals.png", dpi=150)
    plt.close(fig)


def _plot_by_month(monthly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for city, part in monthly.groupby("city"):
        ax.plot(part["month"], part["mae"], marker="o", label=city.title())
    ax.set_title("Monthly MAE on the held-out test split")
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE (ug/m3)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_month.png", dpi=150)
    plt.close(fig)


def _plot_by_hour(hourly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for city, part in hourly.groupby("city"):
        ax.plot(part["hour"], part["mae"], marker="o", ms=3, label=city.title())
    ax.set_title("MAE by hour of day (PKT)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MAE (ug/m3)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_hour.png", dpi=150)
    plt.close(fig)


def _plot_by_bin(bins: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(bins))
    width = 0.38
    ax.bar(x - width / 2, bins["mae"], width, label="MAE")
    ax.bar(x + width / 2, bins["bias"].abs(), width, label="|Bias|")
    ax.set_xticks(x)
    ax.set_xticklabels(bins["pm_bin"], rotation=20, ha="right")
    ax.set_title("Error by true PM2.5 category")
    ax.set_ylabel("ug/m3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_by_pm_bin.png", dpi=150)
    plt.close(fig)


def _plot_confusion(table: pd.DataFrame):
    norm = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)))
    ax.set_yticks(range(len(table.index)))
    short = [c.replace("Unhealthy for Sensitive Groups", "Sensitive") for c in table.columns]
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("AQI category confusion (row-normalised)")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            value = norm.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_confusion.png", dpi=150)
    plt.close(fig)


def _plot_lahore_smog(df: pd.DataFrame):
    part = df[(df["city"] == "lahore") & (df["datetime_pkt"] >= "2026-09-01")].copy()
    if part.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(part["datetime_pkt"], part[TARGET], label="Actual", lw=1.4)
    ax.plot(part["datetime_pkt"], part["prediction"], label="Forecast", lw=1.1)
    ax.plot(part["datetime_pkt"], part["pm25_now"], label="Persistence (now)", lw=0.8, alpha=0.6)
    ax.axhline(125.4, color="red", ls="--", lw=0.8, label="Very Unhealthy threshold")
    ax.set_title("Lahore: September-October 2026 smog onset")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_lahore_smog.png", dpi=150)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for city, part in df.groupby("city"):
        ax.scatter(part[TARGET], part["prediction"], s=5, alpha=0.25, label=city.title())
    upper = float(max(df[TARGET].max(), df["prediction"].max()))
    ax.plot([0, upper], [0, upper], color="black", lw=1, label="Perfect forecast")
    ax.set_xlabel("Actual PM2.5 (ug/m3)")
    ax.set_ylabel("Predicted PM2.5 (ug/m3)")
    ax.set_title("Actual vs predicted (test set)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_scatter.png", dpi=150)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                if float(value).is_integer():
                    values.append(str(int(value)))
                else:
                    values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    REPORTS.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(config.PROCESSED_DIR / "test.csv", parse_dates=["datetime_pkt"])
    model = load_model()
    test["prediction"] = model.predict(test[FEATURE_COLS])
    test["residual"] = test[TARGET] - test["prediction"]
    test["abs_error"] = test["residual"].abs()
    test["month"] = test["datetime_pkt"].dt.month
    test["hour"] = test["datetime_pkt"].dt.hour
    test["date"] = test["datetime_pkt"].dt.date
    test["true_category"] = _categories(test[TARGET])
    test["pred_category"] = _categories(test["prediction"])

    overall = _safe_metrics(test[TARGET], test["prediction"])
    by_city = _group_report(test, "city")
    by_month_city = _group_report(test, "month")
    by_month_city["city"] = "both"

    monthly = (
        test.assign(month_label=test["datetime_pkt"].dt.strftime("%Y-%m"))
        .groupby(["city", "month_label"])
        .apply(
            lambda part: pd.Series(
                {
                    "rows": len(part),
                    "mae": (part[TARGET] - part["prediction"]).abs().mean(),
                    "bias": (part[TARGET] - part["prediction"]).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    monthly_by_city = monthly.rename(columns={"month_label": "month"})

    hourly = _group_report(test.assign(hour=test["hour"]), "hour")
    hourly["city"] = "both"
    hourly_by_city = (
        test.groupby(["city", "hour"])
        .apply(
            lambda part: pd.Series(
                {"mae": (part[TARGET] - part["prediction"]).abs().mean()}
            ),
            include_groups=False,
        )
        .reset_index()
    )

    bin_edges = [0, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
    bin_labels = ["Good", "Moderate", "Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]
    test["pm_bin"] = pd.cut(test[TARGET], bins=bin_edges, labels=bin_labels, right=True)
    by_bin = _group_report(test.dropna(subset=["pm_bin"]), "pm_bin")

    regime, spike_threshold = _regime_frame(test)
    confusion = _confusion(test)

    worst = (
        test.sort_values("abs_error", ascending=False)
        .head(25)[
            [
                "datetime_pkt",
                "city",
                TARGET,
                "prediction",
                "pm25_now",
                "residual",
                "abs_error",
                "true_category",
                "pred_category",
            ]
        ]
        .copy()
    )
    worst.columns = [
        "datetime_pkt",
        "city",
        "actual_pm25",
        "predicted_pm25",
        "pm25_now",
        "residual",
        "abs_error",
        "true_category",
        "pred_category",
    ]

    day_errors = (
        test.groupby(["date", "city"])
        .apply(
            lambda part: pd.Series(
                {
                    "rows": len(part),
                    "mae": (part[TARGET] - part["prediction"]).abs().mean(),
                    "bias": (part[TARGET] - part["prediction"]).mean(),
                    "actual_max": part[TARGET].max(),
                    "pred_max": part["prediction"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    worst_days = day_errors[day_errors["rows"] >= 12].sort_values("mae", ascending=False).head(15)

    # Persist tables.
    by_month_city.to_csv(REPORTS / "error_by_month.csv", index=False)
    by_city.to_csv(REPORTS / "error_by_city.csv", index=False)
    hourly.to_csv(REPORTS / "error_by_hour.csv", index=False)
    by_bin.to_csv(REPORTS / "error_by_pm_bin.csv", index=False)
    regime.to_csv(REPORTS / "error_by_regime.csv", index=False)
    worst.to_csv(REPORTS / "worst_predictions.csv", index=False)
    worst_days.to_csv(REPORTS / "worst_days.csv", index=False)
    confusion.to_csv(REPORTS / "error_confusion.csv")

    _plot_residuals(test)
    _plot_by_month(monthly_by_city)
    _plot_by_hour(hourly_by_city)
    _plot_by_bin(by_bin)
    _plot_confusion(confusion)
    _plot_lahore_smog(test)
    _plot_scatter(test)

    # Key numbers used in the written report.
    lahore = by_city[by_city["city"] == "lahore"].iloc[0]
    karachi = by_city[by_city["city"] == "karachi"].iloc[0]
    lahore_oct = test[
        (test["city"] == "lahore")
        & (test["datetime_pkt"] >= "2026-10-01")
        & (test["datetime_pkt"] < "2026-11-01")
    ]
    lahore_oct_metrics = (
        _safe_metrics(lahore_oct[TARGET], lahore_oct["prediction"])
        if len(lahore_oct)
        else {}
    )
    lahore_oct_bias = (
        float((lahore_oct[TARGET] - lahore_oct["prediction"]).mean())
        if len(lahore_oct)
        else float("nan")
    )
    spike_row = regime[regime["regime"] == "spike"]
    onset_row = regime[regime["regime"] == "onset"]
    recovery_row = regime[regime["regime"] == "recovery"]

    summary = {
        "overall": overall,
        "spike_threshold_pm25": spike_threshold,
        "by_city": by_city.to_dict(orient="records"),
        "by_regime": regime.to_dict(orient="records"),
        "lahore_october_2026": lahore_oct_metrics,
        "worst_days": worst_days.head(10).to_dict(orient="records"),
        "confusion": confusion.to_dict(),
    }
    (REPORTS / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Markdown report with generated tables.
    top_failures = worst.head(10).copy()
    top_failures["datetime_pkt"] = top_failures["datetime_pkt"].astype(str)
    for col in ["actual_pm25", "predicted_pm25", "pm25_now", "residual", "abs_error"]:
        top_failures[col] = top_failures[col].round(1)

    episode = ""
    if not spike_row.empty:
        s = spike_row.iloc[0]
        episode += (
            f"- Top-5% spikes (actual >= {spike_threshold:.1f} ug/m3): "
            f"{int(s['rows'])} hours, MAE {s['mae']:.2f}, bias {s['bias']:+.2f}, "
            f"underprediction {s['underprediction_rate']:.1%}.\n"
        )
    if not onset_row.empty:
        o = onset_row.iloc[0]
        episode += (
            f"- Smog onsets (actual >= 125 and rise >= 50 from now): "
            f"{int(o['rows'])} hours, MAE {o['mae']:.2f}, bias {o['bias']:+.2f}.\n"
        )
    if not recovery_row.empty:
        r = recovery_row.iloc[0]
        episode += (
            f"- Clearing/recovery (now exceeds actual by >= 50): "
            f"{int(r['rows'])} hours, MAE {r['mae']:.2f}, bias {r['bias']:+.2f}.\n"
        )

    report = f"""# Error Analysis

Generated by `python -m src.error_analysis` on the held-out test split
({len(test):,} hours, {test['datetime_pkt'].min()} to {test['datetime_pkt'].max()} PKT).

## 1. Overall and per-city error

| Scope | Rows | MAE | RMSE | R2 | AQI accuracy | Bias |
|---|---:|---:|---:|---:|---:|---:|
| Overall | {len(test):,} | {overall['mae']:.2f} | {overall['rmse']:.2f} | {overall['r2']:.4f} | {overall['aqi_accuracy']:.1%} | {test['residual'].mean():+.2f} |
| Karachi | {int(karachi['rows']):,} | {karachi['mae']:.2f} | {karachi['rmse']:.2f} | {karachi['r2']:.4f} | {karachi['aqi_accuracy']:.1%} | {karachi['bias']:+.2f} |
| Lahore | {int(lahore['rows']):,} | {lahore['mae']:.2f} | {lahore['rmse']:.2f} | {lahore['r2']:.4f} | {lahore['aqi_accuracy']:.1%} | {lahore['bias']:+.2f} |

Lahore's MAE is **{lahore['mae'] / karachi['mae']:.1f}x** Karachi's and its AQI accuracy
is **{(karachi['aqi_accuracy'] - lahore['aqi_accuracy']) * 100:.1f} percentage points** lower.

## 2. When the model fails

{episode}
- Lahore, October 2026: MAE {lahore_oct_metrics.get('mae', float('nan')):.2f},
  bias {lahore_oct_bias:+.2f}, AQI accuracy
  {lahore_oct_metrics.get('aqi_accuracy', float('nan')):.1%}.
- The largest single errors are concentrated in Lahore during rapid transitions:
  the model overpredicts on clearing days (for example 29 October, when the current
  reading was 250-300 ug/m3 but the next day fell to 55-90) and underpredicts on
  abrupt spike days (for example 30 October). In both cases the lag features and
  the 24-hour-ahead outcome disagree.

## 3. Error by true PM2.5 category

{_markdown_table(by_bin[['pm_bin', 'rows', 'mae', 'rmse', 'bias', 'underprediction_rate']])}

## 4. Error by month (both cities)

{_markdown_table(by_month_city[['month', 'rows', 'mae', 'rmse', 'bias', 'aqi_accuracy']])}

## 5. AQI category confusion (row-normalised)

{_markdown_table(confusion.div(confusion.sum(axis=1).replace(0, np.nan), axis=0).round(2).rename_axis('true_category').reset_index())}

## 6. Worst city-days

{_markdown_table(worst_days[['date', 'city', 'rows', 'mae', 'bias', 'actual_max', 'pred_max']].head(10))}

## 7. Largest individual failures

{_markdown_table(top_failures.drop(columns=['true_category', 'pred_category']).head(10))}

## 8. Figures

- `reports/figures/error_residuals.png`
- `reports/figures/error_by_month.png`
- `reports/figures/error_by_hour.png`
- `reports/figures/error_by_pm_bin.png`
- `reports/figures/error_confusion.png`
- `reports/figures/error_lahore_smog.png`
- `reports/figures/error_scatter.png`

## 9. Interpretation

1. Karachi is close to a well-behaved persistence problem: the city's PM2.5 is
   smoother, so the ensemble tracks it tightly.
2. Lahore has a heavy-tailed distribution. The model overpredicts when pollution
   clears quickly (recent lag values stay high) and underpredicts when a spike
   arrives abruptly (there is little warning in the inputs available 24 hours
   earlier). The October 2026 test window contains both failure modes.
3. Errors are therefore **regime-dependent**, not uniformly distributed. Aggregate
   MAE hides the fact that the model is much better in stable hours than in smog
   transitions.
4. The AQI confusion matrix shows that most mistakes are one category away. This is
   safer than jumping across several categories, but still material for health advice.
"""
    (REPORTS / "error_analysis.md").write_text(report, encoding="utf-8")

    print(f"Overall: MAE {overall['mae']:.2f} | RMSE {overall['rmse']:.2f} | "
          f"R2 {overall['r2']:.4f} | AQI {overall['aqi_accuracy']:.1%}")
    print(f"Karachi: MAE {karachi['mae']:.2f} | AQI {karachi['aqi_accuracy']:.1%}")
    print(f"Lahore:  MAE {lahore['mae']:.2f} | AQI {lahore['aqi_accuracy']:.1%}")
    print(f"Wrote {REPORTS / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
