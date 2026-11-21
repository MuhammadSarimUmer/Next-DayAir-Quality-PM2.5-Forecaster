"""
Upload trained pipeline and Model Card to Hugging Face Hub.
"""

import os
from pathlib import Path
from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
MODEL_FILE = ROOT / "models" / "final_pipeline.joblib"

MODEL_CARD_CONTENT = """---
language:
- en
license: mit
tags:
- tabular
- time-series
- air-quality
- environmental-science
- scikit-learn
- lightgbm
- random-forest
datasets:
- openaq
metrics:
- mae
- rmse
- r2
- accuracy
model-index:
- name: Next-Day PM2.5 Forecaster (Karachi & Lahore)
  results:
  - task:
      type: tabular-regression
      name: Air Quality Forecasting
    dataset:
      type: openaq-pakistan
      name: Karachi & Lahore Ground Sensors
    metrics:
    - type: mae
      value: 13.21
      name: Test MAE (µg/m³)
    - type: rmse
      value: 23.43
      name: Test RMSE (µg/m³)
    - type: r2
      value: 0.6762
      name: Test R²
    - type: accuracy
      value: 71.83
      name: EPA AQI Category Accuracy (%)
---

# Next-Day Air Quality (PM2.5) Forecaster for Karachi & Lahore

> **Course:** CT-378 Machine Learning, Semester Project  
> **Team:** Khubsoorat Shklen (Muhammad Sarim Umer, Muhammad Feras Malik, Muhammad Ahmed Khan)  
> **Institution:** Department of Computer Science & Information Technology, NED University

---

## 1. Model Summary

This repository hosts the production **Next-Day Air Quality Forecaster** for **Karachi** and **Lahore**, Pakistan.
The model predicts ground-level **PM2.5 concentration 24 hours into the future ($t + 24\\text{h}$)** in $\\mu\\text{g/m}^3$ using recent air quality trends and weather features. The quantitative forecast is subsequently mapped into standard **US EPA AQI health categories** (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous) with actionable public health advisories.

## 2. Model Architecture

The model is an engineered **Blended Ensemble** combining three complementary model paradigms:
1. **L2-Regularized Linear Model (Ridge Regression, $\\alpha=10.0$):** Provides stable low-variance baseline predictions.
2. **Bagged Decision Trees (Random Forest, 120 estimators, depth 12):** Captures non-linear thresholds and feature splits.
3. **Gradient Boosted Decision Trees (LightGBM, 300 estimators, 45 leaves, lr=0.03):** Optimizes residual corrections sequentially.

- **Preprocessing:** `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder` for the city identifier.
- **Input Features (39 features):**
  - Current PM2.5 and historical lags ($1\\text{h}, 3\\text{h}, 6\\text{h}, 12\\text{h}, 24\\text{h}, 168\\text{h}$)
  - Rolling aggregations ($6\\text{h}, 24\\text{h}, 72\\text{h}$ means, $24\\text{h}$ min/max/std)
  - Momentum & physical indicators ($3\\text{h}$ velocity, $6\\text{h}$ and $24\\text{h}$ change, 2nd-derivative acceleration)
  - Meteorological variables from Open-Meteo at time $t$ (temperature, humidity, surface pressure, wind speed/direction, precipitation, cloud cover)
  - Atmospheric stagnation index: $\\frac{1}{\\text{wind speed}} \\times \\frac{\\text{relative humidity}}{100}$
  - Punjab seasonal crop-burning curve and city-specific winter smog risk
  - Calendar & diurnal cycles (hour sine/cosine, weekday, month, winter flag)

## 3. Training & Validation Setup

- **Dataset:** 24,599 hourly observations collected from 28 ground monitors (12 in Karachi, 16 in Lahore) via OpenAQ (AirGradient sensors) paired with Open-Meteo Historical Weather API.
- **Split Scheme:** Strict chronological 70/15/15 train/validation/test split with an **enforced 24-hour purge gap** to prevent temporal label leakage.
- **Zero Future Weather Leakage:** All weather inputs are observed at prediction time $t$ or preceding 24 hours (no oracle $t+24$ weather).

## 4. Benchmark Performance

Evaluated on the held-out chronological test set (3,678 samples, August 15 – October 31):

| Metric | 24-Hour Persistence Baseline | Final Blended Ensemble | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Overall MAE** | $14.64\\,\\mu\\text{g/m}^3$ | **$13.21\\,\\mu\\text{g/m}^3$** | **+9.7%** |
| **Overall RMSE** | $27.17\\,\\mu\\text{g/m}^3$ | **$23.43\\,\\mu\\text{g/m}^3$** | **+13.8%** |
| **Overall $R^2$** | $0.5765$ | **$0.6762$** | **+0.0997** |
| **EPA AQI Accuracy** | $69.83\\%$ | **$71.83\\%$** | **+2.0 pp** |

### Per-City Breakdown

| City | Test MAE | Test RMSE | Test $R^2$ | EPA AQI Accuracy | Accuracy within $\\pm 1$ Category |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Karachi** | **$6.15\\,\\mu\\text{g/m}^3$** | $9.26\\,\\mu\\text{g/m}^3$ | **$0.7405$** | **$82.44\\%$** | **$99.84\\%$** |
| **Lahore** | **$20.27\\,\\mu\\text{g/m}^3$** | $31.82\\,\\mu\\text{g/m}^3$ | **$0.5735$** | **$61.23\\%$** | **$96.41\\%$** |

*Note on Lahore:* Lahore's $R^2$ exceeds the 24h persistence baseline ($0.4542$) by **+0.1193** (+26% relative increase in variance explained). Severe smog spikes ($>150\\,\\mu\\text{g/m}^3$) are detected with 65.6% precision and 63.0% recall.

## 5. Usage in Python

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Download model pipeline from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="MuhammadSarimUmer/next-day-pm25-forecaster",
    filename="final_pipeline.joblib"
)
pipeline = joblib.load(model_path)

# Example input DataFrame (containing all 39 feature columns)
# df = pd.DataFrame([sample_feature_dict])
# pred_pm25 = pipeline.predict(df)[0]
# print(f"Predicted PM2.5 in 24 hours: {pred_pm25:.1f} µg/m³")
```

## 6. Known Limitations

1. **Regression toward the Mean:** On high-pollution spikes in Lahore ($>125.4\\,\\mu\\text{g/m}^3$), the model tends to underpredict peak values (mean bias: $-33.4\\,\\mu\\text{g/m}^3$) due to mean-squared and mean-absolute loss penalization across the year.
2. **Sensor Grade:** Station observations come from low-cost AirGradient sensors aggregated at the city level, suitable for public advisory but not regulatory compliance.

---
**Repository:** [GitHub](https://github.com/MuhammadSarimUmer/Next-DayAir-Quality-PM2.5-Forecaster)  
**Live Space:** [Hugging Face Space](https://huggingface.co/spaces/MuhammadSarimUmer/next-day-air-quality-forecaster)
"""


def upload_model():
    vals = dotenv_values(ENV_FILE)
    token = vals.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is missing from .env file!")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/next-day-pm25-forecaster"

    print(f"Target repository: {repo_id}")
    print("Creating repository on Hugging Face Model Hub (if it doesn't exist)...")
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print(f"Repository ready: {repo_url}")

    # 1. Upload Model Card README.md
    print("Uploading Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add comprehensive model card"
    )
    print("✓ Model card uploaded.")

    # 2. Upload final_pipeline.joblib
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")

    print(f"Uploading model weights ({MODEL_FILE.name})...")
    api.upload_file(
        path_or_fileobj=str(MODEL_FILE),
        path_in_repo="final_pipeline.joblib",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained 39-feature blended ensemble pipeline"
    )
    print("✓ Model weights uploaded.")

    print("\n" + "=" * 70)
    print(f"SUCCESS: Model published at https://huggingface.co/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    upload_model()
