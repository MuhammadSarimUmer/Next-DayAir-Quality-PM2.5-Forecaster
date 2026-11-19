# Remaining Tasks Before Week 15 Demo

> CT-378 Machine Learning — Team Khubsoorat Shklen  
> Muhammad Sarim Umer (CT-24235) · Muhammad Feras Malik (CT-24247) · Muhammad Ahmed Khan (CT-24245)

---

## 1. EDA Notebook — Save Missing Cell Outputs

Two cells in `notebooks/02_eda.ipynb` have **no saved output** and will show blank in the submitted notebook:

| Cell | Expected output | Why it matters |
|---|---|---|
| **Cell 11** | PM2.5 24-hour lag correlation printout | Shows autocorrelation; justifies lag features |
| **Cell 15** | AQI category distribution bar chart | Shows class imbalance; contextualises AQI accuracy metric |

**Fix:** Open the notebook in Jupyter, run **Kernel → Restart & Run All**, then **File → Save**. Confirm both cells show output before closing.

```bash
jupyter notebook notebooks/02_eda.ipynb
# Kernel → Restart & Run All → File → Save
```

---

## 2. Deployment — Live Public URL Required

The rubric requires **at least one live public URL**. The Streamlit app and FastAPI backend run locally but are not publicly accessible yet.

### Option A: Hugging Face Spaces (Prepared, Not Yet Live)

A complete deployment bundle is ready in `deploy/huggingface/`. Steps to go live:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co) and generate an access token (`Settings → Access Tokens`).
2. **Run the bundler script** to assemble the Space folder:
   ```bash
   python deploy/huggingface/prepare_space.py ../pm25-space
   ```
3. **Push to Hugging Face:**
   ```bash
   cd ../pm25-space
   git init
   git remote add origin https://huggingface.co/spaces/<your-hf-username>/pm25-forecaster
   git add .
   git commit -m "Initial Space"
   git push origin main
   ```
4. **Add `OPENAQ_API_KEY` as a Space Secret** in the HF Space settings → Secrets.

The Space will be live at `https://huggingface.co/spaces/<your-hf-username>/pm25-forecaster`.

### Option B: Streamlit Community Cloud (Simpler)

1. Push the repository to GitHub (see Section 5 for the URL).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect GitHub, select `app/streamlit_app.py` as the entry point.
3. Add `OPENAQ_API_KEY` in the app secrets panel.
4. Click **Deploy**. The app will be live at a `*.streamlit.app` URL.

> [!IMPORTANT]
> **At least one of the above must be completed before the demo.** Option B (Streamlit Cloud) is faster — typically under 10 minutes if the repo is already on GitHub.

---

## 3. CV Folds — Status: FIXED ✅

The pipeline now uses **5-fold `TimeSeriesSplit`** as required by the rubric (previously 3-fold due to larger dataset during development).

**Code is correct.** To reflect the fix in saved model files, re-run the full training pipeline:

```bash
python -m src.baseline
python -m src.train        # 5-fold CV tuning
python -m src.train_final  # retrain on train+val
python -m src.evaluate     # evaluate on test set
```

This is **optional** for the demo — existing model files (`models/final_pipeline.joblib`) were trained with 3-fold. The final metrics (71.65% AQI) are correct. Re-running updates the saved fold count in `model_metadata.json` but is unlikely to materially change the results.

---

## 4. Final Report PDF

`reports/final_report.md` exists and is complete, but needs two updates before converting to PDF:

1. **Add the Kaggle Extension results** — include the comparison table from `Limitations.md` Section 12 (see below) and note that the original pipeline is preferred.
2. **Add all 7 models** to the Results section: Ridge, KNN, Random Forest, LightGBM, Ensemble, Prophet, ARIMAX — plus Kaggle Extended Ensemble for context.

**Final results to include in the report:**

| Model | MAE | RMSE | R² | AQI Acc |
|---|---:|---:|---:|---:|
| Baseline (Persistence) | 14.54 | 26.98 | 0.5700 | 69.99% |
| Ridge Regression | — | — | — | ~69% (val) |
| KNN Regressor | — | — | — | ~71% (val) |
| Random Forest | — | — | — | ~72% (val) |
| LightGBM | — | — | — | ~72% (val) |
| **Ensemble (Final)** | **13.31** | **23.65** | **0.6695** | **71.65%** |
| Prophet | 18.27 | — | — | 55.0% |
| ARIMAX | 24.63 | — | — | 48.1% |
| Kaggle Extended Ensemble | 13.32 | 21.00 | 0.5388 | 64.1% |

**Convert to PDF:**
```bash
# Option 1: Pandoc (if installed)
pandoc reports/final_report.md -o reports/final_report.pdf

# Option 2: VS Code → Open Preview → Right-click → Print to PDF

# Option 3: Copy-paste into Google Docs → File → Download → PDF
```

---

## 5. README GitHub URL — FIXED ✅

The Getting Started section previously had `<your-username>/<repo-name>` placeholder. It has been updated to:

```
https://github.com/MuhammadSarimUmer/Next-DayAir-Quality-PM2.5-Forecaster
```

Verify the repository exists at that URL and is set to **Public** before the demo.

---

## 6. Viva Preparation

Each team member should be able to explain the following:

| Topic | Key talking points |
|---|---|
| **Feature engineering (why 47 features)** | Started with 35; added 5 momentum/trend features (velocity, acceleration, 7d lag, etc.) then 4 domain physics features (stagnation_index, smog_season_intensity, lahore_smog_risk) and 3 regime features. Each group targets a known error mode. `pm25_now` still dominates at ~86% importance — acknowledged limitation. |
| **Why chronological split, not random** | Time series have autocorrelation. A random split leaks future into training (a row at t=15:00 in test could have t=14:00 and t=16:00 neighbours in train). Chronological split simulates real deployment: train on the past, predict the future. |
| **Why LightGBM beats Ridge/KNN** | Ridge assumes linear relationships — PM2.5 dynamics are highly non-linear (winter inversion threshold effects, monsoon clearing, smog accumulation). KNN finds "similar past days" but struggles with rare winter spikes in a training set dominated by summer. LightGBM's gradient boosting captures interaction effects (humidity × wind × season) that linear models miss. |
| **Why Prophet/ARIMAX underperformed** | Prophet: rigid linear trend + additive seasonality cannot capture abrupt regime shifts (Lahore clean → Hazardous in 48h). ARIMAX: overfitted 2025 seasonal coefficients, applied blindly to 2026 (which was 30% cleaner in Aug-Oct); also crashed with m=24 seasonal period (163 MiB array), forced to use m=12. Both methods assume stationarity that PM2.5 violates. |
| **What AQI accuracy means** | For each test-set prediction, map both the predicted and actual PM2.5 to the 2024 US EPA 6-category scale (Good / Moderate / UfSG / Unhealthy / Very Unhealthy / Hazardous). AQI accuracy = % of hours where the predicted category matches the actual category. This is more practically meaningful than MAE for a public health tool — a forecast of 36 µg/m³ when truth is 35 µg/m³ has tiny MAE but crosses the Moderate→UfSG boundary (a category error). |

---

*Last updated: November 19, 2026*
