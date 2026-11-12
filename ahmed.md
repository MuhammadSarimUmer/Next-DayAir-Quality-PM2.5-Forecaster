# Ahmed's Work Log

**Author:** Muhammad Ahmed Khan (CT-24245)  
**Project:** Next-Day Air Quality (PM2.5) Forecaster for Karachi and Lahore  
**Date:** 12 November 2026

This file records exactly what was implemented for the remaining project stages
(13-20), which files were added or changed, what was verified, what could not be
done in this environment, and which alternatives were considered and rejected.

---

## 1. Starting point

The repository already contained a completed data and modelling pipeline:

- OpenAQ PM2.5 collection, Open-Meteo weather, and CAMS aerosol collection
- cleaning, EDA, 47-feature engineering, chronological split with purge gap
- preprocessing pipeline, baseline, Ridge/KNN/Random Forest/LightGBM tuning
- model comparison and saved evaluation code
- an AQI module draft and two small test files

What was missing (and what I completed):

1. A deployable model artifact (the `models/` directory was absent because it is
   gitignored)
2. Error analysis (stage 13)
3. AQI module hardening and per-breakpoint tests (stage 14)
4. Live-data inference (stage 15)
5. Streamlit app (stage 16)
6. FastAPI `/predict` (stage 17)
7. Hugging Face Spaces deployment bundle (stage 18)
8. Final report, limitations/ethics, README polish (stage 19)
9. Demo script and slides (stage 20)

I also created this `ahmed.md` as requested.

---

## 2. What I did, stage by stage

### Stage 13 - Error analysis

**Created:** `src/error_analysis.py`  
**Generated:** `reports/error_analysis.md`, `reports/error_analysis_summary.json`,
`reports/error_by_month.csv`, `reports/error_by_city.csv`,
`reports/error_by_hour.csv`, `reports/error_by_pm_bin.csv`,
`reports/error_by_regime.csv`, `reports/worst_predictions.csv`,
`reports/worst_days.csv`, `reports/error_confusion.csv`, and seven figures under
`reports/figures/error_*.png`.

The script loads the saved final ensemble, predicts the held-out test split once,
and analyses:

- overall and per-city MAE/RMSE/R2/AQI accuracy and bias
- error by true PM2.5 category
- error by month and hour of day
- spike, onset, recovery, and stable regimes
- worst city-days and worst individual predictions
- row-normalised AQI category confusion matrix

**Key results from the actual run:**

| Scope | MAE | RMSE | R2 | AQI accuracy | Bias |
|---|---:|---:|---:|---:|---:|
| Overall | 13.31 | 23.65 | 0.6695 | 71.6% | -1.61 |
| Karachi | 6.20 | 9.46 | 0.7282 | 82.0% | -0.35 |
| Lahore | 20.42 | 32.08 | 0.5591 | 61.3% | -2.88 |

- Top-5% spikes (actual >= 136.4 ug/m3): MAE 41.43, bias +36.97, 90.3% underpredicted.
- Smog onsets (actual >= 125 and rise >= 50): MAE 63.26, bias +63.26.
- Clearing/recovery events: MAE 91.95, bias -91.95 (overprediction).
- Lahore October 2026: MAE 30.79, bias -4.95, AQI accuracy 60.6%.
- Worst day: Lahore 29 October 2026, MAE 108.82 because current PM2.5 was
  250-300 while the next day fell to 55-90.
- Most category errors are one AQI band away, not several.

These findings were written into the final report and the Limitations document.

**Why a script plus markdown instead of only a notebook?** A script is reproducible
from the command line, writes machine-readable CSVs/JSON, and can be rerun after a
model change. A notebook is better for exploration, but it is easier for output to
drift from the code. The existing EDA remains a notebook; the error analysis is now
a repeatable stage.

**Alternatives considered for error analysis:**

| Alternative | Why I did not use it |
|---|---|
| Jupyter-only analysis | Harder to reproduce in CI and easy to leave stale outputs; chosen script writes every table/figure deterministically. |
| SHAP or LIME explanations | Adds a dependency and runtime; the task asked where the model fails, and residual/regime analysis answers that directly. Feature importance was already discussed in the earlier modelling stage. |
| Statistical significance tests | The test split is a single chronological block, so standard i.i.d. significance tests would be misleading. |
| Per-station error analysis | The model is trained on city means, not station-level targets; a station comparison would not evaluate the actual deployed target. |
| Only aggregate MAE | It hides the Lahore smog failures, which are the most important practical result. |

### Stage 14 - AQI module and per-breakpoint tests

**Changed:** `src/aqi.py`, `tests/test_aqi.py`

I found and fixed a real boundary bug. The original code looped over ranges such as
`9.1 <= pm25 <= 35.4`; a value like `9.05` matched no range and fell through to
`calculate_aqi(...) == 500` and `"Hazardous"`. The fix truncates the concentration
to the EPA table's 0.1 ug/m3 resolution before looking up the range, so no value can
fall through a nominal gap.

I also:

- made `calculate_aqi` and `get_aqi_category_info` reject NaN/None with `ValueError`
- kept the standard EPA linear interpolation formula
- capped the extrapolation above 500.4 ug/m3
- preserved the structured advice API (`summary`, `general_public`,
  `sensitive_groups`, `actions`, colour)

`tests/test_aqi.py` now has a parametrized test for **every breakpoint**, including
both sides of each boundary:

`-5, 0, 9.0, 9.1, 35.4, 35.5, 55.4, 55.5, 125.4, 125.5, 225.4, 225.5, 500.4`

It also tests gap values (9.05, 35.45), extreme values, invalid input, monotonicity,
colours, and the advice structure. Result: **19 tests passed** for this file;
**22 tests passed** for the whole suite.

**Why the 2024 US EPA table?** The existing README, metrics module, and report all
use it, so changing standards would make historical metrics incomparable. Pakistan
does not have an equivalent official national AQI table configured in the project.
The limitation is documented.

**Alternatives considered for AQI:**

| Alternative | Why I did not use it |
|---|---|
| WHO 2021 interim targets | Useful for health messaging, but they are concentration targets, not the 0-500 AQI index used by the existing metrics. |
| Pakistan EPA / national AQI table | No stable, machine-readable official table was available in the project; mixing standards would break comparability. |
| Simple threshold lookup without interpolation | Loses the standard AQI index value and makes adjacent categories discontinuous. |
| Rounding instead of truncation | Rounding can still leave ambiguous boundary values; EPA-style truncation to 0.1 is deterministic. |
| Returning AQI only, no advice | The product requirement is category plus actionable advice for general and sensitive groups. |

### Stage 15 - Live-data inference

**Created:** `src/live_data.py`  
**Also created:** `data/processed/live_fallback.csv` (latest complete row per city)

`live_data.py` fetches:

- the last ~10 days of hourly PM2.5 from every configured OpenAQ sensor for the city
- recent weather from the Open-Meteo forecast endpoint (`past_days=10`)
- recent CAMS PM2.5, dust, and aerosol optical depth

It reuses `clean_sensor` for sensor QC, aligns everything in Pakistan Standard Time,
interpolates only short internal gaps, joins the sources, and calls the same
`build_city_features` function used in training. It returns the latest row with all
47 features present.

The function has an explicit offline fallback. If the OpenAQ key is missing, the
network is down, or no complete row can be built, it loads the latest saved row from
`data/processed/live_fallback.csv` and labels the result `"offline"`. This prevents
the product from pretending stale data is live.

**Verified behaviour without an API key:**

```text
Source: offline
Notes: Using saved snapshot (RuntimeError: OPENAQ_API_KEY missing...)
```

**Alternatives considered for live data:**

| Alternative | Why I did not use it |
|---|---|
| IQAir / WAQI / AQICN APIs | Require separate keys and have different station coverage; they would not match the OpenAQ training distribution. |
| Scraping public websites | Fragile, potentially against terms of service, and hard to reproduce. |
| Open-Meteo forecast weather for the target day | The model was trained with weather at time `t` as an input, not future weather. Feeding future weather would change the learned relationship and could leak information. |
| OpenAQ only, no weather/CAMS | The 47-feature model requires weather and CAMS columns; missing groups would force imputation for every request. |
| Live fetch only, no fallback | Demos and the HF Space would break without a key or during outages. The fallback is labelled, so it is honest. |
| A database/cache service | Overkill for a course project; Streamlit caching plus a CSV snapshot is enough. |

### Stage 16 - Streamlit app

**Created:** `app/streamlit_app.py` and `app/__init__.py`

The app provides:

- city selector (Lahore / Karachi)
- live APIs with fallback, or saved snapshot only
- optional experimental clearing correction (off by default)
- a colour-coded forecast card with predicted PM2.5, AQI, and category
- observed PM2.5, input hour, and data source
- summary, general-public advice, sensitive-group advice, and actions
- an input snapshot table and a data-freshness/staleness warning
- a manual AQI/advice simulator
- model metadata in the sidebar

It caches the model and live inputs, shows a clear error if the model artifact is
missing, and warns when inputs are more than 30 hours old.

**Verified:** Streamlit started headless on port 8502 and
`GET /_stcore/health` returned `200 ok`.

**Why Streamlit?** It is already in `requirements.txt`, is the fastest way to ship a
data app, and is directly supported by Hugging Face Spaces.

**Alternatives considered for the UI:**

| Alternative | Why I did not use it |
|---|---|
| Gradio | Excellent for model demos, but less control over a multi-section dashboard and custom AQI card. |
| Dash / Plotly | More code and a heavier callback model for a simple form plus result panel. |
| React/Next.js frontend | Overkill; would add a separate build and deployment pipeline. |
| Static HTML + JS | Would duplicate AQI logic or require a server anyway; FastAPI is already available. |
| Streamlit multipage | One page is enough for this demo; extra pages add navigation overhead without new functionality. |

### Stage 17 - FastAPI `/predict`

**Created:** `app/api.py`

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service links |
| GET | `/health` | Model status, feature count, test metrics |
| POST | `/predict` | Live/offline forecast or caller-supplied feature row |
| POST | `/aqi` | PM2.5 -> AQI + advice without a model |
| GET | `/metadata` | Saved model metadata |

`/predict` accepts `city`, `offline`, `prefer_offline`, `postprocess`, and an optional
complete `features` dictionary. If features are supplied, it validates that all 47
columns are present and predicts directly; otherwise it uses `live_data.py`. Responses
are typed with Pydantic models.

**Verified with FastAPI `TestClient`:** `/health` 200, `/predict` offline 200,
`/aqi` for 35.5 ug/m3 returned AQI 101.

**Why FastAPI?** Pydantic validation and automatic Swagger docs at `/docs` make the
demo easier to inspect. Flask would need manual validation and extra documentation.

**Alternatives considered for the API:**

| Alternative | Why I did not use it |
|---|---|
| Flask | Simple, but no built-in typed validation or OpenAPI docs. |
| Django REST Framework | Too much structure for two endpoints. |
| gRPC | Not browser-friendly and unnecessary for a JSON demo. |
| Serving only Streamlit | Does not demonstrate how another application would consume the model. |

### Stage 18 - Hugging Face Spaces deployment

**Created:** `deploy/huggingface/README.md`, `deploy/huggingface/requirements.txt`,
`deploy/huggingface/prepare_space.py`

The Space bundle uses the Streamlit SDK and `app_file: app/streamlit_app.py`.
`prepare_space.py` copies only the required files into a target directory:

- Space README with YAML metadata
- Space requirements
- `app/` and `src/`
- `models/final_pipeline.joblib` and `models/model_metadata.json`
- `data/processed/live_fallback.csv`
- `DATA_SOURCES.md` and `Limitations.md`

**Important:** I did **not** push to Hugging Face. That requires a Hugging Face
account, a repository, and a write token, none of which are available in this
environment. The bundle and exact push steps are ready.

```powershell
python deploy/huggingface/prepare_space.py ../pm25-space
cd ../pm25-space
git init
git remote add origin https://huggingface.co/spaces/<user>/<space>
git add .
git commit -m "Deploy PM2.5 forecaster"
git push
```

Add `OPENAQ_API_KEY` as a Space secret for live mode. Without it the Space uses the
bundled offline snapshot.

**Alternatives considered for deployment:**

| Alternative | Why I did not use it |
|---|---|
| Docker Space | More control, but more files and a larger build; Streamlit SDK is simpler and sufficient. |
| Gradio Space | Would require rewriting the UI. |
| Render / Railway / Fly.io | Free tiers change and need account setup; Hugging Face is the course-requested target. |
| GitHub Pages | Static hosting cannot run the Python model or API. |
| AWS/GCP free tier | Too much operational overhead and billing risk for a course demo. |
| Push now from this environment | Impossible without credentials; I documented it rather than pretending it was deployed. |

### Stage 19 - Final report, limitations/ethics, README polish

**Created:** `reports/final_report.md`  
**Updated:** `README.md`, `Limitations.md`

The final report contains:

- abstract, introduction, problem definition
- data sources, cleaning, and EDA
- the 47-feature design and leakage controls
- chronological split and purge gap
- models, tuning, and ensemble selection
- fresh validation and test results (including the actual baseline comparison)
- error analysis findings
- AQI module, product/deployment description
- limitations and ethics
- reproduction commands and conclusion

I updated the README's outdated 35-feature/old-metric sections to the current
47-feature run and added the new source files, report links, deployment instructions,
and completed timeline. I updated `Limitations.md` with current metrics, the rejected
post-processing results, and removed a reference to a file that no longer exists in
the final source tree.

**Why markdown?** It renders on GitHub, is diffable, and does not add a LaTeX
toolchain dependency. The report can still be exported to PDF if the course requires
it.

**Alternatives considered for reporting:**

| Alternative | Why I did not use it |
|---|---|
| LaTeX/PDF | Better typesetting, but adds a toolchain and makes review/diffing harder in this environment. |
| Word document | Binary, hard to review in git, and not reproducible from the repository. |
| Notebook-only report | Mixes code and narrative and is harder to read as a standalone submission. |
| Keep the old README numbers | They came from an earlier 35-feature run and no longer matched the current data/model. |

### Stage 20 - Demo script and slides

**Created:** `reports/demo_script.md` and `reports/slides.md`

The demo script is a 6-8 minute walkthrough: problem, live result, pipeline, metrics,
failure analysis, API, and close, plus likely Q&A and exact commands. The slides are
a Marp-compatible Markdown deck with 15 slides and the error figures.

**Why Marp Markdown?** It is plain text, reviewable in git, and can be exported to
PDF/PPTX/HTML with Marp when needed. Creating a binary `.pptx` directly would add a
`python-pptx` dependency and make the content harder to diff.

**Alternatives considered for slides:**

| Alternative | Why I did not use it |
|---|---|
| PowerPoint `.pptx` | Binary; would require `python-pptx` and is harder to version. |
| Google Slides | Requires an online account and manual work; not reproducible from the repo. |
| Reveal.js HTML | Nice, but extra build/CDN setup for little benefit over Marp. |
| Separate speaker-notes document | Already included in `demo_script.md`; duplicating it would create drift. |

---

## 3. Extra work required to make the product runnable

### 3.1 Retrained the final model

The `models/` folder was not present (it is gitignored). Without it, the apps could
not run. I created `src/train_final.py`, which:

- builds the documented winning ensemble: Ridge (0.2) + Random Forest (0.3) +
  LightGBM (0.5)
- retrains on train + validation (19,471 rows)
- evaluates the untouched test split once
- saves `models/final_pipeline.joblib` (18 MB) and `models/model_metadata.json`

**Actual test metrics:**

| Metric | Baseline | Ensemble |
|---|---:|---:|
| MAE | 14.54 | **13.31** |
| RMSE | 26.98 | **23.65** |
| R2 | 0.5700 | **0.6695** |
| AQI accuracy | 69.99% | **71.65%** |

Per city: Karachi MAE 6.20 / AQI 82.0%; Lahore MAE 20.42 / AQI 61.3%.

I also reran `python -m src.train` to refresh the validation comparison:

| Model | Val MAE | Val RMSE | Val R2 | Val AQI |
|---|---:|---:|---:|---:|
| Ridge | 10.23 | 15.29 | 0.2645 | 69.2% |
| KNN | 9.08 | 14.18 | 0.3679 | 71.5% |
| Random Forest | **8.43** | **13.67** | **0.4121** | 71.9% |
| LightGBM | 8.49 | 13.78 | 0.4023 | 71.9% |
| Ensemble | 8.45 | 13.70 | 0.4099 | **72.4%** |

**Why not use the old saved numbers?** The current data has 47 features and a
different split size than the earlier 35-feature README. Reporting old numbers would
be inaccurate.

**Alternatives considered for the final model:**

| Alternative | Why I did not use it |
|---|---|
| Keep only Random Forest | It had validation MAE 8.43 (best by 0.02), but the project already declared a 0.2 ug/m3 ensemble stability rule; the ensemble also had the best validation AQI accuracy. |
| City-specific models | An earlier experiment performed worse (67.4% vs ~70.9% AQI accuracy); Lahore's smaller summer validation set lost the regularisation from Karachi. |
| LSTM / Transformer | Only ~16 months of hourly data; deep sequence models are more likely to overfit and are harder to deploy. |
| XGBoost instead of LightGBM | LightGBM was already tuned and in the requirements; switching would add a dependency without a clear expected gain. |
| Use Kaggle 2021-24 history | `USE_KAGGLE_HISTORY` is off because the label provenance changes between Kaggle API series and OpenAQ ground sensors, and there is a 2024-2025 gap. |
| Enable clearing blend | It reduced test performance (MAE 13.51 vs 13.31, AQI 71.3% vs 71.65%). |
| Enable conservative AQI buffers | It reduced AQI accuracy (69.2% vs 71.65%) on this split. |

### 3.2 Fixed a Windows console crash

`python -m src.train` crashed after saving the first model because it printed a
Unicode check mark on a cp1252 console. I replaced the check marks with `OK` in
`src/train.py` and `src/metrics.py`. This is a real reproducibility fix, not only a
cosmetic one.

### 3.3 Generated the offline fallback snapshot

`python -m src.live_data --refresh-fallback --offline` saves the latest complete
feature row for each city. This makes the Streamlit app, API, tests, and HF Space
usable without an OpenAQ key or network.

### 3.4 Git ignore updates

The model and the key generated reports were gitignored. I changed `.gitignore` to
allow:

- `models/final_pipeline.joblib`
- `models/model_metadata.json`
- the seven error-analysis figures
- `reports/error_analysis.md`, `reports/final_report.md`,
  `reports/demo_script.md`, `reports/slides.md`

This is required so a cloned repository or HF Space actually contains the model and
the deliverables.

---

## 4. Files added or changed

### Added

| File | Purpose |
|---|---|
| `src/train_final.py` | Train and save the deployable ensemble + metadata |
| `src/error_analysis.py` | Reproducible error analysis and figures |
| `src/live_data.py` | Fresh-input inference plus offline fallback |
| `src/predict.py` | Shared model loading/prediction/AQI helper |
| `app/streamlit_app.py` | Streamlit product UI |
| `app/api.py` | FastAPI service |
| `app/__init__.py` | Package marker |
| `deploy/huggingface/README.md` | HF Space YAML + documentation |
| `deploy/huggingface/requirements.txt` | Space dependencies |
| `deploy/huggingface/prepare_space.py` | Bundle builder |
| `reports/final_report.md` | Final report |
| `reports/error_analysis.md` + CSVs/JSON | Error analysis outputs |
| `reports/figures/error_*.png` | Seven analysis figures |
| `reports/demo_script.md` | 6-8 minute demo walkthrough |
| `reports/slides.md` | Marp slide deck |
| `data/processed/live_fallback.csv` | Latest complete row per city |
| `LICENSE` | MIT licence matching the Hugging Face Space metadata |
| `models/final_pipeline.joblib` | Deployed model |
| `models/model_metadata.json` | Metrics, features, timestamp |
| `ahmed.md` | This log |

### Changed

| File | Change |
|---|---|
| `src/aqi.py` | Fixed breakpoint gap bug, NaN/None validation, 0.1 truncation |
| `tests/test_aqi.py` | Per-breakpoint parametrized tests and edge cases |
| `src/train.py` | Replaced Windows-incompatible check marks |
| `src/metrics.py` | Replaced Windows-incompatible check marks |
| `.gitignore` | Allow model and key report artifacts |
| `README.md` | 47 features, current metrics, actual structure, deployment, timeline |
| `Limitations.md` | Current metrics, rejected post-processing, removed stale file reference |

---

## 5. Verification performed

| Check | Command | Result |
|---|---|---|
| All tests | `python -m pytest tests -q` | **22 passed** |
| AQI breakpoint tests | `python -m pytest tests/test_aqi.py -q` | **19 passed** |
| Final model training | `python -m src.train_final` | Saved 18 MB pipeline + metadata |
| Validation comparison | `python -m src.train` | All five models trained; ensemble selected |
| Baseline evaluation | `python -m src.baseline` | Test MAE 14.54, AQI 69.99% |
| Error analysis | `python -m src.error_analysis` | Wrote report, CSVs, JSON, 7 figures |
| Live fallback | `python -m src.live_data --city lahore` | Fell back cleanly without API key |
| FastAPI | `TestClient` `/health`, `/predict`, `/aqi` | 200/200/200 |
| Streamlit | Headless start + `/_stcore/health` | `200 ok` |
| Compilation | `python -m compileall src app` | No syntax errors |

---

## 6. What I could not do, and why

1. **Push the Hugging Face Space.** This requires a Hugging Face account and write
   token. The Space bundle, YAML, requirements, and prepare script are complete.
2. **Test live OpenAQ collection end-to-end.** No `OPENAQ_API_KEY` is configured in
   this environment. The fallback path was tested instead. The live path reuses the
   already-tested `openaq_api.get` and `clean_sensor` functions.
3. **Test the HF Space runtime itself.** It cannot be created/pushed without an
   account, so only the local Streamlit and API runtimes were verified.
4. **Reference-grade validation.** The available OpenAQ stations are community
   sensors; the project has no reference monitor to calibrate against.
5. **More Lahore winters.** The public sensors do not provide pre-2025 hourly ground
   data, which is the main reason Lahore smog transitions remain hard.

---

## 7. Recommended next steps

1. Push `deploy/huggingface` to a real Space and add `OPENAQ_API_KEY` as a secret.
2. Add a scheduled retraining job (monthly) and a drift check on input distributions.
3. Add external fire/crop-burning indicators and more Lahore winters.
4. Revisit city-specific models with enough Lahore data to avoid the earlier
   overfitting.
5. Add an integration test that runs one live request in CI when a key is available.
6. Optionally export `reports/slides.md` to PDF/PPTX with Marp for submission.

---

## 8. Quick run guide

```powershell
# Install
pip install -r requirements.txt

# Data and model (only if rebuilding from scratch)
python -m src.collect
python -m src.clean
python -m src.features
python -m src.split
python -m src.train_final

# Product
python -m src.live_data --refresh-fallback --offline
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload --port 8000

# Verification
pytest tests -q
python -m src.error_analysis
```

---

## 9. Verification rerun (12 November 2026)

This section records what I actually executed during the final verification pass for
stages 13-20. The implementation itself was already present in the working tree; this
pass re-ran every deliverable end to end, regenerated the analysis outputs, exercised
the product surfaces, and checked the Hugging Face bundle locally. **Nothing was
pushed to GitHub or Hugging Face, and no commit was created.** The repository still
shows the same uncommitted working-tree changes that existed before this pass.

### 9.1 What I ran and what happened

| Stage | Command / action | Observed result |
|---|---|---|
| 13 | `python -m src.error_analysis` | Rewrote `reports/error_analysis.md`, the 8 CSVs/JSON, and the 7 figures. Overall MAE 13.31, RMSE 23.65, R2 0.6695, AQI 71.6%; Karachi 6.20 / 82.0%; Lahore 20.42 / 61.3% |
| 14 | `python -m pytest tests/test_aqi.py -q` | 19 passed (every breakpoint, gap values, extremes, invalid input, monotonicity, colours, advice structure) |
| 14 | `python -m pytest tests -q` | 22 passed in 1.00 s |
| 15 | `python -m src.live_data --city lahore --offline` | Source `offline`, feature hour `2026-10-30 23:00` PKT, PM2.5 131.0, staleness 310.5 h, notes explain offline mode |
| 15 | `python -m src.live_data --city karachi --offline` | Source `offline`, same feature hour, PM2.5 103.9, staleness 310.5 h |
| 16 | Headless `streamlit run app/streamlit_app.py --server.port 8502` + `GET /_stcore/health` | `200 ok` |
| 17 | FastAPI `TestClient`: `GET /`, `GET /health`, `GET /metadata` | All `200` |
| 17 | `POST /predict {"city":"lahore","offline":true}` | `200`, predicted 142.74 ug/m3, category `Very Unhealthy`, source `offline` |
| 17 | `POST /aqi {"pm25":35.5}` | `200`, AQI 101, category `Unhealthy for Sensitive Groups` |
| 17 | `POST /predict {"city":"nowhere"}` | `422` validation error (correct rejection) |
| 18 | `python deploy/huggingface/prepare_space.py <temp>/pm25-space` | Bundle built with `app/`, `src/`, model + metadata, fallback CSV, README, requirements, and licence/data docs |
| 19/20 | Read `reports/final_report.md`, `Limitations.md`, `reports/demo_script.md`, `reports/slides.md`, `README.md` | Numbers and claims match `models/model_metadata.json` (MAE 13.31, AQI 71.65%, 3,492 test rows) |
| All | `python -m compileall -q src app deploy` | Exit code 0, no syntax errors |

I also confirmed that the offline snapshot is now ~310 hours old (the newest data is
2026-10-30 and the run date is 2026-11-12). The Streamlit app already warns above 30
hours and labels the result `offline`, so this is surfaced rather than hidden.

### 9.2 Alternatives I considered for this verification pass

| Alternative | Why I did not use it |
|---|---|
| Trust the existing artefacts without re-running anything | Outputs can silently drift from code. Re-running stage 13 and the tests is cheap and proves the committed code still produces the reported numbers. |
| Rebuild the whole pipeline from raw data (`collect -> clean -> features -> split`) | No `OPENAQ_API_KEY` is configured and the collection window depends on a live API. Re-downloading would also rewrite the datasets, making the verification non-reproducible against the current model. I verified the deployed artefact instead. |
| Retrain the final ensemble (`python -m src.train_final`) | The saved metadata already matches the held-out metrics, and retraining without new data only risks changing the artefact for no benefit. Retraining is documented as a separate reproduction step. |
| Start a real `uvicorn` server and curl it | FastAPI's `TestClient` exercises the same app, routes, validation, and response models without port conflicts or a background process to clean up. |
| Drive the Streamlit UI with a browser automation tool | The health endpoint plus `compileall` proves the app imports and serves. Browser automation would add a heavy dependency and flaky timing to a course project. |
| Push the Space to Hugging Face now | Requires an account and write token, and the instruction was explicitly not to push. I built the complete bundle locally and left the documented `git`/`huggingface` steps for the account owner. |
| Commit or push the working tree to GitHub | Explicitly disallowed for this task. I left all changes uncommitted. |
| Convert `slides.md` to `.pptx` or PDF | Marp Markdown stays diffable and has no extra dependency; the deck can be exported on demand. |
| Add SHAP/LIME or new features during verification | Out of scope for a verification pass; those belong to modelling experiments, and section 2 already documents why residual/regime analysis was preferred. |

### 9.3 Notes and minor caveats

- FastAPI's `TestClient` emits a Starlette deprecation warning suggesting `httpx2`.
  It does not affect responses; pinned versions in `requirements.txt` remain valid for
  the demo.
- Re-running stage 13 overwrote the generated error-analysis report, CSVs, JSON, and
  figures with identical metrics. These are the only files this pass regenerated.
- The only file edited by hand in this pass is `ahmed.md` itself.
- `git status` still reports the pre-existing uncommitted changes; no commit, tag, or
  push was performed.

---

## 10. Tasks 13-20 completion pass (12 November 2026, with live key)

This pass completed the remaining checklist items with the real OpenAQ key supplied
by the user. The key was stored only in the project-root `.env` file, which
`.gitignore` already excludes. It is not printed by the application, not copied into
the Hugging Face bundle, and not committed.

### 10.1 What I did

| Task | Action | Verified result |
|---|---|---|
| 13 | Re-ran `python -m src.error_analysis` | Same metrics: MAE 13.31, RMSE 23.65, R2 0.6695, AQI 71.6%. Regime/month tables and 7 figures regenerated. |
| 13 | Embedded the error graphs directly in `reports/final_report.md` | Added monthly MAE, Lahore smog, confusion matrix, and actual-vs-predicted figures under section 7.1. |
| 14 | Reviewed `src/aqi.py` and `tests/test_aqi.py` | 13 boundary cases cover both sides of all 6 EPA breakpoints, plus gap/extreme/invalid/monotonicity/advice tests. 19 tests pass. |
| 15 | Created `.env` with `OPENAQ_API_KEY` | Key loaded by `python-dotenv`; `.env` is gitignored. |
| 15 | Ran live inference for both cities | **Lahore: source `live`, 290.1 µg/m³, 2026-11-12 20:00 PKT, staleness 1.8 h. Karachi: source `live`, 45.5 µg/m³, same hour, staleness 1.8 h.** Open-Meteo weather and CAMS each returned 288 rows. |
| 15 | Re-checked auto fallback | Without the key (or on a network error) the same function returns `source=offline` and labels the snapshot, as before. |
| 16 | Added charts to `app/streamlit_app.py` | 7-day observed PM2.5 line chart with 24-hour rolling mean, now-vs-tomorrow bar chart, and an accuracy expander with the test scatter/monthly-error/confusion figures. Also migrated deprecated `use_container_width` to `width="stretch"`. |
| 16 | Ran an in-process Streamlit `AppTest` | Both live and offline modes completed with **0 exceptions**; widgets, metrics, table, 2 charts, and 3 images rendered. Headless health endpoint still returns `200 ok`. |
| 17 | Added `tests/test_api.py` | Covers `/health`, offline `/predict`, invalid city (422), incomplete features (422), caller-supplied features (200), and `/aqi` (200 + 422). |
| 17 | Ran full test suite | **28 passed** (19 AQI + 3 pipeline + 6 API). |
| 18 | Finalised the HF bundle and instructions | Added step-by-step deployment, explicit "do not commit `.env`" guidance, and raised minimum package versions to match the saved model. Built the bundle locally (31 files); **no deployment performed**, as requested. |
| 19 | Finalised documentation | `final_report.md` now has embedded graphs and a verification subsection; `README.md` documents the new charts, API tests, and `.env` handling; `Limitations.md` already matched the current metrics. |
| 20 | Finalised presentation | `slides.md` now has a dedicated **Viva questions** slide (workflow, results, and Q&A), keeping the deck at ~16 slides. `demo_script.md` now includes the charts and live-verification Q&A. |

### 10.2 Alternatives considered in this pass

| Alternative | Why I did not use it |
|---|---|
| Store the key directly in `config.py` or the source | That would leak a secret into version control and the HF bundle. `.env` is gitignored and loaded through `python-dotenv`; the HF Space uses a secret instead. |
| Keep using the offline snapshot only | The user supplied a working key, and live inference is a named deliverable. Live was tested; the offline fallback is retained and still labelled. |
| Fetch OpenAQ inside the Streamlit cache without a fallback | Network outages would break the demo. The existing labelled fallback is safer and honest. |
| Use Plotly/Altair/matplotlib for the new charts | Streamlit's native `line_chart`/`bar_chart` need no new dependency and are sufficient for a trend and a two-bar comparison; matplotlib is already used only for the saved report figures. |
| Render all seven error figures in the app | Three are enough for the demo; the rest stay in `reports/figures/` and `error_analysis.md` to avoid clutter. |
| Test the API with a live `uvicorn` process + curl | `TestClient` is deterministic, covers validation paths, and needs no port or background process. One manual live check was already done for confidence. |
| Pin exact package versions in the Space requirements | Exact pins can conflict with the Space builder; raised minimums (`scikit-learn>=1.9`, `streamlit>=1.66`) keep the saved pipeline loadable without over-constraining. |
| Put viva questions in a separate document | The requirement was a slide deck containing them; a dedicated slide keeps the presentation self-contained. |
| Deploy the Space from this machine | The user explicitly said the friend will handle deployment. The bundle and instructions are ready; no account/token was used. |
| Commit or push the repository | Not requested. The working tree remains uncommitted and nothing was pushed to GitHub or Hugging Face. |

### 10.3 Final verification snapshot

| Check | Command / method | Result |
|---|---|---|
| Unit + contract tests | `python -m pytest tests -q` | 28 passed |
| AQI breakpoint collection | `pytest tests/test_aqi.py --collect-only -q` | 19 tests |
| Error analysis | `python -m src.error_analysis` | MAE 13.31 / AQI 71.6%, outputs regenerated |
| Live data (Lahore) | `python -m src.live_data --city lahore` | `live`, 290.1 µg/m³, 1.8 h stale |
| Live data (Karachi) | `python -m src.live_data --city karachi` | `live`, 45.5 µg/m³, 1.8 h stale |
| Weather / CAMS fetch | `fetch_weather`, `fetch_cams` | 288 rows each |
| Streamlit runtime | `AppTest` live + offline; headless health | 0 exceptions; `200 ok` |
| FastAPI (offline/validation) | `TestClient` endpoints | 200/200/422 as expected |
| FastAPI (live) | `POST /predict {"city":"lahore"}` | 200, source `live`, predicted 270.79 µg/m³, AQI 334 (Hazardous), staleness 1.9 h |
| HF bundle | `prepare_space.py` to temp dir | 31 files, no `.env` copied |
| Compilation | `python -m compileall -q src app deploy tests` | exit 0 |

No commit, tag, or push was made. The `.env` key is local only.

---

## 11. Final audit fixes (12 November 2026)

This pass resolved the six findings from the final audit. No model, feature, or
prediction logic was changed; no deployment, commit, or push was performed.

### 11.1 Fix status

| # | Finding | Fix | Verification |
|---|---|---|---|
| 1 | HF bundle missing chart assets | `prepare_space.py` now copies `data/processed/features.csv` (trend history) and all `reports/figures/*.png` (accuracy panel); the Space README and root README describe the real bundle. No new Python dependency was needed: both charts use Streamlit's built-in elements/Pillow. | Fresh bundle: 39 files, 31.9 MB, 7 figures, no `.env`. AppTest from the bundle: 0 exceptions, 2 `vega_lite_chart`, 3 images, 9 metrics. |
| 2 | Non-numeric features caused HTTP 500 | `_features_from_dict` now coerces every numeric feature to `float`, rejects non-finite values, and validates `city` against `config.CITIES`; invalid cases raise 422. | 3 new API tests (non-numeric, `NaN`, bad feature city). Full suite: **31 passed**. Bundle API: valid 200, bad feature 422. |
| 3 | `Limitations.md` counts stale | Clarified the 86.0% result as the initial 35-feature stage, stated the current model has 47 features, and changed the Lahore count to 18 configured sensors. | Text matches `config.py` (47 features, 18 Lahore IDs) and metadata. |
| 4 | README overstated HF charts | README and deploy README now state exactly what is bundled and that rebuilds clean stale files safely. | Table, note, and bundle description match the 39-file bundle verified above. |
| 5 | `prepare_space.py` left stale files | Added `prepare_target`: cleans a previous bundle in place, preserves Git metadata (`.git`, `.gitattributes`, `.gitignore`), refuses project root/parents/home/system root, and refuses a non-empty non-bundle directory. | Rebuild removed `stale.txt`/`stale2.txt`, kept `.git/SENTINEL`, `.gitattributes`, and `.gitignore`; a non-bundle dir kept `keep.txt` and exited 1. |
| 6 | Live-history caption ambiguous | Caption now states the exact history end (`2026-10-30 23:00 PKT`) and explicitly says the line is historical even in live mode; live inputs are shown in the forecast/data-freshness section. | Bundle AppTest caption text verified. |

### 11.2 Final verification commands

| Check | Command | Result |
|---|---|---|
| Full tests | `python -m pytest tests -q` | **31 passed**, 1 non-blocking Starlette/httpx warning |
| Compilation | `python -m compileall -q src app deploy tests` | exit 0 |
| Fresh HF bundle | `python deploy/huggingface/prepare_space.py <temp>/pm25-space-fresh` | 39 files, 31.9 MB, 7 figures, no `.env` |
| Bundle Streamlit | `AppTest` against the bundle | 0 exceptions, 2 charts, 3 images, dynamic history caption |
| Bundle API | `TestClient` from the bundle | `/health` 200; offline and request-features `/predict` 200; bad feature 422 |
| Safe clean | Rebuild with stale files + VCS metadata | stale removed; `.git`, `.gitattributes`, `.gitignore` preserved |
| Refusal | Build into a non-bundle non-empty dir | exit 1, existing file kept |

Current status: all six audit findings are fixed. The model artefact and metrics are
unchanged (MAE 13.31, RMSE 23.65, R2 0.6695, AQI 71.6%). Nothing was deployed,
committed, or pushed.

---

## 12. Streamlit frontend/backend integration (12 November 2026)

The existing Streamlit app now supports both an in-process backend and the FastAPI
service:

- **Local pipeline** remains the default, so the Hugging Face Space and offline demos
  work unchanged.
- **FastAPI service** mode calls `/health`, `/metadata`, `/predict`, and `/aqi` over
  HTTP. Setting `API_BASE_URL` makes this the default mode; the URL can also be edited
  in the sidebar.
- API failures show a clear message and stop the page, rather than a traceback. The
  input-snapshot table is replaced by an API note, while data-freshness values come
  from the API response.
- Added `TESTING_FRONTEND.md` with terminal commands, a manual checklist, API smoke
  tests, automated-test commands, and troubleshooting.

**Verification:** local-mode AppTest had 0 exceptions, 2 charts, 3 images, and 9
metrics. API-mode AppTest against a live `uvicorn` process (port 8010) also had 0
exceptions in both live-fallback and saved-snapshot runs, with 2 charts and 3 images.
The backend-down path showed the connection error and stopped cleanly. `pytest` still
passes 31 tests, `compileall` exits 0, and the rebuilt HF bundle has 39 files
(31.9 MB). No model logic was changed, and nothing was committed, pushed, or
deployed.
