---
title: Next-Day PM2.5 Forecast
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.66.1"
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Next-Day PM2.5 Forecast for Karachi and Lahore

This Space serves a 24-hour-ahead PM2.5 forecast for Karachi and Lahore,
Pakistan. It combines recent OpenAQ PM2.5 readings with Open-Meteo weather and
CAMS regional aerosol inputs, then translates the predicted concentration into
a US EPA AQI category and practical health advice.

## Notes

- Forecasts are for research and public awareness, not medical or regulatory
  decisions.
- OpenAQ stations are community/low-cost monitors, not reference-grade
  instruments.
- Set `OPENAQ_API_KEY` in **Settings -> Variables and secrets** to enable live
  OpenAQ collection. Without it, the app automatically uses the bundled
  `data/processed/live_fallback.csv` snapshot.
- Model artifact: `models/final_pipeline.joblib` (VotingRegressor: Ridge 20%,
  Random Forest 30%, LightGBM 50%).
- Held-out test metrics: MAE 13.31 ug/m3, RMSE 23.65, R2 0.6695, AQI category
  accuracy 71.6%.
- Chart assets are bundled: `data/processed/features.csv` supplies the saved
  history for the trend chart, and `reports/figures/*.png` supplies the accuracy
  panel. No extra charting package is required beyond the Space requirements.

## Deployment steps (for the account owner)

1. Create a new **Streamlit** Space at <https://huggingface.co/new-space>.
2. From the project root, build the bundle:

   ```powershell
   python deploy/huggingface/prepare_space.py ../pm25-space
   ```

   Re-running this command cleans stale files from a previous bundle in place
   while preserving its Git metadata (including `.git`). A non-empty folder that
   is not a previous Space bundle is refused rather than deleted.

3. Push the bundle to the Space:

   ```powershell
   cd ../pm25-space
   git init
   git add .
   git commit -m "Deploy PM2.5 forecaster"
   git remote add origin https://huggingface.co/spaces/<user>/<space>
   git push
   ```

   Hugging Face may ask for a username and an **access token** with write
   permission. Do not commit the token.

4. In the Space, open **Settings -> Variables and secrets** and add
   `OPENAQ_API_KEY` as a **secret**. The key in the local `.env` is deliberately
   not copied into the bundle. Without the secret, the Space uses the bundled
   offline snapshot and labels it as offline.

5. Wait for the Space build to finish, then open the app URL. The first request
   can take a few seconds while the 18 MB model loads.

The bundler copies only what the Space needs: `app/`, `src/`, the model and
metadata, `data/processed/live_fallback.csv`, `data/processed/features.csv` (the
trend-chart history), `reports/figures/*.png` (the accuracy panel), this README,
requirements, and the licence, data-source, and limitation documents. The bundle
is about 32 MB. It does not copy `.env`, raw data, notebooks, or the full reports
directory.

## Attribution

- OpenAQ and the original sensor providers (see `DATA_SOURCES.md`)
- Open-Meteo, licensed under CC BY 4.0
- AQI breakpoints: US EPA 2024 PM2.5 table
