"""Streamlit demo: next-day PM2.5 forecast + AQI health advice (Task 16).

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from src import config
from src.aqi import get_health_advice
from src.live_data import get_live_features
from src.predict import load_model, forecast, model_available

MODEL_METRICS = config.ROOT / "models" / "model_metadata.json"
ENV_API_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")

st.set_page_config(
    page_title="Next-Day PM2.5 Forecast | Karachi & Lahore",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_model()


@st.cache_data(ttl=1800, show_spinner="Fetching the latest air-quality and weather inputs...")
def _cached_live_inputs(city: str, offline: bool):
    return get_live_features(city, offline=offline, prefer_offline=offline)


def _metadata() -> dict:
    if MODEL_METRICS.exists():
        return json.loads(MODEL_METRICS.read_text(encoding="utf-8"))
    return {}


def _api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return f"HTTP {response.status_code}: {detail}"


@st.cache_data(ttl=60, show_spinner=False)
def _api_health(base_url: str) -> dict:
    response = requests.get(f"{base_url}/health", timeout=10)
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def _api_metadata(base_url: str) -> dict:
    response = requests.get(f"{base_url}/metadata", timeout=10)
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return response.json()


@st.cache_data(ttl=1800, show_spinner="Calling the FastAPI service...")
def _api_predict(base_url: str, city: str, offline: bool, postprocess: bool) -> dict:
    response = requests.post(
        f"{base_url}/predict",
        json={
            "city": city,
            "offline": offline,
            "prefer_offline": offline,
            "postprocess": postprocess,
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return response.json()


def _api_aqi(base_url: str, pm25: float) -> dict:
    response = requests.post(f"{base_url}/aqi", json={"pm25": pm25}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(_api_error(response))
    return response.json()


@st.cache_data(show_spinner=False)
def _recent_history(city: str, hours: int = 168) -> pd.DataFrame:
    """Last ``hours`` of observed PM2.5 plus a 24-hour rolling mean."""
    path = config.FEATURES_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        usecols=["datetime_pkt", "city", "pm25_now"],
        parse_dates=["datetime_pkt"],
    )
    part = df[df["city"].str.lower() == city].sort_values("datetime_pkt").tail(hours)
    if part.empty:
        return pd.DataFrame()
    out = part.set_index("datetime_pkt")[["pm25_now"]].rename(
        columns={"pm25_now": "Observed PM2.5"}
    )
    out["24h rolling mean"] = out["Observed PM2.5"].rolling(24, min_periods=1).mean()
    return out


st.title("Next-Day Air Quality Forecast")
st.caption("PM2.5 24 hours ahead for Karachi and Lahore, with EPA AQI category and health advice.")

with st.sidebar:
    st.header("Forecast settings")
    city = st.selectbox("City", options=["lahore", "karachi"], format_func=str.title)
    live_mode = st.radio(
        "Input source",
        options=["Live APIs (fallback to snapshot)", "Saved snapshot only"],
        index=0,
    )
    backend = st.radio(
        "Backend",
        options=["Local pipeline", "FastAPI service"],
        index=1 if ENV_API_URL else 0,
        help=(
            "Local pipeline runs inference inside Streamlit. FastAPI service sends "
            "the request to the REST API."
        ),
    )
    use_api = backend == "FastAPI service"
    api_base = ENV_API_URL or "http://127.0.0.1:8000"
    if use_api:
        api_base = st.text_input("API base URL", value=api_base).strip().rstrip("/")
    postprocess = st.checkbox(
        "Apply experimental Lahore smog-recovery correction",
        value=False,
        help=(
            "Blends toward short-horizon momentum when pollution is clearing quickly. "
            "It did not improve test metrics, so it is off by default."
        ),
    )
    refresh = st.button("Refresh inputs", width="stretch")
    if refresh:
        st.cache_data.clear()
    st.divider()
    st.subheader("Model")
    st.caption("Voting ensemble: Ridge (20%) + Random Forest (30%) + LightGBM (50%)")
    if use_api:
        if not api_base:
            st.error("Enter the FastAPI base URL, for example http://127.0.0.1:8000.")
            st.stop()
        try:
            health = _api_health(api_base)
            if health.get("status") != "ok":
                st.error(
                    f"FastAPI backend reports `{health.get('status')}`. Check "
                    "`models/final_pipeline.joblib` on the backend."
                )
                st.stop()
            meta = _api_metadata(api_base)
        except Exception as exc:  # noqa: BLE001 - show a useful connection error
            st.error(f"FastAPI backend error at `{api_base}`: {exc}")
            st.info(
                "Start it with `uvicorn app.api:app --reload --port 8000`, then "
                "reload this page or press Refresh inputs."
            )
            st.stop()
    else:
        meta = _metadata()
    raw = meta.get("test_metrics_raw", {})
    if raw:
        st.metric("Held-out test MAE", f"{raw.get('mae', 0):.2f} ug/m3")
        st.metric("Held-out AQI accuracy", f"{raw.get('aqi_accuracy', 0):.1%}")
        st.caption(
            f"Trained on {meta.get('fit_rows', 0):,} rows | {meta.get('n_features', '?')} features"
        )
    st.divider()
    st.caption(
        "Forecasts are research/awareness tools, not medical or regulatory advice. "
        "Community sensors are not reference-grade monitors."
    )

offline = live_mode.startswith("Saved")

if use_api:
    try:
        result = _api_predict(api_base, city, offline, postprocess)
        live_inputs = None
    except Exception as exc:  # noqa: BLE001 - show a useful backend error
        st.error(f"FastAPI request failed: {exc}")
        st.info("Check that the backend is running and the API URL is correct.")
        st.stop()
else:
    if not model_available():
        st.error(
            "Model artifact is missing. Run `python -m src.train_final` in the project "
            "root, then reload this page."
        )
        st.stop()
    try:
        live_inputs = _cached_live_inputs(city, offline)
        result = forecast(city, model=_load_model(), live=live_inputs, postprocess=postprocess)
    except Exception as exc:  # noqa: BLE001 - show a useful error instead of a traceback
        st.error(f"Could not produce a forecast: {exc}")
        st.info(
            "Check `.env` for OPENAQ_API_KEY, or switch to 'Saved snapshot only'. "
            "The snapshot is generated with `python -m src.live_data --refresh-fallback`."
        )
        st.stop()

left, right = st.columns([1.15, 1])

with left:
    color = result["hex_color"]
    st.markdown(
        f"""
        <div style="background:{color};padding:1.2rem 1.4rem;border-radius:12px;
                    color:#111;border:1px solid rgba(0,0,0,.15)">
          <div style="font-size:.9rem;text-transform:uppercase;letter-spacing:.08em">
            Tomorrow's forecast for {result['city']}
          </div>
          <div style="font-size:3.2rem;font-weight:700;line-height:1.1">
            {result['predicted_pm25']:.1f}
            <span style="font-size:1.1rem">ug/m3 PM2.5</span>
          </div>
          <div style="font-size:1.35rem;font-weight:600">
            AQI {result['aqi']} - {result['category']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed now", f"{result['observed_pm25']:.1f} ug/m3")
    c2.metric("Input hour", str(result["feature_timestamp"])[5:16])
    c3.metric("Data source", result["source"].title())

    st.subheader("What this means")
    st.write(result["summary"])
    st.markdown(f"**General public:** {result['general_public']}")
    st.markdown(f"**Sensitive groups:** {result['sensitive_groups']}")
    st.markdown("**Recommended actions**")
    for action in result["actions"]:
        st.markdown(f"- {action}")

with right:
    st.subheader("Input snapshot")
    if live_inputs is not None:
        feature_row = live_inputs["features"].iloc[0]
        display_cols = [
            "pm25_now",
            "pm25_lag_24h",
            "pm25_roll_mean_24h",
            "pm25_roll_mean_72h",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cams_pm2_5",
            "stagnation_index",
        ]
        table = pd.DataFrame(
            {
                "Feature": display_cols,
                "Value": [round(float(feature_row[c]), 2) for c in display_cols],
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        feature_timestamp = live_inputs["feature_timestamp"]
        observed_at = live_inputs["observed_at"]
        stale_hours = live_inputs["stale_hours"]
        notes = live_inputs["notes"]
    else:
        st.caption(
            "The FastAPI service builds the feature row internally. Freshness "
            "details returned by the backend are shown below."
        )
        feature_timestamp = result["feature_timestamp"]
        observed_at = result["observed_at"]
        stale_hours = result["stale_hours"]
        notes = result["notes"]

    st.subheader("Data freshness")
    st.write(f"Feature hour: `{feature_timestamp}` PKT")
    st.write(f"Latest sensor hour: `{observed_at}` PKT")
    st.write(f"Staleness: `{stale_hours:.1f}` hours")
    if notes:
        st.warning(notes)
    if stale_hours > 30:
        st.warning(
            "Inputs are more than 30 hours old. Treat this forecast as a demo, not a live alert."
        )

    with st.expander("Manual AQI / advice simulator"):
        manual = st.slider("PM2.5 (ug/m3)", 0.0, 400.0, float(result["predicted_pm25"]), 0.5)
        manual_info = None
        if use_api:
            try:
                manual_info = _api_aqi(api_base, manual)
            except Exception as exc:  # noqa: BLE001 - keep the page usable
                st.error(f"Backend AQI request failed: {exc}")
        else:
            manual_info = get_health_advice(manual)
        if manual_info is not None:
            st.markdown(
                f"**AQI {manual_info['aqi']} - {manual_info['category']}**: "
                f"{manual_info['summary']}"
            )
            st.write(manual_info["general_public"])

st.divider()
st.subheader("Recent trend and tomorrow's forecast")
chart_col, compare_col = st.columns([2, 1])

with chart_col:
    history = _recent_history(city)
    if history.empty:
        st.info("Historical PM2.5 series is not available in the saved dataset.")
    else:
        st.line_chart(history, height=300)
        history_end = pd.Timestamp(history.index.max())
        st.caption(
            "Observed city-average PM2.5 from the saved training history, ending "
            f"{history_end:%Y-%m-%d %H:%M} PKT, with a 24-hour rolling mean. This "
            "context line is historical even in live mode; the forecast card above "
            "uses the freshest inputs shown in Data freshness."
        )

with compare_col:
    comparison = pd.DataFrame(
        {"PM2.5 (ug/m3)": [result["observed_pm25"], result["predicted_pm25"]]},
        index=["Now", "Tomorrow (+24h)"],
    )
    st.bar_chart(comparison, height=300, color="#4C78A8")
    st.caption("Latest observed value versus the model's 24-hour-ahead forecast.")

with st.expander("How accurate is the model? (held-out test set)"):
    raw = meta.get("test_metrics_raw", {})
    if raw:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{raw.get('mae', 0):.2f} ug/m3")
        m2.metric("RMSE", f"{raw.get('rmse', 0):.2f} ug/m3")
        m3.metric("R2", f"{raw.get('r2', 0):.4f}")
        m4.metric("AQI accuracy", f"{raw.get('aqi_accuracy', 0):.1%}")
    for name, caption in [
        ("error_scatter.png", "Actual vs predicted PM2.5 on the held-out test set."),
        ("error_by_month.png", "Monthly MAE; October is the hardest month."),
        ("error_confusion.png", "AQI category confusion; most errors are one band away."),
    ]:
        fig = config.FIG_DIR / name
        if fig.exists():
            st.image(str(fig), caption=caption, width="stretch")

st.divider()
st.caption(
    "Data: OpenAQ community sensors (PM2.5) and Open-Meteo/CAMS (weather and regional aerosol). "
    "AQI categories follow the 2024 US EPA PM2.5 breakpoints. See README and Limitations.md."
)
