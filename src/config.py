
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

for d in (RAW_DIR, PROCESSED_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = PROCESSED_DIR / "clean_hourly.csv"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
GAP_REPORT_FILE = REPORTS_DIR / "gap_report.csv"
SPLIT_INFO_FILE = PROCESSED_DIR / "split_info.json"

# ---------- APIs ----------
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo CAMS reanalysis (independent of ground sensors; regional smoke/aerosol signal)
CAMS_API_VARS = ["pm2_5", "dust", "aerosol_optical_depth"]
CAMS_COLS = ["cams_pm2_5", "cams_dust", "cams_aod"]

# ---------- collection ----------
# NOTE: OpenAQ community sensors for Karachi/Lahore only have ~1 year 4 months of data.
# Attempted to extend to 2022 but sensors were installed ~2024-2025.
# Documented in Limitations.md.
START_DATE = "2025-06-23"
END_DATE = "2026-10-31"

# Kaggle 2021-24 hourly API history (datasets/2021-24); OpenAQ wins on overlap.
USE_KAGGLE_HISTORY = False
KAGGLE_HISTORY_END = "2024-12-12"
WEATHER_START_DATE = "2021-08-24" if USE_KAGGLE_HISTORY else START_DATE
CAMS_START_DATE = WEATHER_START_DATE

CITIES = {
    "karachi": {"lat": 24.8607, "lon": 67.0011},
    "lahore": {"lat": 31.5204, "lon": 74.3587},
}

# OpenAQ PM2.5 SENSOR IDs (not location IDs)
# Selected from your stations_candidates.csv output.
SELECTED_SENSORS = {
    "karachi": [
        13368464,  # Urban Resource Center
        13369654,  # NED University City Campus
        13370571,  # Agha Khan University IED
        13377694,  # Karachi Metropolitan Corporation
        13381251,  # Aman Khan Kaka Khail
        13406434,  # NED Main Campus
        13407059,  # Urban Unit Karachi
        13408226,  # Asif Gulistan-e-Johar
        13409122,  # Engr Muzaffar
        13414781,  # Orangi Pilot Project
        13490916,  # Climate Action Center
        13533643,  # G3 Engineering Consultants
    ],

    "lahore": [
        13137726,  # Barki
        13153587,  # Civil Secretariat
        13154528,  # Learning Alliance International DHA
        13155469,  # Foot & Mouth Disease Research Centre
        13156083,  # HBFC Society DHA Phase 5
        13182280,  # Sandha Road
        13183459,  # Ravi Road
        13184347,  # Shalimar Town
        13190998,  # Punjab University
        13192746,  # Model Town
        13223107,  # College Road
        13230942,  # Kahna Hospital
        13238783,  # Gulberg III
        13300980,  # Cantonment
        13302769,  # Samanabad
        13341452,  # Forman Christian College
        13817570,  # Thokar Niaz Baig
        14599816,  # ARC Lahore
    ],
}

# ---------- weather ----------
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "wind_direction_10m",
]

# ---------- cleaning ----------
PKT_TZ = "Asia/Karachi"

PM25_MIN = 0.0
PM25_MAX = 2000.0  # allow extreme smog hours in Kaggle API series

STUCK_HOURS = 8
MIN_COVERAGE_PCT = 50
MAX_INTERP_HOURS = 3

# ---------- features / split ----------
HORIZON_HOURS = 24

LAG_HOURS = [1, 3, 6, 12, 24]
ROLL_HOURS = [6, 24, 72]

WINTER_MONTHS = [11, 12, 1, 2]

INCLUDE_PM25_NOW = True
EXTRA_FEATURES = True

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Fewer CV folds when train set is large (Kaggle history ~50k rows).
CV_SPLITS = 3 if USE_KAGGLE_HISTORY else 5
