"""
Deploy the Next-Day PM2.5 Forecaster Streamlit app to Hugging Face Spaces.
"""

from pathlib import Path
from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
APP_FILE = ROOT / "app.py"
MODEL_FILE = ROOT / "models" / "final_pipeline.joblib"

SPACE_README = """---
title: Next-Day PM2.5 Forecaster
emoji: 🌫️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# 🌫️ Next-Day Air Quality (PM2.5) Forecaster for Karachi & Lahore

An operational machine learning system predicting **ground-level PM2.5 concentration 24 hours ahead** ($t \\to t+24\\text{h}$) for Karachi and Lahore, Pakistan.

- **Course:** CT-378 Machine Learning, Semester Project
- **Team:** Khubsoorat Shklen (Muhammad Sarim Umer, Muhammad Feras Malik, Muhammad Ahmed Khan)
- **Institution:** Department of Computer Science & Information Technology, NED University

### Features
1. **Interactive Form:** Select city (Karachi / Lahore) and weather parameters with presets for clean days and smog spikes.
2. **24h Quantitative Forecast:** Predicted PM2.5 in $\\mu\\text{g/m}^3$ with 95% confidence intervals derived from empirical test residuals.
3. **EPA AQI Health Advisory:** Actionable recommendations for the general public and sensitive groups.
4. **Explanatory Trajectory:** Visualizes the 24-hour bridging path and key environmental drivers.
"""

SPACE_REQUIREMENTS = """streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
scipy>=1.11.0
matplotlib>=3.8.0
joblib>=1.3.0
huggingface_hub>=0.20.0
"""


def deploy_space():
    vals = dotenv_values(ENV_FILE)
    token = vals.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN missing from .env file!")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/next-day-air-quality-forecaster"

    print(f"Target Hugging Face Space: {repo_id}")
    print("Creating Space on Hugging Face (if it doesn't exist)...")
    space_url = api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
        private=False
    )
    print(f"Space URL: {space_url}")

    # 1. Upload README.md (Space configuration)
    print("Uploading Space configuration (README.md)...")
    api.upload_file(
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add Space configuration and metadata"
    )

    # 2. Upload requirements.txt
    print("Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj=SPACE_REQUIREMENTS.encode("utf-8"),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add dependencies for Streamlit Space"
    )

    # 3. Upload app.py
    print("Uploading app.py...")
    api.upload_file(
        path_or_fileobj=str(APP_FILE),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add Streamlit application"
    )

    # 4. Upload models/final_pipeline.joblib
    if MODEL_FILE.exists():
        print("Uploading models/final_pipeline.joblib...")
        api.upload_file(
            path_or_fileobj=str(MODEL_FILE),
            path_in_repo="models/final_pipeline.joblib",
            repo_id=repo_id,
            repo_type="space",
            commit_message="Add trained model weights"
        )

    print("\n" + "=" * 70)
    print(f"SUCCESS: Space deployed and building at https://huggingface.co/spaces/{repo_id}")
    print("=" * 70)


if __name__ == "__main__":
    deploy_space()
