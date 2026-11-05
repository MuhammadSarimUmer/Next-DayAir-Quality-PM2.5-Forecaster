import time
import requests
from src import config


def get(path: str, params: dict | None = None, retries: int = 5) -> dict:
    """GET from OpenAQ v3 with auth header, polite rate limiting and retry on 429/5xx."""
    if not config.OPENAQ_API_KEY:
        raise RuntimeError("OPENAQ_API_KEY missing. Copy .env.example to .env and set it.")

    url = f"{config.OPENAQ_BASE}{path}"
    headers = {"X-API-Key": config.OPENAQ_API_KEY}

    for attempt in range(retries):
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 200:
            time.sleep(1.1)  # stay under the per-minute rate limit
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 5 * (attempt + 1)
            print(f"  [{resp.status_code}] retrying in {wait}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()

    raise RuntimeError(f"Failed after {retries} retries: {url}")