"""
air_source.py — AirLung Twin
============================
LIVE air quality for Lahore from Open-Meteo Air Quality API (free, NO API key).
Returns recent history + a real 5-7 day forecast. This replaces OpenAQ.

Open-Meteo gives hourly PM2.5, PM10, NO2, O3, SO2, CO and US AQI in µg/m³.
If the network is unavailable, falls back to a clearly-labelled offline sample
so the dashboard still runs.

Docs: https://open-meteo.com/en/docs/air-quality-api
"""
from __future__ import annotations
import datetime as _dt
import numpy as np
import pandas as pd

LAHORE_LAT, LAHORE_LON = 31.5497, 74.3436
_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
         "sulphur_dioxide", "carbon_monoxide", "us_aqi"]
# rename to the short names the rest of the app uses
_RENAME = {"pm2_5": "pm25", "pm10": "pm10", "nitrogen_dioxide": "no2",
           "ozone": "o3", "sulphur_dioxide": "so2",
           "carbon_monoxide": "co", "us_aqi": "us_aqi"}


def _request(past_days: int, forecast_days: int) -> dict:
    import requests
    params = {
        "latitude": LAHORE_LAT, "longitude": LAHORE_LON,
        "hourly": ",".join(_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto",
    }
    r = requests.get(_BASE, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _to_frame(hourly: dict) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for v in _VARS:
        df[_RENAME[v]] = pd.to_numeric(pd.Series(hourly.get(v)), errors="coerce")
    # PM10 occasionally missing -> estimate from PM2.5 so charts stay complete
    df["pm10"] = df["pm10"].fillna(df["pm25"] * 1.4)
    return df.dropna(subset=["pm25"]).reset_index(drop=True)


def _offline_sample(hours_past=168, hours_future=120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realistic Lahore-like fallback when offline (clearly labelled)."""
    rng = np.random.default_rng(11)
    n = hours_past + hours_future
    start = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=hours_past)
    ts = pd.date_range(start, periods=n, freq="h")
    base = 150 + 40 * np.sin(np.arange(n) / 12.0)           # daily cycle
    pm25 = (base + np.cumsum(rng.normal(0, 3, n))).clip(40, 400)
    df = pd.DataFrame({
        "timestamp": ts, "pm25": pm25.round(1), "pm10": (pm25 * 1.4).round(1),
        "no2": rng.uniform(20, 60, n).round(1), "o3": rng.uniform(10, 50, n).round(1),
        "so2": rng.uniform(5, 25, n).round(1), "co": rng.uniform(200, 800, n).round(0),
        "us_aqi": np.clip(pm25 * 1.3, 0, 500).round(0),
    })
    split = df["timestamp"] <= pd.Timestamp.now()
    return df[split].reset_index(drop=True), df[~split].reset_index(drop=True)


def load_air(past_days: int = 7, forecast_days: int = 5):
    """Return (history_df, forecast_df, source_label, is_live).

    history_df / forecast_df columns: timestamp, pm25, pm10, no2, o3, so2, co, us_aqi
    """
    try:
        data = _request(past_days, forecast_days)
        df = _to_frame(data["hourly"])
        now = pd.Timestamp.now(tz=df["timestamp"].dt.tz) if df["timestamp"].dt.tz else pd.Timestamp.now()
        hist = df[df["timestamp"] <= now].reset_index(drop=True)
        fcst = df[df["timestamp"] > now].reset_index(drop=True)
        if len(hist) < 6:                      # too little real data -> fallback
            raise ValueError("insufficient live rows")
        latest = hist["timestamp"].iloc[-1]
        is_live = (pd.Timestamp.now(tz=latest.tz) - latest) <= pd.Timedelta(hours=6)
        label = f"Open-Meteo · Lahore ({'live' if is_live else 'recent'})"
        return hist, fcst, label, is_live
    except Exception as e:
        print("[air_source] live fetch failed, using offline sample:", e)
        hist, fcst = _offline_sample()
        return hist, fcst, "Offline sample (no network)", False


if __name__ == "__main__":
    h, f, src, live = load_air()
    print("source:", src, "| live:", live)
    print("history rows:", len(h), "| forecast rows:", len(f))
    print(h.tail(3).to_string(index=False))