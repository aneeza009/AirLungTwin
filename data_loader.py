"""
AirLung Twin — data_loader.py  (30% checkpoint)
===============================================
Loads and cleans the FOUR finalized datasets into the shape the dashboard
(`app.py`) consumes, and computes the spirometry baseline + derived stats.

Sources (files live in ./data):
  1. Air quality, LIVE-first fallback chain (AIR_QUALITY_SOURCE = "openaq"):
       a. OpenAQ v3 — freshest Lahore station, last 30 days hourly PM2.5
          (key in data/"AQ api key.txt"; cached 1 h in data/openaq_lahore.csv)
       b. WAQI / aqicn.org — current + recent daily (token: env WAQI_TOKEN or
          data/"waqi token.txt"); AQI index converted back to µg/m³
  2. PLOS ONE — Lahore 2024 hourly PM2.5/PM10 (lahore.xlsx inside the s001 zip)
               static, offline last resort
  3. PhysioNet BIDMC — one recording = the "patient" vitals
               data/bidmc_01_Numerics.csv  ->  spo2, heart_rate, resp
  4. NHANES spirometry — FEV1/FVC baseline anchor
               data/NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv

Target shape consumed by app.py:
  df: timestamp | pm25 | pm10 | spo2 | heart_rate | resp   (aligned on one axis)
  scalars: fev1_fvc_baseline (%), fvc_ml, fev1_ml, obstruction_flag (bool)
  derived: spo2_baseline, spo2_deviation, pm25_spo2_corr,
           aq_latest_time, city_pm25_24h, city_aqi

NOTE on alignment: air quality (hourly) and vitals (per-second) are recorded
separately — no simultaneous patient recording exists at the 30% stage. For the
fusion visualization both series are placed, index-aligned, on one shared
synthetic timeline. This is a display alignment, not a claim of co-measurement.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Which air-quality series the dashboard DISPLAYS:
#   "openaq" -> LIVE chain: OpenAQ v3 (freshest Lahore station, last 30 days hourly)
#               -> WAQI / aqicn.org (current + recent daily, needs a free token)
#               -> PLOS ONE Lahore 2024 (static, offline last resort)
#   "plos"   -> PLOS ONE only (offline; no network calls)
AIR_QUALITY_SOURCE = "openaq"

DATA_DIR = Path(__file__).parent / "data"

PLOS_ZIP = DATA_DIR / "pone.0349858.s001.zip"
PLOS_LAHORE_XLSX = DATA_DIR / "plos_raw" / "lahore.xlsx"          # extracted copy
PLOS_MEMBER = "My_dataset_Air_quality/Dataset/lahore.xlsx"        # path inside zip
BIDMC_CSV = DATA_DIR / "bidmc_01_Numerics.csv"
NHANES_CSV = DATA_DIR / "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
OPENAQ_KEY_FILE = DATA_DIR / "AQ api key.txt"
OPENAQ_CACHE = DATA_DIR / "openaq_lahore.csv"
WAQI_TOKEN_FILE = DATA_DIR / "waqi token.txt"     # or env var WAQI_TOKEN

LAHORE_LATLON = (31.5497, 74.3436)
# Pin a specific OpenAQ location id here (e.g. 4527035 = Civil Secretariat) to always
# use that station. None -> auto-pick the freshest station nearest the city centre.
OPENAQ_LOCATION_ID: int | None = None
LIVE_MAX_AGE_H = 6        # a reading older than this is NOT called "live"
CACHE_MAX_AGE_H = 1       # re-hit the API at most once an hour
HISTORY_DAYS = 30

PM25_HAZARD = 150  # µg/m³ threshold used for "% time hazardous"


# ----------------------------------------------------------------------
# US EPA AQI <-> concentration  (pre-2024 breakpoints: the scale aqicn.org uses
# and the one app.aqi_category() is built on)
# ----------------------------------------------------------------------
_PM25_BP = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
_PM10_BP = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
            (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400),
            (505, 604, 401, 500)]


def pm25_to_aqi(pm25: float) -> int:
    """PM2.5 concentration (µg/m³) -> US AQI (0-500)."""
    c = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi in _PM25_BP:
        if c <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (max(c, c_lo) - c_lo) + i_lo)
    return 500


def _aqi_to_conc(aqi: float, table) -> float:
    a = max(0.0, float(aqi))
    for c_lo, c_hi, i_lo, i_hi in table:
        if a <= i_hi:
            return (c_hi - c_lo) / (i_hi - i_lo) * (max(a, i_lo) - i_lo) + c_lo
    return float(table[-1][1])


def aqi_to_pm25(aqi: float) -> float:
    return round(_aqi_to_conc(aqi, _PM25_BP), 1)


def aqi_to_pm10(aqi: float) -> float:
    return round(_aqi_to_conc(aqi, _PM10_BP), 1)


def _is_fresh(aq: pd.DataFrame | None, max_age_h: float = LIVE_MAX_AGE_H) -> bool:
    if aq is None or aq.empty or aq["pm25"].notna().sum() == 0:
        return False
    last = pd.to_datetime(aq["timestamp"], utc=True).max()
    return (pd.Timestamp.now(tz="UTC") - last) <= pd.Timedelta(hours=max_age_h)


# ----------------------------------------------------------------------
# 1a.  AIR QUALITY — OpenAQ v3 (primary live source)
# ----------------------------------------------------------------------
def _read_openaq_key() -> str | None:
    try:
        key = OPENAQ_KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        return None


def _pick_openaq_stations(results: list[dict]) -> list[tuple[dict, dict]]:
    """Rank Lahore stations: reporting within LIVE_MAX_AGE_H, then has pm10, then nearest.

    Fixes the old bug of taking the FIRST station with a pm25 sensor (the US
    Diplomatic Post, which stopped reporting in 2025 -> 2019-era rows).
    """
    now = pd.Timestamp.now(tz="UTC")
    ranked = []
    for st in results:
        if OPENAQ_LOCATION_ID is not None and st.get("id") != OPENAQ_LOCATION_ID:
            continue
        sensors = {}
        for s in st.get("sensors", []):
            p = (s.get("parameter") or {}).get("name")
            if p in ("pm25", "pm10") and p not in sensors:
                sensors[p] = s["id"]
        last = pd.to_datetime((st.get("datetimeLast") or {}).get("utc"), utc=True, errors="coerce")
        if "pm25" not in sensors or pd.isna(last):
            continue
        if OPENAQ_LOCATION_ID is None and (now - last) > pd.Timedelta(hours=LIVE_MAX_AGE_H):
            continue
        ranked.append(("pm10" not in sensors, st.get("distance") or 9e9, st, sensors))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return [(st, sensors) for _, _, st, sensors in ranked]


def fetch_openaq_lahore(days: int = HISTORY_DAYS, timeout: int = 20) -> pd.DataFrame | None:
    """Live Lahore hourly PM2.5 (+PM10 when the station has it) from OpenAQ v3.

    Returns DataFrame(timestamp, pm25, pm10, station) for the last `days`, or None.
    Any network/key/parse problem is swallowed (the caller falls back).
    """
    key = _read_openaq_key()
    if not key:
        print("[openaq] no API key found -> skipping")
        return None
    try:
        import requests
    except ImportError:
        print("[openaq] 'requests' not installed -> skipping")
        return None

    headers = {"X-API-Key": key}
    base = "https://api.openaq.org/v3"
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        loc = requests.get(
            f"{base}/locations",
            params={"coordinates": f"{LAHORE_LATLON[0]},{LAHORE_LATLON[1]}",
                    "radius": 25000, "limit": 1000},
            headers=headers, timeout=timeout,
        )
        loc.raise_for_status()
        candidates = _pick_openaq_stations(loc.json().get("results", []))
        if not candidates:
            print("[openaq] no Lahore station reporting pm25 recently")
            return None

        for station, sensors in candidates[:3]:          # try a few in case one is patchy
            frames = {}
            for param, sid in sensors.items():
                r = requests.get(f"{base}/sensors/{sid}/hours",
                                 params={"datetime_from": since, "limit": 1000},
                                 headers=headers, timeout=timeout)
                r.raise_for_status()
                recs = [{"timestamp": ((m.get("period") or {}).get("datetimeFrom") or {}).get("utc"),
                         param: m.get("value")} for m in r.json().get("results", [])]
                if recs:
                    frames[param] = pd.DataFrame(recs)
            if "pm25" not in frames or len(frames["pm25"]) < 24:
                continue
            aq = frames["pm25"]
            if "pm10" in frames:
                aq = aq.merge(frames["pm10"], on="timestamp", how="left")
            else:
                aq["pm10"] = np.nan
            aq["timestamp"] = pd.to_datetime(aq["timestamp"], utc=True, errors="coerce")
            aq["pm25"] = pd.to_numeric(aq["pm25"], errors="coerce")
            aq = (aq.dropna(subset=["timestamp", "pm25"])
                    .query("pm25 >= 0 and pm25 < 1000")          # drop sensor glitches
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp"))
            aq["station"] = station.get("name", "?").strip()
            aq = aq[["timestamp", "pm25", "pm10", "station"]].reset_index(drop=True)
            aq.to_csv(OPENAQ_CACHE, index=False)
            print(f"[openaq] {aq['station'].iloc[0]}: {len(aq)} hourly rows -> {OPENAQ_CACHE.name}")
            return aq
        print("[openaq] candidate stations returned too little data")
        return None
    except Exception as exc:  # network, auth, schema drift — all non-fatal
        print(f"[openaq] live fetch failed ({exc.__class__.__name__}: {exc})")
        return None


def refresh_openaq_cache(max_age_hours: float = CACHE_MAX_AGE_H) -> pd.DataFrame | None:
    """Use data/openaq_lahore.csv if it was fetched recently AND its data is current,
    else re-fetch. (The old check used file mtime only, so a freshly-copied file
    full of 2019 rows counted as "fresh".)"""
    try:
        if OPENAQ_CACHE.exists():
            age_h = (time.time() - OPENAQ_CACHE.stat().st_mtime) / 3600
            cached = pd.read_csv(OPENAQ_CACHE)
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, errors="coerce")
            if age_h < max_age_hours and "station" in cached and _is_fresh(cached):
                print(f"[openaq] cache fresh ({age_h:.1f}h old) -> {OPENAQ_CACHE.name}")
                return cached
        return fetch_openaq_lahore()
    except Exception as exc:
        print(f"[openaq] cache refresh skipped ({exc.__class__.__name__}: {exc})")
        return None


# ----------------------------------------------------------------------
# 1b.  AIR QUALITY — WAQI / aqicn.org (live fallback)
# ----------------------------------------------------------------------
def _read_waqi_token() -> str | None:
    tok = os.environ.get("WAQI_TOKEN", "").strip()
    if tok:
        return tok
    try:
        return WAQI_TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def fetch_waqi_lahore(timeout: int = 15) -> pd.DataFrame | None:
    """Lahore from the WAQI feed (aqicn.org). Free token: https://aqicn.org/data-platform/token/

    WAQI reports US AQI *index* values, not concentrations, so they are converted
    back to µg/m³. It only gives the current reading + a few daily averages, so
    the series is short (daily points + 'now').
    """
    token = _read_waqi_token()
    if not token:
        print("[waqi] no token (env WAQI_TOKEN or data/waqi token.txt) -> skipping")
        return None
    try:
        import requests
        lat, lon = LAHORE_LATLON
        r = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/",
                         params={"token": token}, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "ok":
            print(f"[waqi] API said: {j.get('data')}")
            return None
        d = j["data"]
        station = (d.get("city") or {}).get("name", "Lahore")
        now_ts = pd.to_datetime((d.get("time") or {}).get("iso"), utc=True, errors="coerce")
        iaqi = d.get("iaqi") or {}
        today = (now_ts if pd.notna(now_ts) else pd.Timestamp.now(tz="UTC")).normalize()

        daily = (d.get("forecast") or {}).get("daily") or {}
        pm10_by_day = {x["day"]: x.get("avg") for x in daily.get("pm10", [])}
        recs = []
        for x in daily.get("pm25", []):                       # past days only, no forecast
            day = pd.Timestamp(x["day"], tz="UTC")
            if day < today and x.get("avg") is not None:
                p10 = pm10_by_day.get(x["day"])
                recs.append({"timestamp": day + pd.Timedelta(hours=12),
                             "pm25": aqi_to_pm25(x["avg"]),
                             "pm10": aqi_to_pm10(p10) if p10 is not None else np.nan})
        if "pm25" in iaqi and pd.notna(now_ts):
            p10 = (iaqi.get("pm10") or {}).get("v")
            recs.append({"timestamp": now_ts, "pm25": aqi_to_pm25(iaqi["pm25"]["v"]),
                         "pm10": aqi_to_pm10(p10) if p10 is not None else np.nan})
        if not recs:
            print("[waqi] feed had no pm25")
            return None
        aq = pd.DataFrame(recs).sort_values("timestamp").reset_index(drop=True)
        aq["station"] = station
        print(f"[waqi] {station}: {len(aq)} points")
        return aq
    except Exception as exc:
        print(f"[waqi] fetch failed ({exc.__class__.__name__}: {exc})")
        return None


# ----------------------------------------------------------------------
# 1c.  AIR QUALITY — PLOS ONE Lahore 2024 (static, last resort)
# ----------------------------------------------------------------------
def _ensure_plos_lahore() -> Path:
    """Extract lahore.xlsx from the PLOS s001 zip if not already extracted."""
    if PLOS_LAHORE_XLSX.exists():
        return PLOS_LAHORE_XLSX
    if not PLOS_ZIP.exists():
        raise FileNotFoundError(f"Missing {PLOS_ZIP}")
    PLOS_LAHORE_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PLOS_ZIP) as z:
        with z.open(PLOS_MEMBER) as src:
            PLOS_LAHORE_XLSX.write_bytes(src.read())
    return PLOS_LAHORE_XLSX


def load_plos_lahore() -> pd.DataFrame:
    """Real Lahore hourly PM2.5/PM10 from the PLOS ONE dataset -> timestamp, pm25, pm10."""
    path = _ensure_plos_lahore()
    raw = pd.read_excel(path, engine="openpyxl")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    aq = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce"),
        "pm25": pd.to_numeric(raw["pm25"], errors="coerce"),
        "pm10": pd.to_numeric(raw["pm10"], errors="coerce"),
    })
    aq = (aq.dropna(subset=["timestamp", "pm25"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    return aq


def load_air_quality() -> tuple[pd.DataFrame, str]:
    """Return (air-quality df, source label). Fallback: OpenAQ -> WAQI -> PLOS.

    A live source is only accepted if its latest reading is within LIVE_MAX_AGE_H,
    so the LIVE badge in app.py is never shown over stale data.
    """
    if AIR_QUALITY_SOURCE == "openaq":
        live = refresh_openaq_cache()
        if _is_fresh(live) and live["pm25"].notna().sum() >= 24:
            return live, f"OpenAQ · {live['station'].iloc[0]}, Lahore (live)"
        print("[air] OpenAQ unusable -> trying WAQI")
        waqi = fetch_waqi_lahore()
        if _is_fresh(waqi):
            return waqi, f"WAQI/aqicn · {waqi['station'].iloc[0]} (live)"
        print("[air] WAQI unusable -> falling back to PLOS Lahore (static)")
    return load_plos_lahore(), "PLOS ONE Lahore (2024, static)"


# ----------------------------------------------------------------------
# 3.  VITALS  (BIDMC — one recording = the patient)
# ----------------------------------------------------------------------
def load_bidmc() -> pd.DataFrame:
    """BIDMC numerics -> timestamp, spo2, heart_rate, resp (1 Hz)."""
    raw = pd.read_csv(BIDMC_CSV)
    raw.columns = [str(c).strip() for c in raw.columns]  # strip ' HR' etc.
    time_col = next((c for c in raw.columns if c.lower().startswith("time")), raw.columns[0])
    seconds = pd.to_numeric(raw[time_col], errors="coerce").fillna(0)
    vit = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-10-24 08:00") + pd.to_timedelta(seconds, unit="s"),
        "spo2": pd.to_numeric(raw.get("SpO2"), errors="coerce"),
        "heart_rate": pd.to_numeric(raw.get("HR"), errors="coerce"),
        "resp": pd.to_numeric(raw.get("RESP"), errors="coerce"),
    })
    vit = vit.dropna(subset=["spo2", "heart_rate", "resp"]).reset_index(drop=True)
    return vit


# ----------------------------------------------------------------------
# 4.  SPIROMETRY  (NHANES — FEV1/FVC baseline anchor)
# ----------------------------------------------------------------------
def load_nhanes_baseline() -> dict:
    """Pick one adult subject as the spirometry anchor; return baseline scalars.

    The cleaned NHANES set stores FEV1/FVC in LITRES (Baseline_FEV1_L, Baseline_FVC_L)
    plus a precomputed ratio. We select the first acceptable adult (Age >= 18) so the
    anchor is deterministic across runs.
    """
    nh = pd.read_csv(NHANES_CSV)
    nh.columns = [str(c).strip() for c in nh.columns]

    fev1 = pd.to_numeric(nh["Baseline_FEV1_L"], errors="coerce")
    fvc = pd.to_numeric(nh["Baseline_FVC_L"], errors="coerce")
    age = pd.to_numeric(nh["Age"], errors="coerce")

    mask = (age >= 18) & fev1.notna() & fvc.notna() & (fvc > 0)
    idx = nh.index[mask]
    if len(idx) == 0:                       # fall back to any valid row
        idx = nh.index[fev1.notna() & fvc.notna() & (fvc > 0)]
    i = idx[0]

    fev1_ml = round(float(fev1[i]) * 1000, 1)
    fvc_ml = round(float(fvc[i]) * 1000, 1)
    ratio = round(fev1_ml / fvc_ml * 100, 2)
    return {
        "fev1_ml": fev1_ml,
        "fvc_ml": fvc_ml,
        "fev1_fvc_baseline": ratio,
        "obstruction_flag": bool(ratio < 70.0),   # FEV1/FVC < 0.70 rule
        "seqn": int(pd.to_numeric(nh.loc[i, "SEQN"], errors="coerce")),
        "age": float(age[i]),
    }


# ----------------------------------------------------------------------
# ASSEMBLE  — align air quality + vitals on one axis, compute derived stats
# ----------------------------------------------------------------------
def build_dataset() -> dict:
    aq, aq_source = load_air_quality()
    vit = load_bidmc()
    spx = load_nhanes_baseline()

    n = len(vit)  # vitals drive the fine-grained axis (per-second recording)
    # Evenly sample the full air-quality history down to n points so the whole
    # PM trend is represented alongside the vitals window.
    pos = np.linspace(0, len(aq) - 1, n).round().astype(int)
    aq_s = aq.iloc[pos].reset_index(drop=True)

    df = pd.DataFrame({
        "timestamp": vit["timestamp"],              # shared synthetic timeline
        "pm25": aq_s["pm25"].round(2).to_numpy(),
        "pm10": aq_s["pm10"].round(2).to_numpy(),
        "spo2": vit["spo2"].round(2).to_numpy(),
        "heart_rate": vit["heart_rate"].round(1).to_numpy(),
        "resp": vit["resp"].round(1).to_numpy(),
    })
    df["pm10"] = df["pm10"].fillna(df["pm25"] * 1.4)  # PLOS pm10 is complete; guard anyway

    # Derived (no ML): SpO2 deviation from its own baseline + PM2.5<->SpO2 corr.
    spo2_baseline = round(float(df["spo2"].median()), 2)
    spo2_deviation = round(float(df["spo2"].iloc[-1] - spo2_baseline), 2)
    corr = df["pm25"].corr(df["spo2"])
    pm25_spo2_corr = round(float(corr), 3) if pd.notna(corr) else float("nan")

    # "Generic city AQI" — what a public AQI site (Punjab EPA, aqicn) reports:
    # the 24-h mean PM2.5 of the raw city series converted to US AQI.
    aq_ts = pd.to_datetime(aq["timestamp"], utc=True)
    last_24h = aq.loc[aq_ts >= aq_ts.max() - pd.Timedelta(hours=24), "pm25"]
    city_pm25_24h = round(float(last_24h.mean()), 1)

    return {
        "df": df,
        "aq_source": aq_source,
        "aq_date_range": (aq["timestamp"].min(), aq["timestamp"].max()),
        "aq_latest_time": aq_ts.max(),
        "city_pm25_24h": city_pm25_24h,
        "city_aqi": pm25_to_aqi(city_pm25_24h),
        "spo2_baseline": spo2_baseline,
        "spo2_deviation": spo2_deviation,
        "pm25_spo2_corr": pm25_spo2_corr,
        **spx,
    }


if __name__ == "__main__":
    d = build_dataset()
    df = d["df"]
    print("air-quality source :", d["aq_source"])
    print("aq date range      :", d["aq_date_range"])
    print("city PM2.5 24h/AQI :", d["city_pm25_24h"], "/", d["city_aqi"])
    print("rows               :", len(df))
    print(df.head())
    print("mean pm25          :", round(df["pm25"].mean(), 1))
    print("mean spo2          :", round(df["spo2"].mean(), 1))
    print("FEV1/FVC baseline  :", d["fev1_fvc_baseline"], "%  (SEQN", d["seqn"], "age", d["age"], ")")
    print("FEV1/FVC obstruct  :", d["obstruction_flag"])
    print("spo2 deviation     :", d["spo2_deviation"])
    print("pm25<->spo2 corr   :", d["pm25_spo2_corr"])
