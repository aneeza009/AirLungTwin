"""
AirLung Twin — data_loader.py  (30% checkpoint)
===============================================
Loads and cleans the FOUR finalized datasets into the shape the dashboard
(`app.py`) consumes, and computes the spirometry baseline + derived stats.

Sources (files live in ./data):
  1. OpenAQ  — Lahore PM2.5/PM10, live via API  (key in data/"AQ api key.txt")
               best-effort; result cached to data/openaq_lahore.csv
  2. PLOS ONE — Lahore historical PM2.5/PM10 (lahore.xlsx inside the s001 zip)
               the RELIABLE, offline air-quality source the dashboard uses
  3. PhysioNet BIDMC — one recording = the "patient" vitals
               data/bidmc_01_Numerics.csv  ->  spo2, heart_rate, resp
  4. NHANES spirometry — FEV1/FVC baseline anchor
               data/NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv

Target shape consumed by app.py:
  df: timestamp | pm25 | pm10 | spo2 | heart_rate | resp   (aligned on one axis)
  scalars: fev1_fvc_baseline (%), fvc_ml, fev1_ml, obstruction_flag (bool)
  derived: spo2_baseline, spo2_deviation, pm25_spo2_corr

NOTE on alignment: air quality (hourly) and vitals (per-second) are recorded
separately — no simultaneous patient recording exists at the 30% stage. For the
fusion visualization both series are placed, index-aligned, on one shared
synthetic timeline. This is a display alignment, not a claim of co-measurement.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Which air-quality series the dashboard DISPLAYS:
#   "plos"   -> PLOS ONE Lahore 2024: complete, aligned real pm25 + pm10  (recommended)
#   "openaq" -> OpenAQ live station: real pm25 (pm10 often absent -> estimated), stale ts
# Either way the OpenAQ API is still queried (best-effort) and cached to
# data/openaq_lahore.csv, so the live pipeline is demonstrated regardless.
AIR_QUALITY_SOURCE = "plos"

DATA_DIR = Path(__file__).parent / "data"

PLOS_ZIP = DATA_DIR / "pone.0349858.s001.zip"
PLOS_LAHORE_XLSX = DATA_DIR / "plos_raw" / "lahore.xlsx"          # extracted copy
PLOS_MEMBER = "My_dataset_Air_quality/Dataset/lahore.xlsx"        # path inside zip
BIDMC_CSV = DATA_DIR / "bidmc_01_Numerics.csv"
NHANES_CSV = DATA_DIR / "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"
OPENAQ_KEY_FILE = DATA_DIR / "AQ api key.txt"
OPENAQ_CACHE = DATA_DIR / "openaq_lahore.csv"

PM25_HAZARD = 150  # µg/m³ threshold used for "% time hazardous"


# ----------------------------------------------------------------------
# 1 + 2.  AIR QUALITY  (OpenAQ live, else PLOS Lahore)
# ----------------------------------------------------------------------
def _read_openaq_key() -> str | None:
    try:
        key = OPENAQ_KEY_FILE.read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        return None


def fetch_openaq_lahore(hours: int = 720, timeout: int = 15) -> pd.DataFrame | None:
    """Best-effort pull of Lahore PM2.5/PM10 from the OpenAQ v3 API.

    Returns a cleaned DataFrame (timestamp, pm25, pm10) on success, else None.
    Any network/key/parse problem is swallowed so the dashboard still runs
    offline on the PLOS Lahore file.
    """
    key = _read_openaq_key()
    if not key:
        print("[openaq] no API key found -> skipping live fetch")
        return None
    try:
        import requests
    except ImportError:
        print("[openaq] 'requests' not installed -> skipping live fetch")
        return None

    headers = {"X-API-Key": key}
    base = "https://api.openaq.org/v3"
    try:
        # Find a Lahore location that reports pm25.
        loc = requests.get(
            f"{base}/locations",
            params={"coordinates": "31.5497,74.3436", "radius": 25000, "limit": 100},
            headers=headers, timeout=timeout,
        )
        loc.raise_for_status()
        results = loc.json().get("results", [])
        if not results:
            print("[openaq] no Lahore locations returned")
            return None

        # Use ONE station so pm25/pm10 share a timeline: the first that has pm25
        # (and its pm10 if that same station reports it).
        sensors: dict[str, int] = {}
        for station in results:
            found = {}
            for s in station.get("sensors", []):
                p = (s.get("parameter") or {}).get("name")
                if p in ("pm25", "pm10") and p not in found:
                    found[p] = s["id"]
            if "pm25" in found:
                sensors = found
                break
        if "pm25" not in sensors:
            print("[openaq] no pm25 sensor among Lahore stations")
            return None

        frames = {}
        for param, sid in sensors.items():
            r = requests.get(
                f"{base}/sensors/{sid}/measurements",
                params={"limit": hours}, headers=headers, timeout=timeout,
            )
            r.raise_for_status()
            rows = r.json().get("results", [])
            recs = []
            for m in rows:
                period = m.get("period") or {}
                ts = (period.get("datetimeFrom") or {}).get("utc")
                recs.append({"timestamp": ts, param: m.get("value")})
            if recs:
                frames[param] = pd.DataFrame(recs)

        if "pm25" not in frames:
            return None
        aq = frames["pm25"]
        if "pm10" in frames:
            aq = aq.merge(frames["pm10"], on="timestamp", how="outer")
        aq["timestamp"] = pd.to_datetime(aq["timestamp"], utc=True, errors="coerce")
        aq = aq.dropna(subset=["timestamp"]).sort_values("timestamp")
        if "pm10" not in aq:
            aq["pm10"] = np.nan
        aq = aq[["timestamp", "pm25", "pm10"]].reset_index(drop=True)
        aq.to_csv(OPENAQ_CACHE, index=False)
        print(f"[openaq] fetched {len(aq)} rows -> {OPENAQ_CACHE.name}")
        return aq
    except Exception as exc:  # network, auth, schema drift — all non-fatal
        print(f"[openaq] live fetch failed ({exc.__class__.__name__}: {exc}) -> using PLOS")
        return None


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


def refresh_openaq_cache(max_age_hours: int = 24) -> pd.DataFrame | None:
    """Keep data/openaq_lahore.csv fresh via the live API (best-effort, offline-safe).

    Skips the network call when the cache is younger than max_age_hours, so repeat
    launches are instant and work offline after the first successful fetch.
    """
    try:
        if OPENAQ_CACHE.exists():
            age_h = (time.time() - OPENAQ_CACHE.stat().st_mtime) / 3600
            if age_h < max_age_hours:
                print(f"[openaq] cache fresh ({age_h:.1f}h old) -> {OPENAQ_CACHE.name}")
                return pd.read_csv(OPENAQ_CACHE, parse_dates=["timestamp"])
        return fetch_openaq_lahore()
    except Exception as exc:
        print(f"[openaq] cache refresh skipped ({exc.__class__.__name__}: {exc})")
        return None


def load_air_quality() -> tuple[pd.DataFrame, str]:
    """Return (air-quality df, source label) for the dashboard.

    Always refreshes the OpenAQ live cache (proves the API pipeline). The DISPLAY
    source follows AIR_QUALITY_SOURCE: PLOS gives complete real pm25+pm10, while
    "openaq" uses the live pull when it has usable pm25 (else falls back to PLOS).
    """
    live = refresh_openaq_cache()
    if AIR_QUALITY_SOURCE == "openaq":
        if live is not None and live["pm25"].notna().sum() >= 24:
            return live, "OpenAQ Lahore (live)"
        print("[openaq] live series unusable -> falling back to PLOS Lahore")
    return load_plos_lahore(), "PLOS ONE Lahore (2024)"


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

    return {
        "df": df,
        "aq_source": aq_source,
        "aq_date_range": (aq["timestamp"].min(), aq["timestamp"].max()),
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
    print("rows               :", len(df))
    print(df.head())
    print("mean pm25          :", round(df["pm25"].mean(), 1))
    print("mean spo2          :", round(df["spo2"].mean(), 1))
    print("FEV1/FVC baseline  :", d["fev1_fvc_baseline"], "%  (SEQN", d["seqn"], "age", d["age"], ")")
    print("FEV1/FVC obstruct  :", d["obstruction_flag"])
    print("spo2 deviation     :", d["spo2_deviation"])
    print("pm25<->spo2 corr   :", d["pm25_spo2_corr"])
