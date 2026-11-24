"""
data_loader.py — AirLung Twin  (single entry point: build_dataset())
====================================================================
New architecture (true to the proposal):

   Open-Meteo live air  ──►  Khazina's CNN-1D forecast  (environment twin)
                         └─►  Patient twin [profile + NHANES baseline + response model]
                                                           (patient twin — the novelty)

Drops BIDMC and OpenAQ. Air is LIVE (air_source.py). The patient's vitals are
MODEL-PREDICTED by patient_twin.py (not a borrowed dataset). FEV1/FVC baseline
stays REAL from NHANES. build_dataset() stays the single data entry point so the
dashboard and the future ESP32/avatar phases all read from one place.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import air_source
import patient_twin

DATA_DIR = Path(__file__).parent / "data"
NHANES_CSV = DATA_DIR / "NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv"


# ----- real lung baseline (NHANES), with a safe default if the file is absent -----
def load_nhanes_baseline() -> dict:
    try:
        nh = pd.read_csv(NHANES_CSV)
        nh.columns = [str(c).strip() for c in nh.columns]
        fev1 = pd.to_numeric(nh["Baseline_FEV1_L"], errors="coerce")
        fvc = pd.to_numeric(nh["Baseline_FVC_L"], errors="coerce")
        age = pd.to_numeric(nh["Age"], errors="coerce")
        mask = (age >= 18) & fev1.notna() & fvc.notna() & (fvc > 0)
        i = (nh.index[mask] if mask.any() else nh.index[fev1.notna() & fvc.notna() & (fvc > 0)])[0]
        fev1_ml, fvc_ml = round(float(fev1[i]) * 1000, 1), round(float(fvc[i]) * 1000, 1)
        return {"fev1_ml": fev1_ml, "fvc_ml": fvc_ml,
                "fev1_fvc": round(fev1_ml / fvc_ml * 100, 1),
                "seqn": int(pd.to_numeric(nh.loc[i, "SEQN"], errors="coerce")),
                "age": float(age[i]), "source": "NHANES (real)"}
    except Exception as e:
        print("[data_loader] NHANES file not found, using default baseline:", e)
        return {"fev1_ml": 3000.0, "fvc_ml": 3850.0, "fev1_fvc": 77.9,
                "seqn": None, "age": None, "source": "default (NHANES file missing)"}


def _us_aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (Sensitive)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


def build_dataset(profile_key: str = patient_twin.DEFAULT_PROFILE,
                  with_forecast: bool = True) -> dict:
    # 1) LIVE air + real forecast
    hist, meteo_fcst, aq_source, is_live = air_source.load_air()

    # 2) Khazina's CNN-1D forecast (optional; falls back gracefully)
    model_fcst, model_metrics = (pd.DataFrame(), {"model": "disabled"})
    if with_forecast:
        try:
            import aqi_forecast
            model_fcst, model_metrics = aqi_forecast.forecast_pm25(hist)
        except Exception as e:
            print("[data_loader] forecast skipped:", e)

    # Blend forecasts: PM2.5 ahead comes from OUR CNN-1D model; the other gases
    # (PM10, NO2, O3, SO2) come from the Open-Meteo forecast. Together they form the
    # patient's combined-exposure forecast. (If the CNN is unavailable, PM2.5 stays
    # from Open-Meteo so nothing breaks.)
    if len(meteo_fcst) and len(model_fcst):
        cnn = model_fcst.set_index("timestamp")["pm25"]
        meteo_fcst = meteo_fcst.copy()
        meteo_fcst["pm25"] = [cnn.get(t, v) for t, v in zip(meteo_fcst["timestamp"], meteo_fcst["pm25"])]

    # 3) real lung baseline
    nh = load_nhanes_baseline()

    # 4) patient twin response for the chosen profile
    patient = patient_twin.simulate(hist, profile_key, nh["fev1_fvc"])

    # 5) "generic city AQI" (what a public AQI site shows: 24-h city average)
    tail = hist[hist["timestamp"] >= hist["timestamp"].max() - pd.Timedelta(hours=24)]
    city_aqi = round(float(tail["us_aqi"].mean()), 0) if "us_aqi" in hist else None
    city_pm25_24h = round(float(tail["pm25"].mean()), 0)

    return {
        "air_history": hist,
        "meteo_forecast": meteo_fcst,       # physics forecast from Open-Meteo
        "model_forecast": model_fcst,       # CNN-1D forecast (Khazina)
        "model_metrics": model_metrics,
        "aq_source": aq_source, "is_live": is_live,
        "aq_latest_time": pd.to_datetime(hist["timestamp"].iloc[-1]),
        "city_aqi": city_aqi, "city_aqi_cat": _us_aqi_category(city_aqi) if city_aqi else "n/a",
        "city_pm25_24h": city_pm25_24h,
        "nhanes": nh,
        "patient": patient,                 # for the default profile; app recomputes on switch
        "profile_key": profile_key,
    }


if __name__ == "__main__":
    d = build_dataset()
    print("air source :", d["aq_source"], "| live:", d["is_live"])
    print("latest air :", d["aq_latest_time"])
    print("city 24h   : PM2.5", d["city_pm25_24h"], "| AQI", d["city_aqi"], d["city_aqi_cat"])
    print("baseline   : FEV1/FVC", d["nhanes"]["fev1_fvc"], "%", "(", d["nhanes"]["source"], ")")
    print("forecast   :", d["model_metrics"])
    p = d["patient"]
    print("patient    :", p["profile_name"], "->", p["status"], "|", p["advice"])