"""
patient_twin.py — AirLung Twin  (THE CORE NOVELTY)
==================================================
The "patient half" of the digital twin. It does NOT read vitals from a dataset.
A twin SIMULATES the patient: given the live air + a PROFILE + a real BASELINE,
it models how THAT patient's breathing responds.

Honest framing (use this in the demo):
  - The patient responds to a COMBINED exposure from ALL pollutants (PM2.5, PM10,
    NO2, O3, SO2), each weighted against its health guideline — not PM2.5 alone.
  - Peak flow and the risk level are the defensible outputs. SpO2 is a MODELLED
    estimate (real SpO2 depends on much more than air), shown as an indicator.
  - When ESP32 sensors are added, real readings replace the modelled ones and this
    model becomes the prior knowledge that interprets them.
Grounding: asthma panel studies (higher pollution -> lower peak flow / more
symptoms), e.g. PMC9028503. Coefficients are illustrative, to be calibrated later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLEAN = 15.0   # PM2.5-equivalent "clean" reference (µg/m³)

# health guidelines per pollutant (µg/m³) and how much each one's EXCESS counts
# as PM2.5-equivalent exposure (PM2.5 dominates; the others add).
GUIDELINES = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40}
WEIGHTS = {"pm10": 0.5, "no2": 0.4, "o3": 0.3, "so2": 0.3}
POLLUTANT_NAMES = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃", "so2": "SO₂"}

PROFILES = {
    "healthy": dict(name="Healthy adult", severity="none",
                    personal_best_pef=600, pef_sensitivity=0.05, spo2_base=98, spo2_sensitivity=0.004),
    "mild_asthma": dict(name="Mild asthmatic", severity="mild",
                        personal_best_pef=500, pef_sensitivity=0.15, spo2_base=97, spo2_sensitivity=0.012),
    "moderate_asthma": dict(name="Moderate asthmatic (obstructed)", severity="moderate",
                            personal_best_pef=380, pef_sensitivity=0.30, spo2_base=96, spo2_sensitivity=0.022),
}
DEFAULT_PROFILE = "mild_asthma"


def _col(air, name):
    try:
        return np.asarray(air[name], dtype=float)
    except Exception:
        return np.asarray(0.0)


def combined_exposure(air):
    """PM2.5-equivalent exposure from ALL pollutants. air = DataFrame/dict of columns."""
    eff = _col(air, "pm25").astype(float)
    eff = eff.copy() if eff.ndim else eff
    for k, w in WEIGHTS.items():
        eff = eff + w * np.clip(_col(air, k) - GUIDELINES[k], 0, None)
    return eff


def pollutant_contributions(row) -> list[tuple[str, float]]:
    """How much each pollutant adds to the combined exposure right now (for the
    'what's driving the risk' chart). row = a single air reading (Series/dict)."""
    def g(n):
        try: return float(row[n])
        except Exception: return 0.0
    out = [("PM2.5", max(g("pm25") - CLEAN, 0.0))]
    for k, w in WEIGHTS.items():
        out.append((POLLUTANT_NAMES[k], w * max(g(k) - GUIDELINES[k], 0.0)))
    return [(n, round(v, 1)) for n, v in out if v > 0.5]


def profile_baseline(profile_key: str, nhanes_fev1_fvc: float) -> float:
    sev = PROFILES[profile_key]["severity"]
    if sev == "none": return round(nhanes_fev1_fvc, 1)
    if sev == "mild": return round(nhanes_fev1_fvc * 0.92, 1)
    return round(min(nhanes_fev1_fvc * 0.82, 69.0), 1)


def respond(air, profile_key: str):
    """Vectorised response from COMBINED exposure. Returns arrays."""
    p = PROFILES[profile_key]
    eff = combined_exposure(air)
    excess = np.clip(eff - CLEAN, 0, None)
    pef = p["personal_best_pef"] - np.minimum(p["pef_sensitivity"] * excess, 0.5 * p["personal_best_pef"])
    pef_pct = pef / p["personal_best_pef"] * 100.0
    spo2 = np.clip(p["spo2_base"] - np.minimum(p["spo2_sensitivity"] * excess, 6.0), 85, 100)
    return {"eff": np.round(eff, 1), "pef_pct": np.round(pef_pct, 0), "spo2": np.round(spo2, 1)}


def risk_and_advice(exposure: float, pef_pct: float, spo2: float, severity: str):
    """Status + go/no-go. `exposure` is the combined (all-pollutant) exposure.
    Sensitive patients cross each threshold at LOWER exposure -> same air, different advice."""
    watch, elev, high = {"none": (140, 240, 340), "mild": (60, 130, 240), "moderate": (35, 85, 150)}[severity]
    if pef_pct < 60 or spo2 < 92 or exposure > high:
        return "HIGH RISK", "#D7263D", "No — stay indoors, use inhaler + purifier, monitor closely."
    if pef_pct < 75 or spo2 < 95 or exposure > elev:
        return "ELEVATED", "#F46036", "No — avoid going outside; keep the inhaler close."
    if pef_pct < 90 or exposure > watch:
        return "WATCH", "#F0A202", "Short trips only — limit outdoor exertion."
    return "STABLE", "#2E933C", "Yes — safe for this patient to be outside."


def simulate(air_history: pd.DataFrame, profile_key: str, nhanes_fev1_fvc: float) -> dict:
    r = respond(air_history, profile_key)
    out = air_history[["timestamp", "pm25"]].copy()
    out["spo2"], out["pef_pct"] = r["spo2"], r["pef_pct"]

    pm_now = float(air_history["pm25"].iloc[-1])
    eff_now = float(r["eff"][-1]); spo2_now = float(r["spo2"][-1]); pef_now = float(r["pef_pct"][-1])
    sev = PROFILES[profile_key]["severity"]
    status, color, advice = risk_and_advice(eff_now, pef_now, spo2_now, sev)
    return {
        "series": out, "profile_name": PROFILES[profile_key]["name"],
        "baseline_fev1_fvc": profile_baseline(profile_key, nhanes_fev1_fvc),
        "obstruction": profile_baseline(profile_key, nhanes_fev1_fvc) < 70,
        "pm25_now": round(pm_now, 0), "exposure_now": round(eff_now, 0),
        "spo2_now": round(spo2_now, 1), "pef_pct_now": round(pef_now, 0),
        "status": status, "status_color": color, "advice": advice,
        "contributions": pollutant_contributions(air_history.iloc[-1]),
    }


def forecast_response(air_forecast: pd.DataFrame, profile_key: str) -> pd.DataFrame:
    """Feed the AQI FORECAST (all pollutants) through the patient model -> forecast
    this patient's breathing + hourly status ahead."""
    if air_forecast is None or len(air_forecast) == 0:
        return pd.DataFrame(columns=["timestamp", "pm25", "spo2", "pef_pct", "status", "color"])
    r = respond(air_forecast, profile_key)
    sev = PROFILES[profile_key]["severity"]
    out = air_forecast[["timestamp", "pm25"]].copy()
    out["exposure"], out["spo2"], out["pef_pct"] = r["eff"], r["spo2"], r["pef_pct"]
    rows = [risk_and_advice(e, pf, sp, sev) for e, pf, sp in zip(r["eff"], r["pef_pct"], r["spo2"])]
    out["status"] = [x[0] for x in rows]; out["color"] = [x[1] for x in rows]
    return out.reset_index(drop=True)


def dos_and_donts(status: str, obstruction: bool):
    """Patient-tailored advice (plain text; the UI adds tick/cross icons)."""
    base = {
        "STABLE": (["Normal outdoor activity is fine", "A walk or light exercise is OK",
                    "Stay hydrated as usual"], ["Still avoid smoke and bonfires"]),
        "WATCH": (["Keep your reliever inhaler with you", "Keep outdoor trips short",
                   "Wear a mask on busy roads"], ["Avoid hard exercise outdoors", "Don't leave windows open long"]),
        "ELEVATED": (["Stay indoors where you can", "Keep windows closed", "Keep the inhaler within reach"],
                     ["Avoid going outside", "No outdoor exertion"]),
        "HIGH RISK": (["Stay indoors", "Run an air purifier if you have one",
                       "Wear an N95 mask if you must go out", "Monitor symptoms closely"],
                      ["Do not go outside", "No physical exertion", "Do not open windows"]),
    }
    do, dont = base.get(status, base["STABLE"])
    if obstruction:
        do = do + ["Follow your prescribed controller medication"]
    return do, dont


if __name__ == "__main__":
    air = pd.DataFrame({"timestamp": pd.date_range("2026-11-24", periods=3, freq="h"),
                        "pm25": [40, 90, 160], "pm10": [70, 130, 220],
                        "no2": [30, 60, 90], "o3": [40, 80, 160], "so2": [10, 30, 50]})
    for k in PROFILES:
        s = simulate(air, k, 78.0)
        print(f"{s['profile_name']:32s} PM2.5={s['pm25_now']:.0f} exposure={s['exposure_now']:.0f} "
              f"SpO2={s['spo2_now']:.0f} PEF={s['pef_pct_now']:.0f}% -> {s['status']}")
    print("drivers:", simulate(air, "moderate_asthma", 78.0)["contributions"])