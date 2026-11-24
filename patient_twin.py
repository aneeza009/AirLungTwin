"""
patient_twin.py — AirLung Twin  (THE CORE NOVELTY)
==================================================
This is the "patient half" of the digital twin. It does NOT read vitals from any
dataset. A digital twin SIMULATES the patient: given the live air + a patient
PROFILE + a real lung BASELINE, it computes how THAT patient's breathing responds.

Three parts (per the AirLung Twin proposal):
  1. PROFILE   — who the patient is (asthma severity, sensitivity). You define it.
  2. BASELINE  — FEV1/FVC, real, from NHANES (constant, location-independent).
  3. RESPONSE  — air-driven model: PM2.5 -> this patient's peak-flow drop, SpO2
                 dip and risk level, using published exposure-response behaviour.

IMPORTANT (say this in the demo): the patient vitals here are MODEL-PREDICTED
("simulated"), not measured. When the ESP32 sensors are added later, real SpO2/HR
replace the simulated values and this same model becomes the "prior knowledge"
that interprets them. Nothing here is thrown away.

Exposure-response is grounded in asthma panel-study behaviour (higher PM2.5 ->
lower peak expiratory flow, more so for asthmatics):
  e.g. Associations between daily ambient air pollution and pulmonary function
  in asthma patients, PMC9028503. Coefficients below are illustrative and are
  meant to be calibrated against real ESP32 patient data later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLEAN_PM25 = 15.0   # WHO-ish "clean" reference; response grows above this

# --- PATIENT PROFILES -------------------------------------------------
# personal_best_pef: the patient's own best peak flow (L/min)
# pef_sensitivity:   L/min drop per µg/m³ of PM2.5 above CLEAN_PM25
# spo2_sensitivity:  % SpO2 drop per µg/m³ above CLEAN_PM25
# fev1_fvc:          lung baseline (%). Healthy uses the real NHANES value;
#                    asthmatic profiles are derived from it (lower = obstructed).
PROFILES = {
    "healthy": dict(
        name="Healthy adult", severity="none",
        personal_best_pef=600, pef_sensitivity=0.05,
        spo2_base=98, spo2_sensitivity=0.004),
    "mild_asthma": dict(
        name="Mild asthmatic", severity="mild",
        personal_best_pef=500, pef_sensitivity=0.15,
        spo2_base=97, spo2_sensitivity=0.012),
    "moderate_asthma": dict(
        name="Moderate asthmatic (obstructed)", severity="moderate",
        personal_best_pef=380, pef_sensitivity=0.30,
        spo2_base=96, spo2_sensitivity=0.022),
}
DEFAULT_PROFILE = "mild_asthma"


def profile_baseline(profile_key: str, nhanes_fev1_fvc: float) -> float:
    """Lung baseline FEV1/FVC (%) for a profile, anchored on the REAL NHANES value."""
    sev = PROFILES[profile_key]["severity"]
    if sev == "none":    return round(nhanes_fev1_fvc, 1)
    if sev == "mild":    return round(nhanes_fev1_fvc * 0.92, 1)
    return round(min(nhanes_fev1_fvc * 0.82, 69.0), 1)      # obstructed (<70)


def respond(pm25, profile_key: str):
    """Vectorised response. pm25 may be a scalar or array.
    Returns dict of arrays/scalars: pef (L/min), pef_pct (% of personal best),
    spo2 (%)."""
    p = PROFILES[profile_key]
    pm = np.asarray(pm25, dtype=float)
    excess = np.clip(pm - CLEAN_PM25, 0, None)
    pef = p["personal_best_pef"] - np.minimum(
        p["pef_sensitivity"] * excess, 0.5 * p["personal_best_pef"])
    pef_pct = pef / p["personal_best_pef"] * 100.0
    spo2 = np.clip(p["spo2_base"] - np.minimum(p["spo2_sensitivity"] * excess, 6.0), 85, 100)
    return {"pef": np.round(pef, 0), "pef_pct": np.round(pef_pct, 0), "spo2": np.round(spo2, 1)}


def risk_and_advice(pm25: float, pef_pct: float, spo2: float):
    """Combine air + predicted response into a status + plain go/no-go advice."""
    if pef_pct < 60 or spo2 < 92 or pm25 > 250:
        return "HIGH RISK", "#D7263D", "No — stay indoors, use inhaler/purifier, monitor closely."
    if pef_pct < 75 or spo2 < 95 or pm25 > 150:
        return "ELEVATED", "#F46036", "No — avoid going out; keep the inhaler close."
    if pef_pct < 90 or pm25 > 55:
        return "WATCH", "#F0A202", "Short trips only; limit outdoor exertion."
    return "STABLE", "#2E933C", "Yes — safe for this patient to be outdoors."


def simulate(air_history: pd.DataFrame, profile_key: str, nhanes_fev1_fvc: float) -> dict:
    """Produce the patient twin's response aligned to the air history + a 'now' summary."""
    r = respond(air_history["pm25"].to_numpy(), profile_key)
    out = air_history[["timestamp", "pm25"]].copy()
    out["spo2"] = r["spo2"]
    out["pef_pct"] = r["pef_pct"]

    pm_now = float(air_history["pm25"].iloc[-1])
    spo2_now = float(r["spo2"][-1])
    pef_now = float(r["pef_pct"][-1])
    status, color, advice = risk_and_advice(pm_now, pef_now, spo2_now)
    return {
        "series": out,                       # timestamp, pm25, spo2, pef_pct
        "profile_name": PROFILES[profile_key]["name"],
        "baseline_fev1_fvc": profile_baseline(profile_key, nhanes_fev1_fvc),
        "obstruction": profile_baseline(profile_key, nhanes_fev1_fvc) < 70,
        "pm25_now": round(pm_now, 0),
        "spo2_now": round(spo2_now, 1),
        "pef_pct_now": round(pef_now, 0),
        "status": status, "status_color": color, "advice": advice,
    }


if __name__ == "__main__":
    # quick self-test: same air, three patients -> different responses
    air = pd.DataFrame({"timestamp": pd.date_range("2026-11-24", periods=5, freq="h"),
                        "pm25": [30, 90, 160, 220, 300]})
    for k in PROFILES:
        s = simulate(air, k, nhanes_fev1_fvc=78.0)
        print(f"{s['profile_name']:32s} PM2.5={s['pm25_now']:.0f} "
              f"SpO2={s['spo2_now']:.1f} PEF={s['pef_pct_now']:.0f}% -> {s['status']}")