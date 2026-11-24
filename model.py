"""
AirLung Twin — model.py   ===  MODEL v0  (30% checkpoint)  ===
=================================================================
First fitted model: ordinary least-squares linear regression

        SpO2  =  intercept  +  slope · PM2.5

fitted on the dashboard dataframe from data_loader.build_dataset().

What it is / is NOT (say this in the demo):
  * It IS a real fitted model (coefficients, R², significance) that replaces the
    hand-written thresholds' guesswork about "how much SpO2 drops at this PM2.5".
  * It is NOT the prediction model. The full personalised prediction model
    (Khazeena's thesis direction) replaces fit_pm25_spo2() next phase (Jan–Feb).
  * Until co-measured patient + air data exists, the air and vitals series are
    only display-aligned (see data_loader NOTE), so the fitted slope reflects that
    alignment, not physiology. The model reports whether its slope is significant
    and the dashboard only uses it for status when it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODEL_NAME = "Model v0 · linear PM2.5→SpO₂ (OLS)"
SIG_LEVEL = 0.05


@dataclass
class Pm25Spo2Model:
    intercept: float
    slope: float            # SpO2 % change per 1 µg/m³ PM2.5
    r2: float
    p_value: float          # two-sided test of slope != 0 (normal approx.; n is large)
    n: int
    pm25_range: tuple[float, float]

    @property
    def name(self) -> str:
        return MODEL_NAME

    @property
    def per_10(self) -> float:
        """SpO2 change per +10 µg/m³ PM2.5 — the human-readable effect size."""
        return self.slope * 10

    @property
    def shows_inverse_effect(self) -> bool:
        """True only if the data supports 'more PM2.5 -> lower SpO2'."""
        return self.slope < 0 and self.p_value < SIG_LEVEL

    def predict_spo2(self, pm25: float) -> float:
        return self.intercept + self.slope * float(pm25)

    def expected_drop(self, pm25: float, spo2_baseline: float) -> float:
        """Model-expected SpO2 drop below the patient's baseline at this PM2.5 (>= 0).

        Zero when the model found no significant inverse effect, so the status
        never leans on a relation the data doesn't support.
        """
        if not self.shows_inverse_effect:
            return 0.0
        return max(0.0, float(spo2_baseline) - self.predict_spo2(pm25))

    def summary(self) -> str:
        sig = "significant" if self.p_value < SIG_LEVEL else "not significant"
        return (f"SpO₂ = {self.intercept:.2f} {self.slope:+.4f}·PM2.5  "
                f"(R² = {self.r2:.3f}, p = {self.p_value:.2g} → {sig}, n = {self.n})")


def fit_pm25_spo2(df: pd.DataFrame) -> Pm25Spo2Model:
    """Fit SpO2 ~ PM2.5 by OLS. === MODEL SLOT: swap for the trained model next phase ==="""
    d = df[["pm25", "spo2"]].dropna()
    x, y = d["pm25"].to_numpy(float), d["spo2"].to_numpy(float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return Pm25Spo2Model(float(np.mean(y)) if n else float("nan"), 0.0, 0.0, 1.0, n,
                             (float(x.min()) if n else 0.0, float(x.max()) if n else 0.0))

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    se = math.sqrt(ss_res / (n - 2) / float(((x - x.mean()) ** 2).sum())) if ss_res > 0 else 0.0
    if se > 0:
        p = math.erfc(abs(slope / se) / math.sqrt(2))
    else:
        p = 0.0 if slope != 0 else 1.0

    return Pm25Spo2Model(float(intercept), float(slope), float(r2), float(p), n,
                         (float(x.min()), float(x.max())))
