"""
aqi_forecast.py — AirLung Twin
==============================
Replicates Khazina Naveed's thesis approach: time-series AQI forecasting with a
1-D Convolutional Neural Network (CNN-1D) — her best model (R2 ~ 0.999).

Method (same framing as the thesis):
  - take the recent hourly PM2.5 history,
  - normalise, build sliding windows (lookback -> next-step),
  - train a small CNN-1D, forecast the next `horizon` hours,
  - report RMSE / MAE / R2 (the metrics she used).

TensorFlow is OPTIONAL. If it isn't installed, this falls back to a seasonal-naive
forecast (repeat the last 24-h shape) so the dashboard always has a forecast.
Run `python aqi_forecast.py` to train on live data and print the metrics for the demo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LOOKBACK = 24     # hours of history the model looks at
HORIZON = 24      # hours ahead to predict


def _windows(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def _cnn1d_forecast(pm25: np.ndarray, horizon: int):
    """Khazina's CNN-1D. Returns (forecast, metrics_dict). Raises if TF missing."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input

    lo, hi = float(pm25.min()), float(pm25.max())
    rng = (hi - lo) or 1.0
    norm = (pm25 - lo) / rng

    X, y = _windows(norm, LOOKBACK)
    X = X[..., None]                                   # (samples, lookback, 1)
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    model = Sequential([
        Input((LOOKBACK, 1)),
        Conv1D(64, 3, activation="relu"),
        Conv1D(64, 3, activation="relu"),
        Flatten(), Dense(50, activation="relu"), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr, ytr, epochs=30, batch_size=16, verbose=0)

    # metrics on held-out test (same metrics as the thesis)
    pred = model.predict(Xte, verbose=0).ravel()
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)) * rng)
    mae = float(np.mean(np.abs(pred - yte)) * rng)
    ss_res = np.sum((yte - pred) ** 2); ss_tot = np.sum((yte - yte.mean()) ** 2) or 1
    r2 = float(1 - ss_res / ss_tot)

    # iterative multi-step forecast
    win = norm[-LOOKBACK:].copy()
    fc = []
    for _ in range(horizon):
        nxt = float(model.predict(win[None, :, None], verbose=0).ravel()[0])
        fc.append(nxt); win = np.append(win[1:], nxt)
    forecast = np.clip(np.array(fc) * rng + lo, 0, None)
    return forecast, {"model": "CNN-1D (Khazina)", "rmse": round(rmse, 2),
                      "mae": round(mae, 2), "r2": round(r2, 4)}


def _naive_forecast(pm25: np.ndarray, horizon: int):
    """Fallback: repeat the last 24-h daily shape around the recent mean."""
    last_day = pm25[-24:] if len(pm25) >= 24 else pm25
    reps = int(np.ceil(horizon / len(last_day)))
    fc = np.tile(last_day, reps)[:horizon]
    return np.clip(fc, 0, None), {"model": "seasonal-naive (TF not installed)",
                                  "rmse": None, "mae": None, "r2": None}


def forecast_pm25(air_history: pd.DataFrame, horizon: int = HORIZON):
    """Return (forecast_df, metrics). forecast_df: timestamp, pm25 (predicted)."""
    pm = air_history["pm25"].to_numpy(float)
    try:
        fc, metrics = _cnn1d_forecast(pm, horizon)
    except Exception as e:
        print("[aqi_forecast] CNN-1D unavailable, using fallback:", e)
        fc, metrics = _naive_forecast(pm, horizon)
    last_t = pd.to_datetime(air_history["timestamp"].iloc[-1])
    ts = pd.date_range(last_t + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.DataFrame({"timestamp": ts, "pm25": np.round(fc, 1)}), metrics


if __name__ == "__main__":
    from air_source import load_air
    hist, _, src, _ = load_air()
    fc, m = forecast_pm25(hist)
    print("air source:", src)
    print("model:", m["model"], "| RMSE:", m["rmse"], "| MAE:", m["mae"], "| R2:", m["r2"])
    print(fc.head().to_string(index=False))