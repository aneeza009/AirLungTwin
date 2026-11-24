"""
AirLung Twin — 30% Implementation Dashboard (patient-centric)
=============================================================
FYP: AirLung Twin — IoMT-Enabled Cognitive Digital Twin for Respiratory Health
Team: Muneefah Shahzad (FA23-BCS-144), Aneeza Naheen (FA23-BCS-180), Adan Zia (FA23-BCS-021)

Drops straight into the repo: consumes data_loader.build_dataset() (the real four
datasets). Falls back to sample data only if data_loader can't be imported, so it
still runs standalone.

WHAT THIS VERSION ADDS (supervisor feedback, 23 Nov):
  * PATIENT-FIRST, not air-first — the screen is about ONE asthma patient and how the
    air affects THEIR breathing (the project's novelty / the gap we solved).
  * INTERPRETATION layer — plain-language sentences from the numbers, not raw values.
  * The RELATION made explicit — PM2.5 vs SpO2 fusion + scatter with correlation r.
  * DATA MODE badge (LIVE vs STATIC) from data_loader's aq_source — honest about live.
  * Rule-based STATUS now, with a clearly marked MODEL SLOT for the prediction model.
  * Structured so the moment live/real-patient data flows in (and later the avatar app),
    this same code extends — build_dataset() stays the single data entry point.

Run:  pip install -r requirements.txt   then   python app.py   ->  http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

from model import fit_pm25_spo2, MODEL_NAME

PATIENT_LABEL = "Asthma patient (demo)"   # single-patient design, per proposal
PM25_HAZARD = 150

# ======================================================================
# === DATA: real via data_loader, else sample fallback ================
# ======================================================================
try:
    from data_loader import build_dataset
    DATA = build_dataset()
    df = DATA["df"]
    fev1_fvc_baseline = DATA["fev1_fvc_baseline"]
    obstruction_flag = DATA["obstruction_flag"]
    AQ_SOURCE = DATA.get("aq_source", "dataset")
    spo2_baseline = DATA.get("spo2_baseline")
    corr = DATA.get("pm25_spo2_corr")
    city_pm25_24h, city_aqi = DATA["city_pm25_24h"], DATA["city_aqi"]
    aq_latest_time = DATA.get("aq_latest_time")
    _REAL = True
except Exception as e:                      # standalone fallback (no data_loader present)
    print("data_loader not available, using sample data:", e)
    rng = np.random.default_rng(7)
    n = 240
    ts = pd.date_range("2026-11-24 06:00", periods=n, freq="5min")
    pm25 = 120 + np.cumsum(rng.normal(0, 3, n)).clip(-50, 160)
    spo2 = (98 - (pm25 - 120).clip(0, None) * 0.02 + rng.normal(0, 0.3, n)).clip(88, 100)
    df = pd.DataFrame({"timestamp": ts, "pm25": pm25.round(1), "pm10": (pm25 * 1.4).round(1),
                       "spo2": spo2.round(1),
                       "heart_rate": (80 + (pm25 - 120).clip(0, None) * 0.04).round(0),
                       "resp": (15 + (pm25 - 120).clip(0, None) * 0.015).round(1)})
    fev1_fvc_baseline, obstruction_flag = 70.0, False
    AQ_SOURCE, spo2_baseline, corr, _REAL = "Sample data", None, None, False
    city_pm25_24h, aq_latest_time = round(float(df["pm25"].tail(24).mean()), 1), None
    city_aqi = None

if spo2_baseline is None:
    spo2_baseline = round(float(df["spo2"].quantile(0.95)), 1)
if corr is None or (isinstance(corr, float) and np.isnan(corr)):
    corr = float(df["pm25"].corr(df["spo2"]))

# LIVE only if data_loader accepted a fresh live feed (it labels those "(live)")
DATA_MODE = "LIVE" if "(live)" in AQ_SOURCE.lower() else "STATIC"
AQ_SOURCE_LINE = AQ_SOURCE + (f" · last reading {aq_latest_time:%d %b %H:%M} UTC"
                              if aq_latest_time is not None and DATA_MODE == "LIVE" else "")

# === MODEL v0: fitted PM2.5 -> SpO2 relation (model.py) — replaced by the trained
# prediction model next phase. Fitted on the same df the charts show.
MODEL = fit_pm25_spo2(df)
# ======================================================================

def aqi_category(pm):
    if pm <= 12: return "Good", "#36B37E"
    if pm <= 35.4: return "Moderate", "#FFD93D"
    if pm <= 55.4: return "Unhealthy (Sensitive)", "#FF9F43"
    if pm <= 150.4: return "Unhealthy", "#FF6B6B"
    if pm <= 250.4: return "Very Unhealthy", "#C44CFF"
    return "Hazardous", "#B33030"

if city_aqi is None:
    try:
        from data_loader import pm25_to_aqi
        city_aqi = pm25_to_aqi(city_pm25_24h)
    except Exception:
        city_aqi = None

latest = df.iloc[-1]
spo2_drop = round(float(spo2_baseline) - float(latest["spo2"]), 1)
cat, cat_color = aqi_category(latest["pm25"])

# ===== STATUS  === MODEL SLOT ===
# Model v0 supplies the EXPECTED SpO2 drop at the current PM2.5; status takes the worse
# of observed vs model-expected drop, then applies clinical-style thresholds.
# Next phase: the trained prediction model replaces MODEL.expected_drop().
model_drop = round(MODEL.expected_drop(latest["pm25"], spo2_baseline), 2)
model_spo2_now = MODEL.predict_spo2(latest["pm25"])

def patient_status(pm25, observed_drop, expected_drop):
    drop = max(observed_drop, expected_drop)
    if pm25 > PM25_HAZARD and drop >= 2: return "ELEVATED RISK", "#FF5C5C"
    if pm25 > 55 or drop >= 1:           return "WATCH", "#FFB020"
    return "STABLE", "#36B37E"
status_text, status_color = patient_status(latest["pm25"], spo2_drop, model_drop)

# ---- generic health advice by AQI band, but applied to THIS patient ----
def advice(pm):
    if pm <= 35.4: return "Safe for the patient to be outdoors."
    if pm <= 55.4: return "Sensitive patient: limit prolonged outdoor exertion."
    if pm <= 150.4: return "Asthma patient should stay indoors; keep inhaler close."
    return "Avoid outdoor exposure; mask + purifier indoors; monitor SpO₂."

def build_insights():
    out = [(f"Air right now is {cat} — PM2.5 {latest['pm25']:.0f} µg/m³. {advice(latest['pm25'])}", cat_color)]
    if spo2_drop >= 1:
        out.append((f"This patient's SpO₂ is {spo2_drop:.1f}% below their baseline "
                    f"({spo2_baseline:.0f}%) — early respiratory stress under this air.", "#FF9F43"))
    else:
        out.append((f"SpO₂ is holding near baseline ({spo2_baseline:.0f}%) — no stress right now.", "#36B37E"))
    if MODEL.shows_inverse_effect:
        out.append((f"{MODEL_NAME}: every +10 µg/m³ PM2.5 lowers this patient's SpO₂ by "
                    f"{-MODEL.per_10:.2f}% (r = {corr:.2f}); at today's air it expects "
                    f"{model_spo2_now:.1f}%.", "#8AB4FF"))
    else:
        out.append((f"{MODEL_NAME}: no significant PM2.5→SpO₂ effect in the current data "
                    f"(r = {corr:.2f}, p = {MODEL.p_value:.2g}). Expected — the vitals are a reference "
                    f"recording, not measured during this air; the link appears once co-measured "
                    f"data flows in.", "#8AB4FF"))
    out.append((f"Lung baseline FEV1/FVC = {fev1_fvc_baseline:.0f}% "
                f"({'below normal / obstructed' if obstruction_flag else 'normal range'}); "
                f"every reading is judged against this anchor.", "#C7A0FF"))
    return out

def generic_advice(pm):
    # what a public AQI site tells EVERYONE (US EPA wording, condensed)
    if pm <= 12: return "Air quality is satisfactory for everyone."
    if pm <= 35.4: return "Unusually sensitive people should reduce prolonged exertion."
    if pm <= 55.4: return "Sensitive groups should reduce prolonged outdoor exertion."
    if pm <= 150.4: return "Everyone should reduce prolonged outdoor exertion."
    if pm <= 250.4: return "Everyone should avoid prolonged outdoor exertion."
    return "Everyone should avoid all outdoor activity."

def personal_advice():
    tips = [advice(latest["pm25"])]
    if obstruction_flag:
        tips.append("Obstructed lung baseline — act one AQI band earlier than the public advice.")
    if spo2_drop >= 1:
        tips.append(f"SpO₂ already {spo2_drop:.1f}% under baseline — use reliever; seek care if it falls below 92%.")
    return " ".join(tips)

def compare_row(label, generic, patient):
    return html.Div(className="cmp-row", children=[
        html.Div(label, className="cmp-k"), html.Div(generic, className="cmp-g"),
        html.Div(patient, className="cmp-p")])

def comparison_panel():
    g_cat, g_col = aqi_category(city_pm25_24h)
    model_line = (f"{model_spo2_now:.1f}% expected at PM2.5 {latest['pm25']:.0f} "
                  f"({MODEL.per_10:+.2f}% per +10 µg/m³)" if MODEL.shows_inverse_effect
                  else "no significant air effect fitted yet (see model note)")
    return html.Div(className="panel", children=[
        html.Div("Generic city AQI vs this patient", className="sec"),
        html.Div(className="cmp-row cmp-h", children=[
            html.Div(""), html.Div("Generic city AQI (Punjab EPA / aqicn style)"),
            html.Div("This patient — AirLung Twin")]),
        compare_row("Headline",
                    html.Span([html.B(f"AQI {city_aqi if city_aqi is not None else '—'}",
                                      style={"color": g_col}), f" · {g_cat}"]),
                    html.Span([html.B(status_text, style={"color": status_color}),
                               f" · SpO₂ {latest['spo2']:.1f}% vs baseline {spo2_baseline:.0f}%"])),
        compare_row("Based on", f"Lahore 24-h mean PM2.5 {city_pm25_24h:.0f} µg/m³ — same number for everyone",
                    f"PM2.5 now {latest['pm25']:.0f} µg/m³ + this patient's vitals + FEV1/FVC "
                    f"{fev1_fvc_baseline:.0f}%"),
        compare_row("Breathing response", "not measured",
                    f"observed {(-spo2_drop):+.1f}% vs baseline; model: {model_line}"),
        compare_row("Advice", generic_advice(city_pm25_24h), personal_advice()),
    ])

SIGNALS = {
    "pm25": {"label": "PM2.5", "unit": "µg/m³", "color": "#6C5CE7"},
    "pm10": {"label": "PM10", "unit": "µg/m³", "color": "#00B8D9"},
    "spo2": {"label": "SpO₂", "unit": "%", "color": "#36B37E"},
    "heart_rate": {"label": "Heart Rate", "unit": "bpm", "color": "#FF5C8A"},
    "resp": {"label": "Respiratory Rate", "unit": "br/min", "color": "#FFB020"},
}

def kpi(title, value, sub, color):
    return html.Div(className="card", children=[
        html.Div(title, className="k-t", style={"color": color}),
        html.Div(value, className="k-v"), html.Div(sub, className="k-s")])

app = Dash(__name__)
app.title = "AirLung Twin — Patient Dashboard"

app.layout = html.Div(className="page", children=[
    html.Div(className="hdr", children=[
        html.Div([html.Div("AirLung Twin", className="brand"),
                  html.Div(f"Patient: {PATIENT_LABEL}", className="sub")]),
        html.Div([html.Span("DATA MODE ", className="mode-l"),
                  html.Span(DATA_MODE, className="mode-v"),
                  html.Div(AQ_SOURCE_LINE, className="mode-src")], className="mode"),
        html.Div([html.Div("CURRENT STATUS", className="st-l"),
                  html.Div(status_text, className="st-v", style={"color": status_color})],
                 className="status", style={"borderColor": status_color}),
    ]),

    html.Div(className="insights", children=[html.Div("What this means", className="sec")] +
             [html.Div("• " + t, className="ins", style={"color": c}) for t, c in build_insights()]),

    html.Div(className="kpi-row", children=[
        kpi("PM2.5 NOW", f"{latest['pm25']:.0f}", cat, cat_color),
        kpi("SpO₂ NOW", f"{latest['spo2']:.1f}%", f"baseline {spo2_baseline:.0f}%", "#36B37E"),
        kpi("HEART RATE", f"{latest['heart_rate']:.0f}", "bpm", "#FF5C8A"),
        kpi("RESP. RATE", f"{latest['resp']:.1f}", "breaths/min", "#FFB020"),
        kpi("FEV1/FVC", f"{fev1_fvc_baseline:.0f}%", "lung baseline (NHANES)", "#C7A0FF"),
    ]),

    html.Div(className="two", children=[
        html.Div(className="panel", children=[
            html.Div("Air ↔ Breathing over time (PM2.5 vs SpO₂)", className="sec"),
            dcc.Graph(id="fusion", config={"displayModeBar": False})]),
        html.Div(className="panel", children=[
            html.Div(f"The relationship — {MODEL_NAME}", className="sec"),
            dcc.Graph(id="scatter", config={"displayModeBar": False}),
            html.Div(MODEL.summary(), className="model-note")]),
    ]),

    comparison_panel(),

    html.Div(className="panel", children=[
        html.Div("Explore any signal", className="sec"),
        html.Div(className="ctl", children=[
            dcc.Dropdown(id="signal", value="pm25", clearable=False,
                         options=[{"label": v["label"], "value": k} for k, v in SIGNALS.items()])]),
        dcc.Graph(id="trend", config={"displayModeBar": False})]),

    html.Div("Status = Model v0 (fitted linear PM2.5→SpO₂, model.py) + threshold rules. It is the "
             "FIRST model, not the prediction model — the trained prediction model replaces it at the "
             "marked MODEL SLOT next phase. Patient vitals use a reference recording until live sensors "
             "are added, so air and vitals are display-aligned, not co-measured. The avatar app (next "
             "presentation) reads this same build_dataset() pipeline.",
             className="foot"),
])

def _dark(fig, h=300):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=55, r=55, t=20, b=35), height=h)
    return fig

@callback(Output("fusion", "figure"), Input("signal", "value"))
def fusion(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pm25"], name="PM2.5 (µg/m³)",
                             line=dict(color="#6C5CE7", width=2)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spo2"], name="SpO₂ (%)", yaxis="y2",
                             line=dict(color="#36B37E", width=2)))
    fig.update_layout(yaxis=dict(title="PM2.5"), yaxis2=dict(title="SpO₂", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    return _dark(fig)

@callback(Output("scatter", "figure"), Input("signal", "value"))
def scatter(_):
    x, y = df["pm25"].to_numpy(float), df["spo2"].to_numpy(float)
    xs = np.linspace(x.min(), x.max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="readings",
                             marker=dict(color="#8AB4FF", size=6, opacity=0.6)))
    fig.add_trace(go.Scatter(x=xs, y=MODEL.intercept + MODEL.slope * xs,
                             mode="lines", name="model v0 fit",
                             line=dict(color="#FF9F43", width=2)))
    fig.update_layout(xaxis=dict(title="PM2.5 (µg/m³)"), yaxis=dict(title="SpO₂ (%)"),
                      legend=dict(orientation="h", y=1.12))
    return _dark(fig)

@callback(Output("trend", "figure"), Input("signal", "value"))
def trend(sig):
    mta = SIGNALS[sig]
    fig = go.Figure(go.Scatter(x=df["timestamp"], y=df[sig], mode="lines",
                               line=dict(color=mta["color"], width=2)))
    fig.update_layout(yaxis_title=f"{mta['label']} ({mta['unit']})")
    return _dark(fig, 320)

app.index_string = """
<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
 body{margin:0;background:#0b1220;color:#cdd6e8;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
 .page{padding:18px 26px;max-width:1200px;margin:auto}
 .hdr{display:flex;align-items:center;gap:16px;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap}
 .brand{font-size:20px;font-weight:800;color:#fff}.sub{font-size:13px;color:#8aa0c6}
 .mode{font-size:12px;text-align:center}.mode-l{color:#7d8aa5}.mode-v{color:#8AB4FF;font-weight:700}
 .mode-src{font-size:10px;color:#6b779a;margin-top:2px}
 .status{border:2px solid;border-radius:12px;padding:8px 16px;text-align:center}
 .st-l{font-size:10px;color:#7d8aa5;letter-spacing:1px}.st-v{font-size:20px;font-weight:800}
 .insights{background:#111a2e;border:1px solid #1e2a44;border-radius:14px;padding:14px 18px;margin-bottom:16px}
 .sec{font-size:13px;font-weight:700;color:#aeb9d4;margin-bottom:8px}
 .ins{font-size:14px;margin:4px 0;line-height:1.5}
 .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}
 .card{background:#111a2e;border:1px solid #1e2a44;border-radius:14px;padding:14px 16px}
 .k-t{font-size:10px;font-weight:700;letter-spacing:1px}.k-v{font-size:30px;font-weight:800;color:#fff;margin-top:3px}
 .k-s{font-size:11px;color:#7d8aa5;margin-top:4px}
 .two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
 .panel{background:#111a2e;border:1px solid #1e2a44;border-radius:14px;padding:16px;margin-top:16px}
 .two .panel{margin-top:0}.ctl{max-width:320px;margin-bottom:8px}
 .model-note{font-size:11px;color:#8aa0c6;margin-top:6px;font-family:ui-monospace,Menlo,monospace}
 .cmp-row{display:grid;grid-template-columns:140px 1fr 1fr;gap:14px;padding:8px 0;border-top:1px solid #1e2a44;font-size:13px;line-height:1.45}
 .cmp-h{border-top:0;font-size:11px;font-weight:700;letter-spacing:.5px;color:#7d8aa5}
 .cmp-k{color:#7d8aa5;font-size:12px}.cmp-g{color:#aeb9d4}.cmp-p{color:#e6ecf8}
 @media(max-width:800px){.cmp-row{grid-template-columns:1fr}.cmp-h>div:first-child{display:none}}
 .foot{font-size:11px;color:#6b779a;margin-top:18px;line-height:1.5}
 @media(max-width:800px){.two{grid-template-columns:1fr}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True)
