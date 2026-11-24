"""
AirLung Twin — 30% Implementation Dashboard (Plotly Dash)
=========================================================
FYP: AirLung Twin — An IoMT-Enabled Cognitive Digital Twin for Respiratory Health
Team: Muneefah Shahzad (FA23-BCS-144), Aneeza Naheen (FA23-BCS-180), Adan Zia (FA23-BCS-021)

30% checkpoint dashboard. Mirrors the supervisor's reference (KPI cards + signal
dropdown + trend chart) and adds a PM2.5-vs-SpO2 overlay. The KPI card set is a
CONFIGURABLE list (KPIS) — add/remove cards freely, driven by the data.

Runs out of the box on realistic SAMPLE data. Swap in your real datasets at the
marked block.

Run:
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:8050
"""

import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input

# ======================================================================
# === REAL DATA ========================================================
# ======================================================================
# All four finalized datasets are loaded + cleaned in data_loader.py:
#   pm25, pm10             -> OpenAQ Lahore (live) / PLOS ONE Lahore
#   spo2, heart_rate, resp -> PhysioNet BIDMC bidmc_01_Numerics.csv
#   fev1_fvc_baseline, fvc_ml, fev1_ml, obstruction_flag -> NHANES spirometry
# build_dataset() returns the DataFrame `df` plus baseline + derived scalars.
# ----------------------------------------------------------------------
from data_loader import build_dataset

DATA = build_dataset()
df = DATA["df"]                                   # timestamp, pm25, pm10, spo2, heart_rate, resp
fev1_ml = DATA["fev1_ml"]
fvc_ml = DATA["fvc_ml"]
fev1_fvc_baseline = DATA["fev1_fvc_baseline"]     # %
obstruction_flag = DATA["obstruction_flag"]       # NHANES FEV1/FVC < 70% rule
AQ_SOURCE = DATA["aq_source"]                     # "OpenAQ Lahore (live)" or "PLOS ONE Lahore"
SPO2_BASELINE = DATA["spo2_baseline"]
SPO2_DEVIATION = DATA["spo2_deviation"]
PM25_SPO2_CORR = DATA["pm25_spo2_corr"]
# ======================================================================
# === END DATA SECTION =================================================
# ======================================================================

PM25_HAZARD = 150  # µg/m³ threshold for "% time hazardous"


def aqi_category(pm):
    if pm <= 12: return "Good"
    if pm <= 35.4: return "Moderate"
    if pm <= 55.4: return "Unhealthy (Sensitive)"
    if pm <= 150.4: return "Unhealthy"
    if pm <= 250.4: return "Very Unhealthy"
    return "Hazardous"


# --- Signals for the trend dropdown ---
SIGNALS = {
    "pm25":       {"label": "PM2.5",      "unit": "µg/m³", "color": "#6C5CE7"},
    "pm10":       {"label": "PM10",       "unit": "µg/m³", "color": "#00B8D9"},
    "spo2":       {"label": "SpO₂",       "unit": "%",     "color": "#36B37E"},
    "heart_rate": {"label": "Heart Rate", "unit": "bpm",   "color": "#FF5C8A"},
    "resp":       {"label": "Resp. Rate", "unit": "br/min","color": "#FFB020"},
}

# --- CONFIGURABLE KPI CARDS ---  (title, value, subtitle, colour)
# Add or remove entries here — the layout flows automatically.
pct_hazard = (df["pm25"] > PM25_HAZARD).mean() * 100
KPIS = [
    # Air quality (OpenAQ live / PLOS ONE Lahore)
    ("MEAN PM2.5",      f"{df['pm25'].mean():.1f}",  f"{AQ_SOURCE} — µg/m³",        "#00B8D9"),
    ("MEAN PM10",       f"{df['pm10'].mean():.1f}",  f"{AQ_SOURCE} — µg/m³",        "#00B8D9"),
    ("PEAK PM2.5",      f"{df['pm25'].max():.1f}",   "Worst exposure — µg/m³",      "#00B8D9"),
    ("% TIME HAZARDOUS",f"{pct_hazard:.0f}%",        f"PM2.5 > {PM25_HAZARD} µg/m³", "#00B8D9"),
    ("AQI CATEGORY",    aqi_category(df['pm25'].mean()), "Based on mean PM2.5",     "#00B8D9"),
    # Vitals (BIDMC — single recording = patient)
    ("MEAN SpO₂",       f"{df['spo2'].mean():.1f}",  "BIDMC — %",                   "#36B37E"),
    ("MIN SpO₂",        f"{df['spo2'].min():.1f}",   "Lowest dip — %",              "#36B37E"),
    ("MEAN HEART RATE", f"{df['heart_rate'].mean():.0f}", "BIDMC — bpm",            "#FF5C8A"),
    ("MAX HEART RATE",  f"{df['heart_rate'].max():.0f}",  "BIDMC — bpm",            "#FF5C8A"),
    ("MEAN RESP. RATE", f"{df['resp'].mean():.1f}",  "BIDMC — breaths/min",         "#FFB020"),
    # Spirometry (NHANES)
    ("FEV1/FVC BASELINE", f"{fev1_fvc_baseline:.1f}", "NHANES anchor — %",          "#6C5CE7"),
    ("FEV1",            f"{fev1_ml:.0f}",            "NHANES baseline — mL",        "#6C5CE7"),
    ("FVC",             f"{fvc_ml:.0f}",             "NHANES baseline — mL",        "#6C5CE7"),
    ("OBSTRUCTION",     "Yes" if obstruction_flag else "No", "FEV1/FVC < 70% / LLN","#6C5CE7"),
    # Derived (no ML)
    ("SpO₂ DEVIATION",  f"{SPO2_DEVIATION:+.2f}",    f"vs baseline {SPO2_BASELINE:.1f}% — pts", "#F78DA7"),
    ("PM2.5 ↔ SpO₂ CORR", f"{PM25_SPO2_CORR:+.2f}",  "Exposure–vitals link (r)",    "#F78DA7"),
]


def kpi_card(title, value, subtitle, color):
    return html.Div(className="card", children=[
        html.Div(title, className="card-title", style={"color": color}),
        html.Div(value, className="card-value"),
        html.Div(subtitle, className="card-sub"),
    ])


app = Dash(__name__)
app.title = "AirLung Twin — 30% Dashboard"

app.layout = html.Div(className="page", children=[
    html.Div([
        html.Div("AirLung Twin — Respiratory Digital Twin (30% Checkpoint)", className="header"),
        html.Div(f"Air quality: {AQ_SOURCE} · Vitals: PhysioNet BIDMC · Spirometry: NHANES",
                 className="subhead"),
    ]),

    html.Div(className="kpi-row", children=[kpi_card(*k) for k in KPIS]),

    html.Div(className="main-row", children=[
        html.Div(className="controls", children=[
            html.Div("Signal", className="ctl-label"),
            dcc.Dropdown(
                id="signal",
                options=[{"label": v["label"], "value": k} for k, v in SIGNALS.items()],
                value="pm25", clearable=False,
            ),
        ]),
        html.Div(className="chart-wrap", children=[dcc.Graph(id="trend", config={"displayModeBar": False})]),
    ]),

    html.Div(className="chart-wrap full", children=[
        html.Div("Air Quality ↔ Vitals Fusion (PM2.5 vs SpO₂)", className="chart-title"),
        dcc.Graph(id="overlay", config={"displayModeBar": False}),
    ]),
])


@callback(Output("trend", "figure"), Input("signal", "value"))
def update_trend(signal):
    meta = SIGNALS[signal]
    fig = go.Figure(go.Scatter(
        x=df["timestamp"], y=df[signal], mode="lines",
        line=dict(color=meta["color"], width=2),
    ))
    fig.update_layout(
        title=f"{meta['label']} Trend", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=30, t=50, b=40),
        yaxis_title=f"{meta['label']} ({meta['unit']})", xaxis_title=None, height=380,
    )
    return fig


@callback(Output("overlay", "figure"), Input("signal", "value"))
def overlay_chart(_):
    """PM2.5 (left axis) vs SpO2 (right axis) — shows the exposure/vitals link."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["pm25"], name="PM2.5 (µg/m³)",
                             line=dict(color="#6C5CE7", width=2), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spo2"], name="SpO₂ (%)",
                             line=dict(color="#36B37E", width=2), yaxis="y2"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=20, b=40), height=360,
        yaxis=dict(title="PM2.5 (µg/m³)"),
        yaxis2=dict(title="SpO₂ (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


app.index_string = """
<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin:0; background:#0b1220; font-family:-apple-system,Segoe UI,Roboto,sans-serif; color:#cdd6e8; }
  .page { padding:18px 26px; }
  .header { font-size:18px; font-weight:700; color:#fff; }
  .subhead { font-size:12px; color:#7d8aa5; margin:4px 0 16px; }
  .kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; }
  .card { background:#111a2e; border:1px solid #1e2a44; border-radius:14px; padding:16px 18px; }
  .card-title { font-size:11px; font-weight:700; letter-spacing:1px; }
  .card-value { font-size:34px; font-weight:800; color:#fff; line-height:1.15; margin-top:4px; }
  .card-sub { font-size:11px; color:#7d8aa5; margin-top:6px; }
  .main-row { display:grid; grid-template-columns:300px 1fr; gap:16px; margin-top:18px; }
  .controls, .chart-wrap { background:#111a2e; border:1px solid #1e2a44; border-radius:14px; padding:16px; }
  .chart-wrap.full { margin-top:16px; }
  .ctl-label, .chart-title { font-size:13px; font-weight:600; color:#aeb9d4; margin-bottom:8px; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
