"""
app.py — AirLung Twin dashboard (clean, light, layperson-readable)
==================================================================
Environment twin (live Open-Meteo air + Khazina's CNN-1D forecast)
      +  Patient twin (profile + real NHANES baseline + air-driven response)
      =  a personal answer: "given THIS air, should THIS patient go outside?"

Built with Dash Bootstrap Components on a light theme for a professional look.
Pick a patient profile (healthy / mild / moderate asthma) to see how the SAME
Lahore air affects DIFFERENT patients — that contrast is the project's novelty.

Run:  pip install -r requirements.txt   then   python app.py  -> http://127.0.0.1:8050
"""
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input

import patient_twin
from data_loader import build_dataset

DATA = build_dataset()                     # one call: live air + forecast + baseline
AIR = DATA["air_history"]
NHANES_FEV1FVC = DATA["nhanes"]["fev1_fvc"]

# --- air "now" (profile-independent) ---
PM25_NOW = round(float(AIR["pm25"].iloc[-1]), 0)
AQI_NOW = DATA["city_aqi"]
AQI_CAT = DATA["city_aqi_cat"]
MODE = "LIVE" if DATA["is_live"] else "RECENT"
SRC_LINE = f'{DATA["aq_source"]} · latest {DATA["aq_latest_time"]:%d %b %H:%M}'

AQI_COLORS = {"Good": "#2E933C", "Moderate": "#F0A202",
              "Unhealthy (Sensitive)": "#F46036", "Unhealthy": "#D7263D",
              "Very Unhealthy": "#8E44AD", "Hazardous": "#6B0F1A"}
AQI_COLOR = AQI_COLORS.get(AQI_CAT, "#555")

PROFILE_OPTIONS = [{"label": v["name"], "value": k} for k, v in patient_twin.PROFILES.items()]


def stat(title, value, caption, color="#2c3e50"):
    return dbc.Card(dbc.CardBody([
        html.Div(title, className="text-muted small text-uppercase fw-bold"),
        html.Div(value, style={"fontSize": "2rem", "fontWeight": "800", "color": color}),
        html.Div(caption, className="text-muted small"),
    ]), className="shadow-sm h-100")


def caption(txt):
    return html.Div("What this shows: " + txt, className="text-muted small fst-italic mt-1")


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "AirLung Twin"

app.layout = dbc.Container(fluid=True, className="py-3", children=[
    # header
    dbc.Row(align="center", className="mb-3", children=[
        dbc.Col([html.H3("AirLung Twin", className="fw-bold mb-0"),
                 html.Div("A respiratory digital twin — how live air affects one patient",
                          className="text-muted small")], md=6),
        dbc.Col(md=3, children=dbc.Card(dbc.CardBody([
            html.Div("AIR DATA", className="small text-muted fw-bold"),
            dbc.Badge(MODE, color="success" if MODE == "LIVE" else "secondary", className="me-1"),
            html.Div(SRC_LINE, className="small text-muted mt-1")]), className="shadow-sm")),
        dbc.Col(md=3, children=[
            html.Div("Patient profile", className="small text-muted fw-bold"),
            dcc.Dropdown(id="profile", options=PROFILE_OPTIONS,
                         value=DATA["profile_key"], clearable=False)]),
    ]),

    # BIG answer
    dbc.Row(className="mb-3", children=dbc.Col(dbc.Card(dbc.CardBody([
        dbc.Row(align="center", children=[
            dbc.Col(md=5, children=[
                html.Div("CURRENT STATUS FOR THIS PATIENT", className="small text-muted fw-bold"),
                html.Div(id="status-text", style={"fontSize": "2rem", "fontWeight": "800"})]),
            dbc.Col(md=7, children=[
                html.Div("Should this patient go outside right now?",
                         className="small text-muted fw-bold"),
                html.Div(id="advice-text", style={"fontSize": "1.25rem", "fontWeight": "700"})]),
        ]),
    ]), id="status-card", className="shadow-sm"))),

    # KPI row
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(md=3, children=stat("PM2.5 now", f"{PM25_NOW:.0f} µg/m³",
                                    f"Air Quality Index {AQI_NOW:.0f} · {AQI_CAT}", AQI_COLOR)),
        dbc.Col(md=3, children=html.Div(id="kpi-spo2")),
        dbc.Col(md=3, children=html.Div(id="kpi-pef")),
        dbc.Col(md=3, children=html.Div(id="kpi-baseline")),
    ]),

    # insights
    dbc.Row(className="mb-3", children=dbc.Col(dbc.Card(dbc.CardBody([
        html.H6("What this means", className="fw-bold"),
        html.Div(id="insights")]), className="shadow-sm"))),

    # charts
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(md=6, children=dbc.Card(dbc.CardBody([
            html.H6("Air vs this patient's predicted breathing", className="fw-bold"),
            dcc.Graph(id="fusion", config={"displayModeBar": False}),
            caption("as PM2.5 (orange) rises, the twin predicts this patient's "
                    "oxygen / SpO₂ (blue) drops. The link is defined by the patient model.")]),
            className="shadow-sm")),
        dbc.Col(md=6, children=dbc.Card(dbc.CardBody([
            html.H6("Air quality forecast (next hours)", className="fw-bold"),
            dcc.Graph(id="forecast", config={"displayModeBar": False}),
            caption("recent live PM2.5 (solid) and the predicted next hours "
                    f"({DATA['model_metrics'].get('model','forecast')}).")]),
            className="shadow-sm")),
    ]),

    # comparison + explore
    dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(md=6, children=dbc.Card(dbc.CardBody([
            html.H6("Generic city AQI  vs  this patient", className="fw-bold"),
            html.Div(id="comparison"),
            caption("a public AQI site gives one city number for everyone; the twin "
                    "turns the same air into advice for THIS patient.")]), className="shadow-sm")),
        dbc.Col(md=6, children=dbc.Card(dbc.CardBody([
            html.H6("Explore any pollutant", className="fw-bold"),
            dcc.Dropdown(id="signal", clearable=False, value="pm25", options=[
                {"label": "PM2.5", "value": "pm25"}, {"label": "PM10", "value": "pm10"},
                {"label": "NO₂", "value": "no2"}, {"label": "Ozone", "value": "o3"},
                {"label": "SO₂", "value": "so2"}], style={"maxWidth": "260px"}),
            dcc.Graph(id="trend", config={"displayModeBar": False}),
            caption("the live Lahore trend for the chosen pollutant.")]), className="shadow-sm")),
    ]),

    html.Div("Patient vitals are MODEL-PREDICTED (simulated), grounded in a real NHANES "
             "lung baseline; they are replaced by real ESP32 sensor readings in the next "
             "phase. Air is live (Open-Meteo). Forecast replicates Khazina Naveed's CNN-1D.",
             className="text-muted small fst-italic"),
])


# ---------- patient-dependent callback ----------
@callback(
    Output("status-text", "children"), Output("status-text", "style"),
    Output("advice-text", "children"), Output("advice-text", "style"),
    Output("status-card", "style"),
    Output("kpi-spo2", "children"), Output("kpi-pef", "children"),
    Output("kpi-baseline", "children"), Output("insights", "children"),
    Output("fusion", "figure"), Output("comparison", "children"),
    Input("profile", "value"))
def update_patient(profile_key):
    p = patient_twin.simulate(AIR, profile_key, NHANES_FEV1FVC)
    col = p["status_color"]
    s_style = {"fontSize": "2rem", "fontWeight": "800", "color": col}
    a_style = {"fontSize": "1.25rem", "fontWeight": "700", "color": col}
    card_style = {"borderLeft": f"8px solid {col}"}

    spo2 = stat("Predicted SpO₂", f"{p['spo2_now']:.1f}%", "oxygen for this patient (modelled)", col)
    pef = stat("Predicted peak flow", f"{p['pef_pct_now']:.0f}%", "of this patient's personal best", col)
    base = stat("FEV1/FVC baseline", f"{p['baseline_fev1_fvc']:.0f}%",
                ("obstructed (<70%)" if p["obstruction"] else "normal range") + " · NHANES anchor",
                "#6C5CE7")

    ins = [
        html.Div(f"• Air is {AQI_CAT} (PM2.5 {PM25_NOW:.0f} µg/m³, AQI {AQI_NOW:.0f}).", style={"color": AQI_COLOR}),
        html.Div(f"• For {p['profile_name']}, the twin predicts SpO₂ ≈ {p['spo2_now']:.1f}% and "
                 f"peak flow ≈ {p['pef_pct_now']:.0f}% of personal best right now.", style={"color": col}),
        html.Div(f"• Lung baseline FEV1/FVC = {p['baseline_fev1_fvc']:.0f}% "
                 f"({'obstructed' if p['obstruction'] else 'normal'}), the anchor the response is judged against.",
                 style={"color": "#6C5CE7"}),
        html.Div(f"• Verdict: {p['advice']}", style={"color": col, "fontWeight": "600"}),
    ]

    ser = p["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser["timestamp"], y=ser["pm25"], name="PM2.5 (µg/m³)",
                             line=dict(color="#F46036", width=2)))
    fig.add_trace(go.Scatter(x=ser["timestamp"], y=ser["spo2"], name="SpO₂ (%)", yaxis="y2",
                             line=dict(color="#2C7BE5", width=2)))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=50, r=50, t=10, b=30),
                      yaxis=dict(title="PM2.5"), yaxis2=dict(title="SpO₂ %", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.15))

    comp = dbc.Table([html.Tbody([
        html.Tr([html.Td("Air number"), html.Td(f"City AQI {AQI_NOW:.0f} · {AQI_CAT}"),
                 html.Td(f"PM2.5 {PM25_NOW:.0f} for this patient")]),
        html.Tr([html.Td("Breathing"), html.Td("not considered"),
                 html.Td(f"SpO₂ {p['spo2_now']:.0f}% · peak flow {p['pef_pct_now']:.0f}%")]),
        html.Tr([html.Td("Advice"), html.Td("same for everyone"),
                 html.Td(p["advice"], style={"color": col, "fontWeight": 600})]),
    ])], bordered=True, size="sm", className="mb-0")
    return (p["status"], s_style, p["advice"], a_style, card_style,
            spo2, pef, base, ins, fig, comp)


# ---------- forecast chart (air only) ----------
@callback(Output("forecast", "figure"), Input("profile", "value"))
def forecast_chart(_):
    hist_tail = AIR.tail(48)
    mf = DATA["model_forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["timestamp"], y=hist_tail["pm25"],
                             name="recent (live)", line=dict(color="#2c3e50", width=2)))
    if len(mf):
        fig.add_trace(go.Scatter(x=mf["timestamp"], y=mf["pm25"], name="forecast",
                                 line=dict(color="#F0A202", width=2, dash="dash")))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=50, r=20, t=10, b=30),
                      yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h", y=1.15))
    return fig


# ---------- pollutant explorer (air only) ----------
@callback(Output("trend", "figure"), Input("signal", "value"))
def trend(sig):
    labels = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "Ozone", "so2": "SO₂"}
    fig = go.Figure(go.Scatter(x=AIR["timestamp"], y=AIR[sig], mode="lines",
                               line=dict(color="#2C7BE5", width=2)))
    fig.update_layout(template="plotly_white", height=280, margin=dict(l=50, r=20, t=10, b=30),
                      yaxis_title=f"{labels[sig]} (µg/m³)")
    return fig


if __name__ == "__main__":
    app.run(debug=True)