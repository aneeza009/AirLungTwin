"""
app.py — AirLung Twin  (professional tabbed dashboard)
======================================================
Full-width, multi-tab layout (like a government AQI portal), driven by one global
patient selector. Live air (Open-Meteo, all pollutants) + CNN-1D forecast + a
patient twin that responds to the COMBINED multi-pollutant exposure.

Tabs:  Overview · 24-Hour Forecast · Air Quality · Trends · About
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input

import patient_twin
from data_loader import build_dataset

DATA = build_dataset()
AIR = DATA["air_history"]
FCAST_AIR = DATA["meteo_forecast"]
MODEL_FC = DATA["model_forecast"]
NHANES_FEV1FVC = DATA["nhanes"]["fev1_fvc"]

PM25_NOW = round(float(AIR["pm25"].iloc[-1]), 0)
AQI_NOW, AQI_CAT = DATA["city_aqi"], DATA["city_aqi_cat"]
MODE = "LIVE" if DATA["is_live"] else "RECENT"
SRC_LINE = f'{DATA["aq_source"]} · updated {DATA["aq_latest_time"]:%d %b %H:%M}'
HISTORY_DAYS = max(1, round((AIR["timestamp"].iloc[-1] - AIR["timestamp"].iloc[0]).total_seconds() / 86400))

AQI_COLORS = {"Good": "#2E933C", "Moderate": "#C9A227", "Unhealthy (Sensitive)": "#E07B39",
              "Unhealthy": "#D7263D", "Very Unhealthy": "#8E44AD", "Hazardous": "#6B0F1A"}
AQI_COLOR = AQI_COLORS.get(AQI_CAT, "#555")
PROFILE_OPTIONS = [{"label": v["name"], "value": k} for k, v in patient_twin.PROFILES.items()]
RISK_LEGEND = [("Stable", "#2E933C"), ("Watch", "#F0A202"), ("Elevated", "#F46036"), ("High risk", "#D7263D")]
POLLUTANTS = {
    "pm25": ("Particulate matter PM2.5", "fine particles that reach deep into the lungs"),
    "pm10": ("Particulate matter PM10", "coarser dust particles you breathe in"),
    "no2": ("Nitrogen dioxide NO₂", "a gas mostly from vehicle traffic"),
    "o3": ("Ozone O₃", "an irritant gas that builds up in sunlight"),
    "so2": ("Sulphur dioxide SO₂", "a gas from burning fuels and industry"),
}
AQI_SCALE = [[0.0, "#2E933C"], [0.14, "#2E933C"], [0.20, "#C9A227"], [0.30, "#E07B39"],
             [0.43, "#D7263D"], [0.60, "#8E44AD"], [1.0, "#6B0F1A"]]


# ---------- figure helpers ----------
def _fig(fig, h=320):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=60, r=55, t=15, b=40),
                      font=dict(color="#2c3e50", size=13), legend=dict(orientation="h", y=1.14))
    return fig


def fig_fusion(p):
    s = p["series"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=s["timestamp"], y=s["pm25"], name="Air pollution (PM2.5)",
                           line=dict(color="#E07B39", width=2.5)))
    f.add_trace(go.Scatter(x=s["timestamp"], y=s["spo2"], name="Patient's estimated blood oxygen",
                           yaxis="y2", line=dict(color="#2C7BE5", width=2.5)))
    f.update_layout(yaxis=dict(title="PM2.5 µg/m³"), yaxis2=dict(title="SpO₂ %", overlaying="y", side="right"))
    return _fig(f, 360)


def fig_breath(profile_key):
    fc = patient_twin.forecast_response(FCAST_AIR, profile_key).head(24)
    if not len(fc):
        return _fig(go.Figure(), 320), [], []
    f = go.Figure()
    f.add_trace(go.Scatter(x=fc["timestamp"], y=fc["exposure"], name="Predicted air exposure (all pollutants)",
                           line=dict(color="#E07B39", width=2.5, dash="dot")))
    f.add_trace(go.Scatter(x=fc["timestamp"], y=fc["spo2"], name="Predicted blood oxygen",
                           yaxis="y2", line=dict(color="#2C7BE5", width=2.5, dash="dot")))
    f.update_layout(yaxis=dict(title="exposure (PM2.5-equivalent)"),
                    yaxis2=dict(title="SpO₂ %", overlaying="y", side="right"))
    blocks = [html.Div(className="tl-block", style={"background": r.color},
                       title=f"{r.timestamp:%a %H:%M} — {r.status} (PM2.5 {r.pm25:.0f})") for r in fc.itertuples()]
    labels = [html.Span(f"{fc['timestamp'].iloc[i]:%H:%M}") for i in range(0, len(fc), 4)]
    return _fig(f, 300), blocks, labels


def fig_air_forecast():
    tail = AIR.tail(48)
    f = go.Figure()
    f.add_trace(go.Scatter(x=tail["timestamp"], y=tail["pm25"], name="Recent (measured)",
                           line=dict(color="#17405f", width=2.5)))
    if len(MODEL_FC):
        f.add_trace(go.Scatter(x=MODEL_FC["timestamp"], y=MODEL_FC["pm25"], name="Predicted (CNN-1D model)",
                               line=dict(color="#E07B39", width=2.5, dash="dot")))
    f.update_layout(yaxis_title="PM2.5 µg/m³")
    return _fig(f, 340)


def fig_drivers():
    c = patient_twin.pollutant_contributions(AIR.iloc[-1]) or [("PM2.5", 0.0)]
    c = sorted(c, key=lambda x: x[1])
    f = go.Figure(go.Bar(x=[v for _, v in c], y=[n for n, _ in c], orientation="h",
                         marker_color="#17405f", text=[f"{v:.0f}" for _, v in c], textposition="outside"))
    f.update_layout(xaxis_title="contribution to this patient's exposure")
    return _fig(f, 300)


def fig_pollutant_bar():
    last = AIR.iloc[-1]
    lim = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40}
    names = [POLLUTANTS[k][0].split(" P")[0].split(" N")[0].split(" O")[0].split(" S")[0] or k for k in lim]
    names = ["PM2.5", "PM10", "NO₂", "O₃", "SO₂"]
    vals = [float(last[k]) for k in lim]
    cols = ["#D7263D" if last[k] > lim[k] * 2 else "#E07B39" if last[k] > lim[k] else "#2E933C" for k in lim]
    f = go.Figure(go.Bar(x=names, y=vals, marker_color=cols, text=[f"{v:.0f}" for v in vals], textposition="outside"))
    f.update_layout(yaxis_title="µg/m³  (green = safe, red = well above limit)")
    return _fig(f, 320)


def fig_heatmap():
    d = AIR.copy()
    d["day"] = d["timestamp"].dt.strftime("%a %d %b")
    d["hour"] = d["timestamp"].dt.hour
    order = d.drop_duplicates("day")["day"].tolist()
    piv = d.pivot_table(index="day", columns="hour", values="us_aqi", aggfunc="mean").reindex(order)
    piv = piv.reindex(columns=range(24))
    f = go.Figure(go.Heatmap(z=piv.values, x=[f"{h:02d}" for h in piv.columns], y=piv.index,
                             colorscale=AQI_SCALE, zmin=0, zmax=350,
                             colorbar=dict(title="AQI"), hovertemplate="%{y} %{x}:00<br>AQI %{z:.0f}<extra></extra>"))
    f.update_layout(xaxis_title="hour of day", yaxis_title="")
    return _fig(f, 360)


# ---------- small UI helpers ----------
def metric(label, value, help_text, color):
    return dbc.Card(className="metric h-100 shadow-sm", children=[
        html.Div(className="metric-bar", style={"background": color}),
        dbc.CardBody([html.Div(label, className="metric-label"),
                      html.Div(value, className="metric-value", style={"color": color}),
                      html.Div(help_text, className="metric-help")])])


def panel(title, *children, sub=None):
    head = [html.Div(title, className="panel-title")]
    if sub:
        head.append(html.Div(sub, className="panel-sub"))
    return dbc.Card(className="panel shadow-sm h-100", children=dbc.CardBody(head + list(children)))


def legend():
    return html.Div(className="legend", children=[
        html.Span(className="lg-item", children=[html.Span(className="lg-sw", style={"background": c}), t])
        for t, c in RISK_LEGEND])


def cc(parts):  # colour-coded caption
    return html.Div(parts, className="cap")


ORANGE = {"color": "#E07B39", "fontWeight": 700}
BLUE = {"color": "#2C7BE5", "fontWeight": 700}


# ---------- tab builders ----------
def tab_overview(p):
    col = p["status_color"]
    status = dbc.Card(className="status-hero shadow-sm", style={"borderLeft": f"12px solid {col}", "background": col + "12"},
                      children=dbc.CardBody(dbc.Row(align="center", children=[
                          dbc.Col(md=5, children=[html.Div(f"AIRLUNG TWIN · {p['profile_name'].upper()}", className="sh-cap"),
                                                  html.Div(p["status"], className="sh-status", style={"color": col})]),
                          dbc.Col(md=7, children=[html.Div("Should this patient go outside right now?", className="sh-q"),
                                                  html.Div(p["advice"], className="sh-advice", style={"color": col})])])))
    metrics = dbc.Row(className="g-4 mt-1", children=[
        dbc.Col(lg=3, md=6, children=metric("Particulate matter PM2.5", f"{PM25_NOW:.0f} µg/m³",
                                            "fine particles in the air right now — the same for everyone", AQI_COLOR)),
        dbc.Col(lg=3, md=6, children=metric("Blood oxygen (estimate)", f"{p['spo2_now']:.0f}%",
                                            "modelled oxygen for this patient — 95–100% is normal", col)),
        dbc.Col(lg=3, md=6, children=metric("Breathing strength (peak flow)", f"{p['pef_pct_now']:.0f}%",
                                            "how forcefully they can breathe out, vs their personal best", col)),
        dbc.Col(lg=3, md=6, children=metric("Lung baseline FEV1/FVC", f"{p['baseline_fev1_fvc']:.0f}%",
                                            ("obstructed, under 70%" if p["obstruction"] else "normal lung function")
                                            + " — fixed from NHANES", "#6C5CE7")),
    ])
    means = panel("What this means for this patient", html.Ul(className="ins", children=[
        html.Li(f"The air in Lahore right now is {AQI_CAT} — PM2.5 {PM25_NOW:.0f} µg/m³, Air Quality Index {AQI_NOW:.0f}. "
                "This number is the same for everyone in the city."),
        html.Li([html.B(f"For a {p['profile_name'].lower()}: "),
                 f"the twin estimates blood oxygen around {p['spo2_now']:.0f}% and breathing strength around "
                 f"{p['pef_pct_now']:.0f}% of their personal best, using all pollutants combined "
                 f"(total exposure {p['exposure_now']:.0f})."]),
        html.Li(f"Their lungs start from a {p['baseline_fev1_fvc']:.0f}% baseline "
                f"({'obstructed' if p['obstruction'] else 'normal'}), so the same air affects them "
                f"{'more' if p['obstruction'] else 'less'} than a healthy person."),
        html.Li([html.B("Verdict: "), html.Span(p["advice"], style={"color": col, "fontWeight": 600})]),
    ]))
    comp = panel("How this is different from a public AQI website",
                 dbc.Row(className="g-3", children=[
                     dbc.Col(md=6, children=html.Div(className="cmp cmp-generic", children=[
                         html.Div("A PUBLIC AQI SITE (e.g. Punjab AQI)", className="cmp-head"),
                         html.Div(f"AQI {AQI_NOW:.0f} · {AQI_CAT}", className="cmp-big"),
                         html.Div("Breathing: not considered", className="cmp-row"),
                         html.Div("Advice: the same for the whole city", className="cmp-row")])),
                     dbc.Col(md=6, children=html.Div(className="cmp cmp-twin", style={"borderColor": col}, children=[
                         html.Div("AIRLUNG TWIN · THIS PATIENT", className="cmp-head", style={"color": col}),
                         html.Div(p["status"], className="cmp-big", style={"color": col}),
                         html.Div(f"Breathing: SpO₂ {p['spo2_now']:.0f}% · peak flow {p['pef_pct_now']:.0f}%", className="cmp-row"),
                         html.Div(f"Advice: {p['advice']}", className="cmp-row", style={"fontWeight": 600, "color": col})])),
                 ]),
                 sub="Public sites describe the air. AirLung Twin turns that same air into a decision for one person.")
    return html.Div([status, metrics,
                     dbc.Row(className="g-4 mt-1", children=[dbc.Col(lg=6, children=means), dbc.Col(lg=6, children=comp)])])


def tab_forecast(p, profile_key):
    fig, blocks, labels = fig_breath(profile_key)
    tl = panel("24-hour breathing-risk forecast",
               legend(),
               html.Div(blocks, className="timeline"),
               html.Div(labels, className="tl-labels"),
               dcc.Graph(figure=fig, config={"displayModeBar": False}),
               cc(["Each block is one hour over the next day, coloured by this patient's predicted risk. ",
                   html.Span("Orange (dotted)", style=ORANGE), " = predicted combined air exposure. ",
                   html.Span("Blue (dotted)", style=BLUE), " = predicted blood oxygen. Both come from the air forecast."]),
               sub="We forecast the air for the next 24 hours, then run it through this patient's lung profile to "
                   "predict how their breathing will change — and when it is safest to go out.")
    do, dont = patient_twin.dos_and_donts(p["status"], p["obstruction"])
    do_c = panel([html.I(className="bi bi-check2-circle me-2"), "What this patient should do"],
                 *[html.Div(className="dd-item", children=[html.I(className="bi bi-check-circle-fill dd-ok"), html.Span(t)]) for t in do])
    dont_c = panel([html.I(className="bi bi-x-circle me-2"), "What this patient should avoid"],
                   *[html.Div(className="dd-item", children=[html.I(className="bi bi-x-circle-fill dd-no"), html.Span(t)]) for t in dont])
    return html.Div([tl, dbc.Row(className="g-4 mt-1", children=[dbc.Col(md=6, children=do_c), dbc.Col(md=6, children=dont_c)])])


def tab_air():
    explore = panel("Explore each pollutant",
                    dcc.Dropdown(id="signal", clearable=False, value="pm25", className="mb-2", style={"maxWidth": "340px"},
                                 options=[{"label": v[0], "value": k} for k, v in POLLUTANTS.items()]),
                    dcc.Graph(id="trend", config={"displayModeBar": False}),
                    html.Div(id="trend-cap", className="cap"),
                    sub="Lahore's air is a mix of several pollutants. Pick one to see its live level over the last week.")
    return html.Div([
        dbc.Row(className="g-4", children=[
            dbc.Col(lg=6, children=panel("Current pollutant levels", dcc.Graph(figure=fig_pollutant_bar(), config={"displayModeBar": False}),
                                         cc("The main pollutants in the air right now, compared with their safe limits. "
                                            "Red bars are well above the limit."),
                                         sub="What you are breathing at this moment.")),
            dbc.Col(lg=6, children=panel("What's driving this patient's risk now", dcc.Graph(figure=fig_drivers(), config={"displayModeBar": False}),
                                         cc("Every pollutant adds to this patient's exposure — not PM2.5 alone. This "
                                            "updates as the live air changes, so a different pollutant can lead at different times."),
                                         sub="How each pollutant contributes to the combined exposure.")),
        ]),
        dbc.Row(className="g-4 mt-1", children=[
            dbc.Col(lg=6, children=panel("Air quality forecast — next hours",
                                         dcc.Graph(figure=fig_air_forecast(), config={"displayModeBar": False}),
                                         cc([html.Span("Solid", style={"color": "#17405f", "fontWeight": 700}),
                                             " = recent measured PM2.5. ",
                                             html.Span("Orange (dotted)", style=ORANGE),
                                             " = predicted next hours from our CNN-1D deep-learning model."]),
                                         sub="Our machine-learning model forecasts how the air will change.")),
            dbc.Col(lg=6, children=explore),
        ]),
    ])


def tab_trends(p, profile_key):
    return html.Div([
        panel(f"How the air affected this patient — last {HISTORY_DAYS} days",
              dcc.Graph(figure=fig_fusion(p), config={"displayModeBar": False}),
              cc([html.Span("Orange", style=ORANGE), " = measured particulate pollution (PM2.5). ",
                  html.Span("Blue", style=BLUE), " = this patient's estimated blood oxygen. "
                  "When pollution rises, their oxygen tends to fall."]),
              sub="A week of history showing the link between the air and this patient's breathing."),
        panel("Air quality calendar — last 7 days",
              dcc.Graph(figure=fig_heatmap(), config={"displayModeBar": False}),
              cc("Each square is one hour, coloured by the Air Quality Index (green = good, red/purple = dangerous). "
                 "This shows when the air is usually worst — often late evening and early morning."),
              sub="The same hour-by-hour view a public AQI portal shows, for context.")
    ])


def tab_about():
    rows = [
        ("Live air (PM2.5, PM10, NO₂, O₃, SO₂, AQI)", "Open-Meteo Air Quality API (free, live, Lahore)", "Real, live"),
        ("Air forecast", "Open-Meteo forecast + a CNN-1D deep-learning model for PM2.5", "Real + ML"),
        ("Lung baseline (FEV1/FVC)", "NHANES spirometry dataset", "Real data"),
        ("Patient oxygen, peak flow, risk, advice", "Our patient response model (computed from the air + profile)", "Modelled"),
        ("Patient profiles (healthy / mild / moderate)", "Defined by us", "Our design"),
    ]
    table = dbc.Table([html.Thead(html.Tr([html.Th("What you see"), html.Th("Where it comes from"), html.Th("Type")])),
                       html.Tbody([html.Tr([html.Td(a), html.Td(b), html.Td(c)]) for a, b, c in rows])],
                      bordered=True, hover=True, className="about-table")
    return html.Div([
        panel("What is AirLung Twin?",
              html.P("AirLung Twin is a digital twin of a respiratory patient. It takes Lahore's live air quality "
                     "and shows how that specific air affects one particular patient's breathing right now — and what "
                     "they should do about it. A public air-quality website tells the whole city one number; our system "
                     "takes that same air, runs it through a chosen patient's lung profile, and gives that one person a "
                     "personal status and advice.", className="about-p"),
              html.P("Only the air and the lung baseline are real datasets. Everything about the patient's breathing is "
                     "calculated by our model — which is exactly what a digital twin does: it simulates the patient. When "
                     "real wearable sensors (ESP32) are added later, real readings replace the modelled ones and the same "
                     "model interprets them.", className="about-p")),
        panel("Where every number comes from", table),
        panel("What we introduced (our contribution)",
              html.Ul(className="ins", children=[
                  html.Li("A patient digital twin: a profile + a real lung baseline + an air-driven response model."),
                  html.Li("Turning all pollutants combined — not just PM2.5 — into one person's oxygen, breathing strength and risk."),
                  html.Li("Personalised advice: the same air gives different answers to a healthy person, a mild asthmatic and a moderate asthmatic."),
                  html.Li("Forecasting the patient, not just the air — predicting how this patient will breathe in the hours ahead."),
              ])),
    ])


def footer():
    return html.Footer(className="site-footer", children=dbc.Container(fluid=True, children=dbc.Row([
        dbc.Col(md=4, children=[html.Div("AirLung Twin", className="ft-title"),
                                html.Div("An IoMT-enabled cognitive digital twin for respiratory health. "
                                         "Final Year Project, BS Computer Science, CUI Lahore.", className="ft-text")]),
        dbc.Col(md=4, children=[html.Div("Data sources", className="ft-head"),
                                html.Div("Air quality: Open-Meteo (live)", className="ft-text"),
                                html.Div("Lung baseline: NHANES spirometry", className="ft-text"),
                                html.Div("Forecast model: CNN-1D (deep learning)", className="ft-text")]),
        dbc.Col(md=4, children=[html.Div("Disclaimer", className="ft-head"),
                                html.Div("Patient vitals are model estimates for demonstration, not medical advice. "
                                         "Real sensor integration is planned for the next phase.", className="ft-text")]),
    ])))


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "AirLung Twin"

app.layout = html.Div([
    html.Div(className="hero", children=dbc.Container(fluid=True, children=dbc.Row(align="center", className="g-3", children=[
        dbc.Col(lg=5, md=12, children=[html.Div("AirLung Twin", className="hero-title"),
                                       html.Div("A respiratory digital twin — how Lahore's live air affects one patient's breathing",
                                                className="hero-sub")]),
        dbc.Col(lg=4, md=7, children=html.Div(className="airpill", children=[
            html.Div([dbc.Badge(MODE, color="danger" if MODE == "LIVE" else "secondary", className="me-2"),
                      html.Span("AIR QUALITY, LIVE", className="airpill-cap")]),
            html.Div([html.Span(f"AQI {AQI_NOW:.0f}", className="airpill-aqi", style={"color": AQI_COLOR}),
                      html.Span(f" · {AQI_CAT}", className="airpill-cat")]),
            html.Div(SRC_LINE, className="airpill-src")])),
        dbc.Col(lg=3, md=5, children=[html.Div("Viewing patient", className="hero-label"),
                                      html.Div(className="dd-wrap", children=[
                                          html.Div(id="sel-patient", className="sel-patient"),
                                          dcc.Dropdown(id="profile", options=PROFILE_OPTIONS, value=DATA["profile_key"],
                                                       clearable=False, searchable=False, className="profile-dd")])]),
    ]))),

    dbc.Container(fluid=True, className="navwrap", children=dcc.Tabs(id="tabs", value="overview", className="tabs", children=[
        dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab-sel"),
        dcc.Tab(label="24-Hour Forecast", value="forecast", className="tab", selected_className="tab-sel"),
        dcc.Tab(label="Air Quality", value="air", className="tab", selected_className="tab-sel"),
        dcc.Tab(label="Trends", value="trends", className="tab", selected_className="tab-sel"),
        dcc.Tab(label="About", value="about", className="tab", selected_className="tab-sel"),
    ])),

    dbc.Container(fluid=True, className="body", children=html.Div(id="tab-content")),
    footer(),
])


@callback(Output("sel-patient", "children"), Input("profile", "value"))
def sel(profile_key):
    return patient_twin.PROFILES[profile_key]["name"]


@callback(Output("tab-content", "children"), Input("tabs", "value"), Input("profile", "value"))
def render(tab, profile_key):
    p = patient_twin.simulate(AIR, profile_key, NHANES_FEV1FVC)
    if tab == "overview": return tab_overview(p)
    if tab == "forecast": return tab_forecast(p, profile_key)
    if tab == "air": return tab_air()
    if tab == "trends": return tab_trends(p, profile_key)
    return tab_about()


@callback(Output("trend", "figure"), Output("trend-cap", "children"), Input("signal", "value"))
def trend(sig):
    name, desc = POLLUTANTS[sig]
    f = go.Figure(go.Scatter(x=AIR["timestamp"], y=AIR[sig], mode="lines", line=dict(color="#2C7BE5", width=2.5)))
    f.update_layout(yaxis_title=f"{name} (µg/m³)")
    return _fig(f, 300), f"This is the live level of {name} in Lahore over the last {HISTORY_DAYS} days — {desc}."


app.index_string = """
<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
 body{background:#eef1f6;font-size:15px;color:#2c3e50;}
 .hero{background:linear-gradient(120deg,#0f2740,#17405f);color:#fff;padding:24px 28px;box-shadow:0 4px 18px rgba(0,0,0,.18);}
 .hero-title{font-size:2.1rem;font-weight:800;}.hero-sub{color:#b7d0e6;font-size:1rem;}
 .hero-label{font-size:.75rem;color:#b7d0e6;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
 .airpill{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:12px 16px;}
 .airpill-cap{font-size:.74rem;letter-spacing:1px;color:#cfe3f3;}.airpill-aqi{font-size:1.7rem;font-weight:800;}
 .airpill-cat{font-size:1.05rem;color:#e8eef5;}.airpill-src{font-size:.76rem;color:#9fb6cc;margin-top:2px;}
 .dd-wrap{position:relative;}
 .profile-dd .Select-control{background:#fff;border-radius:8px;min-height:40px;}
 .sel-patient{position:absolute;left:14px;top:0;height:40px;line-height:40px;color:#1f2d3d;font-weight:600;
   font-size:1rem;pointer-events:none;z-index:5;white-space:nowrap;max-width:calc(100% - 44px);overflow:hidden;text-overflow:ellipsis;}
 /* nav tabs */
 .navwrap{background:#fff;border-bottom:1px solid #dfe4ec;box-shadow:0 2px 6px rgba(0,0,0,.05);padding:0 14px;}
 .tabs{border:none !important;} .tab{border:none !important;background:transparent !important;color:#55617a !important;
   font-weight:600 !important;font-size:1rem !important;padding:16px 22px !important;}
 .tab-sel{color:#17405f !important;border-bottom:3px solid #17405f !important;background:#fff !important;}
 .body{padding:22px 28px 30px;}
 .status-hero{border-radius:14px;}
 .sh-cap{font-size:.78rem;letter-spacing:1px;font-weight:700;color:#55617a;}
 .sh-status{font-size:2.4rem;font-weight:800;line-height:1.1;}
 .sh-q{font-size:.85rem;font-weight:700;color:#55617a;text-transform:uppercase;letter-spacing:.5px;}
 .sh-advice{font-size:1.5rem;font-weight:700;}
 .metric{border:none;border-radius:12px;overflow:hidden;background:#fff;min-height:150px;}
 .metric-bar{height:6px;width:100%;}
 .metric-label{font-size:.82rem;font-weight:700;color:#55617a;text-transform:uppercase;letter-spacing:.4px;}
 .metric-value{font-size:2.1rem;font-weight:800;line-height:1.1;margin:4px 0;}
 .metric-help{font-size:.88rem;color:#6b7589;line-height:1.4;}
 .panel{border:none;border-radius:14px;background:#fff;}
 .panel-title{font-size:1.15rem;font-weight:700;color:#1f2d3d;}
 .panel-sub{font-size:.9rem;color:#6b7589;margin:2px 0 12px;}
 .cap{font-size:.9rem;color:#6b7589;margin-top:8px;line-height:1.5;}
 .ins{margin:0;padding-left:1.2rem;}.ins li{margin:7px 0;color:#2c3e50;line-height:1.55;font-size:1rem;}
 .legend{display:flex;gap:18px;margin-bottom:10px;flex-wrap:wrap;}
 .lg-item{display:flex;align-items:center;gap:7px;font-size:.85rem;color:#55617a;}
 .lg-sw{width:16px;height:16px;border-radius:4px;display:inline-block;}
 .timeline{display:flex;gap:3px;margin-bottom:4px;}
 .tl-block{flex:1;height:40px;border-radius:4px;transition:transform .15s;}.tl-block:hover{transform:scaleY(1.2);}
 .tl-labels{display:flex;justify-content:space-between;font-size:.72rem;color:#8a93a6;margin-bottom:10px;}
 .cmp{border-radius:12px;padding:16px;height:100%;}
 .cmp-generic{background:#f1f3f7;border:1px dashed #c4ccda;}.cmp-twin{background:#fff;border:2px solid;}
 .cmp-head{font-size:.76rem;font-weight:700;letter-spacing:.6px;color:#8a93a6;margin-bottom:6px;}
 .cmp-big{font-size:1.6rem;font-weight:800;color:#55617a;}.cmp-row{font-size:.92rem;color:#55617a;margin-top:5px;}
 .dd-item{display:flex;align-items:center;gap:11px;font-size:1rem;color:#2c3e50;padding:9px 0;border-bottom:1px solid #f0f2f6;}
 .dd-ok{color:#2E933C;font-size:1.3rem;}.dd-no{color:#D7263D;font-size:1.3rem;}
 .about-p{font-size:1rem;line-height:1.65;color:#2c3e50;}
 .about-table{font-size:.95rem;}.about-table th{background:#f1f3f7;color:#1f2d3d;}
 .g-4>*{margin-top:1.25rem;}
 .site-footer{background:#0f2740;color:#b7d0e6;padding:28px;margin-top:30px;}
 .ft-title{font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:6px;}
 .ft-head{font-size:.8rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#8fb3d0;margin-bottom:8px;}
 .ft-text{font-size:.9rem;line-height:1.6;margin-bottom:4px;}
 @media(max-width:900px){.hero-title{font-size:1.6rem;}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True)