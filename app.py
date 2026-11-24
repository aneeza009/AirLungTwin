"""
app.py — AirLung Twin dashboard  (professional, patient-readable, interactive)
=============================================================================
Environment twin (live Open-Meteo air + CNN-1D forecast)
   + Patient twin (profile + real NHANES baseline + air-driven response)
   = a PERSONAL answer, with the AIR and the PATIENT always shown together.

New in this version:
  - Patient BREATHING FORECAST: the AQI forecast fed through the patient model
    -> predicts this patient's oxygen + a colour-coded 24-hour risk timeline.
  - Tailored DO's & DON'Ts (per patient & risk) with gently animated icons.
  - A pollutant BAR chart (varied visual, not everything in boxes).
  - Clear selected-patient label + readable dropdown.
"""
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input

import patient_twin
from data_loader import build_dataset

DATA = build_dataset()
AIR = DATA["air_history"]
FCAST_AIR = DATA["meteo_forecast"]                 # physics forecast, hourly, days ahead
NHANES_FEV1FVC = DATA["nhanes"]["fev1_fvc"]

PM25_NOW = round(float(AIR["pm25"].iloc[-1]), 0)
AQI_NOW = DATA["city_aqi"]
AQI_CAT = DATA["city_aqi_cat"]
MODE = "LIVE" if DATA["is_live"] else "RECENT"
SRC_LINE = f'{DATA["aq_source"]} · updated {DATA["aq_latest_time"]:%d %b %H:%M}'

AQI_COLORS = {"Good": "#2E933C", "Moderate": "#C9A227", "Unhealthy (Sensitive)": "#E07B39",
              "Unhealthy": "#D7263D", "Very Unhealthy": "#8E44AD", "Hazardous": "#6B0F1A"}
AQI_COLOR = AQI_COLORS.get(AQI_CAT, "#555")
PROFILE_OPTIONS = [{"label": v["name"], "value": k} for k, v in patient_twin.PROFILES.items()]

POLLUTANTS = {
    "pm25": ("Particulate matter (PM2.5)", "fine particles that reach deep into the lungs"),
    "pm10": ("Particulate matter (PM10)", "coarser dust particles you breathe in"),
    "no2": ("Nitrogen dioxide (NO₂)", "a gas mostly from vehicle traffic"),
    "o3": ("Ozone (O₃)", "an irritant gas that builds up in sunlight"),
    "so2": ("Sulphur dioxide (SO₂)", "a gas from burning fuels and industry"),
}
# rough WHO/importance ceilings (µg/m³) just to colour the bar chart
LIMITS = {"pm25": 15, "pm10": 45, "no2": 25, "o3": 100, "so2": 40}


def metric(label, value, help_text, color):
    return dbc.Card(className="metric h-100 shadow-sm", children=[
        html.Div(className="metric-bar", style={"background": color}),
        dbc.CardBody([html.Div(label, className="metric-label"),
                      html.Div(value, className="metric-value", style={"color": color}),
                      html.Div(help_text, className="metric-help")])])


def pollutant_bar():
    last = AIR.iloc[-1]
    names = [POLLUTANTS[k][0].split(" (")[0] for k in LIMITS]
    vals = [float(last[k]) for k in LIMITS]
    cols = ["#D7263D" if last[k] > LIMITS[k] * 2 else "#E07B39" if last[k] > LIMITS[k] else "#2E933C"
            for k in LIMITS]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=cols,
                           text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(template="plotly_white", height=270, margin=dict(l=10, r=20, t=10, b=30),
                      xaxis_title="µg/m³ (green = within safe limit, red = well above)",
                      font=dict(color="#2c3e50"))
    return fig


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "AirLung Twin"

app.layout = html.Div([
    html.Div(className="hero", children=dbc.Container(fluid=True, children=dbc.Row(align="center", children=[
        dbc.Col(md=5, children=[html.Div("AirLung Twin", className="hero-title"),
                                html.Div("How Lahore's live air affects one patient's breathing", className="hero-sub")]),
        dbc.Col(md=4, children=html.Div(className="airpill", children=[
            html.Div([dbc.Badge(MODE, color="danger" if MODE == "LIVE" else "secondary", className="me-2"),
                      html.Span("AIR QUALITY (LIVE)", className="airpill-cap")]),
            html.Div([html.Span(f"AQI {AQI_NOW:.0f}", className="airpill-aqi", style={"color": AQI_COLOR}),
                      html.Span(f" · {AQI_CAT}", className="airpill-cat")]),
            html.Div(SRC_LINE, className="airpill-src")])),
        dbc.Col(md=3, children=[html.Div("Viewing patient", className="hero-label"),
                                dcc.Dropdown(id="profile", options=PROFILE_OPTIONS, value=DATA["profile_key"],
                                             clearable=False, className="profile-dd"),
                                html.Div(id="sel-patient", className="sel-patient")]),
    ]))),

    dbc.Container(fluid=True, className="body", children=[
        html.Div(id="status-hero", className="status-hero shadow-sm"),

        dbc.Row(className="g-3 mt-1", children=[
            dbc.Col(md=3, children=metric("Particulate matter (PM2.5)", f"{PM25_NOW:.0f} µg/m³",
                                          "fine particles in the air now · same for everyone", AQI_COLOR)),
            dbc.Col(md=3, children=html.Div(id="m-spo2")),
            dbc.Col(md=3, children=html.Div(id="m-pef")),
            dbc.Col(md=3, children=html.Div(id="m-base")),
        ]),

        dbc.Card(className="panel shadow-sm mt-3", children=dbc.CardBody([
            html.Div("What this means for this patient", className="panel-title"),
            html.Div(id="insights")])),

        # ---------- PATIENT BREATHING FORECAST (air forecast -> patient) ----------
        dbc.Card(className="panel shadow-sm mt-3", children=dbc.CardBody([
            html.Div("This patient's next 24 hours (predicted from the air forecast)", className="panel-title"),
            html.Div(id="timeline", className="timeline"),
            html.Div(id="timeline-labels", className="tl-labels"),
            dcc.Graph(id="breath-fc", config={"displayModeBar": False}),
            html.Div("Each block is one hour, coloured by this patient's predicted risk. The line shows "
                     "their predicted blood oxygen as the forecast air changes.", className="cap")])),

        # ---------- DO's & DON'Ts (tailored, animated icons) ----------
        dbc.Row(className="g-3 mt-1", children=[
            dbc.Col(md=6, children=dbc.Card(className="panel h-100 shadow-sm dd-do", children=dbc.CardBody([
                html.Div("✅ What this patient should do", className="panel-title"), html.Div(id="do-list")]))),
            dbc.Col(md=6, children=dbc.Card(className="panel h-100 shadow-sm dd-dont", children=dbc.CardBody([
                html.Div("⛔ What to avoid", className="panel-title"), html.Div(id="dont-list")]))),
        ]),

        # ---------- air↔patient now  +  current pollutant bar ----------
        dbc.Row(className="g-3 mt-1", children=[
            dbc.Col(md=7, children=dbc.Card(className="panel h-100 shadow-sm", children=dbc.CardBody([
                html.Div("How the air is affecting this patient's breathing (recent)", className="panel-title"),
                dcc.Graph(id="fusion", config={"displayModeBar": False}),
                html.Div(id="fusion-cap", className="cap")]))),
            dbc.Col(md=5, children=dbc.Card(className="panel h-100 shadow-sm", children=dbc.CardBody([
                html.Div("Current pollutant levels", className="panel-title"),
                dcc.Graph(id="bar", figure=pollutant_bar(), config={"displayModeBar": False}),
                html.Div("The main pollutants in the air right now vs their safe limits.", className="cap")]))),
        ]),

        # ---------- comparison + explorer ----------
        dbc.Row(className="g-3 mt-1", children=[
            dbc.Col(md=6, children=dbc.Card(className="panel h-100 shadow-sm", children=dbc.CardBody([
                html.Div("A public AQI site  vs  AirLung Twin", className="panel-title"),
                html.Div(id="comparison"),
                html.Div("A public site gives one number and the same advice to the whole city. "
                         "AirLung Twin turns that same air into advice for THIS patient.", className="cap")]))),
            dbc.Col(md=6, children=dbc.Card(className="panel h-100 shadow-sm", children=dbc.CardBody([
                html.Div("Explore each pollutant", className="panel-title"),
                dcc.Dropdown(id="signal", clearable=False, value="pm25", className="mb-2",
                             options=[{"label": v[0], "value": k} for k, v in POLLUTANTS.items()],
                             style={"maxWidth": "320px"}),
                dcc.Graph(id="trend", config={"displayModeBar": False}),
                html.Div(id="trend-cap", className="cap")]))),
        ]),

        html.Div("Air is live (Open-Meteo). Patient vitals are model-predicted (simulated) on a real "
                 "NHANES lung baseline, replaced by real ESP32 readings next phase. Forecast uses a "
                 "CNN-1D deep-learning model.", className="footnote"),
    ]),
])


def _fig(fig, h=300):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=55, r=55, t=10, b=35),
                      font=dict(color="#2c3e50"))
    return fig


@callback(
    Output("sel-patient", "children"),
    Output("status-hero", "children"), Output("status-hero", "style"),
    Output("m-spo2", "children"), Output("m-pef", "children"), Output("m-base", "children"),
    Output("insights", "children"), Output("fusion", "figure"), Output("fusion-cap", "children"),
    Output("comparison", "children"),
    Output("timeline", "children"), Output("timeline-labels", "children"), Output("breath-fc", "figure"),
    Output("do-list", "children"), Output("dont-list", "children"),
    Input("profile", "value"))
def update_patient(profile_key):
    p = patient_twin.simulate(AIR, profile_key, NHANES_FEV1FVC)
    col = p["status_color"]

    sel = html.Span([html.Span("● ", style={"color": col}), f"Showing: {p['profile_name']}"])

    hero = dbc.Row(align="center", children=[
        dbc.Col(md=5, children=[html.Div(f"AIRLUNG TWIN · {p['profile_name'].upper()}", className="sh-cap"),
                                html.Div(p["status"], className="sh-status")]),
        dbc.Col(md=7, children=[html.Div("Should this patient go outside right now?", className="sh-q"),
                                html.Div(p["advice"], className="sh-advice")])])
    hero_style = {"background": col + "14", "borderLeft": f"10px solid {col}", "color": col}

    spo2 = metric("Blood oxygen (SpO₂)", f"{p['spo2_now']:.0f}%",
                  "how well-oxygenated the blood is · 95–100% is normal", col)
    pef = metric("Breathing strength (peak flow)", f"{p['pef_pct_now']:.0f}%",
                 "how forcefully they can breathe out, vs their personal best", col)
    base = metric("Lung baseline (FEV1/FVC)", f"{p['baseline_fev1_fvc']:.0f}%",
                  ("obstructed lungs (under 70%)" if p["obstruction"] else "normal lung function")
                  + " · fixed from NHANES", "#6C5CE7")

    ins = html.Ul(className="ins", children=[
        html.Li(f"The air is {AQI_CAT} (PM2.5 {PM25_NOW:.0f} µg/m³, AQI {AQI_NOW:.0f}) — same for everyone."),
        html.Li([html.B(f"For {p['profile_name']}: "),
                 f"blood oxygen is predicted at {p['spo2_now']:.0f}% and breathing strength at "
                 f"{p['pef_pct_now']:.0f}% of their best right now."]),
        html.Li(f"Their lungs start from a {p['baseline_fev1_fvc']:.0f}% baseline "
                f"({'obstructed' if p['obstruction'] else 'normal'}), so the same air hits them "
                f"{'harder' if p['obstruction'] else 'less hard'} than a healthy person."),
        html.Li([html.B("Verdict: "), html.Span(p["advice"], style={"color": col})]),
    ])

    ser = p["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser["timestamp"], y=ser["pm25"], name="Air pollution (PM2.5)",
                             line=dict(color="#E07B39", width=2)))
    fig.add_trace(go.Scatter(x=ser["timestamp"], y=ser["spo2"], name="This patient's blood oxygen",
                             yaxis="y2", line=dict(color="#2C7BE5", width=2)))
    fig.update_layout(yaxis=dict(title="PM2.5 µg/m³"), yaxis2=dict(title="SpO₂ %", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.18))
    cap = ("Orange = live particulate pollution. Blue = this patient's predicted blood oxygen. "
           "When pollution rises, this patient's oxygen tends to fall.")

    comp = dbc.Row(className="g-2", children=[
        dbc.Col(md=6, children=html.Div(className="cmp cmp-generic", children=[
            html.Div("A PUBLIC AQI SITE", className="cmp-head"),
            html.Div(f"AQI {AQI_NOW:.0f} · {AQI_CAT}", className="cmp-big"),
            html.Div("Breathing: not considered", className="cmp-row"),
            html.Div("Advice: same for the whole city", className="cmp-row")])),
        dbc.Col(md=6, children=html.Div(className="cmp cmp-twin", style={"borderColor": col}, children=[
            html.Div("AIRLUNG TWIN · THIS PATIENT", className="cmp-head", style={"color": col}),
            html.Div(p["status"], className="cmp-big", style={"color": col}),
            html.Div(f"Breathing: SpO₂ {p['spo2_now']:.0f}% · peak flow {p['pef_pct_now']:.0f}%", className="cmp-row"),
            html.Div(f"Advice: {p['advice']}", className="cmp-row", style={"fontWeight": 600, "color": col})])),
    ])

    # ---- patient forecast: timeline + breathing-forecast chart ----
    fc = patient_twin.forecast_response(FCAST_AIR, profile_key).head(24)
    if len(fc):
        blocks = [html.Div(className="tl-block", style={"background": r.color},
                           title=f"{r.timestamp:%a %H:%M} — {r.status} (PM2.5 {r.pm25:.0f})")
                  for r in fc.itertuples()]
        labels = [html.Span(f"{fc['timestamp'].iloc[i]:%H:%M}", className="tl-lab")
                  for i in range(0, len(fc), 6)]
        bf = go.Figure()
        bf.add_trace(go.Scatter(x=fc["timestamp"], y=fc["pm25"], name="PM2.5 forecast",
                                line=dict(color="#E07B39", width=2, dash="dot")))
        bf.add_trace(go.Scatter(x=fc["timestamp"], y=fc["spo2"], name="predicted blood oxygen",
                                yaxis="y2", line=dict(color="#2C7BE5", width=2)))
        bf.update_layout(yaxis=dict(title="PM2.5 µg/m³"), yaxis2=dict(title="SpO₂ %", overlaying="y", side="right"),
                         legend=dict(orientation="h", y=1.2))
        bf = _fig(bf, 260)
    else:
        blocks, labels, bf = [html.Div("forecast unavailable", className="cap")], [], _fig(go.Figure(), 260)

    do, dont = patient_twin.dos_and_donts(p["status"], p["obstruction"])
    do_list = [html.Div(className="dd-item", children=[html.Span(ic, className="dd-ic"), html.Span(tx)]) for ic, tx in do]
    dont_list = [html.Div(className="dd-item", children=[html.Span(ic, className="dd-ic"), html.Span(tx)]) for ic, tx in dont]

    return (sel, hero, hero_style, spo2, pef, base, ins, _fig(fig), cap, comp,
            blocks, labels, bf, do_list, dont_list)


@callback(Output("trend", "figure"), Output("trend-cap", "children"), Input("signal", "value"))
def trend(sig):
    name, desc = POLLUTANTS[sig]
    fig = go.Figure(go.Scatter(x=AIR["timestamp"], y=AIR[sig], mode="lines", line=dict(color="#2C7BE5", width=2)))
    fig.update_layout(yaxis_title=f"{name} (µg/m³)")
    return _fig(fig, 270), f"This is the live level of {name} in Lahore — {desc}."


app.index_string = """
<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
 body{background:#eceff5;}
 .hero{background:linear-gradient(120deg,#0f2740,#17405f);color:#fff;padding:22px 8px;margin-bottom:18px;box-shadow:0 4px 18px rgba(0,0,0,.18);}
 .hero-title{font-size:1.9rem;font-weight:800;}.hero-sub{color:#aecbe6;font-size:.95rem;}
 .hero-label{font-size:.7rem;color:#aecbe6;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
 .sel-patient{font-size:.8rem;color:#dbe8f5;margin-top:5px;font-weight:600;}
 .airpill{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:10px 14px;}
 .airpill-cap{font-size:.7rem;letter-spacing:1px;color:#cfe3f3;}.airpill-aqi{font-size:1.5rem;font-weight:800;}
 .airpill-cat{font-size:1rem;color:#e8eef5;}.airpill-src{font-size:.72rem;color:#9fb6cc;margin-top:2px;}
 .body{max-width:1250px;}
 .status-hero{border-radius:14px;padding:18px 24px;background:#fff;}
 .sh-cap{font-size:.72rem;letter-spacing:1px;font-weight:700;opacity:.8;}
 .sh-status{font-size:2.1rem;font-weight:800;line-height:1.1;}
 .sh-q{font-size:.8rem;font-weight:700;opacity:.75;text-transform:uppercase;letter-spacing:.5px;}
 .sh-advice{font-size:1.35rem;font-weight:700;}
 .metric{border:none;border-radius:12px;overflow:hidden;background:#fff;}
 .metric-bar{height:5px;width:100%;}
 .metric-label{font-size:.78rem;font-weight:700;color:#55617a;text-transform:uppercase;letter-spacing:.4px;}
 .metric-value{font-size:1.9rem;font-weight:800;line-height:1.1;margin:2px 0;}
 .metric-help{font-size:.76rem;color:#8a93a6;line-height:1.3;}
 .panel{border:none;border-radius:14px;background:#fff;}
 .panel-title{font-size:1rem;font-weight:700;color:#1f2d3d;margin-bottom:10px;}
 .cap{font-size:.8rem;color:#7a8396;font-style:italic;margin-top:6px;line-height:1.4;}
 .ins{margin:0;padding-left:1.1rem;}.ins li{margin:5px 0;color:#2c3e50;line-height:1.5;}
 .timeline{display:flex;gap:3px;margin-bottom:4px;}
 .tl-block{flex:1;height:34px;border-radius:4px;transition:transform .15s;}
 .tl-block:hover{transform:scaleY(1.25);}
 .tl-labels{display:flex;justify-content:space-between;font-size:.68rem;color:#8a93a6;margin-bottom:8px;}
 .cmp{border-radius:12px;padding:14px;height:100%;}
 .cmp-generic{background:#f1f3f7;border:1px dashed #c4ccda;}
 .cmp-twin{background:#fff;border:2px solid;}
 .cmp-head{font-size:.72rem;font-weight:700;letter-spacing:.6px;color:#8a93a6;margin-bottom:6px;}
 .cmp-big{font-size:1.5rem;font-weight:800;color:#55617a;}.cmp-row{font-size:.86rem;color:#55617a;margin-top:4px;}
 .dd-do{border-top:4px solid #2E933C;}.dd-dont{border-top:4px solid #D7263D;}
 .dd-item{display:flex;align-items:center;gap:10px;font-size:.95rem;color:#2c3e50;padding:6px 0;border-bottom:1px solid #f0f2f6;}
 .dd-ic{font-size:1.3rem;display:inline-block;animation:bob 2.4s ease-in-out infinite;}
 .dd-item:nth-child(2n) .dd-ic{animation-delay:.6s;}.dd-item:nth-child(3n) .dd-ic{animation-delay:1.2s;}
 @keyframes bob{0%,100%{transform:translateY(0);}50%{transform:translateY(-3px);}}
 .footnote{font-size:.76rem;color:#8a93a6;font-style:italic;margin:18px 0 8px;line-height:1.5;}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True)