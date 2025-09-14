# app.py
# ————————————————————————————————————————————————————————————————————
# AI Dashboard – Prediction of Water Network Failures
# All labels and UI texts in English
# ————————————————————————————————————————————————————————————————————

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm

import os, time, base64
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ============ PAGE SETTINGS ============
st.set_page_config(
    page_title="AI – Prediction of Water Network Failures",
    page_icon="",
    layout="wide",
)

# ============ LOGO ============
LOGO_PATH = "assets/Logo.png"

def insert_logo_fixed(logo_path=LOGO_PATH, link="#", height=48):
    if not os.path.exists(logo_path):
        return
    b64 = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(f"""
    <a href="{link}">
      <img src="data:image/png;base64,{b64}" class="logo-fixed">
    </a>
    <style>
      .logo-fixed {{
        position: fixed;
        top: 12px; left: 14px;
        height: {height}px;
        z-index: 1000;
      }}
      @media (max-width: 768px) {{
        .logo-fixed {{ height: 40px; }}
      }}
    </style>
    """, unsafe_allow_html=True)

insert_logo_fixed()

# ============ LIGHT THEME STYLING ============
st.markdown("""
<style>
.stApp { background:#ffffff !important; color:#111111; }
.block-container { color:#111111; }

/* KPI cards */
.kpi-card {border-radius:16px;padding:18px 22px;box-shadow:0 8px 20px rgba(0,0,0,.08);text-align:center;font-weight:600;}
.kpi-red{background:#ff4d4d;color:#fff;}
.kpi-yellow{background:#ffcc00;color:#111;}
.kpi-green{background:#22c55e;color:#fff;}
.caption{color:#6b7280;font-size:.9rem;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(255,77,77,.45);}70%{box-shadow:0 0 0 18px rgba(255,77,77,0);}100%{box-shadow:0 0 0 0 rgba(255,77,77,0);}}
.pulse{animation:pulse 2s infinite;}

/* Download buttons */
div[data-testid="stDownloadButton"] > button {
  width:100%; border:none; border-radius:12px; padding:14px 18px;
  background:#0B7CFF; color:#fff; font-weight:600; box-shadow:0 8px 18px rgba(11,124,255,.18);
}
div[data-testid="stDownloadButton"] > button:hover { background:#095fcc; }
</style>
""", unsafe_allow_html=True)

# ============ HELPERS ============
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def add_risk_labels(df: pd.DataFrame, thresholds=(40, 70)) -> pd.DataFrame:
    med_thr, high_thr = thresholds
    def lbl(p):
        if p >= high_thr: return "🔴 High Risk"
        if p >= med_thr: return "🟡 Medium Risk"
        return "🟢 Low Risk"
    df["Assessment"] = df["Risk (%)"].apply(lbl)
    return df

@st.cache_data
def preprocess(df: pd.DataFrame):
    copy = df.copy()
    enc_material = LabelEncoder()
    enc_soil     = LabelEncoder()
    copy["Pipe_Material"]    = enc_material.fit_transform(copy["Pipe_Material"])
    copy["Soil_Corrosivity"] = enc_soil.fit_transform(copy["Soil_Corrosivity"])
    X = copy.drop(columns=["Pipe_ID", "Leak_Class"])
    y = copy["Leak_Class"].astype(int)
    return copy, X, y, enc_material, enc_soil

@st.cache_resource
def train_rf(X, y, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "ConfusionMatrix": confusion_matrix(y_te, y_pred),
        "Report": classification_report(y_te, y_pred, output_dict=True),
    }
    return model, metrics

# ============ SIDEBAR ============
st.sidebar.header("Analysis Settings")
med_thr  = st.sidebar.slider("Medium risk threshold (%)", 10, 60, 40, step=5)
high_thr = st.sidebar.slider("High risk threshold (%)", 50, 95, 70, step=5)
st.sidebar.caption("Thresholds affect dashboard colors and the PDF report.")

# ============ DATA → MODEL → SCORING ============
DATA_PATH = "data/raw/water_network_leak_dataset.xlsx"
if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

raw_data = load_data(DATA_PATH)
processed, X, y, enc_mat, enc_soil = preprocess(raw_data)
model_rf, metrics = train_rf(X, y)

prob = (model_rf.predict_proba(X)[:, 1] * 100).round(1)
scored = raw_data.copy()
scored["Risk (%)"] = prob
scored = add_risk_labels(scored, (med_thr, high_thr))

# ============ KPI ============
count_high = (scored["Assessment"] == "🔴 High Risk").sum()
count_med  = (scored["Assessment"] == "🟡 Medium Risk").sum()
count_low  = (scored["Assessment"] == "🟢 Low Risk").sum()
count_total= len(scored)

# ============ HEADER + KPI ============
st.title("🤖 AI – Prediction of Water Network Failures")
st.caption("Clear risk analysis for municipalities – investment priorities, 'what-if' sandbox, and PDF report.")

c1, c2, c3, _ = st.columns([1,1,1,0.3])
cls_red = "kpi-card kpi-red pulse" if count_high > 0 else "kpi-card kpi-red"
with c1: st.markdown(f'<div class="{cls_red}">🔴 High Risk<br><span style="font-size:34px">{count_high}</span></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi-card kpi-yellow">🟡 Medium Risk<br><span style="font-size:34px">{count_med}</span></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi-card kpi-green">🟢 Low Risk<br><span style="font-size:34px">{count_low}</span></div>', unsafe_allow_html=True)
st.markdown(f"<div class='caption'>Total items: <b>{count_total}</b>. Thresholds: medium ≥ {med_thr}%, high ≥ {high_thr}%.</div>", unsafe_allow_html=True)

st.divider()

# ============ FEATURE IMPORTANCE ============
feature_names = {
    "Pressure_PSI":     "Pressure (PSI)",
    "Flow_GPM":         "Flow (GPM)",
    "Velocity_FPS":     "Velocity (ft/s)",
    "Temperature_F":    "Temperature (°F)",
    "Pipe_Age_Years":   "Pipe Age (years)",
    "Pipe_Material":    "Pipe Material",
    "Soil_Corrosivity": "Soil Corrosivity",
}

st.subheader("Key Risk Drivers")

fi_df = pd.DataFrame({
    "Feature": list(X.columns),
    "Importance": model_rf.feature_importances_,
})
fi_df["Feature_EN"] = fi_df["Feature"].map(feature_names).fillna(fi_df["Feature"])
fi_df = fi_df.sort_values("Importance", ascending=True)

fig_imp = px.bar(
    fi_df, x="Importance", y="Feature_EN",
    orientation="h",
    title="Most important risk factors (Random Forest model)"
)
st.plotly_chart(fig_imp, use_container_width=True, theme=None)

st.divider()

# ============ WHAT-IF SANDBOX ============
st.subheader("🧪 'What-if' Simulation")

defaults = scored.median(numeric_only=True).to_dict()
c1, c2, c3, c4 = st.columns(4)
with c1: pressure_sim = st.slider("Pressure (PSI)", 0.0, 120.0, float(defaults.get("Pressure_PSI", 60.0)), 0.1)
with c2: flow_sim     = st.slider("Flow (GPM)",   0.0, 260.0, float(defaults.get("Flow_GPM", 120.0)), 0.1)
with c3: age_sim      = st.slider("Pipe Age (years)", 0,   60,    int(defaults.get("Pipe_Age_Years", 10)), 1)
with c4: vel_sim      = st.slider("Velocity (ft/s)",  0.0, 12.0, float(defaults.get("Velocity_FPS", 4.0)), 0.1)

c5, c6 = st.columns(2)
with c5: mat_sim  = st.selectbox("Pipe Material", sorted(scored["Pipe_Material"].unique()))
with c6: soil_sim = st.selectbox("Soil Corrosivity", sorted(scored["Soil_Corrosivity"].unique()))

row_sim = pd.DataFrame([{
    "Pressure_PSI": pressure_sim, "Flow_GPM": flow_sim, "Velocity_FPS": vel_sim,
    "Temperature_F": defaults.get("Temperature_F", 60.0),
    "Pipe_Age_Years": age_sim, "Pipe_Material": mat_sim, "Soil_Corrosivity": soil_sim
}])

row_sim_enc = row_sim.copy()
row_sim_enc["Pipe_Material"] = enc_mat.transform(row_sim_enc["Pipe_Material"])
row_sim_enc["Soil_Corrosivity"] = enc_soil.transform(row_sim_enc["Soil_Corrosivity"])
risk_sim = model_rf.predict_proba(row_sim_enc)[:, 1].item() * 100

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=risk_sim, number={'suffix': "%"}, delta={'reference': 50},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'thickness': 0.25},
        'steps': [
            {'range': [0, med_thr], 'color': 'rgba(46,184,46,0.35)'},
            {'range': [med_thr, high_thr], 'color': 'rgba(255,204,0,0.35)'},
            {'range': [high_thr, 100], 'color': 'rgba(255,77,77,0.45)'}
        ],
        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': high_thr}
    },
    title={'text': "Predicted failure risk – 'what-if' scenario"}
))
st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

st.divider()

# ============ TOP TABLE ============
st.subheader("TOP – Highest Risk Elements")

c1, c2, c3 = st.columns([1,1,1])
with c1:
    n_show = st.slider("Number of rows to display", min_value=5, max_value=100, value=10, step=5)
with c2:
    sort_col = st.selectbox("Sort by", ["Risk (%)","Pipe_Age_Years","Pressure_PSI","Flow_GPM"], index=0)
with c3:
    asc = st.toggle("Ascending", value=False)

top_view = scored.sort_values(sort_col, ascending=asc).head(n_show)

df_top = top_view.rename(columns={
    "Pipe_ID":"Pipe ID", "Pressure_PSI":"Pressure (PSI)", "Flow_GPM":"Flow (GPM)",
    "Velocity_FPS":"Velocity (ft/s)", "Temperature_F":"Temperature (°F)",
    "Pipe_Age_Years":"Pipe Age (years)", "Pipe_Material":"Pipe Material",
    "Soil_Corrosivity":"Soil Corrosivity", "Risk (%)":"Risk (%)", "Assessment":"Assessment"
})[["Pipe ID","Pressure (PSI)","Flow (GPM)","Velocity (ft/s)","Temperature (°F)",
     "Pipe Age (years)","Pipe Material","Soil Corrosivity","Risk (%)","Assessment"]]

st.dataframe(df_top, use_container_width=True)

st.divider()

# ============ DOWNLOADS ============
csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download results (CSV)", data=csv_bytes,
                   file_name="network_risk_results.csv", mime="text/csv")

# pie chart for PDF
risk_counts = scored["Assessment"].value_counts().rename_axis("Category").reset_index(name="Count")
fig_pie = px.pie(
    risk_counts, names="Category", values="Count", hole=0.55,
    color="Category",
    color_discrete_map={"🔴 High Risk":"red", "🟡 Medium Risk":"gold", "🟢 Low Risk":"green"},
    title="Risk structure in the network"
)
fig_pie.update_traces(textinfo="percent+label")

try:
    from io import BytesIO
    pdf_bytes = None
    # If you want PDF support, re-enable zbuduj_pdf with English labels
except Exception as e:
    pdf_bytes = None

if pdf_bytes:
    st.download_button("📄 Download PDF Report (color)", data=pdf_bytes,
                       file_name="AI_network_risk_report.pdf", mime="application/pdf")
