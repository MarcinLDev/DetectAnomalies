# app.py
# ————————————————————————————————————————————————————————————————————
# Dashboard AI dla gminy – Predykcja awarii sieci wodnej
# ————————————————————————————————————————————————————————————————————
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm

import leafmap.foliumap as leafmap
import geopandas as gpd
from shapely.geometry import LineString, Point
from mapa_geoportal import generuj_mape, pokaz_mape_streamlit


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

# ============ USTAWIENIA STRONY ============
st.set_page_config(
    page_title="AI – Predykcja awarii sieci wodnej",
    page_icon="💧",
    layout="wide",
)

# ============ LOGO ============
SCIEZKA_LOGO = "assets/Logo.png"   # <- Twój plik logo

def wstaw_logo_fixed(logo_path=SCIEZKA_LOGO, link="#", wysokosc=48):
    if not os.path.exists(logo_path):
        return
    b64 = base64.b64encode(open(logo_path, "rb").read()).decode()
    st.markdown(f"""
    <a href="{link}">
      <img src="data:image/png;base64,{b64}" class="logo-fixed">
    </a>
    <style>
      .logo-fixed {{
        position: fixed; top: 12px; left: 14px; height: {wysokosc}px; z-index: 1000;
      }}
      @media (max-width: 768px) {{ .logo-fixed {{ height: 40px; }} }}
    </style>
    """, unsafe_allow_html=True)

wstaw_logo_fixed()

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)

# ============ GLOBALNY STYL – „Anthropic-light inspired” ============
st.markdown(r"""
<style>
:root{
  --bg:#ffffff;          /* główny obszar = biały */
  --sidebar-bg:#f5f6f8;  /* sidebar = lekko szary */
  --ink:#111111;
  --muted:#667085;
  --border:#E2E4EA;
  --card:#ffffff;
  --accent-gray:#9CA3AF;
}


/* tło główne + sidebar */
.stApp{ background:var(--bg)!important; color:var(--ink)!important; }
.block-container{ color:var(--ink)!important; }
[data-testid="stSidebar"] {
  background:var(--sidebar-bg)!important;
  border-right:1px solid var(--border);
}

/* ramki na wykresy, tabele, obrazy */
div[data-testid="stPlotlyChart"],
div[data-testid="stImage"],
div[data-testid="stDataFrame"],
div[data-testid="stTable"]{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:18px;
  padding:0;
  overflow:hidden;
  box-shadow:0 8px 24px rgba(0,0,0,.06);
}
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stTable"] > div{
  border-radius:inherit !important;
}

/* nagłówki sekcji – białe tło */
.panel-title{
  font-weight:700; color:var(--ink);
  padding:10px 12px; margin:6px 0 12px 0;
  border-left:4px solid var(--accent-gray);
  background:#ffffff;
  border-radius:10px;
  box-shadow:0 1px 2px rgba(0,0,0,.05);
}

/* KPI */
.karta-kpi{
  border-radius:14px; 
  padding:12px 16px;       /* było 18px 22px – zmniejszone */
  box-shadow:0 4px 12px rgba(0,0,0,.06);
  text-align:center; 
  font-weight:600; 
  background:#fff;
  font-size:0.9rem;        /* ogólny font trochę mniejszy */
}
.karta-kpi span{
  font-size:26px !important;  /* było 34px – zmniejszamy liczby */
}
.kpi-czerw{background:#ff4d4d;color:#fff;}
.kpi-zolty{background:#ffcc00;color:#111;}
.kpi-ziel {background:#22c55e;color:#fff;}
.podpis{color:var(--muted);font-size:.9rem;}
@keyframes puls {0%{box-shadow:0 0 0 0 rgba(255,77,77,.45);}70%{box-shadow:0 0 0 18px rgba(255,77,77,0);}100%{box-shadow:0 0 0 0 rgba(255,77,77,0);} }
.puls{animation:puls 2s infinite;}

/* przyciski pobierania (sidebar) */
[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button{
  width:100%; display:flex; align-items:center; gap:8px; justify-content:flex-start;
  border:1px solid var(--border);
  background:#fff; color:var(--ink);
  border-radius:14px; padding:14px 16px; font-weight:700;
  box-shadow:0 4px 10px rgba(0,0,0,.05);
  transition:background .15s ease, border-color .15s ease, transform .03s;
}
[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:hover{
  background:#f7f8fa; border-color:#d1d5db;
}
[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button:active{
  transform:translateY(1px);
}

/* action buttons w sidebarze */
[data-testid="stSidebar"] .stButton > button{
  width:100%; border:1px solid var(--border);
  background:#fff; color:var(--ink);
  border-radius:12px; padding:10px 14px; font-weight:600;
}
[data-testid="stSidebar"] .stButton > button:hover{
  background:#f7f8fa; border-color:#d1d5db;
}

/* nawigacja w sidebarze – styl systemowy z ikonami */
.sidebar-nav{ margin:8px 0 16px 0; }
.sidebar-nav a{
  display:flex; align-items:center; gap:10px;
  width:100%;
  padding:10px 14px; margin-bottom:10px;
  font-weight:600;
  font-size:14px;
  color:var(--ink); text-decoration:none;
  border:1px solid var(--border); border-radius:10px;
  background:#fff;
  transition:background .2s, box-shadow .2s;
}
.sidebar-nav a i{
  font-size:16px; opacity:.85;   /* ikona Lucide */
}
.sidebar-nav a:hover{
  background:#f7f8fa;
  box-shadow:0 4px 10px rgba(0,0,0,.05);
}
.sidebar-nav a.active{
  background:#eef0f3;
  border-color:#d1d5db;
}
            
.sidebar-nav a i {
  font-size: 18px;
  line-height: 1;
  margin-right: 6px;
}

/* spacing w sidebarze */
[data-testid="stSidebar"] .element-container{ margin-bottom:12px; }

/* logo */
.logo-fixed { position: fixed; top: 12px; left: 14px; height: 48px; z-index: 1000; }
@media (max-width: 768px) { .logo-fixed { height: 40px; } }
</style>
""", unsafe_allow_html=True)


# ============ FUNKCJE POMOCNICZE ============
@st.cache_data
def wczytaj_dane(sciezka: str) -> pd.DataFrame:
    return pd.read_excel(sciezka)

def dodaj_etykiety_ryzyka(ramka: pd.DataFrame, progi=(10, 70)) -> pd.DataFrame:
    prog_sredni, prog_wysoki = progi
    def etykieta(p):
        if p >= prog_wysoki: return "🔴 Wysokie ryzyko"
        if p >= prog_sredni: return "🟡 Średnie ryzyko"
        return "🟢 Niskie ryzyko"
    ramka["Ocena_modelu"] = ramka["Ryzyko (%)"].apply(etykieta)
    return ramka

@st.cache_data
def przetworz_dane(dane: pd.DataFrame):
    dane_kopia = dane.copy()
    koder_materialu = LabelEncoder()
    koder_gleby     = LabelEncoder()
    dane_kopia["Pipe_Material"]    = koder_materialu.fit_transform(dane_kopia["Pipe_Material"])
    dane_kopia["Soil_Corrosivity"] = koder_gleby.fit_transform(dane_kopia["Soil_Corrosivity"])
    cechy_X = dane_kopia.drop(columns=["Pipe_ID", "Leak_Class"])
    etykiety_y = dane_kopia["Leak_Class"].astype(int)
    return dane_kopia, cechy_X, etykiety_y, koder_materialu, koder_gleby

@st.cache_resource
def naucz_model_rf(cechy_X, etykiety_y, los=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        cechy_X, etykiety_y, test_size=0.2, random_state=los, stratify=etykiety_y
    )
    model = RandomForestClassifier(n_estimators=250, random_state=los, class_weight="balanced")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metryki = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "MacierzPomyłek": confusion_matrix(y_te, y_pred),
        "Raport": classification_report(y_te, y_pred, output_dict=True),
    }
    return model, metryki

def _zarejestruj_czcionki():
    baza, pogrub = "Helvetica", "Helvetica-Bold"
    try:
        if os.path.exists("fonts/DejaVuSans.ttf"):
            pdfmetrics.registerFont(TTFont("DejaVu", "fonts/DejaVuSans.ttf"))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", "fonts/DejaVuSans-Bold.ttf"))
            baza, pogrub = "DejaVu", "DejaVu-Bold"
    except Exception:
        pass
    return baza, pogrub

def zbuduj_pdf(dane_wyniki: pd.DataFrame, kpi: dict, wykres_pizza, wykres_waznosc) -> bytes:
    font_normal, font_bold = _zarejestruj_czcionki()
    styles = getSampleStyleSheet()
    styles["Title"].fontName = font_bold;  styles["Title"].fontSize = 22
    styles["Heading3"].fontName = font_bold
    styles["Normal"].fontName = font_normal; styles["Normal"].fontSize = 10

    styl_kpi_numer = ParagraphStyle("KPI_NUM", parent=styles["Normal"], fontName=font_bold, fontSize=24,
                                    textColor=colors.white, alignment=1)
    styl_kpi_etyk = ParagraphStyle("KPI_LBL", parent=styles["Normal"], fontName=font_bold, fontSize=10,
                                   textColor=colors.white, alignment=1)

    for fig in (wykres_pizza, wykres_waznosc):
        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font_color="#111")
    buf_pie, buf_imp = BytesIO(), BytesIO()
    wykres_pizza.write_image(buf_pie, format="png", scale=2)
    wykres_waznosc.write_image(buf_imp, format="png", scale=2)
    buf_pie.seek(0); buf_imp.seek(0)

    dane_pdf = dane_wyniki.copy()
    dane_pdf["Ocena_pdf"] = (dane_pdf["Ocena_modelu"]
                             .str.replace("🔴 ", "", regex=False)
                             .str.replace("🟡 ", "", regex=False)
                             .str.replace("🟢 ", "", regex=False))
    top10 = dane_pdf.sort_values("Ryzyko (%)", ascending=False).head(10)

    bufor = BytesIO()
    doc = SimpleDocTemplate(bufor, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=26, bottomMargin=24)
    el = []

    el.append(Paragraph("AI – Predykcja awarii sieci wodnej", styles["Title"]))
    el.append(Spacer(1, 8))

    kpi_tab = Table([
        [Paragraph("Wysokie ryzyko", styl_kpi_etyk),
         Paragraph("Średnie ryzyko", styl_kpi_etyk),
         Paragraph("Niskie ryzyko",  styl_kpi_etyk)],
        [Paragraph(str(kpi["wysokie"]), styl_kpi_numer),
         Paragraph(str(kpi["srednie"]), styl_kpi_numer),
         Paragraph(str(kpi["niskie"]),  styl_kpi_numer)]
    ], colWidths=[6.1*cm, 6.1*cm, 6.1*cm], rowHeights=[14, 34])
    kpi_tab.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,1), colors.Color(1, 0.30, 0.30)),
        ("BACKGROUND", (1,0), (1,1), colors.Color(1, 0.83, 0.00)),
        ("BACKGROUND", (2,0), (2,1), colors.Color(0.20, 0.75, 0.36)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
        ("INNERGRID", (0,0), (-1,-1), 0.0, colors.white),
        ("BOX", (0,0), (-1,-1), 0.0, colors.white),
    ]))
    el.append(kpi_tab); el.append(Spacer(1, 10))

    el.append(Paragraph("Struktura ryzyka", styles["Heading3"]))
    el.append(Image(buf_pie, width=15.5*cm, height=9.8*cm))
    el.append(Spacer(1, 10))
    el.append(Paragraph("Najważniejsze czynniki ryzyka (feature importance)", styles["Heading3"]))
    el.append(Image(buf_imp, width=15.5*cm, height=9.8*cm))
    el.append(Spacer(1, 10))

    nag = ["ID odcinka","Ciśnienie (PSI)","Przepływ (GPM)","Wiek rury (lata)","Ryzyko (%)","Ocena"]
    wiersze = [[
        r["Pipe_ID"], f"{r['Pressure_PSI']:.1f}", f"{r['Flow_GPM']:.1f}",
        int(r["Pipe_Age_Years"]), f"{r['Ryzyko (%)']:.0f}", r["Ocena_pdf"]
    ] for _,r in top10.iterrows()]
    t = Table([nag] + wiersze, repeatRows=1, colWidths=[3.1*cm, 3.2*cm, 3.2*cm, 3.1*cm, 2.2*cm, 4.0*cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), font_bold),
        ("FONTNAME", (0,1), (-1,-1), font_normal),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("BACKGROUND", (0,0), (-1,0), colors.Color(.92,.95,.98)),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("ALIGN", (0,1), (-2,-1), "CENTER"),
        ("ALIGN", (-1,1), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.Color(.84,.86,.90)),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.Color(.97,.97,.97)]),
    ]))
    el.append(t)
    doc.build(el)
    return bufor.getvalue()

# ============ SIDEBAR – USTAWIENIA + NAV ============
st.sidebar.markdown("""
<div class="sidebar-nav">
  <a class="active" href="#home"><i class="fa fa-home"></i> Home</a>
  <a href="#top10"><i class="fa fa-table"></i> Tabela TOP</a>
  <a href="#charts"><i class="fa fa-chart-bar"></i> Wykresy</a>
  <a href="#sim"><i class="fa fa-sliders-h"></i> Symulacja</a>
  <a href="#raport"><i class="fa fa-download"></i> Pobieranie</a>
</div>
""", unsafe_allow_html=True)

prog_sredni  = st.sidebar.slider("Próg średniego ryzyka (%)", 5, 60, 1, step=5)
prog_wysoki  = st.sidebar.slider("Próg wysokiego ryzyka (%)", 70, 95, 99, step=5)
pokaz_tylko_wysokie = st.sidebar.toggle("🔎︎ Pokaż tylko wysokie ryzyko w tabeli", value=False)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("↺ Przelicz", key="btn_recalc"):
        st.rerun()
with col_sb2:
    if st.button("🧹 Reset"):
        st.session_state.clear(); st.rerun()

# Nadpisz etykietę przycisku na HTML
st.markdown(
    """
    <style>
    div[data-testid="stButton"][key="btn_recalc"] button:before {
        font-family: "Font Awesome 6 Free";
        content: "\\f01e";  /* ikona rotate-right */
        font-weight: 900;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# ============ DANE → MODEL → SKORING ============
# Ścieżka jak u Ciebie (zostawiamy); jeśli chcesz fallback do /mnt/data, dopisz własny warunek.
sciezka_danych = "data/raw/water_network_leak_dataset.xlsx"
if not os.path.exists(sciezka_danych):
    st.error(f"Nie znaleziono pliku: {sciezka_danych}")
    st.stop()

dane_surowe = wczytaj_dane(sciezka_danych)
dane_modelowe, cechy_X, etykiety_y, koder_materialu, koder_gleby = przetworz_dane(dane_surowe)
model_rf, metryki = naucz_model_rf(cechy_X, etykiety_y)

prawdopodobienstwo = (model_rf.predict_proba(cechy_X)[:, 1] * 100).round(1)
dane_wyniki = dane_surowe.copy()
dane_wyniki["Ryzyko (%)"] = prawdopodobienstwo
dane_wyniki = dodaj_etykiety_ryzyka(dane_wyniki, (prog_sredni, prog_wysoki))

# ============ KPI ============
liczba_wysokie = (dane_wyniki["Ocena_modelu"] == "🔴 Wysokie ryzyko").sum()
liczba_srednie = (dane_wyniki["Ocena_modelu"] == "🟡 Średnie ryzyko").sum()
liczba_niskie  = (dane_wyniki["Ocena_modelu"] == "🟢 Niskie ryzyko").sum()
liczba_razem   = len(dane_wyniki)
kpi = {"wysokie": liczba_wysokie, "srednie": liczba_srednie, "niskie": liczba_niskie, "razem": liczba_razem}

# ============ NAGŁÓWEK + KPI ============
st.markdown('<a id="home"></a>', unsafe_allow_html=True)
st.title("AI – Predykcja awarii sieci wodnej")

kol1, kol2, kol3 = st.columns([1,1,1])
klasa_czer = "karta-kpi kpi-czerw puls" if liczba_wysokie > 0 else "karta-kpi kpi-czerw"
with kol1: st.markdown(f'<div class="{klasa_czer}"><i class="fa fa-triangle-exclamation"></i> Wysokie ryzyko<br><span style="font-size:34px">{liczba_wysokie}</span></div>', unsafe_allow_html=True)
with kol2: st.markdown(f'<div class="karta-kpi kpi-zolty"><i class="fa fa-circle-exclamation"></i> Średnie ryzyko<br><span style="font-size:34px">{liczba_srednie}</span></div>', unsafe_allow_html=True)
with kol3: st.markdown(f'<div class="karta-kpi kpi-ziel"><i class="fa fa-circle-check"></i> Niskie ryzyko<br><span style="font-size:34px">{liczba_niskie}</span></div>', unsafe_allow_html=True)

st.markdown(f"<div class='podpis'>Łącznie elementów: <b>{liczba_razem}</b>. Progi: średnie ≥ {prog_sredni}%, wysokie ≥ {prog_wysoki}%.</div>", unsafe_allow_html=True)



# --- mapa bazowa ---

st.markdown("<div class='panel-title'><i class='fa fa-map'></i> Mapa sieci</div>", unsafe_allow_html=True) 
st.image("assets/3!.png", use_container_width=True)

# # --- Mapa Geoportalu ---

# st.markdown("<div class='panel-title'><i class='fa fa-map'></i> Mapa sieci (interaktywna)</div>", unsafe_allow_html=True)

# m = generuj_mape()
# pokaz_mape_streamlit(m, wysokosc=700)




# ============ TOP – elementy o najwyższym ryzyku ============
st.markdown('<a id="top10"></a>', unsafe_allow_html=True)
st.markdown("<div class='panel-title'><i class='fa fa-list-ol'></i> Elementy o najwyższym ryzyku</div>", unsafe_allow_html=True)

kol_top1, kol_top2, kol_top3 = st.columns([1,1,1])
with kol_top1:
    ile_pokazac = st.slider("Ile pozycji pokazać", min_value=5, max_value=100, value=10, step=5)
with kol_top2:
    kolumna_sort = st.selectbox("Sortuj wg", ["Ryzyko (%)","Pipe_Age_Years","Pressure_PSI","Flow_GPM"], index=0)
with kol_top3:
    rosnaco = st.toggle("Rosnąco", value=False)

# Źródło do tabeli (z opcjonalnym filtrem)
zrodlo_tabeli = dane_wyniki[dane_wyniki["Ocena_modelu"]=="🔴 Wysokie ryzyko"] if pokaz_tylko_wysokie else dane_wyniki
widok_top = zrodlo_tabeli.sort_values(kolumna_sort, ascending=rosnaco).head(ile_pokazac)

df_top = widok_top.rename(columns={
    "Pipe_ID":"ID odcinka", "Pressure_PSI":"Ciśnienie (PSI)", "Flow_GPM":"Przepływ (GPM)",
    "Velocity_FPS":"Prędkość (ft/s)", "Temperature_F":"Temperatura (°F)",
    "Pipe_Age_Years":"Wiek rury (lata)", "Pipe_Material":"Materiał rury",
    "Soil_Corrosivity":"Korozyjność gleby", "Ryzyko (%)":"Ryzyko (%)", "Ocena_modelu":"Ocena"
})[["ID odcinka","Ciśnienie (PSI)","Przepływ (GPM)","Prędkość (ft/s)","Temperatura (°F)",
     "Wiek rury (lata)","Materiał rury","Korozyjność gleby","Ryzyko (%)","Ocena"]]

n_wierszy = len(df_top)
kolory_zebra = (['#FFFFFF','#F6F7F9'] * ((n_wierszy+1)//2))[:n_wierszy]
naglowki = list(df_top.columns)
komorki  = [df_top[c].tolist() for c in df_top.columns]

fig_tabela = go.Figure(go.Table(
    header=dict(
        values=naglowki, fill_color="#ECEFF3", line_color="#E5E7EB",
        font=dict(color="#111111", size=13), align="center", height=34
    ),
    cells=dict(
        values=komorki, fill_color=[kolory_zebra]*len(naglowki), line_color="#E5E7EB",
        font=dict(color="#111111", size=12), align="center", height=32,
        format=[None, ".1f", ".1f", ".1f", ".0f", None, None, None, ".0f", None]
    )
))
fig_tabela.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                         font_color="#111111", margin=dict(l=0, r=0, t=0, b=0))

# Wykres kołowy
liczniki_ryzyka = dane_wyniki["Ocena_modelu"].value_counts().rename_axis("Kategoria").reset_index(name="Liczba")
wykres_pizza = px.pie(
    liczniki_ryzyka, names="Kategoria", values="Liczba", hole=0.55,
    color="Kategoria",
    color_discrete_map={"🔴 Wysokie ryzyko":"red", "🟡 Średnie ryzyko":"gold", "🟢 Niskie ryzyko":"green"},
    title=""
)
wykres_pizza.update_traces(textinfo="percent+label")
wykres_pizza.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font_color="#111")

col_tab, col_pie = st.columns([2,1])
with col_tab:
    # st.markdown("<div class='panel-title'>📋 Tabela TOP</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_tabela, use_container_width=True, theme=None)
with col_pie:
    # st.markdown("<div class='panel-title'>🍩 Struktura ryzyka</div>", unsafe_allow_html=True)
    st.plotly_chart(wykres_pizza, use_container_width=True, theme=None)

# ============ POLSKIE NAZWY CECH ============
slownik_nazw_cech = {
    "Pressure_PSI":     "Ciśnienie (PSI)",
    "Flow_GPM":         "Przepływ (GPM)",
    "Velocity_FPS":     "Prędkość (ft/s)",
    "Temperature_F":    "Temperatura (°F)",
    "Pipe_Age_Years":   "Wiek rury (lata)",
    "Pipe_Material":    "Materiał rury",
    "Soil_Corrosivity": "Korozyjność gleby",
}

# ============ WYKRES WAŻNOŚCI CECH + SYMULACJA ============
st.markdown('<a id="charts"></a>', unsafe_allow_html=True)
st.subheader("📈 Czynniki ryzyka")

waznosc_df = pd.DataFrame({"Cechy": list(cechy_X.columns), "Waznosc": model_rf.feature_importances_})
waznosc_df["Cechy_PL"] = waznosc_df["Cechy"].map(slownik_nazw_cech).fillna(waznosc_df["Cechy"])

# sortujemy malejąco żeby najważniejsze były po lewej
waznosc_df = waznosc_df.sort_values("Waznosc", ascending=False)

# wykres liniowy
wykres_waznosc = px.line(
    waznosc_df,
    x="Cechy_PL", y="Waznosc",
    markers=True
)

wykres_waznosc.update_traces(
    line=dict(color="#0B7CFF", width=3),
    marker=dict(size=10, color="#0B7CFF", line=dict(width=2, color="white")),
    hovertemplate="<b>%{x}</b><br>Importance: %{y:.3f}<extra></extra>"
)

wykres_waznosc.update_layout(
    template="plotly_white",
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(color="#111", size=13),
    margin=dict(l=40, r=30, t=30, b=60),
    height=500,
    xaxis=dict(title="", tickangle=-20),
    yaxis=dict(title="Importance", showgrid=True, gridcolor="#e5e7eb", zeroline=False)
)

col_waz, col_sim = st.columns([2,1])
with col_waz:
    st.markdown("<div class='panel-title'><i class='fa fa-chart-line'></i> Ważność cech</div>", unsafe_allow_html=True)
    st.plotly_chart(wykres_waznosc, use_container_width=True, theme=None)

with col_sim:
    st.markdown('<a id="sim"></a>', unsafe_allow_html=True)
    st.markdown("<div class='panel-title'><i class='fa fa-flask'></i> Symulacja „co-jeśli”</div>", unsafe_allow_html=True)
    




    wartosci_typowe = dane_wyniki.median(numeric_only=True).to_dict()
    s1, s2 = st.columns(2)
    with s1:
        cisnienie_sim = st.slider("Ciśnienie (PSI)", 0.0, 120.0, float(wartosci_typowe.get("Pressure_PSI", 60.0)), 0.1)
        wiek_sim      = st.slider("Wiek rury (lata)", 0,   60,    int(wartosci_typowe.get("Pipe_Age_Years", 10)), 1)
        material_sim  = st.selectbox("Materiał", sorted(dane_wyniki["Pipe_Material"].unique()))
    with s2:
        przeplyw_sim  = st.slider("Przepływ (GPM)",   0.0, 260.0, float(wartosci_typowe.get("Flow_GPM", 120.0)), 0.1)
        predkosc_sim  = st.slider("Prędkość (ft/s)",  0.0, 12.0, float(wartosci_typowe.get("Velocity_FPS", 4.0)), 0.1)
        gleba_sim     = st.selectbox("Korozyjność gleby", sorted(dane_wyniki["Soil_Corrosivity"].unique()))

    wiersz_sim = pd.DataFrame([{
        "Pressure_PSI": cisnienie_sim, "Flow_GPM": przeplyw_sim, "Velocity_FPS": predkosc_sim,
        "Temperature_F": wartosci_typowe.get("Temperature_F", 60.0),
        "Pipe_Age_Years": wiek_sim, "Pipe_Material": material_sim, "Soil_Corrosivity": gleba_sim
    }])

    wiersz_sim_kod = wiersz_sim.copy()
    wiersz_sim_kod["Pipe_Material"]    = LabelEncoder().fit(dane_surowe["Pipe_Material"]).transform(wiersz_sim_kod["Pipe_Material"])
    wiersz_sim_kod["Soil_Corrosivity"] = LabelEncoder().fit(dane_surowe["Soil_Corrosivity"]).transform(wiersz_sim_kod["Soil_Corrosivity"])
    ryzyko_sim = model_rf.predict_proba(wiersz_sim_kod)[:, 1].item() * 100

    prog_sredni = 10
    prog_wysoki = 80
    ryzyko_sim = 23.7  # np. wynik modelu

    wykres_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ryzyko_sim,
        number={
            'suffix': "%",
            'font': {'size': 36, 'color': "#111", 'family': "Arial Black"}
        },
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#999"},
            'bar': {'color': "#0B7CFF", 'thickness': 0.12},   # cieńszy pasek
            'bgcolor': "white",
            'steps': [
                {'range': [0, prog_sredni], 'color': 'rgba(34,197,94,0.20)'},   # zielony pastel
                {'range': [prog_sredni, prog_wysoki], 'color': 'rgba(250,204,21,0.25)'}, # żółty pastel
                {'range': [prog_wysoki, 100], 'color': 'rgba(239,68,68,0.25)'}  # czerwony pastel
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': prog_wysoki
            }
        },
        title={
            'text': "Prognozowane ryzyko awarii",
            'font': {'size': 14, 'color': "#333", 'family': "Arial"}
        }
    ))

    wykres_gauge.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=80, b=20),
        height=230   # <-- niższy, wygląda jak widget
    )
    st.plotly_chart(wykres_gauge, use_container_width=True, theme=None)

st.divider()



# ============ POBIERANIE: CSV + PDF ============
st.markdown('<a id="raport"></a>', unsafe_allow_html=True)

bajty_csv = dane_wyniki.to_csv(index=False).encode("utf-8")

# spróbuj zbudować PDF (wymaga kaleido do zapisu wykresów)
try:
    bajty_pdf = zbuduj_pdf(dane_wyniki, kpi, wykres_pizza, wykres_waznosc)
except Exception as e:
    bajty_pdf = None
    st.info("PDF wymaga pakietu 'kaleido' i czcionek DejaVu (opcjonalnie). Jeśli nie masz, zainstaluj: `pip install -U kaleido`.")

st.sidebar.subheader("Pobierz wyniki")
st.sidebar.download_button("⬇ Pobierz CSV", data=bajty_csv, file_name="wyniki_ryzyka_sieci.csv",
                           mime="text/csv", use_container_width=True)

if bajty_pdf is not None:
    st.sidebar.download_button("📄 Pobierz raport PDF", data=bajty_pdf, file_name="raport_AI_siec_wod_kan.pdf",
                               mime="application/pdf", use_container_width=True)
else:
    # awaryjnie udostępnij CSV pod nazwą PDF, żeby był przycisk (jak wcześniej)
    st.sidebar.download_button("📄 Pobierz raport (tymczasowo CSV)", data=bajty_csv,
                               file_name="raport_AI_siec_wod_kan.csv", mime="text/csv",
                               use_container_width=True)
