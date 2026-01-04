import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

st.title("🗺️ Test mapy Geoportalu")

# Tworzymy mapę
m = folium.Map(location=[53.4738, 18.7555], zoom_start=17)

# Ortofotomapa z Geoportalu
folium.raster_layers.WmsTileLayer(
    url="https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/StandardResolution",
    layers="Raster",
    name="Ortofotomapa",
    fmt="image/png",
    transparent=False,
    version="1.1.1",
    crs="EPSG:4326",
).add_to(m)

# Działki EGiB
folium.raster_layers.WmsTileLayer(
    url="https://mapy.geoportal.gov.pl/wss/service/PZGIK/EGIB/WMS",
    layers="dzialki,numery_dzialek",
    name="Działki ewidencyjne",
    fmt="image/png",
    transparent=True,
    version="1.1.1",
    crs="EPSG:4326",
).add_to(m)

# Panel warstw
folium.LayerControl(collapsed=False).add_to(m)

# Wyświetlenie w Streamlit
st_folium(m, height=700)