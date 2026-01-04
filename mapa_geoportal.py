# mapa_geoportal.py
# ————————————————————————————————
# Interaktywna mapa BDOT10k + OSM (działa lokalnie w Streamlit)
# ————————————————————————————————
import folium
from streamlit_folium import st_folium
from folium import plugins


def generuj_mape():
    """
    Tworzy interaktywną mapę:
      - OpenStreetMap (działa zawsze)
      - BDOT10k (warstwa topograficzna Geoportalu)
    """

    # --- Utworzenie mapy ---
    m = folium.Map(location=[53.4738, 18.7555], zoom_start=16, control_scale=True)

    # --- OpenStreetMap (tło) ---
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors"
    ).add_to(m)

    # --- BDOT10k z Geoportalu ---
    bdot_url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT10k/WMS"
    folium.raster_layers.WmsTileLayer(
        url=bdot_url,
        layers="BDOT10k",
        name="Topografia (BDOT10k)",
        fmt="image/png",
        transparent=True,
        version="1.1.1",
        crs="EPSG:4326",
        attr="Źródło: GUGiK BDOT10k"
    ).add_to(m)

    # --- Narzędzia i kontrolki ---
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.LocateControl(auto_start=False).add_to(m)
    plugins.MousePosition(prefix="Współrzędne:").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def pokaz_mape_streamlit(mapa, wysokosc=700):
    """Renderuje mapę w Streamlit."""
    st_folium(mapa, height=wysokosc, width=None)
