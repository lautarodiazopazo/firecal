"""
FireCal · Calendario armonizado de focos de calor (MODIS + VIIRS)
NASA Space Apps Challenge 2026 — proyecto del equipo, no es una herramienta oficial de NASA.

Correr:
    streamlit run app/app.py
"""
import datetime as dt
import io
import math
import os
import time
import zipfile

import duckdb
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import box, shape
from streamlit_folium import st_folium

import noticias

# ------------------------------------------------------------------ config
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_APP = os.environ.get("FIRE_APP_DATA", os.path.join(APP_DIR, "data", "app_data.parquet"))
DATA_1KM = os.environ.get("FIRE_1KM_DATA", os.path.join(APP_DIR, "..", "data", "salida", "armonizado.parquet"))
DATA_RAW = os.environ.get("FIRE_RAW_DATA", os.path.join(APP_DIR, "..", "data", "parquet"))

GRID = 0.1                                   # grado de la grilla de la app
AQUA_START = dt.date(2002, 7, 4)             # antes: sólo Terra
VIIRS_START = dt.date(2012, 1, 20)           # desde acá: VIIRS S-NPP
SNPP_END = dt.date(2026, 11, 2)              # después: NOAA-20
REF_START = 2003                             # años de referencia (climatología)
MAX_RAW_ROWS = 1_500_000                     # límite de descarga de focos originales

PRESETS = {  # rectángulos aproximados: (lat_min, lat_max, lon_min, lon_max)
    "Mendoza (Argentina)": (-37.6, -32.0, -70.6, -66.5),
    "Sierras de Córdoba (Argentina)": (-33.5, -29.5, -65.5, -63.8),
    "Delta del Paraná (Argentina)": (-34.4, -32.0, -60.8, -58.3),
    "Gran Chaco": (-26.0, -18.0, -64.0, -58.0),
    "Pantanal": (-20.5, -15.5, -58.5, -54.5),
    "Cerrado / MATOPIBA (Brasil)": (-12.0, -4.0, -48.0, -42.0),
    "Arco de deforestación amazónico": (-12.0, -7.0, -66.0, -52.0),
    "Sudamérica completa": (-56.0, 13.0, -82.0, -34.0),
}

ACK = ("We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource "
       "Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere "
       "Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) "
       "and NASA's Earth Science Data and Information System (ESDIS).")

# ------------------------------------------------------------------ textos
TXT = {
    "es": {
        "title": "🔥 Calendario armonizado de focos de calor",
        "subtitle": "MODIS (2000–2012) + VIIRS (2012–hoy) en una sola serie consistente · Sudamérica",
        "lang": "Idioma",
        "zone": "1 · Zona de interés",
        "zone_mode": "Cómo elegir la zona",
        "m_draw": "Dibujar en el mapa", "m_coords": "Coordenadas", "m_preset": "Zonas de ejemplo",
        "draw_help": "Usá las herramientas ▭ (rectángulo) o ⬠ (polígono) arriba a la izquierda del mapa.",
        "lat_min": "Latitud mín.", "lat_max": "Latitud máx.", "lon_min": "Longitud mín.", "lon_max": "Longitud máx.",
        "preset": "Zona", "preset_note": "Rectángulos aproximados, sólo de ejemplo.",
        "period": "2 · Período", "range": "Rango de fechas",
        "options": "Opciones", "excl_terra": "Ocultar nov 2000 – jun 2002 (sólo Terra, menor confianza)",
        "ramp": "Rampa de color", "log": "Escala logarítmica",
        "no_zone": "👈 Elegí una zona: dibujala en el mapa, ingresá coordenadas o elegí una zona de ejemplo.",
        "no_data": "No hay focos en esta zona y período.",
        "small_zone": "La zona es muy chica para la grilla de 0,1° (~11 km); se usan las celdas que la tocan.",
        "low_counts": "Pocos focos en esta zona: los valores mensuales son ruidosos y la armonización es menos confiable.",
        "k_total": "Focos en el período", "k_total_h": "Focos armonizados: celdas de 1 km con fuego por día, en escala VIIRS S-NPP.",
        "k_peak": "Mes típicamente más activo", "k_year": "Año más activo",
        "k_now": "Mes actual vs. mediana", "k_now_h": "Comparado con la mediana de los mismos días del mismo mes, {a}–{b}.",
        "tabs": ["📅 Calendario", "📍 Mes actual vs. historia", "📈 Estacionalidad", "📊 Serie anual", "🗺️ Mapa", "📰 Noticias", "⬇️ Descargas"],
        "cal_mode": "Mostrar", "cal_abs": "Cantidad de focos", "cal_anom": "Anomalía vs. mediana del mes (%)",
        "cal_note": "× = sólo Terra (menor confianza; la escala de color no los usa para no aplastar el resto) · ○ = mes incompleto. Pasá el mouse para ver detalles. "
                    "La cámara de la barra del gráfico lo descarga como PNG.",
        "year": "Año", "month": "Mes", "focos": "Focos armonizados", "median": "Mediana del mes",
        "anom": "Anomalía", "source": "Sensor", "partial": "Mes incompleto",
        "cmp_pick": "Mes a analizar", "cmp_days": "Días comparados: 1 al {d}",
        "cmp_cur": "Este mes", "cmp_med": "Mediana {a}–{b}", "cmp_pct": "Percentil",
        "cmp_rank": "Puesto", "cmp_rank_v": "{r}° de {n} años",
        "cum_title": "Focos acumulados en el mes", "day": "Día del mes", "iqr": "Rango típico (P25–P75)",
        "bars_title": "Mismo período en cada año",
        "seas_year": "Destacar año", "seas_title": "Perfil estacional: mediana y rango típico ({a}–{b})",
        "ann_raw": "Mostrar serie sin armonizar", "ann_title": "Total anual de focos",
        "harm": "Armonizado", "terra_y": "incluye meses sólo Terra (menor confianza)", "raw": "Sin armonizar (celdas-día crudas)", "partial_y": "año incompleto",
        "ann_note": "La serie sin armonizar muestra el salto artificial de 2012 (cambio MODIS → VIIRS) que corrige la armonización.",
        "map_title": "Focos en la zona y período (celdas de 0,1°)",
        "dl_intro": "Todas las descargas son ZIP con el CSV y un LEEME.txt con unidades, método y la cita de FIRMS.",
        "dl_month": "Calendario mensual de la zona", "dl_daily": "Focos diarios por celda de 0,1°",
        "dl_1km": "Focos armonizados por celda de 1 km y día", "dl_raw": "Focos originales de FIRMS (sin armonizar)",
        "dl_prep": "Preparar", "dl_get": "Descargar", "dl_missing": "No disponible en esta instalación (falta {f}).",
        "dl_toolarge": "Son {n:,} focos: reducí la zona o el período (máximo {m:,}).",
        "dl_rows": "{n:,} filas listas.",
        "n_intro": "Titulares de prensa (GDELT, desde 2017) y eventos destacados (Wikipedia) sobre incendios en los lugares "
                   "con más focos de tu zona, para el período elegido. Se muestran sólo titulares y enlaces; el contenido es de cada medio.",
        "n_places": "Lugares con más focos en la zona y período", "n_none_places": "No hay focos para ubicar lugares.",
        "n_extra": "Palabras clave adicionales (opcional)", "n_months": "Meses a consultar en prensa (máx. 3)",
        "n_btn": "🔎 Buscar noticias", "n_press": "📰 Prensa (GDELT)", "n_wiki": "📚 Eventos destacados (Wikipedia)",
        "n_pre2017": "GDELT cubre desde 2017: para años anteriores sólo hay eventos de Wikipedia.",
        "n_rate": "GDELT está limitando consultas desde esta red (muchos usuarios a la vez). Probá de nuevo en un minuto.",
        "n_media": "Sólo medios reconocidos (lista curada) en español, portugués o inglés · 📍 medio de la zona · 🌐 internacional o regional.",
        "n_filtered": "{n} artículos encontrados, pero ninguno de medios reconocidos de la lista.",
        "n_err": "No se pudo consultar GDELT ({e}).", "n_empty": "Sin resultados.",
        "n_wait": "Consultando… (GDELT permite 1 consulta cada 5 segundos)",
        "footer": "Proyecto del equipo BurnInHell para el NASA Space Apps Challenge 2026 (sede Mendoza). No es una herramienta oficial de NASA.",
        "src": {"Terra": "MODIS Terra × k", "MODIS": "MODIS × k", "SNPP": "VIIRS S-NPP", "NOAA20": "VIIRS NOAA-20 × k"},
        "months": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
    },
    "en": {
        "title": "🔥 Harmonized active-fire calendar",
        "subtitle": "MODIS (2000–2012) + VIIRS (2012–today) in one consistent series · South America",
        "lang": "Language",
        "zone": "1 · Area of interest",
        "zone_mode": "How to choose the area",
        "m_draw": "Draw on the map", "m_coords": "Coordinates", "m_preset": "Example areas",
        "draw_help": "Use the ▭ (rectangle) or ⬠ (polygon) tools at the top-left of the map.",
        "lat_min": "Min latitude", "lat_max": "Max latitude", "lon_min": "Min longitude", "lon_max": "Max longitude",
        "preset": "Area", "preset_note": "Approximate rectangles, for illustration only.",
        "period": "2 · Period", "range": "Date range",
        "options": "Options", "excl_terra": "Hide Nov 2000 – Jun 2002 (Terra only, lower confidence)",
        "ramp": "Color ramp", "log": "Log scale",
        "no_zone": "👈 Choose an area: draw it on the map, enter coordinates or pick an example area.",
        "no_data": "No hotspots in this area and period.",
        "small_zone": "The area is very small for the 0.1° (~11 km) grid; cells touching it are used.",
        "low_counts": "Few hotspots in this area: monthly values are noisy and harmonization is less reliable.",
        "k_total": "Hotspots in period", "k_total_h": "Harmonized hotspots: 1-km cells with fire per day, on the VIIRS S-NPP scale.",
        "k_peak": "Typically busiest month", "k_year": "Busiest year",
        "k_now": "Current month vs. median", "k_now_h": "Compared with the median of the same days of the same month, {a}–{b}.",
        "tabs": ["📅 Calendar", "📍 Current month vs. history", "📈 Seasonality", "📊 Annual series", "🗺️ Map", "📰 News", "⬇️ Downloads"],
        "cal_mode": "Show", "cal_abs": "Hotspot count", "cal_anom": "Anomaly vs. monthly median (%)",
        "cal_note": "× = Terra only (lower confidence; excluded from the color scale so they don't flatten the rest) · ○ = incomplete month. Hover for details. "
                    "The camera icon in the chart toolbar downloads it as PNG.",
        "year": "Year", "month": "Month", "focos": "Harmonized hotspots", "median": "Monthly median",
        "anom": "Anomaly", "source": "Sensor", "partial": "Incomplete month",
        "cmp_pick": "Month to analyze", "cmp_days": "Days compared: 1 to {d}",
        "cmp_cur": "This month", "cmp_med": "Median {a}–{b}", "cmp_pct": "Percentile",
        "cmp_rank": "Rank", "cmp_rank_v": "#{r} of {n} years",
        "cum_title": "Cumulative hotspots in the month", "day": "Day of month", "iqr": "Typical range (P25–P75)",
        "bars_title": "Same period in every year",
        "seas_year": "Highlight year", "seas_title": "Seasonal profile: median and typical range ({a}–{b})",
        "ann_raw": "Show non-harmonized series", "ann_title": "Annual hotspot total",
        "harm": "Harmonized", "terra_y": "includes Terra-only months (lower confidence)", "raw": "Non-harmonized (raw cell-days)", "partial_y": "incomplete year",
        "ann_note": "The non-harmonized series shows the artificial 2012 jump (MODIS → VIIRS switch) that harmonization removes.",
        "map_title": "Hotspots in the area and period (0.1° cells)",
        "dl_intro": "Every download is a ZIP with the CSV and a README.txt with units, method and the FIRMS citation.",
        "dl_month": "Monthly calendar of the area", "dl_daily": "Daily hotspots per 0.1° cell",
        "dl_1km": "Harmonized hotspots per 1-km cell and day", "dl_raw": "Original FIRMS hotspots (not harmonized)",
        "dl_prep": "Prepare", "dl_get": "Download", "dl_missing": "Not available in this installation ({f} missing).",
        "dl_toolarge": "That is {n:,} hotspots: reduce the area or period (max {m:,}).",
        "dl_rows": "{n:,} rows ready.",
        "n_intro": "Press headlines (GDELT, since 2017) and notable events (Wikipedia) about fires in the places with the most "
                   "hotspots in your area, for the chosen period. Only headlines and links are shown; content belongs to each outlet.",
        "n_places": "Places with the most hotspots in the area and period", "n_none_places": "No hotspots to locate places.",
        "n_extra": "Extra keywords (optional)", "n_months": "Months to search in the press (max 3)",
        "n_btn": "🔎 Search news", "n_press": "📰 Press (GDELT)", "n_wiki": "📚 Notable events (Wikipedia)",
        "n_pre2017": "GDELT covers 2017 onwards: earlier years only have Wikipedia events.",
        "n_rate": "GDELT is rate-limiting this network (many users at once). Try again in a minute.",
        "n_media": "Recognized outlets only (curated list) in Spanish, Portuguese or English · 📍 local outlet · 🌐 international or regional.",
        "n_filtered": "{n} articles found, but none from recognized outlets in the list.",
        "n_err": "Could not query GDELT ({e}).", "n_empty": "No results.",
        "n_wait": "Searching… (GDELT allows 1 query every 5 seconds)",
        "footer": "Project by team BurnInHell for the NASA Space Apps Challenge 2026 (Mendoza local event). Not an official NASA tool.",
        "src": {"Terra": "MODIS Terra × k", "MODIS": "MODIS × k", "SNPP": "VIIRS S-NPP", "NOAA20": "VIIRS NOAA-20 × k"},
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },
}

RAMPS = {"Viridis": "Viridis", "Inferno": "Inferno", "Cividis": "Cividis", "YlOrRd": "YlOrRd"}
C_MODIS, C_VIIRS, C_RAW, C_CUR = "#3B7DD8", "#E07B39", "#9E9E9E", "#C0392B"

st.set_page_config(page_title="FireCal · Space Apps 2026", page_icon="🔥", layout="wide")


# ------------------------------------------------------------------ datos
@st.cache_resource(show_spinner="Cargando datos…")
def load_data():
    t = pq.read_table(DATA_APP)
    days = t.column("date").cast(pa.int32()).to_numpy()
    d64 = days.astype("datetime64[D]")
    m64 = d64.astype("datetime64[M]")
    data = {
        "g_r": t.column("g_r").to_numpy().astype(np.int16),
        "g_c": t.column("g_c").to_numpy().astype(np.int16),
        "days": days,
        "focos": t.column("focos").to_numpy().astype(np.float32),
        "crudo": t.column("crudo").to_numpy().astype(np.int32),
    }
    dmin = d64.min().astype(object)
    dmax = d64.max().astype(object)
    return data, dmin, dmax


def source_of(d):
    if d < AQUA_START:
        return "Terra"
    if d < VIIRS_START:
        return "MODIS"
    if d <= SNPP_END:
        return "SNPP"
    return "NOAA20"


def cell_selection(data, wkt):
    """Índices de filas y celdas dentro de la geometría."""
    geom = shapely.from_wkt(wkt)
    minx, miny, maxx, maxy = geom.bounds
    r0, r1 = math.floor(miny / GRID), math.floor(maxy / GRID)
    c0, c1 = math.floor(minx / GRID), math.floor(maxx / GRID)
    g_r, g_c = data["g_r"], data["g_c"]
    m = (g_r >= r0) & (g_r <= r1) & (g_c >= c0) & (g_c <= c1)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return idx, False
    rr, cc = g_r[idx].astype(np.int32), g_c[idx].astype(np.int32)
    key = (rr - r0) * (c1 - c0 + 1) + (cc - c0)
    ukey, inv = np.unique(key, return_inverse=True)
    ur, uc = ukey // (c1 - c0 + 1) + r0, ukey % (c1 - c0 + 1) + c0
    inside = shapely.contains_xy(geom, (uc + 0.5) * GRID, (ur + 0.5) * GRID)
    small = False
    if not inside.any():  # zona más chica que una celda: usar celdas que la tocan
        boxes = shapely.box(uc * GRID, ur * GRID, (uc + 1) * GRID, (ur + 1) * GRID)
        inside = shapely.intersects(geom, boxes)
        small = True
    return idx[inside[inv]], small


@st.cache_data(show_spinner=False, max_entries=16)
def zone_daily(wkt):
    data, dmin, dmax = load_data()
    idx, small = cell_selection(data, wkt)
    d0 = np.datetime64(dmin, "D").astype(np.int64)
    n = (np.datetime64(dmax, "D") - np.datetime64(dmin, "D")).astype(int) + 1
    if idx.size:
        dd = data["days"][idx] - d0
        f = np.bincount(dd, weights=data["focos"][idx], minlength=n)
        c = np.bincount(dd, weights=data["crudo"][idx], minlength=n)
    else:
        f = c = np.zeros(n)
    dates = pd.date_range(dmin, dmax, freq="D")
    df = pd.DataFrame({"date": dates, "focos": f, "crudo": c})
    df["year"], df["month"], df["day"] = df.date.dt.year, df.date.dt.month, df.date.dt.day
    return df, small, int(idx.size)


@st.cache_data(show_spinner=False, max_entries=16)
def zone_cells(wkt, d0, d1):
    data, _, _ = load_data()
    idx, _ = cell_selection(data, wkt)
    days = data["days"][idx]
    a = np.datetime64(d0, "D").astype(np.int64)
    b = np.datetime64(d1, "D").astype(np.int64)
    idx = idx[(days >= a) & (days <= b)]
    df = pd.DataFrame({"g_r": data["g_r"][idx], "g_c": data["g_c"][idx], "focos": data["focos"][idx],
                       "days": data["days"][idx]})
    return df


def ref_years(dmax):
    last_full = dmax.year if dmax == dt.date(dmax.year, 12, 31) else dmax.year - 1
    return REF_START, last_full


# ------------------------------------------------------------------ cálculos
def monthly_table(daily, d0, d1, dmin, dmax, ref):
    """Tabla año × mes dentro del rango + mediana de referencia por mes."""
    ra, rb = ref
    full = daily.groupby(["year", "month"], as_index=False)[["focos", "crudo"]].sum()
    clim = full[(full.year >= ra) & (full.year <= rb)].groupby("month").focos.median()
    sel = daily[(daily.date.dt.date >= d0) & (daily.date.dt.date <= d1)]
    mt = sel.groupby(["year", "month"], as_index=False).agg(focos=("focos", "sum"), crudo=("crudo", "sum"),
                                                          ndays=("focos", "size"))
    mt["mlen"] = [pd.Period(f"{y}-{m:02d}").days_in_month for y, m in zip(mt.year, mt.month)]
    mt["partial"] = mt.ndays < mt.mlen
    mt["median"] = mt.month.map(clim)
    mt["anom"] = np.where(mt["median"] > 0, (mt.focos - mt["median"]) / mt["median"] * 100, np.nan)
    mt["source"] = [source_of(dt.date(y, m, 15)) for y, m in zip(mt.year, mt.month)]
    return mt, clim


def compare_month(daily, y, m, dmax, ref):
    ra, rb = ref
    mlen = pd.Period(f"{y}-{m:02d}").days_in_month
    D = dmax.day if (y, m) == (dmax.year, dmax.month) else mlen
    cur = daily[(daily.year == y) & (daily.month == m) & (daily.day <= D)].focos.sum()
    vals = {}
    for yy in range(ra, rb + 1):
        if yy == y:
            continue
        vals[yy] = daily[(daily.year == yy) & (daily.month == m) & (daily.day <= D)].focos.sum()
    v = np.array(list(vals.values()))
    med = float(np.median(v)) if v.size else np.nan
    pct = float((np.sum(v < cur) + 0.5 * np.sum(v == cur)) / v.size * 100) if v.size else np.nan
    rank = int(np.sum(v > cur)) + 1
    return dict(D=D, cur=float(cur), med=med, p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                pct=pct, rank=rank, n=v.size + 1, vals=vals,
                anom=(cur - med) / med * 100 if med > 0 else np.nan)


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.0f}"
    return s.replace(",", ".") if st.session_state.get("lang", "es") == "es" else s


def readme(T, desc, units):
    return (f"{desc}\n\nUnits / Unidades: {units}\n\n"
            "Method / Método: MODIS (Terra+Aqua, 1 km) and VIIRS (S-NPP, NOAA-20, 375 m) active-fire hotspots from "
            "NASA FIRMS, filtered (nominal/high confidence, vegetation fires, static sources removed), counted as "
            "1-km cell-days and harmonized to the VIIRS S-NPP scale with region-specific (1°x1°) factors calibrated "
            "on 2012-2017 and validated on 2018-2020.\n\n"
            f"Data acknowledgment:\n{ACK}\n\n{T['footer']}\n")


def zip_bytes(name, df, T, desc, units):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.csv", df.to_csv(index=False))
        z.writestr("LEEME_README.txt", readme(T, desc, units))
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
lang = st.sidebar.radio("🌐", ["Español", "English"], horizontal=True, label_visibility="collapsed")
L = "es" if lang == "Español" else "en"
st.session_state["lang"] = L
T = TXT[L]
MONTHS = T["months"]

try:
    data, DMIN, DMAX = load_data()
except FileNotFoundError:
    st.error(f"No encuentro los datos de la app: {DATA_APP}\n\nCorré primero `python app/preparar_datos_app.py`.")
    st.stop()
REF = ref_years(DMAX)

st.sidebar.header(T["zone"])
mode = st.sidebar.radio(T["zone_mode"], [T["m_draw"], T["m_preset"], T["m_coords"]])
geom = None
if mode == T["m_preset"]:
    pname = st.sidebar.selectbox(T["preset"], list(PRESETS))
    a, b, c, d = PRESETS[pname]
    geom = box(c, a, d, b)
    st.sidebar.caption(T["preset_note"])
elif mode == T["m_coords"]:
    c1, c2 = st.sidebar.columns(2)
    la0 = c1.number_input(T["lat_min"], -60.0, 15.0, -37.6, 0.1)
    la1 = c2.number_input(T["lat_max"], -60.0, 15.0, -32.0, 0.1)
    lo0 = c1.number_input(T["lon_min"], -85.0, -30.0, -70.6, 0.1)
    lo1 = c2.number_input(T["lon_max"], -85.0, -30.0, -66.5, 0.1)
    if la1 > la0 and lo1 > lo0:
        geom = box(lo0, la0, lo1, la1)
else:
    st.sidebar.info(T["draw_help"])

st.sidebar.header(T["period"])
rng = st.sidebar.date_input(T["range"], value=(DMIN, DMAX), min_value=DMIN, max_value=DMAX, format="DD/MM/YYYY")
D0, D1 = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (DMIN, DMAX))

with st.sidebar.expander(T["options"]):
    excl_terra = st.checkbox(T["excl_terra"], value=False)
    ramp = st.selectbox(T["ramp"], list(RAMPS), index=0)
    logscale = st.checkbox(T["log"], value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Data: NASA FIRMS (MODIS C6.1, VIIRS 375 m S-NPP / NOAA-20). " + T["footer"])

# ------------------------------------------------------------------ cabecera + mapa de selección
st.title(T["title"])
st.caption(T["subtitle"])

fmap = folium.Map(location=[-20, -60], zoom_start=3, tiles="OpenStreetMap", control_scale=True,
                  scrollWheelZoom=False)
if geom is not None:
    folium.GeoJson(shapely.geometry.mapping(geom),
                   style_function=lambda _: {"color": C_CUR, "weight": 2, "fillOpacity": 0.08}).add_to(fmap)
    minx, miny, maxx, maxy = geom.bounds
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])
if mode == T["m_draw"]:
    Draw(export=False, draw_options={"polyline": False, "circle": False, "marker": False,
                                     "circlemarker": False, "polygon": True, "rectangle": True},
         edit_options={"edit": False}).add_to(fmap)
out = st_folium(fmap, height=380, use_container_width=True, key=f"map_{mode}",
                returned_objects=["last_active_drawing"] if mode == T["m_draw"] else [])
if mode == T["m_draw"] and out and out.get("last_active_drawing"):
    g = out["last_active_drawing"].get("geometry")
    if g and g.get("type") in ("Polygon", "MultiPolygon"):
        geom = shape(g)

if geom is None or geom.is_empty:
    st.info(T["no_zone"])
    st.stop()

WKT = shapely.to_wkt(geom, rounding_precision=4)
daily, small, nrows = zone_daily(WKT)
if small:
    st.warning(T["small_zone"])
if nrows == 0 or daily.focos.sum() == 0:
    st.warning(T["no_data"])
    st.stop()

mt, clim = monthly_table(daily, D0, D1, DMIN, DMAX, REF)
if excl_terra:
    mt = mt[mt.source != "Terra"]
annual_ref = daily[(daily.year >= REF[0]) & (daily.year <= REF[1])].groupby("year").focos.sum()
if annual_ref.median() < 100:
    st.warning(T["low_counts"])

# ------------------------------------------------------------------ KPIs
cmp_now = compare_month(daily, DMAX.year, DMAX.month, DMAX, REF)
yr = mt[(mt.year >= REF[0]) & (mt.year <= REF[1])].groupby("year").focos.sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric(T["k_total"], fmt(mt.focos.sum()), help=T["k_total_h"])
k2.metric(T["k_peak"], MONTHS[int(clim.idxmax()) - 1] if clim.max() > 0 else "–")
k3.metric(T["k_year"], str(int(yr.idxmax())) if len(yr) else "–", fmt(yr.max()) if len(yr) else None,
          delta_color="off")
k4.metric(f"{T['k_now']} ({MONTHS[DMAX.month - 1]} {DMAX.year})", fmt(cmp_now["cur"]),
          None if np.isnan(cmp_now["anom"]) else f"{cmp_now['anom']:+.0f} %",
          delta_color="inverse", help=T["k_now_h"].format(a=REF[0], b=REF[1]))

tabs = st.tabs(T["tabs"])

# ------------------------------------------------------------------ 1. calendario
with tabs[0]:
    cmode = st.radio(T["cal_mode"], [T["cal_abs"], T["cal_anom"]], horizontal=True)
    years = sorted(mt.year.unique())
    ylab = [str(y) for y in years]
    piv = lambda col: mt.pivot(index="year", columns="month", values=col).reindex(index=years, columns=range(1, 13))
    # tope de color: meses que no son sólo-Terra (si no, 2000-2002 aplasta el resto)
    zcap = mt.loc[mt.source != "Terra", "focos"].max() if (mt.source != "Terra").any() else mt.focos.max()
    Z, MED, AN = piv("focos"), piv("median"), piv("anom")
    SRC, PART = piv("source"), piv("partial")
    custom = np.dstack([MED.values, AN.values,
                        np.vectorize(lambda s: T["src"].get(s, "") if isinstance(s, str) else "")(SRC.values.astype(object)),
                        np.where(PART.values == True, "✓", "")])
    if cmode == T["cal_abs"]:
        zv = np.log10(1 + Z.values) if logscale else Z.values
        cb = dict(title=T["focos"])
        if logscale:
            mx = np.nanmax(Z.values)
            ticks = [v for v in [0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6] if v <= mx * 1.01]
            cb.update(tickvals=[np.log10(1 + v) for v in ticks], ticktext=[fmt(v) for v in ticks])
        zmax = np.log10(1 + zcap) if logscale else zcap
        hm = go.Heatmap(z=zv, x=MONTHS, y=ylab, colorscale=RAMPS[ramp], colorbar=cb, customdata=custom,
                        zmin=0, zmax=zmax,
                        text=Z.values, xgap=1, ygap=1,
                        hovertemplate=(f"<b>%{{x}} %{{y}}</b><br>{T['focos']}: %{{text:,.0f}}<br>"
                                       f"{T['median']}: %{{customdata[0]:,.0f}}<br>{T['anom']}: %{{customdata[1]:+.0f}} %<br>"
                                       f"{T['source']}: %{{customdata[2]}}<br>{T['partial']}: %{{customdata[3]}}<extra></extra>"))
    else:
        lim = np.nanpercentile(np.abs(AN.values), 95) if np.isfinite(AN.values).any() else 100
        hm = go.Heatmap(z=AN.values, x=MONTHS, y=ylab, colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
                        colorbar=dict(title=f"{T['anom']} (%)"), customdata=custom, text=Z.values, xgap=1, ygap=1,
                        hovertemplate=(f"<b>%{{x}} %{{y}}</b><br>{T['anom']}: %{{z:+.0f}} %<br>"
                                       f"{T['focos']}: %{{text:,.0f}}<br>{T['median']}: %{{customdata[0]:,.0f}}<br>"
                                       f"{T['source']}: %{{customdata[2]}}<extra></extra>"))
    fig = go.Figure(hm)
    terra = mt[mt.source == "Terra"]
    part = mt[mt.partial]
    if len(terra):
        fig.add_trace(go.Scatter(x=[MONTHS[m - 1] for m in terra.month], y=terra.year.astype(str), mode="markers",
                                 marker=dict(symbol="x-thin", size=8, line=dict(width=1.5, color="white")),
                                 hoverinfo="skip", showlegend=False))
    if len(part):
        fig.add_trace(go.Scatter(x=[MONTHS[m - 1] for m in part.month], y=part.year.astype(str), mode="markers",
                                 marker=dict(symbol="circle-open", size=9, line=dict(width=1.5, color="#333")),
                                 hoverinfo="skip", showlegend=False))
    fig.update_yaxes(autorange="reversed", type="category", title=T["year"])
    fig.update_xaxes(side="top")
    fig.update_layout(height=max(380, 22 * len(years) + 120), margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white")
    st.plotly_chart(fig, config={"toImageButtonOptions": {"filename": "calendario_focos"}})
    st.caption(T["cal_note"])

# ------------------------------------------------------------------ 2. mes actual vs historia
with tabs[1]:
    opts = [(y, m) for y in range(DMAX.year, REF_START - 1, -1) for m in range(12, 0, -1)
            if dt.date(y, m, 1) <= DMAX]
    ym = st.selectbox(T["cmp_pick"], opts, format_func=lambda t: f"{MONTHS[t[1] - 1]} {t[0]}")
    y, m = ym
    C = compare_month(daily, y, m, DMAX, REF)
    st.caption(T["cmp_days"].format(d=C["D"]))
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(T["cmp_cur"], fmt(C["cur"]), None if np.isnan(C["anom"]) else f"{C['anom']:+.0f} %", delta_color="inverse")
    a2.metric(T["cmp_med"].format(a=REF[0], b=REF[1]), fmt(C["med"]))
    a3.metric(T["cmp_pct"], "–" if np.isnan(C["pct"]) else f"P{C['pct']:.0f}")
    a4.metric(T["cmp_rank"], T["cmp_rank_v"].format(r=C["rank"], n=C["n"]))

    col1, col2 = st.columns(2)
    # acumulado diario
    mlen = pd.Period(f"{y}-{m:02d}").days_in_month
    cums = []
    for yy in range(REF[0], REF[1] + 1):
        if yy == y:
            continue
        s = daily[(daily.year == yy) & (daily.month == m)].set_index("day").focos.reindex(range(1, mlen + 1)).fillna(0)
        cums.append(s.cumsum().values)
    cums = np.array(cums)
    dd = np.arange(1, mlen + 1)
    q25, q50, q75 = np.percentile(cums, [25, 50, 75], axis=0)
    cur = daily[(daily.year == y) & (daily.month == m) & (daily.day <= C["D"])].set_index("day").focos.cumsum()
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=np.r_[dd, dd[::-1]], y=np.r_[q75, q25[::-1]], fill="toself", line=dict(width=0),
                            fillcolor="rgba(120,120,120,0.25)", name=T["iqr"], hoverinfo="skip"))
    f2.add_trace(go.Scatter(x=dd, y=q50, line=dict(color="#555", dash="dash"),
                            name=T["cmp_med"].format(a=REF[0], b=REF[1])))
    f2.add_trace(go.Scatter(x=cur.index, y=cur.values, line=dict(color=C_CUR, width=3),
                            name=f"{MONTHS[m - 1]} {y}"))
    f2.update_layout(title=T["cum_title"], xaxis_title=T["day"], yaxis_title=T["focos"], height=380,
                     legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=50, b=10),
                     hovermode="x unified")
    col1.plotly_chart(f2)
    # barras por año
    vy = pd.Series(C["vals"]).sort_index()
    vy.loc[y] = C["cur"]
    vy = vy.sort_index()
    f3 = go.Figure(go.Bar(x=vy.index, y=vy.values, marker_color=[C_CUR if k == y else "#B0B7C3" for k in vy.index],
                          hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    f3.add_hline(y=C["med"], line_dash="dash", line_color="#555",
                 annotation_text=T["cmp_med"].format(a=REF[0], b=REF[1]), annotation_position="top left")
    f3.update_layout(title=f"{T['bars_title']} ({MONTHS[m - 1]} 1–{C['D']})", yaxis_title=T["focos"], height=380,
                     margin=dict(l=10, r=10, t=50, b=10))
    col2.plotly_chart(f3)

# ------------------------------------------------------------------ 3. estacionalidad
with tabs[2]:
    full = daily.groupby(["year", "month"], as_index=False).focos.sum()
    ref_m = full[(full.year >= REF[0]) & (full.year <= REF[1])]
    q = ref_m.groupby("month").focos.quantile([0.25, 0.5, 0.75]).unstack()
    yrs = sorted(full.year.unique(), reverse=True)
    hy = st.selectbox(T["seas_year"], yrs, index=0)
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=MONTHS + MONTHS[::-1], y=np.r_[q[0.75].values, q[0.25].values[::-1]], fill="toself",
                            fillcolor="rgba(224,123,57,0.25)", line=dict(width=0), name=T["iqr"], hoverinfo="skip"))
    f4.add_trace(go.Scatter(x=MONTHS, y=q[0.5].values, line=dict(color=C_VIIRS, width=3), name=T["median"]))
    hy_s = full[full.year == hy].set_index("month").focos.reindex(range(1, 13))
    f4.add_trace(go.Scatter(x=MONTHS, y=hy_s.values, line=dict(color=C_CUR, width=2), mode="lines+markers",
                            name=str(hy)))
    f4.update_layout(title=T["seas_title"].format(a=REF[0], b=REF[1]), yaxis_title=T["focos"], height=420,
                     hovermode="x unified", legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(f4)

# ------------------------------------------------------------------ 4. serie anual
with tabs[3]:
    show_raw = st.checkbox(T["ann_raw"], value=True)
    ya = mt.groupby("year").agg(focos=("focos", "sum"), crudo=("crudo", "sum"), ndays=("ndays", "sum")).reset_index()
    ya["full"] = [nd >= (366 if pd.Timestamp(f"{yy}-12-31").is_leap_year else 365) for yy, nd in zip(ya.year, ya.ndays)]
    ya["color"] = [C_MODIS if yy < 2012 else C_VIIRS for yy in ya.year]
    ya.loc[ya.year <= 2002, "color"] = "#A9C6EE"
    ya.loc[~ya.full, "color"] = "#D0D0D0"
    note = np.where(ya.full, "", T["partial_y"]).astype(object)
    note[ya.year.values <= 2002] = note[ya.year.values <= 2002] + " · " + T["terra_y"]
    f5 = go.Figure(go.Bar(x=ya.year, y=ya.focos, marker_color=ya.color, name=T["harm"],
                          customdata=note,
                          hovertemplate="%{x}: %{y:,.0f} %{customdata}<extra></extra>"))
    if show_raw:
        f5.add_trace(go.Scatter(x=ya.year, y=ya.crudo, mode="lines+markers", line=dict(color=C_RAW, dash="dot"),
                                name=T["raw"]))
    f5.add_vline(x=2011.5, line_dash="dot", line_color="#888", annotation_text="MODIS → VIIRS",
                 annotation_position="top")
    f5.update_layout(title=T["ann_title"], yaxis_title=T["focos"], height=420, legend=dict(orientation="h", y=-0.15),
                     margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(f5)
    st.caption(T["ann_note"])

# ------------------------------------------------------------------ 5. mapa
with tabs[4]:
    cells = zone_cells(WKT, D0, D1)
    agg = cells.groupby(["g_r", "g_c"], as_index=False).focos.sum()
    agg = agg[agg.focos > 0].sort_values("focos").tail(40000)
    if len(agg):
        agg["lat"], agg["lon"] = (agg.g_r + 0.5) * GRID, (agg.g_c + 0.5) * GRID
        minx, miny, maxx, maxy = geom.bounds
        span = max(maxx - minx, (maxy - miny) * 1.3, 0.2)
        f6 = go.Figure(go.Scattermap(
            lat=agg.lat, lon=agg.lon, mode="markers",
            marker=dict(size=7, color=np.log10(1 + agg.focos), colorscale=RAMPS[ramp], opacity=0.85,
                        colorbar=dict(title=T["focos"], tickvals=[np.log10(1 + v) for v in [1, 10, 100, 1000, 10000]],
                                      ticktext=["1", "10", "100", "1.000", "10.000"])),
            text=[fmt(v) for v in agg.focos], hovertemplate="%{lat:.2f}, %{lon:.2f}<br>%{text}<extra></extra>"))
        f6.update_layout(map=dict(style="open-street-map", center=dict(lat=(miny + maxy) / 2, lon=(minx + maxx) / 2),
                                  zoom=max(1, min(10, math.log2(360 / span) - 0.6))),
                         height=560, margin=dict(l=0, r=0, t=30, b=0), title=T["map_title"])
        st.plotly_chart(f6)
    else:
        st.info(T["no_data"])

# ------------------------------------------------------------------ 6. noticias
with tabs[5]:
    st.caption(T["n_intro"])
    cells_n = zone_cells(WKT, D0, D1)
    agg_n = cells_n.groupby(["g_r", "g_c"], as_index=False).focos.sum()
    provs, ctrs = noticias.places_for((agg_n.g_r + 0.5) * GRID, (agg_n.g_c + 0.5) * GRID, agg_n.focos)
    if provs or ctrs:
        st.markdown(f"**{T['n_places']}:** " + ", ".join(f"{p} ({noticias.COUNTRY_ES.get(c, c)})" for p, c in provs)
                    + ("" if provs else ", ".join(noticias.COUNTRY_ES.get(c, c) for c in ctrs)))
    else:
        st.info(T["n_none_places"])
    extra = st.text_input(T["n_extra"], placeholder="ej.: Córdoba sierras" if L == "es" else "e.g. Pantanal")
    # meses candidatos: dentro del rango y desde 2017, ordenados por anomalía
    cand = mt[(mt.year >= noticias.GDELT_START.year)].copy()
    cand["exceso"] = cand.focos - cand["median"].fillna(0)   # focos por encima de lo normal
    cand = cand.sort_values(["exceso", "focos"], ascending=False)
    labels = {(int(r.year), int(r.month)): f"{MONTHS[int(r.month) - 1]} {int(r.year)}"
              + ("" if np.isnan(r.anom) else f" ({r.anom:+.0f} %)") for r in cand.itertuples()}
    keys = list(labels)
    if (D1 - D0).days <= 92 and D1 >= noticias.GDELT_START:
        default = [k for k in keys if dt.date(k[0], k[1], 1) <= D1][:3]
    else:
        default = keys[:3]
    sel_months = st.multiselect(T["n_months"], keys, default=default, max_selections=3,
                                format_func=lambda k: labels[k])
    if D0 < noticias.GDELT_START:
        st.caption(T["n_pre2017"])
    if st.button(T["n_btn"], type="primary"):
        places = [p for p, _ in provs] + ctrs
        q = noticias.gdelt_query(places, extra)
        res = {"press": [], "errors": [], "wiki": []}
        with st.spinner(T["n_wait"]):
            for i, (yy, mm) in enumerate(sorted(sel_months)):
                a = max(dt.date(yy, mm, 1), D0)
                b = min(dt.date(yy, mm, pd.Period(f"{yy}-{mm:02d}").days_in_month), D1)
                if "rate" in res["errors"]:      # GDELT nos está limitando: no insistir
                    res["press"].append(((yy, mm), []))
                    continue
                if i:
                    time.sleep(5.5)
                arts, err = noticias.gdelt_search(q, a, b)
                res["press"].append(((yy, mm), arts))
                if err:
                    res["errors"].append(err)
            res["wiki"] = noticias.wiki_events(provs, ctrs, D0.year, D1.year)
        st.session_state["news"] = res
    res = st.session_state.get("news")
    if res:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader(T["n_press"])
            if any(e == "rate" for e in res["errors"]):
                st.warning(T["n_rate"])
            elif res["errors"]:
                st.warning(T["n_err"].format(e=res["errors"][0]))
            st.caption(T["n_media"])
            for (yy, mm), arts in res["press"]:
                raw_n = len(arts)
                st.markdown(f"**{MONTHS[mm - 1]} {yy}**")
                if not arts:
                    st.caption(T["n_empty"])
                arts = noticias.filtrar_medios(arts, ctrs)
                if not arts and raw_n:
                    st.caption(T["n_filtered"].format(n=raw_n))
                for a in arts:
                    d = a["date"]
                    dd = f"{d[6:8]}/{d[4:6]}/{d[:4]}" if len(d) == 8 else ""
                    st.markdown(f"- [{a['title'] or a['url']}]({a['url']})  \n  <small>{'📍 ' if a.get('origen') == 'local' else '🌐 '}{a['domain']} · {dd} · {a['lang']}</small>",
                                unsafe_allow_html=True)
        with c2:
            st.subheader(T["n_wiki"])
            if not res["wiki"]:
                st.caption(T["n_empty"])
            for a in res["wiki"]:
                st.markdown(f"- [{a['title']}]({a['url']})  \n  <small>Wikipedia ({a['lang']})</small>",
                            unsafe_allow_html=True)
        st.caption("Fuentes / Sources: GDELT Project (gdeltproject.org) · Wikipedia (CC BY-SA) · "
                   "Natural Earth (public domain).")

# ------------------------------------------------------------------ 7. descargas
with tabs[6]:
    st.caption(T["dl_intro"])
    tag = f"{D0:%Y%m%d}_{D1:%Y%m%d}"
    units = "harmonized hotspots = 1-km cells with fire per day, VIIRS S-NPP scale"
    desc_zone = f"Area (WKT, EPSG:4326): {WKT}\nPeriod: {D0} to {D1}"

    # 1) mensual
    out_m = mt[["year", "month", "focos", "crudo", "median", "anom", "source", "partial"]].rename(columns={
        "focos": "focos_armonizados", "crudo": "celdas_dia_sin_armonizar", "median": f"mediana_{REF[0]}_{REF[1]}",
        "anom": "anomalia_pct", "source": "sensor", "partial": "mes_incompleto"})
    st.download_button(f"⬇️ {T['dl_month']}", zip_bytes(f"calendario_mensual_{tag}", out_m.round(2), T,
                                                         desc_zone, units),
                       file_name=f"calendario_mensual_{tag}.zip", mime="application/zip")

    # 2) diario por celda 0,1°
    st.markdown(f"**{T['dl_daily']}**")
    if st.button(T["dl_prep"], key="p_daily"):
        c = zone_cells(WKT, D0, D1)
        dfc = pd.DataFrame({"fecha": c.days.values.astype("datetime64[D]"),
                            "lat": ((c.g_r + 0.5) * GRID).round(2), "lon": ((c.g_c + 0.5) * GRID).round(2),
                            "focos_armonizados": c.focos.round(3)}).sort_values(["fecha", "lat", "lon"])
        st.session_state["z_daily"] = zip_bytes(f"focos_diarios_0p1_{tag}", dfc, T, desc_zone, units)
        st.caption(T["dl_rows"].format(n=len(dfc)))
    if "z_daily" in st.session_state:
        st.download_button(T["dl_get"], st.session_state["z_daily"], file_name=f"focos_diarios_0p1_{tag}.zip",
                           mime="application/zip", key="d_daily")

    minx, miny, maxx, maxy = geom.bounds
    # 3) 1 km armonizado
    st.markdown(f"**{T['dl_1km']}**")
    if not os.path.exists(DATA_1KM):
        st.caption(T["dl_missing"].format(f=os.path.basename(DATA_1KM)))
    elif st.button(T["dl_prep"], key="p_1km"):
        q = duckdb.sql(f"""SELECT acq_date AS fecha, CAST(lat AS DOUBLE) lat, CAST(lon AS DOUBLE) lon, fuente,
                                  round(peso, 4) AS focos_armonizados
                           FROM read_parquet('{DATA_1KM}')
                           WHERE lat BETWEEN {miny} AND {maxy} AND lon BETWEEN {minx} AND {maxx}
                             AND acq_date BETWEEN '{D0}' AND '{D1}'""").df()
        q = q[shapely.contains_xy(geom, q.lon.values, q.lat.values)] if len(q) else q
        if len(q) > MAX_RAW_ROWS:
            st.warning(T["dl_toolarge"].format(n=len(q), m=MAX_RAW_ROWS))
        else:
            st.session_state["z_1km"] = zip_bytes(f"focos_armonizados_1km_{tag}", q.sort_values("fecha"), T,
                                                  desc_zone, units)
            st.caption(T["dl_rows"].format(n=len(q)))
    if "z_1km" in st.session_state:
        st.download_button(T["dl_get"], st.session_state["z_1km"], file_name=f"focos_armonizados_1km_{tag}.zip",
                           mime="application/zip", key="d_1km")

    # 4) focos originales FIRMS
    st.markdown(f"**{T['dl_raw']}**")
    if not os.path.isdir(DATA_RAW):
        st.caption(T["dl_missing"].format(f="data/parquet"))
    elif st.button(T["dl_prep"], key="p_raw"):
        q = duckdb.sql(f"""SELECT latitude, longitude, datetime_utc, satellite, sensor, conf_raw AS confidence,
                                  conf_class, frp, daynight, type
                           FROM read_parquet('{DATA_RAW}/**/*.parquet', hive_partitioning=true)
                           WHERE year BETWEEN {D0.year} AND {D1.year}
                             AND latitude BETWEEN {miny} AND {maxy} AND longitude BETWEEN {minx} AND {maxx}
                             AND acq_date BETWEEN '{D0}' AND '{D1}'
                             AND satellite IN ('Terra', 'Aqua', 'SNPP', 'NOAA20')""").df()
        q = q[shapely.contains_xy(geom, q.longitude.values, q.latitude.values)] if len(q) else q
        if len(q) > MAX_RAW_ROWS:
            st.warning(T["dl_toolarge"].format(n=len(q), m=MAX_RAW_ROWS))
        else:
            st.session_state["z_raw"] = zip_bytes(f"focos_firms_originales_{tag}", q.sort_values("datetime_utc"), T,
                                                  desc_zone, "one row per FIRMS hotspot (pixel), not harmonized")
            st.caption(T["dl_rows"].format(n=len(q)))
    if "z_raw" in st.session_state:
        st.download_button(T["dl_get"], st.session_state["z_raw"], file_name=f"focos_firms_originales_{tag}.zip",
                           mime="application/zip", key="d_raw")

# ------------------------------------------------------------------ pie
st.markdown("---")
st.caption(f"{T['footer']}  \n{ACK}")
