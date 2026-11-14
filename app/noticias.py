"""
Búsqueda de noticias y eventos de incendios para una zona y período.

Fuentes (abiertas, sin clave):
  * GDELT DOC 2.0 API — prensa mundial, desde 2017. Límite: 1 consulta cada 5 s.
  * Wikipedia (MediaWiki API) — eventos destacados, cualquier año.
Lugares: Natural Earth admin-1 (dominio público), simplificado a Sudamérica.

Sólo se muestran titulares y enlaces; el contenido pertenece a cada medio.
"""
import datetime as dt
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import shapely
from shapely.geometry import shape

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN1 = os.path.join(APP_DIR, "data", "sa_admin1.geojson")
CACHE = os.path.join(APP_DIR, "data", "cache_noticias.json")
UA = {"User-Agent": "FireCal/0.1 (NASA Space Apps 2026 team project; github.com/lautarodiazopazo/firecal)"}
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_START = dt.date(2017, 1, 1)

FIRE_TERMS = '(wildfire OR wildfires OR "forest fire" OR incendio OR incendios OR "incendio forestal" OR queimada OR queimadas)'
TITLE_FIRE = re.compile(r"incendi|incêndi|fuego|quema|queimad|wildfire|fire|bushfire|forest fire", re.I)
YEAR = re.compile(r"\b(19|20)\d{2}\b")

COUNTRY_PT = {"Brazil": "Brasil"}
COUNTRY_ES = {"Argentina": "Argentina", "Brazil": "Brasil", "Chile": "Chile", "Bolivia": "Bolivia",
              "Paraguay": "Paraguay", "Uruguay": "Uruguay", "Peru": "Perú", "Colombia": "Colombia",
              "Venezuela": "Venezuela", "Ecuador": "Ecuador", "Guyana": "Guyana", "Suriname": "Surinam",
              "France": "Guayana Francesa"}


def _plain(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ------------------------------------------------------------------ caché en disco (best effort)
def _cache_load():
    try:
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _cache_get(key, max_age_h=72):
    c = _cache_load().get(key)
    if c and time.time() - c["t"] < max_age_h * 3600:
        return c["v"]
    return None


def _cache_put(key, value):
    try:
        c = _cache_load()
        c[key] = {"t": time.time(), "v": value}
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(c, fh, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ lugares
_ADMIN = None


def _admin():
    global _ADMIN
    if _ADMIN is None:
        with open(ADMIN1, encoding="utf-8") as fh:
            fc = json.load(fh)
        _ADMIN = [(f["properties"], shape(f["geometry"])) for f in fc["features"]]
    return _ADMIN


def places_for(lat, lon, w, top=3):
    """Provincias y países donde cayó la actividad (ponderado por focos)."""
    lat, lon, w = np.asarray(lat), np.asarray(lon), np.asarray(w, dtype=float)
    prov, country = {}, {}
    if lat.size == 0:
        return [], []
    for props, geom in _admin():
        minx, miny, maxx, maxy = geom.bounds
        m = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        if not m.any():
            continue
        inside = shapely.contains_xy(geom, lon[m], lat[m])
        s = float(w[m][inside].sum())
        if s > 0:
            prov[(props["prov"], props["country"])] = s
            country[props["country"]] = country.get(props["country"], 0) + s
    tot = sum(country.values()) or 1
    provs = sorted(prov.items(), key=lambda kv: -kv[1])
    provs = [k for k, v in provs if v / tot >= 0.05][:top]
    ctrs = [k for k, v in sorted(country.items(), key=lambda kv: -kv[1]) if v / tot >= 0.10][:2]
    return provs, ctrs


# ------------------------------------------------------------------ GDELT
def gdelt_query(places, extra=""):
    names = sorted({_plain(p) for p in places if p})
    loc = " OR ".join(f'"{n}"' if " " in n else n for n in names)
    q = FIRE_TERMS
    if loc:
        q += f" ({loc})" if len(names) > 1 else f" {loc}"
    if extra.strip():
        q += " " + extra.strip()
    return q


def gdelt_search(query, d0, d1, maxrecords=40, retries=2, wait=6.0):
    """Devuelve (artículos, error). d0/d1: date."""
    key = f"gdelt|{query}|{d0}|{d1}|{maxrecords}"
    hit = _cache_get(key)
    if hit is not None:
        return hit, None
    params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": maxrecords,
              "sort": "HybridRel", "startdatetime": f"{d0:%Y%m%d}000000", "enddatetime": f"{d1:%Y%m%d}235959"}
    err = None
    for i in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, headers=UA, timeout=15)
            if r.status_code == 429:
                err = "rate"
                if i < retries - 1:
                    time.sleep(wait)
                continue
            r.raise_for_status()
            txt = r.text.strip()
            arts = json.loads(txt).get("articles", []) if txt.startswith("{") else []
            out = [{"title": a.get("title", "").strip(), "url": a.get("url"), "domain": a.get("domain"),
                    "date": a.get("seendate", "")[:8], "lang": a.get("language", ""),
                    "country": a.get("sourcecountry", "")} for a in arts if a.get("url")]
            _cache_put(key, out)
            return out, None
        except Exception as e:  # noqa: BLE001
            err = str(e)[:120]
            time.sleep(2)
    return [], err


# ------------------------------------------------------------------ Wikipedia
def wiki_search(term, lang, limit=30):
    key = f"wiki|{lang}|{term}|{limit}"
    hit = _cache_get(key, max_age_h=24 * 14)
    if hit is not None:
        return hit
    try:
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": term, "format": "json",
                                 "srlimit": limit, "srprop": "snippet"}, headers=UA, timeout=10)
        r.raise_for_status()
        res = [{"title": s["title"], "lang": lang,
                "url": f"https://{lang}.wikipedia.org/wiki/{s['title'].replace(' ', '_')}",
                "snippet": re.sub(r"<[^>]+>", "", s.get("snippet", ""))}
               for s in r.json().get("query", {}).get("search", [])]
        _cache_put(key, res)
        return res
    except Exception:  # noqa: BLE001
        return []


def wiki_events(provs, countries, y0, y1, max_items=12):
    """Artículos de Wikipedia sobre incendios en esos lugares cuyo título menciona un año del período."""
    terms = []
    for p, c in provs:
        terms += [("es", f"incendios {p}")] + ([("pt", f"incêndios {p}")] if c == "Brazil" else [])
    for c in countries:
        terms += [("es", f"incendios forestales {COUNTRY_ES.get(c, c)}"), ("en", f"wildfires {c}")]
        if c == "Brazil":
            terms += [("pt", "incêndios florestais Brasil"), ("pt", "queimadas Brasil")]
    if y0 == y1:
        terms = [(lang, f"{t} {y0}") for lang, t in terms]
    # nombres que tienen que aparecer en el título o el resumen (sin tildes, en minúscula)
    names = {_plain(p).lower() for p, _ in provs}
    for c in countries:
        names |= {_plain(c).lower(), _plain(COUNTRY_ES.get(c, c)).lower(), _plain(COUNTRY_PT.get(c, c)).lower()}
    if "Brazil" in countries:
        names |= {"amazon", "amazonia", "pantanal", "cerrado"}
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda lt: wiki_search(lt[1], lt[0]), terms))
    seen, out = set(), []
    for res in results:
        for a in res:
            if a["url"] in seen or not TITLE_FIRE.search(a["title"]):
                continue
            text = _plain(a["title"] + " " + a.get("snippet", "")).lower()
            if not any(n in text for n in names):
                continue
            years = [int(m.group(0)) for m in YEAR.finditer(a["title"])]
            if not years or not any(y0 <= y <= y1 for y in years):
                continue
            seen.add(a["url"])
            a["year"] = max(y for y in years if y0 <= y <= y1)
            out.append(a)
    out.sort(key=lambda a: (-a["year"], a["title"]))
    return out[:max_items]
