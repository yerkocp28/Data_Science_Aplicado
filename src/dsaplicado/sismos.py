"""Datos de sismos de Chile: extracción del catálogo del CSN y consulta al USGS.

El Centro Sismológico Nacional (CSN) publica una página HTML por día en
sismologia.cl. El Servicio Geológico de Estados Unidos (USGS) ofrece una API
pública con el catálogo mundial. Usar ambas fuentes permite validar los datos.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL_CSN = "https://www.sismologia.cl/sismicidad/catalogo/{a:%Y}/{a:%m}/{a:%Y%m%d}.html"
URL_USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CABECERA = {"User-Agent": "Proyecto educativo Data Science Aplicado (github.com/yerkocp28)"}

# Recuadro que cubre Chile continental y su zona de subducción.
RECUADRO_CHILE = {"minlatitude": -56.0, "maxlatitude": -17.0, "minlongitude": -76.0, "maxlongitude": -66.0}

COLUMNAS = ["id", "fecha_utc", "fecha_local", "referencia", "latitud", "longitud", "profundidad_km",
            "magnitud", "tipo_magnitud"]


# --- CSN ---------------------------------------------------------------------------


def parsear_catalogo_diario(html: str) -> pd.DataFrame:
    """Extrae la tabla de sismos de una página diaria del CSN."""
    sopa = BeautifulSoup(html, "html.parser")
    tabla = sopa.find("table", class_="detalle")
    filas = []
    if tabla is None:
        return pd.DataFrame(columns=COLUMNAS)
    for tr in tabla.find_all("tr"):
        celdas = tr.find_all("td")
        if len(celdas) != 5:
            continue
        enlace = celdas[0].find("a")
        local, referencia = celdas[0].get_text("|", strip=True).split("|", 1)
        latitud, longitud = celdas[2].get_text("|", strip=True).split("|")
        magnitud, _, tipo = celdas[4].get_text(strip=True).partition(" ")
        filas.append(
            {
                "id": Path(enlace["href"]).stem if enlace else None,
                "fecha_utc": celdas[1].get_text(strip=True),
                "fecha_local": local,
                "referencia": referencia,
                "latitud": float(latitud),
                "longitud": float(longitud),
                "profundidad_km": float(celdas[3].get_text(strip=True).replace("km", "")),
                "magnitud": float(magnitud),
                "tipo_magnitud": tipo.strip(),
            }
        )
    df = pd.DataFrame(filas, columns=COLUMNAS)
    for columna in ("fecha_utc", "fecha_local"):
        df[columna] = pd.to_datetime(df[columna])
    return df


def descargar_dia_csn(dia: date, sesion: requests.Session, reintentos: int = 3) -> pd.DataFrame:
    """Descarga y parsea el catálogo de un día, con reintentos ante fallas de red."""
    for intento in range(reintentos):
        try:
            resp = sesion.get(URL_CSN.format(a=dia), headers=CABECERA, timeout=30)
            if resp.status_code == 404:
                return pd.DataFrame(columns=COLUMNAS)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return parsear_catalogo_diario(resp.text)
        except requests.RequestException:
            if intento == reintentos - 1:
                raise
            time.sleep(5 * (intento + 1))
    return pd.DataFrame(columns=COLUMNAS)


def cargar_catalogo_csn(anio: int, carpeta: str | Path, pausa: float = 1.0) -> pd.DataFrame:
    """Catálogo del CSN de un año completo, con caché en CSV.

    Hace una solicitud por día con una pausa entre ellas para no sobrecargar el
    sitio. La primera descarga de un año tarda varios minutos.
    """
    ruta = Path(carpeta) / f"csn_{anio}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, parse_dates=["fecha_utc", "fecha_local"], dtype={"id": str})
    sesion = requests.Session()
    partes = []
    dia = date(anio, 1, 1)
    while dia.year == anio and dia <= date.today():
        partes.append(descargar_dia_csn(dia, sesion))
        dia += timedelta(days=1)
        time.sleep(pausa)
    catalogo = pd.concat([p for p in partes if not p.empty], ignore_index=True)
    catalogo = catalogo.drop_duplicates("id").sort_values("fecha_utc").reset_index(drop=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    catalogo.to_csv(ruta, index=False)
    return catalogo


# --- USGS --------------------------------------------------------------------------


def cargar_catalogo_usgs(anio: int, carpeta: str | Path, magnitud_minima: float = 2.5) -> pd.DataFrame:
    """Catálogo del USGS para el recuadro de Chile en un año, con caché en CSV."""
    ruta = Path(carpeta) / f"usgs_{anio}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, parse_dates=["fecha_utc"])
    parametros = {"format": "csv", "starttime": f"{anio}-01-01", "endtime": f"{anio + 1}-01-01",
                  "minmagnitude": magnitud_minima, "orderby": "time-asc", **RECUADRO_CHILE}
    resp = requests.get(URL_USGS, params=parametros, headers=CABECERA, timeout=120)
    resp.raise_for_status()
    crudo = pd.read_csv(io.StringIO(resp.text))
    catalogo = pd.DataFrame({
        "id": crudo["id"],
        "fecha_utc": pd.to_datetime(crudo["time"]).dt.tz_localize(None),
        "latitud": crudo["latitude"],
        "longitud": crudo["longitude"],
        "profundidad_km": crudo["depth"],
        "magnitud": crudo["mag"],
        "tipo_magnitud": crudo["magType"],
        "lugar": crudo["place"],
    })
    ruta.parent.mkdir(parents=True, exist_ok=True)
    catalogo.to_csv(ruta, index=False)
    return catalogo


# --- Análisis --------------------------------------------------------------------------


def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia sobre la superficie terrestre con la fórmula del haversine."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def emparejar_catalogos(
    csn: pd.DataFrame, usgs: pd.DataFrame, segundos: float = 30, km: float = 100
) -> pd.DataFrame:
    """Empareja cada sismo del USGS con el sismo del CSN más cercano en tiempo.

    Se acepta la pareja si ocurre dentro de la ventana de tiempo y de distancia.
    Cada sismo del CSN se usa una sola vez.
    """
    a = usgs.sort_values("fecha_utc").reset_index(drop=True)
    b = csn.sort_values("fecha_utc").reset_index(drop=True)
    unido = pd.merge_asof(
        a, b.add_suffix("_csn"), left_on="fecha_utc", right_on="fecha_utc_csn",
        direction="nearest", tolerance=pd.Timedelta(seconds=segundos),
    )
    unido = unido.dropna(subset=["id_csn"])
    unido["distancia_km"] = distancia_km(unido["latitud"], unido["longitud"], unido["latitud_csn"], unido["longitud_csn"])
    unido["desfase_s"] = (unido["fecha_utc"] - unido["fecha_utc_csn"]).dt.total_seconds()
    unido = unido[unido["distancia_km"] <= km]
    unido = unido.sort_values("desfase_s", key=np.abs).drop_duplicates("id_csn")
    return unido.sort_values("fecha_utc").reset_index(drop=True)


def magnitud_completitud(magnitudes: pd.Series, paso: float = 0.1) -> float:
    """Magnitud de completitud por el método de máxima curvatura.

    Es la magnitud más frecuente del catálogo. Bajo ella, la red no detecta todos
    los sismos. Se suma la corrección habitual de 0,2 propuesta por Woessner y Wiemer.
    """
    redondeadas = (magnitudes / paso).round() * paso
    return float(redondeadas.value_counts().idxmax() + 0.2)


def ajustar_gutenberg_richter(magnitudes: pd.Series, mc: float, paso: float = 0.1) -> dict[str, float]:
    """Valor b de Gutenberg-Richter por máxima verosimilitud, con el estimador de Aki y Utsu.

    La ley dice que log10 N(M >= m) = a - b m. El valor b mide la proporción de
    sismos pequeños frente a grandes y suele estar cerca de uno.
    """
    m = magnitudes[magnitudes >= mc - paso / 2]
    n = len(m)
    b = np.log10(np.e) / (m.mean() - (mc - paso / 2))
    error = 2.3 * b**2 * np.sqrt(((m - m.mean()) ** 2).sum() / (n * (n - 1)))  # Shi y Bolt, 1982
    a = np.log10(n) + b * mc
    return {"b": float(b), "error_b": float(error), "a": float(a), "n": int(n), "mc": float(mc)}
