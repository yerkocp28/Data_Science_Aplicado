"""Descarga y caché de datos públicos: precios de acciones e indicadores de Chile.

Cada función de carga guarda una copia en CSV. Así los notebooks son
reproducibles aunque la fuente cambie o no haya conexión.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# Acciones del IPSA con historia completa desde 2019 en Yahoo Finance.
# Se excluye LATAM (LTM.SN) porque su quiebra y reestructuración de 2020-2022
# distorsiona cualquier estimación de retorno y covarianza.
ACCIONES_IPSA = {
    "BCI.SN": "BCI",
    "BSANTANDER.SN": "Santander",
    "CHILE.SN": "Banco de Chile",
    "CENCOSUD.SN": "Cencosud",
    "CMPC.SN": "CMPC",
    "COPEC.SN": "Copec",
    "ENELCHILE.SN": "Enel Chile",
    "FALABELLA.SN": "Falabella",
    "SQM-B.SN": "SQM-B",
}

# ETF It Now IPSA: replica el índice IPSA y cotiza en pesos chilenos.
BENCHMARK_IPSA = {"CFMITNIPSA.SN": "ETF IPSA"}

URL_MINDICADOR = "https://mindicador.cl/api/{codigo}/{anio}"


def raiz_repositorio(desde: str | Path | None = None) -> Path:
    """Busca hacia arriba la carpeta que contiene pyproject.toml.

    Permite que los notebooks usen rutas relativas al repositorio sin importar
    desde qué carpeta se ejecutan.
    """
    actual = Path(desde or Path.cwd()).resolve()
    for carpeta in (actual, *actual.parents):
        if (carpeta / "pyproject.toml").exists():
            return carpeta
    raise FileNotFoundError("No se encontró la raíz del repositorio (pyproject.toml).")


def descargar_precios(tickers: dict[str, str], inicio: str, fin: str) -> pd.DataFrame:
    """Descarga precios de cierre ajustados desde Yahoo Finance.

    Parameters
    ----------
    tickers : dict
        Mapa de ticker de Yahoo a nombre legible, que se usa como columna.
    inicio, fin : str
        Fechas en formato AAAA-MM-DD. El día final no se incluye.
    """
    import yfinance as yf

    precios = yf.download(
        list(tickers), start=inicio, end=fin, auto_adjust=True, progress=False
    )["Close"]
    precios = precios.rename(columns=tickers)[list(tickers.values())]
    precios.index = pd.to_datetime(precios.index).tz_localize(None)
    precios.index.name = "fecha"
    return precios


def cargar_precios(
    ruta: str | Path,
    tickers: dict[str, str],
    inicio: str,
    fin: str,
    actualizar: bool = False,
) -> pd.DataFrame:
    """Lee precios desde el CSV en caché o los descarga si no existe."""
    ruta = Path(ruta)
    if ruta.exists() and not actualizar:
        return pd.read_csv(ruta, index_col="fecha", parse_dates=True)
    precios = descargar_precios(tickers, inicio, fin)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    precios.to_csv(ruta, float_format="%.6f")
    return precios


def parsear_mindicador(respuesta: dict) -> pd.Series:
    """Convierte la respuesta JSON de mindicador.cl en una serie ordenada por fecha."""
    serie = respuesta.get("serie", [])
    if not serie:
        return pd.Series(dtype=float, name=respuesta.get("codigo"))
    df = pd.DataFrame(serie)
    # Las fechas vienen en UTC a las 03:00 o 04:00, que es medianoche en Chile.
    fechas = pd.to_datetime(df["fecha"], utc=True).dt.tz_convert("America/Santiago")
    df["fecha"] = fechas.dt.tz_localize(None).dt.normalize()
    return (
        df.drop_duplicates("fecha")
        .set_index("fecha")["valor"]
        .astype(float)
        .sort_index()
        .rename(respuesta.get("codigo"))
    )


def descargar_indicador(codigo: str, anios: range | list[int]) -> pd.Series:
    """Descarga un indicador de mindicador.cl, por ejemplo tpm, dolar, ipc o imacec."""
    partes = []
    for anio in anios:
        resp = requests.get(URL_MINDICADOR.format(codigo=codigo, anio=anio), timeout=60)
        resp.raise_for_status()
        partes.append(parsear_mindicador(resp.json()))
    serie = pd.concat(partes).sort_index()
    serie = serie[~serie.index.duplicated()]
    serie.index.name = "fecha"
    return serie.rename(codigo)


def cargar_indicador(
    ruta: str | Path, codigo: str, anios: range | list[int], actualizar: bool = False
) -> pd.Series:
    """Lee un indicador desde el CSV en caché o lo descarga si no existe."""
    ruta = Path(ruta)
    if ruta.exists() and not actualizar:
        return pd.read_csv(ruta, index_col="fecha", parse_dates=True)[codigo]
    serie = descargar_indicador(codigo, anios)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    serie.to_csv(ruta)
    return serie
