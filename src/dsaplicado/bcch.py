"""Cliente de la API de la Base de Datos Estadísticos (BDE) del Banco Central de Chile.

La API requiere registro gratuito en si3.bcentral.cl. El token se lee, en este
orden, desde la variable de entorno BCCH_TOKEN, desde la línea BCCH_TOKEN del
archivo .env o desde el archivo api_banco_central.txt con la línea token=...
Los dos archivos están excluidos de git y nunca deben subirse.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL_API = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"


class ErrorBCCh(RuntimeError):
    """Error al consultar la API del Banco Central."""


def leer_token(raiz: str | Path | None = None) -> str:
    """Obtiene el token sin mostrarlo."""
    if os.environ.get("BCCH_TOKEN"):
        return os.environ["BCCH_TOKEN"].strip()
    raiz = Path(raiz or Path.cwd())
    for archivo, patron in ((".env", r"^\s*BCCH_TOKEN\s*=\s*(\S+)"), ("api_banco_central.txt", r"^\s*token\s*[=:]\s*(\S+)")):
        ruta = raiz / archivo
        if ruta.exists():
            coincidencia = re.search(patron, ruta.read_text(encoding="utf-8-sig"), re.M | re.I)
            if coincidencia:
                return coincidencia.group(1).strip("\"'")
    raise ErrorBCCh("No se encontró el token del Banco Central en el entorno, en .env ni en api_banco_central.txt.")


def parsear_serie(respuesta: dict) -> pd.Series:
    """Convierte la respuesta de GetSeries en una serie numérica, sin los días sin dato."""
    if respuesta.get("Codigo") != 0:
        raise ErrorBCCh(f"La API respondió con código {respuesta.get('Codigo')}: {respuesta.get('Descripcion')}")
    info = respuesta["Series"]
    observaciones = pd.DataFrame(info.get("Obs") or [])
    if observaciones.empty:
        return pd.Series(dtype=float, name=info.get("seriesId"))
    observaciones = observaciones[observaciones["statusCode"] == "OK"]
    valores = pd.to_numeric(observaciones["value"], errors="coerce")
    fechas = pd.to_datetime(observaciones["indexDateString"], format="%d-%m-%Y")
    return pd.Series(valores.to_numpy(), index=fechas, name=info.get("seriesId")).dropna().sort_index()


class ClienteBCCh:
    """Cliente con una pausa corta entre consultas. El token nunca aparece en los mensajes."""

    def __init__(self, token: str | None = None, pausa: float = 1.0):
        self._token = token or leer_token()
        self._pausa = pausa
        self._ultima = 0.0

    def serie(self, codigo: str, inicio: date, fin: date) -> pd.Series:
        espera = self._pausa - (time.monotonic() - self._ultima)
        if espera > 0:
            time.sleep(espera)
        parametros = {"token": self._token, "function": "GetSeries", "timeseries": codigo,
                      "firstdate": f"{inicio:%Y-%m-%d}", "lastdate": f"{fin:%Y-%m-%d}"}
        try:
            resp = requests.get(URL_API, params=parametros, timeout=120)
        except requests.RequestException as error:
            raise ErrorBCCh(f"Falló la conexión al pedir {codigo}: {type(error).__name__}") from None
        self._ultima = time.monotonic()
        try:
            respuesta = resp.json()
        except ValueError:
            raise ErrorBCCh(f"La API no devolvió JSON para {codigo}. Revisa que el token sea válido.") from None
        return parsear_serie(respuesta)


def cargar_series(
    codigos: dict[str, str], inicio: date, fin: date, archivo: str | Path, cliente: ClienteBCCh | None = None
) -> pd.DataFrame:
    """Descarga varias series y las guarda juntas en un CSV de caché.

    ``codigos`` mapea el código del Banco Central a un nombre de columna. Una
    serie que falla se omite con un aviso, para no perder las demás.
    """
    ruta = Path(archivo)
    if ruta.exists():
        return pd.read_csv(ruta, index_col="fecha", parse_dates=True)
    cliente = cliente or ClienteBCCh()
    columnas = {}
    for codigo, nombre in codigos.items():
        try:
            columnas[nombre] = cliente.serie(codigo, inicio, fin)
        except ErrorBCCh as error:
            warnings.warn(f"Se omite {codigo}: {error}", stacklevel=2)
    if not columnas:
        raise ErrorBCCh("Ninguna serie devolvió datos; no se guarda en caché.")
    tabla = pd.DataFrame(columnas)
    tabla.index.name = "fecha"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta)
    return tabla
