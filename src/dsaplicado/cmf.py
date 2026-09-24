"""Cliente de la API BEST de la Comisión para el Mercado Financiero (CMF).

La API entrega series y cuadros estadísticos del sistema financiero chileno.
Documentación: https://best.cmfchile.cl/api

La clave es gratuita y personal. Se lee desde la variable de entorno CMF_API_KEY
o desde un archivo .env en la raíz del repositorio. Ese archivo está excluido de
git y nunca debe subirse.

Límites de la API: 10 consultas por minuto, 100 por día y 3.000 por mes. Cada
consulta de rango cubre como máximo doce meses. El cliente respeta el ritmo y
guarda cada cuadro en caché para no repetir consultas.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL_API = "https://apibest.cmfchile.cl/api/v1/"
URL_CATALOGO = "https://best-sbif-api.azurewebsites.net/public/descargar/archivo-publico/catalogo-csv"
PAUSA_MINIMA = 6.5  # segundos entre consultas, para no superar 10 por minuto

MENSAJES_HTTP = {
    401: "La clave no es válida o no fue enviada.",
    403: "La clave no tiene permiso para este recurso.",
    404: "El cuadro o la serie no existe, o no tiene datos en ese rango.",
    429: "Se superó el límite de consultas. Espera un minuto, o hasta mañana si se agotó la cuota diaria.",
}


class ErrorCMF(RuntimeError):
    """Error al consultar la API de la CMF."""


def leer_clave(raiz: str | Path | None = None) -> str:
    """Obtiene la clave desde el entorno o desde el archivo .env, sin mostrarla."""
    if os.environ.get("CMF_API_KEY"):
        return os.environ["CMF_API_KEY"].strip()
    ruta = Path(raiz or Path.cwd()) / ".env"
    if ruta.exists():
        coincidencia = re.search(r"^\s*CMF_API_KEY\s*=\s*(\S+)", ruta.read_text(encoding="utf-8-sig"), re.M)
        if coincidencia:
            return coincidencia.group(1).strip("\"'")
    raise ErrorCMF("No se encontró CMF_API_KEY en el entorno ni en el archivo .env.")


def rangos_anuales(inicio: date, fin: date) -> list[tuple[date, date]]:
    """Divide un período en tramos de hasta doce meses, el máximo que acepta la API."""
    tramos = []
    desde = inicio
    while desde <= fin:
        hasta = min(date(desde.year + 1, desde.month, 1) - timedelta(days=1), fin)
        tramos.append((desde, hasta))
        desde = hasta + timedelta(days=1)
    return tramos


def parsear_cuadro(respuesta: dict) -> pd.DataFrame:
    """Convierte la respuesta de un cuadro en una tabla larga.

    Columnas: fecha, codigo_serie, nombre (la descripción corta, que en los
    cuadros por institución es el nombre del banco) y valor. La API a veces
    repite series dentro de un mismo cuadro; los duplicados se eliminan.
    """
    filas = []
    for serie in respuesta.get("series", []):
        info = serie.get("serieInfo", {})
        for punto in serie.get("valores", []):
            filas.append({
                "fecha": pd.to_datetime(str(punto["fecha"]), format="%Y%m%d"),
                "codigo_serie": info.get("cod_serie"),
                "nombre": info.get("descripcion_corta"),
                "valor": punto.get("valor"),
            })
    tabla = pd.DataFrame(filas, columns=["fecha", "codigo_serie", "nombre", "valor"])
    return tabla.drop_duplicates(["fecha", "codigo_serie"]).sort_values(["codigo_serie", "fecha"]).reset_index(drop=True)


class ClienteCMF:
    """Cliente con ritmo controlado. La clave nunca aparece en mensajes ni registros."""

    def __init__(self, clave: str | None = None, pausa: float = PAUSA_MINIMA):
        self._clave = clave or leer_clave()
        self._pausa = pausa
        self._ultima = 0.0
        self.consultas = 0

    def _get(self, ruta: str) -> dict:
        espera = self._pausa - (time.monotonic() - self._ultima)
        if espera > 0:
            time.sleep(espera)
        resp = requests.get(URL_API + ruta, headers={"x-api-key": self._clave, "Accept": "application/json"}, timeout=90)
        self._ultima = time.monotonic()
        self.consultas += 1
        if resp.status_code != 200:
            detalle = MENSAJES_HTTP.get(resp.status_code, f"La API respondió con código {resp.status_code}.")
            raise ErrorCMF(f"{detalle} Recurso: {ruta}")
        return resp.json()

    def uso(self) -> dict:
        """Consultas usadas y disponibles en el día y el mes."""
        return self._get("user/stats")["usage"]

    def cuadro(self, tag: str, inicio: date, fin: date) -> pd.DataFrame:
        """Descarga un cuadro completo entre dos fechas, en tramos de doce meses."""
        partes = []
        for desde, hasta in rangos_anuales(inicio, fin):
            respuesta = self._get(f"cuadros/data/{tag}/range/{desde:%Y%m%d}/{hasta:%Y%m%d}")
            partes.append(parsear_cuadro(respuesta))
        tabla = pd.concat(partes, ignore_index=True)
        return tabla.drop_duplicates(["fecha", "codigo_serie"]).reset_index(drop=True)


def cargar_cuadro(
    tag: str, inicio: date, fin: date, carpeta: str | Path, cliente: ClienteCMF | None = None
) -> pd.DataFrame:
    """Lee un cuadro desde la caché o lo descarga si no existe.

    La caché se identifica por el cuadro y el rango de fechas, así que cambiar el
    rango genera una descarga nueva.
    """
    ruta = Path(carpeta) / f"{tag.replace('$', 'S')}_{inicio:%Y%m}_{fin:%Y%m}.csv"
    if ruta.exists():
        return pd.read_csv(ruta, parse_dates=["fecha"])
    tabla = (cliente or ClienteCMF()).cuadro(tag, inicio, fin)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla


def cargar_catalogo(carpeta: str | Path) -> pd.DataFrame:
    """Catálogo público de cuadros y series de la API BEST, con caché local."""
    ruta = Path(carpeta) / "catalogo_best.csv"
    if not ruta.exists():
        resp = requests.get(URL_CATALOGO, timeout=120)
        resp.raise_for_status()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(resp.content)
    return pd.read_csv(ruta, sep=";", encoding="latin-1", low_memory=False)


def codigo_institucion(codigo_serie: str) -> str | None:
    """Extrae el código de la institución de una serie de un cuadro por institución.

    Por ejemplo, de CMF_BCOS_IND_BASILEA3_PATEFE_APR_AGIFI_BICE_STO_RAZ_PORC_MONT
    obtiene BICE. Usar el código evita problemas con nombres escritos de distinta
    forma entre cuadros, como con y sin tilde.
    """
    coincidencia = re.search(r"AGIFI_([A-Z0-9]+?)_(?:STO|PORC|MM)", codigo_serie)
    return coincidencia.group(1) if coincidencia else None
