"""Herramientas de series de tiempo: transformaciones, estacionariedad y evaluación de pronósticos.

La evaluación usa origen móvil: en cada fecha de corte el modelo se estima solo
con los datos disponibles hasta esa fecha y pronostica los meses siguientes.
Así se mide el desempeño que el modelo habría tenido en tiempo real.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

Pronosticador = Callable[[pd.Series, int], np.ndarray]


# --- Transformaciones ------------------------------------------------------------


def indice_desde_variaciones(variaciones_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """Construye un índice encadenado a partir de variaciones porcentuales mensuales."""
    return (base * (1 + variaciones_pct / 100).cumprod()).rename("indice")


def variacion_anual(indice: pd.Series, periodos: int = 12) -> pd.Series:
    """Variación porcentual respecto del mismo período del año anterior."""
    return (100 * (indice / indice.shift(periodos) - 1)).rename("variacion_anual")


def a_mensual(serie: pd.Series, como: str = "mean") -> pd.Series:
    """Agrega una serie diaria a mensual, fechada el primer día de cada mes."""
    mensual = serie.resample("MS").agg(como)
    return mensual.dropna()


# --- Estacionariedad ---------------------------------------------------------------


def pruebas_raiz_unitaria(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Pruebas ADF y KPSS para varias series.

    Las dos pruebas tienen hipótesis nulas opuestas. ADF supone raíz unitaria y
    KPSS supone estacionariedad. Cuando ambas coinciden, la conclusión es sólida.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    filas = {}
    for nombre, serie in series.items():
        x = serie.dropna()
        with warnings.catch_warnings():
            # KPSS avisa cuando el p-valor sale de su tabla y ADF anuncia cambios de API.
            warnings.simplefilter("ignore")
            p_adf = adfuller(x, autolag="AIC")[1]
            p_kpss = kpss(x, regression="c", nlags="auto")[1]
        if p_adf < 0.05 and p_kpss >= 0.05:
            conclusion = "estacionaria"
        elif p_adf >= 0.05 and p_kpss < 0.05:
            conclusion = "raíz unitaria"
        else:
            conclusion = "no concluyente"
        filas[nombre] = {"p_adf": p_adf, "p_kpss": p_kpss, "conclusion": conclusion}
    return pd.DataFrame(filas).T


# --- Evaluación de pronósticos -------------------------------------------------------


def evaluar_origen_movil(
    serie: pd.Series,
    pronosticadores: dict[str, Pronosticador],
    inicio: str | pd.Timestamp,
    horizonte: int = 12,
    paso: int = 1,
) -> pd.DataFrame:
    """Evalúa pronosticadores con origen móvil y ventana de estimación creciente.

    Cada pronosticador recibe la serie hasta el origen, inclusive, y el horizonte.
    Debe devolver un arreglo con ``horizonte`` pronósticos.

    Solo se usan orígenes con todos sus valores reales disponibles.

    Returns
    -------
    DataFrame largo con origen, modelo, horizonte, fecha, pronostico, real y error.
    """
    serie = serie.dropna()
    fechas = serie.index
    primero = fechas.searchsorted(pd.Timestamp(inicio))
    filas = []
    for pos in range(primero, len(serie) - horizonte + 1, paso):
        historia = serie.iloc[:pos]
        reales = serie.iloc[pos : pos + horizonte]
        for nombre, funcion in pronosticadores.items():
            prono = np.asarray(funcion(historia, horizonte), dtype=float)
            for h in range(horizonte):
                filas.append(
                    {
                        "origen": fechas[pos - 1],
                        "modelo": nombre,
                        "horizonte": h + 1,
                        "fecha": reales.index[h],
                        "pronostico": prono[h],
                        "real": reales.iloc[h],
                    }
                )
    resultado = pd.DataFrame(filas)
    resultado["error"] = resultado["real"] - resultado["pronostico"]
    return resultado


def resumen_errores(evaluacion: pd.DataFrame, metrica: str = "rmse") -> pd.DataFrame:
    """Tabla de error por modelo y horizonte. Métricas: rmse o mae."""
    if metrica == "rmse":
        agg = evaluacion.groupby(["horizonte", "modelo"])["error"].apply(lambda e: np.sqrt(np.mean(e**2)))
    elif metrica == "mae":
        agg = evaluacion.groupby(["horizonte", "modelo"])["error"].apply(lambda e: np.mean(np.abs(e)))
    else:
        raise ValueError("La métrica debe ser rmse o mae.")
    return agg.unstack("modelo")


def prueba_diebold_mariano(
    errores_a: np.ndarray, errores_b: np.ndarray, horizonte: int = 1
) -> dict[str, float]:
    """Prueba de Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold.

    Contrasta si dos pronósticos tienen el mismo error cuadrático medio. Un
    estadístico negativo indica que el modelo A es más preciso que el B.
    La varianza considera la autocorrelación que generan los horizontes largos.
    """
    a, b = np.asarray(errores_a, dtype=float), np.asarray(errores_b, dtype=float)
    d = a**2 - b**2
    n = len(d)
    media = d.mean()
    centrado = d - media
    gamma = [np.dot(centrado[k:], centrado[: n - k]) / n for k in range(horizonte)]
    varianza = (gamma[0] + 2 * sum(gamma[1:])) / n
    if varianza <= 0:
        return {"estadistico": float("nan"), "p_valor": float("nan")}
    correccion = np.sqrt((n + 1 - 2 * horizonte + horizonte * (horizonte - 1) / n) / n)
    estadistico = correccion * media / np.sqrt(varianza)
    p_valor = 2 * stats.t.sf(abs(estadistico), df=n - 1)
    return {"estadistico": float(estadistico), "p_valor": float(p_valor)}
