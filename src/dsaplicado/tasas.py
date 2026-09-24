"""Traspaso de tasas de política a tasas bancarias, la "beta de depósitos".

En gestión de balance (ALM) y en riesgo de tasa de interés del libro de banca
(IRRBB), la beta de depósitos indica qué fracción de un cambio en la tasa de
referencia se traspasa a la tasa que el banco paga por sus depósitos. Una beta
de 0,8 significa que un alza de 100 puntos base en la TPM sube el costo de los
depósitos en 80 puntos base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _rezagos(serie: pd.Series, rezagos: int, prefijo: str) -> pd.DataFrame:
    return pd.concat({f"{prefijo}{k}": serie.shift(k) for k in range(rezagos + 1)}, axis=1)


def beta_traspaso(tasa: pd.Series, referencia: pd.Series, rezagos: int = 3) -> dict[str, float]:
    """Beta de traspaso con un modelo de rezagos distribuidos en primeras diferencias.

    Estima el cambio mensual de la tasa como función del cambio de la referencia
    en el mismo mes y en los ``rezagos`` meses anteriores. El impacto es el
    coeficiente del mismo mes y la beta de largo plazo es la suma de todos.
    Los errores estándar son robustos a autocorrelación y heterocedasticidad.
    """
    cambios = pd.concat([tasa.rename("tasa"), referencia.rename("ref")], axis=1).diff()
    datos = pd.concat([cambios["tasa"], _rezagos(cambios["ref"], rezagos, "L")], axis=1).dropna()
    ajuste = sm.OLS(datos["tasa"], sm.add_constant(datos.drop(columns="tasa"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": rezagos})
    restriccion = np.r_[0.0, np.ones(rezagos + 1)]
    prueba = ajuste.t_test(restriccion)
    return {
        "impacto": float(ajuste.params["L0"]),
        "largo_plazo": float(ajuste.params.drop("const").sum()),
        "error_estandar": float(np.asarray(prueba.sd).item()),
        "r2": float(ajuste.rsquared),
        "observaciones": int(ajuste.nobs),
    }


def beta_asimetrica(tasa: pd.Series, referencia: pd.Series, rezagos: int = 3) -> dict[str, float]:
    """Betas de largo plazo separadas para alzas y bajas de la referencia, con su prueba de igualdad."""
    cambios = pd.concat([tasa.rename("tasa"), referencia.rename("ref")], axis=1).diff()
    alza = cambios["ref"].clip(lower=0)
    baja = cambios["ref"].clip(upper=0)
    datos = pd.concat([cambios["tasa"], _rezagos(alza, rezagos, "A"), _rezagos(baja, rezagos, "B")], axis=1).dropna()
    ajuste = sm.OLS(datos["tasa"], sm.add_constant(datos.drop(columns="tasa"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": rezagos})
    nombres = list(ajuste.params.index)
    restriccion = np.zeros(len(nombres))
    for k in range(rezagos + 1):
        restriccion[nombres.index(f"A{k}")] = 1
        restriccion[nombres.index(f"B{k}")] = -1
    return {
        "beta_alzas": float(sum(ajuste.params[f"A{k}"] for k in range(rezagos + 1))),
        "beta_bajas": float(sum(ajuste.params[f"B{k}"] for k in range(rezagos + 1))),
        "p_igualdad": float(ajuste.t_test(restriccion).pvalue),
    }


def beta_ciclo(tasa: pd.Series, referencia: pd.Series, inicio: str, fin: str) -> float:
    """Beta acumulada de un ciclo: cambio de la tasa dividido por el cambio de la referencia."""
    delta_ref = referencia.loc[fin] - referencia.loc[inicio]
    if delta_ref == 0:
        raise ValueError("La referencia no cambió en el período; la beta del ciclo no está definida.")
    return float((tasa.loc[fin] - tasa.loc[inicio]) / delta_ref)


# --- Curva de rendimientos: Nelson-Siegel -------------------------------------------

LAMBDA_DIEBOLD_LI = 0.7308  # en años; maximiza la carga de curvatura cerca de 2,5 años


def cargas_nelson_siegel(plazos: np.ndarray, lam: float = LAMBDA_DIEBOLD_LI) -> np.ndarray:
    """Cargas de nivel, pendiente y curvatura de Nelson-Siegel para plazos en años."""
    t = np.asarray(plazos, dtype=float)
    x = lam * t
    pendiente = (1 - np.exp(-x)) / x
    curvatura = pendiente - np.exp(-x)
    return np.column_stack([np.ones_like(t), pendiente, curvatura])


def ajustar_nelson_siegel(plazos: np.ndarray, tasas: np.ndarray, lam: float = LAMBDA_DIEBOLD_LI) -> dict[str, float]:
    """Ajusta nivel, pendiente y curvatura por mínimos cuadrados con lambda fijo.

    Con lambda fijo el ajuste es lineal, estable y rápido, como en Diebold y Li (2006).
    Convención: la pendiente es tasa larga menos tasa corta, por eso es igual a menos beta1.
    """
    x = cargas_nelson_siegel(plazos, lam)
    y = np.asarray(tasas, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuo = y - x @ beta
    return {"nivel": float(beta[0]), "beta1": float(beta[1]), "curvatura": float(beta[2]),
            "pendiente": float(-beta[1]), "rmse": float(np.sqrt(np.mean(residuo**2)))}


def curva_nelson_siegel(parametros: dict[str, float], plazos: np.ndarray, lam: float = LAMBDA_DIEBOLD_LI) -> np.ndarray:
    """Tasas de la curva ajustada en los plazos pedidos."""
    beta = np.array([parametros["nivel"], parametros["beta1"], parametros["curvatura"]])
    return cargas_nelson_siegel(plazos, lam) @ beta


# --- Bonos: precio, duración y convexidad -------------------------------------------


def flujos_bono(cupon: float, plazo: float, frecuencia: int = 2, nominal: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Fechas de pago en años y montos de un bono bullet con cupón fijo anual en porcentaje."""
    n = int(round(plazo * frecuencia))
    tiempos = np.arange(1, n + 1) / frecuencia
    montos = np.full(n, nominal * cupon / 100 / frecuencia)
    montos[-1] += nominal
    return tiempos, montos


def precio_con_curva(tiempos: np.ndarray, montos: np.ndarray, tasas_cero: np.ndarray) -> float:
    """Valor presente descontando cada flujo con su tasa cero, en porcentaje y capitalización continua."""
    return float(np.sum(montos * np.exp(-np.asarray(tasas_cero) / 100 * tiempos)))


def duracion_convexidad(tiempos: np.ndarray, montos: np.ndarray, tasas_cero: np.ndarray) -> dict[str, float]:
    """Precio, duración y convexidad frente a un desplazamiento paralelo de la curva.

    Con capitalización continua, la duración es el plazo promedio de los flujos
    ponderado por su valor presente, y la convexidad el promedio de los plazos al cuadrado.
    """
    vp = montos * np.exp(-np.asarray(tasas_cero) / 100 * tiempos)
    precio = vp.sum()
    return {"precio": float(precio), "duracion": float((vp * tiempos).sum() / precio),
            "convexidad": float((vp * tiempos**2).sum() / precio)}


# --- Escenarios de shock de tasas de Basilea para IRRBB ---------------------------------


def escenarios_basilea(plazos: np.ndarray, paralelo: float, corto: float, largo: float, x: float = 4.0) -> pd.DataFrame:
    """Los seis escenarios estándar de IRRBB de Basilea (BCBS, 2016), en las mismas unidades de los shocks.

    El shock corto decae con el plazo como exp(-t/x). Empinamiento y aplanamiento
    combinan los shocks corto y largo con los pesos del estándar.
    """
    t = np.asarray(plazos, dtype=float)
    s_corto = corto * np.exp(-t / x)
    s_largo = largo * (1 - np.exp(-t / x))
    return pd.DataFrame({
        "paralelo arriba": np.full_like(t, paralelo),
        "paralelo abajo": np.full_like(t, -paralelo),
        "empinamiento": -0.65 * np.abs(s_corto) + 0.9 * np.abs(s_largo),
        "aplanamiento": 0.8 * np.abs(s_corto) - 0.6 * np.abs(s_largo),
        "corto arriba": s_corto,
        "corto abajo": -s_corto,
    }, index=pd.Index(t, name="plazo"))
