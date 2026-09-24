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
