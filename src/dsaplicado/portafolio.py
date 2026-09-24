"""Construcción y evaluación de portafolios de media-varianza.

Convención: los pesos se aplican a retornos simples, porque el retorno de un
portafolio es el promedio ponderado de retornos simples y no de logarítmicos.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DIAS_HABILES = 252


def retornos_simples(precios: pd.DataFrame) -> pd.DataFrame:
    """Retornos diarios simples, sin la primera fila vacía."""
    return precios.pct_change(fill_method=None).dropna(how="all")


def retornos_log(precios: pd.DataFrame) -> pd.DataFrame:
    """Retornos diarios logarítmicos, sin la primera fila vacía."""
    return np.log(precios).diff().dropna(how="all")


def estadisticas_anualizadas(retornos: pd.DataFrame, rf_anual: float = 0.0) -> pd.DataFrame:
    """Retorno medio, volatilidad y Sharpe anualizados por activo."""
    media = retornos.mean() * DIAS_HABILES
    vol = retornos.std() * np.sqrt(DIAS_HABILES)
    return pd.DataFrame(
        {"retorno_anual": media, "volatilidad_anual": vol, "sharpe": (media - rf_anual) / vol}
    )


def rendimiento_portafolio(pesos: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Retorno esperado y volatilidad de un portafolio."""
    pesos = np.asarray(pesos)
    return float(pesos @ mu), float(np.sqrt(pesos @ cov @ pesos))


def _optimizar(objetivo: Callable, n: int, cota_max: float, restricciones: list | None = None) -> np.ndarray:
    """Optimiza con pesos no negativos que suman uno y un tope por activo."""
    if cota_max * n < 1 - 1e-9:
        raise ValueError("El tope por activo es demasiado bajo para que los pesos sumen uno.")
    base = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(
        objetivo,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cota_max)] * n,
        constraints=base + (restricciones or []),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"La optimización no convergió: {res.message}")
    pesos = np.clip(res.x, 0.0, None)
    return pesos / pesos.sum()


def minima_varianza(cov: np.ndarray, cota_max: float = 1.0) -> np.ndarray:
    """Pesos del portafolio de mínima varianza, sin ventas cortas."""
    cov = np.asarray(cov)
    return _optimizar(lambda w: w @ cov @ w, cov.shape[0], cota_max)


def maximo_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, cota_max: float = 1.0) -> np.ndarray:
    """Pesos del portafolio tangente, que maximiza el índice de Sharpe."""
    mu, cov = np.asarray(mu), np.asarray(cov)

    def sharpe_negativo(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    return _optimizar(sharpe_negativo, len(mu), cota_max)


def frontera_eficiente(
    mu: np.ndarray, cov: np.ndarray, n_puntos: int = 40, cota_max: float = 1.0
) -> pd.DataFrame:
    """Puntos de la frontera eficiente: mínima volatilidad para cada retorno objetivo."""
    mu, cov = np.asarray(mu), np.asarray(cov)
    w_min = minima_varianza(cov, cota_max)
    ret_min = float(w_min @ mu)
    # El retorno máximo alcanzable respeta el tope: se llenan primero los activos de mayor retorno.
    restante, ret_max = 1.0, 0.0
    for i in np.argsort(mu)[::-1]:
        peso = min(cota_max, restante)
        ret_max += peso * mu[i]
        restante -= peso
        if restante <= 1e-12:
            break
    filas = []
    for objetivo in np.linspace(ret_min, ret_max, n_puntos):
        restr = [{"type": "eq", "fun": lambda w, o=objetivo: w @ mu - o}]
        try:
            w = _optimizar(lambda w: w @ cov @ w, len(mu), cota_max, restr)
        except RuntimeError:
            continue
        ret, vol = rendimiento_portafolio(w, mu, cov)
        filas.append({"retorno": ret, "volatilidad": vol})
    return pd.DataFrame(filas)


def simular_portafolios(
    mu: np.ndarray, cov: np.ndarray, n: int = 10_000, rf: float = 0.0, semilla: int = 42
) -> pd.DataFrame:
    """Portafolios con pesos aleatorios uniformes sobre el simplex, vía Dirichlet."""
    rng = np.random.default_rng(semilla)
    pesos = rng.dirichlet(np.ones(len(mu)), size=n)
    ret = pesos @ np.asarray(mu)
    vol = np.sqrt(np.einsum("ij,jk,ik->i", pesos, np.asarray(cov), pesos))
    return pd.DataFrame({"retorno": ret, "volatilidad": vol, "sharpe": (ret - rf) / vol})


def beta_capm(
    retornos_activos: pd.DataFrame, retornos_mercado: pd.Series, rf_diario: float | pd.Series = 0.0
) -> pd.DataFrame:
    """Regresión CAPM de excesos de retorno: alfa anualizado, beta y R cuadrado."""
    datos = retornos_activos.join(retornos_mercado.rename("_mercado"), how="inner").dropna()
    rf = rf_diario.reindex(datos.index).ffill() if isinstance(rf_diario, pd.Series) else rf_diario
    exceso = datos.sub(rf, axis=0)
    x = exceso.pop("_mercado")
    filas = {}
    for activo, y in exceso.items():
        beta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
        alfa = y.mean() - beta * x.mean()
        residuo = y - alfa - beta * x
        r2 = 1 - residuo.var(ddof=1) / y.var(ddof=1)
        filas[activo] = {"alfa_anual": alfa * DIAS_HABILES, "beta": beta, "r2": r2}
    return pd.DataFrame(filas).T


def backtest_rebalanceo(
    retornos: pd.DataFrame,
    estrategia: Callable[[pd.DataFrame], np.ndarray],
    ventana: int = 504,
    frecuencia: str = "QE",
    costo: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest fuera de muestra con rebalanceo periódico.

    En cada fecha de rebalanceo la estrategia recibe solo los últimos ``ventana``
    días disponibles y entrega pesos. Entre rebalanceos los pesos derivan con los
    precios, como ocurre en un portafolio real que no se ajusta a diario.

    El parámetro ``costo`` es el costo proporcional por unidad transada, por
    ejemplo 0.002 para 20 puntos base. Se descuenta el día siguiente a cada
    rebalanceo, según la rotación efectiva frente a los pesos que habían derivado.

    Returns
    -------
    retornos_portafolio : Series con el retorno diario del portafolio, neto de costos.
    historial_pesos : DataFrame con los pesos de cada rebalanceo y la columna ``rotacion``.
    """
    retornos = retornos.dropna()
    fechas = retornos.index
    # Último día hábil de cada período, con historia suficiente para estimar.
    ultimos = fechas.to_series().resample(frecuencia).max().dropna()
    cortes = [f for f in ultimos if fechas.get_loc(f) + 1 >= ventana]
    if not cortes:
        raise ValueError("No hay historia suficiente para la ventana de estimación.")

    series, pesos_hist, rotaciones = [], {}, {}
    pesos_derivados = None  # la primera compra parte desde caja
    for i, corte in enumerate(cortes):
        pos = fechas.get_loc(corte)
        pesos = np.asarray(estrategia(retornos.iloc[pos + 1 - ventana : pos + 1]), dtype=float)
        pesos_hist[corte] = pesos
        rotacion = float(np.abs(pesos - pesos_derivados).sum()) if pesos_derivados is not None else 1.0
        rotaciones[corte] = rotacion
        fin = cortes[i + 1] if i + 1 < len(cortes) else fechas[-1]
        periodo = retornos.loc[(fechas > corte) & (fechas <= fin)]
        if periodo.empty:
            continue
        crecimiento = (1 + periodo).cumprod()
        valor = crecimiento @ pesos
        ret = valor.pct_change()
        ret.iloc[0] = valor.iloc[0] - 1
        ret.iloc[0] = (1 + ret.iloc[0]) * (1 - costo * rotacion) - 1
        series.append(ret)
        final = crecimiento.iloc[-1].to_numpy() * pesos
        pesos_derivados = final / final.sum()
    historial = pd.DataFrame(pesos_hist, index=retornos.columns).T
    historial["rotacion"] = pd.Series(rotaciones)
    return pd.concat(series).rename("retorno"), historial


def serie_drawdown(retornos: pd.Series) -> pd.Series:
    """Caída porcentual desde el máximo previo del valor acumulado."""
    valor = (1 + retornos).cumprod()
    return valor / valor.cummax() - 1


def metricas_desempeno(retornos: pd.Series, rf_anual: float = 0.0) -> dict[str, float]:
    """Retorno anual compuesto, volatilidad, Sharpe y máximo drawdown."""
    retornos = retornos.dropna()
    anios = len(retornos) / DIAS_HABILES
    cagr = (1 + retornos).prod() ** (1 / anios) - 1
    vol = retornos.std() * np.sqrt(DIAS_HABILES)
    return {
        "retorno_anual": cagr,
        "volatilidad_anual": vol,
        "sharpe": (cagr - rf_anual) / vol,
        "max_drawdown": serie_drawdown(retornos).min(),
    }
