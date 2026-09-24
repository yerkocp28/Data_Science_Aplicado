"""Métricas de riesgo de mercado: VaR, Expected Shortfall y backtesting.

Convención de signos: el VaR y el ES se expresan como pérdidas positivas.
Un VaR de 0.03 al 99% significa que la pérdida diaria supera el 3% en el 1%
de los días.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

METODOS = ("historico", "normal", "t_student", "ewma")


# --- Medidas puntuales ---------------------------------------------------------


def var_historico(retornos: pd.Series | np.ndarray, nivel: float = 0.99) -> float:
    """VaR por simulación histórica: cuantil empírico de las pérdidas."""
    return float(-np.quantile(np.asarray(retornos), 1 - nivel))


def es_historico(retornos: pd.Series | np.ndarray, nivel: float = 0.99) -> float:
    """ES histórico: pérdida promedio en los días iguales o peores que el VaR."""
    r = np.asarray(retornos)
    return float(-r[r <= -var_historico(r, nivel)].mean())


def var_normal(mu: float, sigma: float, nivel: float = 0.99) -> float:
    """VaR paramétrico suponiendo retornos normales."""
    return float(-(mu + sigma * stats.norm.ppf(1 - nivel)))


def es_normal(mu: float, sigma: float, nivel: float = 0.99) -> float:
    """ES paramétrico normal."""
    z = stats.norm.ppf(nivel)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - nivel))


def ajustar_t(retornos: pd.Series | np.ndarray) -> tuple[float, float, float]:
    """Ajusta una t de Student por máxima verosimilitud: grados de libertad, media y escala."""
    gl, loc, escala = stats.t.fit(np.asarray(retornos))
    return float(gl), float(loc), float(escala)


def var_t(gl: float, loc: float, escala: float, nivel: float = 0.99) -> float:
    """VaR paramétrico con distribución t de Student, que captura colas pesadas."""
    return float(-(loc + escala * stats.t.ppf(1 - nivel, gl)))


def es_t(gl: float, loc: float, escala: float, nivel: float = 0.99) -> float:
    """ES paramétrico con distribución t de Student. Requiere más de un grado de libertad."""
    if gl <= 1:
        raise ValueError("El ES de una t de Student requiere más de un grado de libertad.")
    q = stats.t.ppf(nivel, gl)
    return float(-loc + escala * stats.t.pdf(q, gl) * (gl + q**2) / ((gl - 1) * (1 - nivel)))


def volatilidad_ewma(retornos: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad diaria EWMA de RiskMetrics, pronosticada para el día siguiente.

    El valor en la fecha t usa información hasta t-1, así que sirve como pronóstico.
    """
    r2 = retornos.pow(2).to_numpy()
    var = np.empty(len(r2))
    var[0] = r2[: min(len(r2), 30)].mean()
    for t in range(1, len(r2)):
        var[t] = lam * var[t - 1] + (1 - lam) * r2[t - 1]
    return pd.Series(np.sqrt(var), index=retornos.index, name="vol_ewma")


# --- Pronósticos móviles --------------------------------------------------------


def pronosticar_var_es(
    retornos: pd.Series,
    metodo: str,
    nivel: float = 0.99,
    ventana: int = 250,
    lam: float = 0.94,
    recalibrar_cada: int = 1,
) -> pd.DataFrame:
    """Pronóstico diario de VaR y ES para cada día usando solo información previa.

    La fila de la fecha t contiene el VaR y el ES estimados al cierre de t-1,
    listos para compararse con el retorno realizado en t.

    ``recalibrar_cada`` solo afecta al método t de Student: los grados de libertad
    se reestiman cada tantos días, mientras la media y la escala se actualizan a
    diario. El ajuste por máxima verosimilitud es lento y los grados de libertad
    cambian poco de un día a otro.
    """
    if metodo not in METODOS:
        raise ValueError(f"Método desconocido: {metodo}. Opciones: {METODOS}")
    r = retornos.dropna()
    if metodo == "ewma":
        sigma = volatilidad_ewma(r, lam)
        z = stats.norm.ppf(nivel)
        salida = pd.DataFrame(
            {"var": z * sigma, "es": sigma * stats.norm.pdf(z) / (1 - nivel)}, index=r.index
        )
        return salida.iloc[ventana:]

    valores = r.to_numpy()
    filas = []
    gl = None
    for t in range(ventana, len(r)):
        muestra = valores[t - ventana : t]
        if metodo == "historico":
            filas.append((var_historico(muestra, nivel), es_historico(muestra, nivel)))
        elif metodo == "normal":
            mu, sigma = muestra.mean(), muestra.std(ddof=1)
            filas.append((var_normal(mu, sigma, nivel), es_normal(mu, sigma, nivel)))
        else:
            if gl is None or (t - ventana) % recalibrar_cada == 0:
                gl, loc, escala = ajustar_t(muestra)
                gl = max(gl, 2.1)  # evita un ES infinito cuando la cola es extremadamente pesada
            else:
                loc, escala = stats.t.fit(muestra, fix_df=gl)[1:]
            filas.append((var_t(gl, loc, escala, nivel), es_t(gl, loc, escala, nivel)))
    return pd.DataFrame(filas, index=r.index[ventana:], columns=["var", "es"])


# --- Backtesting ------------------------------------------------------------------


def excepciones(retornos: pd.Series, var: pd.Series) -> pd.Series:
    """Indicador de excepción: la pérdida del día superó el VaR pronosticado."""
    alineado = pd.concat([retornos.rename("r"), var.rename("var")], axis=1, join="inner").dropna()
    return (alineado["r"] < -alineado["var"]).astype(int).rename("excepcion")


def prueba_kupiec(exc: pd.Series | np.ndarray, nivel: float = 0.99) -> dict[str, float]:
    """Prueba de cobertura incondicional de Kupiec, también llamada POF.

    Contrasta si la proporción de excepciones coincide con la esperada, 1 - nivel.
    """
    exc = np.asarray(exc)
    n, x = len(exc), int(exc.sum())
    p = 1 - nivel
    tasa = x / n
    log_nula = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_alt = xlogy(n - x, 1 - tasa) + xlogy(x, tasa)
    lr = float(-2 * (log_nula - log_alt))
    return {
        "observaciones": n,
        "excepciones": x,
        "esperadas": n * p,
        "tasa": tasa,
        "estadistico": lr,
        "p_valor": float(stats.chi2.sf(lr, 1)),
    }


def prueba_christoffersen(exc: pd.Series | np.ndarray) -> dict[str, float]:
    """Prueba de independencia de Christoffersen.

    Contrasta si una excepción hoy vuelve más probable otra mañana. Si las
    excepciones se agrupan, el modelo reacciona tarde a los cambios de volatilidad.
    """
    exc = np.asarray(exc).astype(int)
    previo, actual = exc[:-1], exc[1:]
    n00 = int(((previo == 0) & (actual == 0)).sum())
    n01 = int(((previo == 0) & (actual == 1)).sum())
    n10 = int(((previo == 1) & (actual == 0)).sum())
    n11 = int(((previo == 1) & (actual == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_nula = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = float(-2 * (log_nula - log_alt))
    return {
        "prob_exc_tras_normal": pi01,
        "prob_exc_tras_exc": pi11,
        "estadistico": lr,
        "p_valor": float(stats.chi2.sf(lr, 1)),
    }


def semaforo_basilea(n_excepciones: int) -> str:
    """Zona del semáforo de Basilea para 250 días de VaR al 99%."""
    if n_excepciones <= 4:
        return "verde"
    if n_excepciones <= 9:
        return "amarilla"
    return "roja"


def resumen_backtest(
    retornos: pd.Series, pronostico: pd.DataFrame, nivel: float = 0.99
) -> dict[str, float | str]:
    """Kupiec, Christoffersen, semáforo del último año y severidad de las excepciones."""
    exc = excepciones(retornos, pronostico["var"])
    kup = prueba_kupiec(exc, nivel)
    ind = prueba_christoffersen(exc)
    dias = exc.index[exc == 1]
    perdida_media = float(-retornos.loc[dias].mean()) if len(dias) else float("nan")
    es_medio = float(pronostico.loc[dias, "es"].mean()) if len(dias) else float("nan")
    return {
        "excepciones": kup["excepciones"],
        "esperadas": kup["esperadas"],
        "tasa": kup["tasa"],
        "p_kupiec": kup["p_valor"],
        "p_christoffersen": ind["p_valor"],
        "excepciones_ultimo_anio": int(exc.iloc[-250:].sum()),
        "semaforo_ultimo_anio": semaforo_basilea(int(exc.iloc[-250:].sum())),
        "perdida_media_en_exc": perdida_media,
        "es_medio_en_exc": es_medio,
    }
