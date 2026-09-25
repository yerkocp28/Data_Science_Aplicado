"""Fórmulas del programa FRM Parte I, organizadas por libro.

Cada función implementa una fórmula estándar del programa y se valida en las
pruebas contra ejemplos conocidos de los textos de referencia, como Hull.
Las tasas y volatilidades se expresan en decimales (0,05 es 5%) y los plazos en años.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# =====================================================================================
# Libro 1. Fundamentos de la gestión de riesgos
# =====================================================================================


def indice_sharpe(retorno: float, volatilidad: float, rf: float) -> float:
    """Exceso de retorno por unidad de riesgo total."""
    return (retorno - rf) / volatilidad


def indice_treynor(retorno: float, beta: float, rf: float) -> float:
    """Exceso de retorno por unidad de riesgo sistemático."""
    return (retorno - rf) / beta


def alfa_jensen(retorno: float, beta: float, rf: float, retorno_mercado: float) -> float:
    """Retorno por sobre lo que predice el CAPM."""
    return retorno - (rf + beta * (retorno_mercado - rf))


def indice_informacion(retornos: pd.Series, referencia: pd.Series, periodos: int = 252) -> float:
    """Exceso de retorno sobre la referencia dividido por el error de seguimiento, anualizados."""
    exceso = (retornos - referencia).dropna()
    return float(exceso.mean() * periodos / (exceso.std() * np.sqrt(periodos)))


def indice_sortino(retornos: pd.Series, objetivo_anual: float, periodos: int = 252) -> float:
    """Como Sharpe, pero penaliza solo la volatilidad bajo el retorno objetivo."""
    objetivo = objetivo_anual / periodos
    abajo = np.minimum(retornos - objetivo, 0)
    desviacion_baja = np.sqrt((abajo**2).mean()) * np.sqrt(periodos)
    return float((retornos.mean() * periodos - objetivo_anual) / desviacion_baja)


# =====================================================================================
# Libro 2. Análisis cuantitativo
# =====================================================================================


def varianza_largo_plazo_garch(omega: float, alfa: float, beta: float) -> float:
    """Varianza de largo plazo de un GARCH(1,1): omega / (1 - alfa - beta)."""
    if alfa + beta >= 1:
        raise ValueError("Con alfa + beta >= 1 no existe varianza de largo plazo.")
    return omega / (1 - alfa - beta)


def pronostico_varianza_garch(varianza_hoy: float, omega: float, alfa: float, beta: float, dias: int) -> float:
    """Varianza esperada en ``dias`` días: V_L + (alfa + beta)^dias · (varianza_hoy - V_L)."""
    largo_plazo = varianza_largo_plazo_garch(omega, alfa, beta)
    return largo_plazo + (alfa + beta) ** dias * (varianza_hoy - largo_plazo)


def vida_media(persistencia: float) -> float:
    """Días para que un shock de varianza se reduzca a la mitad."""
    return float(np.log(0.5) / np.log(persistencia))


def varianza_ewma(retornos: np.ndarray, lam: float = 0.94, inicial: float | None = None) -> np.ndarray:
    """Varianza EWMA de RiskMetrics. El valor t usa retornos hasta t-1."""
    r = np.asarray(retornos, dtype=float)
    var = np.empty(len(r))
    var[0] = inicial if inicial is not None else r[: min(30, len(r))].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var


# =====================================================================================
# Libro 3. Mercados y productos financieros
# =====================================================================================


def tasa_continua(tasa: float, capitalizaciones: int) -> float:
    """Convierte una tasa con m capitalizaciones al año en su equivalente continua."""
    return capitalizaciones * np.log(1 + tasa / capitalizaciones)


def tasa_desde_continua(tasa_continua_: float, capitalizaciones: int) -> float:
    """Convierte una tasa continua en su equivalente con m capitalizaciones al año."""
    return capitalizaciones * (np.exp(tasa_continua_ / capitalizaciones) - 1)


def tasas_forward(plazos: np.ndarray, tasas_cero: np.ndarray) -> np.ndarray:
    """Tasas forward continuas entre plazos consecutivos: (R2·T2 - R1·T1) / (T2 - T1)."""
    t, r = np.asarray(plazos, dtype=float), np.asarray(tasas_cero, dtype=float)
    return (r[1:] * t[1:] - r[:-1] * t[:-1]) / (t[1:] - t[:-1])


def precio_forward(spot: float, tasa: float, plazo: float, rendimiento: float = 0.0) -> float:
    """Precio forward con costo de acarreo continuo: S · exp((r - q) · T).

    ``rendimiento`` es un dividendo continuo, la tasa extranjera en un forward de
    divisas, o el rendimiento de conveniencia menos el costo de almacenaje en commodities.
    """
    return float(spot * np.exp((tasa - rendimiento) * plazo))


def forward_divisas(spot: float, tasa_local: float, tasa_extranjera: float, plazo: float) -> float:
    """Paridad cubierta de tasas con capitalización anual: S · ((1 + r_local) / (1 + r_ext))^T."""
    return spot * ((1 + tasa_local) / (1 + tasa_extranjera)) ** plazo


def razon_cobertura_minima_varianza(correlacion: float, vol_spot: float, vol_futuro: float) -> float:
    """h* = ρ · σ_S / σ_F."""
    return correlacion * vol_spot / vol_futuro


def contratos_cobertura(razon: float, exposicion: float, tamano_contrato: float) -> float:
    """Número óptimo de contratos: h* · Q_A / Q_F."""
    return razon * exposicion / tamano_contrato


def contratos_ajuste_beta(beta_actual: float, beta_objetivo: float, valor_cartera: float, valor_futuro: float) -> float:
    """Contratos de futuros sobre índice para llevar la beta de la cartera a la objetivo.

    Un número negativo significa vender futuros.
    """
    return (beta_objetivo - beta_actual) * valor_cartera / valor_futuro


def pago_estrategia(precios_finales: np.ndarray, patas: list[tuple[str, float, float, float]]) -> np.ndarray:
    """Pago neto al vencimiento de una estrategia de opciones, incluidas las primas.

    Cada pata es (tipo, precio de ejercicio, cantidad, prima), con tipo "call",
    "put" o "subyacente". Una cantidad negativa es una posición vendida.
    """
    s = np.asarray(precios_finales, dtype=float)
    total = np.zeros_like(s)
    for tipo, strike, cantidad, prima in patas:
        if tipo == "call":
            pago = np.maximum(s - strike, 0)
        elif tipo == "put":
            pago = np.maximum(strike - s, 0)
        elif tipo == "subyacente":
            pago = s - strike  # strike hace de precio de compra
        else:
            raise ValueError(f"Tipo desconocido: {tipo}")
        total += cantidad * (pago - prima)
    return total


def valor_swap_tasa(nocional: float, tasa_fija: float, plazos_pago: np.ndarray, tasas_cero: np.ndarray,
                    frecuencia: int = 2) -> float:
    """Valor de un swap para quien paga fijo y recibe flotante, con el enfoque de bonos.

    Al inicio de un período, el bono flotante vale el nocional. El valor es
    B_flotante - B_fijo, descontando con tasas cero continuas.
    """
    t, z = np.asarray(plazos_pago, dtype=float), np.asarray(tasas_cero, dtype=float)
    factores = np.exp(-z * t)
    bono_fijo = nocional * tasa_fija / frecuencia * factores.sum() + nocional * factores[-1]
    return float(nocional - bono_fijo)


def tasa_swap_par(plazos_pago: np.ndarray, tasas_cero: np.ndarray, frecuencia: int = 2) -> float:
    """Tasa fija que hace que el swap valga cero al inicio."""
    factores = np.exp(-np.asarray(tasas_cero, dtype=float) * np.asarray(plazos_pago, dtype=float))
    return float(frecuencia * (1 - factores[-1]) / factores.sum())


def cuota_hipotecaria(principal: float, tasa_anual: float, meses: int) -> float:
    """Cuota mensual fija de un crédito hipotecario."""
    i = tasa_anual / 12
    return float(principal * i / (1 - (1 + i) ** -meses))


def smm_desde_cpr(cpr: float) -> float:
    """Tasa de prepago mensual (SMM) equivalente a una tasa anual (CPR)."""
    return float(1 - (1 - cpr) ** (1 / 12))


def flujos_hipoteca(principal: float, tasa_anual: float, meses: int, cpr: float = 0.0) -> pd.DataFrame:
    """Calendario de pagos de un pool hipotecario con prepago constante."""
    i, smm = tasa_anual / 12, smm_desde_cpr(cpr)
    saldo, filas = principal, []
    for mes in range(1, meses + 1):
        if saldo <= 1e-9:
            break
        cuota = cuota_hipotecaria(saldo, tasa_anual, meses - mes + 1)
        interes = saldo * i
        amortizacion = cuota - interes
        prepago = (saldo - amortizacion) * smm
        filas.append({"mes": mes, "interes": interes, "amortizacion": amortizacion, "prepago": prepago,
                      "flujo_total": interes + amortizacion + prepago, "saldo_final": saldo - amortizacion - prepago})
        saldo -= amortizacion + prepago
    return pd.DataFrame(filas)


def exposicion_neta(exposiciones: pd.DataFrame, multilateral: bool) -> float:
    """Exposición total del sistema, bilateral o con una contraparte central.

    ``exposiciones.loc[i, j]`` es lo que i le debe a j. En el caso bilateral cada
    par compensa sus deudas mutuas; con una contraparte central, cada participante
    compensa todas sus posiciones contra la cámara.
    """
    m = exposiciones.to_numpy(dtype=float)
    if multilateral:
        neto = m.sum(axis=0) - m.sum(axis=1)  # lo que cada uno recibe menos lo que paga
        return float(np.clip(neto, 0, None).sum())
    neto_pares = np.triu(np.abs(m - m.T), k=1)
    return float(neto_pares.sum())


# =====================================================================================
# Libro 4. Valoración y modelos de riesgo
# =====================================================================================


def _d1_d2(s, k, r, sigma, t, q):
    d1 = (np.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    return d1, d1 - sigma * np.sqrt(t)


def black_scholes(s: float, k: float, r: float, sigma: float, t: float, tipo: str = "call", q: float = 0.0) -> float:
    """Precio de Black-Scholes-Merton de una opción europea.

    Con ``q`` igual a la tasa extranjera es el modelo de Garman-Kohlhagen para divisas.
    """
    d1, d2 = _d1_d2(s, k, r, sigma, t, q)
    if tipo == "call":
        return float(s * np.exp(-q * t) * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2))
    return float(k * np.exp(-r * t) * stats.norm.cdf(-d2) - s * np.exp(-q * t) * stats.norm.cdf(-d1))


def griegas(s: float, k: float, r: float, sigma: float, t: float, tipo: str = "call", q: float = 0.0) -> dict[str, float]:
    """Delta, gamma, vega, theta y rho de Black-Scholes-Merton.

    Vega y rho se expresan por unidad de volatilidad y de tasa (multiplicar por
    0,01 para un punto porcentual); theta por año.
    """
    d1, d2 = _d1_d2(s, k, r, sigma, t, q)
    n1 = stats.norm.pdf(d1)
    eq, er = np.exp(-q * t), np.exp(-r * t)
    gamma = eq * n1 / (s * sigma * np.sqrt(t))
    vega = s * eq * n1 * np.sqrt(t)
    if tipo == "call":
        delta = eq * stats.norm.cdf(d1)
        theta = -s * eq * n1 * sigma / (2 * np.sqrt(t)) + q * s * eq * stats.norm.cdf(d1) - r * k * er * stats.norm.cdf(d2)
        rho = k * t * er * stats.norm.cdf(d2)
    else:
        delta = eq * (stats.norm.cdf(d1) - 1)
        theta = -s * eq * n1 * sigma / (2 * np.sqrt(t)) - q * s * eq * stats.norm.cdf(-d1) + r * k * er * stats.norm.cdf(-d2)
        rho = -k * t * er * stats.norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def arbol_binomial(s: float, k: float, r: float, sigma: float, t: float, pasos: int, tipo: str = "put",
                   americana: bool = True, q: float = 0.0) -> float:
    """Precio de una opción con el árbol binomial de Cox, Ross y Rubinstein."""
    dt = t / pasos
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    descuento = np.exp(-r * dt)
    precios = s * u ** np.arange(pasos, -1, -1) * d ** np.arange(0, pasos + 1)
    valores = np.maximum(precios - k, 0) if tipo == "call" else np.maximum(k - precios, 0)
    for paso in range(pasos - 1, -1, -1):
        precios = s * u ** np.arange(paso, -1, -1) * d ** np.arange(0, paso + 1)
        valores = descuento * (p * valores[:-1] + (1 - p) * valores[1:])
        if americana:
            ejercicio = np.maximum(precios - k, 0) if tipo == "call" else np.maximum(k - precios, 0)
            valores = np.maximum(valores, ejercicio)
    return float(valores[0])


def perdida_esperada(pd_: float, lgd: float, ead: float) -> float:
    """EL = PD · LGD · EAD."""
    return pd_ * lgd * ead


def perdida_inesperada(pd_: float, lgd: float, ead: float) -> float:
    """Desviación estándar de la pérdida de un préstamo con LGD fija: EAD · LGD · sqrt(PD · (1 - PD))."""
    return float(ead * lgd * np.sqrt(pd_ * (1 - pd_)))


def tasa_default_peor_caso(pd_: float, correlacion: float, confianza: float = 0.999) -> float:
    """Tasa de default en el peor caso del modelo de Vasicek, la base de la fórmula de capital de Basilea."""
    return float(stats.norm.cdf((stats.norm.ppf(pd_) + np.sqrt(correlacion) * stats.norm.ppf(confianza))
                                / np.sqrt(1 - correlacion)))


def transicion_multiperiodo(matriz: pd.DataFrame, periodos: int) -> pd.DataFrame:
    """Matriz de transición de ratings a varios años, bajo el supuesto de Markov."""
    resultado = np.linalg.matrix_power(matriz.to_numpy(dtype=float), periodos)
    return pd.DataFrame(resultado, index=matriz.index, columns=matriz.columns)


def perdidas_operacionales(frecuencia_anual: float, mu_log: float, sigma_log: float, simulaciones: int = 100_000,
                           semilla: int = 0) -> np.ndarray:
    """Pérdida anual simulada con el enfoque de distribución de pérdidas (LDA).

    El número de eventos sigue una Poisson y el monto de cada uno una lognormal.
    """
    rng = np.random.default_rng(semilla)
    eventos = rng.poisson(frecuencia_anual, simulaciones)
    montos = rng.lognormal(mu_log, sigma_log, eventos.sum())
    indices = np.repeat(np.arange(simulaciones), eventos)
    return np.bincount(indices, weights=montos, minlength=simulaciones)


def componente_indicador_negocio(bi: float) -> float:
    """Componente del indicador de negocio (BIC) del método estándar de Basilea III para riesgo operacional.

    Coeficientes marginales de 12%, 15% y 18% por tramos de 1.000 y 30.000 millones de euros.
    """
    tramos = [(1_000.0, 0.12), (30_000.0, 0.15), (np.inf, 0.18)]
    total, inferior = 0.0, 0.0
    for superior, coeficiente in tramos:
        if bi > inferior:
            total += (min(bi, superior) - inferior) * coeficiente
        inferior = superior
    return total


def dv01(precio: float, duracion_modificada: float) -> float:
    """Cambio de precio ante un alza de un punto base: D · P · 0,0001."""
    return duracion_modificada * precio * 0.0001
