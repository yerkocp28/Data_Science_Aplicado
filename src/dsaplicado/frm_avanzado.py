"""Fórmulas avanzadas del programa FRM Parte I, complemento del módulo ``frm``.

Incluye versiones en Python de los motores de cálculo del proyecto frm-prep
(valor del dinero en el tiempo al estilo de la calculadora BA II Plus y ratios de
capital de Basilea III) y modelos de mayor nivel: curva cero por bootstrapping,
opciones exóticas con fórmula cerrada, VaR de Cornish-Fisher y de componentes,
teoría de valores extremos, capital IRB, modelo de Merton, pruebas de estrés
inversas y coberturas de griegas.

Las tasas y volatilidades van en decimales (0,05 es 5%) y los plazos en años,
salvo que la función diga otra cosa.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from dsaplicado import frm

# =====================================================================================
# Valor del dinero en el tiempo. Convención de signos de la BA II Plus: los flujos que
# entran y salen tienen signos opuestos. ``r`` es la tasa por período en decimal.
# =====================================================================================


def valor_presente(n: float, r: float, pago: float = 0.0, vf: float = 0.0) -> float:
    """PV de una anualidad de ``n`` pagos más un valor final, con signo opuesto a los flujos."""
    if r == 0:
        return -(pago * n + vf)
    return -(pago * (1 - (1 + r) ** -n) / r + vf * (1 + r) ** -n)


def valor_futuro(n: float, r: float, vp: float = 0.0, pago: float = 0.0) -> float:
    """FV de un monto inicial y una anualidad, con signo opuesto a los flujos."""
    if r == 0:
        return -(vp + pago * n)
    return -(vp * (1 + r) ** n + pago * ((1 + r) ** n - 1) / r)


def cuota(n: float, r: float, vp: float, vf: float = 0.0) -> float:
    """PMT: pago periódico constante que salda ``vp`` y deja ``vf`` al final."""
    if n == 0:
        return 0.0
    if r == 0:
        return -(vp + vf) / n
    factor = (1 + r) ** n
    return -(vp * factor + vf) * r / (factor - 1)


def numero_periodos(r: float, vp: float, pago: float = 0.0, vf: float = 0.0) -> float:
    """N: número de períodos que igualan los flujos."""
    if r == 0:
        return -(vp + vf) / pago if abs(pago) > 0 else float("nan")
    if abs(pago) < 1e-10:
        return float(np.log(-vf / vp) / np.log(1 + r))
    razon = (pago - vf * r) / (pago + vp * r)
    return float(np.log(razon) / np.log(1 + r)) if razon > 0 else float("nan")


def tasa_por_periodo(n: float, vp: float, pago: float = 0.0, vf: float = 0.0) -> float:
    """I/Y en decimal: la tasa por período que hace cero el valor de los flujos."""
    def f(r):
        if abs(r) < 1e-12:
            return vp + pago * n + vf
        return vp * (1 + r) ** n + pago * ((1 + r) ** n - 1) / r + vf
    return float(optimize.brentq(f, -0.99, 10.0, xtol=1e-14))


def van(tasa: float, flujos: np.ndarray) -> float:
    """Valor actual neto; ``flujos[0]`` ocurre en t = 0."""
    flujos = np.asarray(flujos, dtype=float)
    return float(np.sum(flujos / (1 + tasa) ** np.arange(len(flujos))))


def tir(flujos: np.ndarray) -> float:
    """Tasa interna de retorno. Devuelve NaN si los flujos no cambian de signo.

    Con varios cambios de signo puede haber varias raíces; se devuelve la más cercana a 10%.
    """
    flujos = np.asarray(flujos, dtype=float)
    if np.all(flujos >= 0) or np.all(flujos <= 0):
        return float("nan")
    rejilla = np.concatenate([np.linspace(-0.99, 1, 400), np.linspace(1, 100, 200)[1:]])
    valores = np.array([van(r, flujos) for r in rejilla])
    cambios = np.where(np.sign(valores[:-1]) != np.sign(valores[1:]))[0]
    if len(cambios) == 0:
        return float("nan")
    i = cambios[np.argmin(np.abs(rejilla[cambios] - 0.1))]
    return float(optimize.brentq(van, rejilla[i], rejilla[i + 1], args=(flujos,), xtol=1e-14))


def tabla_amortizacion(principal: float, r: float, n: int) -> pd.DataFrame:
    """Tabla de un crédito de cuota fija: cuota, interés, amortización y saldo por período."""
    columnas = ["periodo", "cuota", "interes", "amortizacion", "saldo"]
    if n <= 0:
        return pd.DataFrame(columns=columnas)
    pago = abs(cuota(n, r, principal))
    saldo, filas = principal, []
    for k in range(1, n + 1):
        interes = saldo * r
        amortizacion = pago - interes
        saldo -= amortizacion
        filas.append([k, pago, interes, amortizacion, 0.0 if abs(saldo) < 1e-6 else saldo])
    return pd.DataFrame(filas, columns=columnas)


def fraccion_anio(inicio: date, fin: date, convencion: str = "act/360") -> float:
    """Fracción de año entre dos fechas según la convención de conteo de días."""
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    dias = (fin - inicio).days
    if convencion == "act/360":
        return dias / 360
    if convencion == "act/365":
        return dias / 365
    if convencion == "30/360":
        d1, d2 = min(inicio.day, 30), fin.day
        if d1 == 30 and d2 == 31:
            d2 = 30
        return (360 * (fin.year - inicio.year) + 30 * (fin.month - inicio.month) + d2 - d1) / 360
    if convencion == "act/act":
        total = 0.0
        for anio in range(inicio.year, fin.year + 1):
            a = max(inicio, pd.Timestamp(anio, 1, 1))
            b = min(fin, pd.Timestamp(anio + 1, 1, 1))
            if b > a:
                total += (b - a).days / (366 if pd.Timestamp(anio, 12, 31).dayofyear == 366 else 365)
        return total
    raise ValueError(f"Convención desconocida: {convencion}")


# =====================================================================================
# Capital regulatorio de Basilea III
# =====================================================================================

MINIMOS_BASILEA = {"cet1": 0.045, "tier1": 0.06, "total": 0.08, "apalancamiento": 0.03}
COLCHON_CONSERVACION = 0.025


def ratios_capital(cet1: float, at1: float, t2: float, apr: float, exposicion: float) -> dict[str, float]:
    """Capital de nivel 1 y total, y los cuatro ratios de Basilea III en decimales."""
    tier1 = cet1 + at1
    total = tier1 + t2

    def dividir(a, b):
        return a / b if b > 0 else float("nan")

    return {"tier1": tier1, "total": total, "cet1_ratio": dividir(cet1, apr), "tier1_ratio": dividir(tier1, apr),
            "total_ratio": dividir(total, apr), "apalancamiento_ratio": dividir(tier1, exposicion)}


def cumplimiento_basilea(ratios: dict[str, float], contraciclico: float = 0.0, sistemico: float = 0.0) -> pd.DataFrame:
    """Compara los ratios con los mínimos y con los colchones que se cubren con CET1.

    El colchón combinado es el de conservación (2,5%), más el contracíclico y el
    cargo sistémico que fije el regulador. El apalancamiento no lleva colchón.
    """
    colchon = COLCHON_CONSERVACION + contraciclico + sistemico
    filas = []
    for metrica, clave in [("cet1", "cet1_ratio"), ("tier1", "tier1_ratio"), ("total", "total_ratio"),
                           ("apalancamiento", "apalancamiento_ratio")]:
        valor = ratios[clave]
        minimo = MINIMOS_BASILEA[metrica]
        con_colchon = None if metrica == "apalancamiento" else minimo + colchon
        filas.append({"metrica": metrica, "valor": valor, "minimo": minimo, "minimo_con_colchon": con_colchon,
                      "cumple_minimo": bool(valor >= minimo),
                      "cumple_colchon": bool(valor >= (minimo if con_colchon is None else con_colchon))})
    return pd.DataFrame(filas).set_index("metrica")


def apr_estandar(exposiciones: np.ndarray, ponderaciones: np.ndarray) -> float:
    """Activos ponderados por riesgo del método estándar: suma de exposición por ponderación."""
    return float(np.dot(exposiciones, ponderaciones))


# =====================================================================================
# Libro 1. Fundamentos
# =====================================================================================


def pesos_minima_varianza(cov: np.ndarray) -> np.ndarray:
    """Cartera de mínima varianza global, con venta corta permitida: Σ⁻¹1 / 1'Σ⁻¹1."""
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.ones(len(cov)))
    return x / x.sum()


def pesos_tangencia(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Cartera tangente de máximo Sharpe, con venta corta permitida: Σ⁻¹(μ − rf) normalizado."""
    x = np.linalg.solve(np.asarray(cov, dtype=float), np.asarray(mu, dtype=float) - rf)
    return x / x.sum()


def retorno_apt(rf: float, cargas: np.ndarray, primas: np.ndarray) -> float:
    """Retorno esperado de la APT: rf más la suma de cargas por primas de los factores."""
    return float(rf + np.dot(cargas, primas))


def cartera_arbitraje(cargas: np.ndarray, retornos_esperados: np.ndarray) -> np.ndarray:
    """Cartera de costo cero y sin exposición a los factores, con retorno esperado positivo.

    ``cargas`` tiene una fila por activo y una columna por factor. Busca pesos con
    suma cero y exposición nula a cada factor; si alguno rinde distinto de cero, hay
    arbitraje. Los pesos se escalan para que la posición larga sume 1. Devuelve
    ceros si no hay arbitraje.
    """
    b = np.asarray(cargas, dtype=float).reshape(len(retornos_esperados), -1)
    restricciones = np.vstack([np.ones(b.shape[0]), b.T])
    nucleo = linalg.null_space(restricciones)
    if nucleo.size == 0:
        return np.zeros(b.shape[0])
    direccion = nucleo @ (nucleo.T @ np.asarray(retornos_esperados, dtype=float))
    if np.allclose(direccion, 0, atol=1e-12):
        return np.zeros(b.shape[0])
    return direccion / direccion[direccion > 0].sum()


def ratio_safety_first(retorno_esperado: float, volatilidad: float, umbral: float) -> float:
    """Ratio safety-first de Roy: (E[R] − umbral) / σ. Se elige la cartera con el mayor."""
    return (retorno_esperado - umbral) / volatilidad


def m_cuadrado(retorno: float, volatilidad: float, rf: float, volatilidad_mercado: float) -> float:
    """M² de Modigliani: retorno de la cartera llevada a la volatilidad del mercado."""
    return rf + (retorno - rf) * volatilidad_mercado / volatilidad


def agregar_riesgos(montos: np.ndarray, correlacion: np.ndarray) -> float:
    """Agrega VaR o capitales con una matriz de correlación: raíz de m' C m."""
    m = np.asarray(montos, dtype=float)
    return float(np.sqrt(m @ np.asarray(correlacion, dtype=float) @ m))


def raroc(ingresos: float, costos: float, perdida_esperada: float, capital_economico: float,
          retorno_capital: float = 0.0) -> float:
    """RAROC: (ingresos − costos − pérdida esperada + retorno del capital) / capital económico."""
    return (ingresos - costos - perdida_esperada + retorno_capital * capital_economico) / capital_economico


def spread_cds(intensidad: float, recuperacion: float) -> float:
    """Triángulo del crédito: spread ≈ λ · (1 − R)."""
    return intensidad * (1 - recuperacion)


def intensidad_desde_spread(spread: float, recuperacion: float) -> float:
    """Tasa de riesgo (hazard rate) implícita en un spread: λ ≈ s / (1 − R)."""
    return spread / (1 - recuperacion)


def pd_condicional_vasicek(pd_: float, correlacion: float, factor: np.ndarray) -> np.ndarray:
    """Probabilidad de incumplimiento dado el factor sistemático M en el modelo de un factor."""
    return stats.norm.cdf((stats.norm.ppf(pd_) - np.sqrt(correlacion) * np.asarray(factor)) / np.sqrt(1 - correlacion))


def perdidas_copula_gaussiana(pd_: float, correlacion: float, n_creditos: int, simulaciones: int = 100_000,
                              lgd: float = 1.0, semilla: int = 0) -> np.ndarray:
    """Pérdida de una cartera homogénea como fracción del nocional, con cópula gaussiana de un factor.

    Dado el factor común M, los incumplimientos son independientes, así que su número
    sigue una binomial con la probabilidad condicional de Vasicek.
    """
    rng = np.random.default_rng(semilla)
    m = rng.standard_normal(simulaciones)
    incumplimientos = rng.binomial(n_creditos, pd_condicional_vasicek(pd_, correlacion, m))
    return lgd * incumplimientos / n_creditos


def perdida_tramo(perdidas: np.ndarray, inicio: float, fin: float) -> np.ndarray:
    """Pérdida de un tramo [inicio, fin] como fracción de su tamaño."""
    return np.clip(np.asarray(perdidas) - inicio, 0, fin - inicio) / (fin - inicio)


# =====================================================================================
# Libro 2. Análisis cuantitativo
# =====================================================================================


def probabilidad_posterior(previa: float, p_dato_si: float, p_dato_no: float) -> float:
    """Teorema de Bayes para una hipótesis binaria: P(H | dato)."""
    return previa * p_dato_si / (previa * p_dato_si + (1 - previa) * p_dato_no)


def potencia_prueba_media(diferencia: float, desviacion: float, n: int, alfa: float = 0.05,
                          dos_colas: bool = True) -> float:
    """Potencia de una prueba z sobre la media cuando la diferencia verdadera es ``diferencia``."""
    z = stats.norm.ppf(1 - alfa / 2) if dos_colas else stats.norm.ppf(1 - alfa)
    desplazamiento = diferencia / (desviacion / np.sqrt(n))
    potencia = 1 - stats.norm.cdf(z - desplazamiento)
    if dos_colas:
        potencia += stats.norm.cdf(-z - desplazamiento)
    return float(potencia)


def error_estandar_sharpe(sharpe: float, n: int, asimetria: float = 0.0, curtosis: float = 3.0) -> float:
    """Error estándar del Sharpe estimado con n observaciones (Mertens, 2002).

    Con retornos normales se reduce a raíz de (1 + SR²/2) / n. La asimetría negativa
    y las colas pesadas lo aumentan. El Sharpe va en la frecuencia de los datos.
    """
    return float(np.sqrt((1 + 0.5 * sharpe**2 - asimetria * sharpe + (curtosis - 3) / 4 * sharpe**2) / n))


def r2_ajustado(r2: float, n: int, k: int) -> float:
    """R² ajustado con n observaciones y k regresores, sin contar la constante."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def estadistico_f(r2_completo: float, r2_restringido: float, n: int, k: int, q: int) -> float:
    """Estadístico F para probar q restricciones en un modelo con k regresores."""
    return ((r2_completo - r2_restringido) / q) / ((1 - r2_completo) / (n - k - 1))


def factores_inflacion_varianza(x: pd.DataFrame) -> pd.Series:
    """VIF de cada regresor: 1 / (1 − R²) de su regresión contra los demás."""
    resultado = {}
    valores = x.to_numpy(dtype=float)
    for j, nombre in enumerate(x.columns):
        otros = np.column_stack([np.ones(len(x)), np.delete(valores, j, axis=1)])
        beta, *_ = np.linalg.lstsq(otros, valores[:, j], rcond=None)
        residuo = valores[:, j] - otros @ beta
        r2 = 1 - residuo.var() / valores[:, j].var()
        resultado[nombre] = 1 / (1 - r2)
    return pd.Series(resultado, name="VIF")


def pronostico_ar1(ultimo: float, media: float, phi: float, horizontes: np.ndarray) -> np.ndarray:
    """Pronóstico de un AR(1): la desviación respecto de la media decae con φ^h."""
    return media + phi ** np.asarray(horizontes, dtype=float) * (ultimo - media)


def rho_desde_tau(tau: float) -> float:
    """Correlación lineal de una cópula gaussiana implícita en la tau de Kendall: sin(πτ/2)."""
    return float(np.sin(np.pi * tau / 2))


def rho_desde_spearman(rho_s: float) -> float:
    """Correlación lineal de una cópula gaussiana implícita en la rho de Spearman: 2 sin(πρ_s/6)."""
    return float(2 * np.sin(np.pi * rho_s / 6))


def dependencia_cola_inferior(x: np.ndarray, y: np.ndarray, cuantil: float = 0.05) -> float:
    """Probabilidad empírica de que Y caiga bajo su cuantil dado que X cae bajo el suyo."""
    x, y = np.asarray(x), np.asarray(y)
    cola_x = x <= np.quantile(x, cuantil)
    return float(np.mean(y[cola_x] <= np.quantile(y, cuantil)))


def bootstrap_bloques(datos: np.ndarray, estadistico, largo_bloque: int, repeticiones: int = 1_000,
                      semilla: int = 0) -> np.ndarray:
    """Bootstrap de bloques móviles: remuestrea bloques contiguos para conservar la dependencia."""
    datos = np.asarray(datos)
    n = len(datos)
    rng = np.random.default_rng(semilla)
    bloques = int(np.ceil(n / largo_bloque))
    inicios = rng.integers(0, n - largo_bloque + 1, size=(repeticiones, bloques))
    indices = (inicios[:, :, None] + np.arange(largo_bloque)).reshape(repeticiones, -1)[:, :n]
    return np.array([estadistico(datos[i]) for i in indices])


def ridge_cerrado(x: np.ndarray, y: np.ndarray, penalizacion: float) -> np.ndarray:
    """Coeficientes de la regresión ridge sin constante: (X'X + λI)⁻¹ X'y."""
    x = np.asarray(x, dtype=float)
    return np.linalg.solve(x.T @ x + penalizacion * np.eye(x.shape[1]), x.T @ np.asarray(y, dtype=float))


def metricas_confusion(vp: int, fp: int, fn: int, vn: int) -> dict[str, float]:
    """Exactitud, precisión, sensibilidad, especificidad y F1 desde la matriz de confusión."""
    precision = vp / (vp + fp)
    sensibilidad = vp / (vp + fn)
    return {"exactitud": (vp + vn) / (vp + fp + fn + vn), "precision": precision, "sensibilidad": sensibilidad,
            "especificidad": vn / (vn + fp), "f1": 2 * precision * sensibilidad / (precision + sensibilidad)}


# =====================================================================================
# Libro 3. Mercados y productos financieros
# =====================================================================================


def qx_gompertz_makeham(edades: np.ndarray, a: float = 0.0005, b: float = 0.00004, gamma: float = 0.095) -> np.ndarray:
    """Probabilidad de morir dentro de un año con la ley de Gompertz-Makeham, μ(x) = a + b·e^{γx}."""
    x = np.asarray(edades, dtype=float)
    integral = a + b * np.exp(gamma * x) * (np.exp(gamma) - 1) / gamma
    return 1 - np.exp(-integral)


def prima_seguro_vida(qx: np.ndarray, tasa: float, suma_asegurada: float = 1.0) -> dict[str, float]:
    """Prima nivelada anual de un seguro de vida temporal.

    ``qx`` son las probabilidades de muerte en cada año del contrato. La prima se
    paga al inicio de cada año si el asegurado vive, y el beneficio al final del año
    de la muerte. La prima iguala el valor presente esperado de ambos lados.
    """
    qx = np.asarray(qx, dtype=float)
    vivo_inicio = np.concatenate([[1.0], np.cumprod(1 - qx)[:-1]])
    anios = np.arange(1, len(qx) + 1)
    vp_beneficios = suma_asegurada * np.sum(vivo_inicio * qx / (1 + tasa) ** anios)
    vp_primas_unitarias = np.sum(vivo_inicio / (1 + tasa) ** (anios - 1))
    return {"vp_beneficios": float(vp_beneficios), "prima_anual": float(vp_beneficios / vp_primas_unitarias)}


def comisiones_fondo(retornos_brutos: np.ndarray, gestion: float = 0.02, desempeno: float = 0.20,
                     umbral: float = 0.0, marca_agua: bool = True, inicial: float = 100.0) -> pd.DataFrame:
    """Valor de una cuota con comisiones de gestión y de desempeño cobradas al cierre de cada año.

    La comisión de gestión se cobra sobre el valor al inicio del año. La de desempeño
    se cobra sobre la ganancia que excede el umbral y, con marca de agua (high-water
    mark), solo sobre el valor por encima del máximo neto anterior.
    """
    valor, maximo, filas = inicial, inicial, []
    for anio, r in enumerate(np.asarray(retornos_brutos, dtype=float), start=1):
        c_gestion = gestion * valor
        despues_gestion = valor * (1 + r) - c_gestion
        referencia = max(valor, maximo) if marca_agua else valor
        c_desempeno = desempeno * max(despues_gestion - referencia * (1 + umbral), 0.0)
        neto = despues_gestion - c_desempeno
        filas.append({"anio": anio, "retorno_bruto": r, "comision_gestion": c_gestion,
                      "comision_desempeno": c_desempeno, "valor_neto": neto, "retorno_neto": neto / valor - 1})
        valor = neto
        maximo = max(maximo, neto)
    return pd.DataFrame(filas).set_index("anio")


def cuenta_margen(precios: pd.Series, posicion: float, margen_inicial: float, margen_mantencion: float) -> pd.DataFrame:
    """Liquidación diaria de una posición en futuros con llamados de margen.

    ``posicion`` es la cantidad de unidades del subyacente, positiva si está larga.
    Cuando el saldo cae bajo el margen de mantención, se repone hasta el inicial.
    """
    saldo, filas = margen_inicial, []
    precio_anterior = precios.iloc[0]
    for fecha, precio in precios.iloc[1:].items():
        ganancia = posicion * (precio - precio_anterior)
        saldo += ganancia
        llamado = margen_inicial - saldo if saldo < margen_mantencion else 0.0
        saldo += llamado
        filas.append({"fecha": fecha, "precio": precio, "ganancia_diaria": ganancia, "llamado": llamado, "saldo": saldo})
        precio_anterior = precio
    return pd.DataFrame(filas).set_index("fecha")


def llamado_colateral(exposicion: float, colateral_actual: float, umbral: float = 0.0,
                      monto_minimo: float = 0.0) -> float:
    """Transferencia de colateral bajo un CSA: lo que falta sobre el umbral, si supera el monto mínimo.

    Un valor positivo lo entrega la contraparte; uno negativo se le devuelve.
    """
    requerido = max(exposicion - umbral, 0.0)
    transferencia = requerido - colateral_actual
    return transferencia if abs(transferencia) >= monto_minimo else 0.0


def cascada_incumplimiento(perdida: float, capas: dict[str, float]) -> pd.DataFrame:
    """Reparte la pérdida del incumplimiento de un miembro entre las capas de una cámara, en orden."""
    restante, filas = perdida, []
    for nombre, tamano in capas.items():
        usado = min(restante, tamano)
        filas.append({"capa": nombre, "tamano": tamano, "usado": usado,
                      "porcentaje_usado": usado / tamano if tamano else 0.0})
        restante -= usado
    filas.append({"capa": "pérdida no cubierta", "tamano": np.nan, "usado": restante, "porcentaje_usado": np.nan})
    return pd.DataFrame(filas).set_index("capa")


def rendimiento_conveniencia(spot: float, futuro: float, tasa: float, plazo: float, almacenamiento: float = 0.0) -> float:
    """Rendimiento de conveniencia implícito en un futuro: F = S·e^{(r + u − y)T}."""
    return float(tasa + almacenamiento - np.log(futuro / spot) / plazo)


def limites_opciones(s: float, k: float, r: float, t: float, q: float = 0.0) -> pd.DataFrame:
    """Cotas de no arbitraje de calls y puts europeas y americanas."""
    s_desc, k_desc = s * np.exp(-q * t), k * np.exp(-r * t)
    return pd.DataFrame({
        "minimo": [max(s_desc - k_desc, 0), max(k_desc - s_desc, 0), max(s_desc - k_desc, s - k, 0), max(k - s, 0)],
        "maximo": [s_desc, k_desc, s, k],
    }, index=["call europea", "put europea", "call americana", "put americana"])


def opcion_binaria(s: float, k: float, r: float, sigma: float, t: float, tipo: str = "call", pago: str = "efectivo",
                   q: float = 0.0, monto: float = 1.0) -> float:
    """Opción digital: paga ``monto`` (efectivo) o el activo si termina dentro del dinero."""
    d1 = (np.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    signo = 1 if tipo == "call" else -1
    if pago == "efectivo":
        return float(monto * np.exp(-r * t) * stats.norm.cdf(signo * d2))
    return float(s * np.exp(-q * t) * stats.norm.cdf(signo * d1))


def call_asiatica_geometrica(s: float, k: float, r: float, sigma: float, t: float, q: float = 0.0) -> float:
    """Call sobre el promedio geométrico continuo: Black-Scholes con σ/√3 y rendimiento (r + q + σ²/6)/2."""
    return frm.black_scholes(s, k, r, sigma / np.sqrt(3), t, "call", q=(r + q + sigma**2 / 6) / 2)


def call_asiatica_geometrica_discreta(s: float, k: float, r: float, sigma: float, t: float, n: int,
                                      q: float = 0.0) -> float:
    """Call sobre el promedio geométrico de n observaciones equiespaciadas en (0, T].

    El logaritmo del promedio es normal, con media ln S + (r − q − σ²/2)·T(n + 1)/(2n)
    y varianza σ²·T(n + 1)(2n + 1)/(6n²). Sirve como variable de control exacta para
    la asiática aritmética con las mismas fechas de observación.
    """
    media = np.log(s) + (r - q - sigma**2 / 2) * t * (n + 1) / (2 * n)
    varianza = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (media - np.log(k) + varianza) / np.sqrt(varianza)
    d2 = d1 - np.sqrt(varianza)
    return float(np.exp(-r * t) * (np.exp(media + varianza / 2) * stats.norm.cdf(d1) - k * stats.norm.cdf(d2)))


def call_barrera_abajo(s: float, k: float, h: float, r: float, sigma: float, t: float, q: float = 0.0,
                       variante: str = "out") -> float:
    """Call con barrera inferior ``h``, down-and-out o down-and-in, con monitoreo continuo (Hull)."""
    if h >= s:
        raise ValueError("La barrera inferior debe estar bajo el precio actual.")
    raiz = sigma * np.sqrt(t)
    lam = (r - q + sigma**2 / 2) / sigma**2
    vainilla = frm.black_scholes(s, k, r, sigma, t, "call", q)
    eq, er = s * np.exp(-q * t), k * np.exp(-r * t)
    if h <= k:
        y = np.log(h**2 / (s * k)) / raiz + lam * raiz
        entrada = eq * (h / s) ** (2 * lam) * stats.norm.cdf(y) - er * (h / s) ** (2 * lam - 2) * stats.norm.cdf(y - raiz)
        salida = vainilla - entrada
    else:
        x1 = np.log(s / h) / raiz + lam * raiz
        y1 = np.log(h / s) / raiz + lam * raiz
        salida = (eq * stats.norm.cdf(x1) - er * stats.norm.cdf(x1 - raiz)
                  - eq * (h / s) ** (2 * lam) * stats.norm.cdf(y1) + er * (h / s) ** (2 * lam - 2) * stats.norm.cdf(y1 - raiz))
        entrada = vainilla - salida
    return float(salida if variante == "out" else entrada)


def opcion_intercambio(s1: float, s2: float, sigma1: float, sigma2: float, rho: float, t: float,
                       q1: float = 0.0, q2: float = 0.0) -> float:
    """Fórmula de Margrabe: derecho a recibir el activo 1 entregando el activo 2, pago máx(S1 − S2, 0)."""
    sigma = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    d1 = (np.log(s1 / s2) + (q2 - q1 + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s1 * np.exp(-q1 * t) * stats.norm.cdf(d1) - s2 * np.exp(-q2 * t) * stats.norm.cdf(d2))


def black_scholes_vectorial(s: np.ndarray, k: float, r: float, sigma: float, t: float, tipo: str = "call",
                            q: float = 0.0) -> np.ndarray:
    """Black-Scholes-Merton para un vector de precios del subyacente, útil para revaluar escenarios."""
    s = np.asarray(s, dtype=float)
    d1 = (np.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if tipo == "call":
        return s * np.exp(-q * t) * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2)
    return k * np.exp(-r * t) * stats.norm.cdf(-d2) - s * np.exp(-q * t) * stats.norm.cdf(-d1)


def black_76(f: float, k: float, r: float, sigma: float, t: float, tipo: str = "call") -> float:
    """Modelo de Black para opciones europeas sobre futuros o forwards."""
    return frm.black_scholes(f, k, r, sigma, t, tipo, q=r)


def simular_gbm(s: float, deriva: float, sigma: float, t: float, pasos: int, trayectorias: int,
                semilla: int = 0, antiteticas: bool = False) -> np.ndarray:
    """Trayectorias de un movimiento browniano geométrico, con variables antitéticas opcionales."""
    rng = np.random.default_rng(semilla)
    dt = t / pasos
    mitad = trayectorias // 2 if antiteticas else trayectorias
    z = rng.standard_normal((mitad, pasos))
    if antiteticas:
        z = np.vstack([z, -z])
    incrementos = (deriva - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * z
    log_s = np.cumsum(incrementos, axis=1)
    return s * np.exp(np.hstack([np.zeros((len(z), 1)), log_s]))


def precio_montecarlo(pagos: np.ndarray, r: float, t: float) -> tuple[float, float]:
    """Precio como valor presente del pago promedio, y su error estándar."""
    descontados = np.exp(-r * t) * np.asarray(pagos, dtype=float)
    return float(descontados.mean()), float(descontados.std(ddof=1) / np.sqrt(len(descontados)))


def bootstrap_par(plazos: np.ndarray, tasas_par: np.ndarray, frecuencia: int = 2,
                  factores_conocidos: np.ndarray | None = None) -> np.ndarray:
    """Factores de descuento a partir de tasas par en una rejilla regular de 1/frecuencia.

    Cada tasa par es el cupón de un bono que vale 100: 1 = c/f · Σ DF_j + DF_n. Los
    primeros factores pueden venir dados, por ejemplo desde tasas cero de corto plazo.
    """
    plazos = np.asarray(plazos, dtype=float)
    tasas_par = np.asarray(tasas_par, dtype=float)
    if not np.allclose(plazos, np.arange(1, len(plazos) + 1) / frecuencia):
        raise ValueError("Los plazos deben ser 1/f, 2/f, 3/f, ...")
    factores = np.zeros(len(plazos))
    conocidos = 0 if factores_conocidos is None else len(factores_conocidos)
    if conocidos:
        factores[:conocidos] = factores_conocidos
    for i in range(conocidos, len(plazos)):
        c = tasas_par[i] / frecuencia
        factores[i] = (1 - c * factores[:i].sum()) / (1 + c)
    return factores


def tasas_cero_continuas(plazos: np.ndarray, factores: np.ndarray) -> np.ndarray:
    """Tasa cero con capitalización continua: −ln(DF)/t."""
    return -np.log(np.asarray(factores, dtype=float)) / np.asarray(plazos, dtype=float)


def forwards_periodo(plazos: np.ndarray, factores: np.ndarray) -> np.ndarray:
    """Tasa forward simple de cada período entre dos nodos: (DF_{i−1}/DF_i − 1)/Δt."""
    plazos, factores = np.asarray(plazos, dtype=float), np.asarray(factores, dtype=float)
    previos = np.concatenate([[1.0], factores[:-1]])
    dt = np.diff(np.concatenate([[0.0], plazos]))
    return (previos / factores - 1) / dt


def precio_bono(cupon: float, rendimiento: float, plazo: float, frecuencia: int = 2, nominal: float = 100.0) -> float:
    """Precio de un bono bullet con cupón y rendimiento anuales en decimal, capitalizados con la frecuencia."""
    n = int(round(plazo * frecuencia))
    periodos = np.arange(1, n + 1)
    flujos = np.full(n, nominal * cupon / frecuencia)
    flujos[-1] += nominal
    return float(np.sum(flujos / (1 + rendimiento / frecuencia) ** periodos))


def rendimiento_bono(precio: float, cupon: float, plazo: float, frecuencia: int = 2, nominal: float = 100.0) -> float:
    """Rendimiento al vencimiento que iguala el precio."""
    return float(optimize.brentq(lambda y: precio_bono(cupon, y, plazo, frecuencia, nominal) - precio, -0.5, 2.0,
                                 xtol=1e-14))


def metricas_bono(cupon: float, rendimiento: float, plazo: float, frecuencia: int = 2,
                  nominal: float = 100.0) -> dict[str, float]:
    """Precio, duración de Macaulay y modificada, convexidad y DV01 con el rendimiento del bono."""
    n = int(round(plazo * frecuencia))
    periodos = np.arange(1, n + 1)
    flujos = np.full(n, nominal * cupon / frecuencia)
    flujos[-1] += nominal
    base = 1 + rendimiento / frecuencia
    vp = flujos / base**periodos
    precio = vp.sum()
    macaulay = (vp * periodos / frecuencia).sum() / precio
    modificada = macaulay / base
    convexidad = (flujos * periodos * (periodos + 1) / base ** (periodos + 2)).sum() / (precio * frecuencia**2)
    return {"precio": float(precio), "macaulay": float(macaulay), "modificada": float(modificada),
            "convexidad": float(convexidad), "dv01": float(modificada * precio * 1e-4)}


def cambio_precio_aproximado(duracion_modificada: float, convexidad: float, cambio_rendimiento: float) -> float:
    """Aproximación de segundo orden del cambio porcentual del precio: −D·Δy + ½·C·Δy²."""
    return -duracion_modificada * cambio_rendimiento + 0.5 * convexidad * cambio_rendimiento**2


def contratos_duracion(valor_cartera: float, duracion_cartera: float, precio_futuro: float, duracion_futuro: float,
                       duracion_objetivo: float = 0.0) -> float:
    """Contratos de futuros de tasa para llevar la duración de la cartera al objetivo; negativo es vender."""
    return (duracion_objetivo - duracion_cartera) * valor_cartera / (precio_futuro * duracion_futuro)


def ajuste_convexidad(sigma: float, t1: float, t2: float) -> float:
    """Ajuste de convexidad de Ho-Lee: tasa forward = tasa del futuro − ½σ²·t1·t2."""
    return 0.5 * sigma**2 * t1 * t2


def smm_psa(meses: np.ndarray, velocidad_psa: float = 100.0) -> np.ndarray:
    """Tasa de prepago mensual del modelo PSA: el CPR sube 0,2% al mes hasta 6% en el mes 30."""
    cpr = np.minimum(0.002 * np.asarray(meses, dtype=float), 0.06) * velocidad_psa / 100
    return 1 - (1 - cpr) ** (1 / 12)


# =====================================================================================
# Libro 4. Valoración y modelos de riesgo
# =====================================================================================


def medida_espectral(perdidas: np.ndarray, aversion: float) -> float:
    """Medida de riesgo espectral con función de aversión exponencial.

    Pondera las pérdidas ordenadas con φ(p) = k·e^{−k(1−p)} / (1 − e^{−k}); a mayor
    aversión k, más peso en la cola. Es coherente porque los pesos crecen con p.
    """
    ordenadas = np.sort(np.asarray(perdidas, dtype=float))
    bordes = np.linspace(0, 1, len(ordenadas) + 1)
    acumulada = (np.exp(-aversion * (1 - bordes)) - np.exp(-aversion)) / (1 - np.exp(-aversion))
    return float(np.dot(np.diff(acumulada), ordenadas))


def var_cornish_fisher(media: float, desviacion: float, asimetria: float, curtosis_exceso: float,
                       confianza: float = 0.99) -> float:
    """VaR con la expansión de Cornish-Fisher, que corrige el cuantil normal por asimetría y curtosis."""
    z = stats.norm.ppf(1 - confianza)
    z_cf = (z + (z**2 - 1) * asimetria / 6 + (z**3 - 3 * z) * curtosis_exceso / 24
            - (2 * z**3 - 5 * z) * asimetria**2 / 36)
    return float(-(media + desviacion * z_cf))


def descomposicion_var(posiciones: np.ndarray, cov: np.ndarray, confianza: float = 0.99,
                       nombres: list[str] | None = None) -> pd.DataFrame:
    """VaR individual, marginal y de componentes de una cartera normal, por el teorema de Euler.

    El VaR de componentes suma exactamente el VaR total de la cartera.
    """
    x = np.asarray(posiciones, dtype=float)
    cov = np.asarray(cov, dtype=float)
    z = stats.norm.ppf(confianza)
    sigma_p = np.sqrt(x @ cov @ x)
    marginal = z * (cov @ x) / sigma_p
    componente = x * marginal
    return pd.DataFrame({"posicion": x, "var_individual": z * np.sqrt(np.diag(cov)) * np.abs(x),
                         "var_marginal": marginal, "var_componente": componente,
                         "porcentaje": componente / componente.sum()}, index=nombres)


def ajustar_pot(perdidas: np.ndarray, umbral: float) -> dict[str, float]:
    """Ajusta una Pareto generalizada a los excesos sobre el umbral (peaks over threshold)."""
    perdidas = np.asarray(perdidas, dtype=float)
    excesos = perdidas[perdidas > umbral] - umbral
    xi, _, beta = stats.genpareto.fit(excesos, floc=0)
    return {"xi": float(xi), "beta": float(beta), "umbral": float(umbral), "n_excesos": int(len(excesos)),
            "n_total": int(len(perdidas))}


def var_es_pot(ajuste: dict[str, float], confianza: float = 0.99) -> dict[str, float]:
    """VaR y ES desde un ajuste de Pareto generalizada, con las fórmulas de McNeil y Frey (2000)."""
    xi, beta, u = ajuste["xi"], ajuste["beta"], ajuste["umbral"]
    proporcion = ajuste["n_total"] / ajuste["n_excesos"] * (1 - confianza)
    var = u + beta / xi * (proporcion ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    return {"var": float(var), "es": float(es)}


def volatilidad_implicita(precio: float, s: float, k: float, r: float, t: float, tipo: str = "call",
                          q: float = 0.0) -> float:
    """Volatilidad que iguala el precio de Black-Scholes-Merton al precio observado."""
    return float(optimize.brentq(lambda v: frm.black_scholes(s, k, r, v, t, tipo, q) - precio, 1e-6, 5.0, xtol=1e-12))


def pd_acumulada_constante(pd_anual: float, anios: np.ndarray) -> np.ndarray:
    """Probabilidad acumulada de incumplimiento con una PD anual constante: 1 − (1 − PD)^n."""
    return 1 - (1 - pd_anual) ** np.asarray(anios, dtype=float)


def pd_marginal(pd_acumuladas: np.ndarray) -> np.ndarray:
    """PD incondicional de cada año: diferencia de las acumuladas."""
    return np.diff(np.concatenate([[0.0], np.asarray(pd_acumuladas, dtype=float)]))


def correlacion_irb(pd_: np.ndarray) -> np.ndarray:
    """Correlación de activos de Basilea para empresas: entre 12% y 24%, menor cuanto mayor la PD."""
    peso = (1 - np.exp(-50 * np.asarray(pd_))) / (1 - np.exp(-50))
    return 0.12 * peso + 0.24 * (1 - peso)


def capital_irb(pd_: float, lgd: float, vencimiento: float = 2.5, correlacion: float | None = None) -> dict[str, float]:
    """Requerimiento de capital por unidad de exposición del método IRB para empresas.

    K = LGD · [N((N⁻¹(PD) + √R·N⁻¹(0,999)) / √(1 − R)) − PD] · ajuste de vencimiento.
    La ponderación de riesgo es 12,5 · K.
    """
    r = float(correlacion_irb(pd_)) if correlacion is None else correlacion
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    ajuste = (1 + (vencimiento - 2.5) * b) / (1 - 1.5 * b)
    peor_caso = frm.tasa_default_peor_caso(pd_, r, 0.999)
    k = lgd * (peor_caso - pd_) * ajuste
    return {"correlacion": r, "pd_peor_caso": float(peor_caso), "ajuste_vencimiento": float(ajuste),
            "capital": float(k), "ponderacion": float(12.5 * k)}


def pd_merton(activos: float, deuda: float, deriva: float, sigma_activos: float, t: float = 1.0) -> dict[str, float]:
    """Distancia al incumplimiento y PD del modelo de Merton: la empresa quiebra si V_T < D."""
    dd = (np.log(activos / deuda) + (deriva - sigma_activos**2 / 2) * t) / (sigma_activos * np.sqrt(t))
    return {"distancia": float(dd), "pd": float(stats.norm.cdf(-dd))}


def activos_merton(patrimonio: float, sigma_patrimonio: float, deuda: float, r: float, t: float = 1.0) -> dict[str, float]:
    """Valor y volatilidad de los activos implícitos en el patrimonio, resolviendo el sistema de Merton.

    El patrimonio es una call sobre los activos con strike igual a la deuda, y
    σ_E·E = N(d1)·σ_V·V.
    """
    def ecuaciones(x):
        v, s = x
        d1 = (np.log(v / deuda) + (r + s**2 / 2) * t) / (s * np.sqrt(t))
        return [frm.black_scholes(v, deuda, r, s, t, "call") - patrimonio,
                stats.norm.cdf(d1) * s * v - sigma_patrimonio * patrimonio]

    inicial = [patrimonio + deuda * np.exp(-r * t), sigma_patrimonio * patrimonio / (patrimonio + deuda)]
    v, s = optimize.fsolve(ecuaciones, inicial, xtol=1e-12)
    return {"activos": float(v), "sigma_activos": float(s)}


def z_altman(capital_trabajo: float, utilidades_retenidas: float, ebit: float, patrimonio_mercado: float,
             ventas: float, activos: float, pasivos: float) -> float:
    """Z de Altman (1968) para manufactureras abiertas en bolsa. Bajo 1,81 es zona de peligro; sobre 2,99, segura."""
    return (1.2 * capital_trabajo / activos + 1.4 * utilidades_retenidas / activos + 3.3 * ebit / activos
            + 0.6 * patrimonio_mercado / pasivos + 1.0 * ventas / activos)


def componente_perdidas(perdidas_anuales_promedio: float) -> float:
    """Componente de pérdidas del método estándar de riesgo operacional: 15 veces la pérdida anual promedio."""
    return 15 * perdidas_anuales_promedio


def multiplicador_perdidas_internas(lc: float, bic: float) -> float:
    """ILM = ln(e − 1 + (LC/BIC)^0,8). Vale 1 cuando las pérdidas son proporcionales al tamaño."""
    return float(np.log(np.e - 1 + (lc / bic) ** 0.8))


def capital_riesgo_operacional(bi: float, perdidas_anuales_promedio: float | None = None) -> dict[str, float]:
    """Capital del método estándar (SMA): BIC · ILM. Sin historial de pérdidas, el ILM es 1."""
    bic = frm.componente_indicador_negocio(bi)
    if perdidas_anuales_promedio is None:
        return {"bic": bic, "lc": float("nan"), "ilm": 1.0, "capital": bic}
    lc = componente_perdidas(perdidas_anuales_promedio)
    ilm = multiplicador_perdidas_internas(lc, bic)
    return {"bic": bic, "lc": lc, "ilm": ilm, "capital": bic * ilm}


def estres_inverso(exposiciones: np.ndarray, cov: np.ndarray, perdida_objetivo: float) -> dict:
    """Escenario más probable que produce una pérdida dada, con factores normales.

    Minimiza la distancia de Mahalanobis x'Σ⁻¹x sujeto a −e'x = L. La solución es
    x* = −L·Σe / (e'Σe) y la distancia es L / raíz(e'Σe).
    """
    e = np.asarray(exposiciones, dtype=float)
    cov = np.asarray(cov, dtype=float)
    varianza = e @ cov @ e
    distancia = perdida_objetivo / np.sqrt(varianza)
    return {"shocks": -perdida_objetivo * (cov @ e) / varianza, "distancia": float(distancia),
            "probabilidad_una_cola": float(stats.norm.sf(distancia))}


def neutralizar_griegas(cartera: dict[str, float], instrumentos: pd.DataFrame,
                        griegas_objetivo: tuple[str, ...] = ("gamma", "vega")) -> pd.Series:
    """Cantidades de cada opción para anular las griegas indicadas, y del subyacente para la delta.

    ``instrumentos`` tiene una fila por opción y columnas con sus griegas. El
    subyacente tiene delta 1 y gamma y vega cero, así que se usa al final.
    """
    a = instrumentos[list(griegas_objetivo)].to_numpy(dtype=float).T
    b = -np.array([cartera[g] for g in griegas_objetivo])
    cantidades = np.linalg.solve(a, b)
    delta_restante = cartera["delta"] + float(instrumentos["delta"].to_numpy(dtype=float) @ cantidades)
    resultado = pd.Series(cantidades, index=instrumentos.index, dtype=float)
    resultado["subyacente"] = -delta_restante
    return resultado


def atribucion_taylor(delta: float, gamma: float, vega: float, theta: float, cambio_precio: float,
                      cambio_vol: float, tiempo: float) -> dict[str, float]:
    """Explica el cambio de valor de una opción con su expansión de Taylor de segundo orden."""
    partes = {"delta": delta * cambio_precio, "gamma": 0.5 * gamma * cambio_precio**2,
              "vega": vega * cambio_vol, "theta": theta * tiempo}
    partes["total"] = sum(partes.values())
    return partes


def factores_desde_bonos(flujos: np.ndarray, precios: np.ndarray) -> dict[str, np.ndarray]:
    """Factores de descuento implícitos en bonos con flujos en fechas comunes, por mínimos cuadrados.

    ``flujos`` tiene una fila por bono y una columna por fecha. Si hay más bonos que
    fechas, los residuos distintos de cero revelan precios inconsistentes entre sí.
    """
    flujos = np.asarray(flujos, dtype=float)
    precios = np.asarray(precios, dtype=float)
    factores, *_ = np.linalg.lstsq(flujos, precios, rcond=None)
    return {"factores": factores, "residuos": precios - flujos @ factores}


def replicar_flujos(objetivo: np.ndarray, flujos: np.ndarray) -> np.ndarray:
    """Cantidades de cada bono que replican exactamente los flujos objetivo."""
    return np.linalg.solve(np.asarray(flujos, dtype=float).T, np.asarray(objetivo, dtype=float))
