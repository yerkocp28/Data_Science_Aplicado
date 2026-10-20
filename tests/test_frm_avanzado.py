"""Pruebas del módulo frm_avanzado.

Se validan contra tres fuentes: las pruebas de los motores de frm-prep (valor del
dinero en el tiempo y Basilea III), las respuestas de su banco de ejercicios
resueltos y ejemplos publicados en Hull y en el Comité de Basilea.
"""

import numpy as np
import pandas as pd
import pytest

from dsaplicado import frm, frm_avanzado as fa

# --- Valor del dinero en el tiempo, mismos casos que lib/finance.test.ts de frm-prep -----


def test_valor_futuro_y_presente_son_inversos():
    vf = fa.valor_futuro(10, 0.05, 1000)
    assert vf == pytest.approx(-1628.894627, abs=1e-4)
    assert fa.valor_presente(10, 0.05, 0, vf) == pytest.approx(1000, abs=1e-6)


def test_cuota_numero_de_periodos_y_tasa():
    assert fa.cuota(360, 0.005, 200_000) == pytest.approx(-1199.101, abs=1e-2)
    assert fa.numero_periodos(0.05, -1000, 0, 2000) == pytest.approx(14.2067, abs=1e-3)
    assert fa.tasa_por_periodo(10, -1000, 0, 1628.894627) == pytest.approx(0.05, abs=1e-6)


def test_tasa_cero_no_divide_por_cero():
    assert fa.valor_futuro(10, 0, 100, 10) == pytest.approx(-200)
    assert fa.valor_presente(10, 0, 10, 100) == pytest.approx(-200)
    assert fa.cuota(10, 0, 100, 100) == pytest.approx(-20)


def test_van_y_tir():
    assert fa.van(0.1, [-100, 60, 60]) == pytest.approx(4.132231, abs=1e-6)
    assert fa.tir([-100, 60, 60]) == pytest.approx(0.130662, abs=1e-5)
    for flujos in ([-500, 100, 100, 100, 100, 100, 100], [-2000, 0, 0, 3000], [-100, 230, -132]):
        assert fa.van(fa.tir(flujos), flujos) == pytest.approx(0, abs=1e-6)
    assert np.isnan(fa.tir([100, 50, 50]))


def test_tabla_amortizacion():
    tabla = fa.tabla_amortizacion(1000, 0.1, 3)
    assert tabla["cuota"].iloc[0] == pytest.approx(402.1148, abs=1e-3)
    assert tabla["interes"].iloc[0] == pytest.approx(100)
    assert tabla["saldo"].iloc[-1] == pytest.approx(0, abs=1e-6)
    larga = fa.tabla_amortizacion(200_000, 0.005, 360)
    assert larga["amortizacion"].sum() == pytest.approx(200_000, abs=1e-2)
    assert fa.tabla_amortizacion(1000, 0.05, 0).empty


def test_convenciones_de_conteo_de_dias():
    assert fa.fraccion_anio("2026-03-01", "2026-09-01", "30/360") == pytest.approx(0.5)
    assert fa.fraccion_anio("2024-01-01", "2025-01-01", "act/act") == pytest.approx(1.0)
    assert fa.fraccion_anio("2026-01-01", "2026-04-01", "act/360") == pytest.approx(90 / 360)


# --- Basilea III, mismos casos que lib/basel.test.ts de frm-prep -------------------------


def test_ratios_de_capital():
    r = fa.ratios_capital(cet1=100, at1=20, t2=30, apr=1000, exposicion=2500)
    assert (r["tier1"], r["total"]) == (120, 150)
    assert r["cet1_ratio"] == pytest.approx(0.10)
    assert r["tier1_ratio"] == pytest.approx(0.12)
    assert r["total_ratio"] == pytest.approx(0.15)
    assert r["apalancamiento_ratio"] == pytest.approx(0.048)
    assert fa.cumplimiento_basilea(r)[["cumple_minimo", "cumple_colchon"]].all().all()


def test_cumplimiento_en_zona_marginal_y_bajo_el_minimo():
    marginal = fa.cumplimiento_basilea(fa.ratios_capital(50, 0, 0, 1000, 3000)).loc["cet1"]
    assert marginal["cumple_minimo"] and not marginal["cumple_colchon"]
    bajo = fa.cumplimiento_basilea(fa.ratios_capital(40, 0, 0, 1000, 2000)).loc["cet1"]
    assert not bajo["cumple_minimo"]
    umbrales = fa.cumplimiento_basilea(fa.ratios_capital(70, 15, 20, 1000, 3000))["minimo_con_colchon"]
    assert umbrales["cet1"] == pytest.approx(0.07) and umbrales["total"] == pytest.approx(0.105)
    assert fa.apr_estandar([1000, 500, 400], [0.2, 1.0, 0.75]) == pytest.approx(1000)


# --- Respuestas del banco de ejercicios de frm-prep ---------------------------------------


def test_ejercicios_libro_1():
    assert fa.ratio_safety_first(0.10, 0.12, 0.03) == pytest.approx(0.583, abs=1e-3)
    cov = np.array([[0.20**2, 0.3 * 0.20 * 0.10], [0.3 * 0.20 * 0.10, 0.10**2]])
    w = np.array([0.6, 0.4])
    assert np.sqrt(w @ cov @ w) == pytest.approx(0.1374, abs=1e-4)
    minima = fa.pesos_minima_varianza(np.diag([0.12**2, 0.18**2]))
    assert minima[0] == pytest.approx(0.692, abs=1e-3)


def test_ejercicios_libro_2():
    assert fa.probabilidad_posterior(0.02, 0.95, 0.05) == pytest.approx(0.279, abs=1e-3)
    assert fa.valor_futuro(8, 0.06, -5000) == pytest.approx(7969.24, abs=1e-2)


def test_ejercicios_libro_3():
    assert fa.precio_bono(0.0, 0.05, 4, 1) == pytest.approx(82.27, abs=1e-2)
    assert fa.precio_bono(0.05, 0.06, 3, 1) == pytest.approx(97.33, abs=1e-2)
    assert fa.precio_bono(0.06, 0.08, 2, 2) == pytest.approx(96.37, abs=1e-2)
    assert fa.cambio_precio_aproximado(7, 100, 0.01) == pytest.approx(-0.065)
    assert frm.precio_forward(1.30, 0.04, 1, rendimiento=0.02) == pytest.approx(1.3263, abs=1e-4)
    assert 70 * np.exp(0.07) == pytest.approx(75.08, abs=1e-2)
    assert (60 - 2) * np.exp(0.05) == pytest.approx(60.97, abs=1e-2)
    assert 4 + 50 - 50 * np.exp(-0.05) == pytest.approx(6.44, abs=1e-2)


def test_ejercicios_libro_4():
    assert fa.agregar_riesgos([3, 4], [[1, 0.5], [0.5, 1]]) == pytest.approx(6.08, abs=1e-2)
    assert fa.pd_acumulada_constante(0.03, 2) == pytest.approx(0.0591, abs=1e-4)
    assert 100 * (1 + fa.cambio_precio_aproximado(6, 90, 0.015)) == pytest.approx(92.01, abs=1e-2)
    assert frm.black_scholes(50, 50, 0.05, 0.20, 1) == pytest.approx(5.23, abs=1e-2)


# --- Libro 1 ----------------------------------------------------------------------------


def test_cartera_tangente_maximiza_sharpe():
    mu, rf = np.array([0.08, 0.12, 0.10]), 0.03
    vol = np.array([0.15, 0.25, 0.20])
    cov = np.outer(vol, vol) * np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]])
    w = fa.pesos_tangencia(mu, cov, rf)

    def sharpe(p):
        return (p @ mu - rf) / np.sqrt(p @ cov @ p)

    rng = np.random.default_rng(0)
    aleatorias = rng.dirichlet(np.ones(3), 2000)
    assert w.sum() == pytest.approx(1)
    assert sharpe(w) >= max(sharpe(p) for p in aleatorias)


def test_cartera_de_arbitraje_apt():
    cargas = np.array([[1.0], [1.5], [2.0]])
    alineados = fa.cartera_arbitraje(cargas, np.array([0.08, 0.10, 0.12]))
    assert np.allclose(alineados, 0)
    w = fa.cartera_arbitraje(cargas, np.array([0.08, 0.10, 0.14]))
    assert w.sum() == pytest.approx(0, abs=1e-12)
    assert w @ cargas[:, 0] == pytest.approx(0, abs=1e-12)
    assert w @ np.array([0.08, 0.10, 0.14]) > 0


def test_medidas_de_desempeno_y_raroc():
    assert fa.m_cuadrado(0.14, 0.28, 0.04, 0.14) == pytest.approx(0.09)
    assert fa.raroc(10, 3, 2, 25, 0.04) == pytest.approx(0.24)
    assert fa.spread_cds(0.02, 0.4) == pytest.approx(0.012)
    assert fa.intensidad_desde_spread(0.012, 0.4) == pytest.approx(0.02)


def test_perdidas_de_cartera_y_tramos():
    perdidas = fa.perdidas_copula_gaussiana(0.02, 0.3, 125, simulaciones=200_000, semilla=1)
    assert perdidas.mean() == pytest.approx(0.02, rel=0.03)
    independientes = fa.perdidas_copula_gaussiana(0.02, 1e-9, 10_000, simulaciones=2_000)
    assert independientes.std() < 0.002
    assert np.allclose(fa.perdida_tramo(np.array([0.0, 0.04, 0.10]), 0.03, 0.07), [0, 0.25, 1])


# --- Libro 2 ----------------------------------------------------------------------------


def test_potencia_y_error_estandar_del_sharpe():
    assert fa.potencia_prueba_media(0, 1, 100) == pytest.approx(0.05)
    assert fa.potencia_prueba_media(0.5, 1, 100) > 0.99
    assert fa.error_estandar_sharpe(0.1, 100) == pytest.approx(np.sqrt(1.005 / 100))
    assert fa.error_estandar_sharpe(0.1, 100, asimetria=-1, curtosis=6) > fa.error_estandar_sharpe(0.1, 100)


def test_regresion_r2_ajustado_f_y_vif():
    assert fa.r2_ajustado(0.5, 101, 5) == pytest.approx(1 - 0.5 * 100 / 95)
    assert fa.estadistico_f(0.30, 0.0, 104, 3, 3) == pytest.approx((0.30 / 3) / (0.70 / 100))
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.standard_normal((5000, 3)), columns=["a", "b", "c"])
    assert np.allclose(fa.factores_inflacion_varianza(x), 1, atol=0.01)
    x["d"] = x["a"] + 0.1 * rng.standard_normal(5000)
    assert fa.factores_inflacion_varianza(x)["d"] > 50


def test_series_copulas_y_ridge():
    assert np.allclose(fa.pronostico_ar1(5, 2, 0.5, [1, 2]), [3.5, 2.75])
    assert fa.rho_desde_tau(0.5) == pytest.approx(np.sqrt(2) / 2)
    assert fa.rho_desde_spearman(1.0) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(20_000), rng.standard_normal(20_000)
    assert fa.dependencia_cola_inferior(x, y, 0.05) == pytest.approx(0.05, abs=0.02)
    assert fa.dependencia_cola_inferior(x, x, 0.05) == pytest.approx(1.0)
    from sklearn.linear_model import Ridge
    xm, ym = rng.standard_normal((200, 4)), rng.standard_normal(200)
    assert np.allclose(fa.ridge_cerrado(xm, ym, 3.0), Ridge(alpha=3.0, fit_intercept=False).fit(xm, ym).coef_)


def test_bootstrap_de_bloques_y_confusion():
    datos = np.arange(100.0)
    medias = fa.bootstrap_bloques(datos, np.mean, largo_bloque=10, repeticiones=500)
    assert medias.shape == (500,)
    assert medias.mean() == pytest.approx(49.5, abs=3)
    m = fa.metricas_confusion(vp=40, fp=10, fn=20, vn=130)
    assert m["precision"] == pytest.approx(0.8)
    assert m["sensibilidad"] == pytest.approx(40 / 60)
    assert m["exactitud"] == pytest.approx(0.85)


# --- Libro 3 ----------------------------------------------------------------------------


def test_seguro_de_vida_temporal():
    assert fa.prima_seguro_vida([0.0, 0.0], 0.05)["prima_anual"] == 0
    assert fa.prima_seguro_vida([0.01], 0.05, 1000)["prima_anual"] == pytest.approx(10 / 1.05)
    qx = fa.qx_gompertz_makeham(np.arange(30, 90))
    assert np.all(np.diff(qx) > 0)


def test_comisiones_con_marca_de_agua():
    tabla = fa.comisiones_fondo([0.30, -0.20, 0.15], gestion=0.02, desempeno=0.20)
    assert tabla.loc[1, "comision_desempeno"] == pytest.approx(0.2 * 28)
    assert tabla.loc[2, "comision_desempeno"] == 0
    # En el año 3 la cuota no supera su máximo anterior de 122,4, así que no paga desempeño.
    assert tabla.loc[3, "comision_desempeno"] == 0
    sin_marca = fa.comisiones_fondo([0.30, -0.20, 0.15], marca_agua=False)
    assert sin_marca.loc[3, "comision_desempeno"] > 0


def test_margen_colateral_y_cascada():
    precios = pd.Series([100.0, 98.0, 95.0, 97.0])
    cuenta = fa.cuenta_margen(precios, posicion=100, margen_inicial=1000, margen_mantencion=750)
    assert cuenta["llamado"].tolist() == [0.0, 500.0, 0.0]
    assert cuenta["saldo"].iloc[-1] == pytest.approx(1200)
    assert fa.llamado_colateral(12e6, 0, umbral=10e6, monto_minimo=0.5e6) == pytest.approx(2e6)
    assert fa.llamado_colateral(10.3e6, 0, umbral=10e6, monto_minimo=0.5e6) == 0
    cascada = fa.cascada_incumplimiento(150, {"margen inicial": 80, "fondo del miembro": 30, "capital propio": 20,
                                              "fondo mutualizado": 100})
    assert cascada.loc["fondo mutualizado", "usado"] == pytest.approx(20)
    assert cascada.loc["pérdida no cubierta", "usado"] == 0


def test_commodities_y_cotas_de_opciones():
    assert fa.rendimiento_conveniencia(70, 70 * np.exp(0.03), 0.05, 1, 0.02) == pytest.approx(0.04)
    cotas = fa.limites_opciones(50, 45, 0.05, 1)
    assert cotas.loc["call europea", "minimo"] == pytest.approx(50 - 45 * np.exp(-0.05))
    assert cotas.loc["put americana", "maximo"] == 45


def test_opciones_exoticas():
    s, k, r, sigma, t = 50, 50, 0.1, 0.4, 1
    efectivo = fa.opcion_binaria(s, k, r, sigma, t, "call", "efectivo")
    activo = fa.opcion_binaria(s, k, r, sigma, t, "call", "activo")
    assert activo - k * efectivo == pytest.approx(frm.black_scholes(s, k, r, sigma, t))
    assert efectivo + fa.opcion_binaria(s, k, r, sigma, t, "put") == pytest.approx(np.exp(-r * t))
    # Hull, ejemplo de opción asiática: el promedio geométrico vale 5,13.
    assert fa.call_asiatica_geometrica(s, k, r, sigma, t) == pytest.approx(5.13, abs=0.01)
    vainilla = frm.black_scholes(100, 100, 0.05, 0.25, 1)
    for h, strike in ((90, 100), (90, 85)):
        entrada = fa.call_barrera_abajo(100, strike, h, 0.05, 0.25, 1, variante="in")
        salida = fa.call_barrera_abajo(100, strike, h, 0.05, 0.25, 1, variante="out")
        assert entrada + salida == pytest.approx(frm.black_scholes(100, strike, 0.05, 0.25, 1))
    assert fa.call_barrera_abajo(100, 100, 1e-6, 0.05, 0.25, 1) == pytest.approx(vainilla)
    # Margrabe con un activo sin volatilidad y rendimiento r es una call con strike fijo.
    assert fa.opcion_intercambio(50, 45, 0.3, 0.0, 0.0, 1, 0.0, 0.05) == pytest.approx(frm.black_scholes(50, 45, 0.05, 0.3, 1))


def test_asiatica_geometrica_discreta():
    continua = fa.call_asiatica_geometrica(50, 50, 0.1, 0.4, 1)
    assert fa.call_asiatica_geometrica_discreta(50, 50, 0.1, 0.4, 1, 1_000_000) == pytest.approx(continua, abs=1e-4)
    # Con una sola observación es una call europea.
    assert fa.call_asiatica_geometrica_discreta(50, 50, 0.1, 0.4, 1, 1) == pytest.approx(frm.black_scholes(50, 50, 0.1, 0.4, 1))
    rutas = fa.simular_gbm(50, 0.1, 0.4, 1, 12, 200_000, semilla=5)
    geometrico = np.exp(np.log(rutas[:, 1:]).mean(axis=1))
    precio, error = fa.precio_montecarlo(np.maximum(geometrico - 50, 0), 0.1, 1)
    assert precio == pytest.approx(fa.call_asiatica_geometrica_discreta(50, 50, 0.1, 0.4, 1, 12), abs=3 * error)


def test_barrera_coincide_con_montecarlo():
    pasos = 1000
    rutas = fa.simular_gbm(100, 0.05, 0.25, 1, pasos, 20_000, semilla=3)
    pago = np.where(rutas.min(axis=1) > 90, np.maximum(rutas[:, -1] - 100, 0), 0)
    precio, error = fa.precio_montecarlo(pago, 0.05, 1)
    # Corrección de Broadie, Glasserman y Kou para monitoreo discreto.
    barrera_continua = 90 * np.exp(-0.5826 * 0.25 * np.sqrt(1 / pasos))
    assert precio == pytest.approx(fa.call_barrera_abajo(100, 100, barrera_continua, 0.05, 0.25, 1), abs=3 * error)


def test_black_scholes_vectorial_coincide_con_el_escalar():
    precios = np.array([80.0, 100.0, 120.0])
    for tipo in ("call", "put"):
        vector = fa.black_scholes_vectorial(precios, 100, 0.05, 0.2, 0.5, tipo, q=0.02)
        escalar = [frm.black_scholes(s, 100, 0.05, 0.2, 0.5, tipo, q=0.02) for s in precios]
        assert np.allclose(vector, escalar)


def test_antiteticas_y_black_76():
    rutas = fa.simular_gbm(100, 0.05, 0.2, 1, 1, 10_000, antiteticas=True)
    assert rutas.shape == (10_000, 2)
    assert fa.black_76(100, 95, 0.05, 0.2, 0.5) == pytest.approx(frm.black_scholes(100, 95, 0.05, 0.2, 0.5, q=0.05))


def test_bootstrap_de_curva_plana():
    plazos = np.arange(1, 21) / 2
    factores = fa.bootstrap_par(plazos, np.full(20, 0.05))
    assert np.allclose(factores, 1.025 ** -np.arange(1, 21))
    assert np.allclose(fa.forwards_periodo(plazos, factores), 0.05)
    assert np.allclose(fa.tasas_cero_continuas(plazos, factores), 2 * np.log(1.025))
    with pytest.raises(ValueError):
        fa.bootstrap_par(np.array([1.0, 3.0]), np.array([0.05, 0.05]))


def test_metricas_de_bonos():
    cero = fa.metricas_bono(0.0, 0.06, 5, 1)
    assert cero["macaulay"] == pytest.approx(5)
    assert cero["modificada"] == pytest.approx(5 / 1.06)
    assert cero["convexidad"] == pytest.approx(5 * 6 / 1.06**2)
    m = fa.metricas_bono(0.05, 0.06, 10, 2)
    assert fa.rendimiento_bono(m["precio"], 0.05, 10, 2) == pytest.approx(0.06)
    dy = 1e-4
    numerica = (fa.precio_bono(0.05, 0.06 - dy, 10) - fa.precio_bono(0.05, 0.06 + dy, 10)) / (2 * dy * m["precio"])
    assert m["modificada"] == pytest.approx(numerica, rel=1e-6)


def test_futuros_de_tasas_y_prepago():
    # Hull: cartera de 10 millones con duración 6,8 y futuro de 93.062,50 con duración 9,2.
    assert fa.contratos_duracion(10e6, 6.8, 93_062.50, 9.2) == pytest.approx(-79.42, abs=0.01)
    # Hull: σ = 1,2%, futuro a 8 años sobre una tasa a 3 meses.
    assert fa.ajuste_convexidad(0.012, 8, 8.25) == pytest.approx(0.00475, abs=1e-5)
    assert fa.smm_psa(30) == pytest.approx(frm.smm_desde_cpr(0.06))
    assert fa.smm_psa(10, 150) == pytest.approx(frm.smm_desde_cpr(0.03))


# --- Libro 4 ----------------------------------------------------------------------------


def test_medida_espectral_va_de_la_media_al_peor_caso():
    perdidas = np.arange(1, 101)
    assert fa.medida_espectral(perdidas, 1e-6) == pytest.approx(50.5, abs=0.01)
    assert fa.medida_espectral(perdidas, 1000) == pytest.approx(100, abs=0.1)
    valores = [fa.medida_espectral(perdidas, k) for k in (1, 5, 20)]
    assert valores == sorted(valores)


def test_cornish_fisher_y_descomposicion_del_var():
    assert fa.var_cornish_fisher(0, 1, 0, 0, 0.99) == pytest.approx(2.3263, abs=1e-4)
    assert fa.var_cornish_fisher(0, 1, -0.5, 3, 0.99) > 2.3263
    cov = np.array([[0.02**2, 0.5 * 0.02 * 0.01], [0.5 * 0.02 * 0.01, 0.01**2]])
    tabla = fa.descomposicion_var([100, 200], cov, 0.99)
    total = 2.3263478740 * np.sqrt(np.array([100, 200]) @ cov @ np.array([100, 200]))
    assert tabla["var_componente"].sum() == pytest.approx(total)
    assert tabla["porcentaje"].sum() == pytest.approx(1)


def test_valores_extremos_recuperan_la_cola():
    rng = np.random.default_rng(0)
    perdidas = rng.pareto(3, 50_000) + 1
    ajuste = fa.ajustar_pot(perdidas, np.quantile(perdidas, 0.95))
    assert ajuste["xi"] == pytest.approx(1 / 3, abs=0.05)
    resultado = fa.var_es_pot(ajuste, 0.999)
    assert resultado["var"] == pytest.approx(0.001 ** (-1 / 3), rel=0.05)
    assert resultado["es"] > resultado["var"]


def test_volatilidad_implicita():
    precio = frm.black_scholes(100, 105, 0.03, 0.27, 0.75, "put")
    assert fa.volatilidad_implicita(precio, 100, 105, 0.03, 0.75, "put") == pytest.approx(0.27, abs=1e-8)


def test_capital_irb_de_basilea():
    # Nota explicativa del Comité de Basilea: PD de 1%, LGD de 45% y M de 2,5 dan 92,32%.
    resultado = fa.capital_irb(0.01, 0.45)
    assert resultado["ponderacion"] == pytest.approx(0.9232, abs=1e-4)
    assert float(fa.correlacion_irb(1e-9)) == pytest.approx(0.24, abs=1e-6)
    assert float(fa.correlacion_irb(0.99)) == pytest.approx(0.12, abs=1e-6)
    assert np.allclose(fa.pd_marginal([0.01, 0.03, 0.06]), [0.01, 0.02, 0.03])


def test_modelo_de_merton_ejemplo_de_hull():
    activos = fa.activos_merton(patrimonio=3, sigma_patrimonio=0.80, deuda=10, r=0.05, t=1)
    assert activos["activos"] == pytest.approx(12.40, abs=0.01)
    assert activos["sigma_activos"] == pytest.approx(0.2123, abs=1e-4)
    assert fa.pd_merton(activos["activos"], 10, 0.05, activos["sigma_activos"])["pd"] == pytest.approx(0.127, abs=1e-3)
    assert fa.z_altman(1, 1, 1, 1, 1, 10, 10) == pytest.approx(0.75)


def test_riesgo_operacional_estandar():
    assert fa.multiplicador_perdidas_internas(100, 100) == pytest.approx(1.0)
    alto = fa.capital_riesgo_operacional(5_000, perdidas_anuales_promedio=100)
    assert alto["bic"] == pytest.approx(720)
    assert alto["ilm"] > 1
    assert fa.capital_riesgo_operacional(500)["capital"] == pytest.approx(60)


def test_estres_inverso():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    e = np.array([100.0, 50.0])
    resultado = fa.estres_inverso(e, cov, 30)
    assert -e @ resultado["shocks"] == pytest.approx(30)
    assert resultado["shocks"] @ np.linalg.solve(cov, resultado["shocks"]) == pytest.approx(resultado["distancia"] ** 2)


def test_neutralizar_griegas():
    instrumentos = pd.DataFrame({"delta": [0.6, 0.5], "gamma": [0.5, 0.8], "vega": [2.0, 1.2]}, index=["opcion 1", "opcion 2"])
    cartera = {"delta": 0.0, "gamma": -5000.0, "vega": -8000.0}
    cantidades = fa.neutralizar_griegas(cartera, instrumentos)
    opciones = cantidades[instrumentos.index]
    assert cartera["gamma"] + instrumentos["gamma"] @ opciones == pytest.approx(0, abs=1e-8)
    assert cartera["vega"] + instrumentos["vega"] @ opciones == pytest.approx(0, abs=1e-8)
    assert cartera["delta"] + instrumentos["delta"] @ opciones + cantidades["subyacente"] == pytest.approx(0, abs=1e-8)
    partes = fa.atribucion_taylor(0.5, 0.04, 20, -5, 2, 0.01, 1 / 365)
    assert partes["total"] == pytest.approx(1 + 0.08 + 0.2 - 5 / 365)


def test_factores_desde_bonos_y_replica():
    factores_verdaderos = np.array([0.97, 0.93, 0.89])
    flujos = np.array([[105, 0, 0], [6, 106, 0], [4, 4, 104], [8, 8, 108]])
    precios = flujos @ factores_verdaderos
    resultado = fa.factores_desde_bonos(flujos, precios)
    assert np.allclose(resultado["factores"], factores_verdaderos)
    assert np.allclose(resultado["residuos"], 0)
    cantidades = fa.replicar_flujos(np.array([5, 5, 105]), flujos[:3])
    assert np.allclose(cantidades @ flujos[:3], [5, 5, 105])
