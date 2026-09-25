"""Pruebas de las fórmulas FRM contra ejemplos conocidos de los textos de referencia."""

import numpy as np
import pandas as pd
import pytest

from dsaplicado import frm

# --- Libro 1 -------------------------------------------------------------------------


def test_medidas_de_desempeno():
    assert frm.indice_sharpe(0.12, 0.20, 0.04) == pytest.approx(0.40)
    assert frm.indice_treynor(0.12, 0.8, 0.04) == pytest.approx(0.10)
    # CAPM predice 4% + 1,2 · (10% - 4%) = 11,2%; el alfa es 13% - 11,2%.
    assert frm.alfa_jensen(0.13, 1.2, 0.04, 0.10) == pytest.approx(0.018)


def test_sortino_solo_penaliza_caidas():
    subidas = pd.Series([0.01, 0.02, -0.01, 0.03] * 50)
    mas_subidas = pd.Series([0.01, 0.05, -0.01, 0.03] * 50)
    assert frm.indice_sortino(mas_subidas, 0.0) > frm.indice_sortino(subidas, 0.0)


def test_indice_informacion_cero_si_replica_la_referencia():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    ruido = pd.Series(np.random.default_rng(1).normal(0, 0.001, 500))
    assert abs(frm.indice_informacion(r + ruido, r)) < 2.5


# --- Libro 2 -------------------------------------------------------------------------


def test_garch_largo_plazo_ejemplo_hull():
    # Hull: omega = 0,000002, alfa = 0,13, beta = 0,86 -> V_L = 0,0002, volatilidad diaria 1,4%.
    vl = frm.varianza_largo_plazo_garch(0.000002, 0.13, 0.86)
    assert vl == pytest.approx(0.0002)
    assert np.sqrt(vl) == pytest.approx(0.01414, abs=1e-4)
    with pytest.raises(ValueError):
        frm.varianza_largo_plazo_garch(0.000002, 0.2, 0.8)


def test_pronostico_garch_converge_al_largo_plazo():
    v = frm.pronostico_varianza_garch(0.0004, 0.000002, 0.13, 0.86, dias=2000)
    assert v == pytest.approx(0.0002, rel=1e-3)
    assert frm.vida_media(0.5) == pytest.approx(1.0)


def test_ewma_usa_retorno_anterior():
    var = frm.varianza_ewma(np.array([0.0, 0.1, 0.0]), lam=0.9, inicial=0.0)
    assert var[1] == pytest.approx(0.0)
    assert var[2] == pytest.approx(0.1 * 0.01)


# --- Libro 3 -------------------------------------------------------------------------


def test_conversion_de_tasas_ejemplos_hull():
    assert frm.tasa_continua(0.10, 2) == pytest.approx(0.09758, abs=1e-5)
    assert frm.tasa_desde_continua(0.08, 4) == pytest.approx(0.08081, abs=1e-5)
    assert frm.tasa_desde_continua(frm.tasa_continua(0.07, 12), 12) == pytest.approx(0.07)


def test_tasas_forward_ejemplo_hull():
    # Hull, tabla de tasas cero 3%, 4%, 4,6%, 5% a 1, 2, 3 y 4 años: forwards 5%, 5,8%, 6,2%.
    f = frm.tasas_forward(np.array([1, 2, 3, 4]), np.array([0.03, 0.04, 0.046, 0.05]))
    assert f == pytest.approx([0.05, 0.058, 0.062])


def test_precio_forward_ejemplo_hull():
    assert frm.precio_forward(40, 0.05, 0.25) == pytest.approx(40.50, abs=0.01)
    # Con rendimiento igual a la tasa, el forward es igual al spot.
    assert frm.precio_forward(100, 0.05, 1, rendimiento=0.05) == pytest.approx(100)


def test_forward_divisas_paridad_cubierta():
    f = frm.forward_divisas(900, 0.05, 0.04, 1)
    assert f == pytest.approx(900 * 1.05 / 1.04)


def test_cobertura_minima_varianza_ejemplo_hull():
    # Ejemplo del combustible de aviones: rho 0,928, sigma_S 0,0263, sigma_F 0,0313 -> h* 0,78.
    h = frm.razon_cobertura_minima_varianza(0.928, 0.0263, 0.0313)
    assert h == pytest.approx(0.78, abs=1e-3)
    assert frm.contratos_cobertura(h, 2_000_000, 42_000) == pytest.approx(37.13, abs=0.05)


def test_ajuste_de_beta_con_futuros():
    # Reducir la beta de 1,2 a 0,6 en una cartera de 5 millones con futuros de 250.000 por contrato.
    assert frm.contratos_ajuste_beta(1.2, 0.6, 5_000_000, 250_000) == pytest.approx(-12)


def test_paridad_put_call_y_estrategias():
    s, k, r, sigma, t = 50, 50, 0.05, 0.3, 1
    c, p = frm.black_scholes(s, k, r, sigma, t, "call"), frm.black_scholes(s, k, r, sigma, t, "put")
    assert c + k * np.exp(-r * t) == pytest.approx(p + s)
    precios = np.array([40.0, 50.0, 60.0])
    straddle = frm.pago_estrategia(precios, [("call", 50, 1, 0), ("put", 50, 1, 0)])
    assert list(straddle) == [10, 0, 10]
    bull = frm.pago_estrategia(precios, [("call", 45, 1, 0), ("call", 55, -1, 0)])
    assert list(bull) == [0, 5, 10]


def test_swap_a_tasa_par_vale_cero():
    plazos = np.arange(0.5, 5.01, 0.5)
    ceros = 0.05 + 0.002 * plazos
    par = frm.tasa_swap_par(plazos, ceros)
    assert frm.valor_swap_tasa(100, par, plazos, ceros) == pytest.approx(0, abs=1e-10)
    # Si las tasas suben, quien paga fijo gana.
    assert frm.valor_swap_tasa(100, par, plazos, ceros + 0.01) > 0


def test_hipoteca_ejemplo_clasico():
    assert frm.cuota_hipotecaria(200_000, 0.06, 360) == pytest.approx(1199.10, abs=0.01)
    assert frm.smm_desde_cpr(0.06) == pytest.approx(0.005143, abs=1e-6)
    flujos = frm.flujos_hipoteca(100_000, 0.06, 360, cpr=0.0)
    assert flujos["amortizacion"].sum() == pytest.approx(100_000)
    con_prepago = frm.flujos_hipoteca(100_000, 0.06, 360, cpr=0.10)
    assert (con_prepago["amortizacion"] + con_prepago["prepago"]).sum() == pytest.approx(100_000)
    assert con_prepago["interes"].sum() < flujos["interes"].sum()


def test_compensacion_multilateral_reduce_exposicion():
    # A le debe 100 a B, B le debe 100 a C, C le debe 100 a A: con cámara central todo se anula.
    m = pd.DataFrame([[0, 100, 0], [0, 0, 100], [100, 0, 0]], index=list("ABC"), columns=list("ABC"))
    assert frm.exposicion_neta(m, multilateral=False) == 300
    assert frm.exposicion_neta(m, multilateral=True) == 0


# --- Libro 4 -------------------------------------------------------------------------


def test_black_scholes_ejemplo_hull():
    assert frm.black_scholes(42, 40, 0.10, 0.20, 0.5, "call") == pytest.approx(4.76, abs=0.005)
    assert frm.black_scholes(42, 40, 0.10, 0.20, 0.5, "put") == pytest.approx(0.81, abs=0.005)


def test_griegas_ejemplo_hull():
    g = frm.griegas(49, 50, 0.05, 0.20, 20 / 52, "call")
    assert g["delta"] == pytest.approx(0.522, abs=0.001)
    assert g["gamma"] == pytest.approx(0.066, abs=0.001)
    assert g["vega"] == pytest.approx(12.1, abs=0.05)
    assert g["theta"] == pytest.approx(-4.31, abs=0.01)
    assert g["rho"] == pytest.approx(8.91, abs=0.01)


def test_griegas_coinciden_con_diferencias_finitas():
    base = dict(s=100, k=95, r=0.04, sigma=0.25, t=0.75, tipo="put", q=0.02)
    g = frm.griegas(**base)
    h = 0.01
    arriba = frm.black_scholes(**{**base, "s": 100 + h})
    abajo = frm.black_scholes(**{**base, "s": 100 - h})
    assert g["delta"] == pytest.approx((arriba - abajo) / (2 * h), abs=1e-5)


def test_arbol_binomial_put_americana_ejemplo_hull():
    # Hull: S = 50, K = 50, r = 10%, sigma = 40%, T = 5 meses, 5 pasos -> 4,49.
    assert frm.arbol_binomial(50, 50, 0.10, 0.40, 5 / 12, 5, "put", americana=True) == pytest.approx(4.49, abs=0.005)


def test_arbol_converge_a_black_scholes():
    europea = frm.arbol_binomial(50, 52, 0.05, 0.30, 2, 500, "put", americana=False)
    assert europea == pytest.approx(frm.black_scholes(50, 52, 0.05, 0.30, 2, "put"), abs=0.01)


def test_credito_perdidas():
    assert frm.perdida_esperada(0.02, 0.4, 1_000_000) == pytest.approx(8_000)
    assert frm.perdida_inesperada(0.02, 0.4, 1_000_000) == pytest.approx(56_000, rel=1e-3)
    # Sin correlación, el peor caso tiende a la PD; con más correlación, empeora.
    assert frm.tasa_default_peor_caso(0.01, 1e-9) == pytest.approx(0.01, abs=1e-4)
    assert frm.tasa_default_peor_caso(0.01, 0.2) > frm.tasa_default_peor_caso(0.01, 0.1)


def test_transicion_multiperiodo():
    m = pd.DataFrame([[0.9, 0.1], [0.0, 1.0]], index=["A", "D"], columns=["A", "D"])
    dos = frm.transicion_multiperiodo(m, 2)
    assert dos.loc["A", "D"] == pytest.approx(0.19)
    assert np.allclose(dos.sum(axis=1), 1)


def test_perdidas_operacionales_media():
    perdidas = frm.perdidas_operacionales(10, 0.0, 1.0, simulaciones=200_000)
    assert perdidas.mean() == pytest.approx(10 * np.exp(0.5), rel=0.02)


def test_componente_indicador_negocio():
    assert frm.componente_indicador_negocio(800) == pytest.approx(96)
    assert frm.componente_indicador_negocio(2_000) == pytest.approx(120 + 150)
    assert frm.componente_indicador_negocio(40_000) == pytest.approx(120 + 29_000 * 0.15 + 10_000 * 0.18)


def test_dv01():
    assert frm.dv01(100, 7.5) == pytest.approx(0.075)
