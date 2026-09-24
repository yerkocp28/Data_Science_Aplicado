import numpy as np
import pandas as pd
import pytest

from dsaplicado import portafolio as pf


@pytest.fixture
def retornos_sinteticos():
    rng = np.random.default_rng(0)
    fechas = pd.bdate_range("2020-01-01", periods=800)
    cov = np.array([[0.0004, 0.0001, 0.0], [0.0001, 0.0002, 0.00005], [0.0, 0.00005, 0.0003]])
    datos = rng.multivariate_normal([0.0005, 0.0003, 0.0004], cov, size=len(fechas))
    return pd.DataFrame(datos, index=fechas, columns=["A", "B", "C"])


def test_minima_varianza_dos_activos_coincide_con_formula_cerrada():
    cov = np.array([[0.04, 0.006], [0.006, 0.01]])
    w = pf.minima_varianza(cov)
    peso_a = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert w == pytest.approx([peso_a, 1 - peso_a], abs=1e-6)


def test_pesos_optimos_suman_uno_y_respetan_tope():
    mu = np.array([0.10, 0.12, 0.08, 0.15])
    cov = np.diag([0.04, 0.05, 0.02, 0.09])
    for w in (pf.minima_varianza(cov, cota_max=0.4), pf.maximo_sharpe(mu, cov, 0.02, cota_max=0.4)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-12).all() and (w <= 0.4 + 1e-9).all()


def test_tope_imposible_lanza_error():
    with pytest.raises(ValueError):
        pf.minima_varianza(np.eye(3), cota_max=0.3)


def test_maximo_sharpe_supera_a_portafolios_aleatorios():
    mu = np.array([0.10, 0.12, 0.08, 0.15])
    cov = np.array(
        [[0.04, 0.01, 0.0, 0.01], [0.01, 0.05, 0.01, 0.0], [0.0, 0.01, 0.02, 0.0], [0.01, 0.0, 0.0, 0.09]]
    )
    w = pf.maximo_sharpe(mu, cov, rf=0.02)
    ret, vol = pf.rendimiento_portafolio(w, mu, cov)
    sim = pf.simular_portafolios(mu, cov, n=5000, rf=0.02)
    assert (ret - 0.02) / vol >= sim["sharpe"].max() - 1e-9


def test_frontera_es_creciente_en_retorno_y_volatilidad():
    mu = np.array([0.10, 0.12, 0.08, 0.15])
    cov = np.diag([0.04, 0.05, 0.02, 0.09])
    fr = pf.frontera_eficiente(mu, cov, n_puntos=15)
    assert fr["retorno"].is_monotonic_increasing
    assert fr["volatilidad"].is_monotonic_increasing


def test_beta_recupera_parametro_simulado():
    rng = np.random.default_rng(1)
    mercado = pd.Series(rng.normal(0.0004, 0.01, 3000))
    activo = 0.0001 + 1.3 * mercado + rng.normal(0, 0.005, 3000)
    res = pf.beta_capm(activo.to_frame("X"), mercado)
    assert res.loc["X", "beta"] == pytest.approx(1.3, abs=0.03)


def test_backtest_sin_mirar_el_futuro(retornos_sinteticos):
    vistos = []

    def estrategia(ventana):
        vistos.append(ventana.index.max())
        return np.full(ventana.shape[1], 1 / ventana.shape[1])

    ret, pesos = pf.backtest_rebalanceo(retornos_sinteticos, estrategia, ventana=250, frecuencia="QE")
    # Cada período se estima con datos hasta el corte y se evalúa después del corte.
    assert all(f in pesos.index for f in vistos)
    assert pesos["rotacion"].iloc[0] == 1.0
    assert ret.index.min() > pesos.index.min()
    assert not ret.index.duplicated().any()


def test_backtest_pesos_iguales_con_deriva_coincide_con_valor_manual(retornos_sinteticos):
    ret, pesos = pf.backtest_rebalanceo(
        retornos_sinteticos, lambda v: np.full(3, 1 / 3), ventana=250, frecuencia="QE"
    )
    pesos = pesos.drop(columns="rotacion")
    corte, siguiente = pesos.index[0], pesos.index[1]
    periodo = retornos_sinteticos.loc[(retornos_sinteticos.index > corte) & (retornos_sinteticos.index <= siguiente)]
    valor_manual = ((1 + periodo).prod() / 3).sum()
    assert (1 + ret.loc[periodo.index]).prod() == pytest.approx(valor_manual)


def test_metricas_desempeno_drawdown():
    r = pd.Series([0.10, -0.20, 0.05])
    m = pf.metricas_desempeno(r)
    assert m["max_drawdown"] == pytest.approx(-0.20)


def test_costos_reducen_el_retorno_segun_rotacion(retornos_sinteticos):
    alternar = iter([np.array([1.0, 0, 0]), np.array([0, 1.0, 0])] * 20)
    estrategia = lambda v: next(alternar)
    bruto, hist = pf.backtest_rebalanceo(retornos_sinteticos, estrategia, ventana=250, frecuencia="QE")
    alternar = iter([np.array([1.0, 0, 0]), np.array([0, 1.0, 0])] * 20)
    neto, _ = pf.backtest_rebalanceo(retornos_sinteticos, estrategia, ventana=250, frecuencia="QE", costo=0.01)
    # Pasar de un activo a otro rota el 200% del portafolio.
    assert hist["rotacion"].iloc[1] == pytest.approx(2.0)
    primer_dia = neto.index[0]
    assert (1 + neto.loc[primer_dia]) == pytest.approx((1 + bruto.loc[primer_dia]) * 0.99)
