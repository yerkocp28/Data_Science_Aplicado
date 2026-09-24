import numpy as np
import pandas as pd
import pytest

from dsaplicado import tasas


@pytest.fixture
def referencia():
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2010-01-01", periods=240, freq="MS")
    return pd.Series(np.cumsum(rng.choice([-0.5, -0.25, 0, 0, 0.25, 0.5], size=240)), index=fechas)


def test_beta_recupera_traspaso_distribuido(referencia):
    rng = np.random.default_rng(1)
    cambio = referencia.diff().fillna(0)
    # 50% del cambio el mismo mes y 30% el mes siguiente: beta de largo plazo 0,8.
    tasa = (0.5 * cambio + 0.3 * cambio.shift(1).fillna(0) + rng.normal(0, 0.01, len(cambio))).cumsum()
    res = tasas.beta_traspaso(tasa, referencia, rezagos=3)
    assert res["impacto"] == pytest.approx(0.5, abs=0.03)
    assert res["largo_plazo"] == pytest.approx(0.8, abs=0.05)
    assert res["error_estandar"] > 0


def test_asimetria_detecta_traspaso_distinto(referencia):
    cambio = referencia.diff().fillna(0)
    tasa = (0.9 * cambio.clip(lower=0) + 0.4 * cambio.clip(upper=0)).cumsum()
    res = tasas.beta_asimetrica(tasa, referencia, rezagos=2)
    assert res["beta_alzas"] == pytest.approx(0.9, abs=0.01)
    assert res["beta_bajas"] == pytest.approx(0.4, abs=0.01)
    assert res["p_igualdad"] < 0.01


def test_beta_ciclo():
    fechas = pd.date_range("2021-01-01", periods=3, freq="MS")
    ref = pd.Series([1.0, 5.0, 11.0], index=fechas)
    tasa = pd.Series([0.5, 3.0, 8.5], index=fechas)
    assert tasas.beta_ciclo(tasa, ref, "2021-01-01", "2021-03-01") == pytest.approx(0.8)
    with pytest.raises(ValueError):
        tasas.beta_ciclo(tasa, pd.Series(1.0, index=fechas), "2021-01-01", "2021-03-01")


def test_nelson_siegel_recupera_parametros():
    plazos = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 10])
    verdad = {"nivel": 6.0, "beta1": -1.5, "curvatura": 0.8}
    tasas_curva = tasas.curva_nelson_siegel(verdad, plazos)
    ajuste = tasas.ajustar_nelson_siegel(plazos, tasas_curva)
    assert ajuste["nivel"] == pytest.approx(6.0)
    assert ajuste["pendiente"] == pytest.approx(1.5)
    assert ajuste["rmse"] == pytest.approx(0, abs=1e-10)


def test_duracion_de_bono_cero_cupon_es_su_plazo():
    tiempos, montos = np.array([5.0]), np.array([100.0])
    res = tasas.duracion_convexidad(tiempos, montos, np.array([5.0]))
    assert res["duracion"] == pytest.approx(5.0)
    assert res["convexidad"] == pytest.approx(25.0)
    assert res["precio"] == pytest.approx(100 * np.exp(-0.25))


def test_duracion_predice_cambio_de_precio():
    tiempos, montos = tasas.flujos_bono(cupon=5, plazo=10)
    curva = np.full_like(tiempos, 5.0)
    base = tasas.duracion_convexidad(tiempos, montos, curva)
    nuevo = tasas.precio_con_curva(tiempos, montos, curva + 0.1)  # 10 puntos base
    aproximado = base["precio"] * (1 - base["duracion"] * 0.001 + 0.5 * base["convexidad"] * 0.001**2)
    assert nuevo == pytest.approx(aproximado, rel=1e-6)


def test_escenarios_basilea_formas():
    plazos = np.array([0.25, 1, 5, 20])
    esc = tasas.escenarios_basilea(plazos, paralelo=4, corto=5, largo=3)
    assert (esc["paralelo arriba"] == 4).all()
    assert esc["corto arriba"].is_monotonic_decreasing
    assert esc.loc[0.25, "empinamiento"] < 0 < esc.loc[20, "empinamiento"]
    assert esc.loc[0.25, "aplanamiento"] > 0 > esc.loc[20, "aplanamiento"]
