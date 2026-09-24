import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dsaplicado import riesgo as rg


def test_var_y_es_historicos_en_muestra_conocida():
    r = -np.arange(1, 101) / 1000  # pérdidas de 0.1% a 10%
    assert rg.var_historico(r, 0.95) == pytest.approx(0.09505, rel=1e-3)
    assert rg.es_historico(r, 0.95) >= rg.var_historico(r, 0.95)


def test_var_normal_coincide_con_cuantil():
    assert rg.var_normal(0.0, 0.01, 0.99) == pytest.approx(0.02326, abs=1e-5)
    assert rg.es_normal(0.0, 0.01, 0.975) == pytest.approx(0.02338, abs=1e-5)


def test_es_t_coincide_con_simulacion():
    gl, escala = 4.0, 0.01
    muestra = stats.t.rvs(gl, scale=escala, size=2_000_000, random_state=3)
    assert rg.es_t(gl, 0.0, escala, 0.99) == pytest.approx(rg.es_historico(muestra, 0.99), rel=0.02)


def test_es_t_converge_a_normal_con_muchos_grados_de_libertad():
    assert rg.es_t(1e6, 0.0, 0.01, 0.99) == pytest.approx(rg.es_normal(0.0, 0.01, 0.99), rel=1e-4)


def test_ewma_usa_solo_informacion_previa():
    r = pd.Series([0.0] * 40 + [0.10] + [0.0] * 5)
    vol = rg.volatilidad_ewma(r)
    assert vol.iloc[40] == pytest.approx(vol.iloc[39])  # el choque aún no se conoce
    assert vol.iloc[41] > vol.iloc[40]


def test_kupiec_no_rechaza_con_tasa_exacta_y_rechaza_con_exceso():
    exacto = np.zeros(1000)
    exacto[:10] = 1
    assert rg.prueba_kupiec(exacto, 0.99)["p_valor"] == pytest.approx(1.0)
    exceso = np.zeros(1000)
    exceso[:30] = 1
    assert rg.prueba_kupiec(exceso, 0.99)["p_valor"] < 0.001


def test_kupiec_sin_excepciones_es_finito():
    res = rg.prueba_kupiec(np.zeros(500), 0.99)
    assert np.isfinite(res["estadistico"]) and res["excepciones"] == 0


def test_christoffersen_detecta_agrupamiento():
    disperso = np.zeros(1000)
    disperso[::100] = 1
    agrupado = np.zeros(1000)
    agrupado[500:510] = 1
    assert rg.prueba_christoffersen(disperso)["p_valor"] > 0.05
    assert rg.prueba_christoffersen(agrupado)["p_valor"] < 0.001


@pytest.mark.parametrize("n,zona", [(0, "verde"), (4, "verde"), (5, "amarilla"), (9, "amarilla"), (10, "roja")])
def test_semaforo_basilea(n, zona):
    assert rg.semaforo_basilea(n) == zona


@pytest.mark.parametrize("metodo", rg.METODOS)
def test_pronostico_alineado_y_positivo(metodo):
    rng = np.random.default_rng(5)
    r = pd.Series(rng.standard_t(5, 600) * 0.01, index=pd.bdate_range("2021-01-01", periods=600))
    pron = rg.pronosticar_var_es(r, metodo, 0.99, ventana=250)
    assert pron.index[0] == r.index[250]
    assert (pron["var"] > 0).all() and (pron["es"] >= pron["var"] - 1e-12).all()


def test_pronostico_historico_no_usa_el_dia_evaluado():
    r = pd.Series(np.full(300, -0.001), index=pd.bdate_range("2021-01-01", periods=300))
    r.iloc[260] = -0.5
    pron = rg.pronosticar_var_es(r, "historico", 0.99, ventana=250)
    assert pron.loc[r.index[260], "var"] == pytest.approx(0.001)
