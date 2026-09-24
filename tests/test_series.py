import numpy as np
import pandas as pd
import pytest

from dsaplicado import riesgo as rg
from dsaplicado import series as st


def test_indice_y_variacion_anual_son_consistentes():
    variaciones = pd.Series([1.0] * 24, index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    indice = st.indice_desde_variaciones(variaciones)
    anual = st.variacion_anual(indice).dropna()
    assert indice.iloc[0] == pytest.approx(101.0)
    assert anual.iloc[0] == pytest.approx(100 * (1.01**12 - 1))


def test_a_mensual_fecha_el_primer_dia_del_mes():
    diaria = pd.Series(range(60), index=pd.bdate_range("2024-01-01", periods=60), dtype=float)
    mensual = st.a_mensual(diaria, "last")
    assert all(f.day == 1 for f in mensual.index)
    assert mensual.iloc[0] == diaria.loc["2024-01"].iloc[-1]


def test_raiz_unitaria_distingue_paseo_aleatorio_de_ruido():
    rng = np.random.default_rng(0)
    ruido = pd.Series(rng.normal(size=500))
    paseo = ruido.cumsum()
    tabla = st.pruebas_raiz_unitaria({"ruido": ruido, "paseo": paseo})
    assert tabla.loc["ruido", "conclusion"] == "estacionaria"
    assert tabla.loc["paseo", "conclusion"] == "raíz unitaria"


def test_origen_movil_no_usa_informacion_futura():
    serie = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2015-01-01", periods=60, freq="MS"))
    vistos = []

    def ultimo_valor(historia, h):
        vistos.append(historia.index[-1])
        return np.repeat(historia.iloc[-1], h)

    ev = st.evaluar_origen_movil(serie, {"ingenuo": ultimo_valor}, inicio="2018-01-01", horizonte=3)
    assert (ev["fecha"] > ev["origen"]).all()
    assert max(vistos) == serie.index[-4]
    # En una tendencia lineal de pendiente uno, el error del ingenuo es igual al horizonte.
    assert (ev["error"] == ev["horizonte"]).all()


def test_resumen_errores_rmse():
    ev = pd.DataFrame({"horizonte": [1, 1, 2, 2], "modelo": ["a"] * 4, "error": [3.0, -4.0, 0.0, 0.0]})
    tabla = st.resumen_errores(ev)
    assert tabla.loc[1, "a"] == pytest.approx(np.sqrt(12.5))
    assert tabla.loc[2, "a"] == 0.0


def test_diebold_mariano_detecta_modelo_mas_preciso():
    rng = np.random.default_rng(3)
    bueno, malo = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    res = st.prueba_diebold_mariano(bueno, malo)
    assert res["estadistico"] < 0 and res["p_valor"] < 0.01
    parecido = st.prueba_diebold_mariano(bueno, bueno + rng.normal(0, 0.01, 300))
    assert parecido["p_valor"] > 0.05


def test_es_de_t_estandarizada_coincide_con_simulacion():
    nu = 5.0
    q, es = rg._cuantil_y_es_t_estandarizada(nu, 0.99)
    muestra = np.random.default_rng(1).standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    assert q == pytest.approx(-np.quantile(muestra, 0.01), rel=0.01)
    assert es == pytest.approx(-muestra[muestra <= -q].mean(), rel=0.02)


def test_garch_sin_mirar_el_futuro():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.standard_t(5, 700) * 0.01, index=pd.bdate_range("2020-01-01", periods=700))
    base = rg.pronosticar_var_es_garch(r, ventana_minima=500, recalibrar_cada=50)
    alterado = r.copy()
    alterado.iloc[650:] = alterado.iloc[650:] * 10  # cambiar el futuro no debe mover pronósticos previos
    otro = rg.pronosticar_var_es_garch(alterado, ventana_minima=500, recalibrar_cada=50)
    assert base.index[0] == r.index[500]
    assert np.allclose(base.loc[: r.index[650], "var"], otro.loc[: r.index[650], "var"])
    assert (base["var"] > 0).all() and (base["es"] > base["var"]).all()
