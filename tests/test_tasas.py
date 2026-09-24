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
