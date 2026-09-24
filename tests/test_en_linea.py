import numpy as np
import pandas as pd
import pytest
from river import metrics, naive_bayes
from sklearn.naive_bayes import GaussianNB

from dsaplicado import en_linea


class ModeloEspia:
    """Modelo de prueba que registra el orden de las llamadas."""

    def __init__(self):
        self.llamadas = []
        self.visto = set()

    def predict_one(self, x):
        self.llamadas.append(("predict", x["id"]))
        return "a" if x["id"] in self.visto else None

    def learn_one(self, x, y):
        self.llamadas.append(("learn", x["id"]))
        self.visto.add(x["id"])


def test_prequential_predice_antes_de_aprender():
    flujo = [({"id": i}, "a") for i in range(4)]
    modelo = ModeloEspia()
    en_linea.evaluar_prequential(modelo, flujo, metrics.Accuracy(), cada=2)
    for k in range(4):
        assert modelo.llamadas[2 * k] == ("predict", k)
        assert modelo.llamadas[2 * k + 1] == ("learn", k)


def test_prequential_aprende_un_problema_separable():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": np.r_[rng.normal(-3, 1, 1000), rng.normal(3, 1, 1000)]})
    y = pd.Series([0] * 1000 + [1] * 1000)
    orden = rng.permutation(2000)
    res = en_linea.evaluar_prequential(naive_bayes.GaussianNB(), en_linea.a_flujo(X.iloc[orden], y.iloc[orden]),
                                       metrics.Accuracy(), cada=500)
    assert res["metrica"] > 0.97
    assert list(res["curva"]["observaciones"]) == [500, 1000, 1500, 2000]


def test_lotes_congelado_evalua_solo_despues_del_entrenamiento():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": np.r_[rng.normal(-3, 1, 500), rng.normal(3, 1, 500)]})
    y = pd.Series([0] * 500 + [1] * 500)
    orden = rng.permutation(1000)
    res = en_linea.evaluar_lotes_congelado(GaussianNB(), X.iloc[orden], y.iloc[orden], 300, metrics.Accuracy(), cada=100)
    assert res["curva"]["observaciones"].iloc[0] == 400
    assert res["metrica"] > 0.97


def test_exponentes_de_complejidad():
    filas = np.array([1000, 2000, 4000, 8000] * 3)
    columnas = np.repeat([5, 10, 20], 4)
    tiempos = pd.DataFrame({"filas": filas, "columnas": columnas, "segundos": 1e-6 * filas**1.5 * columnas**0.5})
    res = en_linea.exponentes_complejidad(tiempos)
    assert res["exponente_filas"] == pytest.approx(1.5)
    assert res["exponente_columnas"] == pytest.approx(0.5)
    assert res["r2"] == pytest.approx(1.0)


def test_medir_tiempos_recorre_la_grilla():
    llamadas = []
    res = en_linea.medir_tiempos(lambda X, y: llamadas.append(X.shape),
                                 lambda n, p: (np.zeros((n, p)), np.zeros(n)), [10, 20], [2, 3], repeticiones=2)
    assert len(res) == 4 and len(llamadas) == 8
    assert set(zip(res["filas"], res["columnas"])) == {(10, 2), (10, 3), (20, 2), (20, 3)}


def test_prequential_con_periodo_de_entrenamiento_no_cuenta_en_la_metrica():
    # Un modelo que se equivoca siempre en las primeras 100 y acierta después.
    class Modelo:
        n = 0
        def predict_one(self, x):
            return "b" if self.n < 100 else "a"
        def learn_one(self, x, y):
            self.n += 1

    flujo = [({"v": i}, "a") for i in range(300)]
    res = en_linea.evaluar_prequential(Modelo(), flujo, metrics.Accuracy(), cada=100, evaluar_desde=100)
    assert res["metrica"] == 1.0
    assert list(res["curva"]["observaciones"]) == [200, 300]


def test_jerarquia_alerce_cubre_las_15_clases():
    clases = {"E", "RRL", "QSO", "LPV", "AGN", "YSO", "SNIa", "Blazar", "Periodic-Other", "CV/Nova",
              "DSCT", "CEP", "SNII", "SNIbc", "SLSN"}
    assert set(en_linea.JERARQUIA_ALERCE) == clases
    assert en_linea.JERARQUIA_ALERCE["LPV"] == "Periódico"
    assert set(en_linea.JERARQUIA_ALERCE.values()) == {"Transitorio", "Estocástico", "Periódico"}


def test_preparar_alerce_une_y_selecciona_variables(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    clases = rng.choice(["E", "QSO", "SNIa"], n)
    pd.DataFrame({"oid": [f"ZTF{i}" for i in range(n)], "classALeRCE": clases, "ra": 0.0}).to_csv(
        tmp_path / en_linea.ARCHIVOS_ALERCE[0], index=False)
    informativa = np.where(clases == "E", 0.0, np.where(clases == "QSO", 5.0, 10.0)) + rng.normal(0, 0.1, n)
    caracteristicas = pd.DataFrame({"oid": [f"ZTF{i}" for i in range(n)] + ["ZTF_sin_etiqueta"],
                                    "util": np.r_[informativa, 1.0], "ruido": rng.normal(size=n + 1)})
    caracteristicas.to_csv(tmp_path / en_linea.ARCHIVOS_ALERCE[1], index=False)
    tabla = en_linea.preparar_alerce(tmp_path, n_variables=1)
    assert len(tabla) == n
    assert list(tabla.columns) == ["oid", "classALeRCE", "clase_superior", "util"]
    assert set(tabla["clase_superior"]) == {"Periódico", "Estocástico", "Transitorio"}


def test_imputador_en_linea_completa_faltantes_con_la_media_acumulada():
    X = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [10.0, 20.0, 30.0]})
    flujo = list(en_linea.a_flujo(X, pd.Series([0, 1, 0])))
    assert flujo[2][0] == {"a": None, "b": 30.0}
    imputador = en_linea.imputador_en_linea(X.columns)
    for x, _ in flujo[:2]:
        imputador.learn_one(x)
    completado = imputador.transform_one(flujo[2][0])
    assert completado["a"] == 2.0
