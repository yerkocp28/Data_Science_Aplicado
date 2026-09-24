from datetime import date

import pytest

from dsaplicado import cmf


def test_leer_clave_desde_archivo_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CMF_API_KEY", raising=False)
    (tmp_path / ".env").write_text('CMF_API_KEY="abc-123"\n', encoding="utf-8")
    assert cmf.leer_clave(tmp_path) == "abc-123"


def test_variable_de_entorno_tiene_prioridad(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CMF_API_KEY=archivo\n", encoding="utf-8")
    monkeypatch.setenv("CMF_API_KEY", "entorno")
    assert cmf.leer_clave(tmp_path) == "entorno"


def test_sin_clave_lanza_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CMF_API_KEY", raising=False)
    with pytest.raises(cmf.ErrorCMF, match="No se encontró"):
        cmf.leer_clave(tmp_path)


def test_rangos_anuales_no_superan_doce_meses():
    tramos = cmf.rangos_anuales(date(2019, 8, 1), date(2022, 3, 31))
    assert tramos[0] == (date(2019, 8, 1), date(2020, 7, 31))
    assert tramos[-1] == (date(2021, 8, 1), date(2022, 3, 31))
    for desde, hasta in tramos:
        assert (hasta.year - desde.year) * 12 + hasta.month - desde.month < 12
    # Los tramos son contiguos y cubren todo el período.
    for (_, fin_a), (inicio_b, _) in zip(tramos, tramos[1:]):
        assert (inicio_b - fin_a).days == 1


def test_parsear_cuadro_elimina_series_repetidas():
    respuesta = {
        "cuadroInfo": {"tag": "X"},
        "series": [
            {"serieInfo": {"cod_serie": "S_BICE", "descripcion_corta": "Banco Bice"},
             "valores": [{"fecha": 20260101, "valor": 15.1}, {"fecha": 20260201, "valor": 15.2}]},
            {"serieInfo": {"cod_serie": "S_BICE", "descripcion_corta": "Banco Bice"},
             "valores": [{"fecha": 20260101, "valor": 15.1}, {"fecha": 20260201, "valor": 15.2}]},
            {"serieInfo": {"cod_serie": "S_BCI", "descripcion_corta": "BCI"},
             "valores": [{"fecha": 20260101, "valor": 13.0}]},
        ],
    }
    tabla = cmf.parsear_cuadro(respuesta)
    assert len(tabla) == 3
    assert set(tabla["nombre"]) == {"Banco Bice", "BCI"}
    assert str(tabla["fecha"].iloc[0].date()) == "2026-01-01"


def test_error_http_no_expone_la_clave(monkeypatch):
    class RespuestaFalsa:
        status_code = 401

    monkeypatch.setattr(cmf.requests, "get", lambda *a, **k: RespuestaFalsa())
    cliente = cmf.ClienteCMF(clave="secreta-123", pausa=0)
    with pytest.raises(cmf.ErrorCMF) as error:
        cliente.uso()
    assert "secreta-123" not in str(error.value)
    assert "clave no es válida" in str(error.value)


@pytest.mark.parametrize("serie,codigo", [
    ("CMF_BCOS_IND_BASILEA3_PATEFE_APR_AGIFI_BICE_STO_RAZ_PORC_MONT", "BICE"),
    ("CMF_CONT_PROV_BANC_CCO_INCUMP_AGIFI_BSANT_PORC_MONT", "BSANT"),
    ("SBIF_CONT_EPLME_ACTIV_COL_MM$_AGIFI_BITAUCORP_STO_MONT", "BITAUCORP"),
    ("SBIF_CONT_EPLME_ACTIV_COL_MM$_STO_MONT", None),
])
def test_codigo_institucion(serie, codigo):
    assert cmf.codigo_institucion(serie) == codigo


def test_parsear_serie_individual():
    respuesta = {"serieInfo": {"cod_serie": "CMF_DEPCAP_DEP_30H60D_AGIFI_BCHI_PORC", "descripcion_corta": "Banco de Chile"},
                 "valores": [{"fecha": 20260401, "valor": 4.01}, {"fecha": 20260501, "valor": 4.0}]}
    tabla = cmf.parsear_serie(respuesta)
    assert len(tabla) == 2
    assert cmf.codigo_institucion(tabla["codigo_serie"].iloc[0]) == "BCHI"


def test_cuadro_vacio_no_se_guarda_en_cache(tmp_path, monkeypatch):
    class ClienteVacio:
        def cuadro(self, *args):
            return cmf.parsear_cuadro({"series": []})

    with pytest.raises(cmf.ErrorCMF, match="no devolvió datos"):
        cmf.cargar_cuadro("X", date(2025, 1, 1), date(2025, 12, 31), tmp_path, ClienteVacio())
    assert not list(tmp_path.iterdir())


def test_cargar_series_omite_las_que_fallan(tmp_path):
    class ClienteParcial:
        def serie(self, codigo, inicio, fin):
            if "MALA" in codigo:
                raise cmf.ErrorCMF("sin datos")
            return cmf.parsear_serie({"serieInfo": {"cod_serie": codigo, "descripcion_corta": codigo},
                                      "valores": [{"fecha": 20260101, "valor": 1.0}]})

    archivo = tmp_path / "series.csv"
    with pytest.warns(UserWarning, match="MALA"):
        tabla = cmf.cargar_series(["S_BUENA", "S_MALA"], date(2026, 1, 1), date(2026, 1, 31), archivo, ClienteParcial())
    assert list(tabla["codigo_serie"]) == ["S_BUENA"]
    assert archivo.exists()
