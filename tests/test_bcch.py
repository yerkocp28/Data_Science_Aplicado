from datetime import date

import pandas as pd
import pytest

from dsaplicado import bcch


def test_leer_token_desde_archivo_del_usuario(tmp_path, monkeypatch):
    monkeypatch.delenv("BCCH_TOKEN", raising=False)
    (tmp_path / "api_banco_central.txt").write_text("token = abc123\n", encoding="utf-8")
    assert bcch.leer_token(tmp_path) == "abc123"


def test_env_tiene_prioridad_sobre_archivo(tmp_path, monkeypatch):
    monkeypatch.delenv("BCCH_TOKEN", raising=False)
    (tmp_path / ".env").write_text("CMF_API_KEY=x\nBCCH_TOKEN=desde_env\n", encoding="utf-8")
    (tmp_path / "api_banco_central.txt").write_text("token=desde_txt\n", encoding="utf-8")
    assert bcch.leer_token(tmp_path) == "desde_env"


def test_sin_token_lanza_error(tmp_path, monkeypatch):
    monkeypatch.delenv("BCCH_TOKEN", raising=False)
    with pytest.raises(bcch.ErrorBCCh):
        bcch.leer_token(tmp_path)


def test_parsear_serie_descarta_dias_sin_dato():
    respuesta = {"Codigo": 0, "Descripcion": "Success", "Series": {
        "seriesId": "F022.SPC.TIN.AN02.NO.Z.D",
        "Obs": [{"indexDateString": "10-09-2026", "value": "5.03", "statusCode": "OK"},
                {"indexDateString": "12-09-2026", "value": "NaN", "statusCode": "ND"},
                {"indexDateString": "11-09-2026", "value": "5.02", "statusCode": "OK"}]}}
    serie = bcch.parsear_serie(respuesta)
    assert list(serie.index.strftime("%Y-%m-%d")) == ["2026-09-10", "2026-09-11"]
    assert serie.iloc[0] == pytest.approx(5.03)


def test_parsear_serie_con_error_de_la_api():
    with pytest.raises(bcch.ErrorBCCh, match="-50"):
        bcch.parsear_serie({"Codigo": -50, "Descripcion": "An internal error has occurred"})


def test_cargar_series_omite_las_que_fallan(tmp_path):
    class ClienteFalso:
        def serie(self, codigo, inicio, fin):
            if codigo == "MALA":
                raise bcch.ErrorBCCh("sin datos")
            return pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]), name=codigo)

    archivo = tmp_path / "tasas.csv"
    with pytest.warns(UserWarning, match="MALA"):
        tabla = bcch.cargar_series({"BUENA": "b", "MALA": "m"}, date(2026, 1, 1), date(2026, 1, 2), archivo, ClienteFalso())
    assert list(tabla.columns) == ["b"]
    assert archivo.exists()
