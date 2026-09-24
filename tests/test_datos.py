from dsaplicado.datos import parsear_mindicador


def test_parsear_mindicador_ordena_y_convierte_fechas_a_hora_chilena():
    respuesta = {
        "codigo": "tpm",
        "serie": [
            {"fecha": "2024-12-30T03:00:00.000Z", "valor": 5},
            {"fecha": "2024-07-01T04:00:00.000Z", "valor": 5.75},
        ],
    }
    serie = parsear_mindicador(respuesta)
    assert serie.name == "tpm"
    assert [str(f.date()) for f in serie.index] == ["2024-07-01", "2024-12-30"]
    assert serie.iloc[0] == 5.75


def test_parsear_mindicador_vacio():
    assert parsear_mindicador({"codigo": "ipc", "serie": []}).empty


def test_raiz_repositorio_desde_subcarpeta(tmp_path):
    from dsaplicado.datos import raiz_repositorio

    (tmp_path / "pyproject.toml").write_text("")
    sub = tmp_path / "finanzas" / "figuras"
    sub.mkdir(parents=True)
    assert raiz_repositorio(sub) == tmp_path.resolve()


def test_separar_etiquetas_respeta_distancia_minima():
    from dsaplicado.graficos import separar_etiquetas

    res = separar_etiquetas({"a": 1.0, "b": 1.01, "c": 1.02, "d": 5.0}, 0.1)
    valores = sorted(res.values())
    assert all(b - a >= 0.1 - 1e-12 for a, b in zip(valores, valores[1:]))
