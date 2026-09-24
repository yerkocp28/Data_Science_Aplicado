import numpy as np
import pandas as pd
import pytest

from dsaplicado import sismos

HTML_CSN = """
<html><body>
<table class="sismologia detalle">
  <tr><th>Fecha Local / Lugar</th><th>Fecha UTC</th><th>Latitud / Longitud</th><th>Profundidad</th><th>Magnitud</th></tr>
  <tr>
    <td><a href="/sismicidad/informes/2025/06/303976.html">2025-06-15 19:09:15</a><br>68 km al SO de Ollagüe</td>
    <td>2025-06-15 23:09:15</td>
    <td>-21.784<br> -68.527</td>
    <td>122 km</td>
    <td class="magnitud">3.2 Ml</td>
  </tr>
  <tr>
    <td><a href="/sismicidad/informes/2025/06/303967.html">2025-06-15 18:09:34</a><br>98 km al O de San Antonio de los Cobres</td>
    <td>2025-06-15 22:09:34</td>
    <td>-24.394<br> -67.272</td>
    <td>194 km</td>
    <td class="magnitud">4.1 Mww</td>
  </tr>
</table>
</body></html>
"""


def test_parsear_catalogo_diario():
    tabla = sismos.parsear_catalogo_diario(HTML_CSN)
    assert len(tabla) == 2
    primera = tabla.iloc[0]
    assert primera["id"] == "303976"
    assert primera["referencia"] == "68 km al SO de Ollagüe"
    assert primera["latitud"] == pytest.approx(-21.784)
    assert primera["profundidad_km"] == 122
    assert primera["magnitud"] == 3.2 and primera["tipo_magnitud"] == "Ml"
    assert tabla.iloc[1]["tipo_magnitud"] == "Mww"
    assert (tabla["fecha_utc"] - tabla["fecha_local"]).dt.total_seconds().eq(4 * 3600).all()


def test_pagina_sin_tabla_devuelve_tabla_vacia():
    assert sismos.parsear_catalogo_diario("<html><body>Sin datos</body></html>").empty


def test_distancia_haversine_conocida():
    # Santiago a Valparaíso, cerca de 100 km en línea recta.
    assert sismos.distancia_km(-33.45, -70.66, -33.05, -71.62) == pytest.approx(99, abs=5)
    assert sismos.distancia_km(0, 0, 0, 0) == 0


def test_emparejar_catalogos_usa_tiempo_y_distancia():
    csn = pd.DataFrame({"id": ["a", "b"], "fecha_utc": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 12:00:00"]),
                        "latitud": [-30.0, -20.0], "longitud": [-71.0, -69.0], "magnitud": [4.0, 3.0]})
    usgs = pd.DataFrame({"id": ["x", "y", "z"],
                         "fecha_utc": pd.to_datetime(["2025-01-01 10:00:05", "2025-01-01 12:00:10", "2025-01-01 15:00:00"]),
                         "latitud": [-30.1, -25.0, -20.0], "longitud": [-71.1, -69.0, -69.0], "magnitud": [4.2, 3.1, 5.0]})
    pares = sismos.emparejar_catalogos(csn, usgs, segundos=30, km=100)
    # x calza con a; y está a tiempo de b pero a más de 500 km; z no tiene pareja en el tiempo.
    assert list(pares["id"]) == ["x"]
    assert pares["id_csn"].iloc[0] == "a"


def test_gutenberg_richter_recupera_valor_b():
    rng = np.random.default_rng(0)
    b_real, mc = 1.0, 3.0
    # Magnitudes continuas desde mc - 0,05, porque el intervalo de 3,0 cubre de 2,95 a 3,05 al redondear.
    magnitudes = mc - 0.05 + rng.exponential(1 / (b_real * np.log(10)), size=20000)
    redondeadas = pd.Series(np.round(magnitudes, 1))
    ajuste = sismos.ajustar_gutenberg_richter(redondeadas, mc=mc)
    assert ajuste["b"] == pytest.approx(b_real, abs=0.03)
    assert 0 < ajuste["error_b"] < 0.03


def test_magnitud_completitud_por_maxima_curvatura():
    rng = np.random.default_rng(1)
    completos = 3.0 + rng.exponential(0.43, 5000)
    incompletos = rng.uniform(1.5, 3.0, 800)  # bajo 3.0 la red detecta solo una parte
    mc = sismos.magnitud_completitud(pd.Series(np.concatenate([completos, incompletos])))
    assert mc == pytest.approx(3.2, abs=0.15)
