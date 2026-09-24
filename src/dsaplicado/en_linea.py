"""Aprendizaje en línea frente a aprendizaje por lotes.

Un modelo por lotes se entrena una vez con todos los datos disponibles. Un
modelo en línea aprende de a una observación a la vez, sin volver a ver las
anteriores, lo que le permite trabajar con flujos continuos de datos y
adaptarse cuando el fenómeno cambia.

La evaluación prequential ("predecir y luego aprender") recorre el flujo en
orden: para cada observación el modelo primero predice, se registra si acertó,
y recién después aprende de ella. Así todas las observaciones sirven para
evaluar y para entrenar, sin fuga de información.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests

URL_PULSAR = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
COLUMNAS_PULSAR = [
    "perfil_media", "perfil_desviacion", "perfil_curtosis", "perfil_asimetria",
    "dmsnr_media", "dmsnr_desviacion", "dmsnr_curtosis", "dmsnr_asimetria", "pulsar",
]


# --- Datos ------------------------------------------------------------------------


def cargar_pulsar(carpeta: str | Path) -> pd.DataFrame:
    """Conjunto HTRU2 de candidatos a púlsar, del repositorio de la UCI, con caché local.

    Ocho variables continuas describen cada candidato: cuatro del perfil de pulso
    integrado y cuatro de la curva DM-SNR. La clase vale 1 para púlsares reales.
    """
    ruta = Path(carpeta) / "htru2.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    resp = requests.get(URL_PULSAR, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archivo:
        crudo = archivo.read("HTRU_2.csv").decode("utf-8")
    # El archivo original usa saltos de línea antiguos de Mac.
    tabla = pd.read_csv(io.StringIO(crudo.replace("\r\n", "\n").replace("\r", "\n")), header=None,
                        names=COLUMNAS_PULSAR)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla


def a_flujo(X: pd.DataFrame, y: pd.Series) -> Iterable[tuple[dict, object]]:
    """Recorre una tabla como un flujo de pares (observación como diccionario, etiqueta).

    Los valores faltantes se entregan como None, que es lo que reconoce el
    imputador de river. Un NaN se propagaría por todo el modelo.
    """
    columnas = list(X.columns)
    for fila, etiqueta in zip(X.itertuples(index=False, name=None), y):
        yield {c: (None if pd.isna(v) else v) for c, v in zip(columnas, fila)}, etiqueta


def imputador_en_linea(columnas: Iterable[str]):
    """Imputador de river que reemplaza cada faltante por la media acumulada de su variable."""
    from river import preprocessing, stats

    return preprocessing.StatImputer(*[(c, stats.Mean()) for c in columnas])


# --- Evaluación prequential ---------------------------------------------------------


def evaluar_prequential(modelo, flujo: Iterable[tuple[dict, object]], metrica, cada: int = 500,
                        evaluar_desde: int = 0) -> dict:
    """Evalúa un modelo de river prediciendo cada observación antes de aprender de ella.

    Con ``evaluar_desde`` mayor que cero, el modelo aprende de las primeras
    observaciones sin que cuenten en la métrica. Sirve para compararlo con un
    modelo por lotes entrenado con esas mismas observaciones.

    Devuelve la métrica final, la curva de la métrica cada ``cada`` observaciones,
    el tiempo total y el número de observaciones.
    """
    curva = []
    inicio = time.perf_counter()
    n = 0
    for x, y in flujo:
        prediccion = modelo.predict_one(x)
        if prediccion is not None and n >= evaluar_desde:
            metrica.update(y, prediccion)
        modelo.learn_one(x, y)
        n += 1
        if n % cada == 0 and n > evaluar_desde:
            curva.append((n, metrica.get()))
    segundos = time.perf_counter() - inicio
    if not curva or curva[-1][0] != n:
        curva.append((n, metrica.get()))
    return {"metrica": metrica.get(), "curva": pd.DataFrame(curva, columns=["observaciones", "metrica"]),
            "segundos": segundos, "observaciones": n}


def evaluar_lotes_congelado(modelo, X: pd.DataFrame, y: pd.Series, n_entrenamiento: int, metrica,
                            cada: int = 500) -> dict:
    """Entrena un modelo de scikit-learn con las primeras observaciones y lo evalúa sin reentrenar.

    Representa la práctica habitual de entrenar una vez y poner el modelo en
    producción. La curva mide la métrica acumulada sobre el resto del flujo,
    comparable con la evaluación prequential de un modelo en línea.
    """
    inicio = time.perf_counter()
    modelo.fit(X.iloc[:n_entrenamiento], y.iloc[:n_entrenamiento])
    predicciones = modelo.predict(X.iloc[n_entrenamiento:])
    curva = []
    for k, (real, pred) in enumerate(zip(y.iloc[n_entrenamiento:], predicciones), start=1):
        metrica.update(real, pred)
        if k % cada == 0:
            curva.append((n_entrenamiento + k, metrica.get()))
    segundos = time.perf_counter() - inicio
    return {"metrica": metrica.get(), "curva": pd.DataFrame(curva, columns=["observaciones", "metrica"]),
            "segundos": segundos, "observaciones": len(y)}


# --- Complejidad computacional ------------------------------------------------------


def medir_tiempos(
    entrenar: Callable[[np.ndarray, np.ndarray], None],
    generar: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    filas: Iterable[int],
    columnas: Iterable[int],
    repeticiones: int = 3,
) -> pd.DataFrame:
    """Mide el tiempo de entrenamiento para cada combinación de filas y columnas.

    Versión actualizada del evaluador de complejidad de la tesis. Usa la mediana
    de varias repeticiones para reducir el ruido del sistema operativo.
    """
    filas_resultado = []
    for n in filas:
        for p in columnas:
            X, y = generar(n, p)
            tiempos = []
            for _ in range(repeticiones):
                inicio = time.perf_counter()
                entrenar(X, y)
                tiempos.append(time.perf_counter() - inicio)
            filas_resultado.append({"filas": n, "columnas": p, "segundos": float(np.median(tiempos))})
    return pd.DataFrame(filas_resultado)


def exponentes_complejidad(tiempos: pd.DataFrame) -> dict[str, float]:
    """Estima a y b en tiempo = c · filas^a · columnas^b con una regresión en logaritmos.

    Un exponente de 1 indica crecimiento lineal; de 2, cuadrático.
    """
    datos = tiempos[tiempos["segundos"] > 0]
    X = np.column_stack([np.ones(len(datos)), np.log(datos["filas"]), np.log(datos["columnas"])])
    coef, *_ = np.linalg.lstsq(X, np.log(datos["segundos"]), rcond=None)
    ajuste = X @ coef
    residuo = np.log(datos["segundos"]) - ajuste
    total = np.log(datos["segundos"]) - np.log(datos["segundos"]).mean()
    return {"exponente_filas": float(coef[1]), "exponente_columnas": float(coef[2]),
            "r2": float(1 - (residuo**2).sum() / (total**2).sum())}


# --- ALeRCE ---------------------------------------------------------------------------

URL_ALERCE = "https://zenodo.org/api/records/4279623/files/files_for_lc_classifier_SanchezSaez2020.tar.gz/content"
ARCHIVOS_ALERCE = ("labeled_set_lc_classifier_SanchezSaez_2020.csv", "features_for_lc_classifier_20200609.csv")

# Jerarquía del clasificador de curvas de luz de ALeRCE (Sánchez-Sáez y otros, 2021).
JERARQUIA_ALERCE = {
    **{c: "Transitorio" for c in ("SNIa", "SNIbc", "SNII", "SLSN")},
    **{c: "Estocástico" for c in ("QSO", "AGN", "Blazar", "YSO", "CV/Nova")},
    **{c: "Periódico" for c in ("LPV", "E", "DSCT", "RRL", "CEP", "Periodic-Other")},
}


def descargar_alerce(destino: str | Path) -> None:
    """Descarga el archivo de Zenodo y extrae solo el conjunto etiquetado y las características.

    El archivo comprimido pesa 1,1 GB. Se procesa como flujo, sin guardarlo completo.
    """
    import tarfile

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    with requests.get(URL_ALERCE, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for miembro in tar:
                nombre = Path(miembro.name).name
                if miembro.isfile() and nombre in ARCHIVOS_ALERCE:
                    with tar.extractfile(miembro) as origen, open(destino / nombre, "wb") as salida:
                        while bloque := origen.read(1 << 20):
                            salida.write(bloque)


def preparar_alerce(origen: str | Path, n_variables: int = 40, semilla: int = 0) -> pd.DataFrame:
    """Une etiquetas y características y conserva las variables más informativas.

    Las variables se eligen por la importancia de un bosque aleatorio entrenado
    solo con una mitad de los objetos, para no usar información de la otra mitad.
    Los valores se redondean a seis cifras significativas para reducir el tamaño.
    """
    from sklearn.ensemble import RandomForestClassifier

    origen = Path(origen)
    etiquetas = pd.read_csv(origen / ARCHIVOS_ALERCE[0], usecols=["oid", "classALeRCE"])
    ids = set(etiquetas["oid"])
    partes = [bloque[bloque["oid"].isin(ids)]
              for bloque in pd.read_csv(origen / ARCHIVOS_ALERCE[1], chunksize=100_000)]
    tabla = etiquetas.merge(pd.concat(partes), on="oid", how="inner").copy()
    tabla["clase_superior"] = tabla["classALeRCE"].map(JERARQUIA_ALERCE)

    variables = [c for c in tabla.columns if c not in ("oid", "classALeRCE", "clase_superior")]
    entrenamiento = tabla.sample(frac=0.5, random_state=semilla)
    bosque = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=semilla)
    bosque.fit(entrenamiento[variables].fillna(-999), entrenamiento["classALeRCE"])
    importancia = pd.Series(bosque.feature_importances_, index=variables).sort_values(ascending=False)
    elegidas = list(importancia.index[:n_variables])

    compacta = tabla[["oid", "classALeRCE", "clase_superior"] + elegidas].copy()
    compacta[elegidas] = compacta[elegidas].apply(lambda s: s.map(lambda v: float(f"{v:.6g}") if pd.notna(v) else v))
    return compacta


def cargar_alerce(carpeta: str | Path, origen: str | Path | None = None) -> pd.DataFrame:
    """Conjunto etiquetado de ALeRCE en versión compacta, con caché comprimida.

    Si la caché no existe, descarga los datos de Zenodo (1,1 GB, varios minutos)
    en ``origen`` o en una carpeta temporal, y los prepara.
    """
    ruta = Path(carpeta) / "alerce_etiquetados.csv.gz"
    if ruta.exists():
        return pd.read_csv(ruta)
    import tempfile

    origen = Path(origen or tempfile.mkdtemp())
    if not all((origen / a).exists() for a in ARCHIVOS_ALERCE):
        descargar_alerce(origen)
    tabla = preparar_alerce(origen)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False, compression="gzip")
    return tabla
