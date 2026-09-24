# Archivo

Trabajos anteriores que se conservan como registro. No se mantienen ni se garantiza que se puedan ejecutar: usan rutas locales de mi equipo de entonces y versiones antiguas de las librerías.

## Tesis: aprendizaje en línea para clasificación

Carpeta [tesis_alerce](tesis_alerce/).

Compara modelos entrenados por lotes con modelos de aprendizaje en línea, que se actualizan a medida que llegan los datos. Se usaron tres conjuntos de datos:

- **ALeRCE.** Características de curvas de luz del broker astronómico chileno ALeRCE.
- **Estrellas púlsar.** Conjunto público HTRU2 de candidatos a púlsar.
- **Datos simulados.** Para controlar el tamaño de la muestra y medir complejidad computacional.

Los modelos incluyen Naive Bayes, árboles de decisión, bagging, AdaBoost y Oza bagging en línea con scikit-multiflow. El archivo `evaluador_complejidad.py` estima empíricamente cómo crece el tiempo de entrenamiento con el número de filas y columnas.

## Scraping de sismología, 2022

Carpeta [web_scraping_sismos_2022](web_scraping_sismos_2022/).

Primer ejercicio de extracción del catálogo de sismos de sismologia.cl con BeautifulSoup. Quedó incompleto. El proyecto completo está en [Sismicidad en Chile](../01_analisis_de_datos/02_sismicidad_chile.ipynb).
