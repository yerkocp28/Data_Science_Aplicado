# Aprendizaje en línea: la tesis, actualizada

Versión reproducible de mi tesis de Ingeniería Estadística en la Universidad de Santiago de Chile, de 2020: [Implementación de un algoritmo de clasificación en línea usando técnicas de Ensemble learning](TESIS_Yerko_Carre%C3%B1o.pdf), con Felipe Elorrieta López como profesor guía.

La tesis implementaba los ensambles en línea de bagging y boosting propuestos por Oza (2005) y los comparaba con sus versiones por lotes, usando datos de estrellas púlsar y de alertas astronómicas del telescopio ZTF procesadas por el broker chileno ALeRCE.

**Pregunta.** ¿Cuándo conviene un modelo que aprende de un flujo continuo de datos, frente a uno entrenado una sola vez con todo?

## Notebooks

| Notebook | Qué hace |
|---|---|
| [01. Datos de ALeRCE](01_datos_alerce.ipynb) | Describe los 123.496 objetos etiquetados, su jerarquía de clases y sus valores faltantes |
| [02. Por lotes frente a en línea](02_lotes_vs_en_linea.ipynb) | Compara nueve modelos en ALeRCE, en púlsares y en un flujo simulado con cambio de concepto |
| [03. Complejidad computacional](03_complejidad_computacional.ipynb) | Mide cómo crece el tiempo de entrenamiento y cuánto cuesta mantener un modelo al día |

La carpeta [legacy](legacy/) conserva los notebooks originales de la tesis y documenta los problemas que se corrigieron.

## Datos

- **ALeRCE.** Conjunto etiquetado del clasificador de curvas de luz, publicado por Sánchez-Sáez y otros (2021, The Astronomical Journal 161, 141) en [Zenodo](https://zenodo.org/records/4279623) con licencia CC BY 4.0. El repositorio guarda una versión compacta con las 40 variables más informativas.
- **Púlsares HTRU2.** Candidatos a púlsar del repositorio de aprendizaje automático de la UCI.
- **Flujos simulados.** Generador Agrawal de la librería river, con un cambio de regla a mitad del flujo.

## Hallazgos

- **Con datos estables gana el aprendizaje por lotes.** En ALeRCE, un bosque aleatorio alcanza un F1 macro de 0,97 frente a 0,90 del mejor modelo en línea.
- **Cuando el problema cambia, gana el aprendizaje en línea.** Tras un cambio de regla, un árbol entrenado por lotes cae a 47% de exactitud, mientras un árbol de Hoeffding adaptativo se recupera hasta 99,8%.
- **Las clases raras son difíciles de aprender en línea.** El bosque por lotes identifica al 91% de los transitorios y el en línea al 59%.
- **El bagging no mejora a Naive Bayes,** ni por lotes ni en línea. El boosting ayuda por lotes, pero empeora en línea con clases desbalanceadas.
- **Mantener un modelo al día es más barato en línea.** Reentrenar un bosque aleatorio cada 1.000 observaciones cuesta unas cien veces más que actualizar un árbol de Hoeffding.

![Cambio de concepto](figuras/cambio_de_concepto.png)

## Qué cambió respecto de la tesis original

| Aspecto | Tesis original | Esta versión |
|---|---|---|
| Librería | scikit-multiflow, hoy discontinuada | river, su sucesora |
| Datos de ALeRCE | 38.702 objetos, filas incompletas eliminadas | 123.496 objetos publicados, faltantes imputados |
| Jerarquía | LPV agrupadas como estocásticas | Jerarquía oficial, con LPV como periódicas |
| Métrica | Exactitud | F1 macro y F1 de la clase rara |
| Evaluación | Separaciones distintas para cada enfoque | Misma segunda mitad del flujo para todos los modelos |
| Reproducibilidad | Rutas locales | Datos en caché y funciones con pruebas en el paquete |
