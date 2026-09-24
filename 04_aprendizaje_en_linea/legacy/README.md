# Versión original de la tesis

Notebooks y código de la tesis tal como se escribieron en 2020. Se conservan como registro. El documento de la tesis y la versión actualizada y reproducible están en la carpeta superior.

## Contenido

| Archivo | Qué hacía |
|---|---|
| `datos_alerce.ipynb` | Clasificación de objetos de ALeRCE por lotes y en línea, con Naive Bayes, bagging y boosting |
| `datos_pulsar.ipynb` | La misma comparación con el conjunto de púlsares HTRU2 |
| `modelos_datos_simulados.ipynb` | Bagging y boosting en línea con datos simulados del generador Waveform |
| `evaluador_complejidad.py` | Clase que estima empíricamente cómo crece el tiempo de entrenamiento con filas y columnas |

## Por qué ya no se ejecutan

- **Rutas locales.** Los datos se leían desde carpetas de mi computador de entonces, como la instalación de Anaconda.
- **Librería discontinuada.** scikit-multiflow dejó de mantenerse y no funciona con versiones actuales de Python. Su sucesora es river, que usa la versión actualizada.

## Problemas metodológicos encontrados al revisarla

| Problema | Consecuencia | Corrección en la versión actual |
|---|---|---|
| Se eliminaban todas las filas con algún faltante | Quedaba solo un cuarto de los objetos, y no al azar | Imputación por la media sin mirar datos futuros |
| El identificador del objeto entraba como variable predictora | Un número de catálogo no tiene información física y puede generar resultados espurios | Solo se usan características físicas |
| Las estrellas LPV se agrupaban como estocásticas | No coincide con la jerarquía oficial de ALeRCE, donde son periódicas | Jerarquía oficial de Sánchez-Sáez y otros (2021) |
| Se usaba la exactitud como métrica | Con clases muy desbalanceadas, premia ignorar la clase rara | F1 macro y F1 de la clase de interés |
| Regresión logística sin estandarizar las variables | No convergía y rendía 49% de exactitud | Todos los modelos usan el mismo preprocesamiento |
| Bagging con 1.000 modelos base | Mucho tiempo de cómputo para un modelo estable como Naive Bayes, al que el bagging mejora poco | Se usan 10 modelos base |
