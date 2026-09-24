# Análisis de datos

## 01. Segmentación de los bancos chilenos

[Abrir el notebook](01_segmentacion_bancos.ipynb)

**Pregunta.** ¿Qué grupos de bancos existen en Chile según su solvencia, rentabilidad, riesgo de crédito, eficiencia y modelo de negocio?

**Datos.** Indicadores mensuales por banco de la API BEST de la Comisión para el Mercado Financiero (CMF): solvencia de Basilea III, rentabilidad sobre patrimonio, cartera deteriorada, morosidad, provisiones, gastos operacionales y composición de la cartera de crédito. El corte transversal usa el promedio entre agosto de 2025 y julio de 2026, y la evolución histórica parte en 2019.

**Método.**

- Una variable por dimensión económica, con logaritmos para las variables muy asimétricas.
- Clustering k-means y jerárquico de Ward, con el número de grupos elegido por el coeficiente de silueta.
- Análisis de componentes principales para ubicar a los bancos en un mapa y entender qué separa a los grupos.
- Comparación entre métodos con el índice de Rand ajustado.

**Conclusiones.**

- El sistema tiene tres grupos: banca universal, banca de consumo y banca mayorista extranjera.
- La banca universal concentra casi todo el crédito del sistema.
- Todos los bancos superan con holgura el mínimo legal de capital de 8% de los activos ponderados por riesgo.
- Falabella es un banco híbrido: según el método queda en la banca de consumo o en la universal.

![Mapa de los bancos](figuras/mapa_bancos.png)

## Datos de la CMF

Los cuadros descargados quedan en caché en la carpeta de datos, así que el notebook se ejecuta sin conexión y sin clave. Para actualizar los datos se necesita una clave gratuita de la API BEST, que se solicita en best.cmfchile.cl. La clave se guarda en un archivo `.env` en la raíz del repositorio, con la línea `CMF_API_KEY=tu_clave`. Ese archivo está excluido de git.

La API permite 10 consultas por minuto y 100 por día. El cliente del paquete respeta ese ritmo y nunca muestra la clave en sus mensajes.
