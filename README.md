# Data Science Aplicado: finanzas cuantitativas y riesgo

Proyectos de análisis de datos, finanzas y series de tiempo con datos públicos de Chile, desarrollados por **Yerko Carreño**, analista de riesgo financiero.

Cada proyecto responde una pregunta concreta, usa datos reproducibles y termina con conclusiones y limitaciones explícitas. Las funciones comunes viven en un paquete de Python con pruebas automáticas.

## Proyectos

| Área | Proyecto | Estado | Qué muestra |
|---|---|---|---|
| Finanzas | [Optimización de portafolio con acciones del IPSA](02_finanzas/01_optimizacion_portafolio.ipynb) | Listo | Markowitz, CAPM y backtest fuera de muestra con costos |
| Finanzas | [Riesgo de mercado: VaR, ES y backtesting](02_finanzas/02_metricas_de_riesgo.ipynb) | Listo | Cuatro métodos de VaR, Kupiec, Christoffersen y semáforo de Basilea |
| Finanzas | Curva de tasas de Chile | Próximo | Nelson-Siegel, duración, convexidad y choques de tasas |
| Series de tiempo | [Descripción y descomposición](03_series_de_tiempo/01_descripcion_y_descomposicion.ipynb) | Listo | Estacionalidad del IPC, raíz unitaria y autocorrelación |
| Series de tiempo | [Modelos univariados](03_series_de_tiempo/02_modelos_univariados.ipynb) | Listo | SARIMA y ETS contra la meta del 3%, con origen móvil |
| Series de tiempo | [Modelos multivariados](03_series_de_tiempo/03_modelos_multivariados.ipynb) | Listo | VAR, causalidad de Granger, impulso-respuesta y pronóstico |
| Series de tiempo | [Volatilidad con GARCH](03_series_de_tiempo/04_volatilidad_garch.ipynb) | Listo | GARCH y GJR con colas t, y VaR condicional |
| Análisis de datos | Sismicidad en Chile | Próximo | Scraping, limpieza y ley de Gutenberg-Richter |
| Análisis de datos | [Segmentación de bancos chilenos](01_analisis_de_datos/01_segmentacion_bancos.ipynb) | Listo | Clustering con datos de la API de la CMF, k-means, Ward y PCA |

## Resultados destacados

**Optimización de portafolio.** Las tres estrategias diversificadas superaron al ETF IPSA entre 2021 y 2026, ya descontados los costos. Sin embargo, con cinco años de datos la ventaja del portafolio de máximo Sharpe no es estadísticamente significativa.

![Backtest de portafolios](02_finanzas/figuras/backtest_portafolios.png)

**Riesgo de mercado.** Los retornos de acciones chilenas tienen colas mucho más pesadas que una normal. El VaR con t de Student es el único de los cuatro métodos que pasa las pruebas de Kupiec y Christoffersen.

![Backtest de VaR](02_finanzas/figuras/backtest_var.png)

**Segmentación de bancos.** Con datos de la API de la CMF, los 17 bancos chilenos se agrupan en banca universal, banca de consumo y banca mayorista extranjera. La banca universal concentra casi todo el crédito, y todos los bancos superan con holgura el mínimo de capital de Basilea III.

![Mapa de los bancos chilenos](01_analisis_de_datos/figuras/mapa_bancos.png)

**Series de tiempo.** Un SARIMA simple pronostica la inflación tan bien como un VAR con cuatro variables, y antes de 2020 la meta del 3% fue el mejor pronóstico. En riesgo, un GARCH con colas t pasa todas las pruebas de backtesting y corrige la subestimación del Expected Shortfall.

![Comparación de pronósticos de inflación](03_series_de_tiempo/figuras/comparacion_pronosticos.png)

## Estructura

```
├── 01_analisis_de_datos/ segmentación de bancos con datos de la CMF
├── 02_finanzas/          notebooks de finanzas y sus figuras
├── 03_series_de_tiempo/  notebooks de series de tiempo, figuras y resultados
├── src/dsaplicado/       paquete compartido
│   ├── datos.py          descarga y caché de Yahoo Finance y mindicador.cl
│   ├── cmf.py            cliente de la API BEST de la CMF, con control de ritmo y caché
│   ├── portafolio.py     optimización, CAPM y backtest con rebalanceo
│   ├── riesgo.py         VaR, ES, EWMA, GARCH y pruebas de backtesting
│   ├── series.py         estacionariedad, origen móvil y prueba de Diebold-Mariano
│   └── graficos.py       estilo gráfico común y paleta apta para daltonismo
├── tests/                pruebas unitarias del paquete
├── datos/                datos públicos en caché para reproducir los resultados
└── archivo/              trabajos anteriores que no se mantienen
```


## Cómo reproducir

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # pruebas del paquete
jupyter lab                      # abrir los notebooks
```

Los notebooks leen los datos desde `datos/`, así que entregan los mismos resultados sin conexión. Para actualizarlos, cada función de carga acepta `actualizar=True`. Actualizar los datos de la CMF requiere además una clave gratuita de su API, guardada en un archivo `.env` que git ignora.

## Fuentes de datos

- **Yahoo Finance.** Precios diarios ajustados por dividendos de acciones del IPSA y del ETF It Now IPSA.
- **mindicador.cl.** IPC, IMACEC, Tasa de Política Monetaria y dólar observado, publicados por el Banco Central de Chile.
- **API BEST de la CMF.** Indicadores de solvencia, rentabilidad, riesgo de crédito y actividad por banco, publicados por la Comisión para el Mercado Financiero.

## Trabajos anteriores

La carpeta [archivo](archivo/) conserva mi tesis sobre clasificación con aprendizaje en línea y un ejercicio de scraping de 2022. Se mantienen como registro y no se actualizan.

## Sobre mí

Analista de riesgo financiero, ingeniero en estadística y magíster en Business Analytics. Candidato al FRM Part I en noviembre de 2026.

Mi foco profesional es riesgo de mercado, ALM e IRRBB, riesgo de liquidez, VaR y Expected Shortfall, pruebas de estrés, validación de modelos y regulación bancaria de Basilea III.

**Herramientas:** Python, R, SQL, estadística, series de tiempo, machine learning, Power BI y Tableau.

Estoy desarrollando proyectos de riesgo más extensos en repositorios independientes:

1. Validación de modelos de VaR y Expected Shortfall
2. Motor de riesgo IRRBB y ALM
3. Riesgo de contraparte y colaterales
4. Riesgo de crédito bajo IFRS 9
5. Laboratorio integrado de riesgo bancario

## Licencia

Código bajo licencia MIT. Los datos pertenecen a sus respectivas fuentes.
