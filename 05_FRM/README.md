# FRM Parte I: casos trabajados

Casos prácticos para preparar el examen FRM Parte I de GARP, organizados por los cuatro libros del programa. Cada caso explica el concepto con su fórmula, lo aplica a datos reales de Chile cuando es posible, interpreta el resultado y resume lo que suele evaluarse en el examen.

## El examen

| Libro | Peso aproximado en el examen | Notebook |
|---|---|---|
| 1. Fundamentos de la gestión de riesgos | 20% | [01_fundamentos_gestion_riesgos](01_fundamentos_gestion_riesgos.ipynb) |
| 2. Análisis cuantitativo | 20% | [02_analisis_cuantitativo](02_analisis_cuantitativo.ipynb) |
| 3. Mercados y productos financieros | 30% | [03_mercados_y_productos](03_mercados_y_productos.ipynb) |
| 4. Valoración y modelos de riesgo | 30% | [04_valoracion_y_modelos_de_riesgo](04_valoracion_y_modelos_de_riesgo.ipynb) |

La Parte I tiene 100 preguntas de selección múltiple en cuatro horas.

## Casos por libro

### Libro 1. Fundamentos de la gestión de riesgos

| Caso | Tema | Qué se aprende |
|---|---|---|
| 1 | CAPM y línea del mercado de valores | Beta, retorno exigido y alfa con acciones del IPSA |
| 2 | Medidas de desempeño | Sharpe, Treynor, alfa de Jensen, índice de información y Sortino, y cuándo usar cada una |
| 3 | Modelos multifactoriales | Sensibilidad de las acciones chilenas al mercado, al dólar y al cobre |
| 4 | La decisión de cobertura | Un exportador que fija su tipo de cambio con forwards entre 2010 y 2026 |
| 5 | Desastres financieros | Apalancamiento de LTCM y lecciones de Metallgesellschaft, Barings, Société Générale y el London Whale |

### Libro 2. Análisis cuantitativo

| Caso | Tema | Qué se aprende |
|---|---|---|
| 1 | Momentos y distribución | Asimetría, curtosis y Jarque-Bera del IPSA y del dólar |
| 2 | Pruebas de hipótesis | Por qué harían falta 35 años de datos para confirmar un Sharpe de 0,33 |
| 3 | Regresión y diagnóstico | Heterocedasticidad, errores robustos de White y Durbin-Watson |
| 4 | Series de tiempo | Raíz unitaria del dólar y regresión espuria |
| 5 | Volatilidad | EWMA frente a GARCH, varianza de largo plazo y vida media de los shocks |
| 6 | Correlación | Correlación EWMA entre el IPSA y el dólar, y un error de alineación de datos |
| 7 | Simulación | Monte Carlo con variables antitéticas y bootstrap del VaR |
| 8 | Aprendizaje automático | Regresión logística para incumplimientos, curva ROC y sobreajuste |

### Libro 3. Mercados y productos financieros

| Caso | Tema | Qué se aprende |
|---|---|---|
| 1 | Tasas de interés | Capitalización, tasas forward de la curva swap chilena y valoración de un FRA |
| 2 | Futuros | Llamados de margen de un exportador durante el estallido social de 2019 |
| 3 | Forwards de divisas | Paridad cubierta de tasas para el dólar y por qué el forward no pronostica |
| 4 | Cobertura con futuros | Razón de mínima varianza, efectividad y ajuste de beta |
| 5 | Opciones | Estrategias sobre el dólar y paridad put-call |
| 6 | Swaps | Valor de mercado de un swap en pesos pactado hace dos años |
| 7 | Commodities | Costo de acarreo, contango y backwardation con el cobre |
| 8 | Hipotecas y MBS | Efecto del prepago en los flujos y la vida promedio |
| 9 | Cámaras de compensación | Compensación bilateral frente a multilateral |

### Libro 4. Valoración y modelos de riesgo

| Caso | Tema | Qué se aprende |
|---|---|---|
| 1 | VaR y Expected Shortfall | Tres métodos, raíz del tiempo y el contraejemplo de subaditividad |
| 2 | Duración y convexidad | DV01, cobertura entre bonos y duraciones parciales ante cambios de pendiente |
| 3 | Árboles binomiales | Convergencia a Black-Scholes y valor del ejercicio anticipado |
| 4 | Black-Scholes y griegas | Opciones sobre el dólar y cobertura delta en 65 trimestres reales |
| 5 | Riesgo de crédito | Pérdida esperada, modelo de Vasicek y matrices de transición |
| 6 | Riesgo operacional | Distribución de pérdidas por simulación y método estándar de Basilea III |
| 7 | Pruebas de estrés | Estallido social, pandemia, ciclo de alzas de la TPM y aranceles de 2025 |

## Fórmulas validadas

Las fórmulas están en el módulo [frm.py](../src/dsaplicado/frm.py) del paquete. Sus pruebas las comparan con ejemplos conocidos de los textos de referencia, como Hull: el precio de 4,76 de una call con Black-Scholes, las griegas del ejemplo de 49 contra 50, la put americana binomial de 4,49, la razón de cobertura de 0,78 del combustible de aviones o la cuota hipotecaria de 1.199,10.

## Otros notebooks del repositorio que cubren el programa

| Tema del programa | Notebook |
|---|---|
| Backtesting de VaR: Kupiec, Christoffersen y semáforo de Basilea | [Riesgo de mercado](../02_finanzas/02_metricas_de_riesgo.ipynb) |
| Teoría moderna de carteras y frontera eficiente | [Optimización de portafolio](../02_finanzas/01_optimizacion_portafolio.ipynb) |
| Curva de tasas, componentes principales y escenarios de IRRBB | [Curva de tasas](../02_finanzas/04_curva_de_tasas.ipynb) |
| GARCH con colas pesadas y VaR condicional | [Volatilidad con GARCH](../03_series_de_tiempo/04_volatilidad_garch.ipynb) |
| Modelos autorregresivos y pronóstico | [Modelos univariados](../03_series_de_tiempo/02_modelos_univariados.ipynb) |

## Advertencias

- **Apoyo, no reemplazo.** Estos casos no reemplazan las lecturas oficiales de GARP. Los nombres de los temas son aproximados y el programa cambia cada año.
- **Supuestos ilustrativos.** Algunos parámetros son supuestos, como los márgenes del futuro, la cartera de créditos o las pérdidas operacionales, y así se indica en cada caso.
- **Aproximaciones de datos.** El dólar observado hace de precio del futuro y el ETF IPSA de futuro sobre el índice, porque no hay datos públicos gratuitos de esos contratos.
