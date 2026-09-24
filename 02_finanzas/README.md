# Finanzas

Dos análisis sobre acciones chilenas del IPSA entre enero de 2019 y agosto de 2026. El primero construye portafolios y el segundo mide su riesgo de cola.

## 01. Optimización de portafolio

[Abrir el notebook](01_optimizacion_portafolio.ipynb)

**Pregunta.** ¿Se puede construir un portafolio de acciones chilenas con mejor relación retorno-riesgo que el IPSA, y esa ventaja se mantiene fuera de muestra?

**Método.**

- Estadísticas de retorno, volatilidad y correlación de nueve acciones del IPSA.
- Regresión CAPM contra el ETF It Now IPSA, con la TPM como tasa libre de riesgo.
- Frontera eficiente sin ventas cortas y con un máximo de 30% por acción.
- Backtest con rebalanceo trimestral, ventana de estimación de dos años y costo de 20 puntos base por unidad transada.

**Conclusiones.**

- Las tres estrategias diversificadas superaron al ETF IPSA en el período fuera de muestra.
- La frontera dentro de muestra exagera los beneficios, porque se calcula conociendo qué acciones subieron.
- Los intervalos de confianza del índice de Sharpe se superponen. Cinco años no bastan para distinguir habilidad de suerte.
- Mínima varianza es la alternativa más defendible, porque no depende de estimar retornos esperados.

![Frontera eficiente](figuras/frontera_eficiente.png)

## 02. Riesgo de mercado: VaR, ES y backtesting

[Abrir el notebook](02_metricas_de_riesgo.ipynb)

**Pregunta.** ¿Qué método de VaR habría medido correctamente el riesgo diario de una cartera de acciones chilenas entre 2020 y 2026?

**Método.**

- VaR al 99% por simulación histórica, normal, t de Student y EWMA de RiskMetrics.
- Pronósticos diarios con ventana móvil, usando solo información anterior a cada día.
- Pruebas de Kupiec para la frecuencia de excepciones y de Christoffersen para su agrupamiento.
- Semáforo de Basilea sobre ventanas de 250 días y revisión de la severidad frente al ES.

**Conclusiones.**

- Los retornos tienen colas mucho más pesadas que una normal.
- La t de Student es el único método que pasa ambas pruebas, a cambio de un VaR promedio mayor.
- El histórico y el normal aciertan en la frecuencia de excepciones, pero estas llegan agrupadas.
- En los días de excepción, la pérdida real supera al ES pronosticado en los cuatro métodos.

![Semáforo de Basilea](figuras/semaforo_basilea.png)

## Supuestos comunes

- **Universo.** BCI, Santander, Banco de Chile, Cencosud, CMPC, Copec, Enel Chile, Falabella y SQM-B. Se excluye LATAM por su quiebra y reestructuración entre 2020 y 2022.
- **Sesgo de supervivencia.** Las acciones se eligieron con información de hoy, lo que favorece a las estrategias frente al índice.
- **Datos.** Los precios quedan en caché en la carpeta de datos, así que los resultados son reproducibles.
