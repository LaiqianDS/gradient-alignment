# Protocolo de evaluación (train/val/test) y plan de análisis

**Estado: pendiente** — propuesta cerrada en sesión de trabajo (2026-06-10) y revisada el 2026-06-11 (diagnóstico matizado, alternativas A–D con pros/contras, lecturas robustas de curva añadidas a la propuesta). Enviada al tutor para consulta antes de implementar. Cuando se confirme, baja al log de [[2 - Decisiones]] y se actualiza [[1 - Diseño]].

## Contexto

El pipeline actual (`src/data.py`, `src/train.py`) usa solo dos conjuntos: se entrena con todo el train y el **test se evalúa cada época**, alimentando tanto la monitorización como todos los indicadores de eficiencia (`final_test_acc`, `best_test_acc`, `test_loss_auc`, `epochs_to_threshold`). El mismo dato hace así doble papel: monitoriza *y* certifica.

**Matiz importante (revisión 2026-06-11):** en este pipeline no se toma ninguna decisión mirando el test — presupuesto de épocas fijo, sin early stopping, sin selección de checkpoint, rejilla de LR preespecificada. El test se *observa*, no se *consulta*; el mismo espíritu que Jiang et al. 2020, cuyo criterio de parada depende solo del train loss. Los problemas reales son tres, más concretos que un "agujero metodológico":

1. **Sesgo de extremo.** `best_test_acc` (máximo) y el mejor loss (mínimo) son extremos de una serie ruidosa, sesgados en proporción a la volatilidad de la curva. La volatilidad depende del LR, y las métricas de ruido de gradiente (GNS, NGV) plausiblemente la predicen → confound potencial entre predictor y VD **fabricado por el estimador**. Cambiar test por val no lo corrige: es propiedad del máximo, no del conjunto donde se mide. Afecta también al cruce de umbral (VD1) cerca del umbral.
2. **Circularidad de calibración.** Los umbrales (`DATASET_BUDGET`) salen de curvas de test del pilot, y `epochs_to_threshold` se mide después sobre ese mismo test.
3. **Flanco en la defensa.** "Evaluasteis test cada época" es pega previsible de tribunal, aunque ninguna decisión esté contaminada.

El precedente externo no desempata: DAWNBench —el estándar de *time-to-accuracy*— lee el cruce de umbral directamente sobre la accuracy de test reportada cada época, así que las 2 vías son defendibles por práctica común y las 3 vías por protocolo clásico. Lo que ningún precedente cercano corrige es (1); eso lo cubre la adición de lecturas robustas de esta propuesta.

Ningún dataset trae un conjunto de validación propio: MNIST/CIFAR solo distribuyen train+test, y en Tiny-ImageNet el `val/` público ya hace de test (las etiquetas del test oficial no son públicas). Un conjunto de validación solo puede salir del train.

## Propuesta

### Dentro de cada run: split train/val/test con lecturas robustas

- **Val = 10% del train**, estratificado por clase, con **semilla de split fija e independiente de la semilla del run**: todos los runs ven exactamente la misma partición; la única aleatoriedad entre seeds sigue siendo inicialización y orden de batches (que es lo que se estudia). Ej.: CIFAR-10 queda 45k train / 5k val / 10k test.
- **Roles únicos, sin cruces.** El modelo entrena solo con el train recortado; la **probe** de métricas de gradiente se muestrea solo del train recortado (representa los datos que el optimizador desciende); la monitorización por época y todos los indicadores de eficiencia (mejor época, AUC de la curva, épocas-hasta-umbral) se calculan sobre **val**; el **test se evalúa una única vez al final** del run → `final_test_acc` + `final_test_f1_macro`, la variable objetivo limpia.
- **Lecturas robustas de la curva** *(añadido 2026-06-11)*: épocas-hasta-umbral (VD1) y mejor loss de monitorización (VD3) se leen sobre la curva de val suavizada con **mediana móvil centrada de 3 épocas** (lectura primaria); la curva cruda se reporta como análisis de sensibilidad. La AUC (VD2) no cambia: integrar ya es robusto. Esto ataca el problema (1) del contexto, que el split por sí solo no corrige — y que se agravaría al pasar de test (10k) a val (5k en CIFAR), una curva más ruidosa.
- **Figura de sanidad preespecificada** *(añadido 2026-06-11)*: scatter `final_val_acc` vs `final_test_acc` sobre los ~960 runs. Gratis (ambos números ya están en el summary) y recupera el diagnóstico de concordancia que se pierde al dejar de evaluar test por época.
- **Regla que lo resume:** train optimiza, **val monitoriza y calibra, test certifica** — exactamente una vez por run. (Reformulada: la versión anterior "val decide" atribuía a val decisiones que no toma — nada decide durante el entrenamiento — y un tribunal podría desmontarla.)

### Entre runs: análisis agregado pre-especificado

- **Sin selección previa a la afirmación.** Para cada métrica de gradiente se contrasta H₀ ("no hay asociación con la eficiencia") sobre los ~960 runs completos, reportando **todas** las métricas con sus estadísticos — no solo la ganadora. Coherente con la decisión "Se mide todo, siempre".
- **Corrección por comparaciones múltiples: FDR (Benjamini–Hochberg)**, no Bonferroni — las métricas están correlacionadas entre sí y Bonferroni sería brutalmente conservador; FDR es el trade-off correcto cuando el objetivo es un mapa de qué métricas señalan algo. El detalle —estructura de familias (por predictor, para caer en independencia entre celdas), supuesto de validez de BH bajo dependencia y cota de sensibilidad Benjamini–Yekutieli— se especifica en [[Plan de análisis congelado]].
- **Plan de análisis congelado antes de mirar resultados**, registrado en docs/: estadístico (Spearman/Pearson), ventanas primarias, nivel de agregación, criterio de exclusión de runs divergentes — ver [[Plan de análisis congelado]]. Lo que se desvíe del plan se reporta igualmente, etiquetado como exploratorio. Esto cubre el riesgo que la corrección múltiple no cubre: las bifurcaciones de análisis post-hoc (jardín de senderos, Gelman & Loken).

## Alternativas consideradas y descartadas

- **A — Mantener train/test (2 vías) + plan congelado.** Coste cero (sin recalibrar umbrales, sin relanzar pilot, sin perder 10% de train), coincide con la práctica DAWNBench/MLPerf y las decisiones de entrenamiento ya dependen solo del train. Pero deja abiertos los problemas (2) y (3), no hay variable certificada para el constructo de rendimiento final, y el riesgo es **asimétrico**: si tutor o tribunal exigen el split después de lanzar la matriz, rehacer ~960 runs es carísimo; adoptarlo ahora cuesta un relanzamiento de pilot. Descartada por ese riesgo asimétrico.
- **B — Split de 3 vías con lecturas crudas (la versión 2026-06-10 de esta propuesta).** Resuelve (2) y (3) pero no toca (1): el sesgo de extremo se muda a val, más pequeño y ruidoso, restando potencia justo a VD1 donde la nota de potencia del [[Plan de análisis congelado]] ya va corta. Refinada en la propuesta actual (split + lecturas robustas) en lugar de descartada.
- **D — 3 vías de libro: checkpoint de mejor val certificado en test.** Haría literal el "val decide, test certifica" y VD4 pasaría a "rendimiento alcanzable con early stopping". Pero añade lógica de checkpointing a 960 runs, cambia el significado de VD4 (deja de ser "qué obtienes con presupuesto fijo", que es el marco de la tesis) y no aporta nada a las hipótesis de velocidad. Sobrecomplicación sin ganancia.
- **Holdout de semillas (descubrir en seeds 0–2, confirmar en 3–4).** Propuesto inicialmente como defensa contra la selección post-hoc de métricas, pero retirado: con análisis agregado + FDR no hay paso de selección previo a la afirmación, y partir las muestras restaría potencia a los tests de hipótesis justo donde se necesita. Queda como **opción de cierre** solo si la memoria acaba destacando "la métrica X es *la* recomendada" — ese highlight sí sería una selección, y confirmarlo en seeds no miradas sería un broche elegante.

## Implicaciones prácticas

- **Cambios de código** (contenidos): `build_dataloaders` pasa a devolver 3 loaders con split estratificado fijo; `train.py` evalúa val por época (`val_loss`, `val_acc`) y test una sola vez al final (añadiendo F1-macro al summary); `efficiency_summary` y `threshold_acc` leen columnas de val, con VD1/VD3 sobre la curva suavizada (~5 líneas extra; la curva cruda completa sigue en `trajectory.parquet`, así que todo es recomputable post-hoc).
- **Recalibrar umbrales en el pilot.** Los `DATASET_BUDGET` (0.97/0.75/0.35/0.25) se calibraron pensando en test-acc; pasan a chequearse sobre la curva de **val suavizada** (la que define VD1) con un train un 10% menor. El pilot ya previsto es el sitio para verificar que siguen alcanzables-pero-no-triviales — este cambio entra **antes** de relanzar el pilot, no después.
- **Plan de análisis congelado:** definir allí VD1/VD3 sobre mediana-3 con crudo como sensibilidad, y añadir la figura val↔test a §Agregación.
- **Notas de honestidad para la memoria:** (1) el "test" de Tiny-ImageNet es su val público — práctica estándar, pero se declara; (2) en datasets balanceados F1-macro ≈ accuracy, así que F1 se reporta como verificación de robustez, no como hallazgo independiente; (3) el split de val es fijo y compartido por todos los runs — deliberado, porque el objeto de estudio es la variación por seed/LR y no la varianza del estimador (Bouthillier et al. 2021 recomiendan aleatorizar splits cuando lo que se compara son métodos; no es el caso aquí), y se declara.

## Referencias (verificar contra PDF al congelar)

- **DAWNBench** — el cruce de umbral se lee sobre test-acc por época: [reglas de submission](https://github.com/stanford-futuredata/dawn-bench-entries), [análisis v1](https://dawn.cs.stanford.edu/news/analysis-dawnbench-v1-time-accuracy-benchmark-deep-learning). *Verificada 2026-06-11.*
- **Jiang et al. 2020** (Fantastic Generalization Measures, ICLR) — criterio de parada solo sobre train loss; precedente del estudio correlacional a gran escala. *De memoria, en cola de lectura.*
- **Bouthillier et al. 2021** (Accounting for Variance in ML Benchmarks, MLSys) — fuentes de varianza y splits; base de la nota de honestidad (3). *De memoria, en cola de lectura.*

## Siguientes pasos (al confirmar el tutor)

1. Implementar el split + lecturas suavizadas en `src/data.py` + `src/train.py` → verificar con los tests existentes y un run corto de MNIST.
2. Registrar la decisión en [[2 - Decisiones]] y actualizar [[1 - Diseño]] (setup de entrenamiento + VD).
3. Escribir el plan de análisis congelado en docs/ (estadístico, ventanas, agregación, exclusiones, FDR, definiciones suavizadas de VD1/VD3, figura val↔test).
4. Relanzar el pilot de calibración con el split nuevo y revisar presupuestos/umbrales.
