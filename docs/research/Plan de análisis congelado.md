# Plan de análisis estadístico (preregistro)

**Estado: CONGELADO el 2026-08-01.** Todas sus puertas están cerradas (§Qué falta para congelar) y el documento vive ya bajo control de versiones, en `docs/research/`. El commit que lo introduce precede al primer commit de `reports/`, de modo que el propio historial de git certifica que este plan es anterior a los datos que va a analizar.

A partir de aquí rige la **política de enmiendas** (§Qué congela este documento): cualquier cambio se anota con fecha y motivo en §Historial de revisiones, se marca como enmienda y se declara en la memoria; una enmienda posterior a haber mirado resultados degrada a exploratorio el contraste que toca.

Sus dos puertas externas se cerraron antes: el tutor confirmó el protocolo de evaluación el 2026-06-12, y el **pilot de calibración** cerró el 2026-06-17, con su evidencia registrada el 2026-07-17. Ambas constan en [[2 - Decisiones]].

Redactado el 2026-06-10, revisado a fondo el 2026-06-11 y en cuatro pasadas el 2026-08-01; ver §Historial de revisiones.

## Guía rápida (leer esto primero)

Versión corta del documento, en lenguaje llano. Todo lo que sigue a esta sección es el detalle que la sostiene; nada de aquí sustituye a nada de allí.

**Qué se pregunta.** Si mirar los gradientes durante las primeras épocas de un entrenamiento sirve para predecir cómo va a ir el entrenamiento entero. Y sobre todo, si sirve **más** que mirar simplemente la curva de validación, que es gratis.

**Sobre qué datos.** 960 entrenamientos, agrupados en 24 **celdas**. Una celda es una combinación de dataset, arquitectura y optimizador, y contiene 40 runs (8 learning rates × 5 semillas). La celda es la unidad que importa: dentro de una celda los 40 runs no son independientes, porque el learning rate los agrupa de 5 en 5.

**Cómo se analiza, en dos pasos.** Primero, dentro de cada celda, se mide la correlación de Spearman entre cada métrica temprana y cada indicador de eficiencia. Eso da 24 correlaciones por métrica, y son solo **descriptivas**. Después, esas 24 correlaciones se llevan a un segundo test (Wilcoxon) que pregunta si, en conjunto, se apartan de cero. **Ese segundo paso es el que decide.** La idea es la de un metaanálisis: cada celda es un pequeño estudio, y la conclusión sale de combinarlos.

**Por qué en dos pasos.** Porque los runs de una misma celda están correlacionados entre sí y cualquier p-valor calculado ahí sale demasiado optimista. Entre celdas no hay ese problema: no comparten runs.

**Qué se decide y cómo:**

| | Pregunta | Se concluye que sí cuando |
|---|---|---|
| **H1** | ¿Alguna métrica de gradiente predice algo? | al menos una tiene correlación mediana ≥ 0,3 y sobrevive a la corrección múltiple |
| **H2** | ¿Aporta algo que la curva de validación no diera ya? | igual, pero descontando antes el baseline gratuito. **Es la hipótesis decisiva** |
| **H3** | ¿Qué familia de métricas gana? | dentro de cada celda se compara la mediana de las dos familias, y esa diferencia se apara de cero entre celdas |
| **H4** | ¿La señal está ya al 10% del entrenamiento? | test de no-inferioridad: esperar al 50% no mejora en más de 0,1 |
| **H5** | ¿Funciona igual con SGD que con Adam? | los signos coinciden en los 12 pares de celdas por encima del azar |
| **H6** | ¿Los signos son los que la teoría predice? | los signos coinciden con una tabla congelada de antemano, por encima del azar |

**El resultado negativo también cuenta.** Si H1 sale que sí pero H2 sale que no, la conclusión es que las métricas de gradiente son redundantes con la curva de loss, y eso es una contribución válida de la tesis: instrumentar gradientes cuesta el doble de cómputo y no compensa. Precisamente porque ese negativo es un entregable, H2 lleva un **test de equivalencia** además del test normal, de modo que se pueda *afirmar* que no aporta, en vez de solo *no encontrar* que aporte. Sin eso, "no aporta nada" y "no hemos tenido potencia para verlo" serían el mismo resultado en el papel.

**Las tres reglas que evitan hacer trampa sin querer:**

1. Este documento se congela **antes** de mirar ningún resultado, y su commit va antes que el primer commit de datos, para que el propio historial de git lo certifique.
2. Los runs que no llegan al umbral o que divergen **no se tiran**: entran con el peor rango. Tirarlos sesgaría el estudio, porque son justo los extremos del barrido de learning rate.
3. Todo lo que se haga y no esté escrito aquí se reporta igual, pero etiquetado como **exploratorio**.

**Lo que este estudio no puede afirmar.** Todo lo que mide es asociación **dentro de muestra**: no se reserva ningún dato para validar. La consistencia entre las 24 celdas es evidencia de que la relación viaja entre configuraciones, pero no es lo mismo que una validación predictiva con datos retenidos, y así se declara.

**Qué se sabe ya de sus límites.** Con las 24 celdas completas el diseño detecta sin problema el efecto que declara interesante. Si la matriz no termina, hacen falta al menos 18 celdas completas para que el análisis confirmatorio siga teniendo sentido. H5 solo puede confirmar invariancia casi perfecta. Y H2, por ser una correlación parcial, tiene menos margen que H1: de ahí su brazo de equivalencia. Los números están en §Nota de potencia y son reproducibles.

## Resumen

Este documento fija, antes de mirar ningún resultado, cómo se analizan los ~960 runs. En concreto fija cinco cosas: qué variables dependientes se usan, qué estadístico y a qué nivel, qué reglas de censura y exclusión se aplican, qué corrección por comparaciones múltiples se hace, y qué criterio decide cada hipótesis H1-H6.

La idea central es la **inferencia en dos etapas**. Dentro de cada celda, el ρ de Spearman es solo un estadístico-resumen y se trata como descriptivo, porque los 40 runs de una celda no son independientes: el learning rate los agrupa. La confirmación ocurre un nivel más arriba, entre celdas, donde los runs sí son disjuntos, con un Wilcoxon de rangos con signo.

La corrección múltiple es Benjamini-Hochberg sobre familias preespecificadas, con la cota Benjamini-Yekutieli como blindaje.

Todo lo que el análisis final haga y no esté escrito aquí se reporta igual, pero etiquetado como **exploratorio** (el "jardín de senderos que se bifurcan" de Gelman y Loken).

## Qué congela este documento (y qué no)

**Congela:** estadístico primario y su nivel de inferencia, VD primaria, ventana primaria, reglas de censura y exclusión, familias de corrección múltiple, contraste por hipótesis y nivel de agregación.

**No congela** dos cosas. Primera, los valores numéricos de presupuesto y umbral por dataset: los fija el pilot de calibración, y este plan los referencia de forma paramétrica. Segunda, qué lista de métricas *destaca* la memoria al final: hay una poda posterior con prueba (decisión "Poda de métricas redundantes, con prueba"), y esa poda no cambia qué se contrasta, solo qué se resalta al escribir.

**Política de enmiendas.** Un preregistro sin regla de cambio no es más honesto que no tenerlo: la tentación no es romperlo de golpe, es retocarlo poco a poco sin dejar rastro. Cualquier modificación posterior a la congelación se anota en §Historial de revisiones con fecha y motivo, se marca explícitamente como **enmienda** y no como redacción original, y se declara en la memoria. Una enmienda hecha después de haber mirado resultados de la matriz degrada a **exploratorio** el contraste que toca, sin excepción y aunque el cambio parezca inocuo.

**Congelar el texto no congela la implementación.** Este documento fija qué se contrasta y con qué criterio se decide. No fija el código que lo calcula, y ahí quedan decisiones abiertas que sí afectan a los números.

Dos ejemplos concretos: cómo se manejan los empates en Spearman, y qué le pasa a una correlación parcial cuando recibe un bloque grande de rangos empatados en el peor valor por censura. Lo segundo no es un caso raro, es el caso esperado en los extremos de la rejilla de learning rate.

El mecanismo que cierra ese hueco es el **dry-run del pipeline sobre datos sintéticos con efecto plantado**, no el criterio que a uno le parezca razonable el día que aparezca el problema. Ese dry-run forma parte del plan, no es un extra opcional, y se ejecuta antes de tocar `reports/`.

**Por qué el pilot no contamina este plan** (tres salvaguardas):

1. **Sus datos no entran en el análisis.** Los 24 runs del pilot sirven para calibrar presupuesto, umbral y overhead. No se usan para contrastar ninguna hipótesis y no forman parte del dataset de análisis.
2. **No había nada que se pudiera manipular.** La calibración ocurre antes de que sea posible estimar ninguna correlación métrica↔VD, porque con un solo run por celda no hay dispersión intra-celda que correlacionar. Es decir, no se pudo elegir un umbral "que favoreciera" a una métrica ni queriendo.
3. **Los criterios se escribieron antes de ejecutarlo** (decisión del pilot, 2026-06-09, en [[2 - Decisiones]]). Eran dos: el presupuesto es la meseta de la curva de monitorización más un margen, redondeado a múltiplo de 20; y el umbral es el que CNN y ResNet cruzan en torno al 30-60% del presupuesto al learning rate central. `run_pilot.py --report` imprime la evidencia, el investigador aplica esos criterios, y la decisión se registra en [[2 - Decisiones]] con sus datos antes de lanzar la matriz.

## Unidad de análisis y datos

**Tres niveles, y conviene no confundirlos.** La **unidad de medida** es el run: hay ~960. La **condición** es la celda, o sea la combinación dataset × arquitectura × optimizador: hay 24, con 40 runs cada una (8 learning rates × 5 seeds). Y la **unidad de inferencia confirmatoria es la celda**, no el run (ver §Estadístico).

**De dónde salen los datos.** Los predictores por ventana, de `metrics_at_window.parquet`. Las VD, de `summary.json` y `trajectory.parquet`. Todo por run, dentro de `reports/`.

**Instrumentación completa, sin selección previa** (decisión cerrada el 2026-07-17, [[2 - Decisiones]]). Las ocho métricas de gradiente y el baseline se miden en **todas** las runs y en **todas** las épocas, sobre el probe fijo. No se preselecciona ningún subconjunto para ahorrar coste.

Esto es una decisión del preregistro y no solo de ejecución, por un motivo concreto: descartar métricas antes de medirlas equivale a elegir qué hipótesis se pueden contrastar según el presupuesto de cómputo en vez de según la pregunta de investigación. H1 y H3 comparan métricas entre sí, así que quedarían apoyadas en un conjunto recortado por razones ajenas a la ciencia. El coste medido en el pilot es de ~2,08x el wall-clock de entrenamiento en el peor caso (fc × tiny_imagenet: 2153 s de métricas frente a 1993 s de train), dentro de la cota <3-4x fijada de antemano. La **poda** de métricas es posterior y solo afecta a lo que la memoria destaca, nunca a lo que se contrasta.

**De dónde viene la variación que se correlaciona.** Dentro de cada celda, la dispersión del predictor la genera el barrido de learning rate; las seeds aportan réplicas. Los dos ejes se analizan juntos dentro de la celda, sin promediar sobre seeds antes de correlacionar.

**Pseudo-replicación reconocida.** Los 40 runs de una celda no son intercambiables: forman 8 grupos de learning rate con 5 réplicas cada uno, de modo que el tamaño muestral efectivo está entre 8 y 40, no en 40. Por eso el ρ por celda es **descriptivo**, una observación del mapa y nada más, y la confirmación se hace entre celdas, donde los runs sí son disjuntos.

**Qué afirma el contraste intra-celda, y qué no.** Dentro de una celda, predictor y VD covarían en buena parte *a través del learning rate*. Así que un ρ alto en una celda no demuestra, por sí solo, que la métrica aporte información que el learning rate (conocido gratis y de antemano) no llevara ya.

Lo que la tesis afirma es la **transferibilidad**. La relación entre learning rate y eficiencia es específica de cada celda y no se conoce a priori. En cambio, un signo y una magnitud de ρ que se mantienen consistentes a lo largo de las 24 celdas sí son información que viaja entre configuraciones, y eso es lo que miden la etapa 2, H5 y la tabla de signos. El valor incremental dentro de la celda queda vigilado aparte, por H2 contra los baselines de loss.

## Variables

### Dependientes (eficiencia)

Todas las VD de eficiencia se leen sobre la curva de **validación**, nunca sobre la de test.

Hay una diferencia de suavizado entre ellas, y tiene una razón. VD1 y VD3 se leen sobre la curva **suavizada con mediana móvil centrada de 3 épocas**, porque ambas dependen de un punto concreto de la curva (dónde cruza el umbral, dónde está el mínimo) y un pico de ruido en una sola época las desplazaría. VD2 integra la curva **cruda**, porque integrar ya es una operación robusta y suavizar antes sería redundante. La curva cruda se conserva siempre en `trajectory.parquet` como análisis de sensibilidad. (Protocolo de evaluación, 2026-06-12, [[2 - Decisiones]].)

| VD | Definición | Curva | Notas |
|---|---|---|---|
| **VD1 (primaria)** | épocas-hasta-umbral de val-acc del dataset | suavizada | no cruza → **censurado** (§Censura) |
| **VD2** | AUC de la val loss dentro del presupuesto | cruda | caveat de sobreconfianza (abajo) |
| **VD3** | mejor val loss dentro del presupuesto | suavizada | caveat de sobreconfianza (abajo) |
| **VD4** | `final_test_acc`, evaluada una única vez al final del run | — | la variable objetivo certificada |
| **VD5 (gap, primaria de generalización)** | `final_gap_loss = final_test_loss − final_train_eval_loss` (positivo = sobreajuste) | — | subconjunto fijo del train (test-sized, `SPLIT_SEED`); familia propia (suelo + parcial) |
| **VD6** | `final_gap_acc = final_train_eval_acc − final_test_acc` | — | robustez del gap, mismo sentido |

**`seconds_to_threshold` queda fuera del análisis confirmatorio.** El wall-clock está confundido por la contención de la máquina (decisión "Timing por run"), así que medir "segundos hasta el umbral" mezcla la eficiencia del entrenamiento con lo ocupada que estuviera la GPU. Se usa solo como exploratorio, restando la columna acumulada `metric_seconds`, y nunca para afirmaciones entre celdas.

**Aviso sobre VD2 y VD3.** La val loss puede empeorar por sobreconfianza mientras la val accuracy sigue mejorando (Ru et al., arXiv:2006.04492, Ap. C.1-C.2), y esto pasa sobre todo en la cola del presupuesto. Es decir, el AUC de la loss y la mejor loss pueden dar peor nota a un run que en realidad está aprendiendo mejor. Esa es una de las razones por las que la VD primaria es VD1, que va sobre accuracy. VD2 y VD3 se interpretan con este sesgo declarado por escrito.

**VD5 y VD6, el gap de generalización** (decisión 2026-06-14, [[2 - Decisiones]]). Son escalares medidos al final del run sobre un subconjunto fijo y estratificado del train, del tamaño del test y elegido con `SPLIT_SEED`, idéntico en todos los runs. La versión en loss es la primaria, porque es la escala en la que están formuladas las garantías teóricas de GSNR y gradient disparity; la versión en accuracy acompaña como robustez.

Estas dos forman **familia de corrección propia**, con dos controles registrados de antemano. El primero es el **suelo de ajuste**: los contrastes del gap excluyen los runs cuyo `final_train_eval_acc` no llegue a un mínimo calibrado en el pilot, aunque esos runs siguen contando en las demás familias. El segundo es una **correlación parcial** por `final_train_eval_loss`, que responde a la pregunta de si la métrica predice el gap más allá de cuánto se ajustó el modelo al train.

### Predictores

Son las 8 métricas del registro completo más los baselines, todas medidas en cada ventana `f` y leídas de `metrics_at_window.parquet`. Se ordenan en tres niveles según lo que cuesta obtenerlas:

- **Nivel 0, sin tocar el gradiente:** `val-acc@f` (el titular), `val-loss@f` y TSE-EMA@f. Son la curva de monitorización y la train loss suavizada, leídas en la misma fracción del presupuesto. Este nivel es el **rival a batir de H2**. No tiene contraste confirmatorio propio, porque sus rechazos están garantizados "por construcción" (ver la tabla de signos); se reporta como referencia descriptiva del mapa.
- **Nivel 1, gradiente barato:** normalized gradient variance (NGV), gradient noise scale (GNS) y gradient disparity.
- **Nivel 2, las retadoras:** gradient confusion, stiffness, m-coherence, GSNR y GWA.

En todos los contrastes se reportan **todos** los predictores con sus estadísticos, salgan significativos o no. No hay ningún filtrado previo.

#### Composición del nivel 0 — justificación y fuentes (revisión 2026-06-11, fuentes verificadas sobre los textos originales)

El nivel 0 debe ser *el mejor predictor obtenible sin instrumentar el gradiente* ([[1 - Diseño]] §Baselines):

**`val-acc@f` es el titular**, por dos razones. Está en la misma escala que define VD1, y es la señal que la literatura de optimización de hiperparámetros usa de forma estándar para rankear configuraciones que difieren en learning rate, que es justo el eje que genera la dispersión de cada celda. Successive halving, Hyperband y ASHA promueven candidatos por métrica de validación temprana (Li et al. 2018, arXiv:1603.06560; Li et al. 2020, arXiv:1810.05934), y rankear por validación tras una sola época ya es casi óptimo para selección top-k (Egele et al. 2024, arXiv:2404.04111). Se lee de la curva ya registrada, así que no exige ningún cambio de pipeline.

**TSE-EMA se mantiene, pero sin ser titular.** Nunca fue validado para rankear hiperparámetros: sus propios autores lo declaran fuera de scope (Ru et al., arXiv:2006.04492, §4.2, *"Verifying the quality of various estimators for predicting the generalisation performance across different hyperparameters lies outside the scope of this paper"*).

Y hay un motivo de diseño para desconfiar aquí en concreto. En este estudio la train loss temprana es casi monótona en el learning rate, y el modo de fallo clásico de cualquier baseline basado en la curva son las **curvas que se cruzan**: un learning rate pequeño acumula menos loss al principio y acaba peor (el *short-horizon bias* de Wu et al. 2018, arXiv:1803.02021, y Li, Wei y Ma 2019, arXiv:1907.04595). Se conserva de todos modos porque es el mejor de su familia en el régimen para el que fue diseñado (benchmark de 31 predictores en White et al. 2021, arXiv:2104.01177), porque cuesta cero, y porque la matriz será de paso uno de los primeros tests controlados de TSE variando el learning rate. Caveats verificados con citas en [[Speedy Performance Estimation for Neural Architecture Search]].

**En las ventanas de este estudio, las variantes de TSE degeneran.** Con f entre el 5 y el 10% de presupuestos de 20 a 80 épocas, la suma cubre entre 1 y 8 épocas. A esa escala, TSE-EMA con γ = 0,999 apenas decae (0,999⁸ ≈ 0,992) y colapsa en la suma simple, y TSE-E con E = 1 *es literalmente* `train-loss@f`. El propio paper encuentra E = 1 como la mejor variante, lo cual apunta a que la señal está en el nivel reciente suavizado de la curva y no en la historia integrada. Dicho de otro modo: en este estudio el nivel 0 es, de hecho, "el nivel suavizado de la curva en f".

**La pendiente de la curva queda fuera del nivel 0.** Ninguna fuente consigue aislar valor de la pendiente por encima del nivel, y tres líneas independientes apuntan a que el nivel suavizado domina: el resultado 1-Epoch de Egele et al. 2024, el E = 1 óptimo de Ru et al., y Baker et al. 2017 (arXiv:1705.10823), que mezcla niveles y derivadas sin hacer ablación interna. Queda como exploratorio si algún resultado lo pide.

**El learning rate no entra como covariable del nivel 0.** Más allá de las razones de §Alternativas (ver "Condicionar por LR"), los valores de hiperparámetros por sí solos son predictores débiles comparados con las features de la curva: Baker et al. 2017 obtienen R² de 0,18 en su espacio de solo hiperparámetros, frente a 0,95 usando la curva.

**Comparador emparejado de VD1 (solo descriptivo).** Consiste en *predecir* las épocas-hasta-umbral invirtiendo un ajuste power-law de 3 parámetros sobre la curva de val-acc dentro de la ventana f. La power-law es la familia paramétrica con mejor respaldo para curvas de iteraciones (Kadra et al. 2023, arXiv:2302.00441; como alternativa amortizada existe LC-PFN, Adriaensen et al. 2023, arXiv:2310.20447).

Se construye aquí porque no existe baseline publicado para el objetivo "épocas hasta umbral"; lo más cercano en la literatura es invertir curvas extrapoladas. Se reporta como referencia descriptiva del mapa, igual que el resto del nivel 0: queda fuera de las familias confirmatorias y fuera del conjunto de covariables de H2, para no quemar grados de libertad cuando solo hay 40 runs por celda.

**Contexto de exigencia.** Vale la pena dejar claro cuánto se está pidiendo aquí: ninguno de los tres papers comparables del corpus usó un baseline basado en la curva de loss. Hölzl 2025 (arXiv:2510.25480) compara contra validation splits 90/10 y 99/1, LabelWave y gradient disparity. Forouzesh y Thiran 2021 (arXiv:2107.06665) usan k-fold CV. Y Jiang et al. 2020 (arXiv:1912.02178) lo controlan *por diseño*, entrenando todos los modelos hasta la misma loss final, precisamente porque *"otherwise one can simply use cross-entropy loss value to predict generalization"*. El protocolo de correlación parcial de este plan es por tanto más estricto que el estándar publicado, y así se declara en la memoria.

### Ventanas

`f` es la fracción del presupuesto de épocas en la que se lee el predictor. Se usan cinco valores, con papeles distintos:

- **f = 0,10 es la primaria.** Es la apuesta que hace H4: que la señal ya está ahí, al 10% del entrenamiento.
- **f ∈ {0,05, 0,25, 0,50} son secundarias.** Forman el barrido con el que H4 comprueba si esperar más mejora algo.
- **f = 1,0 es solo referencia de saturación.** Leer el predictor al final del entrenamiento no es "predicción temprana", así que no entra en ninguna afirmación predictiva del estudio.

## Estadístico

La inferencia va en **dos etapas**. Es el enfoque llamado *summary statistics*, estándar en análisis de grupo en neuroimagen (Holmes y Friston 1998), y equivale a un metaanálisis en el que cada celda hace de estudio individual.

1. **Etapa 1, descriptiva, dentro de cada celda: ρ de Spearman** entre cada predictor@f y cada VD, sobre los 40 runs de la celda. Se usa Spearman y no Pearson porque trabaja con rangos, y eso le permite absorber no linealidades y outliers. Pero sobre todo, y esto es lo importante aquí, le permite meter los runs censurados como peor rango en vez de descartarlos. El ρ̂ de cada celda es el estadístico-resumen que alimenta la etapa 2. Sus p-valores se anotan en el mapa, pero solo como descriptivos, porque son anticonservadores: el n efectivo de una celda es menor que 40 por el clustering de learning rate.
2. **Etapa 2, confirmatoria, entre celdas: Wilcoxon de rangos con signo**, por métrica, sobre los ρ de las celdas elegibles contra 0 y bilateral. Este nivel es robusto a la dependencia que hay dentro de las celdas: bajo la nula los ρ̂ se centran en 0 sea cual sea el agrupamiento por learning rate, y las celdas no comparten runs. Su validez descansa en que la distribución de ρ̂ sea simétrica **bajo la nula**, que es justo donde se sostiene (sin asociación, los ρ̂ se reparten simétricamente en torno a 0). No exige simetría bajo la alternativa.

**Tamaño de efecto de la etapa 2.** Se reportan cuatro cifras: la **pseudomediana de Hodges-Lehmann con su intervalo de confianza exacto**, la mediana muestral, el IQR y la fracción de celdas con signo consistente.

La pseudomediana de Hodges-Lehmann es la mediana de las medias de Walsh, es decir, de todos los promedios por parejas de los ρ̂. Es la titular por una razón técnica: es el estimador que el Wilcoxon de rangos con signo localiza de hecho, mientras que la mediana muestral es otro estimador distinto. Bajo simetría los dos coinciden, pero ρ está acotado en [−1, 1] y su distribución se vuelve asimétrica cuanto más lejos de 0 esté el centro, que es precisamente el caso interesante.

El intervalo de confianza importa aparte, y no es sustituible por el IQR. El IQR describe cuánto varían las celdas entre sí; el IC describe cuánta incertidumbre hay sobre el centro. Solo el segundo da la **precisión** del resumen cross-celda, y sin él el resultado se reporta sin barra de error.

El criterio numérico de H1 y H2 se evalúa sobre la mediana, por continuidad con la redacción de [[1 - Diseño]]. Ambas cifras se publican juntas y cualquier discrepancia entre ellas se declara.

Complementos:

**Pearson r por celda, como secundario.** Solo sobre runs no censurados y con VD finita. Sirve para ver si la conclusión depende de haber elegido un estadístico de rangos.

**Sensibilidad a la pseudo-replicación.** Se repite la etapa 1 sobre las medianas por learning rate, lo que deja n = 8 por celda en vez de 40, y se comprueba que el mapa (medianas cross-celda y signos) no cambia. Si cambiara, significaría que el resultado dependía de contar las seeds como observaciones independientes.

**Valor incremental (H2): correlación parcial de Spearman.** El estadístico-resumen por celda es la correlación que le queda a la métrica tras descontar el baseline gratuito, y el contraste confirmatorio es el mismo Wilcoxon cross-celda, lo que mantiene H2 coherente con el estadístico primario y con el tratamiento de la censura por rangos.

**La covariable primaria es `val-acc@f` sola (k = 1); la parcial sobre los tres baselines (k = 3) es sensibilidad.** Este orden se invirtió el 2026-08-01 respecto a la redacción anterior, y conviene explicar por qué, porque el instinto dice lo contrario.

La intuición de que condicionar por tres baselines es "más estricto" solo vale si los tres aportan ajuste distinto. Aquí no lo hacen: §Predictores demuestra que en estas ventanas TSE-EMA degenera hasta colapsar en `train-loss@f`, y `val-loss@f` y `val-acc@f` son dos lecturas de la misma curva de validación. Son covariables casi colineales, así que k = 3 produce **casi el mismo ajuste** que k = 1 pero con bastante más varianza en el estimador. El coste no está en los grados de libertad (df = n − 2 − k = 35 frente a 37, marginal) sino en la inflación de varianza por colinealidad, que la redacción anterior no consideraba. Gastar potencia ahí es especialmente caro porque H2 es la hipótesis decisiva y su potencia es ya el punto más débil del diseño (§Nota de potencia).

Además, `val-acc@f` sola *es* la barra que [[1 - Diseño]] §Baselines define con sus propias palabras: "un predictor muy fuerte y casi gratis: si las métricas de gradiente no lo superan, no hay aporte que defender". La barra es una, no tres. La versión k = 3 se reporta siempre como sensibilidad, y con ella se reportan las correlaciones por pares entre los tres baselines dentro de cada celda, que son el diagnóstico que respalda esta decisión.

**ΔR² como comprobación de un sesgo concreto, no como segunda opinión.** Se calcula el ΔR² de añadir la métrica a un modelo que ya lleva el baseline, sobre los runs **no censurados**. Su papel no es opinar otra vez sobre H2 sino vigilar una limitación específica de la parcial: una parcial de Spearman es una parcial de Pearson calculada sobre los rangos, así que solo elimina el efecto **lineal en el espacio de rangos** del baseline, y el bloque de empates que la censura introduce es precisamente una no linealidad en ese espacio. Es decir, la parcial puede **infra-ajustar** por el baseline, y ese sesgo va en dirección anticonservadora para H2. El ΔR² corre sobre la población donde ese bloque de empates no existe: si H2 se sostiene en ambas, la objeción queda cerrada. Si discrepan, **manda la parcial** por cobertura de población (incluye censurados), pero la discrepancia se reporta como lo que es, evidencia de que el sesgo está actuando.

**Brazo de equivalencia de H2** (añadido el 2026-08-01, antes de ver datos). H2 tiene dos salidas que la tesis declara como contribución: que las métricas aporten, y que no aporten. El problema es que un contraste que solo puede rechazar la nula convierte la segunda en "no hemos encontrado nada", y eso es compatible con dos cosas muy distintas: con que sean realmente redundantes, y con que no hubiera potencia para verlo (§Nota de potencia: con una parcial real de 0,15 la detección es del 0,46).

Para poder **afirmar** el negativo, cada métrica que no rechace la nula pasa por un test de equivalencia TOST sobre la misma parcial, con el mismo Wilcoxon cross-celda, margen **δ_H2 = 0,15** y α = 0,05 unilateral. El resultado de H2 por métrica es por tanto ternario: **aporta** (se rechaza la nula), **no aporta** (se rechaza la nula de equivalencia, es decir, la parcial cross-celda es demostrablemente menor que δ_H2) o **inconcluso** (ninguna de las dos, y se declara sin adornos). Ninguna métrica puede caer en las dos primeras a la vez.

**De dónde sale δ_H2 = 0,15.** El margen no viene de la tabla de Cohen sino de la pregunta de la tesis, que es si instrumentar el gradiente compensa. El coste medido es de ~2,08x el wall-clock de entrenamiento (§Unidad de análisis), así que la métrica tiene que aportar información suficiente para justificar duplicar el presupuesto de cómputo frente a un baseline que es gratis. Una parcial cross-celda por debajo de 0,15 explica menos del 2,3% de la varianza de rangos residual, y a ese precio eso es irrelevante para la decisión de cualquier practicante.

El valor se fija aquí, antes de ver datos. Está en el mismo orden de magnitud que el δ = 0,1 de H4 sin ser el mismo número, porque responden a preguntas distintas: allí el margen viene de la precisión de la propia mediana, aquí de la relevancia práctica frente a un coste conocido.

**Comparación entre familias (H3): diferencia pareada de medianas por celda.** Dentro de cada celda se calcula la mediana de \|ρ\| de la familia de alineación y la de la familia de variabilidad, y se toma su diferencia; el contraste es el Wilcoxon cross-celda sobre esas 24 diferencias. Que sea **pareado dentro de celda** controla la dificultad del dataset por construcción, y que compare medianas y no el máximo lo hace insensible al tamaño de las familias: bajo la nula, la mediana de 5 valores y la de 3 estiman la misma cantidad, así que su diferencia está centrada en cero (la de 3 es más ruidosa, nada más).

Esto sustituye al criterio anterior, que contaba en cuántas celdas la métrica de **mayor ΔR²** pertenecía a cada familia y exigía 16 de 24. Ese criterio estaba sesgado de forma grave y no reparable subiendo el umbral: alineación tiene 5 métricas y variabilidad 3, así que bajo la nula de que las 8 son intercambiables el argmax cae en alineación con probabilidad 5/8, lo que da **15 celdas de 24 esperadas por puro azar** y una probabilidad de **0,42** de superar el umbral de 16 sin que exista ningún efecto de familia. Cualquier criterio construido sobre un argmax hereda el sesgo del tamaño de los grupos que compara. Detectado y sustituido el 2026-08-01, antes de ver datos.

**Binomial de concordancia de signos: una herramienta, dos usos.** H5 y H6 son el mismo test exacto (binomial bilateral contra 0,5) aplicado a niveles distintos: H5 sobre los 12 pares SGD↔Adam pregunta si el signo de ρ coincide entre optimizadores, y H6 sobre las 24 celdas de cada métrica pregunta si el signo observado coincide con el que predice la tabla congelada. Verlo junto simplifica la implementación y el capítulo de metodología.

**Sobre validación fuera de muestra.** Todo este diseño mide asociación **dentro de muestra**: el ρ se calcula sobre los mismos runs, sin reservar nada. La afirmación de transferibilidad (§Unidad de análisis) se sostiene sobre la consistencia entre celdas que mide la etapa 2, no sobre una validación con datos retenidos. Se consideró añadir un leave-one-cell-out sobre el signo y se descartó el 2026-08-01: su información marginal sobre lo que ya dicen el Wilcoxon cross-celda y la fracción de celdas con signo consistente es pequeña, y no compensa añadir un contraste, una familia de corrección y una salvedad de potencia más. Queda como línea de trabajo futuro, y la limitación se declara en la memoria: este estudio evidencia consistencia entre configuraciones, no validación predictiva con retención.

**Lateralidad.** Todos los tests son **bilaterales** a α = 0,05 antes de corregir, con dos excepciones que son unilaterales por ser de no-inferioridad o equivalencia: H4 y el brazo TOST de H2. H6 predice signo, pero el signo se evalúa por concordancia, no con tests unilaterales.

**Reproducibilidad.** El análisis corre sobre Python gestionado por `uv`, con `scipy` como única dependencia estadística. Los tests exactos (Wilcoxon de rangos con signo, binomial) se toman de `scipy.stats` y no se reimplementan. Todo componente aleatorio (permutaciones, bootstrap, las simulaciones de §Nota de potencia) lleva semilla fija y declarada en el código, y las versiones de librería usadas para los números finales se registran en la memoria. El pipeline confirmatorio vive en `src/` y se ejecuta como script, no desde un notebook: los notebooks solo preparan figuras a partir de lo que el backend devuelve.

## Censura y exclusiones

**Censura en VD1.** Un run que no cruza el umbral dentro del presupuesto recibe el **peor rango**, empatado con los demás censurados. Ni se le asigna "infinito" ni se elimina. El motivo es que los extremos de la rejilla de learning rate divergen o no llegan *por diseño*, y por tanto son parte legítima del eje de eficiencia que el estudio quiere medir.

**Divergencia, lado VD.** Un run con NaN o Inf en la loss de entrenamiento o de monitorización en cualquier época recibe el peor rango en VD1, VD2 y VD3, empatado con los censurados, y queda excluido de Pearson y del ΔR². Se reporta el recuento por celda.

**Divergencia, lado predictor** (regla verificada el 2026-07-25). El valor de una métrica en la ventana `f` es válido si y solo si el run seguía finito *en esa época*. Que diverja después de `f` no invalida la medición: de hecho ese es el caso más informativo del estudio, una métrica temprana sana que anticipa un run catastrófico.

Lo delicado es que la comprobación no se puede hacer columna a columna. En un run divergente real (`fc/mnist/sgd` a lr = 1,0) salen NaN 23 de las 27 columnas de métrica, pero cuatro salen **0,0 finito**: `stiffness/sign_global`, `stiffness/sign_within`, `stiffness/sign_between` y `confusion/frac_neg`. El motivo es que la fracción de comparaciones de signo sobre gradientes que son NaN es 0. Es un valor que parece medido y no lo es.

De ahí la regla: la validez del predictor se decide con un indicador por (run, época), construido con `train_loss` y las columnas de coseno de esa fila. Cuando la fila no es válida se marcan como faltantes *todas* las columnas de esa métrica en ese run, incluidas las de signo. Sin esta regla, esas cuatro columnas entrarían en Spearman con un 0,0 sistemático concentrado en los learning rates altos, que son también los lentos y los censurados. Eso sería correlación fabricada por el estimador, no correlación medida.

**Celda degenerada en VD1.** Una celda con más del 80% de runs censurados, o con menos de 2 valores distintos de VD1, sale de los análisis de VD1 para *todas* las métricas. Su ρ̂ no es informativo y solo metería ruido en la etapa 2. Esa celda se mantiene en VD2 y VD3, y se reporta cuáles fueron. [[1 - Diseño]] ya anticipa que ocurrirá con FC sobre CIFAR-100 y Tiny-ImageNet. La etapa 2 opera sobre las celdas elegibles restantes.

**Atenuación desigual de ρ por censura, y cómo se vigila** (añadido el 2026-08-01, antes de ver datos). Meter los censurados como peor rango empatado es la decisión correcta, pero tiene un efecto que el plan no recogía: un bloque grande de rangos empatados **comprime mecánicamente el \|ρ\| alcanzable**. Una celda con el 50% de runs censurados tiene un techo de \|ρ\| más bajo que una con el 5%, sin que eso diga nada sobre la métrica.

Esto importa porque la etapa 2 compara los ρ̂ entre celdas como si fueran comparables, y la tasa de censura **está correlacionada con la dificultad de la celda**, que es justo el confusor que la inferencia en dos etapas existía para evitar. Excluir las celdas por encima del 80% no basta: entre el 0% y el 80% la atenuación varía de forma continua.

No se intenta corregir el ρ̂, porque eso sería inventar un estimador sin respaldo. Se vigila con dos comprobaciones preespecificadas que no requieren ningún cálculo nuevo: **(a)** se reporta la tasa de censura por celda junto a cada ρ̂, y se repite la etapa 2 restringida a las celdas por debajo del 25% de censura, comprobando que la conclusión no cambia; **(b)** se usan **VD2 y VD3 como control**, porque no tienen censura en absoluto, así que si la asociación sobrevive sobre ellas la censura no puede estar fabricando el resultado. Ambas se reportan siempre, no solo si el resultado incomoda.

**Exclusión total, y solo por fallo técnico.** Un run sin `summary.json` no existe para el launcher: se relanza, no se analiza. No hay ninguna otra causa de exclusión total preespecificada.
**Matriz incompleta** (regla añadida el 2026-08-01, antes de ver ningún dato). Todo el resto del plan asume las 24 celdas completas, pero el cómputo es una sola GPU durante unos 6 días seguidos, así que hay que escribir de antemano qué se hace si se acaba el tiempo. Decidirlo entonces sería exactamente el grado de libertad que un preregistro existe para eliminar.

La regla tiene tres partes. Primera: una celda entra en la **etapa 2 solo si está completa**, es decir, con sus 40 runs y su `summary.json`; las incompletas quedan fuera de todo contraste confirmatorio y se reportan como tales, indicando cuántos runs tenían. Segunda: el análisis confirmatorio **no se ejecuta con menos de 18 celdas elegibles**. Ese umbral está calculado, no estimado (§Nota de potencia): a la mediana de ρ = 0,30 que el propio H1 declara como efecto mínimo de interés, el Wilcoxon de etapa 2 tiene potencia 0,95 con 18 celdas al α corregido por BH, pero solo 0,69 con 12 y 0,54 con 10. Por debajo de 18, el estudio se reporta como incompleto y todo pasa a exploratorio. Tercera: H5 exige sus **12 pares SGD↔Adam completos**, porque con menos su binomial exacto solo distingue concordancia perfecta del azar; con menos de 12 se declara inconcluso en vez de correrse sobre una fracción.

Y una salvaguarda que las acompaña: el **orden de ejecución de la matriz se fija y se registra antes de lanzarla**, de modo que qué celdas acaben completas lo determine el calendario y no una elección tomada después de ver resultados.

**Missingness por métrica.** Si una métrica falla en tiempo de ejecución (`measure` aísla los fallos, así que un fallo no tumba el run), ese run sale de los tests de *esa* métrica y solo de esa. Se reporta el recuento. Si una métrica falla en más del 5% de los runs de una celda, se señala como descarte silencioso de facto y la celda se marca para esa métrica.

**Suelo de ajuste, solo para la familia del gap (VD5/VD6).** Los contrastes del gap excluyen los runs cuyo `final_train_eval_acc` no alcance un mínimo por dataset. Es el análogo al descarte de modelos no convergidos de Jiang et al. Esos runs **siguen contando** para velocidad y para rendimiento final: solo salen de la familia del gap.

**Valores del suelo, fijados el 2026-08-01 antes de ver datos de matriz: MNIST 0,97; CIFAR-10 0,65; CIFAR-100 0,35; Tiny-ImageNet 0,20.** Son exactamente los umbrales de accuracy que VD1 ya usa sobre val, aplicados aquí sobre train.

La elección es deliberadamente conservadora y conviene explicar por qué no se calibró sobre el pilot como decía la versión anterior de esta línea. El pilot no puede calibrar este número con honestidad: da **un** valor por celda, al learning rate central, a la seed 0 y medido al presupuesto **doblado**, de modo que su distribución está sesgada al alza y no cubre el eje que de verdad la puebla en la matriz, que es el barrido de learning rate de 3,5 décadas. Los extremos bajos de ese barrido apenas ajustarán, y el pilot no los contiene. Sus valores observados fueron MNIST 0,999-1,000; CIFAR-10 0,867-1,000; CIFAR-100 0,701-0,9998; Tiny 0,350-1,000 (0,380-1,000 en las re-corridas al presupuesto real).

Reutilizar el umbral de VD1 tiene una ventaja que ninguna calibración nueva podría igualar: **no introduce ningún número nuevo**. Ese umbral ya está fijado, ya está registrado con su evidencia y ya se calibró con criterios escritos de antemano, así que reutilizarlo añade cero grados de libertad al investigador. Y el argumento se dice en una frase: un run que ni siquiera alcanza **sobre train** la accuracy que el estudio exige **sobre val** es, por definición, un run que no aprendió.

El suelo se declara como lo que es, un filtro de "no aprendió", y **no** como calibración fina del confusor. La alternativa, un suelo relativo al mejor ajuste de cada celda, exigiría mirar datos de la matriz, que es justo lo que congelar prohíbe. Se asume por tanto que este suelo puede quedarse corto para su función, y esa carga la lleva el otro control, la parcial por `final_train_eval_loss`. Se declara como limitación.

El motivo es que, a presupuesto fijo, el gap crudo confunde dos cosas: cuánto sobreajusta el modelo y cuánto llegó a ajustar el train. Un modelo que no aprendió nada tiene un gap pequeño y parece generalizar de maravilla. La respuesta de este plan son dos controles combinados, el suelo (que excluye) y la parcial por `final_train_eval_loss` (§Estadístico). Es una respuesta analítica, más débil que un control experimental, y se declara como limitación.

## Corrección por comparaciones múltiples

Se usa **Benjamini-Hochberg (BH) a q = 0,05**, no Bonferroni, por tres motivos. Los predictores están correlacionados entre sí. El objetivo del estudio es un mapa de qué métricas señalan algo, no una única afirmación. Y Bonferroni, que controla la tasa de error por familia, sería brutalmente conservador aplicado a cientos de tests. BH controla la FDR de forma demostrada bajo independencia (Benjamini y Hochberg, 1995) y bajo dependencia positiva del tipo PRDS (Benjamini y Yekutieli, 2001).

**Hay un punto delicado de validez, y conviene decirlo abiertamente.** Los tests de este plan son bilaterales, y PRDS *no se cumple en general* para tests bilaterales de correlación: Yekutieli (2008) da contraejemplos explícitos en configuraciones gaussianas bilaterales. Es decir, la garantía de BH deja de ser automática justo en el caso que aquí interesa, que es juntar en una misma familia predictores correlacionados con signos esperados opuestos (por ejemplo gradient confusion, que se espera +, frente a stiffness, que se espera −).

La dependencia se descompone en dos partes limpias. **Entre celdas** los runs son disjuntos, así que los p-valores son independientes. **Dentro de una celda** los 11 predictores comparten los mismos 40 runs, así que están correlacionados. O sea, el problema vive dentro de la celda, no entre celdas.

A eso se suma un segundo problema independiente: cada p-valor por celda es de por sí anticonservador, porque el n efectivo es menor que 40 (§Unidad de análisis). Las dos cosas juntas son la razón de que el nivel confirmatorio se suba al cross-celda.

**Familias preespecificadas.** El principio es que la unidad de corrección coincida con la unidad de afirmación: H1 y H2 afirman *por métrica y al nivel cross-celda*, mientras que el mapa por celda es descriptivo. De ahí salen tres familias:

1. **Familia confirmatoria primaria:** los **8 Wilcoxon cross-celda**, uno por métrica de gradiente, en la combinación (f = 0,10, VD1). Es una sola pasada de BH sobre 8 p-valores. Como los 8 tests comparten celdas y runs, cabe esperar dependencia positiva, y como blindaje se reporta también la cota BY, que a este nivel es barata (c(8) ≈ 2,72). Los baselines quedan fuera de esta familia, porque son referencia descriptiva (§Predictores).
2. **Familia confirmatoria de H2:** los 8 Wilcoxon análogos, pero sobre las correlaciones parciales. Pasada de BH propia, también de 8, con su cota BY.
3. **Familia de H6:** los 8 binomiales de concordancia de signos, uno por métrica sobre sus 24 celdas. Pasada de BH propia de 8, con su cota BY. H5 no forma familia: es un único binomial por métrica sobre las que superaron H1, y su recuento se reporta sin pasada propia.
4. **Mapa por celda, descriptivo:** **una pasada de BH por predictor** sobre sus 24 celdas, en cada combinación de ventana y VD. Son familias de 24 p-valores. Aquí la validez es limpia: las celdas usan runs disjuntos, así que los 24 p-valores son **independientes** y BH vale por el resultado de 1995. Reagrupar por predictor en vez de por celda elimina por construcción la objeción PRDS/bilateral, que solo surgiría al agrupar los 11 predictores correlacionados *dentro* de una misma celda. Estas pasadas sirven para anotar el mapa y **ninguna afirmación se apoya en ellas**, porque sus p-valores individuales son anticonservadores.

**Robustez y cota conservadora:**

**Cota de sensibilidad BY.** Se reporta además **Benjamini-Yekutieli**, que es válido bajo dependencia *arbitraria*, incluida la negativa. Funciona como cota dura: lo que sobrevive a BY es robusto a cualquier estructura de dependencia que haya. En el nivel confirmatorio sale barata, con c(8) ≈ 2,72. En el mapa por celda, haber reagrupado por predictor también la abarata: c(24) ≈ 3,78 frente al c(264) ≈ 6,15 que costaría una pasada única.

**Validación por permutación, opcional, como blindaje.** Permutar la VD dentro de cada celda y recalcular Spearman estima empíricamente la nula *conjunta*, sin asumir PRDS en ningún momento. Hay una variante que respeta el clustering y es la preferible aquí: permutar grupos de learning rate completos (bloques de 5 seeds) en vez de runs sueltos. Es coherente con el bootstrap previsto para H4, y se extiende a la etapa 2 sin más que recalcular los Wilcoxon sobre los ρ permutados, lo que da la nula conjunta de la familia confirmatoria.

**Control entre familias.** Se gestiona por jerarquía de titulares: lo secundario no asciende a titular. Si alguien exigiera garantía formal sobre el conjunto de todo lo reportado, el marco de inferencia selectiva de Benjamini y Bogomolov (2014) formaliza exactamente esta estructura de dos niveles. Es la respuesta preparada por si el tribunal aprieta.

Se reportan siempre los q-valores (BH, y BY como cota) junto a los p crudos. Lo que solo sobreviva en familias secundarias, es decir en otras ventanas y otras VD con la misma estructura de tres niveles, no asciende a titular.

**Storey queda descartado como primario.** El q-valor adaptativo de Storey gana potencia estimando π₀ en vez de asumir π₀ ≈ 1. Pero es menos estable, y presenta rechazos paradójicos cuando muchas de las nulas son falsas (Reiss et al.). Para una tesis prima la estabilidad sobre la potencia extra, así que queda como mucho para un análisis de sensibilidad.

### Referencias (corrección múltiple)

- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B 57(1):289–300. — procedimiento BH original; control de FDR bajo independencia.
- Benjamini, Y. & Yekutieli, D. (2001). *The control of the FDR in multiple testing under dependency.* Annals of Statistics 29(4):1165–1188. — validez bajo PRDS y la corrección BY bajo dependencia arbitraria.
- Yekutieli, D. (2008). — contraejemplos de no-PRDS en tests gaussianos bilaterales; para el caso concreto de tests de correlación bilaterales ver [*Asymptotic control of FWER… application to correlation tests* (arXiv 2007.00909)](https://arxiv.org/pdf/2007.00909).
- Storey, J. D. (2002). [*A direct approach to false discovery rates* (q-valor adaptativo)](http://genomics.princeton.edu/storeylab/papers/directfdr.pdf) — más potencia, menos estable.
- [*Paradoxical results of adaptive false discovery rate procedures in neuroimaging studies*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3699340/) — por qué no se adopta Storey como primario.
- Benjamini, Y. & Bogomolov, M. (2014). [*Selective inference on multiple families of hypotheses* (JRSS-B 76:297–318)](https://academic.oup.com/jrsssb/article/76/1/297/7075946) — FDR jerárquica sobre familias.
- [*Conditional calibration for false discovery rate control under dependence* (Fithian & Lei)](https://www.stat.berkeley.edu/~wfithian/fdr-dependence.pdf) — panorama de control de FDR bajo dependencia.
- [*Multiple testing under negative dependence* (arXiv 2212.09706)](https://arxiv.org/pdf/2212.09706) — resultados recientes para dependencia negativa.

## Contrastes por hipótesis

| Hipótesis | Contraste | Criterio |
|---|---|---|
| **H1** — existencia | Wilcoxon cross-celda, familia confirmatoria primaria | ≥1 métrica de gradiente con \|mediana de ρ\| ≥ 0,3 sobre las celdas elegibles **y** q < 0,05 |
| **H2** — valor incremental (decisiva) | Wilcoxon cross-celda sobre las parciales, familia H2, **más brazo de equivalencia TOST** con δ_H2 = 0,15 para las que no rechacen | Resultado ternario por métrica: **aporta** (≥1 métrica con q < 0,05), **no aporta** (se rechaza la equivalencia: redundante con la curva de loss, negativo válido y afirmable) o **inconcluso** (ni una cosa ni la otra, se declara sin adornos). H1 ✓ con H2 "no aporta" es el negativo fuerte que la tesis anuncia como contribución |
| **H3** — familia ganadora | Wilcoxon cross-celda sobre la **diferencia pareada** de medianas de \|ρ\| por familia dentro de cada celda (§Estadístico) | una familia domina si la diferencia mediana cross-celda se aparta de 0 con q < 0,05; si no, H3 queda sin resolver y así se reporta. No se fija un umbral de magnitud porque no hay ninguno defendible a priori: se reporta el tamaño de efecto con su IC y se interpreta. Insesgado respecto al tamaño de las familias (5 vs 3), a diferencia del criterio de argmax que sustituye |
| **H4** — suficiencia temprana | no-inferioridad: Wilcoxon unilateral sobre dᵢ − δ | "satura pronto" si se rechaza H₀: mediana(d) ≥ δ, con δ = 0,1; solo métricas que pasaron H2; puede salir "inconcluso" y se reporta como tal. Se reporta **siempre la sd(d) alcanzada**, porque de ella depende por completo la potencia del contraste (§Nota de potencia), y un inconcluso debe distinguir si lo fue por efecto o por dispersión |
| **H5** — invariancia cross-optimizador | binomial exacto contra 0,5 sobre los 12 pares de celdas SGD↔Adam (seeds compartidas); exige los 12 pares completos y **solo se corre sobre las métricas que superaron H1** | concordancia de signo de ρ por par. La restricción a las que pasan H1 es la misma disciplina que ya aplica H4 y evita que el signo de un ρ ≈ 0, que es una moneda al aire, entre al test como si fuera información. Se reporta junto a la mediana de \|ρ\|. **Dos limitaciones declaradas:** los 12 pares no son independientes (comparten 4 datasets y 3 arquitecturas), y con 12 pares el contraste solo confirma invariancia casi perfecta (potencia 0,98 a concordancia 0,95, pero **0,39** a 0,75), así que un no-rechazo no es evidencia de no-invariancia |
| **H6** — mecanismo, con signo | **binomial exacto contra 0,5** por métrica sobre sus 24 celdas, familia de corrección propia | fracción de celdas cuyo signo coincide con el de la tabla congelada abajo. Bajo ausencia de mecanismo esa fracción es 0,5, que es la nula que faltaba: hasta el 2026-08-01 H6 era la única hipótesis sin criterio de falsación, pese a que [[1 - Diseño]] la describe como "prueba más exigente que la magnitud". Se reporta con la mediana de ρ cross-celda. Los signos marcados como *extrapolados* en la tabla se contrastan aparte de los *fuertes*: una discordancia en un extrapolado falsifica la extrapolación, no el paper |
| **Gap — doble disociación** (generalización) | Wilcoxon cross-celda sobre las parciales (controladas por `final_train_eval_loss`), familia FDR propia del gap; suelo de ajuste por `final_train_eval_acc` (§Censura) | las métricas que reclaman generalización (GSNR, GWA, gradient disparity, stiffness, m-coherence) se asocian más al gap que al AUC de velocidad, y las de velocidad al revés. **Criterio de la disociación (añadido el 2026-08-01):** para cada métrica se toma dentro de cada celda la diferencia \|ρ(gap)\| − \|ρ(VD2)\| y se contrasta con el mismo Wilcoxon cross-celda; la disociación se declara si el signo de esa diferencia es el predicho y q < 0,05 en **ambos** grupos de métricas, no solo en uno. Es la misma herramienta que H3 y no añade maquinaria. Sin este criterio la doble disociación era una afirmación sin test, el mismo hueco que tenía H6. H1/H2/H6 corren también contra VD5/VD6 |

**Detalle de H4.** Para cada celda elegible se calcula dᵢ = \|ρᵢ@0,50\| − \|ρᵢ@0,10\|, es decir, cuánto mejora la correlación por esperar del 10% al 50% del presupuesto. El margen preespecificado es **δ = 0,1**, entendido como la mínima diferencia de \|ρ\| con relevancia práctica, que además está en el orden de la precisión de la propia mediana cross-celda.

El marco es TOST / no-inferioridad (Lakens 2017), elegido porque es falsable en ambas direcciones: puede concluir que satura, que no satura, o que no hay evidencia suficiente. Como descriptivo de apoyo por celda se acompaña de un intervalo de confianza para la diferencia de correlaciones dependientes, bien por bootstrap BCa (10 000 remuestras de runs), bien con los ICs cerrados de Zou (2007).

### Tabla de signos esperados (vs VD1, épocas-hasta-umbral: menor = más rápido)

| Métrica | Signo esperado | Base |
|---|---|---|
| gradient confusion | + | fuerte (Sankararaman et al.) |
| stiffness (intra-clase) | − | fuerte (Fort et al.) |
| m-coherence | − | fuerte vía CGH (Chatterjee 2020: más coherencia → la loss cae más rápido; medido en train). Chatterjee & Zielinski no afirma velocidad |
| gradient disparity | + | extrapolada (Forouzesh predice test error, no velocidad) |
| NGV | + | fuerte (Faghri et al.) |
| GNS | + | fuerte (McCandlish ec. 2.7/D.1: más pasos a batch fijo; requiere B ≲ 𝓑 y LR razonable, y el GNS medido depende del LR) |
| GSNR | − | extrapolada (Liu solo afirma el gap; vs VD4: + también extrapolada, derivada del gap) |
| GWA | − | extrapolada (Hölzl calla sobre velocidad; vs VD4: + fuerte, con matices) |
| TSE / val-loss@f (baselines) | + | por construcción |
| val-acc@f (baseline) | − | por construcción |

Signos verificados contra los papers el 2026-07-17 (veredictos y citas en [[2 - Decisiones]]). El reparto de predicciones fuertes queda así: GSNR la lleva sobre el gap (− vía OSGR; su + vs VD4 es derivación, no claim del paper); GWA la lleva sobre VD4 (+; Pearson 0,99 solo ConvNeXt/CIFAR-10, 0,92 cross-arquitectura, medida sobre el máximo de toda la trayectoria, no en ventana temprana); ambas columnas VD1 se evalúan aparte como extrapolación.

**Signos esperados frente al gap** (`final_gap_loss`, donde positivo significa más sobreajuste; decisión 2026-06-14, verificados contra los papers el 2026-07-17):

- **GSNR: −**, con base fuerte. Viene del OSGR, ecuación 22 de Liu, y encaja bien porque el gap de ese paper está en loss, que es exactamente la cantidad de VD5.
- **Stiffness intra-clase: −.**
- **M-coherence: −**, con una salvedad que hace el propio paper: la conexión entre coherencia y generalización es "complicated", porque con 100% de label noise la coherencia también alcanza valores altos. Según su teoría, lo informativo es la coherencia **temprana**.
- **GWA: −**, pero solo direccional y cualitativa. El paper nunca mide test loss menos train loss; su claim cuantitativo fuerte es frente a VD4.
- **Gradient disparity: +.** Es la señal de sobreajuste de su paper, así que más disparity implica más gap.

**Cuidado con un cruce de signos que despista.** GSNR y GWA son **+** frente a VD4 (`final_test_acc`, donde más alto es mejor) pero **−** frente al gap (donde más alto es peor). No es una contradicción: es que VD4 y el gap apuntan a sentidos opuestos de "bueno".

Las métricas de velocidad (gradient confusion, NGV, GNS) **no tienen predicción direccional propia sobre el gap**; McCandlish lo declara explícitamente fuera de su modelo (caveat 6). Su asociación se reporta igualmente, pero la predicción fuerte para ellas es la doble disociación: que se asocien menos con el gap que con la velocidad.

## Agregación entre celdas

**Por celda primero, agregado después.** Es la guardia contra la paradoja de Simpson: si se juntaran los ~960 runs de golpe, la distinta dificultad de cada dataset podría fabricar una correlación que no existe dentro de ninguna celda.

**El resumen cross-celda no es solo un resumen, es el contraste confirmatorio.** Consta de la mediana y el IQR de ρ sobre las celdas elegibles, la fracción de celdas con signo consistente, y el Wilcoxon de etapa 2, todo por métrica y por ventana. Ese es el formato del "mapa" que reporta la memoria.

**La independencia entre celdas es aproximada, y se declara como tal.** Las celdas no comparten runs, pero sí comparten datasets, arquitecturas y seeds: una misma arquitectura con una misma seed arranca de la misma inicialización. Es una limitación reconocida. La dependencia residual que cabe esperar es débil y positiva, y si no lo fuera, la cota BY cubre el caso.

**Agregado pooled, como secundario.** Correlación sobre los ~960 runs, pero con los rangos calculados *dentro* de cada celda, lo que equivale a estandarizar por condición. Los modelos de efectos mixtos quedan como exploratorio, por si el resumen anterior se queda corto.

**Figura de sanidad val↔test, descriptiva.** Un scatter de `final_val_acc` frente a `final_test_acc` sobre los ~960 runs, preespecificado en la decisión del protocolo de evaluación ([[2 - Decisiones]]). Recupera el diagnóstico de concordancia que se pierde al no evaluar test por época. No alimenta ningún contraste.

**Reporte por celda para todas las familias** (Dziugaite et al. 2020). Además del resumen cross-celda, para cada métrica se reporta la **distribución** de asociaciones por celda: la mediana y también el peor caso, no solo el estadístico agregado. La razón la da bien el propio paper: promediar la calidad predictiva entre entornos no es resumen suficiente, porque *"a satisfying theory cannot simply predict well on average"*.

## Nota de potencia

Recalculada por simulación Monte Carlo el 2026-08-01 (20 000 réplicas por punto, ICs de Wilson sobre la estimación), antes de ver ningún dato de matriz. No hay fórmula cerrada para un Wilcoxon de rangos con signo aplicado a coeficientes de correlación, así que la vía es simular bajo la verdad asumida y analizar con el test exacto que se va a usar. Reproducible con `src/power_analysis.py`. La versión anterior de esta sección daba cifras estimadas de oído: su punto central resultó correcto, pero omitía la corrección por multiplicidad y no cubría H2, H4 ni H5.

**α efectivos.** La potencia se calcula al α que el criterio realmente exige, no a 0,05: los criterios de H1 y H2 piden q < 0,05 bajo BH sobre familias de 8, cuyo peor caso es α = 0,05/8 = **0,00625**, y la cota dura BY usa α = 0,05/(8·c(8)) = **0,00230** con c(8) = 2,718. Calcular a 0,05 y decidir a q es el error de potencia más común, y es el que cometía la versión anterior de esta nota.

**Etapa 2 (confirmatoria), con las 24 celdas.** A la mediana real de ρ = 0,30 que H1 declara como efecto mínimo de interés y dispersión cross-celda de 0,25, la potencia es 1,000 en crudo, **0,993 bajo BH** y 0,982 bajo BY: la afirmación original queda confirmada y es incluso conservadora. A mediana 0,20 cae a 0,956 / 0,785 / 0,659, así que la garantía vale para el umbral declarado y no por debajo. El **efecto mínimo detectable** al 80% de potencia con 24 celdas es una mediana de ρ de 0,153 en crudo, **0,205 bajo BH** y 0,226 bajo BY: el SESOI de H1 (0,30) queda holgadamente por encima del MDE, que es la relación que debe cumplirse para que el criterio no esté limitado por potencia.

**Etapa 2 con matriz incompleta.** Potencia a mediana 0,30 y dispersión 0,25, por número de celdas elegibles: 24 → 0,993 (BH) / 0,982 (BY); 20 → 0,976 / 0,933; 18 → 0,949 / 0,880; 16 → 0,900 / 0,809; 14 → 0,822 / 0,653; 12 → 0,685 / 0,440; 10 → 0,542 / 0,291. De aquí sale el **mínimo de 18 celdas** de §Censura, calculado y no razonado por analogía.

**Suelo discreto, anterior a la potencia.** El Wilcoxon de rangos con signo es discreto: con n celdas, el p bilateral más pequeño que puede producir es 2^(1−n). Con 8 celdas eso es 0,0078, **mayor** que el α de BH (0,00625), de modo que el test no puede rechazar ni con las 8 celdas apuntando en la misma dirección con ρ = 0,9. El suelo duro es de 9 celdas para BH y de 10 para BY. Es un modo de fallo que ninguna curva de potencia muestra y que solo aparece al corregir por multiplicidad.

**H2 (decisiva), sobre las parciales.** H2 no corre sobre el ρ crudo sino sobre la parcial condicionada por tres baselines, uno de ellos `val-acc@f`, fuerte por construcción (§Predictores). Las parciales encogen respecto al ρ crudo, así que heredar la potencia de H1 es un error de escala. Potencia con 24 celdas bajo BH según la mediana real de la parcial: 0,30 → 0,995; 0,25 → 0,949; 0,20 → 0,790; 0,15 → **0,464**; 0,10 → 0,164. Consecuencia que gobierna el diseño del contraste: un aporte incremental real pero modesto (parcial ≈ 0,15) se detectaría menos de la mitad de las veces, y como el negativo de H2 es una **contribución declarada** de la tesis y no un subproducto, no basta con no rechazar. Por eso H2 lleva brazo de equivalencia (§Estadístico y §Contrastes): sin él, "las métricas son redundantes con la curva de loss" y "no hemos tenido potencia para verlo" serían indistinguibles en el informe final.

**H4 (suficiencia temprana).** Su potencia depende de la dispersión cross-celda de dᵢ = \|ρᵢ@0,50\| − \|ρᵢ@0,10\|, desconocida a priori. Bajo saturación real (mediana de d = 0) y α = 0,05 unilateral con 24 celdas: sd(d) = 0,10 → 0,999; 0,15 → 0,929; 0,20 → 0,745; 0,30 → 0,450. Como d es una diferencia de correlaciones **dependientes** medidas sobre los mismos runs, lo esperable es sd pequeña y potencia alta, pero es una conjetura y se declara como tal. La sd(d) alcanzada se reporta siempre junto al resultado de H4, y si el contraste sale inconcluso se distingue explícitamente si fue por efecto o por dispersión. Es estimable pronto con las primeras celdas completas, dentro de los diagnósticos de adecuación del diseño.

**H5 (invariancia cross-optimizador).** Binomial exacto bilateral sobre los 12 pares SGD↔Adam a α = 0,05. Potencia según la concordancia real: 0,75 → **0,391**; 0,85 → 0,736; 0,95 → 0,980; 1,00 → 1,000. Es decir, **H5 solo puede confirmar invariancia casi perfecta**: una invariancia parcial genuina (3 de cada 4 pares concordantes) se declararía no significativa la mayoría de las veces. Se asume como limitación explícita en vez de descubrirse a posteriori, es la razón por la que el resultado de H5 se reporta siempre junto a la mediana de \|ρ\|, y es la razón por la que H5 exige los 12 pares completos: con 8 pares el p mínimo alcanzable es 0,0078 y con 6 pares harían falta los 6 concordantes para rechazar siquiera.

**H6 (concordancia de signos).** Binomial exacto bilateral sobre las 24 celdas de cada métrica, contra 0,5. Potencia según la concordancia real: 0,65 → 0,211; 0,75 → 0,607; 0,85 → **0,943**; 0,95 → 1,000. Es decir, detecta concordancia fuerte y no la moderada, y el efecto mínimo detectable al 80% está en torno a 0,80. Se declara como limitación: un no-rechazo de H6 no es evidencia de que la teoría se equivoque.

**Familia del gap (VD5/VD6).** Usa el mismo Wilcoxon de etapa 2, así que la tabla de arriba la cubre, con una salvedad: el suelo de ajuste excluye runs, de modo que su n por celda es menor que 40 y sus ρ̂ algo más ruidosos. La potencia efectiva de la familia del gap es por tanto **algo menor** que la de la familia primaria, y no se simula aparte porque cuánto menor depende de cuántos runs caigan bajo el suelo, que es dato de matriz. Se reporta el n efectivo por celda junto a cada resultado del gap.

**H3 (familia ganadora).** Usa la misma maquinaria de etapa 2 (Wilcoxon cross-celda sobre 24 diferencias pareadas), así que su potencia es la de la etapa 2 leída sobre la escala de la diferencia de medianas de \|ρ\| entre familias. No se simula aparte porque la tabla de etapa 2 ya la cubre; lo que no se puede anticipar es la magnitud de esa diferencia, que no tiene precedente en la literatura.

**Etapa 1 (descriptiva, contexto del mapa).** Con n = 40 por celda, el test de Spearman a α = 0,05 bilateral tiene 0,77 de potencia para \|ρ\| = 0,44 y solo **0,43** para \|ρ\| = 0,3 (Fisher-z con el factor de inflación 1,06 propio de Spearman; sin él saldría 0,47, que es la cifra que daba la versión anterior). Sus p-valores son además anticonservadores porque el n efectivo está entre 8 y 40 por el clustering de LR: al n efectivo de 8, la potencia para \|ρ\| = 0,3 baja a 0,10. Doble motivo por el que las anotaciones por celda son descriptivas y ninguna afirmación se apoya en una celda suelta.

**Salvedad del método.** Las simulaciones modelan los ρ de celda como normales truncados a (−1, 1). La distribución real será asimétrica y probablemente más dispersa en las celdas difíciles, así que las potencias de arriba son un caso algo optimista. El sesgo va en la dirección que refuerza los mínimos exigidos (18 celdas, 12 pares), no en la que los relaja.

## Alternativas consideradas y descartadas

**Inferencia confirmatoria por celda**, que era el plan original. Se descartó por dos fallos. Uno, la pseudo-replicación por learning rate, que hace anticonservadores los p-valores. Dos, un criterio de H1 incoherente con la potencia real del diseño. Reagrupar las familias de BH por predictor arreglaba la dependencia *entre tests*, pero no la calibración de cada p-valor individual, que era el otro problema. Las dos piezas se conservan, pero para el mapa descriptivo.

**Inferencia intra-celda consciente del cluster.** Se miraron tres vías y ninguna compensa. El bootstrap por clusters de learning rate se queda corto, porque 8 clusters son pocos para remuestrear. Los modelos mixtos con el learning rate como efecto son artillería difícil de justificar y de explicar en un TFG. Y rmcorr (Bakdash y Marusich 2017) responde en realidad a otra pregunta: la asociación a learning rate fijo, entre seeds, donde la varianza es minúscula. Las tres son sobrecomplicación frente a la etapa 2, y quedan como exploratorio.

**Condicionar por learning rate** para aislar el aporte de la métrica. Suena razonable y no lo es, por dos motivos. Eliminaría por diseño casi toda la varianza que el estudio genera, dejando solo la de las seeds. Y la relación entre learning rate y eficiencia es no monótona, así que una parcial de rangos tampoco la limpiaría. La respuesta correcta al confusor del learning rate es otra: la transferibilidad cross-celda, más H2 contra los baselines de loss.
**Contar celdas con \|ρ̂\| ≥ 0,3 sin exigir significancia en cada una**, y testar ese recuento contra su expectativa bajo la nula (~6% de celdas por azar si ρ = 0). Es válido y es lo más fiel a la literalidad de [[1 - Diseño]]. Se descartó como primario por dos motivos: mezcla magnitud y recuento en un criterio menos estándar que el Wilcoxon agregado, y ese 6% de referencia es optimista cuando el n efectivo es menor que 40. Puede reportarse como secundario.

**H4 solo descriptivo**, es decir, dibujar la curva de ρ frente a la ventana sin ningún test. Se descartó porque dejaba H4 sin contraste falsable. La curva se mantiene como figura, pero el contraste es el de no-inferioridad.

**Bonferroni y Storey.** Ver §Corrección por comparaciones múltiples.

## Qué falta para congelar

1. ~~**Confirmación del tutor** al protocolo train/val/test~~ — **cerrada el 2026-06-12**: la monitorización de VD1–VD3 lee val (suavizada para VD1/VD3) y VD4 existe; protocolo implementado ([[2 - Decisiones]]).
2. ~~**Pilot de calibración:** fija los valores definitivos de presupuesto y umbral por dataset que parametrizan VD1/VD2~~ — **cerrada el 2026-06-17**, registrada con su evidencia el 2026-07-17 ([[2 - Decisiones]]): MNIST 20 / 0,97; CIFAR-10 40 / 0,65; CIFAR-100 40 / 0,35; Tiny-ImageNet 40 / 0,20. La estructura del plan nunca dependió de ellos.
3. ~~**Suelo de ajuste del gap:** sigue sin valor fijado (aplazado a propósito el 2026-07-17)~~ — **cerrada el 2026-08-01**: fijado en §Censura como umbral absoluto por dataset (MNIST 0,97; CIFAR-10 0,65; CIFAR-100 0,35; Tiny 0,20), reutilizando los umbrales de VD1 aplicados sobre train. No se calibró sobre el pilot porque el pilot no puede calibrarlo con honestidad (un valor por celda, al LR central, a presupuesto doblado); el razonamiento completo está en §Censura.

**Ya no queda ninguna puerta abierta.** Lo que falta es el acto formal: mover este documento a `docs/research/` y registrar la congelación con fecha en [[2 - Decisiones]].
4. ~~**Decisión de instrumentación** (mantener la medición completa, cerrada el 2026-07-17 en [[2 - Decisiones]]): incorporar su texto aquí~~ — **cerrada el 2026-08-01**: incorporada como bullet propio en §Unidad de análisis y datos, con su justificación de preregistro y el coste medido.
5. ~~**Cobertura de columnas:** comprobar, contra el esquema de un run del pilot, que toda cantidad invocada por §Contrastes existe ya como columna~~ — **cerrada el 2026-08-01**: verificada contra el esquema real de un run del pilot; las seis VD, los ocho predictores, los tres del nivel 0, los ingredientes de la regla de censura del lado predictor y las columnas de timing están todos presentes. No falta ninguna, así que ninguna carencia del plan obliga a re-correr la matriz. Evidencia en [[3 - Progreso]] §Pasos inmediatos. El dry-run completo del pipeline sobre datos sintéticos con efecto plantado sigue pendiente y puede ejecutarse mientras la matriz corre.
6. ~~**Regla de matriz incompleta** y **política de enmiendas**~~ — huecos detectados y **cerrados el 2026-08-01**: la primera en §Censura y exclusiones, la segunda (con la salvedad de que congelar el texto no congela la implementación) en §Qué congela este documento.

El gap de generalización quedó **confirmado e integrado el 2026-06-14** (decisión en [[2 - Decisiones]]; propuesta original en el histórico git de `pending/Gap de generalización como variable objetivo.md`): su familia de contrastes (suelo de ajuste, parcial por `final_train_eval_loss`, doble disociación sobre parciales) y el reporte por celda para todas las familias ya están en el cuerpo del plan. Lo único que el gap añade a la calibración del pilot es el **suelo de ajuste** (distribución de `final_train_eval_acc` por celda), con los mismos criterios pre-escritos antes de lanzar la rejilla.

Al cerrarse todo: mover este documento a `docs/research/`, registrar la congelación en [[2 - Decisiones]] con fecha, y solo entonces mirar resultados de la matriz.

## Historial de revisiones

- **2026-06-10** — primera redacción, antes de ejecutar pilot y matriz.
- **2026-06-11** — pasada crítica con cuatro correcciones. (1) Pseudo-replicación intra-celda reconocida (el LR clusteriza los 40 runs) → respuesta central: **inferencia en dos etapas**, el ρ por celda baja a descriptivo y la confirmación sube al nivel cross-celda. (2) Criterio de H1 retirado: exigía significancia por celda en ≥13/24, pero la propia nota de potencia estima que un ρ = 0,3 real solo sale significativo en ~11/24 celdas sin corregir — exigía de facto ρ ≳ 0,45 y desalineaba el plan con la barra de falsación de [[1 - Diseño]]. (3) Criterio de H4 retirado: "el IC de la diferencia contiene 0" afirmaba la nula y, con ICs anchos por n = 40 y correlaciones dependientes, se "confirmaba" casi gratis → sustituido por no-inferioridad. (4) Huecos en las reglas de pilot y celdas degeneradas completados; corrección sobre `--report`: imprime evidencia y la decisión es del investigador (la redacción anterior afirmaba que implementaba una regla mecánica). La misma fecha revisa la composición del nivel 0 tras revisión bibliográfica de TSE con fuentes verificadas: `val-acc@f` pasa a titular, TSE-EMA se mantiene sin titularidad.
- **2026-06-12** — dependencia del tutor cerrada: VD1–VD3 leen val y VD4 existe; lecturas suavizadas de VD1/VD3 incorporadas a §Variables; figura de sanidad val↔test añadida a §Agregación.
- **2026-06-14** — gap de generalización confirmado por el tutor e integrado: VD5/VD6 en §Variables, familia de contrastes y doble disociación en §Contrastes, signos vs el gap en la tabla, suelo de ajuste en §Censura, reporte por celda en §Agregación. Deja de ser bloque condicional; el pilot calibra además el suelo de ajuste.
- **2026-07-17** — tabla de signos verificada contra los 6 PDFs prioritarios (§Tabla de signos y su nota; veredictos en [[2 - Decisiones]]): m-coherence vs VD1 cambia de paper base, GSNR vs VD4 baja a extrapolada, GWA vs gap baja a direccional cualitativa, GNS confirmada con condiciones.
- **2026-07-25** — §Censura desdobla la regla de divergencia en lado VD y lado predictor, tras verificar sobre un run divergente real que cuatro columnas de signo devuelven 0,0 finito en vez de NaN. §Qué falta actualizado: el pilot ya no bloquea; quedan el suelo de ajuste del gap, el texto de la decisión de instrumentación y la cobertura de columnas.
- **2026-08-01** — última pasada antes de congelar, toda ella anterior a ver ningún dato de matriz. Cuatro cambios: (1) §Censura gana la **regla de matriz incompleta**, que el plan no tenía y que era un grado de libertad grande, porque con una única GPU y ~6 días de cómputo el escenario "se acabó el tiempo con 700 de 960 runs" es realista y decidir entonces qué se analiza es exactamente lo que un preregistro existe para impedir; (2) §Qué congela gana la **política de enmiendas** y la salvedad de que congelar el texto no congela la implementación, con el dry-run sobre datos sintéticos como mecanismo que cierra las decisiones de código todavía abiertas; (3) §Unidad de análisis incorpora la **decisión de instrumentación** del 2026-07-17 con su coste medido, por ser decisión del preregistro; (4) §Qué falta queda reducida a una sola puerta, el suelo de ajuste del gap, tras cerrarse la verificación de cobertura de columnas.
- **2026-08-01 (cuarta pasada, revisión de la cadena objetivo → hipótesis → test).** Todo anterior a ver datos. Cinco cambios, cuatro de los cuales **quitan** complejidad o reparan algo roto. (1) **H3 cambia de estadístico.** Su criterio anterior contaba en cuántas celdas la métrica de mayor ΔR² pertenecía a cada familia, exigiendo 16 de 24, y estaba gravemente sesgado: alineación tiene 5 métricas y variabilidad 3, así que bajo la nula el argmax cae en alineación con probabilidad 5/8, lo que da 15 celdas esperadas por azar y una probabilidad de **0,42** de superar el umbral sin efecto alguno. Sustituido por la diferencia pareada de medianas de \|ρ\| por familia dentro de cada celda, con el mismo Wilcoxon cross-celda: insesgada respecto al tamaño de familia y sin maquinaria nueva. (2) **H2 invierte sus covariables**: la parcial primaria pasa a ser sobre `val-acc@f` sola, con la de tres baselines como sensibilidad, porque los tres son casi colineales y k = 3 daba casi el mismo ajuste con más varianza, gastando potencia justo en la hipótesis decisiva. El ΔR² se reencuadra como la comprobación específica del infra-ajuste de la parcial en presencia de empates por censura, no como segunda opinión. (3) **H6 gana su nula**: binomial exacto de concordancia de signos contra 0,5, con familia propia. Era la única hipótesis sin criterio de falsación pese a estar descrita como la prueba más exigente. (4) **H5 se restringe a las métricas que superaron H1**, igual que H4 se restringe a las que pasaron H2, y declara la no independencia de sus 12 pares. (5) §Censura gana la vigilancia de la **atenuación desigual de ρ por censura**, que ningún documento recogía: los bloques de empates comprimen el \|ρ\| alcanzable de forma proporcional a la censura, que está correlacionada con la dificultad de la celda. No se corrige el estimador; se vigila con dos comprobaciones que no requieren cálculo nuevo (restringir la etapa 2 a celdas con <25% de censura, y usar VD2/VD3, que no tienen censura, como control). (6) **La doble disociación del gap gana criterio**, que era el mismo hueco de H6: era la predicción titular de la familia de generalización y estaba escrita como afirmación, sin test. Se contrasta con la diferencia pareada \|ρ(gap)\| − \|ρ(VD2)\| por celda y el mismo Wilcoxon cross-celda, exigiendo el signo predicho en **ambos** grupos de métricas. Misma herramienta que H3, sin maquinaria nueva. Se **descartó** además añadir una validación leave-one-cell-out: su información marginal sobre el Wilcoxon cross-celda es pequeña y no compensaba la superficie añadida; la limitación de que todo el estudio es asociación dentro de muestra se declara en su lugar.
- **2026-08-01 (tercera pasada).** Se cierra la última puerta: el **suelo de ajuste del gap** queda fijado en §Censura como umbral absoluto por dataset, reutilizando los umbrales de accuracy de VD1 aplicados sobre train (MNIST 0,97; CIFAR-10 0,65; CIFAR-100 0,35; Tiny 0,20). Se documenta por qué no se calibró sobre el pilot, que es lo que decía la redacción anterior: el pilot da un solo valor por celda, al LR central y a presupuesto doblado, así que no cubre el barrido de LR que es justo lo que puebla esa distribución en la matriz. Reutilizar un número ya congelado añade cero grados de libertad; el precio es que el suelo puede quedarse corto, y esa limitación se declara. Además, pasada de legibilidad sobre todo el documento (sin cambios de contenido) y **guía rápida** al principio para lectores que no van a leer las 380 líneas.
- **2026-08-01 (segunda pasada, auditoría de potencia).** §Nota de potencia rehecha por simulación Monte Carlo (`src/power_analysis.py`), también antes de ver datos. Confirma el punto central del plan (24 celdas, mediana ρ = 0,30 → potencia 0,993 bajo BH) y corrige cuatro cosas que no estaban. (1) La nota calculaba a α = 0,05 ignorando la **multiplicidad**, cuando el criterio decide a q; se añaden los α efectivos de BH y BY. (2) El mínimo de celdas para matriz incompleta sube de 12 a **18**, ahora calculado: con 12 la potencia bajo BH al propio SESOI del plan es 0,69. Se documenta además el **suelo discreto** del Wilcoxon (p mínimo = 2^(1−n)), por el que con menos de 9 celdas el test no puede rechazar bajo BH tenga el efecto que tenga. (3) **H2 gana un brazo de equivalencia TOST** con δ_H2 = 0,15 anclado en el coste de instrumentación: sin él, el negativo que la tesis anuncia como contribución era indistinguible de la falta de potencia (con una parcial real de 0,15 la detección es del 0,46), y su resultado pasa a ser ternario. (4) H4 y H5 reciben nota de potencia propia, que no tenían: H5 queda declarado como capaz de confirmar solo invariancia casi perfecta (potencia 0,39 a concordancia 0,75) y exige los 12 pares, y H4 pasa a reportar siempre la sd(d) de la que depende. Además, el tamaño de efecto de la etapa 2 pasa a ser la **pseudomediana de Hodges-Lehmann con IC exacto** (el estimador que el Wilcoxon localiza de hecho, y la única forma de dar precisión y no solo dispersión), y §Estadístico gana una nota de reproducibilidad.

## Referencias generales del plan

*(Las de §Corrección múltiple están verificadas con enlace; las siguientes están citadas de memoria — comprobar título/año exactos antes de citarlas en la memoria.)*

- Gelman, A. & Loken, E. (2014). *The garden of forking paths.* — multiplicidad de análisis posibles; motiva el preregistro.
- Holmes, A. P. & Friston, K. J. (1998). *Generalisability, random effects and population inference.* NeuroImage. — enfoque *summary statistics* en dos etapas: estadístico por unidad, inferencia entre unidades.
- Lakens, D. (2017). *Equivalence tests: a practical primer for t-tests, correlations, and meta-analyses.* Social Psychological and Personality Science. — TOST y márgenes de equivalencia/no-inferioridad; base del criterio de H4.
- Zou, G. Y. (2007). *Toward using confidence intervals to compare correlations.* Psychological Methods. — ICs para diferencias de correlaciones dependientes (descriptivo de H4).
- Bakdash, J. Z. & Marusich, L. R. (2017). *Repeated measures correlation.* Frontiers in Psychology. — alternativa intra-celda descartada en §Alternativas.
