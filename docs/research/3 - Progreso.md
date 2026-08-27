## Estado actual (rev. 2026-08-27)

Aquí viven el estado y el plan. El *qué se decidió y por qué* está en [[2 - Decisiones]] y el *estado resultante del diseño* en [[1 - Diseño]]; este documento no los repite, los referencia. La retrospectiva del plan de abril a junio y las fases 0 a 6, todas cerradas o superseded, se retiraron el 2026-08-26 y siguen en el historial de git. El 2026-08-27 se deshizo la primera pasada de la fase A, código y texto incluidos, para rehacerla después de la fase 0 y con el estilo de [[Estilo de redacción - notas del TFM HOFT]] ya aplicado.

- **Matriz: TERMINADA.** 960 de 960 runs, las 24 celdas con sus 40 cada una, versionados en git desde el commit `f6df900` (2026-08-22), así que existen fuera de esta máquina. Coste real 121,7 h de reloj: 97,6 h de entrenamiento y 24,1 h de instrumentación, con un sobrecoste máximo por celda de 2,048x, dentro de la cota.
- **Lo único que bloquea: no hay método de análisis.** El plan preregistrado se retiró el 2026-08-25 y las seis hipótesis siguen en [[1 - Diseño]] como afirmaciones falsables **sin criterio de decisión**. Bloquea las fases B a G.
- **Sigue sin calcularse ninguna correlación entre métrica y variable dependiente.** El método de contraste se fija antes de mirar ningún resultado, y esa disciplina deja de valer en cuanto se mire una correlación antes de tenerlo escrito.
- **Código.** `src/analysis.py` reúne los diagnósticos de sanidad: validez, identidades, degeneración, tendencia y redundancia. **No hay ninguna figura, ningún módulo de estilo de figuras y ni una línea de análisis confirmatorio**: el 2026-08-27 se retiró todo el estilo que había, así que la fase 0 parte de cero en ese punto y con las notas de HOFT como referencia.
- **Qué hay realmente registrado por época.** `trajectory.parquet` trae **23 columnas de métrica de gradiente** (no 8: cada métrica emite varias claves) más **4 columnas de TSE**, más la curva cruda `train_loss`/`val_loss`/`val_acc` y los dos relojes. `metrics_at_window.parquet` son esas mismas columnas en 5 filas por run, una por ventana. Consecuencia operativa: antes de contrastar hay que fijar **una columna titular por métrica** (`analysis.py::headline_columns()` existe para eso), o la multiplicidad se dispara sin que ningún objetivo lo pida.
- **Memoria.** Terminados introducción, estado del arte y fundamentos. De metodología viven §Configuración del entrenamiento, §Calibración y, dentro de §Protocolo de análisis, la población de análisis y el tratamiento del censurado, más §Amenazas a la validez, cuyo nombre y contenido revisa la fase 0. Siguen vacías §Hipótesis, §Variables, §Ventana temporal, §Matriz y §Protocolo de evaluación, y esta última **ya la referencia otro capítulo**. Resultados, implementación y conclusiones son títulos sin prosa. **Cero figuras.** Faltan los tres resúmenes de `main.tex`, los agradecimientos y los dos anexos.
- **Promesas colgantes en la memoria.** Las de falsación (`introduccion.tex:54` y `:68`, `estado-del-arte.tex:111`, `fundamentos.tex:161` y `:250`) las cumple la fase B y no son errores, son anuncios. La que sí puede no cumplirse es la del **frente de Pareto**, prometida en `estado-del-arte.tex:111` y con sección propia en `resultados.tex:137`; ver §Problemas conocidos.
- **Pilot:** ejecutado y leído; presupuestos y umbrales en `config.py::DATASET_BUDGET` y en los 24 YAML. Aviso: `reports_pilot/` está en `.gitignore`, así que la evidencia que justifica esos números existe **solo en el disco local**.
- **Lista de métricas:** cerrada con la implementación. Variabilidad (normalized variance, GNS simple, GSNR) y alineación (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline.

## Problemas conocidos de los objetivos (rev. 2026-08-26)

Ninguno obliga a cambiar los seis objetivos de `introduccion.tex`. Se resuelven declarando alcance, y hay que resolverlos **antes** de escribir el objetivo al que afectan.

- **OE2 y la velocidad: el baseline lee la misma curva que define la variable.** VD1 es "en qué época la accuracy de validación cruza τ", y la segunda mitad del predictor de referencia es "la accuracy de validación en la ventana f". Son la misma curva leída en dos puntos, así que descontar el baseline para predecir VD1 es descontar un prefijo del propio resultado. Hay además un solape entre la ventana de medida y el propio cruce del umbral, que **la fase A tiene que cuantificar** antes de que OE2 y OE4 se puedan escribir: si en una ventana ya ha cruzado buena parte de los runs, esa ventana no anticipa la velocidad, la describe. El contraste de OE2 será por tanto informativo sobre todo en las variables que la curva de validación no define, es decir, la accuracy de test y los dos gaps.
- **OE3 y el solape entre familias.** `noise_scale/simple` (variabilidad) y `mcoh/global` (alineación) son la misma cantidad reparametrizada: `GNS = M/α − 1` exacto (Ecuación `eq:gns-mcoh` de `fundamentos.tex`), y el Spearman intra-run medido sobre los 960 sale **−1,000**. Además GSNR es el recíproco por parámetro de NGV. Comparar familia contra familia sin podar antes cuenta el mismo número una vez en cada bando. **La poda con prueba deja de ser mejora de presentación y pasa a ser requisito previo de OE3.**
- **El coste por métrica no está medido y no es separable.** `summary.json` trae un único `metric_seconds` por run, y `metrics_runner.measure` construye **un solo** barrido per-sample que consumen seis de las ocho métricas, así que su coste marginal individual no existe ni a posteriori. El objetivo general promete "la mejor capacidad predictiva por unidad de coste" y hay un frente de Pareto prometido en `estado-del-arte.tex:111` con sección propia en `resultados.tex:137`. Con los 960 runs **no es computable**. Dos salidas honestas: apoyarse en la columna de requisito de cálculo que ya publica `tab:sota-metricas`, o medir el coste por métrica con un micro-benchmark aparte, minutos y fuera de la matriz. En cualquier caso, retirar §Frente de Pareto como sección propia y doblarla dentro de OE2, que ya dice "a un coste justificado". **Sin decidir.**

## Plan por objetivos (rev. 2026-08-26; objetivo: primera semana de septiembre de 2026)

**Metodología de trabajo, fijada el 2026-08-27: una cosa cada vez, y en tres carriles a la vez.** Son dos mitades que se sostienen la una a la otra, y gobiernan todo lo que queda del proyecto.

*Una cosa cada vez.* En cada momento hay un solo tema abierto, y no se abre otro hasta cerrarlo. Lo que aparezca por el camino se anota donde corresponda y se sigue con lo que había. Dispersarse deja tres cosas al setenta por ciento, y tres cosas al setenta por ciento valen cero.

*Tres carriles a la vez.* Ese tema único avanza en los tres sitios en la misma tanda: el **código** de `src/` y `tests/`, la **documentación** del vault, y la **memoria** en `thesis/`. No se adelanta el código para escribir el texto más tarde, porque el texto que se deja para más tarde no se escribe. Si un paso no toca alguno de los tres carriles, se dice por qué en vez de dejarlo en silencio.

**La regla de fase, y no se negocia: una fase, un objetivo, un entregable completo.** Una fase se cierra cuando su código está escrito y probado, sus números calculados sobre `reports/`, su figura hecha, y su texto redactado en el `.tex` correspondiente. Nada pasa a la fase siguiente con algo a medias, porque lo que queda a medias es lo que no se hace.

**Segunda regla, del 2026-08-26:** toda cuenta que se ejecute tiene que poder responder de qué objetivo es y qué se haría distinto si saliera al revés. Si no hay respuesta, no se ejecuta. Y el contraste no puede ser solo visual: cada objetivo lleva su prueba de hipótesis, que hay que reconstruir porque la anterior se retiró.

**Regla de figuras, del 2026-08-26:** una figura, una afirmación, con la evidencia numérica visible. Si no se puede resumir en una frase que empiece por "esta figura demuestra que", está mostrando datos y no un resultado, y su sitio es un anexo o el repositorio.

**Esto es estructura, no compromiso.** Lo único fijo son las reglas de arriba. Todo lo que sigue es un esqueleto para no perderse, y **cada fase puede cambiar**: su método, su contenido, sus entregables, su orden, y una fase puede partirse en dos o desaparecer. Cada fase se abre decidiendo qué se hace en ella y registrando esa decisión en [[2 - Decisiones]] antes de programar nada. Si al abrir una fase resulta que las siguientes ya no tienen sentido tal como están escritas aquí, se reescriben aquí y se sigue: este documento va detrás del trabajo, no delante.

### Fase 0: poner el LaTeX al día (paralela a la fase A)

No depende de ningún resultado, así que corre en paralelo. Recoge en la memoria todo lo que hoy ya se sabe, para que el análisis se escriba sobre un documento al día y no sobre uno a medias.

- [x] **Plantilla de capítulo de HOFT.** Hecha el 2026-08-27. **Norma:** la plantilla rige en los capítulos de cuerpo, del 2 al 6, y no en la Introducción ni en las Conclusiones, que ya cierran con «Estructura de la memoria» y con la memoria misma; HOFT tampoco las incluye. Ese papel lo hace «Posicionamiento de este trabajo» en Estado del arte, y Fundamentos tiene sección propia; faltan las de Metodología, Implementación y Resultados, que se escriben junto al cuerpo de su capítulo y nunca antes, para no dejar un título sin prosa. La instrucción queda como comentario en esos tres ficheros. Los tres capítulos escritos ya abrían con su hoja de ruta, así que el trabajo real fue poner al día la de la Introducción, que nombraba §Amenazas a la validez tras el renombrado a §Riesgos, llamaba «rejilla» a la matriz de experimentos y se saltaba dos secciones de nueve.
- [ ] **Definir el estilo de figuras desde cero**, con las notas de HOFT §Estilo de las figuras como referencia. No hay nada heredado: los dos intentos anteriores se retiraron.
- [x] **Unificar la terminología en inglés.** Hecho el 2026-08-27, decisión y detalle en [[2 - Decisiones]]. A inglés *epoch*, *learning rate* y *seed*; "conjunto de medición" pasa a "batch de medición" y queda definido en `fundamentos.tex` antes de usarse; todo anglicismo va en cursiva **siempre**, que es la norma de la escuela y deroga la convención anterior de cursiva solo en la primera aparición.
- [ ] Secciones de metodología que no dependen del análisis: variables, ventana temporal, matriz de experimentos y protocolo de evaluación. **§Matriz de experimentos escrita el 2026-08-27**: la celda como unidad de análisis, por qué el presupuesto va a los *learning rates* y no a las *seeds*, rejilla por optimizador y no por arquitectura, extremos divergentes como recorrido deliberado del eje de eficiencia, y perillas congeladas; con la tabla «Ejes de la matriz de experimentos». Cifras verificadas contra `src/config.py`, que es la fuente única de la rejilla. Quedan variables, ventana temporal y protocolo de evaluación.
- [ ] Capítulo de implementación, hoy seis títulos sin prosa.
- [ ] Revisar §Protocolo de análisis y §Riesgos: qué se queda, cómo se llama y qué se va al capítulo de resultados por estar duplicado.
- [ ] Cerrar las promesas colgantes que ya se pueden cerrar sin tener el método de análisis.
- [ ] **Criterio de cierre:** ninguna sección con título y sin prosa salvo las que dependen de resultados, y ninguna referencia cruzada que apunte a una sección vacía. **Medido el 2026-08-27: 13 referencias apuntan hoy a algo vacío.** Eran 14; §Matriz de experimentos cerró dos y añadió una nueva a §Protocolo de análisis, a propósito, porque es la remisión que impide escribir dos veces el procedimiento de correlación. Cinco apuntan a los capítulos de Implementación, Resultados y Conclusiones desde la Introducción; las ocho restantes, a secciones de Metodología. Ojo: LaTeX no detecta ninguna de las dos cosas, porque un `\ref` resuelve igual aunque la sección esté vacía, así que «cero referencias indefinidas» no vale como comprobación y hay que contarlas aparte.

### Fase A: datos y validez

No ataca ningún objetivo: establece que los datos sirven y qué se puede calcular con ellos. **Regla de ejecución: un lado cada vez**, predictores y variables dependientes por separado.

- [ ] Validez de las columnas registradas sobre los 960 runs: rango teórico, identidades exactas y columnas ausentes, valores nulos o imposibles, runs que se quedaron sin aprender estancadas, etc.
- [ ] Mapa de lo computable celda a celda: qué runs entran en el análisis y cuántos quedan para cada variable dependiente.
- [ ] Rango dinámico del predictor: si la métrica se mueve más por el learning rate que por la semilla.
- [ ] Solape entre la ventana de medida y el cruce del umbral, que decide en qué ventanas se puede analizar la velocidad.
- [ ] Concordancia entre validación y test, que decide si las variables leídas sobre la curva de validación sirven.
- [ ] **Criterio de cierre:** veredicto de validez escrito, mapa tabulado, y las decisiones que se deriven reflejadas en el `.tex`.

### Fase B: OE1, existencia

- [ ] Decidir el método y registrarlo. Programarlo con pruebas sobre datos sintéticos de efecto conocido, antes de tocar `reports/`.
- [ ] Aquí se reconstruye la **prueba de hipótesis**, una sola vez y para todos los objetivos: qué se calcula dentro de cada celda, cómo se agregan las 24, cómo entra el censurado y cómo se corrige la multiplicidad.
- [ ] **Texto:** §Hipótesis (H1) de `metodologia.tex`, hoy vacía, y la parte de contraste de §Protocolo de análisis, más su sección de resultados.
- [ ] **Criterio de cierre:** H1 respondida con su número, su figura y su párrafo.

### Fase C: OE2, valor incremental (el decisivo)

- [ ] Método para descontar el predictor de referencia (TSE y validación temprana), con la asimetría de §Problemas conocidos declarada por escrito.
- [ ] Aquí se resuelve también el eje de coste, doblado dentro de este objetivo en vez de como frente de Pareto propio.
- [ ] **Criterio de cierre:** H2 respondida sobre las seis VD, con la limitación de la velocidad declarada, no omitida.

### Fase D: OE3, comparación de familias

- [ ] **Requisito previo:** poda con prueba del par `noise_scale/simple` ≡ `mcoh/global`, y del solape NGV/GSNR. Sin poda, este objetivo no es contestable.
- [ ] **Criterio de cierre:** H3 respondida sobre la lista podada, y la poda documentada como resultado propio.

### Fase E: OE4, suficiencia temprana

- [ ] Barrido de ventanas 5/10/25/50 %, teniendo en cuenta el solape medido en A4.
- [ ] **Criterio de cierre:** H4 respondida, con la ventana mínima defendible dicha explícitamente.

### Fase F: OE5, robustez entre optimizadores

- [ ] Comparación pareada sobre los 12 pares de celdas que solo difieren en el optimizador.
- [ ] **Criterio de cierre:** H5 respondida, declarando que 12 pares dan poca potencia y que un no rechazo no prueba invariancia.

### Fase G: OE6, concordancia con la literatura

- [ ] Contraste del signo observado contra el predicho por cada paper. La tabla de signos ya está verificada contra los PDFs (2026-07-17, [[Datos experimentales]] §5.3) y la salvedad de signo de GWA está en `fundamentos.tex:159`.
- [ ] **Criterio de cierre:** H6 respondida, separando los signos que el paper afirma de los que esta memoria extrapola.

### Fase H: cierre de la memoria

- [ ] **Conclusiones**, una por objetivo.
- [ ] **Cerrar las promesas colgantes** listadas en §Estado actual, incluida la decisión sobre el frente de Pareto.
- [ ] **Declarar el encuadre del análisis:** posterior a los datos y anterior a los resultados, con las fechas que lo sostienen.
- [ ] **Notas de honestidad:** Tiny-ImageNet usa su val público como test, F1-macro ≈ accuracy en datasets balanceados, y el split es fijo y compartido por todos los runs.

### Fase I: obligatorio para el depósito (independiente, en cualquier hueco)

No depende de ningún resultado, así que puede adelantarse desde ya. Lo que es escritura de capítulos se ha movido a la fase 0; aquí queda solo lo administrativo y lo que solo se puede hacer al final.

- [ ] **Los tres resúmenes** de `main.tex`, los agradecimientos, y los **dos anexos** (ODS, obligatorio, 1-2 páginas; configuración y reproducibilidad).
- [ ] Formato UPV/ETSINF, Turnitin y entrega por EBRON.
- [ ] Borrador al tutor → incorporar feedback.
- [ ] Slides de defensa (10-15, ~15 min).

### Higiene, barato y en cualquier hueco

- [ ] **Copia de seguridad de `reports_pilot/`.** Está en `.gitignore` y solo existe en el disco local: si se pierde, se pierde la evidencia que justifica los presupuestos y umbrales congelados.
- [ ] **Etiquetar en git la versión del código que produjo los 960 runs.** Hoy no hay ninguna etiqueta que los una, así que reproducir la matriz obliga a buscar el commit a mano.
