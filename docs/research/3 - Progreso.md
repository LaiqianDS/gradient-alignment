## Estado actual (rev. 2026-08-29)

Aquí viven el estado y el plan. El *qué se decidió y por qué* está en [[2 - Decisiones]] y el *estado resultante del diseño* en [[1 - Diseño]]; este documento no los repite, los referencia. Tampoco guarda el camino recorrido, que ya está en el historial de git, solo dónde está el trabajo hoy. **La fase 0 quedó cerrada el 2026-08-28** y lo siguiente es la fase A.

- **Matriz: TERMINADA.** 960 de 960 runs, las 24 celdas con sus 40 cada una, versionados en git desde el commit `f6df900` (2026-08-22), así que existen fuera de esta máquina. Coste real 121,7 h de reloj, 97,6 h de entrenamiento y 24,1 h de instrumentación.
- **Siguiente paso, acordado con Lai el 2026-09-02: revisar la memoria párrafo por párrafo antes de abrir la fase B.** Un capítulo por turno, en este orden: Resultados, Metodología, Fundamentos, Estado del arte, Introducción, Desarrollo y Trabajos futuros. En cada párrafo se comprueban cuatro cosas: que no deje una promesa colgando, que todo término esté explicado antes de usarse, que cada número coincida con lo que sale de `reports/`, y los tics de estilo de [[Estilo de redacción - notas del TFM HOFT]]. Al cerrar cada capítulo se dice qué cambió y qué no se puede cerrar porque depende de un resultado que no existe. **Lai ha pedido no abrir la fase B ni las C a G hasta nuevo aviso.** La decisión sobre las nubes de puntos sigue pendiente y se toma con él. **Resultados quedó revisado el 2026-09-03** (siete cifras corregidas sobre `reports/`, dos incoherencias de §Rango dinámico, seis términos explicados). Quedan de ese capítulo: §Calibración del *pilot* vacía, escribible ya sin la fase B; siete títulos de la fase B sin prosa, y Lai decide si se comentan; hoja de ruta y conclusiones del capítulo, que esperan al cuerpo. El siguiente capítulo es Metodología.
- **Lo único que bloquea: no hay método de análisis.** Las seis hipótesis siguen en [[1 - Diseño]] como afirmaciones falsables **sin criterio de decisión**. Bloquea las fases B a G. Como el plan preregistrado se retiró, el análisis que se haga es **posterior a los datos**, y así hay que presentarlo en la memoria.
- **Sigue sin calcularse ninguna correlación entre métrica y variable dependiente.** El método de contraste se fija antes de mirar ningún resultado, y esa disciplina deja de valer en cuanto se mire una correlación antes de tenerlo escrito.
- **Código.** `src/analysis.py` es el backend de sanidad de las columnas y tiene el rango dinámico del punto 3; `src/efficiency.py` tiene por unidad el run y la celda: censo de salud, mapa de lo computable, solape del punto 4 (regla en `AHEAD_FLOOR`) y concordancia del punto 5 (banda en `NOISE_SIGMAS`). Los tamaños de las particiones viven en `config.py::VAL_SIZE` y `TEST_SIZE`. Las figuras viven en `src/figures.py`, una función por figura, **seis en total** (`ventana-lr`, `rango-celda`, `rango-columnas`, `solape-celda`, `solape-mapa`, `val-test`), con estilo de artículo desde el 2026-09-02 (Okabe-Ito, TeX Gyre Heros). **Sigue sin haber una línea de análisis confirmatorio.** Suite en **274** pruebas verdes.
- **Runs con NaN.** 154 nunca aprendieron (115 por colapso, 39 por divergencia) y 11 se recuperaron; los recuentos completos están en [[2 - Decisiones]] (2026-08-31). El cero falso de `stiffness/sign_*` y `confusion/frac_neg` está parcheado en `reports/` desde el 2026-08-29, **así que se comporta como cualquier valor ausente y ninguna fase posterior necesita una excepción**.
- **Qué hay realmente registrado por época.** `trajectory.parquet` trae **23 columnas de métrica de gradiente** (no 8: cada métrica emite varias claves) más **cuatro columnas de TSE**, más la curva cruda `train_loss`/`val_loss`/`val_acc` y los dos relojes. `metrics_at_window.parquet` son esas mismas columnas en cinco filas por run, una por ventana.
- **Memoria.** Escritos enteros Introducción, Estado del arte, Fundamentos, Desarrollo y Trabajos futuros. De Metodología **solo queda vacía §Protocolo de análisis**, que escribe la fase B; §Variables tiene desde el 2026-09-03 la Tabla «Columna titular de cada predictor», que fija la columna que representa a cada métrica. De Resultados están escritas §Qué variables quedan medidas, §Rango dinámico del predictor, §Solape entre la ventana y el desenlace y §Concordancia entre validación y test, las cuatro de la fase A; las otras ocho secciones y Conclusiones siguen siendo títulos sin prosa, y la hoja de ruta y las conclusiones del capítulo se escriben con el cuerpo. §Hipótesis da las seis afirmaciones y lo que refutaría a cada una, sin criterio numérico y sin ninguna palabra que sugiera preinscripción, porque el método es posterior a los datos. **Seis figuras**, las 6.1 a 6.6; sigue faltando la única del capítulo de Desarrollo, el diagrama del *pipeline*, anotada como comentario en su fichero. Faltan los tres resúmenes de `main.tex`, los agradecimientos y los dos anexos. Compila en 67 páginas, sin referencias sin resolver ni cajas desbordadas.
- **Qué hay escrito del censurado, con precisión.** `metodologia.tex` §Variables declara la **convención de registro**, que un valor censurado se anota como ausente y nunca como el presupuesto, y remite su tratamiento en el contraste a §Protocolo de análisis. La **población de análisis**, es decir qué entrenamientos entran, no está escrita en ningún sitio. Las dos cosas son trabajo de la fase B.
- **Promesas colgantes en la memoria: diez.** Seis apuntan a §Protocolo de análisis, una a `ch:resultados`, dos a `ch:conclusiones` y una a `sec:determinismo` del anexo de configuración. Ninguna es ya trabajo de fase 0. Las de falsación (`introduccion.tex:54` y `:68`, `estado-del-arte.tex:111`, `fundamentos.tex:170` y `:259`) las cumple la fase B y no son errores, son anuncios. **LaTeX no detecta ninguna**, porque un `\ref` resuelve igual aunque la sección esté vacía, así que «cero referencias indefinidas» no vale como comprobación y hay que contarlas aparte, comparando cada `\label` con la prosa que le sigue. Lección del recuento: escribir una sección sube tantas como baja, porque cada remisión honesta abre una colgante nueva.
- **Pilot:** ejecutado y leído; los presupuestos que salieron de él están en `config.py::DATASET_BUDGET` y en los 24 YAML. Los umbrales ya no, desde el 2026-09-01, y viven en `config.py::THRESHOLD_ACC` calculados sobre la matriz. Desde el 2026-08-29 `reports_pilot/` está versionado, con los seis `testfix_40ep/` de Tiny dentro, así que la evidencia que justifica esos números ya existe fuera de esta máquina.

## Vigentes: a tener en cuenta en todo lo que queda de análisis (rev. 2026-08-31)

Lista de comprobación para abrir cualquier sesión de análisis. Entra aquí lo que **condiciona un cálculo futuro** y se olvidaría con facilidad: trampas medidas, equivalencias entre columnas, y decisiones abiertas que cambian un número. No entra el estado del trabajo, que está arriba, ni el porqué de cada cosa, que está en [[2 - Decisiones]]. Una línea por asunto y un puntero. Cuando algo deja de condicionar, se borra de aquí.

- **El método es posterior a los datos.** La matriz acabó el 2026-08-22 y el plan se retiró el 2026-08-25. Ninguna frase, en el vault o en la memoria, puede sugerir preinscripción.
- **El umbral de VD1 es por dataset y arquitectura desde el 2026-09-01** ([[2 - Decisiones]]), y sale de una norma reproducible: el valor redondo más alto que alcanza al menos el 60 % de los entrenamientos que aprendieron. Si se recalcula, se recalcula con la norma y no a mano. Y **VD1 se recalcula siempre en la capa de análisis** desde `trajectory.parquet`: el campo `epochs_to_threshold` de los 960 `summary.json` está calculado con los umbrales viejos y **no se puede leer**.
- **El learning rate mueve el predictor y también la variable a predecir**, medido el 2026-09-02. Un reparto de 0,90 dice que nueve décimas de esa columna son el learning rate leído de otra manera, así que la correlación de la fase B será en buena parte una correlación entre dos efectos de una causa común. Declarado en `resultados.tex` §Rango dinámico del predictor, y hay que volver a decirlo al interpretar cada contraste.
- **El predictor de referencia gana en rango dinámico a todas las métricas caras**, TSE 0,97 y la curva de validación 0,95 frente a 0,92 la mejor instrumentada. Es un mal presagio para H2 y hay que tenerlo delante al escribir la fase C.
- **VD1 y el predictor de referencia leen la misma curva.** Descontar la validación temprana para predecir VD1 es descontar un prefijo del propio resultado. OE2 será informativo sobre todo en VD4, VD5 y VD6.
- **El solape está medido por celda desde el 2026-09-02, en pares, y acota OE4 sobre la velocidad.** VD1 sirve en la ventana del 5 % en 22 celdas y en la del 10 % en 12, y en ninguna al 25 % ni al 50 %. VD3 da 23, 17, 9 y 2 celdas, y donde falla es por sobreajuste, no por velocidad. VD2 sirve hasta el 25 % en las 24 y al 50 % cae sobre la línea por construcción, «descrita a medias». La regla es la mitad por delante (`efficiency.py::AHEAD_FLOOR`) y **la cuenta es un suelo**, porque no haber cruzado no es estar por decidir. Qué hace la fase B con los runs ya cruzados dentro de una ventana que sirve **sigue sin decidir**. Detalle en [[2 - Decisiones]].
- **GWA va al revés que su artículo.** Aquí se mide sobre $\nabla\ell$ bruto, así que la predicción heredada es que una GWA **baja** acompañe a mejor generalización. Conversión en `fundamentos.tex:168`, con el mismo aviso en el enunciado de H6 de [[1 - Diseño]] y en la tabla de signos de [[Datos experimentales]] §5.3, que es el material contra el que se contrasta H6.
- **En los 41 runs divergidos, `final_test_acc` y `final_gap_acc` traen número y no son mediciones**, porque `argmax` sobre logits NaN devuelve siempre el índice 0. Son las dos únicas columnas donde "no falta ningún valor" engaña. Los 41 son todos de SGD: Adam no divergió nunca.
- **`noise_scale/simple` y `mcoh/global` son la misma cantidad** (GNS = M/α − 1 exacto, Spearman intra-run −1,000), y GSNR es el recíproco por parámetro de NGV. Sin poda previa, OE3 cuenta el mismo número en los dos bandos.
- **Cada métrica entra por su columna titular**, fijada el 2026-09-03 en la tabla de `metodologia.tex` §Variables y en `analysis.py::headline_columns()`. Las demás columnas de las 27 no se contrastan, o solo como exploratorio.
- **154 runs nunca aprendieron y los 154 tienen causa conocida**, 115 de colapso y 39 de divergencia. Se identifican por `learned` de `efficiency.py::run_health`: mejor val-accuracy suavizada por debajo de **1,25 veces el azar** (`CHANCE_MARGIN`); 1,2 y 1,25 seleccionan los mismos 154. **Distinguir dos usos, que no son el mismo:** se excluyen del denominador que **calibra** el umbral de VD1 (norma del 2026-09-01), porque no cruzarían ninguno; que entren o no en cada **contraste** sigue **sin decidir** y es de la fase B.
- **La censura de VD1 se cuenta en pares, no en runs.** Un run censurado sigue siendo comparable con cualquiera que cruzó. 15 cruces de 40 conservan el 61,5 % de los pares, no el 37,5 %. Con n = 40 en las 24 celdas, esa fracción es una función estrictamente creciente del número de cruces, así que da la escala honesta y **no una evidencia aparte**, porque los dos números dicen lo mismo.
- **Los 24 YAML de `experiments/` llevan el umbral viejo por conjunto de datos.** `run_matrix.py --init` no pisa ficheros existentes, así que siguen ahí desde que se corrió la matriz. No afecta a ningún cálculo, porque el análisis lee `config.py::THRESHOLD_ACC` y nunca los YAML, pero engaña a quien los abra.
- **El suelo de ajuste del gap sigue sin valor fijado**, el mínimo de `final_train_eval_acc` para los contrastes de VD5 y VD6 (aplazado a propósito el 2026-07-17). Es un filtro de análisis, así que fijarlo no obliga a re-correr nada.

## Problemas conocidos de los objetivos (rev. 2026-08-26)

Ninguno obliga a cambiar los seis objetivos de `introduccion.tex`. Se resuelven declarando alcance, y hay que resolverlos **antes** de escribir el objetivo al que afectan.

- **OE2 y la velocidad.** El baseline lee la misma curva que define la variable. VD1 es "en qué época la accuracy de validación cruza τ", y la segunda mitad del predictor de referencia es "la accuracy de validación en la ventana f". Son la misma curva leída en dos puntos, así que descontar el baseline para predecir VD1 es descontar un prefijo del propio resultado. Con el solape medido el 2026-09-02 pasa lo mismo, porque si en una ventana ya ha cruzado buena parte de los runs, esa ventana no anticipa la velocidad, la describe, y en un run ya cruzado la validación leída en la ventana vale ya el umbral. Está declarado en `resultados.tex` §Solape, y hay que repetirlo al redactar OE2 y OE4.
- **OE3 y el solape entre familias.** Comparar familia contra familia sin podar antes cuenta el mismo número una vez en cada bando, porque la escala de ruido y la m-coherencia son la misma cantidad reparametrizada y GSNR es el recíproco por parámetro de NGV. **La poda con prueba deja de ser mejora de presentación y pasa a ser requisito previo de OE3.**

## Plan por objetivos (rev. 2026-08-26; objetivo: primera semana de septiembre de 2026)

**Metodología de trabajo, fijada el 2026-08-27: una cosa cada vez, y en tres carriles a la vez.** Son dos mitades que se sostienen la una a la otra, y gobiernan todo lo que queda del proyecto.

*Una cosa cada vez.* En cada momento hay un solo tema abierto, y no se abre otro hasta cerrarlo. Lo que aparezca por el camino se anota donde corresponda y se sigue con lo que había. Dispersarse deja tres cosas al setenta por ciento, y tres cosas al setenta por ciento valen cero.

*Tres carriles a la vez.* Ese tema único avanza en los tres sitios en la misma tanda: el **código** de `src/` y `tests/`, la **documentación** del vault, y la **memoria** en `thesis/`. No se adelanta el código para escribir el texto más tarde, porque el texto que se deja para más tarde no se escribe. Si un paso no toca alguno de los tres carriles, se dice por qué en vez de dejarlo en silencio.

**La regla de fase, y no se negocia: una fase, un objetivo, un entregable completo.** Una fase se cierra cuando su código está escrito y probado, sus números calculados sobre `reports/`, su figura hecha, y su texto redactado en el `.tex` correspondiente. Nada pasa a la fase siguiente con algo a medias, porque lo que queda a medias es lo que no se hace.

**Segunda regla, del 2026-08-26:** toda cuenta que se ejecute tiene que poder responder de qué objetivo es y qué se haría distinto si saliera al revés. Si no hay respuesta, no se ejecuta. Y el contraste no puede ser solo visual: cada objetivo lleva su prueba de hipótesis, que hay que reconstruir porque la anterior se retiró.

**Regla de figuras, del 2026-08-26:** una figura, una afirmación, con la evidencia numérica visible. Si no se puede resumir en una frase que empiece por "esta figura demuestra que", está mostrando datos y no un resultado, y su sitio es un anexo o el repositorio.

**Regla de capítulo, del 2026-08-27:** la plantilla de [[Estilo de redacción - notas del TFM HOFT]] rige en los capítulos de cuerpo, del 2 al 6, y no en la Introducción, las Conclusiones ni Trabajos futuros, que cierran solos. La hoja de ruta y las conclusiones de un capítulo se escriben junto a su cuerpo y **nunca antes**, para no dejar un título sin prosa. Solo falta la de Resultados, y la instrucción está como comentario en ese fichero.

**Regla de registro, del 2026-08-29:** el vault se poda mientras se trabaja, no al final. Una entrada de [[2 - Decisiones]] guarda cuatro cosas y ninguna más: qué se decidió, por qué en la forma más corta que aún lo defienda ante un tribunal, la evidencia medida con su número, y las trampas que le costarían tiempo a quien lo toque después. Se va todo lo demás: el camino hasta la decisión, las alternativas que ya no informan nada, el detalle de la verificación, y lo que el código o la memoria ya cuentan mejor. **Lo que no se poda nunca son las decisiones y sus fechas**, porque son la evidencia del encuadre que la fase H tiene que declarar, que el análisis es posterior a los datos y anterior a los resultados.

**Esto es estructura, no compromiso.** Lo único fijo son las reglas de arriba. Todo lo que sigue es un esqueleto para no perderse, y **cada fase puede cambiar**: su método, su contenido, sus entregables, su orden, y una fase puede partirse en dos o desaparecer. Cada fase se abre decidiendo qué se hace en ella y registrando esa decisión en [[2 - Decisiones]] antes de programar nada. Si al abrir una fase resulta que las siguientes ya no tienen sentido tal como están escritas aquí, se reescriben aquí y se sigue: este documento va detrás del trabajo, no delante.

### Fase 0: poner el LaTeX al día

**Cerrada el 2026-08-28.** Puso la memoria al día con todo lo que ya se sabía, sin depender de ningún resultado: plantilla de capítulo, estilo de figuras, terminología en inglés, los capítulos de Metodología, Desarrollo y Trabajos futuros, y la revisión de Lai sobre el texto. Cada decisión está en [[2 - Decisiones]], y lo que sigue vivo de esta fase está arriba, en §Estado actual y en las reglas.

### Fase A: datos y validez

No ataca ningún objetivo. Establece que los datos sirven y qué se puede calcular con ellos. **Regla de ejecución: un lado cada vez**, predictores y variables dependientes por separado.

Plan de ejecución, decidido el 2026-08-31: `src/analysis.py` para las columnas y `src/efficiency.py` para runs y celdas; orden 1 y 2, luego 3 (lado predictor) y 4 y 5 (lado variable dependiente), porque el 3 y el 4 descartan dos de las tres causas de un coeficiente nulo antes de mirar ninguna correlación. El método de cada punto se decidió al llegar a él y está en [[2 - Decisiones]] con su fecha; la evidencia completa y sus lecturas están en `resultados.tex`.

- [x] **Puntos 1 y 2 (del 2026-08-29 al 2026-09-01):** columnas válidas sobre los 960 y un cero falso parcheado en `reports/`; censo de runs y mapa de lo computable; umbral por conjunto de datos y arquitectura, y figura `ventana-lr`. Escrito en `resultados.tex` §Qué variables quedan medidas.
- [x] **Lección del carril de LaTeX (2026-09-01, repetida el 2026-09-03):** los números del vault se vuelven a medir sobre `reports/` antes de entrar en la memoria, no se copian. Dos salieron mal el día 1 y uno más el día 3.
- [x] **Punto 3, rango dinámico (2026-09-02):** la causa dos de un coeficiente nulo queda descartada en casi toda la rejilla. §Rango dinámico del predictor, figuras `rango-celda` y `rango-columnas`.
- [x] **Punto 4, solape (2026-09-02):** acotada, no descartada; para la velocidad solo las ventanas del 5 y el 10 % anticipan algo. §Solape entre la ventana y el desenlace, figuras `solape-celda` y `solape-mapa`.
- [x] **Punto 5, concordancia validación-test (2026-09-02):** las variables leídas sobre validación no son un artefacto de la partición. §Concordancia entre validación y test, figura `val-test`.
- [ ] **Decisión pendiente antes de la fase B, y con fecha.** Comprobar que las relaciones son monótonas, que es lo que sostiene la elección de tau ([[2 - Decisiones]] §Pendientes), exige mirar nubes de puntos de métrica contra variable dependiente, justo lo que se evita hasta tener el método escrito. Opciones: mirarlo abierto y registrarlo, mirarlo sobre una porción declarada de antemano, o fijar el estadístico a ciegas. Propuesta del 2026-09-02, sin decidir: una porción declarada, la ventana del 5 %, las ocho columnas titulares, la variable principal de cada constructo y las dos celdas de ejemplo de la memoria, mirando solo la forma y sin anotar ningún coeficiente. Lai la aplaza hasta después de la revisión de la memoria.
- [ ] **Criterio de cierre:** veredicto de validez escrito, mapa tabulado, y las decisiones que se deriven reflejadas en el `.tex`. Los cinco puntos están cerrados y escritos; falta la decisión sobre las nubes de puntos, y con ella se cierra la fase.

**Las fases B a G son una máquina y seis consultas (visto el 2026-08-31).** No son seis trabajos. La fase B construye el contraste entero y lo estrena con OE1; las otras cinco añaden una idea cada una encima de la misma maquinaria, y OE6 no calcula nada nuevo, solo lee signos. La forma propuesta de esa máquina, las cinco elecciones que se toman una vez, está en [[2 - Decisiones]] §Pendientes, sin decidir. **Reordenación que sale de ahí:** la poda de métricas redundantes se adelanta de la fase D a antes de la fase B, porque decide qué columnas entran como predictor en *todos* los objetivos y no solo en OE3.

Entrada común a todas: los predictores de `metrics_at_window.parquet` (5 filas por run, una por ventana), las seis VD de `summary.json`, y el mapa de la fase A para la n de cada casilla.

### Fase B: OE1, existencia

- [ ] **Requisito previo, adelantado desde la fase D:** poda con prueba del par `noise_scale/simple` ≡ `mcoh/global` y del solape NGV/GSNR. Si se confirma al agregar, la familia de variabilidad podría quedarse en **una sola cantidad**, y una de sus tres métricas resulta ser de alineación. Eso es resultado propio y a la vez amenaza para OE3.
- [ ] Decidir el método y registrarlo. Programarlo con pruebas sobre datos sintéticos de efecto conocido, antes de tocar `reports/`: un pipeline de correlación no validado contra un efecto que sabes que existe no distingue "no hay señal" de "tengo un bug".
- [ ] Aquí se reconstruye la **prueba de hipótesis**, una sola vez y para todos los objetivos: qué se calcula dentro de cada celda, cómo se agregan las 24, cómo entra el censurado y cómo se corrige la multiplicidad.
- [ ] **Trampa:** un coeficiente nulo tiene tres causas que desde fuera no se distinguen, que la métrica no prediga, que no varíe dentro de la celda más allá del ruido de semilla, o que el resultado ya hubiera ocurrido al medir. Las descartan los puntos 3 y 4 de la fase A, y por eso van antes.
- [ ] **Texto:** §Hipótesis (H1) de `metodologia.tex`, hoy vacía, y la parte de contraste de §Protocolo de análisis, más su sección de resultados. Ahí entran también la población de análisis y el tratamiento del censurado, que no están escritos. **Tres deudas más que Resultados le deja por escrito a §Protocolo de análisis (contadas el 2026-09-03):** qué hace el estadístico con una relación en U, que un coeficiente de rangos leería como ausencia de señal (§Rango dinámico); qué hace el contraste con los entrenamientos que ya cruzaron dentro de una ventana que sirve, porque quitarlos recorta el extremo rápido y dejarlos mezcla detectar con predecir (§Solape); y cómo se interpreta el confusor del learning rate, que mueve a la vez métrica y variable dependiente (§Rango dinámico).
- [ ] **Criterio de cierre:** H1 respondida con su número, su figura y su párrafo.

### Fase C: OE2, valor incremental (el decisivo)

- [ ] Método para descontar el predictor de referencia (TSE, titular `tse/ema_0_999`, y la validación temprana), con la asimetría de §Problemas conocidos declarada por escrito. Vía principal propuesta: **comparación pareada por celda** del coeficiente de la métrica contra el del baseline, contrastada sobre las 24. Responde literalmente H2 y es lo más interpretable. La correlación parcial de rangos queda como comprobación secundaria.
- [ ] El eje de coste tiene sección propia, §Coste y capacidad predictiva; aquí entra solo en cuanto OE2 pregunta si ese coste está justificado.
- [ ] **Criterio de cierre:** H2 respondida sobre las seis VD, con la limitación de la velocidad declarada, no omitida.

### Fase D: OE3, comparación de familias

- [ ] La poda se hace antes, en la fase B. Aquí se documenta **como resultado propio**: demostrar que dos métricas de familias distintas son la misma cantidad es un hallazgo, no limpieza.
- [ ] **Contingencia:** si la variabilidad queda en una métrica, comparar una familia de cuatro contra una de una no es lo que promete el objetivo. Habría que reformular OE3 como "qué métricas, ya podadas, predicen", declarando por qué.
- [ ] **Criterio de cierre:** H3 respondida sobre la lista podada, y la poda documentada como resultado propio.

### Fase E: OE4, suficiencia temprana

- [ ] Barrido de ventanas 5/10/25/50 %, con el 100 % como ancla, teniendo en cuenta el solape medido en la fase A.
- [ ] **Trampa, ya medida el 2026-09-02:** comparar ventanas para la velocidad está contaminado. VD1 solo anticipa algo al 5 % (22 celdas) y al 10 % (12 celdas); al 25 % y al 50 % describe el pasado en las 24. VD3 aguanta hasta el 10 % en 17 celdas y VD2 hasta el 25 % en todas. VD4, VD5 y VD6 se miden al final del run y no tienen ese problema, así que OE4 es limpio sobre esas tres y sobre la velocidad se restringe a las ventanas y celdas que deja el mapa de `efficiency.py::window_overlap`.
- [ ] **Criterio de cierre:** H4 respondida, con la ventana mínima defendible dicha explícitamente.

### Fase F: OE5, robustez entre optimizadores

- [ ] Comparación pareada sobre los 12 pares de celdas que solo difieren en el optimizador.
- [ ] **Asimetría medida entre los dos brazos:** los 41 runs divergidos son todos de SGD, y Adam no divergió ni una vez en sus 480. Tenerlo delante al interpretar cualquier diferencia.
- [ ] **Criterio de cierre:** H5 respondida, declarando que 12 pares dan poca potencia y que un no rechazo no prueba invariancia.

### Fase G: OE6, concordancia con la literatura

- [ ] Contraste del signo observado contra el predicho por cada paper. La tabla de signos ya está verificada contra los PDFs (2026-07-17, [[Datos experimentales]] §5.3) y la salvedad de signo de GWA está en `fundamentos.tex:168`. **No calcula nada nuevo**, consume la salida de la fase B, así que cabe en cualquier hueco en cuanto B esté.
- [ ] **Criterio de cierre:** H6 respondida, separando los signos que el paper afirma de los que esta memoria extrapola.

### Fase H: cierre de la memoria

- [ ] **Conclusiones**, una por objetivo.
- [ ] **Cerrar las promesas colgantes** listadas en §Estado actual.
- [ ] **Declarar el encuadre del análisis:** posterior a los datos y anterior a los resultados, con las fechas que lo sostienen.
- [ ] **Notas de honestidad:** Tiny-ImageNet usa su val público como test, y el split es fijo y compartido por todos los runs. La de F1-macro ≈ accuracy ya está medida y escrita en `resultados.tex` §Concordancia entre validación y test (2026-09-02); solo hay que citarla.

### Fase I: obligatorio para el depósito (independiente, en cualquier hueco)

No depende de ningún resultado, así que puede adelantarse desde ya. Lo que es escritura de capítulos se ha movido a la fase 0; aquí queda solo lo administrativo y lo que solo se puede hacer al final.

- [ ] **Los tres resúmenes** de `main.tex`, los agradecimientos, y los **dos anexos** (ODS, obligatorio, 1-2 páginas; configuración y reproducibilidad).
- [ ] Formato UPV/ETSINF, Turnitin y entrega por EBRON.
- [ ] Borrador al tutor → incorporar feedback.
- [ ] Slides de defensa (10-15, ~15 min).
