## Estado actual (rev. 2026-08-29)

Aquí viven el estado y el plan. El *qué se decidió y por qué* está en [[2 - Decisiones]] y el *estado resultante del diseño* en [[1 - Diseño]]; este documento no los repite, los referencia. Tampoco guarda el camino recorrido, que ya está en el historial de git, solo dónde está el trabajo hoy. **La fase 0 quedó cerrada el 2026-08-28** y lo siguiente es la fase A.

- **Matriz: TERMINADA.** 960 de 960 runs, las 24 celdas con sus 40 cada una, versionados en git desde el commit `f6df900` (2026-08-22), así que existen fuera de esta máquina. Coste real 121,7 h de reloj, 97,6 h de entrenamiento y 24,1 h de instrumentación.
- **Lo único que bloquea: no hay método de análisis.** Las seis hipótesis siguen en [[1 - Diseño]] como afirmaciones falsables **sin criterio de decisión**. Bloquea las fases B a G. Como el plan preregistrado se retiró, el análisis que se haga es **posterior a los datos**, y así hay que presentarlo en la memoria.
- **Sigue sin calcularse ninguna correlación entre métrica y variable dependiente.** El método de contraste se fija antes de mirar ningún resultado, y esa disciplina deja de valer en cuanto se mire una correlación antes de tenerlo escrito.
- **Código.** `src/analysis.py` reúne los diagnósticos de sanidad de las columnas y apunta por defecto a `reports/`. `src/efficiency.py` tiene por unidad el run y la celda, ahí viven el censo de salud y el mapa de lo computable, y ahí irán los puntos 4 y 5. Las figuras viven en `src/figures.py`, una función por figura, y ahí irán también las de las fases B a G. **La primera figura ya existe** (`ventana-lr`, del punto 2, ya referenciada desde Resultados), y **sigue sin haber una línea de análisis confirmatorio**. `figstyle.py` registra las cuatro variantes de Pagella y no solo la redonda, así que un rótulo en cursiva sale en la cursiva del cuerpo; el modo matemático de matplotlib se probó y se descartó, porque imprime un aviso de tipografía en cada figura. Suite en **250** pruebas verdes.
- **Cuidado con los runs con NaN, y sobre todo con cómo se cuentan.** 165 tienen la firma en alguna época, 41 divergidos y 124 colapsados, y 133 la tienen en todas, 39 y 94. De los 165, **154 nunca aprendieron y 11 se recuperaron**, entre ellos el único ResNet-18 afectado, que colapsa cinco épocas de 40 y acaba a 24,7 veces el azar. Los 133 son el 13,9 % de los runs pero solo el 7 % de las horas, porque son todos FC y CNN. El cero falso de `stiffness/sign_*` y `confusion/frac_neg` está parcheado en `reports/` desde el 2026-08-29, **así que eso ya no hay que tratarlo: se comporta como cualquier valor ausente y ninguna fase posterior necesita una excepción**. El detalle está en [[2 - Decisiones]].
- **Qué hay realmente registrado por época.** `trajectory.parquet` trae **23 columnas de métrica de gradiente** (no 8: cada métrica emite varias claves) más **cuatro columnas de TSE**, más la curva cruda `train_loss`/`val_loss`/`val_acc` y los dos relojes. `metrics_at_window.parquet` son esas mismas columnas en cinco filas por run, una por ventana.
- **Memoria.** Escritos enteros Introducción, Estado del arte, Fundamentos, Desarrollo y Trabajos futuros. De Metodología **solo queda vacía §Protocolo de análisis**, que escribe la fase B. De Resultados solo está escrita §Qué variables quedan medidas, la apertura del capítulo; las otras ocho secciones y Conclusiones siguen siendo títulos sin prosa, y la hoja de ruta y las conclusiones del capítulo se escriben con el cuerpo. §Hipótesis da las seis afirmaciones y lo que refutaría a cada una, sin criterio numérico y sin ninguna palabra que sugiera preinscripción, porque el método es posterior a los datos. **Una figura**, la 6.1; sigue faltando la única del capítulo de Desarrollo, el diagrama del *pipeline*, anotada como comentario en su fichero. Faltan los tres resúmenes de `main.tex`, los agradecimientos y los dos anexos. Compila en 59 páginas, sin referencias sin resolver ni cajas desbordadas.
- **Qué hay escrito del censurado, con precisión.** `metodologia.tex` §Variables declara la **convención de registro**, que un valor censurado se anota como ausente y nunca como el presupuesto, y remite su tratamiento en el contraste a §Protocolo de análisis. La **población de análisis**, es decir qué entrenamientos entran, no está escrita en ningún sitio. Las dos cosas son trabajo de la fase B.
- **Promesas colgantes en la memoria: diez.** Seis apuntan a §Protocolo de análisis, una a `ch:resultados`, dos a `ch:conclusiones` y una a `sec:determinismo` del anexo de configuración. Ninguna es ya trabajo de fase 0. Las de falsación (`introduccion.tex:54` y `:68`, `estado-del-arte.tex:111`, `fundamentos.tex:170` y `:259`) las cumple la fase B y no son errores, son anuncios. **LaTeX no detecta ninguna**, porque un `\ref` resuelve igual aunque la sección esté vacía, así que «cero referencias indefinidas» no vale como comprobación y hay que contarlas aparte, comparando cada `\label` con la prosa que le sigue. Lección del recuento: escribir una sección sube tantas como baja, porque cada remisión honesta abre una colgante nueva.
- **Pilot:** ejecutado y leído; los presupuestos que salieron de él están en `config.py::DATASET_BUDGET` y en los 24 YAML. Los umbrales ya no, desde el 2026-09-01, y viven en `config.py::THRESHOLD_ACC` calculados sobre la matriz. Desde el 2026-08-29 `reports_pilot/` está versionado, con los seis `testfix_40ep/` de Tiny dentro, así que la evidencia que justifica esos números ya existe fuera de esta máquina.

## Vigentes: a tener en cuenta en todo lo que queda de análisis (rev. 2026-08-31)

Lista de comprobación para abrir cualquier sesión de análisis. Entra aquí lo que **condiciona un cálculo futuro** y se olvidaría con facilidad: trampas medidas, equivalencias entre columnas, y decisiones abiertas que cambian un número. No entra el estado del trabajo, que está arriba, ni el porqué de cada cosa, que está en [[2 - Decisiones]]. Una línea por asunto y un puntero. Cuando algo deja de condicionar, se borra de aquí.

- **El método es posterior a los datos.** La matriz acabó el 2026-08-22 y el plan se retiró el 2026-08-25. Ninguna frase, en el vault o en la memoria, puede sugerir preinscripción.
- **El umbral de VD1 es por dataset y arquitectura desde el 2026-09-01** ([[2 - Decisiones]]), y sale de una norma reproducible: el valor redondo más alto que alcanza al menos el 60 % de los entrenamientos que aprendieron. Si se recalcula, se recalcula con la norma y no a mano. Y **VD1 se recalcula siempre en la capa de análisis** desde `trajectory.parquet`: el campo `epochs_to_threshold` de los 960 `summary.json` está calculado con los umbrales viejos y **no se puede leer**.
- **VD1 y el predictor de referencia leen la misma curva.** Descontar la validación temprana para predecir VD1 es descontar un prefijo del propio resultado. OE2 será informativo sobre todo en VD4, VD5 y VD6.
- **Solape entre la ventana de medida y el cruce del umbral.** Con los umbrales del 2026-09-01, el 11 % de los cruces ya ha ocurrido al medir la ventana del 5 %, el 35 % al llegar al 10 % y el 72 % al 25 %. Antes eran 30, 56 y 82. Mejora mucho y **no lo arregla**: en la ventana del 25 % el suceso ya ha pasado para siete de cada diez. Falta el desglose por celda, que es el punto 4 de la fase A, y condiciona OE4.
- **GWA va al revés que su artículo.** Aquí se mide sobre $\nabla\ell$ bruto, así que la predicción heredada es que una GWA **baja** acompañe a mejor generalización. Conversión en `fundamentos.tex:168`, con el mismo aviso en el enunciado de H6 de [[1 - Diseño]] y en la tabla de signos de [[Datos experimentales]] §5.3, que es el material contra el que se contrasta H6.
- **En los 41 runs divergidos, `final_test_acc` y `final_gap_acc` traen número y no son mediciones**, porque `argmax` sobre logits NaN devuelve siempre el índice 0. Son las dos únicas columnas donde "no falta ningún valor" engaña. Los 41 son todos de SGD: Adam no divergió nunca.
- **`noise_scale/simple` y `mcoh/global` son la misma cantidad** (GNS = M/α − 1 exacto, Spearman intra-run −1,000), y GSNR es el recíproco por parámetro de NGV. Sin poda previa, OE3 cuenta el mismo número en los dos bandos.
- **27 columnas de predictor, no 8.** Hay que fijar una columna titular por métrica (`analysis.py::headline_columns()`) antes de contrastar, o la multiplicidad se dispara sin que ningún objetivo lo pida.
- **154 runs nunca aprendieron y los 154 tienen causa conocida**, 115 de colapso y 39 de divergencia. Se identifican por `learned` de `efficiency.py::run_health`: mejor val-accuracy suavizada por debajo de **1,25 veces el azar** (`CHANCE_MARGIN`); 1,2 y 1,25 seleccionan los mismos 154. **Distinguir dos usos, que no son el mismo:** se excluyen del denominador que **calibra** el umbral de VD1 (norma del 2026-09-01), porque no cruzarían ninguno; que entren o no en cada **contraste** sigue **sin decidir** y es de la fase B.
- **La censura de VD1 se cuenta en pares, no en runs.** Un run censurado sigue siendo comparable con cualquiera que cruzó. 15 cruces de 40 conservan el 61,5 % de los pares, no el 37,5 %. Con n = 40 en las 24 celdas, esa fracción es una función estrictamente creciente del número de cruces, así que da la escala honesta y **no una evidencia aparte**, porque los dos números dicen lo mismo.
- **Los 24 YAML de `experiments/` llevan el umbral viejo por conjunto de datos.** `run_matrix.py --init` no pisa ficheros existentes, así que siguen ahí desde que se corrió la matriz. No afecta a ningún cálculo, porque el análisis lee `config.py::THRESHOLD_ACC` y nunca los YAML, pero engaña a quien los abra.
- **El suelo de ajuste del gap sigue sin valor fijado**, el mínimo de `final_train_eval_acc` para los contrastes de VD5 y VD6 (aplazado a propósito el 2026-07-17). Es un filtro de análisis, así que fijarlo no obliga a re-correr nada.

## Problemas conocidos de los objetivos (rev. 2026-08-26)

Ninguno obliga a cambiar los seis objetivos de `introduccion.tex`. Se resuelven declarando alcance, y hay que resolverlos **antes** de escribir el objetivo al que afectan.

- **OE2 y la velocidad.** El baseline lee la misma curva que define la variable. VD1 es "en qué época la accuracy de validación cruza τ", y la segunda mitad del predictor de referencia es "la accuracy de validación en la ventana f". Son la misma curva leída en dos puntos, así que descontar el baseline para predecir VD1 es descontar un prefijo del propio resultado. Con el solape que mide el punto 4 de la fase A pasa lo mismo, porque si en una ventana ya ha cruzado buena parte de los runs, esa ventana no anticipa la velocidad, la describe. Hay que declararlo por escrito antes de redactar OE2 y OE4.
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

Plan de ejecución. El código se reparte por unidad de observación, decidido el 2026-08-31 y registrado en [[2 - Decisiones]]: `src/analysis.py` es el backend de sanidad de las columnas y lo que mira runs y celdas va a `src/efficiency.py`. Orden: 1 y 2 primero, porque fijan qué columnas sirven y con qué n hablan las demás; después el 3, que es lado predictor, y el 4 y el 5, que son lado variable dependiente. El 3 y el 4 son los que ponen techo a lo que se puede encontrar, y por eso van antes de mirar ninguna correlación: sin ellos, un resultado nulo posterior tiene tres causas que desde fuera no se distinguen, que la métrica no prediga, que no varíe entre los runs de una misma celda, o que la variable a predecir ya hubiera ocurrido al medirla. La figura de la fase es una, la del punto 2; los otros cuatro puntos son tablas o frases, y una tabla en la memoria se gana con un hallazgo, no por defecto.

- [x] **Puntos 1 y 2, cerrados, con sus veredictos y fechas en [[2 - Decisiones]].** El 2026-08-29 se retiraron del vault y de la memoria los criterios de decisión supervivientes del plan retirado, se puso [[Métricas]] al día con el código y se validaron las columnas sobre los 960, con cero valores fuera de rango, cero infinitos, las cinco identidades exactas sin violaciones y ninguna columna ausente en ningún run. El 2026-08-31 se validaron los runs con el censo de `src/efficiency.py`. El 2026-09-01 se recontó el mapa sobre los umbrales por arquitectura, se cerró el carril de LaTeX del umbral y el mapa de disponibilidad se sustituyó por la figura `ventana-lr`. **Regla de la fase, todavía en vigor:** el método de cada uno de los cinco puntos se decide al llegar a él, no los cinco por adelantado, y se registra en [[2 - Decisiones]] ahí mismo.
- [x] **Los números del mapa, recontados el 2026-09-01 sobre los umbrales por arquitectura.** Las cinco VD que no dependen de un umbral están disponibles en las **24 celdas** con 30 runs o más (919 a 921 de 960, mínimo de celda 30), y sus pérdidas son los 41 divergidos, todos de SGD, 28 con FC y 13 con CNN, ninguno con ResNet-18. VD1 también existe en las **24 celdas**, con 571 cruces de 960, de 15 a 37 por celda y 22 celdas en 20 o más; medida en pares comparables cada celda conserva del 62 % al 99 %, mediana 82 % y 10 celdas por encima del 85 %. La peor es Tiny-ImageNet con FC y Adam. Los sigue calculando `efficiency.py`, con `vd_status`, `availability_by_cell`, `vd1_information` y `crossing_by_lr`.
- [x] **Lección del carril de LaTeX, del 2026-09-01.** Al reproducir los números del vault sobre `reports/` para escribirlos en la memoria salieron dos mal, ya corregidos en el log de ese día. Los números del vault se vuelven a medir antes de entrar en la memoria, no se copian.
- [ ] **Después: rango dinámico del predictor**, si la métrica se mueve más por el learning rate que por la semilla. *Entrada:* las 27 columnas de `metrics_at_window.parquet` en las ventanas tempranas, más el censo para saber qué runs son medición. *Por qué va antes de la fase B:* descarta una de las tres causas indistinguibles de un coeficiente nulo, que la métrica no varíe dentro de la celda más allá del ruido de semilla. *Salida:* por celda y métrica, cuánta de su variación pone el LR y cuánta la semilla. **Mira solo el lado predictor, así que no gasta la disciplina de no mirar resultados.**
- [ ] **Decisión que hay que abrir antes del punto 4, y de forma explícita.** Comprobar que las relaciones son monótonas, que es lo que sostiene la elección de tau ([[2 - Decisiones]] §Pendientes), exige mirar nubes de puntos de métrica contra variable dependiente, o sea mirar justo lo que se evita hasta tener el método escrito. No es fatal, el análisis ya está declarado posterior a los datos, pero tiene que ser una decisión con fecha y no un desliz. Opciones: mirarlo abierto y registrarlo, mirarlo sobre una porción declarada de antemano, o fijar el estadístico a ciegas asumiendo que una relación en U se leería como ausencia de señal.
- [ ] Solape entre la ventana de medida y el cruce del umbral, que decide en qué ventanas se puede analizar la velocidad.
- [ ] Concordancia entre validación y test, que decide si las variables leídas sobre la curva de validación sirven.
- [ ] **Criterio de cierre:** veredicto de validez escrito, mapa tabulado, y las decisiones que se deriven reflejadas en el `.tex`.

**Las fases B a G son una máquina y seis consultas (visto el 2026-08-31).** No son seis trabajos. La fase B construye el contraste entero y lo estrena con OE1; las otras cinco añaden una idea cada una encima de la misma maquinaria, y OE6 no calcula nada nuevo, solo lee signos. La forma propuesta de esa máquina, las cinco elecciones que se toman una vez, está en [[2 - Decisiones]] §Pendientes, sin decidir. **Reordenación que sale de ahí:** la poda de métricas redundantes se adelanta de la fase D a antes de la fase B, porque decide qué columnas entran como predictor en *todos* los objetivos y no solo en OE3.

Entrada común a todas: los predictores de `metrics_at_window.parquet` (5 filas por run, una por ventana), las seis VD de `summary.json`, y el mapa de la fase A para la n de cada casilla.

### Fase B: OE1, existencia

- [ ] **Requisito previo, adelantado desde la fase D:** poda con prueba del par `noise_scale/simple` ≡ `mcoh/global` y del solape NGV/GSNR. Si se confirma al agregar, la familia de variabilidad podría quedarse en **una sola cantidad**, y una de sus tres métricas resulta ser de alineación. Eso es resultado propio y a la vez amenaza para OE3.
- [ ] Decidir el método y registrarlo. Programarlo con pruebas sobre datos sintéticos de efecto conocido, antes de tocar `reports/`: un pipeline de correlación no validado contra un efecto que sabes que existe no distingue "no hay señal" de "tengo un bug".
- [ ] Aquí se reconstruye la **prueba de hipótesis**, una sola vez y para todos los objetivos: qué se calcula dentro de cada celda, cómo se agregan las 24, cómo entra el censurado y cómo se corrige la multiplicidad.
- [ ] **Trampa:** un coeficiente nulo tiene tres causas que desde fuera no se distinguen, que la métrica no prediga, que no varíe dentro de la celda más allá del ruido de semilla, o que el resultado ya hubiera ocurrido al medir. Las descartan los puntos 3 y 4 de la fase A, y por eso van antes.
- [ ] **Texto:** §Hipótesis (H1) de `metodologia.tex`, hoy vacía, y la parte de contraste de §Protocolo de análisis, más su sección de resultados. Ahí entran también la población de análisis y el tratamiento del censurado, que no están escritos.
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
- [ ] **Trampa:** comparar ventanas para VD1 está contaminado, porque en la del 25 % el cruce ya ha ocurrido en buena parte de los runs y la métrica "predice" el pasado. VD4, VD5 y VD6 se miden al final del run y no tienen ese problema, así que OE4 es limpio sobre esas tres y hay que declararlo sobre las de velocidad.
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
- [ ] **Notas de honestidad:** Tiny-ImageNet usa su val público como test, F1-macro ≈ accuracy en datasets balanceados, y el split es fijo y compartido por todos los runs.

### Fase I: obligatorio para el depósito (independiente, en cualquier hueco)

No depende de ningún resultado, así que puede adelantarse desde ya. Lo que es escritura de capítulos se ha movido a la fase 0; aquí queda solo lo administrativo y lo que solo se puede hacer al final.

- [ ] **Los tres resúmenes** de `main.tex`, los agradecimientos, y los **dos anexos** (ODS, obligatorio, 1-2 páginas; configuración y reproducibilidad).
- [ ] Formato UPV/ETSINF, Turnitin y entrega por EBRON.
- [ ] Borrador al tutor → incorporar feedback.
- [ ] Slides de defensa (10-15, ~15 min).
