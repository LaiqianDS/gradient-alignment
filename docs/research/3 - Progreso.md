## Estado actual (rev. 2026-08-29)

Aquí viven el estado y el plan. El *qué se decidió y por qué* está en [[2 - Decisiones]] y el *estado resultante del diseño* en [[1 - Diseño]]; este documento no los repite, los referencia. Tampoco guarda el camino recorrido, que ya está en el historial de git, solo dónde está el trabajo hoy. **La fase 0 quedó cerrada el 2026-08-28** y lo siguiente es la fase A, que se escribe desde cero porque su primera pasada se deshizo entera, código y texto incluidos.

- **Matriz: TERMINADA.** 960 de 960 runs, las 24 celdas con sus 40 cada una, versionados en git desde el commit `f6df900` (2026-08-22), así que existen fuera de esta máquina. Coste real 121,7 h de reloj: 97,6 h de entrenamiento y 24,1 h de instrumentación, con un sobrecoste máximo por celda de 2,048x, dentro de la cota.
- **Lo único que bloquea: no hay método de análisis.** Las seis hipótesis siguen en [[1 - Diseño]] como afirmaciones falsables **sin criterio de decisión**. Bloquea las fases B a G. Y como el plan preregistrado se retiró, el análisis que se haga es **posterior a los datos**, y así hay que presentarlo en la memoria.
- **Sigue sin calcularse ninguna correlación entre métrica y variable dependiente.** El método de contraste se fija antes de mirar ningún resultado, y esa disciplina deja de valer en cuanto se mire una correlación antes de tenerlo escrito.
- **Código.** `src/analysis.py` reúne los diagnósticos de sanidad: validez, identidades, columnas ausentes, degeneración, tendencia y redundancia. Desde el 2026-08-29 apunta por defecto a `reports/` y no al piloto. **No hay ninguna figura ni una línea de análisis confirmatorio.** El estilo de figuras sí existe, en `src/figstyle.py`, y la primera figura real llega con la fase A. Suite en **231** pruebas verdes.
- **Cuidado con los 133 runs con NaN.** 39 divergieron (SGD a lr 0,3 y 1,0) y 94 se quedaron planos con la red colapsada; ninguno es ResNet-18, la única arquitectura con normalización por lotes. Juntos son el 13,9 % de los runs pero solo el 7 % de las horas, porque son todos FC y CNN. En los divergidos, `stiffness/sign_*` y `confusion/frac_neg` traían un 0,0 falso y desde el 2026-08-29 traen NaN, parcheado en 41 runs. **Lo que sigue engañando ahí es la accuracy**: la de test vale la frecuencia de la clase 0 de cada dataset, no el azar, porque `argmax` sobre logits NaN devuelve siempre el índice 0. Esas columnas no se han tocado a propósito. El detalle está en [[2 - Decisiones]].
- **Qué hay realmente registrado por época.** `trajectory.parquet` trae **23 columnas de métrica de gradiente** (no 8: cada métrica emite varias claves) más **4 columnas de TSE**, más la curva cruda `train_loss`/`val_loss`/`val_acc` y los dos relojes. `metrics_at_window.parquet` son esas mismas columnas en 5 filas por run, una por ventana. Consecuencia operativa: antes de contrastar hay que fijar **una columna titular por métrica** (`analysis.py::headline_columns()` existe para eso), o la multiplicidad se dispara sin que ningún objetivo lo pida.
- **Memoria.** Escritos enteros Introducción, Estado del arte, Fundamentos, Desarrollo y Trabajos futuros. De Metodología **solo queda vacía §Protocolo de análisis**, que escribe la fase B. Resultados y Conclusiones siguen siendo títulos sin prosa. §Hipótesis da las seis afirmaciones y lo que refutaría a cada una, sin criterio numérico y sin ninguna palabra que sugiera preinscripción, porque el método es posterior a los datos. **Cero figuras**, incluida la única del capítulo de Desarrollo, el diagrama del *pipeline*, anotada como comentario en su fichero. Faltan los tres resúmenes de `main.tex`, los agradecimientos y los dos anexos. Compila en 59 páginas, sin referencias sin resolver ni cajas desbordadas.
- **Qué hay escrito del censurado, con precisión.** `metodologia.tex` §Variables declara la **convención de registro**, que un valor censurado se anota como ausente y nunca como el presupuesto, y remite su tratamiento en el contraste a §Protocolo de análisis. La **población de análisis**, es decir qué entrenamientos entran, no está escrita en ningún sitio. Las dos cosas son trabajo de la fase B.
- **Cuidado al tocar cualquier texto sobre GWA: la dirección correcta en esta memoria es la contraria a la del artículo.** `fundamentos.tex:168` convierte el signo, porque el original define el gradiente como $-\nabla\ell$ y aquí se mide sobre $\nabla\ell$ bruto, de modo que la predicción heredada es que una GWA **baja** acompañe a mejor generalización. Desde el 2026-08-29 el aviso está también en el enunciado de H6 de [[1 - Diseño]] y en la tabla de signos de [[Datos experimentales]] §5.3, que es el material contra el que se contrasta H6.
- **Promesas colgantes en la memoria: diez.** Seis apuntan a §Protocolo de análisis, una a `ch:resultados`, dos a `ch:conclusiones` y una a `sec:determinismo` del anexo de configuración. Ninguna es ya trabajo de fase 0. Las de falsación (`introduccion.tex:54` y `:68`, `estado-del-arte.tex:111`, `fundamentos.tex:170` y `:259`) las cumple la fase B y no son errores, son anuncios. **LaTeX no detecta ninguna**, porque un `\ref` resuelve igual aunque la sección esté vacía, así que «cero referencias indefinidas» no vale como comprobación y hay que contarlas aparte, comparando cada `\label` con la prosa que le sigue. Lección del recuento: escribir una sección sube tantas como baja, porque cada remisión honesta abre una colgante nueva.
- **Pilot:** ejecutado y leído; presupuestos y umbrales en `config.py::DATASET_BUDGET` y en los 24 YAML. Desde el 2026-08-29 `reports_pilot/` está versionado, con los seis `testfix_40ep/` de Tiny dentro, así que la evidencia que justifica esos números ya existe fuera de esta máquina.
- **Lista de métricas:** cerrada con la implementación. Variabilidad (normalized variance, GNS simple, GSNR) y alineación (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline.

## Problemas conocidos de los objetivos (rev. 2026-08-26)

Ninguno obliga a cambiar los seis objetivos de `introduccion.tex`. Se resuelven declarando alcance, y hay que resolverlos **antes** de escribir el objetivo al que afectan.

- **OE2 y la velocidad: el baseline lee la misma curva que define la variable.** VD1 es "en qué época la accuracy de validación cruza τ", y la segunda mitad del predictor de referencia es "la accuracy de validación en la ventana f". Son la misma curva leída en dos puntos, así que descontar el baseline para predecir VD1 es descontar un prefijo del propio resultado. Hay además un solape entre la ventana de medida y el propio cruce del umbral, que **la fase A tiene que cuantificar** antes de que OE2 y OE4 se puedan escribir: si en una ventana ya ha cruzado buena parte de los runs, esa ventana no anticipa la velocidad, la describe. El contraste de OE2 será por tanto informativo sobre todo en las variables que la curva de validación no define, es decir, la accuracy de test y los dos gaps.
- **OE3 y el solape entre familias.** `noise_scale/simple` (variabilidad) y `mcoh/global` (alineación) son la misma cantidad reparametrizada: `GNS = M/α − 1` exacto (Ecuación `eq:gns-mcoh` de `fundamentos.tex`), y el Spearman intra-run medido sobre los 960 sale **−1,000**. Además GSNR es el recíproco por parámetro de NGV. Comparar familia contra familia sin podar antes cuenta el mismo número una vez en cada bando. **La poda con prueba deja de ser mejora de presentación y pasa a ser requisito previo de OE3.**
- **El coste por métrica no está medido y no es separable.** `summary.json` trae un único `metric_seconds` por run, y seis de las ocho métricas comparten un solo barrido per-sample, así que su coste marginal individual no existe ni a posteriori. **Resuelto el 2026-08-27 por la vía del coste asintótico** ([[2 - Decisiones]]): el eje de coste deja de ser continuo y pasa a ser una escala de clases, de modo que lo que se puede dibujar honestamente no es un frente de Pareto, sino una comparación dentro de cada clase y frente al predictor que no deriva nada. La sección de resultados se llama §Coste y capacidad predictiva (`sec:res-coste`) y se mantiene aparte en lugar de doblarse dentro de OE2, porque las clases de coste dan material propio y OE2 pregunta otra cosa. **Esa última parte es decisión mía, revisable.** Sigue disponible el micro-benchmark el día que se quiera el eje continuo: minutos de cómputo, fuera de la matriz.

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

Plan de ejecución. El código va en `src/analysis.py`, que ya es el backend de diagnóstico de sanidad; hoy apunta por defecto a `reports_pilot/` y su `_main` se anuncia como piloto de 1 run por celda, así que repuntarlo a `reports/` es la primera línea del punto 1. Orden: 1 y 2 primero, porque fijan qué columnas sirven y con qué n hablan las demás; después el 3, que es lado predictor, y el 4 y el 5, que son lado variable dependiente. El 3 y el 4 son los que ponen techo a lo que se puede encontrar, y por eso van antes de mirar ninguna correlación: sin ellos, un resultado nulo posterior tiene tres causas que desde fuera no se distinguen, que la métrica no prediga, que no varíe entre los runs de una misma celda, o que la variable a predecir ya hubiera ocurrido al medirla. La figura de la fase es una, el mapa del punto 2; los otros cuatro puntos son tablas o frases, y una tabla en la memoria se gana con un hallazgo, no por defecto.

- [x] Vaciar este documento de lo innecesario de fases previas. Hecho el 2026-08-29.
- [x] Retirar del vault y de la memoria los criterios de decisión supervivientes del plan retirado, y poner [[Métricas]] al día con el código. Hecho el 2026-08-29 y registrado en [[2 - Decisiones]].
- [x] **Apertura de la fase: qué produce, qué no hace, y dónde vive el veredicto en la memoria.** Hecho el 2026-08-29 y registrado en [[2 - Decisiones]]. El método de cada uno de los cinco puntos se decide al llegar a él, no los cinco por adelantado, y se registra ahí mismo.
- [x] **Validez de las columnas sobre los 960.** Hecho el 2026-08-29, con veredicto y hallazgo en [[2 - Decisiones]]: cero valores fuera de rango, cero infinitos, las cinco identidades exactas sin violaciones, y ninguna columna ausente en ningún run. El único hallazgo es el cero falso de `torch.sign(NaN)`, corregido en el código.
- [ ] Validez de los runs: censo de los que se quedaron sin aprender, y comprobar cómo se solapan los 133 con GWA a NaN y los 154 clavados en el azar del criterio del 2026-08-27.
- [ ] Mapa de lo computable celda a celda: qué runs entran en el análisis y cuántos quedan para cada variable dependiente. Parte ya está medida y dispersa en [[2 - Decisiones]]: los 154 clavados en el azar, las seis celdas de FC sin variable de velocidad, y los 227 frente a 239 censurados por optimizador.
- [ ] Rango dinámico del predictor: si la métrica se mueve más por el learning rate que por la semilla.
- [ ] Solape entre la ventana de medida y el cruce del umbral, que decide en qué ventanas se puede analizar la velocidad.
- [ ] Concordancia entre validación y test, que decide si las variables leídas sobre la curva de validación sirven.
- [ ] **Criterio de cierre:** veredicto de validez escrito, mapa tabulado, y las decisiones que se deriven reflejadas en el `.tex`.

### Fase B: OE1, existencia

- [ ] Decidir el método y registrarlo. Programarlo con pruebas sobre datos sintéticos de efecto conocido, antes de tocar `reports/`.
- [ ] Aquí se reconstruye la **prueba de hipótesis**, una sola vez y para todos los objetivos: qué se calcula dentro de cada celda, cómo se agregan las 24, cómo entra el censurado y cómo se corrige la multiplicidad.
- [ ] **Texto:** §Hipótesis (H1) de `metodologia.tex`, hoy vacía, y la parte de contraste de §Protocolo de análisis, más su sección de resultados. Ahí entran también la población de análisis y el tratamiento del censurado, que no están escritos.
- [ ] **Criterio de cierre:** H1 respondida con su número, su figura y su párrafo.

### Fase C: OE2, valor incremental (el decisivo)

- [ ] Método para descontar el predictor de referencia (TSE y validación temprana), con la asimetría de §Problemas conocidos declarada por escrito.
- [ ] El eje de coste tiene sección propia, §Coste y capacidad predictiva; aquí entra solo en cuanto OE2 pregunta si ese coste está justificado.
- [ ] **Criterio de cierre:** H2 respondida sobre las seis VD, con la limitación de la velocidad declarada, no omitida.

### Fase D: OE3, comparación de familias

- [ ] **Requisito previo:** poda con prueba del par `noise_scale/simple` ≡ `mcoh/global`, y del solape NGV/GSNR. Sin poda, este objetivo no es contestable.
- [ ] **Criterio de cierre:** H3 respondida sobre la lista podada, y la poda documentada como resultado propio.

### Fase E: OE4, suficiencia temprana

- [ ] Barrido de ventanas 5/10/25/50 %, teniendo en cuenta el solape medido en la fase A.
- [ ] **Criterio de cierre:** H4 respondida, con la ventana mínima defendible dicha explícitamente.

### Fase F: OE5, robustez entre optimizadores

- [ ] Comparación pareada sobre los 12 pares de celdas que solo difieren en el optimizador.
- [ ] **Criterio de cierre:** H5 respondida, declarando que 12 pares dan poca potencia y que un no rechazo no prueba invariancia.

### Fase G: OE6, concordancia con la literatura

- [ ] Contraste del signo observado contra el predicho por cada paper. La tabla de signos ya está verificada contra los PDFs (2026-07-17, [[Datos experimentales]] §5.3) y la salvedad de signo de GWA está en `fundamentos.tex:168`.
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

### Higiene, barato y en cualquier hueco

- [x] **Copia de seguridad de `reports_pilot/`.** Hecho el 2026-08-29: sale de `.gitignore` y queda versionado, 2,5 MB y 120 ficheros, con los seis `testfix_40ep/` de Tiny dentro. Con ello quedan disponibles las cifras del régimen post-meseta si alguna vez se quieren en la memoria.
