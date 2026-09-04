# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

Qué entra en una entrada: la decisión, su porqué en corto, la evidencia con su número, y las trampas. Nada más. El registro se poda mientras se trabaja, y las decisiones y sus fechas no se podan nunca; la regla completa está en [[3 - Progreso]] §Plan por objetivos.

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] §Plan por objetivos.

**Ninguna pendiente bloquea hoy.** El método de análisis quedó fijado el 2026-09-03 para los seis objetivos y revisado esa misma tarde en diez puntos (entradas de esa fecha); la población de análisis y el tratamiento del censurado están escritos en `metodologia.tex` §Protocolo de análisis. Lo que cada fase decida al abrirse se registra aquí con fecha.

## Tomadas (log)

### 2026-09-04

#### El código queda autocontenido: sin referencias a docs/, al vault ni a la memoria, comentarios al mínimo y código muerto fuera

**Qué se decidió (Lai: «El código no debería depender de docs/ ni nada de eso, ninguna referencia. El código vive autocontenido»; plan aprobado con «Me gusta, adelante»).** Cuatro subagentes sobre ficheros disjuntos y una revisión final. (1) Fuera toda referencia en comentarios, docstrings, nombres e impresiones a `docs/`, al vault, a la memoria («body text», «the protocol», «the objectives»), a fases («phase B»), a hipótesis (H1 a H6, VD1), a fechas y a historias de arreglos; las funciones de `contrast.py` describen la tabla que producen y sus cabeceras por consola también; la figura `signos-h1` pasa a `signos` (fichero renombrado con `git mv`, un `\includegraphics` en Resultados); el test «the name the memoria uses» se renombra. Las citas de artículo en los docstrings de las métricas se quedan, porque son la definición. (2) Comentarios al mínimo: cada regla compartida vive en un solo sitio (TSE por `epoch` en `tse.py`, invariancia al bloque en `primitives.py`, el asignador CUDA en `train.py`, los campos obsoletos de `summary.json` en `efficiency.smoothed_fields`); fuera los que repiten la línea siguiente, las banderas de sección y las listas de reglas de autoría; se quedan los porqués numéricos (NaN de `torch.sign`, cancelación en fp32, memoria). (3) Código muerto fuera: las figuras `crossings_consumed`, `crossing_bands` y `val_test` con sus tests y sus PDF, `figstyle.match_limits`, `analysis.PILOT_DIR`, el parámetro `overwrite` de `init_cells`, dos locales sin uso, un `import numpy` sombreado. (4) Duplicados unificados: `ROOT` solo en `config.py`, `CELL` público en `efficiency.py`, `chance_level` en la figura, `TV_CLASSES` público; `IMG_DIR` pasa a `FIGURE_DIR`, y sigue escribiendo en `thesis/img/`. (5) `pyproject.toml`: `numpy` declarado, `matplotlib` y `scipy` pasan a dependencias porque `efficiency.py` y `figures.py` los importan, `ipykernel` fuera; `uv.lock` regenerado. (6) `README.md` reescrito como README de código, sin enlaces a `docs/` ni a la memoria; `src/metrics/README.md` sin «the thesis», sin fecha de perfilado, sin rayas, y con la `gradient disparity` en variabilidad, como la trata el análisis; `.gitignore` gana `.pytest_cache/`.

**Qué no se tocó, y por qué.** `train.py` no cambia ni una línea de código (solo prosa), porque es el que produjo los 960 entrenamientos; los dos lanzadores siguen siendo casi copias; `tabla_larga_960.parquet` se sigue escribiendo; `metrics/base.py` se queda como contrato; `.vscode/` y `media/` como estaban; no se añadió `ruff` ni ninguna herramienta.

**Evidencia.** Suite completa: 310 pruebas verdes (314 menos las cuatro de lo borrado). `git diff` de `train.py` sin ninguna línea de código cambiada. Las siete figuras regeneradas desde `reports/` y `results/` con el código nuevo son idénticas byte a byte a las del código anterior (un cambio de formateador en los ticks de `solape-celda` se revirtió para que lo fueran). Búsqueda de control sobre `src/`, `tests/`, `experiments/`, `pyproject.toml` y `README.md` con el patrón de punteros: solo la ruta `thesis/img` de `FIGURE_DIR` y las dos líneas del `README` que nombran `docs/` y `thesis/` en la lista de directorios. Diff: 34 ficheros de código, +331 y −862 líneas. La memoria compila igual con la figura renombrada.

- **Trampa:** `config.ROOT` lleva `.resolve()`, así que `EXPERIMENTS_DIR`, `REPORTS_DIR`, `DATA_PATH` y `PILOT_DIR` son rutas resueltas; en este repositorio da lo mismo, con un enlace simbólico no.
- **Trampa:** las etiquetas `fig:signos-h1` y `tab:signos-h1` de `resultados.tex` siguen con ese nombre; son de la memoria, no del código, y cambiarlas toca seis líneas de ese fichero.
- **Trampa:** `matplotlib` y `scipy` ya no son de desarrollo; un `uv sync --no-dev` sigue instalándolos, que es lo que `efficiency.py` necesita.

#### La fase H escribe las Conclusiones una por objetivo, reordena Resultados al orden de los objetivos y cierra Metodología y Resultados

**Qué se decidió (Lai, con la explicación leída, «Adelante»).** (1) `conclusiones.tex` entero: §Cumplimiento de objetivos con un párrafo por objetivo, todos con la misma forma (veredicto, la cifra que lo decide, qué significa, contraste con la literatura) y un párrafo final que responde a la pregunta de investigación; §Aportaciones en cuatro (el banco común con su coste medido, la poda como resultado, el método de contraste, la lectura del signo como composición de curvas), sin métrica nueva; §Limitaciones por lo que el diseño no puede afirmar (un solo eje barrido, tamaño de las comparaciones, qué se mide, alcance, tres notas de honestidad, orden de los hechos), sin pisar Trabajos futuros, que dice qué hacer. (2) `resultados.tex`: hoja de ruta, secciones en el orden de los objetivos (H1, H2, coste, H3, H4, H5, H6; antes H4 iba entre H1 y H2 y las dos se citaban hacia delante y hacia atrás) y §Conclusiones del capítulo en registro descriptivo, sin interpretar. (3) `metodologia.tex`: §Conclusiones del capítulo. (4) `trabajos-futuros.tex` §Intervención: H2 pasa de condicional a indicativo, y lo que queda por probar es una lectura anterior a la primera ventana. (5) Vocabulario de Lai en este carril: «puerta» → «requisito previo» (Resultados ×3, §Protocolo ×1) y «régimen» → «estas condiciones» (Resultados ×2, §Protocolo ×1); quedan «puerta» ×1 y «régimen» ×1 en §Riesgos, «régimen» y «constructo» en §Variables, §Configuración y §Calibración y «régimen» ×2 en Trabajos futuros, que revisa la otra sesión. (6) Las fechas del método viven solo en §Cuándo se fijó; Conclusiones declara el encuadre sin fechas (regla de Lai: sin fechas de decisión en la memoria) y remite ahí, con Nosek et al. 2018 y Willroth y Atherton 2024. (7) El recorte no se hace en esta fase: queda como decisión de Lai con las cifras de abajo.

**Por qué.** La Introducción promete retomar los objetivos uno a uno y la plantilla pide que Resultados describa y Conclusiones interprete; el orden por objetivos pone la hipótesis decisiva junto al requisito previo.

**Evidencia.** Compila en 110 páginas de PDF, 100 impresas más 10 de portada e índices, sin cajas desbordadas ni referencias sin resolver (104 antes de la fase H); cuerpo, de Introducción a Trabajos futuros, 90 impresas, con Metodología en 24 (27 a 50), Resultados en 22 (59 a 80) y Conclusiones en 5 (81 a 85). Promesas colgantes recontadas a mano: cuatro en el PDF, todas hacia Conclusiones (Introducción §Objetivos y §Estructura; Metodología §Cuenta granulada y §Riesgos, la palabra «no identificable»), cerradas; tres comentarios de plantilla (hoja de ruta y cierre de Resultados, cierre de Metodología), cerrados; la condicional sobre H2 de Trabajos futuros, cerrada; queda el comentario de `main.tex` sobre revisar los resúmenes al cerrar los capítulos, que es de la fase I. Palabras comprometidas, comprobadas en `conclusiones.tex`: «no superan» (H2), «no es identificable» (alineación, dos veces), «no haber visto lo contrario» (H5), «cae en parte» (H6). Tics contados sobre las líneas nuevas: cero «y no», «sino», «conviene», «de modo que», frases con «Y» inicial y dos puntos explicativos; cero anglicismos fuera de `\emph{}`; ninguna frase nueva de más de 45 palabras.

- **Trampa:** Conclusiones no lleva ninguna cifra que no esté en Resultados; la única relectura es «D_ref entre 0,23 y 0,46», que junta las dos columnas de `tab:h2`.
- **Trampa:** la nota del seminario (50 a 100 páginas) no dice si cuenta portada, índices, bibliografía y anexos. Si cuenta solo el cuerpo, 90 y no hay que recortar; si cuenta todo, 100 impresas más lo que añadan los resúmenes. Recortar sale de Metodología (24) y de la parte de validez de Resultados (8).
- **Trampa:** las dos sesiones escriben sobre `3 - Progreso.md`. Mientras esta fase escribía, la otra commiteó su línea de la GPU (`649de77`), cambió una frase de §Memoria entre la lectura y la edición, y dejó sin commitear una frase nueva en la fase I (que nada en `reports/` registra la máquina); al commitear el vault de esta fase hay que dejar fuera ese hunk o esperar a que lo commitee, y antes de editar el fichero hay que releerlo.

#### La revisión de las fases C a G corrige H6, que cae en parte, y declara que el jackknife por run es ancho porque los 40 runs de una celda no son independientes

**Qué se decidió (Lai pidió la revisión completa de las fases C a G y después «los cambios pertinentes»).** Nueve cambios, ninguno de método. (1) La disparidad con el accuracy de test entra en H6 como predicción del artículo (Forouzesh y Thiran, signo −): la tabla de signos de [[Datos experimentales]] §5.3 la registraba como base de la extrapolación a la velocidad, [[1 - Diseño]] la cita y `contrast.GOOD_END` ya la usaba como regla del artículo en la lectura de selección; dejarla fuera «por no registrada» era falso y se decidió con el signo a la vista. `contrast.PREDICTED` gana una fila y `results/signos.parquet` se regenera; los otros nueve parquets salen idénticos. (2) `metodologia.tex` §El coeficiente de una celda deja de afirmar que los 40 runs son independientes: la seed fija la inicialización, el orden de los batches y el probe, así que los ocho learning rates de una seed comparten las tres cosas. El jackknife por run se mantiene como intervalo único del capítulo y se declara ancho para la rejilla fija, con la sensibilidad por seed en una frase. (3) H1: solo GWA cumple las dos condiciones de la puerta; la disparidad se da la vuelta en MNIST (dos negativas seguras frente a una positiva); se añade la lectura por signo mayoritario (16 frente a 7 en velocidad, 15 frente a 7 en test), porque la posición en la rejilla no tiene varianza de seed y su intervalo no es comparable. (4) H2 y §Coste: «no añaden nada» pasa a «no superan», y se dice que el diseño no mide lo que añadirían encima de la validación. (5) H4: cuatro métricas no maduran y dos maduran poco (stiffness y confusión, hasta 0,51 y 0,41 con el gap, cinco y cuatro celdas con intervalo); la cuenta de velocidad es sobre 23 celdas, porque tiny-fc-adam no tiene ningún par comparable en la segunda ventana. (6) «A una cola» se sustituye en la hipótesis, el protocolo y Resultados por «con el signo fijado de antemano y el mismo intervalo de dos colas». (7) GWA deja de llamarse «sustituto» de la validación (D_ref 0,45, lejos de 0,8): ordena en parte como ella y peor. (8) H3 declara que el signo mayoritario en crudo se fijó con la tabla de H1 a la vista y qué cambiaría la otra lectura (GNS con el test, del sexto al tercer puesto). (9) Errata 0,33 → 0,32 (GSNR con el gap, `tab:h4`), una frase en §La velocidad, por hitos (D_ref y granulada de esa fila van sobre todos los runs) y dos frases sobre intervalos anchos en H4-velocidad y H5 (poblaciones anidadas, seeds compartidas).

**Por qué.** Una predicción cuyo signo estaba escrito antes de los datos no se puede excluir después de verlo; y un párrafo que dice «independientes» donde no lo son es falso aunque su consecuencia sea benigna.

**Evidencia.** Jackknife por seed (cinco réplicas, quitar los ocho runs de una seed), familia primaria al 5 %, 720 D: error típico 0,58 veces el de quitar un run, de mediana (cuartiles 0,38 y 0,83); con t(4) = 2,78 las celdas seguras suben en casi todas las combinaciones (stiffness con velocidad 14 → 19, disparidad con test 16 → 20, posición 12 → 21) y bajan en tres (GNS con gap 11 → 6, GWA con test 16 → 13, GWA con gap 10 → 8); con 1,96 ningún signo mayoritario de H6 cambia. No se recalcularon H2, H4 ni H5 con ese jackknife. Disparidad con el test: 12 positivas seguras (cnn 6, fc 5, resnet18 1) y cuatro negativas (fc 2, resnet18 2); D_ref con el accuracy de validación +0,26 de mediana, positiva en 16 celdas. Las seis tablas de Resultados, cotejadas número a número contra `results/`: solo la errata.

**Veredicto de H6 desde hoy.** Cae en parte: de siete predicciones, dos se cumplen (GWA con test 15/1, GSNR con gap 11/2, en las tres arquitecturas), tres no reciben apoyo (confusión y escala de ruido con la velocidad, m-coherencia con el gap) y dos van en contra de forma sistemática (stiffness con la velocidad 9 frente a 5, disparidad con el test 12 frente a 4), las dos con la forma de la composición de curvas. Los demás veredictos no cambian.

- **Trampa:** el jackknife por seed no se adopta; con cinco réplicas el error típico es tan incierto que exige una t con cuatro grados de libertad, y el intervalo por run es uno para todo el capítulo. La comparación de H1 «más celdas que la posición en la rejilla» sí depende del método de intervalo, porque `log_lr` no tiene varianza de seed; por eso Resultados la da también por signo mayoritario.
- **Trampa:** la stiffness con el gap (15/1 en el sentido del título de su artículo) sigue fuera del veredicto: nunca tuvo signo escrito en el vault.
- **Trampa:** «no superan» no es «no añaden»; lo segundo pediría la asociación condicionada a la validación, que el 2026-09-03 se retiró a propósito, y las Conclusiones deben usar la primera palabra.

#### La fase G lee H6 como celdas a favor y en contra del signo de cada artículo, y H6 no se refuta: los signos se reproducen en el desenlace y no en la velocidad

**Revisada esa misma tarde (entrada de arriba):** la disparidad con el accuracy de test entra como predicción del artículo y H6 pasa a caer en parte. El resto de la entrada se conserva tal como se escribió.

**Qué se decidió (Lai delegó con la explicación leída).** La regla del protocolo tal cual: por predicción que el artículo enuncia, celdas con el intervalo fuera del cero a favor del signo predicho y en contra, a una cola porque el sentido está fijado de antemano, con el mismo intervalo de todas las tablas y sin bajar la barra a 1,645, para que una celda signifique lo mismo en todo el capítulo (`contrast.PREDICTED`, `concordance_table`, `results/signos.parquet`, con la cuenta repetida por arquitectura). Tres concreciones: la predicción de la m-coherencia sobre el gap cuenta como fuerte, porque el vault la verificó contra el PDF el 2026-07-17 con la salvedad del propio artículo, y el párrafo de H6 de `metodologia.tex` la nombra desde hoy («sobre la velocidad y sobre el gap»); la disparidad no entra al veredicto aunque su artículo afirme correlación positiva con el error de test, porque esa predicción nunca se registró y añadirla con el signo ya visto sería elegir la predicción después del dato, así que se reporta como observación y se dice que sale al revés; lo mismo para la stiffness con el gap, que va en el sentido del título de su artículo. Sin figura, porque son recuentos.

**Por qué.** H6 no calcula nada nuevo: son las cifras de la tabla de H1 partidas por el signo que cada artículo afirma. `PREDICTED` es la tabla de signos de [[Datos experimentales]] §5.3 por variable, con GWA convertida y la m-coherencia sobre la escala de ruido con el signo cambiado; una prueba la ata a `GOOD_END`, que es la misma tabla leída como extremo bueno.

**Evidencia, `results/signos.parquet` del 2026-09-04, celdas a favor/en contra (fc, cnn, resnet18).** Fuertes: GWA con test 15/1 (7/0, 3/0, 5/1; la contraria es tiny-resnet18-sgd); GSNR con gap 11/2 (4/2, 3/0, 4/0; las dos contrarias son cifar10-fc); escala de ruido con velocidad 7/3 (0/1, 2/0, 5/2; las tres contrarias en MNIST); confusión con velocidad 5/4 (4/0, 0/3, 1/1); stiffness con velocidad 5/9 (1/2, 0/4, 4/3); escala de ruido con gap, la m-coherencia, 5/6 (0/2, 1/3, 4/1). Extrapoladas: GWA con velocidad 16/1; GWA con gap 6/4 (0/0, 0/4, 6/0); GSNR con velocidad 7/6 (1/0, 0/5, 6/1) y con test 7/4; disparidad con velocidad 4/9. No registradas: disparidad con test, 4 en el sentido del artículo y 12 en contra; stiffness con gap, 15/1 en el sentido del título.

**Veredicto.** H6 no se refuta: donde hay asociación los sentidos no contradicen de forma sistemática a los predichos. Las dos predicciones sobre el desenlace (GWA con test, GSNR con gap) se cumplen en las tres arquitecturas; las tres sobre la velocidad no reciben apoyo (5 a 7 celdas de 24); y la única en contra, la stiffness (9 frente a 5), tiene la forma de la composición de curvas: al 5 % ordena al revés que el accuracy de validación (D_ref mediana −0,44), así que una stiffness alta marca un run retrasado por su learning rate. Escrito en `resultados.tex` §Concordancia con la literatura, tabla `tab:h6`.

- **Trampa:** los artículos fijaron el signo sobre otros ejes (tiempo, anchura y profundidad, batch, ruido de etiquetas) y aquí solo se mueve el learning rate; una discrepancia se lee primero como composición de dos curvas y ninguna cuenta de aquí refuta un mecanismo.
- **Trampa:** el acuerdo de GWA es el de un sustituto del accuracy de validación con el accuracy de test; confirma al artículo en lo que el artículo mide y no añade prueba sobre los gradientes (H2 ya cayó).
- **Trampa:** la cuenta sobre 24 esconde cambios de bando; la confusión va con su artículo en la red plana (4/0) y en contra en la convolucional (0/3), y por eso la tabla lleva la cuenta por arquitectura.

#### La fase F lee H5 sobre los 12 pares con acuerdo e inversión solo entre D seguras, y H5 no se refuta ni queda demostrada

**Qué se decidió (Lai delegó con la explicación leída).** La regla del protocolo tal cual: por par de celdas, acuerdo o inversión solo cuando las dos D dejan fuera el cero, y la diferencia D con SGD menos D con Adam con el intervalo de sumar las dos varianzas (`contrast.optimizer_table`, `results/optimizadores.parquet`). Se publica al lado el recuento de pares donde la diferencia deja fuera el cero, porque una asociación que conserva el signo pero cambia de tamaño también es un cambio. Sin figura, porque las inversiones van de 0 a 2 y la memoria pasa de 100 páginas. La sección pasa de «Robustez cross-optimizador» a «Robustez entre optimizadores».

**Por qué.** Dos signos que coinciden sin intervalo coinciden por azar la mitad de las veces. Las dos D vienen de entrenamientos distintos, así que no hay jackknife pareado y se suman varianzas.

**Evidencia, `results/optimizadores.parquet` del 2026-09-04.** Inversiones por métrica (velocidad/test/gap): disparidad 1/2/2, stiffness 1/1/0, GNS 0/0/1, GSNR, confusión y GWA 0; posición 1/2/0; gratuitos 0. Pares que invierten: disparidad en cifar10-fc (velocidad y test), mnist-resnet18 (test), cifar100-cnn y tiny-cnn (gap); stiffness en cifar100-resnet18 (velocidad y test); GNS en mnist-resnet18 (gap); posición en mnist-fc (velocidad y test) y mnist-resnet18 (test). Acuerdos con intervalo: métricas de 1 a 6 (GWA 6 en velocidad y test, stiffness 5 en gap); gratuitos 11 o 12 en velocidad, 6 o 7 en test, 10 u 11 en gap. Tamaño (|D| SGD → Adam): disparidad 0,48 → 0,23 en velocidad (4 diferencias fuera del cero), stiffness 0,21 → 0,43 en gap, posición 0,42 → 0,18 en test (6); gratuitos más fuertes con Adam en velocidad (0,73 a 0,78 frente a 0,58 a 0,69) y con SGD en gap (0,56 a 0,62 frente a 0,45 a 0,51).

**Veredicto.** H5 no se refuta y no queda demostrada: ninguna asociación segura cambia de signo en más de dos pares de doce, las inversiones llegan de dos en dos con las dos variables de un mismo par (la forma de una métrica que lee el learning rate), y lo que cambia con el optimizador es el tamaño más que el signo. Escrito en `resultados.tex` §Robustez entre optimizadores, tabla `tab:h5`.

- **Trampa:** 12 pares solo detectan inversiones groseras; «invariante» no se escribe, se escribe «no se ha visto lo contrario».
- **Trampa:** los brazos no son simétricos (41 divergidos, todos SGD; norma del umbral sobre los dos juntos): una diferencia de tamaño se lee primero como población.
- **Trampa:** para las métricas la mayoría de los pares no tienen voz porque una D incluye el cero; los acuerdos son sobre 1 a 6 pares, no sobre 12.

#### La fase D lee H3 como orden de métricas con el signo mayoritario, y H3 cae: la alineación pone la mejor métrica y también la peor

**Qué se decidió (Lai delegó con la explicación leída).** El orden de H3 se calcula con la regla del protocolo, celdas con asociación del signo mayoritario y desempate por la mediana de |D| (`contrast.ranking_table`, `results/ranking.parquet`), con tres concreciones fijadas antes de calcular: el signo mayoritario es el que tiene la D en más celdas y un empate lo decide la mediana; la posición en la rejilla va como fila de referencia en el puesto que le tocaría, sin número; y la segunda mitad de H3, «el orden cambia de bando», se lee repitiendo la cuenta dentro de cada conjunto de datos y de cada arquitectura (`ranking_grupos.parquet`) y mirando solo la primera métrica de cada grupo. La poda se cuenta como resultado propio y sus tres parejas se vuelven a medir con el estadístico del contraste, la D dentro de celda al 5 % sobre los 806 (`efficiency.pair_agreement`, impresa por `efficiency.py`), en vez de copiar las Spearman del 2026-09-03. Las etiquetas de familia salen a `contrast.FAMILY` y `figures.py` las importa.

**Por qué.** La regla estaba escrita en §Protocolo desde el 2026-09-03 y usa las cifras de la tabla de H1, así que el orden se rehace a mano. Contar solo el signo mayoritario castiga a propósito a la métrica que cambia de sentido entre celdas. Medir la poda con D y no con Spearman la pone en la misma escala y el mismo umbral, 0,8, que la redundancia de H2.

**Evidencia, `results/ranking.parquet` y `ranking_grupos.parquet` del 2026-09-04, celdas a favor/en contra.** Velocidad: GWA 16/1 (0,40), stiffness 9/5 (0,35), disparidad 9/4 (0,28), posición 7/5, GSNR 7/6, GNS 7/3, confusión 5/4. Accuracy de test: GWA 15/1, disparidad 12/4, posición 7/6, GSNR 7/4, stiffness 5/6, confusión 5/5, GNS 3/6 (seis negativas con intervalo, mayoría cruda positiva 14/10). Gap: posición 15/0 (0,46), stiffness 15/1 (0,39), disparidad 11/4, GSNR 11/2, confusión 9/5, GNS 6/5, GWA 4/6. Gratuitos: 24, 18 y 16. Por arquitectura: fc, alineación en las tres (GWA, GWA, stiffness, 7/0 cada una); cnn, GSNR, disparidad, stiffness; resnet18, GSNR, GNS, GWA con el gap en signo positivo, 6/0, frente a 4/0 negativo en la cnn y nada en la fc. Por conjunto: MNIST y CIFAR-10 alineación en las tres; CIFAR-100 variabilidad en las tres; Tiny stiffness, disparidad, confusión. Poda: GNS y m-coherencia |D| 1,000 en las 24; NGV y GNS mediana 0,844, mínimo 0,731, 18 celdas ≥ 0,8 (contra GSNR 0,593 y ninguna celda); disparidad y √tr Σ 0,926, mínimo 0,766, 23 celdas ≥ 0,8.

**Veredicto.** H3 cae tal como se enunció: ninguna familia ocupa las primeras posiciones (alineación pone la primera en las tres variables y la última en dos; variabilidad la segunda en las tres) y el orden cambia de bando con la arquitectura y con el conjunto. Lo que la portada sostiene es que la métrica que más celdas ordena en cada variable es de alineación, GWA o stiffness, y eso es una afirmación sobre dos métricas. Escrito en `resultados.tex` §Qué familia predice mejor, tabla `tab:h3`, sin figura.

- **Trampa:** el orden lleva el learning rate dentro, igual que H1; ordena quién lo lee mejor, y lo que ordena sobre las 24 es sobre todo lo que pasa en la red plana y en los dos conjuntos fáciles.
- **Trampa:** la cuenta por grupo tiene 6 u 8 celdas y muchos empates a celdas; ahí decide la mediana de |D| y un ganador por un empate no es un hallazgo. Solo se lee la primera métrica del grupo.
- **Trampa:** la m-coherencia y la escala de ruido dan |D| = 1 exacto porque son la misma cantidad; una D de 1 es identidad, no «asociación perfecta».

#### La fase E compara ventanas con el mismo jackknife pareado, y H4 se sostiene porque las métricas no maduran

**Qué se decidió.** La diferencia |D| al 50 % menos |D| al 5 % se lee con el intervalo jackknife pareado de la fase C, porque las dos D salen de los mismos runs (`contrast.window_table`, `window_counts`, `results/ventanas.parquet` y `ventanas_recuento.parquet`); el protocolo decía «sumar los dos intervalos», y el pareado es más exacto sin ningún concepto nuevo. En velocidad la comparación es entre las dos lecturas por hitos, epoch 1 frente a 2 y 2 frente a 4, sobre poblaciones distintas, y ahí sí se suman las varianzas. Se calculan las tres variables del final y la velocidad; la tabla de la memoria da las dos primarias y la velocidad, y el gap de accuracy va en una frase. `primary_family` admite `window=None` y `incremental.parquet` lleva ya todas las ventanas, así que H2 al 50 % sale gratis.

**Evidencia, `results/ventanas_recuento.parquet` del 2026-09-04.** Accuracy de test: ninguna métrica crece con intervalo en más de 6 celdas (disparidad) ni decrece en más de 4; medianas de |D| 0,18 a 0,34 al 5 % y 0,21 a 0,35 al 50 %; la validación sube de 0,40 a 0,70 (22 crecen, 16 con intervalo) y TSE de 0,33 a 0,50 (18, 13). Gap: la validación pierde |D| al esperar (val_acc 0,53 a 0,33, 20 decrecen, 8 con intervalo); stiffness 0,39 a 0,51 y confusión 0,31 a 0,41 con 5 y 4 celdas con intervalo; TSE supera a val_loss al 50 % 16/8 (9/3). Velocidad: 0 a 2 celdas con intervalo por predictor; en crudo la disparidad crece en 19 (0,28 a 0,44) y la confusión en 17; en riesgo al 10 %, de 8 a 40. H2 al 50 % sobre el accuracy de test: GWA 1/23, ninguna métrica gana en más de 2.

**Veredicto.** H4 se sostiene para las métricas: su |D| no crece con la ventana. La ventana mínima defendible es la primera, y no porque maduren pronto sino porque no maduran; la referencia gratuita sí madura hasta 0,70. Escrito en `resultados.tex` §Análisis agregado y barrido temporal.

- **Trampa:** al 50 % la validación es casi el resultado, así que su crecimiento es trivial y no es «suficiencia temprana» de nada.
- **Trampa:** en el gap los predictores gratuitos pierden |D| al esperar porque el gap es velocidad al revés y la lectura temprana es la que la recoge; no leer eso como «la validación temprana predice el gap».
- **Trampa:** la figura `ventanas` lleva diez líneas por panel; se sostiene por color de familia y marcador, y la afirmación que defiende es una sola, las gratuitas suben y las métricas se quedan planas.

#### La fase C fija tres reglas de lectura de H2 antes de calcular su tabla, y H2 cae

**Qué se decidió (Lai delegó las tres a esta sesión, con la explicación leída).** Superar a la referencia se lee con el intervalo de la diferencia |D| de la métrica menos |D| de la referencia, por jackknife pareado sobre los mismos entrenamientos (`efficiency.d_diff_stats`, columnas `D_diff`, `se_diff` y sus `_land`), con el recuento crudo de celdas al lado como descripción; redundante es |D_ref| ≥ 0,8, el mismo umbral que la poda (`contrast.REDUNDANT_D`); la pérdida por selección se lee por mediana y por celda, con el signo del artículo como primaria y el signo invertido solo como exploratorio (`regret_flipped` en `seleccion.parquet`). Fijadas con H1 leída y la tabla de H2 sin calcular; los medianas de pérdida se habían visto de reojo el 2026-09-03 y así se declara.

**Por qué.** Contar celdas donde |D| supera a |D| trata igual una diferencia de 0,02 que una de 0,4, y el jackknife que ya existía da el intervalo sin ningún concepto nuevo. El 0,8 ya estaba justificado como «nueve de cada diez pares concuerdan». El signo del artículo es la única regla previa a los datos; el invertido es post hoc y se etiqueta así.

**Evidencia, `results/incremental.parquet` y `seleccion.parquet` del 2026-09-04, familia primaria.** Accuracy de test: GWA gana 11 y pierde 13 en crudo, 0 y 8 con intervalo; disparidad 10/14 y 3/8; las otras cuatro ganan 5 o 6; mediana de |D_ref| entre 0,23 y 0,45 y ninguna celda ≥ 0,8; sin MNIST, GWA 9/9 y 0/5. La posición en la rejilla también 11/13. Las dos lecturas de validación son intercambiables (|D_ref| 0,93, 24 celdas ≥ 0,8) y TSE es redundante con val_acc en 17 y pierde 4/19. Gap: referencia |D| mediana 0,54 frente a 0,21 a 0,39 de las métricas; stiffness 7/17 (2/6), GWA 5/19 (0/14); TSE supera a val_loss 14/9 (3/1). Selección: val_acc 0,013, val_loss 0,015, TSE 0,015, GWA 0,023 (9 celdas mejor que val_acc, 2/3/1/3 por conjunto), GSNR 0,035, confusión 0,059, azar 0,070, GNS 0,073, disparidad 0,097, stiffness 0,110; invertido, disparidad 0,023, GNS 0,066, stiffness 0,075, y GWA, GSNR y confusión suben a 0,084 o más. Velocidad, no decisoria: métricas 0 a 4 celdas ganadas, 12 a 19 perdidas con intervalo; val_loss empata con val_acc 10/11.

**Veredicto.** H2 cae: ninguna métrica supera a su referencia con intervalo en más celdas de las que pierde, en ninguna de las dos variables que deciden, y todas pierden más accuracy que la validación al elegir. Escrito en `resultados.tex` §Valor incremental y §Coste y capacidad predictiva.

- **Trampa:** la disparidad con el signo invertido iguala a GWA (0,023); es post hoc y coincide con la D positiva de H1, se dice en una frase y no decide nada.
- **Trampa:** «no redundante y aun así pierde» no es contradictorio: las métricas ordenan distinto que la validación y peor; lo que comparten con ella es el learning rate, y la posición en la rejilla empata con la mejor.
- **Trampa:** TSE supera a val_loss en el gap (14/9); los dos son gratuitos y los dos son lecturas de la velocidad, así que no es valor incremental de ninguna métrica.
- **Trampa:** la selección es top-1 entre ocho learning rates con la población de los 806; un run muerto sería la peor elección y no entra.

### 2026-09-03

#### Diez desviaciones del método tras la revisión externa, posteriores a la tabla larga y a la primera lectura de H1

**Qué se decidió (Lai, a propuesta de esta sesión, tarde del 2026-09-03).** Diez cambios sobre el método fijado por la mañana y cinco cosas que se dejan como estaban. Salen de la revisión externa de la memoria (`docs/research/writing/Revisión externa de la memoria.md`, gitignorada), hecha como revisor de ML ajeno al proyecto y sin calcular ninguna correlación entre una métrica y una variable dependiente.

1. **Suavizado.** `train.median3` pasa a mediana de tres donde hay tres *epochs* y valor crudo en la primera y en la última. La versión anterior tomaba la media de dos en los bordes, que cruza el umbral cuando ninguna de las dos *epochs* lo cruza. Todo lo que depende del suavizado se recalcula en la capa de análisis desde `trajectory.parquet`: VD1, VD3 (`best_val_loss`) y `best_val_acc`; los tres campos suavizados de `summary.json` quedan obsoletos, no solo `epochs_to_threshold`.
2. **H2 contra una referencia nombrada de antemano.** El *accuracy* de validación en la ventana para el *accuracy* de test y el *loss* de validación en la ventana para el *gap*; TSE y la otra lectura de validación se publican al lado sin elegir el máximo. Junto a cada comparación va `D_ref`, la D entre la métrica y su referencia dentro de la celda, que dice si la métrica es la referencia con otro nombre. Sustituye a «el mejor de tres», que es el máximo de tres números con ruido y sube la vara por construcción. La promesa de «descontar» de OE2, del Estado del arte y de H2 se reescribe como «supera a la referencia y no es redundante con ella».
3. **Lectura de selección para el *accuracy* de test.** En cada celda y *seed*, el *learning rate* que la métrica al 5 % señala como mejor, tomando como bueno el extremo que su artículo llama bueno (tabla de signos de [[Datos experimentales]] §5.3, GWA con el signo convertido), frente al mejor *learning rate* real de esa *seed*, en puntos de *accuracy* de test perdidos; lo mismo eligiendo con el *accuracy* de validación al 5 %, y la elección al azar como vara. Es el protocolo de estimación de rendimiento de NAS y HPO, responde a la pregunta de la Introducción en sus unidades y no depende de la forma pico o valle. Solo para el *accuracy* de test. `contrast.selection_regret`.
4. **La posición en la rejilla como predictor de coste cero.** `log_lr`, el logaritmo del *learning rate*, entra como predictor en la tabla larga y en las figuras. Es el techo de cualquier lectura monótona del *learning rate*: una métrica que no lo supera es el *learning rate* leído de otra forma.
5. **Incertidumbre por celda, y agregación sin prueba binomial ni Benjamini-Hochberg.** Cada D lleva un error típico por *jackknife* (quitar un run, recalcular, la dispersión de las réplicas) y un intervalo normal al 95 %. Una celda muestra asociación si el intervalo excluye el cero. H1 se lee como cuántas celdas de 24 la muestran y con qué signo, con el desglose por conjunto de datos; el recuento de signos queda como descripción de consistencia y la referencia binomial (18 de 24, 10 de 12) se cita solo como vara. Sin valores p no hay nada que corregir, así que Benjamini-Hochberg sale; la multiplicidad se controla con la familia primaria pequeña y etiquetando el resto como exploratorio. La figura de signos gana los intervalos.
6. **H5 y H6.** H5 es la diferencia pareada D con SGD menos D con Adam por par de celdas, con su intervalo; una inversión se cuenta solo cuando las dos D excluyen el cero y difieren de signo, y un no rechazo no prueba invariancia. H6 es el acuerdo del signo con el predicho, a una cola, contado solo sobre las celdas cuyo intervalo excluye el cero y solo para las predicciones que el artículo enuncia. Para la alineación, donde la *seed* no mueve la métrica, la conclusión sobre «más allá del *learning rate*» se escribe «no identificable con este diseño» y nunca «sin señal».
7. **Cuenta granulada estratificada.** Concordantes menos discordantes sumados sobre todos los pares dentro de un *learning rate*, dividido por el total de esos pares, con los *learning rates* de al menos tres runs como hasta ahora, y con su número de pares y su *jackknife*. La media de las D por *learning rate* pesaba igual un grupo con un par y otro con diez.
8. **Poda completa.** Sale también `var/normalized` (NGV): es la escala de ruido estimada con diez submuestras de 25 en vez de con los 256, y la regla de la mañana la comparó con GSNR y nunca con la escala de ruido. La *gradient disparity* pasa a la familia de variabilidad en `analysis.SPECS`, porque es √(2·tr Σ/51). Quedan seis métricas de gradiente: escala de ruido, GSNR y disparidad en variabilidad; *stiffness* intra-clase, *gradient confusion* y GWA en alineación.
9. **Velocidad por hitos.** Para VD1, en cada ventana, la D se calcula también solo entre los runs que aún no habían cruzado al cerrar la ventana, censurados incluidos, con el número en riesgo al lado (`D_land`, `n_land`); esa es la lectura primaria en las ventanas del 5 y el 10 %, y la D sobre todos los runs se publica como secundaria. Así «ya cruzó» no cuenta como predicción, y la regla de la mitad por delante deja de decidir qué celdas se leen. H4 sobre la velocidad se declara en *epochs* absolutas: 1 frente a 2 en MNIST y 2 frente a 4 en los demás.
10. **Población y variables.** Los 806 siguen como población primaria. La pasada con los 960 se conserva en `results/` pero no entra en la memoria como sensibilidad, porque no es un contraste más estricto sino otro distinto: los colapsados llevan constantes y GWA a cero exacto. VD3 pasa al constructo de rendimiento final como lectura de apoyo. El suelo del *gap* deja 693 de 806 con mínimo 20 por celda.

**Lo que se deja como estaba, y por qué.** No se niegan las columnas de GWA para recuperar la convención del artículo: la convención actual está escrita de forma coherente en seis sitios y cambiarla con dos sesiones abre más errores de los que cierra; se escribe en Fundamentos su forma cerrada como margen normalizado. No se añaden variables dependientes nuevas, porque la multiplicidad ya es el problema. No se cambia la columna titular de la *stiffness*: el coseno global a normas iguales es (α − 1)/(M − 1) y sería redundante con la escala de ruido. No se comprime Resultados ahora: se retiran solo las figuras `solape-bandas` y `solape-cruces` y la compresión queda para la fase H (esa misma noche, a petición de Lai, se retiró también `val-test`, se borró `solape-mapa` del todo porque no se entendía en ninguna de sus dos versiones, y se recortaron §Concordancia, §Rango dinámico, §Riesgos, §Protocolo, §Calibración, §Cálculo de las métricas y el anexo de configuración, de 101 a 96 páginas; la proporción de Resultados se juzga en la fase H). No se reabre el micro-benchmark de coste (decisión de Lai del 2026-08-27); se añade el sobrecoste medido por celda.

**Cronología, para declararla en la memoria.** Método fijado por la mañana. Tabla larga, figura `signos-h1` y lectura de H1 en `resultados.tex` §Correlaciones por condición (recuento de signos con Benjamini-Hochberg) por la tarde, del flujo de análisis. Revisión externa hecha sin calcular ninguna correlación métrica-VD; sus motivos son los de arriba, todos de estructura. Las diez desviaciones se decidieron después de conocer esa primera lectura, y así se declara en `metodologia.tex` §Cuándo se fijó y en la fase H, con Willroth y Atherton 2024 como forma de reportarlas. Ninguna se justifica por el valor de un coeficiente.

**Evidencia medida sobre `reports/`.** Suavizado: los 12 umbrales de la norma del 60 % no cambian; los 806 que aprendieron siguen siendo 806 pero cambia uno, sale `fc_cifar100_sgd_lr0.3_seed2` (cociente 1,14) y entra `fc_cifar100_adam_lr0.01_seed4` (1,32), los dos colapsados, y 1,2 y 1,25 ya no seleccionan el mismo conjunto (153 frente a 154); los cruces pasan de 571 a 573 y 17 runs cambian de *epoch*; los cruces en la *epoch* 1 bajan de 17 a 2; `best_val_acc` cambia en 270 runs y `best_val_loss` en 213 (mediana del cambio relativo 0,2 %, máximo 29 %; en 64 cambia la *epoch* del mínimo); la fracción de cruces consumida al cerrar el 5 % baja del 10,7 al 10,3 % y la del 10 % queda en el 34,6 %. Poda: NGV contra escala de ruido, Spearman 0,94 dentro de celda al 5 % (mínimo 0,79) y −0,94 contra la m-coherencia; disparidad contra `noise_scale/tr_sigma`, Spearman 0,98, cociente medido 0,038 frente a 0,039 teórico. GWA contra el *accuracy* de validación al 5 %: −0,83. Techos de los estimadores: la escala de ruido satura en M − 1 = 255 con mediana 207; NGV satura en K = 10 con mediana 8,1.

- **Trampa:** el *jackknife* supone runs independientes, que lo son porque las *seeds* son independientes y el *learning rate* es un factor fijo del diseño; el intervalo dice cuánto se movería la D con otras *seeds* sobre la misma rejilla y nada sobre otra rejilla, otro conjunto ni otra arquitectura.
- **Trampa:** un intervalo que excluye el cero no dice que la métrica añada nada al *learning rate*; para eso están `log_lr`, `D_ref` y la cuenta granulada.
- **Trampa:** las ventanas de velocidad siguen siendo las *epochs* 1 y 2 en MNIST y 2 y 4 en los demás; la lectura por hitos no cambia eso.
- **Trampa:** la convención de signo de GWA no cambia; toda predicción heredada de su artículo se sigue leyendo con el signo convertido.

#### La poda deja la variabilidad en tres cantidades, la regla se lee en valor absoluto y de la pareja idéntica se queda la escala de ruido

**Superada esa misma tarde por la desviación 8 de la entrada anterior: NGV sale también, porque es la escala de ruido con diez submuestras y esta regla nunca la comparó con ella.** Se conserva porque `metodologia.tex` §Poda declara que un primer criterio comparó NGV solo con GSNR.

**Qué se decidió.** Tres cosas al ejecutar el paso 1 de la fase B. La regla de poda entre NGV y GSNR se lee sobre el valor absoluto de la D: se queda una si la mediana de |D| sobre las 24 celdas alcanza 0,8. Con los datos no se alcanza, así que NGV y GSNR entran las dos en el contraste y la familia de variabilidad queda en tres cantidades, NGV, escala de ruido y GSNR. De la pareja idéntica sale la m-coherencia y se queda la escala de ruido; la predicción de signo del artículo de la m-coherencia (H6) se lee sobre la escala con el signo cambiado, porque GNS = M/α − 1 es decreciente en α.

**Por qué.** Para un estadístico de rangos, dos métricas que ordenan los runs exactamente al revés llevan la misma información que dos que ordenan igual, y GSNR y NGV ordenan al revés por construcción. La escala de ruido se queda porque es la métrica barata del nivel 1 de [[1 - Diseño]] y la m-coherencia la retadora del nivel 2, y una retadora que es la barata con otro nombre no reta nada.

**Evidencia, sobre los 806 que aprendieron, ventana del 5 %, celdas de 24.** D entre NGV y GSNR: mediana −0,59, mediana de |D| 0,59, negativa en 16 celdas y ninguna con |D| ≥ 0,8; las ocho positivas son CIFAR-100 y Tiny-ImageNet con la red *fully connected* y la convolucional, donde las dos ni siquiera coinciden en la dirección. D entre escala de ruido y m-coherencia: exactamente −1 en las 24, que confirma la identidad sobre los datos y valida `efficiency.somers_d`. El recuento de formas de la cuarta vía, ya en `efficiency.shape_census`, reproduce los números de la entrada de las nubes; `declared_cells` devuelve 211 ternas celda, predictor y variable, en 47 pares distintos de celda y predictor. Suite en 285 pruebas.

- **Trampa:** la poda se calcula sobre los que aprendieron y no sobre los 960; un run clavado en el azar aporta una constante y una constante solo fabrica empates.
- **Trampa:** el signo dispar entre celdas es una observación para Resultados, no un argumento a favor ni en contra de la poda.
- **Trampa:** `somers_d` vive en `efficiency.py` y el módulo de contraste del paso 2 lo importa de ahí; no reescribirlo.

#### El resto del proyecto va en tres tandas con dos flujos, y §Protocolo de análisis lo escribe la fase B para los seis objetivos

**Qué se decidió (Lai, a propuesta de esta sesión).** Tanda 1, desde hoy: el flujo de análisis, esta sesión, abre la fase B; el flujo de memoria, la otra sesión, hace §Calibración del *pilot*, la revisión de Metodología, Fundamentos, Estado del arte, Introducción, Desarrollo y Trabajos futuros, el diagrama del *pipeline* y los dos anexos. Tanda 2, cuando exista la tabla larga de la fase B: una sesión hace las fases C y E, que calculan algo nuevo, y la otra D, F y G. Tanda 3, en serie: fase H y el resto de la I. §Protocolo de análisis lo escribe entero la fase B, porque las elecciones sirven a los seis objetivos; C a G solo escriben su sección de Resultados.

**Reparto de ficheros.** El flujo de análisis es dueño del código nuevo en `src/` y `tests/`, de §Protocolo de análisis y de las secciones de `resultados.tex` de la fase B en adelante. El de memoria, de los demás capítulos, de `main.tex`, de los anexos y de §Calibración del *pilot*. Los dos escriben en el vault y en `figures.py` solo añadiendo, releen el fichero antes de cada edición y commitean solo sus ficheros.

**Por qué.** La fase B produce un dato, la tabla larga, y las ramas solo lo leen; el árbol de git es uno solo y sin ramas, así que el paralelismo se sostiene sobre la propiedad de ficheros. La tanda 1 se cerró el 2026-09-03 y todo quedó commiteado esa noche (`ab1755d`, `a6b08cd`, `86b3880`); Lai decide si la tanda 2 va en una sesión o en dos.

- **Trampa:** «una cosa cada vez» sigue rigiendo dentro de cada sesión, y el cuello de botella es Lai, que confirma cada paso de los dos flujos.
- **Trampa:** el anexo ODS tiene dos entregables, el `.tex` y el `.docx` de la plantilla oficial para Ebrón (tabla de los 17 con alto, medio, bajo y no procede, y reflexión de 500 a 1.500 palabras); los dos existen desde el 2026-09-03 y el `.docx` se regenera con `thesis/anexo-ods-docx.py` si cambia el `.tex`.

#### El método de la fase B queda fijado para los seis objetivos, antes de calcular ningún coeficiente

**Es el método anterior a cualquier coeficiente, y cinco de sus ocho elecciones cambiaron esa misma tarde** (entrada «Diez desviaciones»): la 3 pasó a granulada estratificada, la 4 a intervalos por *jackknife* sin prueba binomial, la 5 perdió Benjamini-Hochberg, la 6 sacó también NGV y la 7 pasó a la referencia nombrada con `D_ref` y la lectura de selección. Se conserva entera porque la cronología es la evidencia de la fase H.

**Qué se decidió.** Ocho elecciones, cada una con su base, cierran la pendiente del 2026-08-25.

1. **Población.** Los 806 que aprendieron son la población primaria de todo contraste; los 960 se reportan como sensibilidad. Base: Jiang et al. 2020 descartan los modelos que no alcanzan su criterio de entrenamiento; aquí, con los 154 muertos dentro, cualquier contraste detecta muerte, que es binaria y que la validación temprana detecta sin coste.
2. **Estadístico intra-celda.** Concordantes menos discordantes entre pares comparables, la D de Somers, que es la C de Harrell reescalada, C = (D + 1)/2, y coincide con tau donde no hay empates ni censura. Comparable: valores distintos de la variable dependiente; en la velocidad, además, al menos uno de los dos cruzó, y un cruce en la última *epoch* contra un censurado es comparable con el cruce como más rápido. Base: Harrell et al. 1982 y 1996, Newson 2002. Sin valor p por celda, porque los 40 no son intercambiables, están estructurados por *learning rate*; por celda va el tamaño de efecto con su número de pares.
3. **Cuenta granulada.** La misma D entre las cinco *seeds* de un mismo *learning rate*, promediada sobre los *learning rates* de la celda con al menos tres runs. Es el coeficiente granulado de Jiang et al. 2020 y la única cuenta que no es *learning rate* leído de otra manera.
4. **Agregación.** Las 24 D siempre a la vista y un recuento de signos con referencia binomial, sin supuesto sobre la distribución de las 24. Verificado con `scipy`: 18 de 24 baja del 5 % a dos colas (p = 0,023) y 17 no (0,064); para los 12 pares de OE5, 10 de 12 (0,039) y 9 no (0,146). Se declara que las 24 no son independientes y se enseña la consistencia dentro de cada conjunto de datos.
5. **Familia primaria.** Ventana del 5 %, una variable por constructo (*epochs* hasta el umbral, *accuracy* de test, *gap* de *loss*) y las métricas que sobrevivan a la poda, con Benjamini-Hochberg 1995 dentro de la familia de cada objetivo. Todo lo demás, ventanas y variables de apoyo, exploratorio y etiquetado.
6. **Poda.** Sale una de escala de ruido y m-coherencia por identidad exacta. NGV contra GSNR se decide con la D entre las dos dentro de cada celda al 5 %: si la mediana sobre las 24 alcanza 0,8, nueve de cada diez pares concuerdan y se queda una. Con la familia de variabilidad reducida, H3 pasa a ranking de métricas individuales con la familia como etiqueta. **Ejecutada el mismo día** (entrada de arriba): la regla se lee en |D|, no se alcanza, 0,59, y entran las dos; sale la m-coherencia y se queda la escala de ruido.
7. **OE2.** Comparación pareada por celda del valor absoluto de la D de la métrica contra el mejor de los tres predictores gratuitos de esa celda, con recuento de signos; secundaria, la misma comparación sobre la cuenta granulada. Se retira la correlación parcial de rangos. En velocidad se declara que la referencia es un prefijo de la propia variable y H2 se decide sobre *accuracy* de test y *gap*.
8. **Suelo del gap.** Entran en los contrastes de VD5 y VD6 los runs cuya *accuracy* de train final alcanza el umbral τ de su celda. Propuesta propia: Jiang et al. paran todos los modelos en un mismo *loss* de train para que el propio *loss* no prediga, y este suelo hace lo mismo dentro del presupuesto fijo. Falta medir cuántos runs deja por celda.

**Por qué así y no como se propuso el 2026-08-31.** El test de signos se explica en una frase y no supone nada sobre las 24; la cuenta granulada responde mejor que una parcial a «qué aporta la métrica descontado el *learning rate*», que es lo que la referencia codifica.

**Evidencia previa, medida el 2026-09-03 sobre los que aprendieron.** La *seed* apenas mueve las métricas de alineación (0,03 a 0,08 frente a 0,13 de referencia) y algo las de variabilidad (0,17 y 0,12), así que la cuenta granulada solo tiene con qué trabajar en variabilidad, predicción registrada antes de calcularla; empates en la velocidad, mediana del 13 % de los pares por celda y máximo 44 %. Las dos cifras están en [[3 - Progreso]] §Vigentes.

- **Trampa:** toda D se calcula desde `trajectory.parquet` y `summary.json` con VD1 recalculada; `epochs_to_threshold` del resumen es obsoleto.
- **Trampa:** TSE es una suma acumulada, 8,2 al 5 % en un run cuyo *loss* es 3,7; a ventana fija su orden es el de la media del *loss* de las primeras *epochs*. Es *loss* temprano con otro nombre.
- **Trampa:** H1 será casi trivial por el *learning rate*, que explica 0,9 de la dispersión; se presenta como puerta y no como hallazgo.
- **Trampa (superada esa tarde por la lectura por hitos):** los runs ya cruzados dentro de una ventana que sirve se quedaban, con la fracción consumida declarada junto al coeficiente.
- **Trampa:** en MNIST el *accuracy* de test no tiene recorrido entre los sanos (2026-09-02).

#### La decisión de las nubes de puntos se cierra sin mirar ninguna: la forma se lee lado a lado a lo largo del *learning rate*

**Qué se decidió.** No se mira ninguna nube de métrica contra variable dependiente antes de fijar el estadístico. En su lugar se clasifica la forma de cada lado por separado a lo largo de los ocho *learning rates*, con la mediana de las *seeds* que aprendieron y solo los *learning rates* con al menos tres, en cinco clases: sube, baja, pico, valle e irregular, con una tolerancia del 5 % del recorrido para que una diferencia de ruido no cuente como cambio de sentido. Dentro de una celda las dos cosas son funciones del *learning rate*, así que la relación entre ellas es la composición de las dos curvas, y una métrica monótona con una variable en pico o valle no puede dar una relación monótona. Esas celdas se declaran por métrica antes de calcular, y no se cambia de estadístico después. El recuento vive hoy en la carpeta temporal y entra en `efficiency.py` con prueba en el primer paso de código de la fase B; su párrafo va a §Protocolo de análisis.

**Por qué.** Mirar las nubes gasta la única protección que queda, que el método se fije antes de ver ningún coeficiente métrica-VD. Mirar cada lado contra el *learning rate* no la gasta, es lo que la fase A ya hacía, y basta para saber dónde tau no puede ver nada.

**Evidencia, sobre los que aprendieron, celdas de 24.** Variables dependientes: *epochs* hasta el umbral valle en 19 (censurados colocados como los más lentos), mejor *loss* valle en 19, *accuracy* de test pico en 17, *gap* de *loss* pico en 13, *gap* de *accuracy* pico en 14, AUC valle en 13. Predictores al 5 %: *loss* de validación valle en 21, *accuracy* de validación pico en 20, TSE valle en 19, *gradient disparity* pico en 17, *stiffness* valle en 7 y baja en 7, y las otras seis irregulares en la mayoría (NGV 18, GNS 16, m-coherencia 16, GSNR 11, confusión 11, GWA 11). Celdas donde la composición no puede ser monótona: entre 1 y 7 por métrica, la peor la *stiffness*, 6 con la velocidad y 7 con el *accuracy* de test. Con tolerancia del 10 % los recuentos se mueven entre una y tres celdas.

- **Trampa:** el valle de la velocidad es en parte convención, porque los censurados van como los más lentos y quedan en los dos extremos de la rejilla.
- **Trampa:** la tolerancia del 5 % la fija esta entrada; cambiarla exige rehacer el recuento y decirlo con fecha.
- **Trampa:** los tres predictores gratuitos tienen la forma exacta de la variable dependiente y las métricas de gradiente no. Es H2 vista desde un solo lado.

#### El mapa de solape baja a la variable principal, y el volumen de cruces sale a figura aparte

**Qué se decidió.** `solape-mapa` pasa de tres paneles a uno, solo VD1, porque el panel del AUC enseñaba una supervivencia que el texto tenía que desmentir. Se añaden `solape-cruces` y `solape-bandas`, que cuentan entrenamientos y sirven para dimensionar, no para decidir, y `solape-celda` gana un panel con los cruces acumulados.

**Superada esa misma noche.** `solape-cruces` y `solape-bandas` salieron de la memoria con las diez desviaciones, y `solape-mapa` se borró del todo, función, prueba y PDF, por decisión de Lai, porque no se entendía ni como mapa de calor ni como un punto por celda sobre el eje. Los recuentos por celda y ventana viven en la prosa de §Solape, y la única figura de esa sección es `solape-celda`.

#### La columna titular de cada predictor pasa a la memoria, tal como estaba marcada en el código

**Qué se decidió.** Las once columnas que `analysis.py::SPECS` marca como `headline` (las ocho métricas, TSE con `tse/ema_0_999`, y `val_loss` y `val_acc`) se escriben en `metodologia.tex` §Variables como la Tabla «Columna titular de cada predictor», y Resultados §Rango dinámico las cita por esa tabla. Sin cambiar ninguna: son las que usó el punto 3 de la fase A y las que dibuja la figura de rango por columnas.

**Por qué.** Resultados usaba «columna titular» sin definirla y Metodología decía que la elección quedaba para el protocolo de análisis, que no existe. La elección estaba en el código desde antes de la fase A y no en este registro.

- **Trampa.** El protocolo de la fase B hereda la lista. Cambiar una columna obliga a rehacer el punto 3 y su figura, y a decirlo aquí con fecha.

### 2026-09-02

#### El punto 5 compara validación y test al final del entrenamiento, en orden con tau y en nivel contra el ruido binomial

**Qué se decidió.** Solo coexisten al final, así que ahí se comparan `final_val_acc` con `final_test_acc` y el `val_loss` de la última *epoch* con `final_test_loss`. Orden: tau-b de Kendall por celda. Nivel: validación menos test por run contra la desviación típica binomial de dos *accuracies* sobre los tamaños de las particiones, `NOISE_SIGMAS = 2`; los tamaños suben a `config.py::VAL_SIZE` y `TEST_SIZE`. Fuera los 41 divergidos. Dos pasadas. Vive en `efficiency.py::val_test_agreement` y `agreement_summary`. Cierra de paso la nota de honestidad F1 macro ≈ *accuracy* de la fase H. Método, evidencia y lecturas en `resultados.tex` §Concordancia entre validación y test.

**Por qué esas dos cosas.** El orden es lo que consume el estadístico de la fase B; el nivel es lo que hace que un umbral fijado sobre validación signifique lo mismo sobre datos nuevos. No entra ninguna métrica, así que es la correlación entre dos VD que la fase A tiene permitida.

**Evidencia, sobre los 919 con medición.** Tau del *accuracy* por celda entre 0,71 y 0,97, medianas por conjunto de 0,84 (MNIST) a 0,93; nivel a cero, con el 95 % de los runs a menos de 0,011 de la diagonal y el más lejano a 0,019 (corregido el 2026-09-03: la cifra «entre −0,013 y 0,013» escrita el día 2 no se reproduce sobre ninguna población); F1 macro menos *accuracy* con mediana 0,002 y máximo 0,048 entre los 790 runs cuyo mejor val-*accuracy* supera tres veces el azar. Solo con los que aprendieron, las medianas de tau bajan entre 0,01 y 0,04.

- **Trampa para la fase B: en MNIST, VD4 no tiene recorrido entre los sanos.** La tau mínima, 0,71, sale de 40 runs que terminan en un tramo de 0,008 de test, el tamaño del ruido; solo con los que aprendieron, cinco de las seis celdas de MNIST terminan en menos de 0,05. Cualquier correlación con VD4 en MNIST es en buena parte correlación con ruido de medida.
- **Trampa: la referencia binomial es una cota, no una predicción.** Supone cada ejemplo como una moneda independiente, y los runs clavados tienen *accuracy* fijada por la estratificación, sin varianza; por eso el observado queda por debajo del 4,6 % esperado.
- **Trampa: el acuerdo está comprobado al final y solo al final**, porque el test se mira una vez. Se declara así en la memoria.
- **Una figura y no dos, y desde el 2026-09-03 ninguna.** `val-test` se retiró de la memoria esa noche porque comprueba algo que el diseño garantiza; la sección quedó en tres párrafos con sus números, y la función sigue en `figures.py`.

#### El punto 4 mide el solape en pares, sobre las tres variables de velocidad, y fija la mitad como regla

**Qué se decidió.** Una ventana se cierra en la *epoch* en la que se lee el predictor y un suceso en esa misma *epoch* cuenta como ya ocurrido. Suceso de VD1, el cruce; de VD3, la *epoch* del mínimo del *loss* suavizado; VD2 no tiene suceso y se mide la parte del área que la ventana aún no fija. Lo que sigue por delante se cuenta en pares comparables, con la fórmula de la censura del 2026-08-31. **Regla:** una ventana sirve para una variable en una celda cuando al menos la mitad sigue por delante, `efficiency.py::AHEAD_FLOOR`. Dos pasadas. Vive en `efficiency.py::window_overlap`, `overlap_summary` y `vd1_consumed_pooled`, con la ventana leída de `metrics_at_window.parquet`. Método, evidencia y lecturas en `resultados.tex` §Solape entre la ventana y el desenlace.

**Por qué la mitad, y cuándo se fijó.** Tolerancia cero vacía el mapa por un solo run madrugador, y sin regla se incumple lo que `metodologia.tex` §Riesgos promete. La regla se fijó después de conocer las cifras agregadas del 2026-09-01 y antes del desglose por celda, así que no es ciega y la memoria lo dice.

**Evidencia, sobre los 960.** Celdas que conservan al menos la mitad, por ventana: VD1 22, 12, 0 y 0; VD3 23, 17, 9 y 2; VD2 24, 24, 24 y 13. Solo con los que aprendieron, VD1 y VD2 no cambian y VD3 da 23, 16, 9 y 3. Consecuencia: para la velocidad, OE4 queda acotado a las ventanas del 5 y el 10 %.

- **Trampa: la cuenta es un suelo.** Un run al 95 % del umbral cuenta como por delante. El solape real es mayor.
- **Trampa: VD2 al 50 % cae sobre la línea por construcción**, 0,51 frente al 0,50 de una curva plana, y se declara «descrita a medias».
- **Trampa: VD3 mide el sobreajuste donde el mínimo llega pronto.** En CIFAR-10 con ResNet-18 y Adam, 37 de 40 runs tocan fondo en la *epoch* 10 o antes (corregido el 2026-09-03; aquí decía «los 40 antes de la 10»).
- **Trampa: el suavizado mira una *epoch* adelante.** La mediana centrada usa la siguiente, luego VD1 lleva una *epoch* de anticipación dentro. Manda la coherencia con VD1 tal como está definida.
- **Trampa: las ventanas de MNIST son las *epochs* 1, 2, 5 y 10 y las demás 2, 4, 10 y 20.** Comparar el solape entre conjuntos mezcla tiempo absoluto y relativo; dentro de una celda es constante.
- **Lo que no decidía**, qué hacer con los runs ya cruzados dentro de una ventana que sirve, lo cerró la lectura por hitos del 2026-09-03; desde entonces la regla de la mitad describe y no decide.
- **Cuatro figuras, por decisión de Lai:** `solape-celda`, `solape-cruces`, `solape-bandas` y `solape-mapa`. Solo `solape-celda` queda en la memoria desde el 2026-09-03; ver la entrada de esa fecha.

#### El estilo de figuras pasa a color de artículo, y se retiran tres restricciones

**Qué se decidió (Lai).** Salen de `figstyle.py` la separación mínima de luminancia, el emparejamiento color-trazo y la tipografía del cuerpo dentro de las figuras. Paleta **Okabe-Ito** y tipografía **TeX Gyre Heros**. Se mantiene lo que no es estilo: ancho final sin escalado, prohibición de `bbox_inches="tight"`, salida en PDF, eje hasta el cero y escala compartida entre paneles.

**Por qué.** El depósito por Ebrón es digital y Lai confirmó que nunca habrá entrega en papel monocromo, así que el color puede informar solo. Efecto lateral: con la sans y `mathtext.fontset` en `stixsans` desaparece el aviso de tipografía y vuelve la cursiva parcial dentro de un rótulo.

- **Regla nueva: una figura es autocontenida.** Dentro no entra ningún identificador del código ni abreviatura que no esté en la memoria, y los nombres son los del cuerpo. Salió de dos defectos reales, un rótulo «LR» y once barras con las claves del parquet.
- **Trampa:** `tests/test_figstyle.py` ya no vigila la luminancia; vigila que el ciclo asigne la paleta en su orden declarado, porque el color sigue al concepto y nunca al rango.

#### El punto 3 mide el rango dinámico con eta cuadrado sobre puestos, y no entra ninguna biblioteca nueva

**Qué se decidió.** Dentro de cada celda, columna y ventana temprana, los valores se sustituyen por su puesto y la dispersión se parte entre grupos y dentro de grupos, una vez por learning rate y otra por semilla, contra la referencia exacta (k−1)/(n−1). Vive en `analysis.py::dynamic_range_report` y `dynamic_range_summary`. Dos pasadas. Método, evidencia y lecturas en `resultados.tex` §Rango dinámico del predictor.

**Por qué un tamaño de efecto y no una prueba.** La pregunta es cuánto sitio tiene el predictor para moverse; un valor p serían 2.592 contrastes sin que ningún objetivo los pida. El cociente es el ε² de Kruskal-Wallis, así que `scipy` solo aportaría ese valor p.

**Evidencia, ventana del 5 % y solo los que aprendieron.** Las ocho métricas superan su línea, de 0,45 a 0,92; de las 768 casillas solo 29 caen en la línea o por debajo, todas de variabilidad o de m-coherencia. La causa dos de un coeficiente nulo queda descartada en casi toda la rejilla. Las dos trampas que condicionan las fases B y C, que el predictor de referencia gana en rango dinámico (0,97 y 0,95 frente a 0,92) y que el learning rate es causa común de métrica y variable dependiente, están en [[3 - Progreso]] §Vigentes.

- **Trampa: una columna constante da cero partido por cero y devuelve NaN a propósito.** Con un cero, una celda muerta se leería como una celda donde manda la semilla.
- **Trampa: la celda de ejemplo de la memoria usa los 39 runs con valor y la figura de 24 celdas solo los que aprendieron.** Son poblaciones distintas y la memoria lo dice desde el 2026-09-03.
- **Cambio de plan, con fecha.** La fase A decía que solo el punto 2 llevaba figura; el punto 3 se lleva dos, `rango-celda` y `rango-columnas`, por decisión de Lai.

#### Se podan los dos documentos del vault, y salen los dos últimos restos del plan de análisis retirado

**Qué se decidió.** Poda con la regla de registro del 2026-08-29. El hallazgo que justifica la entrada: el barrido del 2026-08-29 dejó dos criterios de decisión sin retirar, el «ΔR², no ρ crudo» de las decisiones de ejecución del 2026-06-09 y el margen δ_H2 = 0,15 de la entrada del 2026-08-08. Los dos nombran un estadístico o un umbral para H2, y se retiran con fecha porque la declaración de la fase H se apoya en este log: un criterio que muere pide su línea igual que uno que nace.

- **Trampa:** los dos iban dentro de entradas sobre otra cosa. Un barrido futuro lee las entradas enteras y no solo sus títulos.

### 2026-09-01

#### La figura de la fase A deja de ser el mapa de disponibilidad y pasa a ser la ventana de learning rates

**Qué se decidió.** El mapa de 24 celdas por 6 VD se retira entero, con su PDF y su función. En su sitio, `ventana-lr`, que recorre los ocho learning rates de cada optimizador y marca cuántos de los cinco runs cruzan el umbral, con leyenda discreta porque el dato solo toma seis valores. Los recuentos pasan a prosa en `resultados.tex` §Qué variables quedan medidas y `efficiency.py::availability_by_cell` los sigue calculando.

**Por qué.** Su afirmación murió con el umbral por arquitectura, que dejó VD1 en las 24 celdas; cinco de sus seis columnas decían el mismo número y la sexta contaba runs cuando la censura se mide en pares. Lo que afirma la figura nueva está en la memoria: tramo contiguo de learning rates en las 24 celdas, desplazado hacia arriba con la profundidad, y confundido con el umbral, que sube con ella.

- **Trampa: la figura agrega los cinco runs de cada posición** y no dice cuál cruza; el detalle run a run está en `efficiency.py::vd_status`. La trampa del eje x no compartido dejó de existir el 2026-09-02, cuando los dos paneles pasaron a un eje común.

#### El umbral de VD1 pasa a ser por conjunto de datos y arquitectura

**Qué se decidió.** El umbral τ de val-*accuracy* deja de ser uno por conjunto de datos y pasa a ser uno por conjunto de datos y arquitectura, doce valores, compartido entre optimizadores a propósito.

| conjunto de datos | FC | CNN | ResNet-18 | τ anterior |
|---|---|---|---|---|
| MNIST | 0,975 | 0,98 | 0,99 | 0,97 |
| CIFAR-10 | 0,50 | 0,60 | 0,75 | 0,65 |
| CIFAR-100 | 0,20 | 0,30 | 0,40 | 0,35 |
| Tiny-ImageNet | 0,08 | 0,22 | 0,36 | 0,20 |

**Por qué.** Un umbral por conjunto era una vara que FC no alcanzaba nunca fuera de MNIST y que ResNet-18 saltaba de inmediato. La correlación se calcula dentro de una celda, donde la arquitectura es constante. Se comparte entre optimizadores porque OE5 compara los doce pares de celdas que solo difieren en el optimizador.

**El criterio es una norma reproducible, no preinscripción.** τ es el valor redondo más alto que alcanza al menos el 60 % de los runs que aprendieron; redondo es múltiplo de 0,05, de 0,005 en MNIST y de 0,02 en Tiny-ImageNet, redondeando hacia abajo. El 60 % es la cobertura más alta a la que ninguna de las doce combinaciones degenera: al 65 %, Tiny con FC tendría que bajar a 0,06 y ahí dos tercios de sus cruces caen en la primera *epoch*. El denominador son los 806 runs que aprendieron, `learned` de `efficiency.py::run_health` con `CHANCE_MARGIN = 1,25`; los otros 154 tienen causa mecánica y no cruzarían ningún umbral. La norma promete un mínimo, no una cuota: las coberturas reales van del 62 al 89 %. Se fijó mirando los datos, como todo lo posterior al 2026-08-22.

**Evidencia, sobre los 960.** Los cruces pasan de 494 a 571 (573 con el suavizado corregido del 2026-09-03) y las celdas con VD1 de 18 a 24, con mínimo 15 por celda. Los cruces ya ocurridos al medir la ventana del 5 % bajan del 30 al 11 %, y los del 10 % del 56 al 35 %.

- **Trampa: `epochs_to_threshold` de los 960 `summary.json` queda obsoleto**, calculado con los umbrales viejos. VD1 se recalcula siempre en la capa de análisis desde `trajectory.parquet`; `reports/` no se toca.
- **Trampa: la calibración se hace sobre los datos que se analizan.** τ se eligió mirando la distribución de resultados y nunca la relación entre una métrica y VD1, así que no infla ninguna correlación, pero sí escoge la versión de la variable con más varianza. Escrito en `metodologia.tex` §Calibración.
- **No se cumplió el criterio de calibración preescrito del pilot**, que pedía cruces hacia el 30-60 % del presupuesto con el learning rate central; con esta tabla las medianas van del 8 al 25 %. Está dicho en `metodologia.tex` §Calibración desde el 2026-09-03.

### 2026-08-31

#### El mapa de lo computable cuenta tres estados, y la censura de VD1 se mide en pares y no en runs

**Qué se decidió.** Por cada run y VD, tres estados: el valor está, falta, o está pero no es una medición. El tercero existe solo en `final_test_acc` y `final_gap_acc` de los 41 divergidos, donde `argmax` sobre logits NaN devuelve el índice 0. El mapa señala y no excluye. La censura de VD1 se cuenta en pares comparables: un censurado se ordena contra cualquier run que cruzó, y con 15 cruces de 40 quedan el 61,5 % de los pares y no el 37,5 % que sugiere contar runs. Vive en `efficiency.py::vd_status`, `availability_by_cell` y `vd1_information`; la memoria lo cuenta en §Qué variables quedan medidas.

- **Es censura de tipo I, administrativa:** presupuesto fijo, igual para todos los runs del conjunto y conocido antes de mirar resultados, el caso que la estadística de supervivencia trata de serie.
- **Sin medir, y declarado:** no se distingue run a run entre «iba subiendo y se acabó el presupuesto» y «estaba en meseta por debajo del umbral».

#### El censo de runs cierra los cuatro recuentos: 154 nunca aprendieron y ninguno queda sin explicación

**Qué se decidió.** `efficiency.py::run_health` da dos columnas separadas, `learned` (cómo acabó: mejor val-*accuracy* suavizada por encima de 1,25 veces el azar) y `failure` (qué firma apareció: divergencia, colapso o ninguna), más `*_frac` con la proporción del run. No excluye a nadie. Dos columnas porque `resnet18_cifar100_sgd_lr1.0_seed2` colapsa 5 *epochs* de 40, se recupera y acaba a 24,7 veces el azar.

**Evidencia, sobre los 960.** 165 runs con firma en alguna *epoch* (41 divergidos, 124 colapsados) y 133 en todas (39 y 94). 154 nunca aprendieron, 115 por colapso y 39 por divergencia, cero sin causa; 1,2 y 1,25 seleccionan los mismos 154. ResNet-18, la única con normalización por lotes, no perdió ni un run.

- **Trampa: el margen es una línea recta sobre un continuo.** Once runs muestran firma y aprenden igual, uno a 1,35 veces el azar. «Aprendió» no es «sirve»; esa segunda decisión es de la fase B.

#### El código de run y celda va a `efficiency.py`, y `analysis.py` se queda con las columnas

El corte va por unidad de observación: `analysis.py` trabaja sobre columnas de `trajectory.parquet`; lo que mira runs y celdas y lee `summary.json` va a `src/efficiency.py`, nombre que continúa `train.py::efficiency_summary`. Se partió entonces porque nada en `src/` importaba `analysis` y salía gratis. Ninguna correlación métrica-VD vive ahí: tendrá casa propia en la fase B.

#### El número de clases sube a `config.py`; el nivel de azar se queda fuera

`NUM_CLASSES` se define en `config.py` y `data.py` lo lee, como `SPLIT_SEED`: es un hecho del conjunto que usan los dos lados, e importar `data` desde el análisis ataría la capa de análisis a torch a cambio de un entero. El nivel de azar 1/K y el factor 1,25 no suben, porque son criterio nuestro y revisable, y viven en el módulo de análisis.

### 2026-08-29

#### Punto 1 de la fase A: las columnas son válidas, y un cero falso que confirmaba la teoría

**Qué pasó.** Al validar las columnas sobre los 960 runs, `stiffness/sign_global`, `sign_within`, `sign_between` y `confusion/frac_neg` daban un cero exacto en los 41 runs divergidos, donde los gradientes son NaN, porque `torch.sign(NaN)` devuelve 0,0 y `NaN < 0` es falso. Corregido con dos líneas y una prueba cada una, en `stiffness.py` y `gradient_confusion.py`.

**Por qué importaba.** El punto falso caía en la esquina del gráfico, métrica en su mínimo y resultado en el peor, así que una correlación habría leído «poca alineación temprana, peor resultado final», que es la dirección que predice la teoría. El fallo no metía ruido, metía señal, y era invisible para el chequeo de validez, porque cero está dentro de rango. Lo delató solo compararlo con la columna hermana: el punto 1 tiene que mirar los runs y no solo las columnas.

**Veredicto sobre los 960.** Ningún valor fuera de rango ni infinito en 33.600 filas por 30 columnas, las cinco identidades exactas sin violación, y las mismas 41 columnas en todos los runs, así que ninguna métrica lanzó nunca una excepción. Los NaN tienen dos causas y las dos son el learning rate alto: 41 divergidos (39 enteros, todos SGD en los dos valores más altos de su rejilla, y 2 a mitad) y 94 colapsados con `gwa/score_mean` a cero exacto, ninguno ResNet-18.

**`reports/` se parchea, decisión de Lai:** son la fuente de la verdad y se corrigen en vez de enmascararse al leer. La regla exige coseno del mismo ámbito a NaN y celda a 0,0 exacto, porque un subconjunto de pares vacío da un 0,0 legítimo. 41 runs, 82 ficheros, 6.484 celdas de 0,0 a NaN; no se re-entrena nada.

- **Aguas abajo no hay nada que hacer:** desde este día `reports/` trae el NaN y se comporta como cualquier valor ausente.
- **En la memoria no se narra, decisión de Lai.** La memoria describe un método y reporta resultados, y ningún resultado dependió del fallo, porque se encontró antes de la primera correlación. La única obligación es no afirmar lo contrario: no escribir que `reports/` es exactamente lo que produjeron las ejecuciones.
- **Las dos columnas de *accuracy* de los divergidos no se parchean.** `evaluate_test` calcula de verdad lo que el modelo predijo; declararlas inservibles es interpretación y es de la fase B.
- **El NaN de GWA no es un defecto:** `_gwa_aggregate` devuelve NaN a propósito cuando la varianza de los cosenos es cero.

#### El `README.md` certificaba lo contrario de la verdad; el pilot pasa a versionado y `CLAUDE.md` no se versiona nunca

El `README.md` afirmaba que el historial de git certificaba que el plan precedía a los datos; corregido con la cronología, la matriz terminó el 22 y el plan se retiró el 25. `reports_pilot/` pasa a versionado, 2,5 MB con los seis `testfix_40ep/` de Tiny y los `summary.json` que se auto-declaran corruptos con `_tiny_test_note`, porque es la evidencia de `DATASET_BUDGET`. `CLAUDE.md` no se versiona nunca, decisión de Lai, porque es un fichero de instrucciones para una IA y el repositorio acaba depositado; queda en `.gitignore` con el motivo al lado.

#### La fase A cuenta pero no excluye, y su veredicto se parte en dos en la memoria

La fase A cuenta, no excluye: un run clavado en el azar tiene *accuracy* de test y entra en el recuento, y descartarlo es método y es fase B. La correlación entre dos VD, el punto 5, está permitida porque no entra ninguna métrica. El veredicto va partido, decisión de Lai: los números a Resultados, porque son algo calculado sobre la matriz, y la regla que se derive, la población de análisis, a §Protocolo de análisis, que escribe la fase B.

#### Se retiran los últimos criterios de decisión supervivientes, y `Métricas.md` se pone al día con el código

Desde el 2026-08-25 el proyecto afirmaba que ninguna hipótesis tenía criterio de decisión, y quedaban cinco restos: el corte |ρ| < 0,3 de [[1 - Diseño]], el «Spearman + Pearson con FDR» de su diagrama, el censurado como peor rango, la estandarización o efectos mixtos para agregar, y el FDR de `resultados.tex`. Los cinco se retiran y esta fecha es la evidencia para la fase H.

- **Trampa que habría invertido un resultado.** H6 daba «GWA alta → mejor generalización», el signo del artículo, que define el gradiente como −∇ℓ. Aquí se mide sobre ∇ℓ bruto, así que la predicción heredada es la contraria; el aviso está en el enunciado de H6 y en la tabla de signos de [[Datos experimentales]] §5.3.
- **Lección.** [[Métricas]] describía la v1 del código y clasificaba el coste al revés; una optimización que cambia la estructura del cálculo invalida en silencio la documentación escrita en términos de esa estructura.

### 2026-08-27

#### La comparación SGD↔Adam se hace sobre las ocho posiciones, no sobre el solape de rejillas

Las rejillas de LR de SGD y Adam **se solapan en seis de sus ocho valores**, de 3e-4 a 1e-1; solo SGD tiene 0,3 y 1, y solo Adam tiene 3e-5 y 1e-4. Es consecuencia mecánica del desplazamiento, porque una década son dos saltos en una rejilla espaciada a medias décadas. **La comparación pareada de OE5 se hace sobre las ocho posiciones** y no sobre esos seis valores compartidos al mismo LR nominal.

- **Decide el reparto de los fallos, medido sobre los 960 `summary.json`.** Con las ocho posiciones, SGD tiene 72 entrenamientos clavados en el azar de 480 y Adam 82 de 480, un 15 % frente a un 17 %, y los dos fallan por el mismo lado, el de paso demasiado grande. Restringido al solape, SGD cae a 3 de 300 y Adam se queda en 82 de 300, un 1 % frente a un 27 %. La restricción convierte una comparación equilibrada en una desequilibrada, porque **el mismo valor nominal no es el mismo régimen** en los dos optimizadores: sobre el solape, SGD recorrería de lento a bueno sin fallar casi nunca y Adam de bueno a muerto uno de cada cuatro.
- **De paso queda validado el desplazamiento de 10×**, que hasta hoy era una suposición tomada de los valores por defecto canónicos y del *momentum* 0,9, que amplifica el paso efectivo de SGD unas diez veces. La matriz completa lo comprueba a posteriori: que los dos optimizadores fallen casi al mismo ritmo, 72 frente a 82 muertos y 227 frente a 239 censurados, es lo que se observa cuando las dos rejillas cubren tramos comparables de sus rangos respectivos.
- **Se descarta también el solape como comprobación de robustez**, que fue la primera propuesta. Sería un control peor que aquello que controla: con un desequilibrio propio del 1 % frente al 27 %, una discrepancia en OE5 no permitiría distinguir un desplazamiento mal elegido de un subconjunto torcido.

#### El coste por métrica se dará en notación asintótica; el micro-benchmark queda aparcado

**El coste por métrica no existe en los 960 entrenamientos**, y no es que no se midiera: cada `summary.json` guarda un único `metric_seconds`, y `stream_shared` ejecuta el barrido por ejemplo una sola vez para seis de las ocho métricas, así que su coste marginal individual no es recuperable ni a posteriori. Peor aún, con un barrido compartido **"el coste de la m-coherencia" no está bien definido**: si se calculan las seis, su coste marginal es casi nulo; si se calcula sola, paga el barrido entero. Ese es el eje que `estado-del-arte.tex:111` promete como frente de Pareto.

**Decisión de Lai (2026-08-27): primero el coste teórico en notación asintótica; el micro-benchmark se aparca.** El coste asintótico es mejor que el reloj para este fin, porque no depende de la máquina ni de qué otras métricas corran a la vez, y separa las ocho en ligas según cuántas pasadas hacia atrás exigen, que es la unidad que manda.

- **Consecuencia que hay que asumir:** con coste asintótico el eje deja de ser continuo y pasa a ser una escala de tres o cuatro niveles, así que lo que se puede dibujar honestamente **no es un frente de Pareto** literal. Resuelto el mismo día en los cuatro sitios de la memoria que lo prometían: lo que se compara es dentro de cada clase de coste y frente al predictor de referencia, que no calcula ningún gradiente, en §Coste y capacidad predictiva de `resultados.tex` (`sec:res-coste`). La sección se mantiene en lugar de doblarse dentro de OE2, porque las clases de coste dan material propio y OE2 pregunta otra cosa; **esa parte es decisión mía y es revisable**.
- **Decidido por Lai: la tabla del estado del arte no se toca y el coste va en una tabla nueva**, `tab:coste-metricas` en `desarrollo.tex` §Cálculo de las métricas, con un solo propósito, lo que cuesta medir y lo que ahorra el barrido compartido. La memoria se queda fuera de esa tabla y sigue contada en prosa, porque responde a otra pregunta.
- El micro-benchmark sigue disponible el día que se quiera el eje continuo: minutos de cómputo, fuera de la matriz y sin tocar `reports/`.

**Resultado de la derivación**, con cada término trazado a la línea que lo produce. Las derivadas no separan a ninguna métrica, porque las ocho recorren el *batch* de medición una vez. Lo que separa es la aritmética posterior, en dos niveles: `M·P` las de momentos y `M²·P` las de pares. En el margen, con el barrido ya hecho, **seis de las ocho no derivan nada**: `P` la coherencia y la escala de ruido, `P·log P` el GSNR, `M` la GWA, `M²` la *stiffness* y `M²·log M` la confusión. Solo la varianza normalizada y la disparidad siguen pagando sus diez y cinco pasadas por submuestra. Medir el registro completo cuesta unas tres pasadas sobre el *batch*, no nueve.

- **Aviso para cuando se escriban los resultados:** el reloj `metric_seconds` incluye la aritmética del predictor de referencia, que además rehace el historial entero cada época y crece con el cuadrado del número de épocas. La cantidad es despreciable, pero `metric_seconds` no es exactamente "lo que cuesta instrumentar el gradiente".

#### Estilo de figuras: al ancho final, con la tipografía del cuerpo, y el color nunca solo

Definido desde cero en `src/figstyle.py`, con las notas de [[Estilo de redacción - notas del TFM HOFT]] §Estilo de las figuras como referencia, y aprobado por Lai sobre una demo con datos inventados. No añade ninguna biblioteca: matplotlib ya estaba en el grupo de desarrollo.

Medidas tomadas antes de decidir, no supuestas: el bloque de texto son **15 cm** (A4 con 3 cm de margen por lado, `tfgetsinf.cls:45`), el cuerpo son **10 pt en Palatino** (`\LoadClass{book}` sin opción de tamaño, más `mathpazo`), y **TeX Gyre Pagella**, el Palatino libre, está en el árbol de TeX Live y matplotlib puede cargarlo.

- **La figura se genera al ancho final y LaTeX no la escala nunca.** Solo hay dos anchos, el completo de 15 cm y uno estrecho de 10 cm, y los fija el módulo, de modo que en el `.tex` no debe aparecer ningún `width=0,8\textwidth`. Esto ataca el defecto que se midió en HOFT, donde dos gráficas quedaron con los rótulos a la mitad del tamaño del texto por exportarlas a un tamaño y encogerlas a otro.
- **Trampa que anula la decisión anterior si se toca:** `bbox_inches="tight"`, el idioma habitual para guardar en matplotlib, recorta el PDF al contenido y por tanto **cambia el tamaño final**. El módulo usa `constrained_layout`, que mantiene el tamaño y mete el contenido dentro. Está avisado en el código porque es justo lo que alguien "arregla" sin saber que lo rompe.
- **El color no lleva nunca información él solo (regla retirada el 2026-09-02, ver la entrada de esa fecha):** el ciclo empareja cada color con su propio trazo. La primera paleta candidata **suspendió** la prueba de gris, con naranja y verde a 0,011 de luminancia, o sea el mismo gris; la definitiva tiene una separación mínima de 0,096, y una prueba exige que el mínimo supere 0,05.
- **Reglas que puso Lai al aprobarlo (2026-08-27).** Salida **siempre en PDF**, porque se quiere máxima calidad; un PNG es previsualización y nunca entregable. **Ningún eje cortado por encima del cero**, que exagera las diferencias. Y **dos paneles de la misma cantidad comparten escala**, porque dibujar cada uno en su rango hace parecer iguales valores que no lo son. Las dos últimas están en `include_zero()` y `match_limits()` en vez de en un comentario, con pruebas, porque la demo aprobada las incumplía las dos.
- **Separación deliberada de HOFT:** fuera los ejes superior y derecho. Allí las cuatro gráficas conservan la caja cerrada, pero eso es el valor por defecto de matplotlib y no una decisión suya.
- **Se copia la regla de figura o tabla de HOFT**, que es observable y no admite discusión: cómo cambia algo al mover una perilla continua va a gráfica; qué gana en qué prueba va a tabla con el mejor valor en negrita; cómo funciona un mecanismo va a esquema. Junto con la regla ya vigente de una figura, una afirmación.

#### `seconds_to_threshold` queda registrada pero no se declara variable del estudio

Todos los runs escriben `seconds_to_threshold` junto a `epochs_to_threshold`, pero no está entre las seis VD de [[1 - Diseño]]. **Se menciona en §Variables como disponible y no se declara variable del estudio.** Es la lectura de velocidad honesta al comparar SGD con Adam, porque su paso no cuesta lo mismo, así que borrarla del texto sería esconder un dato que existe; declararla, en cambio, añade una séptima familia de correlaciones a un trabajo cuya multiplicidad ya es el problema abierto, y esa elección pertenece al método de análisis y no a la sección de variables. Si el método la necesita, se declara entonces y se registra aquí.

#### Terminología: anglicismos bien conocidos, y siempre en cursiva

Se admiten en la memoria los términos ingleses de uso corriente en el campo, y todo anglicismo va en cursiva **cada vez que aparece**, no solo la primera. Esto último deroga la convención vigente desde julio, que reservaba la cursiva a la primera aparición de cada término.

- **Por qué la cursiva siempre.** Lo manda la norma de la escuela: el material de seminarios dice "cursiva para palabras extranjeras" y remite al DLE para distinguir el extranjerismo crudo, que va en cursiva, del adaptado, que va en redonda. Ninguno de los nuestros está adaptado en el DLE. Son unas 160 cursivas en 43 páginas, algo más de tres por página.
- **Pasan a inglés:** *epoch*, *learning rate* y *seed*. Con dos consecuencias gramaticales que hubo que resolver a mano: *epoch* y *seed* heredan el género femenino de "época" y "semilla", de modo que los artículos y adjetivos que ya estaban siguen concordando; *learning rate*, en cambio, es masculino en el uso habitual.
- **"Conjunto de medición" pasa a "batch de medición".** El nombre anterior era ambiguo en castellano, porque se lee igual como "el conjunto de las mediciones", que es lo contrario de lo que designa. **Se descartó "probe"**, que es el nombre del concepto en el código y encaja por significado, porque en aprendizaje automático *probe* ya designa otra cosa muy conocida, el clasificador lineal entrenado sobre representaciones congeladas. Falla justo el criterio de admisión: es conocido, pero por otra cosa. El término queda definido formalmente en `fundamentos.tex` §Geometría del gradiente, antes de su primer uso.
- **"Tamaño de lote" pasa a "tamaño de batch"**, aplicación de la decisión del 2026-07-04, que ya fijaba *batch* frente a "lote".
- **Regla para los nombres de métrica:** acrónimo en redonda, palabra inglesa en cursiva. Quedan en redonda GSNR, GWA, TSE, NGV y GNS, y también "m-coherencia", que es una adaptación al castellano. Van en cursiva *stiffness*, *gradient disparity* y *gradient confusion*, que son sintagmas ingleses usados como nombre. Se aplica lo mismo a *minibatch*, *dropout* y *weight decay*.
- **Se quedan en castellano**, porque cambiarlos sería anglicismo por anglicismo y no por claridad: "conjunto de datos" frente a *dataset*, "submuestra", "ventana", "banco de pruebas" y los conjuntos de entrenamiento, validación y test.
- **Ampliación del mismo día, al escribir §Configuración del entrenamiento: el nombre técnico inglés gana a la traducción forzada.** La regla anterior admitía el término inglés cuando no había equivalente; ahora se admite también cuando el equivalente existe pero es una perífrasis que nadie usa. Entran así la red *fully connected* en lugar de "red totalmente conectada", y *max pooling*, *adaptive average pooling*, *stem* y *stride*, que en castellano solo se dicen describiéndolos. El criterio de corte sigue siendo la claridad y no el anglicismo por el anglicismo: "red convolucional" y "perceptrón multicapa" se quedan como están, porque son de uso normal en castellano.
- **Trampa al marcar cursivas:** marcar *batch* vuelve a marcar el que ya está dentro de *batch normalization*, y deja cursivas anidadas.

### 2026-08-25

#### Se retira el plan de análisis

Se borran `4 - Análisis.md` (el plan preregistrado y congelado), `src/power_analysis.py` con sus pruebas, y el §Protocolo de análisis de [[1 - Diseño]]. Las seis hipótesis se conservan en [[1 - Diseño]] como afirmaciones falsables, ahora sin criterio de decisión.

**Por qué.** El plan había crecido hasta un punto en que ya no se entendía por completo, y un método que no se entiende no se puede defender ante un tribunal. Se prefiere partir de una base comprensible y construir el análisis desde ahí, hipótesis por hipótesis.

**Consecuencia que hay que declarar en la memoria.** El plan estaba congelado y commiteado antes del primer resultado, lo que permitía afirmar que el análisis precedía a los datos. Al retirarlo, el análisis que se haga es **posterior a los datos** y así debe presentarse. El plan retirado sigue íntegro en el historial de git por si hiciera falta recuperarlo.

### 2026-08-08

Las tres entradas de esta fecha salen de la primera revisión de la matriz en marcha, con 268 runs terminados. Ninguna se tomó habiendo calculado ninguna correlación entre métrica y VD, que a esta fecha seguían sin existir. Ninguna toca `src/`, porque la matriz corre lanzando un proceso nuevo por run, así que cualquier edición del código habría entrado en vigor en el run siguiente y habría partido la matriz en dos versiones.

#### El coste de instrumentación documentado era anterior al barrido compartido

La cifra de 3,21x que este log venía citando como peor caso procede del pilot, que se ejecutó el 2026-06-15. El barrido compartido entró dos días después (commit `8566fc3`, `perf(metrics): share one per-sample gradient sweep per probe`). **Todas las cifras de coste anteriores a esa fecha eran, por tanto, de una versión del código que ya no es la que corre la matriz.** El peor caso real está medido sobre la matriz y vive en la entrada del 2026-07-17.

La causa es la optimización y no otra cosa: el tiempo de entrenamiento por época es el mismo (1,671 s en el pilot frente a 1,638 s en la matriz), el de medición se reduce a la mitad, y las trayectorias de la misma configuración salen **idénticas bit a bit** durante las 40 épocas, que es justo lo que promete el invariante de la ruta compartida.

**La corrección no cambió ninguna decisión.** El coste servía para justificar cuánto había que exigirle a una métrica de gradiente para que mereciera la pena, no para calcularlo, y 2x sigue siendo un coste alto.

#### El techo de la red FC fuera de MNIST

Verificado sobre los 960 runs (2026-08-25). Los techos de val-accuracy medidos para FC: 0,584 y 0,569 en CIFAR-10; 0,295 y 0,289 en CIFAR-100; 0,114 y 0,108 en Tiny-ImageNet. Con el umbral único por conjunto de datos vigente entonces (0,65 / 0,35 / 0,20), ninguna de esas seis celdas alcanzaba nunca el umbral, con ninguna de las 8 tasas de aprendizaje ni ninguna de las 5 semillas. Es el motivo por el que el umbral pasó a depender también de la arquitectura el 2026-09-01.

El contraste que lo explica: la misma red FC **sí** pasa el umbral en MNIST, con 0,987 y 0,985 contra 0,97. No es falta de presupuesto ni un fallo de configuración. El pilot corrió esa configuración al presupuesto doblado, 80 épocas, y tocó techo en la época 40 para quedarse plano el resto. Es el techo de la arquitectura.

#### Muchos entrenamientos se quedan clavados en el azar, y se detectan por un cero exacto

Verificado sobre los 960 runs (2026-08-25). En las tasas de aprendizaje altas hay entrenamientos que no aprenden nada. Su accuracy de validación se queda en el azar. Son **154 de 960**, repartidos por 16 de las 24 celdas, así que no es una curiosidad de un conjunto de datos concreto. El censo del 2026-08-31 los reconcilia con el resto de recuentos.

**El mecanismo, comprobado.** Las ReLU mueren, la última capa recibe un vector de entrada exactamente nulo y el clasificador solo emite su sesgo. El gradiente del peso de la última capa es entonces exactamente cero, y como GWA protege su norma con un `clamp_min(EPS)`, `gwa/score_mean` sale **0,0 exacto**. Un cero exacto en coma flotante no sale por casualidad, así que sirve de firma: aparece en 124 runs y, en los afectados, ocupa de media el 74 % de las épocas. La firma es fuerte, pero no infalible: de esos 124, 115 acaban clavados y **9 no**.

Qué se hace con estos runs al analizar se decidió el 2026-09-03: fuera de la población primaria de todo contraste, los 806 que aprendieron, y dentro de la pasada con 960 que queda solo en `results/`. Matiz medido el 2026-08-25: en la variable de velocidad no hay nada que decidir, porque todo run clavado es además un run que nunca cruza el umbral, y la cuenta de runs con velocidad medida sale idéntica con ellos y sin ellos. La decisión solo afecta a las otras cinco variables dependientes, donde los clavados sí tienen valor.

### 2026-08-05

#### Corregido el estadístico de degeneración de los diagnósticos de sanidad

El diagnóstico que responde "¿esta métrica llega a moverse dentro de un entrenamiento?" estaba **mal**. Medía `within_std / RMS(within_std entre runs)`, una normalización contra una referencia calculada **entre** entrenamientos y dominada por el de mayor escala, así que el resultado ordenaba por escala y no por movimiento. Marcaba como degenerada la `val_loss` de los seis runs de MNIST, cuya val loss recorre el rango 0,18 a 0,02.

**Sustituto.** `signal_to_jitter = std(valores) / std(primeras diferencias)` dentro de cada run. Numerador y denominador escalan igual con la métrica, así que el cociente no depende de las unidades ni de la escala. Y trae su propia referencia en vez de un umbral a mano: una trayectoria que sea ruido blanco alrededor de una constante cumple `std(diff) = √2 · std(valores)`, luego su cociente vale 1/√2 ≈ 0,71.

**Qué cambia en las conclusiones.** El orden se invierte en la cabeza y en la cola. `var/normalized` y `noise_scale/simple` figuraban como las más sanas (0,89 y 0,87 del máximo) y son en realidad las que menos se distinguen del temblor (0,81 y 0,82 frente al 0,71 del ruido puro), junto con `stiffness/cos_global` y `mcoh/global`. Y `gwa/value` figuraba como la más muerta (0,003) sin serlo: sus valores oscilan entre ±1,5 en un run de ResNet sobre MNIST y ±4·10⁻⁵ en uno de FC sobre Tiny-ImageNet, seis órdenes de magnitud, que es justo lo que la normalización anterior confundía con inmovilidad.

#### Sin veredicto binario sobre degeneración

La versión corregida deja de emitir una etiqueta `degenerate` y publica solo la comparación contra la referencia (`below_noise`). El motivo es que 1/√2 es el valor **asintótico**. Una métrica que fuera puro ruido cae a un lado o a otro de la línea aproximadamente la mitad de las veces, de modo que un booleano por run afirma una decisión que el estadístico no sostiene con una sola trayectoria. Lo que se lee es la distribución completa de los 24 runs frente a la línea.

### 2026-07-17

#### Coste de instrumentación: se mantiene la medición completa

Cierra la decisión abierta desde el pilot. Se mantiene la medición tal cual: registro completo de las 8 métricas + baseline al final de cada época, sobre la probe fija de M=256, en toda la rejilla. La prioridad declarada es disponer de datos suficientes: la serie temporal completa por época.

- **Por qué.** El peor caso, ya medido sobre la matriz completa, es **2,048x** el wall-clock de un run sin instrumentar, en `fc × cifar100 × sgd`, dentro de la cota <3-4x fijada. Conservar la serie completa preserva la elección de ventanas a posteriori y la línea exploratoria post-meseta anotada como trabajo futuro.
- **Alternativa que sigue disponible.** Fusionar los 2 batch-sweeps restantes (NGV, gradient disparity) en el sweep compartido es una palanca válida de ingeniería, pero no es bit-idéntica: cambia los valores en ~1e-6 frente a los que los tests pinean. Queda como optimización futura si el coste apretara. Bajar la cadencia de medición y submuestrear la probe quedan descartadas, la primera porque pierde resolución de trayectoria y la segunda porque M=256 está congelado por comparabilidad.
- **Consecuencia.** La matriz completa consumió **121,7 h** de reloj: 97,6 h de entrenamiento y 24,1 h de instrumentación, sumadas sobre los 960 `summary.json`.

#### Presupuestos definitivos del pilot (registro formal)

Cierra el "registrar aquí los valores finales con su evidencia" de la decisión del pilot (2026-06-09). Los valores operan en `config.py::DATASET_BUDGET` y en los 24 YAML desde el 2026-06-17 (evidencia: `run_pilot.py --report` sobre `reports_pilot/`; el pilot corrió con los presupuestos candidatos doblados, 40/80/120/160 épocas). Los umbrales que esta entrada fijó dejaron de valer el 2026-09-01, cuando pasaron a depender también de la arquitectura.

- **Presupuestos finales (épocas):** MNIST 20; CIFAR-10, CIFAR-100 y Tiny-ImageNet 40.
- **Evidencia de presupuesto** (época de meseta de val-loss por celda, absoluta): MNIST 6-17, CIFAR-10 3-19, CIFAR-100 3-8, Tiny-ImageNet 1-8. Los presupuestos finales cubren la meseta con margen y conservan el múltiplo de 20 que hace exacto el snap de `windows`. CIFAR-100 baja de 60 a 40 y Tiny-ImageNet de 80 a 40 (recorte del pico de meseta: épocas muertas multiplicadas por ~960 runs); MNIST (20) y CIFAR-10 (40) conservan su candidato.
- **Contexto de ejecución corregido (2026-07-17):** hay una única GPU disponible, no un cluster (pese a la asunción de la decisión 2026-06-09), así que el troceado por nodos no aplica y la matriz se ejecuta por tandas con la reanudación del launcher.
- **Salvedad Tiny.** Los campos test/gap de `reports_pilot/` para Tiny-ImageNet son los corruptos del bug pre-fix; la calibración usó solo el lado val y el timing (sanos). Desde el 2026-07-17 la referencia corregida (re-run post-fix a 40 épocas) vive dentro de cada run del pilot (`testfix_40ep/`), el `summary.json` corrupto se auto-declara vía la clave `_tiny_test_note`, y `reports_validity/` quedó retirada tras la fusión.
- **Qué NO cerraba esta entrada.** El suelo de ajuste del gap (mínimo de `final_train_eval_acc` para los contrastes de VD5/VD6) se dejó sin valor a propósito; se fijó el 2026-09-03 en el τ de la celda y deja 693 de 806. Aplazarlo no cuesta nada: es un filtro de análisis, no un knob de entrenamiento (`final_train_eval_acc` queda registrado en cada `summary.json`), así que fijarlo o revisarlo después solo toca código de análisis, nunca obliga a re-correr runs.

#### Tabla de signos de H6 verificada contra los papers

Verificación de la tabla contra los PDFs del vault (GSNR/Liu, Coherent Gradients/Chatterjee, Making Coherence/Chatterjee & Zielinski, GWA/Hölzl, GNS/McCandlish, TSE/Ru). La tabla corregida vive en [[Datos experimentales]] §5.3; cambios y evidencia clave:

- **m-coherence vs VD1: sigue −, pero la base fuerte cambia de paper.** Chatterjee & Zielinski no afirma velocidad (su α es eficiencia por paso, definicional; las menciones de velocidad son citas a terceros). El claim explícito es de Chatterjee 2020 (CGH): "we expect that greater the agreement in per-example gradients, the faster loss should decrease" (§2.2) y "as noise increases, the time taken to reach a given level of accuracy (i.e., realized learning rate) increases" (§2.3). Matiz: medido sobre train accuracy; la extensión a val es razonada.
- **GSNR vs VD4: de fuerte a extrapolada.** El paper solo afirma el gap ("larger GSNR during training process leads to better generalization performance", vía OSGR, ec. 22; el gap es en loss, la misma cantidad que VD5); no hay claim de test accuracy. Su predicción fuerte es − vs el gap. Matices: la teoría se deriva en fase temprana (favorece la ventana del TFG) pero con full-batch GD, no SGD.
- **GWA vs gap: de fuerte a direccional cualitativa.** El claim cuantitativo del paper es vs test accuracy (Fig. 3: Pearson 0,99 solo ConvNeXt/CIFAR-10; 0,92 cross-arquitectura; medido sobre max de toda la trayectoria, no ventana temprana; su criterio de early stopping descarta el primer 10% como warm-up). El gap operativo (test loss − train loss) nunca se mide. **Corrección 2026-08-05:** con la medida libre de escala, GWA en ventana temprana marca 1,08 en su valor titular y 1,37 en la media de scores, por encima del 0,71 del ruido puro, y solo su curtosis (0,91) queda pegada a esa línea. Es débil, no plana, y de hecho `var/normalized` y `mcoh/global` se mueven **menos** que GWA sobre la trayectoria completa. La rebaja se mantiene, y se apoya solo en el argumento bibliográfico, que es el que decide.
- **GNS vs VD1: + confirmado con condiciones.** Base formal ec. 2.7/D.1 (δS = 1 + 𝓑/B a B fijo). Condiciones: régimen B ≲ 𝓑, LR bien ajustado, y el GNS medido depende del LR ("it is not consistent at different learning rates", Ap. A.1), relevante porque el TFG barre LR a B fijo. Su silencio de gap es correcto (caveat 6 del paper).
- **m-coherence vs gap: − confirmado con la salvedad del propio paper** ("this connection is complicated": con 100% label noise la coherencia también sube; lo informativo es la coherencia temprana). Trayectoria esperada para los diagnósticos: no monótona en general ("broad parabolic trajectory"); a granularidad de época con labels reales, decreciente hacia ~1 tras un pico muy temprano.
- **TSE: definiciones y caveats confirmados.** Corrección literal de la cita de §4.2: termina "outside the scope of this paper", no "of our work". γ=0,999 es el default recomendado de §4.1, no la constante definicional (§2 introduce TSE-EMA con γ=0,9). Aviso de archivo: el PDF local es la versión NeurIPS sin apéndices; los apéndices C.1-C.2 (overconfidence, base del caveat de VD2/VD3) solo están en el arXiv v2, que hay que archivar en `Papers/PDFs/`.

La lectura humana de estos papers sigue pendiente y es valiosa para el estado del arte.

### 2026-06-14

#### Gap de generalización: tercer constructo de variable dependiente

Confirmado por el tutor el 2026-06-14 e implementado el mismo día en `src/data.py` + `src/train.py`. Añade el constructo *generalización* a las variables dependientes de [[1 - Diseño]], junto a velocidad y rendimiento final.

- **Qué se mide.** Cinco claves nuevas en `summary.json`: `final_gap_loss = final_test_loss − final_train_eval_loss` (primaria; positivo = sobreajuste), `final_gap_acc = final_train_eval_acc − final_test_acc` (robustez, mismo sentido), sus términos `final_train_eval_loss`/`final_train_eval_acc`, y `final_test_loss`.
- **Cómo.** Una pasada `evaluate()` extra al final del run, en modo eval y con los mismos pesos, sobre un subconjunto fijo y estratificado por clase del train recortado, de tamaño igual al test y muestreado con `SPLIT_SEED` (idéntico en todos los runs, independiente de la semilla del run; `build_train_eval_loader` en `data.py`). Coste: segundos por run, una vez.
- **Confound conocido** (el presupuesto de épocas es fijo): a presupuesto fijo, un gap pequeño puede significar que el modelo generaliza bien o que no ha aprendido lo suficiente, y las dos cosas se confunden. Quien analice el gap tiene que separarlas. Decidido el 2026-09-03: un suelo de ajuste, `final_train_eval_acc` ≥ τ de la celda, y el *gap* se lee con la velocidad delante.
- **Respaldo.** La cantidad (riesgo de test − riesgo empírico) y el rol (gap como variable dependiente de un estudio correlacional) son de la literatura: Jiang et al. 2020 (arXiv:1912.02178), Dziugaite et al. 2020 (arXiv:2010.11924); incluso la estimación del término de train por submuestreo tiene precedente en Jiang §3. Lo propio del TFG son los dos controles.

### 2026-06-12

#### Protocolo de evaluación: train optimiza, val monitoriza, test certifica

Confirmado por el tutor (respuesta rápida del 2026-06-12: particiones típicas de cada dataset, sin validación cruzada, semillas múltiples sobre train, val para evaluar convergencia, test para el resultado final) e implementado el mismo día en `src/data.py` + `src/train.py`. Los tamaños de partición resultantes están en [[1 - Diseño]] §Setup de entrenamiento.

- **Roles únicos, sin cruces.** El modelo entrena con el train recortado; la probe de métricas se muestrea de ese mismo train; la monitorización por época y todos los indicadores de eficiencia leen val; el test se evalúa exactamente una vez al final, produciendo `final_test_acc` y `final_test_f1_macro` (vía matriz de confusión en torch, sin dependencias nuevas).
- **Lecturas estables de la curva.** VD1 (épocas-hasta-umbral) y VD3 (mejor loss) se leen sobre la curva de val suavizada con mediana móvil centrada de 3 épocas (`median3` en `train.py`; desde el 2026-09-03 el borde toma el valor crudo, ver «Diez desviaciones»). Motivo: los extremos de una serie ruidosa están sesgados en proporción a su volatilidad, la volatilidad depende del LR y las métricas de ruido de gradiente plausiblemente la predicen. Sin suavizado, el propio estimador crearía un confusor entre predictor y VD. VD2 (AUC) integra la curva cruda: integrar ya amortigua el ruido. La curva cruda completa queda en `trajectory.parquet`, todo recomputable post-hoc.
- **Por qué.** Tres problemas del setup de 2 vías: el sesgo de extremo (arriba), la circularidad de calibración (umbrales calibrados sobre curvas de test del pilot y `epochs_to_threshold` medido después sobre ese mismo test) y la objeción previsible en la defensa ("evaluasteis test cada época"), aunque ninguna decisión de entrenamiento mirase el test (presupuesto fijo, sin early stopping, rejilla preespecificada).
- **Notas de honestidad para la memoria.** (1) El "test" de Tiny-ImageNet es su val público, práctica estándar, se declara. (2) F1-macro ≈ accuracy en balanceados: verificación, no hallazgo. (3) El split es fijo y compartido por todos los runs, decisión deliberada: lo estudiado es la variación por seed/LR, no la varianza del estimador (Bouthillier et al. 2021 recomiendan aleatorizar splits cuando se comparan métodos; no es el caso), y se declara.

### 2026-06-10

#### Timing por run: dos relojes, no uno

Cada run cronometra por separado el entrenamiento y la instrumentación (`src/train.py`): `summary.json` gana `total_seconds`, `metric_seconds` (acumulado alrededor de cada bloque de medición, con `synchronize` en cuda/mps para que los kernels asíncronos se atribuyan al reloj correcto) y `train_seconds` = total − metric.

- **Por qué dos relojes y no uno.** El overhead de la instrumentación (per-sample grads vía vmap sobre la matriz M×P) escala con el tamaño del modelo y con la densidad de probes: un único wall-clock sesgaría las comparaciones de tiempo entre celdas a favor de los modelos pequeños.
- **Timestamps por fila.** Toda fila de `trajectory.parquet` lleva `elapsed_seconds` y `metric_seconds` acumulados; eso habilita `seconds_to_threshold` junto a `epochs_to_threshold`, la velocidad en wall-clock, más honesta al comparar SGD↔Adam (coste por paso distinto). Es cruda (incluye instrumentación hasta ese punto); la corrección post-hoc es restar la columna acumulada, sin relanzar nada.
- **Convenciones.** `evaluate()` cuenta como entrenamiento (práctica estándar de cualquier run); solo `measure` + baseline TSE van al reloj de overhead. El `synchronize` se hace solo alrededor de los probes (infrecuentes y ya caros), nunca por paso de optimización. El wall-clock es señal de presupuesto y anomalías, no métrica científica, así que no se correlaciona como si fuera limpio.

### 2026-06-09

#### Pilot de calibración: un run por celda, presupuesto doblado

Concreta el "se calibran en el pilot" de presupuestos y umbrales (decisión "Matriz de runs congelada") en un protocolo ejecutable: `src/run_pilot.py`, módulo aparte del launcher de producción.

- **Qué corre.** Un run por celda (24 en total), LR en el centro de la rejilla (SGD 1e-2, Adam 1e-3, los defaults canónicos de cada optimizador), seed 0 y **el doble del presupuesto candidato** (40/80/120/160 épocas). La asimetría que lo justifica: recortar una curva generosa a posteriori es gratis, estirar una corta es relanzar. El presupuesto define `progress_frac`, las ventanas y el AUC, así que debe quedar bien fijado *antes* de los ~960 runs. El coste del pilot (24 runs a 2×) es ~5% del de la matriz.
- **Aislado de `reports/` a propósito.** Los pilots escriben en `reports_pilot/`: `run_matrix` da por hecho un punto de la rejilla si existe `reports/<run_name>/summary.json`, y un pilot con LR de rejilla y seed 0 dentro de `reports/` se contabilizaría después como run de producción, entrenado con el presupuesto viejo.
- **Módulo aparte y no flag `--pilot`**, para no llenar de condicionales (out_dir, épocas, ejes de barrido) el launcher de producción justo antes de usarlo en serio. `run_pilot` reutiliza de `run_matrix` el naming y las celdas (identidad espejada por construcción) y puede retirarse tras la calibración.

#### Justificación valor a valor de los hiperparámetros congelados

La matriz congelada fija números concretos (`src/config.py::FIXED_KNOBS`, `DATASET_BUDGET`; escritos explícitamente en cada YAML de `experiments/`). Aquí queda el porqué de cada uno; los que ya tienen decisión propia (rejilla de LR, seeds, métricas) solo se referencian.

- **`batch_size = 128`.** Tres razones. (1) *Comparabilidad de las métricas*: varias métricas dependen del batch de entrenamiento. La gradient disparity no es comparable entre runs con batch distinto (su varianza decrece como $1/m$; aviso documentado en la nota de Forouzesh & Thiran) y el GNS se lee relativo a B ($\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$ por TLC). Barrer B mezclaría el predictor con el hiperparámetro; fijarlo lo neutraliza. (2) *Es el valor del corpus*: Sankararaman et al. (gradient confusion) usan exactamente mini-batches de 128 en el mismo trío MNIST/CIFAR-10/CIFAR-100. (3) *Práctico*: cabe en memoria con las cuatro combinaciones dataset×modelo y da longitudes de época razonables (391 pasos/época en CIFAR, 469 en MNIST, 782 en Tiny-ImageNet).
- **`momentum = 0.9` (SGD).** Es el default canónico de la literatura (lo usan los baselines del corpus que entrenan con momentum, p. ej. Chatterjee & Zielinski). No es pregunta de la tesis, así que no se barre: cada eje extra multiplica la matriz. Dos consecuencias deliberadas: (a) las métricas leen el gradiente bruto ∇L, nunca el update con momentum, así que el valor no entra en las métricas, solo da forma a la trayectoria (mismo argumento que el weight decay); (b) la rejilla de LR de SGD está centrada *asumiendo* 0.9 (el paso efectivo estacionario se amplifica ~1/(1−β) = 10×): cambiar el momentum obligaría a recentrar la rejilla.
- **`weight_decay = 0`.** Ya justificado en [[1 - Diseño]] §Matriz de runs: las métricas leen ∇L de la pérdida, así que el decay no entra en su valor; se fija a 0 solo para no introducir un eje de trayectoria extra. Coincide además con el setup de varios papers del corpus (Sankararaman y Chatterjee & Zielinski entrenan sin weight decay precisamente para aislar la dinámica).
- **`probe_size = 256` (M).** Equilibrio entre memoria y estadística. *Memoria*: la matriz de gradientes per-sample pesa ~M×P×4 bytes. *Estadística*: M=256 da 256·255/2 ≈ 32.6k pares para las métricas del Gram per-ejemplo (confusion, stiffness, m-coherence), varianza muestral sobrada para un estimador por medición. *Comparabilidad*: M se congela porque cambia la varianza de todos los estimadores; con probes de distinto tamaño entre runs, las comparaciones entre modelos dejarían de ser válidas (es la razón del aviso de memoria en `train.py` en lugar de un límite silencioso). El probe además es fijo durante el run (mismas 256 muestras siempre): la serie temporal mide la evolución del modelo, no el remuestreo.
- **`windows = [0.05, 0.10, 0.25, 0.50, 1.0]`.** Los cuatro primeros son el barrido de fracción temprana del diseño (§Ventana temporal: el barrido es en sí un resultado reportable, H4), espaciados ~geométricamente como la rejilla de LR; el 1.0 ancla el extremo "entrenamiento completo" como referencia de saturación. Los valores se eligieron junto a los presupuestos para que cada fracción caiga exacta en frontera de época.
- **Presupuestos de épocas 20/40/60/80 (MNIST/CIFAR-10/CIFAR-100/Tiny-ImageNet).** Escalan con la dificultad del dataset: sin augmentation las curvas se aplanan antes que en los schedules SOTA, y el presupuesto se dimensiona para que los runs bien ajustados lleguen a meseta sin gastar épocas muertas en ~960 runs. Todos múltiplos de 20 *a propósito*: cada fracción de `windows` cae exacta en frontera de época (0.05×20=1, 0.25×60=15…), así que el snap a posteriori no introduce desfase. Son puntos de partida: el pilot puede moverlos (los YAML editados a mano sobreviven a `--init`).
- **Seeds `{0,1,2,3,4}` y rejilla de LR.** Justificados en sus decisiones propias (abajo en esta misma fecha): 5 seeds compartidas para comparación pareada SGD↔Adam y 8 LR por optimizador priorizando dispersión del predictor.
- **Métricas = todas, sin knob.** Ya no hay valor que justificar: se eliminó el knob `active_metrics` (de `Config`, `FIXED_KNOBS`, los 24 YAML y el runner), así que el código no *puede* producir runs con subconjuntos de métricas. Se computa el registro completo en toda la rejilla y la lista reportada se poda después con prueba.

#### Rejilla de LR uniforme por optimizador

Al implementar el lanzador (`src/config.py::LR_GRID`, `src/run_matrix.py`) la rejilla de SGD quedó distinta de la congelada en la matriz: una sola rejilla log-espaciada en medias décadas por optimizador, **idéntica para FC, CNN y ResNet-18** (SGD `{3e-4 … 1.0}`), en lugar de `{0.005 … 0.5}` para CNN/ResNet-18 con FC desplazada una década abajo. Adam no cambia. Se adopta la versión implementada como decisión y se actualiza [[1 - Diseño]] §Matriz de runs.

- **Por qué uniforme y no por modelo.** El lanzador deriva la identidad de cada run (y su directorio de salida) solo de (modelo, dataset, optimizador, lr, seed); una rejilla por modelo añadiría lógica condicional y rompería la simetría de la rejilla sin cambiar lo que se mide. En su lugar, un rango ancho (3,5 décadas, vs. 2 de la spec original) cubre a la vez los óptimos bajos de FC y los altos de CNN/ResNet-18.
- **El coste es asumible por diseño.** En cada celda sobran puntos en un extremo u otro (divergen o no alcanzan umbral), pero esos runs censurados son justo los que pueblan el eje de eficiencia (VD1), y con 40 runs/celda hay margen sobre el suelo n ≥ 30.
- **Simetría SGD↔Adam.** Misma forma de rejilla, desplazada una década (el paso efectivo de Adam va preescalado por 1/√v): la comparación pareada entre optimizadores (H5) no confunde forma de rejilla con efecto del optimizador.

#### Matriz de runs congelada

Resuelve el budget de cómputo y cierra la variante de ResNet. Spec ejecutable en [[1 - Diseño]] §Matriz de runs.

- **Rejilla completa, sin recortar.** Se ejecuta la matriz entera: {MNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet} × {FC, CNN simple, ResNet-18} × {SGD, Adam} = 24 celdas. El presupuesto de cómputo (riesgo #1 de [[1 - Diseño]]) deja de ser limitante y se descarta el subset "~18-24 runs".
- **Profundidad por celda.** 8 LR × 5 seeds = 40 runs/celda → ~960 runs, por encima del suelo n ≥ 30. A conteo fijo se prioriza tener más LR distintos (dan la dispersión del predictor) sobre más seeds (que dan intervalos de confianza). Mismas seeds {0,1,2,3,4} en todas las celdas para comparación pareada entre SGD y Adam (sostiene H5).
- **Tiny-ImageNet entra.** Como cabe en cómputo, se confirma el condicional anterior: cuarto dataset, sube el techo de dificultad sobre CIFAR-100. Actualizado [[1 - Diseño]] §Setup de entrenamiento.
- **ResNet-18 fija la variante.** La rejilla se congela con ResNet-18 (adaptada a imágenes pequeñas, ya en código). Cierra la pendiente "Variante de ResNet".
- **Todas las métricas implementadas en toda la rejilla.** Computar el conjunto completo de antemano no contradice "no añadir métricas a posteriori": la lista *reportada* se decide luego por poda con prueba. (Corregido: esta entrada decía que el cluster hacía viables las caras y que en ResNet-18 las per-sample iban last-layer-only. Ninguna de las dos se sostiene. Hay una sola GPU desde el 2026-07-17, y lo que hace viables las caras es el troceado en filas del barrido per-sample, que recorre todos los parámetros; GWA es la única métrica last-layer.)

#### Decisiones de ejecución

Refinan el diseño cerrado de [[1 - Diseño]].

- **Medir durante todo el trayecto.** Las métricas se registran a lo largo de *todo* el entrenamiento, no solo en la fracción $f$. Da la serie temporal completa y permite elegir el $f$ predictivo a posteriori.
- **Poda de métricas redundantes, con prueba.** Si dos métricas se comportan casi igual y miden casi lo mismo (pares colineales: GNS≈B·NGV, GSNR primo de NGV, clúster del Gram per-ejemplo), descartar una para aligerar análisis y redacción, pero solo demostrándolo (correlación alta, comportamiento solapado). Permite el análisis a nivel de familias.
- **Baseline = loss (confirmado).** El baseline es la curva de loss (TSE + val-loss tempranas); toda métrica de gradiente se juzga por su valor incremental sobre ella. Detalle en [[1 - Diseño]] §Baselines y §Hipótesis a contrastar (H2).

### 2026-05-14

#### Setup base: datasets y arquitecturas

Fija el núcleo experimental del estudio. Detalle en [[1 - Diseño]] §Convergencia de la literatura.

- **Setup mínimo por convergencia de la literatura.** Datasets MNIST + CIFAR-10 + CIFAR-100; familias de arquitectura FC + CNN simple + ResNet; optimizadores SGD y Adam (mínimo). Es el núcleo común de los 15 papers con setup.
- **Variante de ResNet, abierta.** El resto es firme; la variante concreta quedó pendiente (cerrada el 2026-06-09 con ResNet-18).
