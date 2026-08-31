# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

Qué entra en una entrada: la decisión, su porqué en corto, la evidencia con su número, y las trampas. Nada más. El registro se poda mientras se trabaja, y las decisiones y sus fechas no se podan nunca; la regla completa está en [[3 - Progreso]] §Plan por objetivos.

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] (Plan hasta la entrega).

**El método de análisis, entero (abierto el 2026-08-25).** Los 960 entrenamientos están hechos y sus datos versionados en `reports/`. No hay método definido para analizarlos: el plan anterior se retiró (ver el log del 2026-08-25). Lo que queda por decidir es, hipótesis por hipótesis, con qué cuenta concreta se responde. Es la única pendiente, y bloquea el capítulo de resultados.

Tamaño del problema de multiplicidad, para tenerlo presente al decidir: `metrics_at_window.parquet` registra **27 columnas de predictor**, porque cada métrica escribe varias variantes (*stiffness* seis, *gradient confusion* cinco, GSNR tres, TSE cuatro, el resto una o dos). Una correlación consume los 40 entrenamientos de una celda, así que cada elección de predictor, ventana e indicador de eficiencia produce 24 coeficientes, uno por celda. Con 27 predictores, 4 ventanas tempranas y los indicadores de eficiencia en juego, el factorial completo son miles de contrastes sobre las mismas 24 celdas. El método tiene que decidir qué subconjunto se contrasta y cómo se corrige la multiplicidad, no barrerlo entero.

**El umbral de VD1, en consulta con el tutor (abierto el 2026-08-31).** Los umbrales de val-accuracy son por dataset y no por modelo (0,97 / 0,65 / 0,35 / 0,20), calibrados sobre el pilot el 2026-06-17 y por tanto sobre datos distintos de los que se analizan. La consecuencia medida es que 466 de 960 runs quedan censurados y que las seis celdas de FC fuera de MNIST no tienen VD1 en ningún run. Tres caminos, cada uno con su precio. **Dejarlo y tratar la censura en el estadístico**, con concordancia sobre pares comparables: precio, esas seis celdas se declaran como limitación y se cubren con VD2 y VD3. **Umbral por modelo**: rescataría FC, pero subiría el umbral de ResNet-18 hacia su techo y le quitaría la cobertura casi completa que hoy tiene (35 y 40 cruces de 40 en CIFAR-10), y cambiaría el constructo de VD1 de "épocas hasta resolver la tarea" a "épocas hasta llegar al techo de esta arquitectura", con reescritura de diseño y metodología. **Bajar el umbral por dataset**: menos censura, pero adelanta los cruces y agrava el solape con la ventana de medida, que es justo lo que hace contestable OE4. El intercambio no tiene óptimo: el mismo knob juega en contra de sí mismo. **Cambiar de idea es barato en cómputo**, porque ninguna métrica depende del umbral y VD1 se recalcula desde la curva de val guardada en `trajectory.parquet` sin reentrenar; lo caro es el texto, `config.py`, los 24 YAML, [[1 - Diseño]] y `metodologia.tex`. Mientras esté abierta, el mapa de la fase A se calcula con los umbrales actuales y se recalcula si cambian.

El 2026-08-27 se retiró de este log la tanda de decisiones del 2026-08-26, la que abría la primera pasada de la fase A, junto con el código y el texto que produjo. La fase A se rehace después de la fase 0, así que sus decisiones se vuelven a tomar entonces y se registran aquí con la fecha en que se tomen. Lo único que sobrevive de aquella pasada es la **convención de registro** del censurado, que `metodologia.tex` §Variables declara: un valor censurado se anota como ausente y nunca como el presupuesto. La población de análisis y el tratamiento del censurado en el contraste **no** están escritos en ninguna parte, y son trabajo de la fase B (corregido el 2026-08-29; esta entrada daba por escritas las dos cosas).

## Tomadas (log)

### 2026-08-31

#### El mapa de lo computable cuenta tres estados, y la censura de VD1 se mide en pares y no en runs

**Qué cuenta el mapa.** Por cada run y cada variable dependiente, tres estados: el valor está, el valor falta, o el valor está pero no es una medición. El tercero existe solo en un sitio conocido, los 41 runs divergidos en `final_test_acc` y `final_gap_acc`, donde `argmax` sobre logits NaN devuelve siempre el índice 0 y la cifra resultante es la frecuencia de la clase 0. El mapa no quita nada a nadie: la fase A señala y la fase B interpreta, que es la línea trazada el 2026-08-29.

**Por qué la censura de VD1 se cuenta en pares.** Un run censurado no es un dato perdido. No se puede ordenar contra otro censurado, pero sí contra cualquier run que sí cruzó, porque tardó más que él. Lo que una correlación de rangos consume son pares comparables, no observaciones sueltas. En una celda con 15 cruces de 40 quedan 480 de los 780 pares, el 61,5 %, y no el 37,5 % que sugiere contar runs. Contada en runs, la censura de este trabajo parece casi el doble de grave de lo que es.

**Evidencia medida sobre los 960.** 466 runs sin VD1. De ellos **240 son las seis celdas de FC fuera de MNIST**, donde no cruza ninguno, y los otros **226 se reparten por las dieciocho celdas restantes**. En esas dieciocho la fracción de pares comparables va del 61,5 % al 100 %, y once pasan del 85 %. El dato incómodo: solo **8 de las 24 celdas** llegan a 30 runs con VD1, que era el suelo de n fijado al congelar la rejilla.

**Hay dos censuras distintas y no significan lo mismo.** En las dieciocho celdas vivas es censura de tipo I, administrativa: el presupuesto es fijo, igual para todos los runs del dataset y conocido antes de mirar resultados, que es el caso más benigno y el que la estadística de supervivencia trata de serie. En las seis de FC no lo es: la curva había llegado a meseta por debajo del umbral y no habría cruzado nunca, con evidencia directa del pilot, que corrió esa configuración a presupuesto doblado y la vio plana desde la época 40. Tratar las dos con la misma herramienta rompe el supuesto de que el suceso acaba ocurriendo. El mapa las separa por la distancia al umbral, que es descriptiva y no exige criterio nuevo.

- **Trampa: la ausencia de VD1 no es un agujero de datos, es un resultado.** Que FC no llegue al umbral en tres datasets es el techo de la arquitectura, medido, y así hay que escribirlo. Lo que se pierde no es información sobre esas celdas, es una de las tres formas de medir velocidad en ellas; VD2 y VD3 no se censuran nunca.
- **Trampa: VD4 y VD6 están completas al 100 % y 41 de sus valores no son mediciones.** Es la única pareja de variables donde "no falta ningún valor" induce a error, y es justo por eso por lo que el mapa tiene tres estados y no dos.
- **Sin medir, y declarado:** dentro de las dieciocho celdas vivas no se distingue run a run entre "iba subiendo y se acabó el presupuesto" y "estaba en meseta por debajo del umbral". La separación por techo está probada a nivel de celda para FC, no a nivel de run.

#### El censo de runs cierra los cuatro recuentos: 154 nunca aprendieron y ninguno queda sin explicación

**Qué etiqueta y qué no.** `src/efficiency.py::run_health` da a cada run dos columnas separadas. `learned` dice cómo acabó, si superó 1,25 veces el azar. `failure` dice qué firma apareció en alguna época, divergencia, colapso o ninguna. Las columnas `*_frac` dicen en qué proporción del run. No excluye a nadie, que es la regla de la fase.

**Por qué dos columnas y no una.** La primera versión daba una sola etiqueta y clasificaba `resnet18_cifar100_sgd_lr1.0_seed2` como colapsado: colapsa en 5 de sus 40 épocas, se recupera y acaba a 24,7 veces el azar. Con una etiqueta, un run sano se leía como una ruina. Cómo acabó y qué le pasó son dos preguntas y piden dos columnas.

**Los cuatro recuentos del vault eran correctos y les faltaba el adjetivo.** El mismo dato se lee distinto según se pregunte "roto en alguna época" o "roto en todas", y ninguna entrada decía cuál. Medido sobre los 960: **165** runs tienen firma en alguna época, 41 divergidos y 124 colapsados, y **133** la tienen en todas, 39 y 94. Los 124 y los 94 del cero exacto de GWA son ese mismo par. Desde ahora la extensión va en el dato, en `nan_frac`, y no en la prosa.

**154 sin aprender, y ninguno sin causa conocida.** 115 colapsados más 39 divergidos dan exactamente los 154, y **cero runs se quedan en el azar sin firma mecánica**. El umbral tampoco importa: 1,2 y 1,25 seleccionan los mismos 154, así que la contradicción entre las entradas del 25 y del 27 no cambia ningún resultado ni toca el reparto por optimizador que sostiene la decisión de OE5.

**ResNet-18 no perdió ni un run**, los 320 de sus ocho celdas. Es la única de las tres arquitecturas con normalización por lotes, y el único ResNet-18 que llegó a colapsar se recuperó solo.

- **Trampa: el margen es una línea recta sobre un continuo.** Once runs muestran firma y aprenden igual, algunos por muy poco. `fc_cifar100_sgd_lr0.3_seed2` colapsa en el 92,5 % de sus épocas y acaba a 1,35 veces el azar, justo por encima de la línea. "Aprendió" no es lo mismo que "sirve", y esa segunda decisión es de la fase B.
- **Trampa: en los divergidos la accuracy de test sigue sin ser una medición**, y este censo no la toca (ver la entrada del 2026-08-29). `learned` se lee sobre la curva de validación, que sí es real hasta la época en que el run revienta.

#### El código de run y celda va a `efficiency.py`, y `analysis.py` se queda con las columnas

**El corte va por unidad de observación.** `analysis.py` trabaja sobre columnas de `trajectory.parquet` y se queda tal cual. El código que mira runs y celdas, que lee `summary.json`, va a un módulo nuevo, `src/efficiency.py`, y lo crea el punto 1b. Son los cuatro puntos que quedan de la fase A: 1b, 2, 4 y 5.

**Por qué ahora y no más tarde.** Partir sale gratis hoy, porque **nada en `src/` importa `analysis`**: sus únicos consumidores son `tests/test_analysis.py` y su propio `_main` de consola. En cuanto la fase B empiece a importar de estos módulos, mover una función entre ellos ya tocará ficheros de terceros.

**Por qué ese nombre.** Continúa vocabulario que ya existe en el código y en la memoria: las seis VD las produce `train.py::efficiency_summary` y se llaman indicadores de eficiencia. No inventa una palabra nueva para algo que ya la tiene.

- **Debilidad asumida del nombre.** Describe bien los puntos 2, 4 y 5 y solo de refilón el 1b, que es validez y no eficiencia. Se acepta porque en un `src/` plano de doce ficheros se busca por nombre de función y por docstring, no por nombre de fichero, y porque partir en dos módulos para unas trescientas líneas es más estructura de la que el problema pide.
- **Lo que ese módulo no va a contener:** ninguna correlación entre métrica y variable dependiente. Eso tendrá casa propia en la fase B, así que el nombre no encierra nada.

#### El número de clases sube a `config.py`; el nivel de azar nunca ha existido en el código

**`NUM_CLASSES` se define en `config.py` y `data.py` lo lee**, igual que ya hacía con `SPLIT_SEED`. Es un hecho del conjunto de datos que usan los dos lados del proyecto, y config es donde este repositorio ya guarda lo que usan los dos lados. Descartado importar `data` desde el análisis, que cuesta 1,40 s frente a 0,03 s y ataría la capa de análisis a torch para siempre a cambio de un entero.

**El nivel de azar 1/K y el factor 1,25 no suben a config**, porque no son diseño sino criterio nuestro, discutible y revisable. Viven en el módulo de análisis, para que cambiarlos no toque el diseño.

**Y el hallazgo que obliga al punto 1b: ese criterio nunca ha estado en el código.** Buscado en `src/`, en `tests/` y en los dos únicos scripts de análisis que han existido, `plots.py` borrado el 2026-08-27 y `power_analysis.py` borrado el 2026-08-25: no aparece en ninguno. El recuento de 154 se calculó a mano en una sesión y solo sobrevivió el resultado, escrito en prosa. De ahí que el vault se contradiga a sí mismo: la entrada del 2026-08-25 escribe el umbral como 1,2 y la del 2026-08-27 como 1,25, y las dos dicen 154.

- **Cuatro recuentos sueltos que no cuadran, y el 1b los deja en uno.** 133 runs con NaN, 154 clavados en el azar, 124 con la firma de `gwa/score_mean` a cero exacto y 115 de esos que acaban clavados. La entrada del 27 llama a 154 "el mismo recuento" que esa firma, que la del 25 cifra en 124.
- **Trampa: el 154 sostiene una decisión ya tomada.** Con él se decidió el 2026-08-27 que OE5 compara las ocho posiciones de cada rejilla y no el solape de seis, apoyándose en 72 muertos de 480 en SGD frente a 82 de 480 en Adam. Si al recalcularlo el reparto cambia, hay que releer esa decisión y no solo corregir la cifra.

### 2026-08-29

#### Punto 1 de la fase A: las columnas son válidas, y un cero falso que confirmaba la teoría

**Resumen, en las seis preguntas que hay que saber responder.**

*Qué pasó.* Al validar las columnas sobre los 960 runs, cuatro de ellas daban un cero exacto en 41 entrenamientos mientras las columnas hermanas, calculadas sobre la misma matriz y en la misma función, daban NaN.

*Cuál era el error.* En esos 41 el entrenamiento divergió y los gradientes son NaN. Casi toda la aritmética propaga el NaN, pero dos operaciones no: `torch.sign(NaN)` devuelve 0,0 en PyTorch, y `NaN < 0` es falso. Así que `stiffness/sign_global`, `sign_within`, `sign_between` y `confusion/frac_neg` publicaban un cero limpio donde no se había medido nada.

*Cómo se corrigió.* Dos líneas, una en `stiffness.py` y otra en `gradient_confusion.py`, para que esas operaciones propaguen el NaN como hace el resto, con una prueba cada una. Y los datos ya escritos: 6.484 celdas en 82 ficheros pasaron de 0,0 a NaN.

*Si tocó lo correcto.* Sí. La regla del parche exige dos condiciones a la vez, así que los 17 ceros legítimos de la matriz siguen intactos, y el código nuevo da diferencia 0,0 frente al viejo sobre 200 matrices sin NaN.

*Cómo perjudicaba.* En los runs sanos `stiffness/sign_within` vale 0,90 de media; en las 1.621 celdas afectadas valía 0,00 exacto, y el resultado final de esos runs es el peor posible. El punto falso caía en la esquina del gráfico, métrica en su mínimo y resultado en el peor, así que una correlación habría leído "poca alineación temprana, peor resultado final", que es **la dirección que predice la teoría**.

*Por qué convenía arreglarlo.* Porque no metía ruido, metía señal. Un error que mete ruido hace fallar por prudente y se nota; este hace acertar por accidente y no se nota nunca, porque el resultado sale como se esperaba. Y era **invisible para el propio chequeo de validez**: cero está dentro de rango, no es NaN, no es infinito y no rompe ninguna identidad, así que pasaba las cinco comprobaciones. Lo delató solo compararlo con la columna hermana. De ahí que el punto 1 tenga que mirar los runs y no solo las columnas.

**Veredicto sobre los 960.** Ningún valor se sale de su rango teórico y ningún valor es infinito, sobre 33.600 filas por 30 columnas. Las cinco identidades exactas se cumplen sin una sola violación en 32.182 filas. Y **ninguna columna falta nunca**: los 960 runs registran las mismas 41 columnas, comprobado leyendo el esquema de cada parquet, así que en toda la matriz ninguna métrica llegó a lanzar una excepción. Lo único anómalo son los NaN.

**Los NaN tienen dos causas, y las dos son el learning rate demasiado alto.** 39 runs divergieron entero, todos SGD en los dos valores más altos de su rejilla (29 a lr 1,0 y 10 a lr 0,3), más 2 que divergieron a mitad (`fc_mnist_sgd_lr0.3`, semillas 0 y 1). Otros 94 no divergieron: se quedaron con la pérdida clavada en ln(K) y la accuracy en el azar, con `gwa/score_mean` valiendo **cero exacto en sus 3.320 filas** y `var/avg` en 6·10⁻⁷, el rastro de una red cuyas activaciones ocultas colapsaron a cero y en la que solo sobrevive el gradiente del sesgo de la última capa. **Ni un solo ResNet-18 aparece en ninguno de los dos grupos**, que es la única de las tres arquitecturas con normalización por lotes.

**Y `reports/` se parchea, decisión de Lai:** esos datos son la fuente de la verdad del trabajo, así que se corrigen en vez de enmascararse al leer. La regla se aplica por fila y exige las dos condiciones, coseno del mismo ámbito a NaN **y** celda a 0,0 exacto, porque un subconjunto de pares vacío da un 0,0 legítimo con coseno sano y ésos no se tocan (hay 11 y 6 de esos en la matriz). Resultado: 41 runs, 82 ficheros, 1.418 filas de trayectoria y 203 de ventanas, 6.484 celdas de 0,0 a NaN, sin que se mueva ninguna otra columna. **No se re-entrena nada.**

- **Aguas abajo no hay que hacer nada, y ese era el objetivo del parche.** Desde el 2026-08-29 `reports/` ya trae el NaN, así que ninguna fase posterior tiene que acordarse de estas cuatro columnas ni escribir una excepción para estos 41 runs: se comportan como cualquier valor ausente. Lo único que sigue pidiendo criterio son las dos columnas de accuracy, y ese criterio es de la fase B.
- **En la memoria no se narra, decisión de Lai.** La memoria describe un método y reporta resultados, no el historial de desarrollo, igual que no enumera ningún otro fallo corregido mientras se construía. El código entregado es el corregido, los datos entregados son consistentes con él, y **ningún resultado dependió nunca del fallo**, porque se encontró antes de calcular la primera correlación.
- **La única regla que sí obliga: no afirmar lo contrario.** Callarlo vale; escribir que `reports/` es exactamente lo que produjeron las ejecuciones, o que no se aplicó ningún posprocesado, no, porque sería falso y el historial de git lo enseña. Basta con no escribir esa frase.
- **En esos 39 runs ninguna de las seis VD es una medición.** Cuatro están ausentes: épocas-hasta-umbral, AUC, mejor val-loss y gap de pérdida. Las otras dos traen número pero no significan lo que parecen: la accuracy de test vale un único valor por dataset (0,100 en CIFAR-10, 0,010 en CIFAR-100, 0,098 en MNIST, 0,005 en Tiny) porque un modelo con logits NaN pasa por un `argmax` que devuelve siempre el índice 0, así que esa cifra es la frecuencia de la clase 0 en cada test y no el azar. El gap de accuracy sale casi nulo por lo mismo.
- **Esas dos columnas de accuracy no se parchean, y la línea es deliberada.** El parche corrige lo que el código, ya arreglado, emitiría distinto. `evaluate_test` no está arreglado ni tiene que estarlo: calcula de verdad la accuracy de lo que el modelo predijo. Declarar esas dos cifras inservibles es interpretación, y la interpretación es de la fase B. La fase A las deja donde están y las señala.
- **El NaN de GWA no es un defecto.** `_gwa_aggregate` detecta que la varianza de los cosenos es cero y devuelve NaN a propósito, porque la curtosis de una constante no existe. Conserva `gwa/score_mean`, que sí está definida.

#### El `README.md` certificaba lo contrario de la verdad; el pilot pasa a versionado y `CLAUDE.md` no se versiona nunca

**El `README.md` afirmaba que el plan de análisis precedía a los datos**, con estas palabras: "the git history itself certifies that the plan precedes the data". Cuatro líneas más arriba, el mismo fichero dice que el plan se retiró el 2026-08-25. Corregido con la cronología escrita: la matriz terminó el 22 y el plan se retiró el 25, así que el método es posterior a los datos. **Trampa:** el barrido que quitó esa afirmación del vault miró `docs/` y `thesis/`, y por eso sobrevivió en la puerta de entrada del repositorio.

**`reports_pilot/` pasa a versionado.** Son 2,5 MB y 120 ficheros, con los seis `testfix_40ep/` de Tiny dentro y los seis `summary.json` que se auto-declaran corruptos con la clave `_tiny_test_note`. Es la evidencia que justifica los presupuestos y umbrales de `config.py::DATASET_BUDGET` y hasta hoy vivía en un solo disco. Sale con él `reports_validity/` del `.gitignore`, que era un resto de la primera pasada retirada de la fase A y habría dejado sin versionar en silencio cualquier salida futura con ese nombre.

**`CLAUDE.md` no se versiona nunca (decisión de Lai).** Es un fichero de instrucciones para una IA y el repositorio acaba depositado. Se queda en `.gitignore` con el motivo escrito al lado, para que nadie lo "arregle" más adelante. Consecuencia asumida: sus 13 KB de arquitectura existen solo en local, y la fuente autorizada del estado sigue siendo el vault, que sí está versionado.

#### La fase A cuenta pero no excluye, y su veredicto se parte en dos en la memoria

**La fase A cuenta, no excluye.** Un run clavado en el azar tiene accuracy de test, así que entra en el recuento. Si además se descarta, eso es método y es fase B. Mezclar las dos cosas convertiría el recuento en una decisión tomada mirando los datos y sin registrarla.

**Dónde vive el veredicto en la memoria (decisión de Lai): partido en dos.** Los números van a una sección nueva al principio de `resultados.tex`, porque son algo calculado sobre la matriz. La regla que se derive de ellos, la población de análisis, va a §Protocolo de análisis y la escribe la fase B. Separa el dato del método, que es la frontera que ya usa el resto de la memoria. Se descarta meter las dos cosas en Metodología, que pondría resultados calculados en el capítulo que describe lo que se va a hacer.

- **Trampa que parece romper la regla de un lado cada vez y no la rompe.** El punto 5, la concordancia entre validación y test, sí es una correlación, pero entre dos variables dependientes. No entra ninguna métrica de gradiente, así que no toca la relación que la fase A tiene prohibido mirar.
- **El punto 1 se ejecuta sobre los 960, sin excluir a nadie.** Un chequeo de validez tiene que ver los casos patológicos, que es donde aparecen los fallos: los 154 clavados en el azar son justo donde una métrica emite un valor sin sentido, y de hecho ahí `gwa/score_mean` vale cero exacto.

#### Se retiran los últimos criterios de decisión supervivientes, y `Métricas.md` se pone al día con el código

Desde el 2026-08-25 el proyecto afirma que ninguna hipótesis tiene criterio de decisión, y **no era cierto**: quedaban cinco restos del plan retirado. [[1 - Diseño]] fijaba un corte en |ρ| < 0,3 como falsación, prometía "Spearman + Pearson con corrección FDR" en el diagrama del procedimiento, prescribía tratar el censurado como peor rango, y prescribía estandarizar dentro de condición o usar efectos mixtos para agregar; `resultados.tex:15` llevaba el último FDR de la memoria. Los cinco se retiran y el método vuelve a estar entero por definir.

Importa por la fase H y no por higiene: la declaración de que el análisis es posterior a los datos y anterior a los resultados solo se sostiene si el repositorio no contiene ningún criterio escrito de antes. Desde hoy no lo contiene, y esta fecha es la evidencia.

- **Trampa que habría invertido un resultado.** El enunciado de H6 daba "GWA alta → mejor generalización", que es el signo del artículo, porque Hölzl define el gradiente como $-\nabla\ell$. Aquí se mide sobre $\nabla\ell$ bruto, así que la predicción heredada es la contraria. La conversión estaba hecha en `fundamentos.tex:168`, pero no en el enunciado de la hipótesis ni en la tabla de signos de [[Datos experimentales]] §5.3, que es justo el material contra el que se contrasta H6. Las dos llevan ya el aviso.
- **[[Métricas]] describía un código que dejó de existir.** Decía que la gradient confusion usa 50 pares y comparte un barrido con `cos_sim_batches`, una métrica que no existe, cuando el código reduce del Gram del barrido compartido con las 256 muestras. Decía "las diez métricas del registry", que son ocho. Daba por pendiente el last-layer de la stiffness en ResNet-18, resuelto con el troceado en filas. Y clasificaba el coste **al revés**, llamando baratas a `var/normalized` y `gradient_disparity`, que son las dos únicas que todavía pagan pasadas hacia atrás propias.
- **La causa se repite y conviene reconocerla.** Las cuatro salen de la misma raíz: el documento se escribió en junio describiendo la v1 y el barrido compartido entró dos días después. Es el mismo patrón del 3,21x corregido el 2026-08-08. Una optimización que cambia la estructura del cálculo invalida en silencio toda la documentación escrita en términos de esa estructura, y nada avisa.

### 2026-08-27

#### La comparación SGD↔Adam se hace sobre las ocho posiciones, no sobre el solape de rejillas

Las rejillas de LR de SGD y Adam **se solapan en seis de sus ocho valores**, de 3e-4 a 1e-1; solo SGD tiene 0,3 y 1, y solo Adam tiene 3e-5 y 1e-4. Es consecuencia mecánica del desplazamiento, porque una década son dos saltos en una rejilla espaciada a medias décadas. **La comparación pareada de OE5 se hace sobre las ocho posiciones** y no sobre esos seis valores compartidos al mismo LR nominal.

- **Decide el reparto de los fallos, medido sobre los 960 `summary.json`.** Con las ocho posiciones, SGD tiene 72 entrenamientos clavados en el azar de 480 y Adam 82 de 480, un 15 % frente a un 17 %, y los dos fallan por el mismo lado, el de paso demasiado grande. Restringido al solape, SGD cae a 3 de 300 y Adam se queda en 82 de 300, un 1 % frente a un 27 %. La restricción convierte una comparación equilibrada en una desequilibrada, porque **el mismo valor nominal no es el mismo régimen** en los dos optimizadores: sobre el solape, SGD recorrería de lento a bueno sin fallar casi nunca y Adam de bueno a muerto uno de cada cuatro.
- **De paso queda validado el desplazamiento de 10×**, que hasta hoy era una suposición tomada de los valores por defecto canónicos y del *momentum* 0,9, que amplifica el paso efectivo de SGD unas diez veces. La matriz completa lo comprueba a posteriori: que los dos optimizadores fallen casi al mismo ritmo, 72 frente a 82 muertos y 227 frente a 239 censurados, es lo que se observa cuando las dos rejillas cubren tramos comparables de sus rangos respectivos.
- **Se descarta también el solape como comprobación de robustez**, que fue la primera propuesta. Sería un control peor que aquello que controla: con un desequilibrio propio del 1 % frente al 27 %, una discrepancia en OE5 no permitiría distinguir un desplazamiento mal elegido de un subconjunto torcido.
- **Criterio de "clavado en el azar"** usado en estas cuentas: *accuracy* de validación máxima por debajo de 1,25 veces el azar del conjunto de datos. Da exactamente 154 entrenamientos, el mismo recuento que la firma `gwa/score_mean` = 0,0 que ya estaba registrada, aunque no se ha comprobado que sean el mismo conjunto de 154.

#### El coste por métrica se dará en notación asintótica; el micro-benchmark queda aparcado

**El coste por métrica no existe en los 960 entrenamientos**, y no es que no se midiera: cada `summary.json` guarda un único `metric_seconds`, y `stream_shared` ejecuta el barrido por ejemplo una sola vez para seis de las ocho métricas, así que su coste marginal individual no es recuperable ni a posteriori. Peor aún, con un barrido compartido **"el coste de la m-coherencia" no está bien definido**: si se calculan las seis, su coste marginal es casi nulo; si se calcula sola, paga el barrido entero. Ese es el eje que `estado-del-arte.tex:111` promete como frente de Pareto.

**Decisión de Lai (2026-08-27): primero el coste teórico en notación asintótica; el micro-benchmark se aparca.** El coste asintótico es mejor que el reloj para este fin, porque no depende de la máquina ni de qué otras métricas corran a la vez, y separa las ocho en ligas según cuántas pasadas hacia atrás exigen, que es la unidad que manda.

- **Consecuencia que hay que asumir:** con coste asintótico el eje deja de ser continuo y pasa a ser una escala de tres o cuatro niveles, así que lo que se puede dibujar honestamente **no es un frente de Pareto** literal. Resuelto el mismo día en los cuatro sitios de la memoria que lo prometían: lo que se compara es dentro de cada clase de coste y frente al predictor de referencia, que no calcula ningún gradiente, en §Coste y capacidad predictiva de `resultados.tex` (`sec:res-coste`). La sección se mantiene en lugar de doblarse dentro de OE2, porque las clases de coste dan material propio y OE2 pregunta otra cosa; **esa parte es decisión mía y es revisable**.
- **Decidido por Lai: la tabla del estado del arte no se toca y el coste va en una tabla nueva**, `tab:coste-metricas` en `implementacion.tex` §Cálculo de las métricas, con un solo propósito, lo que cuesta medir y lo que ahorra el barrido compartido. La memoria se queda fuera de esa tabla y sigue contada en prosa, porque responde a otra pregunta. Queda descartada la propuesta anterior de repartir el coste entre Fundamentos e Implementación, que partía en dos capítulos justo el contraste que importa.
- El micro-benchmark sigue disponible el día que se quiera el eje continuo: minutos de cómputo, fuera de la matriz y sin tocar `reports/`.

**Resultado de la derivación**, con cada término trazado a la línea que lo produce. Las derivadas no separan a ninguna métrica, porque las ocho recorren el *batch* de medición una vez. Lo que separa es la aritmética posterior, en dos niveles: `M·P` las de momentos y `M²·P` las de pares. En el margen, con el barrido ya hecho, **seis de las ocho no derivan nada**: `P` la coherencia y la escala de ruido, `P·log P` el GSNR, `M` la GWA, `M²` la *stiffness* y `M²·log M` la confusión. Solo la varianza normalizada y la disparidad siguen pagando sus diez y cinco pasadas por submuestra. Medir el registro completo cuesta unas tres pasadas sobre el *batch*, no nueve.

- **Dos matices que cambian la lectura.** La GWA solo lee la última capa, pero deriva la red entera para cada ejemplo y descarta el resto, así que su ventaja está en la aritmética; y esa capa no siempre es pequeña, en la CNN sobre Tiny-ImageNet son 102.400 de 116.936 pesos. El término cuadrático domina sobre las derivadas solo en la red *fully connected*, donde una pasada hacia atrás cuesta del orden de `P`; en las convolucionales la comparación puede invertirse.
- **Aviso para cuando se escriban los resultados:** el reloj `metric_seconds` incluye la aritmética del predictor de referencia, que además rehace el historial entero cada época y crece con el cuadrado del número de épocas. La cantidad es despreciable, pero `metric_seconds` no es exactamente "lo que cuesta instrumentar el gradiente".

#### Estilo de figuras: al ancho final, con la tipografía del cuerpo, y el color nunca solo

Definido desde cero en `src/figstyle.py`, con las notas de [[Estilo de redacción - notas del TFM HOFT]] §Estilo de las figuras como referencia, y aprobado por Lai sobre una demo con datos inventados. No añade ninguna biblioteca: matplotlib ya estaba en el grupo de desarrollo.

Medidas tomadas antes de decidir, no supuestas: el bloque de texto son **15 cm** (A4 con 3 cm de margen por lado, `tfgetsinf.cls:45`), el cuerpo son **10 pt en Palatino** (`\LoadClass{book}` sin opción de tamaño, más `mathpazo`), y **TeX Gyre Pagella**, el Palatino libre, está en el árbol de TeX Live y matplotlib puede cargarlo.

- **La figura se genera al ancho final y LaTeX no la escala nunca.** Solo hay dos anchos, el completo de 15 cm y uno estrecho de 10 cm, y los fija el módulo, de modo que en el `.tex` no debe aparecer ningún `width=0,8\textwidth`. Esto ataca el defecto que se midió en HOFT, donde dos gráficas quedaron con los rótulos a la mitad del tamaño del texto por exportarlas a un tamaño y encogerlas a otro.
- **Trampa que anula la decisión anterior si se toca:** `bbox_inches="tight"`, el idioma habitual para guardar en matplotlib, recorta el PDF al contenido y por tanto **cambia el tamaño final**. El módulo usa `constrained_layout`, que mantiene el tamaño y mete el contenido dentro. Está avisado en el código porque es justo lo que alguien "arregla" sin saber que lo rompe.
- **El color no lleva nunca información él solo:** el ciclo empareja cada color con su propio trazo. La primera paleta candidata **suspendió** la prueba de gris, con naranja y verde a 0,011 de luminancia, o sea el mismo gris; la definitiva tiene una separación mínima de 0,096, y una prueba exige que el mínimo supere 0,05.
- **Reglas que puso Lai al aprobarlo (2026-08-27).** Salida **siempre en PDF**, porque se quiere máxima calidad; un PNG es previsualización y nunca entregable. **Ningún eje cortado por encima del cero**, que exagera las diferencias. Y **dos paneles de la misma cantidad comparten escala**, porque dibujar cada uno en su rango hace parecer iguales valores que no lo son. Las dos últimas están en `include_zero()` y `match_limits()` en vez de en un comentario, con pruebas, porque la demo aprobada las incumplía las dos.
- **Separación deliberada de HOFT:** fuera los ejes superior y derecho. Allí las cuatro gráficas conservan la caja cerrada, pero eso es el valor por defecto de matplotlib y no una decisión suya.
- **Se copia la regla de figura o tabla de HOFT**, que es observable y no admite discusión: cómo cambia algo al mover una perilla continua va a gráfica; qué gana en qué prueba va a tabla con el mejor valor en negrita; cómo funciona un mecanismo va a esquema. Junto con la regla ya vigente de una figura, una afirmación.
- **Qué NO se ha escrito, y a propósito:** ninguna ayuda para gráficas concretas, porque la primera figura real llega con la fase A y escribirlas ahora sería adivinar qué forma tendrán.

#### `seconds_to_threshold` queda registrada pero no se declara variable del estudio

Todos los runs escriben `seconds_to_threshold` junto a `epochs_to_threshold`, pero no está entre las seis VD de [[1 - Diseño]]. **Se menciona en §Variables como disponible y no se declara variable del estudio.** Es la lectura de velocidad honesta al comparar SGD con Adam, porque su paso no cuesta lo mismo, así que borrarla del texto sería esconder un dato que existe; declararla, en cambio, añade una séptima familia de correlaciones a un trabajo cuya multiplicidad ya es el problema abierto, y esa elección pertenece al método de análisis y no a la sección de variables. Si el método la necesita, se declara entonces y se registra aquí.

#### Terminología: anglicismos bien conocidos, y siempre en cursiva

Se admiten en la memoria los términos ingleses de uso corriente en el campo, y todo anglicismo va en cursiva **cada vez que aparece**, no solo la primera. Esto último deroga la convención vigente desde julio, que reservaba la cursiva a la primera aparición de cada término.

- **Por qué la cursiva siempre.** Lo manda la norma de la escuela: el material de seminarios dice "cursiva para palabras extranjeras" y remite al DLE para distinguir el extranjerismo crudo, que va en cursiva, del adaptado, que va en redonda. Ninguno de los nuestros está adaptado en el DLE. Son unas 160 cursivas en 43 páginas, algo más de tres por página.
- **Pasan a inglés:** *epoch* y *epochs* (47), *learning rate* y *learning rates* (10), *seed* y *seeds* (8). Con dos consecuencias gramaticales que hubo que resolver a mano: *epoch* y *seed* heredan el género femenino de "época" y "semilla", de modo que los artículos y adjetivos que ya estaban siguen concordando; *learning rate*, en cambio, es masculino en el uso habitual, así que sus diez apariciones se reescribieron una a una.
- **"Conjunto de medición" pasa a "batch de medición"** (17). El nombre anterior era ambiguo en castellano, porque se lee igual como "el conjunto de las mediciones", que es lo contrario de lo que designa. **Se descartó "probe"**, que es el nombre del concepto en el código y encaja por significado, porque en aprendizaje automático *probe* ya designa otra cosa muy conocida, el clasificador lineal entrenado sobre representaciones congeladas. Falla justo el criterio de admisión: es conocido, pero por otra cosa. El término queda definido formalmente en `fundamentos.tex` §Geometría del gradiente, antes de su primer uso.
- **"Tamaño de lote" pasa a "tamaño de batch"** (2). No es una decisión nueva sino la aplicación de la del 2026-07-04, que ya fijaba *batch* frente a "lote" y no se había aplicado en el estado del arte.
- **Regla para los nombres de métrica:** acrónimo en redonda, palabra inglesa en cursiva. Quedan en redonda GSNR, GWA, TSE, NGV y GNS, y también "m-coherencia", que es una adaptación al castellano. Van en cursiva *stiffness*, *gradient disparity* y *gradient confusion*, que son sintagmas ingleses usados como nombre. Se aplica lo mismo a *minibatch*, *dropout* y *weight decay*.
- **Se quedan en castellano**, porque cambiarlos sería anglicismo por anglicismo y no por claridad: "conjunto de datos" frente a *dataset*, "submuestra", "ventana", "banco de pruebas" y los conjuntos de entrenamiento, validación y test.
- **Ampliación del mismo día, al escribir §Configuración del entrenamiento: el nombre técnico inglés gana a la traducción forzada.** La regla anterior admitía el término inglés cuando no había equivalente; ahora se admite también cuando el equivalente existe pero es una perífrasis que nadie usa. Entran así la red *fully connected* en lugar de "red totalmente conectada" (tres apariciones reescritas, en introducción, fundamentos y metodología), y *max pooling*, *adaptive average pooling*, *stem* y *stride*, que en castellano solo se dicen describiéndolos. El criterio de corte sigue siendo la claridad y no el anglicismo por el anglicismo: "red convolucional" y "perceptrón multicapa" se quedan como están, porque son de uso normal en castellano.
- **Trampa que costó una reparación:** al marcar *batch* se volvió a marcar el que ya estaba dentro de *batch normalization*, y quedaron tres cursivas anidadas.

### 2026-08-25

#### Se retira el plan de análisis

Se borran `4 - Análisis.md` (el plan preregistrado y congelado), `src/power_analysis.py` con sus pruebas, y el §Protocolo de análisis de [[1 - Diseño]]. Las seis hipótesis se conservan en [[1 - Diseño]] como afirmaciones falsables, ahora sin criterio de decisión.

**Por qué.** El plan había crecido hasta un punto en que ya no se entendía por completo, y un método que no se entiende no se puede defender ante un tribunal. Se prefiere partir de una base comprensible y construir el análisis desde ahí, hipótesis por hipótesis.

**Consecuencia que hay que declarar en la memoria.** El plan estaba congelado y commiteado antes del primer resultado, lo que permitía afirmar que el análisis precedía a los datos. Al retirarlo, el análisis que se haga es **posterior a los datos** y así debe presentarse. El plan retirado sigue íntegro en el historial de git por si hiciera falta recuperarlo.

### 2026-08-08

Las tres entradas de esta fecha salen de la primera revisión de la matriz en marcha (268 runs terminados: MNIST completo, el smoke de Tiny y media celda de `fc × cifar10 × sgd`). Ninguna se ha tomado habiendo calculado ninguna correlación entre métrica y VD, que a esta fecha siguen sin existir. Ninguna toca `src/`, porque la matriz corre lanzando un proceso nuevo por run (`run_matrix.py:201`), así que cualquier edición del código entraría en vigor en el run siguiente y partiría la matriz en dos versiones, que es una contaminación que no se repara después.

#### El coste de instrumentación documentado era anterior al barrido compartido

La cifra de 3,21x que este log venía citando como peor caso procede del pilot, que se ejecutó el 2026-06-15. El barrido compartido entró dos días después (commit `8566fc3`, `perf(metrics): share one per-sample gradient sweep per probe`). **Todas las cifras de coste del vault eran, por tanto, de una versión del código que ya no es la que corre la matriz.**

Medido ahora celda a celda contra el pilot, el sobrecoste baja de 3,09x a 2,04x en `fc × cifar10 × sgd`, de 2,40x a 1,65x en `resnet18 × mnist`, de 1,77x a 1,40x en `fc × mnist` y de 1,05x a 1,02x en `cnn × mnist`. El peor caso medido sobre la matriz es **2,04x**. Las dos celdas que encabezaban el pilot (`fc × cifar100` y `fc × tiny_imagenet`) todavía no han corrido; escalando el factor observado quedarían alrededor de 2,1x, dentro de la cota <3-4x con bastante más holgura que antes.

La causa es la optimización y no otra cosa: el tiempo de entrenamiento por época es el mismo (1,671 s en el pilot frente a 1,638 s en la matriz), el de medición se reduce a la mitad, y las trayectorias de la misma configuración salen **idénticas bit a bit** durante las 40 épocas, que es justo lo que promete el invariante de la ruta compartida.

**La corrección no cambió ninguna decisión.** El coste servía para justificar cuánto había que exigirle a una métrica de gradiente para que mereciera la pena, no para calcularlo, y 2x sigue siendo un coste alto. (El margen concreto, δ_H2 = 0,15, pertenecía al plan de análisis retirado el 2026-08-25.)

#### FC no alcanza el umbral en CIFAR-10, CIFAR-100 ni Tiny-ImageNet

Verificado sobre los 960 runs (2026-08-25). Ninguna de las seis celdas de FC fuera de MNIST alcanza nunca su umbral de val-accuracy, con ninguna de las 8 tasas de aprendizaje ni ninguna de las 5 semillas. Los techos medidos: 0,584 y 0,569 contra un umbral de 0,65 en CIFAR-10; 0,295 y 0,289 contra 0,35 en CIFAR-100; 0,114 y 0,108 contra 0,20 en Tiny-ImageNet.

El contraste que lo explica: la misma red FC **sí** pasa el umbral en MNIST, con 0,987 y 0,985 contra 0,97. No es falta de presupuesto ni un fallo de configuración. El pilot corrió esa configuración al presupuesto doblado, 80 épocas, y tocó techo en la época 40 para quedarse plano el resto. Es el techo de la arquitectura.

Consecuencia práctica: en esas seis celdas, "cuántas épocas tarda en llegar al umbral" no existe para ningún run. Cualquier análisis que use ese indicador trabaja con 18 de las 24 celdas.

#### Muchos entrenamientos se quedan clavados en el azar, y se detectan por un cero exacto

Verificado sobre los 960 runs (2026-08-25). En las tasas de aprendizaje altas hay entrenamientos que no aprenden nada. Su accuracy de validación se queda en el azar. Son **154 de 960** por el criterio de "mejor val-acc por debajo de 1,2 veces el azar", repartidos por 16 de las 24 celdas, así que no es una curiosidad de un conjunto de datos concreto.

**El mecanismo, comprobado.** Las ReLU mueren, la última capa recibe un vector de entrada exactamente nulo y el clasificador solo emite su sesgo. El gradiente del peso de la última capa es entonces exactamente cero, y como GWA protege su norma con un `clamp_min(EPS)`, `gwa/score_mean` sale **0,0 exacto**. Un cero exacto en coma flotante no sale por casualidad, así que sirve de firma: aparece en 124 runs y, en los afectados, ocupa de media el 74% de las épocas.

**La firma es fuerte, pero no infalible.** De los 124 runs con cero exacto, 115 acaban clavados y **9 no**, así que no discrimina sin ambigüedad, como sí parecía con 268 runs.

Qué se hace con estos runs al analizar está **por decidir**, como el resto del análisis. Matiz medido el 2026-08-25: en la variable de velocidad no hay nada que decidir, porque todo run clavado es además un run que nunca cruza el umbral, y la cuenta de runs con velocidad medida sale idéntica con ellos y sin ellos en las 24 celdas. La decisión solo afecta a las otras cinco variables dependientes, donde los clavados sí tienen valor (ver [[3 - Progreso]] §Estado actual).

### 2026-08-05

#### Corregido el estadístico de degeneración de los diagnósticos de sanidad

El diagnóstico que responde "¿esta métrica llega a moverse dentro de un entrenamiento?" estaba **mal**, y la figura del notebook mostraba fielmente el resultado equivocado. Medía `within_std / RMS(within_std entre runs)`: una normalización contra una referencia calculada **entre** entrenamientos, dominada por el de mayor escala. El resultado ordenaba por escala, no por movimiento.

**Cómo se detectó.** Marcaba como degenerada la `val_loss` de los 13 runs de menor escala, entre ellos los seis de MNIST, cuya val loss obviamente se mueve: recorre el rango 0,18 a 0,02, casi un orden de magnitud. La referencia la fijaba `fc_tiny_imagenet_sgd`, con una desviación de 19,3 frente a 0,005 en MNIST.

**Sustituto.** `signal_to_jitter = std(valores) / std(primeras diferencias)` dentro de cada run. Numerador y denominador escalan igual con la métrica, así que el cociente no depende de las unidades ni de la escala. Y trae su propia referencia en vez de un umbral a mano: una trayectoria que sea ruido blanco alrededor de una constante cumple `std(diff) = √2 · std(valores)`, luego su cociente vale 1/√2 ≈ 0,71.

**Qué cambia en las conclusiones.** El orden se invierte en la cabeza y en la cola. `var/normalized` y `noise_scale/simple` figuraban como las más sanas (0,89 y 0,87 del máximo) y son en realidad las que menos se distinguen del temblor (0,81 y 0,82 frente al 0,71 del ruido puro), junto con `stiffness/cos_global` y `mcoh/global`. Y `gwa/value` figuraba como la más muerta (0,003) sin serlo: sus valores oscilan entre ±1,5 en un run de ResNet sobre MNIST y ±4·10⁻⁵ en uno de FC sobre Tiny-ImageNet, seis órdenes de magnitud, que es justo lo que la normalización anterior confundía con inmovilidad.

**Consecuencia sobre la entrada del 2026-08-01.** La corroboración del pilot que allí se cita para GWA no se sostiene tal cual; ver la corrección anotada en esa entrada. El argumento bibliográfico, que es el que decide, no se toca.

#### Sin veredicto binario sobre degeneración

La versión corregida deja de emitir una etiqueta `degenerate` y publica solo la comparación contra la referencia (`below_noise`). El motivo es que 1/√2 es el valor **asintótico**. Una métrica que fuera puro ruido cae a un lado o a otro de la línea aproximadamente la mitad de las veces, de modo que un booleano por run afirma una decisión que el estadístico no sostiene con una sola trayectoria. Lo que se lee es la distribución completa de los 24 runs frente a la línea.

### 2026-07-17

#### Coste de instrumentación: se mantiene la medición completa

Cierra la decisión abierta desde el pilot ([[3 - Progreso]]). Se mantiene la medición tal cual: registro completo de las 8 métricas + baseline al final de cada época, sobre la probe fija de M=256, en toda la rejilla. La prioridad declarada es disponer de datos suficientes: la serie temporal completa por época.

- **Por qué.** El peor caso, ya medido sobre la matriz completa, es **2,048x** el wall-clock de un run sin instrumentar, en `fc × cifar100 × sgd`, dentro de la cota <3-4x fijada. Cifra corregida el 2026-08-08: las lecturas anteriores salían del pilot, anterior al barrido compartido, y sobrestimaban. La conclusión de mantener la medición completa no cambia. Conservar la serie completa preserva la elección de ventanas a posteriori y la línea exploratoria post-meseta anotada como trabajo futuro.
- **Alternativas descartadas.** Bajar la cadencia de medición (pierde resolución de trayectoria y complica el snap exacto de ventanas); submuestrear la probe (M=256 está congelado por comparabilidad cross-celda: tocarlo introduce un confusor); fusionar las 2 batch-sweeps restantes (NGV, gradient disparity) en el sweep compartido (palanca válida de ingeniería, pero no bit-idéntica, cambios ~1e-6 frente a los valores que los tests pinean; queda como optimización futura si el coste apretara).
- **Consecuencia.** La proyección de ~147 GPU-h del registro de abajo sobrestimaba, porque salía del pilot. La matriz completa consumió **121,7 h** de reloj: 97,6 h de entrenamiento y 24,1 h de instrumentación, sumadas sobre los 960 `summary.json`.

#### Presupuestos y umbrales definitivos del pilot (registro formal)

Cierra el "registrar aquí los valores finales con su evidencia" de la decisión del pilot (2026-06-09). Los valores operan en `config.py::DATASET_BUDGET` y en los 24 YAML desde el 2026-06-17; esta entrada añade el registro con su evidencia (`run_pilot.py --report` sobre `reports_pilot/`; el pilot corrió con los presupuestos candidatos doblados: 40/80/120/160 épocas).

- **Valores finales (épocas / umbral de val-acc):** MNIST 20 / 0,97; CIFAR-10 40 / 0,65; CIFAR-100 40 / 0,35; Tiny-ImageNet 40 / 0,20.
- **Evidencia de presupuesto** (época de meseta de val-loss por celda, absoluta): MNIST 6-17, CIFAR-10 3-19, CIFAR-100 3-8, Tiny-ImageNet 1-8. Los presupuestos finales cubren la meseta con margen y conservan el múltiplo de 20 que hace exacto el snap de `windows`. CIFAR-100 baja de 60 a 40 y Tiny-ImageNet de 80 a 40 (recorte del pico de meseta: épocas muertas multiplicadas por ~960 runs); MNIST (20) y CIFAR-10 (40) conservan su candidato.
- **Evidencia de umbral.** CIFAR-10 baja de 0,75 a 0,65: el techo de val-acc de CNN es ~0,72-0,73, con 0,75 la celda quedaba censurada por techo, no por lentitud (verificado sobre la curva suavizada: con 0,65 la CNN a LR centrado cruza en la época 4-5, un 10-12% del presupuesto). Tiny-ImageNet baja de 0,25 a 0,20: el margen de CNN sobre 0,25 era ~0,007 (frágil frente a la varianza entre seeds); con 0,20 sube a ~0,06 y la CNN a LR centrado cruza en la época 3-4 (8-10% del presupuesto). MNIST (0,97) y CIFAR-100 (0,35) se conservan; la censura esperada queda en FC (ya anticipada en [[1 - Diseño]]).
- **Cruce temprano asumido.** Con los umbrales finales, todos los cruces de CNN/ResNet a LR centrado caen al 5-15% del presupuesto (recomputados el 2026-07-17 sobre la curva suavizada desde `trajectory.parquet`; la banda 30-60% preescrita no se alcanza en ningún dataset en el punto central). Se acepta: subir umbrales rozaría el techo de FC (MNIST) o acercaría la censura de CNN (CIFAR-10/100, Tiny), y el rango dinámico de VD1 lo puebla la rejilla completa de LR, no el punto central.
- **Coste proyectado.** Escalando el wall-clock por celda del pilot al presupuesto final × 40 runs/celda: MNIST ~9,5 h; CIFAR-10 ~21,6 h; CIFAR-100 ~21,7 h; Tiny-ImageNet ~93,7 h (~64% del total). Total ~147 GPU-h, desde ~250 con los candidatos. Contexto de ejecución corregido (2026-07-17): hay una única GPU disponible, no un cluster (pese a la asunción de la decisión 2026-06-09), así que ~147 GPU-h ≈ ~6 días de GPU continuos; el troceado por nodos no aplica y la matriz se ejecuta por tandas con la reanudación del launcher.
- **Salvedad Tiny.** Los campos test/gap de `reports_pilot/` para Tiny-ImageNet son los corruptos del bug pre-fix; la calibración usó solo el lado val y el timing (sanos). Desde el 2026-07-17 la referencia corregida (re-run post-fix a 40 épocas) vive dentro de cada run del pilot (`testfix_40ep/`), el `summary.json` corrupto se auto-declara vía la clave `_tiny_test_note`, y `reports_validity/` quedó retirada tras la fusión.
- **Qué NO cierra esta entrada.** El suelo de ajuste del gap (mínimo de `final_train_eval_acc` para los contrastes de VD5/VD6) sigue sin valor fijado, y el 2026-07-17 se decide dejarlo así a propósito hasta el acto de congelación. Aplazarlo no cuesta nada: es un filtro de análisis, no un knob de entrenamiento (`final_train_eval_acc` queda registrado en cada `summary.json`), así que fijarlo o revisarlo después solo toca código de análisis, nunca obliga a re-correr runs.

#### Tabla de signos de H6 verificada contra los papers

Verificación de la tabla contra los PDFs del vault (GSNR/Liu, Coherent Gradients/Chatterjee, Making Coherence/Chatterjee & Zielinski, GWA/Hölzl, GNS/McCandlish, TSE/Ru). La tabla corregida vive en [[Datos experimentales]] §5.3; cambios y evidencia clave:

- **m-coherence vs VD1: sigue −, pero la base fuerte cambia de paper.** Chatterjee & Zielinski no afirma velocidad (su α es eficiencia por paso, definicional; las menciones de velocidad son citas a terceros). El claim explícito es de Chatterjee 2020 (CGH): "we expect that greater the agreement in per-example gradients, the faster loss should decrease" (§2.2) y "as noise increases, the time taken to reach a given level of accuracy (i.e., realized learning rate) increases" (§2.3). Matiz: medido sobre train accuracy; la extensión a val es razonada.
- **GSNR vs VD4: de fuerte a extrapolada.** El paper solo afirma el gap ("larger GSNR during training process leads to better generalization performance", vía OSGR, ec. 22; el gap es en loss, la misma cantidad que VD5); no hay claim de test accuracy. Su predicción fuerte es − vs el gap. Matices: la teoría se deriva en fase temprana (favorece la ventana del TFG) pero con full-batch GD, no SGD.
- **GWA vs gap: de fuerte a direccional cualitativa.** El claim cuantitativo del paper es vs test accuracy (Fig. 3: Pearson 0,99 solo ConvNeXt/CIFAR-10; 0,92 cross-arquitectura; medido sobre max de toda la trayectoria, no ventana temprana; su criterio de early stopping descarta el primer 10% como warm-up). El gap operativo (test loss − train loss) nunca se mide. **Corrección 2026-08-05:** el pilot no corrobora la rebaja, como esta entrada decía. Ese apoyo salía del estadístico de degeneración roto (ver la entrada del 2026-08-05): con la medida libre de escala, GWA en ventana temprana marca 1,08 en su valor titular y 1,37 en la media de scores, por encima del 0,71 del ruido puro, y solo su curtosis (0,91) queda pegada a esa línea. Es débil, no plana, y de hecho `var/normalized` y `mcoh/global` se mueven **menos** que GWA sobre la trayectoria completa. La rebaja se mantiene, pero se apoya solo en el argumento bibliográfico, que es el que decide.
- **GNS vs VD1: + confirmado con condiciones.** Base formal ec. 2.7/D.1 (δS = 1 + 𝓑/B a B fijo). Condiciones: régimen B ≲ 𝓑, LR bien ajustado, y el GNS medido depende del LR ("it is not consistent at different learning rates", Ap. A.1), relevante porque el TFG barre LR a B fijo. Su silencio de gap es correcto (caveat 6 del paper).
- **m-coherence vs gap: − confirmado con la salvedad del propio paper** ("this connection is complicated": con 100% label noise la coherencia también sube; lo informativo es la coherencia temprana). Trayectoria esperada para los diagnósticos: no monótona en general ("broad parabolic trajectory"); a granularidad de época con labels reales, decreciente hacia ~1 tras un pico muy temprano.
- **TSE: definiciones y caveats confirmados.** Corrección literal de la cita de §4.2: termina "outside the scope of this paper", no "of our work" (corregido en el plan; la nota del paper ya la tenía bien). γ=0,999 es el default recomendado de §4.1, no la constante definicional (§2 introduce TSE-EMA con γ=0,9). Aviso de archivo: el PDF local es la versión NeurIPS sin apéndices; los apéndices C.1-C.2 (overconfidence, base del caveat de VD2/VD3) solo están en el arXiv v2, conviene archivar esa versión en `Papers/PDFs/`.

La lectura humana de estos papers sigue pendiente y es valiosa para el estado del arte.

### 2026-06-14

#### Gap de generalización: tercer constructo de variable dependiente

Confirmado por el tutor el 2026-06-14 e implementado el mismo día en `src/data.py` + `src/train.py`. Añade el constructo *generalización* a las variables dependientes de [[1 - Diseño]], junto a velocidad y rendimiento final.

- **Qué se mide.** Cinco claves nuevas en `summary.json`: `final_gap_loss = final_test_loss − final_train_eval_loss` (primaria; positivo = sobreajuste), `final_gap_acc = final_train_eval_acc − final_test_acc` (robustez, mismo sentido), sus términos `final_train_eval_loss`/`final_train_eval_acc`, y `final_test_loss` (la evaluación final ya recorría el test; solo se le añadió la loss).
- **Cómo.** Una pasada `evaluate()` extra al final del run, en modo eval y con los mismos pesos, sobre un subconjunto fijo y estratificado por clase del train recortado, de tamaño igual al test y muestreado con `SPLIT_SEED` (idéntico en todos los runs, independiente de la semilla del run; `build_train_eval_loader` en `data.py`). Coste: segundos por run, una vez. Solo toca la evaluación final, como el protocolo ya implementado.
- **Confound conocido** (el presupuesto de épocas es fijo): a presupuesto fijo, un gap pequeño puede significar que el modelo generaliza bien o que no ha aprendido lo suficiente, y las dos cosas se confunden. Quien analice el gap tiene que separarlas, por ejemplo excluyendo los runs que no aprenden el train o controlando por `final_train_eval_loss`. Cómo hacerlo está por decidir.
- **Respaldo.** La cantidad (riesgo de test − riesgo empírico) y el rol (gap como variable dependiente de un estudio correlacional) son de la literatura: Jiang et al. 2020 (arXiv:1912.02178), Dziugaite et al. 2020 (arXiv:2010.11924); incluso la estimación del término de train por submuestreo tiene precedente en Jiang §3. Lo propio del TFG son los dos controles.
- **Qué queda.** El pilot calibra el suelo de ajuste (distribución de `final_train_eval_acc` por celda), sin impacto en presupuestos ni umbrales.

### 2026-06-12

#### Protocolo de evaluación: train optimiza, val monitoriza, test certifica

Confirmado por el tutor (respuesta rápida del 2026-06-12: particiones típicas de cada dataset, sin validación cruzada, semillas múltiples sobre train, val para evaluar convergencia, test para el resultado final) e implementado el mismo día en `src/data.py` + `src/train.py`.

- **Split.** Test oficial intacto; val extraído del train, estratificado por clase, con semilla de split fija e independiente de la semilla del run (`SPLIT_SEED` en `data.py`). Todos los runs ven la misma partición, y la única aleatoriedad entre seeds sigue siendo la inicialización y el orden de batches, el objeto de estudio. Tamaños por convención de cada dataset, no por regla uniforme (decisión sobre la respuesta del tutor, que pedía "las particiones típicas"): MNIST 50k/10k/10k (la convención clásica, la que el tutor recordaba), CIFAR-10/100 45k/5k/10k (He et al. 2015, ResNet) y Tiny-ImageNet 90k/10k/10k. En Tiny-ImageNet su `val/` público hace de test (las etiquetas del test oficial no son públicas) y el val de monitorización replica ese tamaño.
- **Roles únicos, sin cruces.** El modelo entrena con el train recortado; la probe de métricas se muestrea de ese mismo train; la monitorización por época y todos los indicadores de eficiencia leen val; el test se evalúa exactamente una vez al final, produciendo `final_test_acc` y `final_test_f1_macro` (vía matriz de confusión en torch, sin dependencias nuevas). En datasets balanceados F1 ≈ acc: se reporta como verificación, no como hallazgo.
- **Lecturas estables de la curva.** VD1 (épocas-hasta-umbral) y VD3 (mejor loss) se leen sobre la curva de val suavizada con mediana móvil centrada de 3 épocas (`median3` en `train.py`; la ventana encoge en los bordes). Motivo: los extremos de una serie ruidosa están sesgados en proporción a su volatilidad, la volatilidad depende del LR y las métricas de ruido de gradiente plausiblemente la predicen. Sin suavizado, el propio estimador crearía un confusor entre predictor y VD. VD2 (AUC) integra la curva cruda: integrar ya amortigua el ruido. La curva cruda completa queda en `trajectory.parquet`, todo recomputable post-hoc.
- **Por qué.** Tres problemas del setup de 2 vías: el sesgo de extremo (arriba), la circularidad de calibración (umbrales calibrados sobre curvas de test del pilot y `epochs_to_threshold` medido después sobre ese mismo test) y la objeción previsible en la defensa ("evaluasteis test cada época"), aunque ninguna decisión de entrenamiento mirase el test (presupuesto fijo, sin early stopping, rejilla preespecificada).
- **Notas de honestidad para la memoria.** (1) El "test" de Tiny-ImageNet es su val público, práctica estándar, se declara. (2) F1-macro ≈ accuracy en balanceados: verificación, no hallazgo. (3) El split es fijo y compartido por todos los runs, decisión deliberada: lo estudiado es la variación por seed/LR, no la varianza del estimador (Bouthillier et al. 2021 recomiendan aleatorizar splits cuando se comparan métodos; no es el caso), y se declara.
- **Qué queda.** Relanzar el pilot con el split nuevo: los umbrales candidatos se calibraron pensando en test-acc y pasan a chequearse sobre la curva de val suavizada con un train menor.

### 2026-06-10

#### Timing por run: dos relojes, no uno

Cada run cronometra por separado el entrenamiento y la instrumentación (`src/train.py`): `summary.json` gana `total_seconds`, `metric_seconds` (acumulado alrededor de cada bloque de medición, con `synchronize` en cuda/mps para que los kernels asíncronos se atribuyan al reloj correcto) y `train_seconds` = total − metric.

- **Por qué dos relojes y no uno.** El overhead de la instrumentación (per-sample grads vía vmap sobre la matriz M×P) escala con el tamaño del modelo y con la densidad de probes: un único wall-clock sesgaría las comparaciones de tiempo entre celdas a favor de los modelos pequeños. Evidencia local: en un run corto fc/MNIST el overhead fue ~43% del wall-clock total (incluye el warmup de compilación de vmap del primer probe).
- **Timestamps por fila.** Toda fila de `trajectory.parquet` lleva `elapsed_seconds` y `metric_seconds` acumulados; eso habilita `seconds_to_threshold` junto a `epochs_to_threshold`, la velocidad en wall-clock, más honesta al comparar SGD↔Adam (coste por paso distinto). Es cruda (incluye instrumentación hasta ese punto); la corrección post-hoc es restar la columna acumulada, sin relanzar nada.
- **Convenciones.** `evaluate()` cuenta como entrenamiento (práctica estándar de cualquier run); solo `measure` + baseline TSE van al reloj de overhead. El `synchronize` se hace solo alrededor de los probes (infrecuentes y ya caros), nunca por paso de optimización. El wall-clock es señal de presupuesto y anomalías, no métrica científica, así que no se correlaciona como si fuera limpio.
- **Qué cierra.** El criterio "overhead <3-4x" ([[3 - Progreso]]) y las "GPU-h reales por run" que el pilot debía validar (decisión del pilot, 2026-06-09) se leen ahora directamente de cada `summary.json`.

### 2026-06-09

#### Pilot de calibración: un run por celda, presupuesto doblado

Concreta el "se calibran en el pilot" de presupuestos y umbrales (decisión "Matriz de runs congelada") en un protocolo ejecutable: `src/run_pilot.py`, módulo aparte del launcher de producción.

- **Qué corre.** Un run por celda (24 en total), LR en el centro de la rejilla (SGD 1e-2, Adam 1e-3, los defaults canónicos de cada optimizador), seed 0 y **el doble del presupuesto candidato** (40/80/120/160 épocas). La asimetría que lo justifica: recortar una curva generosa a posteriori es gratis, estirar una corta es relanzar. El presupuesto define `progress_frac`, las ventanas y el AUC, así que debe quedar bien fijado *antes* de los ~960 runs. El coste del pilot (24 runs a 2×) es ~5% del de la matriz.
- **Qué responde.** (1) *Presupuesto*: dónde se aplana la test loss de los runs bien ajustados → presupuesto final = meseta + margen, redondeado a múltiplo de 20 (conserva el ajuste exacto de `windows`). (2) *Umbral*: debe cruzarse hacia el 30-60% del presupuesto por CNN/ResNet a LR centrado; cruzado en la época 1 no discrimina velocidad, cruzado por casi nadie censura media matriz. La decisión la toma el investigador y no el script, que solo imprime la evidencia por celda.
- **Aislado de `reports/` a propósito.** Los pilots escriben en `reports_pilot/`: `run_matrix` da por hecho un punto de la rejilla si existe `reports/<run_name>/summary.json`, y un pilot con LR de rejilla y seed 0 dentro de `reports/` se contabilizaría después como run de producción, entrenado con el presupuesto viejo.
- **Módulo aparte y no flag `--pilot`**, para no llenar de condicionales (out_dir, épocas, ejes de barrido) el launcher de producción justo antes de usarlo en serio. `run_pilot` reutiliza de `run_matrix` el naming y las celdas (identidad espejada por construcción) y puede retirarse tras la calibración.
- **También cierra** las validaciones que el diseño dejaba para el pilot: overhead real de las métricas caras (m-coherence, gradient confusion), redundancia GNS≈B·NGV, centrado de la rejilla de LR y GPU-h reales por run.
- **Tras el pilot:** fijar presupuesto/umbral definitivos editando los 24 YAML de celda *y* `config.py::DATASET_BUDGET` (la fuente de celdas regeneradas), y registrar aquí los valores finales con su evidencia.

#### Justificación valor a valor de los hiperparámetros congelados

La matriz congelada fija números concretos (`src/config.py::FIXED_KNOBS`, `DATASET_BUDGET`; escritos explícitamente en cada YAML de `experiments/`). Aquí queda el porqué de cada uno; los que ya tienen decisión propia (rejilla de LR, seeds, métricas) solo se referencian.

- **`batch_size = 128`.** Tres razones. (1) *Comparabilidad de las métricas*: varias métricas dependen del batch de entrenamiento. La gradient disparity no es comparable entre runs con batch distinto (su varianza decrece como $1/m$; aviso documentado en la nota de Forouzesh & Thiran) y el GNS se lee relativo a B ($\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$ por TLC). Barrer B mezclaría el predictor con el hiperparámetro; fijarlo lo neutraliza. (2) *Es el valor del corpus*: Sankararaman et al. (gradient confusion) usan exactamente mini-batches de 128 en el mismo trío MNIST/CIFAR-10/CIFAR-100. (3) *Práctico*: cabe en memoria con las cuatro combinaciones dataset×modelo y da longitudes de época razonables (391 pasos/época en CIFAR, 469 en MNIST, 782 en Tiny-ImageNet).
- **`momentum = 0.9` (SGD).** Es el default canónico de la literatura (lo usan los baselines del corpus que entrenan con momentum, p. ej. Chatterjee & Zielinski). No es pregunta de la tesis, así que no se barre: cada eje extra multiplica la matriz. Dos consecuencias deliberadas: (a) las métricas leen el gradiente bruto ∇L, nunca el update con momentum, así que el valor no entra en las métricas, solo da forma a la trayectoria (mismo argumento que el weight decay); (b) la rejilla de LR de SGD está centrada *asumiendo* 0.9 (el paso efectivo estacionario se amplifica ~1/(1−β) = 10×): cambiar el momentum obligaría a recentrar la rejilla.
- **`weight_decay = 0`.** Ya justificado en [[1 - Diseño]] §Matriz de runs: las métricas leen ∇L de la pérdida, así que el decay no entra en su valor; se fija a 0 solo para no introducir un eje de trayectoria extra. Coincide además con el setup de varios papers del corpus (Sankararaman y Chatterjee & Zielinski entrenan sin weight decay precisamente para aislar la dinámica).
- **`probe_size = 256` (M).** Equilibrio entre memoria y estadística. *Memoria*: la matriz de gradientes per-sample pesa ~M×P×4 bytes; con M=256 los modelos FC/CNN caben holgados y ResNet-18 se cubre con la decisión last-layer-only (con sus ~11M de parámetros completos serían ~11 GB). *Estadística*: M=256 da 256·255/2 ≈ 32.6k pares para las métricas del Gram per-ejemplo (confusion, stiffness, m-coherence), varianza muestral sobrada para un estimador por medición. *Comparabilidad*: M se congela porque cambia la varianza de todos los estimadores; con probes de distinto tamaño entre runs, las comparaciones entre modelos dejarían de ser válidas (es la razón del aviso de memoria en `train.py` en lugar de un límite silencioso). El probe además es fijo durante el run (mismas 256 muestras siempre): la serie temporal mide la evolución del modelo, no el remuestreo.
- **`windows = [0.05, 0.10, 0.25, 0.50, 1.0]`.** Los cuatro primeros son el barrido de fracción temprana del diseño (§Ventana temporal: el barrido es en sí un resultado reportable, H4), espaciados ~geométricamente como la rejilla de LR; el 1.0 ancla el extremo "entrenamiento completo" como referencia de saturación. Los valores se eligieron junto a los presupuestos para que cada fracción caiga exacta en frontera de época (ver siguiente punto).
- **Presupuestos de épocas 20/40/60/80 (MNIST/CIFAR-10/CIFAR-100/Tiny-ImageNet).** Escalan con la dificultad del dataset: sin augmentation las curvas se aplanan antes que en los schedules SOTA, y el presupuesto se dimensiona para que los runs bien ajustados lleguen a meseta sin gastar épocas muertas en ~960 runs. Todos múltiplos de 20 *a propósito*: cada fracción de `windows` cae exacta en frontera de época (0.05×20=1, 0.25×60=15…), así que el snap a posteriori no introduce desfase. Son puntos de partida: el pilot puede moverlos (los YAML editados a mano sobreviven a `--init`).
- **Umbrales de accuracy 0.97/0.75/0.35/0.25.** Calibrados *por dataset* (no por modelo) para ser alcanzables pero no triviales sin augmentation. Quedan por debajo del techo razonable de las arquitecturas competentes bien ajustadas (FC llega a ~0.98 en MNIST; CNN/ResNet-18 a ~0.75-0.85 en CIFAR-10; ResNet-18 a ~0.4-0.5 en CIFAR-100 y ~0.3-0.4 en Tiny-ImageNet), pero son lo bastante altos para que el número de épocas hasta cruzarlos tenga rango dinámico: un umbral que todo run cruza en la época 1 no discrimina velocidad. El umbral único por dataset hace VD1 comparable dentro de cada celda y entre celdas del mismo dataset; el precio asumido es que las arquitecturas débiles quedan censuradas (FC en CIFAR-100/Tiny-ImageNet, y previsiblemente buena parte de FC en CIFAR-10), y esas celdas se analizan con las VD secundarias. Igual que los presupuestos, se recalibran tras el pilot si quedan mal centrados.
- **Seeds `{0,1,2,3,4}` y rejilla de LR.** Justificados en sus decisiones propias (abajo en esta misma fecha): 5 seeds compartidas para comparación pareada SGD↔Adam y 8 LR por optimizador priorizando dispersión del predictor.
- **Métricas = todas, sin knob.** Ya no hay valor que justificar: se eliminó el knob `active_metrics` (de `Config`, `FIXED_KNOBS`, los 24 YAML y el runner), así que el código no *puede* producir runs con subconjuntos de métricas. Se computa el registro completo en toda la rejilla y la lista reportada se poda después con prueba.

#### Rejilla de LR uniforme por optimizador

Al implementar el lanzador (`src/config.py::LR_GRID`, `src/run_matrix.py`) la rejilla de SGD quedó distinta de la congelada en la matriz: una sola rejilla log-espaciada en medias décadas por optimizador, **idéntica para FC, CNN y ResNet-18** (SGD `{3e-4 … 1.0}`), en lugar de `{0.005 … 0.5}` para CNN/ResNet-18 con FC desplazada una década abajo. Adam no cambia. Se adopta la versión implementada como decisión y se actualiza [[1 - Diseño]] §Matriz de runs.

- **Por qué uniforme y no por modelo.** El lanzador deriva la identidad de cada run (y su directorio de salida) solo de (modelo, dataset, optimizador, lr, seed); una rejilla por modelo añadiría lógica condicional y rompería la simetría de la rejilla sin cambiar lo que se mide. En su lugar, un rango ancho (3,5 décadas, vs. 2 de la spec original) cubre a la vez los óptimos bajos de FC y los altos de CNN/ResNet-18.
- **El coste es asumible por diseño.** En cada celda sobran puntos en un extremo u otro (divergen o no alcanzan umbral), pero esos runs censurados son justo los que pueblan el eje de eficiencia (VD1), y con 40 runs/celda hay margen sobre el suelo n ≥ 30.
- **Simetría SGD↔Adam.** Misma forma de rejilla, desplazada una década (el paso efectivo de Adam va preescalado por 1/√v): la comparación pareada entre optimizadores (H5) no confunde forma de rejilla con efecto del optimizador.
- **Riesgo y plan de contingencia.** Medias décadas dan menos resolución alrededor del óptimo que la rejilla original; si tras el pilot el óptimo de alguna celda queda descentrado o entre puntos, se recalibra el centro (ya previsto en la spec congelada).

#### Matriz de runs congelada

Resuelve el budget de cómputo y cierra la variante de ResNet. Spec ejecutable en [[1 - Diseño]] §Matriz de runs.

- **Rejilla completa, sin recortar.** Se ejecuta la matriz entera: {MNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet} × {FC, CNN simple, ResNet-18} × {SGD, Adam} = 24 celdas. Habilitado por disponer de GPU/cluster dedicado: el presupuesto de cómputo (riesgo #1 de [[1 - Diseño]]) deja de ser limitante y se descarta el subset "~18-24 runs".
- **Profundidad por celda.** 8 LR × 5 seeds = 40 runs/celda → ~960 runs, por encima del suelo n ≥ 30. A conteo fijo se prioriza tener más LR distintos (dan la dispersión del predictor) sobre más seeds (que dan intervalos de confianza). Mismas seeds {0,1,2,3,4} en todas las celdas para comparación pareada entre SGD y Adam (sostiene H5).
- **Tiny-ImageNet entra.** Como cabe en cómputo, se confirma el condicional anterior: cuarto dataset, sube el techo de dificultad sobre CIFAR-100. Actualizado [[1 - Diseño]] §Setup de entrenamiento.
- **ResNet-18 fija la variante.** La rejilla se congela con ResNet-18 (adaptada a imágenes pequeñas, ya en código). Cierra la pendiente "Variante de ResNet".
- **Todas las métricas implementadas en toda la rejilla.** Computar el conjunto completo de antemano no contradice "no añadir métricas a posteriori": la lista *reportada* se decide luego por poda con prueba (pendiente "Lista definitiva de métricas" + decisión de poda de abajo). (Corregido: esta entrada decía que el cluster hacía viables las caras y que en ResNet-18 las per-sample iban last-layer-only. Ninguna de las dos se sostiene. Hay una sola GPU desde el 2026-07-17, y lo que hace viables las caras es el troceado en filas del barrido per-sample, que recorre todos los parámetros; GWA es la única métrica last-layer.)

#### Decisiones de ejecución

Refinan el diseño cerrado de [[1 - Diseño]].

- **Entrenar hasta convergencia, medir durante todo el trayecto.** Cada run se entrena hasta una convergencia definida de antemano (umbral ε sobre la pérdida, el mismo que ancla "épocas-hasta-umbral") y las métricas se registran a lo largo de *todo* el entrenamiento, no solo en la fracción $f$. Da la serie temporal completa y permite elegir el $f$ predictivo a posteriori.
- **Doble eje temporal en el análisis: época y % de convergencia.** Correlacionar por época absoluta y también normalizando por el porcentaje de convergencia hacia ε. "Época 10" no significa lo mismo entre problemas de distinta dificultad; el eje "fracción del camino a ε" hace comparables las curvas (mitiga el confusor de dificultad de [[1 - Diseño]] §Confusores).
- **Tiny-ImageNet: lo incorporamos si es posible (de momento, sí).** La intención por defecto es sumarlo a los datasets, para que el estudio sea *más completo* y suba el techo de dificultad por encima de CIFAR-100. Solo lo dejaremos fuera si no cabe en el presupuesto de cómputo (sobre todo por las métricas caras). Si finalmente entra, actualizar [[1 - Diseño]] §Setup de entrenamiento.
- **Poda de métricas redundantes, con prueba.** Si dos métricas se comportan casi igual y miden casi lo mismo (pares colineales: GNS≈B·NGV, GSNR primo de NGV, clúster del Gram per-ejemplo), descartar una para aligerar análisis y redacción, pero solo demostrándolo (correlación alta, comportamiento solapado). Permite el análisis a nivel de familias.
- **Varias runs, varias seeds.** Múltiples runs variando seed (y demás ejes) para tener réplicas con las que hacer tests de hipótesis sobre las correlaciones, no leer un único número. Conecta con el objetivo n ≥ 30 por celda (riesgo #1 de [[1 - Diseño]]).
- **Baseline = loss (confirmado).** El baseline es la curva de loss (TSE + val-loss tempranas); toda métrica de gradiente se juzga por su valor incremental sobre ella (ΔR², no ρ crudo). Detalle en [[1 - Diseño]] §Baselines y §Hipótesis a contrastar (H2).

### 2026-05-14

#### Setup base: datasets y arquitecturas

Fija el núcleo experimental del estudio. Detalle en [[1 - Diseño]] §Convergencia de la literatura.

- **Setup mínimo por convergencia de la literatura.** Datasets MNIST + CIFAR-10 + CIFAR-100; familias de arquitectura FC + CNN simple + ResNet; optimizadores SGD y Adam (mínimo). Es el núcleo común de los 15 papers con setup.
- **Variante de ResNet, abierta.** El resto es firme; la variante concreta quedó pendiente (cerrada el 2026-06-09 con ResNet-18).
