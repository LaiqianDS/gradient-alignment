# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

Qué entra en una entrada: la decisión, su porqué en corto, la evidencia con su número, y las trampas. Nada más. El registro se poda mientras se trabaja, y las decisiones y sus fechas no se podan nunca; la regla completa está en [[3 - Progreso]] §Plan por objetivos.

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] §Plan por objetivos.

**El método de análisis, entero (abierto el 2026-08-25).** Los 960 entrenamientos están hechos y sus datos versionados en `reports/`. No hay método definido para analizarlos: el plan anterior se retiró (ver el log del 2026-08-25). Lo que queda por decidir es, hipótesis por hipótesis, con qué cuenta concreta se responde. Bloquea el capítulo de resultados.

Tamaño del problema de multiplicidad, para tenerlo presente al decidir: `metrics_at_window.parquet` registra **27 columnas de predictor**, porque cada métrica escribe varias variantes (*stiffness* seis, *gradient confusion* cinco, GSNR tres, TSE cuatro, el resto una o dos). Una correlación consume los 40 entrenamientos de una celda, así que cada elección de predictor, ventana e indicador de eficiencia produce 24 coeficientes, uno por celda. Con 27 predictores, 4 ventanas tempranas y los indicadores de eficiencia en juego, el factorial completo son miles de contrastes sobre las mismas 24 celdas. El método tiene que decidir qué subconjunto se contrasta y cómo se corrige la multiplicidad, no barrerlo entero. Las columnas titulares están fijadas desde el 2026-09-03 (tabla de `metodologia.tex` §Variables), así que el punto de partida son 8 métricas por 4 ventanas por 6 VD.

**Forma propuesta, sin decidir (2026-08-31).** Cinco elecciones que se toman una sola vez y sirven a los seis objetivos, porque las fases B a G son una máquina y seis consultas ([[3 - Progreso]] §Plan por objetivos).

- **Unidad.** Dentro de una celda, un run es una observación: predictor de `metrics_at_window.parquet` en la ventana f, variable dependiente de `summary.json`. n = 40 menos lo que diga el mapa. No hay alternativa razonable, es lo que impone el diseño.
- **Estadístico intra-celda: tau de Kendall.** El motivo no es el habitual. Tau se define contando pares concordantes, y su generalización censurada, la D de Somers, es el **mismo** estadístico sobre los pares comparables, así que VD1 censurada y las otras cinco usan uno solo. Spearman no tiene versión censurada y obligaría a dos estadísticos y a explicar por qué. Precio, cosmético: tau sale más pequeño que Spearman sobre los mismos datos y no es comparable con el de otros artículos.
- **Agregación de las 24 celdas.** Enseñar las 24 siempre y contrastarlas con Wilcoxon de rangos con signo contra cero, que encaja con la palabra "consistente" de H1. **Limitación que hay que declarar y no esconder:** las 24 celdas no son réplicas independientes, comparten 4 datasets y 5 semillas, y agregar antes por dataset dejaría 4 unidades, demasiado pocas.
- **Multiplicidad.** Una combinación primaria declarada por objetivo *antes* de calcular, más Benjamini-Hochberg dentro de la familia de cada objetivo; el resto, etiquetado como exploratorio. Con las columnas titulares son 8 métricas x 4 ventanas tempranas x 6 VD = 192 combinaciones, cada una con sus 24 celdas. No es preinscripción y no se puede fingir que lo sea, pero sí es una regla fijada antes de mirar y con fecha.
- **OE2.** Comparación pareada por celda del coeficiente de la métrica contra el del baseline, contrastada sobre las 24 con un test de signos. La correlación parcial de rangos, como comprobación secundaria.

**Riesgo que invalidaría la segunda elección:** todo esto supone relación monótona. Con un barrido de LR de 3,5 décadas una relación en U es posible, y cualquier estadístico de rangos la leería como ausencia de señal. Antes de fijar el estadístico hay que mirar nubes de puntos, y eso no se ha hecho.

El 2026-08-27 se retiró de este log la tanda de decisiones del 2026-08-26, la que abría la primera pasada de la fase A, junto con el código y el texto que produjo. Lo único que sobrevive de aquella pasada es la **convención de registro** del censurado, que `metodologia.tex` §Variables declara: un valor censurado se anota como ausente y nunca como el presupuesto. La población de análisis y el tratamiento del censurado en el contraste **no** están escritos en ninguna parte, y son trabajo de la fase B.

## Tomadas (log)

### 2026-09-03

#### El mapa de solape baja a la variable principal, y el volumen de cruces sale a figura aparte

**Qué se decidió.** `solape-mapa` pasa de tres paneles a uno, solo VD1. VD2 y VD3 se cuentan en prosa. Se añaden dos figuras: `solape-cruces`, los cruces ya ocurridos al cerrar cada ventana sumados sobre las 24 celdas, y `solape-bandas`, esa misma cuenta contra la parte del presupuesto consumida, en dos paneles apilados, por conjunto de datos y por arquitectura. `solape-celda` gana un panel con los cruces acumulados.

**Por qué.** El panel del AUC enseñaba supervivencia en las 24 celdas y el párrafo siguiente tenía que retirarla, porque el área supera toda ventana por construcción del trapecio. Un tercio de la figura sostenía un resultado que el texto desmiente. Lo que faltaba en cambio era el volumen: ni la celda de ejemplo ni el mapa dicen cuántos cruces hay detrás de cada ventana, y esa es la magnitud del problema.

- **Trampa: las dos figuras nuevas cuentan entrenamientos y la regla decide en pares.** Sirven para dimensionar, no para decidir. Quien decide sigue siendo el mapa, y la memoria lo dice en el propio párrafo.
- **Trampa: los dos cortes de `solape-bandas` heredan el confusor del umbral.** MNIST cruza tarde porque se le pide 0,975 y la red plana pronto porque se le pide el umbral más bajo, así que ninguno separa la dificultad de la exigencia. Declarado en la memoria.
- **Trampa: las bandas son cuartiles sobre seis u ocho celdas.** Con grupos más pequeños el recorrido se colapsaría sobre la mediana sin avisar.

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
- **Una figura y no dos.** `val-test`, dos paneles, *accuracy* y *loss*, un punto por run coloreado por conjunto. El mapa por celda no se dibuja porque las 24 tau tienen una sola estructura, la de las celdas sin recorrido, y eso lo dice una frase.

#### El punto 4 mide el solape en pares, sobre las tres variables de velocidad, y fija la mitad como regla

**Qué se decidió.** Una ventana se cierra en la *epoch* en la que se lee el predictor y un suceso en esa misma *epoch* cuenta como ya ocurrido. Suceso de VD1, el cruce; de VD3, la *epoch* del mínimo del *loss* suavizado; VD2 no tiene suceso y se mide la parte del área que la ventana aún no fija. Lo que sigue por delante se cuenta en pares comparables, con la fórmula de la censura del 2026-08-31. **Regla:** una ventana sirve para una variable en una celda cuando al menos la mitad sigue por delante, `efficiency.py::AHEAD_FLOOR`. Dos pasadas. Vive en `efficiency.py::window_overlap`, `overlap_summary` y `vd1_consumed_pooled`, con la ventana leída de `metrics_at_window.parquet`. Método, evidencia y lecturas en `resultados.tex` §Solape entre la ventana y el desenlace.

**Por qué la mitad, y cuándo se fijó.** Tolerancia cero vacía el mapa por un solo run madrugador, y sin regla se incumple lo que `metodologia.tex` §Riesgos promete. La regla se fijó después de conocer las cifras agregadas del 2026-09-01 y antes del desglose por celda, así que no es ciega y la memoria lo dice.

**Evidencia, sobre los 960.** Celdas que conservan al menos la mitad, por ventana: VD1 22, 12, 0 y 0; VD3 23, 17, 9 y 2; VD2 24, 24, 24 y 13. Solo con los que aprendieron, VD1 y VD2 no cambian y VD3 da 23, 16, 9 y 3. Consecuencia: para la velocidad, OE4 queda acotado a las ventanas del 5 y el 10 %.

- **Trampa: la cuenta es un suelo.** Un run al 95 % del umbral cuenta como por delante. El solape real es mayor.
- **Trampa: VD2 al 50 % cae sobre la línea por construcción**, 0,51 frente al 0,50 de una curva plana, y se declara «descrita a medias».
- **Trampa: VD3 mide el sobreajuste donde el mínimo llega pronto.** En CIFAR-10 con ResNet-18 y Adam, 37 de 40 runs tocan fondo en la *epoch* 10 o antes (corregido el 2026-09-03; aquí decía «los 40 antes de la 10»).
- **Trampa: el suavizado mira una *epoch* adelante.** La mediana centrada usa la siguiente, luego VD1 lleva una *epoch* de anticipación dentro. Manda la coherencia con VD1 tal como está definida.
- **Trampa: las ventanas de MNIST son las *epochs* 1, 2, 5 y 10 y las demás 2, 4, 10 y 20.** Comparar el solape entre conjuntos mezcla tiempo absoluto y relativo; dentro de una celda es constante.
- **Lo que no decide**, anotado como deuda de §Protocolo de análisis en [[3 - Progreso]]: qué hace la fase B con los runs ya cruzados dentro de una ventana que sirve.
- **Cuatro figuras, por decisión de Lai:** `solape-celda`, `solape-cruces`, `solape-bandas` y `solape-mapa`. La forma que tomaron está en la entrada del 2026-09-03.

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

**Evidencia, sobre los 960.** Los cruces pasan de 494 a 571 y las celdas con VD1 de 18 a 24, con mínimo 15 por celda. Los cruces ya ocurridos al medir la ventana del 5 % bajan del 30 al 11 %, y los del 10 % del 56 al 35 %.

- **Trampa: `epochs_to_threshold` de los 960 `summary.json` queda obsoleto**, calculado con los umbrales viejos. VD1 se recalcula siempre en la capa de análisis desde `trajectory.parquet`; `reports/` no se toca.
- **Trampa: la calibración se hace sobre los datos que se analizan.** τ se eligió mirando la distribución de resultados y nunca la relación entre una métrica y VD1, así que no infla ninguna correlación, pero sí escoge la versión de la variable con más varianza. Escrito en `metodologia.tex` §Calibración.
- **No se cumplió el criterio de calibración preescrito del pilot**, que pedía cruces hacia el 30-60 % del presupuesto con el learning rate central; con esta tabla las medianas van del 8 al 25 %. Lo tiene que decir §Calibración del pilot de Resultados cuando se escriba.

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
- **Decidido por Lai: la tabla del estado del arte no se toca y el coste va en una tabla nueva**, `tab:coste-metricas` en `implementacion.tex` §Cálculo de las métricas, con un solo propósito, lo que cuesta medir y lo que ahorra el barrido compartido. La memoria se queda fuera de esa tabla y sigue contada en prosa, porque responde a otra pregunta.
- El micro-benchmark sigue disponible el día que se quiera el eje continuo: minutos de cómputo, fuera de la matriz y sin tocar `reports/`.

**Resultado de la derivación**, con cada término trazado a la línea que lo produce. Las derivadas no separan a ninguna métrica, porque las ocho recorren el *batch* de medición una vez. Lo que separa es la aritmética posterior, en dos niveles: `M·P` las de momentos y `M²·P` las de pares. En el margen, con el barrido ya hecho, **seis de las ocho no derivan nada**: `P` la coherencia y la escala de ruido, `P·log P` el GSNR, `M` la GWA, `M²` la *stiffness* y `M²·log M` la confusión. Solo la varianza normalizada y la disparidad siguen pagando sus diez y cinco pasadas por submuestra. Medir el registro completo cuesta unas tres pasadas sobre el *batch*, no nueve.

- **Dos matices que cambian la lectura.** La GWA solo lee la última capa, pero deriva la red entera para cada ejemplo y descarta el resto, así que su ventaja está en la aritmética; y esa capa no siempre es pequeña, en la CNN sobre Tiny-ImageNet son 102.400 de 116.936 pesos. El término cuadrático domina sobre las derivadas solo en la red *fully connected*, donde una pasada hacia atrás cuesta del orden de `P`; en las convolucionales la comparación puede invertirse.
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

Qué se hace con estos runs al analizar está **por decidir**, como el resto del análisis. Matiz medido el 2026-08-25: en la variable de velocidad no hay nada que decidir, porque todo run clavado es además un run que nunca cruza el umbral, y la cuenta de runs con velocidad medida sale idéntica con ellos y sin ellos. La decisión solo afecta a las otras cinco variables dependientes, donde los clavados sí tienen valor.

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
- **Qué NO cierra esta entrada.** El suelo de ajuste del gap (mínimo de `final_train_eval_acc` para los contrastes de VD5/VD6) sigue sin valor fijado, y el 2026-07-17 se decide dejarlo así a propósito hasta el acto de congelación. Aplazarlo no cuesta nada: es un filtro de análisis, no un knob de entrenamiento (`final_train_eval_acc` queda registrado en cada `summary.json`), así que fijarlo o revisarlo después solo toca código de análisis, nunca obliga a re-correr runs.

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
- **Confound conocido** (el presupuesto de épocas es fijo): a presupuesto fijo, un gap pequeño puede significar que el modelo generaliza bien o que no ha aprendido lo suficiente, y las dos cosas se confunden. Quien analice el gap tiene que separarlas, por ejemplo excluyendo los runs que no aprenden el train o controlando por `final_train_eval_loss`. Cómo hacerlo está por decidir.
- **Respaldo.** La cantidad (riesgo de test − riesgo empírico) y el rol (gap como variable dependiente de un estudio correlacional) son de la literatura: Jiang et al. 2020 (arXiv:1912.02178), Dziugaite et al. 2020 (arXiv:2010.11924); incluso la estimación del término de train por submuestreo tiene precedente en Jiang §3. Lo propio del TFG son los dos controles.

### 2026-06-12

#### Protocolo de evaluación: train optimiza, val monitoriza, test certifica

Confirmado por el tutor (respuesta rápida del 2026-06-12: particiones típicas de cada dataset, sin validación cruzada, semillas múltiples sobre train, val para evaluar convergencia, test para el resultado final) e implementado el mismo día en `src/data.py` + `src/train.py`. Los tamaños de partición resultantes están en [[1 - Diseño]] §Setup de entrenamiento.

- **Roles únicos, sin cruces.** El modelo entrena con el train recortado; la probe de métricas se muestrea de ese mismo train; la monitorización por época y todos los indicadores de eficiencia leen val; el test se evalúa exactamente una vez al final, produciendo `final_test_acc` y `final_test_f1_macro` (vía matriz de confusión en torch, sin dependencias nuevas).
- **Lecturas estables de la curva.** VD1 (épocas-hasta-umbral) y VD3 (mejor loss) se leen sobre la curva de val suavizada con mediana móvil centrada de 3 épocas (`median3` en `train.py`; la ventana encoge en los bordes). Motivo: los extremos de una serie ruidosa están sesgados en proporción a su volatilidad, la volatilidad depende del LR y las métricas de ruido de gradiente plausiblemente la predicen. Sin suavizado, el propio estimador crearía un confusor entre predictor y VD. VD2 (AUC) integra la curva cruda: integrar ya amortigua el ruido. La curva cruda completa queda en `trajectory.parquet`, todo recomputable post-hoc.
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
