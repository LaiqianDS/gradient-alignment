# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] (Plan hasta la entrega).

**El método de análisis, entero (abierto el 2026-08-25).** Los 960 entrenamientos están hechos y sus datos versionados en `reports/`. No hay método definido para analizarlos: el plan anterior se retiró (ver el log del 2026-08-25). Lo que queda por decidir es, hipótesis por hipótesis, con qué cuenta concreta se responde. Es la única pendiente, y bloquea el capítulo de resultados.

Tamaño del problema de multiplicidad, para tenerlo presente al decidir: `metrics_at_window.parquet` registra **27 columnas de predictor**, porque cada métrica escribe varias variantes (*stiffness* seis, *gradient confusion* cinco, GSNR tres, TSE cuatro, el resto una o dos). Una correlación consume los 40 entrenamientos de una celda, así que cada elección de predictor, ventana e indicador de eficiencia produce 24 coeficientes, uno por celda. Con 27 predictores, 4 ventanas tempranas y los indicadores de eficiencia en juego, el factorial completo son miles de contrastes sobre las mismas 24 celdas. El método tiene que decidir qué subconjunto se contrasta y cómo se corrige la multiplicidad, no barrerlo entero.

El 2026-08-27 se retiró de este log la tanda de decisiones del 2026-08-26, la que abría la primera pasada de la fase A, junto con el código y el texto que produjo. La fase A se rehace después de la fase 0, así que sus decisiones se vuelven a tomar entonces y se registran aquí con la fecha en que se tomen. Lo que sobrevive de aquella pasada, porque está aprobado y escrito en la memoria, son la población de análisis y el tratamiento del censurado de `metodologia.tex`.

## Tomadas (log)

### 2026-08-27

#### La comparación SGD↔Adam se hace sobre las ocho posiciones, no sobre el solape de rejillas

Las rejillas de LR de SGD y Adam **se solapan en seis de sus ocho valores**, de 3e-4 a 1e-1; solo SGD tiene 0,3 y 1, y solo Adam tiene 3e-5 y 1e-4. Es consecuencia mecánica del desplazamiento, porque una década son dos saltos en una rejilla espaciada a medias décadas. De ahí la pregunta de si la comparación pareada de OE5 no sería más limpia sobre esos seis valores compartidos, al mismo LR nominal, en lugar de por posición en la rejilla. **Se decide mantener las ocho posiciones** y no usar el solape.

- **Decide el reparto de los fallos, medido sobre los 960 `summary.json`.** Con las ocho posiciones, SGD tiene 72 entrenamientos clavados en el azar de 480 y Adam 82 de 480, un 15 % frente a un 17 %, y los dos fallan por el mismo lado, el de paso demasiado grande. Restringido al solape, SGD cae a 3 de 300 y Adam se queda en 82 de 300, un 1 % frente a un 27 %. La restricción convierte una comparación equilibrada en una desequilibrada, porque **el mismo valor nominal no es el mismo régimen** en los dos optimizadores: sobre el solape, SGD recorrería de lento a bueno sin fallar casi nunca y Adam de bueno a muerto uno de cada cuatro.
- **De paso queda validado el desplazamiento de 10×**, que hasta hoy era una suposición tomada de los valores por defecto canónicos y del *momentum* 0,9, que amplifica el paso efectivo de SGD unas diez veces. El piloto no pudo comprobarlo porque solo ejecutó el punto central de cada rejilla. La matriz completa sí lo comprueba a posteriori: que los dos optimizadores fallen casi al mismo ritmo, 72 frente a 82 muertos y 227 frente a 239 censurados, es lo que se observa cuando las dos rejillas cubren tramos comparables de sus rangos respectivos.
- **Se descarta también el solape como comprobación de robustez**, que fue la primera propuesta. Sería un control peor que aquello que controla: con un desequilibrio propio del 1 % frente al 27 %, una discrepancia en OE5 no permitiría distinguir un desplazamiento mal elegido de un subconjunto torcido.
- **Criterio de "clavado en el azar"** usado en estas cuentas: *accuracy* de validación máxima por debajo de 1,25 veces el azar del conjunto de datos. Da exactamente 154 entrenamientos, el mismo recuento que la firma `gwa/score_mean` = 0,0 que ya estaba registrada, aunque no se ha comprobado que sean el mismo conjunto de 154.

#### Terminología: anglicismos bien conocidos, y siempre en cursiva

Se admiten en la memoria los términos ingleses de uso corriente en el campo, y todo anglicismo va en cursiva **cada vez que aparece**, no solo la primera. Esto último deroga la convención vigente desde julio, que reservaba la cursiva a la primera aparición de cada término.

- **Por qué la cursiva siempre.** Lo manda la norma de la escuela: el material de seminarios dice "cursiva para palabras extranjeras" y remite al DLE para distinguir el extranjerismo crudo, que va en cursiva, del adaptado, que va en redonda. Ninguno de los nuestros está adaptado en el DLE. Son unas 160 cursivas en 43 páginas, algo más de tres por página.
- **Pasan a inglés:** *epoch* y *epochs* (47), *learning rate* y *learning rates* (10), *seed* y *seeds* (8). Con dos consecuencias gramaticales que hubo que resolver a mano: *epoch* y *seed* heredan el género femenino de "época" y "semilla", de modo que los artículos y adjetivos que ya estaban siguen concordando; *learning rate*, en cambio, es masculino en el uso habitual, así que sus diez apariciones se reescribieron una a una.
- **"Conjunto de medición" pasa a "batch de medición"** (17). El nombre anterior era ambiguo en castellano, porque se lee igual como "el conjunto de las mediciones", que es lo contrario de lo que designa. **Se descartó "probe"**, que es el nombre del concepto en el código y encaja por significado, porque en aprendizaje automático *probe* ya designa otra cosa muy conocida, el clasificador lineal entrenado sobre representaciones congeladas. Falla justo el criterio de admisión: es conocido, pero por otra cosa. El término queda definido formalmente en `fundamentos.tex` §Geometría del gradiente, antes de su primer uso.
- **"Tamaño de lote" pasa a "tamaño de batch"** (2). No es una decisión nueva sino la aplicación de la del 2026-07-04, que ya fijaba *batch* frente a "lote" y no se había aplicado en el estado del arte.
- **Regla para los nombres de métrica:** acrónimo en redonda, palabra inglesa en cursiva. Quedan en redonda GSNR, GWA, TSE, NGV y GNS, y también "m-coherencia", que es una adaptación al castellano. Van en cursiva *stiffness*, *gradient disparity* y *gradient confusion*, que son sintagmas ingleses usados como nombre. Se aplica lo mismo a *minibatch*, *dropout* y *weight decay*.
- **Se quedan en castellano**, porque cambiarlos sería anglicismo por anglicismo y no por claridad: "conjunto de datos" frente a *dataset*, "submuestra", "ventana", "banco de pruebas" y los conjuntos de entrenamiento, validación y test.
- **Verificado.** Compila en 43 páginas, cero referencias indefinidas y cero cajas desbordadas. Trampa que costó una reparación: al marcar *batch* se volvió a marcar el que ya estaba dentro de *batch normalization*, y quedaron tres cursivas anidadas.

### 2026-08-25

#### Se retira el plan de análisis

Se borran `4 - Análisis.md` (el plan preregistrado y congelado), `src/power_analysis.py` con sus pruebas, y el §Protocolo de análisis de [[1 - Diseño]]. Las seis hipótesis se conservan en [[1 - Diseño]] como afirmaciones falsables, ahora sin criterio de decisión.

**Por qué.** El plan había crecido hasta un punto en que ya no se entendía por completo, y un método que no se entiende no se puede defender ante un tribunal. Se prefiere partir de una base comprensible y construir el análisis desde ahí, hipótesis por hipótesis.

**Consecuencia que hay que declarar en la memoria.** El plan estaba congelado y commiteado antes del primer resultado, lo que permitía afirmar que el análisis precedía a los datos. Al retirarlo, el análisis que se haga es **posterior a los datos** y así debe presentarse. El plan retirado sigue íntegro en el historial de git por si hiciera falta recuperarlo.

### 2026-08-08

Las tres entradas de esta fecha salen de la primera revisión de la matriz en marcha (268 runs terminados: MNIST completo, el smoke de Tiny y media celda de `fc × cifar10 × sgd`). Ninguna se ha tomado habiendo calculado ninguna correlación entre métrica y VD, que a esta fecha siguen sin existir. Ninguna toca `src/`: la matriz corre lanzando un proceso nuevo por run (`run_matrix.py:201`), así que cualquier edición del código entraría en vigor en el run siguiente y partiría la matriz en dos versiones, que es una contaminación que no se repara después.

#### El coste de instrumentación documentado era anterior al barrido compartido

La cifra de 3,21x que este log venía citando como peor caso procede del pilot, que se ejecutó el 2026-06-15. El barrido compartido entró dos días después (commit `8566fc3`, `perf(metrics): share one per-sample gradient sweep per probe`). **Todas las cifras de coste del vault eran, por tanto, de una versión del código que ya no es la que corre la matriz.**

Medido ahora celda a celda contra el pilot, el sobrecoste baja de 3,09x a 2,04x en `fc × cifar10 × sgd`, de 2,40x a 1,65x en `resnet18 × mnist`, de 1,77x a 1,40x en `fc × mnist` y de 1,05x a 1,02x en `cnn × mnist`. El peor caso medido sobre la matriz es **2,04x**. Las dos celdas que encabezaban el pilot (`fc × cifar100` y `fc × tiny_imagenet`) todavía no han corrido; escalando el factor observado quedarían alrededor de 2,1x, dentro de la cota <3-4x con bastante más holgura que antes.

Que la causa es la optimización y no otra cosa se comprueba con tres hechos: el tiempo de entrenamiento por época es el mismo (1,671 s en el pilot frente a 1,638 s en la matriz), el de medición se reduce a la mitad, y las trayectorias de la misma configuración salen **idénticas bit a bit** durante las 40 épocas. Eso último es exactamente lo que promete el invariante de que la ruta compartida y la independiente coincidan hasta el último bit, así que el hallazgo confirma la optimización además de corregir el número.

**La corrección no cambió ninguna decisión.** El coste servía para justificar cuánto había que exigirle a una métrica de gradiente para que mereciera la pena, no para calcularlo, y 2x sigue siendo un coste alto. (El margen concreto, δ_H2 = 0,15, pertenecía al plan de análisis retirado el 2026-08-25.)

#### FC no alcanza el umbral en CIFAR-10, CIFAR-100 ni Tiny-ImageNet

Verificado sobre los 960 runs (2026-08-25). Ninguna de las seis celdas de FC fuera de MNIST alcanza nunca su umbral de val-accuracy, con ninguna de las 8 tasas de aprendizaje ni ninguna de las 5 semillas. Los techos medidos: 0,584 y 0,569 contra un umbral de 0,65 en CIFAR-10; 0,295 y 0,289 contra 0,35 en CIFAR-100; 0,114 y 0,108 contra 0,20 en Tiny-ImageNet.

El contraste que lo explica: la misma red FC **sí** pasa el umbral en MNIST, con 0,987 y 0,985 contra 0,97. No es falta de presupuesto ni un fallo de configuración. El pilot corrió esa configuración al presupuesto doblado, 80 épocas, y tocó techo en la época 40 para quedarse plano el resto. Es el techo de la arquitectura.

Consecuencia práctica: en esas seis celdas, "cuántas épocas tarda en llegar al umbral" no existe para ningún run. Cualquier análisis que use ese indicador trabaja con 18 de las 24 celdas.

#### Muchos entrenamientos se quedan clavados en el azar, y se detectan por un cero exacto

Verificado sobre los 960 runs (2026-08-25). En las tasas de aprendizaje altas hay entrenamientos que no aprenden nada: su accuracy de validación se queda en el azar. Son **154 de 960** por el criterio de "mejor val-acc por debajo de 1,2 veces el azar", repartidos por 16 de las 24 celdas. No es una curiosidad de un conjunto de datos concreto.

**El mecanismo, comprobado.** Las ReLU mueren, la última capa recibe un vector de entrada exactamente nulo y el clasificador solo emite su sesgo. El gradiente del peso de la última capa es entonces exactamente cero, y como GWA protege su norma con un `clamp_min(EPS)`, `gwa/score_mean` sale **0,0 exacto**. Un cero exacto en coma flotante no sale por casualidad, así que sirve de firma: aparece en 124 runs y, en los afectados, ocupa de media el 74% de las épocas.

**Lo que NO se sostiene.** La versión anterior de esta entrada, escrita con 268 runs, decía que el cero exacto "discrimina sin ambigüedad". Sobre los 960 no es así: de los 124 runs con cero exacto, 115 acaban clavados y **9 no**. La firma es fuerte, no infalible.

Qué se hace con estos runs al analizar está **por decidir**, como el resto del análisis. Matiz medido el 2026-08-25: en la variable de velocidad no hay nada que decidir, porque todo run clavado es además un run que nunca cruza el umbral, y la cuenta de runs con velocidad medida sale idéntica con ellos y sin ellos en las 24 celdas. La decisión solo afecta a las otras cinco variables dependientes, donde los clavados sí tienen valor (ver [[3 - Progreso]] §Estado actual).

### 2026-08-05

#### Corregido el estadístico de degeneración de los diagnósticos de sanidad

El diagnóstico que responde "¿esta métrica llega a moverse dentro de un entrenamiento?" estaba **mal**, y la figura del notebook mostraba fielmente el resultado equivocado. Medía `within_std / RMS(within_std entre runs)`: una normalización contra una referencia calculada **entre** entrenamientos, dominada por el de mayor escala. El resultado ordenaba por escala, no por movimiento.

**Cómo se detectó.** Marcaba como degenerada la `val_loss` de los 13 runs de menor escala, entre ellos los seis de MNIST, cuya val loss obviamente se mueve: recorre el rango 0,18 a 0,02, casi un orden de magnitud. La referencia la fijaba `fc_tiny_imagenet_sgd`, con una desviación de 19,3 frente a 0,005 en MNIST. Una magnitud que no puede ser degenerada salía marcada en más de la mitad de los runs.

**Sustituto.** `signal_to_jitter = std(valores) / std(primeras diferencias)` dentro de cada run. Numerador y denominador escalan igual con la métrica, así que el cociente no depende de las unidades ni de la escala. Y trae su propia referencia en vez de un umbral a mano: una trayectoria que sea ruido blanco alrededor de una constante cumple `std(diff) = √2 · std(valores)`, luego su cociente vale 1/√2 ≈ 0,71. Se conserva como prueba de regresión que mide la misma curva en unidades separadas por un factor de un millón y exige idéntico resultado.

**Qué cambia en las conclusiones.** El orden se invierte en la cabeza y en la cola. `var/normalized` y `noise_scale/simple` figuraban como las más sanas (0,89 y 0,87 del máximo) y son en realidad las que menos se distinguen del temblor (0,81 y 0,82 frente al 0,71 del ruido puro), junto con `stiffness/cos_global` y `mcoh/global`. Y `gwa/value` figuraba como la más muerta (0,003) sin serlo: sus valores oscilan entre ±1,5 en un run de ResNet sobre MNIST y ±4·10⁻⁵ en uno de FC sobre Tiny-ImageNet, seis órdenes de magnitud, que es justo lo que la normalización anterior confundía con inmovilidad.

**Consecuencia sobre la entrada del 2026-08-01.** La corroboración del pilot que allí se cita para GWA no se sostiene tal cual; ver la corrección anotada en esa entrada. El argumento bibliográfico, que es el que decide, no se toca.

#### Sin veredicto binario sobre degeneración

La versión corregida deja de emitir una etiqueta `degenerate` y publica solo la comparación contra la referencia (`below_noise`). El motivo es que 1/√2 es el valor **asintótico**: una métrica que fuera puro ruido cae a un lado o a otro de la línea aproximadamente la mitad de las veces, de modo que un booleano por run afirma una decisión que el estadístico no sostiene con una sola trayectoria. Lo que se lee es la distribución completa de los 24 runs frente a la línea. Nada del plan congelado depende de esta etiqueta: sus reglas de exclusión son sobre degeneración de VD1 por celda, no sobre métricas.

### 2026-08-01

#### Orden de ejecución de la matriz, fijado antes de lanzarla

Lo exige la regla de matriz incompleta del plan: si el cómputo no llega a los 960 runs, qué celdas acaben completas tiene que estar determinado por un orden escrito de antemano y no por una elección posterior a ver resultados. Queda así:

1. **`--dataset mnist --model cnn`** (40 runs, horas). Es una celda barata y completa, con sus 8 LR y sus 5 seeds, sobre la que correr los diagnósticos de adecuación del diseño el primer día. No se desperdicia nada: son 40 puntos legítimos de la rejilla.
2. **El resto de MNIST**, luego **CIFAR-10**, luego **CIFAR-100**, y **Tiny-ImageNet al final**, que es el orden natural de `enumerate_runs` y también el de coste creciente (Tiny es ~64% del total).
3. Dentro de cada dataset, el orden de `enumerate_runs` sin alterar.

El motivo de sacar `mnist × cnn` fuera del orden natural es de diagnóstico, no de resultados: el pilot no puede decir si el barrido de LR genera rango dinámico en el predictor a f = 0,10, porque tiene un run por celda y sin variación de LR ni de seed. Si el diseño falla por ahí, conviene saberlo el día uno con la celda más barata y no el día seis con Tiny a medias. La elección es anterior a ver ningún dato y se fija aquí precisamente para que no pueda serlo después.

**Reconciliación con [[1 - Diseño]].** Los criterios originales de H1, H3 y H4 quedan superseded por el plan, con una nota fechada en el propio diseño en vez de una edición silenciosa, para que el historial muestre que se interrogaron antes de existir datos. Se corrige además el enunciado de H5, que presentaba la invariancia de signo como consecuencia lógica de la decisión raw-grad: no lo es, es una afirmación empírica independiente.

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
- **Cruce temprano asumido.** Con los umbrales finales, todos los cruces de CNN/ResNet a LR centrado caen al 5-15% del presupuesto (recomputados el 2026-07-17 sobre la curva suavizada desde `trajectory.parquet`; la banda 30-60% preescrita no se alcanza en ningún dataset en el punto central). Se acepta: subir umbrales rozaría el techo de FC (MNIST) o acercaría la censura de CNN (CIFAR-10/100, Tiny), y el rango dinámico de VD1 lo puebla la rejilla completa de LR, no el punto central. El artefacto de `--report` que motivó la confusión (recomputaba las épocas del pilot desde el `DATASET_BUDGET` ya editado y leía el `epochs_to_threshold` calculado con los umbrales candidatos) se corrigió el mismo día: ahora lee las épocas reales de cada run desde `trajectory.parquet` y recomputa el cruce contra el umbral vigente.
- **Coste proyectado.** Escalando el wall-clock por celda del pilot al presupuesto final × 40 runs/celda: MNIST ~9,5 h; CIFAR-10 ~21,6 h; CIFAR-100 ~21,7 h; Tiny-ImageNet ~93,7 h (~64% del total). Total ~147 GPU-h, desde ~250 con los candidatos. Contexto de ejecución corregido (2026-07-17): hay una única GPU disponible, no un cluster (pese a la asunción de la decisión 2026-06-09), así que ~147 GPU-h ≈ ~6 días de GPU continuos; el troceado por nodos no aplica y la matriz se ejecuta por tandas con la reanudación del launcher.
- **Salvedad Tiny.** Los campos test/gap de `reports_pilot/` para Tiny-ImageNet son los corruptos del bug pre-fix; la calibración usó solo el lado val y el timing (sanos). Desde el 2026-07-17 la referencia corregida (re-run post-fix a 40 épocas) vive dentro de cada run del pilot (`testfix_40ep/`), el `summary.json` corrupto se auto-declara vía la clave `_tiny_test_note`, y `reports_validity/` quedó retirada tras la fusión.
- **Qué NO cierra esta entrada.** El suelo de ajuste del gap (mínimo de `final_train_eval_acc` para los contrastes de VD5/VD6) sigue sin valor fijado, y el 2026-07-17 se decide dejarlo así a propósito hasta el acto de congelación. Aplazarlo no cuesta nada: es un filtro de análisis, no un knob de entrenamiento (`final_train_eval_acc` queda registrado en cada `summary.json`), así que fijarlo o revisarlo después solo toca código de análisis, nunca obliga a re-correr runs. Sigue en pie el requisito del plan: debe quedar fijado y registrado al congelar, antes de mirar ningún resultado de la matriz.

#### Tabla de signos de H6 verificada contra los papers

Verificación previa a la congelación ([[3 - Progreso]]), realizada con 6 subagentes de lectura sobre los PDFs del vault (GSNR/Liu, Coherent Gradients/Chatterjee, Making Coherence/Chatterjee & Zielinski, GWA/Hölzl, GNS/McCandlish, TSE/Ru). La tabla corregida vive en [[Datos experimentales]] §5.3; cambios y evidencia clave:

- **m-coherence vs VD1: sigue −, pero la base fuerte cambia de paper.** Chatterjee & Zielinski no afirma velocidad (su α es eficiencia por paso, definicional; las menciones de velocidad son citas a terceros). El claim explícito es de Chatterjee 2020 (CGH): "we expect that greater the agreement in per-example gradients, the faster loss should decrease" (§2.2) y "as noise increases, the time taken to reach a given level of accuracy (i.e., realized learning rate) increases" (§2.3). Matiz: medido sobre train accuracy; la extensión a val es razonada.
- **GSNR vs VD4: de fuerte a extrapolada.** El paper solo afirma el gap ("larger GSNR during training process leads to better generalization performance", vía OSGR, ec. 22; el gap es en loss, la misma cantidad que VD5); no hay claim de test accuracy. Su predicción fuerte es − vs el gap. Matices: la teoría se deriva en fase temprana (favorece la ventana del TFG) pero con full-batch GD, no SGD.
- **GWA vs gap: de fuerte a direccional cualitativa.** El claim cuantitativo del paper es vs test accuracy (Fig. 3: Pearson 0,99 solo ConvNeXt/CIFAR-10; 0,92 cross-arquitectura; medido sobre max de toda la trayectoria, no ventana temprana; su criterio de early stopping descarta el primer 10% como warm-up). El gap operativo (test loss − train loss) nunca se mide. **Corrección 2026-08-05:** la frase original añadía aquí que el pilot corroboraba la rebaja porque GWA quedaba casi constante en ventanas tempranas. Ese hallazgo salía del estadístico de degeneración roto (ver la entrada del 2026-08-05) y no se sostiene: con la medida libre de escala, GWA en ventana temprana marca 1,08 en su valor titular y 1,37 en la media de scores, por encima del 0,71 del ruido puro, y solo su curtosis (0,91) queda pegada a esa línea. Es débil, no plana, y de hecho `var/normalized` y `mcoh/global` se mueven **menos** que GWA sobre la trayectoria completa. La rebaja se mantiene, pero se apoya solo en el argumento bibliográfico, que es el que decide.
- **GNS vs VD1: + confirmado con condiciones.** Base formal ec. 2.7/D.1 (δS = 1 + 𝓑/B a B fijo). Condiciones: régimen B ≲ 𝓑, LR bien ajustado, y el GNS medido depende del LR ("it is not consistent at different learning rates", Ap. A.1), relevante porque el TFG barre LR a B fijo. Su silencio de gap es correcto (caveat 6 del paper).
- **m-coherence vs gap: − confirmado con la salvedad del propio paper** ("this connection is complicated": con 100% label noise la coherencia también sube; lo informativo es la coherencia temprana). Trayectoria esperada para los diagnósticos: no monótona en general ("broad parabolic trajectory"); a granularidad de época con labels reales, decreciente hacia ~1 tras un pico muy temprano.
- **TSE: definiciones y caveats confirmados.** Corrección literal de la cita de §4.2: termina "outside the scope of this paper", no "of our work" (corregido en el plan; la nota del paper ya la tenía bien). γ=0,999 es el default recomendado de §4.1, no la constante definicional (§2 introduce TSE-EMA con γ=0,9). Aviso de archivo: el PDF local es la versión NeurIPS sin apéndices; los apéndices C.1-C.2 (overconfidence, base del caveat de VD2/VD3) solo están en el arXiv v2, conviene archivar esa versión en `Papers/PDFs/`.

La lectura humana de estos papers sigue pendiente y es valiosa para el estado del arte, pero la verificación de la tabla ya no bloquea la congelación.

### 2026-06-14

#### Gap de generalización: tercer constructo de variable dependiente

Confirmado por el tutor el 2026-06-14 e implementado el mismo día en `src/data.py` + `src/train.py`. Cierra la propuesta de `pending/` del 2026-06-10 (revisada el 2026-06-11; texto completo en el histórico git de `pending/Gap de generalización como variable objetivo.md`). Añade el constructo *generalización* a las variables dependientes de [[1 - Diseño]], junto a velocidad y rendimiento final.

- **Qué se mide.** Cinco claves nuevas en `summary.json`: `final_gap_loss = final_test_loss − final_train_eval_loss` (primaria; positivo = sobreajuste), `final_gap_acc = final_train_eval_acc − final_test_acc` (robustez, mismo sentido), sus términos `final_train_eval_loss`/`final_train_eval_acc`, y `final_test_loss` (la evaluación final ya recorría el test; solo se le añadió la loss).
- **Cómo.** Una pasada `evaluate()` extra al final del run, en modo eval y con los mismos pesos, sobre un subconjunto fijo y estratificado por clase del train recortado, de tamaño igual al test y muestreado con `SPLIT_SEED` (idéntico en todos los runs, independiente de la semilla del run; `build_train_eval_loader` en `data.py`). Coste: segundos por run, una vez. Solo toca la evaluación final, como el protocolo ya implementado.
- **Confound conocido** (el presupuesto de épocas es fijo): a presupuesto fijo, un gap pequeño puede significar que el modelo generaliza bien o que no ha aprendido lo suficiente, y las dos cosas se confunden. Quien analice el gap tiene que separarlas, por ejemplo excluyendo los runs que no aprenden el train o controlando por `final_train_eval_loss`. Cómo hacerlo está por decidir.
- **Respaldo.** La cantidad (riesgo de test − riesgo empírico) y el rol (gap como variable dependiente de un estudio correlacional) son de la literatura: Jiang et al. 2020 (arXiv:1912.02178), Dziugaite et al. 2020 (arXiv:2010.11924); incluso la estimación del término de train por submuestreo tiene precedente en Jiang §3. Lo propio del TFG son los dos controles.
- **Verificación.** Suite en verde (tests nuevos: subconjunto train-eval estratificado/fijo/del tamaño del test; claves del gap y sus signos en el smoke test) + run corto de MNIST: `final_gap_loss` ≈0.02 a 1 época y ≈0.09 a 12 (positivo y monótono, signo correcto).
- **Qué queda.** El pilot calibra el suelo de ajuste (distribución de `final_train_eval_acc` por celda). Sin impacto en presupuestos ni umbrales.

### 2026-06-12

#### Protocolo de evaluación: train optimiza, val monitoriza, test certifica

Confirmado por el tutor (respuesta rápida del 2026-06-12: particiones típicas de cada dataset, sin validación cruzada, semillas múltiples sobre train, val para evaluar convergencia, test para el resultado final) e implementado el mismo día en `src/data.py` + `src/train.py`. Cierra la propuesta de `pending/` del 2026-06-10 (revisada el 2026-06-11; el texto completo, con el diagnóstico de los tres problemas y las alternativas A/B/D descartadas, queda en el histórico git de `pending/Protocolo de evaluación y plan de análisis.md`).

- **Split.** Test oficial intacto; val extraído del train, estratificado por clase, con semilla de split fija e independiente de la semilla del run (`SPLIT_SEED` en `data.py`). Todos los runs ven la misma partición, y la única aleatoriedad entre seeds sigue siendo la inicialización y el orden de batches, el objeto de estudio. Tamaños por convención de cada dataset, no por regla uniforme (decisión sobre la respuesta del tutor, que pedía "las particiones típicas"): MNIST 50k/10k/10k (la convención clásica, la que el tutor recordaba), CIFAR-10/100 45k/5k/10k (He et al. 2015, ResNet) y Tiny-ImageNet 90k/10k/10k. En Tiny-ImageNet su `val/` público hace de test (las etiquetas del test oficial no son públicas) y el val de monitorización replica ese tamaño.
- **Roles únicos, sin cruces.** El modelo entrena con el train recortado; la probe de métricas se muestrea de ese mismo train; la monitorización por época y todos los indicadores de eficiencia leen val; el test se evalúa exactamente una vez al final, produciendo `final_test_acc` y `final_test_f1_macro` (vía matriz de confusión en torch, sin dependencias nuevas). En datasets balanceados F1 ≈ acc: se reporta como verificación, no como hallazgo.
- **Lecturas estables de la curva.** VD1 (épocas-hasta-umbral) y VD3 (mejor loss) se leen sobre la curva de val suavizada con mediana móvil centrada de 3 épocas (`median3` en `train.py`; la ventana encoge en los bordes). Motivo: los extremos de una serie ruidosa están sesgados en proporción a su volatilidad, la volatilidad depende del LR y las métricas de ruido de gradiente plausiblemente la predicen. Sin suavizado, el propio estimador crearía un confusor entre predictor y VD. VD2 (AUC) integra la curva cruda: integrar ya amortigua el ruido. La curva cruda completa queda en `trajectory.parquet`, todo recomputable post-hoc.
- **Por qué.** Tres problemas del setup de 2 vías: el sesgo de extremo (arriba), la circularidad de calibración (umbrales calibrados sobre curvas de test del pilot y `epochs_to_threshold` medido después sobre ese mismo test) y la objeción previsible en la defensa ("evaluasteis test cada época"), aunque ninguna decisión de entrenamiento mirase el test (presupuesto fijo, sin early stopping, rejilla preespecificada).
- **Figura de sanidad preespecificada.** Scatter `final_val_acc` vs `final_test_acc` sobre los ~960 runs (ambos en `summary.json`): recupera el diagnóstico de concordancia que se pierde al dejar de evaluar test por época.
- **Notas de honestidad para la memoria.** (1) El "test" de Tiny-ImageNet es su val público, práctica estándar, se declara. (2) F1-macro ≈ accuracy en balanceados: verificación, no hallazgo. (3) El split es fijo y compartido por todos los runs, decisión deliberada: lo estudiado es la variación por seed/LR, no la varianza del estimador (Bouthillier et al. 2021 recomiendan aleatorizar splits cuando se comparan métodos; no es el caso), y se declara.
- **Verificación.** Suite completa en verde (166 tests; nuevos: split estratificado determinista/disjunto/completo, mediana-3 con ventanas de borde, umbral insensible a un pico de una época, claves nuevas del summary en el smoke test) + run corto de MNIST por CLI: curva de val por época, test único final (acc 0.9728, F1 0.9726), `best_val_*` y `epochs_to_threshold` verificados a mano contra la curva suavizada.
- **Qué queda.** Relanzar el pilot de calibración con el split nuevo: los `DATASET_BUDGET` (0.97/0.75/0.35/0.25) se calibraron pensando en test-acc y pasan a chequearse sobre la curva de val suavizada con un train menor. `run_pilot.py --report` ya lee las columnas nuevas (`best_val_acc`, `best_val_loss`, meseta sobre `val_loss`).

### 2026-06-10

#### Timing por run: dos relojes, no uno

Cada run cronometra por separado el entrenamiento y la instrumentación (`src/train.py`): `summary.json` gana `total_seconds`, `metric_seconds` (acumulado alrededor de cada bloque de medición, con `synchronize` en cuda/mps para que los kernels asíncronos se atribuyan al reloj correcto) y `train_seconds` = total − metric.

- **Por qué dos relojes y no uno.** El overhead de la instrumentación (per-sample grads vía vmap sobre la matriz M×P) escala con el tamaño del modelo y con la densidad de probes: un único wall-clock sesgaría las comparaciones de tiempo entre celdas a favor de los modelos pequeños. Evidencia local: en un run corto fc/MNIST el overhead fue ~43% del wall-clock total (incluye el warmup de compilación de vmap del primer probe).
- **Timestamps por fila.** Toda fila de `trajectory.parquet` lleva `elapsed_seconds` y `metric_seconds` acumulados; eso habilita `seconds_to_threshold` junto a `epochs_to_threshold`, la velocidad en wall-clock, más honesta al comparar SGD↔Adam (coste por paso distinto). Es cruda (incluye instrumentación hasta ese punto); la corrección post-hoc es restar la columna acumulada, sin relanzar nada.
- **Convenciones.** `evaluate()` cuenta como entrenamiento (práctica estándar de cualquier run); solo `measure` + baseline TSE van al reloj de overhead. El `synchronize` se hace solo alrededor de los probes (infrecuentes y ya caros), nunca por paso de optimización. El wall-clock es señal de presupuesto y anomalías, no métrica científica: en cluster compartido está confundido por contención de otros jobs, así que no se correlaciona como si fuera limpio.
- **Qué cierra.** El criterio "overhead <3-4x" ([[3 - Progreso]], semanas 1-2) y las "GPU-h reales por run" que el pilot debía validar (decisión del pilot, 2026-06-09) se leen ahora directamente de cada `summary.json`; `run_pilot.py --report` añade la columna `time` por celda para proyectar el coste de los ~960 runs.
- **Verificación.** Tests en tres niveles: unitario determinista (`seconds_to_threshold` extrae el elapsed de la época correcta), invariantes sobre el run de humo (los relojes suman, columnas monótonas y coherentes con el summary) y atribución por inyección (`tests/test_timing.py`: un `sleep` dentro de `measure` debe aterrizar en `metric_seconds`, nunca en `train_seconds`).

### 2026-06-09

#### Pilot de calibración: un run por celda, presupuesto doblado

Concreta el "se calibran en el pilot" de presupuestos y umbrales (decisión "Matriz de runs congelada") en un protocolo ejecutable: `src/run_pilot.py`, módulo aparte del launcher de producción.

- **Qué corre.** Un run por celda (24 en total), LR en el centro de la rejilla (SGD 1e-2, Adam 1e-3, los defaults canónicos de cada optimizador), seed 0 y **el doble del presupuesto candidato** (40/80/120/160 épocas). La asimetría que lo justifica: recortar una curva generosa a posteriori es gratis, estirar una corta es relanzar. El presupuesto define `progress_frac`, las ventanas y el AUC, así que debe quedar bien fijado *antes* de los ~960 runs. El coste del pilot (24 runs a 2×) es ~5% del de la matriz.
- **Qué responde.** (1) *Presupuesto*: dónde se aplana la test loss de los runs bien ajustados → presupuesto final = meseta + margen, redondeado a múltiplo de 20 (conserva el ajuste exacto de `windows`). (2) *Umbral*: debe cruzarse hacia el 30–60% del presupuesto por CNN/ResNet a LR centrado; cruzado en la época 1 no discrimina velocidad, cruzado por casi nadie censura media matriz. `--report` imprime por celda best acc/loss, época de meseta (primera a <2% de la mejor loss) y época de cruce del umbral candidato; la decisión la toma el investigador, no el script.
- **Aislado de `reports/` a propósito.** Los pilots escriben en `reports_pilot/` (gitignored): `run_matrix` da por hecho un punto de la rejilla si existe `reports/<run_name>/summary.json`, y un pilot con LR de rejilla y seed 0 dentro de `reports/` se contabilizaría después como run de producción, entrenado con el presupuesto viejo.
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
- **Umbrales de accuracy 0.97/0.75/0.35/0.25.** Calibrados *por dataset* (no por modelo) para ser alcanzables pero no triviales sin augmentation. Quedan por debajo del techo razonable de las arquitecturas competentes bien ajustadas (FC llega a ~0.98 en MNIST; CNN/ResNet-18 a ~0.75–0.85 en CIFAR-10; ResNet-18 a ~0.4–0.5 en CIFAR-100 y ~0.3–0.4 en Tiny-ImageNet), pero son lo bastante altos para que el número de épocas hasta cruzarlos tenga rango dinámico: un umbral que todo run cruza en la época 1 no discrimina velocidad. El umbral único por dataset hace VD1 comparable dentro de cada celda y entre celdas del mismo dataset; el precio asumido es que las arquitecturas débiles quedan censuradas (FC en CIFAR-100/Tiny-ImageNet, y previsiblemente buena parte de FC en CIFAR-10), y esas celdas se analizan con las VD secundarias. Igual que los presupuestos, se recalibran tras el pilot si quedan mal centrados.
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
- **Horizonte septiembre.** La rejilla completa asume el Plan B de septiembre ([[3 - Progreso]]); no se compromete a entrar antes del 22-jun.

#### Decisiones de ejecución

Refinan el diseño cerrado de [[1 - Diseño]].

- **Entrenar hasta convergencia, medir durante todo el trayecto.** Cada run se entrena hasta una convergencia definida de antemano (umbral ε sobre la pérdida, el mismo que ancla "épocas-hasta-umbral") y las métricas se registran a lo largo de *todo* el entrenamiento, no solo en la fracción $f$. Da la serie temporal completa y permite elegir el $f$ predictivo a posteriori.- **Doble eje temporal en el análisis: época y % de convergencia.** Correlacionar por época absoluta y también normalizando por el porcentaje de convergencia hacia ε. "Época 10" no significa lo mismo entre problemas de distinta dificultad; el eje "fracción del camino a ε" hace comparables las curvas (mitiga el confusor de dificultad de [[1 - Diseño]] §Confusores).
- **Tiny-ImageNet: lo incorporamos si es posible (de momento, sí).** La intención por defecto es sumarlo a los datasets, para que el estudio sea *más completo* y suba el techo de dificultad por encima de CIFAR-100. Solo lo dejaremos fuera si no cabe en el presupuesto de cómputo (sobre todo por las métricas caras). Si finalmente entra, actualizar [[1 - Diseño]] §Setup de entrenamiento.
- **Poda de métricas redundantes, con prueba.** Si dos métricas se comportan casi igual y miden casi lo mismo (pares colineales: GNS≈B·NGV, GSNR primo de NGV, clúster del Gram per-ejemplo), descartar una para aligerar análisis y redacción, pero solo demostrándolo (correlación alta, comportamiento solapado). Permite el análisis a nivel de familias.
- **Varias runs, varias seeds.** Múltiples runs variando seed (y demás ejes) para tener réplicas con las que hacer tests de hipótesis sobre las correlaciones, no leer un único número. Conecta con el objetivo n ≥ 30 por celda (riesgo #1 de [[1 - Diseño]]).
- **Baseline = loss (confirmado).** El baseline es la curva de loss (TSE + val-loss tempranas); toda métrica de gradiente se juzga por su valor incremental sobre ella (ΔR², no ρ crudo). Detalle en [[1 - Diseño]] §Baselines y §Hipótesis a contrastar (H2).

### 2026-05-14

#### Setup base: datasets y arquitecturas

Fija el núcleo experimental del estudio. Detalle en [[1 - Diseño]] §Convergencia de la literatura.

- **Setup mínimo por convergencia de la literatura.** Datasets MNIST + CIFAR-10 + CIFAR-100; familias de arquitectura FC + CNN simple + ResNet; optimizadores SGD y Adam (mínimo). Es el núcleo común de los 15 papers con setup.
- **Variante de ResNet, abierta.** El resto es firme; la variante concreta quedó pendiente (cerrada el 2026-06-09 con ResNet-18).
