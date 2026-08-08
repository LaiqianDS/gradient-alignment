# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] (Pasos inmediatos).

**Ninguna a 2026-08-02.** La última que quedaba, la congelación del plan de análisis, se cerró ese día (ver el log). Lo que queda hasta lanzar la matriz ya no son decisiones sino ejecución.

## Tomadas (log)

### 2026-08-08

Las tres entradas de esta fecha salen de la primera revisión de la matriz en marcha (268 runs terminados: MNIST completo, el smoke de Tiny y media celda de `fc × cifar10 × sgd`). Ninguna se ha tomado habiendo calculado ninguna correlación entre métrica y VD, que a esta fecha siguen sin existir. Ninguna toca `src/`: la matriz corre lanzando un proceso nuevo por run (`run_matrix.py:201`), así que cualquier edición del código entraría en vigor en el run siguiente y partiría la matriz en dos versiones, que es una contaminación que no se repara después.

#### El coste de instrumentación documentado era anterior al barrido compartido

La cifra de 3,21x que este log y [[4 - Análisis]] venían citando como peor caso procede del pilot, que se ejecutó el 2026-06-15. El barrido compartido entró dos días después (commit `8566fc3`, `perf(metrics): share one per-sample gradient sweep per probe`). **Todas las cifras de coste del vault eran, por tanto, de una versión del código que ya no es la que corre la matriz.**

Medido ahora celda a celda contra el pilot, el sobrecoste baja de 3,09x a 2,04x en `fc × cifar10 × sgd`, de 2,40x a 1,65x en `resnet18 × mnist`, de 1,77x a 1,40x en `fc × mnist` y de 1,05x a 1,02x en `cnn × mnist`. El peor caso medido sobre la matriz es **2,04x**. Las dos celdas que encabezaban el pilot (`fc × cifar100` y `fc × tiny_imagenet`) todavía no han corrido; escalando el factor observado quedarían alrededor de 2,1x, dentro de la cota <3-4x con bastante más holgura que antes.

Que la causa es la optimización y no otra cosa se comprueba con tres hechos: el tiempo de entrenamiento por época es el mismo (1,671 s en el pilot frente a 1,638 s en la matriz), el de medición se reduce a la mitad, y las trayectorias de la misma configuración salen **idénticas bit a bit** durante las 40 épocas. Eso último es exactamente lo que promete el invariante de que la ruta compartida y la independiente coincidan hasta el último bit, así que el hallazgo confirma la optimización además de corregir el número.

**δ_H2 se queda en 0,15.** El plan usa el coste como justificación del margen, no como fórmula que lo calcule, y 2x sigue siendo un coste que justifica exigir esa parcial. Bajarlo solo restaría potencia a la hipótesis decisiva. La corrección se declara igualmente como enmienda en [[4 - Análisis]] por ser factual y afectar a dos secciones del preregistro.

#### `fc × cifar10` degenera en VD1, y la proyección deja VD1 justo en el suelo de 18 celdas

Los 26 runs terminados de `fc × cifar10 × sgd` están **censurados los 26**: el umbral de CIFAR-10 es 0,65 de val-acc y el mejor run llega a 0,584. Por la regla de celda degenerada, una celda con más del 80% de censura sale de VD1 para todas las métricas.

No es falta de presupuesto. El pilot corrió esa configuración al presupuesto doblado, 80 épocas, y tocó techo en 0,584 justo en la época 40 para quedarse plano el resto. Una red totalmente conectada no llega a 0,65 en CIFAR-10; es el techo de la arquitectura, y por eso `fc × cifar10 × adam` caerá igual.

[[4 - Análisis]] anticipaba que esto pasaría con FC sobre CIFAR-100 y Tiny-ImageNet, es decir 4 celdas. Con CIFAR-10 dentro son **6**, y la proyección deja VD1 con **18 celdas elegibles**, que es exactamente el suelo por debajo del cual el propio plan manda declarar el estudio incompleto y pasar todo a exploratorio. Margen cero. Quedan además dos celdas en riesgo razonable de degenerar también, `cnn × cifar100` contra 0,35 y `cnn × tiny_imagenet` contra 0,20.

**No se toca el umbral de CIFAR-10.** Sería mover un criterio congelado después de ver que una celda no lo alcanza, y el propio plan degradaría VD1 a exploratoria. Conviene dejar anotado que la restricción es del preregistro y no del dato: recalculando `epochs_to_threshold` desde `trajectory.parquet` con la misma mediana móvil de tres épocas sale idéntico al de `summary.json` en **268 de 268** runs, así que un cambio de umbral no costaría cómputo. Si el tutor decidiera que la calibración estuvo mal hecha, es corregible a coste cero, pero como enmienda declarada y con esa consecuencia asumida.

#### Runs colapsados: la regla de divergencia no los cubría, y solo afectan a GWA

De los 268 runs, 11 divergen con NaN, que es lo previsto. Pero hay **26 que colapsan sin divergir**: se quedan finitos, con la loss de entrenamiento clavada en 2,30 que es ln(10), y la accuracy en el 10% que es el azar. Están en `cnn × mnist` (21) y `fc × mnist × adam` (5), todos en las tasas de aprendizaje altas.

Lo que les pasa es que las ReLU mueren, la última capa recibe un vector de entrada exactamente nulo y el clasificador solo emite su sesgo. Importa porque la regla de divergencia del lado predictor (2026-07-25) decide con la finitud de la fila, y estas filas son perfectamente finitas: el caso se le escapa. Y como se concentra en las tasas altas, que son también las lentas y las censuradas, es la misma estructura de confusión que aquella regla existía para impedir.

**El alcance real es mucho más estrecho de lo que parecía, y conviene ser preciso sobre por qué.** Los gradientes por muestra no son nulos (`var/avg` = 1,9·10⁻⁶ > 0 en todas las épocas), así que la mayoría de columnas son mediciones reales de una geometría degenerada: `gsnr/mean`, `mcoh/global` y los cosenos de confusión salen con valores ordinarios. Incluso `stiffness/cos_within` = 1,0 exacto es real, porque con la entrada de la última capa a cero el único gradiente no nulo es el del sesgo, que es literalmente el mismo vector para todas las muestras de una clase.

La única cantidad fabricada es **`gwa/score_mean`**. GWA usa el gradiente del peso de la última capa, que aquí sí es exactamente cero, y su norma se protege con `clamp_min(EPS)`, de modo que el coseno sale 0/EPS = 0,0 exacto. El coseno de un vector nulo no existe; ese 0,0 es un relleno, y es justo "un valor que parece medido y no lo es". `gwa/kurt` y `gwa/value` ya salen NaN por la rama documentada de varianza nula.

**Regla, y se aplica en el análisis, no en el entrenamiento.** Un `gwa/score_mean` exactamente igual a 0,0 se marca como faltante para GWA en ese (run, época), igual que hace la regla de 2026-07-25 con las columnas de signo. Discrimina sin ambigüedad: los 22 runs que lo cumplen están colapsados y ninguno de los 242 sanos lo cumple. Los runs colapsados **siguen dentro** del estudio para todo lo demás, porque un run que colapsa es genuinamente ineficiente y una métrica que lo anticipe temprano es genuinamente útil, que es la misma lógica por la que los censurados entran con el peor rango en vez de tirarse.

Como el valor 0,0 es detectable a posteriori sobre lo ya registrado, esto **no obliga a re-correr nada ni a tocar `src/` mientras la matriz corre**. Arreglar `gwa.py` para que devuelva NaN en vez de 0,0 queda pendiente para cuando la matriz termine, y es cosmético una vez la regla está escrita.

### 2026-08-05

#### Renombrado del plan congelado a `4 - Análisis.md`

El preregistro pasa a llamarse [[4 - Análisis]] y entra en la serie numerada. El motivo es de orden de lectura: los cuatro documentos que mandan son ahora *qué y por qué* (1), *decisiones* (2), *estado* (3) y *cómo se decide cada hipótesis* (4), y el número lo dice sin que haya que leer el índice. No se pierde la marca de congelado: el estado sigue en la primera línea del propio documento y el acto quedó registrado en la entrada del 2026-08-02, con su commit. Se actualizaron todos los enlaces del vault, la referencia del README raíz, la del notebook y la del backend de análisis, que además apuntaba todavía a la ruta antigua en `pending/`.

#### Corregido el estadístico de degeneración de los diagnósticos de sanidad

El diagnóstico que responde "¿esta métrica llega a moverse dentro de un entrenamiento?" estaba **mal**, y la figura del notebook mostraba fielmente el resultado equivocado. Medía `within_std / RMS(within_std entre runs)`: una normalización contra una referencia calculada **entre** entrenamientos, dominada por el de mayor escala. El resultado ordenaba por escala, no por movimiento.

**Cómo se detectó.** Marcaba como degenerada la `val_loss` de los 13 runs de menor escala, entre ellos los seis de MNIST, cuya val loss obviamente se mueve: recorre el rango 0,18 a 0,02, casi un orden de magnitud. La referencia la fijaba `fc_tiny_imagenet_sgd`, con una desviación de 19,3 frente a 0,005 en MNIST. Una magnitud que no puede ser degenerada salía marcada en más de la mitad de los runs.

**Sustituto.** `signal_to_jitter = std(valores) / std(primeras diferencias)` dentro de cada run. Numerador y denominador escalan igual con la métrica, así que el cociente no depende de las unidades ni de la escala. Y trae su propia referencia en vez de un umbral a mano: una trayectoria que sea ruido blanco alrededor de una constante cumple `std(diff) = √2 · std(valores)`, luego su cociente vale 1/√2 ≈ 0,71. Se conserva como prueba de regresión que mide la misma curva en unidades separadas por un factor de un millón y exige idéntico resultado.

**Qué cambia en las conclusiones.** El orden se invierte en la cabeza y en la cola. `var/normalized` y `noise_scale/simple` figuraban como las más sanas (0,89 y 0,87 del máximo) y son en realidad las que menos se distinguen del temblor (0,81 y 0,82 frente al 0,71 del ruido puro), junto con `stiffness/cos_global` y `mcoh/global`. Y `gwa/value` figuraba como la más muerta (0,003) sin serlo: sus valores oscilan entre ±1,5 en un run de ResNet sobre MNIST y ±4·10⁻⁵ en uno de FC sobre Tiny-ImageNet, seis órdenes de magnitud, que es justo lo que la normalización anterior confundía con inmovilidad.

**Consecuencia sobre la entrada del 2026-08-01.** La corroboración del pilot que allí se cita para GWA no se sostiene tal cual; ver la corrección anotada en esa entrada. El argumento bibliográfico, que es el que decide, no se toca.

#### Sin veredicto binario sobre degeneración

La versión corregida deja de emitir una etiqueta `degenerate` y publica solo la comparación contra la referencia (`below_noise`). El motivo es que 1/√2 es el valor **asintótico**: una métrica que fuera puro ruido cae a un lado o a otro de la línea aproximadamente la mitad de las veces, de modo que un booleano por run afirma una decisión que el estadístico no sostiene con una sola trayectoria. Lo que se lee es la distribución completa de los 24 runs frente a la línea. Nada del plan congelado depende de esta etiqueta: sus reglas de exclusión son sobre degeneración de VD1 por celda, no sobre métricas.

### 2026-08-02

#### Congelación del plan de análisis (acto formal)

El plan se movió de `pending/` a `docs/research/` y se commiteó en `e433377`, con lo que queda bajo control de versiones y **anterior al primer commit de resultados**. Esto cierra la última pendiente del proyecto.

**Por qué el acto importaba y no era trámite.** El plan había sido borrado de git en el commit `563d5a5` y desde entonces vivía solo en `pending/`, que está en `.gitignore`. Es decir, existía el documento pero **no existía ninguna prueba verificable de que precediera a los datos**, que es justamente lo que un preregistro tiene que poder demostrar. El commit es lo que crea esa prueba: no hay que creerse que el plan es anterior a los resultados, se comprueba en el historial.

**El orden queda garantizado por construcción.** `reports/` se versiona a propósito (decisión 2026-07-25, hace además de backup incremental de la Fase 4), así que el primer commit de resultados dejará su propia marca temporal en el historial, por detrás de `e433377`. Se verificó explícitamente que `reports/` sigue **sin** estar en `.gitignore`.

**Qué rige desde ahora.** No se mira ningún resultado de matriz fuera de lo que el plan prescribe. Toda modificación posterior pasa por la política de enmiendas del propio plan: se anota con fecha y motivo en su §Historial de revisiones, se marca como enmienda y se declara en la memoria, y si la enmienda es posterior a haber visto resultados degrada a exploratorio el contraste que toca.

**Contexto del commit.** Fue uno de cinco commits del 2026-08-02, ordenados a propósito: la escritura atómica de `summary.json` primero (prerrequisito para lanzar), después `src/power_analysis.py` (porque el plan lo cita como su fuente de reproducibilidad y no debe referenciar código inexistente), después la congelación, y al final la redacción de la memoria y el notebook del pilot, separados por no tener relación con el acto.

### 2026-08-01

#### Revisión estadística del plan de análisis, antes de congelarlo

Pasada crítica sobre la cadena objetivo → hipótesis → test, con las skills de estadística y con simulación. Toda ella anterior a existir ningún dato de matriz. El detalle vive en [[4 - Análisis]] §Historial; aquí queda el *qué y por qué*.

**Suelo de ajuste del gap, cerrado.** Umbral absoluto por dataset reutilizando los umbrales de accuracy que VD1 ya usa sobre val, aplicados sobre train: MNIST 0,97; CIFAR-10 0,65; CIFAR-100 0,35; Tiny 0,20. No se calibra sobre el pilot porque el pilot no puede hacerlo con honestidad (un valor por celda, al LR central, a presupuesto doblado, sin cubrir el barrido de LR que es lo que puebla esa distribución en la matriz). Reutilizar un número ya congelado y ya justificado añade cero grados de libertad al investigador, y el argumento es de una frase: un run que no alcanza sobre train la accuracy que el estudio exige sobre val no aprendió. Se declara como filtro de "no aprendió" y no como calibración fina; la carga del confusor la lleva la parcial por `final_train_eval_loss`.

**La nota de potencia estaba calculada al α equivocado.** Se recalculó por simulación Monte Carlo (`src/power_analysis.py`, con tests). Confirma el punto central del plan (24 celdas, mediana ρ = 0,30 → potencia 0,993) pero calculaba a α = 0,05 cuando el criterio decide a q, ignorando la multiplicidad. De paso apareció el **suelo discreto** del Wilcoxon: con menos de 9 celdas el p mínimo alcanzable supera el α corregido, así que el test no puede rechazar tenga el efecto que tenga. De ahí el mínimo de 18 celdas elegibles para la regla de matriz incompleta, que antes era un 12 puesto por analogía.

**H2 gana un brazo de equivalencia y cambia de covariable.** El negativo de H2 es una contribución declarada de la tesis, pero un contraste que solo puede rechazar la nula lo convierte en "no encontramos nada", indistinguible de la falta de potencia (con una parcial real de 0,15 la detección es del 0,46). Se añade un TOST con δ = 0,15 anclado en el coste de instrumentación medido (~2,1x según la lectura de entonces; el peor caso real es 3,2x, ver la corrección del 2026-08-05, que solo refuerza el ancla), no en una convención: por debajo de esa parcial, la métrica explica menos del 2,3% de varianza residual, irrelevante a ese precio. Y la parcial primaria pasa de tres covariables a **una** (`val-acc@f`): las tres son casi colineales, así que k = 3 daba casi el mismo ajuste con más varianza, gastando potencia justo en la hipótesis decisiva.

**El criterio de H3 estaba sesgado y se sustituye.** Contaba en cuántas celdas la métrica de mayor ΔR² pertenecía a cada familia, exigiendo 16 de 24. Pero alineación tiene 5 métricas y variabilidad 3, así que bajo la nula el argmax cae en alineación con probabilidad 5/8: **15 celdas esperadas por azar**, y probabilidad **0,42** de superar el umbral sin efecto alguno. El criterio favorecía a la familia que el título apuesta. Se sustituye por la diferencia pareada de medianas de |ρ| por familia dentro de cada celda, insesgada respecto al tamaño de familia y sin maquinaria nueva.

**H6 tenía que tener una nula y no la tenía.** Era la única hipótesis sin criterio de falsación, pese a estar descrita como "prueba más exigente que la magnitud". Se le da el binomial exacto de concordancia de signos contra 0,5, el mismo test que H5. **H5**, a su vez, se restringe a las métricas que superaron H1 (misma disciplina que H4) y declara la no independencia de sus 12 pares.

**Atenuación desigual de ρ por censura.** Ningún documento la recogía: un bloque de rangos empatados comprime el |ρ| alcanzable en proporción a la censura, que está correlacionada con la dificultad de la celda, o sea el confusor que la inferencia en dos etapas existía para evitar. No se corrige el estimador; se vigila con dos comprobaciones que no cuestan cálculo nuevo (etapa 2 restringida a celdas con <25% de censura, y VD2/VD3 como control sin censura).

**Descartado: validación leave-one-cell-out.** Se consideró añadir un contraste fuera de muestra, porque todo el diseño es asociación dentro de muestra. Se descarta: su información marginal sobre el Wilcoxon cross-celda y la fracción de signos consistentes es pequeña, y no compensa añadir un contraste, una familia de corrección y una salvedad de potencia. Queda como trabajo futuro y la limitación se declara en la memoria.

#### Orden de ejecución de la matriz, fijado antes de lanzarla

Lo exige la regla de matriz incompleta del plan: si el cómputo no llega a los 960 runs, qué celdas acaben completas tiene que estar determinado por un orden escrito de antemano y no por una elección posterior a ver resultados. Queda así:

1. **`--dataset mnist --model cnn`** (40 runs, horas). Es una celda barata y completa, con sus 8 LR y sus 5 seeds, sobre la que correr los diagnósticos de adecuación del diseño el primer día. No se desperdicia nada: son 40 puntos legítimos de la rejilla.
2. **El resto de MNIST**, luego **CIFAR-10**, luego **CIFAR-100**, y **Tiny-ImageNet al final**, que es el orden natural de `enumerate_runs` y también el de coste creciente (Tiny es ~64% del total).
3. Dentro de cada dataset, el orden de `enumerate_runs` sin alterar.

El motivo de sacar `mnist × cnn` fuera del orden natural es de diagnóstico, no de resultados: el pilot no puede decir si el barrido de LR genera rango dinámico en el predictor a f = 0,10, porque tiene un run por celda y sin variación de LR ni de seed. Si el diseño falla por ahí, conviene saberlo el día uno con la celda más barata y no el día seis con Tiny a medias. La elección es anterior a ver ningún dato y se fija aquí precisamente para que no pueda serlo después.

**Reconciliación con [[1 - Diseño]].** Los criterios originales de H1, H3 y H4 quedan superseded por el plan, con una nota fechada en el propio diseño en vez de una edición silenciosa, para que el historial muestre que se interrogaron antes de existir datos. Se corrige además el enunciado de H5, que presentaba la invariancia de signo como consecuencia lógica de la decisión raw-grad: no lo es, es una afirmación empírica independiente.

### 2026-07-17

#### Coste de instrumentación: se mantiene la medición completa

Cierra la decisión abierta desde el pilot ([[3 - Progreso]], Pasos inmediatos). Se mantiene la medición tal cual: registro completo de las 8 métricas + baseline al final de cada época, sobre la probe fija de M=256, en toda la rejilla. La prioridad declarada es disponer de datos suficientes: la serie temporal completa por época.

- **Por qué.** El peor caso medido es ~3,2x el wall-clock de un run sin instrumentar, dentro de la cota <3-4x fijada. (Corregido el 2026-08-05: esta entrada citaba fc × tiny_imagenet a 2,08x; ese es el run de mayor coste absoluto, pero el de mayor cociente es fc × cifar100, donde medir cuesta 2,21 s por cada segundo de entrenar. La conclusión de mantener la medición completa no cambia.) Conservar la serie completa preserva la elección de ventanas a posteriori y la línea exploratoria post-meseta anotada como trabajo futuro.
- **Alternativas descartadas.** Bajar la cadencia de medición (pierde resolución de trayectoria y complica el snap exacto de ventanas); submuestrear la probe (M=256 está congelado por comparabilidad cross-celda: tocarlo introduce un confusor); fusionar las 2 batch-sweeps restantes (NGV, gradient disparity) en el sweep compartido (palanca válida de ingeniería, pero no bit-idéntica, cambios ~1e-6 frente a los valores que los tests pinean; queda como optimización futura si el coste apretara).
- **Consecuencia.** El coste aceptado ya está incluido en la proyección de ~147 GPU-h del registro de abajo (el pilot midió las métricas dentro del wall-clock por celda).

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

Verificación previa a la congelación ([[3 - Progreso]], Pasos inmediatos), realizada con 6 subagentes de lectura sobre los PDFs del vault (GSNR/Liu, Coherent Gradients/Chatterjee, Making Coherence/Chatterjee & Zielinski, GWA/Hölzl, GNS/McCandlish, TSE/Ru). La tabla corregida vive en [[Datos experimentales]] §5.3 y en [[4 - Análisis]]; cambios y evidencia clave:

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
- **Controles pre-registrados** (contra el confound del presupuesto fijo): suelo de ajuste (los contrastes del gap excluyen runs cuyo `final_train_eval_acc` no alcance un mínimo calibrado en el pilot; esos runs siguen contando para velocidad y rendimiento) y correlación parcial de Spearman por `final_train_eval_loss`. Más reporte por celda (mediana + peor caso) para todas las familias. Detalle en [[4 - Análisis]].
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
- **Todas las métricas implementadas en toda la rejilla.** El cluster hace viables las caras (m-coherence, gradient confusion); en ResNet-18 las per-sample van last-layer-only. Computar el conjunto completo de antemano no contradice "no añadir métricas a posteriori": la lista *reportada* se decide luego por poda con prueba (pendiente "Lista definitiva de métricas" + decisión de poda de abajo).
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
