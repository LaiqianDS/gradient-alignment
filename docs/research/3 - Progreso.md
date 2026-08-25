## Plan original (27 abril → 22 junio 2026): retrospectiva de lo completado
### Semana 20-24 abril (retrospectiva)
- [x] Búsqueda inversa de papers: ver más recientes para mapear qué está hecho y qué no
- [x] GitHub repo
- [x] EBRON: título más amplio, resumen, palabras clave

### Semana actual (27 abril - 3 mayo). Setup y decisiones
- [x] Búsqueda inversa de papers
- [x] GitHub repo
- [x] EBRON: título, resumen, palabras clave
- [x] Decidir modelos (FC, CNN simple, ResNet) y datasets (MNIST, CIFAR-10, CIFAR-100) finales
- [x] Revisar papers e identificar familias de métricas
- [x] Montar pipeline base: carga datos, modelos, bucle entrenamiento, logging (W&B/TensorBoard), semillas fijas
- [x] Montar setup LaTex
- [x] **Entregable:** Repo entrena modelos×datasets con loss/accuracy logged

### Semanas 1-2 (4-17 mayo). Métricas de alineación
- [x] Implementar todas las métricas elegidas
- [x] Sanity checks sintéticos (gradientes paralelos → cosine ~1, random → ~0)
- [x] Granularidad: global por batch primero; por capa si hay tiempo
- [x] **Entregable:** Métricas integradas + tests sanity documentados
- [x] **Criterio éxito:** Valores coherentes ✓ (tests sintéticos; 212 verdes a 2026-08-01) y overhead <3-4x ✓. El ratio se lee desde `metric_seconds`/`train_seconds` en `summary.json` (separados por run desde el 2026-06-10): peor caso medido sobre la matriz ~2,04x el wall-clock de un run sin instrumentar, dentro de la cota (corregido dos veces: el 2026-08-05 porque la lectura citaba el run de mayor coste **absoluto** y no el de mayor cociente, y el 2026-08-08 porque toda cifra tomada del pilot es anterior al barrido compartido y sobrestima). Cerrado con la decisión de coste de instrumentación del 2026-07-17 ([[2 - Decisiones]]).

## Plan por fases (actualizado 2026-06-12)

Sustituye a las semanas 3-7 del plan original, obsoletas desde que la rejilla completa fijó el horizonte de septiembre (2026-06-09; el antiguo "Plan B septiembre" pasa a ser el plan). Las fases van en orden de dependencia, no de fechas: la 1 corre en paralelo a la 0, y dentro de la 1 los cuatro frentes son independientes entre sí.


### Fase 0: confirmación del tutor (bloqueante externo)

- [x] Protocolo de evaluación: confirmado por el tutor el 2026-06-12 (particiones típicas de cada dataset, sin validación cruzada, val para convergencia, test al final) → implementado el mismo día, ver fase 2 y [[2 - Decisiones]]
- [x] Gap de generalización: confirmado por el tutor el 2026-06-14 e implementado el mismo día (tercer constructo de VD, el gap test−train; ver [[2 - Decisiones]] y VD5-VD6 en [[1 - Diseño]])

### Fase 1: trabajo no bloqueado (en paralelo a la fase 0)

1. **Pipeline de análisis** (`src/`): cargar `reports/` y responder cada hipótesis. Sigue sin escribirse, y el método que debía implementar se retiró el 2026-08-25, así que hoy está por definir antes que por programar. Ver §Pasos inmediatos.
	- **Verificación:** probar el pipeline sobre datos sintéticos con efecto conocido antes de tocar `reports/`. Si algo no es computable tal como está escrito, se descubre ahí y no sobre los datos reales.
2. **Redacción de lo que no depende de resultados:** estructura real en `thesis/main.tex` (hoy es la plantilla de ejemplo), introducción, estado del arte (desde las notas de `Papers/`), metodología (desde [[1 - Diseño]] y [[2 - Decisiones]]; el contenido ya está escrito, es pasarlo a memoria). Target 50-80 páginas, claridad > extensión.
4. **Logística de cómputo:** entorno reproducible con `uv`, datasets descargados, lanzador reanudable, smoke test de 1 run real en GPU. El cómputo es una única GPU dedicada, no un cluster (contexto corregido el 2026-07-17).
	- **Criterio de éxito:** un run de la matriz corre de principio a fin sin intervención manual.

### Fase 2: implementar el protocolo de evaluación (al cerrar la fase 0)

- [x] Split de 3 vías (`data.py`: 3 loaders, split estratificado con semilla fija, tamaños convencionales por dataset) + lecturas suavizadas de VD1/VD3 y test único final con F1-macro (`train.py`); implementado el 2026-06-12, ver [[2 - Decisiones]]
- [x] **Verificación del protocolo:** 166 tests verdes (split determinista/estratificado, mediana-3, umbral insensible a picos) + run corto de MNIST con las columnas nuevas en `summary.json` y lecturas verificadas a mano
- [x] Registrado en [[2 - Decisiones]] y actualizado [[1 - Diseño]] (setup de entrenamiento + VD)
- [x] Gap de generalización (pasada `evaluate()` final sobre subset fijo de train): implementado el 2026-06-14 (ver [[2 - Decisiones]]); verificado: `final_gap_loss` ≈ 0,02 a 1 época y ≈ 0,09 a 12 (positivo y monótono)

### Fase 3: pilot de calibración + congelación del plan

- [x] Lanzar los 24 runs (con el protocolo de la fase 2 ya dentro) y leer el report:
	```bash
	uv run python src/run_pilot.py            # reanudable; --dataset para correr solo una parte
	uv run python src/run_pilot.py --status
	uv run python src/run_pilot.py --report   # tabla de calibración por dataset
	```
- [x] Calibrar con los criterios preescritos (decisión 2026-06-09): presupuestos/umbrales finales escritos en los 24 YAML **y** `config.py::DATASET_BUDGET`. Registrados con su evidencia en [[2 - Decisiones]] (2026-07-17)
- [x] Chequeos adicionales que el pilot cierra: overhead <3-4x (`metric_seconds`/`train_seconds`), ninguna métrica falla sistemáticamente, redundancia GNS ≈ B·NGV, centrado de la rejilla de LR, GPU-h proyectadas para ~960 runs, suelo de ajuste del gap (distribución de `final_train_eval_acc`)
- [x] **Congelar el plan de análisis.** Se hizo el 2026-08-02 y se **deshizo** el 2026-08-25: el plan quedó retirado (ver [[2 - Decisiones]])
- [x] **Criterio de éxito:** presupuestos y umbrales registrados con su evidencia

### Fase 4: matriz completa (~960 runs)

- [x] Lanzar por tandas, monitorizar con `--status`, relanzar pendientes (el launcher reanuda)
- [x] QA continuo y descriptivo: divergencias y censura por celda, missingness por métrica, runs sin `summary.json` relanzados
- [x] `reports/` versionado en git (commit `f6df900`, 2026-08-22): hace de copia de seguridad y de marca temporal
- [x] **Criterio de éxito:** 24/24 celdas completas, 960/960 runs, dataset íntegro y copiado

### Fase 5: análisis

- [ ] **Definir el método de análisis, que hoy no existe** (ver §Pasos inmediatos). El plan anterior se retiró el 2026-08-25
- [ ] Implementarlo en `src/` con sus pruebas y ejecutarlo sobre `reports/`
- [ ] Poda de métricas redundantes con prueba (GNS ≡ M/mcoh − 1, clúster del Gram) para la lista *reportada*
- [ ] Si no hay señal: resultado negativo documentado, que sigue siendo un resultado válido del diseño
- [ ] **Entregable:** tablas y figuras finales + texto de resultados

### Fase 6: redacción final y cierre

- [ ] Resultados + discusión (las limitaciones ya están redactadas en los docs metodológicos) + conclusiones
- [ ] Notas de honestidad de los pending docs a la memoria (Tiny-ImageNet val-como-test, F1 ≈ acc, split fijo compartido)
- [ ] Formato UPV/ETSINF, anexo ODS (1-2 páginas), Turnitin
- [ ] Borrador al tutor → incorporar feedback → entrega EBRON
- [ ] Slides defensa (10-15, ~15 min)
- [ ] **Entregable:** memoria entregada + presentación lista

## Estado actual (rev. 2026-08-25)

- **Matriz: TERMINADA.** 960 de 960 runs, las 24 celdas completas con sus 40 runs. Verificado contando los `summary.json` en disco. Los datos están versionados en git desde el commit `f6df900` (2026-08-22), así que existen fuera de esta máquina.
- **Coste real:** 121,7 h de reloj, de las que 97,6 h son entrenamiento y 24,1 h instrumentación. La proyección del pilot era ~147 GPU-h y sobrestimaba un 21%. El peor sobrecoste por celda es 2,048x, en `fc × cifar100 × sgd`, dentro de la cota <3-4x con holgura.
- **Plan de análisis: RETIRADO el 2026-08-25.** Se borraron el plan preregistrado, su código de potencia y el §Protocolo de análisis del diseño. Las seis hipótesis siguen en [[1 - Diseño]] como afirmaciones falsables, **sin criterio de decisión**. Motivo y consecuencias en [[2 - Decisiones]]. Lo que hay que declarar en la memoria: el análisis que se haga es posterior a los datos.
- **Código de análisis: solo sanidad.** `src/analysis.py` comprueba validez, identidades entre columnas, degeneración, tendencia y redundancia. No hay ni una línea de análisis confirmatorio. Suite en 241 pruebas verdes.
- **Dos hechos verificados sobre los 960 que condicionan cualquier análisis** (detalle en [[2 - Decisiones]]): FC no alcanza nunca su umbral en CIFAR-10, CIFAR-100 ni Tiny, así que en seis de las 24 celdas la variable "épocas hasta el umbral" no existe para ningún run; y 154 runs se quedan clavados en el azar, con una firma detectable, `gwa/score_mean` igual a 0,0 exacto.
- **Memoria: cuatro capítulos redactados y tres esqueletos.** Introducción (2.039 palabras), estado del arte (3.056), fundamentos (3.306) y metodología (4.578) son prosa real. Implementación está a medias (1.457 palabras: solo §Verificación). Resultados (43 palabras), conclusiones (13) y los dos anexos son títulos con comentarios. **No hay ni una figura** en toda la memoria, y los tres resúmenes de `main.tex` siguen en `????`.
- **Pilot:** ejecutado y leído; presupuestos y umbrales en `config.py::DATASET_BUDGET` y en los 24 YAML. Aviso: `reports_pilot/` está en `.gitignore`, así que la evidencia que justifica esos números existe **solo en el disco local**.
- **Lista de métricas:** cerrada con la implementación: variabilidad (normalized variance, GNS simple, GSNR) y alineación (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline.

## Pasos inmediatos (rev. 2026-08-25)

**Lo único que bloquea: no hay método de análisis.** Los datos están completos y el capítulo de resultados no puede escribirse sin decidir cómo se responde cada hipótesis.

- [ ] **Definir el método, hipótesis por hipótesis.** Empezar por una sola, la que mejor se entienda, y escribir en lenguaje llano qué pregunta hace y con qué cuenta concreta se responde. No hace falta recuperar la maquinaria anterior. Dos cosas hay que decidirlas de forma explícita, porque los datos las imponen: qué se hace con los 154 runs clavados en el azar, y qué se hace con las seis celdas donde la variable de velocidad no existe.
- [ ] **Escribir ese análisis en `src/`,** con sus pruebas, como el resto del repo.
- [ ] **Redactar resultados y conclusiones,** que dependen del punto anterior.
- [ ] **Figuras.** Hoy hay cero en la memoria. `src/plots.py` existe y fija el estilo, pero no lo llama nadie.
- [ ] **Copia de seguridad de `reports_pilot/`.** Está en `.gitignore` y solo existe en local: si se pierde ese disco, se pierde la evidencia de la calibración.
- [ ] **Completar el capítulo de implementación,** que hoy solo tiene §Verificación.
- [ ] **Los tres resúmenes y el anexo ODS,** obligatorios para el depósito.
