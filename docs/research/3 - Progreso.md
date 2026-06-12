## Plan original (27 abril → 22 junio 2026) — retrospectiva de lo completado
### Semana 20-24 abril (retrospectiva)
- [x] Búsqueda inversa de papers — ver más recientes para mapear qué está hecho y qué no
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
- [ ] **Criterio éxito:** Valores coherentes ✓ (tests sintéticos, 130 verdes); overhead <3-4x → se mide en el pilot run (ver "Pasos inmediatos"; desde 2026-06-10 cada run separa `metric_seconds`/`train_seconds` en `summary.json`, así que el ratio se lee directo)

## Plan por fases (actualizado 2026-06-12)

Sustituye a las semanas 3-7 del plan original, obsoletas desde que la rejilla completa fijó el horizonte de septiembre (2026-06-09; el antiguo "Plan B septiembre" pasa a ser el plan). Las fases van en orden de dependencia, no de fechas: la 1 corre en paralelo a la 0, y dentro de la 1 los cuatro frentes son independientes entre sí.

### Fase 0 — Confirmación del tutor (bloqueante externo)

- [x] Protocolo de evaluación: confirmado por el tutor el 2026-06-12 (particiones típicas de cada dataset, sin validación cruzada, val para convergencia, test al final) → implementado el mismo día, ver fase 2 y [[2 - Decisiones]]
- [ ] Gap de generalización: la respuesta rápida no lo menciona — pedir confirmación expresa. El plan de análisis ya no depende del tutor (su otra dependencia es el pilot)

### Fase 1 — Trabajo no bloqueado (en paralelo a la fase 0)

1. **Pipeline de análisis** (`src/analysis/` o equivalente): implementar el plan estadístico en código antes de congelarlo — carga de `reports/`, Spearman por celda con censura por rangos, Wilcoxon cross-celda (etapa 2), familias BH + cota BY, parciales de H2, no-inferioridad de H4, binomial de H5, figuras preespecificadas.
	- **Verificación (dry-run del preregistro):** datos sintéticos con efecto plantado (ρ conocido) → el pipeline lo recupera; con ρ = 0 → falsos positivos ≈ nominal tras BH. Si algo del plan no es computable tal como está escrito, se descubre aquí, antes de congelar — no sobre los datos reales.
2. **Cola de lectura, priorizada por lo que bloquea la congelación.** Primero los 6 to-read de la tabla de signos: GSNR (Liu), Coherent Gradients + Making Coherence (m-coherence), GWA (Hölzl), GNS (McCandlish), TSE (Ru) — la tabla del [[Plan de análisis congelado]] exige revisar signos contra los papers al congelar. Después, las citas "de memoria" de los pending docs sin nota propia (Jiang 2020, Bouthillier 2021, Holmes & Friston, Lakens, Zou). Los de optimizadores (Adam, RMSProp) y NTK pueden esperar al estado del arte.
	- **Entregable:** tabla de signos verificada contra los papers + citas de los pending docs comprobadas.
3. **Redacción de lo que no depende de resultados:** estructura real en `thesis/main.tex` (hoy es la plantilla de ejemplo), introducción, estado del arte (desde las notas de `Papers/`), metodología (desde [[1 - Diseño]] y [[2 - Decisiones]] — el contenido ya está escrito, es pasarlo a memoria). Target 50-80 páginas, claridad > extensión.
4. **Logística de cluster:** entorno reproducible en nodo (uv), datasets descargados, scripts de lanzamiento (job arrays / troceado por dataset), smoke test de 1 run real en GPU.
	- **Criterio de éxito:** un run de la matriz corre en el cluster de principio a fin sin intervención manual.

### Fase 2 — Implementar el protocolo de evaluación (al cerrar la fase 0)

- [x] Split de 3 vías (`data.py`: 3 loaders, split estratificado con semilla fija, tamaños convencionales por dataset) + lecturas suavizadas de VD1/VD3 y test único final con F1-macro (`train.py`) — implementado el 2026-06-12, ver [[2 - Decisiones]]
- [x] **Verificación del protocolo:** 166 tests verdes (split determinista/estratificado, mediana-3, umbral insensible a picos) + run corto de MNIST con las columnas nuevas en `summary.json` y lecturas verificadas a mano
- [x] Registrado en [[2 - Decisiones]] y actualizado [[1 - Diseño]] (setup de entrenamiento + VD)
- [ ] Gap de generalización (pasada `evaluate()` final sobre subset fijo de train) — espera confirmación expresa del tutor (fase 0); verificación al implementarlo: `final_gap_loss` ≈ 0 al principio y crece con épocas extra

### Fase 3 — Pilot de calibración + congelación del plan

- [ ] Lanzar los 24 runs (con el protocolo de la fase 2 ya dentro) y leer el report:
	```bash
	uv run python src/run_pilot.py            # reanudable; --dataset para trocear por nodo
	uv run python src/run_pilot.py --status
	uv run python src/run_pilot.py --report   # tabla de calibración por dataset
	```
- [ ] Calibrar con los criterios preescritos (decisión 2026-06-09): presupuesto = meseta + margen, múltiplo de 20; umbral cruzado al 30-60% del presupuesto por CNN/ResNet → editar los 24 YAML **y** `config.py::DATASET_BUDGET`, registrar valores y evidencia en [[2 - Decisiones]]
- [ ] Chequeos adicionales que el pilot cierra: overhead <3-4x (`metric_seconds`/`train_seconds`), ninguna métrica falla sistemáticamente, redundancia GNS ≈ B·NGV, centrado de la rejilla de LR, GPU-h proyectadas para ~960 runs, suelo de ajuste del gap (distribución de `final_train_eval_acc`)
- [ ] **Congelar el plan de análisis:** valores del pilot + tabla de signos verificada → mover [[Plan de análisis congelado]] a `docs/research/`, registrar la congelación con fecha
- [ ] **Criterio de éxito:** presupuestos/umbrales registrados con su evidencia; plan congelado y fechado. A partir de aquí no se mira ningún resultado de matriz sin plan congelado

### Fase 4 — Matriz completa (~960 runs)

- [ ] Lanzar por tandas (troceado por dataset/nodo), monitorizar con `--status`, relanzar pendientes (el launcher reanuda)
- [ ] QA continuo, descriptivo y sin mirar hipótesis: recuento de divergencias y censura por celda, missingness por métrica (>5% en una celda → marcarla, según el plan), runs sin `summary.json` → relanzar
- [ ] Backup incremental de `reports/` fuera del cluster
- [ ] **Criterio de éxito:** 24/24 celdas completas, dataset íntegro y copiado

### Fase 5 — Análisis

- [ ] Ejecutar el pipeline (validado en fase 1) sobre los datos reales: mapa descriptivo por celda + contrastes confirmatorios H1-H6 + figuras preespecificadas (val↔test, ρ vs ventana, heatmaps por familia)
- [ ] Todo lo no preespecificado se reporta etiquetado como exploratorio
- [ ] Poda de métricas redundantes con prueba (GNS≈B·NGV, clúster del Gram) para la lista *reportada*
- [ ] Si hay señal fuerte y sobra tiempo: intervención simple (LR adaptativo por alineación) vs baseline; opción de cierre: confirmar la métrica destacada en seeds no miradas
- [ ] Si no hay señal: resultado negativo documentado con su nota de potencia — resultado válido del diseño
- [ ] **Entregable:** tablas y figuras finales + texto de resultados con etiquetas confirmatorio/exploratorio

### Fase 6 — Redacción final y cierre

- [ ] Resultados + discusión (las limitaciones ya están redactadas en los docs metodológicos) + conclusiones
- [ ] Notas de honestidad de los pending docs a la memoria (Tiny-ImageNet val-como-test, F1 ≈ acc, split fijo compartido)
- [ ] Formato UPV/ETSINF, anexo ODS (1-2 páginas), Turnitin
- [ ] Borrador al tutor → incorporar feedback → entrega EBRON
- [ ] Slides defensa (10-15, ~15 min)
- [ ] **Entregable:** memoria entregada + presentación lista

## Estado actual (2026-06-12)

- **Decisiones:** dos pendientes en [[2 - Decisiones]]: confirmación expresa del gap (externa) y congelación del plan de análisis (espera pilot). El protocolo de evaluación se confirmó e implementó el 2026-06-12.
- **Repo:** pipeline single-run completo con el protocolo de evaluación dentro (split train/val/test estratificado fijo con tamaños convencionales, monitorización por val, test único final con F1-macro, lecturas suavizadas mediana-3), métricas en `src/metrics/` con runner integrado, launchers de matriz y pilot, timing de dos relojes. 166 tests pasan. Falta el código de análisis estadístico (fase 1.1) y el gap (tras confirmación expresa).
- **Pilot:** desbloqueado — el split ya está dentro del pipeline; listo para relanzar la calibración (presupuestos/umbrales se chequean sobre la curva de val suavizada). Idealmente tras cerrar el gap, que también toca la evaluación final.
- **Lectura:** 6/16 papers (`status: read`). Cubierto el núcleo métrica/baseline; 6 de los 10 pendientes son los de la tabla de signos (prioridad de la fase 1.2). Detalle abajo en "Cola de lectura".
- **Memoria:** plantilla ETSINF compilando, sin contenido propio aún (fase 1.3).
- **Lista de métricas:** cerrada con la implementación — variabilidad (normalized variance, GNS simple, GSNR) + alineación/coherencia (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline obligatorio.

## Pasos inmediatos

1. **Seguimiento al tutor** (fase 0) — queda solo la confirmación expresa del gap; VD1-VD4 ya están fijadas e implementadas.
2. **Mientras tanto, fase 1 por apalancamiento:** el pipeline de análisis (1.1) y la lectura de la tabla de signos (1.2) son lo que más acerca la congelación del plan; redacción (1.3) y logística de cluster (1.4) rellenan el resto del paralelo. Nada de la fase 1 depende del tutor.
3. **Pilot de calibración** (fase 3, en cuanto el protocolo esté dentro). Un run por celda (24), LR centrado, seed 0, presupuesto doblado. Sustituye al pilot reducido MNIST×FC×SGD: calibrar presupuestos y umbrales exige ver todas las celdas, y de paso valida el pipeline, el overhead real de las métricas (<3-4x) y las GPU-h por run que fijan el coste total (ambos medibles desde 2026-06-10: cada run loguea `total/metric/train_seconds` y el `--report` muestra el tiempo por celda — decisión "Timing por run" en [[2 - Decisiones]]). Protocolo y justificación en [[2 - Decisiones]]. Cómo ejecutarlo:

   ```bash
   uv run python src/run_pilot.py                         # lanza los 24 (reanudable: relanzar ejecuta solo pendientes)
   uv run python src/run_pilot.py --dataset tiny_imagenet # opcional: trocear por nodo del cluster (--model/--optimizer igual)
   uv run python src/run_pilot.py --status                # progreso hecho/pendiente
   uv run python src/run_pilot.py --report                # al acabar: tabla de calibración por dataset
   ```

   Con el report en la mano: fijar presupuesto (meseta + margen, múltiplo de 20) y umbral (cruzado al 30–60% del presupuesto por CNN/ResNet) editando los 24 YAML de celda **y** `config.py::DATASET_BUDGET`, registrar los valores finales en [[2 - Decisiones]], y lanzar la rejilla con `run_matrix.py`.


## Decisiones

El registro de decisiones (pendientes + log de las tomadas) vive en [[2 - Decisiones]].


## Cola de lectura

### No leídos

```dataview
TABLE relevance AS "Prio", tfg_note AS "Por qué"
FROM "docs/research/Papers"
WHERE status = "to-read"
SORT file.name ASC
```

### Leídos

```dataview
TABLE relevance AS "Prio", tfg_note AS "Por qué"
FROM "docs/research/Papers"
WHERE status = "read"
SORT file.name ASC
```

## Backlog / ideas sueltas
- Toy problem para visualización pedagógica de métricas
- Revisar qué modelos plantean los papers (probablemente conjunto fijo recurrente)
- SGD básico como baseline mínimo
- Predicción de convergencia como salida secundaria del análisis
