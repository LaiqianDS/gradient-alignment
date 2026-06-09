## Plan actualizado (27 abril → 22 junio 2026)
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
- [ ] **Criterio éxito:** Valores coherentes ✓ (tests sintéticos, 130 verdes); overhead <3-4x → se mide en el pilot run (ver "Pasos inmediatos")

### Semana 3 (18-24 mayo). Experimentos principales
- [ ] Matriz: rejilla completa congelada 2026-06-09 — 24 celdas, ~960 runs (8 LR × 5 seeds/celda) sobre GPU/cluster. Spec en [[1 - Diseño]] §Matriz de runs. (El subset ~18-24 queda sin efecto: el cómputo ya no aprieta.)
- [ ] Logging completo: alineación + loss/accuracy
- [ ] Métricas eficiencia: steps hasta loss objetivo, accuracy final, AUC
- [ ] **Entregable:** Dataset resultados (CSV/JSON) por configuración
- [ ] **Criterio éxito:** Datos limpios, reproducibles, sin runs corruptos

### Semana 4 (25-31 mayo). Análisis correlación + intervención
- [ ] Spearman/Pearson entre métricas tempranas (épocas 1, 3, 5, 10) y eficiencia final
- [ ] Robustez across LRs y optimizadores
- [ ] Visualizaciones: scatter, heatmaps correlación, curvas métrica vs época
- [ ] Si hay señal: intervención simple (LR adaptativo según alineación) vs baseline
- [ ] Si no hay señal: documentar resultado negativo honestamente
- [ ] **Entregable:** Tablas correlación + figuras + resultados intervención (o análisis de ausencia de señal)

### Semanas 5-6 (1-14 junio). Redacción memoria
- [ ] Estructura: Introducción, Estado del arte, Metodología, Experimentos y resultados, Conclusiones, Anexos
- [ ] Estado del arte: adaptar (no copiar) doc GradientInstability
- [ ] Target 50-80 páginas, claridad > extensión
- [ ] **Entregable:** Borrador completo

### Semana 7 (15-21 junio). Revisión + entrega
- [ ] Borrador a tutor → incorporar feedback
- [ ] Turnitin + verificar similitud
- [ ] Formato UPV/ETSINF
- [ ] Reflexión ODS anexo (1-2 páginas)
- [ ] Subir memoria EBRON antes 22 junio 14:00
- [ ] Slides defensa (10-15, ~15 min)
- [ ] **Entregable:** Memoria entregada + presentación lista

### Plan B septiembre
Si correlación + intervención + redacción no caben antes 22 junio: cortar en mejor punto, defender lo hecho en septiembre con experimentos extendidos.

## Estado actual (2026-06-09)

- **Setup:** decisiones cerradas (datasets, arquitecturas, optimizadores) — ver [[1 - Diseño]] §Diseño experimental. Rejilla completa congelada 2026-06-09: 24 celdas, ~960 runs.
- **Repo:** pipeline single-run completo (config YAML, datos, modelos, bucle de entrenamiento, logging a parquet), métricas implementadas en `src/metrics/` con runner integrado, y launcher de matriz (`src/run_matrix.py`) con los 24 YAML de celdas en `experiments/`. 130 tests pasan.
- **Lectura:** 6/16 papers (`status: read`). Cubierto el núcleo métrica/baseline; pendientes los teóricos (NTK, GSNR, Coherent Gradients) y los de optimizadores (Adam, RMSProp). Detalle abajo en "Cola de lectura".
- **Calendario:** semanas 1-2 completadas (métricas + sanity checks). Las fechas de mayo quedaron obsoletas al fijar el timeline de septiembre con la rejilla completa (2026-06-09); la secuencia de semanas 3-7 sigue válida como orden de trabajo. Siguiente paso: pilot run, que además cierra el criterio de overhead pendiente.
- **Lista de métricas:** cerrada con la implementación — variabilidad (normalized variance, GNS simple, GSNR) + alineación/coherencia (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline obligatorio.

## Pasos inmediatos

Los bloqueantes previos (lista de métricas, budget de cómputo, grid de hiperparámetros) se cerraron el 2026-06-09 — registro en [[2 - Decisiones]]. Quedan dos antes de lanzar la rejilla completa:

1. **Pilot de calibración.** Un run por celda (24), LR centrado, seed 0, presupuesto doblado. Sustituye al pilot reducido MNIST×FC×SGD: calibrar presupuestos y umbrales exige ver todas las celdas, y de paso valida el pipeline, el overhead real de las métricas (<3-4x) y las GPU-h por run que fijan el coste total. Protocolo y justificación en [[2 - Decisiones]]. Cómo ejecutarlo:

   ```bash
   uv run python src/run_pilot.py                         # lanza los 24 (reanudable: relanzar ejecuta solo pendientes)
   uv run python src/run_pilot.py --dataset tiny_imagenet # opcional: trocear por nodo del cluster (--model/--optimizer igual)
   uv run python src/run_pilot.py --status                # progreso hecho/pendiente
   uv run python src/run_pilot.py --report                # al acabar: tabla de calibración por dataset
   ```

   Con el report en la mano: fijar presupuesto (meseta + margen, múltiplo de 20) y umbral (cruzado al 30–60% del presupuesto por CNN/ResNet) editando los 24 YAML de celda **y** `config.py::DATASET_BUDGET`, registrar los valores finales en [[2 - Decisiones]], y lanzar la rejilla con `run_matrix.py`.
2. **Preregistrar análisis estadístico.** Spearman primaria + Benjamini-Hochberg/FDR. Documentar antes de ver resultados, evita p-hacking ex-post.


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
