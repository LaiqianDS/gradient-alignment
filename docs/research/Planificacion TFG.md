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
- [x] Revisar papers e identificar familias de métricas (selección definitiva pendiente, ver "Pasos inmediatos")
- [ ] Montar pipeline base: carga datos, modelos, bucle entrenamiento, logging (W&B/TensorBoard), semillas fijas
- [x] Montar setup LaTex
- [ ] **Entregable:** Repo entrena modelos×datasets con loss/accuracy logged

### Semanas 1-2 (4-17 mayo). Métricas de alineación
- [ ] Implementar todas las métricas elegidas
- [ ] Sanity checks sintéticos (gradientes paralelos → cosine ~1, random → ~0)
- [ ] Granularidad: global por batch primero; por capa si hay tiempo
- [ ] **Entregable:** Métricas integradas + tests sanity documentados
- [ ] **Criterio éxito:** Valores coherentes, overhead <3-4x

### Semana 3 (18-24 mayo). Experimentos principales
- [ ] Matriz: modelos × datasets × LRs × optimizadores (SGD, Adam). Subset representativo (~18-24 runs) si cómputo aprieta
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

## Estado actual (2026-05-30)

- **Setup:** decisiones cerradas (datasets, arquitecturas, optimizadores mínimos) — ver `Estado TFG.md` §Diseño experimental.
- **Repo:** iniciado, con funcionalidad de seeds para reproducibilidad. Pipeline base de entrenamiento/logging y las métricas de alineación **aún no implementados**.
- **Lectura:** 6/16 papers (`status: read`). Cubierto el núcleo métrica/baseline; pendientes los teóricos (NTK, GSNR, Coherent Gradients) y los de optimizadores (Adam, RMSProp). Detalle abajo en "Cola de lectura".
- **Calendario:** este documento sitúa esta semana en *análisis de correlación* (semana 4, 25-31 mayo), pero el pipeline y las métricas siguen pendientes → retraso real respecto al plan. Vigilar el Plan B de septiembre.
- **Lista de métricas:** familias fijadas (alineación/coherencia + variabilidad estocástica); la selección definitiva sigue sin cerrar (bloqueante, ver "Pasos inmediatos").

## Pasos inmediatos

Orden de prioridad. Pasos 1 y 2 son bloqueantes: sin ellos, lanzar experimentos = desperdicio.

1. **Cerrar lista definitiva de métricas.** Antes de tocar código. Evita p-hacking. Candidatas ya identificadas en dos familias (alineación/coherencia y variabilidad estocástica); decidir cuáles entran y cuáles no.
2. **Calcular budget de cómputo total.** runs × arquitectura × dataset × optimizador × LR × seeds, con n ≥ 30 por celda. Si no cuadra, recortar condiciones ahora, no después. Vinculado al riesgo #1 de `Estado TFG.md`.
3. **Decidir grid de hiperparámetros.**
   - Optimizadores: SGD + Adam (mínimo).
   - LRs: barrido por condición.
   - Presupuesto de épocas fijo por dataset.
   - Umbral de accuracy primario por dataset (define VD1, "épocas hasta umbral").
4. **Pilot run reducido.** MNIST × FC × SGD × 3 LRs × 5 seeds. Valida pipeline (logging de gradientes, cálculo de métricas, storage) antes de escalar. Detecta bugs baratos.
5. **Priorizar métricas baratas primero.** cosine similarity batch-wise, normalized variance, GSNR, gradient noise scale. Diferir m-coherence y gradient confusion (riesgo #2: posible abandono si overhead inviable). Medir overhead real en el pilot.
6. **Preregistrar análisis estadístico.** Spearman primaria + Benjamini-Hochberg/FDR. Documentar antes de ver resultados. Evita p-hacking ex-post.
7. **Setup del repo experimental.** Configuración via YAML/Hydra, seeds fijas, storage de trayectorias por step, separación raw/processed.
8. **Revisar título de la memoria.** Vinculado al riesgo #4 de `Estado TFG.md`. Decidir si "alineación" sigue siendo eje central o si conviene framing más neutro (p. ej. "métricas tempranas de gradiente").

## Cola de lectura

El estado lo deriva Dataview del frontmatter de cada nota, así que marcar un paper como `read` lo mueve automáticamente de la lista de no leídos a la de leídos.

### No leídos

```dataview
TABLE relevance AS "Prio", tfg_note AS "Por qué"
FROM "docs/research/Papers"
WHERE status = "to-read"
SORT file.name ASC
```

### Leídos

```dataview
TABLE relevance AS "Prio", last_review AS "Revisado", tfg_note AS "Por qué"
FROM "docs/research/Papers"
WHERE status = "read"
SORT file.name ASC
```

## Backlog / ideas sueltas
- Toy problem para visualización pedagógica de métricas
- Revisar qué modelos plantean los papers (probablemente conjunto fijo recurrente)
- SGD básico como baseline mínimo
- Predicción de convergencia como salida secundaria del análisis
- Respecto a [[A Study of Gradient Variance in Deep Learning]] → LR adaptativo basado en varianza normalizada
