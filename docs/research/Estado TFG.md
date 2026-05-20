## Framing

Estudio correlacional. Objetivo único: cuantificar en qué medida métricas de variabilidad y alineación de gradientes medidas en la fase inicial del entrenamiento predicen indicadores de eficiencia del entrenamiento completo. La componente de intervención (early stopping basado en la señal) queda explícitamente fuera del alcance del TFG por restricciones de tiempo. Puede mencionarse como trabajo futuro.

## Hipótesis operativa

La variabilidad y/o alineación de los gradientes, medida a través de distintas métricas en una fracción inicial del entrenamiento, correlaciona significativamente con indicadores de eficiencia del entrenamiento completo, bajo variaciones de learning rate y optimizador, en arquitecturas de visión por computador.

Hipótesis falsada si las correlaciones son débiles (|ρ| < 0.3) o inestables entre configuraciones en la mayoría de condiciones. Un resultado negativo con análisis robusto sigue siendo contribución válida.

## Diseño experimental

### Variables dependientes (eficiencia del entrenamiento)

1. Número de épocas hasta alcanzar un umbral de accuracy predefinido por dataset (primaria). Runs que no alcancen el umbral se tratan como censurados.
2. Área bajo la curva de test loss dentro de un presupuesto fijo de épocas.
3. Mejor test loss alcanzada dentro de ese presupuesto (secundaria).

### Variables independientes (métricas tempranas)

Pendiente de cerrar la lista definitiva antes de empezar los experimentos. Candidatas actuales repartidas en dos familias:

- **Alineación / coherencia direccional**: cosine similarity entre gradientes de batches, gradient confusion, stiffness, m-coherence.
- **Variabilidad estocástica**: gradient noise scale, normalized gradient variance.

La lista se cierra antes de ejecutar los experimentos. No se añaden métricas a posteriori.

### Ventana temporal

Fracciones fijas del presupuesto total de entrenamiento. Barrido en 5%, 10%, 25%, 50%. El barrido en sí mismo es un resultado reportable (cuán temprano basta para predecir).

### Setup de entrenamiento

- Datasets: MNIST, CIFAR-10, CIFAR-100. Decidido 2026-05-14.
- Arquitecturas: FC, CNN simple, ResNet-18. Decidido 2026-05-14.
- Label noise: descartado en v1. Backlog si sobra tiempo (replicaría Forouzesh / Chatterjee&Zielinski).
- Learning rates: varios por condición.
- Optimizadores: al menos SGD y Adam.

## Protocolo de análisis

- Correlaciones Spearman (primaria) y Pearson (secundaria) entre cada métrica temprana y cada variable dependiente.
- Se reportan todas las correlaciones, tanto significativas como no significativas, para todas las métricas.
- Corrección por comparaciones múltiples (Benjamini-Hochberg / FDR).
- Lista de métricas cerrada antes de ejecutar los experimentos para evitar p-hacking.
- Análisis por condición (arquitectura × dataset) y agregado, para evaluar robustez cross-setting.

## Convergencia de la literatura (15 papers revisados)

Extraído de `metrics.md`, `datasets.md` y `models.md`. Justifica el setup propuesto.

**Datasets** (núcleo común):
- CIFAR-10: 12/15 papers.
- MNIST: 10/15.
- CIFAR-100: 6/15.
- ImageNet: 4/15 — fuera de scope por coste computacional.

**Arquitecturas** (familias dominantes):
- MLPs / Fully-Connected: 9/15.
- ResNets (con ResNet-18 recurrente en Faghri, Forouzesh, Chatterjee & Zielinski, Liu): 8/15.
- CNNs no-ResNet (típicamente 3 capas conv con filtros 3×3): 8/15.
- ViT / ConvNeXt: 2/15 (Hölzl, Fort) — fuera de scope.

**Métricas tempranas** (dos familias en `Estado TFG.md` confirmadas por la literatura):
- **Alineación / coherencia direccional** (7 papers): NTK alignment (Shan & Bordelon), GWA (Hölzl), m-coherence (Chatterjee & Zielinski), stiffness (Fort et al.), gradient confusion η (Sankararaman et al.), gradient disparity $\|g_i - g_j\|_2$ (Forouzesh & Thiran), Coherent Gradients $f_t^p$ (Chatterjee).
- **Variabilidad estocástica** (3 papers): normalized variance $\mathbb{V}[g]/\mathbb{E}[g]^2$ (Faghri et al.), GSNR $\tilde{g}^2/\rho^2$ (Liu et al.), gradient noise scale $B_{\text{simple}} = \operatorname{tr}(\Sigma)/\|G\|^2$ (McCandlish et al.).

**Implicación para el TFG**:
- Setup fijado: MNIST + CIFAR-10 + CIFAR-100 × {FC, CNN simple, ResNet-18} × ambas familias de métricas.
- Coincide con el setup propuesto en este documento.
- ImageNet, transformers y dominios non-vision (Atari, Dota, MNLI) quedan explícitamente fuera del scope.

## Riesgos abiertos

1. **Número de runs por condición.** Correlaciones con n pequeña son inútiles. Objetivo mínimo n ≥ 30 por celda (arquitectura × dataset × optimizador). Pendiente hacer cuentas de cómputo total y recortar condiciones antes de empezar si no cuadra.
2. **Coste computacional de métricas caras** (gradient confusion, m-coherence). Posible abandono si el overhead es inviable.
3. **Diferenciación frente a literatura existente** (McCandlish 2018, Faghri 2020). Aporte a defender: comparativa rigurosa entre múltiples familias de métricas, barrido en fracciones tempranas, análisis de robustez cross-architecture/cross-dataset. Debe ser explícito en la intro.
4. **Coherencia título–contenido.** El título actual enfatiza "alineación". Si el análisis acaba apoyándose más en métricas de variabilidad, revisar el título de la memoria (el de EBRON ya está comprometido).

## Próximos pasos

Orden de prioridad. Pasos 1 y 2 son bloqueantes: sin ellos, lanzar experimentos = desperdicio.

1. **Cerrar lista definitiva de métricas.** Antes de tocar código. Evita p-hacking. Candidatas ya identificadas en dos familias (alineación/coherencia y variabilidad estocástica); decidir cuáles entran y cuáles no.
2. **Calcular budget de cómputo total.** runs × arquitectura × dataset × optimizador × LR × seeds, con n ≥ 30 por celda. Si no cuadra, recortar condiciones ahora, no después. Vinculado al riesgo #1.
3. **Decidir grid de hiperparámetros.**
   - Optimizadores: SGD + Adam (mínimo).
   - LRs: barrido por condición.
   - Presupuesto de épocas fijo por dataset.
   - Umbral de accuracy primario por dataset (define VD1, "épocas hasta umbral").
4. **Pilot run reducido.** MNIST × FC × SGD × 3 LRs × 5 seeds. Valida pipeline (logging de gradientes, cálculo de métricas, storage) antes de escalar. Detecta bugs baratos.
5. **Priorizar métricas baratas primero.** cosine similarity batch-wise, normalized variance, GSNR, gradient noise scale. Diferir m-coherence y gradient confusion (riesgo #2: posible abandono si overhead inviable). Medir overhead real en el pilot.
6. **Preregistrar análisis estadístico.** Spearman primaria + Benjamini-Hochberg/FDR. Documentar antes de ver resultados. Evita p-hacking ex-post.
7. **Setup del repo experimental.** Configuración via YAML/Hydra, seeds fijas, storage de trayectorias por step, separación raw/processed.
8. **Revisar título de la memoria.** Vinculado al riesgo #4. Decidir si "alineación" sigue siendo eje central o si conviene framing más neutro (p. ej. "métricas tempranas de gradiente").