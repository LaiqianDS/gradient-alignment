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
- [ ] **Criterio éxito:** Valores coherentes ✓ (tests sintéticos, 130 verdes); overhead <3-4x → se mide en el pilot run (ver "Pasos inmediatos"; desde 2026-06-10 cada run separa `metric_seconds`/`train_seconds` en `summary.json`, así que el ratio se lee directo).

## Plan por fases (actualizado 2026-06-12)

Sustituye a las semanas 3-7 del plan original, obsoletas desde que la rejilla completa fijó el horizonte de septiembre (2026-06-09; el antiguo "Plan B septiembre" pasa a ser el plan). Las fases van en orden de dependencia, no de fechas: la 1 corre en paralelo a la 0, y dentro de la 1 los cuatro frentes son independientes entre sí.


### Fase 0: confirmación del tutor (bloqueante externo)

- [x] Protocolo de evaluación: confirmado por el tutor el 2026-06-12 (particiones típicas de cada dataset, sin validación cruzada, val para convergencia, test al final) → implementado el mismo día, ver fase 2 y [[2 - Decisiones]]
- [x] Gap de generalización: confirmado por el tutor el 2026-06-14 e implementado el mismo día (tercer constructo de VD, el gap test−train; ver [[2 - Decisiones]] y VD5-VD6 en [[1 - Diseño]])

### Fase 1: trabajo no bloqueado (en paralelo a la fase 0)

1. **Pipeline de análisis** (`src/analysis/` o equivalente): implementar el plan estadístico en código antes de congelarlo: carga de `reports/`, Spearman por celda con censura por rangos, Wilcoxon cross-celda (etapa 2), familias BH + cota BY, parciales de H2, no-inferioridad de H4, binomial de H5, figuras preespecificadas.
	- **Verificación (dry-run del preregistro):** datos sintéticos con efecto plantado (ρ conocido) → el pipeline lo recupera; con ρ = 0 → falsos positivos ≈ nominal tras BH. Si algo del plan no es computable tal como está escrito, se descubre aquí, antes de congelar, no sobre los datos reales.
2. **Cola de lectura, priorizada por lo que bloquea la congelación.** Primero los 6 to-read de la tabla de signos: GSNR (Liu), Coherent Gradients + Making Coherence (m-coherence), GWA (Hölzl), GNS (McCandlish), TSE (Ru); la tabla del [[Plan de análisis congelado]] exige revisar signos contra los papers al congelar. Después, las citas "de memoria" de los pending docs sin nota propia (Jiang 2020, Bouthillier 2021, Holmes & Friston, Lakens, Zou). Los de optimizadores (Adam, RMSProp) pueden esperar al estado del arte.
	- **Entregable:** tabla de signos verificada contra los papers + citas de los pending docs comprobadas.
3. **Redacción de lo que no depende de resultados:** estructura real en `thesis/main.tex` (hoy es la plantilla de ejemplo), introducción, estado del arte (desde las notas de `Papers/`), metodología (desde [[1 - Diseño]] y [[2 - Decisiones]]; el contenido ya está escrito, es pasarlo a memoria). Target 50-80 páginas, claridad > extensión.
4. **Logística de cluster:** entorno reproducible en nodo (uv), datasets descargados, scripts de lanzamiento (job arrays / troceado por dataset), smoke test de 1 run real en GPU.
	- **Criterio de éxito:** un run de la matriz corre en el cluster de principio a fin sin intervención manual.

### Fase 2: implementar el protocolo de evaluación (al cerrar la fase 0)

- [x] Split de 3 vías (`data.py`: 3 loaders, split estratificado con semilla fija, tamaños convencionales por dataset) + lecturas suavizadas de VD1/VD3 y test único final con F1-macro (`train.py`); implementado el 2026-06-12, ver [[2 - Decisiones]]
- [x] **Verificación del protocolo:** 166 tests verdes (split determinista/estratificado, mediana-3, umbral insensible a picos) + run corto de MNIST con las columnas nuevas en `summary.json` y lecturas verificadas a mano
- [x] Registrado en [[2 - Decisiones]] y actualizado [[1 - Diseño]] (setup de entrenamiento + VD)
- [x] Gap de generalización (pasada `evaluate()` final sobre subset fijo de train): implementado el 2026-06-14 (ver [[2 - Decisiones]]); verificado: `final_gap_loss` ≈ 0,02 a 1 época y ≈ 0,09 a 12 (positivo y monótono)

### Fase 3: pilot de calibración + congelación del plan

- [x] Lanzar los 24 runs (con el protocolo de la fase 2 ya dentro) y leer el report:
	```bash
	uv run python src/run_pilot.py            # reanudable; --dataset para trocear por nodo
	uv run python src/run_pilot.py --status
	uv run python src/run_pilot.py --report   # tabla de calibración por dataset
	```
- [x] Calibrar con los criterios preescritos (decisión 2026-06-09): presupuestos/umbrales finales escritos en los 24 YAML **y** `config.py::DATASET_BUDGET`. Pendiente: registrar los valores y su evidencia en [[2 - Decisiones]]
- [x] Chequeos adicionales que el pilot cierra: overhead <3-4x (`metric_seconds`/`train_seconds`), ninguna métrica falla sistemáticamente, redundancia GNS ≈ B·NGV, centrado de la rejilla de LR, GPU-h proyectadas para ~960 runs, suelo de ajuste del gap (distribución de `final_train_eval_acc`)
- [ ] **Congelar el plan de análisis:** valores del pilot + tabla de signos verificada → mover [[Plan de análisis congelado]] a `docs/research/`, registrar la congelación con fecha
- [ ] **Criterio de éxito:** presupuestos/umbrales registrados con su evidencia; plan congelado y fechado. A partir de aquí no se mira ningún resultado de matriz sin plan congelado

### Fase 4: matriz completa (~960 runs)

- [ ] Lanzar por tandas (troceado por dataset/nodo), monitorizar con `--status`, relanzar pendientes (el launcher reanuda)
- [ ] QA continuo, descriptivo y sin mirar hipótesis: recuento de divergencias y censura por celda, missingness por métrica (>5% en una celda → marcarla, según el plan), runs sin `summary.json` → relanzar
- [ ] Backup incremental de `reports/` fuera del cluster
- [ ] **Criterio de éxito:** 24/24 celdas completas, dataset íntegro y copiado

### Fase 5: análisis

- [ ] Ejecutar el pipeline (validado en fase 1) sobre los datos reales: mapa descriptivo por celda + contrastes confirmatorios H1-H6 + figuras preespecificadas (val↔test, ρ vs ventana, heatmaps por familia)
- [ ] Todo lo no preespecificado se reporta etiquetado como exploratorio
- [ ] Poda de métricas redundantes con prueba (GNS≈B·NGV, clúster del Gram) para la lista *reportada*
- [ ] Si hay señal fuerte y sobra tiempo: intervención simple (LR adaptativo por alineación) vs baseline; opción de cierre: confirmar la métrica seleccionada en seeds no miradas
- [ ] Si no hay señal: resultado negativo documentado con su nota de potencia, resultado válido del diseño
- [ ] **Entregable:** tablas y figuras finales + texto de resultados con etiquetas confirmatorio/exploratorio

### Fase 6: redacción final y cierre

- [ ] Resultados + discusión (las limitaciones ya están redactadas en los docs metodológicos) + conclusiones
- [ ] Notas de honestidad de los pending docs a la memoria (Tiny-ImageNet val-como-test, F1 ≈ acc, split fijo compartido)
- [ ] Formato UPV/ETSINF, anexo ODS (1-2 páginas), Turnitin
- [ ] Borrador al tutor → incorporar feedback → entrega EBRON
- [ ] Slides defensa (10-15, ~15 min)
- [ ] **Entregable:** memoria entregada + presentación lista

## Estado actual (rev. 2026-06-26)

- **Decisiones:**
  Fase 0 cerrada: protocolo de evaluación confirmado e implementado el 2026-06-12 y gap de generalización confirmado e implementado el 2026-06-14 (ver [[2 - Decisiones]]).
  Queda una pendiente: congelar el plan de análisis.
  Sus dos puertas (confirmación del tutor y calibración del pilot) están cerradas, así que falta solo el paso formal.
- **Repo:**
  Pipeline single-run completo con el protocolo de evaluación dentro: split train/val/test estratificado fijo, monitorización por val, test único final con F1-macro, gap de generalización sobre subset fijo de train, y lecturas suavizadas mediana-3.
  Métricas en `src/metrics/` con runner integrado, launchers de matriz y pilot, y timing de dos relojes.
  Backend de análisis en `src/analysis.py` (loaders + diagnósticos de métricas) con notebook de sanity sobre el pilot.
- **Pilot:**
  Ejecutado y leído; la calibración se cerró el 2026-06-17.
  El número de épocas y los umbrales se determinaron sobre `reports_pilot/` (meseta de val-loss, cruce del umbral en val y timing), datos válidos para todos los datasets incluido Tiny pese al bug de test/gap.
  Presupuestos finales en épocas, escritos en `config.py::DATASET_BUDGET` y en los 24 YAML: mnist 20, cifar10 40, cifar100 40, tiny_imagenet 40.
  Umbrales de accuracy: mnist 0.97, cifar10 0.65, cifar100 0.35, tiny_imagenet 0.20.
  Criterio: cifar10 se bajó de 0.75 para recuperar a CNN, que quedaba censurada; tiny_imagenet de 0.25 para ensanchar el margen de CNN (0.007 era frágil); cifar100 y tiny_imagenet se recortaron del pico de meseta.
  Coste proyectado de la matriz: de ~250 a ~150 GPU-h, con tiny_imagenet en ~63% del total.
  Falta registrar estos valores con su evidencia en [[2 - Decisiones]].
- **Bug de test de Tiny ImageNet (cerrado 2026-06-17):**
  El split de test usaba `ImageFolder(val/)` sobre un `val/` plano, mandando las 10.000 imágenes a una sola clase (test-acc azar ~0.5%, gap inflado ~0.43 fijo en todo el dataset).
  `data.py` ahora etiqueta el val oficial leyendo `val_annotations.txt` y mapeando cada wnid con el `class_to_idx` del train.
  Verificado entrenando 8 épocas (test 0.251 ≈ val 0.249, gap 0.079) y con una regresión hermética en `tests/test_data.py`.
  No hace falta re-correr el pilot de Tiny: el bug solo tocaba los campos de test/gap, y la calibración usó val y tiempos, que nunca estuvieron rotos.
  Separación de fuentes para Tiny: las épocas y umbrales salieron de `reports_pilot/` (lado val + timing, sanos), mientras que el **test-acc/gap corregido** vive en `reports_validity/` (las 6 mismas celdas de Tiny re-corridas con el fix; gitignored, artefacto de referencia; salvedad en [[Datos experimentales]] §5.1).
- **Coste de instrumentación (observación; decisión de diseño abierta):**
  En las celdas caras las métricas se acercan al coste de entrenar; el peor caso medido es fc × tiny_imagenet, con 2153 s de métricas frente a 1993 s de train, es decir ~2.08x el wall-clock de entrenamiento.
  Eso cae dentro de la cota <3-4x que el proyecto se fijó, pero la decisión de qué hacer sigue abierta y debe entrar en el preregistro antes de congelar (ver Pasos inmediatos).
- **Matriz (~960 runs):**
  Aún no lanzada; lista para arrancar en cuanto se congele el plan.
- **Lectura:**
  6/16 papers (`status: read`, según el último registro); pendientes los 6 de la tabla de signos, prioridad para la congelación.
  La tabla de signos esperados ya está redactada en [[Datos experimentales]] §5.3; queda verificarla contra esos PDFs.
- **Memoria:**
  Estructura propia en marcha; redactados intro, estado del arte y fundamentos (2026-06-16) y metodología arrancada (hipótesis H1-H6 y setup).
  Compila limpio.
- **Lista de métricas:**
  Cerrada con la implementación: variabilidad (normalized variance, GNS simple, GSNR) + alineación/coherencia (m-coherence, stiffness, gradient disparity, gradient confusion, GWA), más TSE como baseline obligatorio.

## Pasos inmediatos (rev. 2026-06-26)

**Marcapasos del calendario:**
La cadena que queda es congelar el plan → lanzar la matriz (~960 runs, el tramo más largo de septiembre) → análisis.
Lo que bloquea ahora es la congelación; el gap, el protocolo y la calibración del pilot ya están cerrados.

**Ya cerrado** (hacia el arranque de la matriz; se conserva marcado, no se elimina):

- [x] **Protocolo de evaluación** confirmado e implementado (2026-06-12) y **gap de generalización** (2026-06-14); detalle en el plan por fases (Fase 0 y Fase 2) y en [[2 - Decisiones]].
- [x] **Pilot de calibración** ejecutado y leído; presupuestos/umbrales finales escritos en `config.py::DATASET_BUDGET` + los 24 YAML (2026-06-17).
- [x] **Bug de test de Tiny ImageNet** corregido y verificado (2026-06-17): test 0.251 ≈ val 0.249, gap 0.079; regresión hermética en `tests/test_data.py`.
- [x] **`reports_validity/`**: 6 celdas de Tiny re-corridas con el fix, fuente del **test-acc/gap** corregido de Tiny (la calibración de épocas/umbrales salió de `reports_pilot/`); gitignored, artefacto de referencia; salvedad en [[Datos experimentales]] §5.1.

**Para congelar la preregistración** (la decisión de instrumentación va *dentro* del preregistro, así que precede al acto formal de congelar; lanzar la matriz va al final):

- [ ] **Cerrar la decisión de coste de instrumentación** en las celdas caras (fc/resnet × tiny_imagenet).
  Es la única tarea que aún implica una decisión de diseño, y debe quedar escrita dentro del preregistro antes de congelarlo.
  Las palancas son mantener la medición tal cual, bajar la cadencia de ventanas, o submuestrear el probe; M=256 está congelado por comparabilidad cross-celda, así que tocar el probe introduce un confusor.
  Evidencia para decidir: el peor caso (fc × tiny) es ~2.08x el wall-clock de entrenar, dentro de la cota <3-4x del proyecto.
- [ ] **Cerrar el presupuesto total de cómputo:**
  confirmar que los ~150 GPU-h entran en el cluster con el plazo de septiembre (tiny_imagenet es ~63%, donde más rinde cualquier optimización de la tarea anterior).
- [ ] **Verificar la tabla de signos** contra los 6 PDFs de prioridad alta (GSNR, Coherent Gradients + Making Coherence, GWA, GNS, TSE) y los signos de H6.
  Prioridad a las 3 filas extrapoladas (gradient disparity, GSNR, GWA), que aún no están fijadas contra su paper.
  La tabla ya está redactada en [[Datos experimentales]] §5.3.
- [ ] **Alinear el framing antiguo de "beat TSE-EMA"** que aún arrastran [[Métricas]] (bloque TSE) y el paréntesis de H2 en [[1 - Diseño]].
- [ ] **Registrar en [[2 - Decisiones]]** los presupuestos/umbrales finales del pilot con su evidencia; los valores ya están en `config.py` y los YAML, falta el registro formal.
- [ ] **Congelar:**
  mover [[Plan de análisis congelado]] de `pending/` a `docs/research/` con fecha, incorporando la decisión de instrumentación.
  A partir de aquí no se mira ningún resultado de matriz sin plan congelado.

**Logística (tuya, en paralelo):**

- [ ] **Smoke test en cluster:**
  1 run real de principio a fin si no se ha hecho, antes de la matriz.

**Tras congelar:**

- [ ] **Lanzar la matriz** por tandas (troceado por dataset/nodo) una vez congelado el plan; el launcher reanuda los pendientes por `summary.json`.
  El detalle posterior (QA continuo, backup incremental, ejecución del pipeline de análisis y figuras) está en el plan por fases de arriba (Fase 4 y Fase 5).

**No bloqueante, desde ya:**

- [ ] **Redacción en paralelo** (no depende de resultados):
  completar metodología y dejar listo lo que no espera datos (implementación); intro, estado del arte y fundamentos ya redactados.


## Cola de lectura

El `status` de cada paper vive en su frontmatter (`Papers/`); esta es su foto estática, ordenada por prioridad. 6/16 leídos.

### No leídos (10)

- **[[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]]** (high): Origen de `gwa` (coseno entre el gradiente per-sample y los pesos del clasificador final; casi gratis, last-layer; Pearson 0.99 con test accuracy). Competidor más reciente (2025), uno de los 3 papers más comparables.
- **[[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]]** (high): Origen de `m_coherence`: formalización escalable y O(m) de la Coherent Gradients Hypothesis (coherencia per-sample normalizada, rango [0, m]; 1 = límite ortogonal). Núcleo de la familia alineación.
- **[[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning|Defazio & Bottou 2019]]** (high): Muestra empíricamente que SVRG no reduce la varianza en redes modernas (la ratio de varianza SVRG/SGD supera 1). Cierra el eje varianza con SVRG y Faghri: la varianza es señal diagnóstica, no un objetivo a minimizar.
- **[[Speedy Performance Estimation for Neural Architecture Search|Ru 2021]]** (high): Origen del baseline `tse_ema` (suma exponencialmente ponderada de la train loss, coste cero). Miembro del nivel 0 junto a val-acc@f y val-loss@f. Uno de los 3 papers más comparables.
- **[[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu 2020]]** (high): Origen de `gsnr` (gradient signal-to-noise ratio por parámetro; GSNR alto → menor gap de generalización), con marco teórico OSGR. Eje varianza; es el inverso conceptual de la varianza normalizada de Faghri.
- **[[An Empirical Model of Large-Batch Training|McCandlish 2018]]** (medium): Origen de `gns_simple` (gradient noise scale: ruido del gradiente relativo a su norma) que predice a priori el batch size crítico. Eje varianza; caso canónico de métrica de ruido → elección de hiperparámetro.
- **[[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Chatterjee 2020]]** (medium): Coherent Gradients Hypothesis: ejemplos similares producen gradientes similares, y las direcciones que muchos ejemplos refuerzan son las que generalizan. Marco conceptual de la familia alineación (su estimador operativo es m-coherence).
- **[[Adam - A Method for Stochastic Optimization|Kingma & Ba 2015]]** (low): Optimizador del sweep (SGD vs Adam), no métrica. Normaliza el gradiente por una estimación de su segundo momento (un SNR por parámetro), intuición que motiva el eje varianza del TFG.
- **[[RMSProp - Divide the gradient by a running average of its recent magnitude|Tieleman & Hinton 2012]]** (low): Optimizador adaptativo: divide el gradiente por la raíz de un EMA de su magnitud cuadrática. Related-work, no métrica ni parte del sweep; ese divisor (SNR por parámetro) es el antecedente conceptual del eje varianza.

### Leídos (6)

- **[[A Study of Gradient Variance in Deep Learning|Faghri 2020]]** (high): Origen de `normalized_variance` (NGV: varianza del gradiente relativa a su media, inverso del SNR). Eje varianza; hallazgo contraintuitivo principal: la NGV crece durante el entrenamiento en CIFAR/ImageNet en vez de decrecer.
- **[[Disparity Between Batches as a Signal for Early Stopping|Forouzesh & Thiran 2021]]** (high): Origen de `gradient_disparity` (distancia L2 media entre gradientes de batches independientes; Pearson 0.957 entre la GD train-train y la GD train-val, lo que justifica la versión train-train como proxy). Uno de los 3 papers más comparables.
- **[[Stiffness - A New Perspective on Generalization in Neural Networks|Fort 2019]]** (high): Origen de `stiffness` (coseno/signo entre gradientes per-sample de dos ejemplos: alto si actualizar con uno mejora el otro), desagregado within/between clase. Operador base de la familia alineación; decae al empezar el overfitting.
- **[[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent|Sankararaman 2020]]** (high): Origen de `gradient_confusion` (mínimo coseno entre gradientes de pares de ejemplos; más confusión = SGD más lento). Caso worst-case de la familia alineación; la sobreparametrización por anchura la reduce.
- **[[An overview of gradient descent optimization algorithms|Ruder 2017]]** (medium): Survey didáctico de variantes de SGD (momentum, Adagrad, RMSProp, Adam). Background: respalda acotar el sweep a SGD+Adam y medir sobre el gradiente bruto. No aporta métrica.
- **[[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction|Johnson & Zhang 2013]]** (low): Stochastic Variance Reduced Gradient: reduce la varianza del gradiente con un control variate para acelerar SGD. Related-work del eje varianza; su supuesto de varianza decreciente falla en deep learning real → motiva medir la varianza en vez de reducirla.

## Backlog / ideas sueltas
- Toy problem para visualización pedagógica de métricas
- Revisar qué modelos plantean los papers (probablemente conjunto fijo recurrente)
- SGD básico como baseline mínimo
- Predicción de convergencia como salida secundaria del análisis
