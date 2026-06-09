# Decisiones

Registro único de decisiones del TFG. Dos partes: lo que **falta por decidir** y lo que **ya se decidió** (cronológico, lo más reciente arriba). Cuando una pendiente se cierra, baja al log y se actualiza el diseño en [[1 - Diseño]].

El *qué decidimos y por qué* vive aquí; el *estado resultante del diseño*, en [[1 - Diseño]]; el *calendario y avance*, en [[3 - Progreso]].

## Pendientes (sin cerrar)

Bloquean experimentos. La acción para resolverlas vive en [[3 - Progreso]] (Pasos inmediatos).

- **Lista definitiva de métricas.** Las dos familias están fijadas (alineación/coherencia + variabilidad estocástica), pero la selección concreta dentro de cada una no está cerrada. Bloqueante antes de experimentos para evitar p-hacking.
- **Variante de ResNet.** Familia decidida (FC, CNN simple, ResNet); falta fijar la variante concreta. Candidata: ResNet-18.

## Tomadas (log)

### 2026-06-09 — Decisiones de ejecución

Refinan el diseño cerrado de [[1 - Diseño]].

- **Entrenar hasta convergencia, medir durante todo el trayecto.** Cada run se entrena hasta una convergencia definida de antemano (umbral ε sobre la pérdida, el mismo que ancla "épocas-hasta-umbral") y las métricas se registran a lo largo de *todo* el entrenamiento, no solo en la fracción $f$. Da la serie temporal completa y permite elegir el $f$ predictivo a posteriori.
- **NTK: se menciona, no se calcula.** El NTK alignment se descarta del cómputo. Su justificación teórica descansa en el límite de anchura infinita —irreal en la práctica— y su cálculo es caro (jacobiano per-ejemplo). Se menciona en la redacción como marco teórico de la alineación, no como métrica medida.
- **Doble eje temporal en el análisis: época y % de convergencia.** Correlacionar por época absoluta y también normalizando por el porcentaje de convergencia hacia ε. "Época 10" no significa lo mismo entre problemas de distinta dificultad; el eje "fracción del camino a ε" hace comparables las curvas (mitiga el confusor de dificultad de [[1 - Diseño]] §Confusores).
- **Tiny ImageNet — lo incorporamos si es posible (de momento, sí).** La intención por defecto es sumarlo a los datasets, para que el estudio sea **más completo** y suba el techo de dificultad por encima de CIFAR-100. Solo lo dejaremos fuera si no cabe en el presupuesto de cómputo (sobre todo por las métricas caras). Si finalmente entra, actualizar [[1 - Diseño]] §Setup de entrenamiento.
- **Poda de métricas redundantes, con prueba.** Si dos métricas se comportan casi igual y miden casi lo mismo (pares colineales: GNS≈B·NGV, GSNR primo de NGV, clúster del Gram per-ejemplo), descartar una para aligerar análisis y redacción, pero solo demostrándolo (correlación alta, comportamiento solapado). Habilita análisis a nivel de familias.
- **Varias runs, varias seeds.** Múltiples runs variando seed (y demás ejes) para tener réplicas con las que hacer tests de hipótesis sobre las correlaciones, no leer un único número. Conecta con el objetivo n ≥ 30 por celda (riesgo #1 de [[1 - Diseño]]).
- **Baseline = loss (confirmado).** El baseline es la curva de loss (TSE + val-loss tempranas); toda métrica de gradiente se juzga por su valor incremental sobre ella (ΔR², no ρ crudo). Detalle en [[1 - Diseño]] §Baselines y §Hipótesis a contrastar (H2).

### 2026-05-14 — Setup base: datasets y arquitecturas

- **Decisión:** datasets MNIST + CIFAR-10 + CIFAR-100; familias de arquitectura FC + CNN simple + ResNet; optimizadores SGD y Adam (mínimo).
- **Por qué:** convergencia de la literatura — el núcleo común de los 15 papers con setup. Ver [[1 - Diseño]] §Convergencia de la literatura.
- **Estado:** firme. Queda abierta la variante concreta de ResNet (ver Pendientes).
