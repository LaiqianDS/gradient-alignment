## Plan actualizado (27 abril → 22 junio 2026)
### Semana 20-24 abril (retrospectiva)
- [x] Búsqueda inversa de papers — ver más recientes para mapear qué está hecho y qué no
- [x] GitHub repo
- [x] EBRON: título más amplio, resumen, palabras clave

### Semana actual (27 abril - 3 mayo). Setup y decisiones
- [x] Búsqueda inversa de papers
- [x] GitHub repo
- [x] EBRON: título, resumen, palabras clave
- [x] Decidir modelos (FC, CNN simple, ResNet-18) y datasets (MNIST, CIFAR-10, CIFAR-100) finales
- [x] Revisar papers y cerrar lista de métricas a implementar (más de 4)
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

## Cola de lectura de papers

Orden recomendado para los 13 pendientes (status `to-read`). Núcleo antes de soporte; conceptual antes de escalable. Mínimo viable para cerrar diseño experimental: 1–5.

### Prioridad alta — núcleo (métrica / baseline / método)
- [x] 1. [[Stiffness - A New Perspective on Generalization in Neural Networks|Stiffness]] (Fort, 2019) — operador base de alineación
- [ ] 2. [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Making Coherence]] (Chatterjee & Zielinski, 2020) — versión escalable de Stiffness + LR
- [ ] 3. [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent|Gradient Confusion]] (Sankararaman, 2020) — alineación ↔ velocidad SGD
- [ ] 4. [[Disparity Between Batches as a Signal for Early Stopping|Disparity Between Batches]] (Forouzesh, 2021) — baseline directo del TFG
- [ ] 5. [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Gradient-Weight Alignment]] (Hölzl, 2025) — competidor reciente, posicionamiento
- [ ] 6. [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|GSNR]] (Liu, 2020) — eje varianza con marco teórico
- [ ] 7. [[Speedy Performance Estimation for Neural Architecture Search|Speedy NAS]] (Ru, 2021) — plantilla metodológica de proxy temprano

### Prioridad media — soporte / related work / contexto
- [ ] 8. [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Coherent Gradients]] (Chatterjee, 2019) — marco conceptual previo
- [ ] 9. [[An Empirical Model of Large-Batch Training|Empirical Model Large-Batch]] (McCandlish, 2018) — fuente `gns_simple`
- [ ] 10. [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training|NTK Alignment]] (Shan & Bordelon, 2021) — fundamento teórico kernel
- [ ] 11. [[An overview of gradient descent optimization algorithms|Ruder overview]] (2017) — capítulo fundamentos

### Prioridad baja — background optimizadores (lectura mínima)
- [ ] 12. [[Adam - A Method for Stochastic Optimization|Adam]] (Kingma & Ba, 2015)
- [ ] 13. [[RMSProp - Divide the gradient by a running average of its recent magnitude|RMSProp]] (Tieleman & Hinton, 2012)

## Backlog / ideas sueltas
- Toy problem para visualización pedagógica de métricas
- Revisar qué modelos plantean los papers (probablemente conjunto fijo recurrente)
- SGD básico como baseline mínimo
- Predicción de convergencia como salida secundaria del análisis
- Respecto a [[A Study of Gradient Variance in Deep Learning]] → LR adaptativo basado en varianza normalizada
