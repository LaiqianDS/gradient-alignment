---
authors:
  - Florian A. Hölzl
year: 2025
status: to-read
relevance: high
last_review: 2026-05-07
url: https://arxiv.org/abs/2510.25480
---

# Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks

## Summary

### Contextualización

El trabajo se enmarca en la búsqueda de proxies de generalización computables en tiempo de entrenamiento (train-time) para tareas de clasificación supervisada con pérdida de entropía cruzada. La práctica habitual para diagnosticar sobreajuste y monitorizar la dinámica de optimización consiste en separar un conjunto de validación independiente y verificar el desempeño sobre datos i.i.d. respecto al conjunto de entrenamiento. Esta estrategia tiene tres limitaciones que motivan el paper: (i) reduce los datos etiquetados disponibles para el ajuste, (ii) presupone independencia distribucional entre entrenamiento y validación, y (iii) ofrece una visión agregada que no permite atribuir el comportamiento del modelo a muestras concretas del dataset. Trabajos previos sobre coherencia de gradientes (gradient coherence, stiffness, Coherent Gradients, GSNR), curvatura del paisaje de pérdida y métodos basados en cambios de predicción como LabelWave o Gradient Disparity (GD) intentan estimar el gap de generalización sin validación, pero presentan problemas de coste computacional, almacenamiento de gradientes por muestra, escalabilidad limitada o sensibilidad a hiperparámetros. La tesis del autor es que medir la dirección de los gradientes per-sample respecto a los pesos del modelo permite caracterizar la convergencia direccional descrita por la teoría de Ji & Telgarsky sobre flujo gradiente con cross-entropy, evitando estadísticas pairwise costosas.

### Aportación: Gradient-Weight Alignment (GWA)

La contribución central es la métrica Gradient-Weight Alignment (GWA), un proxy escalable de generalización calculable de forma online durante el entrenamiento. GWA cuantifica la coherencia entre los gradientes per-sample y los pesos del modelo, recogiendo tanto la contribución específica de cada ejemplo como la dinámica global del dataset. El autor muestra que: (1) GWA predice con precisión el punto óptimo de early stopping y puede sustituir al validation set; (2) permite comparar modelos a través de runs distintos mediante el alineamiento máximo alcanzado; (3) identifica muestras influyentes, outliers y etiquetas ruidosas a partir de los scores per-sample; (4) escala a entrenamiento desde cero y a fine-tuning, manteniéndose robusta bajo ruido de entrada y de etiquetas. La intuición geométrica (Fig. 1) es que durante la fase de aprendizaje efectivo los gradientes individuales están direccionalmente alineados con los pesos (régimen coherente), mientras que la inicialización y el sobreajuste corresponden a regímenes incoherentes con direcciones de gradiente dispersas u ortogonales.

### Metodología

**Definición formal.** Sea $\mathbf{g}_T(x_i) = -\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_T, x_i)$ el gradiente negativo de la pérdida respecto a los pesos $\mathbf{w}_T$ en la época $T$. El score de alineamiento per-sample se define como similitud coseno:
$$\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T) = \frac{\mathbf{g}_T(x_i)\cdot \mathbf{w}_T}{\|\mathbf{g}_T(x_i)\|\,\|\mathbf{w}_T\|}.$$
Sea $\mathcal{G}_T = \{\gamma(x_i, \mathbf{w}_T)\}_{i=0}^N$ el conjunto de scores per-sample en la época $T$, $\mathcal{A}_T$ su distribución empírica y $M_T^{(k)}$ su $k$-ésimo momento. GWA se define como la esperanza con corrección por exceso de curtosis:
$$\text{GWA}_T = \frac{\mathbb{E}_i[\mathcal{A}_T]}{\text{Kurt}_i[\mathcal{A}_T] + \beta} = \frac{M_T^{(1)}}{M_T^{(4)}/(M_T^{(2)})^2 - 3 + \beta}.$$
Se elige $\beta = 1.2$ para compensar el exceso de curtosis de la uniforme sobre $[-1,1]$ (caso límite no esperado en práctica) y garantizar denominador no negativo. La corrección por curtosis se justifica por la teoría long-tail del deep learning: muestras raras o atípicas tienen influencia desproporcionada y producen distribuciones leptocúrticas que reducen GWA, señalando patrones de aprendizaje problemáticos.

**Conexión con teoría.** GWA se inspira en el resultado de Ji & Telgarsky de convergencia direccional bajo flujo gradiente con cross-entropy: en datos perfectamente clasificables, gradiente y pesos convergen en la misma dirección, $\mathbb{E}_i[\gamma(x_i, \mathbf{w}_T)] \to 1$ asintóticamente. En la práctica con datasets ruidosos, la distribución de scores ofrece una medida más rica que la convergencia ideal.

**Estimador escalable.** Para evitar materializar gradientes completos de la red por muestra (prohibitivo en datasets grandes y propenso a degradación de la similitud coseno por alta dimensionalidad), el estimador propuesto restringe el cálculo al clasificador lineal final, donde el gradiente se computa en forma cerrada $\mathbf{g}_t(x_i) = -z_i \cdot (\hat{y}_i - y_i)^\top$, con $z_i$ la representación latente, $\hat{y}_i$ los logits softmax e $y_i$ el target. Esta elección se apoya en que el clasificador lineal proporciona la señal más directa de la tarea aprendida; capas más profundas degradan la estimación. El segundo aspecto crítico es cuándo medir el alineamiento: en lugar de recomputar scores sobre el dataset completo en un paso fijo, los momentos centrales $\hat{M}_T^{(k)}$ se estiman acumulando alineamientos a lo largo de los pasos de actualización de una época. El Algoritmo 1 resume el procedimiento: para cada minibatch $\mathcal{B}_{T,t}$ se hace un forward pass estándar, se computan logits y latents, se obtiene $\mathbf{g}_t(x_i)$ en forma cerrada, se calcula $\gamma(x_i, \mathbf{w}_{T,t})$ y se almacena; al final de la época se devuelve GWA según la Ec. (2). El sesgo introducido por la deriva de pesos durante la época es despreciable: comparado con un offline estimator a $\mathbf{w}_0$ se obtiene un sesgo de primer orden proporcional al learning rate (medias de 0.04 a 0.020 para $\eta$ entre 0.1 y 0.001), y respecto al midpoint $\mathbf{w}_{\text{mid}}$ el sesgo dominante de primer orden se cancela. La correlación online vs. offline es casi perfecta (Fig. 6).

### Datasets y modelos

Las tareas de clasificación cubren CIFAR-10, CIFAR-10-N (variante con etiquetas ruidosas humanas en niveles 0%, 9%, 17% y 40%), CIFAR-C (perturbaciones de entrada), ImageNet-1k con sus splits V2 y ReaL, ImageNet-C, ImageNet-21k (preentrenamiento), iNat18 y Places365 (fine-tuning). Las arquitecturas son ConvNeXt-Femto (training from scratch en CIFAR e ImageNet), ViT/S-16 (training from scratch) y ViT/B-16 preentrenado en ImageNet-21k para fine-tuning. Los splits de validación son 90/10% y 99/1%, y los hiperparámetros se detallan en Tabla 4 (Adam/AdamW para training from scratch (ConvNeXt-P en CIFAR con Adam lr 0.001; ViT en CIFAR-10/-N con Adam lr 0.0001; ConvNeXt-F y ViT/S-16 en ImageNet-1k con AdamW lr 0.001), SGD para fine-tuning, batch sizes 512/1024/512, schedulers Cosine y WarmupCosine). El overhead computacional medido en ViT/S-16 sobre ImageNet-1k con NVIDIA RTX A6000 es de ~2.5 segundos por época (1861 vs. 1867 imágenes/s), inferior al coste de evaluar un validation set del 1% (16 segundos por época), con memoria GPU pico idéntica.

### Métricas y resultados

GWA se compara contra (i) early stopping con val sets 90/10 y 99/1, (ii) LabelWave (cambio de predicciones) y (iii) Gradient Disparity (GD). En la Tabla 1, GWA iguala o supera a la mayoría de criterios: en CIFAR-10/CIFAR-10-N obtiene en promedio +0.4% sobre val set estándar y +0.67% sobre LabelWave; en ImageNet-1k con ViT bate al 99/1 split mientras elimina la dependencia de hold-out. En CIFAR-10 con 0%/9%/17% de ruido alcanza 81.57/78.93/75.70 (ViT) y 89.73/86.08/82.55 (ConvNeXt). El máximo alineamiento $\max_T \mathbb{E}[\mathcal{A}_T]$ correlaciona fuertemente con la accuracy de test sobre ConvNeXt y ViT en CIFAR-10/CIFAR-C: Pearson 0.99/0.98 y Spearman 0.97/0.93 en CIFAR-10; Pearson 0.99/0.94 y Spearman 0.96/0.92 en CIFAR-C ($p<0.001$). En robustez (Tabla 2), modelos seleccionados con GWA mejoran +0.55% promedio en CIFAR-C y +0.67% en ImageNet-C respecto a val set 10%. Los scores per-sample $\gamma(x_i, \mathbf{w}_T)$ identifican casi todas las muestras mal etiquetadas en CIFAR-10-N (Fig. 4) sin técnicas explícitas de detección, y reflejan el sesgo de simplicidad: alineamientos positivos altos al inicio corresponden a ejemplos visualmente simples, mientras que los aumentos posteriores incorporan ejemplos progresivamente más complejos. En fine-tuning (Tabla 3, Fig. 5), GWA muestra una caída inicial seguida de un mínimo y posterior incremento; el criterio adaptado (mínimo inicial, máximo posterior) iguala o supera al val set 1%/10% en ImageNet-1k, iNat18 y Places365.

### Conclusiones

El autor concluye que GWA proporciona una métrica fiable de tiempo de entrenamiento que captura dinámicas dataset-level y sample-level, sirviendo como criterio de early stopping robusto frente a ruido de etiquetas y bajo régimen de datos. Se elimina la necesidad de validation sets y se ofrece una herramienta diagnóstica para comparación de modelos e identificación de muestras influyentes. Las limitaciones reconocidas incluyen la posibilidad de distribuciones bimodales tardías que el estimador con curtosis no captura bien y la sensibilidad de la magnitud del score a la dimensionalidad latente, mitigable con Johnson-Lindenstrauss Transform. Direcciones futuras: extensión al setting self-supervised, modalidades de texto con pérdida autorregresiva e integración con aproximaciones de matrix information theory (MIR/HDR) y neural collapse.

![[Pasted image 20260421195041.png]]

## Medición y pipeline

**Métrica concreta.** Gradient-Weight Alignment (GWA), propuesta por Hölzl (2025). En su forma per-sample se define como la similitud coseno (con signo) entre el gradiente negativo de la pérdida $\mathbf{g}_T(x_i) = -\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_T, x_i)$ y el vector de pesos $\mathbf{w}_T$: $\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T)$. El agregado escalar de época se obtiene como esperanza corregida por exceso de curtosis (Ec. (2) del paper). Para la integración train-time en esta tesis interesa la variante per-layer aplicada al minibatch: $\gamma_l = \cos\text{sim}(\mathbf{g}_l, \mathbf{w}_l)$, con $\mathbf{g}_l$ el gradiente del minibatch sobre los parámetros de la capa $l$ y $\mathbf{w}_l$ sus pesos actuales.

**Entradas.** Parámetros actuales $\mathbf{w}_l$ y gradiente de minibatch $\mathbf{g}_l$ por capa $l$. No requiere forward adicional: se reutiliza el gradiente ya materializado por `loss.backward()`.

**Cuándo computar.** Cada paso durante la ventana temprana de entrenamiento (objeto del estudio) y cada época una vez estabilizada la dinámica. El coste marginal es prácticamente nulo porque se reusa el backward existente.

**Coste.** Por capa, un producto interno $\mathcal{O}(P_l)$ y dos normas $\mathcal{O}(P_l)$. Total $\mathcal{O}(P)$ sobre los parámetros del modelo, despreciable frente al forward/backward.

**Integración en el pipeline.**

```python
# tras loss.backward(), antes de optimizer.step()
gwa_per_layer = {}
for name, p in model.named_parameters():
    if p.grad is None or p.numel() == 0:
        continue
    g = p.grad.detach().flatten()
    w = p.data.detach().flatten()
    gwa_per_layer[name] = torch.nn.functional.cosine_similarity(
        g.unsqueeze(0), w.unsqueeze(0)
    ).item()
gwa_mean = sum(gwa_per_layer.values()) / len(gwa_per_layer)
logger.log_scalars({"gwa/mean": gwa_mean, **{f"gwa/{k}": v for k, v in gwa_per_layer.items()}}, step=step)
```

**Consideraciones.** GWA por capa es más informativa que el escalar global porque preserva heterogeneidad entre bloques (especialmente relevante en CNN y ResNet, donde primeras capas convolucionales y clasificador final exhiben dinámicas distintas). Hay que filtrar parámetros con $\mathbf{w}_l \approx \mathbf{0}$ inicial (bias típicamente inicializados a cero) para evitar coseno indefinido o numéricamente inestable. El valor absoluto y la escala dependen del esquema de inicialización (Kaiming, Xavier), por lo que las comparaciones intra-arquitectura son más robustas que las inter-arquitectura. Al ser una métrica reciente (2025) y directamente formulada como proxy train-time, es comparable de forma natural con las otras métricas cerradas (cosine similarity entre gradientes de minibatch, gradient confusion, stiffness, m-coherence, GNS, varianza normalizada del gradiente).

**Logging.** GWA por capa y promedio agregado. Frecuencia: cada paso durante la ventana temprana (primeras épocas, foco del análisis predictivo), reduciendo a cada época una vez sale del régimen transitorio. Persistir en formato largo (`step, layer, gwa`) para facilitar la correlación posterior con métricas de eficiencia de entrenamiento completo.

## Notes

### Uso en el TFG

- **Métrica que origina.** Es la fuente de `gwa` en el `METRIC_REGISTRY` (familia alineación). Score per-sample $\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T)$: coseno entre el gradiente per-sample y los pesos del clasificador final. Agregado de época con corrección por exceso de curtosis $\text{GWA}_T = M^{(1)} / (\text{Kurt} - 3 + \beta)$, con $\beta = 1.2$.
- **Cómo se usa (casi gratis, online, last-layer).** Estimador cerrado de la última capa $\mathbf{g}_t(x_i) = -z_i (\hat{y}_i - y_i)^\top$ (con $z_i$ latente penúltima, $\hat{y}_i = \text{softmax}(\text{logits})$): se acumula online en el forward de training, $O(B \cdot d_z \cdot C)$ por minibatch, **sin backward extra**. Es la única métrica del registro que no consume ninguno de los tres sweeps compartidos (engancha hooks al forward, acumula momentos por época).
- **Señal.** $\uparrow$ mejor: el paper reporta Pearson 0.99 / Spearman 0.97 entre $\max_T \mathbb{E}[\mathcal{A}_T]$ y test accuracy en CIFAR-10. Cadencia `epoch` (acumulación online, scalar al cerrar la época).
- **Pitfalls / decisiones.** (1) Excluir el bias del clasificador (norma cero $\Rightarrow$ coseno indefinido). (2) Welford para los momentos $M^{(2..4)}$; computar la curtosis desde $\sum \gamma^k$ crudos es numéricamente inestable. (3) $\beta = 1.2$ mantiene el denominador positivo en el límite uniforme; añadir `clamp` $\max(\cdot, \epsilon)$ como salvavida. (4) Convención de signo: el paper usa $\mathbf{g} = -\nabla L$; el pipeline trabaja sobre $\nabla L$ bruto, así que se guarda $\gamma_i$ crudo y se documenta el signo en la capa de análisis. (5) En arquitectura FC el clasificador final **es** la red completa (caso degenerado, marcar con flag explícito).
- **Limitaciones heredadas del paper.** Distribuciones bimodales tardías que la corrección de curtosis no captura bien; la magnitud del score depende de la dimensionalidad latente $d_z$ (mitigable con Johnson-Lindenstrauss). Refuerzan la preferencia por comparaciones intra-arquitectura.
- **Paper comparable más reciente (delta a argumentar).** Junto con Ru et al. (2021, TSE) y Forouzesh & Thiran (2021, Gradient Disparity), es uno de los **tres papers más comparables** al TFG (mismo problema: proxy train-time barato de generalización) y el más reciente. La intro de la tesis debe argumentar el delta frente a este: GWA es un proxy puntual *post-hoc* para early stopping y selección de modelo sobre una métrica/arquitectura; el TFG propone un estudio **correlacional cerrado** que mide 10 métricas (7 alineación + 3 varianza) + baseline en ventanas tempranas (5/10/25/50% de épocas), sobre gradiente **bruto** $\nabla L$ (no solo last-layer), barriendo SGD/Adam y FC/CNN/ResNet, para predecir generalización **y** eficiencia sin optimizar nada.

## Papers relacionados

- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — familia alineación; cosine-stiffness es el mismo coseno entre gradientes per-sample, pero entre pares de muestras en vez de gradiente-vs-pesos.
- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — familia alineación; m-coherence mide alineamiento gradiente-vs-gradiente promedio, alternativa per-sample escalable como GWA.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — familia alineación; gradient confusion (min coseno entre gradientes per-sample) comparte la lectura geométrica de coherencia direccional.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — familia alineación; marco CGH del que GWA hereda la intuición de coherencia $\to$ generalización.
- [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training]] — familia alineación vía kernel-target alignment, también restringible a last-layer; alineamiento como predictor de aprendizaje.
- [[Disparity Between Batches as a Signal for Early Stopping]] — paper comparable directo: proxy train-time barato para early stopping (GD); GWA lo usa como baseline competidor en sus tablas.
- [[Speedy Performance Estimation for Neural Architecture Search]] — paper comparable directo: TSE/TSE-EMA, mismo problema de predicción temprana barata; es el baseline del TFG.

## Otros papers interesantes a revisar

- **Directional convergence and alignment in deep learning** (Ji & Telgarsky, 2020) — base teórica directa de GWA: bajo flujo gradiente con cross-entropy en datos separables, gradiente y pesos convergen direccionalmente ($\mathbb{E}[\gamma] \to 1$). arXiv:2006.06657.
- **Estimating Training Data Influence by Tracing Gradient Descent (TracIn)** (Pruthi et al., 2020) — usa el gradiente per-sample (también restringido a last-layer por coste) para atribuir influencia y detectar etiquetas ruidosas, igual que el uso sample-level de $\gamma(x_i, \mathbf{w}_T)$. arXiv:2002.08484.
- **Prevalence of Neural Collapse during the terminal phase of deep learning training** (Papyan, Han & Donoho, 2020) — explica la geometría last-layer (features y pesos del clasificador colapsando alineados) que sustenta por qué el estimador last-layer de GWA es informativo. arXiv:2008.08186 / DOI:10.1073/pnas.2015509117.
- **Characterizing signal propagation to close the performance gap in unnormalized ResNets** / línea de proxies *zero-cost* NAS como **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah et al., 2021) — proxies train-time/at-init (SNIP, GraSP, synflow, jacov) sobre gradientes; marco comparativo natural para situar el coste y poder predictivo de GWA. arXiv:2101.08134.
- **Gradient Starvation: A Learning Proclivity in Neural Networks** (Pezeshki et al., 2021) — analiza cómo la dinámica de gradientes condiciona qué features se aprenden y la generalización; complementa la lectura de coherencia direccional con el sesgo de simplicidad que GWA observa empíricamente. arXiv:2011.09468.

## Cited By
