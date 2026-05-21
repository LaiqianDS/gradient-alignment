---
authors:
  - Stanislav Fort
  - Paweł Krzysztof Nowak
  - Stanisław Jastrzębski
  - Srini Narayanan
year: 2019
status: to-read
relevance: high
last_review: 2026-05-07
url: https://arxiv.org/abs/1901.09491
---
# Stiffness: A New Perspective on Generalization in Neural Networks

## Summary

### Contextualización

El trabajo se sitúa en el problema central de la teoría moderna del deep learning: comprender por qué las redes neuronales sobreparametrizadas, capaces de memorizar datos arbitrariamente etiquetados (Zhang et al., 2016; Arpit et al., 2017), generalizan bien a datos no vistos cuando se entrenan con descenso por gradiente. Los autores enmarcan su aportación dentro de la línea de investigación que estudia la similitud entre las salidas de la red para entradas próximas (Schoenholz et al., 2016; Novak et al., 2018) y que culmina en la teoría del Neural Tangent Kernel (NTK) de Jacot et al. (2018). El objetivo es proporcionar una herramienta operativa que permita diagnosticar cuándo la red está aprendiendo características transferibles entre ejemplos (generalización) frente a cuándo se limita a ajustarse de forma específica al conjunto de entrenamiento (overfitting o memorización).

### Aportación

La contribución principal es la introducción del concepto de **stiffness** (rigidez) como métrica para caracterizar la generalización. La idea operativa es directa: dados dos ejemplos $X_1$ y $X_2$ con etiquetas $y_1$ e $y_2$, se calcula el gradiente $\vec{g}_1 = \nabla_W \mathcal{L}(f_W(X_1), y_1)$ y se aplica un paso infinitesimal en la dirección $-\vec{g}_1$ que por construcción reduce la pérdida en $X_1$. La stiffness mide qué le ocurre simultáneamente a la pérdida sobre $X_2$. Si $\Delta\mathcal{L}_2 < 0$ los dos puntos son rígidos (positive stiffness): mejorar uno mejora también el otro y la red está aprendiendo features compartidas. Si $\Delta\mathcal{L}_2 > 0$ son anti-stiff (negative stiffness): el ejemplo $X_2$ empeora, señal de memorización local. La equivalencia matemática clave es que esta variación coincide con el alineamiento $\vec{g}_1 \cdot \vec{g}_2$ entre gradientes evaluados en ambos puntos, conectando la métrica directamente con el NTK y con el estudio del Hessiano del paisaje de pérdida.

### Metodología

Los autores formalizan dos definiciones complementarias:

- **Sign-stiffness**: $S_{\mathrm{sign}}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\mathrm{sign}(\vec{g}_1 \cdot \vec{g}_2)]$, esperanza del signo del producto de gradientes; toma valores en $[-1,1]$ y resulta más informativa para stiffness entre clases.
- **Cosine-stiffness**: $S_{\cos}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\cos(\vec{g}_1,\vec{g}_2)]$, con $\cos(\vec{g}_1,\vec{g}_2) = (\vec{g}_1/|\vec{g}_1|)\cdot(\vec{g}_2/|\vec{g}_2|)$; preferida para stiffness intra-clase.

Por construcción, la stiffness de un punto consigo mismo es 1. Distinguen tres regímenes de muestreo de pares: train-train, train-val y val-val, observando que los tres se comportan de forma muy similar, lo que refuerza que la stiffness train-val está directamente ligada a la generalización (transferencia de la mejora del entrenamiento a validación). Definen además la **matriz de class stiffness** $C(c_a, c_b) = \mathbb{E}_{X_1\in c_a, X_2\in c_b, X_1\neq X_2}[S((X_1,y_1),(X_2,y_2))]$, cuyas entradas diagonales miden generalización intra-clase y las extra-diagonales la transferencia entre clases. El resumen entre clases se obtiene como $S_{\mathrm{between\ classes}} = \frac{1}{N_c(N_c-1)}\sum_{c_1}\sum_{c_2\neq c_1} C(c_1,c_2)$.

El protocolo experimental consiste en: (1) entrenar la red durante cierto número de pasos sobre el set de train, (2) congelar pesos y, para cada modo (train-train, train-val, val-val), recorrer tuplas de imágenes calculando $\vec{g}_1, \vec{g}_2$ y los valores $\mathrm{sign}(\vec{g}_1\cdot\vec{g}_2)$ y $\cos(\vec{g}_1,\vec{g}_2)$, (3) registrar la distancia en el espacio de entrada y otras features. Usaron subconjuntos fijos (≈500 imágenes para 10 clases y ≈3000 para 100 clases) suficientes para reducir incertidumbre estadística. Las entradas se preprocesaron con media cero, varianza unitaria y normalización a la esfera unidad ($|\vec{X}|=1$); el optimizador fue Adam con tasas de aprendizaje constantes y batch por defecto de 32.

### Datasets y modelos

Los experimentos cubren cuatro datasets de visión: **MNIST** (LeCun y Cortes, 2010), **FASHION MNIST** (Xiao et al., 2017), **CIFAR-10** y **CIFAR-100** (Krizhevsky, 2009). Las arquitecturas son: una red **fully-connected** ReLU de 3 capas $X \to 500 \to 300 \to 100 \to y$; una **CNN** de 3 capas con filtros $3\times3$ y canales $16, 32, 32$ seguidos de max-pooling $2\times2$ y una capa final FC; y una **ResNet20v1** (He et al., 2015) en la implementación de Chollet et al. (2015), sin batch normalization. Adicionalmente, validan los hallazgos sobre una tarea de NLP: un modelo **BERT** (Devlin et al., 2018) fine-tuned sobre el dataset **MNLI** (Williams et al., 2017).

### Métricas

Las magnitudes seguidas son: (i) loss y accuracy en train y validación a lo largo de las épocas; (ii) elementos diagonales de la matriz de class stiffness (within-class) y suma extra-diagonal (between-classes), tanto en sign como en cosine; (iii) curvas de stiffness vs distancia en el espacio de entrada, donde la distancia se define como $\mathrm{distance}(\vec{X}_1,\vec{X}_2) = 1 - \frac{\vec{X}_1\cdot\vec{X}_2}{|\vec{X}_1||\vec{X}_2|}$, acotada en $[0,2]$; (iv) la **dynamic critical length** $\xi$, definida como la distancia umbral en la cual la stiffness intra-clase, en promedio, cruza el cero (ajuste lineal de stiffness vs distancia). Esta longitud caracteriza el tamaño típico de los parches del espacio de entrada que se mueven juntos bajo un paso de gradiente, conectando directamente con la noción de "regiones rígidas" de la función aprendida.

### Conclusiones

Los resultados empíricos articulan varias conclusiones. En primer lugar, la stiffness es una herramienta diagnóstica del overfitting: en estados iniciales del entrenamiento la stiffness intra-clase es alta y entre clases crece a medida que la red aprende features compartidas; con el inicio del overfitting (marcado en la Figura 3 con la zona naranja) tanto la stiffness within-class como between-classes regresan hacia 0, indicando que las actualizaciones de gradiente respecto a un ejemplo dejan de beneficiar incluso a los demás miembros de su misma clase. Esto sugiere que la métrica podría usarse como criterio de early stopping observable únicamente sobre el train set. En segundo lugar, la stiffness es sensible al contenido semántico: en CIFAR-100 la matriz de class stiffness exhibe estructura de coarse-grain alineada con super-clases (grupos de 5 clases semánticamente conectadas) e incluso super-super-clases (distinción living/non-living), lo que demuestra que la red captura jerarquías semánticas más allá del nivel de etiqueta usado en el entrenamiento. En tercer lugar, la dynamic critical length $\xi$ depende sistemáticamente del learning rate: tasas de aprendizaje mayores producen $\xi$ menores, es decir, funciones aprendidas localmente más maleables (lower stiffness) y más fáciles de doblar mediante actualizaciones de gradiente, incluso cuando logran accuracy comparable. Este hallazgo apunta al rol regularizador del learning rate sobre el tipo de función aprendida y no solo sobre la velocidad de convergencia. Finalmente, las observaciones sobre BERT-MNLI replican el patrón de visión, sugiriendo que stiffness es un fenómeno general. Los autores conjeturan como dirección futura su uso como parámetro guía para meta-learning y neural architecture search, ya que arquitecturas con sesgos inductivos como la localidad de las CNN se traducen en propiedades de stiffness más altas.

## Medición y pipeline

**Métrica concreta.** Se adopta la *stiffness* de Fort et al. en sus dos formulaciones complementarias sobre un conjunto de pares de ejemplos $(x_i, x_j)$ con $i \neq j$:

- Sign-stiffness: $S_{\mathrm{sign}} = \mathbb{E}_{i\neq j}\big[\mathrm{sign}(\nabla_W \ell_i \cdot \nabla_W \ell_j)\big]$, más informativa para la variante *between-class*.
- Cosine-stiffness: $S_{\cos} = \mathbb{E}_{i\neq j}\big[\cos(\nabla_W \ell_i, \nabla_W \ell_j)\big] = \mathbb{E}_{i\neq j}\big[\tfrac{\nabla_W \ell_i \cdot \nabla_W \ell_j}{\|\nabla_W \ell_i\|\,\|\nabla_W \ell_j\|}\big]$, preferida para la variante *within-class*.

Siguiendo el paper, se distinguirán ambas variantes: *within-class* (promedio restringido a pares con $y_i = y_j$, captura generalización intra-clase) y *between-class* (pares con $y_i \neq y_j$, captura transferencia entre clases).

**Entradas.** Gradientes por ejemplo $\nabla_W \ell(x_k; W)$ sobre un *probe set* fijo y disjunto del batch de entrenamiento (subconjunto del set de validación o muestra aleatoria del train). Tamaños sugeridos en línea con el paper: $N \approx 500$ ejemplos para datasets de 10 clases (MNIST, Fashion-MNIST, CIFAR-10), estratificado por clase.

**Cuándo computar.** Una medición por época durante toda la corrida; en la fase inicial (primeras 5-10 épocas), cada $K$ pasos para muestrear con mayor resolución la *ventana temprana* relevante para la predicción.

**Coste.** $N$ forward+backward por medición para obtener los gradientes y $O(N^2)$ evaluaciones de coseno/signo sobre los pares. Memoria: $N$ vectores de gradiente; si $|W|$ es grande, almacenar por capa o usar proyecciones aleatorias para reducir huella.

**Integración en el pipeline.** Pseudocódigo:

```python
probe_X, probe_y = sample_probe_set(val_set, N, stratify=True)
def stiffness_step(model, probe_X, probe_y):
    grads = []
    for x_k, y_k in zip(probe_X, probe_y):
        model.zero_grad()
        loss = criterion(model(x_k.unsqueeze(0)), y_k.unsqueeze(0))
        g = torch.autograd.grad(loss, model.parameters())
        grads.append(flatten(g))            # o per-sample via functorch/vmap
    G = torch.stack(grads)                  # (N, |W|)
    Gn = G / G.norm(dim=1, keepdim=True)
    C = Gn @ Gn.T                           # (N, N) cos-stiffness
    S_cos = C[~eye(N)].mean()
    S_sign = C.sign()[~eye(N)].mean()
    return S_cos, S_sign, C
```

Sobre $C$ se calcula adicionalmente la *class stiffness matrix* agregando por pares de clases, de donde se derivan los promedios within/between.

**Consideraciones.** La stiffness decae a medida que avanza el entrenamiento y tiende a 0 al iniciarse el overfitting; por tanto, la *ventana temprana* es crítica para la señal predictiva sobre la eficiencia. La matriz por clases aporta más información que el escalar agregado (estructura semántica, jerarquías). Con $N$ grande la memoria y el $O(N^2)$ crecen rápidamente: muestrear, usar gradientes por capa o aplicar libs de gradiente per-sample (`functorch.vmap`, `torch.func.grad`) para acelerar el bucle.

**Logging.** Por época se registran: $S_{\cos}$ y $S_{\mathrm{sign}}$ escalares (within, between, global), promedios por clase (diagonal y extra-diagonal de $C$) y, cada $K$ épocas, la matriz $N\times N$ completa como *heatmap* para análisis cualitativo posterior.

## Notes

### Uso en el TFG

- **Métrica que origina.** Es el origen de `stiffness` (familia alineación). Sobre gradientes per-sample del gradiente bruto $g_i = \nabla_W \ell(f_W(x_i), y_i)$ se computan cos-stiffness $S_{\cos}(i,j) = \cos(g_i, g_j) = \tfrac{g_i \cdot g_j}{\|g_i\|\,\|g_j\|}$ y sign-stiffness $S_{\text{sign}}(i,j) = \text{sign}(g_i \cdot g_j)$.
- **Cómo se usa.** Se agrega sobre un probe estratificado $M = 256$ distinguiendo pares within-class (mismo label) y between-class (distinto label): $\bar S_{\cos}^{\text{within}} = \mathbb{E}_{i\neq j,\, y_i=y_j}[S_{\cos}(i,j)]$ y $\bar S_{\cos}^{\text{between}} = \mathbb{E}_{i\neq j,\, y_i\neq y_j}[S_{\cos}(i,j)]$. La señal predictiva vive en la ventana temprana (5/10/25/50% de épocas): la stiffness decae hacia 0 al iniciarse el overfitting.
- **Señal.** $\bar S_{\cos}^{\text{within}}$ ↑ mejor (clases con features compartidos → SGD que transfiere mejoras intra-clase); $\bar S_{\cos}^{\text{between}} \approx 0$ esperable. Se reportan también `cos`, `sign` globales.
- **Pitfall de memoria.** El cuello es la gram $G G^\top$ con coste de memoria $M \cdot P$ floats = **12GB para $M = 256$ en ResNet-18 fp32**, bloqueante sin mitigación. Solución: chunk en $M$ acumulando filas, o reducir $P$.
- **Decisión de scope.** **Default last-layer-only** en ResNet-18 ($P_{\text{eff}} \approx 5k$ en lugar de 11.7M), consistente con `gwa`; Fort reporta stiffness por bloques, así que sigue siendo teóricamente significativo. En FC/MLP y CNN-small: full-parameter (P pequeño).
- **Sweep compartido y descarte.** Comparte el barrido per-sample $\nabla L$ con `m_coherence` ($M=1024$) y `gsnr` ($M=512$). NO se implementa el análisis "stiffness vs distancia en el input" ni la dynamic critical length $\xi$: añaden una dimensión continua (distancia 2D) fuera del scope puramente correlacional del TFG.

## Cited By
[[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]]
[[Speedy Performance Estimation for Neural Architecture Search]]
[[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]]
[[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]]
[[Disparity Between Batches as a Signal for Early Stopping]]

## Papers relacionados

- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — Misma familia alineación; m-coherence $\alpha_m$ es el agregado escalar del mismo producto $g_z\cdot g_{z'}$ entre per-sample grads y comparte el sweep $\nabla L$ con stiffness.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — Cita a stiffness; la CGH formaliza el mismo principio (la generalización viene del alineamiento entre gradientes per-sample) que aquí se mide vía within/between-class.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — Cita a stiffness; gradient confusion es el peor caso (mín. coseno) del mismo objeto coseno entre gradientes per-sample/per-batch.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — Cita a stiffness; proxy train-time de generalización con la misma decisión de last-layer-only para acotar memoria.
- [[Disparity Between Batches as a Signal for Early Stopping]] — Cita a stiffness; usa alineamiento (distancia L2) entre gradientes como señal de early stopping, mismo régimen de ventana temprana.
- [[Speedy Performance Estimation for Neural Architecture Search]] — Cita a stiffness y comparte la motivación NAS/meta-learning que los autores conjeturan como uso futuro de la métrica.
- [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training]] — La equivalencia stiffness $\leftrightarrow g_i\cdot g_j$ es exactamente el NTK empírico; este paper formaliza el kernel alignment al que stiffness apunta.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — Familia alineación/varianza; comparte el sweep per-sample $\nabla L$ con stiffness y conecta alineamiento de gradientes con generalización.

## Otros papers interesantes a revisar

- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** (Jacot, Gabriel & Hongler, 2018) — Marco NTK que fundamenta la equivalencia stiffness = $g_i\cdot g_j$; cita central del paper. arXiv:1806.07572.
- **Sensitivity and Generalization in Neural Networks: an Empirical Study** (Novak et al., 2018) — Línea directa de la que parte Fort (sensibilidad de salidas a entradas próximas); evidencia empírica de la conexión sensibilidad-generalización. arXiv:1802.08760.
- **A Closer Look at Memorization in Deep Networks** (Arpit et al., 2017) — Caracteriza memorización vs aprendizaje de patrones; complementa la interpretación de stiffness baja como memorización local. arXiv:1706.05394.
- **Gradient Descent Happens in a Tiny Subspace** (Gur-Ari, Roberts & Dyer, 2018) — El gradiente vive en un subespacio de baja dimensión dominado por el Hessiano; relevante para justificar last-layer-only y proyecciones al acotar $M\cdot P$. arXiv:1812.04754.
- **What Can Linearized Neural Networks Actually Say About Generalization?** (Ortiz-Jiménez, Moosavi-Dezfooli & Frossard, 2021) — Examina cuándo el NTK empírico predice generalización; matiza el alcance de proxies basados en alineamiento de gradientes como stiffness. arXiv:2106.06770.
