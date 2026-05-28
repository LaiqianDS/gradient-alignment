
El gradiente per-sample $g_i = \nabla_W \ell(x_i, y_i; W)$ es el gradiente de la pérdida calculado para un único ejemplo, antes de cualquier promedio sobre el minibatch. Es el objeto fundamental sobre el que se construyen casi todas las métricas modernas de alineación, coherencia y varianza, porque toda la riqueza estadística del minibatch se pierde en el momento en que reduces los $g_i$ a su media.

El gradiente del minibatch es simplemente $\hat g = \tfrac{1}{B}\sum_i g_i$, pero las $g_i$ individuales contienen mucha más información. Su distribución (covarianza, asimetría, curtosis) describe cuánto ruido hay; sus productos internos por pares describen si los ejemplos cooperan o se cancelan ([[Stiffness]], m-coherence, [[Gradient confusion]]); y su similitud con cantidades de referencia da pie a proxies de generalización como GWA (gradiente per-sample contra los pesos del clasificador) o KTA (gradientes contra el target $yy^\top$).

Un ejemplo concreto del cambio de granularidad. En un batch de 128 imágenes de CIFAR-10, el gradiente medio es un único vector en $\mathbb{R}^p$ con $p$ del orden de millones. Los 128 gradientes per-sample son una matriz de $128 \times p$, mucho más informativa. Si calculas la matriz de Gram $G_{ij} = \langle g_i, g_j\rangle$ de tamaño $128 \times 128$, ya puedes leer directamente qué ejemplos están alineados (entrada grande positiva), cuáles son ortogonales (entrada cercana a cero) y cuáles entran en conflicto (entrada negativa). Eso te lo da el dato per-sample, pero no el gradiente promedio.

En frameworks como PyTorch se materializan eficientemente con `torch.func.vmap(grad(loss_fn))`, que aplica el cálculo de gradiente de forma vectorizada por ejemplo sin tener que iterar el optimizador uno por uno. Sin esta primitiva habría que hacer un forward-backward por ejemplo (128 veces más caro), o aproximar con técnicas como "expand-and-jacobian", que tampoco escalan. La existencia de vmap en autograd moderno es lo que ha hecho viable medir alineación en tiempo de entrenamiento sin un coste prohibitivo, y por eso muchas de las métricas que aparecen en los papers de los últimos años son recientes a pesar de que conceptualmente se podían haber definido hace tiempo.

## Enlaces

- Base teórica de m-coherence sobre gradientes per-sample: [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]]
- Coherencia per-sample y su rol en generalización: [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]]
- Uso en la métrica GWA (gradiente per-sample vs pesos): [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]]
- Stiffness sobre pares per-sample: [[Stiffness - A New Perspective on Generalization in Neural Networks]]
