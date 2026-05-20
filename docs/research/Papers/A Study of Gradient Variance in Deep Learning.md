---
authors:
  - Fartash Faghri
  - David Duvenaud
  - David J. Fleet
  - Jimmy Ba
year: 2020
status: read
relevance: high
last_review: 2026-05-07
url: https://arxiv.org/pdf/2007.04532
---

# A Study of Gradient Variance in Deep Learning

## Summary

### Contextualización

El paper se sitúa en el problema clásico de la optimización estocástica en deep learning: cuando se entrena un modelo con mini-batch SGD, el gradiente estocástico es un estimador insesgado del gradiente medio sobre el conjunto de entrenamiento, y la calidad de este estimador (cuantificada por su varianza) condiciona tanto la velocidad de convergencia como la estabilidad del entrenamiento. La literatura previa (Bottou et al. 2018; Robbins y Monro 1951) demuestra cotas de convergencia que mejoran a medida que la varianza del gradiente disminuye, y predicen una mejora lineal con el tamaño del mini-batch. Sin embargo, en la práctica, trabajos como Shallue et al. (2018) o Zhang et al. (2019) observan **rendimientos decrecientes** (*diminishing returns*) al aumentar el tamaño del mini-batch: a partir de cierto umbral, doblar el batch ya no acelera el entrenamiento. La hipótesis habitual atribuye este fenómeno a los beneficios del ruido para encontrar *flat minima* y mejorar la generalización (Goodfellow y Vinyals 2015; Keskar et al. 2017), pero esto no explica por qué los rendimientos decrecientes también aparecen en la *training loss*. Faghri et al. argumentan que la varianza del gradiente como estadístico está infraestudiada empíricamente en deep learning real, y proponen estudiarla sistemáticamente. El hueco que llena el paper es doble: (i) caracterizar empíricamente la distribución de gradientes durante el entrenamiento de redes profundas y (ii) proponer un estadístico mejor que la varianza cruda para correlacionar con la dificultad de optimización.

### Aportación

Las contribuciones principales son cuatro:

1. **Vista distribucional del gradiente**: en lugar de tratar el mini-batch gradient solo como estimador puntual de la media, lo modelan como muestreo de una distribución potencialmente estructurada (multimodal, clusterizada).
2. **Gradient Clustering (GC)**: un método de muestreo estratificado que prueba teóricamente reduce la varianza del estimador del gradiente medio si los datos se agrupan en clusters ponderados en el espacio de gradientes (Proposición 3.1).
3. **Implementación eficiente de GC** mediante aproximaciones rank-1 por capa, evitando calcular gradientes individuales y manteniendo el coste computacional dentro de un factor moderado del back-prop.
4. **Normalized variance** como estadístico alternativo: un escalar que correlaciona mejor con la velocidad de convergencia que la varianza absoluta y es comparable entre problemas de distinta escala.

Empíricamente, el hallazgo más sorprendente es que, **contrariamente al supuesto común**, la varianza del gradiente *aumenta* durante el entrenamiento en CIFAR-10 e ImageNet, y que learning rates más pequeños coinciden con varianza más alta.

### Metodología

Sea un conjunto de entrenamiento $\{x_i\}_{i=1}^N$ y los gradientes por muestra $g_i = \frac{\partial}{\partial \theta}\ell(x_i; \theta)$. Para una partición en $K$ subconjuntos de tamaño $N_k$, el estimador de gradiente con muestreo estratificado es:

$$\hat{g}(a) = \frac{1}{N}\sum_{k=1}^{K} N_k \, g_{j,k}, \quad j \sim \mathbb{U}[1, N_k]$$

La **Proposición 3.1** demuestra que este estimador es insesgado ($\mathbb{E}[\hat{g}] = g$) y que su varianza es $\mathbb{V}[\hat{g}] = N^{-2}\sum_{k=1}^{K} N_k^2 \mathbb{V}[g_{j,k}]$. Minimizar esta expresión equivale a un problema de **clustering ponderado**:

$$\min_{C, a} \sum_{k=1}^{K}\sum_{i=1}^{N} N_k \, \|C_k - g_i\|^2 \, \mathbb{I}(a_i = k)$$

donde $C_k$ es el centro del cluster $k$. El factor $N_k$ extra (frente a K-Means estándar) penaliza clusters grandes con datos dispersos. El algoritmo alterna entre un paso de **Asignación** ($\mathcal{A}$: $a_i = \arg\min_k N_k \|C_k - g_i\|^2$) y un paso de **Update** ($\mathcal{U}$: $C_k = \frac{1}{N_k}\sum_i g_i \mathbb{I}(a_i=k)$), de forma análoga a Lloyd's algorithm pero con asignaciones duras.

El reto computacional es que materializar gradientes por muestra es prohibitivo. La solución es una **aproximación rank-1** por capa: para una capa fully-connected con $\theta \in \mathbb{R}^{I \times O}$, el gradiente por muestra factoriza como $g = AD^\top$ con $A \in \mathbb{R}^{I \times N}$ (activaciones de entrada) y $D \in \mathbb{R}^{O \times N}$ (gradientes respecto a las salidas). Cada centro de cluster se aproxima como $C_k = c_k d_k^\top$ con $c_k \in \mathbb{R}^I$, $d_k \in \mathbb{R}^O$. Bajo el supuesto $\mathbb{E}[A_i D_i] = \mathbb{E}[A_i]\mathbb{E}[D_i]$ (similar a K-FAC, Martens y Grosse 2015), las actualizaciones cerradas son $c_k = \frac{1}{N_k}\sum_i A_i \mathbb{I}(a_i=k)$ y $d_k = \frac{1}{N_k}\sum_i D_i \mathbb{I}(a_i=k)$. La extensión a capas convolucionales se desarrolla en el apéndice B.1. El paso $\mathcal{A}$ se ejecuta cada pocas épocas; $\mathcal{U}$ puede actualizarse online con mini-batches. El overhead total es a lo sumo $2\times$ el coste del back-prop estándar.

### Datasets y modelos

Los experimentos cubren tanto benchmarks de visión como un modelo teórico:

- **MNIST** (LeCun et al. 1998) con un MLP de tres capas fully-connected ($784 \to 1024 \to 1024 \to 10$), ReLU, sin dropout, learning rate $0.02$, weight decay $5\times10^{-4}$, momentum $0.5$.
- **CIFAR-10** (Krizhevsky et al. 2009) con **ResNet8** (sin batch normalization), learning rate $0.01$, weight decay $5\times10^{-4}$, momentum $0.9$, $80\,000$ iteraciones, decay del LR en las iteraciones $40\,000$ y $60\,000$ por factor $0.1$.
- **CIFAR-100** con **ResNet32**, learning rate inicial $0.1$, resto de hiperparámetros como CIFAR-10.
- **ImageNet** (Deng et al. 2009) con **ResNet18**, learning rate $0.1$, weight decay $1\times10^{-4}$, momentum $0.9$, schedule de decay similar a CIFAR-10.
- **Random Features (RF) models** (Rahimi y Recht 2007) con activación ReLU, dimensión oculta fija $h_s = 1000$ y número de muestras de entrenamiento variable de forma que el ratio de sobreparametrización $h_s/N$ recorre $h_s/N \in [0.1, 10]$. El dataset sintético $\{(x_i, y_i)\} \in \mathbb{R}^I \times \{\pm 1\}$ se genera por un modelo "teacher" con pesos $\theta_1 \in \mathbb{R}^{I \times h_t}$, $\theta_2 \in \mathbb{R}^{h_t \times 1}$ y bias $b$, entrenando con pérdida cross-entropy y mini-batch de tamaño 10.

Los baselines de comparación son: SG-B (mini-batch SGD), SG-2B (mini-batch del doble de tamaño, que reduce varianza por factor 2 a coste $2\times$), **SVRG** (Johnson y Zhang 2013) y **GC** (el método propuesto). El tamaño de mini-batch es $B=128$ en todos los benchmarks de imagen.

### Métricas

Definen dos métricas centrales sobre snapshots fijos del modelo, calculando estadísticos a partir de decenas de mini-batches muestreados:

- **Average Variance**: promedio sobre las coordenadas del parámetro de la varianza del estimador del gradiente medio. Equivale a la traza normalizada de la matriz de covarianza.
- **Normalized Variance**: $\mathbb{V}[g] / \mathbb{E}[g]^2$, donde el cociente se interpreta como el inverso de un *signal-to-noise ratio* (SNR). Si la varianza normalizada supera 1, el ruido domina sobre la señal del gradiente.

También reportan **training loss**, **accuracy** y la varianza máxima sobre las últimas iteraciones (para los RF models). Las curvas se reportan frente a número de iteraciones, no wall-clock time.

### Conclusiones

Los hallazgos principales son:

1. **La varianza absoluta no es comparable entre problemas** (CIFAR-10 alcanza $\sim 10^{-4}$, ImageNet $< 10^{-6}$), pero la varianza normalizada sí: en MNIST decrece monotónicamente, en CIFAR-10 e ImageNet *crece* durante el entrenamiento, especialmente tras los drops del learning rate.
2. **La varianza normalizada correlaciona mejor con la velocidad de convergencia** que la varianza cruda, confirmando su utilidad como estadístico diagnóstico.
3. **Strong Growth Condition** (Schmidt y Le Roux 2013) parece cumplirse en MNIST (varianza tiende a cero, $\sim 10^{-8}$), explicando por qué SVRG funciona bien ahí pero falla en deep learning real.
4. En CIFAR-10, **GC reduce la varianza** especialmente cuando hay duplicados o etiquetas corruptas (label corruption del 10%), respaldando la motivación original; sin embargo, en ImageNet GC se solapa con SG-B, sugiriendo que la distribución de gradientes ahí carece de estructura clusterizada explotable.
5. **Sobreparametrización**: en RF models con $h_s/N$ alto, todos los métodos tienen varianza similar y baja, y GC pierde su ventaja. La ganancia de GC sobre SVRG es robusta, pero SVRG colapsa en régimen sobreparametrizado.
6. **Learning rates pequeños coinciden con varianza más alta**, lo cual contradice asunciones simplistas y sugiere que la dependencia del ruido sobre los parámetros actuales es no trivial.
7. La caída inmediata de la loss tras un *learning rate drop* no se explica únicamente por reducción de varianza.

Los autores reconocen como limitaciones que algunas contribuciones son empíricas y carecen de respaldo teórico cerrado, que GC no siempre mejora la optimización pese a reducir la varianza, que en ImageNet la estructura clusterizada parece ausente o residir en subespacios no capturados por la aproximación rank-1, y que **explotar la varianza para mejorar la optimización sigue siendo un problema abierto**. Las implicaciones del trabajo apuntan a que el diseño de optimización *distribution-aware* podría beneficiarse de modelar la distribución completa del gradiente y no solo su media, y que la varianza normalizada es una métrica útil como estadístico diagnóstico de la dificultad de optimización.

## Medición y pipeline

**Métrica concreta.** Se adopta la *normalized gradient variance* (NGV) de Faghri et al., definida coordenada a coordenada como el cociente entre la varianza del estimador del gradiente medio y el cuadrado de la media, $\text{NGV} = \mathbb{V}[g] / \mathbb{E}[g]^2$, interpretado como el inverso de un signal-to-noise ratio. En la práctica se reporta el escalar agregado $\text{NGV} = \text{tr}(\text{Cov}(g)) / \|\mathbb{E}[g]\|^2$, equivalente a la traza normalizada de la matriz de covarianza dividida por la norma cuadrada del gradiente medio. Es este estadístico normalizado, y no la varianza absoluta, el que el paper destaca como predictor de la velocidad de progreso de SGD entre problemas de distinta escala.

**Entradas.** Para un peso $w$ fijo, se muestrean $K$ mini-batches independientes y se calculan los gradientes $g_1, \dots, g_K$ con el mismo tamaño de batch usado en entrenamiento. El paper computa el estadístico **por capa** y agrega después, lo que permite localizar la fuente de ruido; aquí se loguea NGV global y NGV por capa.

**Cuándo computar.** La medición es costosa, por lo que se recomienda hacerla cada época (o cada $K$ pasos en regímenes largos) sobre un snapshot fijo $w_t$, no en cada iteración de entrenamiento.

**Coste.** $K$ pasadas backward adicionales sin actualización de pesos; en memoria, basta con acumular $\sum g_i$ y $\sum \|g_i\|^2$ de forma streaming, evitando almacenar los $K$ vectores completos.

**Integración pipeline (pseudocódigo).**

```
freeze(w_t)
S, Q = 0, 0.0
for k in 1..K:
    x, y = sample_independent_minibatch()
    g_k  = backward(loss(model(x), y))   # sin optimizer.step()
    S   += g_k
    Q   += ||g_k||^2
mu       = S / K
var_tr   = Q / K - ||mu||^2              # tr(Cov(g))
ngv      = var_tr / (||mu||^2 + eps)
log(ngv)
```

**Consideraciones.** Faghri et al. observan contraintuitivamente que la NGV **no decrece monotónicamente**: en CIFAR-10 e ImageNet crece durante el entrenamiento, sobre todo tras los *learning rate drops*, por lo que la ventana de medición condiciona la interpretación. La NGV por capa puede divergir del valor global cuando ciertas capas dominan la covarianza; conviene loguear ambos. La NGV está estrechamente emparentada con el *gradient noise scale* de McCandlish et al., que reescala una cantidad análoga para predecir el batch size crítico.

**Logging.** Por época: NGV global, NGV por capa y, opcionalmente, un histograma de las normas $\|g_i\|$ para diagnosticar colas pesadas en la distribución de gradientes.

## Notes
