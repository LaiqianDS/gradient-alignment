---
authors:
  - Mahsa Forouzesh
year: 2021
status: read
relevance: high
url: https://link.springer.com/chapter/10.1007/978-3-030-86520-7_14
tfg_role:
  - metric
  - baseline
tfg_note: "Origen de `gradient_disparity` (distancia L2 media entre gradientes de batches independientes; Pearson 0.957 entre la GD train-train y la GD train-val, lo que justifica la versión train-train como proxy). Uno de los 3 papers más comparables: ellos optimizan early-stopping sin val set, el TFG sólo mide la correlación."
---

# Disparity Between Batches as a Signal for Early Stopping

## Summary

### Contextualización

El early stopping clásico continúa siendo una de las técnicas más extendidas para evitar el sobreajuste en redes neuronales profundas entrenadas con métodos iterativos como el descenso por gradiente. El protocolo habitual consiste en separar un subconjunto del dataset disponible como conjunto de validación etiquetado de manera fiable y monitorizar sobre él la pérdida o el error de clasificación durante el entrenamiento. La optimización se detiene en cuanto el rendimiento sobre validación deja de mejorar mientras el de entrenamiento sigue avanzando. Esta estrategia, sin embargo, presenta limitaciones prácticas relevantes. Por una parte, exige reservar datos que dejan de estar disponibles para entrenamiento, lo que se agrava en aplicaciones con datos escasos o costosos como el dataset MRNet de resonancias de rodilla. Por otra, cuando las etiquetas presentan ruido provocado por anotadores no expertos, tareas complejas o ataques adversariales, el conjunto de validación puede contener una fracción elevada de muestras incorrectas y dejar de funcionar como estimador insesgado del error de test. La k-fold cross-validation mitiga parcialmente estos problemas, pero a costa de un sobrecoste computacional muy elevado y de entrenar el modelo únicamente sobre una fracción $(1 - 1/k)$ del dataset en cada ronda.

### Aportación

Forouzesh y Thiran proponen la *gradient disparity* (GD) como métrica de early stopping que no requiere conjunto de validación. La GD se define como la distancia $\ell_2$ entre los vectores gradiente calculados sobre dos mini-batches independientes muestreados del propio conjunto de entrenamiento. La motivación teórica parte de un upper bound probabilístico sobre la diferencia entre los errores de clasificación de un mini-batch cuando la red ha sido actualizada con ese mismo mini-batch frente a cuando ha sido actualizada con otro distinto del mismo dataset. Los autores demuestran empíricamente que la GD es una señal fiable de overfitting, particularmente útil cuando los datos son limitados (porque permite usar todas las muestras para entrenamiento) y cuando contienen etiquetas ruidosas (porque señala overfitting mejor que la pérdida de validación contaminada). Observan además una correlación positiva fuerte entre la GD y el error de generalización entre los conjuntos de entrenamiento y test, junto con una sensibilidad informativa al nivel de label noise.

### Metodología

Sean $S_1$ y $S_2$ dos mini-batches de tamaños $m_1$ y $m_2$ con vectores gradiente $g_1$ y $g_2$ respectivamente. Tras una iteración SGD que actualiza el vector de parámetros $w$ a $w_1 = w - \gamma \nabla L_{S_1}(h_w)$, los autores definen la *generalization penalty* $\mathcal{R}_2 = L_{S_2}(h_{w_1}) - L_{S_2}(h_{w_2})$, que mide cuánto se penaliza la generalización al haber elegido $S_1$ en lugar de $S_2$. Adaptando el marco PAC-Bayesiano y modelando los predictores como gaussianos $\mathcal{N}(w_i, \sigma^2 I)$, derivan el Teorema 1: el bound sobre $\mathbb{E}[\mathcal{R}_1] + \mathbb{E}[\mathcal{R}_2]$ depende de divergencias KL que se reducen a $\mathrm{KL}(Q_1 \| Q_2) = \tfrac{1}{2}\tfrac{\gamma^2}{\sigma^2}\|g_1 - g_2\|_2^2$. La gradient disparity entre dos batches queda entonces definida como $\mathcal{D}_{i,j} = \|g_i - g_j\|_2$ (Eq. 6) y, para mejorar la estimación, se promedia sobre $s$ batches con $s \ll B$ (típicamente $s = 5$). El algoritmo de early stopping consiste en monitorizar la GD y detener la optimización según uno de dos thresholds: $t_1$ = la GD ha aumentado durante 5 epochs **posiblemente no consecutivos** desde el inicio del entrenamiento, o $t_2$ = 5 epochs consecutivos (patience). Los números de las Tablas 3 y 4 que se citan abajo corresponden a $t_1$, y la Tabla 1 reporta ambos. Para corregir el desajuste de escala entre etapas tempranas (donde tanto $g_i$ como $L_{S_i}$ son grandes) y tardías del entrenamiento, los valores de loss se reescalan dentro de cada batch antes de calcular la GD media. Los autores extienden el análisis más allá de SGD vainilla a SGD con momento, Adagrad, Adadelta y Adam, observando que la GD aparece igualmente en el término KL multiplicada por factores que dependen de promedios decadentes de gradientes pasados.

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas

Las magnitudes reportadas son test loss, test accuracy, test AUC (en MRNet), training accuracy, valores de gradient disparity, coeficiente de correlación de Pearson entre GD y error de test, y sensibilidad de cada método al threshold de early stopping según la Eq. 14 del paper extendido. Los resultados clave muestran que en MRNet la GD mejora más de un 1% el AUC de test frente a 5-fold CV **en promedio sobre las tres tareas** (Tabla 1; en Meniscal el delta individual es 0.55); en CIFAR-10/VGG-13 con datos limitados alcanza 36.96% de test accuracy frente a 35.98% de 5-fold CV; en MNIST/AlexNet limitado obtiene 79.12% frente a 62.62% de 5-fold CV (Tabla 3); con 50% label noise en CIFAR-100/ResNet-18 alcanza 3.68% top-1 frente a 1.59% de 10-fold CV, y en MNIST/AlexNet noisy logra 97.32% frente a 97.28% (Tabla 4). La sensibilidad al threshold también favorece a la GD: 0.916 (accuracy) y 0.886 (loss) frente a 1.613 y 1.019 de CV (Tabla 5). El $\rho = 0.957$ sobre 220 configuraciones experimentales (Fig. 2) es la correlación entre la GD **train-train** y la GD **train-val**, y es la evidencia que justifica medir la GD puramente sobre el set de entrenamiento como proxy. No es la correlación entre GD y error de test, que el paper afirma solo cualitativamente ("strong positive correlation with the test error", p. 219).

### Conclusiones

La gradient disparity supera al early stopping basado en validation set, incluyendo k-fold CV y la variante $k^+$-fold CV, cuando los datos son limitados o ruidosos. En el escenario con datos abundantes y limpios ambas técnicas son comparables, pero la GD evita el coste de reservar un conjunto de validación. La GD refleja además fielmente el nivel de label noise incluso en etapas tempranas del entrenamiento, donde el gap de generalización todavía no lo hace. Decrece con el tamaño del training set, aumenta con el batch size (correlacionado con el test error en este caso, contraintuitivamente) y disminuye con el ancho de la red (esto último para la GD **normalizada por el número de parámetros**, matiz explícito del paper). Una limitación del método es que pertenece a la familia de métricas basadas en similitud entre vectores gradiente y puede dejar de ser informativa cuando los gradientes son muy pequeños. Los autores observan empíricamente, sin embargo, que el momento en que la GD detecta overfitting precede al colapso de la pérdida de entrenamiento, por lo que esta limitación no compromete su utilidad práctica como criterio de parada. Como dirección futura mencionan el escenario de epoch-wise double-descent.

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** Se adopta la *gradient disparity* de Forouzesh y Thiran sobre el gradiente bruto $\nabla L$ del modelo completo evaluado en los pesos $w$ del paso actual. Para dos mini-batches $S_i, S_j$ independientes muestreados del training set, con gradientes $g_i = \nabla L_{S_i}(h_w)$ y $g_j = \nabla L_{S_j}(h_w)$, la distancia pareada es

$$\mathcal{D}_{i,j} = \|g_i - g_j\|_2,$$

y el estimador escalar promedia las $\binom{s}{2}$ distancias sobre $s$ batches con $s \ll B$ (el paper fija $s = 5$, equivalente a 10 pares): $\overline{\mathcal{D}} = \binom{s}{2}^{-1} \sum_{i<j} \mathcal{D}_{i,j}$. Conviene insistir en que se trata de una **magnitud $\ell_2$ sin normalizar, no una similitud coseno**: la cota PAC-Bayes del paper depende explícitamente de $\|g_1 - g_2\|_2^2$ a través de $\mathrm{KL}(Q_1\|Q_2) = \tfrac{1}{2}\tfrac{\gamma^2}{\sigma^2}\|g_1-g_2\|_2^2$, y por tanto cualquier normalización por $\|g\|$ rompe la conexión teórica.

**Entradas.** Los $s$ vectores gradiente $g_i = \nabla L_{S_i}(h_w)$ se calculan sobre mini-batches *independientes* extraídos del propio training set sobre los **mismos pesos $w$ congelados** del paso de medición. No se requiere conjunto de validación. El gradiente del modelo completo se concatena en un único vector para el cálculo escalar; opcionalmente puede descomponerse por capa para diagnóstico estructural sin afectar al criterio agregado.

**Granularidad temporal y estructural.** La cadencia natural es **una medición por época**, suficiente para reproducir el protocolo del paper (patience $p = 5$ epochs sobre la serie de GD). Frecuencias mayores (cada $N$ steps) son posibles pero multiplican el coste; la cadencia debe fijarse desde el inicio para que las series sean comparables entre runs. A nivel estructural el reporte es **global por defecto**, con descomposición **opcional por capa** para análisis comparativo entre FC, CNN y ResNet.

**Coste.** Cada medición requiere $s$ pares de forward+backward extra (≈ $5\times$ el coste de un step de training con $s=5$), sin llamar a `optimizer.step()`. La memoria adicional es del orden del tamaño del vector de parámetros, suficiente para almacenar un único gradiente porque las distancias se acumulan en streaming. El sobrecoste por época es por tanto barato y constante.

**Truco práctico.** Si el dataloader del training loop ya entrega mini-batches del mismo tamaño $m$, basta con **consumir $s$ batches sucesivos del propio iterador** antes del siguiente `optimizer.step()`: esto evita instanciar un sampler separado y garantiza la independencia muestral siempre que los batches no se solapen con el step de actualización.

**Integración en el pipeline.** En el punto de medición se congela el optimizador, se muestrean $s$ batches frescos, se computa $g_i$ por backward sin update y se acumulan las $\binom{s}{2}$ distancias $\ell_2$. Pseudocódigo:

```python
def gradient_disparity(model, loss_fn, sampler, s=5):
    grads = []
    for _ in range(s):
        xb, yb = next(sampler)              # mini-batch independiente
        model.zero_grad()
        loss = loss_fn(model(xb), yb)       # rescalar loss por batch si procede
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(g.detach().clone())
    dists = [torch.norm(grads[i] - grads[j])
             for i in range(s) for j in range(i+1, s)]
    return torch.stack(dists).mean().item()
```

**Logging.** Por época se registra un escalar `gd/scalar` (la GD global media sobre los $\binom{s}{2}$ pares) junto con `epoch`, `step`, `train_loss` y `test_loss` para la correlación posterior con eficiencia y generalización. Si se activa la descomposición estructural, se añade una serie por capa bajo el patrón `gd/per_layer/{name}`, manteniendo la convención de claves jerárquicas usada en el resto del registry. Estos logs alimentan el cálculo de epochs-to-threshold, AUC de test loss y best test loss en el análisis posterior.

**Interpretación de la señal.** A diferencia de las métricas coseno del TFG, en `gd/scalar` (y en su desglose `gd/per_layer/{name}`) la convención es **cuanto más alto, peor**: una disparidad $\overline{\mathcal{D}}$ creciente indica que los gradientes de mini-batches independientes se separan en norma $\ell_2$. El paper lo interpreta como señal de sobreajuste y lo emplea como criterio de parada precisamente porque crece monótonamente en la cola tardía del entrenamiento. La lectura, sin embargo, no es monótona en el régimen global: durante la fase de fit conviven una norma de gradiente grande y batches coherentes, de modo que $\overline{\mathcal{D}}$ tiende a bajar. El cambio de pendiente que marca el inicio del overfitting solo se identifica de forma fiable **leyendo `gd/scalar` junto a `train_loss`**, esperando a que el descenso del entrenamiento se acompañe de un repunte sostenido de la disparidad. Dos matices son importantes para evitar leer cifras absolutas: la magnitud de $\overline{\mathcal{D}}$ **crece con el batch size** aun cuando la correlación con el test error se mantiene, por lo que los valores no son comparables cross-batch-size sin renormalizar. Además, aunque la cota PAC-Bayes prohíbe dividir por $\|g\|$ —`gd/scalar` debe permanecer sin normalizar—, sí conviene registrar en paralelo $\overline{\|g\|}$ para detectar la contracción tardía del gradiente que aplana la señal sin que ello implique mejora de generalización. Operativamente, el objetivo del diseño es que $\overline{\mathcal{D}}$ se mantenga **lo más baja posible** durante la fase de fit y que su repunte temprano funcione como bandera de generalización futura, leída siempre en contexto con la dinámica de pérdida y norma.

**Avisos.** Tres cuestiones requieren atención. Primero, el **tamaño del mini-batch afecta directamente la magnitud** de $\mathcal{D}_{i,j}$ porque la varianza del estimador gradiente decrece como $1/m$, de modo que la GD no es comparable entre runs con batch size distinto sin renormalizar. El paper observa además que la GD *aumenta* con el batch size manteniendo correlación con el test error, comportamiento contraintuitivo que conviene reportar explícitamente. Son dos ejes distintos: el decaimiento $1/m$ es la varianza *intra-estimador* a batch size fijo, mientras que el aumento con el batch size compara la GD *entre* distintos tamaños de batch. Segundo, aunque la cota teórica prohíbe normalizar por $\|g\|$, sí es útil **registrar en paralelo $\|g\|$ media** (o $\mathcal{D}_{i,j}/\overline{\|g\|}$ como diagnóstico secundario) para detectar la contracción de gradientes al final del entrenamiento, momento en que la señal absoluta puede saturar. Tercero, la **sensibilidad al threshold** de early stopping (Eq. 14 del paper) es 0.916/0.886 para la GD frente a 1.613/1.019 para CV: la métrica es robusta a la elección de patience. La comparación entre optimizadores, sin embargo, requiere precaución porque en Adam, momentum, Adagrad o Adadelta el término KL recoge factores adicionales que dependen de promedios decadentes de gradientes pasados y la escala absoluta de la GD deja de ser directamente comparable con la de SGD.

## Notes

### Uso en el TFG

- **Métrica que origina**: `gradient_disparity` (familia alineación, `METRIC_REGISTRY`). Es la fuente directa de esta métrica en el registro cerrado de 10.
- **Cómo se usa**: distancia $\ell_2$ media entre los gradientes de $s$ mini-batches *independientes* sobre el gradiente bruto $\nabla L$, $\mathcal{D}_{i,j} = \|g_i - g_j\|_2$ con $\overline{\mathcal{D}} = \binom{s}{2}^{-1}\sum_{i<j}\mathcal{D}_{i,j}$ y $s = 5$ (10 pares, según el paper). Se mide en las ventanas tempranas (5/10/25/50% de épocas) y se correlaciona con generalización/eficiencia; **no** se usa el criterio de early stopping con patience del paper.
- **Señal**: $\downarrow$ mejor (disparidad baja $\Rightarrow$ batches coherentes). El paper reporta Pearson $\rho = 0.957$ entre la GD train-train y la GD train-val sobre 220 configuraciones (Fig. 2), lo que legitima la versión train-train como proxy; su relación con el test error se afirma cualitativamente.
- **Pitfalls/decisiones clave**: es una **magnitud $\ell_2$, NO coseno** $\Rightarrow$ **no normalizar**: la cota PAC-Bayes del paper depende de $\|g_1 - g_2\|_2^2$ sin normalizar (vía $\mathrm{KL}(Q_1\|Q_2) = \tfrac{1}{2}\tfrac{\gamma^2}{\sigma^2}\|g_1-g_2\|_2^2$); por ello **contrae cuando $\|g\|\to 0$** tarde en el entrenamiento y su escala absoluta no es comparable entre optimizadores (en Adam/momentum el término KL recoge factores de promedios decadentes). Los $s$ batches deben ser **independientes** (muestreo sin solape con el siguiente train step).
- **Implementación v1**: la métrica divide su propio probe en $s = 5$ sub-batches disjuntos sobre pesos congelados y emite solo `gd/scalar` (la versión per-layer y el sweep compartido K=10 con otras métricas quedan para v2; hoy no hay caché de sweeps). El re-escalado de loss por batch que el paper aplica se omite deliberadamente (se mide en ventanas fijas, no como criterio de parada).
- **NOTA — paper comparable (delta a argumentar)**: es **uno de los 3 papers más comparables** al TFG (junto a Ru et al. 2021 y Hölzl 2025) por atacar el mismo problema —predicción temprana y barata de generalización/early-stopping. La intro debe argumentar el **delta explícito**: ellos *optimizan* un criterio de parada single-metric sin val set; el TFG es **correlacional**, no optimiza, evalúa 10 métricas + baseline en un sweep controlado (SGD/Adam × FC/CNN/ResNet × MNIST/CIFAR-10/(ImageNet)) y sitúa GD como **una métrica más** dentro de un estudio comparativo de proxies, midiendo su poder predictivo relativo en ventanas tempranas frente a alternativas de alineación y varianza.

## Papers relacionados

- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — mismo problema (proxy train-time de generalización); compara explícitamente contra Gradient Disparity como baseline. Uno de los 3 comparables.
- [[Speedy Performance Estimation for Neural Architecture Search]] — mismo problema (predicción temprana barata sin val set); TSE-EMA es el baseline del TFG. Uno de los 3 comparables.
- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — familia alineación: similitud entre gradientes per-sample (coseno/signo) como proxy de generalización; alternativa normalizada a la distancia $\ell_2$ de GD.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — familia alineación; gradient confusion sobre pares de gradientes de batch, comparte el mismo *batch-grad sweep* que GD.
- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — familia alineación; m-coherence mide alineación de gradientes como señal de generalización a lo largo del entrenamiento.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — fundamento conceptual de la familia alineación (CGH); racionaliza por qué la coherencia/disparidad entre gradientes predice generalización.

## Otros papers interesantes a revisar

- **Fantastic Generalization Measures and Where to Find Them** (Jiang et al., 2019) — estudio empírico a gran escala de medidas de generalización y su correlación causal; marco de referencia para situar GD frente a otros proxies. arXiv:1912.02178
- **Predicting Neural Network Accuracy from Weights** (Unterthiner et al., 2020) — predicción de accuracy desde estadísticos baratos del modelo; comparable como proxy temprano sin val set. arXiv:2002.11448
- **Robust Early-Learning: Hindering the Memorization of Noisy Labels** (Xia et al., 2021) — early stopping/learning bajo label noise, escenario central de Forouzesh & Thiran. ICLR 2021 (OpenReview `Eql5b1_hTE4`)
- **A PAC-Bayesian Approach to Spectrally-Normalized Margin Bounds for Neural Networks** (Neyshabur et al., 2018) — base PAC-Bayes de la que parte la derivación de la GD; útil para contextualizar la cota teórica. arXiv:1707.09564
