---
authors:
  - Jinlong Liu
  - Yunzhi Bai
  - Guoqing Jiang
  - Ting Chen
  - Huayan Wang
year: 2020
status: to-read
relevance: high
url: https://arxiv.org/abs/2001.07384
last_review: 2026-05-07
tfg_role:
  - metric
  - theory
---

# Understanding Why Neural Networks Generalize Well Through GSNR of Parameters

## Summary

### Contextualización

El trabajo aborda una de las paradojas centrales de la teoría moderna del deep learning: las redes neuronales profundas contienen habitualmente muchos más parámetros entrenables que muestras de entrenamiento y, según las teorías clásicas de la generalización basadas en dimensión VC o complejidad de Rademacher, deberían exhibir un *gap* de generalización elevado, esto es, una diferencia grande entre la pérdida en entrenamiento y la pérdida en test. La práctica contradice esta predicción de forma sistemática, ya que las DNN logran *gaps* notablemente pequeños incluso en régimen muy sobreparametrizado. Trabajos previos han atribuido este fenómeno a propiedades de la optimización por descenso de gradiente (Hardt et al., 2015; Zhang et al., 2016) o a la geometría del paisaje de pérdida y la curvatura de los mínimos hallados (Keskar et al., 2016; Dinh et al., 2017), pero el mecanismo concreto que vincula la dinámica del entrenamiento con la capacidad final de generalización seguía sin caracterizarse de forma cuantitativa. Liu et al. (ICLR 2020) cubren ese hueco proponiendo un puente analítico entre la estadística de los gradientes durante el entrenamiento y el *gap* de generalización resultante.

### Aportación

La contribución central es la introducción del **Gradient Signal-to-Noise Ratio (GSNR)** como métrica por parámetro que conecta de forma cuantitativa la dinámica del descenso de gradiente con la generalización de la red. El GSNR de un parámetro $\theta_j$ se define como el cociente entre el cuadrado de la media de su gradiente sobre la distribución de datos y la varianza de dicho gradiente sobre la misma distribución, es decir, $r(\theta_j) = \tilde{g}^2(\theta_j) / \rho^2(\theta_j)$, donde $\tilde{g}(\theta_j) = \mathbb{E}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$ y $\rho^2(\theta_j) = \mathrm{Var}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$. La intuición es directa: un GSNR alto sobre un parámetro indica que la media de los gradientes per-sample domina a su varianza, lo que ocurre cuando la mayoría de las muestras coincide en la dirección óptima de actualización y, por tanto, el parámetro está capturando una característica compartida y transferible entre ejemplos; un GSNR bajo señala lo contrario, gradientes dispersos que tiran del parámetro en direcciones contradictorias y que probablemente se ajustan a peculiaridades de muestras concretas. Los autores demuestran que mantener un GSNR elevado durante el entrenamiento implica un *gap* de generalización pequeño, y prueban analíticamente que la dinámica del descenso de gradiente en DNN, a diferencia de la de modelos lineales como regresión logística o SVM, produce naturalmente un GSNR elevado mediante un proceso vinculado al aprendizaje de características (*feature learning*).

### Metodología

El núcleo teórico es la derivación del **One-Step Generalization Ratio (OSGR)**, definido como $R(\mathcal{Z}, n) = \mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$, es decir, la razón entre la mejora esperada de la pérdida en test y la mejora esperada de la pérdida en entrenamiento tras un único paso de descenso de gradiente. Bajo la *non-overfitting limit approximation* (Asunción 2.3.1, que postula que $g_D(\theta)$ y $g_{D'}(\theta)$ siguen la misma distribución) y considerando el límite de tasa de aprendizaje pequeña ($\lambda \to 0$), los autores derivan una expresión cerrada que enlaza el OSGR con los GSNR de los parámetros: $R(\mathcal{Z}, n) = 1 - \tfrac{1}{n}\sum_j W_j \cdot \tfrac{1}{r_j + 1/n}$, donde $W_j$ es un peso proporcional a la contribución del parámetro $j$ a la disminución de la pérdida de entrenamiento y se cumple $\sum_j W_j = 1$. La expresión deja transparente que, cuando los $r_j$ son grandes, el OSGR tiende a 1 (el test loss decrece prácticamente al mismo ritmo que el train loss y el *gap* es pequeño), mientras que valores pequeños de $r_j$ empujan el OSGR hacia cero o incluso lo hacen negativo, generando *gaps* grandes. El paso clave en la derivación consiste en escribir $g_D(\theta) = \tilde{g}(\theta) + \epsilon$ y $g_{D'}(\theta) = \tilde{g}(\theta) + \epsilon'$ con $\epsilon$ y $\epsilon'$ ruidos independientes de media cero y varianza $\sigma^2(\theta) = \rho^2(\theta)/n$, y en aplicar la aproximación lineal $\Delta L \approx -\Delta\theta \cdot \partial L/\partial\theta$. La sección 3 complementa la derivación con un análisis de la dinámica de capas plenamente conectadas (ecuaciones 27 a 34, Apéndice B) que muestra cómo el primer término del cambio en un paso de $g_D^{(l)}$ tiene signo opuesto a los pesos $W^{(l)}_{s,c}$, lo que genera un proceso de retroalimentación positiva: los pesos cuyo signo es contrario al de su gradiente medio tienden a aumentar su valor absoluto, incrementando el numerador del GSNR y, con él, la coherencia de las actualizaciones a lo largo del entrenamiento.

### Datasets y modelos

Los experimentos se llevan a cabo principalmente sobre **MNIST** con una CNN simple de dos bloques Conv-ReLU-MaxPooling más dos capas plenamente conectadas (Tabla 2, $p \in \{6, 8, 10, 12, 14, 16, 18, 20\}$ canales) y sobre **CIFAR-10** con una red más profunda con Batch Normalization (cuatro bloques Conv-BN-ReLU-Conv-BN-ReLU-MaxPooling más tres FC, Tabla 3). Adicionalmente se utiliza **ResNet18** para comparar el GSNR con etiquetas reales frente al obtenido con etiquetas aleatorizadas, así como un *toy dataset* sintético generado por $y = x_0 x_1 + \epsilon$ con un MLP de dos capas (dos entradas, $N$ neuronas ocultas y una salida) que permite aislar el papel del *feature learning*. Las configuraciones experimentales varían el tamaño del conjunto de entrenamiento ($n \in \{1000, 2000, 4000, 6000, 8000, 10000, 15000\}$ en MNIST), la probabilidad de ruido en las etiquetas ($p_{random} \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$) y la anchura de la red, todo ello entrenado con descenso de gradiente puro (no SGD) y tasa de aprendizaje pequeña ($\lambda = 0.001$) para mantenerse cerca del régimen analizado teóricamente.

### Métricas

Las magnitudes seguidas a lo largo del entrenamiento son las siguientes. En primer lugar, los **valores de GSNR** estimados por parámetro, $r_{j,t} \approx g_{D,j,t}^2 / \rho^2_{D,j,t}$ (ec. 25), junto con la **train loss** $L[D]$ y la **test loss** $L[D']$ medidas en los mismos pasos. En segundo lugar, el **OSGR empírico** estimado promediando sobre $M = 10$ conjuntos de entrenamiento (ec. 23 y 24) y la **correlación de Pearson** entre el lado izquierdo (OSGR por definición) y el lado derecho (OSGR como función del GSNR) de la ec. 19, que en MNIST alcanza valores entre 0.907 y 0.968 a lo largo del entrenamiento, validando empíricamente la identidad teórica. En tercer lugar, la proporción $p_{same\_sign}$ de parámetros cuyo gradiente tiene el mismo signo entre muestras distintas, que crece desde aproximadamente el 50% propio de la inicialización aleatoria hasta cerca del 56% durante el entrenamiento; este incremento aparentemente modesto basta para que la media domine a la varianza del gradiente y, por tanto, para producir GSNR alto. En cuarto lugar, la correlación entre los pesos $W^{(l)}_{s,c}$ y la variación $\Delta g^{(l)}_{D,s,c}$, que resulta negativa (especialmente acusada en el top-10% de pesos de mayor valor absoluto, donde alrededor del 80% presentan signo opuesto a su gradiente medio) y confirma el proceso de retroalimentación positiva sobre $(g^{(l)}_{D,s,c})^2$. Finalmente, el GSNR medio para configuraciones congeladas frente a no congeladas en el experimento del MLP de dos capas, que demuestra que en ausencia de *feature learning* el GSNR decrece monótonamente.

### Conclusiones

Los resultados establecen, empírica y teóricamente, que un **GSNR alto durante el entrenamiento es equivalente a un *gap* de generalización pequeño**: el OSGR se aproxima a 1 cuando la mayor parte de los parámetros mantiene gradientes consistentes entre muestras. Más aún, los autores observan que la curva de GSNR en DNN entrenadas con datos reales presenta un **proceso ascendente característico en las primeras etapas del entrenamiento**, mientras que con etiquetas aleatorizadas el GSNR permanece bajo durante toda la corrida; esa diferencia explica por qué redes con la misma capacidad de memorización pueden generalizar de forma radicalmente distinta. La causa subyacente es el **aprendizaje de características** propio de las DNN: a medida que las capas inferiores aprenden representaciones útiles, los gradientes que las capas superiores ven por muestra se vuelven más coherentes entre sí, elevando el numerador del GSNR sin elevar simultáneamente su denominador. En modelos lineales (regresión logística, SVM) este mecanismo no existe y el GSNR decrece monótonamente durante el entrenamiento, lo que coincide con sus mayores *gaps* de generalización empíricos. Los autores postulan que este vínculo entre dinámica de optimización y generalización es probablemente la clave del éxito empírico de las DNN y, además, abre la puerta a usar el GSNR como diagnóstico temprano sobre el conjunto de entrenamiento sin necesidad de un conjunto de validación.

## Medición y pipeline

**Métrica concreta.** Se adopta el GSNR por parámetro de Liu et al. (2020), $r(\theta_j) = \tilde g(\theta_j)^2 / \rho^2(\theta_j)$, calculado sobre los gradientes per-sample $g(x,y,\theta_j) = \partial \ell(f_W(x), y)/\partial \theta_j$ del gradiente bruto $\nabla L$, donde la esperanza y la varianza se toman sobre la distribución empírica de un *probe set* fijo de tamaño $M$. Junto al valor por parámetro se reportan agregaciones globales y por capa, y opcionalmente la fracción $p_{same\_sign}$ como diagnóstico complementario que conecta GSNR con la lectura signo-a-signo del paper.

**Entradas.** El estimador requiere gradientes per-sample $g_{i,j}$ para cada ejemplo $i \in \{1,\dots,M\}$ y cada parámetro $j$, lo que excluye el gradiente promediado por minibatch que devuelve `backward()` por defecto. En la práctica se obtiene con `torch.func.vmap(grad(loss_fn))` aplicado a un lote de $M$ ejemplos del *probe set* (en este TFG, $M = 512$ estratificado por clase). De cada parámetro $\theta_j$ se derivan tres estadísticos: la media $\tilde g_j = \tfrac{1}{M}\sum_i g_{i,j}$, la varianza $\rho^2_j = \tfrac{1}{M-1}\sum_i (g_{i,j} - \tilde g_j)^2$ (estimador insesgado) y la ratio $r_j = \tilde g_j^2 / (\rho^2_j + \varepsilon)$.

**OSGR.** Bajo la *non-overfitting limit approximation* se cumple $R(\mathcal{Z}, n) = 1 - \tfrac{1}{n}\sum_j W_j / (r_j + 1/n)$ con $\sum_j W_j = 1$, donde $W_j$ pondera la contribución del parámetro $j$ a la reducción de la pérdida de entrenamiento. La estimación empírica del OSGR como cantidad observada (no derivada) exige $M$ runs independientes de entrenamiento para promediar $\mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$, incompatible con un único run por composición experimental; por ello el OSGR queda fuera del registro derivado en este TFG y solo se loggea el lado derecho de la igualdad como diagnóstico interno cuando es informativo.

**Granularidad.** Una medición por época durante toda la corrida sobre el mismo *probe set* para garantizar comparabilidad entre pasos. Se computa por parámetro (resolución máxima, como en el paper) y se agrega a dos niveles adicionales: por capa, $\text{GSNR}^{(\ell)} = \operatorname{mean}_{j\in\ell}\, r_j$, y global, $\text{GSNR}_{\text{global}} = \operatorname{mean}_j\, r_j$. Se conserva además un histograma por época de los $r_j$ para inspección cualitativa y detección de colas pesadas.

**Coste.** Gradientes per-sample sobre $M = 512$ con varianza por parámetro vía `vmap`; el coste dominante es una pasada *forward-backward* vectorizada equivalente a un minibatch de tamaño $M$ pero con tensor de gradientes de dimensión $(M, P)$. La memoria $\mathcal{O}(M \cdot P)$ se evita con un estimador *streaming* tipo Welford: se acumulan $S_j = \sum_i g_{i,j}$ y $Q_j = \sum_i g_{i,j}^2$ por parámetro y se obtienen media y varianza online con memoria $\mathcal{O}(P)$ y un solo barrido sobre $M$. El *probe set* se comparte con `m_coherence` y `stiffness` para amortizar el barrido per-sample de $\nabla L$.

**Claves de logging.** Se registran como mínimo `gsnr/mean`, `gsnr/median` y `gsnr/p95` (la cola pesada de $r_j$ desplaza la media respecto a la mediana y la sola media es engañosa), `gsnr/per_layer/{name}` para cada capa relevante, `osgr/value` cuando aplica el diagnóstico interno y `p_same_sign` como complemento signo-a-signo. Opcionalmente se almacena el histograma `gsnr/hist` por época.

**Interpretación de la señal.** La convención es uniforme y alineada con el teorema OSGR del paper: **mayor GSNR = mejor generalización**, porque $r(\theta_j) = \tilde g_j^2/\rho_j^2$ es un cociente señal/ruido y un valor alto significa que la media del gradiente per-sample domina a su varianza, esto es, que las muestras coinciden en la dirección de actualización del parámetro. La lectura, sin embargo, no se agota en una sola cifra. `gsnr/median` es el indicador más fiable del estado típico del modelo porque es robusto a la cola pesada de la distribución de $r_j$; cuanto más alto, mejor. `gsnr/mean` apunta en la misma dirección pero queda sesgado al alza por unos pocos parámetros dominantes, de modo que su lectura aislada puede sugerir un régimen mejor del real y conviene contrastarla siempre con la mediana. `gsnr/p95` describe la cola alta de "parámetros estrella" que concentran la señal más limpia: ↑ p95 = mejor, y un p95 que sube mientras la mediana se estanca señala que la mejora se está concentrando en pocos parámetros en lugar de propagarse. `gsnr/per_layer/{name}` permite leer heterogeneidad por capa; diferencias grandes entre capas con GSNR alta y capas dominadas por ruido suelen identificar candidatas a regularización o congelación. `osgr/value` se interpreta directamente como eficiencia teórica de generalización por paso (↑ mejor, con cota natural en 1) y `p_same_sign` como concordancia signo-a-signo entre lotes per-sample (↑ mejor, partiendo del 50% propio de la inicialización aleatoria). Operativamente, el objetivo del diseño es elevar `gsnr/median` y `osgr/value` durante el régimen temprano del entrenamiento sin que ello se reduzca a inflar la media por culpa de unos pocos parámetros. Dos cautelas conviene fijar: parámetros con gradiente próximo a cero (ReLUs muertas, capas congeladas, *biases* iniciales) producen GSNR ruidoso y la lectura "↑ mejor" solo es informativa tras filtrar ese ruido numérico; y la agregación correcta es **mean, no sum**, porque la suma sobre parámetros no es comparable entre arquitecturas con distinto $P$ total.

**Trucos.** El estimador *streaming* de Welford evita materializar $(M, P)$ y permite escalar $M$ sin coste de memoria adicional. Conviene mantener el *probe set* fijo a lo largo del entrenamiento (incluyendo orden de iteración) para que las trayectorias por parámetro y por capa sean comparables entre épocas. Cuando se usa `vmap`, fijar `randomness='same'` en capas estocásticas garantiza que dropout/augmentations no contaminen la varianza atribuida al gradiente.

**Gotchas.** Los parámetros con gradiente cercano a cero (ReLUs muertas, capas congeladas, *biases* inicializados a cero) producen GSNR ruidoso porque tanto $\tilde g_j^2$ como $\rho^2_j$ caen al ruido numérico; conviene filtrarlos por umbral sobre $\|g_j\|$ o equivalente antes de agregar, y añadir un $\varepsilon$ por parámetro al denominador para evitar divisiones inestables. La cola pesada de $r_j$ hace que la media global sea sensible a unos pocos parámetros dominantes, por lo que se reportan también mediana y p95. La agregación correcta es **mean, no sum**: la suma es incomparable entre arquitecturas con distinto número total de parámetros $P$. Para la varianza se usa el estimador insesgado ($\div\, M-1$).

**Pseudocódigo PyTorch.**

```python
from torch.func import vmap, grad, functional_call

def per_sample_loss(params, buffers, x, y):
    out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
    return criterion(out, y.unsqueeze(0))

params  = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

# Welford streaming sobre M ejemplos del probe set fijo
S  = {k: torch.zeros_like(v) for k, v in params.items()}
Q  = {k: torch.zeros_like(v) for k, v in params.items()}
M  = 0
for x_batch, y_batch in probe_loader:               # batches del probe set
    grads = vmap(grad(per_sample_loss), in_dims=(None, None, 0, 0))(
        params, buffers, x_batch, y_batch
    )                                               # dict de tensores (B, *shape)
    for k, g in grads.items():
        S[k] += g.sum(dim=0)
        Q[k] += g.pow(2).sum(dim=0)
    M += x_batch.size(0)

mu   = {k: S[k] / M for k in S}
var  = {k: (Q[k] - M * mu[k].pow(2)) / (M - 1) for k in S}
gsnr = {k: mu[k].pow(2) / (var[k] + eps) for k in mu}

# Filtrado de parámetros muertos antes de agregar
alive  = {k: mu[k].abs() > tau for k in mu}
flat   = torch.cat([gsnr[k][alive[k]].flatten() for k in gsnr])
log = {
    "gsnr/mean":   flat.mean(),
    "gsnr/median": flat.median(),
    "gsnr/p95":    flat.quantile(0.95),
    **{f"gsnr/per_layer/{k}": gsnr[k][alive[k]].mean() for k in gsnr},
}
```

## Notes

- Métrica de **eje varianza**: directamente comparable con normalized gradient variance (Faghri) y con el gradient noise scale de McCandlish.
- A diferencia de Faghri, centrado en velocidad de convergencia, GSNR ata varianza relativa con generalización dentro de un marco teórico explícito (OSGR).
- Encaje en TFG: candidata fuerte como métrica temprana, ya que fundamenta teóricamente por qué baja varianza relativa predice mejor rendimiento final.

### Uso en el TFG

- **Métrica que origina.** `gsnr` (familia varianza). SNR por parámetro $r(\theta_j) = \tilde g(\theta_j)^2 / \rho^2(\theta_j)$, con $\tilde g(\theta_j) = \mathbb{E}_i[g_{i,j}]$ y $\rho^2(\theta_j) = \operatorname{Var}_i[g_{i,j}]$ sobre el gradiente bruto $\nabla L$.
- **Cómo se usa.** Estimación per-sample sobre `probe_set` fijo ($M = 512$) con acumuladores streaming $S_j = \sum_i g_{i,j}$, $Q_j = \sum_i g_{i,j}^2$; agregación por **mean** sobre parámetros ($\text{GSNR}_{\text{global}} = \operatorname{mean}_j r_j$), medida en ventanas tempranas (5/10/25/50% de épocas) y correlacionada con generalización y eficiencia. Comparte el barrido per-sample $\nabla L$ con `m_coherence` y `stiffness`.
- **Señal.** ↑ mejor (GSNR alto $\Rightarrow$ *gap* de generalización pequeño; la curva ascendente temprana sobre datos reales es justo el régimen que el TFG explota).
- **Pitfalls y decisiones.** Agregar con **mean, nunca sum** (la suma es incomparable entre arquitecturas de distinto $P$); $\rho_j^2$ **unbiased** ($\div\, M-1$); `eps` por parámetro para $\rho_j^2 \to 0$ (dead ReLU, capas congeladas); cola pesada de $r_j$ $\Rightarrow$ loguear también **median** y **p95**, no solo la media.
- **Qué NO se implementa.** El **OSGR** ($R(\mathcal{Z},n) = 1 - \tfrac{1}{n}\sum_j W_j/(r_j + 1/n)$) requiere $M$ runs de entrenamiento independientes, incompatible con un run por composición; GSNR es el único candidato del registro derivado de este paper.

## Papers relacionados

- [[A Study of Gradient Variance in Deep Learning]] — misma familia varianza; la normalized variance $\mathbb{V}[g]/\mathbb{E}[g]^2$ es el inverso del SNR que mide GSNR.
- [[An Empirical Model of Large-Batch Training]] — familia varianza; el gradient noise scale $\operatorname{tr}(\Sigma)/\|G\|^2$ es otra ratio señal/ruido del gradiente.
- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — comparte el sweep per-sample $\nabla L$ y el mismo problema (consistencia de gradientes ↔ generalización), conectado vía $p_{same\_sign}$.
- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — comparte el sweep per-sample $\nabla L$; alineación per-sample como proxy de generalización con la misma intuición de gradientes coherentes.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — mismo problema: el acuerdo entre gradientes per-sample en régimen sobreparametrizado controla SGD y generalización.
- [[Adam - A Method for Stochastic Optimization]] — la ratio $\hat m_t/\sqrt{\hat v_t}$ de Adam es el antecedente conceptual del SNR por parámetro que formaliza GSNR.

## Otros papers interesantes a revisar

- **Fantastic Generalization Measures and Where to Find Them** (Jiang et al., 2020) — estudio empírico a gran escala de más de 40 medidas de generalización (varias basadas en gradiente y varianza) con análisis causal; marco de referencia para situar GSNR frente a otras métricas. arXiv:1912.02178
- **Stochastic Gradient Descent as Approximate Bayesian Inference** (Mandt et al., 2017) — modela el ruido de SGD como proceso de Ornstein-Uhlenbeck; fundamenta la lectura de $\rho^2(\theta_j)$ como varianza del estimador estocástico. arXiv:1704.04289
- **On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima** (Keskar et al., 2017) — vincula tamaño de batch y *gap* de generalización vía nitidez del mínimo; complementa la explicación que GSNR ofrece del mismo *gap*. arXiv:1609.04836
- **The Break-Even Point on Optimization Trajectories of Deep Neural Networks** (Jastrzębski et al., 2020) — analiza el espectro de la covarianza del gradiente en la fase temprana del entrenamiento; directamente relevante para medir varianza de gradiente en ventanas tempranas. arXiv:2002.09572

## Cited By
