---
authors:
  - Jinlong Liu
  - Guoqing Jiang
  - Yunzhi Bai
  - Ting Chen
  - Huayan Wang
year: 2020
status: to-read
relevance: high
url: https://arxiv.org/abs/2001.07384
tfg_role:
  - metric
  - theory
tfg_note: "Origen de `gsnr` (gradient signal-to-noise ratio por parámetro; GSNR alto → menor gap de generalización), con marco teórico OSGR. Eje varianza; es el inverso conceptual de la varianza normalizada de Faghri."
---

# Understanding Why Neural Networks Generalize Well Through GSNR of Parameters

## Summary

### Contextualización

El trabajo aborda una de las paradojas centrales de la teoría moderna del deep learning: las redes neuronales profundas contienen habitualmente muchos más parámetros entrenables que muestras de entrenamiento. Según las teorías clásicas de la generalización basadas en dimensión VC o complejidad de Rademacher, deberían exhibir un *gap* de generalización elevado, esto es, una diferencia grande entre la pérdida en entrenamiento y la pérdida en test. La práctica contradice esta predicción de forma sistemática, ya que las DNN logran *gaps* notablemente pequeños incluso en régimen muy sobreparametrizado. Trabajos previos han atribuido este fenómeno a propiedades de la optimización por descenso de gradiente (Hardt et al., 2015; Zhang et al., 2016) o a la geometría del paisaje de pérdida y la curvatura de los mínimos hallados (Keskar et al., 2016; Dinh et al., 2017). Sin embargo, el mecanismo concreto que vincula la dinámica del entrenamiento con la capacidad final de generalización seguía sin caracterizarse de forma cuantitativa. Liu et al. (ICLR 2020) cubren ese hueco proponiendo un puente analítico entre la estadística de los gradientes durante el entrenamiento y el *gap* de generalización resultante.

### Aportación

La contribución central es la introducción del **Gradient Signal-to-Noise Ratio (GSNR)** como métrica por parámetro que conecta de forma cuantitativa la dinámica del descenso de gradiente con la generalización de la red. El GSNR de un parámetro $\theta_j$ se define como el cociente entre el cuadrado de la media de su gradiente sobre la distribución de datos y la varianza de dicho gradiente sobre la misma distribución, es decir, $r(\theta_j) = \tilde{g}^2(\theta_j) / \rho^2(\theta_j)$, donde $\tilde{g}(\theta_j) = \mathbb{E}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$ y $\rho^2(\theta_j) = \mathrm{Var}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$. La intuición es directa: un GSNR alto sobre un parámetro indica que la media de los gradientes per-sample domina a su varianza, lo que ocurre cuando la mayoría de las muestras coincide en la dirección óptima de actualización y, por tanto, el parámetro está capturando una característica compartida y transferible entre ejemplos. Un GSNR bajo señala lo contrario: gradientes dispersos que tiran del parámetro en direcciones contradictorias y que probablemente se ajustan a peculiaridades de muestras concretas. Los autores demuestran que mantener un GSNR elevado durante el entrenamiento implica un *gap* de generalización pequeño. Además, prueban analíticamente que la dinámica del descenso de gradiente en DNN, a diferencia de la de modelos lineales como regresión logística o SVM, produce naturalmente un GSNR elevado mediante un proceso vinculado al aprendizaje de características (*feature learning*).

### Metodología

El núcleo teórico es la derivación del **One-Step Generalization Ratio (OSGR)**, definido como $R(\mathcal{Z}, n) = \mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$, es decir, la razón entre la mejora esperada de la pérdida en test y la mejora esperada de la pérdida en entrenamiento tras un único paso de descenso de gradiente. Bajo la *non-overfitting limit approximation* (Asunción 2.3.1, que postula que $g_D(\theta)$ y $g_{D'}(\theta)$ siguen la misma distribución) y considerando el límite de tasa de aprendizaje pequeña ($\lambda \to 0$), los autores derivan una expresión cerrada que enlaza el OSGR con los GSNR de los parámetros: $R(\mathcal{Z}, n) = 1 - \tfrac{1}{n}\sum_j W_j \cdot \tfrac{1}{r_j + 1/n}$, donde la definición exacta del peso (ec. 22, vía ec. 21) es $W_j := \mathbb{E}_D(\Delta L_j[D])/\mathbb{E}_D(\Delta L[D]) = (\tilde{g}_j^2 + \rho_j^2/n)/\sum_k(\tilde{g}_k^2 + \rho_k^2/n)$. Este peso es la contribución del parámetro $j$ a la disminución de la pérdida de entrenamiento, con $\sum_j W_j = 1$ por construcción. La forma simplificada $W_j \approx \tilde{g}_j^2 / \sum_k \tilde{g}_k^2$ es la aproximación de $n$ grande (régimen de la ec. 25), no la definición. La expresión deja claro que, cuando los $r_j$ son grandes, el OSGR tiende a 1 (el test loss decrece prácticamente al mismo ritmo que el train loss y el *gap* es pequeño). Valores pequeños de $r_j$, en cambio, empujan el OSGR hacia cero (su cota inferior bajo la identidad de la ec. 19, alcanzada cuando $r_j \to 0$), generando *gaps* grandes. Empíricamente el OSGR puede llegar a ser ligeramente negativo solo cuando la *non-overfitting limit approximation* deja de cumplirse, esto es, como artefacto de la aproximación y no como predicción de la forma cerrada (que está acotada en $[0,1]$). El paso clave en la derivación consiste en escribir $g_D(\theta) = \tilde{g}(\theta) + \epsilon$ y $g_{D'}(\theta) = \tilde{g}(\theta) + \epsilon'$ con $\epsilon$ y $\epsilon'$ ruidos independientes de media cero y varianza $\sigma^2(\theta) = \rho^2(\theta)/n$, y en aplicar la aproximación lineal $\Delta L \approx -\Delta\theta \cdot \partial L/\partial\theta$. La sección 3 complementa la derivación con un análisis de la dinámica de capas plenamente conectadas (ecs. 27–28 en §3.3 del texto principal; ecs. 29–34 en el Apéndice B) que muestra cómo el primer término del cambio en un paso de $g_D^{(l)}$ tiene signo opuesto a los pesos $W^{(l)}_{s,c}$, lo que genera un proceso de retroalimentación positiva. Los pesos cuyo signo es contrario al de su gradiente medio tienden a aumentar su valor absoluto, incrementando el numerador del GSNR y, con él, la coherencia de las actualizaciones a lo largo del entrenamiento.

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas

Las magnitudes seguidas a lo largo del entrenamiento son las siguientes. En primer lugar, los **valores de GSNR** estimados por parámetro, $r_{j,t} \approx g_{D,j,t}^2 / \rho^2_{D,j,t}$ (ec. 25), junto con la **train loss** $L[D]$ y la **test loss** $L[D']$ medidas en los mismos pasos. En segundo lugar, el **OSGR empírico** estimado promediando sobre $M = 10$ conjuntos de entrenamiento (ec. 23 y 24) y la **correlación de Pearson** entre el lado izquierdo (OSGR por definición) y el lado derecho (OSGR como función del GSNR) de la ec. 19. Esta correlación alcanza en MNIST valores entre 0.907 y 0.968 a lo largo del entrenamiento, validando empíricamente la identidad teórica. En tercer lugar, la proporción $p_{same\_sign}$ de parámetros cuyo gradiente tiene el mismo signo entre muestras distintas, que crece desde aproximadamente el 50% propio de la inicialización aleatoria hasta cerca del 56% durante el entrenamiento. Este incremento aparentemente modesto basta para que la media domine a la varianza del gradiente y, por tanto, para producir GSNR alto. En cuarto lugar, la correlación entre los pesos $W^{(l)}_{s,c}$ y la variación $\Delta g^{(l)}_{D,s,c}$, que el paper acota a la **fase temprana** ("In the early training stage, they are indeed negatively correlated"; la Fig. 6a muestra que se vuelve positiva, ~+0.4, pasada la época ~100). En esa fase la correlación es especialmente acusada en el top-10% de pesos de mayor valor absoluto (alrededor del 80% presentan signo opuesto a su gradiente medio), confirmando el proceso de retroalimentación positiva sobre $(g^{(l)}_{D,s,c})^2$. Finalmente, el GSNR medio para configuraciones congeladas frente a no congeladas en el experimento del MLP de dos capas, que demuestra que en ausencia de *feature learning* el GSNR decrece monótonamente.

### Conclusiones

Los resultados establecen, empírica y teóricamente, que un **GSNR alto durante el entrenamiento es equivalente a un *gap* de generalización pequeño**: el OSGR se aproxima a 1 cuando la mayor parte de los parámetros mantiene gradientes consistentes entre muestras. Más aún, los autores observan que la curva de GSNR en DNN entrenadas con datos reales presenta un **proceso ascendente característico en las primeras etapas del entrenamiento**, mientras que con etiquetas aleatorizadas el GSNR permanece bajo durante toda la corrida. Esa diferencia explica por qué redes con la misma capacidad de memorización pueden generalizar de forma radicalmente distinta. La causa subyacente es el **aprendizaje de características** propio de las DNN: a medida que las capas inferiores aprenden representaciones útiles, los gradientes que las capas superiores ven por muestra se vuelven más coherentes entre sí, elevando el numerador del GSNR sin elevar simultáneamente su denominador. En modelos lineales (regresión logística, SVM) este mecanismo no existe y el GSNR decrece monótonamente durante el entrenamiento, lo que coincide con sus mayores *gaps* de generalización empíricos. Los autores postulan que este vínculo entre dinámica de optimización y generalización es probablemente la clave del éxito empírico de las DNN y, además, abre la puerta a usar el GSNR como diagnóstico temprano sobre el conjunto de entrenamiento sin necesidad de un conjunto de validación.

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** Se adopta el GSNR por parámetro de Liu et al. (2020), $r(\theta_j) = \tilde g(\theta_j)^2 / \rho^2(\theta_j)$, calculado sobre los gradientes per-sample $g(x,y,\theta_j) = \partial \ell(f_W(x), y)/\partial \theta_j$ del gradiente bruto $\nabla L$, donde la esperanza y la varianza se toman sobre la distribución empírica de un *probe set* fijo de tamaño $M$. Junto al valor por parámetro se reportan agregaciones globales y por capa, y opcionalmente la fracción $p_{same\_sign}$ como diagnóstico complementario que conecta GSNR con la lectura signo-a-signo del paper.

**Entradas.** El estimador requiere gradientes per-sample $g_{i,j}$ para cada ejemplo $i \in \{1,\dots,M\}$ y cada parámetro $j$, lo que excluye el gradiente promediado por minibatch que devuelve `backward()` por defecto. En la práctica se obtiene con `torch.func.vmap(grad(loss_fn))` aplicado a un lote de $M$ ejemplos del *probe set* (en este TFG, el probe compartido de $M = 256$ muestreado con `randperm`; la estratificación por clase queda pendiente). De cada parámetro $\theta_j$ se derivan tres estadísticos: la media $\tilde g_j = \tfrac{1}{M}\sum_i g_{i,j}$, la varianza $\rho^2_j = \tfrac{1}{M-1}\sum_i (g_{i,j} - \tilde g_j)^2$ (estimador insesgado) y la ratio $r_j = \tilde g_j^2 / (\rho^2_j + \varepsilon)$.

**OSGR.** Bajo la *non-overfitting limit approximation* se cumple $R(\mathcal{Z}, n) = 1 - \tfrac{1}{n}\sum_j W_j / (r_j + 1/n)$ con $\sum_j W_j = 1$, donde $W_j$ pondera la contribución del parámetro $j$ a la reducción de la pérdida de entrenamiento. La estimación empírica del OSGR como cantidad observada (no derivada) exige $M$ runs independientes de entrenamiento para promediar $\mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$, incompatible con un único run por composición experimental; por ello el OSGR queda fuera del registro derivado en este TFG y solo se loggea el lado derecho de la igualdad como diagnóstico interno cuando es informativo.

**Granularidad.** Una medición por época durante toda la corrida sobre el mismo *probe set* para garantizar comparabilidad entre pasos. Se computa por parámetro (resolución máxima, como en el paper) y se agrega globalmente, $\text{GSNR}_{\text{global}} = \operatorname{mean}_j\, r_j$; la agregación por capa y el histograma por época quedan para v2.

**Coste.** Gradientes per-sample sobre $M = 256$ con varianza por parámetro vía `vmap`; el coste dominante es una pasada *forward-backward* vectorizada equivalente a un minibatch de tamaño $M$ pero con tensor de gradientes de dimensión $(M, P)$. La v1 **materializa** esa matriz $[M, P]$ (no hay Welford streaming; con ResNet-18 y $M=256$ son ~12 GB fp32, riesgo abierto registrado en [[Métricas]]); el streaming $\mathcal{O}(P)$ con acumuladores $S_j$/$Q_j$ queda como optimización. El *probe set* es el mismo que usan `m_coherence` y `stiffness`, pero cada métrica recomputa su propio barrido en v1 (sin caché compartida).

**Claves de logging.** El código emite `gsnr/mean`, `gsnr/median` y `gsnr/p95` (la cola pesada de $r_j$ desplaza la media respecto a la mediana y la sola media es engañosa). `gsnr/per_layer/{name}`, `gsnr/hist`, `osgr/value` y `p_same_sign` quedan para v2 (no se emiten).

**Interpretación de la señal.** La convención es uniforme y alineada con el teorema OSGR del paper: **mayor GSNR = mejor generalización**, porque $r(\theta_j) = \tilde g_j^2/\rho_j^2$ es un cociente señal/ruido. Un valor alto significa que la media del gradiente per-sample domina a su varianza, esto es, que las muestras coinciden en la dirección de actualización del parámetro. La lectura, sin embargo, no se agota en una sola cifra. `gsnr/median` es el indicador más fiable del estado típico del modelo porque es robusto a la cola pesada de la distribución de $r_j$; cuanto más alto, mejor. `gsnr/mean` apunta en la misma dirección pero queda sesgado al alza por unos pocos parámetros dominantes, de modo que su lectura aislada puede sugerir un régimen mejor del real y conviene contrastarla siempre con la mediana. `gsnr/p95` describe la cola alta de "parámetros estrella" que concentran la señal más limpia: ↑ p95 = mejor, y un p95 que sube mientras la mediana se estanca señala que la mejora se está concentrando en pocos parámetros en lugar de propagarse. `gsnr/per_layer/{name}` permite leer heterogeneidad por capa; diferencias grandes entre capas con GSNR alta y capas dominadas por ruido suelen identificar candidatas a regularización o congelación. `osgr/value` se interpreta directamente como eficiencia teórica de generalización por paso (↑ mejor, con cota natural en 1) y `p_same_sign` como concordancia signo-a-signo entre lotes per-sample (↑ mejor, partiendo del 50% propio de la inicialización aleatoria). Operativamente, el objetivo del diseño es elevar `gsnr/median` y `osgr/value` durante el régimen temprano del entrenamiento sin que ello se reduzca a inflar la media por culpa de unos pocos parámetros.

**Trucos.** Un estimador *streaming* de Welford evitaría materializar $(M, P)$ y permitiría escalar $M$ sin coste de memoria adicional (pendiente; la v1 materializa la matriz). Conviene mantener el *probe set* fijo a lo largo del entrenamiento (incluyendo orden de iteración) para que las trayectorias por parámetro y por capa sean comparables entre épocas. Cuando se usa `vmap`, fijar `randomness='same'` en capas estocásticas garantiza que dropout/augmentations no contaminen la varianza atribuida al gradiente.

**Avisos.** Los parámetros con gradiente cercano a cero (ReLUs muertas, capas congeladas, *biases* inicializados a cero) producen GSNR ruidoso porque tanto $\tilde g_j^2$ como $\rho^2_j$ caen al ruido numérico; conviene filtrarlos por umbral sobre $\|g_j\|$ o equivalente antes de agregar, y añadir un $\varepsilon$ por parámetro al denominador para evitar divisiones inestables. La cola pesada de $r_j$ hace que la media global sea sensible a unos pocos parámetros dominantes, por lo que se reportan también mediana y p95. La agregación correcta es **mean, no sum**: la suma es incomparable entre arquitecturas con distinto número total de parámetros $P$. Para la varianza se usa el estimador insesgado ($\div\, M-1$).

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
- **Cómo se usa.** Estimación per-sample sobre el `probe_set` fijo compartido ($M = 256$, matriz $[M, P]$ materializada en v1; streaming pendiente); agregación por **mean** sobre parámetros ($\text{GSNR}_{\text{global}} = \operatorname{mean}_j r_j$), medida en ventanas tempranas (5/10/25/50% de épocas) y correlacionada con generalización y eficiencia. Usa el mismo probe que `m_coherence` y `stiffness`, aunque cada métrica recomputa su barrido en v1.
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
