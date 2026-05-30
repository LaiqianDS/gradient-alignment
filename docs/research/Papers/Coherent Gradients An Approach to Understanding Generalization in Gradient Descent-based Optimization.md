---
authors:
  - Satrajit Chatterjee
year: 2019
status: to-read
relevance: medium
last_review: 2026-05-07
url: https://arxiv.org/abs/2002.10657
tfg_role:
  - theory
  - related-work
tfg_note: "Marco conceptual previo de coherencia de gradientes."
---

# Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization

## Summary

### Contextualización

El trabajo se enmarca en una de las preguntas abiertas centrales del aprendizaje profundo: por qué las redes neuronales entrenadas con descenso de gradiente (GD) generalizan bien sobre datos reales pese a disponer de capacidad efectiva suficiente para ajustar mapeos arbitrarios entre entradas y salidas, incluso bajo etiquetas aleatorias. Esta paradoja, popularizada por Zhang et al. (2017), se manifiesta empíricamente cuando se entrena la misma red sobre un dataset real $S$ y sobre una variante $S'$ con etiquetas aleatorizadas: en ambos casos se alcanza alta exactitud de entrenamiento, pero la exactitud de test sólo es buena en el primero. Las cotas clásicas de complejidad (VC-dimension, complejidad de Rademacher) resultan vacuas en el régimen sobreparametrizado, y los enfoques alternativos basados en márgenes espectralmente normalizados (Bartlett et al., 2017), normas path-based group (Neyshabur et al., 2018) o compresión (Arora et al., 2018) tampoco proporcionan, según Chatterjee, una explicación causal completa. El autor observa además que el fenómeno no es exclusivo de redes neuronales: random forests y árboles de decisión muestran un comportamiento análogo, donde la única forma de ajustar etiquetas aleatorias es construir árboles mucho más profundos —uno por ejemplo en el límite—, lo que sugiere preguntarse si GD aplica un mecanismo similar de detección de regularidades comunes.

### Aportación: la hipótesis de Coherent Gradients (CGH)

El núcleo del paper es una hipótesis sobre la dinámica de GD que explica simultáneamente la buena generalización en datos reales y la capacidad de ajustar ruido cuando es necesario. La articulación fundamental es la siguiente: los gradientes son coherentes, es decir, ejemplos similares (o partes similares de ejemplos) producen gradientes per-sample similares, mientras que ejemplos disimilares producen gradientes disimilares. Puesto que el gradiente global es la suma de los gradientes per-sample, resulta más fuerte en aquellas direcciones donde los gradientes individuales se refuerzan mutuamente y más débil donde difieren. Como los parámetros se actualizan proporcionalmente al gradiente, cambian más rápido en las direcciones de gradiente fuerte y, en consecuencia, los cambios durante el entrenamiento están sesgados hacia aquellos que benefician simultáneamente a muchos ejemplos en lugar de a unos pocos. La conexión con la generalización se articula vía estabilidad algorítmica (Bousquet & Elisseeff, 2002; Shalev-Shwartz et al., 2010): las direcciones fuertes son estables porque la presencia o ausencia de un único ejemplo no las altera, mientras que las direcciones débiles son sensibles a ejemplos individuales y, por tanto, fuente de sobreajuste. La hipótesis no es sólo descriptiva sino prescriptiva: sugiere modificaciones concretas a GD para suprimir las direcciones débiles y reducir overfitting.

### Metodología

El autor desarrolla un argumento heurístico de primer orden. Para un paso de GD con tasa $\alpha$, la reducción de pérdida es $\Delta L \approx -\alpha \cdot \|g_t\|^2$, y dado que $g_t = \sum_e g_t^e$, la norma al cuadrado se descompone como

$$\|g_t\|^2 = \sum_e \|g_t^e\|^2 + \sum_{e \neq e'} \langle g_t^e, g_t^{e'} \rangle,$$

donde el segundo término captura el grado de alineamiento (coherencia) entre gradientes per-sample. Si todos los $\|g_t^e\|$ son comparables, gradientes ortogonales producen $\|g_t\|^2 \approx m \cdot \|g^\circ\|^2$, mientras que gradientes alineados producen $\|g_t\|^2 \approx m^2 \cdot \|g^\circ\|^2$, prediciendo aprendizaje cuadráticamente más rápido bajo coherencia. Las predicciones se contrastan mediante dos bloques de experimentos de intervención. El primero reduce la similitud entre ejemplos inyectando label noise al 25%, 50%, 75% y 100%, separa las muestras en pristine (etiqueta original) y corrupt (etiqueta permutada) y monitoriza la fracción de la reducción de pérdida atribuible a cada grupo, definida como

$$f_t^p = \frac{\langle g_t, g_t^p \rangle}{\langle g_t, g_t \rangle}, \quad f_t^c = \frac{\langle g_t, g_t^c \rangle}{\langle g_t, g_t \rangle}, \quad f_t^p + f_t^c = 1.$$

El segundo bloque introduce **winsorized SGD**, una modificación novel: para cada parámetro entrenable $w$ se computan los gradientes per-sample $g_w^e$, se calculan los percentiles $c$ y $100 - c$ sobre los ejemplos del minibatch ($l_w$ y $u_w$) y se aplica el gradiente clipeado $g_w^c = \sum_e \mathrm{clip}(g_w^e, l_w, u_w)$. El hiperparámetro $c \in [0, 50]$ controla el nivel de winsorization ($c=0$ es SGD estándar; $c=2$ recorta los 2 valores extremos por cada lado). La motivación es análoga al uso de un tamaño mínimo de hoja en random forests: limitar el efecto de las direcciones que sólo benefician a unos pocos ejemplos. Se contrasta también con simulaciones bajo el modelo nulo (permutando aleatoriamente las designaciones pristine/corrupt) para evaluar significación estadística.

### Datasets y modelos

Para mantener una línea base limpia, los experimentos se restringen a MNIST (60.000 ejemplos de entrenamiento, 10.000 de test, dígitos 28×28 en escala de grises). En la primera serie se entrena una red totalmente conectada con una capa oculta de 2048 ReLUs y softmax de 10 vías, inicialización Xavier, vanilla SGD sin momentum, cross entropy, learning rate constante de 0.1, minibatch de 100 y 10⁵ pasos (≈170 épocas), sin regularización explícita. Se eligen arquitecturas fully-connected (no convolucionales) para evitar que la inductive bias arquitectónica contamine los resultados. Para los experimentos de winsorized SGD, por el alto coste computacional de mantener gradientes per-sample, se reduce a una red más pequeña con 3 capas ocultas de 256 ReLUs cada una, entrenada durante 60.000 pasos (100 épocas) con $\text{lr} = 0.1$, minibatch de 100 y $c \in \{0, 1, 2, 4, 8\}$. El autor menciona experimentos preliminares en otras arquitecturas y datasets que sustentan la generalidad de las observaciones, pero no los reporta en el cuerpo principal.

### Métricas

Se reportan training accuracy y validation/test accuracy a lo largo del entrenamiento, training loss, fracción de ejemplos aprendidos en función del paso, las medidas instantáneas de coherencia $f_t^p$ y $f_t^c$ y sus integrales acumuladas

$$i_t^p = \frac{1}{|p|} \sum_{t'=0}^{t} \langle g_{t'}, g_{t'}^p \rangle, \quad i_t^c = \frac{1}{|c|} \sum_{t'=0}^{t} \langle g_{t'}, g_{t'}^c \rangle,$$

y un overfit ajustado definido como $\text{ta} - [\varepsilon \cdot (1/10) + (1 - \varepsilon) \cdot \text{va}]$ para corregir el hecho de que las etiquetas de test no están aleatorizadas.

### Conclusiones

Los experimentos respaldan cualitativamente la CGH. En primer lugar, al aumentar el ruido de etiquetas, el realized learning rate disminuye porque los gradientes per-sample se vuelven menos alineados; los ejemplos pristine se aprenden más rápido que los corrupt, y a una tasa cercana a la del caso 0% de ruido, mientras que los corrupt se aprenden a una tasa cercana a la del 100%. En segundo lugar, las trayectorias de $f_t^p$ y $f_t^c$ muestran un cruce: inicialmente los pristine dominan la reducción de pérdida pese a tener menos masa relativa en algunos casos —en la práctica del paper, $f_t^p \approx 0.7$ indica que en torno al 70% de la mejora viene del subconjunto pristine—, y sólo al final del entrenamiento los corrupt acaparan el descenso de la norma del gradiente; este patrón no aparece en los mundos nulos (excepto débilmente en el 75%), lo que refuerza la significación. En tercer lugar, winsorized SGD con $c > 1$ impide que la training accuracy supere la proper accuracy del dataset (es decir, evita memorizar etiquetas corruptas) y la tasa de overfit decrece monótonamente con $c$; sin embargo, valores grandes de $c$ degradan también la capacidad de fit y, por construcción, el paso ya no es estrictamente de descenso. El autor concluye que la CGH ofrece una explicación unificada de varios fenómenos empíricos —aprendizaje más lento con etiquetas aleatorias, robustez a label noise, beneficio del early stopping, mejora con mayor capacidad, esquemas de inicialización adversarial y detección de patrones comunes incluso bajo etiquetas aleatorias— y abre direcciones para SGD modificadas con garantías de generalización (y privacidad), así como conexiones con la Lottery Ticket Hypothesis y con métricas afines como stiffness (Fort et al., 2019) y gradient confusion (Sankararaman et al., 2019).

## Medición y pipeline

**Métrica concreta.** El paper introduce la coherencia de gradientes como concepto cualitativo y la operacionaliza, en los experimentos de label noise, mediante dos escalares que cuantifican la fracción de la reducción de pérdida atribuible a cada partición del minibatch:

$$f_t^p = \frac{\langle g_t, g_t^p \rangle}{\langle g_t, g_t \rangle}, \qquad f_t^c = \frac{\langle g_t, g_t^c \rangle}{\langle g_t, g_t \rangle}, \qquad f_t^p + f_t^c = 1.$$

La intuición se apoya en la descomposición $\|g\|^2 = \sum_i \|g_i\|^2 + 2\sum_{i<j} \langle g_i, g_j \rangle$, que separa explícitamente la energía no alineada (suma de normas individuales) de la señal alineada (suma de productos internos cruzados); cuando una partición $p$ aporta una fracción desproporcionada del producto $\langle g_t, g_t \rangle$ frente a su masa relativa, esa partición concentra el alineamiento útil del paso. La extensión natural más allá de pristine/corrupt es definir, para cualquier partición $A$ del batch, $f_t^A = \langle g_t, g_t^A \rangle / \langle g_t, g_t \rangle$.

**Entradas.** Gradientes per-sample $g_t^e \in \mathbb{R}^P$ de los ejemplos del minibatch en el paso $t$ y gradiente global del batch $g_t = \sum_e g_t^e$, todos sobre la pérdida no agregada. Se requiere además una etiqueta de partición $\pi(e) \in \{\text{pristine}, \text{corrupt}\}$ (o cualquier partición arbitraria del batch) que indique a qué subgrupo pertenece cada ejemplo.

**Cuándo computar.** La métrica original es instantánea por step: el paper la reporta a lo largo del entrenamiento para visualizar el cruce pristine/corrupt. Para reducir varianza y coste de logging, una agregación por época (media de $f_t^p$ sobre los steps de la época) es razonable. Granularidad complementaria: global sobre todos los parámetros, por capa filtrando $g_t^e$ a los parámetros del módulo correspondiente, y por partición arbitraria (no sólo pristine/corrupt).

**Coste.** Obtener los $|B|$ gradientes per-sample del minibatch domina el coste; en PyTorch se implementa con `torch.func.vmap` sobre `torch.func.grad` (o `functorch` equivalente). Memoria: $O(|B| \cdot P)$ si se materializan todos los $g_t^e$, o $O(P)$ por partición si se acumula $g_t^p$ y $g_t^c$ en streaming —basta con sumar gradientes per-sample en su acumulador de partición y luego calcular $\langle g_t, g_t^p \rangle = \langle g_t^p + g_t^c, g_t^p \rangle$.

**Trucos.** La descomposición $\|g\|^2 = \sum_i \|g_i\|^2 + 2\sum_{i<j} \langle g_i, g_j \rangle$ permite separar de un solo barrido la energía aligned ($\sum_{i<j}\langle g_i, g_j\rangle$) de la energía no aligned ($\sum_i \|g_i\|^2$); el cociente $2\sum_{i<j}\langle g_i, g_j\rangle / \|g\|^2$ es una fracción global de alineamiento que no requiere partición etiquetada y se puede loggear como sanity check de la métrica m-coherence (continuación directa del paper).

**Integración en el pipeline.** Pseudocódigo:

```python
# Entradas: minibatch (x, y, partition), partition[e] in {"p", "c"}
per_sample_grads = vmap(grad(loss_fn))(params, x, y)   # (|B|, P)

g_p = per_sample_grads[partition == "p"].sum(dim=0)    # acumulado pristine
g_c = per_sample_grads[partition == "c"].sum(dim=0)    # acumulado corrupt
g_t = g_p + g_c                                        # gradiente global

denom = (g_t * g_t).sum()
f_p = (g_t * g_p).sum() / denom
f_c = (g_t * g_c).sum() / denom                        # f_p + f_c == 1

# Descomposición aligned vs no aligned (sin necesitar partición)
energy_self = per_sample_grads.pow(2).sum()            # Σ ||g_i||²
energy_total = denom                                   # ||g_t||²
aligned_fraction = (energy_total - energy_self) / energy_total

logger.log_scalar("cgh/f_pristine", f_p.item(), step=t)
logger.log_scalar("cgh/f_corrupt", f_c.item(), step=t)
logger.log_scalar("cgh/aligned_fraction", aligned_fraction.item(), step=t)
```

**Claves de logging.** `cgh/f_pristine`, `cgh/f_corrupt`, `cgh/aligned_fraction`; opcionalmente `cgh/f_pristine_layer_k` desagregando por bloque de parámetros y las integrales acumuladas $i_t^p, i_t^c$ del paper como `cgh/i_pristine`, `cgh/i_corrupt`.

**Gotchas.** Las fracciones $f_t^p$ y $f_t^c$ requieren disponer de una partición pristine/corrupt etiquetada, lo que sólo es natural en experimentos con label noise inyectado de forma controlada. En datasets limpios no existe tal partición; la adaptación pasa por (i) reutilizar la métrica como agregado por clase o por subgrupo demográfico, (ii) loggear únicamente `aligned_fraction` (independiente de partición), o (iii) sustituir por m-coherence (Chatterjee & Zielinski, 2020) que normaliza el mismo objeto sin requerir etiquetado del batch. Conviene además separar el cómputo del paso de entrenamiento para no contaminar la dinámica de SGD y mantener el subconjunto de evaluación fijo entre épocas para reducir varianza inter-medición.

## Notes

### Uso en el TFG

- **Rol: motivación / marco conceptual, no métrica del registry.** Chatterjee (2019) es la cita central de la *intro* y el *related work* de toda la familia de alineación del TFG. Articula la **Coherent Gradients Hypothesis (CGH)** —ejemplos similares producen gradientes per-sample similares, y como el gradiente global es su suma, las direcciones reforzadas por muchos ejemplos son las que dominan la actualización y, vía estabilidad algorítmica, las que generalizan. Responde al *por qué* medimos alineación, no aporta un estimador concreto al `METRIC_REGISTRY`.
- **Fórmula clave (descomposición que motiva el TFG).** $\|g_t\|^2 = \sum_e \|g_t^e\|^2 + \sum_{e \neq e'} \langle g_t^e, g_t^{e'} \rangle$. El primer término es siempre positivo; el **segundo término** (suma de productos internos per-sample) es exactamente la alineación que el TFG cuantifica. La intuición cuadrática del paper —gradientes ortogonales dan $\|g_t\|^2 \approx m\|g^\circ\|^2$ frente a $\approx m^2\|g^\circ\|^2$ bajo alineación perfecta— es la base conceptual de que más coherencia temprana debería predecir convergencia más rápida.
- **Qué NO se implementa: $f_t^p$ y $f_t^c$.** Las métricas de fracción de reducción de pérdida atribuible a pristine/corrupt, $f_t^p = \langle g_t, g_t^p\rangle/\langle g_t, g_t\rangle$ y $f_t^c = \langle g_t, g_t^c\rangle/\langle g_t, g_t\rangle$ (con $f_t^p + f_t^c = 1$), y sus integrales $i_t^p, i_t^c$, **quedan fuera del registry**: exigen *label noise* (partición pristine/corrupt del minibatch) que se descarta en la v1 del TFG. Tampoco se reproduce *winsorized SGD*: el TFG es correlacional sobre $\nabla L$ bruto, no interviene la dinámica de optimización.
- **Qué SÍ hereda el TFG.** El estimador escalable y formalmente justificado que sí se loggea es **m-coherence** (ver [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]]), continuación directa de este paper con cómputo $O(m)$ en streaming. El desdoblamiento background conceptual (Chatterjee 2019) → método operacional (m-coherence) es útil para la narrativa de la tesis.
- **Conexión con el registry.** El segundo término de la descomposición es la magnitud común a tres métricas de alineación: `m_coherence` lo normaliza por $\sum_e\|g_t^e\|^2$; `stiffness` ([[Stiffness - A New Perspective on Generalization in Neural Networks]]) mide el mismo producto interno per-sample como coseno/signo, con desagregación intra/entre clase; `gradient_confusion` ([[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]]) captura el caso adverso ($\langle g_i, g_j\rangle$ muy negativo). La CGH es el paraguas teórico que justifica por qué estos tres proxies deberían correlacionar con generalización y eficiencia.

### Discrepancias detectadas

- **URL de arXiv.** El frontmatter apunta a `arxiv.org/abs/2002.10657`, identificador que en arXiv corresponde a un preprint posterior; la primera versión del trabajo "Coherent Gradients" (Chatterjee, 2019) circula con identificador `arXiv:2002.10657` desde febrero de 2020, mientras que el título y autoría suelen citarse con año 2019 por la fecha del workshop original. Conviene homogeneizar año y URL al criterio bibliográfico del TFG (mantener 2019 como año de referencia o actualizar a 2020 según la convención fijada).
- **Operacionalización de coherencia.** Versiones previas del archivo introducían un estimador propio $\|\sum_i g_i\|^2 / (N \sum_i \|g_i\|^2)$ como "coherencia" supuestamente derivada de este paper. El paper no define ese ratio: las métricas explícitas son $f_t^p$ y $f_t^c$ (que requieren partición pristine/corrupt) y la descomposición heurística. El estimador escalable de tipo cociente normalizado es la **m-coherence** de Chatterjee & Zielinski (2020); atribuirlo a este paper inducía a confusión.

## Papers relacionados

- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — continuación directa (Chatterjee & Zielinski); convierte la coherencia cualitativa de la CGH en m-coherence, el estimador escalable que sí entra al registry.
- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — citado por el paper; operacionaliza el mismo producto interno per-sample (coseno/signo) que el 2º término de la descomposición, con desagregación intra/entre clase.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — citado por el paper; gradient confusion es el caso adverso de la alineación (gradientes anti-correlacionados) y conecta coherencia con sobreparametrización y velocidad de SGD.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — mismo problema (generalización vía estadística de gradientes per-sample); su argumento de same-sign gradients es el análogo por-parámetro de la coherencia por-dirección.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — misma familia (proxy de alineación train-time para predecir generalización); hereda la motivación de la CGH para un score per-sample barato.
- [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training]] — noción de alineación afín (kernel-target alignment del NTK); enlaza coherencia de gradientes con la teoría de núcleos tangentes.

## Otros papers interesantes a revisar

- **Understanding deep learning requires rethinking generalization** (Zhang, Bengio, Hardt, Recht, Vinyals, 2017) — la paradoja de generalización (ajustar etiquetas aleatorias con la misma red) que motiva todo el paper y el TFG. arXiv:1611.03530.
- **Stability and Generalization** (Bousquet & Elisseeff, 2002) — base de estabilidad algorítmica que Chatterjee usa para ligar coherencia con generalización (direcciones fuertes = estables ante remoción de un ejemplo). JMLR 2:499–526.
- **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks** (Frankle & Carbin, 2019) — conexión que el propio paper señala: las direcciones coherentes/fuertes se relacionan con subredes entrenables. arXiv:1803.03635.
- **Gradient Descent Quantizes ReLU Network Features / SGD on Neural Networks Learns Functions of Increasing Complexity** (Nakkiran et al., 2019) — evidencia complementaria de que SGD aprende primero patrones simples comunes a muchos ejemplos, en línea con la CGH. arXiv:1905.11604.

## Cited By
