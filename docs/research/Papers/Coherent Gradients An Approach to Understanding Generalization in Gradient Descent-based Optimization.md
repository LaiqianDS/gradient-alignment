---
authors:
  - Satrajit Chatterjee
year: 2019
status: to-read
relevance: medium
last_review: 2026-05-07
url: https://arxiv.org/abs/2002.10657
---

# Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization

## Summary

**Contextualización.** El trabajo aborda una de las preguntas abiertas centrales en aprendizaje profundo: por qué las redes neuronales entrenadas con descenso de gradiente (GD) generalizan bien sobre datos reales pese a tener capacidad efectiva suficiente para ajustar mapeos arbitrarios desde entradas a salidas, incluso etiquetas aleatorias. Esta paradoja, popularizada por Zhang et al. (2017), se manifiesta empíricamente cuando se entrena la misma red sobre un dataset real $S$ y sobre una variante $S'$ con etiquetas aleatorizadas: en ambos casos se alcanza alta exactitud de entrenamiento, pero la exactitud de test sólo es buena en el primero. Las cotas clásicas de complejidad (VC-dimension, complejidad de Rademacher) resultan vacuas en este régimen sobreparametrizado, y enfoques alternativos basados en márgenes espectralmente normalizados (Bartlett et al., 2017), normas path-based group (Neyshabur et al., 2018) o compresión (Arora et al., 2018) tampoco proporcionan, según el autor, una explicación causal completa de la generalización en la práctica. Chatterjee observa además que el fenómeno no es exclusivo de redes neuronales: random forests y árboles de decisión muestran un comportamiento análogo, donde la única forma de ajustar etiquetas aleatorias es construir árboles mucho más profundos (uno por ejemplo en el límite), y la pregunta natural es si GD aplica un mecanismo similar de detección de regularidades comunes.

**Aportación: la hipótesis de Coherent Gradients (CGH).** El núcleo del paper es una hipótesis sobre la dinámica de GD que explica simultáneamente buena generalización en datos reales y la capacidad de ajustar ruido cuando es necesario. Sus enunciados fundamentales son: (1) los gradientes son coherentes, es decir, ejemplos similares (o partes similares de ejemplos) producen gradientes per-sample similares, mientras que ejemplos disimilares producen gradientes disimilares; (2) puesto que el gradiente global es la suma de los gradientes per-sample, resulta más fuerte en aquellas direcciones donde los gradientes individuales se refuerzan mutuamente y más débil donde difieren; (3) como los parámetros se actualizan proporcionalmente al gradiente, cambian más rápido en las direcciones de gradiente fuerte; y (4) en consecuencia, los cambios durante el entrenamiento están sesgados hacia aquellos que benefician simultáneamente a muchos ejemplos en lugar de a unos pocos. La conexión con generalización se articula vía estabilidad algorítmica (Bousquet & Elisseeff, 2002; Shalev-Shwartz et al., 2010): las direcciones fuertes son estables porque la presencia o ausencia de un único ejemplo no las altera, mientras que las direcciones débiles son sensibles a ejemplos individuales y, por tanto, fuente de sobreajuste. La hipótesis no es sólo descriptiva sino prescriptiva: sugiere modificaciones concretas a GD para suprimir las direcciones débiles y reducir overfitting.

**Metodología.** El autor desarrolla un argumento heurístico de primer orden: para un paso de GD con tasa $\alpha$, la reducción de pérdida es $\Delta L \approx -\alpha \cdot \|g_t\|^2$, y dado que $g_t = \sum_e g_t^e$, se descompone

$$\|g_t\|^2 = \sum_e \|g_t^e\|^2 + \sum_{e \neq e'} \langle g_t^e, g_t^{e'} \rangle,$$

donde el segundo término captura el grado de alineamiento (coherencia) entre gradientes per-sample. Si todos los $\|g_t^e\|$ son comparables, gradientes ortogonales producen $\|g_t\|^2 \approx m \cdot \|g^\circ\|^2$ mientras que gradientes alineados producen $\|g_t\|^2 \approx m^2 \cdot \|g^\circ\|^2$, prediciendo aprendizaje cuadráticamente más rápido bajo coherencia. Las predicciones se contrastan mediante experimentos de intervención organizados en dos bloques: (i) experimentos que reducen la similitud entre ejemplos inyectando label noise al 25%, 50%, 75% y 100%, donde se separan las muestras en pristine (etiqueta original) y corrupt (etiqueta permutada) y se monitoriza la fracción de la reducción de pérdida atribuible a cada grupo, definida como

$$f_t^p = \frac{\langle g_t, g_t^p \rangle}{\langle g_t, g_t \rangle}, \quad f_t^c = \frac{\langle g_t, g_t^c \rangle}{\langle g_t, g_t \rangle}, \quad f_t^p + f_t^c = 1;$$

(ii) experimentos de winsorized SGD, una modificación novel donde, para cada parámetro entrenable $w$, se computan los gradientes per-sample $g_w^e$, se calculan los percentiles $c$ y $100 - c$ sobre los ejemplos del minibatch ($l_w$ y $u_w$), y se aplica el gradiente clipeado $g_w^c = \sum_e \mathrm{clip}(g_w^e,\, l_w,\, u_w)$. El hiperparámetro $c \in [0, 50]$ controla el nivel de winsorization ($c=0$ es SGD estándar; $c=2$ recorta los 2 valores extremos por cada lado). La motivación es análoga al uso de un tamaño mínimo de hoja en random forests: limitar el efecto de las direcciones que sólo benefician a unos pocos ejemplos. Se contrasta también con simulaciones bajo el modelo nulo (permutando aleatoriamente las designaciones pristine/corrupt) para evaluar significación estadística.

**Datasets y modelos.** Para mantener una línea base limpia, los experimentos se restringen a MNIST (60.000 ejemplos de entrenamiento, 10.000 de test, dígitos 28×28 en escala de grises). En la primera serie de experimentos se entrena una red totalmente conectada con una capa oculta de 2048 ReLUs y softmax de 10 vías, inicialización Xavier, vanilla SGD sin momentum, cross entropy, learning rate constante de 0.1, minibatch de 100 y 10⁵ pasos (~170 épocas), sin regularización explícita. Se eligen arquitecturas fully-connected (no convolucionales) para evitar que la inductive bias arquitectónica contamine los resultados. Para los experimentos de winsorized SGD, debido al alto coste computacional de mantener gradientes per-sample, se reduce a una red más pequeña con 3 capas ocultas de 256 ReLUs cada una, entrenada durante 60.000 pasos (100 épocas) con $\text{lr} = 0.1$, minibatch de 100 y $c \in \{0, 1, 2, 4, 8\}$. El autor menciona experimentos preliminares en otras arquitecturas y datasets que sustentan la generalidad de las observaciones, pero no los reporta en el cuerpo principal.

**Métricas.** Se reportan training accuracy y validation/test accuracy a lo largo del entrenamiento, training loss, fracción de ejemplos aprendidos en función del paso, las medidas de coherencia $f_t^p$ y $f_t^c$ instantáneas y sus integrales acumuladas

$$i_t^p = \frac{1}{|p|} \sum_{t'=0}^{t} \langle g_{t'}, g_{t'}^p \rangle, \quad i_t^c = \frac{1}{|c|} \sum_{t'=0}^{t} \langle g_{t'}, g_{t'}^c \rangle,$$

y un overfit ajustado definido como $\text{ta} - [\varepsilon \cdot (1/10) + (1 - \varepsilon) \cdot \text{va}]$ para corregir el hecho de que las etiquetas de test no están aleatorizadas.

**Conclusiones.** Los experimentos respaldan cualitativamente la CGH. Primero, al aumentar el ruido de etiquetas, el realized learning rate disminuye porque los gradientes per-sample se vuelven menos alineados; los ejemplos pristine se aprenden más rápido que los corrupt, y a una tasa cercana a la del caso 0% de ruido, mientras que los corrupt se aprenden a una tasa cercana a la del 100%. Segundo, las trayectorias de $f_t^p$ y $f_t^c$ muestran un cruce: inicialmente los pristine dominan la reducción de pérdida pese a tener menos masa relativa en algunos casos, y sólo al final del entrenamiento los corrupt acaparan el descenso de la norma del gradiente; este patrón no aparece en los mundos nulos (excepto débilmente en el 75%), reforzando la significación. Tercero, winsorized SGD con $c > 1$ impide que la training accuracy supere la proper accuracy del dataset (es decir, evita memorizar etiquetas corruptas), y la tasa de overfit decrece monótonamente con $c$; sin embargo, valores grandes de $c$ degradan también la capacidad de fit y, por construcción, el paso ya no es estrictamente de descenso. El autor concluye que la CGH ofrece una explicación unificada de varios fenómenos empíricos —aprendizaje más lento con etiquetas aleatorias, robustez a label noise, beneficio del early stopping, mejora con mayor capacidad, esquemas de inicialización adversarial y detección de patrones comunes incluso bajo etiquetas aleatorias— y abre direcciones para SGD modificadas con garantías de generalización (y privacidad), así como conexiones con la Lottery Ticket Hypothesis y con métricas afines como stiffness (Fort et al., 2019) y gradient confusion (Sankararaman et al., 2019).

## Medición y pipeline

**Métrica concreta.** El paper introduce la *coherencia de gradientes* como concepto cualitativo: el grado en que los gradientes per-sample $g_i$ se refuerzan mutuamente al sumarse. Una operacionalización natural, derivada directamente de la descomposición heurística $\|\sum_i g_i\|^2 = \sum_i \|g_i\|^2 + \sum_{i \neq j} \langle g_i, g_j \rangle$, consiste en computar sobre un subconjunto de $N$ ejemplos del training set la fracción de magnitud retenida tras la suma vectorial:

$$\text{coherencia} \approx \frac{\|\sum_i g_i\|^2}{N \cdot \sum_i \|g_i\|^2},$$

valor acotado en $[1/N,\, 1]$ que tiende a $1$ bajo alineamiento perfecto y a $1/N$ bajo ortogonalidad. Este paper aporta el marco conceptual (background del TFG); el estimador escalable y formalmente justificado a loggear es la **m-coherence** propuesta en [[Making Coherence Out of Nothing At All]] (Chatterjee & Zielinski, 2022), que admite cómputo en streaming sin materializar la matriz de Gram.

**Entradas.** Per-example gradients $g_i \in \mathbb{R}^P$ sobre $N$ ejemplos del training set, computados con la pérdida no agregada.

**Cuándo computar.** Una vez por época sobre un subconjunto fijo de $N$ ejemplos (mismo subconjunto entre épocas para reducir varianza inter-medición). Versión barata: misma fórmula que m-coherence, con coste $O(P)$ por acumulador y un único barrido streaming.

**Coste.** Obtención de $N$ gradientes per-sample vía `torch.func.vmap` / `functorch.grad`; memoria $O(P)$ para los acumuladores (no se almacena el batch completo de gradientes); compute dominado por las $N$ retropropagaciones.

**Integración en el pipeline.** Pseudocódigo:

```python
S = torch.zeros(P, device=dev)   # acumulador vectorial Σ g_i
T = 0.0                          # acumulador escalar Σ ||g_i||²
for x_i, y_i in eval_subset:     # N ejemplos
    g_i = per_sample_grad(model, loss_fn, x_i, y_i)  # vmap/functorch
    S += g_i
    T += g_i.pow(2).sum().item()
coherence = (S.pow(2).sum().item()) / (N * T)
logger.log_scalar("coherence/global", coherence, step=epoch)
# repetir por capa: filtrar parámetros de cada módulo y emitir coherence/layer_k
```

**Consideraciones.** El paper aporta el marco conceptual (background del TFG: explica *por qué* el alineamiento de gradientes debería predecir generalización vía estabilidad algorítmica), pero no propone un estimador escalable; la métrica concreta que se loggeará es la implementación m-coherence de Chatterjee & Zielinski. Este desdoblamiento es útil para la narrativa de la tesis (background conceptual → método operacional). Conviene también separar el cálculo de evaluación del paso de entrenamiento para evitar contaminar la dinámica de SGD.

**Logging.** Escalar global `coherence/global` y por capa `coherence/layer_k` (descomposición por bloque de parámetros), una entrada por época. Sirve como baseline conceptual frente al cual comparar variantes (m-coherence escalable, stiffness, gradient confusion, GSNR) en los experimentos de correlación con eficiencia de entrenamiento.

## Cited By

