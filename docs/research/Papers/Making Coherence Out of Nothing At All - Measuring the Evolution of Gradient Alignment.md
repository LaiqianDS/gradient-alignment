---
authors: ["Satrajit Chatterjee", "Piotr Zielinski"]
year: 2020
status: to-read
relevance: high
last_review: 2026-05-07
url: "https://arxiv.org/pdf/2008.01217"
---

# Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment

[arXiv](https://arxiv.org/pdf/2008.01217)

## Summary

**Contextualización.** El trabajo se sitúa en la línea abierta por la *Coherent Gradients hypothesis* (CG) que Chatterjee formula en ICLR 2020. CG propone una explicación unificada de memorización y generalización en redes profundas entrenadas con SGD: como el gradiente del minibatch es la suma de gradientes por ejemplo, los pasos de SGD se sesgan hacia direcciones que reducen la pérdida sobre múltiples ejemplos simultáneamente cuando dichas direcciones existen. En el caso extremo de gradientes por ejemplo perfectamente alineados se obtendría estabilidad algorítmica y, por tanto, generalización; en el extremo de gradientes mutuamente ortogonales no habría estabilidad y se produciría memorización pura. Este paper de Chatterjee y Zielinski (2020) es un follow-up empírico que construye la herramienta de medición que falta para validar y matizar CG en la práctica, abordando dos preguntas abiertas: ¿qué aspecto tiene la coherencia en datasets y arquitecturas reales? y ¿cómo evoluciona a lo largo de toda la trayectoria de entrenamiento, no sólo al inicio?

**Aportación.** La contribución central es **m-coherence** (denotada $\alpha_m$), una métrica de alineamiento de gradientes por ejemplo con una interpretación intuitiva muy directa: sobre una muestra de tamaño $m$, m-coherence indica el número medio de ejemplos (incluyéndose a sí mismo) que se benefician de un pequeño paso a lo largo del gradiente de un ejemplo aleatorio. Frente a métricas previas (pairwise dot product, sign/cosine stiffness de Fort et al. 2019, gradient confusion de Sankararaman et al. 2019), m-coherence es más interpretable, matemáticamente más limpia, escala-invariante en sentido apropiado y, sobre todo, **computacionalmente eficiente**: $O(m)$ en lugar de $O(m^2)$, lo que permite usar muestras dos órdenes de magnitud mayores. Los autores también señalan que el recíproco de $\alpha$ coincide con la *gradient diversity* previamente usada en cotas teóricas (Yin et al. 2018, Jain et al. 2018), conectando la métrica con literatura sobre convergencia de mini-batch SGD.

**Metodología.** Partiendo de la expansión de Taylor de la pérdida por ejemplo $\ell_z(w)$ en torno a $w$ al dar un paso $h_z = -\eta\, g_z$, definen

$$\alpha := \frac{\mathbb{E}[g_z \cdot g]}{\mathbb{E}[g_z \cdot g_z]} = \frac{\mathbb{E}_{z, z'}[g_z \cdot g_{z'}]}{\mathbb{E}_z[g_z \cdot g_z]}.$$

Esta cantidad $\alpha \in [0, 1]$ tiene interpretación de cociente: cambio real en la pérdida global tras un paso de gradiente dividido por el cambio máximo posible si cada $\ell_z$ pudiese optimizarse independientemente. En el límite ortogonal (gradientes mutuamente perpendiculares con misma norma) $\alpha = 1/m$, mientras que con gradientes idénticos $\alpha = 1$. Multiplicando por $m$ se obtiene m-coherence $= m \cdot \alpha$, que vale $1$ en el caso ortogonal y $m$ en el caso de alineación perfecta. La sección 4 generaliza la noción a cualquier distribución de vectores en un espacio euclídeo y demuestra propiedades clave: acotación $0 \leq \alpha(V) \leq 1$ (Teorema 1), invariancia de escala $\alpha(kV) = \alpha(V)$ (Lema 2), amplificación por mini-batching $\alpha(W) \geq \alpha(V)$ con $W = (1/k) \sum v_i$ (Teorema 3 y Corolario 3.1, que muestra que mini-batches incrementan coherencia, lo que obliga a usar gradientes por ejemplo y no de minibatch para medir coherencia correctamente), y efecto de gradientes nulos $\alpha(W) = p \cdot \alpha(V)$ (Lema 4, que justifica que ejemplos ya ajustados con gradiente cercano a cero diluyen la coherencia). El **algoritmo eficiente** explota la identidad $\mathbb{E}[g_z \cdot g] = \mathbb{E}_z[g_z] \cdot \mathbb{E}_z[g_z] = \|g\|^2$: basta mantener dos sumas corrientes ($g$ acumulado y suma de $\|g_z\|^2$) sobre el sample, sin necesidad de almacenar los gradientes por ejemplo, lo que permite cómputo en streaming. Se contrasta con stiffness y gradient confusion: las no-linealidades (sign, normalización por norma, mínimo) y la exclusión $z \neq z'$ hacen difícil ligar esas métricas al cambio real en la loss tras un step, mientras que $\alpha$ se conecta exactamente con la reducción de pérdida.

**Datasets y modelos.** Aunque el resumen del título de tu TFG menciona CIFAR-10, los experimentos principales del paper se realizan en realidad sobre **ImageNet** con tres variantes de ruido en etiquetas (0%, 50%, 100% de etiquetas aleatorizadas). Las arquitecturas usadas son **ResNet-18** (experimento principal) e **Inception-V3** (validación de generalidad). Configuración: SGD con momentum $0.9$, batch size $4096$, schedule de learning rate de Goyal et al. 2017, sin augmentation ni weight decay (para observar memorización en tiempos razonables), tamaño de muestra $m = 40\,356$ ejemplos de entrenamiento fijos. El apéndice incluye experimentos adicionales con 25% y 75% de ruido y duplicación de $m$ a $80\,072$. Las comparaciones con Fort et al. 2019 (MNIST, Fashion MNIST, CIFAR-10/100, MNLI) y Sankararaman et al. 2019 (CIFAR-10/100) se discuten cualitativamente.

**Métricas medidas.** Se trackea m-coherence sobre el modelo completo y, separadamente, sobre tres capas representativas (primera convolución, capa intermedia, capa fully-connected final), junto con loss y top-1 accuracy en train y test, durante toda la trayectoria. La línea horizontal de referencia $1$ marca el límite ortogonal y $m$ marca alineación perfecta.

**Conclusiones.** (1) Con etiquetas reales la coherencia inicial es muy alta ($\sim 10^4$) y decrece progresivamente al ir ajustando ejemplos, asentándose en $1$ cuando todos están ajustados, lo que valida la intuición de CG: datasets reales tienen alta alineación de gradientes, y esta alineación es la que sostiene la generalización. (2) Sorprendentemente, con etiquetas **completamente aleatorias** la coherencia inicial es baja (cerca del límite ortogonal, como CG predice) pero **aumenta** durante el entrenamiento hasta valores de orden $10^2$–$10^3$ entre las épocas 40-60 antes de decaer, indicando que las redes sobre-parametrizadas **crean** patrones comunes incluso cuando no hay nada que generalizar. (3) Existe un transitorio inicial ($\sim 25$ primeros pasos) donde la coherencia cae bruscamente desde casi $m$ hasta menos de $10$ en ambos casos, atribuible a la salida cuasi-uniforme inicial de la red. (4) Tras ese transitorio, ambos regímenes muestran una trayectoria parabólica amplia: la coherencia sube, alcanza un máximo y vuelve a $1$; la diferencia clave es la velocidad de subida, que es inversamente proporcional al ruido en etiquetas. (5) La trayectoria es consistente entre arquitecturas (ResNet vs Inception) y entre capas individuales, sugiriendo que el fenómeno no depende de una capa específica. (6) Las capas convolucionales muestran mayor coherencia que las fully-connected, coherente con el efecto de weight sharing como amplificador de coherencia (consistente con Corolario 3.1). (7) Conceptualmente, el paper separa optimización y generalización: SGD no sólo *explota* coherencia sino que la *crea*, lo que sugiere que CG necesita complementarse con una teoría de optimización de segundo orden que explique cómo emerge esa alineación. La *generalizabilidad* aparece como invariante inductivo que SGD intenta mantener paso a paso, condicionado al nivel de coherencia disponible localmente.

## Medición y pipeline

**Métrica concreta.** Se adopta la m-coherence $\alpha_m(w)$ de Chatterjee y Zielinski (2020). Partiendo de $\alpha := \mathbb{E}_{z, z'}[g_z \cdot g_{z'}] / \mathbb{E}_z[g_z \cdot g_z]$, y dado que sobre una muestra finita de tamaño $m$ se cumple $\sum_{z, z'} g_z \cdot g_{z'} = \|\sum_z g_z\|^2$, la métrica admite la forma escalable

$$\alpha_m = \frac{\|\sum_i g_i\|^2}{m \cdot \sum_i \|g_i\|^2}, \qquad \text{m-coherence} \equiv m \cdot \alpha_m,$$

equivalente al cociente entre el producto interior medio por pares (incluyendo el término $z = z'$) y la norma cuadrática media. Esta identidad reduce el cómputo de $O(m^2)$ (bucle sobre pares) a $O(m)$: basta acumular un vector y un escalar, sin enumerar pares.

**Entradas.** Gradientes por ejemplo $g_i \in \mathbb{R}^P$ (o por micro-batch consistente) sobre una muestra fija de $m$ ejemplos. Los estadísticos suficientes son $S = \sum_i g_i$ (vector $P$-dimensional) y $Q = \sum_i \|g_i\|^2$ (escalar). Ambos son actualizables en streaming, lo que permite no almacenar los $g_i$ individuales.

**Cuándo computar.** Cada época, o cada $K$ pasos en transitorios iniciales (donde el paper observa caídas bruscas en los primeros $\sim 25$ pasos). El coste reducido permite logging frecuente sin penalización apreciable.

**Coste.** Requiere $m$ gradientes por ejemplo, calculables eficientemente con `torch.func.vmap` + `grad` (functorch). Memoria: $O(P)$ para $S$ y $O(1)$ extra para $Q$; sin coste cuadrático en $m$.

**Integración en el pipeline.** Pseudocódigo de acumulación sin almacenar g_i:

```python
S = torch.zeros_like(flat_params)   # Σ g_i
Q = 0.0                             # Σ ||g_i||^2
n = 0
for xb, yb in coherence_loader:     # muestra fija de m ejemplos
    per_ex_grads = vmap(grad_fn)(params, xb, yb)   # (b, P)
    g = per_ex_grads.reshape(per_ex_grads.shape[0], -1)
    S += g.sum(dim=0)
    Q += (g * g).sum().item()
    n += g.shape[0]
alpha_m = (S.dot(S).item()) / (n * Q)
m_coherence = n * alpha_m
```

Si se registra *on-the-fly* durante el entrenamiento, pueden reutilizarse los gradientes del propio paso siempre que se mantenga consistencia de batch y modo del modelo.

**Consideraciones.** Conviene reportar $\alpha_m$ **global** y **por capa** (primera convolución, capa intermedia, capa final), siguiendo el protocolo del paper, ya que la señal por capa es más rica y revela el efecto de *weight sharing* (Corolario 3.1). El paper documenta dependencia con la tasa de aprendizaje, por lo que se debe registrar $\alpha_m$ a lo largo del sweep de LR. Para evitar ruido espurio se fija el modelo en modo `eval()` (o consistente) durante la medición, neutralizando BatchNorm y Dropout, y se utiliza siempre la misma muestra fija a lo largo del entrenamiento.

**Logging.** Se registra $\alpha_m$ escalar global por época, $\alpha_m$ por capa seleccionada, y opcionalmente la curva $\alpha_m$ frente a época/paso para visualizar el transitorio inicial y la trayectoria parabólica. Estos valores alimentan después la correlación con las métricas de eficiencia (épocas-hasta-umbral, AUC de la *test loss*, mejor *test loss*).

## Notes

### Uso en el TFG

- **Métrica que origina:** `m_coherence` (familia alineación), núcleo conceptual de toda la familia junto con la *Coherent Gradients Hypothesis*. Es la formalización escala-invariante y $O(m)$ de la coherencia per-sample.
- **Fórmula clave:** $\alpha_m = \dfrac{\|\sum_i g_i\|^2}{\sum_i \|g_i\|^2} = m\cdot\dfrac{\mathbb{E}_{z,z'}[g_z\cdot g_{z'}]}{\mathbb{E}_z[g_z\cdot g_z]} \in [1, m]$ ($1$ = gradientes ortogonales, $m$ = idénticos), sobre el gradiente bruto $\nabla L$.
- **Cómo se usa:** se mide $\alpha_m$ sobre un probe fijo $m \in [512, 2048]$ en las ventanas tempranas (5/10/25/50% de épocas) y se correlaciona (Spearman/Pearson) con eficiencia (épocas-a-umbral, AUC de test loss) y generalización. NO se optimiza. Comparte el barrido per-sample $\nabla L$ con `stiffness` y `gsnr`.
- **Señal:** ↑ mejor — alta coherencia temprana → convergencia rápida + mejor generalización. (Matiz del paper: con etiquetas aleatorias la coherencia *crece* a mitad de entrenamiento; en el TFG sin label noise se espera la trayectoria de etiquetas reales.)
- **Cómputo $O(m)$ streaming:** acumuladores $S=\sum_i g_i$ (vector $P$) y $Q=\sum_i\|g_i\|^2$ (escalar); $\alpha_m = \|S\|^2/Q$. No materializar la matriz $(m,P)$ (≈47 GB para $m{=}1024$, ResNet-18 fp32).
- **Pitfalls/decisiones:** SOLO per-sample — los mini-batches inflan la coherencia (Cor 3.1 del paper), nunca medir per-batch; forzar **fp32** ($Q$ hace underflow en fp16); la diagonal $z=z'$ se mantiene en la definición; `model.eval()` y misma muestra fija en todas las épocas. El recíproco $1/\alpha$ es la *gradient diversity* de las cotas de mini-batch SGD (Yin et al. 2018), útil para conectar con el eje teórico.

## Papers relacionados

- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — paper predecesor del mismo autor (Chatterjee 2019); `m_coherence` operacionaliza su *Coherent Gradients Hypothesis* sobre el término cruzado $\sum_{e\neq e'}\langle g_{te}, g_{te'}\rangle$.
- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — misma familia (alineación per-sample); el paper compara explícitamente $\alpha_m$ frente a sign/cosine stiffness (no-linealidades que dificultan ligar la métrica al cambio real en la loss). Comparte el barrido per-sample $\nabla L$.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — misma familia; el paper contrasta $\alpha_m$ con la *gradient confusion* de Sankararaman (estimador de extremo $z\neq z'$ vs. media incluyendo diagonal).
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — eje alineación/varianza per-sample sobre $\nabla L$; comparte el mismo barrido per-sample y mide coherencia/SNR del gradiente como proxy de generalización.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — mismo problema (proxy de alineación en tiempo de entrenamiento para generalización); comparable directo del delta del TFG.
- [[Disparity Between Batches as a Signal for Early Stopping]] — mismo problema (señal barata de gradientes para predecir generalización/early stopping), familia alineación.

## Otros papers interesantes a revisar

- **Gradient Diversity: a Key Ingredient for Scalable Distributed Learning** (Yin, Pananjady, Lam, Papailiopoulos, Ramchandran, Bartlett, 2018) — define la *gradient diversity*, exactamente el recíproco $1/\alpha$; conecta `m_coherence` con cotas de convergencia de mini-batch SGD y batch size. arXiv:1706.05699.
- **What Can Linearized Neural Networks Actually Say About Generalization?** (Ortiz-Jiménez, Moosavi-Dezfooli, Frossard, 2021) — relaciona alineación de gradientes/kernel con generalización en regímenes no-lazy; complementa la separación optimización/generalización que plantea el paper. arXiv:2106.06770.
- **Gradient Descent Quantizes ReLU Network Features** (Maennel, Bousquet, Gelly, 2018) — explica cómo SGD *crea* estructura/alineación común (eco del hallazgo de coherencia creciente con etiquetas aleatorias). arXiv:1803.08367.
- **An Investigation into Neural Net Optimization via Hessian Eigenvalue Density** (Ghorbani, Krishnan, Xiao, 2019) — curvatura de segundo orden que el paper señala como pieza ausente para explicar la *emergencia* de coherencia. arXiv:1901.10159.

## Cited By
