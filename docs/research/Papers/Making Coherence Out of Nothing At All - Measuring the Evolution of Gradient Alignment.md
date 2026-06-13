---
authors: ["Satrajit Chatterjee", "Piotr Zielinski"]
year: 2020
status: to-read
relevance: high
url: "https://arxiv.org/pdf/2008.01217"
tfg_role:
  - metric
tfg_note: "Origen de `m_coherence`: formalización escalable y O(m) de la Coherent Gradients Hypothesis (coherencia per-sample normalizada, rango [0, m]; 1 = límite ortogonal). Núcleo de la familia alineación."
---

# Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment

[arXiv](https://arxiv.org/pdf/2008.01217)

## Summary

### Contextualización

El trabajo se sitúa en la línea abierta por la *Coherent Gradients hypothesis* (CG) que Chatterjee formula en ICLR 2020. CG propone una explicación unificada de memorización y generalización en redes profundas entrenadas con SGD: como el gradiente del minibatch es la suma de gradientes por ejemplo, los pasos de SGD se sesgan hacia direcciones que reducen la pérdida sobre múltiples ejemplos simultáneamente cuando dichas direcciones existen. En el caso extremo de gradientes por ejemplo perfectamente alineados se obtendría estabilidad algorítmica y, por tanto, generalización; en el extremo de gradientes mutuamente ortogonales no habría estabilidad y se produciría memorización pura. Este paper de Chatterjee y Zielinski (2020) es un follow-up empírico que construye la herramienta de medición que faltaba para validar y matizar CG en la práctica. Aborda dos preguntas abiertas: qué aspecto tiene la coherencia en datasets y arquitecturas reales, y cómo evoluciona a lo largo de toda la trayectoria de entrenamiento y no únicamente al inicio.

### Aportación

La contribución central es la introducción de la **m-coherence**, denotada $\alpha_m$, como métrica de alineamiento de gradientes por ejemplo con una interpretación intuitiva directa: sobre una muestra de tamaño $m$, m-coherence indica el número medio de ejemplos (incluyéndose a sí mismo) que se benefician de un pequeño paso a lo largo del gradiente de un ejemplo aleatorio. Frente a métricas previas como el producto interior por pares, la sign/cosine stiffness de Fort et al. (2019) o la gradient confusion de Sankararaman et al. (2019), $\alpha_m$ resulta más interpretable, matemáticamente más limpia, escala-invariante en el sentido apropiado y, sobre todo, computacionalmente eficiente. Requiere $O(m)$ operaciones en lugar de $O(m^2)$, lo que permite utilizar muestras dos órdenes de magnitud mayores. Los autores también señalan que el recíproco de $\alpha$ (literal del paper: *"the reciprocal of α appears in the theory literature as gradient diversity"*) aparece en cotas teóricas previas (Yin et al. 2018, Jain et al. 2018). Con el factor $m$ explícito, la $\Delta_S(w) = \sum_i\|g_i\|^2/\|\sum_i g_i\|^2$ de Yin et al. es exactamente $1/\alpha_m$, lo que conecta la métrica con la literatura sobre convergencia de mini-batch SGD.

### Metodología

Partiendo de la expansión de Taylor de la pérdida por ejemplo $\ell_z(w)$ alrededor de $w$ tras un paso $h_z = -\eta\, g_z$, los autores definen

$$\alpha := \frac{\mathbb{E}[g_z \cdot g]}{\mathbb{E}[g_z \cdot g_z]} = \frac{\mathbb{E}_{z, z'}[g_z \cdot g_{z'}]}{\mathbb{E}_z[g_z \cdot g_z]}.$$

Esta cantidad $\alpha \in [0, 1]$ admite una lectura como cociente entre el cambio real en la pérdida global tras un paso de gradiente y el cambio máximo posible si cada $\ell_z$ pudiera optimizarse independientemente. En el límite ortogonal, donde los gradientes son mutuamente perpendiculares con misma norma, se cumple $\alpha = 1/m$; en el extremo opuesto, con gradientes idénticos, $\alpha = 1$. Multiplicando por $m$ se obtiene la m-coherence $\alpha_m = m\cdot\alpha$, que vale $1$ en el caso ortogonal y $m$ en el caso de alineación perfecta. La sección 4 del paper generaliza la noción a cualquier distribución de vectores en un espacio euclídeo y demuestra propiedades clave: la acotación $0 \leq \alpha(V) \leq 1$ (Teorema 1), la invariancia de escala $\alpha(kV) = \alpha(V)$ (Lema 2), la amplificación por mini-batching $\alpha(W) \geq \alpha(V)$ con $W = (1/k)\sum v_i$ (Teorema 3 y Corolario 3.1, que muestra que agregar gradientes en mini-batches infla la coherencia y obliga a medirla siempre sobre gradientes por ejemplo) y el efecto de gradientes nulos $\alpha(W) = p\cdot\alpha(V)$ (Lema 4, que justifica que los ejemplos ya ajustados, con gradiente cercano a cero, diluyen la coherencia).

El algoritmo eficiente explota la identidad $\mathbb{E}[g_z \cdot g] = \mathbb{E}_z[g_z]\cdot\mathbb{E}_z[g_z] = \|g\|^2$: basta mantener dos sumas corrientes —el gradiente acumulado $g$ y la suma de cuadrados $\sum\|g_z\|^2$— sobre el sample, sin necesidad de almacenar los gradientes por ejemplo, lo que habilita un cómputo en streaming. El contraste con stiffness y gradient confusion es explícito: las no-linealidades de aquellas métricas (signo, normalización por norma, mínimo) y la exclusión de la diagonal $z = z'$ hacen difícil ligar su valor al cambio real de la pérdida tras un paso. En cambio, $\alpha$ se conecta exactamente con la reducción de pérdida prevista por la expansión de Taylor.

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas

Se sigue la m-coherence sobre el modelo completo y, separadamente, sobre tres capas representativas —primera convolución, capa intermedia y capa fully-connected final— junto con loss y top-1 accuracy en train y test, a lo largo de toda la trayectoria. La línea horizontal de referencia $\alpha_m = 1$ marca el límite ortogonal y $\alpha_m = m$ marca alineación perfecta, lo que facilita la lectura visual de las curvas.

### Conclusiones

Los resultados articulan varios hallazgos. Con etiquetas reales la coherencia inicial es muy alta, del orden de $10^4$, y decrece progresivamente conforme la red ajusta ejemplos, asentándose en $1$ cuando todos están aprendidos. Este patrón valida la intuición de CG según la cual los datasets reales presentan alta alineación de gradientes y esa alineación sostiene la generalización. Sorprendentemente, con etiquetas completamente aleatorias la coherencia inicial es baja, cercana al límite ortogonal como CG predice, pero **aumenta** durante el entrenamiento hasta valores del orden de $10^2$–$10^3$ entre las épocas 40 y 60 antes de decaer. Esto indica que las redes sobreparametrizadas *crean* patrones comunes incluso cuando no hay nada que generalizar. Existe además un transitorio inicial de aproximadamente 25 pasos en el que la coherencia cae bruscamente desde valores cercanos a $m$ hasta menos de $10$ en ambos casos, atribuible a la salida cuasi-uniforme de la red al comienzo. Tras ese transitorio ambos regímenes describen una trayectoria parabólica amplia —la coherencia sube, alcanza un máximo y vuelve a $1$— y la diferencia clave es la velocidad de subida, inversamente proporcional al ruido en etiquetas. La trayectoria es consistente entre arquitecturas (ResNet vs Inception) y entre capas, lo que sugiere que el fenómeno no depende de una capa específica. Las capas convolucionales muestran mayor coherencia que las fully-connected, coherente con el efecto amplificador del weight sharing predicho por el Corolario 3.1. Conceptualmente, el paper separa optimización y generalización: SGD no sólo *explota* la coherencia sino que la *crea*, lo que sugiere que CG debe complementarse con una teoría de optimización de segundo orden que explique cómo emerge esa alineación. La *generalizabilidad* aparece como invariante inductivo que SGD intenta preservar paso a paso, condicionado al nivel de coherencia disponible localmente.

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** Se adopta la m-coherence $\alpha_m(w)$ de Chatterjee y Zielinski (2020) sobre el gradiente bruto $\nabla L$. Partiendo de la definición $\alpha := \mathbb{E}_{z, z'}[g_z \cdot g_{z'}] / \mathbb{E}_z[g_z \cdot g_z]$, y dado que sobre una muestra finita de tamaño $m$ se cumple $\sum_{z, z'} g_z \cdot g_{z'} = \|\sum_z g_z\|^2$, la métrica admite la forma escalable

$$\alpha_m = \frac{\|\sum_i g_i\|^2}{\sum_i \|g_i\|^2}, \qquad \text{m-coherence} \equiv m\cdot\alpha,$$

equivalente al cociente entre el producto interior medio por pares (incluyendo el término $z = z'$) y la norma cuadrática media. La identidad $\mathbb{E}[g_z\cdot g] = \|g\|^2$ es la pieza clave que reduce el cómputo de $O(m^2)$ —el bucle ingenuo sobre pares— a $O(m)$: basta acumular un vector y un escalar, sin enumerar pares ni materializar la matriz $N\times N$.

**Ejemplos canónicos.** Los dos casos límite anclan la lectura de cualquier curva de $\alpha_m$. Cuando los $m$ gradientes por ejemplo son mutuamente ortogonales y de misma norma, $\sum_i g_i$ tiene magnitud $\sqrt{m}\,\|g_i\|$ y el numerador iguala al denominador, dando $\alpha_m = 1$: ningún ejemplo aprovecha el paso de otro y el régimen es de pura memorización. Cuando todos los gradientes son idénticos, $\sum_i g_i = m\,g$ y $\alpha_m = m$: un único paso reduce la pérdida en los $m$ ejemplos simultáneamente, régimen de alineamiento perfecto que es el techo teórico. Durante el entrenamiento real con etiquetas verdaderas, $\alpha_m$ parte cerca del techo, decae rápidamente en los primeros pasos por el transitorio inicial, recorre una trayectoria parabólica amplia y termina asentándose en $1$ a medida que todos los ejemplos quedan ajustados.

**Entradas.** Gradientes por ejemplo $g_i = \nabla_W \ell(x_i; W) \in \mathbb{R}^P$ sobre una muestra fija de $m$ ejemplos, mantenida constante a lo largo del entrenamiento. Los estadísticos suficientes son el vector $S = \sum_i g_i \in \mathbb{R}^P$ y el escalar $Q = \sum_i \|g_i\|^2$, ambos actualizables en streaming, lo que evita almacenar los $g_i$ individuales y mantener una matriz $(m, P)$ en memoria.

**Cálculo eficiente $O(m)$.** El estimador escalable se reescribe como $\alpha_m = m\,\|S\|^2 / (m\cdot Q) = \|S\|^2 / Q$ tras la cancelación, lo que evidencia que el cómputo es lineal en $m$ y en $P$ y trivialmente paralelizable. La equivalencia $\mathbb{E}[g_z\cdot g] = \|g\|^2$ debe validarse al integrar la métrica (ver avisos más abajo): es la identidad que justifica todo el ahorro.

**Granularidad temporal.** Se registra $\alpha_m$ una vez por época durante el grueso del entrenamiento; en la ventana inicial (primeras 5-10 épocas), cada $K$ pasos para muestrear con resolución el transitorio de unos 25 pasos que el paper documenta, donde $\alpha_m$ cae bruscamente desde valores cercanos a $m$ hasta menos de $10$. El coste reducido permite logging frecuente sin penalización apreciable.

**Granularidad estructural.** Se reporta $\alpha_m$ global y por capa, siguiendo el protocolo del paper sobre tres capas representativas: primera convolución, capa intermedia y fully-connected final. La señal por capa es más rica y revela efectos como el weight sharing convolucional (consistente con el Corolario 3.1) que el escalar global agrega.

**Coste.** Requiere $m$ gradientes por ejemplo, calculables eficientemente con `torch.func.vmap` combinado con `grad` (functorch). Tamaño típico $m = 1024$. Memoria: $O(P)$ para $S$ y $O(1)$ extra para $Q$; sin coste cuadrático en $m$. Comparado con la matriz $(m, P)$ que necesitaría una formulación naïve, el ahorro es decisivo (≈47 GB para $m = 1024$ y ResNet-18 fp32 frente a $O(P)$ del estimador streaming).

**Claves de logging.** La v1 emite solo `mcoh/global`; `mcoh/per_layer/{name}` (primera convolución, intermedia y FC final, el protocolo per-layer del paper) queda reservada para v2. De forma complementaria pueden registrarse las trazas frente a paso/época para visualizar el transitorio inicial y la trayectoria parabólica.

**Interpretación de la señal.** La convención es **↑ m-coherence = mejor durante la fase de fit**, alineada con el resto de métricas de alineamiento del TFG y con la *Coherent Gradients Hypothesis*: valores altos de $\alpha_m$ indican que los gradientes per-sample comparten dirección, lo que se traduce en que un paso de SGD reduce simultáneamente la pérdida sobre múltiples ejemplos y, por tanto, en señal generalizable. La lectura, sin embargo, no es monótona en el tiempo, sino **parabólica**: en `mcoh/global` se espera un ascenso rápido durante el aprendizaje de la estructura común (cuanto antes y más alto el pico, mejor *trainability*), un máximo que marca la transición desde fit hacia memorización, y un descenso posterior hacia $\alpha_m \approx 1$ —el régimen ortogonal— a medida que los gradientes residuales se vuelven mutuamente ortogonales sobre los ejemplos aún no ajustados. Un descenso prematuro o un pico bajo son señales tempranas de pobre generalización. La traza por capa `mcoh/per_layer/{name}` enriquece la lectura: la heterogeneidad es informativa, porque permite localizar qué capas siguen extrayendo estructura coherente y cuáles ya han transitado a memorización. La expectativa es que las convolucionales sostengan coherencia más alta que las fully-connected por el efecto amplificador del weight sharing (Corolario 3.1). Aviso crítico de interpretación: la regla ↑ mejor solo es válida si se mide sobre **gradientes per-sample**. Una `mcoh/global` sospechosamente alta o que no decae en absoluto es síntoma probable de estar midiendo sobre gradientes per-batch, lo que infla artificialmente $\alpha_m$ por el mismo Corolario 3.1 y debe diagnosticarse antes de interpretar la curva.

**Avisos.** Tres puntos críticos. Primero, la métrica requiere **gradientes per-sample**: los gradientes de mini-batch inflan $\alpha_m$ por el Corolario 3.1 del paper, lo que invalida cualquier comparación; nunca medir per-batch. Segundo, conviene **validar empíricamente la equivalencia** $\mathbb{E}[g_z\cdot g] = \|g\|^2$ en una pasada inicial (acumulando explícitamente $\sum_z g_z \cdot g$ y comparando con $\|S\|^2/m$) antes de confiar en el estimador streaming. Tercero, **fijar precisión fp32** durante la acumulación: $Q = \sum \|g_z\|^2$ sufre underflow en fp16 cuando $P$ es grande. Además, debe fijarse `model.eval()` (o un modo consistente) para neutralizar BatchNorm y Dropout y reutilizarse la misma muestra fija en todas las épocas.

**Integración en el pipeline.** Pseudocódigo PyTorch del estimador streaming, sin materializar los gradientes por ejemplo:

```python
S = torch.zeros_like(flat_params)   # Σ g_i
Q = 0.0                             # Σ ||g_i||^2
n = 0
for xb, yb in coherence_loader:     # muestra fija de m ejemplos
    per_ex_grads = vmap(grad_fn)(params, xb, yb)   # (b, P) per-sample
    g = per_ex_grads.reshape(per_ex_grads.shape[0], -1)
    S += g.sum(dim=0)
    Q += (g * g).sum().item()
    n += g.shape[0]
alpha_m = (S.dot(S).item()) / Q     # = ||S||^2 / Q  (escala-invariante)
m_coherence = alpha_m               # ya está en escala [0, m]; 1 = límite ortogonal
```

Si se registra *on-the-fly* durante el entrenamiento, pueden reutilizarse los gradientes per-sample del propio paso siempre que se mantenga consistencia de batch y modo del modelo.

## Notes

### Uso en el TFG

- **Métrica que origina.** `m_coherence` (familia alineación), núcleo conceptual de toda la familia junto con la *Coherent Gradients Hypothesis*. Es la formalización escala-invariante y $O(m)$ de la coherencia per-sample.
- **Fórmula clave.** $\alpha_m = \|\sum_i g_i\|^2 / \sum_i \|g_i\|^2 = m\cdot \mathbb{E}_{z,z'}[g_z\cdot g_{z'}]/\mathbb{E}_z[g_z\cdot g_z] \in [0, m]$, sobre el gradiente bruto $\nabla L$. El Teorema 1 da $0 \le \alpha \le 1$, así que $1$ es el **límite ortogonal** (no una cota inferior); $m$ corresponde a gradientes idénticos y los valores $< 1$ indican gradientes anticorrelados, que el paper observa en la primera conv tras alcanzar el 100% de train accuracy.
- **Cómo se usa.** Se mide $\alpha_m$ sobre un probe fijo de $m \in [512, 2048]$ en las ventanas tempranas (5/10/25/50% de épocas) y se correlaciona (Spearman/Pearson) con métricas de eficiencia (épocas-hasta-umbral, AUC de test loss) y generalización. No se optimiza. Comparte el barrido per-sample $\nabla L$ con `stiffness` y `gsnr`.
- **Señal.** Mayor $\alpha_m$ es mejor: alta coherencia temprana se asocia con convergencia más rápida y mejor generalización. (Matiz del paper: con etiquetas aleatorias la coherencia crece a mitad de entrenamiento; en el TFG sin label noise se espera la trayectoria de etiquetas reales.)
- **Cómputo $O(m)$ streaming.** Acumuladores $S = \sum_i g_i$ (vector $P$) y $Q = \sum_i \|g_i\|^2$ (escalar); $\alpha_m = \|S\|^2/Q$. Idealmente sin materializar la matriz $(m, P)$ (≈47 GB para $m = 1024$, ResNet-18 fp32); **la v1 real sí la materializa** (probe compartido $M=256$ vía `per_sample_grad_matrix`) y emite solo `mcoh/global` — el streaming y la versión per-layer quedan como optimización/v2.
- **Pitfalls y decisiones.** Solo per-sample —los mini-batches inflan la coherencia (Cor 3.1), nunca medir per-batch—; forzar fp32 ($Q$ hace underflow en fp16); la diagonal $z = z'$ se mantiene en la definición; `model.eval()` y misma muestra fija en todas las épocas. Sobre la *gradient diversity*: el paper dice literalmente que $1/\alpha$ es la que aparece en la literatura teórica; la $\Delta_S$ de Yin et al. (2018) es $1/\alpha_m$ (difieren en un factor $m$) — útil para conectar con el eje teórico manteniendo el factor explícito.

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
