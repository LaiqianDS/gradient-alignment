---
authors:
  - Diederik P. Kingma
  - Jimmy Ba
year: 2015
status: to-read
relevance: low
last_review: 2026-05-07
url: https://arxiv.org/pdf/1412.6980
tfg_role:
  - background
  - method
---

# Adam - A Method for Stochastic Optimization

## Summary

### Contextualización

El trabajo de Kingma y Ba se enmarca dentro de la optimización estocástica de funciones objetivo escalares y diferenciables, problema central en el aprendizaje profundo y, en general, en el aprendizaje automático moderno. La motivación parte de las limitaciones observadas en los métodos previos basados en gradiente de primer orden. El descenso por gradiente estocástico (SGD) clásico, aunque eficiente y de bajo coste de memoria, es muy sensible a la elección del learning rate y se comporta de forma uniforme sobre todos los parámetros, sin adaptarse a la geometría de la función de coste. Cuando la objetivo es no estacionaria, presenta gradientes ruidosos (por ejemplo, debidos a regularización por dropout) o tiene gradientes muy dispersos (típico en modelos sobre datos de alta dimensión y representaciones tipo bag-of-words), SGD sufre oscilaciones y convergencia lenta.

AdaGrad (Duchi et al., 2011) introdujo un escalado por parámetro acumulando los cuadrados de todos los gradientes pasados, lo cual funciona bien con gradientes dispersos pero presenta una desventaja crítica: el denominador crece monótonamente, decayendo el paso efectivo hasta hacerse demasiado pequeño en regímenes no convexos y no estacionarios, deteniendo el aprendizaje prematuramente. RMSProp (Tieleman y Hinton, 2012) mitiga ese problema sustituyendo la suma acumulada por una media móvil exponencial del cuadrado del gradiente, recuperando la capacidad de adaptación a objetivos no estacionarios, pero carece de un término de corrección de sesgo en la inicialización y no incorpora momento del primer orden de forma directa sobre el gradiente original. Adam se propone como síntesis de las ventajas de AdaGrad (gradientes dispersos) y RMSProp (no estacionariedad), añadiendo además momento sobre el primer momento y una corrección explícita del sesgo de inicialización.

### Aportación

Los autores proponen Adam (Adaptive Moment Estimation), un algoritmo de optimización de primer orden que calcula learning rates adaptativos por parámetro a partir de estimaciones del primer y segundo momento (no centrado) del gradiente. Sus propiedades destacadas son: requisitos de memoria modestos (solo dos vectores de tamaño igual al de los parámetros), invariancia ante reescalados diagonales del gradiente, idoneidad para problemas grandes en datos y/o parámetros, robustez frente a objetivos no estacionarios y gradientes ruidosos o dispersos, e hiperparámetros con interpretación intuitiva que requieren poco ajuste. Además, los autores presentan una variante AdaMax basada en la norma infinito y demuestran formalmente una cota de regret comparable al mejor resultado conocido en optimización convexa online.

### Metodología

El algoritmo Adam (Algoritmo 1 del paper) opera de la siguiente forma. Dados el stepsize $\alpha$, los coeficientes de decaimiento exponencial $\beta_1, \beta_2 \in [0,1)$, una constante de estabilidad numérica $\varepsilon$ y los parámetros iniciales $\theta_0$, se inicializan $m_0 \leftarrow 0$ (vector del primer momento) y $v_0 \leftarrow 0$ (vector del segundo momento), junto con $t \leftarrow 0$. En cada iteración:

$$
\begin{aligned}
& t \leftarrow t + 1 \\
& g_t \leftarrow \nabla_\theta f_t(\theta_{t-1}) \\
& m_t \leftarrow \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\
& v_t \leftarrow \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
& \hat{m}_t \leftarrow m_t / (1 - \beta_1^t) \\
& \hat{v}_t \leftarrow v_t / (1 - \beta_2^t) \\
& \theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)
\end{aligned}
$$

La sección 3 deriva la corrección de sesgo. Como $v_t$ es una media móvil exponencial inicializada a cero, su esperanza es

$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) + \zeta,$$

donde $\zeta \approx 0$ si el segundo momento real es estacionario o si $\beta_2$ está adecuadamente elegido. Dividir por $(1 - \beta_2^t)$ corrige el sesgo hacia cero introducido por la inicialización, especialmente importante en los primeros pasos y cuando $\beta_2$ está próximo a uno (caso típico con gradientes dispersos). Una derivación análoga aplica al primer momento.

La sección 2.1 analiza la magnitud del paso efectivo $\Delta_t = \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t}$. Adam impone dos cotas superiores: $|\Delta_t| \leq \alpha \cdot (1 - \beta_1) / \sqrt{1 - \beta_2}$ cuando $(1 - \beta_1) > \sqrt{1 - \beta_2}$, y $|\Delta_t| \leq \alpha$ en otros casos. Esto define una región de confianza alrededor del parámetro actual, lo que facilita la elección de $\alpha$ a priori. La razón $\hat{m}_t / \sqrt{\hat{v}_t}$ se interpreta como un signal-to-noise ratio (SNR): cuando es pequeña (mayor incertidumbre sobre la dirección real del gradiente) el paso efectivo se reduce automáticamente, ofreciendo una forma de annealing automático cerca de óptimos. El paso es además invariante al reescalado del gradiente.

Los hiperparámetros recomendados por defecto son $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\varepsilon = 10^{-8}$. La sección 4 proporciona el análisis de convergencia en el marco de aprendizaje convexo online de Zinkevich (2003). El regret se define como $R(T) = \sum_t [f_t(\theta_t) - f_t(\theta^*)]$. El Teorema 4.1 establece, bajo gradientes acotados $\|\nabla f_t\|_2 \leq G$, $\|\nabla f_t\|_\infty \leq G_\infty$, distancia entre iterados acotada por $D$ y $D_\infty$, $\beta_1^2 / \sqrt{\beta_2} < 1$, $\alpha_t = \alpha / \sqrt{t}$ y $\beta_{1,t} = \beta_1 \lambda^{t-1}$ con $\lambda \in (0,1)$, una cota de regret que combina términos en $\sqrt{T \hat{v}_{T,i}}$ y $\|g_{1:T,i}\|_2$. El Corolario 4.2 muestra que $R(T)/T = O(1/\sqrt{T})$, garantizando convergencia promedio. Cuando los datos son dispersos, la suma $\sum_i \|g_{1:T,i}\|_2$ puede ser significativamente menor que $d\, G_\infty \sqrt{T}$, alcanzando regret $O(\log d \cdot \sqrt{T})$, comparable al de AdaGrad y mejora sobre el $O(\sqrt{dT})$ de los métodos no adaptativos.

La sección 7 presenta AdaMax, donde el segundo momento se reemplaza por una norma infinito recursiva $u_t = \max(\beta_2 \cdot u_{t-1},\, |g_t|)$, eliminando la necesidad de corregir sesgo del segundo momento y acotando $|\Delta_t| \leq \alpha$ de forma más simple.

### Datasets, modelos y métricas

Los experimentos abarcan tres familias de modelos. En regresión logística L2-regularizada multiclase sobre MNIST (clasificación de los 10 dígitos a partir de vectores de 784 dimensiones, minibatches de 128, αₜ = α/√t), Adam alcanza convergencia similar a SGD con momento de Nesterov y más rápida que AdaGrad. En el problema de clasificación de polaridad sobre IMDB con representaciones bag-of-words de 10.000 palabras y dropout del 50%, Adam y AdaGrad logran el menor training cost, claramente por encima de RMSProp y SGD-Nesterov. Para redes neuronales completamente conectadas con dos capas ocultas de 1.000 unidades ReLU sobre MNIST, con dropout y minibatches de 128, Adam muestra mejor convergencia que AdaGrad, RMSProp, SGD-Nesterov, AdaDelta y el quasi-Newton SFO (Sohl-Dickstein et al., 2014), siendo además 5–10 veces más rápido por iteración que SFO con menor coste de memoria. En redes convolucionales sobre CIFAR-10, con arquitectura c64-c64-c128-1000 (tres etapas alternantes de filtros 5x5 y max-pooling 3x3 con stride 2, seguidas de una capa densa de 1.000 ReLU), minibatches de 128 y dropout en la entrada y la capa densa, Adam y SGD convergen significativamente más rápido que AdaGrad. En este experimento los autores observan que v̂ₜ tiende a cero tras pocas épocas y queda dominado por ε, indicando que el segundo momento es una mala aproximación a la geometría del coste en CNNs y que el speed-up proviene principalmente de la reducción de varianza del minibatch a través del primer momento. Las métricas reportadas son training cost (negative log-likelihood en logística, cross-entropy en redes) y test error implícito en la convergencia. Un experimento adicional sobre un Variational Auto-Encoder (Kingma y Welling, 2013) con 500 unidades ocultas softplus y latente gaussiano de 50 dimensiones cuantifica el efecto del término de corrección de sesgo: con β₂ próximo a 1, omitir la corrección produce inestabilidades y mayor pérdida, especialmente en las primeras épocas.

### Conclusiones

Adam combina las virtudes de AdaGrad (gradientes dispersos) y RMSProp (objetivos no estacionarios), añade momento sobre el primer momento, corrige explícitamente el sesgo de inicialización de las medias móviles exponenciales, exhibe interpretaciones intuitivas (región de confianza, SNR, annealing automático) y consigue una cota de regret O(√T) acorde al estado del arte en optimización convexa online. Empíricamente domina o iguala a métodos contemporáneos en regresión logística, MLPs y CNNs, escalando bien a problemas de alta dimensión con bajo coste de memoria. Es robusto a una amplia gama de problemas no convexos en aprendizaje automático.

## Medición y pipeline

En el contexto del TFG, este paper no aporta una métrica de diagnóstico sino que **define uno de los dos optimizadores del sweep** (junto a SGD). Se emplea con los hiperparámetros estándar recomendados por los autores: $\text{lr} = 10^{-3}$ (o el valor del sweep), $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\varepsilon = 10^{-8}$. Su rol metodológico es servir como segundo régimen optimizador frente al cual contrastar la robustez de las correlaciones métrica↔eficiencia obtenidas con SGD.

Para el análisis cross-optimizador conviene registrar como **variables de control**, paso a paso: (i) el lr efectivo bias-corrected $\alpha \cdot \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$, (ii) las normas medias por capa de $m_t$ (primer momento) y $v_t$ (segundo momento), útiles como diagnóstico de saturación o explosión del preacondicionador, y (iii) la norma del update efectivo $\Delta\theta_t = -\text{lr} \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$.

**Decisión metodológica clave**: las métricas de alineación y varianza del estudio (cosine similarity, m-coherence, GNS, NGV, stiffness) se calculan sobre el **gradiente crudo** $\nabla L(w)$ y no sobre el update preacondicionado de Adam, para garantizar comparabilidad directa entre SGD y Adam en el análisis de robustez.

Integración en el pipeline (PyTorch):

```python
for p in model.parameters():
    state = optimizer.state[p]
    if 'exp_avg' in state:
        m_t = state['exp_avg']           # primer momento
        v_t = state['exp_avg_sq']        # segundo momento
        log_layer_norm(p, m_t.norm(), v_t.norm())
```

Esta extracción se realiza tras `optimizer.step()` y se loggea por capa junto al gradiente crudo. **Consideración**: Adam suele estabilizar la convergencia pero la literatura reporta peor generalización que SGD en ciertos regímenes; comparar las correlaciones métrica↔eficiencia bajo ambos optimizadores forma parte central del análisis de robustez del TFG.

## Notes
- Usable para comentar contexto de métodos adaptativos que usan la varianza del gradiente.
- Adam combina momentum (primer momento) con estimación del segundo momento no centrado (varianza no centrada) del gradiente para escalar el learning rate por parámetro.

### Uso en el TFG

- **Rol: optimizador del sweep, NO métrica.** Adam no aporta ninguna métrica al `METRIC_REGISTRY` cerrado. Es uno de los dos regímenes optimizadores del barrido (SGD vs Adam); las 10 métricas se computan bajo **ambos** para contrastar la robustez de las correlaciones métrica↔eficiencia. Hiperparámetros estándar: $\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.
- **Razón $\hat{m}_t / \sqrt{\hat{v}_t}$ como SNR que motiva el eje varianza.** El paso de Adam es $\Delta\theta_t = -\alpha\,\hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$, donde la razón $\hat{m}_t / \sqrt{\hat{v}_t}$ (primer momento sobre raíz del segundo momento no centrado) se interpreta como un *signal-to-noise ratio* por parámetro (sección 2.1). Esta lectura es la motivación conceptual de las métricas de varianza del registro: `gsnr` ($r(\theta_j) = \tilde{g}(\theta_j)^2 / \rho^2(\theta_j)$, Liu et al. 2020) y `normalized_variance` ($\mathbb{V}[g]/\mathbb{E}[g]^2$, inverso de un SNR, Faghri et al. 2020).
- **Raw-grad rationale (justificación en methods).** Todas las métricas se calculan sobre el gradiente bruto $\nabla L(w)$ y **nunca** sobre la actualización preacondicionada $\hat{m}_t / \sqrt{\hat{v}_t}$. Sin esto, las métricas medidas bajo Adam no serían comparables con las de SGD y se rompería el análisis cross-optimizador: el preacondicionador define al optimizador, no es parte de la señal medida.
- **Snapshot/restore de $m_t, v_t$ en la medición (crítico).** El estado interno de Adam (`exp_avg` = $m_t$, `exp_avg_sq` = $v_t$) acumula entre pasos. Durante la medición fuera del bucle de entrenamiento hay que hacer snapshot de `optimizer.state_dict()` y restaurarlo al terminar; sin restore, el barrido de gradientes contaminaría el estado y el run divergiría tras cada ventana de medida. En SGD puro (sin estado) este cuidado no aplica, pero el pipeline lo trata de forma uniforme.
- **Variables de control cross-optimizador** (no son métricas del registro): lr efectivo bias-corrected $\alpha\sqrt{1-\beta_2^t}/(1-\beta_1^t)$, normas por capa de $m_t$ y $v_t$, y la norma del update $\|\Delta\theta_t\|$, útiles para diagnosticar saturación o explosión del preacondicionador.

## Papers relacionados

- [[RMSProp - Divide the gradient by a running average of its recent magnitude]] — antecedente directo: Adam hereda de RMSProp la media móvil exponencial de $g_t^2$ (segundo momento no centrado) que escala el paso por parámetro.
- [[An overview of gradient descent optimization algorithms]] — review que sitúa Adam en la taxonomía de optimizadores adaptativos; respaldo de la elección del sweep SGD + Adam.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — formaliza el GSNR $\tilde{g}^2/\rho^2$ por parámetro; convierte en métrica explícita el SNR $\hat{m}_t/\sqrt{\hat{v}_t}$ que Adam usa implícitamente.
- [[A Study of Gradient Variance in Deep Learning]] — define `normalized_variance` $\mathbb{V}[g]/\mathbb{E}[g]^2$ como inverso de un SNR; misma intuición señal/ruido que el preacondicionador de Adam.
- [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]] — SVRG; varianza del estimador del gradiente como objeto a controlar, complementario al SNR por parámetro de Adam.

## Otros papers interesantes a revisar

- **Decoupled Weight Decay Regularization (AdamW)** (Loshchilov & Hutter, 2019) — desacopla weight decay del paso adaptativo; variante de Adam dominante en la práctica y candidata natural si el sweep se ampliara. arXiv:1711.05101.
- **On the Convergence of Adam and Beyond (AMSGrad)** (Reddi, Kale & Kumar, 2018) — exhibe un fallo de convergencia de Adam por el segundo momento no monótono y propone AMSGrad; relevante para matizar la cota de regret del paper original. ICLR 2018, arXiv:1904.09237.
- **Adaptive Methods Generalize Worse Than SGD (The Marginal Value of Adaptive Gradient Methods in Machine Learning)** (Wilson et al., 2017) — evidencia empírica de peor generalización de métodos adaptativos frente a SGD; sustenta el contraste cross-optimizador del TFG. arXiv:1705.08292.
- **On the Variance of the Adaptive Learning Rate and Beyond (RAdam)** (Liu et al., 2020) — atribuye la inestabilidad temprana de Adam a la alta varianza del learning rate adaptativo en los primeros pasos; conecta directamente con la ventana de entrenamiento temprano del TFG. arXiv:1908.03265.
