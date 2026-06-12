---
authors:
  - Diederik P. Kingma
  - Jimmy Ba
year: 2015
status: to-read
relevance: low
url: https://arxiv.org/pdf/1412.6980
tfg_role:
  - background
  - method
tfg_note: "Optimizador del sweep (SGD vs Adam), no métrica. Normaliza el gradiente por una estimación de su segundo momento —un SNR por parámetro—, intuición que motiva el eje varianza del TFG."
---

# Adam - A Method for Stochastic Optimization

## Summary

### Contextualización

El trabajo de Kingma y Ba se enmarca dentro de la optimización estocástica de funciones objetivo escalares y diferenciables, problema central en el aprendizaje profundo y, en general, en el aprendizaje automático moderno. La motivación parte de las limitaciones observadas en los métodos previos basados en gradiente de primer orden. El descenso por gradiente estocástico (SGD) clásico, aunque eficiente y de bajo coste de memoria, es muy sensible a la elección del learning rate y se comporta de forma uniforme sobre todos los parámetros, sin adaptarse a la geometría de la función de coste. Cuando la objetivo es no estacionaria, presenta gradientes ruidosos (por ejemplo, debidos a regularización por dropout) o tiene gradientes muy dispersos (típico en modelos sobre datos de alta dimensión y representaciones tipo bag-of-words), SGD sufre oscilaciones y convergencia lenta. AdaGrad (Duchi et al., 2011) introdujo un escalado por parámetro acumulando los cuadrados de todos los gradientes pasados, lo cual funciona bien con gradientes dispersos pero presenta una desventaja crítica: el denominador crece monótonamente, decayendo el paso efectivo hasta hacerse demasiado pequeño en regímenes no convexos y no estacionarios, deteniendo el aprendizaje prematuramente. RMSProp (Tieleman y Hinton, 2012) mitiga ese problema sustituyendo la suma acumulada por una media móvil exponencial del cuadrado del gradiente, recuperando la capacidad de adaptación a objetivos no estacionarios, pero carece de un término de corrección de sesgo en la inicialización y no incorpora momento del primer orden de forma directa sobre el gradiente original. Adam se propone como síntesis de las ventajas de AdaGrad sobre gradientes dispersos y RMSProp sobre objetivos no estacionarios, añadiendo además momento sobre el primer momento y una corrección explícita del sesgo de inicialización.

### Aportación

Los autores proponen Adam (Adaptive Moment Estimation), un algoritmo de optimización de primer orden que calcula learning rates adaptativos por parámetro a partir de estimaciones del primer y segundo momento (no centrado) del gradiente. Sus propiedades destacadas son requisitos de memoria modestos —solo dos vectores de tamaño igual al de los parámetros—, invariancia ante reescalados diagonales del gradiente, idoneidad para problemas grandes en datos y/o parámetros, robustez frente a objetivos no estacionarios y gradientes ruidosos o dispersos, e hiperparámetros con interpretación intuitiva que requieren poco ajuste. Además, los autores presentan una variante AdaMax basada en la norma infinito y demuestran formalmente una cota de regret comparable al mejor resultado conocido en optimización convexa online.

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

La sección 3 del paper deriva la corrección de sesgo. Como $v_t$ es una media móvil exponencial inicializada a cero, su esperanza es $\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) + \zeta$, donde $\zeta \approx 0$ si el segundo momento real es estacionario o si $\beta_2$ está adecuadamente elegido. Dividir por $(1 - \beta_2^t)$ corrige el sesgo hacia cero introducido por la inicialización, especialmente importante en los primeros pasos y cuando $\beta_2$ está próximo a uno (caso típico con gradientes dispersos). Una derivación análoga aplica al primer momento. Con los valores recomendados $\beta_1 = 0.9$ y $\beta_2 = 0.999$, la memoria efectiva del primer momento abarca aproximadamente los últimos diez pasos, mientras que el segundo momento integra una ventana de cerca de mil pasos (inferencia propia vía $1/(1-\beta)$, no un cálculo del paper), y la constante $\varepsilon = 10^{-8}$ evita divisiones por cero en regiones donde el segundo momento queda muy próximo a su origen.

La sección 2.1 analiza la magnitud del paso efectivo $\Delta_t = \alpha \cdot \hat{m}_t / \sqrt{\hat{v}_t}$. Adam impone dos cotas superiores: $|\Delta_t| \leq \alpha \cdot (1 - \beta_1) / \sqrt{1 - \beta_2}$ cuando $(1 - \beta_1) > \sqrt{1 - \beta_2}$, y $|\Delta_t| \leq \alpha$ en otros casos. Esto define una región de confianza alrededor del parámetro actual y facilita la elección de $\alpha$ a priori. La razón $\hat{m}_t / \sqrt{\hat{v}_t}$ se interpreta como un signal-to-noise ratio (SNR) —con signo en la §2.1 del paper, frente a la versión en valor absoluto $|\hat{m}_t|/\sqrt{\hat{v}_t}$ que se loguea más abajo como diagnóstico—: cuando es pequeña —es decir, cuando existe mayor incertidumbre sobre la dirección real del gradiente— el paso efectivo se reduce automáticamente, ofreciendo una forma de annealing automático cerca de óptimos. El paso es además invariante al reescalado del gradiente, propiedad que separa cualitativamente a Adam de SGD: mientras SGD aplica directamente $-\alpha g_t$ y depende fuertemente de la escala del gradiente, Adam normaliza paso a paso por la magnitud típica reciente, lo que estabiliza el régimen efectivo de aprendizaje a través de capas con escalas dispares.

Los hiperparámetros recomendados por defecto son $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\varepsilon = 10^{-8}$. La sección 4 proporciona el análisis de convergencia en el marco de aprendizaje convexo online de Zinkevich (2003). El regret se define como $R(T) = \sum_t [f_t(\theta_t) - f_t(\theta^*)]$. El Teorema 4.1 establece, bajo gradientes acotados $\|\nabla f_t\|_2 \leq G$, $\|\nabla f_t\|_\infty \leq G_\infty$, distancia entre iterados acotada por $D$ y $D_\infty$, $\beta_1^2 / \sqrt{\beta_2} < 1$, $\alpha_t = \alpha / \sqrt{t}$ y $\beta_{1,t} = \beta_1 \lambda^{t-1}$ con $\lambda \in (0,1)$, una cota de regret que combina términos en $\sqrt{T \hat{v}_{T,i}}$ y $\|g_{1:T,i}\|_2$. El Corolario 4.2 muestra que $R(T)/T = O(1/\sqrt{T})$, garantizando convergencia promedio. Cuando los datos son dispersos, la suma $\sum_i \|g_{1:T,i}\|_2$ puede ser significativamente menor que $d\, G_\infty \sqrt{T}$, alcanzando regret $O(\log d \cdot \sqrt{T})$, comparable al de AdaGrad y mejora sobre el $O(\sqrt{dT})$ de los métodos no adaptativos. La sección 7 presenta AdaMax, donde el segundo momento se reemplaza por una norma infinito recursiva $u_t = \max(\beta_2 \cdot u_{t-1},\, |g_t|)$, eliminando la necesidad de corregir sesgo del segundo momento y acotando $|\Delta_t| \leq \alpha$ de forma más simple.

### Datasets, modelos y métricas

Los experimentos abarcan tres familias de modelos. En regresión logística L2-regularizada multiclase sobre MNIST (clasificación de los diez dígitos a partir de vectores de 784 dimensiones, minibatches de 128, $\alpha_t = \alpha/\sqrt{t}$), Adam alcanza convergencia similar a SGD con momento de Nesterov y más rápida que AdaGrad. En el problema de clasificación de polaridad sobre IMDB con representaciones bag-of-words de 10.000 palabras y dropout del 50%, Adam converge tan rápido como AdaGrad, y este supera a SGD-Nesterov "by a large margin" (el paper no menciona a RMSProp en ese pasaje; en la Fig. 1 queda cerca del cluster Adam/AdaGrad). Para redes neuronales completamente conectadas con dos capas ocultas de 1.000 unidades ReLU sobre MNIST, con dropout y minibatches de 128, Adam muestra mejor convergencia que AdaGrad, RMSProp, SGD-Nesterov, AdaDelta y el quasi-Newton SFO (Sohl-Dickstein et al., 2014), siendo además 5 a 10 veces más rápido por iteración que SFO con menor coste de memoria. En redes convolucionales sobre CIFAR-10, con arquitectura c64-c64-c128-1000 (tres etapas alternantes de filtros 5x5 y max-pooling 3x3 con stride 2, seguidas de una capa densa de 1.000 ReLU), minibatches de 128 y dropout en la entrada y la capa densa, Adam y SGD convergen significativamente más rápido que AdaGrad. En este experimento los autores observan que $\hat{v}_t$ tiende a cero tras pocas épocas y queda dominado por $\varepsilon$, indicando que el segundo momento es una mala aproximación a la geometría del coste en CNNs y que el speed-up proviene principalmente de la reducción de varianza del minibatch a través del primer momento. Las métricas reportadas son training cost (negative log-likelihood en logística, cross-entropy en redes) y test error implícito en la convergencia. Un experimento adicional sobre un Variational Auto-Encoder (Kingma y Welling, 2013) con 500 unidades ocultas softplus y latente gaussiano de 50 dimensiones cuantifica el efecto del término de corrección de sesgo: con $\beta_2$ próximo a uno, omitir la corrección produce inestabilidades y mayor pérdida, especialmente en las primeras épocas.

### Conclusiones

Adam combina las virtudes de AdaGrad sobre gradientes dispersos y RMSProp sobre objetivos no estacionarios, añade momento sobre el primer momento, corrige explícitamente el sesgo de inicialización de las medias móviles exponenciales, exhibe interpretaciones intuitivas (región de confianza, SNR, annealing automático) y consigue una cota de regret $O(\sqrt{T})$ acorde al estado del arte en optimización convexa online. Empíricamente domina o iguala a métodos contemporáneos en regresión logística, MLPs y CNNs, escalando bien a problemas de alta dimensión con bajo coste de memoria. Es robusto a una amplia gama de problemas no convexos en aprendizaje automático.

## Medición y pipeline

**Rol metodológico.** En el contexto del TFG este paper no aporta una métrica de diagnóstico al `METRIC_REGISTRY`; define uno de los dos optimizadores del sweep, junto a SGD, frente al cual contrastar la robustez de las correlaciones métrica↔eficiencia. Se emplea con los hiperparámetros estándar recomendados por los autores: $\text{lr} = 10^{-3}$ (o el valor del sweep), $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\varepsilon = 10^{-8}$. Todas las métricas de alineación y varianza del estudio (cosine similarity, m-coherence, GSNR, normalized variance, stiffness) se calculan sobre el gradiente crudo $\nabla L(w)$ y nunca sobre el update preacondicionado $\hat{m}_t/\sqrt{\hat{v}_t}$, porque el preacondicionador define al optimizador y no forma parte de la señal medida; sin esa decisión, los valores bajo Adam y SGD dejarían de ser comparables y se rompería el análisis cross-optimizador.

**Entradas.** Los estados internos del optimizador en PyTorch son la única fuente de datos: `optimizer.state[p]['exp_avg']` corresponde al primer momento $m_t$ y `optimizer.state[p]['exp_avg_sq']` al segundo momento $v_t$. Tras `optimizer.step()` ambos tensores ya están actualizados y disponibles parámetro a parámetro, de modo que la lectura tiene coste cero adicional: los datos ya viven en el estado del optimizador.

**Magnitudes a loguear.** Por capa se registran tres familias de variables de control. Primero, el SNR por parámetro $\text{SNR} = |\hat{m}_t| / (\sqrt{\hat{v}_t} + \varepsilon)$ con bias correction $\hat{m}_t = m_t / (1 - \beta_1^t)$ y $\hat{v}_t = v_t / (1 - \beta_2^t)$, que materializa la cantidad que el paper interpreta como signal-to-noise ratio y motiva conceptualmente las métricas `gsnr` y `normalized_variance`. Segundo, las normas $\|m_t\|_2$ y $\|v_t\|_2$ por capa, útiles para diagnosticar saturación o explosión del preacondicionador. Tercero, el learning rate efectivo bias-corrected $\alpha \cdot \sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$ y la norma del update efectivo $\|\Delta\theta_t\| = \|\alpha \cdot \hat{m}_t/(\sqrt{\hat{v}_t}+\varepsilon)\|$.

**Granularidad temporal.** Por step durante la fase inicial del entrenamiento (cuando bias correction es relevante y los momentos aún están lejos del régimen estacionario) y por época en régimen estable. **Granularidad estructural.** Agregados por capa para las normas y el SNR medio, más un histograma global del SNR por parámetro para visualizar la distribución y detectar colas pesadas.

**Claves de logging sugeridas:** `adam/snr_layer/{name}`, `adam/m_norm/{name}`, `adam/v_norm/{name}`, `adam/lr_eff`, `adam/update_norm/{name}` y el histograma global `adam/snr_hist`.

**Pseudocódigo PyTorch.**

```python
# Tras optimizer.step()
t = optimizer.state[next(iter(model.parameters()))]['step']
bc1 = 1.0 - beta1 ** t
bc2 = 1.0 - beta2 ** t
lr_eff = lr * (bc2 ** 0.5) / bc1
log("adam/lr_eff", lr_eff)

snr_all = []
for name, p in model.named_parameters():
    state = optimizer.state[p]
    if 'exp_avg' not in state:
        continue
    m_t = state['exp_avg']           # primer momento
    v_t = state['exp_avg_sq']        # segundo momento (no centrado)
    m_hat = m_t / bc1
    v_hat = v_t / bc2
    snr = m_hat.abs() / (v_hat.sqrt() + eps)

    log(f"adam/snr_layer/{name}",  snr.mean().item())
    log(f"adam/m_norm/{name}",     m_t.norm().item())
    log(f"adam/v_norm/{name}",     v_t.norm().item())
    log(f"adam/update_norm/{name}", (lr * m_hat / (v_hat.sqrt() + eps)).norm().item())
    snr_all.append(snr.flatten())

log_hist("adam/snr_hist", torch.cat(snr_all))
```

**Gotchas.** El SNR que se loguea aquí usa el valor absoluto $|\hat{m}_t|/(\sqrt{\hat{v}_t}+\varepsilon)$, mientras que la §2.1 del paper define la razón con signo $\hat{m}_t/\sqrt{\hat{v}_t}$; no es una discrepancia, sino la elección deliberada de medir magnitud señal/ruido por coordenada. La corrección de sesgo es crítica durante los primeros pasos: con $\beta_2 = 0.999$, en $t = 10$ se tiene $1 - \beta_2^t \approx 10^{-2}$, así que $\hat{v}_t$ es dos órdenes de magnitud mayor que $v_t$ y olvidar la corrección produce SNRs aparentes inflados. Además, $v_t$ es un segundo momento **no centrado** ($\mathbb{E}[g^2]$, no $\text{Var}(g) = \mathbb{E}[g^2] - \mathbb{E}[g]^2$); por tanto la lectura "SNR = señal/ruido" es estricta solo cuando $\mathbb{E}[g] \approx 0$. Cuando la dirección del gradiente es persistente, $v_t$ sobrestima la varianza real y el SNR de Adam subestima la verdadera relación señal-ruido por parámetro: este sesgo es precisamente lo que `gsnr` (Liu et al., 2020) corrige al usar la varianza centrada $\rho^2(\theta_j)$.

**Snapshot/restore en la medición.** El estado interno de Adam acumula entre pasos, así que durante mediciones realizadas fuera del bucle de entrenamiento es necesario hacer snapshot de `optimizer.state_dict()` y restaurarlo al terminar; sin ese cuidado, el barrido de gradientes contaminaría $m_t$ y $v_t$ y el run divergiría tras cada ventana de medida. En SGD puro no aplica, pero el pipeline lo trata de forma uniforme para minimizar bifurcaciones.

## Notes
- Usable para comentar contexto de métodos adaptativos que usan la varianza del gradiente.
- Adam combina momentum (primer momento) con estimación del segundo momento no centrado (varianza no centrada) del gradiente para escalar el learning rate por parámetro.

### Uso en el TFG

Adam interviene en el TFG estrictamente como optimizador del sweep, no como métrica: no aporta entradas al `METRIC_REGISTRY` cerrado. Es uno de los dos regímenes optimizadores del barrido (SGD frente a Adam) y las diez métricas se computan bajo ambos para contrastar la robustez de las correlaciones métrica↔eficiencia. La razón $\hat{m}_t / \sqrt{\hat{v}_t}$, leída por Kingma y Ba como signal-to-noise ratio por parámetro (sección 2.1), proporciona la motivación conceptual del eje varianza del estudio: las métricas `gsnr` ($r(\theta_j) = \tilde{g}(\theta_j)^2 / \rho^2(\theta_j)$, Liu et al. 2020) y `normalized_variance` ($\mathbb{V}[g]/\mathbb{E}[g]^2$, inverso de un SNR; Faghri et al. 2020) son lecturas explícitas de la misma intuición señal/ruido que Adam usa implícitamente. Por coherencia con el análisis cross-optimizador, todas las métricas operan sobre el gradiente bruto $\nabla L(w)$ y nunca sobre la actualización preacondicionada. Las variables de control derivadas del estado interno —lr efectivo bias-corrected, normas de $m_t$ y $v_t$ por capa, norma del update— se loguean como diagnóstico complementario, no entran al registro de métricas.

## Papers relacionados

- [[RMSProp - Divide the gradient by a running average of its recent magnitude]] — antecedente directo: Adam hereda de RMSProp la media móvil exponencial de $g_t^2$ (segundo momento no centrado) que escala el paso por parámetro.
- [[An overview of gradient descent optimization algorithms]] — review que sitúa Adam en la taxonomía de optimizadores adaptativos; respaldo de la elección del sweep SGD + Adam.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — formaliza el GSNR $\tilde{g}^2/\rho^2$ por parámetro; convierte en métrica explícita el SNR $\hat{m}_t/\sqrt{\hat{v}_t}$ que Adam usa implícitamente.
- [[A Study of Gradient Variance in Deep Learning]] — define `normalized_variance` $\mathbb{V}[g]/\mathbb{E}[g]^2$ como inverso de un SNR; misma intuición señal/ruido que el preacondicionador de Adam.
- [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]] — SVRG; varianza del estimador del gradiente como objeto a controlar, complementario al SNR por parámetro de Adam.

## Otros papers interesantes a revisar

- **Decoupled Weight Decay Regularization (AdamW)** (Loshchilov & Hutter, 2019) — desacopla weight decay del paso adaptativo; variante de Adam dominante en la práctica y candidata natural si el sweep se ampliara. arXiv:1711.05101.
- **On the Convergence of Adam and Beyond (AMSGrad)** (Reddi, Kale & Kumar, 2018) — exhibe un fallo de convergencia de Adam por el segundo momento no monótono y propone AMSGrad; relevante para matizar la cota de regret del paper original. ICLR 2018, arXiv:1904.09237 (id arXiv sin verificar).
- **Adaptive Methods Generalize Worse Than SGD (The Marginal Value of Adaptive Gradient Methods in Machine Learning)** (Wilson et al., 2017) — evidencia empírica de peor generalización de métodos adaptativos frente a SGD; sustenta el contraste cross-optimizador del TFG. arXiv:1705.08292.
- **On the Variance of the Adaptive Learning Rate and Beyond (RAdam)** (Liu et al., 2020) — atribuye la inestabilidad temprana de Adam a la alta varianza del learning rate adaptativo en los primeros pasos; conecta directamente con la ventana de entrenamiento temprano del TFG. arXiv:1908.03265.
