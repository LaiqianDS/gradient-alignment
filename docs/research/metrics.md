# Métricas por Paper

Resumen comparativo de las métricas utilizadas en los 16 papers leídos del TFG. Cada entrada distingue entre la métrica central propuesta o estudiada por el paper y las métricas usadas para la evaluación empírica.

---

## A Study of Gradient Variance in Deep Learning (Faghri et al., 2020)

**Métrica(s) central(es):**
- **Average Variance** — promedio sobre las coordenadas del parámetro de la varianza del estimador del gradiente medio. Equivale a la traza normalizada de la matriz de covarianza del gradiente.
- **Normalized Variance** — $\mathbb{V}[g] / \mathbb{E}[g]^2$. Se interpreta como el inverso de un signal-to-noise ratio (SNR). Si supera 1, el ruido domina sobre la señal del gradiente. Es comparable entre problemas de distinta escala, a diferencia de la varianza absoluta.

**Métricas de evaluación:**
- Training loss — frente al número de iteraciones.
- Accuracy — exactitud sobre el conjunto de entrenamiento/test.
- Varianza máxima en las últimas iteraciones — usada en los RF models para evaluar el régimen sobreparametrizado.

---

## A Theory of Neural Tangent Kernel Alignment and Its Influence on Training (Shan & Bordelon, 2021)

**Métrica(s) central(es):**
- **Kernel Alignment** $A(t)$ — métrica de Cortes et al. (2012):
$$A(t) = \frac{\langle \boldsymbol{y}\boldsymbol{y}^\top, \boldsymbol{K}(\boldsymbol{\theta}) \rangle_F}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F\, \|\boldsymbol{y}\boldsymbol{y}^\top\|_F} = \frac{\boldsymbol{y}^\top \boldsymbol{K}(\boldsymbol{\theta}) \boldsymbol{y}}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F\, \|\boldsymbol{y}\|^2}.$$
Mide la proyección del NTK sobre la dirección del target $\boldsymbol{y}\boldsymbol{y}^\top$.
- **Kernel Specialization Matrix (KSM)** — $\text{KSM}(c,d) = A(\boldsymbol{K}^{c,c}, \boldsymbol{y}_d \boldsymbol{y}_d^\top) / [C^{-1} \sum_{d'} A(\boldsymbol{K}^{d',d'}, \boldsymbol{y}^d \boldsymbol{y}^{d\top})]$. Detecta la alineación preferente de cada subkernel diagonal con su propio target en clasificación multiclase.

**Métricas de evaluación:**
- Pérdida de entrenamiento $\mathcal{L}$ — dinámica de convergencia.
- Norma de Frobenius $\|\boldsymbol{K}\|_F$ — magnitud global del NTK.
- Forma bilineal $B(\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{K} \boldsymbol{z}$ y traza $\text{Tr}(\boldsymbol{K})$ — medida de anisotropía.
- Cosine similarity — entre pesos red/profesor y entre autofunción dominante del NTK y target.
- Comparativas NN vs. KGD vs. aKGD — aislando efectos de estructura frente a magnitud.

---

## Accelerating Stochastic Gradient Descent using Predictive Variance Reduction (Johnson & Zhang, 2013)

**Métrica(s) central(es):**
- No propone una nueva métrica de diagnóstico. Introduce el método SVRG y analiza la **varianza del incremento de pesos** $-\eta(\nabla \psi_i(w) - \nabla \psi_i(\tilde{w}) + \tilde{\mu})$ como objeto a reducir explícitamente para acelerar la convergencia.

**Métricas de evaluación:**
- Training loss $P(w)$ — frente al número de gradientes calculados normalizado por $n$.
- Suboptimalidad $P(w) - P(w_*)$ — residuo de pérdida respecto al óptimo estimado.
- Varianza del estimador estocástico — comparada entre SVRG, SGD y SDCA.
- Test error rate — sobre los conjuntos de test (con splits 50/50 en protein y covtype).

---

## Adam - A Method for Stochastic Optimization (Kingma & Ba, 2015)

**Métrica(s) central(es):**
- No propone una métrica nueva. Adam utiliza dos magnitudes internas para escalar el paso: el primer momento $m_t$ (media móvil exponencial del gradiente) y el segundo momento no centrado $v_t$ (media móvil exponencial de $g_t^2$, proxy de la varianza por parámetro). La razón $\hat{m}_t / \sqrt{\hat{v}_t}$ se interpreta como signal-to-noise ratio.
- Cota de **regret** en marco convexo online de Zinkevich (2003): $R(T) = \sum_t [f_t(\theta_t) - f_t(\theta^*)]$, con $R(T)/T = O(1/\sqrt{T})$.

**Métricas de evaluación:**
- Training cost — negative log-likelihood en regresión logística, cross-entropy en redes.
- Test error — implícito en las curvas de convergencia.
- Convergencia frente a SGD-Nesterov, AdaGrad, RMSProp, AdaDelta y SFO — sobre MNIST, IMDB, MLP MNIST, CNN CIFAR-10 y VAE.

---

## An Empirical Model of Large-Batch Training (McCandlish et al., 2018)

**Métrica(s) central(es):**
- **Gradient Noise Scale** $B_{\text{noise}}$ — versión exacta ponderada por la Hessiana:
$$B_{\text{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G}.$$
- **Simple Gradient Noise Scale** $B_{\text{simple}}$ — aproximación bajo $H \propto I$:
$$B_{\text{simple}} = \frac{\operatorname{tr}(\Sigma)}{\|G\|^2}.$$
Predice el batch size crítico $B_{\text{crit}}$ a partir del cual el speed-up de paralelización deja de ser lineal.

**Métricas de evaluación:**
- Training steps $S$ y ejemplos procesados $E$ — hasta alcanzar un rendimiento objetivo.
- Curva de Pareto hiperbólica $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$.
- $B_{\text{crit}} = E_{\min}/S_{\min}$ — definida como el punto donde la eficiencia de cómputo cae al 50%.
- Error de clasificación, perplejidad, recompensa de episodio, TrueSkill (Dota) — métricas de rendimiento específicas por dominio.

---

## An overview of gradient descent optimization algorithms (Ruder, 2017)

**Métrica(s) central(es):**
- No propone métricas nuevas. Se trata de un review que cataloga optimizadores (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam) sin estudio empírico propio.

**Métricas de evaluación:**
- Velocidad de convergencia (cualitativa) — aproximación al mínimo.
- Capacidad de escapar saddle points.
- Estabilidad frente a oscilaciones.
- Eficiencia computacional — coste por actualización.
- Robustez frente a sparsity de los datos e inicializaciones pobres.
No se reportan accuracy ni curvas cuantitativas: comparaciones cualitativas y referencias externas.

---

## Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization (Chatterjee, 2019)

**Métrica(s) central(es):**
- **Fracción de la reducción de pérdida atribuible a pristine/corrupt** — $f_t^p = \langle g_t, g_t^p\rangle/\langle g_t, g_t\rangle$ y $f_t^c = \langle g_t, g_t^c\rangle/\langle g_t, g_t\rangle$, con $f_t^p + f_t^c = 1$. Cuantifica la coherencia operativa entre subconjuntos del minibatch.
- Versiones acumuladas $i_t^p$ y $i_t^c$ — integrales temporales de las fracciones anteriores.
- **Hipótesis Coherent Gradients (CGH)** — articulada vía $\|g_t\|^2 = \sum_e \|g_{te}\|^2 + \sum_{e\neq e'}\langle g_{te}, g_{te'}\rangle$, donde el segundo término captura el alineamiento entre gradientes per-sample.

**Métricas de evaluación:**
- Training accuracy y test/validation accuracy.
- Training loss.
- Fracción de ejemplos aprendidos frente al paso.
- Overfit ajustado — $ta - [\varepsilon \cdot (1/10) + (1-\varepsilon)\cdot va]$, corrige que las etiquetas de test no estén aleatorizadas.
- Comparación con modelo nulo (permutaciones aleatorias) para significación estadística.

---

## Disparity Between Batches as a Signal for Early Stopping (Forouzesh & Thiran, 2021)

**Métrica(s) central(es):**
- **Gradient Disparity (GD)** — distancia $\ell_2$ entre gradientes de dos mini-batches independientes:
$$\mathcal{D}_{i,j} = \|g_i - g_j\|_2.$$
Deriva de un upper bound PAC-Bayesiano sobre $\mathbb{E}[\mathcal{R}_1] + \mathbb{E}[\mathcal{R}_2]$ donde $\mathrm{KL}(Q_1\|Q_2) = \tfrac{1}{2}\tfrac{\gamma^2}{\sigma^2}\|g_1 - g_2\|_2^2$. Promediada sobre $s$ batches ($s = 5$).

**Métricas de evaluación:**
- Test loss, test accuracy y test AUC (en MRNet).
- Training accuracy.
- Correlación de Pearson — entre GD y error de test ($\rho = 0.957$ en 220 configuraciones).
- Sensibilidad al threshold de early stopping (Eq. 14).
- Comparativa frente a $k$-fold y $k^+$-fold CV — diferencias absolutas de accuracy/AUC.

---

## Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks (Hölzl, 2025)

**Métrica(s) central(es):**
- **Score per-sample** — similitud coseno entre el gradiente per-sample y los pesos:
$$\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T) = \frac{\mathbf{g}_T(x_i)\cdot \mathbf{w}_T}{\|\mathbf{g}_T(x_i)\|\,\|\mathbf{w}_T\|}.$$
- **Gradient-Weight Alignment (GWA)** — esperanza con corrección por exceso de curtosis:
$$\text{GWA}_T = \frac{\mathbb{E}_i[\mathcal{A}_T]}{\operatorname{Kurt}_i[\mathcal{A}_T] + \beta} = \frac{M_T^{(1)}}{M_T^{(4)}/(M_T^{(2)})^2 - 3 + \beta},\quad \beta = 1.2.$$
- Estimador restringido al clasificador lineal final: $\mathbf{g}_t(x_i) = -z_i \cdot (\hat{y}_i - y_i)^\top$.

**Métricas de evaluación:**
- Test accuracy — sobre CIFAR-10, CIFAR-10-N, ImageNet-1k, ImageNet-C, etc.
- Test AUC y robustez (CIFAR-C, ImageNet-C).
- Correlación de **Pearson** y **Spearman** entre $\max_T \mathbb{E}[\mathcal{A}_T]$ y test accuracy — Pearson 0.99/0.98 y Spearman 0.97/0.93 en CIFAR-10.
- Overhead computacional (s/época) y memoria GPU.
- Comparación frente a val set (90/10, 99/1), LabelWave y Gradient Disparity.

---

## Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment (Chatterjee & Zielinski, 2020)

**Métrica(s) central(es):**
- **m-coherence** $\alpha_m = m\cdot\alpha$ — con
$$\alpha := \frac{\mathbb{E}_{z,z'}[g_z\cdot g_{z'}]}{\mathbb{E}_z[g_z\cdot g_z]} = \frac{\mathbb{E}[g_z\cdot g]}{\mathbb{E}[g_z\cdot g_z]}.$$
Interpretable como el número medio de ejemplos del sample que se benefician de un paso a lo largo del gradiente de un ejemplo aleatorio. En el límite ortogonal $\alpha_m = 1$; en alineación perfecta $\alpha_m = m$.
- Cálculo eficiente $O(m)$ vía $\mathbb{E}[g_z\cdot g] = \|g\|^2$ (acumulación streaming de $g$ y de $\sum\|g_z\|^2$).

**Métricas de evaluación:**
- Loss en train y test.
- Top-1 accuracy en train y test.
- m-coherence por capa (primera conv, intermedia, FC final) y global.
- Trayectoria frente al número de pasos.

---

## On the Ineffectiveness of Variance Reduced Optimization for Deep Learning (Defazio & Bottou, 2019)

**Métrica(s) central(es):**
- No propone métrica nueva. Mide directamente la **ratio de varianza** del estimador SVRG respecto al estimador SGD, $\mathrm{Var}[\hat{g}_{\text{SVRG}}]/\mathrm{Var}[\hat{g}_{\text{SGD}}]$, sobre el dataset completo en snapshots fijos del modelo. Ratio $<1$ indica reducción; ratios $\sim 2$ implican que el control variate ha quedado descorrelacionado con el gradiente estocástico, **aumentando** la varianza. Para que SVRG sea rentable la ratio debe estar por debajo de $1/3$.
- Diagnóstico complementario: **distancia iterada** $\|w_k - \tilde{w}\|$ y **curvatura empírica** $\|\tfrac{1}{|S_i|}\sum_{j\in S_i}[f'_j(w_k) - f'_j(\tilde{w})]\|/\|w_k - \tilde{w}\|$ entre el iterado actual y el snapshot, para localizar la causa del fallo.

**Métricas de evaluación:**
- Test error sobre CIFAR-10 (LeNet, DenseNet-40-36, ResNet-110) e ImageNet (ResNet-18); comparación SGD vs SVRG vs SCSG.
- Test error post-fine-tuning activando SVRG desde diferentes epochs en ResNet-50 ($23.61\%$ baseline SGD vs $23.65\text{–}28.60\%$ con SVRG fine-tuning) y DenseNet-169 ($23.22\%$ baseline vs $23.30\text{–}23.38\%$).
- Varianza del paso streaming en función de iteraciones desde el snapshot.

---

## RMSProp - Divide the gradient by a running average of its recent magnitude (Tieleman & Hinton, 2012)

**Métrica(s) central(es):**
- No propone métrica nueva. Introduce el algoritmo RMSProp con la media móvil exponencial del cuadrado del gradiente:
$$\text{MeanSquare}(w, t) = 0.9\, \text{MeanSquare}(w, t-1) + 0.1\left(\tfrac{\partial E}{\partial w}(t)\right)^2,\quad \Delta w(t) \propto \frac{\partial E/\partial w(t)}{\sqrt{\text{MeanSquare}(w, t)}}.$$
Conecta segundo momento no centrado del gradiente con el learning rate efectivo.

**Métricas de evaluación:**
- Comparaciones cualitativas — error de entrenamiento/validación a lo largo de las épocas, oscilaciones, comportamiento en mesetas, condicionamiento del paisaje y velocidad de convergencia. No hay tablas cuantitativas.

---

## Speedy Performance Estimation for Neural Architecture Search (Ru et al., 2021)

**Métrica(s) central(es):**
- **Training Speed Estimator (TSE)** — suma de pérdidas de entrenamiento durante SGD:
$$\text{TSE} = \sum_{t=1}^{T}\frac{1}{B}\sum_{i=1}^{B}\ell(f_{\theta_{t,i}}(\mathbf{X}_i), \mathbf{y}_i).$$
- **TSE-E** — variante con burn-in: $\text{TSE-E} = \sum_{t=T-E+1}^{T}\tfrac{1}{B}\sum_i \ell(\cdot)$.
- **TSE-EMA** — media móvil exponencial con $\gamma = 0.9$ (o $0.999$):
$$\text{TSE-EMA} = \sum_{t=1}^{T}\gamma^{T-t}\frac{1}{B}\sum_i \ell(\cdot).$$

**Métricas de evaluación:**
- Correlación de **Spearman** entre ranking predicho y ranking verdadero en test (también **Kendall** en apéndices).
- Error de test mínimo frente al runtime — eficiencia de búsqueda.
- Precisión media del top-10 de arquitecturas seleccionadas (one-shot).
- Comparación con baselines: TSE base, TLmini, SoVL, VAccES, LcSVR, y zero-cost JacCov/SNIP/SynFlow.

---

## Stiffness: A New Perspective on Generalization in Neural Networks (Fort et al., 2019)

**Métrica(s) central(es):**
- **Sign-stiffness** — $S_{\text{sign}}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\operatorname{sign}(\vec{g}_1\cdot\vec{g}_2)]$, en $[-1,1]$.
- **Cosine-stiffness** — $S_{\cos}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\cos(\vec{g}_1,\vec{g}_2)]$.
- **Class stiffness matrix** $C(c_a, c_b) = \mathbb{E}_{X_1\in c_a, X_2\in c_b, X_1\neq X_2}[S((X_1,y_1),(X_2,y_2))]$ — captura generalización intra-clase (diagonal) y transferencia entre clases (extra-diagonal).
- **Dynamic critical length** $\xi$ — distancia umbral donde la stiffness intra-clase cruza cero (ajuste lineal stiffness vs distancia en input).

**Métricas de evaluación:**
- Loss y accuracy en train y validación.
- Stiffness vs distancia en el espacio de entrada — con $\text{distance}(\vec{X}_1, \vec{X}_2) = 1 - \tfrac{\vec{X}_1\cdot\vec{X}_2}{|\vec{X}_1||\vec{X}_2|}\in[0,2]$.
- Stiffness en regímenes train-train, train-val y val-val.
- Suma extra-diagonal $S_{\text{between classes}} = \tfrac{1}{N_c(N_c-1)}\sum_{c_1}\sum_{c_2\neq c_1} C(c_1,c_2)$.

---

## The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent (Sankararaman et al., 2020)

**Métrica(s) central(es):**
- **Gradient Confusion** $\eta \ge 0$ — máxima similitud negativa entre gradientes per-sample. Las pérdidas $\{f_i\}$ tienen gradient confusion acotada por $\eta$ si para todo $i\neq j$ y un $w\in\mathbb{R}^d$ fijado:
$$\langle \nabla f_i(w), \nabla f_j(w)\rangle \ge -\eta.$$
$\eta$ pequeña $\Rightarrow$ gradientes armoniosos y SGD rápido; $\eta$ grande $\Rightarrow$ anti-correlación entre gradientes y SGD lento.

**Métricas de evaluación:**
- Valores de gradient confusion empíricos — mínimo y densidad de similitudes coseno por pares (100 pares de mini-batches al final de cada epoch).
- Training loss a lo largo de 200 epochs.
- Test accuracy final.
- Ablations de profundidad, anchura, batch normalization y skip connections.

---

## Understanding Why Neural Networks Generalize Well Through GSNR of Parameters (Liu et al., 2020)

**Métrica(s) central(es):**
- **Gradient Signal-to-Noise Ratio (GSNR)** por parámetro:
$$r(\theta_j) = \frac{\tilde{g}^2(\theta_j)}{\rho^2(\theta_j)},\quad \tilde{g}(\theta_j) = \mathbb{E}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)],\quad \rho^2(\theta_j) = \operatorname{Var}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)].$$
- **One-Step Generalization Ratio (OSGR)** — $R(\mathcal{Z}, n) = \mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$. Bajo aproximación de no-overfitting:
$$R(\mathcal{Z}, n) = 1 - \frac{1}{n}\sum_j W_j \cdot \frac{1}{r_j + 1/n},\quad \sum_j W_j = 1.$$

**Métricas de evaluación:**
- Train loss $L[D]$ y test loss $L[D']$.
- OSGR empírico — promediando sobre $M=10$ conjuntos de entrenamiento.
- Correlación de **Pearson** entre lado izquierdo y derecho de la ec. del OSGR — entre 0.907 y 0.968 en MNIST.
- Proporción $p_{\text{same\_sign}}$ — porcentaje de parámetros con gradiente del mismo signo entre muestras (~50% $\to$ ~56% durante el entrenamiento).
- Correlación entre pesos $W^{(l)}_{s,c}$ y $\Delta g^{(l)}_{D,s,c}$ — confirma la retroalimentación positiva.
- GSNR medio para configuraciones congeladas vs. no congeladas — en el toy dataset.

---

## Tabla comparativa

| Paper | Métrica central | Métricas evaluación |
|-------|-----------------|---------------------|
| Faghri et al. (2020) | Average Variance, Normalized Variance $\mathbb{V}[g]/\mathbb{E}[g]^2$ | Training loss, accuracy, varianza máxima |
| Shan & Bordelon (2021) | Kernel Alignment $A(t)$, Kernel Specialization Matrix (KSM) | Loss, $\|K\|_F$, traza, cosine similarity, forma bilineal |
| Johnson & Zhang (2013) | No propone métrica nueva (SVRG, reducción de varianza del estimador) | Training loss, suboptimalidad $P(w)-P(w_*)$, varianza, test error |
| Kingma & Ba (2015) | No propone métrica nueva (Adam usa $m_t$, $v_t$, regret $O(\sqrt{T})$) | Training cost (NLL/CE), test error, convergencia |
| McCandlish et al. (2018) | $B_{\text{noise}} = \operatorname{tr}(H\Sigma)/(G^\top H G)$, $B_{\text{simple}} = \operatorname{tr}(\Sigma)/\|G\|^2$ | $S$, $E$, frente de Pareto, $B_{\text{crit}}$, error/perplejidad/recompensa |
| Ruder (2017) | No propone métrica nueva (review) | Convergencia, saddle points, eficiencia (todo cualitativo) |
| Chatterjee (2019) | Coherent Gradients Hypothesis; $f_t^p$, $f_t^c$ y sus integrales | Train/test accuracy, training loss, overfit ajustado |
| Forouzesh & Thiran (2021) | Gradient Disparity $\mathcal{D}_{i,j} = \|g_i - g_j\|_2$ | Test loss/accuracy/AUC, Pearson, sensibilidad al threshold |
| Hölzl (2025) | Gradient-Weight Alignment (GWA), score $\gamma(x_i, w_T)$ | Test accuracy, Pearson/Spearman, robustez, overhead |
| Chatterjee & Zielinski (2020) | m-coherence $\alpha_m = m\cdot \mathbb{E}[g_z\cdot g]/\mathbb{E}[g_z\cdot g_z]$ | Train/test loss, top-1 accuracy, m-coherence por capa |
| Defazio & Bottou (2019) | No propone métrica nueva (ratio $\mathrm{Var}[\hat{g}_{\text{SVRG}}]/\mathrm{Var}[\hat{g}_{\text{SGD}}]$, distancia iterada, curvatura empírica) | Test error CIFAR-10/ImageNet, fine-tuning vs baseline, varianza streaming |
| Tieleman & Hinton (2012) | No propone métrica nueva (RMSProp, MeanSquare móvil) | Error train/val (cualitativo) |
| Ru et al. (2021) | TSE, TSE-E, TSE-EMA (suma de training losses) | Spearman, Kendall, error de test vs runtime, top-10 accuracy |
| Fort et al. (2019) | Sign-stiffness, Cosine-stiffness, class stiffness matrix, dynamic critical length $\xi$ | Loss/accuracy train-val, stiffness vs distancia, between-classes |
| Sankararaman et al. (2020) | Gradient Confusion $\eta$ ($\langle\nabla f_i,\nabla f_j\rangle \ge -\eta$) | Densidad de similitudes coseno, training loss, test accuracy |
| Liu et al. (2020) | GSNR $r(\theta_j) = \tilde{g}^2/\rho^2$, OSGR $R(\mathcal{Z},n)$ | Train/test loss, OSGR empírico, Pearson, $p_{\text{same\_sign}}$ |
