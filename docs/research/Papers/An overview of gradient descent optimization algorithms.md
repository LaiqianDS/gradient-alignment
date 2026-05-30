---
authors:
  - Sebastian Ruder
year: 2017
status: read
relevance: medium
last_review: 2026-05-07
url: https://arxiv.org/pdf/1609.04747
tfg_role:
  - background
tfg_note: "Survey de fundamentos (SGD, momentum, Adam); solo capítulo de fundamentos."
---

# An overview of gradient descent optimization algorithms

## Summary

### Contextualización

El artículo de Sebastian Ruder es un review didáctico cuyo propósito declarado es dotar al lector de intuiciones operativas sobre los algoritmos de optimización basados en descenso por gradiente que articulan el entrenamiento de redes neuronales modernas. El autor parte de una observación pragmática: pese a estar implementados en las principales librerías de deep learning (lasagne, caffe, keras), estos algoritmos se emplean habitualmente como cajas negras debido a la dificultad de localizar explicaciones prácticas de sus fortalezas y debilidades. El texto, publicado originalmente como entrada de blog en enero de 2016 y posteriormente sistematizado en formato académico, se enmarca en el problema del entrenamiento de redes mediante minimización de una función objetivo $J(\theta)$ parametrizada por $\theta \in \mathbb{R}^d$, donde la actualización se realiza en dirección opuesta al gradiente $\nabla_\theta J(\theta)$ y la learning rate $\eta$ fija el tamaño del paso.

### Aportación

La contribución central es una taxonomía cualitativa de los optimizadores más comunes basados en gradiente, articulada en torno a la motivación que llevó a proponer cada técnica como solución de las limitaciones de la anterior. Ruder no introduce algoritmos nuevos: organiza las contribuciones existentes —Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax y Nadam— en una progresión lógica que facilita comprender por qué cada método añade un mecanismo concreto sobre el precedente. Adicionalmente revisa arquitecturas para SGD paralelo y distribuido (Hogwild!, Downpour SGD, algoritmos delay-tolerant, TensorFlow, Elastic Averaging SGD) y discute estrategias auxiliares como shuffling, curriculum learning, batch normalization, early stopping y gradient noise.

### Metodología

El review presenta primero las tres variantes fundamentales según la cantidad de datos empleada por actualización. **Batch Gradient Descent** calcula el gradiente sobre todo el dataset, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta)$, y garantiza convergencia al mínimo global en superficies convexas y a un mínimo local en no convexas, pero resulta lento e impracticable para datasets grandes. **Stochastic Gradient Descent (SGD)** actualiza con un único ejemplo, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta;\, x^{(i)};\, y^{(i)})$, es más rápido y permite aprendizaje online, aunque introduce una varianza alta que provoca fluctuaciones marcadas. **Mini-batch Gradient Descent**, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta;\, x^{(i:i+n)};\, y^{(i:i+n)})$, combina ambas ideas reduciendo la varianza y aprovechando las optimizaciones matriciales del hardware, con tamaños habituales entre 50 y 256. A continuación Ruder enumera los retos del mini-batch SGD vainilla: la elección del learning rate, los schedules predefinidos incapaces de adaptarse al dataset, la aplicación uniforme del mismo $\eta$ a todos los parámetros —problemática cuando los datos son sparse— y la dificultad de escapar de saddle points (Dauphin et al. 2014), que en alta dimensión resultan más limitantes que los mínimos locales.

Sobre esta base se desarrollan las reglas de actualización de los optimizadores principales. **Momentum** (Qian, 1999) añade una fracción $\gamma$ del vector de actualización previo, $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$, $\theta = \theta - v_t$, con $\gamma$ típicamente $0.9$, acelerando las dimensiones con gradientes coherentes y amortiguando oscilaciones. El ejemplo paradigmático de su utilidad son los barrancos en la superficie de pérdida, comunes alrededor de los óptimos locales, donde SGD vainilla oscila entre las paredes y avanza lentamente por el fondo; con momentum, las componentes oscilatorias se cancelan y la componente consistente a lo largo del barranco se acumula. **Nesterov Accelerated Gradient (NAG)** refina Momentum calculando el gradiente en una posición aproximada futura, $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1})$, lo que aporta *prescience* y permite corregir antes cuando la trayectoria se desvía. Resulta especialmente útil en el entrenamiento de RNNs sobre tareas con dependencias largas, donde la corrección anticipada del momento marca la diferencia.

**Adagrad** (Duchi et al., 2011) cambia el régimen al adaptar $\eta$ por parámetro acumulando los gradientes pasados al cuadrado en una matriz diagonal $G_t$:

$$\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \varepsilon}} \cdot g_{t, i}.$$

Es la opción natural cuando los datos son sparse —embeddings de palabras, modelos de gran vocabulario, sistemas de recomendación— porque los parámetros infrecuentes acumulan poco $G_{t,ii}$ y reciben pasos relativamente grandes cuando finalmente reciben señal. Su debilidad estructural es que el denominador crece monotonamente, lo que hace que el learning rate efectivo decaiga hasta volverse infinitesimal y el entrenamiento se detenga prematuramente. **Adadelta** (Zeiler, 2012) corrige este problema sustituyendo la suma acumulada por una media móvil exponencial $E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$ y reemplazando el numerador $\eta$ por $\mathrm{RMS}[\Delta\theta]_{t-1}$ para igualar unidades hipotéticas, eliminando la necesidad de fijar un learning rate global por defecto. Es preferible cuando se desea un método adaptativo robusto sin sintonizar $\eta$. **RMSprop** (Hinton, no publicado) coincide con la primera derivación de Adadelta,

$$E[g^2]_t = 0.9\, E[g^2]_{t-1} + 0.1\, g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}}\, g_t,$$

con $\eta$ por defecto $0.001$, y se ha consolidado como elección frecuente para RNNs y modelos no estacionarios.

**Adam** (Kingma y Ba, 2015) sintetiza momentum y RMSprop almacenando medias móviles del gradiente y de su cuadrado, $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, aplica corrección de sesgo $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$ y actualiza

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\, \hat{m}_t,$$

con valores por defecto $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$. Es el método por defecto cuando se quiere convergencia rápida en redes profundas o complejas con poco esfuerzo de tuning. **AdaMax** generaliza $v_t$ a la norma $\ell_p$ y, llevándolo al límite $\ell_\infty$, define $u_t = \max(\beta_2 \cdot u_{t-1},\, |g_t|)$, $\theta_{t+1} = \theta_t - (\eta / u_t)\, \hat{m}_t$, con valores por defecto $\eta = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$; resulta útil cuando se desea acotar de forma natural la magnitud del paso por parámetro. **Nadam** (Dozat, 2016) incorpora el momento de Nesterov dentro de Adam reemplazando $\hat{m}_{t-1}$ por $\hat{m}_t$ en la regla expandida,

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\left(\beta_1 \hat{m}_t + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t}\right),$$

y es la alternativa cuando se busca la combinación de adaptatividad de Adam con la corrección anticipada de NAG.

### Tipos de problemas y experimentos

Por tratarse de un review, el artículo no presenta experimentos propios ni un estudio empírico sistemático. Ruder se apoya en comportamientos típicos descritos en la literatura y en visualizaciones cualitativas (Figura 4, de Alec Radford) sobre la función de Beale y un saddle point: en los contornos de pérdida, Adagrad, Adadelta y RMSprop convergen rápido en la dirección correcta mientras que Momentum y NAG se desvían como una bola rodando, si bien NAG corrige antes; en el saddle point, SGD, Momentum y NAG tienen problemas para romper la simetría, mientras que los métodos adaptativos descienden rápidamente por la pendiente negativa con Adadelta a la cabeza. Estas ilustraciones cualitativas sustituyen a una evaluación cuantitativa.

### Métricas

Las métricas referenciadas son las habituales en optimización: convergencia entendida como velocidad de aproximación al mínimo, capacidad de escapar saddle points y estabilidad frente a oscilaciones; eficiencia computacional, medida por el coste por actualización y el aprovechamiento de implementaciones matriciales en mini-batch; y robustez frente a inicializaciones pobres y a sparsity de los datos. No se reportan accuracy ni curvas de pérdida cuantitativas, sino comparaciones cualitativas y referencias a resultados externos —como la observación de Kingma et al. de que la corrección de sesgo de Adam aporta una ventaja marginal frente a RMSprop conforme los gradientes se vuelven más sparse—.

### Conclusiones

Ruder cierra con recomendaciones prácticas que pueden leerse como una guía de elección. Si los datos de entrada son sparse, conviene optar por un método de learning rate adaptativo y evitar el ajuste manual de $\eta$. RMSprop es una extensión de Adagrad que mitiga su decaimiento radical y resulta idéntica a Adadelta salvo por el numerador. Adam añade corrección de sesgo y momento sobre RMSprop y, según Kingma et al., supera ligeramente a RMSprop cuando los gradientes se vuelven sparse, por lo que podría ser la mejor opción global. El autor reconoce que muchos trabajos recientes utilizan SGD vainilla con un schedule simple de annealing y que SGD suele encontrar el mínimo, pero a costa de más tiempo, mejor inicialización y riesgo de quedar atrapado en saddle points; por tanto, si se prioriza la convergencia rápida o se entrena una red profunda o compleja, conviene escoger un método adaptativo. Finalmente, recapitula que ha cubierto las tres variantes (con mini-batch como la más popular), los algoritmos para optimizar SGD (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam), variantes asíncronas y estrategias adicionales como shuffling, curriculum learning, batch normalization y early stopping.

## Medición y pipeline

Este artículo no aporta una métrica para el `METRIC_REGISTRY` del TFG, pero informa decisiones de diseño del pipeline y, sobre todo, fija qué magnitudes del propio optimizador conviene loggear como variables de control. Su rol es de **background / fundamentos**: justifica la elección de SGD (vanilla y con momentum) y Adam como optimizadores del sweep, al ser los dos extremos representativos —descenso por gradiente puro frente a actualización adaptativa con primer y segundo momento— y encuadra las diferencias de comportamiento esperadas entre ambos durante la fase temprana del entrenamiento.

De la taxonomía se derivan tres implicaciones de pipeline. Primero, en SGD (vanilla o con momentum) los estimadores de ruido y varianza del gradiente se calculan de forma directa, pues el minibatch gradient noise es el objeto natural de medida. Segundo, en Adam y RMSprop los momentos adaptativos $m_t$ y $v_t$ alteran la magnitud efectiva del update: si la alineación se midiera sobre el update preacondicionado en lugar del gradiente crudo, las métricas cambiarían de escala e interpretación, por lo que es necesario decidir y documentar explícitamente qué cantidad se loggea. Tercero, en Adagrad y Adadelta aplica la misma consideración por su acumulador diagonal de segundos momentos. Como decisión de diseño, para garantizar la comparabilidad entre SGD y Adam las métricas del TFG (cosine similarity, m-coherence, GNS, NGV, stiffness) se calculan sobre el **gradiente crudo** $\nabla L(w)$ con independencia del optimizador empleado, y esta elección se justifica explícitamente en la memoria.

Más allá del raw-grad rationale, el paper sugiere una batería compacta de **magnitudes del optimizador** que conviene registrar como diagnóstico complementario, con coste prácticamente nulo porque todas ellas viven en el state del optimizer o se derivan en una sola pasada sobre los parámetros. Por step se calculan la **norma global del gradiente** $\|g\|_2$ y, opcionalmente, sus normas por capa $\|g^{(\ell)}\|_2$ —útiles para detectar vanishing/exploding gradients y para anclar el resto de magnitudes—; la **norma del paso** $\|\Delta w\|_2$ tras `optimizer.step()`, que en SGD coincide con $\eta\|g\|$ pero en Adam refleja el preacondicionamiento; el **learning rate efectivo por parámetro**, que en Adagrad es $\eta/\sqrt{G_{t,ii}+\varepsilon}$ y en Adam corresponde a la combinación bias-corrected $\eta\sqrt{1-\beta_2^t}/(1-\beta_1^t)$ multiplicada por $1/(\sqrt{\hat v_t}+\varepsilon)$ por coordenada; y el **coseno entre paso actual y anterior**, $\cos(\Delta w_t, \Delta w_{t-1})$, que cuantifica la coherencia temporal de la trayectoria y detecta oscilaciones, complementando la firma temprana de las métricas de alineación. La frecuencia recomendada es **por step** para $\|g\|$, $\|\Delta w\|$ y el coseno temporal —se necesita resolución fina en la ventana temprana— y **por época** para los lr efectivos agregados (media, mediana, percentiles), que cambian con escala lenta.

Como **claves de log sugeridas** se proponen `opt/grad_norm`, `opt/grad_norm_layer/<l>`, `opt/step_norm`, `opt/step_norm_layer/<l>`, `opt/lr_eff_mean`, `opt/lr_eff_p50`, `opt/lr_eff_p95`, `opt/cos_step_prev` y, específicas de Adam, `opt/m_norm_layer/<l>`, `opt/v_norm_layer/<l>` y `opt/lr_eff_biascorr`. La extracción se hace sobre `optimizer.state[p]` (`exp_avg`, `exp_avg_sq` en Adam; `sum` en Adagrad) tras `optimizer.step()`, sin coste adicional de forward/backward.

Conviene anticipar varios **gotchas**. Comparar lr efectivos entre optimizadores requiere normalizar: el $\eta$ nominal de SGD no es comparable con $\eta/(\sqrt{\hat v_t}+\varepsilon)$ de Adam, y dentro de Adam el lr efectivo varía por parámetro y por step, por lo que tiene sentido reportar la distribución (mediana, p95) más que un escalar. En Adam, los primeros pasos están dominados por la corrección de sesgo y el lr efectivo nominal subestima el real si se omite el factor $\sqrt{1-\beta_2^t}/(1-\beta_1^t)$. La norma del paso $\|\Delta w\|$ es invariante de reescalado de $\eta$ en Adam pero no en SGD, lo que distorsiona comparaciones directas si se varía la learning rate del sweep. Por último, en cualquier régimen de medición fuera del bucle de entrenamiento conviene hacer snapshot y restore de `optimizer.state_dict()` para evitar contaminar $m_t$ y $v_t$ con los gradientes del probe; SGD puro no necesita este cuidado, pero el pipeline lo trata de forma uniforme para mantener la simetría cross-optimizador.

## Notes
- Introductory review

### Uso en el TFG
- **Rol**: review didáctico y **citación de respaldo**. No aporta métrica, baseline ni estudio empírico propio; no entra en el `METRIC_REGISTRY`.
- **Justificar el sweep de optimizadores SGD + Adam**: Ruder ofrece la taxonomía y las razones de elección de cada método, sirviendo de respaldo para acotar el sweep a los dos extremos representativos —descenso por gradiente puro (SGD/momentum) frente a actualización adaptativa con primer y segundo momento (Adam)— y para descartar el resto (Adagrad, Adadelta, RMSprop, AdaMax, Nadam) como casos intermedios sin contraste cualitativo nuevo.
- **Motivar el raw-grad rationale**: los optimizadores adaptativos (Adagrad/Adadelta/RMSprop/Adam) reescalan el gradiente $\nabla L$ mediante acumuladores de segundos momentos, alterando la magnitud e interpretación del update; esto fundamenta medir las 10 métricas sobre el **gradiente bruto** $\nabla L(w)$ y no sobre la update preacondicionada, garantizando comparabilidad SGD↔Adam.
- **Variables de control**: respalda loggear los hiperparámetros del optimizador (lr, $\beta_1$, $\beta_2$, $\varepsilon$, momentum) y, como diagnóstico opcional, las magnitudes del optimizer state ($\|g\|$, $\|\Delta w\|$, lr efectivo, coseno entre pasos consecutivos, normas de $m_t$ y $v_t$) para el análisis de robustez cross-optimizador.
- **Qué NO aporta**: ninguna métrica de diagnóstico, ningún resultado cuantitativo (solo comparaciones cualitativas y figuras ilustrativas), ni evidencia correlacional métrica↔generalización. Se cita en *background / related work* (raw-grad rationale + elección de optimizadores), no en *methods* de las métricas.

## Papers relacionados
- [[Adam - A Method for Stochastic Optimization]] — optimizador del sweep catalogado por Ruder; combina primer momento (momentum) y segundo momento no centrado, base del régimen adaptativo a contrastar con SGD.
- [[RMSProp - Divide the gradient by a running average of its recent magnitude]] — precursor de Adam en la taxonomía; el segundo momento $E[g^2]$ como reescalado por parámetro motiva el raw-grad rationale (medir sobre $\nabla L$, no sobre la update).
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — el segundo momento del gradiente que usan los métodos adaptativos (RMSprop/Adam) reaparece aquí como denominador del GSNR (SNR por parámetro), conectando preacondicionamiento y métrica de varianza.
- [[A Study of Gradient Variance in Deep Learning]] — la varianza/segundo momento del gradiente que escala el paso adaptativo es el objeto de la Normalized Variance ($\mathbb{V}[g]/\mathbb{E}[g]^2$), familia varianza del registro.

## Otros papers interesantes a revisar
- **Decoupled Weight Decay Regularization (AdamW)** (Loshchilov & Hutter, 2019) — corrige el acoplamiento entre weight decay y el preacondicionamiento adaptativo de Adam; refina la elección de optimizadores adaptativos para el sweep. arXiv:1711.05101.
- **On the Convergence of Adam and Beyond (AMSGrad)** (Reddi, Kale & Kumar, 2018) — señala un fallo en la prueba de convergencia de Adam y propone AMSGrad; relevante para acotar las garantías del régimen adaptativo. ICLR 2018, arXiv:1904.09237.
- **The Marginal Value of Adaptive Gradient Methods in Machine Learning** (Wilson et al., 2017) — evidencia empírica de que los métodos adaptativos (Adam/RMSprop) generalizan peor que SGD bien ajustado; respalda directamente el interés del contraste SGD↔Adam del TFG. arXiv:1705.08292.
- **Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (AdaGrad)** (Duchi, Hazan & Singer, 2011) — fuente primaria del primer optimizador adaptativo de la taxonomía de Ruder; útil para citar el origen del acumulador de segundos momentos. JMLR 12:2121-2159.
