---
authors:
  - Sebastian Ruder
year: 2017
status: read
relevance: medium
last_review: 2026-05-07
url: https://arxiv.org/pdf/1609.04747
---

# An overview of gradient descent optimization algorithms

## Summary

### Contextualización

El artículo de Sebastian Ruder (2017, arXiv:1609.04747v2) es un review didáctico cuyo objetivo declarado es proporcionar al lector intuiciones sobre el comportamiento de los distintos algoritmos de optimización basados en descenso por gradiente que se utilizan en deep learning. El autor parte de la constatación de que dichos algoritmos, pese a estar implementados en las principales librerías (lasagne, caffe, keras), suelen emplearse como cajas negras debido a la dificultad de encontrar explicaciones prácticas de sus fortalezas y debilidades. El texto, originalmente publicado como entrada de blog en enero de 2016, fue posteriormente sistematizado en formato académico para servir como referencia de consulta. Se enmarca en el contexto del entrenamiento de redes neuronales mediante minimización de una función objetivo $J(\theta)$ parametrizada por $\theta \in \mathbb{R}^d$, donde la actualización se realiza en dirección opuesta al gradiente $\nabla_\theta J(\theta)$ y la learning rate $\eta$ fija el tamaño del paso.

### Aportación

La aportación principal es una taxonomía y comparación cualitativa de los optimizadores más comunes basados en gradiente, articulada en torno a la motivación que llevó a cada uno a proponerse como solución de los retos del anterior. Ruder no introduce algoritmos nuevos: organiza las contribuciones existentes (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam) en una progresión lógica que facilita comprender por qué cada técnica añade un mecanismo concreto sobre la anterior. Adicionalmente, revisa arquitecturas para SGD paralelo y distribuido (Hogwild!, Downpour SGD, algoritmos delay-tolerant, TensorFlow, Elastic Averaging SGD) y discute estrategias auxiliares (shuffling, curriculum learning, batch normalization, early stopping, gradient noise).

### Metodología

El review presenta primero las tres variantes fundamentales según la cantidad de datos empleada por actualización:

- **Batch Gradient Descent** calcula el gradiente sobre todo el dataset, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta)$, garantiza convergencia al mínimo global en superficies convexas y a un mínimo local en no convexas, pero es lento e impracticable para datasets grandes.
- **Stochastic Gradient Descent (SGD)** actualiza con un solo ejemplo, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta;\, x^{(i)};\, y^{(i)})$, es más rápido y permite aprendizaje online, aunque introduce alta varianza que provoca fluctuaciones.
- **Mini-batch Gradient Descent**, $\theta = \theta - \eta \cdot \nabla_\theta J(\theta;\, x^{(i:i+n)};\, y^{(i:i+n)})$, combina ambos, reduce la varianza y aprovecha optimizaciones matriciales, con tamaños habituales entre 50 y 256.

A continuación, Ruder enumera los retos del mini-batch SGD vainilla: elección del learning rate, schedules predefinidos incapaces de adaptarse al dataset, aplicación uniforme del mismo $\eta$ a todos los parámetros (problemático con datos sparse), y la dificultad de escapar de saddle points (Dauphin et al. 2014) más que de mínimos locales.

Sobre esta base se desarrollan las reglas de actualización de los optimizadores principales.

**Momentum (Qian 1999)** añade una fracción $\gamma$ del vector de actualización previo:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta), \quad \theta = \theta - v_t,$$

con $\gamma$ típicamente $0.9$, acelerando dimensiones con gradientes coherentes y reduciendo oscilaciones.

**Nesterov Accelerated Gradient (NAG)** calcula el gradiente en una posición aproximada futura $\theta - \gamma v_{t-1}$, aportando *prescience*:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1}).$$

**Adagrad (Duchi et al. 2011)** adapta $\eta$ por parámetro acumulando los gradientes pasados al cuadrado en una matriz diagonal $G_t$:

$$\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \varepsilon}} \cdot g_{t, i},$$

ideal para datos sparse pero con learning rate que decae monotonamente hasta volverse infinitesimal.

**Adadelta (Zeiler 2012)** corrige este problema usando una media móvil exponencial $E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$, sustituye el numerador $\eta$ por $\mathrm{RMS}[\Delta\theta]_{t-1}$ para igualar unidades hipotéticas, y elimina la necesidad de fijar $\eta$ por defecto.

**RMSprop (Hinton, no publicado)** coincide con la primera derivación de Adadelta:

$$E[g^2]_t = 0.9\, E[g^2]_{t-1} + 0.1\, g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}}\, g_t,$$

con $\eta$ por defecto $0.001$.

**Adam (Kingma y Ba 2015)** combina momento y RMSprop almacenando medias móviles del gradiente ($m_t$) y de su cuadrado ($v_t$):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2;$$

aplica corrección de sesgo $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$, y actualiza

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\, \hat{m}_t,$$

con valores por defecto $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

**AdaMax** generaliza $v_t$ a la norma $\ell_p$ y, llevándolo al límite $\ell_\infty$, define $u_t = \max(\beta_2 \cdot u_{t-1},\, |g_t|)$, $\theta_{t+1} = \theta_t - (\eta / u_t)\, \hat{m}_t$, con valores por defecto $\eta = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Nadam (Dozat 2016)** incorpora el momento de Nesterov dentro de Adam reemplazando $\hat{m}_{t-1}$ por $\hat{m}_t$ en la regla expandida:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\left(\beta_1 \hat{m}_t + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t}\right).$$

### Tipos de problemas y experimentos

Por tratarse de un review, el artículo no presenta experimentos propios ni un estudio empírico sistemático. Ruder se apoya en comportamientos típicos descritos en la literatura y en visualizaciones cualitativas (Figura 4 de Alec Radford) sobre la función de Beale y un saddle point: en los contornos de pérdida, Adagrad, Adadelta y RMSprop convergen rápido en la dirección correcta mientras Momentum y NAG se desvían como una bola rodando, aunque NAG corrige antes; en el saddle point, SGD, Momentum y NAG tienen problemas para romper la simetría, mientras que los métodos adaptativos descienden rápidamente por la pendiente negativa con Adadelta a la cabeza. Estas ilustraciones cualitativas sustituyen a una evaluación cuantitativa.

### Métricas

Las métricas referenciadas son las habituales en optimización: convergencia (velocidad de aproximación al mínimo, capacidad de escapar saddle points, estabilidad frente a oscilaciones), eficiencia computacional (coste por actualización y aprovechamiento de implementaciones matriciales en mini-batch), y robustez frente a inicializaciones pobres y a sparsity de los datos. No se reportan accuracy ni curvas de pérdida cuantitativas, sino comparaciones cualitativas y referencias a resultados externos (e.g. Kingma et al. para la ventaja marginal de la corrección de sesgo de Adam frente a RMSprop conforme los gradientes se vuelven más sparse).

### Conclusiones

Ruder cierra con recomendaciones prácticas: si los datos de entrada son sparse, conviene usar un método de learning rate adaptativo, evitando el ajuste manual del η. RMSprop es una extensión de Adagrad que mitiga su decaimiento radical y resulta idéntica a Adadelta salvo por el numerador. Adam añade corrección de sesgo y momento sobre RMSprop y, según Kingma et al., supera ligeramente a RMSprop cuando los gradientes se vuelven sparse, por lo que podría ser la mejor opción global. Reconoce que muchos trabajos recientes utilizan SGD vainilla con un schedule simple de annealing y que SGD suele encontrar el mínimo, pero requiere más tiempo, mejor inicialización y puede quedar atrapado en saddle points; por tanto, si se prioriza convergencia rápida o se entrena una red profunda o compleja, conviene escoger un método adaptativo. Finalmente, recapitula que ha cubierto las tres variantes (con mini-batch como la más popular), los algoritmos para optimizar SGD (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam), variantes para SGD asíncrono y estrategias adicionales como shuffling, curriculum learning, batch normalization y early stopping.

## Medición y pipeline

Este artículo no aporta una métrica directa para el TFG, pero informa decisiones de diseño del pipeline. Su rol es de **background / fundamentos**: justifica la elección de SGD (vanilla y con momentum) y Adam como optimizadores del sweep, al ser los dos extremos representativos —descenso por gradiente puro frente a actualización adaptativa con primer y segundo momento— y encuadra las diferencias de comportamiento esperadas entre ambos durante la fase temprana del entrenamiento.

De la taxonomía se derivan tres **implicaciones de pipeline**. Primero, en SGD (vanilla o con momentum) los estimadores de ruido y varianza del gradiente se calculan de forma directa, pues el minibatch gradient noise es el objeto natural de medida. Segundo, en Adam y RMSProp los momentos adaptativos $m_t$ y $v_t$ alteran la magnitud efectiva del update; si la alineación se midiera sobre el `update` preacondicionado en lugar del gradiente crudo, las métricas cambiarían de escala e interpretación, por lo que es necesario decidir y documentar explícitamente qué cantidad se loggea. Tercero, en Adagrad y Adadelta aplica la misma consideración por su acumulador diagonal de segundos momentos.

Del paper conviene **loggear como variables de control** los hiperparámetros del optimizador ($\text{lr}$, $\beta_1$, $\beta_2$, $\varepsilon$, momentum) y, opcionalmente, las normas de $m_t$ y $v_t$ en Adam como diagnóstico para el análisis de robustez cross-optimizador.

**Decisión de diseño**: para garantizar la comparabilidad entre SGD y Adam, las métricas del TFG (cosine similarity, m-coherence, GNS, NGV, stiffness) se calculan sobre el **gradiente crudo** $\nabla L(w)$, con independencia del optimizador empleado. Esta elección se justificará explícitamente en la memoria.

## Notes
- Introductory review
