---
authors:
  - Tijmen Tieleman
  - Geoffrey Hinton
year: 2012
status: to-read
relevance: low
last_review: 2026-05-07
url: https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
tfg_role:
  - background
  - method
---

# RMSProp - Divide the gradient by a running average of its recent magnitude

## Summary

### Contextualización

El material corresponde a la Lecture 6 (slides 6a–6e) del curso *Neural Networks for Machine Learning* impartido por Geoffrey Hinton en Coursera (2012), con la colaboración de Nitish Srivastava y Kevin Swersky. No se trata de un paper formal sino de un slide-deck docente en el que se introduce públicamente por primera vez el algoritmo RMSProp, atribuido a Tijmen Tieleman como trabajo no publicado. Pese a su carácter informal, el documento es la referencia canónica que la literatura cita cuando se invoca el método.

La motivación parte de un análisis de las dificultades del descenso por gradiente estocástico con learning rate global fijo. En superficies de error con forma de cuenco cuadrático mal condicionado (elipses muy alargadas en lugar de círculos), el gradiente apunta con magnitud grande en direcciones donde solo se desea avanzar poco, y con magnitud pequeña en direcciones donde se desearía avanzar mucho. El resultado típico es la oscilación a través del barranco (*ravine*) y un avance lento por su eje principal. Si el learning rate es demasiado grande las oscilaciones divergen; si es demasiado pequeño la convergencia es prohibitiva. Hinton subraya además que en redes multicapa profundas la magnitud del gradiente varía ampliamente entre capas (especialmente con inicializaciones pequeñas), de modo que un único learning rate global no es adecuado. La lección 6d propone usar ganancias locales adaptativas por peso (regla aditiva al concordar el signo del gradiente, multiplicativa al discrepar), inspirándose en Jacobs (1989). La lección 6e da el siguiente paso introduciendo RMSProp como variante mini-batch de **rprop**.

### Aportación

La contribución central es **RMSProp**, una regla de actualización con learning rate adaptativo por parámetro que adapta a mini-batches la idea de rprop. Rprop, en su versión full-batch, ignora la magnitud del gradiente y solo emplea su signo, ajustando multiplicativamente el tamaño de paso por peso (por ejemplo, multiplicar por 1.2 si los signos de los dos últimos gradientes coinciden, por 0.5 si no, manteniendo el paso entre cotas como un millonésimo y 50). Esto da actualizaciones de la misma magnitud para todos los pesos y permite escapar rápido de mesetas con gradientes minúsculos.

Hinton expone por qué rprop falla con mini-batches: la lógica del SGD asume que el gradiente promedia sobre actualizaciones sucesivas de mini-batches; un peso con gradiente +0.1 en nueve mini-batches y −0.9 en el décimo debería permanecer aproximadamente donde está, pero rprop lo incrementaría nueve veces y solo lo decrementaría una vez por la misma magnitud, haciéndolo crecer indebidamente. La pregunta es cómo conjugar la robustez de rprop, la eficiencia de los mini-batches y el promediado efectivo de gradientes.

La respuesta es observar que rprop equivale a usar el gradiente dividiéndolo por su propia magnitud; el problema con mini-batches es que ese divisor cambia mucho de un batch a otro. RMSProp fuerza a que el divisor sea similar entre mini-batches adyacentes manteniéndolo como una **media móvil exponencial del cuadrado del gradiente**.

### Metodología

La regla de actualización completa, tal como aparece en la slide, es:

$$
\text{MeanSquare}(w, t) = 0.9 \, \text{MeanSquare}(w, t-1) + 0.1 \left(\frac{\partial E}{\partial w}(t)\right)^2
$$

A continuación, el gradiente se divide por la raíz cuadrada de esa media móvil (RMS, *root mean square*) antes de aplicar el paso:

$$
\Delta w(t) \propto \frac{\partial E / \partial w(t)}{\sqrt{\text{MeanSquare}(w, t)}}
$$

Hiperparámetros explícitos en las slides: factor de suavizado 0.9 sobre el segundo momento no centrado (es decir, vida media corta sobre los últimos gradientes cuadrados), y un learning rate global multiplicativo. No se introduce explícitamente término de estabilidad numérica $\epsilon$ ni corrección de sesgo del estimador (esos detalles aparecerán formalizados después en Adam). La cita en la slide es literal: *"Dividing the gradient by sqrt(MeanSquare(w,t)) makes the learning work much better (Tijmen Tieleman, unpublished)"*.

Como extensiones discutidas en *Further developments of rmsprop*, Hinton sugiere combinar RMSProp con momentum estándar (señalando que ayuda menos de lo habitual y requiere más investigación), con momentum de Nesterov (Sutskever 2012), donde funciona mejor si el RMS de los gradientes recientes se utiliza para dividir la corrección y no el salto en la dirección del gradiente acumulado, y con learning rates adaptativos por conexión al estilo Jacobs (también pendiente de más investigación). Se menciona como método relacionado el de Yann LeCun en *No more pesky learning rates*.

### Datasets / modelos

Al ser un slide-deck pedagógico, no se reportan experimentos sistemáticos sobre datasets concretos ni curvas cuantitativas. Las recomendaciones de uso son cualitativas y referidas a familias de problemas: redes profundas (especialmente con cuellos de botella estrechos), redes recurrentes y redes anchas y poco profundas. La slide *Summary of learning methods for neural networks* recomienda RMSProp (con momentum opcional) para datasets grandes y redundantes con aprendizaje por mini-batches, y reserva métodos full-batch (gradiente conjugado, LBFGS, learning rates adaptativos, rprop) para datasets pequeños (por ejemplo del orden de 10 000 casos) o sin redundancia.

### Métricas

Las slides operan sobre nociones cualitativas: error de entrenamiento y de validación a lo largo de las épocas, oscilaciones del error, comportamiento en mesetas, condicionamiento de la superficie de error (relación entre eigenvalores en la aproximación cuadrática local) y velocidad de convergencia. No hay tablas con accuracy, perplejidad ni tiempos de wall-clock; las figuras son ilustrativas (cuencos cuadráticos, trayectorias en barrancos, esquemas del método de momentum y de Nesterov).

### Conclusiones

RMSProp se presenta como una receta práctica para entrenamiento por mini-batches en redes neuronales: hereda de rprop la idea de normalizar por la magnitud reciente del gradiente para igualar de facto las magnitudes de actualización entre pesos, pero estabiliza el divisor mediante una media móvil exponencial del cuadrado del gradiente, evitando la patología de rprop con muestreo estocástico. Hinton sugiere su uso como primera opción junto a SGD con momentum y la receta de LeCun para datasets grandes y redundantes. Reconoce explícitamente que no existe una receta única porque las redes y tareas difieren mucho, y que varias combinaciones (RMSProp + momentum, RMSProp + Nesterov, RMSProp + ganancias adaptativas por conexión) requieren más investigación. Para el TFG, la relevancia del documento radica en explicitar la conexión conceptual entre la varianza por parámetro del gradiente (segundo momento no centrado, $E[g^2]$) y el learning rate efectivo, anclando la familia de métodos adaptativos posteriores (Adam, Adadelta) en la idea de escalar el paso por una estimación local de magnitud del gradiente.

## Medición y pipeline

RMSProp se trata en este TFG como **background histórico**: precursor directo de Adam y referencia conceptual para anclar la familia de optimizadores adaptativos en la idea de escalar el paso por una estimación local del segundo momento del gradiente. No constituye una métrica a loggear ni forma parte del sweep cerrado de optimizadores.

**Rol del paper.** RMSProp introduce la pieza que Adam hereda y generaliza: el segundo momento adaptativo $v_t = \rho v_{t-1} + (1-\rho) g_t^2$ con actualización $\Delta w \propto -\eta \cdot g_t / (\sqrt{v_t} + \epsilon)$. La justificación de Adam como optimizador en el sweep se apoya, en parte, en este precedente.

**Decisión metodológica.** El sweep cerrado de la pregunta de investigación (RQ: las métricas de gradiente en etapa temprana predicen la eficiencia de entrenamiento completo) se restringe a **SGD y Adam**, suficientes para el contraste cross-optimizador entre régimen no preacondicionado y régimen con segundo momento más momentum. Añadir RMSProp incrementaría el coste experimental sin aportar contraste cualitativo nuevo: equivale aproximadamente a un caso intermedio (segundo momento sin momentum) entre SGD y Adam.

**Si se incluyera como optimizador opcional** en un análisis de robustez, los diagnósticos serían análogos a los de Adam —norma de $v_t$ por capa y learning rate efectivo $\eta/\sqrt{v_t+\epsilon}$, accesibles vía `optimizer.state[p]['square_avg']` en `torch.optim.RMSprop(lr, alpha=0.99, eps=1e-8)`— y las métricas del estudio (cosine similarity, m-coherence, GNS, NGV, stiffness) se calcularían sobre el **gradiente crudo**, no sobre el update preacondicionado, manteniendo la consistencia con el criterio adoptado para Adam. La inclusión queda relegada a *future work*.

## Notes
- Fuente original: Lecture 6e del curso "Neural Networks for Machine Learning" de Hinton (Coursera, 2012). No hay paper formal.

### Uso en el TFG
- **Rol: related work (junto a Adam), NO métrica del registry.** Precursor histórico del optimizador adaptativo; no aporta ninguna de las 10 métricas + baseline. No entra tampoco en el sweep cerrado (SGD y Adam): equivaldría a un caso intermedio (segundo momento sin momentum), sin contraste cualitativo nuevo.
- **MeanSquare móvil como 2º momento no centrado / proxy de varianza por parámetro.** $\text{MeanSquare}(w,t) = 0.9\,\text{MeanSquare}(w,t-1) + 0.1\,(\partial E/\partial w)^2$ estima $\mathbb{E}[g^2]$ por peso vía EMA; con $\Delta w \propto (\partial E/\partial w)/\sqrt{\text{MeanSquare}(w,t)}$, el learning rate efectivo $\eta/\sqrt{\mathbb{E}[g^2]}$ es exactamente un reescalado tipo SNR por parámetro.
- **Motiva el eje varianza del TFG.** El divisor $\sqrt{\mathbb{E}[g^2]}$ es el antecedente conceptual de las métricas de varianza del registry: `normalized_variance` ($\mathbb{V}[g]/\mathbb{E}[g]^2$, inverso de un SNR) y `gsnr` ($\tilde g^2/\rho^2$). RMSProp muestra que ya en 2012 la magnitud cuadrática del gradiente se usaba como señal local de ruido para regular el paso.
- **Motiva el raw-grad rationale.** RMSProp ilustra que la *update* aplicada ($\propto g/\sqrt{\text{MeanSquare}}$) está reescalada por el propio segundo momento; medir las métricas sobre esa update mezclaría la señal de alineación/varianza con el preacondicionamiento del optimizador. De ahí la decisión de calcular todas las métricas sobre el **gradiente bruto** $\nabla L$, garantizando comparabilidad SGD vs Adam.

## Papers relacionados
- [[Adam - A Method for Stochastic Optimization]] — sucesor directo: formaliza el 2º momento $\hat v_t$ de RMSProp y añade 1er momento, corrección de sesgo y $\epsilon$; razón $\hat m_t/\sqrt{\hat v_t}$ como SNR.
- [[An overview of gradient descent optimization algorithms]] — sitúa RMSProp en la taxonomía de optimizadores adaptativos (Adagrad → Adadelta → RMSprop → Adam); respaldo para justificar el sweep SGD+Adam.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — usa el SNR por parámetro $\tilde g^2/\rho^2$, formalizando como métrica el cociente señal/ruido que RMSProp explota implícitamente en su divisor.
- [[A Study of Gradient Variance in Deep Learning]] — define la varianza normalizada $\mathbb{V}[g]/\mathbb{E}[g]^2$ como inverso de un SNR, el eje varianza que RMSProp motiva históricamente.

## Otros papers interesantes a revisar
- **Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (Adagrad)** (Duchi, Hazan & Singer, 2011) — origen de los learning rates por parámetro escalados por la suma acumulada de gradientes al cuadrado; RMSProp lo corrige sustituyendo la suma por una EMA. JMLR 12:2121-2159.
- **ADADELTA: An Adaptive Learning Rate Method** (Zeiler, 2012) — variante contemporánea de RMSProp que elimina el learning rate global usando una segunda EMA sobre las updates al cuadrado. arXiv:1212.5701.
- **On the Convergence of Adam and Beyond (AMSGrad)** (Reddi, Kale & Kumar, 2018) — analiza fallos de convergencia de los métodos con EMA del 2º momento (RMSProp/Adam) y propone una corrección; contexto teórico del divisor adaptativo. arXiv:1904.09237.
