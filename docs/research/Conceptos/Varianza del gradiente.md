
En SGD, el gradiente que usas en cada paso es un estimador $\hat g$ del gradiente real $\nabla f$ obtenido sobre un minibatch. Su varianza no es residual sino estructural, y persiste incluso en el óptimo, donde $\nabla f = 0$ pero $\mathrm{Var}[\hat g] > 0$. La razón es que cada minibatch apunta a un objetivo ligeramente distinto, y las contribuciones individuales de los ejemplos no se cancelan perfectamente al promediarlas.

Para un minibatch de tamaño $B$ con ejemplos i.i.d., la varianza del estimador es $\mathrm{Var}[\hat g] = \tfrac{1}{B}\,\mathrm{tr}(\Sigma)$ con $\Sigma$ la covarianza per-sample. Esto significa que la varianza escala $1/B$, así que duplicar el batch reduce el ruido a la mitad, pero no a cero. El suelo de ruido viene del hecho de que los ejemplos no son redundantes, y nunca lo serán si el dataset es rico en variedad.

Un ejemplo concreto ayuda a aterrizar la magnitud. Imagina entrenar una red en CIFAR-10 con 50 000 imágenes y un batch de 128. Cada paso ve menos del 0.3 por ciento del dataset, así que el gradiente medio que calculas está rodeado por una nube de ruido cuya desviación típica suele ser del mismo orden que la propia señal del gradiente cerca del óptimo. Cuando subes el batch a 512 (cuatro veces más grande), la varianza cae a un cuarto, así que la desviación típica baja por un factor 2. Para llegar a varianza despreciable necesitarías acercarte al gradiente full-batch, lo cual deja de ser viable computacionalmente con datasets reales.

Esta varianza estructural conecta tres preguntas distintas que a primera vista parecen no relacionadas. Primero, explica por qué hay que decaer el learning rate (ver [[LR decay]]): si no, las iteradas orbitan el óptimo en lugar de converger. Segundo, define el [[Batch size crítico]], es decir, el tamaño de batch a partir del cual la varianza ya es lo bastante baja como para que aumentar más $B$ deje de comprar velocidad de convergencia. Tercero, la varianza está acoplada inversamente con la [[Coherencia de gradientes]]: si los ejemplos están bien alineados (cosenos por pares positivos), la varianza del promedio es baja y la señal domina; si están confusos, lo contrario.

## Enlaces

- Definiciones formales y descomposición empírica: [[A Study of Gradient Variance in Deep Learning]]
- Conexión con batch size y escalado de entrenamiento: [[An Empirical Model of Large-Batch Training]]
- Reducción de varianza por control variates en SGD: [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]]
- Por qué la reducción de varianza fracasa en deep learning: [[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning]]
