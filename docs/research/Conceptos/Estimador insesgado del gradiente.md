
Un estimador estocástico del gradiente es insesgado si $\mathbb{E}[\hat g] = \nabla f$, es decir, si su esperanza coincide exactamente con el gradiente que querrías calcular. Es la propiedad mínima que garantiza que SGD esté optimizando la pérdida correcta, no una distorsión sistemática de ella.

[[Mini-batch SGD]] lo cumple trivialmente cuando los ejemplos se muestrean i.i.d. de la distribución empírica. Si la pérdida total es $f(w) = \tfrac{1}{n}\sum_{i=1}^n \ell_i(w)$, el estimador $\hat g = \tfrac{1}{B}\sum_{i\in S}\nabla \ell_i(w)$ con $S$ uniforme tiene esperanza igual al gradiente full-batch por linealidad. Esa propiedad es lo que permite demostrar convergencia incluso con paso constante (a un entorno del óptimo) y lo que justifica usar las garantías clásicas de optimización estocástica.

Variantes más sofisticadas como SVRG mantienen el insesgamiento aplicando un truco de control variate. Construyen $\hat g_{\text{SVRG}} = \nabla\ell_i(w) - \nabla\ell_i(\tilde w) + \tilde\mu$, donde $\tilde w$ es un snapshot congelado del modelo y $\tilde\mu$ es su gradiente full-batch precalculado. La esperanza del término que se resta es exactamente $\tilde\mu$, así que la corrección no introduce sesgo, pero reduce drásticamente la varianza cuando el iterado actual está cerca del snapshot. Un ejemplo de cuándo aporta: en regresión logística sobre datos tabulares, SVRG converge linealmente y supera a SGD con tasa decreciente.

El insesgamiento se rompe cuando la estructura finite-sum se pierde. Si dentro de la pérdida usas batch normalization, las estadísticas de normalización dependen del minibatch $S$ que estás procesando en ese momento, así que la "función" $\nabla\ell_i$ no es la misma para distintos batches. Lo mismo pasa con data augmentation aleatoria (cada batch ve transformaciones distintas) y con dropout. En esos casos $\nabla\ell_i$ ya no es una función pura del peso, lo que invalida los supuestos de los métodos de reducción de varianza y explica empíricamente por qué SVRG fracasa en redes profundas modernas a pesar de su superioridad teórica en convexo.

## Enlaces

- Construcción del estimador SVRG y su análisis: [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]]
- Cómo BN, augmentation y dropout rompen finite-sum: [[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning]]
- Análisis de varianza del estimador mini-batch en redes: [[A Study of Gradient Variance in Deep Learning]]
