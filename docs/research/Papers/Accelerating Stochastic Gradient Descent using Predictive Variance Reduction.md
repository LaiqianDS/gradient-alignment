---
authors:
  - Rie Johnson
  - Tong Zhang
year: 2013
status: read
relevance: low
last_review: 2026-05-07
url: https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
---

# Accelerating Stochastic Gradient Descent using Predictive Variance Reduction

## Summary

**Contextualización.** El trabajo aborda el problema canónico de minimización empírica de la forma $P(w) = \frac{1}{n} \sum_{i=1}^{n} \psi_i(w)$, donde cada $\psi_i$ representa la pérdida asociada al ejemplo $i$ (por ejemplo, mínimos cuadrados o regresión logística regularizada). El descenso por gradiente clásico (GD) requiere evaluar las $n$ derivadas en cada paso, lo que resulta prohibitivo a gran escala. El descenso por gradiente estocástico (SGD) reduce el coste por iteración a $1/n$ del de GD al muestrear un único índice $i_t$ y aplicar $w^{(t)} = w^{(t-1)} - \eta_t \nabla \psi_{i_t}(w^{(t-1)})$. Sin embargo, dado que $\nabla \psi_{i_t}$ solo coincide con $\nabla P$ en esperanza, el estimador introduce una varianza intrínseca que obliga a usar tasas de aprendizaje decrecientes $\eta_t = O(1/t)$, degradando la convergencia a una tasa sublineal $O(1/t)$ en lugar de la lineal $O((1 - \gamma/L)^t)$ alcanzable por GD bajo suavidad y convexidad fuerte (con constante de Lipschitz $L$ y módulo de convexidad fuerte $\gamma$). Este compromiso entre coste por iteración y velocidad de convergencia es el problema central que motiva el artículo. Trabajos previos (SAG de Le Roux et al., 2012; SDCA de Shalev-Shwartz y Zhang, 2012) ya habían conseguido convergencia lineal mediante reducción de varianza, pero ambos requerían almacenar todos los gradientes individuales o las variables duales, lo que los hace inviables en problemas grandes o no convexos como el entrenamiento de redes neuronales.

**Aportación.** Los autores proponen **Stochastic Variance Reduced Gradient (SVRG)**, un método que reduce explícitamente la varianza del estimador estocástico del gradiente sin requerir almacenamiento de gradientes individuales. La contribución es triple: (1) un algoritmo aplicable a problemas complejos donde SAG/SDCA no son viables (predicción estructurada, redes neuronales); (2) una demostración de convergencia lineal significativamente más simple e intuitiva que las de SAG y SDCA; y (3) una explicación basada en reducción de varianza que se extiende como heurística al caso no convexo, sirviendo como acelerador local para SGD en redes neuronales.

**Metodología.** El núcleo del método consiste en mantener un *snapshot* $\tilde{w}$ del vector de parámetros que se actualiza cada $m$ iteraciones de SGD. En cada *snapshot* se calcula el gradiente completo $\tilde{\mu} = \nabla P(\tilde{w}) = \frac{1}{n} \sum_i \nabla \psi_i(\tilde{w})$ con un solo barrido sobre los datos. La regla de actualización interior reemplaza el gradiente estocástico estándar por el estimador con control variate:

$$w^{(t)} = w^{(t-1)} - \eta_t \left( \nabla \psi_{i_t}(w^{(t-1)}) - \nabla \psi_{i_t}(\tilde{w}) + \tilde{\mu} \right).$$

Como $\mathbb{E}_{i_t}[\nabla \psi_{i_t}(\tilde{w}) - \tilde{\mu}] = 0$, el estimador permanece insesgado: $\mathbb{E}[w^{(t)} | w^{(t-1)}] = w^{(t-1)} - \eta_t \nabla P(w^{(t-1)})$. La clave es que cuando $w^{(t-1)} \to w_*$ y $\tilde{w} \to w_*$, los términos $\nabla \psi_{i_t}(w^{(t-1)}) - \nabla \psi_{i_t}(\tilde{w}) + \tilde{\mu}$ tienden a cero, eliminando la varianza asintóticamente y permitiendo emplear una tasa de aprendizaje $\eta$ constante. El pseudocódigo (Figura 1 del paper) define dos variantes: **opción I**, donde $\tilde{w}_s = w_m$ (último iterado del bucle interior); y **opción II**, donde $\tilde{w}_s = w_t$ con $t$ elegido uniformemente al azar en $\{0, \ldots, m-1\}$. La opción I es la natural en la práctica; la opción II es la que sustenta el análisis teórico. El **análisis de convergencia** se formaliza en el Teorema 1: bajo suavidad de cada $\psi_i$ (constante $L$), convexidad de cada $\psi_i$ y convexidad fuerte de $P$ (módulo $\gamma > 0$), si $m$ es suficientemente grande para que $\alpha = \frac{1}{\gamma \eta (1 - 2L\eta) m} + \frac{2L\eta}{1 - 2L\eta} < 1$, entonces $\mathbb{E}\,P(\tilde{w}_s) - P(w_*) \le \alpha^s \, [P(\tilde{w}_0) - P(w_*)]$, es decir, **convergencia geométrica en esperanza**. La demostración acota $\mathbb{E}\|v_t\|_2^2 \le 4L[P(w_{t-1}) - P(w_*) + P(\tilde{w}) - P(w_*)]$ usando la desigualdad $\|a+b\|_2^2 \le 2\|a\|_2^2 + 2\|b\|_2^2$, la propiedad de varianza $\mathbb{E}\|\xi - \mathbb{E}\xi\|_2^2 \le \mathbb{E}\|\xi\|_2^2$ y la cota $n^{-1} \sum_i \|\nabla \psi_i(w) - \nabla \psi_i(w_*)\|_2^2 \le 2L[P(w) - P(w_*)]$. En el régimen indicativo $L/\gamma = n$, eligiendo $\eta = 0{,}1/L$ y $m = O(n)$ se obtiene $\alpha = 1/2$, lo que requiere procesar $n \ln(1/\epsilon)$ ejemplos para precisión $\epsilon$, frente a los $n^2 \ln(1/\epsilon)$ de GD estándar. Para problemas suaves no fuertemente convexos se obtiene $O(1/T)$, mejorando el $O(1/\sqrt{T})$ de SGD. En problemas no convexos se sugiere inicializar $\tilde{w}_0$ cerca de un mínimo local mediante SGD y aplicar SVRG para acelerar la convergencia local. El paper también ofrece una **interpretación unificadora**: SDCA puede reescribirse como un método de reducción de varianza para SGD, ya que cuando $(w, \alpha) \to (w_*, \alpha_*)$, $\frac{1}{n} \sum_i (\nabla \phi_i(w) + \lambda n \alpha_i)^2 \to 0$, mostrando paralelismo conceptual con SVRG.

**Datasets y modelos.** Los experimentos cubren tanto el régimen convexo como el no convexo. En **convexo** se usa regresión logística multiclase L2-regularizada sobre **MNIST** ($\lambda = 10^{-4}$) y regresión logística L2-regularizada sobre **rcv1.binary** y **covtype.binary** (LIBSVM), **protein** (KDD Cup) y **CIFAR-10** ($\lambda = 10^{-3}$ para CIFAR-10 y $10^{-5}$ para los demás). En **no convexo** se entrenan redes neuronales con una capa oculta totalmente conectada de 100 nodos, activación sigmoide, salida softmax de 10 clases, regularización L2 y mini-batches de tamaño 10 sobre MNIST ($\lambda = 10^{-4}$) y CIFAR-10 ($\lambda = 10^{-3}$). El intervalo de actualización se fija en $m = 2n$ (convexo) y $m = 5n$ (no convexo); SVRG se inicializa con 1 iteración (convexo) o 10 iteraciones (no convexo) de SGD.

**Métricas.** Los autores reportan: (1) **training loss** $P(w)$ frente al número de gradientes calculados normalizado por $n$; (2) **suboptimalidad** o residuo de pérdida $P(w) - P(w_*)$, con $w_*$ estimado mediante GD ejecutado durante mucho tiempo; (3) **varianza del incremento de pesos** $-\eta(\nabla \psi_i(w) - \nabla \psi_i(\tilde{w}) + \tilde{\mu})$ comparada con la de SGD y SDCA; y (4) **test error rate** sobre los conjuntos de prueba (con divisiones aleatorias 50/50 para protein y covtype, que carecen de etiquetas de test). El eje $x$ se mide en número de cómputos de gradiente dividido por $n$ para garantizar comparaciones equitativas.

**Conclusiones.** Los resultados experimentales confirman las predicciones teóricas. En el caso convexo, SVRG es competitivo con SDCA (curvas casi solapadas) y converge claramente más rápido que el SGD mejor ajustado con planificación de tasa de aprendizaje (decaimiento exponencial $\eta(t) = \eta_0 a^{\lfloor t/n \rfloor}$ o $t$-inverso $\eta(t) = \eta_0(1 + b\lfloor t/n \rfloor)^{-1}$). La varianza de SVRG y SDCA decae exponencialmente, mientras que la de SGD con tasa fija permanece alta y la del mejor SGD solo cae por la decadencia forzada de $\eta(t)$. En el caso no convexo (redes neuronales), SVRG reduce la varianza y supera al SGD mejor ajustado, mostrando que la idea de reducción de varianza explícita es aplicable más allá del régimen convexo, con la ventaja decisiva sobre SAG/SDCA de no requerir almacenamiento de gradientes. Los autores señalan como línea futura la validación con redes más grandes y profundas, donde el coste de entrenamiento es crítico.

## Medición y pipeline

**Rol del paper.** SVRG no aporta una métrica para loggear, sino un algoritmo de optimización. Su relevancia para el TFG es indirecta pero fundamental: si la reducción explícita de la varianza del estimador del gradiente acelera la convergencia (resultado teórico del Teorema 1 y confirmado empíricamente), entonces **medir la varianza del gradiente** durante el entrenamiento constituye un proxy plausible de eficiencia de optimización. Esta lectura justifica teóricamente la elección de NGV (Faghri et al.) y GNS (McCandlish et al.) como métricas centrales del eje varianza en el pipeline.

**Métrica derivada.** La cantidad de interés es la traza de la covarianza del gradiente por muestra, $\mathrm{tr}(\mathrm{Cov}(\nabla L_i))$, que cuantifica la varianza total del estimador estocástico. No es una métrica definida en este paper, pero sí consistente con su marco conceptual. Se puede computar bien con gradientes per-sample (coste alto), bien mediante el **estimador two-batch** derivado en la literatura de large-batch training (McCandlish et al., 2018), no en SVRG directamente.

**Cuándo computar.** Por época o cada $K$ pasos, en función del coste; el muestreo de $N$ ejemplos para la estimación condiciona la fidelidad.

**Integración pipeline (estimador two-batch).**

```
g_small = mean_grad(batch B_small)
g_big   = mean_grad(batch B_big)
S_small = ||g_small||^2
S_big   = ||g_big||^2
tr_cov  = (S_small * B_small - S_big * B_big) / (B_big - B_small)
log(step, tr_cov)
```

**Consideraciones.** SVRG podría usarse como **baseline experimental opcional**: entrenar con SVRG y verificar que la varianza medida decae como predice la teoría serviría de sanity check del estimador. Queda fuera del scope cerrado del TFG.

**Decisión.** No se loggea ninguna métrica directa de este paper; se cita como justificación teórica del eje varianza.

## Notes
Es una referencia fundacional, está en NeurIPS, todo el mundo en optimización lo conoce. Una frase: "SVRG reduce explícitamente la varianza del estimador del gradiente para acelerar la convergencia en problemas suaves"

Métodos que aprovechan la varianza del gradiente
- Establecer que la comunidad reconoce que la varianza tiene contenido informativo.
- Marcar que tú no propones un nuevo optimizador, sino una _señal de diagnóstico_ derivada de las mismas magnitudes.
