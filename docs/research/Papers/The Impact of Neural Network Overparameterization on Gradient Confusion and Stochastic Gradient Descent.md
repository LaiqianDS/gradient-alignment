---
authors:
  - Karthik Abinav Sankararaman
  - Soham De
  - Zheng Xu
  - W. Ronny Huang
  - Tom Goldstein
year: 2020
status: read
relevance: high
url: https://proceedings.mlr.press/v119/sankararaman20a/sankararaman20a.pdf
tfg_role:
  - metric
tfg_note: "Origen de `gradient_confusion` (mínimo coseno entre gradientes de pares de ejemplos; más confusión = SGD más lento). Caso worst-case de la familia alineación; la sobreparametrización por anchura la reduce."
---

# The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent

[PDF](https://proceedings.mlr.press/v119/sankararaman20a/sankararaman20a.pdf) · ICML 2020

## Summary

### Contextualización

El descenso por gradiente estocástico y sus variantes con momentum constituyen la rutina estándar de optimización en redes neuronales profundas, en buena medida porque empíricamente convergen rápido y generalizan bien. Sin embargo, la teoría clásica de optimización estocástica para funciones convexas predice que la tasa de aprendizaje debe decaer con el tiempo, típicamente como $O(1/k)$, para garantizar convergencia $\varepsilon$-precisa al minimizador. En la práctica, los profesionales entrenan redes sobreparametrizadas (con muchos más parámetros que muestras de entrenamiento) usando tasas de aprendizaje constantes durante la mayor parte del entrenamiento sin observar estancamiento. La teoría reciente ha introducido la idea de un *noise floor* cuya magnitud depende de la varianza de los gradientes en el minimizador, pero esto no explica plenamente por qué SGD escala tan bien en el régimen sobreparametrizado ni cómo influyen las decisiones arquitectónicas (anchura, profundidad, conexiones residuales, batch normalization) sobre la dinámica del entrenamiento. Es conocimiento común que, bajo inicializaciones gaussianas estándar (Glorot y Bengio, He et al.), las redes más profundas son más difíciles de entrenar, lo que motivó innovaciones como inicializaciones cuidadosas, conexiones residuales y batch normalization. Este trabajo busca un marco teórico unificador que explique por qué la sobreparametrización facilita el entrenamiento con SGD y cómo cada decisión arquitectónica afecta a esa facilidad.

### Aportación

La contribución principal es la introducción del concepto formal de **gradient confusion** (confusión de gradientes), parametrizado por una constante $\eta \geq 0$ que mide la máxima similitud negativa entre los gradientes per-sample. Formalmente, dado un conjunto de $N$ funciones de pérdida $\{f_i\}$, se dice que tienen gradient confusion acotada por $\eta$ si para todo $i \neq j$ y un $w \in \mathbb{R}^d$ fijado se cumple

$$\langle \nabla f_i(w), \nabla f_j(w) \rangle \geq -\eta.$$

La interpretación operativa es transparente. Cuando $\eta$ es pequeña los gradientes de distintas muestras interactúan de manera armoniosa: un paso que reduce la pérdida en un mini-batch no perjudica sustancialmente a los demás, y SGD progresa con rapidez incluso bajo learning rate constante. Cuando $\eta$ es grande aparecen pares de gradientes fuertemente anti-correlacionados, los mini-batches "tiran" en direcciones contrarias y el progreso neto se diluye. La métrica permite ligar de forma directa la geometría del problema con la velocidad de convergencia y, a diferencia de la varianza de gradientes o de la gradient diversity, no requiere asumir gradientes con norma acotada y captura específicamente correlaciones negativas pares.

### Metodología

El trabajo combina análisis teórico con experimentos extensivos. En el plano teórico, los autores prueban primero que bajo suavidad de Lipschitz (A1) y la condición Polyak-Lojasiewicz (A2), el Teorema 3.1 muestra que SGD con tasa de aprendizaje constante converge linealmente a un entorno del mínimo cuyo tamaño es proporcional a $\alpha\eta/(1-\rho)$; el Teorema 3.2 extiende el resultado al caso suave no convexo con cota (constantes exactas a verificar contra el PDF)

$$\min_t \mathbb{E}\|\nabla F(w_t)\|^2 \leq \frac{\rho(F(w_1) - F^*)}{T} + \rho\eta.$$

Cuando $\eta = O(\varepsilon)$, SGD alcanza precisión $\varepsilon$ en $T = O(\log(1/\varepsilon))$ iteraciones sin necesidad de decaer $\alpha$. La sección 4 estudia el efecto de la arquitectura sobre redes totalmente conectadas y convolucionales con activaciones acotadas y diferenciables (sigmoid, tanh, softmax, ReLU bajo norma unidad) y pesos inicializados según la estrategia 4.1 ($W_0 \in \mathbb{R}^{\ell \times d}$ con entradas $\mathcal{N}(0, 1/d)$ y $W_p$ con entradas $\mathcal{N}(0, 1/(\kappa\, \ell_{p-1}))$). El Teorema 4.1 prueba que la cota de gradient confusion (eq. 3) se cumple con probabilidad al menos

$$1 - \beta \cdot \exp(-c_1 \kappa^2 \ell^2) - N^2 \cdot \exp\!\left(-\frac{c\, \ell^2\, \beta\, (\eta - 4)^2}{64\, \zeta_0^4\, (\beta + 2)^4}\right),$$

donde $\ell$ es la anchura máxima, $\beta$ la profundidad y $\zeta_0 := 2\sqrt{\beta}$ (verificado contra el PDF, p. 5: el segundo exponente lleva $\ell^2$, no $\kappa^2$). La lectura es que la anchura mejora la *trainability* y la profundidad la empeora. La sección 5 generaliza este resultado mostrando, vía el Teorema 5.1 y el Corolario 5.1, que bajo pesos pequeños ($\|W_i\| \leq 1$) la dependencia de la anchura desaparece y la probabilidad de cumplir la cota decae con la profundidad para todos los pesos en una bola alrededor del minimizador. La sección 6 cierra el análisis con inicializaciones ortogonales: el Teorema 6.1 demuestra que para redes lineales con matrices de pesos ortogonales y datos i.i.d. en la esfera unidad, la cota se cumple con probabilidad al menos $1 - N^2 \exp(-c\, d\, \eta^2)$, independientemente de la profundidad, lo que sugiere que las inicializaciones ortogonales son una vía prometedora para entrenar redes muy profundas. En el plano empírico, los experimentos miden la mínima similitud coseno entre gradientes y las distribuciones de similitudes de pares para distintas profundidades, anchuras, y combinaciones de skip connections y batch normalization.

### Datasets y modelos

Los experimentos cubren tareas de clasificación de imágenes sobre **CIFAR-10**, **CIFAR-100** y **MNIST** (CIFAR-10 en el cuerpo, el resto en el apéndice A). Los modelos incluyen **wide residual networks (WRNs)** (Zagoruyko y Komodakis, 2016), **redes convolucionales (CNNs)** y **multi-layer perceptrons (MLPs)**, con la notación CNN-$\beta$-$\ell$ para WRNs de profundidad $\beta$ y factor de anchura $\ell$. Para los experimentos sobre profundidad se usan $\beta \in \{16, 22, 28, 34, 40\}$ con anchura fija 2; para anchura, profundidad fija 16 con factores $\{2, 3, 4, 5, 6\}$; y para el estudio de batch norm y skip connections se llega hasta $\beta \in \{16, 22, 28, 34, 40, 52, 76, 100\}$. El optimizador es SGD sin momentum durante 200 epochs, mini-batches de tamaño 128, learning rate decaído por factor 10 en epochs 80 y 160, inicialización MSRA (He et al., 2015), y dropout y weight decay desactivados. Cada experimento se repite cinco veces.

### Métricas

Las métricas reportadas son los **valores de gradient confusion** —aproximados como el mínimo y la densidad de las similitudes coseno por pares: el protocolo exacto (Sec. 7, p. 7) muestrea **100 pares de mini-batches de tamaño 128** al final de cada epoch, calcula el gradiente de cada mini-batch y los cosenos por par, es decir, 100 cosenos por época (no $\binom{100}{2}$ pares de un pool de 100 gradientes, como afirmaban versiones previas de estas notas)—, el **training loss** a lo largo de los 200 epochs, la **test accuracy final**, comparativas a distintas **profundidades y anchuras**, y ablations con **batch normalization on/off** y **skip connections on/off**. Las figuras 2, 3 y 4 sintetizan estos resultados sobre CIFAR-10.

### Conclusiones

La sobreparametrización por anchura reduce la gradient confusion y acelera el entrenamiento de SGD, mientras que el aumento de profundidad la incrementa y dificulta la convergencia. Bajo inicializaciones gaussianas estándar, esto explica la dificultad de entrenar redes muy profundas sin medidas adicionales. Los experimentos confirman que **batch normalization** y **skip connections**, sobre todo en combinación, reducen drásticamente la gradient confusion incluso en redes muy profundas, justificando su éxito empírico en arquitecturas residuales. Las **inicializaciones ortogonales** consiguen, en el caso lineal, gradient confusion independiente de la profundidad, apuntando a una dirección prometedora para entrenar modelos profundos con activaciones no lineales. La gradient confusion ofrece, por tanto, un criterio teórico y medible de trainability que conecta arquitectura, inicialización y dinámica de SGD, y abre la puerta a usarla como proxy para diseñar redes más eficientes y para investigar la conexión con generalización. En resumen, una baja gradient confusion acelera SGD mientras que una alta lo ralentiza; redes más anchas tienden a reducirla, redes más profundas a aumentarla, y los mitigantes habituales en redes profundas (inicializaciones específicas, batch norm, skip connections) actúan precisamente sobre esta cantidad.

## Medición y pipeline

**Métrica concreta.** Se adopta la *gradient confusion* de Sankararaman et al. La definición teórica es una cota inferior sobre los productos internos por pares entre gradientes per-sample (o per-mini-batch): para todo $i \neq j$, $\langle \nabla f_i(w), \nabla f_j(w) \rangle \geq -\eta$. La constante $\eta$ se interpreta como el peor caso de anti-correlación: cuando es pequeña, ningún par de gradientes tira fuertemente en direcciones opuestas y SGD progresa con armonía; cuando es grande, existe al menos un par con coseno marcadamente negativo que frena el avance neto del optimizador. Empíricamente se estima en su versión normalizada vía la **mínima similitud coseno** entre pares de gradientes evaluados en el $w$ actual,

$$\hat\eta = -\min_{i\neq j}\, \cos(\nabla f_i, \nabla f_j) = -\min_{i\neq j}\, \frac{\langle \nabla f_i, \nabla f_j \rangle}{\|\nabla f_i\|\,\|\nabla f_j\|},$$

reportando además la **densidad completa** de los cosenos pares para no perder información de masa. En redes bien entrenadas con anchura suficiente, esta densidad se concentra cerca de cero con cola muy ligera hacia los negativos; en redes profundas mal inicializadas aparece una cola pesada con el mínimo desplazado claramente a la izquierda de $0$, signo de confusión activa. Conviene notar que el estimador empírico $\hat\eta = -\min\cos$ puede ser **negativo** cuando todos los pares están positivamente alineados (es decir, $\min\cos > 0$): el $\eta \geq 0$ teórico se cumple solo porque $\eta$ se **define** como una cota inferior no negativa, mientras que $\hat\eta < 0$ simplemente significa que no se observó ningún par confundido.

**Entradas.** Gradientes calculados en el mismo $w$. El protocolo del paper (Sec. 7): **100 pares de mini-batches de tamaño 128** muestreados al final de cada epoch → 100 cosenos por época, sobre gradientes **per-minibatch**. La v1 del TFG implementa en cambio la versión **per-sample** (un gradiente por ejemplo del probe compartido, $M = 256$) y forma **todos** los $\binom{M}{2}$ pares — más pares que el paper, no un recorte, pero de otra distribución: por la Sec. 8 ($\eta \propto 1/B^2$ con gradientes promediados) los cosenos per-sample tienen colas más pesadas y **no son comparables** con las figuras 2–4 del paper.

**Cuándo computar.** Una medición por epoch (granularidad típica), manteniendo $w$ fijo durante la medición. En entrenamientos largos puede espaciarse a cada $K$ pasos. La métrica es esencialmente **global** sobre el vector de parámetros aplanado; opcionalmente puede desagregarse **por capa** para localizar la fuente de confusión, lo que resulta especialmente útil al estudiar el efecto de batch norm o de las skip connections en bloques profundos.

**Coste.** $M$ pasadas forward+backward sin actualización más $M^2$ productos internos baratos una vez cacheados los gradientes aplanados. El coste de cómputo es **moderado** y, en la práctica, se amortiza reutilizando los mismos mini-batches del *batch-grad sweep* compartido con `cos_sim_batches`, `gradient_disparity` y `normalized_variance`. El cuello de botella es la memoria $M\cdot P$: para ResNet-18 fp32 con $M=50$ son aproximadamente $2{,}3$ GB, lo que obliga a chunkear el producto $G G^\top$ o a desplazar los gradientes a CPU.

**Integración pipeline.** Pseudocódigo del estimador:

```python
def gradient_confusion(model, loss_fn, loader, M=100):
    model.eval()  # congelar BN/Dropout durante la medición
    grads = []
    for _ in range(M):
        xb, yb = next(loader)
        model.zero_grad()
        loss_fn(model(xb), yb).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(g / g.norm())
    G = torch.stack(grads)                        # (M, P)
    sims = G @ G.T                                # matriz M×M de cosenos
    pairs = sims[torch.triu_indices(M, M, offset=1).unbind()]
    eta_hat = -pairs.min().item()                 # peor caso (confusion/eta)
    min_cos = pairs.min().item()                  # ↑ mejor, escala coseno
    return eta_hat, min_cos, pairs                # escalar + escalar + histograma
```

**Claves de logging.** Por checkpoint se registran `confusion/eta` ($\hat\eta = -\min\cos$; el flip de signo es **convención del TFG** — el paper no define $\hat\eta$, reporta el mínimo coseno crudo en sus figuras), `confusion/min_cos` (el mínimo coseno crudo, preferido por consistencia con las demás métricas coseno del TFG donde *mayor es mejor*), `confusion/median_cos`, `confusion/p05_cos` y `confusion/frac_neg`, que correlacionan tanto el peor caso como la masa de la distribución con las métricas de eficiencia (epochs-to-threshold, AUC test loss, best test loss). El histograma completo `confusion/cos_hist` no se emite (el contrato del registry es `dict[str, float]`).

**Interpretación de la señal.** Conviene fijar la convención porque conviven dos signos en el logging. En `confusion/eta` (la $\hat\eta$ del paper, mínimo coseno **con** signo invertido) la lectura natural es **cuanto más alto, peor**: valores grandes indican al menos un par de mini-batches con gradientes en oposición que ralentizan SGD, mientras que valores cercanos a cero indican gradientes que no se contradicen y entrenamiento armonioso con learning rate constante. En `confusion/min_cos` (mínimo coseno **sin** invertir el signo) la lectura se alinea con las demás métricas coseno del TFG: **mayor `min_cos` = mejor** (gradientes más alineados o al menos no opuestos), `min_cos` muy negativo = peor (par fuertemente anti-correlacionado). Los descriptores de masa siguen la misma lógica que `min_cos`: `confusion/median_cos` y `confusion/p05_cos` van con la convención coseno (mayor = mejor), mientras que `confusion/frac_neg` (fracción de pares con coseno negativo) sigue la convención del peor caso (mayor = peor). Operativamente, el objetivo del diseño arquitectónico (anchura, batch normalization, skip connections, inicializaciones ortogonales) es mantener $\hat\eta$ tan bajo como sea posible durante el entrenamiento, y las predicciones empíricas del paper sobre CIFAR-10 son consistentes con esa lectura: anchura $\downarrow\hat\eta$, profundidad $\uparrow\hat\eta$, BN+skip $\downarrow\hat\eta$. Por construcción `eta` es un estimador de extremo y muy ruidoso: si `eta` es alto pero `median_cos` sigue cerca de cero y `p05_cos` no está muy desplazado a negativo, lo que se observa es ruido de cola y no confusion estructural; solo cuando el peor caso y la masa típica se desplazan a la vez hay un problema real de geometría que ralentizará SGD. Además, $\hat\eta = -\min\cos$ puede salir **negativo** cuando todos los pares están positivamente alineados ($\min\cos > 0$); el $\eta \geq 0$ teórico solo se sostiene porque $\eta$ se **define** como una cota inferior no negativa, de modo que un $\hat\eta < 0$ no es un error sino que indica que no se observó ningún par confundido.

**Gotchas.** Por construcción $\hat\eta$ es un estimador de **extremo** y como tal es muy sensible al ruido de muestreo: un único par especialmente desalineado puede dominar el escalar y ocultar el comportamiento típico. La práctica recomendada en el TFG es reportar simultáneamente el **percentil 5%** y la **mediana** del histograma de cosenos junto al mínimo, de modo que la señal robusta provenga de la cola y de la masa central, no del peor par aislado. Otros puntos a vigilar: BatchNorm debe fijarse en modo evaluación para no contaminar gradientes con estadísticas de batch distintas; en Adam los productos internos sobre $\nabla f$ no reflejan la dirección efectiva de actualización (modulada por momentos adaptativos), por lo que conviene reportar $\hat\eta$ sobre el **gradiente bruto** $\nabla L$ y, si se quiere comparar regímenes, opcionalmente también sobre el paso preacondicionado; los mini-batches deben muestrearse **disjuntos** sin reemplazo dentro de la medición.

## Notes

### Uso en el TFG

- **Métrica que origina:** `gradient_confusion` (familia alineación), única métrica del `METRIC_REGISTRY` derivada de este paper. Es el origen formal de la noción de anti-correlación par a par entre gradientes.
- **Fórmula clave y estimador.** Definición teórica: $\langle \nabla f_i(w), \nabla f_j(w)\rangle \ge -\eta$ con $\eta \ge 0$ para todo $i\neq j$ (Def. 2.1, producto interno crudo). En el TFG se estima la versión coseno (normalizada e invariante a escala, legitimada por la Sec. 8 del paper) sobre gradientes **per-sample** del probe: $\hat\eta = -\min_{i\neq j} \frac{g_i \cdot g_j}{\|g_i\|\,\|g_j\|}$ (el flip de signo es convención del TFG).
- **Protocolo y diferencia con el paper.** El paper muestrea 100 **pares de mini-batches** (B=128) por época → 100 cosenos per-minibatch; el TFG forma **todos** los $\binom{M}{2}$ pares de los $M = 256$ gradientes per-sample del probe (32 640 pares — un estimador más denso, pero de la distribución per-sample, con colas más pesadas que la per-minibatch del paper). Cadencia `epoch` dentro de la ventana temprana (5/10/25/50% de épocas).
- **Señal (signo).** $\hat\eta$ mayor = peor (más confusión, SGD más lento). Como `min` es un estimador de extremo ruidoso, se reporta también `min_cos` (↑ mejor, preferida por consistencia con las demás métricas coseno), `median_cos`, `p05_cos` y `frac_neg` para capturar la masa de la distribución, no solo el peor caso.
- **Decisiones de implementación.** Gradiente **bruto** $\nabla L$ (no la update preacondicionada de Adam), `model.eval()` durante la medición (congelar BN/Dropout), muestreo de batches **disjuntos** sin reemplazo. Comparte el **batch-grad sweep** con `cos_sim_batches`, `gradient_disparity` y `normalized_variance` (mismos $K$ batches; ampliar a $M=50$ es la métrica que más lo demanda junto a `normalized_variance`).
- **Pitfall principal.** Memoria $M\cdot P$ es bloqueante en ResNet-18 ($\approx 2.3$ GB fp32 para $M=50$): chunkear el producto $G G^\top$ o pasar los gradientes a CPU. NO se implementan las cotas teóricas (Teoremas 3.1–6.1) ni el OSGR-style; solo el estimador empírico. Los **ablations de overparameterización** (anchura↓confusión, profundidad↑confusión, BN+skip↓confusión) son material de discusión teórica, no se reproducen como experimento propio.

## Papers relacionados

- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — misma familia (alineación); m-coherence es el agregado complementario (media de pares) frente al peor-caso (mínimo) de gradient confusion.
- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — misma familia; cosine-stiffness es el mismo coseno entre gradientes pero per-sample y desglosado within/between clases, no el mínimo per-batch.
- [[Disparity Between Batches as a Signal for Early Stopping]] — comparte el batch-grad sweep y el mismo objeto (discrepancia entre 2 batches); GD usa distancia $\ell_2$ sin normalizar donde aquí se usa el coseno.
- [[A Study of Gradient Variance in Deep Learning]] — comparte el batch-grad sweep; `normalized_variance` es la cara de varianza estocástica del mismo barrido de gradientes de batch.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — mismo problema (alineación de gradientes como motor de generalización); el paper cita la CGH como marco conceptual de fondo.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — mismo problema (proxy barato en train-time para predecir generalización); GWA usa coseno gradiente-peso en lugar de coseno entre gradientes.

## Otros papers interesantes a revisar

- **Gradient Diversity: a Key Ingredient for Scalable Distributed Learning** (Yin, Pananjady, Lam, Papailiopoulos, Ramchandran, Bartlett, 2018) — define *gradient diversity* $\Delta_S(w) = \frac{\sum_i \|\nabla f_i\|^2}{\|\sum_i \nabla f_i\|^2}$, magnitud emparentada con la confusión que regula el batch size admisible sin pérdida de velocidad; el propio Sankararaman la contrasta. arXiv:1706.05699.
- **Stochastic Training is Not Necessary for Generalization** (Geiping, Goldstein et al., 2021) — del mismo grupo (Goldstein); estudia hasta qué punto el ruido de SGD (y por tanto la confusión/diversidad de gradientes) es necesario para generalizar, complementa la lectura sobre el rol del gradiente bruto. arXiv:2109.14119.
- **Gradient Descent Happens in a Tiny Subspace** (Gur-Ari, Roberts, Dyer, 2018) — muestra que el gradiente vive en un subespacio de baja dimensión definido por los top eigenvectores de la Hessiana; relevante para entender por qué los cosenos entre gradientes de batch no son ruido puro. arXiv:1812.04754.
