---
authors: ["Karthik Abinav Sankararaman", "Soham De", "Zheng Xu", "W. Ronny Huang", "Tom Goldstein"]
year: 2020
status: to-read
relevance: high
last_review: "2026-05-07"
url: "https://proceedings.mlr.press/v119/sankararaman20a/sankararaman20a.pdf"
---

# The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent

[PDF](https://proceedings.mlr.press/v119/sankararaman20a/sankararaman20a.pdf) · ICML 2020

## Summary

### Contextualización

El descenso por gradiente estocástico (SGD) y sus variantes con momentum son la rutina estándar de optimización en redes neuronales profundas, en buena medida porque empíricamente convergen rápido y generalizan bien. Sin embargo, la teoría clásica de optimización estocástica para funciones convexas predice que la tasa de aprendizaje debe decaer con el tiempo (típicamente como $O(1/k)$) para garantizar convergencia $\varepsilon$-precisa al minimizador. En la práctica, los profesionales entrenan redes sobreparametrizadas (con muchos más parámetros que muestras de entrenamiento) usando tasas de aprendizaje constantes durante la mayoría del entrenamiento sin observar estancamiento. La teoría reciente ha introducido la idea de un "noise floor" cuya magnitud depende de la varianza de los gradientes en el minimizador, pero esto sigue sin explicar plenamente por qué SGD escala tan bien en el régimen sobreparametrizado ni cómo influyen las decisiones arquitectónicas (anchura, profundidad, conexiones residuales, batch normalization) en la dinámica del entrenamiento. Es conocimiento común que, bajo inicializaciones gaussianas estándar (Glorot/Bengio, He et al.), las redes más profundas son más difíciles de entrenar, lo que motivó innovaciones como inicializaciones cuidadosas, conexiones residuales y batch normalization. Este trabajo busca un marco teórico unificador que explique por qué la sobreparametrización facilita el entrenamiento con SGD y cómo cada decisión arquitectónica afecta a esa facilidad.

### Aportación

Los autores introducen un concepto formal llamado **gradient confusion** (confusión de gradientes), parametrizado por una constante $\eta \geq 0$ que mide la máxima similitud negativa entre los gradientes per-sample. Formalmente, dado un conjunto de $N$ funciones de pérdida $\{f_i\}$, se dice que tienen gradient confusion acotada por $\eta$ si para todo $i \neq j$ y para un $w \in \mathbb{R}^d$ fijado se cumple

$$\langle \nabla f_i(w), \nabla f_j(w) \rangle \geq -\eta.$$

Cuando $\eta$ es grande, los gradientes de distintas muestras pueden estar fuertemente anti-correlacionados (los mini-batches "tiran" en direcciones contrarias), lo que ralentiza SGD; cuando $\eta$ es pequeña, los gradientes interactúan de manera armoniosa y SGD progresa rápidamente. Esta noción permite ligar de forma directa la geometría del problema con la velocidad de convergencia y, a diferencia de la varianza de gradientes o la diversidad de gradientes, no requiere asumir gradientes con norma acotada y captura específicamente correlaciones negativas pares.

### Metodología

El trabajo combina análisis teórico con experimentos extensivos. En el plano teórico:

1. **Convergencia de SGD bajo low gradient confusion**: bajo suavidad de Lipschitz (A1) y la condición Polyak-Lojasiewicz (A2), el Teorema 3.1 muestra que SGD con tasa de aprendizaje constante converge linealmente a un entorno del mínimo cuyo tamaño es proporcional a $\alpha\eta/(1-\rho)$; el Teorema 3.2 extiende el resultado al caso suave no convexo, con cota

$$\min_t \mathbb{E}\|\nabla F(w_t)\|^2 \leq \frac{\rho(F(w_1) - F^*)}{T} + \rho\eta.$$

Cuando $\eta = O(\varepsilon)$, SGD alcanza precisión $\varepsilon$ en $T = O(\log(1/\varepsilon))$ iteraciones sin necesidad de decaer $\alpha$.
2. **Efecto de la arquitectura en inicializaciones gaussianas**: la sección 4 analiza redes totalmente conectadas y convolucionales con activaciones acotadas y diferenciales acotadas (sigmoid, tanh, softmax, ReLU bajo norma unidad) y pesos inicializados con la estrategia 4.1 ($W_0 \in \mathbb{R}^{\ell \times d}$ con entradas $\mathcal{N}(0, 1/d)$ y $W_p$ con entradas $\mathcal{N}(0, 1/(\kappa\, \ell_{p-1}))$). El Teorema 4.1 prueba que la cota de gradient confusion (eq. 3) se cumple con probabilidad al menos

$$1 - \beta \cdot \exp(-c_1 \kappa^2 \ell^2) - N^2 \cdot \exp\!\left(-\frac{c_2\, \kappa^2\, \beta\, (\eta - 4)^2}{64\, \zeta_0^4\, (\beta + 2)^4}\right),$$

donde $\ell$ es la anchura máxima y $\beta$ la profundidad. Esto se traduce en que la anchura mejora la *trainability* y la profundidad la empeora.
3. **Resultado más general sobre profundidad**: la sección 5 (Teorema 5.1, Corolario 5.1) demuestra que, bajo la suposición de pesos pequeños ($\|W_i\| \leq 1$), la dependencia de la anchura desaparece y la probabilidad de cumplir la cota decae con la profundidad para todos los pesos en una bola alrededor del minimizador.
4. **Inicializaciones ortogonales (sección 6)**: el Teorema 6.1 demuestra que para redes lineales con matrices de pesos ortogonales y datos i.i.d. en la esfera unidad, la cota se cumple con probabilidad al menos $1 - N^2 \exp(-c\, d\, \eta^2)$, independientemente de la profundidad $\beta$. Esto sugiere que las inicializaciones ortogonales son una vía prometedora para entrenar redes muy profundas.

En el plano empírico, los experimentos miden la mínima similitud coseno entre gradientes y las distribuciones de similitudes de pares para distintas profundidades, anchuras, y combinaciones de skip connections y batch normalization.

### Datasets y modelos

Los experimentos cubren tareas de clasificación de imágenes en **CIFAR-10**, **CIFAR-100** y **MNIST** (los resultados de CIFAR-10 se presentan en el cuerpo y el resto en el apéndice A). Los modelos incluyen **wide residual networks (WRNs)** (Zagoruyko & Komodakis, 2016), **redes convolucionales (CNNs)** y **multi-layer perceptrons (MLPs)**. Se emplea la notación CNN-$\beta$-$\ell$ para denotar WRNs con profundidad $\beta$ y factor de anchura $\ell$. Para los experimentos sobre profundidad se usan $\beta \in \{16, 22, 28, 34, 40\}$ con anchura fija 2; para anchura, profundidad fija 16 con factores $\{2, 3, 4, 5, 6\}$; y para el estudio de batch norm/skip connections se llega hasta $\beta \in \{16, 22, 28, 34, 40, 52, 76, 100\}$. El optimizador es SGD sin momentum, 200 epochs, mini-batches de tamaño 128, learning rate decaído por factor 10 en epochs 80 y 160, inicialización MSRA (He et al., 2015), y dropout y weight decay desactivados. Cada experimento se repite 5 veces.

### Métricas

Las métricas reportadas son: (i) **valores de gradient confusion**, aproximados como el mínimo y la densidad de las similitudes coseno por pares calculadas sobre 100 pares de mini-batches al final de cada epoch; (ii) **training loss** a lo largo de los 200 epochs; (iii) **test accuracy final** sobre el conjunto de test; (iv) comparativas a distintas **profundidades y anchuras**; y (v) ablations con **batch normalization on/off** y **skip connections on/off**. Las figuras 2, 3 y 4 sintetizan estos resultados para CIFAR-10.

### Conclusiones

La sobreparametrización por anchura reduce la gradient confusion y acelera el entrenamiento de SGD, mientras que el aumento de profundidad la incrementa y dificulta la convergencia. Bajo inicializaciones gaussianas estándar, esto explica la dificultad de entrenar redes muy profundas sin medidas adicionales. Los experimentos confirman que **batch normalization** y **skip connections**, especialmente en combinación, reducen drásticamente la gradient confusion incluso en redes muy profundas, lo que justifica su éxito empírico en arquitecturas residuales. Las **inicializaciones ortogonales** consiguen, en el caso lineal, gradient confusion independiente de la profundidad, apuntando a una dirección prometedora para entrenar modelos profundos con activaciones no lineales. La gradient confusion ofrece, por tanto, un criterio teórico y medible de trainability que conecta arquitectura, inicialización y dinámica de SGD, y abre la puerta a usarla como proxy para diseñar redes más eficientes y para investigar la conexión con generalización.

- Baja gradient confusion → SGD converge más rápido; alta → más lento
- Redes más **anchas** tienden a **reducir** Gradient Confusion
- Redes más **profundas** tienden a **aumentarlo**
- Mitigantes para redes profundas: inicializaciones específicas, batch norm, skip connections

## Medición y pipeline

**Métrica concreta.** La gradient confusion $\zeta$ se define como una cota inferior sobre los productos internos por pares entre gradientes per-sample (o per-mini-batch): para todo $i \neq j$,

$$\langle \nabla f_i(w), \nabla f_j(w) \rangle \geq -\eta.$$

Empíricamente, los autores estiman la cantidad como el **mínimo de la similitud coseno** entre pares de gradientes de mini-batches evaluados en el $w$ actual; es decir, trabajan con la versión normalizada

$$\cos(\nabla f_i, \nabla f_j) = \frac{\langle \nabla f_i, \nabla f_j \rangle}{\|\nabla f_i\| \cdot \|\nabla f_j\|},$$

y reportan también la densidad completa de similitudes pares.

**Entradas.** Se requieren $M$ gradientes calculados en el mismo $w$ sobre $M$ mini-batches muestreados i.i.d. del conjunto de entrenamiento. El paper utiliza **$M = 100$ pares de mini-batches** al final de cada epoch; como alternativa razonable puede tomarse $M \in [50, 200]$ mini-batches y construir los $\binom{M}{2}$ pares.

**Cuándo computar.** Al final de cada epoch (o cada $K$ pasos en entrenamientos largos), manteniendo $w$ fijo durante la medición. El coste por pares es $O(M^2)$.

**Coste.** $M$ forward+backward sin actualización + $M^2$ productos internos (baratos una vez cacheados los gradientes aplanados). Memoria: $M$ vectores de dimensión $|w|$; con redes grandes conviene almacenarlos en CPU o usar subsampling de parámetros.

**Integración pipeline.** Pseudocódigo:

```python
def gradient_confusion(model, loss_fn, loader, M=100):
    model.eval()  # cuidado con BN/dropout
    grads = []
    for _ in range(M):
        xb, yb = next(loader)
        model.zero_grad()
        loss_fn(model(xb), yb).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(g / g.norm())
    G = torch.stack(grads)
    sims = G @ G.T            # matriz M×M de cosenos
    pairs = sims[torch.triu_indices(M, M, offset=1).unbind()]
    return pairs.min().item(), pairs  # escalar + histograma
```

**Consideraciones.** $\zeta$ depende fuertemente de la **anchura y profundidad** (resultado central del paper: la sobreparametrización por anchura reduce la confusión; la profundidad la aumenta). Al medir, BatchNorm debe fijarse en modo evaluación para no contaminar gradientes con estadísticas de batch distintas. En **Adam**, los productos internos sobre $\nabla f$ no reflejan la dirección efectiva de actualización (modulada por momentos adaptativos): conviene reportar $\zeta$ sobre el gradiente crudo y, opcionalmente, sobre el paso preacondicionado.

**Logging.** Por checkpoint: el escalar $\zeta$ (mínimo) y, para señal más rica, un **histograma** de los $\binom{M}{2}$ cosenos pares (media, percentiles 1/5/50, fracción negativa). Esto permite correlacionar tanto el peor caso como la masa de la distribución con las métricas de eficiencia (epochs-to-threshold, AUC test loss, best test loss).

## Notes
