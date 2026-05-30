---
authors:
  - Rie Johnson
  - Tong Zhang
year: 2013
status: read
relevance: low
last_review: 2026-05-07
url: https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf
tfg_role:
  - related-work
  - theory
tfg_note: "SVRG; reducción de varianza para acelerar SGD; related-work."
---

# Accelerating Stochastic Gradient Descent using Predictive Variance Reduction

## Summary

### Contextualización

El trabajo aborda el problema canónico de minimización empírica $P(w) = \frac{1}{n} \sum_{i=1}^{n} \psi_i(w)$, donde cada $\psi_i$ representa la pérdida del ejemplo $i$ (por ejemplo, mínimos cuadrados o regresión logística regularizada). El descenso por gradiente clásico (GD) requiere evaluar las $n$ derivadas en cada paso, lo que resulta prohibitivo a gran escala, mientras que el descenso por gradiente estocástico (SGD) reduce el coste por iteración a $1/n$ del de GD al muestrear un único índice $i_t$ y aplicar $w^{(t)} = w^{(t-1)} - \eta_t \nabla \psi_{i_t}(w^{(t-1)})$. El problema es que $\nabla \psi_{i_t}$ solo coincide con $\nabla P$ en esperanza, de modo que el estimador introduce una varianza intrínseca que obliga a usar tasas de aprendizaje decrecientes $\eta_t = O(1/t)$ y degrada la convergencia a una tasa sublineal $O(1/t)$, frente a la lineal $O((1 - \gamma/L)^t)$ que GD alcanza bajo suavidad y convexidad fuerte (con $L$ la constante de Lipschitz y $\gamma$ el módulo de convexidad fuerte). Este compromiso entre coste por iteración y velocidad de convergencia es el problema central que motiva el artículo. Trabajos previos como SAG (Le Roux et al., 2012) o SDCA (Shalev-Shwartz y Zhang, 2012) ya habían conseguido convergencia lineal mediante reducción de varianza, pero ambos requerían almacenar todos los gradientes individuales o las variables duales, lo que los hace inviables en problemas grandes o no convexos como el entrenamiento de redes neuronales.

### Aportación

Los autores proponen **Stochastic Variance Reduced Gradient (SVRG)**, un método que reduce explícitamente la varianza del estimador estocástico del gradiente sin requerir almacenamiento de gradientes individuales. La contribución es triple: aporta un algoritmo aplicable a problemas complejos donde SAG y SDCA no son viables (predicción estructurada, redes neuronales); ofrece una demostración de convergencia lineal sensiblemente más simple e intuitiva que las de SAG y SDCA; y proporciona una explicación basada en reducción de varianza que se extiende como heurística al caso no convexo, sirviendo como acelerador local de SGD en redes neuronales.

### Metodología

El núcleo del método consiste en mantener un *snapshot* $\tilde{w}$ del vector de parámetros que se actualiza cada $m$ iteraciones de SGD. En cada *snapshot* se calcula el gradiente completo $\tilde{\mu} = \nabla P(\tilde{w}) = \frac{1}{n} \sum_i \nabla \psi_i(\tilde{w})$ con un solo barrido sobre los datos, y la regla de actualización interior reemplaza el gradiente estocástico estándar por el estimador con *control variate*:

$$w^{(t)} = w^{(t-1)} - \eta_t \left( \nabla \psi_{i_t}(w^{(t-1)}) - \nabla \psi_{i_t}(\tilde{w}) + \tilde{\mu} \right).$$

Como $\mathbb{E}_{i_t}[\nabla \psi_{i_t}(\tilde{w}) - \tilde{\mu}] = 0$, el estimador permanece insesgado, esto es, $\mathbb{E}[w^{(t)} \mid w^{(t-1)}] = w^{(t-1)} - \eta_t \nabla P(w^{(t-1)})$. La clave está en que cuando $w^{(t-1)} \to w_*$ y $\tilde{w} \to w_*$ los términos $\nabla \psi_{i_t}(w^{(t-1)}) - \nabla \psi_{i_t}(\tilde{w}) + \tilde{\mu}$ tienden a cero, eliminando la varianza asintóticamente y permitiendo emplear una tasa de aprendizaje $\eta$ constante. El pseudocódigo (Figura 1 del paper) define dos variantes: la **opción I**, donde $\tilde{w}_s = w_m$ (último iterado del bucle interior), que es la natural en la práctica; y la **opción II**, donde $\tilde{w}_s = w_t$ con $t$ elegido uniformemente al azar en $\{0, \ldots, m-1\}$, que es la que sustenta el análisis teórico.

El análisis de convergencia se formaliza en el Teorema 1: bajo suavidad de cada $\psi_i$ (constante $L$), convexidad de cada $\psi_i$ y convexidad fuerte de $P$ (módulo $\gamma > 0$), si $m$ es suficientemente grande para que $\alpha = \frac{1}{\gamma \eta (1 - 2L\eta) m} + \frac{2L\eta}{1 - 2L\eta} < 1$, entonces $\mathbb{E}\,P(\tilde{w}_s) - P(w_*) \le \alpha^s \, [P(\tilde{w}_0) - P(w_*)]$, es decir, **convergencia geométrica en esperanza**. La demostración acota $\mathbb{E}\|v_t\|_2^2 \le 4L[P(w_{t-1}) - P(w_*) + P(\tilde{w}) - P(w_*)]$ apoyándose en la desigualdad $\|a+b\|_2^2 \le 2\|a\|_2^2 + 2\|b\|_2^2$, en la propiedad $\mathbb{E}\|\xi - \mathbb{E}\xi\|_2^2 \le \mathbb{E}\|\xi\|_2^2$ y en la cota $n^{-1} \sum_i \|\nabla \psi_i(w) - \nabla \psi_i(w_*)\|_2^2 \le 2L[P(w) - P(w_*)]$. En el régimen indicativo $L/\gamma = n$, eligiendo $\eta = 0{,}1/L$ y $m = O(n)$ se obtiene $\alpha = 1/2$, lo que requiere procesar $n \ln(1/\epsilon)$ ejemplos para precisión $\epsilon$, frente a los $n^2 \ln(1/\epsilon)$ de GD estándar. Para problemas suaves no fuertemente convexos se recupera $O(1/T)$, mejorando el $O(1/\sqrt{T})$ de SGD, y en el caso no convexo se sugiere inicializar $\tilde{w}_0$ cerca de un mínimo local mediante SGD y aplicar SVRG para acelerar la convergencia local. El paper también ofrece una interpretación unificadora al reescribir SDCA como un método de reducción de varianza para SGD: cuando $(w, \alpha) \to (w_*, \alpha_*)$ se cumple $\frac{1}{n} \sum_i (\nabla \phi_i(w) + \lambda n \alpha_i)^2 \to 0$, mostrando el paralelismo conceptual con SVRG.

### Datasets y modelos

Los experimentos cubren tanto el régimen convexo como el no convexo. En el régimen convexo se entrena regresión logística multiclase L2-regularizada sobre **MNIST** ($\lambda = 10^{-4}$) y regresión logística L2-regularizada sobre **rcv1.binary** y **covtype.binary** (LIBSVM), **protein** (KDD Cup) y **CIFAR-10** ($\lambda = 10^{-3}$ para CIFAR-10 y $10^{-5}$ para los demás). En el régimen no convexo se entrenan redes neuronales con una capa oculta totalmente conectada de 100 nodos, activación sigmoide, salida softmax de 10 clases, regularización L2 y mini-batches de tamaño 10 sobre MNIST ($\lambda = 10^{-4}$) y CIFAR-10 ($\lambda = 10^{-3}$). El intervalo de actualización se fija en $m = 2n$ (convexo) y $m = 5n$ (no convexo), y SVRG se inicializa con una iteración de SGD en el caso convexo y diez en el no convexo.

### Métricas

Los autores reportan la *training loss* $P(w)$ frente al número de gradientes calculados normalizado por $n$, la **suboptimalidad** $P(w) - P(w_*)$ (con $w_*$ estimado mediante GD ejecutado durante mucho tiempo), la **varianza del incremento de pesos** $-\eta(\nabla \psi_i(w) - \nabla \psi_i(\tilde{w}) + \tilde{\mu})$ comparada con la de SGD y SDCA, y el **test error rate** sobre los conjuntos de prueba (con divisiones aleatorias 50/50 para protein y covtype, que carecen de etiquetas de test). El eje $x$ se mide siempre en número de cómputos de gradiente dividido por $n$ para garantizar comparaciones equitativas.

### Conclusiones

Los resultados experimentales confirman las predicciones teóricas. En el caso convexo SVRG es competitivo con SDCA, con curvas casi solapadas, y converge claramente más rápido que el SGD mejor ajustado, ya sea con decaimiento exponencial $\eta(t) = \eta_0 a^{\lfloor t/n \rfloor}$ o con $t$-inverso $\eta(t) = \eta_0(1 + b\lfloor t/n \rfloor)^{-1}$. La varianza de SVRG y SDCA decae exponencialmente, mientras que la de SGD con tasa fija permanece alta y la del mejor SGD solo cae por la decadencia forzada de $\eta(t)$. En el caso no convexo (redes neuronales) SVRG reduce la varianza y supera al SGD mejor ajustado, mostrando que la idea de reducción de varianza explícita es aplicable más allá del régimen convexo, con la ventaja decisiva sobre SAG y SDCA de no requerir almacenamiento de gradientes. Los autores señalan como línea futura la validación con redes más grandes y profundas, donde el coste de entrenamiento es crítico.

## Medición y pipeline

**Rol del paper.** SVRG no aporta una métrica de diagnóstico para registrar, sino un algoritmo de optimización. Conforme a `metrics.md`, Johnson y Zhang (2013) no proponen ninguna métrica nueva: introducen el método SVRG y analizan la **varianza del incremento de pesos** $-\eta(\nabla \psi_i(w) - \nabla \psi_i(\tilde{w}) + \tilde{\mu})$ como objeto a reducir explícitamente para acelerar la convergencia. Su relevancia para el TFG es indirecta pero fundamental: si la reducción explícita de la varianza del estimador del gradiente acelera la convergencia (resultado del Teorema 1, confirmado empíricamente), entonces medir la varianza del gradiente durante el entrenamiento constituye un proxy plausible de eficiencia de optimización. Esta lectura justifica teóricamente la elección de NGV (Faghri et al.) y GNS (McCandlish et al.) como métricas centrales del eje varianza del pipeline. En lo que sigue describimos el procedimiento que se seguiría para reproducir el diagnóstico del paper, es decir, la **ratio de varianzas** $\mathrm{Var}[g_{\text{SVRG}}] / \mathrm{Var}[g_{\text{SGD}}]$, que es la cantidad que el propio Johnson y Zhang miden y que Defazio y Bottou (2019) retoman como sanity check.

**Entradas.** Se necesitan dos magnitudes por paso. Por un lado, el gradiente per-sample $\nabla \psi_i(w)$ evaluado en los pesos actuales $w$, es decir, el mismo objeto que computaría SGD. Por otro lado, el gradiente per-sample $\nabla \psi_i(\tilde{w})$ del mismo índice $i$ pero evaluado en los pesos del *snapshot* $\tilde{w}$. La media completa $\tilde{\mu} = \frac{1}{n}\sum_i \nabla \psi_i(\tilde{w})$ se computa una vez por *snapshot* recorriendo el dataset entero con los pesos congelados $\tilde{w}$. Un *control variate* típico sería entonces el vector $c_i = \nabla \psi_i(\tilde{w}) - \tilde{\mu}$, de media cero, que se resta del gradiente per-sample actual para obtener el estimador $g_{\text{SVRG}, i} = \nabla \psi_i(w) - c_i$.

**Granularidad temporal.** Hay dos escalas naturales. La interior, dentro del bucle de SGD, es por paso o cada $K$ pasos: en cada paso se mide la varianza empírica del minibatch tanto bajo el estimador SVRG como bajo el SGD vanilla, y se reporta la ratio puntual. La exterior es por *snapshot*, es decir, cada $m$ pasos (en el paper $m = 2n$ en el régimen convexo y $m = 5n$ en redes neuronales), cuando se refresca $\tilde{w}$ y se recalcula $\tilde{\mu}$. La señal de interés es la **ratio $\mathrm{Var}[g_{\text{SVRG}}] / \mathrm{Var}[g_{\text{SGD}}]$ promediada por época**: en el caso convexo decae exponencialmente hacia cero (lo que justifica $\eta$ constante), mientras que en deep learning real esa ratio puede estancarse cerca de 1 o incluso superarla cuando $w$ se aleja de $\tilde{w}$.

**Granularidad estructural.** Suele bastar con un escalar global por época, es decir, la varianza calculada concatenando todos los parámetros en un único vector. Para diagnóstico es útil reportar también la versión por capa, computando la traza de la covarianza del gradiente restringido a los parámetros de cada capa: en redes profundas las primeras capas tienden a presentar varianza más alta y peor descorrelación entre $w$ y $\tilde{w}$, y el escalar global puede ocultarlo.

**Coste.** El *snapshot* cuesta una pasada completa sobre los datos, es decir, $O(n)$ forward-backward por época con los pesos $\tilde{w}$ congelados. El *control variate* añade un forward y un backward extra por paso de SGD, ya que cada $i_t$ requiere evaluar $\nabla \psi_{i_t}$ en $w$ y en $\tilde{w}$. En total, el coste por época es aproximadamente el triple del de SGD vanilla (una pasada para $\tilde{\mu}$ más dos evaluaciones por paso). La memoria adicional es modesta: basta con guardar $\tilde{w}$ (mismo tamaño que $w$) y $\tilde{\mu}$ (idem).

**Trucos.** El *snapshot diferido* consiste en no congelar realmente una copia $\tilde{w}$ en GPU, sino guardar los pesos en CPU y mover por capas durante el cómputo de $\nabla \psi_i(\tilde{w})$, abaratando memoria a costa de latencia. El *streaming variance* (algoritmo de Welford) permite mantener una estimación incremental de $\mathbb{E}\|g\|^2$ y de $\|\mathbb{E}[g]\|^2$ sin almacenar los gradientes per-sample, lo que reduce huella de memoria de $O(N \cdot |W|)$ a $O(|W|)$ cuando solo interesa la traza. Si solo se quiere la ratio de varianzas a nivel global, se pueden submuestrear $N$ ejemplos del *probe set* (típicamente $N \approx 256$–$1024$) en lugar de recorrer el dataset entero.

**Claves de log.** Por época se registran, como mínimo, las siguientes claves: `svrg/var_ratio` para la ratio $\mathrm{Var}[g_{\text{SVRG}}] / \mathrm{Var}[g_{\text{SGD}}]$ a nivel global; `svrg/grad_var_snapshot` para la traza de la covarianza $\mathrm{tr}(\mathrm{Cov}(g_{\text{SVRG}}))$ medida justo después de cada *snapshot*, que en el régimen convexo debe decaer exponencialmente; y opcionalmente `svrg/var_ratio/layer_k` para la versión por capa. Una varianza típica del estimador SVRG es del orden de $10^{-2}$–$10^{-1}$ veces la de SGD inmediatamente tras el *snapshot* en problemas convexos, y se acerca a 1 (o supera 1) en redes profundas conforme $w$ se aleja de $\tilde{w}$.

**Gotchas.** El problema principal es la **descorrelación del control variate**: el estimador SVRG solo reduce varianza si $\nabla \psi_i(w)$ y $\nabla \psi_i(\tilde{w})$ están fuertemente correlacionados, lo que exige que $w$ y $\tilde{w}$ permanezcan próximos. En redes profundas con tasas de aprendizaje altas, $\|w - \tilde{w}\|$ crece rápidamente entre *snapshots* y el control variate pierde correlación, llegando incluso a aumentar la varianza (Defazio y Bottou, 2019, observan ratios de hasta 2 en ResNet, y señalan que para que SVRG sea rentable la ratio debe estar por debajo de $1/3$). Esto se traduce en que **SVRG falla en deep learning real**: la *strong growth condition* (varianza del estimador $\to 0$ en $w_*$) que el método presupone se cumple en MNIST y problemas tipo regresión logística, pero no en CIFAR-10 o ImageNet, donde la varianza del gradiente *crece* durante el entrenamiento. Otros riesgos son el desfase entre el batch actual y $\tilde{\mu}$ por batch normalization (los estadísticos de BN difieren entre $w$ y $\tilde{w}$), y la inestabilidad numérica de la varianza empírica con pocos ejemplos.

**Pseudocódigo PyTorch.**

```python
def svrg_var_ratio_step(model, snapshot_model, mu_tilde, batch, criterion):
    # batch: lista de (x_i, y_i) per-sample
    g_sgd_list, g_svrg_list = [], []
    for x_i, y_i in batch:
        # gradiente per-sample en w
        model.zero_grad()
        loss_w = criterion(model(x_i.unsqueeze(0)), y_i.unsqueeze(0))
        g_w = flatten(torch.autograd.grad(loss_w, model.parameters()))

        # gradiente per-sample en snapshot ~w
        snapshot_model.zero_grad()
        loss_wt = criterion(snapshot_model(x_i.unsqueeze(0)), y_i.unsqueeze(0))
        g_wt = flatten(torch.autograd.grad(loss_wt, snapshot_model.parameters()))

        # estimadores
        g_sgd_list.append(g_w)
        g_svrg_list.append(g_w - g_wt + mu_tilde)   # control variate

    G_sgd = torch.stack(g_sgd_list)
    G_svrg = torch.stack(g_svrg_list)
    var_sgd  = (G_sgd  - G_sgd.mean(0)).pow(2).sum(1).mean()
    var_svrg = (G_svrg - G_svrg.mean(0)).pow(2).sum(1).mean()
    return (var_svrg / var_sgd).item()

# por snapshot, cada m pasos:
def refresh_snapshot(model, snapshot_model, train_loader, criterion):
    snapshot_model.load_state_dict(model.state_dict())   # ~w <- w
    mu_tilde = torch.zeros_like(flatten(model.parameters()))
    n = 0
    for x, y in train_loader:
        snapshot_model.zero_grad()
        loss = criterion(snapshot_model(x), y)
        g = flatten(torch.autograd.grad(loss, snapshot_model.parameters()))
        mu_tilde += g * x.size(0); n += x.size(0)
    return mu_tilde / n
```

**Decisión.** No se loggea ninguna métrica directa de este paper en el TFG; SVRG se cita como justificación teórica del eje varianza (junto a NGV y GNS) y, opcionalmente, como sanity check experimental fuera del scope cerrado.

## Notes

Es una referencia fundacional en optimización, publicada en NeurIPS, ampliamente conocida en la comunidad. La idea en una frase es que **SVRG reduce explícitamente la varianza del estimador del gradiente para acelerar la convergencia en problemas suaves**. En el TFG sirve para establecer que la comunidad reconoce que la varianza tiene contenido informativo y para marcar que aquí no se propone un nuevo optimizador sino una *señal de diagnóstico* derivada de las mismas magnitudes que SVRG manipula.

### Uso en el TFG

El rol del paper es de **soporte teórico del eje varianza, no de métrica del registry**. SVRG no aporta ninguna de las diez métricas (ni el baseline TSE-EMA); se cita en *related work* como la pieza que legitima el eje varianza: si reducir explícitamente la varianza del estimador del gradiente acelera la convergencia, entonces medir esa varianza durante el entrenamiento es un proxy plausible de dificultad y de eficiencia de optimización.

La idea operativa y la fórmula del *update* son el corazón conceptual a citar. SVRG mantiene un *snapshot* $\tilde{w}$ y su gradiente full-batch periódico $\tilde{\mu} = \nabla P(\tilde{w}) = \frac{1}{n}\sum_i \nabla\psi_i(\tilde{w})$ como *control variate*, y aplica el paso interior $w \leftarrow w - \eta\,(\nabla\psi_i(w) - \nabla\psi_i(\tilde{w}) + \tilde{\mu})$. El estimador es insesgado porque $\mathbb{E}_i[\nabla\psi_i(\tilde{w}) - \tilde{\mu}] = 0$, y su varianza tiende a cero cuando $w, \tilde{w} \to w_*$, lo que permite tasa $\eta$ constante. El resultado clave que se cita es que, bajo suavidad y convexidad fuerte, esa reducción de varianza convierte la convergencia sublineal $O(1/t)$ de SGD en lineal/geométrica $\alpha^s$ (Teorema 1). Es el argumento causal "menos varianza del gradiente $\Rightarrow$ optimización más rápida" que el TFG invoca para justificar `normalized_variance`, `gns_simple` y `gsnr`.

La **conexión crítica** está en la *strong growth condition*: la aceleración de SVRG presupone que la varianza del estimador decae cerca del óptimo, hipótesis que se cumple en problemas tipo MNIST (los experimentos convexos y no convexos del paper la soportan) pero falla en deep learning real (CIFAR-10, ImageNet), donde la varianza del gradiente crece durante el entrenamiento. Esto motiva la decisión central del TFG: medir la varianza empíricamente en lugar de asumirla decreciente. El enlace con Faghri et al. cierra el círculo, ya que [[A Study of Gradient Variance in Deep Learning]] usa precisamente SVRG como baseline para mostrar este fallo: la reducción de varianza por *control variate* deja de ayudar cuando la varianza no decae, evidenciando empíricamente la ruptura de la *strong growth condition* en redes profundas. Por eso SVRG entra como justificación teórica más contraste, no como método a ejecutar (a lo sumo como *sanity check* opcional fuera de scope).

![[Pasted image 20260526130825.png]]

En síntesis, este método reduce la varianza durante la optimización con garantía teórica y empírica (imagen superior). Cuando optimizamos, actualizamos $w$ moviéndonos en la dirección de un gradiente, pero SGD estima ese gradiente con un único ejemplo elegido al azar (o varios): acierta en media (es insesgado) pero resulta ruidoso, y ese ruido no desaparece en el óptimo porque los gradientes individuales no se anulan aunque su promedio sí lo haga. Por eso SGD necesita ir bajando el paso $\eta$ hasta cero para converger, lo que lo ralentiza. SVRG cambia ese estimador por uno corregido,

$$\nabla\psi_i(w) - \nabla\psi_i(\tilde{w}) + \tilde{\mu},$$

donde $\nabla\psi_i(w)$ es el gradiente actual que usaría SGD, $\nabla\psi_i(\tilde{w})$ es el gradiente del mismo ejemplo pero evaluado con los parámetros del *snapshot* $\tilde{w}$, y $\tilde{\mu}$ es la media del gradiente sobre todos los ejemplos de entrenamiento en $\tilde{w}$. Sigue siendo insesgado, pero su varianza tiende a cero a medida que nos acercamos al óptimo porque el término correctivo cancela el ruido propio de cada ejemplo. Eso permite usar un paso grande y constante, recupera la convergencia lineal y, a diferencia de SAG o SDCA, no exige almacenar todos los gradientes, de modo que también sirve para redes neuronales. El precio es una pasada completa por los datos de vez en cuando para recomputar $\tilde{\mu}$.

## Discrepancias detectadas

La versión previa de este archivo contenía una sección "Integración pipeline (estimador two-batch)" con pseudocódigo del estimador de McCandlish et al. (2018) para $\mathrm{tr}(\mathrm{Cov}(\nabla L_i))$. Ese estimador no aparece en el paper de Johnson y Zhang y pertenece al pipeline de GNS, por lo que se ha sustituido en esta revisión por el pseudocódigo de la ratio $\mathrm{Var}[g_{\text{SVRG}}] / \mathrm{Var}[g_{\text{SGD}}]$, que es la magnitud que el propio SVRG mide en sus experimentos (Figura 2 del paper) y que Defazio y Bottou (2019) retoman como diagnóstico. La referencia al estimador two-batch queda implícita en el enlace con [[An Empirical Model of Large-Batch Training]] dentro de "Papers relacionados".

## Papers relacionados

- [[A Study of Gradient Variance in Deep Learning]] — usa SVRG como baseline y muestra empíricamente que su hipótesis de varianza decreciente (strong growth) falla en deep learning real; fuente directa de `normalized_variance`.
- [[An Empirical Model of Large-Batch Training]] — formaliza la varianza del gradiente como *gradient noise scale* $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma)/\|G\|^2$; mismo eje varianza, medida en vez de reducida (fuente de `gns_simple`).
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — GSNR $= \tilde{g}^2/\rho^2$ es el SNR por parámetro, el inverso conceptual de la varianza que SVRG reduce; fuente de `gsnr`.

## Otros papers interesantes a revisar

- **A Stochastic Gradient Method with an Exponential Convergence Rate for Finite Training Sets (SAG)** (Le Roux, Schmidt & Bach, 2012) — el predecesor directo citado por SVRG: logra convergencia lineal promediando gradientes, pero almacena los $n$ gradientes individuales (inviable a gran escala). Contextualiza por qué SVRG no necesita ese almacenamiento. arXiv:1202.6258
- **Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization (SDCA)** (Shalev-Shwartz & Zhang, 2013) — el otro método de varianza-reducida con el que SVRG se compara y se reinterpreta como reducción de varianza; JMLR 14:567-599. arXiv:1209.1873
- **SpiderBoost / SARAH-type variance reduction for non-convex optimization** (Nguyen et al., SARAH, 2017) — extiende la reducción de varianza tipo SVRG al caso no convexo con garantías más fuertes; útil si se quiere matizar por qué la teoría de SVRG es solo heurística en redes profundas. arXiv:1703.00102
- **On the Ineffectiveness of Variance Reduced Optimization for Deep Learning** (Defazio & Bottou, NeurIPS 2019) — argumenta empíricamente que la reducción de varianza estilo SVRG no acelera el deep learning moderno; refuerza desde otro ángulo la motivación del TFG de medir (no reducir) la varianza. arXiv:1812.04529
