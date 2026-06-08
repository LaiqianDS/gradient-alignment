---
authors:
  - Haozhe Shan
  - Blake Bordelon
year: 2021
status: to-read
relevance: medium
url: https://arxiv.org/abs/2105.14301
tfg_role:
  - theory
  - related-work
tfg_note: "Origen de `ntk_alignment` (kernel-target alignment) y soporte teórico del eje alineación: el NTK se alinea con la tarea durante el entrenamiento y eso acelera el aprendizaje, tanto más cuanto más profunda es la red."
---

# A Theory of Neural Tangent Kernel Alignment and Its Influence on Training

## Summary

### Contextualización

El trabajo aborda una tensión fundamental en la teoría moderna del deep learning. En el régimen de anchura infinita y tasa de aprendizaje pequeña, con una parametrización adecuada, una red neuronal se comporta como un modelo lineal en sus parámetros: entrenarla equivale a hacer *kernel gradient descent* (KGD) sobre el **Neural Tangent Kernel (NTK)**, que permanece estático durante todo el entrenamiento. Este límite, formalizado por Jacot et al. (2018), permite caracterizaciones analíticas precisas tanto de la dinámica como de la generalización, pero al ser estacionario no captura el **aprendizaje de características** (feature learning), justamente el fenómeno que explica buena parte del éxito empírico de las redes profundas finitas. La motivación del paper parte de una observación recurrente en redes prácticas: el NTK no es estático, sino que evoluciona durante el entrenamiento y, lo que es más relevante, evoluciona en una dirección concreta. Diversos estudios empíricos (Baratin et al. 2021; Fort et al. 2020; Geiger et al. 2020; Atanasov et al. 2021; Paccolat et al. 2021) habían documentado que el NTK se **alinea** progresivamente con la función objetivo, un efecto que Shan y Bordelon denominan *kernel alignment*. Su objetivo es proporcionar una explicación teórica y mecanicista de cómo emerge esa alineación durante el entrenamiento y por qué beneficia al aprendizaje.

### Aportación

La contribución se articula en tres frentes. En primer lugar, los autores demuestran que la **alineación del kernel acelera el entrenamiento**: estudian un modelo de juguete llamado Optimal Feature Evolution (OFE) en el que se permite que el NTK evolucione para acelerar la convergencia de la pérdida, y muestran que la alineación con la función objetivo emerge naturalmente como solución óptima de ese problema. La intensidad de la aceleración queda controlada por un único parámetro, la *feature learning rate* $\gamma$, que determina simultáneamente la velocidad del entrenamiento y la alineación final. En segundo lugar, aportan una **teoría analítica** que explica cómo emerge la alineación en redes lineales profundas (Sec. 5) y un tratamiento aproximado del caso ReLU de dos capas (Sec. 6); la teoría predice que la alineación es **mayor en redes más profundas**, predicción que validan numéricamente en redes ReLU. En tercer lugar, identifican y formalizan el fenómeno de **kernel specialization** en redes con múltiples salidas (clasificación multiclase): cada subkernel diagonal asociado a un cabezal de salida se alinea preferentemente con su propia función objetivo, no con las del resto. Para detectarlo definen una *Kernel Specialization Matrix* (KSM) y demuestran que el fenómeno requiere activaciones no lineales y una estructura de datos específica.

### Metodología

El punto de partida es la formulación del NTK. Para una red $f(\boldsymbol{x}, \boldsymbol{\theta})$ entrenada por descenso por gradiente sobre una pérdida $\mathcal{L} = \sum_\mu \ell(f(\boldsymbol{x}^\mu, \boldsymbol{\theta}), y^\mu)$ con $N$ ejemplos, la dinámica de los parámetros es

$$\frac{d\boldsymbol{\theta}}{dt} = -\eta \sum_{\mu=1}^{N} \frac{\partial f(\boldsymbol{x}^\mu, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \frac{\partial \ell(f(\boldsymbol{x}^\mu, \boldsymbol{\theta}), y^\mu)}{\partial f(\boldsymbol{x}^\mu, \boldsymbol{\theta})},$$

y la dinámica inducida sobre la salida se escribe como

$$\frac{df(\boldsymbol{x}, \boldsymbol{\theta})}{dt} = -\eta \sum_\nu K(\boldsymbol{x}, \boldsymbol{x}^\nu; \boldsymbol{\theta}) \frac{\partial \ell}{\partial f(\boldsymbol{x}^\nu, \boldsymbol{\theta})},$$

donde el **NTK** se define por $K(\boldsymbol{x}, \boldsymbol{x}'; \boldsymbol{\theta}) = \frac{\partial f(\boldsymbol{x}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot \frac{\partial f(\boldsymbol{x}', \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$. Sobre el conjunto de entrenamiento el NTK queda completamente descrito por la matriz de Gram $\boldsymbol{K}(\boldsymbol{\theta}) \in \mathbb{R}^{N \times N}$ (con $N$ el número de ejemplos). Para medir cuánto se parece este operador a la tarea, los autores adoptan la métrica de alineación de Cortes et al. (2012):

$$A(t) = \frac{\langle \boldsymbol{y}\boldsymbol{y}^\top, \boldsymbol{K}(\boldsymbol{\theta}) \rangle_F}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F \, \|\boldsymbol{y}\boldsymbol{y}^\top\|_F} = \frac{\boldsymbol{y}^\top \boldsymbol{K}(\boldsymbol{\theta}) \boldsymbol{y}}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F \, \|\boldsymbol{y}\|^2},$$

una cantidad acotada en $[0,1]$ cuando el target es PSD (como aquí, $\boldsymbol{y}\boldsymbol{y}^\top$) que cuantifica qué fracción del NTK se proyecta sobre la dirección del target $\boldsymbol{y}\boldsymbol{y}^\top$ en el espacio de matrices. Sobre esta métrica construyen sus tres análisis. El **modelo OFE** considera KGD con kernel evolutivo $\boldsymbol{K}(t) = \boldsymbol{\Psi}(t)^\top \boldsymbol{\Psi}(t)$, donde $\boldsymbol{\Psi} \in \mathbb{R}^{P \times Q}$ define el feature map. Con pérdida MSE, la dinámica resulta $\mathcal{L}_{t+1}(\boldsymbol{\Psi}) = \|(\boldsymbol{I} - \eta \boldsymbol{\Psi}_t^\top \boldsymbol{\Psi}_t) \boldsymbol{\Delta}_t\|^2$ y el feature map se actualiza por descenso sobre $\mathcal{L}_{t+1}$ con paso $\gamma$. En el límite continuo se obtiene un kernel final $\boldsymbol{K}_\infty = \gamma \boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0$, que es exactamente alineación con la tarea: aumentar $\gamma$ acelera la convergencia y eleva la alineación final.

Para **redes lineales profundas** $f(\boldsymbol{x}) = \boldsymbol{w}^{L+1\top} \boldsymbol{W}^L \cdots \boldsymbol{W}^1 \boldsymbol{x}$ con inicialización pequeña y leyes de conservación $\boldsymbol{W}^\ell \boldsymbol{W}^{\ell\top} \approx \boldsymbol{W}^{\ell+1\top} \boldsymbol{W}^{\ell+1}$, derivan el kernel $K(\boldsymbol{x}, \boldsymbol{x}') = u(t)^{2L-2} \boldsymbol{x}^\top [L \boldsymbol{r}_1(t) \boldsymbol{r}_1(t)^\top + \boldsymbol{I}] \boldsymbol{x}'$ y, en el límite, el resultado central

$$\boldsymbol{K}_\infty \propto L \boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0,$$

cuya escala explícita con $L$ predice **mayor alineación en redes más profundas**. Como corolario, prueban que las redes lineales **no pueden desarrollar kernels especializados**. Para **ReLU de dos capas**, $f(\boldsymbol{x}^\mu) = \sum_i V_i \,\text{ReLU}(\boldsymbol{w}_i \cdot \boldsymbol{x}^\mu)$, descomponen el NTK como $\boldsymbol{K} = \boldsymbol{K}_V + \boldsymbol{K}_W$ y, bajo dos heurísticas (signo de preactivaciones estático y $\boldsymbol{y}^\top \boldsymbol{f}(t) > 0$), derivan dinámicas en la forma bilineal $B(\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{K} \boldsymbol{z}$ que muestran anisotropía dirigida por $\boldsymbol{y}$ en $\boldsymbol{K}_V$ e isotropía en $\boldsymbol{K}_W$. Para clasificación multiclase con mezcla de gaussianas demuestran que, en el límite $\sigma^2 \to 0$, el subkernel $K^{c,c}_{\boldsymbol{W}}$ crece aproximadamente al doble de velocidad en la dirección de $\boldsymbol{y}_c$ que en las de $\boldsymbol{y}_{d \neq c}$, formalizando así la especialización.

### Datasets y modelos usados

Los experimentos abarcan tanto datos reales como sintéticos. Sobre **MNIST** (subconjunto) entrenan un MLP de dos capas con $N = 500$ unidades ReLU para clasificación binaria odd-even (Fig. 1) y MLPs ReLU de $N=500$ para clasificación de 10 clases (Fig. 3A, Fig. 7C-D). Sobre **CIFAR-10** (subconjunto de 100 imágenes, dos clases) entrenan un Wide ResNet de Zagoruyko y Komodakis (2017) con factor de ensanchamiento $k=3$ y $b=2$ bloques (Fig. 2), y una CNN para clasificación de dos clases (Fig. 3B). Los **datos sintéticos** cubren tres familias: targets lineales $y^\mu = \beta \boldsymbol{x}^\mu$ con vectores i.i.d. gaussianos para redes lineales (Fig. 4); una mezcla de 10 gaussianas con $\sigma^2 = 0.01$ y centros mutuamente perpendiculares para validar la especialización en ReLU (Fig. 7); y clasificación binaria aleatoria con $\boldsymbol{x}^\mu \sim \mathcal{N}(\boldsymbol{0}, N^{-1/2}\boldsymbol{I})$ y $y^\mu \in \{\pm 1\}$ para ReLU de dos capas (Fig. 6). Para validar la predicción teórica de que la alineación crece con $L$ entrenan MLPs ReLU de profundidad variable (2, 3, 4, 5 capas ocultas) sobre la tarea MNIST (Fig. 5).

### Métricas

La cantidad central seguida es la alineación NTK-target $A(\boldsymbol{K}, \boldsymbol{y}\boldsymbol{y}^\top)$ de Cortes et al. (2012). Junto a ella reportan la pérdida de entrenamiento $\mathcal{L}$ y la norma de Frobenius $\|\boldsymbol{K}\|_F$ como medidas de dinámica y magnitud global. Para aislar efectos de estructura frente a magnitud comparan la red NN con KGD usando el kernel inicial (KGD) y con KGD usando el kernel inicial reescalado (aKGD). La forma bilineal $B(\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{K} \boldsymbol{z}$ y la traza $\text{Tr}(\boldsymbol{K})$ sirven para detectar anisotropía direccional. La **Kernel Specialization Matrix** $\text{KSM}(c,d) = A(\boldsymbol{K}^{c,c}, \boldsymbol{y}_d \boldsymbol{y}_d^\top) / [C^{-1} \sum_{d'} A(\boldsymbol{K}^{d',d'}, \boldsymbol{y}_{d'} \boldsymbol{y}_{d'}^\top)]$ (forma exacta a verificar contra el PDF) se interpreta así: valores en torno a 1 indican ausencia de especialización (cada subkernel se alinea por igual con todos los targets), entradas diagonales notablemente mayores que 1 con extra-diagonales menores que 1 indican que cada cabezal se ha especializado en su propia clase. En la práctica del paper, la diagonal supera 1.5 mientras que las extra-diagonales caen por debajo de 0.9 en redes ReLU multiclase. Como medida complementaria reportan la similitud coseno entre pesos de red y de profesor, y entre la autofunción dominante del NTK y el target.

### Conclusiones

El trabajo demuestra empírica y teóricamente que la alineación del kernel **acelera el aprendizaje** mediante una anisotropía estructural del NTK que no puede explicarse por un simple aumento de escala. El modelo OFE muestra que la alineación con el target es la solución óptima cuando se permite que el feature map evolucione. La derivación analítica para redes lineales profundas predice (y los experimentos en ReLU confirman) que la alineación es **más fuerte en redes profundas**. Por primera vez, los autores identifican y formalizan la *kernel specialization* en redes con múltiples salidas: cada subkernel diagonal se alinea con su target específico, fenómeno que requiere no linealidades y estructura de datos por clase y que las redes lineales no pueden producir. Como limitaciones reconocen el uso de pérdida MSE, las heurísticas sobre las que se apoyan algunas derivaciones y la restricción al kernel evaluado sobre el conjunto de entrenamiento (no sobre el de test), dejando como dirección futura la conexión rigurosa entre alineación y generalización.

## Medición y pipeline

**Métrica concreta.** La cantidad de interés es el *kernel-target alignment* (KTA) sobre el NTK empírico, en la forma canónica de Cortes et al. (2012):

$$\text{KTA} = \frac{\langle \boldsymbol{K}, \boldsymbol{y}\boldsymbol{y}^\top \rangle_F}{\|\boldsymbol{K}\|_F \cdot \|\boldsymbol{y}\boldsymbol{y}^\top\|_F} = \frac{\boldsymbol{y}^\top \boldsymbol{K} \boldsymbol{y}}{\|\boldsymbol{K}\|_F \, \|\boldsymbol{y}\|^2},$$

donde $\boldsymbol{K} = \boldsymbol{J}\boldsymbol{J}^\top$ es el NTK empírico construido a partir del jacobiano per-ejemplo $\boldsymbol{J} \in \mathbb{R}^{N \times P}$ (con $N$ tamaño del probe y $P$ número de parámetros efectivos), e $\boldsymbol{y}$ son las etiquetas codificadas. Por construcción $\boldsymbol{K}$ es semidefinida positiva; el rango $\text{KTA} \in [0, 1]$ solo se garantiza con un target PSD (one-hot o $\boldsymbol{y}\boldsymbol{y}^\top$), mientras que la codificación $\pm 1$ por pares adoptada aquí (ver Entradas) tiene entradas negativas y produce el rango $[-1, 1]$. En la inicialización aleatoria el valor típico es bajo (orden $1/\sqrt{N}$ en redes amplias) y, conforme avanza el entrenamiento, sube hasta valores en el rango $0.3$–$0.7$ en redes ReLU profundas sobre tareas reales; valores cercanos a $0.9$ corresponden a kernels casi perfectamente alineados con la tarea (régimen analítico en redes lineales muy profundas o tras OFE saturado).

**Entradas.** El ingrediente básico es el jacobiano per-ejemplo de la salida respecto a los parámetros, $\boldsymbol{J}_i = \nabla_\theta f(\boldsymbol{x}_i)$, sobre $N$ puntos. En clasificación multiclase con $C$ clases caben dos estrategias. La primera, fiel al paper, consiste en computar $\nabla_\theta f_c(\boldsymbol{x}_i)$ por clase y construir los subkernels diagonales $\boldsymbol{K}^{c,c}$, que alimentan la *Kernel Specialization Matrix*. La segunda, más barata, reduce la salida a un escalar (logit de la clase correcta o margen) y trabaja con un único NTK agregado de tamaño $N \times N$. Para la métrica del registry del TFG se toma esta segunda vía, con $\boldsymbol{Y}$ codificada one-vs-rest $\pm 1$ por pares ($Y_{ij} = +1$ si $y_i = y_j$, $-1$ en caso contrario) en lugar de one-hot. La codificación $\pm 1$ da un rango $[-1, 1]$ con un punto de alineación nula verdadero (KTA $= 0$), mientras que one-hot comprime el rango a un suelo positivo (≈$1/C$ con clases balanceadas).

**Granularidad temporal.** Se mide una vez por época sobre un probe fijo y disjunto, lo que garantiza comparabilidad entre snapshots. En la *ventana temprana* (primeras 5–10 épocas), donde la teoría OFE predice el grueso de la evolución del kernel, se densifica a cada $K$ pasos para muestrear con mayor resolución. La variante multiclase (KSM) se calcula por defecto cada 5 épocas en CIFAR-10 por su sobrecoste $C\times$, no en cada época.

**Coste y memoria.** Con $N = 512$ y $P = 11.7\text{M}$ parámetros (ResNet-18, FP32), el jacobiano completo ocuparía $N \cdot P \cdot 4 \approx 24$ GB, claramente prohibitivo. Tres mitigaciones son razonables. Primero, restringir el jacobiano a la **última capa** (default en el TFG), con $P_{\text{eff}} \approx 5\text{k}$, lo que reduce la huella a $\approx 10$ MB y se justifica por la teoría aditiva por capa de Shan y Bordelon. Segundo, **proyecciones aleatorias** $\boldsymbol{J} \boldsymbol{R}$ con $\boldsymbol{R} \in \mathbb{R}^{P \times r}$ ($r \ll P$): preservan productos internos en expectativa pero rompen estrictamente la PSD-idad de $\boldsymbol{K}$, así que solo se usan si se renuncia a interpretaciones espectrales. Tercero, **estimar la traza por Hutchinson** $\text{Tr}(\boldsymbol{K}) \approx \mathbb{E}_z[z^\top \boldsymbol{K} z]$ con $z$ Rademacher cuando solo se necesita $\|\boldsymbol{K}\|_F^2 = \text{Tr}(\boldsymbol{K}^2)$. La forma $\boldsymbol{y}^\top \boldsymbol{K} \boldsymbol{y}$ es trivial vía $\|\boldsymbol{J}^\top \boldsymbol{y}\|^2$ y no requiere materializar $\boldsymbol{K}$.

**Pseudocódigo PyTorch.** El esquema de referencia, con reducción escalar de la salida (logit de la clase correcta) y last-layer-only:

```python
import torch
from torch.func import vmap, jacrev, functional_call

# probe fijo, estratificado por clase
X, y = subset(loader, N=512)            # X: (N, ...), y: (N,) int64
params = {k: v.detach() for k, v in model.named_parameters()
          if k.startswith("fc.")}       # last-layer-only
# resto de parámetros congelados (todo menos la última capa)
params_full_frozen = {k: v.detach() for k, v in model.named_parameters()
                      if not k.startswith("fc.")}
buffers = dict(model.named_buffers())

def f_scalar(p, x, yi):
    logits = functional_call(model, {**params_full_frozen, **p}, x.unsqueeze(0))
    return logits[0, yi]                # logit de la clase correcta

# jacobiano per-ejemplo respecto a parámetros last-layer
J = vmap(jacrev(f_scalar), (None, 0, 0))(params, X, y)
J_flat = torch.cat([j.reshape(N, -1) for j in J.values()], dim=1)  # (N, P_eff)

# Y one-vs-rest +-1 por pares
Y = (y.unsqueeze(0) == y.unsqueeze(1)).float() * 2 - 1            # (N, N)
Y_vec = Y.flatten()

# KTA sin materializar K si N pequeño: aquí K si entra en memoria
K = J_flat @ J_flat.T                                              # (N, N)
num = (K * Y).sum()
den = torch.linalg.norm(K, ord="fro") * torch.linalg.norm(Y, ord="fro")
kta = (num / den).item()

log({"ntk/alignment": kta,
     "ntk/frobenius": torch.linalg.norm(K, ord="fro").item(),
     "epoch": epoch})
```

Para la variante KSM se repite el bloque por clase, computando $\boldsymbol{K}^{c,c}$ con los gradientes del logit $c$ y agregando las alineaciones cruzadas con los $\boldsymbol{y}_d \boldsymbol{y}_d^\top$.

**Claves de log.** Se registran tres escalares por época, `ntk/alignment` (KTA global), `ntk/frobenius` (magnitud del kernel) y, cuando procede, `ntk/ksm` (matriz $C \times C$ aplanada o sus estadísticos diagonal/extra-diagonal). Opcionalmente los $k$ autovalores principales de $\boldsymbol{K}$ para inspeccionar el espectro.

**Gotchas.** Cuatro puntos críticos. (1) **Escala vs. estructura**: el KTA divide por $\|\boldsymbol{K}\|_F$, así que un crecimiento aislado de la magnitud no infla la métrica; comparar KTA frente a aKGD (kernel inicial reescalado) aísla el efecto puramente direccional. (2) **Normalización Frobenius**: usar la forma sin centrar (como Shan y Bordelon, no como Cortes original, que sí centra) para mantener consistencia con el paper de referencia; el centrado cambia el rango y rompe la comparabilidad con sus figuras. (3) **One-hot vs. one-vs-rest**: codificar $\boldsymbol{Y}$ como matriz $\pm 1$ por pares y no como one-hot da un rango $[-1, 1]$ con un punto de alineación nula verdadero, mientras que one-hot comprime el rango a un suelo positivo (≈$1/C$ con clases balanceadas). (4) **PSD y proyecciones**: $\boldsymbol{K} = \boldsymbol{J}\boldsymbol{J}^\top$ es PSD por construcción; una proyección aleatoria de $\boldsymbol{J}$ preserva esto, pero submuestrear filas/columnas de $\boldsymbol{K}$ directamente no, así que cualquier reducción debe hacerse sobre $\boldsymbol{J}$.

**Consideraciones sobre la dinámica.** En redes finitas el NTK no es estático (no estamos en *lazy regime*), de modo que la ventana temprana del entrenamiento es la relevante para la hipótesis predictiva del TFG. Por calcularse a partir de $\nabla f$ y no de $\nabla \mathcal{L}$, el sweep de KTA es **independiente** del de `stiffness`, `m_coherence` y `gsnr` y se ejecuta en una corrida separada.

## Notes

### Uso en el TFG

- **Métrica que origina:** `ntk_alignment` (familia alineación). Kernel-Target Alignment de Cortes et al. (2012) sobre el NTK empírico $K_{ij} = \nabla_\theta f(x_i)\cdot\nabla_\theta f(x_j)$, con reducción escalar $f = z_{y_i}$ (logit de la clase correcta) e $Y$ matriz one-vs-rest $\pm 1$ por pares.
- **Cómo se mide:** per-sample Jacobian de $f$ (no de $\ell$) vía `vmap`+`jacrev`; probe $N=512$; default last-layer-only. Sweep separado del de gradientes de pérdida.
- **Señal:** $\uparrow$ mejor — KTA alto acelera el aprendizaje (Thm. 4.1: la alineación con el target es la solución óptima del feature map evolutivo).
- **Pitfalls / decisiones:** $K=JJ^\top$ PSD por construcción; $Y$ one-vs-rest $\pm 1$ (no one-hot), que da rango $[-1, 1]$ con cero de alineación verdadero frente al suelo positivo (≈$1/C$ con clases balanceadas) del one-hot; forma sin centrar como en Shan & Bordelon (Cortes original sí centra). KSM multiclase off-by-default por coste $C\times$.
- **Doble rol:** (1) métrica del registry; (2) soporte teórico para el related work — fundamenta por qué la alineación de gradientes correlaciona con eficiencia, y conecta stiffness y m-coherence (aproximaciones empíricas locales) con la formalización global a nivel kernel. La predicción $\boldsymbol{K}_\infty \propto L\,\boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0$ y la *kernel specialization* multiclase argumentan por qué redes profundas finitas superan al régimen lazy.

## Papers relacionados

- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — familia alineación; cosine/sign-stiffness son la aproximación batch-wise local de lo que el NTK alignment formaliza globalmente.
- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — m-coherence mide alineación de gradientes per-sample; comparte el eje "alineación → eficiencia/generalización" a nivel de $\nabla L$.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — la CGH (alineación de gradientes per-sample explica generalización) es el correlato empírico de la teoría kernel-target.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — mismo recorte last-layer y proxy de alineación train-time; comparable en diseño de medición.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — gradient confusion (coseno mínimo entre gradientes) es la versión worst-case de la (des)alineación que el NTK captura como operador.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — vincula propiedades del gradiente (SNR por parámetro) con generalización; complementa la vía alineación con la vía varianza.

## Otros papers interesantes a revisar

- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** (Jacot, Gabriel & Hongler, 2018) — introduce el NTK y el régimen de anchura infinita; base teórica del operador $K$ que este paper hace evolucionar. arXiv:1806.07572 (NeurIPS 2018).
- **Algorithms for Learning Kernels Based on Centered Alignment** (Cortes, Mohri & Rostamizadeh, 2012) — origen de la métrica Kernel-Target Alignment $A(K,yy^\top)$ que adopta el paper; discute la versión centrada vs. sin centrar. JMLR 13:795–828.
- **Implicit Regularization via Neural Feature Alignment** (Baratin, George, Laurent et al., 2021) — evidencia empírica de que el NTK se alinea con el target durante el entrenamiento (fenómeno que motiva la teoría de Shan & Bordelon). arXiv:2008.00938 (AISTATS 2021).
- **Deep Learning versus Kernel Learning: an Empirical Study of Loss Landscape Geometry and the Time Evolution of the Neural Tangent Kernel** (Fort, Dziugaite, Paul et al., 2020) — caracteriza la evolución temprana del NTK y el cambio rápido de su geometría; respalda usar la ventana temprana como señal. arXiv:2010.15110 (NeurIPS 2020).

## Cited By
