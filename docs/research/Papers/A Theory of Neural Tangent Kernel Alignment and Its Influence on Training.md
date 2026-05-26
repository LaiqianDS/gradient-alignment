---
authors:
  - Haozhe Shan
  - Blake Bordelon
year: 2021
status: to-read
relevance: medium
url: https://arxiv.org/abs/2105.14301
last_review: 2026-05-07
tfg_role:
  - theory
  - related-work
---

# A Theory of Neural Tangent Kernel Alignment and Its Influence on Training

## Summary

### Contextualización

Las dinámicas de entrenamiento y las propiedades de generalización de las redes neuronales pueden caracterizarse en el espacio de funciones a través del **Neural Tangent Kernel (NTK)**. En el régimen de anchura infinita y tasa de aprendizaje pequeña, con cierta parametrización, una red neuronal se comporta como un modelo lineal en sus parámetros: el entrenamiento es equivalente a *kernel gradient descent* (KGD) con un NTK estático. Este límite permite caracterizaciones analíticas precisas del entrenamiento y la generalización, pero, al ser estacionario, **no captura el aprendizaje de características** (feature learning), fenómeno responsable en gran medida del éxito empírico de las redes profundas finitas. En redes prácticas, sin embargo, el NTK evoluciona durante el entrenamiento, y trabajos empíricos recientes (Baratin et al. 2021; Fort et al. 2020; Geiger et al. 2020; Atanasov et al. 2021; Paccolat et al. 2021) han identificado un fenómeno ubicuo: el NTK se **alinea** progresivamente con la función objetivo, un efecto que los autores denominan *kernel alignment*. La motivación del trabajo es proporcionar una explicación teórica y mecanicista de cómo emerge esa alineación durante el entrenamiento y por qué beneficia al aprendizaje.

### Aportación

Las contribuciones principales son tres. (1) Demuestran que la **alineación del kernel acelera el entrenamiento**: estudian un modelo de juguete de Optimal Feature Evolution (OFE) en el que el NTK se permite evolucionar para acelerar la convergencia de la pérdida y muestran que la alineación con la función objetivo emerge naturalmente como solución óptima. La fuerza de la aceleración está controlada por un único parámetro, la *feature learning rate* $\gamma$, que determina tanto la velocidad de entrenamiento como la alineación final. (2) Aportan una **teoría analítica** de cómo emerge la alineación en redes lineales profundas (Sec. 5) y un tratamiento aproximado en redes ReLU de dos capas (Sec. 6); la teoría predice que la alineación es **mayor en redes más profundas**, predicción validada numéricamente en redes ReLU. (3) Identifican y formalizan el fenómeno de **kernel specialization** en redes con múltiples salidas (clasificación multiclase): cada subkernel diagonal asociado a un cabezal de salida se alinea preferentemente con su propia función objetivo, no con las del resto. Definen una *Kernel Specialization Matrix* (KSM) para detectarlo y muestran que requiere activaciones no lineales y estructura de datos específica.

### Metodología

**Formulación del NTK.** Para una red $f(\boldsymbol{x}, \boldsymbol{\theta})$ entrenada por descenso por gradiente sobre una pérdida $\mathcal{L} = \sum_\mu \ell(f(\boldsymbol{x}^\mu, \boldsymbol{\theta}), y^\mu)$ con $P$ ejemplos, la dinámica de los parámetros es

$$\frac{d\boldsymbol{\theta}}{dt} = -\eta \sum_{\mu=1}^{P} \frac{\partial f(\boldsymbol{x}^\mu, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \frac{\partial \ell(f(\boldsymbol{x}^\mu, \boldsymbol{\theta}), y^\mu)}{\partial f(\boldsymbol{x}^\mu, \boldsymbol{\theta})}.$$

La dinámica inducida sobre la salida de la red es

$$\frac{df(\boldsymbol{x}, \boldsymbol{\theta})}{dt} = -\eta \sum_\nu K(\boldsymbol{x}, \boldsymbol{x}^\nu; \boldsymbol{\theta}) \frac{\partial \ell}{\partial f(\boldsymbol{x}^\nu, \boldsymbol{\theta})},$$

donde el **NTK** se define como $K(\boldsymbol{x}, \boldsymbol{x}'; \boldsymbol{\theta}) = \frac{\partial f(\boldsymbol{x}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot \frac{\partial f(\boldsymbol{x}', \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$. Sobre el conjunto de entrenamiento queda completamente descrito por la matriz de Gram $\boldsymbol{K}(\boldsymbol{\theta}) \in \mathbb{R}^{P \times P}$.

**Definición formal de alignment.** Adoptan la métrica de alineación de kernels de Cortes et al. (2012):

$$A(t) = \frac{\langle \boldsymbol{y}\boldsymbol{y}^\top, \boldsymbol{K}(\boldsymbol{\theta}) \rangle_F}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F \, \|\boldsymbol{y}\boldsymbol{y}^\top\|_F} = \frac{\boldsymbol{y}^\top \boldsymbol{K}(\boldsymbol{\theta}) \boldsymbol{y}}{\|\boldsymbol{K}(\boldsymbol{\theta})\|_F \, \|\boldsymbol{y}\|^2}.$$

Esta cantidad mide cuánto del NTK se proyecta sobre la dirección del target $\boldsymbol{y}\boldsymbol{y}^\top$ en el espacio de matrices.

**Modelo OFE.** Consideran KGD con kernel evolutivo $\boldsymbol{K}(t) = \boldsymbol{\Psi}(t)^\top \boldsymbol{\Psi}(t)$, con $\boldsymbol{\Psi} \in \mathbb{R}^{P \times Q}$ definiendo el feature map. Con pérdida MSE, la dinámica es $\mathcal{L}_{t+1}(\boldsymbol{\Psi}) = \|(\boldsymbol{I} - \eta \boldsymbol{\Psi}_t^\top \boldsymbol{\Psi}_t) \boldsymbol{\Delta}_t\|^2$ y el feature map se actualiza por descenso sobre $\mathcal{L}_{t+1}$ con paso $\gamma$. En el límite continuo se obtiene un kernel final $\boldsymbol{K}_\infty = \gamma \boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0$, que es exactamente alineación con la tarea: aumentar $\gamma$ acelera la convergencia y eleva la alineación final.

**Redes lineales profundas.** Para una red lineal de $L$ capas $f(\boldsymbol{x}) = \boldsymbol{w}^{L+1\top} \boldsymbol{W}^L \cdots \boldsymbol{W}^1 \boldsymbol{x}$ con inicialización pequeña y leyes de conservación $\boldsymbol{W}^\ell \boldsymbol{W}^{\ell\top} \approx \boldsymbol{W}^{\ell+1\top} \boldsymbol{W}^{\ell+1}$, derivan el kernel $K(\boldsymbol{x}, \boldsymbol{x}') = u(t)^{2L-2} \boldsymbol{x}^\top [L \boldsymbol{r}_1(t) \boldsymbol{r}_1(t)^\top + \boldsymbol{I}] \boldsymbol{x}'$ y el resultado central

$$\boldsymbol{K}_\infty \propto L \boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0,$$

cuya escala con $L$ predice **mayor alineación en redes más profundas**. Asimismo, prueban que las redes lineales **no pueden desarrollar kernels especializados**.

**Redes ReLU de dos capas.** Para $f(\boldsymbol{x}^\mu) = \sum_i V_i \,\text{ReLU}(\boldsymbol{w}_i \cdot \boldsymbol{x}^\mu)$, descomponen el NTK en $\boldsymbol{K} = \boldsymbol{K}_V + \boldsymbol{K}_W$ y, bajo dos heurísticas (signo de preactivaciones estático y $\boldsymbol{y}^\top \boldsymbol{f}(t) > 0$), derivan dinámicas en la forma bilineal $B(\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{K} \boldsymbol{z}$ que muestran anisotropía dirigida por $\boldsymbol{y}$ en $\boldsymbol{K}_V$ e isotropía en $\boldsymbol{K}_W$. Para clasificación multiclase con mezcla de gaussianas, prueban que en el límite $\sigma^2 \to 0$, el subkernel $K^{c,c}_{\boldsymbol{W}}$ crece aproximadamente al doble de velocidad en la dirección de $\boldsymbol{y}_c$ que en las de $\boldsymbol{y}_{d \neq c}$, formalizando la especialización.

### Datasets y modelos usados

- **MNIST** (subconjunto): MLP de dos capas con $N = 500$ unidades ReLU para clasificación binaria odd-even (Fig. 1) y MLPs ReLU de $N=500$ para clasificación de 10 clases (Fig. 3A, Fig. 7C-D).
- **CIFAR-10** (subconjunto de 100 imágenes, dos clases): Wide ResNet de Zagoruyko y Komodakis (2017) con factor de ensanchamiento $k=3$ y $b=2$ bloques (Fig. 2); CNN para clasificación de dos clases (Fig. 3B).
- **Datos sintéticos**: targets lineales con $y^\mu = \beta \boldsymbol{x}^\mu$ y vectores i.i.d. gaussianos para redes lineales (Fig. 4); mezcla de 10 gaussianas con $\sigma^2 = 0.01$ y centros mutuamente perpendiculares para validar especialización en ReLU (Fig. 7); clasificación binaria aleatoria con $\boldsymbol{x}^\mu \sim \mathcal{N}(\boldsymbol{0}, N^{-1/2}\boldsymbol{I})$ y $y^\mu \in \{\pm 1\}$ para ReLU de dos capas (Fig. 6).
- MLPs ReLU de profundidad variable (2, 3, 4, 5 capas ocultas) sobre la tarea MNIST para validar la predicción de alineación creciente con $L$ (Fig. 5).

### Métricas

- $A(\boldsymbol{K}, \boldsymbol{y}\boldsymbol{y}^\top)$: alineación NTK-target (Cortes et al. 2012).
- Pérdida de entrenamiento $\mathcal{L}$ y norma de Frobenius $\|\boldsymbol{K}\|_F$.
- Comparación de la red NN frente a KGD con kernel inicial (KGD) y a KGD con kernel inicial reescalado (aKGD) para aislar efectos de estructura vs. magnitud.
- Forma bilineal $B(\boldsymbol{z}) = \boldsymbol{z}^\top \boldsymbol{K} \boldsymbol{z}$ y traza $\text{Tr}(\boldsymbol{K})$ para medir anisotropía.
- **Kernel Specialization Matrix** $\text{KSM}(c,d) = A(\boldsymbol{K}^{c,c}, \boldsymbol{y}_d \boldsymbol{y}_d^\top) / [C^{-1} \sum_{d'} A(\boldsymbol{K}^{d',d'}, \boldsymbol{y}^d \boldsymbol{y}^{d\top})]$.
- Cosine similarity entre pesos de red y de profesor, y entre la autofunción dominante del NTK y el target.

### Conclusiones

El trabajo demuestra empírica y teóricamente que la alineación del kernel **acelera el aprendizaje** mediante una anisotropía estructural del NTK que no puede explicarse por un simple aumento de escala. El análisis del modelo OFE muestra que la alineación con el target es la solución óptima cuando se permite que el feature map evolucione. La derivación analítica para redes lineales profundas predice (y los experimentos en ReLU confirman) que la alineación es **más fuerte en redes profundas**. Por primera vez, identifican y formalizan la *kernel specialization* en redes con múltiples salidas: cada subkernel diagonal se alinea con su target específico, fenómeno que requiere no linealidades y estructura de datos por clase y que las redes lineales no pueden producir. Como limitaciones reconocen el uso de pérdida MSE, supuestos heurísticos en las derivaciones y la restricción al kernel evaluado sobre el conjunto de entrenamiento (no sobre el de test), dejando como dirección futura la conexión rigurosa entre alineación y generalización.

## Medición y pipeline

**Métrica concreta.** La cantidad de interés es el *kernel-target alignment* (KTA) sobre el NTK empírico, en la forma canónica de Cortes et al. (2012):

$$\text{KTA} = \frac{\langle \boldsymbol{K}, \boldsymbol{y}\boldsymbol{y}^\top \rangle_F}{\|\boldsymbol{K}\|_F \cdot \|\boldsymbol{y}\boldsymbol{y}^\top\|_F},$$

donde $\boldsymbol{K}_{ij} = \langle \nabla_\theta f(\boldsymbol{x}_i), \nabla_\theta f(\boldsymbol{x}_j) \rangle$ es el NTK empírico evaluado sobre un subconjunto de $N$ ejemplos y $\boldsymbol{y}$ son las etiquetas (codificadas *one-hot* en clasificación, o como vector escalar de margen/logit en su versión reducida).

**Entradas.** Los gradientes per-ejemplo de la salida respecto a los parámetros, $\nabla_\theta f(\boldsymbol{x}_i)$, sobre $N$ puntos. En clasificación multiclase con $C$ clases hay dos opciones: (i) computar $\nabla_\theta f_c(\boldsymbol{x}_i)$ por clase y construir subkernels diagonales $\boldsymbol{K}^{c,c}$ (consistente con la *Kernel Specialization Matrix* del paper), o (ii) reducir la salida a un escalar (margen o logit de la clase correcta) y trabajar con un único NTK agregado.

**Cuándo computar.** Una vez por época, sobre un subconjunto fijo de $N$ ejemplos (p. ej. $N = 256$) para garantizar comparabilidad temporal. La matriz resultante es $N \times N$.

**Coste.** Requiere $N$ *forward passes* más $N$ *jacobian-vector products* (eficientes con `torch.func.jacrev` o `vmap`); la memoria es $\mathcal{O}(N^2)$ para $\boldsymbol{K}$ y el coste dominante es el cálculo de los gradientes per-ejemplo cuando $N$ crece.

**Integración en el pipeline.** Esquema de pseudocódigo:

```python
# Cada época, sobre un subset fijo de N ejemplos
X, y = subset(loader, N)                  # y en one-hot: (N, C)
def f_single(params, x): return model(params, x)
J = vmap(jacrev(f_single), (None, 0))(params, X)   # (N, C, |theta|)
J_flat = J.reshape(N * C, -1)             # o reducir a escalar antes
K = J_flat @ J_flat.T                     # (N*C, N*C) o (N, N)
Y = y.reshape(-1, 1)                      # vector target apilado
kta = (Y.T @ K @ Y) / (norm(K, 'fro') * (Y.T @ Y))
log({"kta": float(kta), "epoch": epoch})
```

**Consideraciones.** En redes finitas el NTK no es estático (no estamos en *lazy regime*), por lo que la ventana temprana de entrenamiento es la relevante para la hipótesis predictiva del TFG. El coste cuadrático en $N$ obliga a submuestrear; alternativas razonables son: KTA por capa (usando solo $\nabla_{\theta^\ell} f$), aproximación por traza $\text{Tr}(\boldsymbol{K})$ o estimación estocástica vía *trace-Hutchinson*. La teoría del paper sostiene que mayor alineación predice mejor generalización, lo que justifica usar KTA como señal temprana.

**Logging.** Por época se registra un escalar KTA; opcionalmente los $k$ autovalores principales de $\boldsymbol{K}$ (para inspeccionar el espectro) y, en multiclase, la matriz KSM completa.

## Notes

### Uso en el TFG

- **Métrica que origina:** `ntk_alignment` (familia alineación). Kernel-Target Alignment de Cortes et al. (2012) sobre el NTK empírico $K_{ij} = \nabla_\theta f(x_i)\cdot\nabla_\theta f(x_j)$: $\text{KTA} = \dfrac{\langle K, Y\rangle_F}{\|K\|_F\,\|Y\|_F}$, con reducción escalar $f = z_{y_i}$ (logit de la clase correcta) e $Y$ matriz one-vs-rest $\pm 1$ por pares ($Y_{ij}=+1$ si $y_i=y_j$, $-1$ si no).
- **Cómo se mide:** per-sample **Jacobian de $f$** (no de la loss $\ell$) vía `vmap`+`jacrev`; probe $N=256$; default **last-layer-only** (teoría aditiva por capa). Por calcular $\nabla f$ y no $\nabla L$, va en un **sweep SEPARADO** del de `stiffness` / `m_coherence` / `gsnr`.
- **Señal:** ↑ mejor — KTA alto acelera el aprendizaje (Thm. 4.1: la alineación con el target es la solución óptima del feature map evolutivo).
- **Pitfalls / decisiones:** $K=JJ^\top$ es PSD por construcción (random projection puede romperlo); usar $Y$ one-vs-rest $\pm 1$, no one-hot (rango trivial $[1/C,1]$); forma **sin centrar** como en Shan & Bordelon (Cortes centra). KSM multiclase off-by-default (coste $C\times$; opcional `every_k_epochs=5` en CIFAR-10).
- **Doble rol:** (1) métrica del registry; (2) **soporte teórico** para el related work — fundamenta por qué la alineación de gradientes correlaciona con eficiencia, y conecta stiffness/cosine batch-wise (aproximaciones empíricas locales) con la formalización global a nivel kernel. La predicción $\boldsymbol{K}_\infty \propto L\,\boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0$ (mayor alineación con profundidad) y la *kernel specialization* multiclase (requiere no linealidad) son argumentos para discutir por qué redes profundas finitas superan al régimen lazy.

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
