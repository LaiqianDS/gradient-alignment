# Datasets por Paper

Este documento consolida los datasets reportados en cada uno de los 15 papers del TFG. Para cada paper se distinguen datasets reales, datasets sintéticos y benchmarks especializados (NAS), incluyendo una breve descripción de su uso.

---

## A Study of Gradient Variance in Deep Learning (Faghri et al., 2020)

- **MNIST** — clasificación de dígitos con MLP fully-connected $784 \to 1024 \to 1024 \to 10$, ReLU.
- **CIFAR-10** — clasificación de imágenes con ResNet8 (sin batch normalization); incluye experimento adicional con label corruption del 10%.
- **CIFAR-100** — clasificación con ResNet32.
- **ImageNet** — clasificación de imágenes con ResNet18.
- **Random Features (RF) models, sintético** — dataset $\{(x_i, y_i)\} \in \mathbb{R}^I \times \{\pm 1\}$ generado por un modelo teacher con activación ReLU para estudiar régimen sobreparametrizado ($h_s/N$ variable, $N \in [0.1, 10]$).

---

## A Theory of Neural Tangent Kernel Alignment and Its Influence on Training (Shan & Bordelon, 2021)

- **MNIST (subconjunto)** — clasificación binaria odd-even y clasificación de 10 clases con MLPs ReLU de $N=500$ unidades.
- **CIFAR-10 (subconjunto de 100 imágenes, dos clases)** — clasificación binaria con Wide ResNet ($k=3$, $b=2$ bloques) y con CNN.
- **Datos sintéticos lineales** — targets $y^\mu = \beta x^\mu$ con vectores i.i.d. gaussianos, para validar la teoría en redes lineales profundas.
- **Mezcla de 10 gaussianas, sintético** — $\sigma^2 = 0.01$ y centros mutuamente perpendiculares; valida la kernel specialization en redes ReLU multiclase.
- **Clasificación binaria aleatoria, sintético** — $x^\mu \sim \mathcal{N}(0, N^{-1/2} I)$ y $y^\mu \in \{\pm 1\}$ para redes ReLU de dos capas.

---

## Accelerating Stochastic Gradient Descent using Predictive Variance Reduction (Johnson & Zhang, 2013)

- **MNIST** — regresión logística L2-regularizada multiclase ($\lambda = 10^{-4}$) en caso convexo; red neuronal con una capa oculta de 100 nodos sigmoid + softmax de 10 clases en caso no convexo.
- **CIFAR-10** — regresión logística L2-regularizada ($\lambda = 10^{-3}$) y red neuronal con una capa oculta de 100 nodos en caso no convexo.
- **rcv1.binary (LIBSVM)** — regresión logística L2-regularizada binaria ($\lambda = 10^{-5}$).
- **covtype.binary (LIBSVM)** — regresión logística L2-regularizada binaria con divisiones aleatorias 50/50 train/test ($\lambda = 10^{-5}$).
- **protein (KDD Cup)** — regresión logística L2-regularizada con divisiones 50/50 train/test ($\lambda = 10^{-5}$).

---

## Adam - A Method for Stochastic Optimization (Kingma & Ba, 2015)

- **MNIST** — regresión logística L2-regularizada multiclase con minibatches de 128; también MLP de dos capas ocultas de 1000 ReLU con dropout.
- **IMDB** — clasificación de polaridad con representaciones bag-of-words de 10000 palabras y dropout del 50%.
- **CIFAR-10** — CNN c64-c64-c128-1000 con tres etapas de filtros 5x5 y max-pooling, capa densa final de 1000 ReLU, dropout en entrada y capa densa, minibatches de 128.
- **MNIST (VAE)** — Variational Autoencoder con 500 unidades softplus ocultas y latente gaussiano de 50 dimensiones para cuantificar el efecto de la corrección de sesgo.

---

## An Empirical Model of Large-Batch Training (McCandlish et al., 2018)

- **MNIST** — clasificación con CNN simple optimizada por SGD.
- **SVHN** — clasificación con CNN simple usando SGD y Adam; también autoencoder y Variational Autoencoder (arquitectura InfoGAN) para generative modeling.
- **CIFAR-10** — clasificación con ResNet-32 + momentum.
- **ImageNet** — clasificación con ResNet-50 + momentum y schedule de learning rate decay.
- **Atari games (7 juegos)** — Alien, Beam Rider, Breakout, Pong, Qbert, Seaquest, Space Invaders, entrenados con A2C + RMSProp (reinforcement learning).
- **Dota (1v1 y 5v5)** — agentes OpenAI Dota entrenados con PPO asíncrono.
- **Billion Word Benchmark** — modelado de lenguaje con LSTM autoregresivo de tamaño 2048 (también 1024 y 512), codificación BPE de vocabulario 40k, optimizado con Adam.

---

## An overview of gradient descent optimization algorithms (Ruder, 2017)

Review/Survey — sin experimentos propios. El autor se apoya en visualizaciones cualitativas y resultados reportados en la literatura (e.g. función de Beale, saddle points, Figura 4 de Alec Radford) para comparar optimizadores.

---

## Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization (Chatterjee, 2019)

- **MNIST** — único dataset reportado. Experimentos con label noise sintético al 25%, 50%, 75% y 100% (etiquetas permutadas aleatoriamente). Arquitecturas: MLP fully-connected con una capa oculta de 2048 ReLUs y, para winsorized SGD, MLP con 3 capas ocultas de 256 ReLUs.

---

## Disparity Between Batches as a Signal for Early Stopping (Forouzesh & Thiran, 2021)

- **MRNet** — dataset de resonancias magnéticas de rodilla (detección de anormalidad, lesiones de ACL y meniscales) para el escenario de datos limitados.
- **MNIST** — subconjunto de 256 muestras con AlexNet (datos limitados); también con 50% de label noise para el escenario ruidoso.
- **CIFAR-10** — subconjunto de 1.28k muestras con VGG-13 (datos limitados).
- **CIFAR-100** — subconjunto de 1.28k muestras con ResNet-18 (datos limitados); también con 50% de label noise para el escenario ruidoso.

---

## Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks (Hölzl, 2025)

- **CIFAR-10** — clasificación con ConvNeXt-Femto y ViT/S-16 (training from scratch); splits de validación 90/10 y 99/1.
- **CIFAR-10-N** — variante con etiquetas humanas ruidosas en niveles 0%, 9%, 17% y 40%.
- **CIFAR-C** — perturbaciones de entrada para evaluación de robustez.
- **ImageNet-1k** — training from scratch con ConvNeXt-F y ViT/S-16; AdamW lr 0.001.
- **ImageNet-V2** — split adicional de ImageNet para evaluar generalización.
- **ImageNet-ReaL** — split adicional de ImageNet para evaluar generalización.
- **ImageNet-C** — perturbaciones de entrada para robustez.
- **ImageNet-21k** — preentrenamiento de ViT/B-16.
- **iNat18** — fine-tuning con ViT/B-16 preentrenado.
- **Places365** — fine-tuning con ViT/B-16 preentrenado.

---

## Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment (Chatterjee & Zielinski, 2020)

- **ImageNet** — experimento principal con tres variantes de label noise sintético (0%, 50%, 100% de etiquetas aleatorizadas; apéndice añade 25% y 75%). Arquitecturas: ResNet-18 (principal) e Inception-V3 (validación de generalidad). Muestra fija de $m = 40\,356$ ejemplos (duplicada a 80072 en apéndice).
- Discusión cualitativa de resultados ajenos sobre **MNIST**, **Fashion MNIST**, **CIFAR-10**, **CIFAR-100** y **MNLI** (citando Fort et al. 2019 y Sankararaman et al. 2019), sin experimentos propios en esos datasets.

---

## RMSProp - Divide the gradient by a running average of its recent magnitude (Tieleman & Hinton, 2012)

Slides/Lecture deck (Coursera, Lecture 6e) — sin experimentos propios sistemáticos. Las recomendaciones de uso son cualitativas: redes profundas, redes recurrentes, redes anchas y poco profundas, datasets grandes y redundantes. No se reportan curvas cuantitativas sobre datasets concretos.

---

## Speedy Performance Estimation for Neural Architecture Search (Ru et al., 2021)

- **NAS-Bench-201 (NB201)** — benchmark NAS con 6466 arquitecturas únicas evaluadas sobre **CIFAR-10**, **CIFAR-100** e **ImageNet-16-120**.
- **NAS-Bench-301** — benchmark NAS con 5000 arquitecturas del espacio DARTS sobre **CIFAR-10**.
- **ResNet/ResNeXt search space** — 50000 arquitecturas sobre **CIFAR-10**.
- **Randomly Wired Neural Networks (RWNN)** — 69×8 arquitecturas sobre **Flower102**.

(Datasets reales implicados: CIFAR-10, CIFAR-100, ImageNet-16-120, Flower102.)

---

## Stiffness: A New Perspective on Generalization in Neural Networks (Fort et al., 2019)

- **MNIST** — clasificación con MLP fully-connected ReLU $X \to 500 \to 300 \to 100 \to y$, CNN de 3 capas, y ResNet20v1 sin batch normalization.
- **Fashion MNIST** — clasificación con las mismas arquitecturas (MLP, CNN, ResNet20v1).
- **CIFAR-10** — clasificación con las mismas arquitecturas.
- **CIFAR-100** — clasificación con las mismas arquitecturas; usado adicionalmente para estudiar la estructura coarse-grain de super-clases en la matriz de class stiffness.
- **MNLI** — clasificación de inferencia textual con BERT fine-tuned, para validar el fenómeno fuera de visión.

---

## The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent (Sankararaman et al., 2020)

- **CIFAR-10** — clasificación con wide residual networks (WRNs) variando profundidad $\beta \in \{16, 22, 28, 34, 40, 52, 76, 100\}$ y anchura $\ell \in \{2, 3, 4, 5, 6\}$. Resultados principales del cuerpo.
- **CIFAR-100** — clasificación con WRNs, CNNs y MLPs (apéndice A).
- **MNIST** — clasificación con WRNs, CNNs y MLPs (apéndice A).

---

## Understanding Why Neural Networks Generalize Well Through GSNR of Parameters (Liu et al., 2020)

- **MNIST** — clasificación con CNN simple de dos bloques Conv-ReLU-MaxPooling + dos FC, variando el número de canales $p \in \{6, 8, 10, 12, 14, 16, 18, 20\}$ y el tamaño del set de entrenamiento $n \in \{1000, 2000, 4000, 6000, 8000, 10000, 15000\}$. Adicionalmente se ensaya label noise con $p_{random} \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$.
- **CIFAR-10** — clasificación con red Conv-BN-ReLU profunda (4 bloques Conv-BN-ReLU-Conv-BN-ReLU-MaxPooling + 3 FC); también con ResNet18 para comparar GSNR entre etiquetas reales y etiquetas aleatorizadas.
- **Toy dataset sintético** — generado por $y = x_0 x_1 + \epsilon$ con MLP de 2 entradas, $N$ neuronas ocultas y 1 salida; usado para aislar el efecto del feature learning sobre el GSNR.

---

## Tabla comparativa

| Paper | Datasets |
|-------|----------|
| A Study of Gradient Variance in Deep Learning | MNIST, CIFAR-10, CIFAR-100, ImageNet, Random Features (sintético) |
| A Theory of Neural Tangent Kernel Alignment | MNIST (subconjunto), CIFAR-10 (subconjunto), datos sintéticos lineales, mezcla de gaussianas, clasificación binaria aleatoria |
| Accelerating SGD using Predictive Variance Reduction | MNIST, CIFAR-10, rcv1.binary, covtype.binary, protein |
| Adam | MNIST, IMDB, CIFAR-10 |
| An Empirical Model of Large-Batch Training | MNIST, SVHN, CIFAR-10, ImageNet, 7 Atari games, Dota 1v1/5v5, Billion Word Benchmark |
| An overview of gradient descent optimization algorithms | Review — sin experimentos propios |
| Coherent Gradients | MNIST (+ label noise sintético 25/50/75/100%) |
| Disparity Between Batches as a Signal for Early Stopping | MRNet, MNIST, CIFAR-10, CIFAR-100 (+ label noise 50%) |
| Gradient-Weight Alignment | CIFAR-10, CIFAR-10-N, CIFAR-C, ImageNet-1k, ImageNet-V2, ImageNet-ReaL, ImageNet-C, ImageNet-21k, iNat18, Places365 |
| Making Coherence Out of Nothing At All | ImageNet (+ label noise 0/25/50/75/100%) |
| RMSProp | Slides — sin experimentos propios |
| Speedy Performance Estimation for NAS | NAS-Bench-201, NAS-Bench-301, CIFAR-10, CIFAR-100, ImageNet-16-120, Flower102 |
| Stiffness | MNIST, Fashion MNIST, CIFAR-10, CIFAR-100, MNLI |
| Gradient Confusion (Sankararaman et al.) | CIFAR-10, CIFAR-100, MNIST |
| GSNR of Parameters | MNIST, CIFAR-10, toy sintético (+ label noise 0-50%) |

---

## Frecuencia de datasets

### Datasets reales (visión)

- **MNIST** — 10 papers: Faghri et al. (Gradient Variance); Shan & Bordelon (NTK Alignment); Johnson & Zhang (SVRG); Kingma & Ba (Adam); McCandlish et al. (Large-Batch); Chatterjee (Coherent Gradients); Forouzesh & Thiran (Gradient Disparity); Fort et al. (Stiffness); Sankararaman et al. (Gradient Confusion); Liu et al. (GSNR).
- **CIFAR-10** — 12 papers: Faghri et al.; Shan & Bordelon; Johnson & Zhang; Kingma & Ba; McCandlish et al.; Forouzesh & Thiran; Hölzl (GWA); Ru et al. (Speedy NAS); Fort et al.; Sankararaman et al.; Liu et al.; (mencionado cualitativamente en Chatterjee & Zielinski).
- **CIFAR-100** — 6 papers: Faghri et al.; Forouzesh & Thiran; Ru et al.; Fort et al.; Sankararaman et al.; (mencionado cualitativamente en Chatterjee & Zielinski).
- **ImageNet** — 4 papers: Faghri et al.; McCandlish et al.; Chatterjee & Zielinski; Hölzl (variante ImageNet-1k).
- **Fashion MNIST** — 1 paper: Fort et al. (también mencionado cualitativamente en Chatterjee & Zielinski).
- **SVHN** — 1 paper: McCandlish et al.
- **CIFAR-10-N** — 1 paper: Hölzl.
- **CIFAR-C** — 1 paper: Hölzl.
- **ImageNet-V2** — 1 paper: Hölzl.
- **ImageNet-ReaL** — 1 paper: Hölzl.
- **ImageNet-C** — 1 paper: Hölzl.
- **ImageNet-21k** — 1 paper: Hölzl (preentrenamiento).
- **ImageNet-16-120** — 1 paper: Ru et al. (vía NAS-Bench-201).
- **iNat18** — 1 paper: Hölzl.
- **Places365** — 1 paper: Hölzl.
- **Flower102** — 1 paper: Ru et al. (vía RWNN).
- **MRNet** — 1 paper: Forouzesh & Thiran.

### Datasets reales (otros dominios)

- **IMDB** — 1 paper: Kingma & Ba.
- **MNLI** — 1 paper: Fort et al. (también discutido en Chatterjee & Zielinski).
- **Billion Word Benchmark** — 1 paper: McCandlish et al.
- **Atari (7 juegos)** — 1 paper: McCandlish et al.
- **Dota 1v1/5v5** — 1 paper: McCandlish et al.
- **rcv1.binary** — 1 paper: Johnson & Zhang.
- **covtype.binary** — 1 paper: Johnson & Zhang.
- **protein (KDD Cup)** — 1 paper: Johnson & Zhang.

### Benchmarks NAS

- **NAS-Bench-201** — 1 paper: Ru et al.
- **NAS-Bench-301** — 1 paper: Ru et al.

### Datasets sintéticos / toy problems

- **Random Features (RF) sintético** — 1 paper: Faghri et al.
- **Datos sintéticos lineales / mezcla de gaussianas / clasificación binaria aleatoria** — 1 paper: Shan & Bordelon.
- **Toy dataset $y = x_0 x_1 + \epsilon$** — 1 paper: Liu et al.
- **Label noise sintético sobre datasets reales** — 6 papers: Faghri et al. (10% en CIFAR-10); Chatterjee (25/50/75/100% en MNIST); Chatterjee & Zielinski (0/25/50/75/100% en ImageNet); Forouzesh & Thiran (50% en MNIST y CIFAR-100); Hölzl (0/9/17/40% en CIFAR-10-N, etiquetas humanas reales); Liu et al. (0-50% en MNIST y comparativa real vs. aleatorio en CIFAR-10/ResNet18).

### Papers sin experimentos propios

- **An overview of gradient descent optimization algorithms** (Ruder, 2017) — Review.
- **RMSProp - Divide the gradient by a running average of its recent magnitude** (Tieleman & Hinton, 2012) — Slides docentes.

---

## Decisiones de implementación (TFG)

Datasets efectivamente cargados en `src/load_data.py` y descargados a `data/`:

- **MNIST** — cubre 10 papers.
- **CIFAR-10** — cubre 12 papers.
- **CIFAR-100** — cubre 6 papers.
- **Fashion-MNIST** — cubre Fort et al.
- **Tiny-ImageNet** — sustituto de **ImageNet-1k** (4 papers: Faghri, McCandlish, Chatterjee & Zielinski, Hölzl).

### Sustitución de ImageNet-1k

El ILSVRC2012 completo (~150 GB, train 138 GB + val 6.3 GB) queda fuera de presupuesto de disco y tiempo de cómputo para el TFG. Se utiliza **Tiny-ImageNet** (200 clases, 64×64, ~240 MB) como proxy de bajo coste. Implicación: los experimentos de escala completa de Faghri et al. (ResNet18), McCandlish et al. (ResNet-50), Chatterjee & Zielinski (ResNet-18 / Inception-V3) y Hölzl (ViT/S-16, ConvNeXt-F) no son reproducibles fielmente; los resultados deberán interpretarse como evidencia cualitativa sobre subconjuntos de menor resolución y deberá mencionarse explícitamente en la memoria.

### Datasets no cargados

- **SVHN** (McCandlish) — disponible vía `torchvision.datasets.SVHN`. No cargado en esta iteración.
- **CIFAR-10-N, CIFAR-C, ImageNet-C, ImageNet-V2, ImageNet-ReaL, ImageNet-21k, iNat18, Places365** (Hölzl) — no cargados; replicación de robustez/noise/transfer queda fuera del alcance inicial.
- **IMDB** (Kingma & Ba) — NLP, fuera del alcance de visión.
- **MNLI, Billion Word Benchmark** (Fort, McCandlish) — NLP, fuera del alcance.
- **Atari, Dota** (McCandlish) — RL, fuera del alcance.
- **rcv1.binary, covtype.binary, protein** (Johnson & Zhang) — LIBSVM clásicos, fuera del alcance.
- **MRNet** (Forouzesh & Thiran) — datos médicos, fuera del alcance.
- **NAS-Bench-201/301, Flower102, ImageNet-16-120** (Ru et al.) — benchmarks NAS, fuera del alcance.
- **Datasets sintéticos** (Random Features, gaussianas, toy $y = x_0 x_1 + \epsilon$) — generables en runtime si se requiere para experimentos teóricos puntuales.

---

**Ruta del documento creado**: `/Users/laiqiands/GitHub/2ndBrain/02 - Files/Uni/TFG/datasets.md`
