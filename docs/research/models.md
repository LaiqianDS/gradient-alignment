# Modelos por Paper

Consolidación de las arquitecturas y modelos utilizados en los experimentos de los 16 papers del TFG. Agrupados por familia. Solo se incluyen modelos efectivamente reportados en las secciones de datasets y modelos de cada paper.

---

## A Study of Gradient Variance in Deep Learning (Faghri et al., 2020)

- **MLPs / Fully-Connected** — MLP de tres capas FC sobre MNIST con arquitectura $784 \to 1024 \to 1024 \to 10$, activación ReLU, sin dropout, lr 0.02, weight decay $5\times10^{-4}$, momentum 0.5.
- **ResNets** — ResNet8 sobre CIFAR-10 (sin batch normalization, lr 0.01, momentum 0.9, 80 000 iteraciones); ResNet32 sobre CIFAR-100 (lr inicial 0.1); ResNet18 sobre ImageNet (lr 0.1, weight decay $1\times10^{-4}$, momentum 0.9).
- **Modelos especiales** — Random Features (RF) models (Rahimi y Recht, 2007) con activación ReLU, dimensión oculta fija $h_s = 1000$, ratio de sobreparametrización $h_s/N \in [0.1, 10]$, modelo teacher $\theta_1 \in \mathbb{R}^{I \times h_t}$, $\theta_2 \in \mathbb{R}^{h_t \times 1}$, cross-entropy, mini-batch 10.
- **Baselines de optimización** — SG-B, SG-2B, SVRG, GC (método propuesto). Mini-batch $B=128$ en todos los experimentos de imagen.

---

## A Theory of Neural Tangent Kernel Alignment and Its Influence on Training (Shan y Bordelon, 2021)

- **MLPs / Fully-Connected** — MLP de dos capas con $N=500$ unidades ReLU sobre MNIST (clasificación binaria odd-even y de 10 clases); MLPs ReLU de profundidad variable (2, 3, 4, 5 capas ocultas) sobre MNIST para validar la predicción $\boldsymbol{K}_\infty \propto L \boldsymbol{y}\boldsymbol{y}^\top + \boldsymbol{K}_0$.
- **CNNs** — CNN sobre subconjunto de CIFAR-10 para clasificación de dos clases.
- **ResNets** — Wide ResNet (Zagoruyko y Komodakis, 2017) con factor de ensanchamiento $k=3$ y $b=2$ bloques sobre subconjunto de CIFAR-10 (100 imágenes, dos clases).
- **Redes lineales profundas** — $f(\boldsymbol{x}) = \boldsymbol{w}^{L+1\top} \boldsymbol{W}^L \cdots \boldsymbol{W}^1 \boldsymbol{x}$ de $L$ capas para el análisis analítico.
- **Modelos teóricos** — redes ReLU de dos capas $f(\boldsymbol{x}^\mu) = \sum_i V_i \,\text{ReLU}(\boldsymbol{w}_i \cdot \boldsymbol{x}^\mu)$ sobre datos sintéticos (targets lineales, mezcla de 10 gaussianas, clasificación binaria aleatoria).

---

## Accelerating Stochastic Gradient Descent using Predictive Variance Reduction (Johnson y Zhang, 2013)

- **Modelos clásicos** — regresión logística multiclase L2-regularizada sobre MNIST ($\lambda = 10^{-4}$); regresión logística L2-regularizada sobre rcv1.binary, covtype.binary (LIBSVM), protein (KDD Cup) y CIFAR-10 ($\lambda = 10^{-3}$ para CIFAR-10, $10^{-5}$ para el resto).
- **MLPs / Fully-Connected** — red neuronal con **una capa oculta totalmente conectada de 100 nodos**, activación sigmoide, salida softmax de 10 clases, regularización L2, mini-batches de tamaño 10 sobre MNIST ($\lambda = 10^{-4}$) y CIFAR-10 ($\lambda = 10^{-3}$).
- **Baselines de optimización** — SVRG (método propuesto), SGD con learning rate scheduling, SDCA, SAG; $m = 2n$ en convexo y $m = 5n$ en no convexo.

---

## Adam - A Method for Stochastic Optimization (Kingma y Ba, 2015)

- **Modelos clásicos** — regresión logística L2-regularizada multiclase sobre MNIST (vectores de 784 dimensiones, minibatch 128, $\alpha_t = \alpha/\sqrt{t}$); regresión logística sobre IMDB con representaciones bag-of-words de 10 000 palabras y dropout 50%.
- **MLPs / Fully-Connected** — red FC con **dos capas ocultas de 1 000 unidades ReLU** sobre MNIST, con dropout y minibatches de 128.
- **CNNs** — arquitectura **c64-c64-c128-1000** sobre CIFAR-10 (tres etapas alternantes de filtros 5x5 y max-pooling 3x3 con stride 2, seguidas de capa densa de 1 000 ReLU), minibatches de 128 y dropout en entrada y capa densa.
- **Modelos especiales** — Variational Auto-Encoder (Kingma y Welling, 2013) con 500 unidades ocultas softplus y latente gaussiano de 50 dimensiones.
- **Baselines de optimización** — Adam (propuesto), AdaMax, AdaGrad, RMSProp, SGD-Nesterov, AdaDelta, SFO (quasi-Newton).

---

## An Empirical Model of Large-Batch Training (McCandlish, Kaplan y Amodei, 2018)

- **CNNs** — CNN simple sobre MNIST con SGD; CNN simple sobre SVHN con SGD y Adam.
- **ResNets** — ResNet-32 con momentum sobre CIFAR-10; ResNet-50 con momentum y schedule de learning rate decay sobre ImageNet.
- **LSTMs / RNNs** — LSTM autoregresivo de tamaño 2048 (también variantes 1024 y 512 para estudio de dependencia con tamaño del modelo) sobre el Billion Word Benchmark con codificación BPE de vocabulario 40k, optimizado con Adam.
- **Modelos especiales** — Variational Autoencoder con arquitectura InfoGAN sobre SVHN; autoencoder simple con la misma arquitectura.
- **Modelos de RL** — siete juegos de Atari (Alien, Beam Rider, Breakout, Pong, Qbert, Seaquest, Space Invaders) entrenados con A2C y RMSProp; agentes Dota 1v1 y 5v5 entrenados con PPO asíncrono.

---

## An overview of gradient descent optimization algorithms (Ruder, 2017)

Review/Slides — sin experimentos propios. El artículo presenta una taxonomía cualitativa de optimizadores (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam) y arquitecturas paralelas/distribuidas para SGD (Hogwild!, Downpour SGD, delay-tolerant algorithms, TensorFlow, Elastic Averaging SGD). Las únicas ilustraciones cualitativas son visualizaciones sobre la función de Beale y un saddle point (Figura 4 de Alec Radford).

---

## Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization (Chatterjee, 2019)

- **MLPs / Fully-Connected** — red FC con **una capa oculta de 2 048 ReLUs** y softmax de 10 vías sobre MNIST, inicialización Xavier, vanilla SGD sin momentum, cross-entropy, lr 0.1, minibatch 100, $10^5$ pasos (~170 épocas), sin regularización explícita.
- **MLPs / Fully-Connected (winsorized SGD)** — red FC reducida con **3 capas ocultas de 256 ReLUs** sobre MNIST, 60 000 pasos (~100 épocas), lr 0.1, minibatch 100, $c \in \{0, 1, 2, 4, 8\}$. Elección de arquitecturas FC (no convolucionales) deliberada para evitar contaminación por inductive bias arquitectónico.

---

## Disparity Between Batches as a Signal for Early Stopping (Forouzesh y Thiran, 2021)

- **CNNs** — AlexNet sobre MNIST (256 muestras y variante con 50% label noise).
- **VGG** — VGG-13 sobre CIFAR-10 (1.28k muestras, datos limitados).
- **ResNets** — ResNet-18 sobre CIFAR-100 (1.28k muestras; también escenario noisy 50% label noise).
- **MLPs / Fully-Connected** — redes totalmente conectadas en estudios de ancho y tamaño (mencionadas sin especificación exacta).
- **Modelos específicos de dominio** — modelo para MRNet (resonancias magnéticas de rodilla, tareas de anormalidad, lesiones ACL y meniscales).

---

## Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks (Hölzl, 2025)

- **Transformers / ViT** — ViT/S-16 (training from scratch en CIFAR-10/-N y ImageNet-1k, Adam lr 0.0001 en CIFAR, AdamW lr 0.001 en ImageNet); ViT/B-16 preentrenado en ImageNet-21k para fine-tuning sobre ImageNet-1k, iNat18 y Places365 (con SGD).
- **CNNs modernas** — ConvNeXt-Femto (training from scratch en CIFAR y ImageNet; Adam lr 0.001 en CIFAR, AdamW lr 0.001 en ImageNet).
- **Configuraciones** — splits 90/10% y 99/1%, batch sizes 512/1024/512, schedulers Cosine y WarmupCosine. Datasets: CIFAR-10, CIFAR-10-N (etiquetas humanas ruidosas 0%, 9%, 17%, 40%), CIFAR-C, ImageNet-1k (V2, ReaL), ImageNet-C, ImageNet-21k (preentrenamiento), iNat18, Places365.

---

## Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment (Chatterjee y Zielinski, 2020)

- **ResNets** — ResNet-18 sobre ImageNet (experimento principal con tres variantes de ruido en etiquetas: 0%, 50%, 100% aleatorizadas; apéndice añade 25% y 75%).
- **CNNs** — Inception-V3 sobre ImageNet (validación de generalidad del fenómeno).
- **Configuración** — SGD con momentum 0.9, batch size 4096, schedule de Goyal et al. 2017, sin augmentation ni weight decay (para observar memorización), tamaño de muestra $m = 40\,356$ ejemplos fijos (y duplicación a 80 072 en apéndice).

---

## On the Ineffectiveness of Variance Reduced Optimization for Deep Learning (Defazio y Bottou, 2019)

- **CNNs** — LeNet-5 modificada con batch-norm y ReLU ($\sim 62\text{k}$ parámetros) sobre CIFAR-10.
- **ResNets** — ResNet-18 escalado a la mitad de feature planes por capa ($\sim 69\text{k}$ parámetros) y ResNet-110 ($\sim 1.7\text{M}$ parámetros) sobre CIFAR-10; ResNet-18 sobre ImageNet (90 epochs); fine-tuning de ResNet-50 sobre ImageNet activando SVRG desde epochs 0/20/40/60/80.
- **DenseNets** — DenseNet-40-36 wide (growth rate 36, depth 40, $\sim 1.5\text{M}$ parámetros, test error $<5\%$) sobre CIFAR-10; fine-tuning de DenseNet-169 sobre ImageNet activando SVRG desde epochs 60/80.
- **Configuración** — batch 128, momentum 0.9, weight decay $10^{-4}$, lr inicial 0.1 con decays $10\times$ en epochs 150 y 220, sampling sin reemplazo, data augmentation (random horizontal flips, random cropping $32\times32$ tras padding de 4 píxeles). Activación ELU como sanity check de suavidad frente a ReLU.
- **Baselines de optimización** — SGD con momentum, SVRG (con transform locking, BN reset, dropout seed reuse), SCSG (streaming SVRG con mega-batch $10\text{–}32\times$ el mini-batch).

---

## RMSProp - Divide the gradient by a running average of its recent magnitude (Tieleman y Hinton, 2012)

Review/Slides — sin experimentos propios. Material correspondiente a la Lecture 6 (slides 6a–6e) del curso *Neural Networks for Machine Learning* de Hinton (Coursera, 2012). Recomendaciones de uso cualitativas referidas a familias de problemas: redes profundas (especialmente con cuellos de botella estrechos), redes recurrentes y redes anchas y poco profundas. La slide *Summary of learning methods for neural networks* recomienda RMSProp (con momentum opcional) para datasets grandes y redundantes y reserva métodos full-batch (gradiente conjugado, LBFGS, rprop) para datasets pequeños.

---

## Speedy Performance Estimation for Neural Architecture Search (Ru et al., 2021)

- **NAS search spaces** — NAS-Bench-201 (NB201, 6 466 arquitecturas únicas sobre CIFAR-10, CIFAR-100, ImageNet-16-120); DARTS (5 000 arquitecturas en CIFAR-10 con NAS-Bench-301); ResNet/ResNeXt search space (50 000 arquitecturas en CIFAR-10); Randomly Wired Neural Networks (RWNN, 69×8 arquitecturas en Flower102).
- **DARTS adicional** — arquitecturas grandes de 20 celdas con tres set-ups distintos (learning rate inicial, scheduler, batch size).
- **Métodos NAS integrados** — query-based (Random Search, Regularised Evolution, Bayesian Optimisation con TSE-EMA $T=10$), NAS one-shot (RandNAS, FairNAS, MultiPaths con $B$ mini-batches adicionales), NAS diferenciable (DARTS, DrNAS con gradiente de TSE sobre 100 mini-batches).
- **Baselines de estimación** — TSE base, TLmini, SoVL, VAccES, LcSVR ($\nu$-SVR), JacCov, SNIP, SynFlow.

---

## Stiffness: A New Perspective on Generalization in Neural Networks (Fort et al., 2019)

- **MLPs / Fully-Connected** — red FC ReLU de 3 capas $X \to 500 \to 300 \to 100 \to y$.
- **CNNs** — CNN de 3 capas con filtros $3\times3$ y canales 16, 32, 32 seguidos de max-pooling $2\times2$ y una capa final FC.
- **ResNets** — ResNet20v1 (He et al., 2015) en la implementación de Chollet et al. (2015), sin batch normalization.
- **Transformers** — BERT (Devlin et al., 2018) fine-tuned sobre MNLI (Williams et al., 2017), como validación en NLP.
- **Configuración** — Adam con learning rates constantes, batch 32, entradas preprocesadas con media cero, varianza unitaria y normalización a esfera unidad ($|\vec{X}|=1$). Datasets: MNIST, FASHION MNIST, CIFAR-10, CIFAR-100.

---

## The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent (Sankararaman et al., 2020)

- **ResNets** — Wide Residual Networks (WRNs) (Zagoruyko y Komodakis, 2016) con notación CNN-$\beta$-$\ell$ para profundidad $\beta$ y factor de anchura $\ell$. Estudio de profundidad: $\beta \in \{16, 22, 28, 34, 40\}$ con anchura fija 2. Estudio de anchura: profundidad fija 16 con factores $\{2, 3, 4, 5, 6\}$. Estudio de batch norm/skip connections: $\beta \in \{16, 22, 28, 34, 40, 52, 76, 100\}$.
- **CNNs** — redes convolucionales (especificación no detallada).
- **MLPs / Fully-Connected** — multi-layer perceptrons (especificación no detallada).
- **Configuración** — SGD sin momentum, 200 epochs, mini-batches 128, lr decaído por factor 10 en epochs 80 y 160, inicialización MSRA (He et al., 2015), dropout y weight decay desactivados, 5 repeticiones. Datasets: CIFAR-10 (cuerpo), CIFAR-100 y MNIST (apéndice A).

---

## Understanding Why Neural Networks Generalize Well Through GSNR of Parameters (Liu et al., 2020)

- **CNNs** — CNN simple sobre MNIST con dos bloques Conv-ReLU-MaxPooling más dos capas FC (Tabla 2, $p \in \{6, 8, 10, 12, 14, 16, 18, 20\}$ canales); red más profunda sobre CIFAR-10 con 4 bloques Conv-BN-ReLU-Conv-BN-ReLU-MaxPooling más 3 FC (Tabla 3).
- **ResNets** — ResNet18 para comparar GSNR con etiquetas reales frente a etiquetas aleatorizadas.
- **MLPs / Fully-Connected** — MLP de dos capas (2 entradas, $N$ neuronas ocultas, 1 salida) sobre toy dataset sintético generado por $y = x_0 x_1 + \epsilon$.
- **Modelos clásicos** (mencionados como contraste teórico) — regresión logística y SVM, donde el GSNR decrece monótonamente al carecer de feature learning.
- **Configuración** — descenso de gradiente puro (no SGD), lr pequeño $\lambda = 0.001$. Variables experimentales: tamaño $n \in \{1000, 2000, 4000, 6000, 8000, 10000, 15000\}$ en MNIST, $p_{random} \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$, anchura.

---

## Tabla comparativa

| Paper | Modelos |
|-------|---------|
| Faghri et al., 2020 | MLP 784-1024-1024-10, ResNet8, ResNet32, ResNet18, Random Features (ReLU, $h_s=1000$) |
| Shan y Bordelon, 2021 | MLP 2 capas $N=500$ ReLU, MLPs profundidad 2-5, CNN, Wide ResNet ($k=3$, $b=2$), redes lineales profundas, ReLU 2 capas |
| Johnson y Zhang, 2013 | Regresión logística L2, MLP 1 capa 100 sigmoid + softmax |
| Kingma y Ba, 2015 | Regresión logística L2, MLP 2x1000 ReLU, CNN c64-c64-c128-1000, VAE (500 softplus, latente 50) |
| McCandlish et al., 2018 | CNN simple, ResNet-32, ResNet-50, LSTM 2048/1024/512, VAE InfoGAN, autoencoder, A2C/PPO Atari/Dota |
| Ruder, 2017 | Review/Slides — sin experimentos propios |
| Chatterjee, 2019 | MLP 1 capa 2048 ReLU, MLP 3 capas 256 ReLU |
| Forouzesh y Thiran, 2021 | AlexNet, VGG-13, ResNet-18, FC, modelo MRNet |
| Hölzl, 2025 | ViT/S-16, ViT/B-16, ConvNeXt-Femto |
| Chatterjee y Zielinski, 2020 | ResNet-18, Inception-V3 |
| Defazio y Bottou, 2019 | LeNet-5 (BN+ReLU), ResNet-18 scaled, ResNet-110, DenseNet-40-36, ResNet-18/50 (ImageNet), DenseNet-169 (ImageNet) |
| Tieleman y Hinton, 2012 | Slides — sin experimentos propios |
| Ru et al., 2021 | NAS-Bench-201, DARTS, ResNet/ResNeXt search space, RWNN |
| Fort et al., 2019 | FC 500-300-100 ReLU, CNN 3 capas (16/32/32), ResNet20v1, BERT |
| Sankararaman et al., 2020 | Wide ResNets ($\beta\in\{16..100\}$, $\ell\in\{2..6\}$), CNNs, MLPs |
| Liu et al., 2020 | CNN MNIST (2 bloques + 2 FC), CNN CIFAR-10 (4 bloques BN + 3 FC), ResNet18, MLP 2 capas, regresión logística, SVM |

---

## Frecuencia de arquitecturas

**MLPs / Fully-Connected** (9 papers): Faghri et al., Shan y Bordelon, Johnson y Zhang, Kingma y Ba, Chatterjee 2019, Forouzesh y Thiran, Fort et al., Sankararaman et al., Liu et al.

**CNNs (no-ResNet)** (9 papers): Shan y Bordelon, Kingma y Ba, McCandlish et al., Forouzesh y Thiran (AlexNet), Chatterjee y Zielinski (Inception-V3), Defazio y Bottou (LeNet-5), Fort et al., Sankararaman et al., Liu et al.

**ResNets / Wide ResNets** (10 papers): Faghri et al. (ResNet8/18/32), Shan y Bordelon (Wide ResNet $k=3$), McCandlish et al. (ResNet-32/50), Forouzesh y Thiran (ResNet-18), Hölzl (vía ConvNeXt — no ResNet estricto, no se contabiliza aquí), Chatterjee y Zielinski (ResNet-18), Defazio y Bottou (ResNet-18 scaled, ResNet-110, ResNet-50), Fort et al. (ResNet20v1), Sankararaman et al. (WRN $\beta$-$\ell$), Liu et al. (ResNet18). Total con ResNet estricto: 9 papers.

**DenseNets** (1 paper): Defazio y Bottou (DenseNet-40-36 wide, DenseNet-169).

**VGG** (1 paper): Forouzesh y Thiran (VGG-13).

**Transformers / ViT / ConvNeXt** (2 papers): Hölzl (ViT/S-16, ViT/B-16, ConvNeXt-Femto), Fort et al. (BERT como validación en NLP).

**LSTMs / RNNs** (1 paper): McCandlish et al. (LSTM 2048/1024/512).

**Modelos clásicos (regresión logística L2, SVM)** (3 papers): Johnson y Zhang, Kingma y Ba, Liu et al. (regresión logística y SVM como contraste teórico).

**Modelos especiales (VAE, Autoencoder, InfoGAN, Random Features)** (3 papers): Faghri et al. (Random Features), Kingma y Ba (VAE), McCandlish et al. (VAE InfoGAN, autoencoder).

**Redes lineales profundas / modelos teóricos** (1 paper): Shan y Bordelon (redes lineales profundas y ReLU 2 capas).

**NAS search spaces** (1 paper): Ru et al. (NAS-Bench-201, DARTS, ResNet/ResNeXt search space, RWNN).

**Modelos de RL (A2C, PPO)** (1 paper): McCandlish et al. (Atari y Dota).

**Modelos específicos de dominio (MRNet)** (1 paper): Forouzesh y Thiran.

**Papers sin experimentos propios** (2 papers): Ruder 2017, Tieleman y Hinton 2012.

---

**Ruta del documento creado:** `/Users/laiqiands/GitHub/2ndBrain/02 - Files/Uni/TFG/models.md`
