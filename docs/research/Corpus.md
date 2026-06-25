# Corpus: datasets y modelos por paper

Qué datasets y arquitecturas usa cada paper del corpus, fusionados como pares **dataset → modelo (config)**, más las **frecuencias** que justifican el setup del TFG y las **decisiones de implementación**. Las frecuencias se cuentan sobre los 14 papers que proponen métrica o setup (los 15 del corpus menos *On the Ineffectiveness of Variance Reduced Optimization*, related-work); alimentan la convergencia de la literatura en [[1 - Diseño]].

---

## A Study of Gradient Variance in Deep Learning (Faghri et al., 2020)

- **MNIST** → MLP FC $784 \to 1024 \to 1024 \to 10$, ReLU (sin dropout, lr 0.02, weight decay $5\times10^{-4}$, momentum 0.5).
- **CIFAR-10** → ResNet8 sin batch normalization (lr 0.01, momentum 0.9, 80 000 iteraciones); incluye experimento con label corruption del 10%.
- **CIFAR-100** → ResNet32 (lr inicial 0.1).
- **ImageNet** → ResNet18 (lr 0.1, weight decay $1\times10^{-4}$, momentum 0.9).
- **Random Features (RF), sintético** → modelo teacher ReLU (Rahimi & Recht, 2007), dimensión oculta $h_s = 1000$, ratio de sobreparametrización $h_s/N \in [0.1, 10]$, cross-entropy, mini-batch 10.
- **Optimización.** Baselines SG-B, SG-2B, SVRG, GC (método propuesto); mini-batch $B = 128$ en todos los experimentos de imagen.

## Accelerating Stochastic Gradient Descent using Predictive Variance Reduction (Johnson & Zhang, 2013)

- **MNIST** → regresión logística L2 multiclase ($\lambda = 10^{-4}$, convexo) y red FC de 1 capa oculta de 100 nodos sigmoid + softmax (no convexo).
- **CIFAR-10** → regresión logística L2 ($\lambda = 10^{-3}$) y red FC de 1 capa oculta de 100 nodos.
- **rcv1.binary, covtype.binary (LIBSVM), protein (KDD Cup)** → regresión logística L2 binaria ($\lambda = 10^{-5}$; covtype y protein con divisiones 50/50 train/test).
- **Optimización.** SVRG (propuesto), SGD con learning rate scheduling, SDCA, SAG; $m = 2n$ en convexo, $m = 5n$ en no convexo.

## Adam - A Method for Stochastic Optimization (Kingma & Ba, 2015)

- **MNIST** → regresión logística L2 multiclase (minibatch 128, $\alpha_t = \alpha/\sqrt{t}$); MLP de 2 capas ocultas de 1000 ReLU con dropout.
- **IMDB** → regresión logística sobre bag-of-words de 10 000 palabras, dropout 50%.
- **CIFAR-10** → CNN c64-c64-c128-1000 (tres etapas de filtros 5×5 + max-pooling 3×3 stride 2, densa final de 1000 ReLU), minibatch 128, dropout en entrada y densa.
- **MNIST (VAE)** → Variational Auto-Encoder (Kingma & Welling, 2013) con 500 unidades softplus y latente gaussiano de 50 dimensiones.
- **Optimización.** Adam (propuesto), AdaMax, AdaGrad, RMSProp, SGD-Nesterov, AdaDelta, SFO.

## An Empirical Model of Large-Batch Training (McCandlish et al., 2018)

- **MNIST** → CNN simple (SGD).
- **SVHN** → CNN simple (SGD y Adam); también autoencoder y VAE (arquitectura InfoGAN).
- **CIFAR-10** → ResNet-32 + momentum.
- **ImageNet** → ResNet-50 + momentum y schedule de learning rate decay.
- **Atari (7 juegos: Alien, Beam Rider, Breakout, Pong, Qbert, Seaquest, Space Invaders)** → A2C + RMSProp (RL).
- **Dota (1v1 y 5v5)** → PPO asíncrono.
- **Billion Word Benchmark** → LSTM autoregresivo 2048 (también 1024 y 512), BPE vocab 40k, Adam.

## An overview of gradient descent optimization algorithms (Ruder, 2017)

Review/Slides, sin experimentos propios. Taxonomía cualitativa de optimizadores (Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam) y arquitecturas paralelas/distribuidas para SGD (Hogwild!, Downpour SGD, delay-tolerant, TensorFlow, Elastic Averaging SGD). Únicas ilustraciones: función de Beale y un saddle point (Figura 4 de Alec Radford).

## Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization (Chatterjee, 2019)

- **MNIST** (único dataset) → MLP FC de 1 capa oculta de 2048 ReLUs, softmax 10 vías (Xavier, vanilla SGD sin momentum, CE, lr 0.1, minibatch 100, $10^5$ pasos ≈ 170 épocas, sin regularización explícita). Para winsorized SGD, MLP de 3 capas ocultas de 256 ReLUs (60 000 pasos ≈ 100 épocas, $c \in \{0,1,2,4,8\}$). Experimentos con label noise 25/50/75/100%. Elección de FC (no convolucional) deliberada para evitar inductive bias arquitectónico.

## Disparity Between Batches as a Signal for Early Stopping (Forouzesh & Thiran, 2021)

- **MNIST** → AlexNet (subconjunto de 256 muestras, datos limitados; variante con 50% label noise).
- **CIFAR-10** → VGG-13 (subconjunto de 1.28k muestras).
- **CIFAR-100** → ResNet-18 (subconjunto de 1.28k muestras; escenario noisy 50% label noise).
- **MRNet** → modelo de dominio para resonancias de rodilla (anormalidad, lesiones ACL y meniscales; datos limitados).
- **FC** → redes totalmente conectadas en estudios de ancho y tamaño (sin especificación exacta).

## Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks (Hölzl, 2025)

- **CIFAR-10, CIFAR-10-N** (etiquetas humanas ruidosas 0/9/17/40%), **CIFAR-C** → ConvNeXt-Femto y ViT/S-16 training from scratch (Adam lr 0.001 ConvNeXt; Adam lr 0.0001 ViT). Splits de validación 90/10 y 99/1.
- **ImageNet-1k** (+ V2, ReaL), **ImageNet-C** → ConvNeXt-F y ViT/S-16 from scratch (AdamW lr 0.001).
- **ImageNet-21k** → preentrenamiento de ViT/B-16.
- **iNat18, Places365** → fine-tuning de ViT/B-16 preentrenado (SGD).
- **Config.** Batch sizes 512/1024/512, schedulers Cosine y WarmupCosine.

## Making Coherence Out of Nothing At All: Measuring the Evolution of Gradient Alignment (Chatterjee & Zielinski, 2020)

- **ImageNet** (experimento principal) → ResNet-18 (principal) e Inception-V3 (validación de generalidad). Tres variantes de label noise sintético (0/50/100% aleatorizadas; apéndice añade 25/75%). SGD momentum 0.9, batch 4096, schedule de Goyal et al. 2017, sin augmentation ni weight decay (para observar memorización), muestra fija $m = 40\,356$ (duplicada a 80 072 en apéndice).
- Discusión cualitativa de resultados ajenos sobre **MNIST, Fashion MNIST, CIFAR-10, CIFAR-100, MNLI** (citando Fort et al. 2019 y Sankararaman et al. 2019), sin experimentos propios.

## On the Ineffectiveness of Variance Reduced Optimization for Deep Learning (Defazio & Bottou, 2019)

*Related-work, fuera del recuento de frecuencias.*

- **CIFAR-10** → LeNet-5 modificada (BN+ReLU, ~62k params); ResNet-18 a media anchura (~69k), ResNet-110 (~1.7M); DenseNet-40-36 wide (growth 36, depth 40, ~1.5M, test error <5%).
- **ImageNet** → ResNet-18 (90 epochs); fine-tuning de ResNet-50 (SVRG desde epochs 0/20/40/60/80) y DenseNet-169 (SVRG desde 60/80).
- **Config.** Batch 128, momentum 0.9, weight decay $10^{-4}$, lr 0.1 con decays $10\times$ en epochs 150 y 220, sin reemplazo, data augmentation (flips + crops $32\times32$ tras padding 4). ELU como sanity check de suavidad. Baselines: SGD+momentum, SVRG (transform locking, BN reset, dropout seed reuse), SCSG.

## RMSProp - Divide the gradient by a running average of its recent magnitude (Tieleman & Hinton, 2012)

Slides/Lecture deck (Coursera, Lecture 6a-6e), sin experimentos propios sistemáticos. Recomendaciones cualitativas por familia de problemas: redes profundas (especialmente con cuellos de botella estrechos), redes recurrentes, redes anchas y poco profundas. La slide *Summary of learning methods* recomienda RMSProp (con momentum opcional) para datasets grandes y redundantes, y reserva métodos full-batch (gradiente conjugado, LBFGS, rprop) para datasets pequeños.

## Speedy Performance Estimation for Neural Architecture Search (Ru et al., 2021)

- **NAS-Bench-201** (6466 arquitecturas únicas) → sobre **CIFAR-10, CIFAR-100, ImageNet-16-120**.
- **NAS-Bench-301** (5000 arquitecturas del espacio DARTS) → sobre **CIFAR-10**.
- **ResNet/ResNeXt search space** (50 000 arquitecturas) → sobre **CIFAR-10**.
- **Randomly Wired Neural Networks (RWNN)** (69×8 arquitecturas) → sobre **Flower102**.
- **DARTS adicional** → arquitecturas grandes de 20 celdas con tres set-ups (lr inicial, scheduler, batch size).
- **Métodos NAS** → query-based (Random Search, Regularised Evolution, BO con TSE-EMA $T=10$), one-shot (RandNAS, FairNAS, MultiPaths), diferenciable (DARTS, DrNAS). Baselines de estimación: TSE base, TLmini, SoVL, VAccES, LcSVR, JacCov, SNIP, SynFlow.

## Stiffness: A New Perspective on Generalization in Neural Networks (Fort et al., 2019)

- **MNIST, Fashion MNIST, CIFAR-10, CIFAR-100** → las mismas tres arquitecturas: MLP FC ReLU $X \to 500 \to 300 \to 100 \to y$; CNN de 3 capas (filtros $3\times3$, canales 16/32/32, max-pool $2\times2$, FC final); ResNet20v1 (He et al. 2015, impl. Chollet) sin batch normalization. CIFAR-100 además para estudiar super-clases en la matriz de class stiffness.
- **MNLI** → BERT (Devlin et al. 2018) fine-tuned, validación del fenómeno fuera de visión.
- **Config.** Adam con lr constante, batch 32, entradas con media cero, varianza unitaria y normalización a esfera unidad ($|\vec X|=1$).

## The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent (Sankararaman et al., 2020)

- **CIFAR-10** (cuerpo) → Wide Residual Networks (Zagoruyko & Komodakis 2016), notación CNN-$\beta$-$\ell$. Estudio de profundidad $\beta \in \{16,22,28,34,40\}$ (anchura fija 2); de anchura $\ell \in \{2,3,4,5,6\}$ (profundidad fija 16); de BN/skip $\beta \in \{16,\dots,100\}$.
- **CIFAR-100, MNIST** (apéndice A) → WRNs, CNNs y MLPs (CNN/MLP sin especificación detallada).
- **Config.** SGD sin momentum, 200 epochs, minibatch 128, lr decaído $10\times$ en epochs 80 y 160, init MSRA (He et al. 2015), dropout y weight decay desactivados, 5 repeticiones.

## Understanding Why Neural Networks Generalize Well Through GSNR of Parameters (Liu et al., 2020)

- **MNIST** → CNN simple de 2 bloques Conv-ReLU-MaxPooling + 2 FC (canales $p \in \{6,\dots,20\}$, tamaño $n \in \{1000,\dots,15000\}$, label noise $p_{random} \in \{0,\dots,0.5\}$).
- **CIFAR-10** → red Conv-BN-ReLU profunda (4 bloques + 3 FC); también ResNet18 para comparar GSNR entre etiquetas reales y aleatorizadas.
- **Toy sintético $y = x_0 x_1 + \epsilon$** → MLP de 2 entradas, $N$ ocultas, 1 salida (aísla el efecto de feature learning sobre GSNR).
- **Contraste teórico** → regresión logística y SVM, donde el GSNR decrece monótonamente al carecer de feature learning.
- **Config.** Descenso de gradiente puro (no SGD), lr pequeño $\lambda = 0.001$.

---

## Frecuencias

### Datasets (sobre los 15 papers con setup)

**Reales (visión).**
- **MNIST** (9 papers): Faghri; Johnson & Zhang; Kingma & Ba; McCandlish; Chatterjee (Coherent Gradients); Forouzesh & Thiran; Fort et al.; Sankararaman; Liu.
- **CIFAR-10** (11 papers): Faghri; Johnson & Zhang; Kingma & Ba; McCandlish; Forouzesh & Thiran; Hölzl; Ru et al.; Fort et al.; Sankararaman; Liu; (cualitativo en Chatterjee & Zielinski).
- **CIFAR-100** (6 papers): Faghri; Forouzesh & Thiran; Ru et al.; Fort et al.; Sankararaman; (cualitativo en Chatterjee & Zielinski).
- **ImageNet** (4 papers): Faghri; McCandlish; Chatterjee & Zielinski; Hölzl (variante ImageNet-1k).
- Otros (visión, 1 paper cada uno): Fashion MNIST (Fort et al.; cualitativo en Chatterjee & Zielinski); SVHN (McCandlish); CIFAR-10-N, CIFAR-C, ImageNet-V2/ReaL/C, ImageNet-21k, iNat18, Places365 (Hölzl); ImageNet-16-120 (Ru, vía NAS-Bench-201); Flower102 (Ru, vía RWNN); MRNet (Forouzesh & Thiran).

**Reales (otros dominios), 1 paper cada uno:** IMDB (Kingma & Ba); MNLI (Fort et al.; discutido en Chatterjee & Zielinski); Billion Word Benchmark, Atari, Dota (McCandlish); rcv1.binary, covtype.binary, protein (Johnson & Zhang).

**Benchmarks NAS:** NAS-Bench-201, NAS-Bench-301 (Ru et al.).

**Sintéticos / toy:** Random Features (Faghri); toy $y = x_0 x_1 + \epsilon$ (Liu). **Label noise sintético sobre datasets reales** (6 papers): Faghri (10% CIFAR-10); Chatterjee (25/50/75/100% MNIST); Chatterjee & Zielinski (0-100% ImageNet); Forouzesh & Thiran (50% MNIST y CIFAR-100); Hölzl (0/9/17/40% CIFAR-10-N, humanas); Liu (0-50% MNIST, real vs. aleatorio en CIFAR-10/ResNet18).

**Sin experimentos propios:** Ruder 2017 (review); Tieleman & Hinton 2012 (slides).

### Arquitecturas (por familia)

- **MLPs / Fully-Connected** (8 papers): Faghri; Johnson & Zhang; Kingma & Ba; Chatterjee 2019; Forouzesh & Thiran; Fort et al.; Sankararaman; Liu.
- **CNNs (no-ResNet)** (8 papers): Kingma & Ba; McCandlish; Forouzesh & Thiran (AlexNet); Chatterjee & Zielinski (Inception-V3); Defazio & Bottou (LeNet-5); Fort et al.; Sankararaman; Liu.
- **ResNets / Wide ResNets** (8 papers con ResNet estricto): Faghri (ResNet8/18/32); McCandlish (ResNet-32/50); Forouzesh & Thiran (ResNet-18); Chatterjee & Zielinski (ResNet-18); Defazio & Bottou (ResNet-18/110/50); Fort et al. (ResNet20v1); Sankararaman (WRN $\beta$-$\ell$); Liu (ResNet18). (Hölzl usa ConvNeXt, no ResNet estricto.)
- **DenseNets** (1): Defazio & Bottou. **VGG** (1): Forouzesh & Thiran. **Transformers / ViT / ConvNeXt** (2): Hölzl (ViT/S-16, ViT/B-16, ConvNeXt-Femto), Fort et al. (BERT, validación NLP). **LSTMs / RNNs** (1): McCandlish.
- **Modelos clásicos (reg. logística L2, SVM)** (3): Johnson & Zhang; Kingma & Ba; Liu (contraste teórico). **Especiales (VAE, autoencoder, InfoGAN, Random Features)** (3): Faghri; Kingma & Ba; McCandlish. **NAS search spaces** (1): Ru et al. **RL (A2C, PPO)** (1): McCandlish. **Dominio (MRNet)** (1): Forouzesh & Thiran. **Sin experimentos propios** (2): Ruder 2017, Tieleman & Hinton 2012.

---

## Decisiones de implementación (TFG)

Datasets efectivamente cargados en `src/data.py` y descargados a `data/`:

- **MNIST** (cubre 10 papers), **CIFAR-10** (12), **CIFAR-100** (6), **Fashion-MNIST** (Fort et al.), **Tiny-ImageNet** (sustituto de ImageNet-1k: Faghri, McCandlish, Chatterjee & Zielinski, Hölzl).

**Sustitución de ImageNet-1k.** El ILSVRC2012 completo (~150 GB) queda fuera de presupuesto de disco y cómputo para el TFG. Se usa **Tiny-ImageNet** (200 clases, 64×64, ~240 MB) como proxy de bajo coste. Implicación: los experimentos de escala completa de Faghri (ResNet18), McCandlish (ResNet-50), Chatterjee & Zielinski (ResNet-18/Inception-V3) y Hölzl (ViT/S-16, ConvNeXt-F) no son reproducibles fielmente; sus resultados se interpretan como evidencia cualitativa sobre subconjuntos de menor resolución, y debe declararse en la memoria.

**Datasets no cargados.** SVHN (McCandlish, disponible vía torchvision); CIFAR-10-N, CIFAR-C, ImageNet-C/V2/ReaL/21k, iNat18, Places365 (Hölzl, robustez/noise/transfer fuera del alcance inicial); IMDB (Kingma & Ba, NLP); MNLI, Billion Word Benchmark (Fort, McCandlish, NLP); Atari, Dota (McCandlish, RL); rcv1/covtype/protein (Johnson & Zhang, LIBSVM); MRNet (Forouzesh & Thiran, médico); NAS-Bench-201/301, Flower102, ImageNet-16-120 (Ru, benchmarks NAS); sintéticos (Random Features, gaussianas, toy $y = x_0x_1+\epsilon$, generables en runtime si se requieren).
