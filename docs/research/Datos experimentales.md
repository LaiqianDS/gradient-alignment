# Datos experimentales: datasets, particiones, arquitecturas y referencias

**Estado: borrador de trabajo (2026-06-20, depurado el 2026-08-25). Hechos verificados contra el código y la bibliografía.** Documento para ordenar las ideas antes de redactar la metodología. Su tema propio son los **datos experimentales**: datasets, particiones, arquitecturas, condiciones de entrenamiento, normalización y las anclas bibliográficas con las que se sitúa el techo razonable de cada celda. Las particiones, clases, arquitecturas y condiciones se derivan del código y se han cotejado línea a línea (`src/data.py`, `src/config.py`, `src/models.py`, `src/train.py`). Las referencias tienen cita firme en su mayoría; lo que queda abierto está en §6.

**Aquí no se redefine nada que sea de otro documento.** Las variables dependientes, las hipótesis H1-H6, los baselines y la matriz de runs viven en [[1 - Diseño]]; la semántica de cada métrica, en [[Métricas]]. Las secciones §4 y §5 de este fichero son punteros a ellos, con una excepción: la tabla de signos esperados de §5.3 vive aquí, porque es lectura de artículos y no método.

---

## 1. Tabla de datasets

Particiones: el **test oficial queda intacto** y la **val se recorta del train oficial** con muestreo estratificado por clase y semilla fija (`SPLIT_SEED = 42`), del tamaño convencional de cada dataset (ninguno trae val etiquetada de fábrica). La estratificación es por clase: cada clase aporta a la val `round(val_size · n_clase / N)` ejemplos, así que la val conserva las proporciones de clase del train. La partición es la misma en los ~960 runs, porque depende solo de `(etiquetas, val_size, SPLIT_SEED)` y nunca de la semilla del run. Fuente: `DATASET_SPECS`, `build_dataloaders` y `stratified_split_indices` en `src/data.py`.

| Dataset | Total usable | Train | Val | Test | Clases | Dim. entrada | Referencia supervisada (de la bibliografía) |
|---|---|---|---|---|---|---|---|
| MNIST | 70 000 | 50 000 | 10 000 | 10 000 | 10 | 1×28×28 | ~99,0 % (CNN tipo LeNet); MLP ~98,3 % |
| CIFAR-10 | 60 000 | 45 000 | 5 000 | 10 000 | 10 | 3×32×32 | ~95 % (ResNet-18 con aug.); MLP ~50–55 % |
| CIFAR-100 | 60 000 | 45 000 | 5 000 | 10 000 | 100 | 3×32×32 | ~77 % (ResNet-18 con aug.) |
| Tiny-ImageNet | 110 000 | 90 000 | 10 000 | 10 000 | 200 | 3×64×64 | ~50 % top-1 (ResNet-18 desde cero, 64×64) |

Notas a la tabla:

- **Total usable** = train oficial + test usado. En Tiny-ImageNet el `val/` público (10 000) hace de **test**, porque las etiquetas del test oficial (otros 10 000) no son públicas; ese test oficial no se usa, así que el total usable es 110 000, no 120 000. El `val/` viene en formato plano (no por carpetas de clase), así que se etiqueta leyendo `val_annotations.txt` y mapeando con el `class_to_idx` del train, para que train y test compartan los mismos índices de clase (cualquier otro orden permutaría las etiquetas).
- MNIST 50k/10k/10k es la convención clásica; CIFAR-10/100 45k/5k/10k sigue a He et al. (2015). En el código, solo MNIST lleva un `assert` que comprueba el tamaño exacto del split (`src/data.py:247`); las otras tres particiones no se asertan, pero salen de los tamaños estándar de torchvision (50k train + 10k test en CIFAR; 100k train + 10k val en Tiny) menos el recorte de val, y coinciden con la convención. La afirmación correcta es "coinciden con la convención", con asert de respaldo solo en MNIST.
- Todos son **clasificación supervisada de etiqueta única**, sin datos extra ni preentrenamiento.

### 1.1 Referencia por (dataset × arquitectura)

Más útil que un único número por dataset, porque el estudio cruza tres arquitecturas y el régimen de cada celda depende de ello (las celdas FC sobre datasets difíciles no alcanzan nunca su umbral, verificado en [[2 - Decisiones]]). Hay que separar **dos regímenes** que la bibliografía suele mezclar:

- **Régimen comparable** (Tabla A): sin data augmentation, desde cero y a resolución nativa, que es justo el de este estudio (§3). Es el techo honesto al que nuestros runs deberían acercarse.
- **Techo con augmentation** (al final de la sección): el mejor resultado publicado, casi siempre con *data augmentation, weight decay, schedule de learning rate y entrenamientos largos*. Solo sirve de cota laxa; nuestros runs deben quedar **por debajo**, y esa distancia es esperada, no un fallo.

Las cifras del régimen comparable no salen de los rankings de SOTA, sino de filas de *ablación* o de *baseline* de trabajos que estudian otra cosa: Scaling MLPs (sesgo inductivo) para el MLP, Chen & Ho (2018) para la CNN pequeña, y ZeroPur o Ko et al. (filas «Vanilla» y «No-aug») para ResNet-18.

**Tabla A. Régimen comparable** (sin augmentation, desde cero, resolución nativa). Banda esperada para nuestros runs:

| Arquitectura | MNIST | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|---|---|---|---|---|
| FC (MLP 1024-1024) | ~98,4 % | ~50–54 % | ~25–29 % | ~8–12 % |
| CNN (3 bloques conv) | ~99,0–99,4 % | ~70–80 % | ~45–55 % | ~20–35 % † |
| ResNet-18 (stem 3×3) | ~99,0–99,5 % | ~84–92 % | ~50–57 % | ~40–50 % † |

† **Sin fuente directa en este régimen.** No hay baseline publicado de una CNN simple ni de ResNet-18 desde cero, sin augmentation y a 64×64 nativo en Tiny-ImageNet (casi todo lleva augmentation, reescala a 224×224 o preentrena). Es una estimación; nuestro número será un punto de datos novedoso, no una réplica.

Anclas del régimen comparable, por arquitectura:

- **FC (MLP).** El mejor anclaje es Scaling MLPs (Bachmann et al., 2023): MLP llano sin augmentation, **54,2 % (CIFAR-10), 28,8 % (CIFAR-100) y 8,5 % (Tiny, 64×64 nativo)**. Su red es más profunda (6 capas) y se entrena más tiempo (100 épocas) que la nuestra (2 capas, 40 épocas), así que son **cotas superiores blandas**: nuestra red, más corta, debería quedar igual o por debajo. En MNIST el MLP llano sin augmentation ronda ~98,4 % (Simard et al., 2003; Srivastava et al., 2014). *Corrección:* el ~50–55 % atribuido antes a Lin et al. (2016) no es verificable como baseline de MLP llano (su resultado real es el modelo Z-Lin, ~70 % con cuello de botella y preentrenamiento, que ya no es un MLP llano); el anclaje correcto es Scaling MLPs.
- **CNN (3 bloques).** El gemelo arquitectónico más cercano es Chen & Ho (2018): 3 capas conv 16/16/32, sin BatchNorm ni dropout, *explícitamente sin augmentation*, con **99,58 % (MNIST), 82,77 % (CIFAR-10) y 55,21 % (CIFAR-100)**. Usan activación GReLU en lugar de ReLU llana, así que son **cotas superiores suaves** para nuestra red. Mantzaris (2025) acota por abajo con un baseline sin augmentation pero con BatchNorm: 77,2 % (CIFAR-10) y 45,3 % (CIFAR-100). De ahí las bandas ~70–80 % y ~45–55 %.
- **ResNet-18 (stem 3×3).** La variante para imagen pequeña (conv1 3×3 stride 1, sin maxpool) es **exactamente** la nuestra (`_build_resnet18` en `src/models.py`). Sin augmentation, ZeroPur (Ye et al., 2024) reporta la fila «Vanilla»: **83,80 % (CIFAR-10) y 57,01 % (CIFAR-100)**. Ko et al. (2024) dan una columna «No-aug» de 92,4 % en CIFAR-10, pero con weight decay alto (0,4) y 200 épocas, así que es una cota optimista frente a nuestro `weight_decay = 0` y 40 épocas; FractalNet (Larsson et al., 2017) corrobora ~55 % en CIFAR-100 sin augmentation. La banda esperada (wd=0, 40 épocas) queda en ~84–92 % y ~50–57 %.
- **MNIST (todas las arquitecturas).** Cualquier red pequeña satura en torno a ~99 %: LeNet-5 da 99,05 % (LeCun et al., 1998) y la CNN y ResNet apenas se distinguen ahí.

**Techo con augmentation** (cota laxa, no comparable 1:1). Para ResNet-18, GenURL (Li et al., 2023) entrena desde cero a resolución nativa con augmentation y weight decay y da **94,55 % (CIFAR-10), 78,07 % (CIFAR-100) y 50,68 % top-1 (Tiny, 400 épocas)**; los números de Tiny por encima de ~60 % suelen venir de reescalar a 224×224 o de preentrenamiento. Para FC y CNN, los mejores resultados publicados exigen **salir de la clase de arquitectura**: β-LASSO (Neyshabur, 2020) llega a ~85 % en CIFAR-10, pero es una red densa que aprende conectividad local tipo convolución, y All-CNN (Springenberg et al., 2015) ~91 %, pero es totalmente convolucional, profunda y con dropout. No son techos del MLP llano ni de la CNN simple.

---

## 2. Diseño de la matriz (24 celdas, ~960 runs)

`dataset (4) × arquitectura (3) × optimizador (2) = 24 celdas`. Cada celda barre `8 LR × 5 seeds = 40 runs`, de modo que `24 × 40 = 960` runs. Fuente: `DATASETS / MODELS / OPTIMIZERS / SEEDS / LR_GRID` en `src/config.py`.

**Arquitecturas** (las tres terminan en una `Linear`, para que las métricas localicen la cabeza como la última capa lineal de la red):

- **FC (MLP).** `Flatten → Linear(C·H·W → 1024) → ReLU → Linear(1024 → 1024) → ReLU → Linear(1024 → num_classes)`. Dos capas ocultas de 1024 con ReLU.
- **CNN.** Tres bloques `Conv 3×3 (padding 1) → ReLU → MaxPool 2` con anchos 16, 32 y 32 (tres reducciones ×2 de la resolución), seguidos de `AdaptiveAvgPool 4×4 → Flatten → Linear(512 → num_classes)`. El pool adaptativo final desacopla la cabeza del tamaño de entrada, así que la misma red vale para 28, 32 y 64 px. (Ojo: además del pool adaptativo final hay un `MaxPool 2` en cada bloque; la descripción de "+ pool adaptativo" se quedaba corta.)
- **ResNet-18.** `torchvision.resnet18(weights=None)` con stem de imagen pequeña: `conv1` se reemplaza por `Conv 3×3 stride 1 padding 1` (sin bias) y el `maxpool` por `Identity`, para no encoger 32×32 demasiado pronto. Ese `conv1` también adapta el número de canales de entrada, así que MNIST en escala de grises funciona sin replicar a tres canales. La cabeza es `fc = Linear(512 → num_classes)`.

**Optimizadores:** SGD (momentum 0,9) y Adam (betas y eps por defecto de PyTorch). Ambos operan sobre el gradiente crudo `∇L`, sin recorte ni regularización, para que las métricas sean comparables entre los dos. `weight_decay = 0` en ambos.

**Rejillas de LR (8 puntos, medio orden de magnitud por paso):**

- SGD: `3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0`.
- Adam: `3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1` (desplazada ~10× hacia abajo porque Adam preescala el paso por `1/√v`).

**Semillas:** 0–4. La misma semilla fija la misma inicialización y el mismo orden de barajado del train (el generador del train loader se siembra con la semilla del run). La partición train/val, en cambio, la fija `SPLIT_SEED = 42` con independencia de la semilla del run.

---

## 3. Condiciones de entrenamiento (lo que fija «mismas condiciones que tú»)

Para que el cotejo con la bibliografía sea honesto, así entrena este estudio (calibrado por el pilot del 2026-06-17; valores en `src/config.py`, semántica de uso en `src/train.py`):

- **Sin data augmentation.** Solo `ToTensor + Normalize` con la media y la desviación del train de cada dataset (estudio sensible al determinismo, [[1 - Diseño]] §Normalización). Son estadísticos de población por canal del split de train, en el rango [0, 1] que produce `ToTensor`:

  | Dataset | media | std |
  |---|---|---|
  | MNIST | (0,1307) | (0,3081) |
  | CIFAR-10 | (0,4914, 0,4822, 0,4465) | (0,2470, 0,2435, 0,2616) |
  | CIFAR-100 | (0,5071, 0,4865, 0,4409) | (0,2673, 0,2564, 0,2762) |
  | Tiny-ImageNet | (0,4802, 0,4481, 0,3975) | (0,2770, 0,2691, 0,2821) |

  Recalcularlos desde cero reproduce estas constantes con un error de ~5e-5 en MNIST/CIFAR y ~6e-4 en Tiny (la tolerancia más floja en Tiny es consistente con diferencias de descodificación JPEG entre versiones de librería).

- **`weight_decay = 0.0`** (sin regularización explícita), **`batch_size = 128`**, **`probe_size = 256`** (fijo: es un knob científico que fija el número de muestras del estimador de gradiente, así que no se mueve por la rejilla).
- El train loader **baraja con un generador sembrado** por la semilla del run y **descarta el último batch parcial** (`drop_last = True`); por eso los ejemplos efectivos por época quedan justo por debajo del nominal (MNIST 50 000 → 390 batches × 128 = 49 920). Val y test no se barajan ni descartan nada.
- El **probe** es un batch fijo de 256 ejemplos del train, determinista en `(dataset, probe_size, semilla del run)`, **congelado y reutilizado** en todas las medidas de métricas a lo largo del entrenamiento (no se vuelve a muestrear cada época).
- **Presupuesto de épocas y umbral de val-acc por dataset** (calibrados en el pilot, por debajo del techo ajustado de cada dataset para que los modelos fuertes crucen sin censurar):

  | Dataset | Épocas | Umbral val-acc (VD1) |
  |---|---|---|
  | MNIST | 20 | 0,97 |
  | CIFAR-10 | 40 | 0,65 |
  | CIFAR-100 | 40 | 0,35 |
  | Tiny-ImageNet | 40 | 0,20 |

  El entrenamiento corre **siempre el presupuesto completo de épocas**: no hay early-stop al cruzar el umbral. El umbral solo se usa **después**, para leer VD1 (épocas hasta cruzarlo) sobre la curva de val-acc suavizada. Por eso VD2 y VD3 («dentro del presupuesto») están bien definidas: la trayectoria completa siempre existe.

Implicación: nuestros números **no** son comparables 1:1 con el SOTA augmentado. Son la referencia de «cuánto se puede esperar entrenando desde cero, supervisado y sin trucos». El gap con la bibliografía es la diferencia atribuible a augmentation, weight decay, schedules de LR y presupuesto.

---

## 4. Métricas a utilizar (predictores)

Los predictores del estudio son las **8 métricas de gradiente** del registro completo (`REGISTRY` en `src/metrics/__init__.py`) más los baselines que no tocan el gradiente. Se miden al final de cada época sobre el probe fijo descrito en §3, y el análisis las lee en las ventanas `f`, la fracción del presupuesto de épocas.

La semántica de cada métrica, sus fórmulas y los nombres de columna que emite viven en [[Métricas]]. Los baselines sin gradiente están en [[1 - Diseño]] §Baselines: son lo que la curva de loss ya da sin coste, y son el rival que las métricas de gradiente tienen que batir para que H2 se sostenga.

---

## 5. Variables dependientes e hipótesis

Las seis variables dependientes (VD1-VD6), con su definición exacta, la curva sobre la que se lee cada una y el nombre de la clave que emite, viven en [[1 - Diseño]] §Variables dependientes. Las convenciones de lectura están implementadas en `efficiency_summary` (`src/train.py`), y los valores calculados están en el `summary.json` de cada uno de los 960 runs.

Aquí no se copian, para que no haya dos versiones que puedan divergir.

`seconds_to_threshold` se mide y se registra, pero conviene no leerlo como indicador de eficiencia: el wall-clock de un run mezcla lo rápido que aprende el modelo con la carga que tuviera la máquina y con el coste de la propia instrumentación, que es de algo más de 2x. Las épocas no tienen ese problema.

### 5.1 Salvedad de datos (pilot, Tiny-ImageNet)

Los runs de `reports_pilot/` para Tiny-ImageNet tienen el `final_test_acc` **mal calculado** (bug previo a la corrección), así que VD4 y, por depender del test, VD5 y VD6 no son fiables en esas celdas del pilot; las métricas del lado val (VD1-VD3) y el timing sí lo son. Desde el 2026-07-17 la referencia corregida vive **dentro de cada run del pilot**: `reports_pilot/<run>/testfix_40ep/` contiene la celda re-corrida con la corrección (mismo LR centrado y seed 0, pero al presupuesto calibrado de 40 épocas, no a las 160 del pilot: horizontes distintos, no intercambiables), y el `summary.json` corrupto se auto-declara con la clave `_tiny_test_note`. La antigua `reports_validity/` quedó retirada tras la fusión. La referencia sirve para apuntar test-acc/gap por celda, no para análisis (un solo LR y seed); como todos los `reports_*` está gitignored, y el test-acc definitivo de Tiny saldrá de la matriz completa, ya con el código corregido.

### 5.2 Hipótesis y contrastes

Las seis hipótesis H1-H6 viven en [[1 - Diseño]] §Hipótesis a contrastar, enunciadas como afirmaciones falsables. El método con el que se decidirá cada una está **por definir**: el plan de análisis anterior se retiró el 2026-08-25 ([[2 - Decisiones]]), así que hoy no existe criterio escrito para ninguna.

Las ventanas `f`, la fracción del presupuesto de épocas en la que se leen las métricas, sí están fijadas en `src/config.py` (`windows = (0.05, 0.10, 0.25, 0.50, 1.0)`) y se registran en `metrics_at_window.parquet`.

### 5.3 Tabla de signos esperados

Qué signo predice el paper de cada métrica frente a VD1 (épocas hasta el umbral, menos es más rápido). Es el material que sostiene H6, así que vive aquí y no en un documento de método.

| Métrica | Signo | Base |
|---|---|---|
| gradient confusion | + | fuerte (Sankararaman et al.) |
| stiffness (intra-clase) | − | fuerte (Fort et al.) |
| m-coherence | − | fuerte vía CGH (Chatterjee 2020: más coherencia → la loss cae más rápido; medido en train). Chatterjee & Zielinski no afirma velocidad |
| gradient disparity | + | extrapolada (predice test error, no velocidad) |
| NGV | + | fuerte (Faghri et al.) |
| GNS | + | fuerte (McCandlish ec. 2.7/D.1: más pasos a batch fijo; requiere B ≲ 𝓑 y LR razonable, y el GNS medido depende del LR) |
| GSNR | − | extrapolada (el paper solo afirma el gap; vs VD4: + también extrapolada, derivada del gap) |
| GWA | − | extrapolada (el paper calla sobre velocidad; vs VD4: + fuerte, con matices) |
| `val-acc@f` (baseline) | − | por construcción |
| TSE / `val-loss@f` (baselines) | + | por construcción |

Verificado contra los papers el 2026-07-17 (veredictos y citas en [[2 - Decisiones]]). El reparto real de predicciones fuertes: GSNR la lleva sobre el gap (signo −, vía OSGR; el gap del paper es en loss, la misma cantidad que VD5), y su + vs VD4 es derivación, no claim del paper. GWA la lleva sobre VD4 (signo +; Pearson 0,99 solo en ConvNeXt/CIFAR-10, 0,92 cross-arquitectura, medida sobre el máximo de alineamiento de la trayectoria completa, no en ventana temprana), y su − vs el gap es direccional cualitativo (el paper nunca mide test loss − train loss). m-coherence mantiene el − vs el gap con la salvedad del propio paper (conexión "complicated": con 100% label noise la coherencia también alcanza valores altos; lo informativo según la teoría es la coherencia temprana, lo que favorece la ventana del TFG). La aparente contradicción de signos (+ vs VD4 pero − vs el gap) es solo aritmética: VD4 y el gap apuntan a sentidos opuestos de «bueno» (más test-acc es mejor, más gap es peor).

---

## 6. Pendiente / por confirmar

- **Citas de la columna bibliográfica.** Régimen comparable (sin augmentation, §1.1 Tabla A) ya anclado: FC con Scaling MLPs (Bachmann et al., 2023); CNN simple con Chen & Ho (2018) y Mantzaris (2025); ResNet-18 con ZeroPur (Ye et al., 2024), Ko et al. (2024) y FractalNet (Larsson et al., 2017); MNIST con LeCun et al. (1998). Techo con augmentation anclado con GenURL (Li et al., 2023), y el split 45k/5k con He et al. (2015). Siguen **sin fuente directa** (estimación, punto de datos novedoso) solo las dos celdas † de la Tabla A: CNN simple y ResNet-18 desde cero, sin augmentation, a 64×64 nativo en Tiny-ImageNet. Decidir si en la memoria va **un** número por dataset o la rejilla 3×4 completa por régimen.
- **Cómo presentar el gap con el SOTA augmentado** en la memoria: como columna «referencia con augmentation» frente a «nuestro mejor run sin augmentation», una vez haya resultados de la matriz.
- **MNIST en la matriz pero ¿en la tabla titular?** Está en las 24 celdas; valorar si en la memoria final se mantiene o se relega a anexo (es el dataset «fácil» que puede dominar la agregación, paradoja de Simpson, [[1 - Diseño]] §Riesgos).

---

## Referencias (anclas de la columna bibliográfica)

Las métricas y baselines de §4–§5 tienen su nota propia en `docs/research/Papers/`. Estas son las anclas de accuracy por dataset, que no viven en esas notas.

**Régimen comparable (sin augmentation, desde cero, resolución nativa):**

- Bachmann, G., Anagnostidis, S. y Hofmann, T. (2023). *Scaling MLPs: A Tale of Inductive Bias.* NeurIPS 2023 (arXiv:2306.13575). MLP llano sin augmentation: CIFAR-10 54,2 %, CIFAR-100 28,8 %, Tiny-ImageNet 8,5 % (red más profunda y entrenada más tiempo que la nuestra: cota superior blanda).
- Chen, Z. y Ho, P.-H. (2018). *Deep Global-Connected Net with the Generalized Multi-Piecewise ReLU Activation.* (arXiv:1807.03116). CNN de 3 conv (16/16/32), sin BatchNorm ni dropout, sin augmentation: MNIST 99,58 %, CIFAR-10 82,77 %, CIFAR-100 55,21 % (activación GReLU: cota superior suave para ReLU llana).
- Mantzaris, A. V. (2025). *Exploring the Hierarchical Reasoning Model for Small Natural-Image Classification Without Augmentation.* (arXiv:2510.03598). CNN pequeña con BatchNorm, sin augmentation: CIFAR-10 77,2 %, CIFAR-100 45,3 % (cota inferior de la banda CNN).
- Ye, X. et al. (2024). *ZeroPur: Succinct Training-Free Adversarial Purification.* (arXiv:2406.03143). Fila «Vanilla» (sin augmentation) de ResNet-18: CIFAR-10 83,80 %, CIFAR-100 57,01 %.
- Ko, G., Kim, H. y Lee, J. (2024). *Learning Infinitesimal Generators of Continuous Symmetries from Data.* NeurIPS 2024 (arXiv:2410.21853). Columna «No-aug» de ResNet-18 en CIFAR-10: 92,4 % (con weight decay 0,4 y 200 épocas: cota optimista).
- Larsson, G., Maire, M. y Shakhnarovich, G. (2017). *FractalNet: Ultra-Deep Neural Networks Without Residuals.* ICLR 2017 (arXiv:1605.07648). ResNet sin augmentation en CIFAR-100: ~55 % (corrobora la banda).
- Simard, P. Y., Steinkraus, D. y Platt, J. C. (2003), *Best Practices for Convolutional Neural Networks* (ICDAR 2003), y Srivastava, N. et al. (2014), *Dropout* (JMLR 15). MLP llano sin augmentation en MNIST: ~98,4 %.
- LeCun, Y., Bottou, L., Bengio, Y. y Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proc. IEEE, 86(11), 2278–2324. LeNet-5 (99,05 %) y LeNet-300-100 en MNIST.

**Techo con augmentation / contexto:**

- Li, S., Liu, Z., Zang, Z., Wu, D., Chen, Z. y Li, S. Z. (2023). *GenURL: A General Framework for Unsupervised Representation Learning.* IEEE TNNLS (arXiv:2110.14553). ResNet-18 desde cero, nativo, con augmentation: CIFAR-10 94,55 %, CIFAR-100 78,07 %, Tiny-ImageNet 50,68 % (400 épocas).
- He, K., Zhang, X., Ren, S. y Sun, J. (2015). *Deep Residual Learning for Image Recognition.* CVPR 2016 (arXiv:1512.03385). Origen de ResNet y del split 45k/5k en CIFAR-10.
- Springenberg, J. T., Dosovitskiy, A., Brox, T. y Riedmiller, M. (2015). *Striving for Simplicity: The All Convolutional Net.* ICLR Workshop (arXiv:1412.6806). All-CNN ~91 % en CIFAR-10: superar el 90 % exige red convolucional profunda, no una CNN simple.
- Neyshabur, B. (2020). *Towards Learning Convolutions from Scratch.* NeurIPS 2020 (arXiv:2007.13657). β-LASSO ~85 % en CIFAR-10: red densa que aprende conectividad local, no un MLP llano.
- Lin, Z., Memisevic, R. y Konda, K. (2016). *How Far Can We Go Without Convolution.* ICLR Workshop (arXiv:1511.02580). Modelo Z-Lin ~70 % en CIFAR-10 con cuello de botella y preentrenamiento; **no** es un baseline de MLP llano (no usar como ~50–55 %).
