---
authors:
  - Stanislav Fort
  - Paweł Krzysztof Nowak
  - Stanisław Jastrzębski
  - Srini Narayanan
year: 2019
status: read
relevance: high
url: https://arxiv.org/abs/1901.09491
tfg_role:
  - metric
tfg_note: "Origen de `stiffness` (coseno/signo entre gradientes per-sample de dos ejemplos: alto si actualizar con uno mejora el otro), desagregado within/between clase. Operador base de la familia alineación; decae al empezar el overfitting."
---
# Stiffness: A New Perspective on Generalization in Neural Networks

## Summary

### Contextualización

El trabajo se sitúa en el problema central de la teoría moderna del deep learning: comprender por qué las redes neuronales sobreparametrizadas, capaces de memorizar datos arbitrariamente etiquetados (Zhang et al., 2016; Arpit et al., 2017), generalizan bien a datos no vistos cuando se entrenan con descenso por gradiente. Los autores se enmarcan dentro de la línea de investigación que estudia la similitud entre las salidas de la red para entradas próximas (Schoenholz et al., 2016; Novak et al., 2018) y que culmina en la teoría del Neural Tangent Kernel (NTK) de Jacot et al. (2018). Su propósito es proporcionar una herramienta operativa, fácil de calcular durante el entrenamiento, que permita diagnosticar cuándo la red está aprendiendo características transferibles entre ejemplos —el régimen de generalización genuina— frente a cuándo se limita a ajustarse de forma específica al conjunto de entrenamiento, es decir, cuándo está memorizando.

### Aportación

La contribución principal es la introducción del concepto de *stiffness* (rigidez) como métrica para caracterizar la generalización. La idea operativa es directa y muy visual. Dados dos ejemplos $X_1$ y $X_2$ con etiquetas $y_1$ e $y_2$, se calcula el gradiente $\vec{g}_1 = \nabla_W \mathcal{L}(f_W(X_1), y_1)$ y se aplica un paso infinitesimal en la dirección $-\vec{g}_1$ que por construcción reduce la pérdida en $X_1$. La stiffness mide qué le ocurre simultáneamente a la pérdida sobre $X_2$. Si $\Delta\mathcal{L}_2 < 0$, los dos puntos son rígidos (*positive stiffness*): mejorar uno mejora también al otro y la red está aprendiendo features compartidas. Si $\Delta\mathcal{L}_2 > 0$ son *anti-stiff* (*negative stiffness*): el ejemplo $X_2$ empeora, señal de memorización local. La equivalencia matemática clave es que esta variación coincide hasta primer orden con el producto $\vec{g}_1 \cdot \vec{g}_2$ entre gradientes evaluados en ambos puntos, lo que conecta la métrica directamente con el NTK empírico y con el estudio del Hessiano del paisaje de pérdida.

Para fijar ideas, imagínese una CNN entrenada sobre CIFAR-10 a mitad de entrenamiento: tomar dos imágenes de la clase "gato" y comprobar que un paso en la dirección de $-\vec{g}_1$ reduce la pérdida en ambas refleja que la red ha extraído un detector de orejas o de bigotes compartido. Si en cambio el paso reduce la pérdida en la imagen 1 pero la incrementa en la 2 a pesar de tener la misma etiqueta, lo que se está aprendiendo son particularidades del primer ejemplo. La stiffness convierte esta intuición en un número.

### Metodología

Los autores formalizan dos definiciones complementarias. La primera es la **sign-stiffness**,

$$S_{\mathrm{sign}}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\mathrm{sign}(\vec{g}_1 \cdot \vec{g}_2)],$$

esperanza del signo del producto de gradientes, que toma valores en $[-1,1]$ y resulta más informativa para stiffness *entre* clases, donde la magnitud importa menos que la dirección general de la transferencia. La segunda es la **cosine-stiffness**,

$$S_{\cos}((X_1,y_1),(X_2,y_2);f) = \mathbb{E}[\cos(\vec{g}_1,\vec{g}_2)],\qquad \cos(\vec{g}_1,\vec{g}_2) = \frac{\vec{g}_1}{\|\vec{g}_1\|}\cdot\frac{\vec{g}_2}{\|\vec{g}_2\|},$$

que se prefiere para stiffness *intra-clase* porque captura mejor la gradación fina del alineamiento. Por construcción, la stiffness de un punto consigo mismo vale 1.

Los autores distinguen tres regímenes de muestreo de pares: *train-train*, *train-val* y *val-val*. Observan que los tres se comportan de forma muy similar, lo que refuerza la idea de que la stiffness *train-val* —la única estrictamente ligada a la generalización en sentido clásico— puede aproximarse usando únicamente el conjunto de entrenamiento. Definen además la **matriz de class stiffness**

$$C(c_a, c_b) = \mathbb{E}_{X_1\in c_a,\, X_2\in c_b,\, X_1\neq X_2}[S((X_1,y_1),(X_2,y_2))],$$

cuyas entradas diagonales miden generalización intra-clase y las extra-diagonales la transferencia entre clases. El resumen entre clases se obtiene como

$$S_{\mathrm{between\ classes}} = \frac{1}{N_c(N_c-1)}\sum_{c_1}\sum_{c_2\neq c_1} C(c_1,c_2).$$

El protocolo experimental procede en tres pasos. Primero, se entrena la red durante un número fijo de iteraciones sobre el conjunto de train. A continuación, se congelan los pesos y, para cada modo de muestreo, se recorre una colección de tuplas calculando $\vec{g}_1$, $\vec{g}_2$, el signo y el coseno. Finalmente, se registran adicionalmente la distancia en el espacio de entrada y otras features relevantes. Los subconjuntos utilizados son fijos: aproximadamente 500 imágenes para datasets de 10 clases y 3000 para datasets de 100 clases, tamaños que los autores muestran suficientes para reducir la incertidumbre estadística. Las entradas se preprocesan con media cero, varianza unitaria y normalización a la esfera unidad ($\|\vec{X}\|=1$); el optimizador es Adam con learning rates constantes y batch por defecto de 32.

### Datasets y modelos

Los experimentos cubren cuatro datasets de visión: MNIST (LeCun y Cortes, 2010), Fashion-MNIST (Xiao et al., 2017), CIFAR-10 y CIFAR-100 (Krizhevsky, 2009). Las arquitecturas son una red fully-connected ReLU de tres capas ocultas, $X \to 500 \to 300 \to 100 \to y$; una CNN de tres capas con filtros $3\times3$ y canales $16, 32, 32$ seguidos de max-pooling $2\times2$ y una capa final FC; y una ResNet20v1 (He et al., 2015) en la implementación de Chollet et al. (2015), notablemente *sin* batch normalization. Adicionalmente, los autores validan los hallazgos sobre una tarea de NLP fine-tuneando BERT (Devlin et al., 2018) sobre MNLI (Williams et al., 2017), lo que sugiere que el fenómeno no se restringe a visión.

### Métricas

Las magnitudes seguidas durante los experimentos son cuatro. Por una parte, los autores registran la loss y la accuracy en train y validación a lo largo de las épocas, lo que sirve para localizar el inicio del overfitting. Por otra, miden los elementos diagonales y extra-diagonales de la matriz de class stiffness, tanto en su variante de signo como de coseno. Adicionalmente, dibujan curvas de stiffness en función de la distancia en el espacio de entrada, donde la distancia se define como

$$\mathrm{distance}(\vec{X}_1,\vec{X}_2) = 1 - \frac{\vec{X}_1\cdot\vec{X}_2}{\|\vec{X}_1\|\,\|\vec{X}_2\|},$$

acotada en $[0,2]$. Por último, introducen la **dynamic critical length** $\xi$, definida como la distancia umbral a la cual la stiffness intra-clase cruza el cero según un ajuste lineal a la curva stiffness-distancia. Esta longitud caracteriza el tamaño típico de los parches del espacio de entrada que se mueven juntos bajo un paso de gradiente y conecta directamente con la noción de "regiones rígidas" de la función aprendida: una $\xi$ grande implica que dos ejemplos relativamente alejados aún experimentan transferencia positiva, mientras que una $\xi$ pequeña indica que la red sólo generaliza en vecindades estrechas.

### Conclusiones

Los resultados empíricos articulan varias conclusiones. En primer lugar, la stiffness funciona como herramienta diagnóstica del overfitting. En estados iniciales del entrenamiento la stiffness intra-clase es alta y la stiffness entre clases crece a medida que la red aprende features compartidas; cuando arranca el overfitting (marcado en la Figura 3 con la zona naranja), tanto la stiffness within-class como between-classes regresan hacia 0, indicando que las actualizaciones de gradiente sobre un ejemplo dejan de beneficiar incluso a los demás miembros de su misma clase. Esto sugiere que la métrica podría usarse como criterio de early stopping observable únicamente sobre el train set, sin necesidad de un val set.

En segundo lugar, la stiffness es sensible al contenido semántico. En CIFAR-100 la matriz de class stiffness exhibe estructura de *coarse-grain* alineada con super-clases (grupos de cinco clases semánticamente conectadas, como los mamíferos pequeños o los vehículos terrestres) e incluso super-super-clases (la distinción living/non-living). Esto demuestra que la red captura jerarquías semánticas más allá del nivel de etiqueta usado en el entrenamiento. Como ejemplo concreto, a modo de ejemplo esperable, las entradas $C(\text{leopard}, \text{tiger})$ aparecen marcadamente más altas que $C(\text{leopard}, \text{truck})$, sin que el modelo haya visto nunca la etiqueta "felino".

En tercer lugar, la dynamic critical length $\xi$ depende sistemáticamente del learning rate: learning rates mayores producen $\xi$ menores, es decir, funciones aprendidas localmente más maleables (menor stiffness) y más fáciles de doblar mediante actualizaciones de gradiente, incluso cuando logran accuracy comparable. Este hallazgo apunta a un rol regularizador del learning rate sobre el *tipo* de función aprendida y no solo sobre la velocidad de convergencia.

Por último, las observaciones sobre BERT-MNLI replican el patrón visto en visión, lo que sugiere que stiffness es un fenómeno general y no un artefacto de las arquitecturas convolucionales. Los autores conjeturan como dirección futura su uso como parámetro guía para meta-learning y neural architecture search, ya que arquitecturas con sesgos inductivos como la localidad de las CNN se traducen sistemáticamente en propiedades de stiffness más altas.

## Medición y pipeline

**Métrica concreta.** Se adopta la stiffness de Fort et al. en sus dos formulaciones complementarias, sobre un conjunto de pares de ejemplos $(x_i, x_j)$ con $i \neq j$:

$$S_{\mathrm{sign}} = \mathbb{E}_{i\neq j}\big[\mathrm{sign}(\nabla_W \ell_i \cdot \nabla_W \ell_j)\big],\qquad S_{\cos} = \mathbb{E}_{i\neq j}\Big[\tfrac{\nabla_W \ell_i \cdot \nabla_W \ell_j}{\|\nabla_W \ell_i\|\,\|\nabla_W \ell_j\|}\Big].$$

La variante de signo es más informativa para la stiffness *between-class*, mientras que la de coseno es preferible para la *within-class*. Siguiendo el paper, se distinguen ambas: within-class restringe la esperanza a pares con $y_i = y_j$ y captura generalización intra-clase; between-class restringe a $y_i \neq y_j$ y mide transferencia entre clases.

**Entradas.** La métrica consume gradientes per-sample $\nabla_W \ell(x_k; W)$ calculados sobre un *probe set* fijo y disjunto del batch de entrenamiento. En la práctica del TFG se usará un probe estratificado por clase, muestreado una sola vez al inicio del run y reutilizado en cada medición para que la varianza temporal refleje cambios reales en el modelo y no resampling. Tamaños sugeridos en línea con el paper: $N \approx 500$ ejemplos para datasets de 10 clases (MNIST, Fashion-MNIST, CIFAR-10). Para el TFG el default operativo es $M = 256$, del mismo bloque de per-sample grads que `m_coherence` y `gsnr` (cada uno con su propio $M$; ver Notes).

**Granularidad temporal.** Una medición por época durante toda la corrida. En la fase inicial —las primeras 5–10 épocas, que es donde Fort et al. muestran la señal más predictiva— se densifica el muestreo a cada $K$ pasos (típicamente $K=50$ o $K=100$ en CIFAR-10) para resolver bien la ventana temprana. Para reporting en runs largos basta con loguear en los hitos 5%, 10%, 25%, 50% y 100% de épocas, que es la convención del comparador de generalización del TFG.

**Granularidad estructural.** Por defecto, en ResNet-18 se computa la stiffness *last-layer-only*: se restringen los gradientes a los parámetros del clasificador lineal final ($P_{\text{eff}} \approx 5\text{k}$ frente a $11.7\text{M}$ del modelo completo), consistente con la elección de `gwa`. Fort et al. reportan stiffness por bloques de la ResNet, lo que avala teóricamente esta restricción. En MLP y CNN pequeña se computa *full-parameter* porque $P$ es manejable. Además del escalar agregado, se loguea la matriz $C(c_a, c_b)$ de class stiffness, que aporta mucha más información (estructura semántica, jerarquías de super-clase).

**Coste y memoria.** El cuello no está en el cómputo escalar de cosenos —que es $O(N^2 P_{\text{eff}})$ pero perfectamente vectorizable como una multiplicación de matrices— sino en almacenar la matriz de gradientes $G \in \mathbb{R}^{N\times P_{\text{eff}}}$. Como referencia: para $N = 256$ y ResNet-18 *full-parameter* ($P \approx 11.7\text{M}$, fp32) la matriz pesa $256 \times 11.7\text{M} \times 4 \text{ B} \approx 12\,\text{GB}$, lo que es bloqueante en una GPU de consumo. Con la restricción a last-layer ($P_{\text{eff}} = 5\text{k}$) baja a $\approx 5\,\text{MB}$, completamente manejable. En MLP/CNN-small con $P \approx 200\text{k}$ y $N = 256$ son $\approx 200\,\text{MB}$, también factible.

**Trucos.** El bucle ingenuo de un forward+backward por ejemplo es lento; conviene usar `torch.func.vmap` o `functorch` para vectorizar los per-sample grads en un solo paso. Si por algún motivo se necesita el cómputo full-parameter en una red grande, las dos mitigaciones estándar son procesar $G$ en chunks acumulando submatrices de cosenos en lugar de la matriz completa, o aplicar una proyección aleatoria $\Phi \in \mathbb{R}^{P\times d}$ con $d \ll P$ que preserva productos internos hasta un factor multiplicativo (Johnson–Lindenstrauss). Una tercera opción es restringir a un subespacio fijo (por capa o por bloque), que es la elección por defecto. El cálculo de $S_{\cos}$ se hace normalizando filas de $G$ y multiplicando $G_n G_n^\top$; el de $S_{\mathrm{sign}}$ aprovecha que el signo solo depende del producto sin normalizar, así que puede leerse de $G G^\top$.

**Claves de log.** Por época se registran como mínimo los escalares `stiffness/cos_within`, `stiffness/cos_between`, `stiffness/sign_within`, `stiffness/sign_between` y los promedios globales `stiffness/cos_global`, `stiffness/sign_global`. Cada $K$ épocas (por ejemplo cada 5) se vuelca la matriz $C$ completa como heatmap para inspección cualitativa, bajo la clave `stiffness/class_matrix`. Si se computa también la diagonal/extra-diagonal por clase, conviene loguearlas como vectores `stiffness/diag_per_class` y `stiffness/offdiag_per_class` para detectar clases que pierden cohesión antes que las demás (señal precoz de overfitting localizado).

**Interpretación de la señal.** Conviene fijar la convención antes de leer las claves, porque la stiffness tiene matices que no se reducen a un único "más alto = mejor". La regla básica es que coseno y signo entre gradientes per-sample alineados indican que el SGD transfiere mejoras entre los ejemplos comparados, así que en `stiffness/cos_within` y `stiffness/sign_within` la lectura natural es **mayor = mejor**: pares de la misma clase con $\cos(g_i,g_j)$ alto significan que la red ha extraído features compartidas y cada paso de gradiente sobre un ejemplo beneficia al resto de su clase. En `stiffness/cos_between` y `stiffness/sign_between` la lectura saludable no es maximizar, sino que se mantengan **cercanos a $0$ y claramente por debajo de la within**: between alta cercana a within indica que el modelo no discrimina las clases, mientras que between ligeramente positiva es esperable porque siempre hay features de bajo nivel compartidos. El indicador robusto es por tanto el *gap* `cos_within - cos_between`, que cuanto mayor es más cohesión intra-clase con separación entre clases refleja. Los promedios globales `stiffness/cos_global` y `stiffness/sign_global` son un resumen menos informativo y se usan sobre todo para inspección rápida. La variante de signo $S_{\mathrm{sign}}$ es más robusta a la magnitud de los gradientes y por eso se prefiere para la lectura between, donde lo que importa es la dirección general de la transferencia; la variante coseno $S_{\cos}$ captura la gradación fina y es preferible para la within. El matiz crítico, y el que diferencia a esta métrica del resto del bloque de alineación, es **temporal**: Fort et al. observan que la stiffness decae hacia $0$ cuando la red entra en overfitting, así que la lectura "↑ within = mejor" solo aplica durante la **fase de fit**, esto es, las primeras épocas y hasta el codo train/val. Una stiffness baja al final del entrenamiento no es necesariamente "peor optimización": es la firma esperable de la transición a memorización. Operativamente, la señal predictiva del proxy vive en la ventana temprana (hitos 5%, 10%, 25%) y conviene evaluarla ahí; medir solo al final no es informativo y puede invertir la interpretación.

**Gotchas.** Hay varios. Primero, la stiffness decae a 0 cuando la red entra en overfitting, así que la *ventana temprana* es la que carga la señal predictiva; medir solo al final del entrenamiento no es informativo. Segundo, el preprocesado importa: normalizar las entradas a la esfera unidad cambia tanto las magnitudes de gradiente como las distancias en input space, y compararse con los números del paper exige replicarlo. Tercero, la BN introduce dependencias entre ejemplos del mismo forward, lo que rompe la noción de gradiente per-sample; o bien se evalúa en modo `eval()` con BN congelada (la opción del paper, que entrena ResNet20 sin BN), o se desacopla cuidadosamente. Cuarto, los pares $i=i$ aportan $\cos = 1$ y sesgan el promedio si no se enmascaran (la máscara `~eye(N)` no es opcional). Quinto, en problemas con $N_c$ clases muy desbalanceadas, conviene estratificar el probe set para que todos los pares within-class tengan soporte razonable.

**Pseudocódigo PyTorch.**

```python
from torch.func import functional_call, vmap, grad

def per_sample_grads(model, params, x, y, loss_fn):
    def loss_one(p, xi, yi):
        out = functional_call(model, p, (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))
    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)

def stiffness_step(model, probe_X, probe_y, loss_fn, last_layer_only=True):
    params = {k: v.detach() for k, v in model.named_parameters()
              if (not last_layer_only) or k.startswith("fc.")}
    grads = per_sample_grads(model, params, probe_X, probe_y, loss_fn)
    G = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)
    mask = ~torch.eye(G.size(0), dtype=torch.bool, device=G.device)
    Gn = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C_cos  = Gn @ Gn.T
    C_sign = (G @ G.T).sign()
    s_cos_global  = C_cos[mask].mean()
    s_sign_global = C_sign[mask].mean()
    same = probe_y.unsqueeze(0) == probe_y.unsqueeze(1)
    s_cos_within  = C_cos[mask & same].mean()
    s_cos_between = C_cos[mask & ~same].mean()
    return dict(cos_global=s_cos_global, sign_global=s_sign_global,
                cos_within=s_cos_within, cos_between=s_cos_between, C=C_cos)
```

**Descarte explícito.** En el TFG no se implementa el análisis "stiffness vs distancia en el input" ni la dynamic critical length $\xi$, porque añaden una dimensión continua (la distancia 2D entrada-entrada) fuera del scope puramente correlacional del trabajo. La estructura semántica se cubre suficientemente con la matriz $C$.

## Notes

### Uso en el TFG

La stiffness es el origen de la métrica `stiffness` dentro de la familia de alineación implementada en `src/`. Sobre gradientes per-sample del gradiente bruto $g_i = \nabla_W \ell(f_W(x_i), y_i)$ se computan la cos-stiffness $S_{\cos}(i,j) = \cos(g_i, g_j) = g_i \cdot g_j / (\|g_i\|\,\|g_j\|)$ y la sign-stiffness $S_{\text{sign}}(i,j) = \text{sign}(g_i \cdot g_j)$.

La agregación se hace sobre un probe estratificado de $M = 256$ ejemplos, distinguiendo pares within-class (mismo label) y between-class (distinto label) para reportar $\bar S_{\cos}^{\text{within}} = \mathbb{E}_{i\neq j,\, y_i=y_j}[S_{\cos}(i,j)]$ y $\bar S_{\cos}^{\text{between}} = \mathbb{E}_{i\neq j,\, y_i\neq y_j}[S_{\cos}(i,j)]$. La señal predictiva vive en la ventana temprana (5%, 10%, 25%, 50% de épocas): la stiffness decae hacia 0 al iniciarse el overfitting, así que es la fase temprana —no el final— la que carga la información.

En cuanto a interpretación de la señal, $\bar S_{\cos}^{\text{within}}$ alto es mejor (clases con features compartidos y SGD que transfiere mejoras intra-clase), $\bar S_{\cos}^{\text{between}} \approx 0$ es esperable, y se reportan también los promedios globales `cos` y `sign`. El cuello operativo es la gram $GG^\top$, con coste de memoria $M \cdot P$ floats que en ResNet-18 fp32 con $M=256$ asciende a $\sim 12\,\text{GB}$ —bloqueante sin mitigación—. La solución por defecto es last-layer-only en ResNet-18 ($P_{\text{eff}} \approx 5\text{k}$ en lugar de $11.7\text{M}$), consistente con `gwa`; en FC/MLP y CNN-small se mantiene full-parameter. El sweep per-sample $\nabla L$ se comparte con `m_coherence` ($M=1024$) y `gsnr` ($M=512$). No se implementa "stiffness vs distancia" ni la dynamic critical length $\xi$.

## Discrepancias detectadas

Se documentan tres discrepancias menores entre el paper y el resumen `metrics.md`, todas marginales:

La primera es de notación: el paper usa $|\vec{g}|$ para la norma euclídea, mientras que en este archivo y en `metrics.md` se ha unificado a $\|\vec{g}\|$, más estándar. No hay impacto matemático.

La segunda concierne al modo de muestreo *train-val*. El paper afirma que train-train, train-val y val-val se comportan de forma muy similar y por eso opta por reportar mayoritariamente train-train en sus figuras. En el TFG se sigue una elección equivalente —probe estratificado disjunto del batch de training—, pero se etiqueta como "val-val" cuando el probe proviene del split de validación. La métrica numérica es la misma; conviene dejar claro en el código qué split se está usando para el probe.

La tercera es la elección de $N$ en el probe. El paper recomienda $N \approx 500$ para 10 clases, mientras que el default del TFG es $M = 256$ por consistencia con `m_coherence` y `gsnr` y por presión de memoria. Esta reducción aumenta la varianza estadística del estimador, especialmente en la diagonal de $C$ donde cada clase contribuye con $\binom{N/N_c}{2}$ pares; conviene tenerlo presente al interpretar variaciones pequeñas entre épocas.

## Cited By
[[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]]
[[Speedy Performance Estimation for Neural Architecture Search]]
[[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]]
[[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]]
[[Disparity Between Batches as a Signal for Early Stopping]]

## Papers relacionados

- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — Misma familia alineación; m-coherence $\alpha_m$ es el agregado escalar del mismo producto $g_z\cdot g_{z'}$ entre per-sample grads y comparte el sweep $\nabla L$ con stiffness.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — Cita a stiffness; la CGH formaliza el mismo principio (la generalización viene del alineamiento entre gradientes per-sample) que aquí se mide vía within/between-class.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — Cita a stiffness; gradient confusion es el peor caso (mín. coseno) del mismo objeto coseno entre gradientes per-sample/per-batch.
- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — Cita a stiffness; proxy train-time de generalización con la misma decisión de last-layer-only para acotar memoria.
- [[Disparity Between Batches as a Signal for Early Stopping]] — Cita a stiffness; usa alineamiento (distancia L2) entre gradientes como señal de early stopping, mismo régimen de ventana temprana.
- [[Speedy Performance Estimation for Neural Architecture Search]] — Cita a stiffness y comparte la motivación NAS/meta-learning que los autores conjeturan como uso futuro de la métrica.
- [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training]] — La equivalencia stiffness $\leftrightarrow g_i\cdot g_j$ es exactamente el NTK empírico; este paper formaliza el kernel alignment al que stiffness apunta.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — Familia alineación/varianza; comparte el sweep per-sample $\nabla L$ con stiffness y conecta alineamiento de gradientes con generalización.

## Otros papers interesantes a revisar

- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** (Jacot, Gabriel & Hongler, 2018) — Marco NTK que fundamenta la equivalencia stiffness = $g_i\cdot g_j$; cita central del paper. arXiv:1806.07572.
- **Sensitivity and Generalization in Neural Networks: an Empirical Study** (Novak et al., 2018) — Línea directa de la que parte Fort (sensibilidad de salidas a entradas próximas); evidencia empírica de la conexión sensibilidad-generalización. arXiv:1802.08760.
- **A Closer Look at Memorization in Deep Networks** (Arpit et al., 2017) — Caracteriza memorización vs aprendizaje de patrones; complementa la interpretación de stiffness baja como memorización local. arXiv:1706.05394.
- **Gradient Descent Happens in a Tiny Subspace** (Gur-Ari, Roberts & Dyer, 2018) — El gradiente vive en un subespacio de baja dimensión dominado por el Hessiano; relevante para justificar last-layer-only y proyecciones al acotar $M\cdot P$. arXiv:1812.04754.
- **What Can Linearized Neural Networks Actually Say About Generalization?** (Ortiz-Jiménez, Moosavi-Dezfooli & Frossard, 2021) — Examina cuándo el NTK empírico predice generalización; matiza el alcance de proxies basados en alineamiento de gradientes como stiffness. arXiv:2106.06770.
