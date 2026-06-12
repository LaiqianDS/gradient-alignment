---
authors:
  - Fartash Faghri
  - David Duvenaud
  - David J. Fleet
  - Jimmy Ba
year: 2020
status: read
relevance: high
url: https://arxiv.org/pdf/2007.04532
tfg_role:
  - metric
  - baseline
tfg_note: "Origen de `normalized_variance` (NGV: varianza del gradiente relativa a su media, inverso del SNR). Eje varianza; hallazgo contraintuitivo clave: la NGV crece durante el entrenamiento en CIFAR/ImageNet en vez de decrecer."
---

# A Study of Gradient Variance in Deep Learning

## Summary

### Contextualización

El paper se sitúa en el problema clásico de la optimización estocástica en deep learning. Cuando se entrena un modelo con mini-batch SGD, el gradiente estocástico es un estimador insesgado del gradiente medio sobre el conjunto de entrenamiento, y la calidad de ese estimador, cuantificada por su varianza, condiciona tanto la velocidad de convergencia como la estabilidad del entrenamiento. La literatura previa (Bottou et al., 2018; Robbins y Monro, 1951) demuestra cotas de convergencia que mejoran a medida que la varianza disminuye y predicen una mejora lineal con el tamaño del mini-batch. Sin embargo, en la práctica trabajos como Shallue et al. (2018) o Zhang et al. (2019) observan rendimientos decrecientes (*diminishing returns*): a partir de cierto umbral, doblar el batch ya no acelera el entrenamiento.

La hipótesis habitual atribuye este fenómeno a los beneficios del ruido para encontrar *flat minima* y mejorar la generalización (Goodfellow y Vinyals, 2015; Keskar et al., 2017), pero esa explicación no aclara por qué los rendimientos decrecientes también aparecen en la *training loss*. Faghri et al. argumentan que la varianza del gradiente como estadístico está infraestudiada empíricamente en deep learning real y proponen abordarla de forma sistemática. El hueco que llena el paper es doble: por un lado, caracterizar empíricamente la distribución de gradientes durante el entrenamiento de redes profundas; por otro, proponer un estadístico mejor que la varianza cruda para correlacionar con la dificultad de optimización.

### Aportación

Las contribuciones principales son cuatro. La primera es una **vista distribucional del gradiente**: en lugar de tratar el mini-batch gradient solo como estimador puntual de la media, lo modelan como muestreo de una distribución potencialmente estructurada y multimodal. La segunda es el **Gradient Clustering (GC)**, un método de muestreo estratificado que demuestra teóricamente reducir la varianza del estimador del gradiente medio si los datos se agrupan en clusters ponderados en el espacio de gradientes (Proposición 3.1). La tercera es una **implementación eficiente de GC** mediante aproximaciones rank-1 por capa, evitando calcular gradientes individuales y manteniendo el coste computacional dentro de un factor moderado del back-prop. La cuarta, la más relevante para este TFG, es la introducción de la **Normalized Variance** como estadístico alternativo: un escalar que correlaciona mejor con la velocidad de convergencia que la varianza absoluta y que sí es comparable entre problemas de distinta escala.

Empíricamente, el hallazgo más sorprendente es que, contrariamente al supuesto común, la varianza del gradiente **aumenta** durante el entrenamiento en CIFAR-10 e ImageNet, y que learning rates más pequeños coinciden con varianza más alta.

### Metodología

Sea un conjunto de entrenamiento $\{x_i\}_{i=1}^N$ y los gradientes por muestra $g_i = \frac{\partial}{\partial \theta}\ell(x_i; \theta)$. Para una partición en $K$ subconjuntos de tamaño $N_k$, el estimador de gradiente con muestreo estratificado es

$$\hat{g}(a) = \frac{1}{N}\sum_{k=1}^{K} N_k \, g_{j,k}, \quad j \sim \mathbb{U}[1, N_k].$$

La **Proposición 3.1** demuestra que este estimador es insesgado ($\mathbb{E}[\hat{g}] = g$) y que su varianza es $\mathbb{V}[\hat{g}] = N^{-2}\sum_{k=1}^{K} N_k^2 \mathbb{V}[g_{j,k}]$. Minimizar esta expresión equivale a un problema de clustering ponderado:

$$\min_{C, a} \sum_{k=1}^{K}\sum_{i=1}^{N} N_k \, \|C_k - g_i\|^2 \, \mathbb{I}(a_i = k),$$

donde $C_k$ es el centro del cluster $k$. El factor $N_k$ extra, frente a K-Means estándar, penaliza clusters grandes con datos dispersos. El algoritmo alterna entre un paso de asignación ($\mathcal{A}$: $a_i = \arg\min_k N_k \|C_k - g_i\|^2$) y un paso de actualización ($\mathcal{U}$: $C_k = \frac{1}{N_k}\sum_i g_i \mathbb{I}(a_i=k)$), de forma análoga al algoritmo de Lloyd pero con asignaciones duras.

El reto computacional es que materializar gradientes por muestra resulta prohibitivo, y la solución es una aproximación rank-1 por capa. Para una capa fully-connected con $\theta \in \mathbb{R}^{I \times O}$, el gradiente por muestra factoriza como $g = AD^\top$ con $A \in \mathbb{R}^{I \times N}$ (activaciones de entrada) y $D \in \mathbb{R}^{O \times N}$ (gradientes respecto a las salidas). Cada centro de cluster se aproxima como $C_k = c_k d_k^\top$ con $c_k \in \mathbb{R}^I$, $d_k \in \mathbb{R}^O$. Bajo el supuesto $\mathbb{E}[A_i D_i] = \mathbb{E}[A_i]\mathbb{E}[D_i]$, similar al de K-FAC (Martens y Grosse, 2015), las actualizaciones cerradas son $c_k = \frac{1}{N_k}\sum_i A_i \mathbb{I}(a_i=k)$ y $d_k = \frac{1}{N_k}\sum_i D_i \mathbb{I}(a_i=k)$. La extensión a capas convolucionales se desarrolla en el apéndice B.1. El paso $\mathcal{A}$ se ejecuta cada pocas épocas; $\mathcal{U}$ puede actualizarse online con mini-batches. El overhead total es a lo sumo $2\times$ el coste del back-prop estándar.

### Datasets y modelos

Los experimentos cubren benchmarks de visión y un modelo teórico. En **MNIST** (LeCun et al., 1998) usan un MLP de tres capas fully-connected ($784 \to 1024 \to 1024 \to 10$) con ReLU, sin dropout, learning rate $0.02$, weight decay $5\times10^{-4}$ y momentum $0.5$. En **CIFAR-10** (Krizhevsky et al., 2009) entrenan una **ResNet8** sin batch normalization con learning rate $0.01$, weight decay $5\times10^{-4}$, momentum $0.9$ y $80\,000$ iteraciones, con decay del LR en las iteraciones $40\,000$ y $60\,000$ por factor $0.1$. En **CIFAR-100** usan **ResNet32** con learning rate inicial $0.1$ y el resto de hiperparámetros como CIFAR-10. En **ImageNet** (Deng et al., 2009) entrenan una **ResNet18** con learning rate $0.1$, weight decay $1\times10^{-4}$, momentum $0.9$ y un schedule de decay similar al de CIFAR-10. Por último, los **Random Features (RF) models** (Rahimi y Recht, 2007) emplean activación ReLU, dimensión oculta fija $h_s = 1000$ y número variable de muestras de entrenamiento de modo que el ratio de sobreparametrización recorra $h_s/N \in [0.1, 10]$; el dataset sintético $\{(x_i, y_i)\} \in \mathbb{R}^I \times \{\pm 1\}$ se genera con un modelo "teacher" con pesos $\theta_1 \in \mathbb{R}^{I \times h_t}$, $\theta_2 \in \mathbb{R}^{h_t \times 1}$ y bias $b$, entrenando con cross-entropy y mini-batch de tamaño 10.

Los baselines son SG-B (mini-batch SGD), SG-2B (mini-batch del doble de tamaño, que reduce varianza por factor 2 al doble de coste), **SVRG** (Johnson y Zhang, 2013) y **GC** (el método propuesto). El tamaño de mini-batch es $B=128$ en todos los benchmarks de imagen.

### Métricas

El paper define dos métricas centrales sobre snapshots fijos del modelo, calculando estadísticos a partir de decenas de mini-batches muestreados. La **Average Variance** es el promedio sobre las coordenadas del parámetro de la varianza del estimador del gradiente medio, equivalente a la traza normalizada de la matriz de covarianza. La **Normalized Variance**, $\mathbb{V}[g] / \mathbb{E}[g]^2$, se interpreta como el inverso de un *signal-to-noise ratio* (SNR): si supera 1, el ruido domina sobre la señal del gradiente. Como referencia de magnitudes empíricas: la Average Variance toma valores en torno a $10^{-4}$ en CIFAR-10 y por debajo de $10^{-6}$ en ImageNet, mientras que en MNIST decae hasta $\sim 10^{-8}$. La Normalized Variance, al ser adimensional, sí es comparable entre estos regímenes.

También reportan **training loss**, **accuracy** y la varianza máxima sobre las últimas iteraciones (en los RF models). Las curvas se presentan frente al número de iteraciones, no en wall-clock time.

### Conclusiones

Los hallazgos principales pueden resumirse así. Primero, la varianza absoluta no es comparable entre problemas (CIFAR-10 alcanza $\sim 10^{-4}$, ImageNet $< 10^{-6}$), pero la varianza normalizada sí lo es: en MNIST decrece monotónicamente, mientras que en CIFAR-10 e ImageNet crece durante el entrenamiento, especialmente tras los drops del learning rate. Segundo, la varianza normalizada correlaciona mejor con la velocidad de convergencia que la varianza cruda, lo que confirma su utilidad como estadístico diagnóstico. Tercero, la *Strong Growth Condition* (Schmidt y Le Roux, 2013) parece cumplirse en MNIST (varianza tendiendo a cero), lo que explica por qué SVRG funciona bien allí pero falla en deep learning real. Cuarto, en CIFAR-10 GC reduce la varianza especialmente cuando hay duplicados o etiquetas corruptas (corrupción del 10%), respaldando la motivación original; sin embargo, en ImageNet GC se solapa con SG-B, lo que sugiere que la distribución de gradientes allí carece de estructura clusterizada explotable. Quinto, en RF models con $h_s/N$ alto, todos los métodos tienen varianza similar y baja, y GC pierde su ventaja, mientras que SVRG colapsa en régimen sobreparametrizado. Sexto, learning rates pequeños coinciden con varianza más alta, lo cual contradice asunciones simplistas y sugiere que la dependencia del ruido sobre los parámetros actuales es no trivial. Y séptimo, la caída inmediata de la loss tras un *learning rate drop* no se explica únicamente por reducción de varianza.

Los autores reconocen como limitaciones que algunas contribuciones son empíricas y carecen de respaldo teórico cerrado, que GC no siempre mejora la optimización pese a reducir la varianza, que en ImageNet la estructura clusterizada parece ausente o residir en subespacios no capturados por la aproximación rank-1, y que explotar la varianza para mejorar la optimización sigue siendo un problema abierto. La implicación del trabajo es que el diseño de optimización *distribution-aware* podría beneficiarse de modelar la distribución completa del gradiente y no solo su media, y que la varianza normalizada es una métrica útil como diagnóstico de la dificultad de optimización.

## Medición y pipeline

**Métrica concreta.** El paper define dos estadísticos que conviene tratar por separado. La **Average Variance** es la traza normalizada de la matriz de covarianza del estimador del gradiente medio, $\overline{\mathbb{V}} = \frac{1}{d}\sum_{j=1}^d \mathbb{V}[g_j]$, una varianza absoluta agregada coordenada a coordenada. La **Normalized Variance** del paper es, literalmente, el cociente por coordenada $\mathbb{V}[g_j] / \mathbb{E}[g_j^2]$ — segundo momento **no central** en el denominador (p. 5, §4: *"divided by its second non-central moment"*) —, promediado sobre coordenadas y acotado en torno a 1 (por eso satura en ImageNet en las Figs. 3d–3f). La forma que el TFG adopta como NGV, $\text{tr}(\text{Cov}(g)) / \|\mathbb{E}[g]\|^2$ (estilo McCandlish), es una **adaptación deliberada**: ambas están relacionadas monótonamente ($NV = \text{NGV}/(1+\text{NGV})$ bajo agregación suma/suma), de modo que se conservan los rankings. El propio paper es internamente inconsistente — escribe $V[g]/E[g^2]$ pero glosa "(normalized gradient larger than one)" —, lo que explica la confusión de versiones previas de estas notas. La varianza normalizada, y no la absoluta, es la que el paper destaca como comparable entre problemas de distinta escala.

**Entradas.** Para un peso $w_t$ fijo se muestrean $K$ mini-batches independientes y se calculan los gradientes $g_1, \dots, g_K$. El protocolo del paper es "sample tens of mini-batches" (App. C), con $T = 50$ estimaciones en MNIST/CIFAR-10/CIFAR-100 y $T = 10$ en ImageNet (Tabla 2); el pipeline del TFG usa $K = 10$ sub-batches disjuntos del probe. La estadística se computa **coordenada a coordenada**: para cada parámetro $j$ se acumulan $\mathbb{E}[g_j]$ y $\mathbb{V}[g_j]$ sobre los $K$ batches, y luego se agrega (suma sobre $j$) para obtener el escalar global. La versión **por capa** es una decisión propia del TFG, no una recomendación del paper (lo per-layer del paper es la aproximación rank-1 de Gradient Clustering, §3.2); permite localizar la fuente del ruido cuando ciertas capas dominan la covarianza.

**Rangos típicos e interpretación.** La Average Variance, al ser absoluta, no es comparable entre escalas: $\sim 10^{-4}$ en CIFAR-10, $< 10^{-6}$ en ImageNet, $\sim 10^{-8}$ en MNIST tardío. La NGV sí es comparable: valores por debajo de 1 indican que la señal del gradiente medio domina sobre el ruido, valores por encima de 1 que el ruido domina y un paso individual de SGD es esencialmente direccional-ruido. Ojo: este umbral 1 solo es coherente con la forma adaptada del TFG ($\text{tr}(\text{Cov})/\|\mathbb{E}[g]\|^2$); bajo la fórmula literal del paper ($V/E[g^2]$, acotada ≈1) el valor poblacional nunca lo supera de forma significativa. Una NGV próxima a 1 es la frontera donde el SNR cae a la unidad y donde, esperablemente, aumentar el batch size deja de compensar (es la región conectada con el *gradient noise scale* de McCandlish et al.).

**Granularidad temporal.** Por época durante toda la corrida, o cada $N$ pasos en regímenes largos (típicamente $N = 500$–$2000$ en CIFAR/ImageNet). Conviene **no** computar la métrica en cada iteración: cada medición congela $w_t$ y consume $K$ pasadas backward adicionales sin actualizar pesos.

**Granularidad estructural.** Loguear NGV global (un escalar por época) y NGV por capa (un escalar por capa y época). En la práctica las primeras capas convolucionales y las capas finales (fully-connected del clasificador) suelen presentar regímenes muy distintos: agregar todo en un único escalar puede ocultar inversiones de tendencia.

**Coste y memoria.** Coste de cómputo: $K$ pasadas backward por medición, sin paso del optimizer; con $K = 30$ y medición cada época en CIFAR-10, equivale aproximadamente a un overhead del 3–5% del coste total de entrenamiento. Memoria: con acumuladores streaming basta con almacenar dos buffers del tamaño del modelo (suma de gradientes y suma de cuadrados); orden de magnitud $\mathcal{O}(d)$ floats, donde $d$ es el número de parámetros. **No** es necesario almacenar los $K$ vectores de gradiente completos: $K \cdot d$ floats sería prohibitivo en ResNet-18 fp32 ($\approx 11.7$M × $K = 30 \approx 1.4$ GB) y se evita por completo con streaming.

**Trucos numéricos.** Acumulación **Welford streaming** para la varianza coordenada a coordenada cuando se quiere precisión numérica (evita el cancelamiento catastrófico de $\mathbb{E}[g^2] - \mathbb{E}[g]^2$ con flotantes). Para el agregado global basta con dos acumuladores $S = \sum_k g_k$ y $Q = \sum_k \|g_k\|^2$, y luego (forma insesgada de Bessel, $\div(K-1)$) $\text{tr}(\text{Cov}) = (Q - \|S\|^2/K)/(K-1)$. Se recomienda fp32 (o fp64 si el modelo está en bf16/fp16) para los acumuladores.

**Claves de log.** Se sugiere la siguiente convención de nombres (compatible con TensorBoard / wandb):

- `var/avg` — Average Variance global (varianza absoluta agregada, $\frac{1}{d}\sum_j \mathbb{V}[g_j]$).
- `var/normalized` — NGV global ($\text{tr}(\text{Cov}) / \|\mathbb{E}[g]\|^2$).
- `var/per_layer/{name}` — NGV por tensor de parámetros, donde `{name}` es el nombre devuelto por `named_parameters()`, esto es, una entrada por tensor de peso/bias (`layer1.0.conv1.weight`, `fc.bias`, etc.), no una por módulo.
- `var/grad_norm_hist` (opcional) — histograma de $\|g_k\|$ sobre los $K$ batches, útil para diagnosticar colas pesadas en la distribución de gradientes.

**Interpretación de la señal.** Conviene fijar la convención por clave porque la lectura "menor es mejor" del registro no se traduce sin matices en una métrica que empíricamente **crece** durante el entrenamiento. En `var/avg` (Average Variance, $\frac{1}{d}\sum_j \mathbb{V}[g_j]$) la convención formal es **cuanto más bajo, mejor** dentro de un mismo problema: una varianza coordenada agregada baja indica que el estimador del gradiente medio es preciso y que un paso de SGD avanza en la dirección de descenso esperada; sin embargo, la magnitud absoluta **no** es comparable entre datasets ($\sim 10^{-4}$ en CIFAR-10, $< 10^{-6}$ en ImageNet, $\sim 10^{-8}$ en MNIST tardío) y solo tiene sentido leerla relativamente a su propia trayectoria. En `var/normalized` (NGV global, $\text{tr}(\text{Cov})/\|\mathbb{E}[g]\|^2$) la convención teórica es la misma — **menor = mejor**, con NGV $<1$ indicando que la señal del gradiente medio domina sobre el ruido y NGV $\gtrsim 1$ que un paso individual es esencialmente direccional-ruido —, pero la lectura operativa requiere cuidado: el paper documenta que la NGV **crece** monotónicamente en CIFAR-10/ImageNet a lo largo del entrenamiento y sube tras los LR drops, así que un aumento sostenido **no** debe interpretarse como degradación de la optimización ni como criterio de parada; es el régimen esperado. La señal útil es comparativa entre arquitecturas, datasets o configuraciones de batch size, no la pendiente intra-corrida. En `var/per_layer/{name}` la convención por capa es **mayor NGV = capa más ruidosa**, y la lectura informativa es la heterogeneidad: si una capa (típicamente las convolucionales tempranas o el fully-connected final) domina la covarianza global, ahí está la fuente del ruido y ahí debe atacarse (re-inicialización, regularización, ajuste de learning rate por grupo). En `var/grad_norm_hist` la lectura no es monótona sino distribucional: colas pesadas o bimodalidad indican una distribución de gradientes con estructura — potencialmente explotable con muestreo estratificado, o señal de outliers/etiquetas corruptas. Operativamente, el objetivo del TFG con esta métrica no es minimizar la NGV (es imposible en CIFAR/ImageNet por construcción) sino usarla como diagnóstico del régimen SNR y validar su correlación con `gns_simple` (CLT predice Spearman $> 0.9$) y `gsnr` (Spearman $\sim 0.6$–$0.8$). Por construcción la NGV es sensible a la norma del gradiente medio en el denominador y muy ruidosa cerca de mínimos locales o tras convergencia parcial de una capa: solo cuando el global y el desglose por capa se mueven a la vez hay un cambio estructural real en la geometría del problema.

**Gotchas.** El más serio es la **degradación del estimador al caer $\|\mathbb{E}[g]\|^2$**: cerca de un mínimo la norma del gradiente medio cae varios órdenes de magnitud. Con el estimador plug-in implementado la NGV **no explota: satura en ≈K** — el denominador $\|\bar g\|^2$ está sesgado al alza en $\text{tr}(\text{Cov})/K$, así que con media nula y $K=10$ el estimador lee ≈10 (verificado numéricamente); el $\varepsilon = 10^{-12}$ solo cubre el 0/0 exacto. Es decir, en el régimen de interés (NGV ≳ 1) la compresión es severa: una NGV real de 10 se lee ≈5, invertible post-hoc vía $\text{NGV} \approx \widehat{\text{NGV}}/(1-\widehat{\text{NGV}}/K)$. Mitigaciones de lectura: reportar log-NGV o limitar el rango ploteado. Segundo, los $K$ batches deben ser **independientes y sin reemplazo dentro de la medición**: la independencia es necesaria pero no suficiente para insesgadez, hay que combinarla con la corrección de Bessel ($\div(K-1)$) del estimador de varianza; reutilizar el mismo batch sesga hacia cero. Tercero, la NGV no es monótona: contra-intuitivamente, en CIFAR-10 e ImageNet **crece** durante el entrenamiento y sube tras los LR drops, así que no debe usarse como criterio de parada simple.

**Pseudocódigo PyTorch.**

```python
import torch

@torch.no_grad()
def reset_acc(model):
    return {p: (torch.zeros_like(p), torch.zeros_like(p)) for p in model.parameters()}

def measure_ngv(model, loss_fn, sampler, K=30, eps=1e-12):
    """Devuelve (avg_var, ngv_global, ngv_per_layer) para los pesos actuales."""
    # Acumuladores streaming: S = sum g_k, Q = sum g_k^2 (coord-a-coord)
    S = {p: torch.zeros_like(p) for p in model.parameters()}
    Q = {p: torch.zeros_like(p) for p in model.parameters()}

    model.eval()  # congelar BN/dropout; los pesos no se actualizan
    for _ in range(K):
        x, y = next(sampler)            # mini-batch independiente, tamaño B
        model.zero_grad(set_to_none=False)
        loss = loss_fn(model(x), y)
        loss.backward()
        for p in model.parameters():
            if p.grad is None: continue
            S[p].add_(p.grad)
            Q[p].add_(p.grad.pow(2))

    avg_var_total, num_total, denom_total = 0.0, 0, 0.0
    ngv_per_layer = {}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        mean = S[p] / K
        var  = (Q[p] - S[p].pow(2) / K) / (K - 1)  # Var[g_j] coord-a-coord, insesgada (Bessel)
        var.clamp_(min=0)                    # ruido fp puede dar negativos
        tr_cov   = var.sum().item()
        mean_sq  = mean.pow(2).sum().item()
        ngv_per_layer[name] = tr_cov / (mean_sq + eps)
        avg_var_total += tr_cov
        num_total     += p.numel()
        denom_total   += mean_sq

    avg_var = avg_var_total / num_total          # Average Variance (escalar)
    ngv     = avg_var_total / (denom_total + eps)  # NGV global
    return avg_var, ngv, ngv_per_layer
```

El bucle se invoca una vez por época (o cada $N$ pasos) sobre un *snapshot* de los pesos, sin invocar `optimizer.step()`. Para acumular con Welford en lugar de $Q$, se sustituyen los dos acumuladores por la actualización incremental clásica de media y varianza online.

**Consideraciones finales.** La NGV está estrechamente emparentada con el *gradient noise scale* de McCandlish et al., que reescala una cantidad análoga para predecir el batch size crítico, y por construcción debe verificar $\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$. Conviene loguear NGV global y por capa simultáneamente: en CIFAR-10/ImageNet la NGV global crece, pero el comportamiento por capa puede ser heterogéneo y revela qué bloques dominan la covarianza.

## Notes

**Aporta al TFG:** métrica `normalized_variance` (NGV), familia varianza.

$$\text{NGV} = \frac{\text{tr}(\text{Cov}(g))}{\|\mathbb{E}[g]\|^2} \quad (\text{inverso de SNR; } <1 \Rightarrow \text{señal domina; adaptación del TFG — la fórmula literal del paper es } V[g_j]/E[g_j^2])$$

**Implementación:** $K = 10$ gradientes de sub-batch del probe (la v1 materializa la pila $[K, P]$; el streaming $S$/$Q$ queda como optimización). Coste bajo (≈3–5% overhead). El estimador plug-in satura en ≈K (ver Gotchas).

**Signo:** menor es mejor.

**Por qué normalizada y no absoluta.** La varianza cruda no es comparable entre escalas (CIFAR-10 ~$10^{-4}$ vs ImageNet $<10^{-6}$). La NGV sí lo es.

**Aviso de nomenclatura.** El paper define dos métricas distintas que conviene no confundir. La *Average Variance* es la traza normalizada de la covarianza (varianza absoluta agregada) y **no** es comparable entre problemas de distinta escala. La *Normalized Variance* del paper, $\mathbb{V}[g_j]/\mathbb{E}[g_j^2]$ (momento no central; la NGV del TFG, $\text{tr}(\text{Cov})/\|\mathbb{E}[g]\|^2$, es su adaptación monótonamente equivalente), **sí** lo es. El término "Normalized Average Variance" no aparece como tal en el paper: mezcla ambos conceptos. Para cualquier comparación cross-problema (datasets o modelos) hay que usar la NGV; la Average Variance cruda solo tiene sentido dentro de un mismo problema.

**Para análisis.** Validar redundancia de NGV frente a `gns_simple` (por CLT, $\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$, Spearman esperado $> 0.9$) y frente a `gsnr` (Spearman ~$0.6$–$0.8$). Dropear cara si la correlación es alta. El hallazgo contraintuitivo sirve como sanity check: la NGV **crece** durante el entrenamiento en CIFAR/ImageNet y sube tras los LR drops; no es monótona.

**Soporte teórico (intro/related).** Justifica el eje varianza del TFG y conecta con Adam/RMSProp (la ratio $\hat m / \sqrt{\hat v}$ es un SNR por parámetro) y con SVRG.

**NO usar.** Gradient Clustering queda fuera de scope porque es un método de optimización, no una métrica correlacional. El paper demuestra empíricamente (solo en algunos casos, sobre todo cuando hay duplicados o etiquetas corruptas) que explotar la distribución de los gradientes mediante un muestreo estratificado puede acelerar el entrenamiento; la figura inferior ilustra cómo GC reduce la varianza antes que mini-batch de tamaño $B$, de tamaño $2B$ y que SVRG.

![[Pasted image 20260521195641.png]]

## Papers relacionados

- [[An Empirical Model of Large-Batch Training]] — misma familia varianza; el gradient noise scale $\mathcal{B}_{\text{simple}} = \operatorname{tr}(\Sigma)/\|G\|^2$ es la misma ratio que NGV salvo escala (CLT: $\mathcal{B}_{\text{simple}}\approx B\cdot\text{NGV}$, Spearman esperado >0.9; fuente de `gns_simple`).
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — familia varianza; GSNR $= \tilde{g}^2/\rho^2$ es el SNR por parámetro, esto es, el inverso de la NGV agregado de otra forma (validar redundancia, Spearman ~0.6–0.8; fuente de `gsnr`).
- [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]] — SVRG aparece como baseline en este paper; su hipótesis de varianza decreciente (*strong growth*) falla en deep learning real, justo lo que la NGV creciente diagnostica.
- [[Adam - A Method for Stochastic Optimization]] — la ratio $\hat m_t/\sqrt{\hat v_t}$ de Adam es el SNR por parámetro que motiva interpretar la NGV como inverso de un signal-to-noise.
- [[RMSProp - Divide the gradient by a running average of its recent magnitude]] — el segundo momento no centrado $\mathbb{E}[g^2]$ que reescala el paso es el mismo estadístico de varianza que entra en la NGV.

## Otros papers interesantes a revisar

- **On the Ineffectiveness of Variance Reduced Optimization for Deep Learning** (Defazio & Bottou, NeurIPS 2019) — confirma desde otro ángulo que SVRG (baseline aquí) no acelera el deep learning moderno; refuerza la lectura empírica de Faghri y la motivación del TFG de medir, no reducir, la varianza. arXiv:1812.04529
- **A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks** (Şimşekli et al., 2019) — el ruido del gradiente es de cola pesada (no gaussiano); directamente relevante para el histograma de normas $\|g_i\|$ que se loguea como diagnóstico y para la vista distribucional de Faghri. arXiv:1901.06053
- **The Break-Even Point on Optimization Trajectories of Deep Neural Networks** (Jastrzębski et al., 2020) — analiza el espectro de $\text{Cov}(g)$ en la fase temprana; pertinente para interpretar el crecimiento no monótono de la NGV y la elección de ventanas tempranas. arXiv:2002.09572
- **Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model** (Zhang et al., 2019) — el modelo cuadrático ruidoso explica los rendimientos decrecientes con el batch size que motivan el noise scale y la NGV. arXiv:1907.04164
