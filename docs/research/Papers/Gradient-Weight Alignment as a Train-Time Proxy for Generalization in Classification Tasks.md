---
authors:
  - Florian A. Hölzl
  - Daniel Rueckert
  - Georgios Kaissis
year: 2025
status: to-read
relevance: high
url: https://arxiv.org/abs/2510.25480
tfg_role:
  - metric
tfg_note: "Origen de `gwa` (coseno entre el gradiente per-sample y los pesos del clasificador final; casi gratis, last-layer; Pearson 0.99 con test accuracy). Competidor más reciente (2025), uno de los 3 papers más comparables."
---

# Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks

## Summary

### Contextualización

El trabajo se enmarca en la búsqueda de proxies de generalización que puedan computarse durante el propio entrenamiento (train-time) en tareas de clasificación supervisada con pérdida de entropía cruzada. La práctica habitual para diagnosticar sobreajuste y monitorizar la dinámica de optimización consiste en reservar un conjunto de validación independiente y verificar el desempeño sobre datos i.i.d. respecto al de entrenamiento. Esta estrategia arrastra tres limitaciones que motivan el paper: reduce los datos etiquetados disponibles para el ajuste, presupone independencia distribucional entre entrenamiento y validación, y ofrece solo una visión agregada que no permite atribuir el comportamiento del modelo a muestras concretas del dataset. Enfoques anteriores intentan estimar el gap de generalización sin validación: coherencia de gradientes (gradient coherence, stiffness, Coherent Gradients, GSNR), curvatura del paisaje de pérdida, o métodos basados en cambios de predicción como LabelWave o Gradient Disparity. Todos tropiezan con coste computacional, almacenamiento de gradientes por muestra, escalabilidad limitada o sensibilidad a hiperparámetros. La tesis del autor es que medir la dirección de los gradientes per-sample respecto a los pesos del modelo permite caracterizar la convergencia direccional descrita por la teoría de Ji y Telgarsky sobre flujo gradiente con cross-entropy, evitando estadísticas pairwise costosas.

### Aportación: Gradient-Weight Alignment (GWA)

La contribución central es la métrica Gradient-Weight Alignment (GWA), un proxy escalable de generalización calculable de forma online durante el entrenamiento. GWA cuantifica la coherencia entre los gradientes per-sample y los pesos del modelo, recogiendo tanto la contribución específica de cada ejemplo como la dinámica global del dataset. El autor muestra que GWA predice con precisión el punto óptimo de early stopping y puede sustituir al validation set. También permite comparar modelos a través de runs distintos mediante el alineamiento máximo alcanzado, identifica muestras influyentes, outliers y etiquetas ruidosas a partir de los scores per-sample, y escala tanto a entrenamiento desde cero como a fine-tuning manteniéndose robusta bajo ruido de entrada y de etiquetas. La intuición geométrica (Fig. 1) es que durante la fase de aprendizaje efectivo los gradientes individuales están direccionalmente alineados con los pesos (régimen coherente), mientras que la inicialización y el sobreajuste corresponden a regímenes incoherentes con direcciones de gradiente dispersas u ortogonales.

### Metodología

La definición formal parte del gradiente negativo de la pérdida respecto a los pesos $\mathbf{w}_T$ en la época $T$, $\mathbf{g}_T(x_i) = -\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_T, x_i)$. El score de alineamiento per-sample se define entonces como la similitud coseno

$$\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T) = \frac{\mathbf{g}_T(x_i)\cdot \mathbf{w}_T}{\|\mathbf{g}_T(x_i)\|\,\|\mathbf{w}_T\|}.$$

A modo de ejemplo, si en una época concreta los gradientes de muchos ejemplos apuntan claramente en la dirección de $\mathbf{w}_T$, los $\gamma(x_i, \mathbf{w}_T)$ individuales serán positivos y cercanos a 1, lo que indica que cada paso de descenso reduce la pérdida de forma coherente para todo el dataset. Si en cambio aparecen muestras con $\gamma$ negativo o cercano a 0, esos puntos están tirando en direcciones opuestas a la mayoritaria: candidatos a outlier o a etiqueta ruidosa.

Sea $\mathcal{G}_T = \{\gamma(x_i, \mathbf{w}_T)\}_{i=1}^N$ el conjunto de scores per-sample en la época $T$, $\mathcal{A}_T$ su distribución empírica y $M_T^{(k)}$ su $k$-ésimo momento. GWA se define como la esperanza con corrección por exceso de curtosis,

$$\text{GWA}_T = \frac{\mathbb{E}_i[\mathcal{A}_T]}{\text{Kurt}_i[\mathcal{A}_T] + \beta} = \frac{M_T^{(1)}}{M_T^{(4)}/(M_T^{(2)})^2 - 3 + \beta}.$$

El término $M^{(4)}/(M^{(2)})^2 - 3$ es el exceso de curtosis (curtosis menos 3, donde 3 corresponde a la normal). El parámetro $\beta = 1.2$ se elige para compensar el exceso de curtosis de una distribución uniforme sobre $[-1,1]$, que es $-1.2$ y que el paper presenta como caso límite "never to be expected in practice". **No es una cota matemática**: la curtosis mínima de cualquier distribución es 1 (dos masas simétricas), con exceso $-2$ y denominador $-0.8 < 0$. La implementación del TFG maneja ese borde con un guard de signo (empuja el denominador en su propia dirección por $\varepsilon$, sin voltear el signo) y devuelve NaN cuando $M^{(2)} = 0$ (curtosis indefinida), no con un clamp. La corrección por curtosis se justifica por la teoría long-tail del deep learning: muestras raras o atípicas tienen influencia desproporcionada y producen distribuciones leptocúrticas que aumentan la curtosis, reduciendo GWA y señalando patrones de aprendizaje problemáticos.

La conexión teórica es directa con el resultado de Ji y Telgarsky sobre convergencia direccional bajo flujo gradiente con cross-entropy: en datos perfectamente clasificables, gradiente y pesos convergen en la misma dirección, $\mathbb{E}_i[\gamma(x_i, \mathbf{w}_T)] \to 1$ asintóticamente. En la práctica con datasets ruidosos, la distribución completa de scores ofrece una medida más rica que la convergencia ideal a uno.

El estimador escalable evita materializar gradientes completos de la red por muestra, algo prohibitivo en datasets grandes y propenso a degradación de la similitud coseno por alta dimensionalidad. Para ello restringe el cálculo al clasificador lineal final, donde el gradiente admite forma cerrada $\mathbf{g}_t(x_i) = -z_i \cdot (\hat{y}_i - y_i)^\top$, con $z_i$ la representación latente (penúltima capa), $\hat{y}_i = \text{softmax}(\text{logits})$ la probabilidad (salida softmax) e $y_i$ el target. La elección se apoya en que el clasificador lineal proporciona la señal más directa de la tarea aprendida; capas más profundas degradan la estimación. El segundo aspecto crítico es cuándo medir el alineamiento: en lugar de recomputar scores sobre el dataset completo en un paso fijo, los momentos centrales $\hat{M}_T^{(k)}$ se estiman acumulando alineamientos a lo largo de los pasos de actualización de una época. El Algoritmo 1 del paper resume el procedimiento: para cada minibatch $\mathcal{B}_{T,t}$ se hace un forward pass estándar, se computan logits y latents, se obtiene $\mathbf{g}_t(x_i)$ en forma cerrada, se calcula $\gamma(x_i, \mathbf{w}_{T,t})$ y se almacena; al cerrar la época se devuelve GWA según la Ec. (2). El sesgo introducido por la deriva de pesos durante la época es despreciable: comparado con un estimador offline a $\mathbf{w}_0$ se obtiene un sesgo de primer orden proporcional al learning rate (verificado contra el PDF, p. 14, A.1: media 0.04 con máximo 0.12 para $\eta = 0.1$; 0.027/0.08 para $\eta = 0.01$; 0.020/0.05 para $\eta = 0.001$). Respecto al midpoint $\mathbf{w}_{\text{mid}}$, el sesgo dominante de primer orden se cancela. La correlación online vs. offline resulta casi perfecta (Fig. 6).

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas y resultados

GWA se compara contra early stopping con val sets 90/10 y 99/1, contra LabelWave (cambio de predicciones) y contra Gradient Disparity (GD). En la Tabla 1 iguala o supera a la mayoría de criterios: en CIFAR-10 y CIFAR-10-N obtiene en promedio +0.4% sobre el val set estándar y +0.67% sobre LabelWave; el claim "On ViT, GWA even outperforms the 99/1% validation split" se sostiene en **CIFAR-10** ("particularly strong on the smaller CIFAR-10 dataset") y elimina además la dependencia del hold-out. En ImageNet-1k/ViT, en cambio, GWA queda por debajo de Val 1% en Val y ReaL y solo gana en V2 por 0.01. En CIFAR-10 con 0%, 9% y 17% de ruido alcanza 81.57/78.93/75.70 con ViT y 89.73/86.08/82.55 con ConvNeXt. El máximo alineamiento $\max_T \mathbb{E}[\mathcal{A}_T]$ correlaciona fuertemente con la accuracy de test en ConvNeXt y ViT sobre CIFAR-10 y CIFAR-C: las correlaciones reportadas alcanzan Pearson 0.99 y 0.98 (Spearman 0.97 y 0.93) en CIFAR-10, y Pearson 0.99 y 0.94 (Spearman 0.96 y 0.92) en CIFAR-C, todas con $p<0.001$. En robustez (Tabla 2), los modelos seleccionados con GWA mejoran en promedio +0.55% sobre CIFAR-C y +0.67% sobre ImageNet-C frente al val set 10%. Los scores per-sample $\gamma(x_i, \mathbf{w}_T)$ identifican casi todas las muestras mal etiquetadas en CIFAR-10-N (Fig. 4) sin necesidad de técnicas explícitas de detección, y reflejan el sesgo de simplicidad: alineamientos positivos altos al inicio corresponden a ejemplos visualmente simples, mientras que los aumentos posteriores incorporan ejemplos progresivamente más complejos. En fine-tuning (Tabla 3, Fig. 5), GWA muestra una caída inicial seguida de un mínimo y un posterior incremento; el criterio adaptado (mínimo inicial, máximo posterior) iguala o supera al val set 1%/10% en ImageNet-1k, iNat18 y Places365.

### Conclusiones

El autor concluye que GWA proporciona una métrica fiable de tiempo de entrenamiento que captura dinámicas dataset-level y sample-level, sirviendo como criterio de early stopping robusto frente a ruido de etiquetas y bajo régimen de datos. Se elimina así la necesidad de validation sets y se ofrece una herramienta diagnóstica para comparación de modelos e identificación de muestras influyentes. Las limitaciones reconocidas incluyen la posibilidad de distribuciones bimodales tardías que el estimador con curtosis no captura bien y la sensibilidad de la magnitud del score a la dimensionalidad latente, mitigable con Johnson-Lindenstrauss Transform. Como direcciones futuras se apuntan la extensión al setting self-supervised, las modalidades de texto con pérdida autorregresiva y la integración con aproximaciones de matrix information theory (MIR/HDR) y neural collapse.

![[Pasted image 20260421195041.png]]

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** Gradient-Weight Alignment (GWA), propuesta por Hölzl (2025). El score per-sample es la similitud coseno (con signo) entre el gradiente negativo de la pérdida $\mathbf{g}_T(x_i) = -\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}_T, x_i)$ y el vector de pesos $\mathbf{w}_T$ del clasificador lineal final, $\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T)$. El agregado escalar por época se obtiene como esperanza corregida por exceso de curtosis, $\text{GWA}_T = M_T^{(1)} / (M_T^{(4)}/(M_T^{(2)})^2 - 3 + \beta)$, con $\beta = 1.2$.

**Entradas.** Los dos ingredientes son el gradiente per-sample $\mathbf{g}_T(x_i)$ y los pesos $\mathbf{w}_T$ del clasificador lineal final. En la variante barata, que es la que se integra en el TFG, el gradiente per-sample admite forma cerrada sobre la última capa, $\mathbf{g}_t(x_i) = -z_i (\hat{y}_i - y_i)^\top$, donde $z_i$ es la representación latente (salida de la penúltima capa), $\hat{y}_i = \text{softmax}(\text{logits})$ es la probabilidad predicha (salida softmax, no un logit) y $y_i$ es el target one-hot. Es decir, basta con disponer de logits, labels y latents, todos ya producidos por el forward pass.

**Pipeline.** A partir de los gradientes per-sample $\mathbf{g}_T(x_i)$ y los pesos $\mathbf{w}_T$ se calcula $\text{score}_i = \cos(\mathbf{g}_T(x_i), \mathbf{w}_T)$ por cada ejemplo del probe set. El agregado escalar de época es entonces $\text{GWA}_T = \mathbb{E}[\text{score}] / (\text{Kurt}[\text{score}] + \beta)$, con $\beta = 1.2$. La curtosis se calcula como exceso, $\text{Kurt} = M^{(4)}/(M^{(2)})^2 - 3$, sobre la distribución empírica de los scores.

**Granularidad temporal.** Para la integración del TFG se distinguen dos cadencias. Una medición por época durante toda la corrida del entrenamiento, suficiente para reproducir la trayectoria $\max_T \mathbb{E}[\mathcal{A}_T]$ del paper. Y mediciones cada $K$ pasos durante la ventana temprana (los primeros 5%, 10%, 25% y 50% de las épocas totales), que es donde se evalúa la capacidad predictiva sobre eficiencia y generalización.

**Granularidad estructural.** Por defecto el cálculo se restringe a la última capa lineal, siguiendo la justificación de Hölzl: es donde el gradiente admite forma cerrada y donde la señal direccional es más informativa de la tarea aprendida. Opcionalmente puede computarse por capa para arquitecturas donde interese contrastar dinámicas locales, pero la versión last-layer-only es la canónica.

**Coste.** Barato. La extracción de $z_i$ y de los logits es parte del forward estándar; los labels $y_i$ están disponibles en el batch. No se requiere un backward adicional para esta variante porque $\mathbf{g}_t(x_i)$ se compone en forma cerrada como producto exterior de tensores ya materializados. El coste dominante por minibatch es $\mathcal{O}(B \cdot d_z \cdot C)$, donde $B$ es el batch size, $d_z$ la dimensión latente y $C$ el número de clases.

**Probe set.** En la v1 real es el probe compartido de $M = 256$ ejemplos muestreado con `randperm` (la estratificación por clase y los tamaños 512/1024 quedan pendientes; la estratificación garantizaría cobertura uniforme y reduciría la varianza del estimador de curtosis, sensible a desbalance).

**Claves de logging.** Para cada época se persisten tres entradas: `gwa/score_mean` (la media $\mathbb{E}[\text{score}]$), `gwa/kurt` (el exceso de curtosis del denominador, antes de sumar $\beta$; Gaussiana ≈ 0) y `gwa/value` (el escalar $\text{GWA}_T$ resultante). El histograma per-sample (`gwa/per_sample_hist`, útil para outliers y muestras mal etiquetadas) no cabe en el contrato `dict[str, float]` del registry y queda retirado de la promesa.

**Interpretación de la señal.** Conviene fijar la convención porque GWA combina una señal cruda, un atenuador y un escalar corregido que se leen de forma distinta. En `gwa/score_mean` la lectura es la canónica del paper y la que se alinea con las métricas coseno del TFG: **mayor `score_mean` = mejor**, porque el resultado de Ji y Telgarsky implica $\mathbb{E}_i[\gamma(x_i, \mathbf{w}_T)] \to 1$ asintóticamente en datos clasificables y los valores empíricos altos correlacionan fuertemente con la accuracy de test ($r \approx 0.99$ en CIFAR-10). En `gwa/value`, que es el escalar corregido $M^{(1)}/(M^{(4)}/(M^{(2)})^2 - 3 + \beta)$, la lectura es la misma —**mayor `value` = mejor generalización**— y es el proxy propiamente dicho del que se extrae el criterio de early stopping vía $\max_T \mathbb{E}[\mathcal{A}_T]$. En cambio `gwa/kurt` es **diagnóstica, no monótona**: no debe leerse como "más alta peor" ni "más alta mejor", sino como un atenuador de la media. Una curtosis alta indica una distribución de cosenos con colas pesadas (pocas muestras dominando la media), que el paper descuenta dividiendo por ella para reflejar la influencia desproporcionada de ejemplos atípicos en escenarios long-tail. Sirve para entender por qué un `score_mean` aparentemente alto puede traducirse en un `value` más modesto. El histograma `gwa/per_sample_hist` es complementario y sirve para inspeccionar la cola izquierda: muestras con $\gamma_i$ muy negativo son candidatas a etiqueta ruidosa o outlier, que es precisamente el uso sample-level que el paper demuestra en CIFAR-10-N. Operativamente, el objetivo del diseño es maximizar `gwa/value` durante la ventana temprana y detectar su pico como señal de parada. Por último, la lectura depende de la **convención de signo**: el paper define $\mathbf{g} = -\nabla L$, así que un cómputo directo sobre $\nabla L$ bruto invierte el signo de `score_mean` y `value` y obliga a leer "menor = mejor", lo que rompe la coherencia con el resto del TFG. La convención debe explicitarse en la capa de análisis.

**Avisos.** La curtosis es muy sensible a outliers numéricos, así que conviene calcular los momentos vía Welford u otro algoritmo estable, no a partir de $\sum \gamma^k$ crudos. Batch normalization puede distorsionar la dinámica de $\mathbf{w}_T$ al introducir reescalados implícitos, por lo que en arquitecturas con BN hay que documentar la convención (medir sobre pesos pre-BN o post-BN) y mantenerla consistente entre runs. El término $\beta$ evita la división por cero, pero conviene además aplicar un `clamp` $\max(\cdot, \epsilon)$ al denominador como salvavidas numérico. El bias del clasificador final debe excluirse: su norma puede ser cero al inicio y dejaría el coseno indefinido. Y la convención de signo del paper ($\mathbf{g} = -\nabla L$) debe documentarse, porque trabajar sobre $\nabla L$ bruto invierte el signo del score.

**Pseudocódigo PyTorch.**

```python
# Suponemos modelo con clasificador final lineal: model.classifier = nn.Linear(d_z, C)
# Y un hook (o forward custom) que expone z_i (latente penúltima capa).

import torch
import torch.nn.functional as F

@torch.no_grad()
def gwa_step(model, batch_x, batch_y, num_classes, accumulator):
    # Forward que devuelve logits y latents z_i (penúltima capa).
    z, logits = model.forward_with_latents(batch_x)        # z: (B, d_z), logits: (B, C)
    y_hat = F.softmax(logits, dim=1)                       # (B, C)
    y_onehot = F.one_hot(batch_y, num_classes).float()     # (B, C)

    # Pesos del clasificador final (excluir bias).
    W = model.classifier.weight.detach()                   # (C, d_z)

    # Gradiente per-sample en forma cerrada: g_i = -z_i (y_hat_i - y_i)^T, shape (C, d_z).
    # cos(g_i, W) = <g_i, W>_F / (||g_i||_F * ||W||_F)
    delta = y_hat - y_onehot                               # (B, C)
    # <g_i, W>_F = -<delta_i z_i^T, W> = -(delta_i^T W z_i)
    dot = -(delta * (z @ W.t())).sum(dim=1)                # (B,)
    g_norm_sq = (delta.pow(2).sum(dim=1)) * (z.pow(2).sum(dim=1))  # ||g_i||^2 = ||delta_i||^2 ||z_i||^2
    g_norm = g_norm_sq.clamp_min(1e-12).sqrt()
    W_norm = W.norm()
    score = dot / (g_norm * W_norm + 1e-12)                # (B,) coseno per-sample

    accumulator.update(score)                              # Welford para M^(1..4)
    return score

def gwa_epoch(accumulator, beta=1.2, eps=1e-12):
    m1 = accumulator.mean()
    m2 = accumulator.central_moment(2)
    m4 = accumulator.central_moment(4)
    if m2 == 0:                                # curtosis indefinida -> NaN honesto
        return {"gwa/score_mean": m1.item(), "gwa/kurt": float("nan"), "gwa/value": float("nan")}
    excess_kurt = m4 / m2.pow(2) - 3.0         # float64; sin eps aditivo (distorsiona)
    denom = excess_kurt + beta
    denom = denom + (eps if denom >= 0 else -eps)   # guard de signo, no max(., eps):
    gwa = m1 / denom                                # max() explotaria con signo erroneo
                                                    # en distribuciones platicurticas
    return {"gwa/score_mean": m1.item(),
            "gwa/kurt": excess_kurt.item(),
            "gwa/value": gwa.item()}
```

El `accumulator` mantiene los momentos centrales de la distribución de scores con un algoritmo numéricamente estable (Welford extendido a momentos de cuarto orden) y permite cerrar el escalar por época sin almacenar todos los scores. Para la versión con probe set estratificado, la llamada a `gwa_step` se restringe a los $M = 512$ o $M = 1024$ ejemplos del probe en lugar de a todo el batch de entrenamiento.

## Notes

### Uso en el TFG

- **Métrica que origina.** Es la fuente de `gwa` en el `METRIC_REGISTRY` (familia alineación). Score per-sample $\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T)$: coseno entre el gradiente per-sample y los pesos del clasificador final. Agregado de época con corrección por exceso de curtosis $\text{GWA}_T = M^{(1)} / (M^{(4)}/(M^{(2)})^2 - 3 + \beta)$, con $\beta = 1.2$.
- **Cómo se usa (last-layer; v1 vía sweep per-sample).** El estimador canónico del paper es la forma cerrada de la última capa $\mathbf{g}_t(x_i) = -z_i (\hat{y}_i - y_i)^\top$ (con $z_i$ latente penúltima, $\hat{y}_i = \text{softmax}(\text{logits})$), acumulable online en el forward sin backward extra. **La v1 del TFG no la implementa así**: obtiene el sweep per-sample completo vía `vmap` y descarta todo salvo `head.weight` (matemáticamente idéntico, verificado a 1.2e-7, pero coste de tramo medio); la forma cerrada con hooks queda como optimización pendiente que dejaría la métrica casi gratis.
- **Señal.** $\uparrow$ mejor: el paper reporta Pearson 0.99 / Spearman 0.97 entre $\max_T \mathbb{E}[\mathcal{A}_T]$ y test accuracy en CIFAR-10. Cadencia `epoch` (acumulación online, scalar al cerrar la época).
- **Pitfalls / decisiones.** (1) Excluir el bias del clasificador (norma cero $\Rightarrow$ coseno indefinido). (2) Momentos centrales en float64 (la v1 usa two-pass sobre el tensor de gammas, no Welford; computar la curtosis desde $\sum \gamma^k$ crudos o en float32 es numéricamente inestable — un EPS aditivo sobre $(M^{(2)})^2$ distorsiona la curtosis y puede voltear el signo de `gwa/value` cuando la dispersión baja de ~1e-3). (3) $\beta = 1.2$ deja el denominador no-negativo solo en el límite uniforme (la cota real del exceso es $-2$): guard de **signo** en el denominador y NaN si $M^{(2)} = 0$, nunca `max(·, ε)`. (4) Convención de signo: el paper usa $\mathbf{g} = -\nabla L$; el pipeline trabaja sobre $\nabla L$ bruto, así que se guarda $\gamma_i$ crudo y se documenta el signo en la capa de análisis. (5) En arquitectura FC el clasificador final es la red completa (caso degenerado, marcar con flag explícito).
- **Limitaciones heredadas del paper.** Distribuciones bimodales tardías que la corrección de curtosis no captura bien; la magnitud del score depende de la dimensionalidad latente $d_z$ (mitigable con Johnson-Lindenstrauss). Refuerzan la preferencia por comparaciones intra-arquitectura.
- **Paper comparable más reciente (delta a argumentar).** Junto con Ru et al. (2021, TSE) y Forouzesh & Thiran (2021, Gradient Disparity), es uno de los tres papers más comparables al TFG (mismo problema: proxy train-time barato de generalización) y el más reciente. La intro de la tesis debe argumentar el delta frente a este: GWA es un proxy puntual *post-hoc* para early stopping y selección de modelo sobre una métrica/arquitectura; el TFG propone un estudio correlacional cerrado que mide 10 métricas (7 alineación + 3 varianza) + baseline en ventanas tempranas (5/10/25/50% de épocas), sobre gradiente bruto $\nabla L$ (no solo last-layer), barriendo SGD/Adam y FC/CNN/ResNet, para predecir generalización y eficiencia sin optimizar nada.

### Discrepancias detectadas

- **Convención de signo de $\mathbf{g}$.** El paper define $\mathbf{g}_T(x_i) = -\nabla_{\mathbf{w}}\mathcal{L}$ (gradiente negativo, dirección de descenso), mientras que el pipeline del TFG opera sobre $\nabla_{\mathbf{w}}\mathcal{L}$ bruto. El coseno invierte de signo entre ambas convenciones, así que el escalar $\gamma_i$ debe interpretarse según el signo documentado; el orden y la magnitud relativos se conservan.
- **Versión per-layer frente a last-layer.** La nota previa del archivo describía un cálculo por capa sobre $\nabla L$ tras `loss.backward()` y antes de `optimizer.step()`. Esa variante es un atajo razonable, pero no es la métrica del paper. El estimador canónico de Hölzl es last-layer-only en forma cerrada $\mathbf{g}_t(x_i) = -z_i (\hat{y}_i - y_i)^\top$, sin backward adicional, y con corrección por curtosis. La variante per-layer queda como opción auxiliar.
- **Coste y ubicación en el sweep.** En el TFG GWA no consume el barrido per-sample $\nabla L$ compartido por `stiffness`, `m_coherence` y `gsnr`: se computa vía hooks al forward y acumulación online de momentos, por lo que su coste marginal sobre el entrenamiento es despreciable.

## Papers relacionados

- [[Stiffness - A New Perspective on Generalization in Neural Networks]] — familia alineación; cosine-stiffness es el mismo coseno entre gradientes per-sample, pero entre pares de muestras en vez de gradiente-vs-pesos.
- [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]] — familia alineación; m-coherence mide alineamiento gradiente-vs-gradiente promedio, alternativa per-sample escalable como GWA.
- [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]] — familia alineación; gradient confusion (min coseno entre gradientes per-sample) comparte la lectura geométrica de coherencia direccional.
- [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]] — familia alineación; marco CGH del que GWA hereda la intuición de coherencia $\to$ generalización.
- [[A Theory of Neural Tangent Kernel Alignment and Its Influence on Training]] — familia alineación vía kernel-target alignment, también restringible a last-layer; alineamiento como predictor de aprendizaje.
- [[Disparity Between Batches as a Signal for Early Stopping]] — paper comparable directo: proxy train-time barato para early stopping (GD); GWA lo usa como baseline competidor en sus tablas.
- [[Speedy Performance Estimation for Neural Architecture Search]] — paper comparable directo: TSE/TSE-EMA, mismo problema de predicción temprana barata; es el baseline del TFG.

## Otros papers interesantes a revisar

- **Directional convergence and alignment in deep learning** (Ji & Telgarsky, 2020) — base teórica directa de GWA: bajo flujo gradiente con cross-entropy en datos separables, gradiente y pesos convergen direccionalmente ($\mathbb{E}[\gamma] \to 1$). arXiv:2006.06657.
- **Estimating Training Data Influence by Tracing Gradient Descent (TracIn)** (Pruthi et al., 2020) — usa el gradiente per-sample (también restringido a last-layer por coste) para atribuir influencia y detectar etiquetas ruidosas, igual que el uso sample-level de $\gamma(x_i, \mathbf{w}_T)$. arXiv:2002.08484.
- **Prevalence of Neural Collapse during the terminal phase of deep learning training** (Papyan, Han & Donoho, 2020) — explica la geometría last-layer (features y pesos del clasificador colapsando alineados) que sustenta por qué el estimador last-layer de GWA es informativo. arXiv:2008.08186 / DOI:10.1073/pnas.2015509117.
- **Characterizing signal propagation to close the performance gap in unnormalized ResNets** / línea de proxies *zero-cost* NAS como **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah et al., 2021) — proxies train-time/at-init (SNIP, GraSP, synflow, jacov) sobre gradientes; marco comparativo natural para situar el coste y poder predictivo de GWA. arXiv:2101.08134.
- **Gradient Starvation: A Learning Proclivity in Neural Networks** (Pezeshki et al., 2021) — analiza cómo la dinámica de gradientes condiciona qué features se aprenden y la generalización; complementa la lectura de coherencia direccional con el sesgo de simplicidad que GWA observa empíricamente. arXiv:2011.09468.
