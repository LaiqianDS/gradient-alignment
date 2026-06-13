---
authors:
  - Sam McCandlish
  - Jared Kaplan
  - Dario Amodei
year: 2018
status: to-read
relevance: medium
url: https://arxiv.org/pdf/1812.06162
tfg_role:
  - metric
tfg_note: "Origen de `gns_simple` (gradient noise scale: ruido del gradiente relativo a su norma) que predice a priori el batch size crítico. Eje varianza; caso canónico de métrica de ruido → elección de hiperparámetro."
---

# An Empirical Model of Large-Batch Training

## Summary

### Contextualización

El paper se inscribe en un problema central del deep learning a escala: el compromiso entre el tamaño de batch durante el entrenamiento, el grado de paralelismo de datos alcanzable y la eficiencia computacional global. La motivación parte de una observación empírica acumulada según la cual el umbral a partir del cual ampliar el batch deja de aportar un speed-up casi lineal varía radicalmente entre dominios. En clasificación sobre ImageNet se han reportado batch sizes efectivos del orden de decenas de miles (8k, 16k, 32k e incluso 64k); en modelos de lenguaje el orden es de miles; y el agente Dota de OpenAI llegó a operar con batches de millones de timesteps. Antes de este trabajo no existía un marco conceptual unificado que explicase cuantitativamente por qué esos límites difieren entre tareas ni cómo anticipar el batch size adecuado para un problema nuevo. El paralelismo de datos exige comunicación rápida entre dispositivos, pero su utilidad real depende de que el batch grande siga siendo algorítmicamente productivo: si el ruido del gradiente ya es bajo, duplicar el batch apenas mejora el paso de optimización y desperdicia cómputo.

### Aportación principal

La contribución central consiste en introducir el gradient noise scale como predictor empírico, simple de medir, del batch size crítico $\mathcal{B}_{\text{crit}}$ por encima del cual la eficiencia de paralelización cae significativamente. Partiendo de hipótesis básicas sobre la pérdida y la covarianza del gradiente por ejemplo, los autores derivan una relación que predice tanto la forma exacta de la curva de Pareto entre el tiempo de entrenamiento ($S$, número de pasos) y el cómputo total ($E$, ejemplos procesados) como el punto de inflexión donde el paralelismo deja de ser productivo. El gradient noise scale es, en esencia, una medida de la relación señal-ruido del gradiente entre ejemplos del dataset, y su valor coincide aproximadamente con el batch crítico para una amplia variedad de tareas, optimizadores y arquitecturas.

### Metodología

La derivación parte de considerar el gradiente estimado $G_{\text{est}}$ sobre un batch de tamaño $B$ como una variable aleatoria con esperanza igual al gradiente verdadero $G$ y covarianza $\Sigma / B$, donde $\Sigma$ es la covarianza por ejemplo. Una expansión de Taylor de segundo orden de la pérdida bajo un paso de tamaño $\epsilon$ en la dirección de $G_{\text{est}}$ proporciona la mejora esperada. Optimizando respecto a $\epsilon$, se obtiene que el step size óptimo decae como

$$\epsilon_{\text{opt}}(B) = \frac{\epsilon_{\max}}{1 + \mathcal{B}_{\text{noise}} / B},$$

y la mejora óptima en pérdida tiene la misma forma funcional. De aquí emerge la definición exacta del noise scale ponderado por la Hessiana,

$$\mathcal{B}_{\text{noise}} = \frac{\mathrm{tr}(H \Sigma)}{G^\top H G}.$$

Bajo el supuesto idealizado de que la Hessiana es proporcional a la identidad, la expresión se reduce a la forma simplificada empleada en la mayoría de experimentos,

$$\mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|G\|^2},$$

esto es, la suma de las varianzas de las componentes del gradiente dividida por la norma global al cuadrado del gradiente, una medida de cuán grande es el gradiente respecto a su variabilidad entre ejemplos. Los autores muestran que en distancia $L^2$ normalizada $\mathbb{E}[\|G_{\text{est}} - G\|^2]/\|G\|^2 = \mathcal{B}_{\text{simple}}/B$, lo que conecta directamente la calidad del estimado de batch con la métrica. Promediando la mejora local sobre toda la trayectoria de entrenamiento se obtiene la ecuación clave del trade-off (forma hiperbólica),

$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1,$$

donde $S_{\min}$ es el número mínimo de pasos posible (límite de batch infinito) y $E_{\min}$ el mínimo de ejemplos procesados (límite de batch unitario). Definen el batch size crítico empírico como $\mathcal{B}_{\text{crit}} = E_{\min}/S_{\min}$, que el modelo predice aproximadamente igual al noise scale promediado adecuadamente. Distinguen así dos versiones de la métrica: el noise scale verdadero $\mathcal{B}_{\text{noise}}$, que requiere productos Hessiano-vector, y el simple noise scale $\mathcal{B}_{\text{simple}}$, computacionalmente mucho más barato. Ambos suelen diferir sólo por un factor multiplicativo pequeño, especialmente cuando se usan esquemas que mejoran el condicionamiento. El apéndice A.1 ofrece además un método sin overhead para estimar $\mathcal{B}_{\text{simple}}$ en entornos data-paralelos, comparando la norma del gradiente local (antes de promediar entre dispositivos) con la norma del gradiente global (después de promediar) y combinándolas mediante medias móviles exponenciales.

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas

Las magnitudes seguidas son los training steps $S$, los ejemplos procesados $E$ hasta alcanzar un nivel objetivo de rendimiento (error de clasificación, perplejidad, recompensa de episodio o TrueSkill en Dota) y la curva de Pareto entre ambos. Para cada tarea se ejecuta una grid search sobre batch size y learning rate, se ajustan los frentes de Pareto a la ecuación hiperbólica y se mide $\mathcal{B}_{\text{crit}} = E_{\min}/S_{\min}$ para varios niveles objetivo. En paralelo se mide $\mathcal{B}_{\text{simple}}$ durante el entrenamiento. El criterio operativo de batch crítico se define como el batch al que la eficiencia de cómputo cae al 50% del óptimo, punto en el cual el training speed también es típicamente el 50% del máximo.

### Resultados principales

El simple noise scale predice $\mathcal{B}_{\text{crit}}$ al nivel de orden de magnitud en todos los dominios estudiados; el paper solo cita textualmente el rango extremo "from 20 (for an SVHN autoencoder) to over 10 million" (p. 9, y referido a $\mathcal{B}_{\text{crit}}$). El rango ~1k–1M para tareas estándar, en cambio, es una lectura de los clusters de la Figura 4 (MNIST/SVHN/CIFAR-10 ~$10^3$, ImageNet ~$10^4$, Billion Word/Dota 1v1 ~$10^5$–$10^6$), no una cita. En CIFAR-10 con ResNet-32, por ejemplo, el modelo predice batches críticos de unos pocos miles consistentes con el frente de Pareto medido. La concordancia es notable considerando que la fórmula se deriva localmente y se aplica a trayectorias completas. Los frentes de Pareto se ajustan muy bien a la ecuación $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$. El learning rate óptimo sigue la forma funcional $\epsilon_{\text{opt}}(B) = \epsilon_{\max}/(1 + \mathcal{B}_*/B)$ para SGD/momentum ($\alpha = 1$), mientras que para Adam y RMSProp se observa una power law con $\alpha \in [0.5, 1.0]$.

Otras observaciones empíricas relevantes acompañan al resultado central. El noise scale crece a lo largo del entrenamiento, típicamente en uno o más órdenes de magnitud, porque $\|G\|$ disminuye conforme se aproxima un mínimo mientras $\mathrm{tr}(\Sigma)$ permanece relativamente constante; el batch crítico depende, por tanto, del nivel de rendimiento al que se entrena. A pérdida fija, el noise scale es aproximadamente independiente del tamaño del modelo (verificado con LSTMs de 512, 1024 y 2048): los modelos más grandes alcanzan mayor noise scale solo porque alcanzan menor pérdida. Las tareas más complejas exhiben mayor noise scale (Dota 5v5 supera a Dota 1v1, RL supera a clasificación y, dentro de Atari, Pong tiene noise scale menor que Beam Rider o Space Invaders), y los modelos generativos sobre SVHN presentan $\mathcal{B}_{\text{simple}}$ significativamente menor que los clasificadores sobre el mismo dataset, sugiriendo que extraen más información por ejemplo. El noise scale depende del learning rate vía una temperatura de entrenamiento $T = \epsilon/\epsilon_{\max}(B)$, con $\mathcal{B}_{\text{noise}}$ proporcional a $1/T$ en equilibrio, lo que explica por qué runs bien ajustadas con distintos batches presentan el mismo noise scale al mismo nivel de pérdida. Variar dinámicamente el batch size proporcionalmente a $\sqrt{\mathcal{B}_{\text{simple}}}$ produce mejoras modestas en eficiencia (estimadas en torno al 4% para SVHN), aunque experimentos preliminares apuntan a beneficios potencialmente mayores.

### Conclusiones

El gradient noise scale, en su forma simplificada $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma)/\|G\|^2$, es un predictor empíricamente sólido (al nivel de orden de magnitud) del batch size crítico que separa el régimen de paralelización lineal del régimen de retornos decrecientes. Esto permite estimar el batch size útil para una tarea nueva a partir de un único entrenamiento, incluso parcial, y reduce drásticamente el coste de las búsquedas por ensayo y error. La curva de Pareto entre pasos y ejemplos procesados sigue una forma hiperbólica universal, y el batch crítico se interpreta como el punto en que se duplica simultáneamente el cómputo mínimo y el tiempo mínimo: un compromiso natural entre eficiencia temporal y computacional. La complejidad de la tarea se refleja en el noise scale, lo que sugiere que problemas más ricos (RL complejo, lenguaje) son más paralelizables. El trabajo aporta además una explicación parcial del rol de la temperatura de entrenamiento y motiva los schedules adaptativos de batch size, aunque su tratamiento se centra deliberadamente en $B$ asumiendo que el learning rate se ajusta posteriormente.

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** Se adopta el simple gradient noise scale de McCandlish et al. en su forma idealizada (Hessiana proporcional a la identidad),

$$\mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|G\|^2},$$

donde $\Sigma$ es la covarianza por ejemplo del gradiente estocástico y $G$ el gradiente verdadero (esperanza sobre el dataset). La versión exacta $\mathcal{B}_{\text{noise}} = \mathrm{tr}(H\Sigma)/(G^\top H G)$ se reserva como referencia teórica: requiere productos Hessiano-vector y se descarta por coste.

**Entradas.** Per-sample gradients sobre $B$ muestras del dataset, $\{g_i = \nabla_W \ell(x_i, y_i)\}_{i=1}^B$, a partir de los cuales se construyen el estimador de la media $G = \tfrac{1}{B}\sum_i g_i$ y el estimador de la covarianza $\Sigma = \mathrm{Cov}[g_i]$. Opcionalmente, si se quiere comparar con la forma exacta, la Hessiana $H$ se invoca sólo a través de productos $Hv$ vía Hutchinson. En la práctica, $\mathrm{tr}(\Sigma)$ y $\|G\|^2$ se computan sin materializar $\Sigma$:

$$\mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|G\|^2} = \frac{\tfrac{1}{B}\sum_{i=1}^B \|g_i\|^2 - \|G\|^2}{\|G\|^2}.$$

Conviene notar que esta forma con la media empírica $G = \tfrac{1}{B}\sum_i g_i$ es **sesgada**: el numerador estima $(1 - 1/B)\,\mathrm{tr}(\Sigma)$ en vez de $\mathrm{tr}(\Sigma)$, esto es, sesgada a la baja por el factor $(B-1)/B$; la ruta insesgada es aplicar la corrección $1/(B-1)$ (factor $B/(B-1)$) o usar el estimador de dos batches. Si el coste per-sample es prohibitivo, sigue siendo válido el estimador de dos batch sizes $B_{\text{small}} < B_{\text{big}}$ del apéndice A.1 del paper, con $\widehat{\|G\|^2} = (B_{\text{big}}\|G_{\text{big}}\|^2 - B_{\text{small}}\|G_{\text{small}}\|^2)/(B_{\text{big}} - B_{\text{small}})$ y $\widehat{\mathrm{tr}(\Sigma)} = (\|G_{\text{small}}\|^2 - \|G_{\text{big}}\|^2)/(1/B_{\text{small}} - 1/B_{\text{big}})$, promediados con EMA antes de dividir.

**Granularidad.** Una medición por época durante toda la corrida; cada $K$ pasos ($K \in [100, 500]$) en la ventana temprana (primeras 5-10 épocas), donde la trayectoria de $\mathcal{B}_{\text{simple}}$ es más informativa. La frecuencia debe bastar para capturar el crecimiento monótono observado en el paper (uno o dos órdenes de magnitud a lo largo del entrenamiento).

**Estructural.** Se registra un escalar global $\mathcal{B}_{\text{simple}}$ y, adicionalmente, una versión por capa $\mathcal{B}_{\text{simple}}^{(\ell)} = \mathrm{tr}(\Sigma^{(\ell)})/\|G^{(\ell)}\|^2$ restringiendo $g_i$ a los parámetros del módulo $\ell$. La heterogeneidad inter-capa señala diferencias de dinámica que el escalar agregado oculta.

**Coste.** Per-sample grads sobre $B$ muestras: vía `torch.func.vmap` (o `functorch`) en una sola pasada vectorizada, o vía microbatching ($B$ backward passes secuenciales) cuando la memoria es ajustada. La variante de dos batch sizes añade sólo dos backward passes adicionales por medición y se reserva si $B$ es demasiado grande para vmap.

**Trucos.** $\mathcal{B}_{\text{simple}}$ es barato porque elude HVPs por completo. Si se desea $\mathrm{tr}(H\Sigma)$ para validar la versión exacta, basta con un estimador de Hutchinson: muestrear $v$ Rademacher y promediar $v^\top H \Sigma v$ sobre per-sample grads, sin formar nunca $H$.

**Claves de logging.** El código emite `noise_scale/simple` (el cociente) y `noise_scale/tr_sigma` (el numerador $\widehat{\mathrm{tr}}(\Sigma)$ plug-in) por época. La clave `noise_scale/noise` queda **reservada** para la variante exacta con Hessiana, descartada por coste, y `noise_scale/per_layer/{name}` para v2 (no se emite). De las normas intermedias, $\|\bar g\|^2$ es recuperable como `tr_sigma / simple`; el resto no se persiste en v1.

**Interpretación de la señal.** Conviene fijar la convención porque, a diferencia del resto de métricas coseno del TFG, el noise scale **no es monótono en calidad**: la lectura es diagnóstica, no del tipo "más alto = mejor" ni "más bajo = mejor". En `noise_scale/simple` un $\mathcal{B}_{\text{simple}}$ alto significa que el gradiente medio es pequeño respecto a la dispersión per-sample y, por tanto, que SGD tolera bien batches grandes (el batch crítico $\mathcal{B}_{\text{crit}}$ es alto y el paralelismo de datos rinde). Un $\mathcal{B}_{\text{simple}}$ bajo indica que cada ejemplo aporta señal coherente y que batches grandes apenas mejoran el paso de optimización (rendimientos decrecientes a partir de $\mathcal{B}_{\text{crit}}$). La métrica no es un proxy directo de generalización ni de velocidad de convergencia: informa del régimen de paralelismo en el que se entrena, no de si el entrenamiento va "bien". En `noise_scale/noise` la lectura es idéntica, sólo que con la ponderación Hessiana exacta y validándose puntualmente frente a la versión simple. En `noise_scale/per_layer/{name}` la misma convención aplica por módulo: capas con $\mathcal{B}_{\text{simple}}^{(\ell)}$ marcadamente mayor que la media señalan dinámicas locales más ruidosas y suelen ser las que dictan el techo efectivo de paralelización. Operativamente, el comportamiento esperado es que $\mathcal{B}_{\text{simple}}$ **crezca** uno o dos órdenes de magnitud a lo largo del entrenamiento (porque $\|G\|^2$ cae al acercarse a un mínimo mientras $\mathrm{tr}(\Sigma)$ se mantiene aproximadamente constante), de modo que un schedule de batch size creciente proporcional a $\sqrt{\mathcal{B}_{\text{simple}}}$ extrae más eficiencia que uno fijo. Por construcción la trayectoria importa más que el valor puntual, y por ruido del estimador conviene leer la curva en log-scale tras EMA sobre numerador y denominador antes de dividir.

**Avisos.** El estimador es ruidoso por construcción: con $B$ pequeño la varianza muestral de $\mathrm{tr}(\Sigma)$ es alta y el cociente puede oscilar drásticamente, por lo que conviene promediar (EMA o ventanas) numerador y denominador por separado antes de dividir. Cuando $\|G\|^2$ es pequeño (cercanía a un mínimo, ramas anti-correladas) el denominador domina la inestabilidad: registrar en log-scale, fijar un suelo numérico y reportar la trayectoria en lugar de valores puntuales. El estimador de dos batch sizes puede dar $\widehat{\mathrm{tr}(\Sigma)} < 0$ cuando el ruido domina; la EMA es el remedio estándar.

**Pseudocódigo.**

```python
def noise_scale_simple(model, loss_fn, batch):
    # batch: (X, y) con B = len(X) muestras
    per_sample_grads = vmap(grad(per_sample_loss))(model.params, batch)
    # per_sample_grads: (B, P) flattened
    G = per_sample_grads.mean(dim=0)              # (P,)
    sq_norms = (per_sample_grads ** 2).sum(dim=1) # (B,)
    G_sq = (G ** 2).sum()
    tr_Sigma = sq_norms.mean() - G_sq             # tr(Cov[g_i]); sesgado a la baja (B-1)/B con la media empirica, corregir con B/(B-1) si se quiere insesgado
    return (tr_Sigma / G_sq).item(), G_sq.item(), tr_Sigma.item()

# Logging por época: aplicar EMA sobre tr_Sigma y G_sq antes de dividir.
# Per-layer: repetir el cálculo restringiendo per_sample_grads a parámetros del módulo.
```

**Consideraciones para el TFG.** $\mathcal{B}_{\text{simple}}$ predice el batch size crítico y, por extensión, la paralelizabilidad esperada del entrenamiento, lo que la convierte en variable explicativa natural de la eficiencia de cómputo. Al crecer uno o más órdenes de magnitud durante el entrenamiento, conviene registrar la trayectoria entera —no un valor puntual— en log-scale. El cómputo por capa es informativo y conceptualmente coherente con la normalized variance (NGV) de Faghri et al., con la que mantiene la relación $\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$. La relación es exacta en esperanza por la Eq. 2.2 ($\mathrm{cov}(G_{\text{est}}) = \Sigma/B$) y la linealidad de la covarianza con muestreo i.i.d., sin necesidad de invocar el CLT. Si la redundancia se confirma en el pilot, una de las dos puede descartarse.

## Discrepancias detectadas

La versión previa del documento mezclaba dos derivaciones del estimador (per-sample vs. dos batch sizes) sin distinguir su rigor estadístico. La forma per-sample con la media empírica ($\mathcal{B}_{\text{simple}} = (\sum_i \|g_i\|^2/B - \|G\|^2)/\|G\|^2$) es en realidad el **estimador naive sesgado** (a la baja por $(B-1)/B$, ver arriba), mientras que el estimador de dos batch sizes ($B_{\text{small}}, B_{\text{big}}$) del apéndice A.1 es el **insesgado**. Esta versión adopta el per-sample con `vmap`/microbatching como ruta principal por su simplicidad, reservando el estimador de dos batch sizes como alternativa de bajo coste e insesgada. Las notas previas reportaban $\mathcal{B}_{\text{simple}} \in [20, 10^7]$; ese es de hecho el único rango que el paper cita textualmente (p. 9, y para $\mathcal{B}_{\text{crit}}$). El rango 1k–1M para tareas estándar es una **interpretación de la Figura 4** (sus clusters centrales), no una cita del paper, y los extremos (autoencoder SVHN y Dota 5v5) deben leerse como casos límite.

## Notes

Es un ejemplo de cómo una métrica relacionada con el ruido del gradiente permite seleccionar un valor para un hiperparámetro —en este caso el batch size— aumentando el paralelismo y reduciendo tiempos sin perjudicar el rendimiento. Los autores parten de la idea de que existe un batch size grande que no degrada el rendimiento, pero ese umbral varía entre problemas, y proponen el noise scale como instrumento para localizarlo a priori.

### Uso en el TFG

La métrica que origina es `gns_simple` (familia varianza estocástica), una de las tres métricas de varianza del `METRIC_REGISTRY` cerrado del TFG. Se utiliza como proxy barato medido en entrenamiento temprano (ventanas 5/10/25/50% de épocas) sobre el gradiente bruto $\nabla L$. No se optimiza el batch size: se mide su trayectoria y se correlaciona (Spearman) con generalización y eficiencia, con cadencia por época. La fórmula clave es $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma)/\|G\|^2$, estimada sin HVPs por la **ruta per-sample** vía `vmap` sobre el probe fijo ($M = 256$), que es la implementada en `src/metrics/gns_simple.py`. El estimador de dos batch sizes del apéndice A.1 ($B_{\text{small}}, B_{\text{big}}$) queda como atajo de reserva si $B$ creciera demasiado para `vmap`. Menor $\mathcal{B}_{\text{simple}}$ implica menos ruido relativo del gradiente; su vínculo con la eficiencia es más sutil que el de NGV, dado que McCandlish lo liga al batch size crítico $\mathcal{B}_{\text{crit}}$, no directamente a épocas-a-umbral. La versión exacta `gns_exact` (ponderada por Hessiana, $\mathrm{tr}(H\Sigma)/(G^\top H G)$) se descarta por el coste de los HVPs y porque el paper muestra que `gns_simple` la aproxima bien. La métrica crece 1-2 órdenes de magnitud durante el entrenamiento ($\|G\| \downarrow$, $\mathrm{tr}(\Sigma)$ estable), por lo que conviene loguear en log-scale y registrar la trayectoria; el estimador puede dar $\widehat{\mathrm{tr}(\Sigma)} < 0$ cuando el ruido domina, lo que se mitiga con EMA opcional sobre numerador y denominador por separado antes de dividir. La relación $\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$ es exacta en esperanza (Eq. 2.2 + linealidad de la covarianza; no requiere CLT), con Spearman esperado $> 0.9$ frente a `normalized_variance`; la redundancia se valida en el pilot y, si se confirma, se descarta la versión más cara.

## Papers relacionados

- [[A Study of Gradient Variance in Deep Learning]] — misma familia (varianza); NGV $\approx \mathcal{B}_{\text{simple}}/B$ por CLT, redundancia directa a validar.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — misma familia (varianza); GSNR es SNR por parámetro, recíproco conceptual del ruido relativo que mide $\mathcal{B}_{\text{simple}}$.
- [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]] — justificación teórica del eje varianza (SVRG reduce explícitamente la varianza del estimador que aquí se cuantifica).
- [[Disparity Between Batches as a Signal for Early Stopping]] — mismo problema (proxy barato de gradiente para predecir generalización); GD también usa pares de batches independientes.
- [[Speedy Performance Estimation for Neural Architecture Search]] — TSE-EMA es el baseline de eficiencia del TFG; predicción temprana de rendimiento, mismo objetivo que el noise scale.
- [[Adam - A Method for Stochastic Optimization]] — sweep de optimizadores (Adam vs SGD); el paper observa que el LR óptimo de Adam sigue power-law con $\alpha \in [0.5, 1]$ frente a $\alpha = 1$ de SGD/momentum.
- [[RMSProp - Divide the gradient by a running average of its recent magnitude]] — usado en los experimentos de RL (A2C) del paper; $\hat v_t$ como segundo momento no centrado motiva la lectura SNR del gradiente.

## Otros papers interesantes a revisar

- **Don't Decay the Learning Rate, Increase the Batch Size** (Smith, Kindermann, Le, 2018) — formaliza el schedule de batch size creciente que McCandlish solo esboza (variar $B \propto \sqrt{\mathcal{B}_{\text{simple}}}$); relevante para interpretar el crecimiento del noise scale. arXiv:1711.00489.
- **Three Factors Influencing Minima in SGD** (Jastrzębski et al., 2018) — relaciona ruido del gradiente, ratio LR/batch y temperatura de SGD ($T = \epsilon/\epsilon_{\max}$), concepto central de la sección de temperatura del paper. arXiv:1711.04623.
- **Measuring the Effects of Data Parallelism on Neural Network Training** (Shallue et al., 2019) — estudio empírico exhaustivo de la curva pasos-vs-batch que valida y matiza el frente de Pareto hiperbólico de McCandlish. arXiv:1811.03600.
- **On the Relation Between the Sharpest Directions of DNN Loss and the SGD Step Length** (Jastrzębski et al., 2019) — conecta la Hessiana (presente en $\mathcal{B}_{\text{noise}}$) con el LR efectivo; ayuda a justificar cuándo la aproximación $H \propto I$ de `gns_simple` es razonable. arXiv:1807.05031.
