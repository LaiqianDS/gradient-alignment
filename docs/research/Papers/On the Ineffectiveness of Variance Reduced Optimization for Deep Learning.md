---
authors:
  - Aaron Defazio
  - Léon Bottou
year: 2019
status: to-read
relevance: high
url: https://arxiv.org/pdf/1812.04529
tfg_role:
  - related-work
tfg_note: "Muestra empíricamente que SVRG no reduce la varianza en redes modernas (la ratio de varianza SVRG/SGD supera 1). Cierra el triángulo del eje varianza con SVRG y Faghri: la varianza es señal diagnóstica, no un objetivo a minimizar."
---

# On the Ineffectiveness of Variance Reduced Optimization for Deep Learning

## Summary

### Contextualización

El trabajo se sitúa entre la teoría de la optimización estocástica con reducción de varianza (SVR) y la práctica del entrenamiento de redes neuronales profundas modernas. Métodos como SVRG (Johnson y Zhang, 2013), SAGA (Defazio et al., 2014a) y sus variantes —SDCA, MISO, Finito, SAG, Catalyst, SARAH— han demostrado convergencia lineal para problemas suaves y fuertemente convexos mediante el uso de *control variates* sobre el estimador estocástico del gradiente. La técnica del control variate es clásica: dado un estimador $X$ de $\mathbb{E}[X]$ y una variable correlacionada $Y$ con esperanza conocida, $Z = X - Y + \mathbb{E}[Y]$ es insesgado y tiene varianza menor que $X$ siempre que $\mathrm{Var}[Y] \le 2\,\mathrm{Cov}[X, Y]$. Aplicado al problema canónico $f(w) = \frac{1}{n}\sum_{i=1}^n f_i(w)$, SVRG mantiene un *snapshot* $\tilde{w}$ con su gradiente completo $f'(\tilde{w})$ y aplica la actualización $w_{k+1} = w_k - \gamma[f'_i(w_k) - f'_i(\tilde{w}) + f'(\tilde{w})]$, eliminando asintóticamente la varianza cuando $w_k, \tilde{w} \to w_*$. Resultados recientes han extendido este marco al no convexo (Allen-Zhu y Hazan, 2016; Reddi et al., 2016) y a problemas tipo *saddle point* (Balamurugan y Bach, 2016). La cuestión abierta —y central del paper— es si esta maquinaria, exitosa en problemas convexos y en benchmarks pequeños tipo MNIST, ofrece alguna ventaja en el entrenamiento real de CNN profundas. El hueco que se cubre es estrictamente empírico: cuantificar cuánta reducción de varianza efectiva se logra en problemas modernos y, sobre todo, mostrar que la respuesta es, en la práctica, ninguna útil.

### Aportación

La aportación principal es negativa y empírica. Los autores identifican y resuelven tres complicaciones prácticas que rompen la estructura *finite-sum* asumida por SVRG en deep learning: *data augmentation* mediante *transform locking*, *batch normalization* mediante *BN reset* y *dropout* mediante reutilización de semillas. A continuación miden directamente la ratio de varianza del estimador SVRG respecto al de SGD en cuatro arquitecturas (LeNet, ResNet-18 reducido, ResNet-110, DenseNet-40-36) sobre CIFAR-10, mostrando que para modelos grandes la varianza *aumenta* en lugar de reducirse durante la mayoría de cada época. El fallo se explica identificando la **distancia iterada** desde el snapshot como factor dominante, no la curvatura (Figura 3). Las variantes *streaming* tipo SCSG (Lei y Jordan, 2017) no mejoran sobre SVRG estándar pese a su capacidad teórica de manejar data augmentation. Los experimentos de fine-tuning en ImageNet (ResNet-50, DenseNet-169) demuestran que activar SVRG en cualquier punto del entrenamiento —desde la época 0, 20, 40, 60 u 80— no produce mejora alguna sobre SGD con momentum.

### Metodología

El estudio se articula en dos niveles. En el nivel de implementación, los autores documentan las tres correcciones necesarias para que SVRG no diverja en entornos modernos. El *transform locking* responde a que, si la transformación de data augmentation aplicada a un ejemplo $x_i$ en el paso de snapshot difiere de la aplicada en los pasos posteriores, el control variate pierde correlación y la varianza se incrementa. La solución es cachear la transformación $T_i$ usada en el snapshot y reutilizarla durante la época siguiente; la Figura 1 lo confirma al mostrar que sin locking la varianza es uniformemente peor. El *BN reset* aborda el hecho de que la dependencia de la pérdida sobre el mini-batch que introduce batch normalization rompe la estructura finite-sum: al computar gradientes en dos puntos distintos $w_k$ y $\tilde{w}$, la media móvil exponencial de las estadísticas de BN —$m_{\text{EMA}} = \frac{9}{10}m_{\text{EMA}} + \frac{1}{10}m$ por defecto en PyTorch— promedia activaciones inconsistentes y causa divergencia. La solución propuesta es almacenar temporalmente las estadísticas antes del paso en $\tilde{w}$ y restaurarlas después, manteniendo la red en modo *train* en ambos puntos. El tratamiento de *dropout* reutiliza la semilla del patrón por *data-point* para evitar que el patrón aleatorio descorrelacione snapshot e iterado actual.

En el nivel de medición, se calcula directamente la varianza del estimador SVRG y se reporta su ratio respecto a la del estimador SGD, usando el dataset completo para minimizar ruido. Una ratio menor que uno indica reducción de varianza; ratios alrededor de dos implican que el control variate ha quedado descorrelacionado con el gradiente estocástico, **aumentando** la varianza. Los autores argumentan que para que SVRG sea rentable la ratio debe estar por debajo de $1/3$ ("below 1/3 to offset the additional computational costs of the method"). La descomposición en factor 2 por doble gradiente → umbral $<1/2$ → $<1/3$ con snapshots es reconstrucción propia de estas notas, no un argumento explícito del paper. La medición se realiza en múltiples puntos *dentro* de cada época (2 %, 11 %, 33 %, 100 % de progreso), no solo entre épocas, capturando la dinámica intra-época, que es crítica.

El análisis del fallo descompone la varianza del estimador en dos factores: la distancia iterada $\|w_k - \tilde{w}\|$ y la constante de Lipschitz empírica $\|\frac{1}{|S_i|}\sum_{j\in S_i}[f'_j(w_k) - f'_j(\tilde{w})]\| / \|w_k - \tilde{w}\|$. La Figura 3 muestra que la curvatura empírica es similar entre LeNet y DenseNet, pero la distancia iterada crece mucho más rápido en DenseNet, identificando la **velocidad de movimiento del iterado** como factor causal del fallo. Para las variantes *streaming* se considera la formulación genérica de streaming SVRG (Frostig et al. 2015; Lei y Jordan 2017), que sustituye el snapshot full-batch por un *mega-batch* de tamaño $B$ típicamente $10\text{–}32\times$ el mini-batch: $w_{k+1} = w_k - \gamma[\frac{1}{|S_k|}\sum_{i\in S_k}(f'_i(w_k) - f'_i(\tilde{w})) + \tilde{g}]$, con $\tilde{g}$ el gradiente del mega-batch. SCSG es el refinamiento que además muestrea los mini-batches interiores desde el propio mega-batch. Las familias adicionales —SAGA-like, dual, Catalyst, SARAH— se descartan por motivos análogos: SAGA almacena $n$ gradientes, lo que resulta inviable a gran escala; los métodos duales no aplican al caso no convexo; Catalyst no alcanza la mejor tasa en no convexo y no se adapta bien a modelos no suaves tipo ReLU; y SARAH++ (Nguyen et al., 2019) sufre acumulación de error en el bucle interior.

### Datasets y modelos

Setup completo (datasets × modelos) en [[Corpus]].

### Métricas

El paper reporta la **ratio de varianza SVRG/SGD** a múltiples porcentajes de progreso intra-época (Figura 2) como métrica central del análisis, junto con la **distancia iterada** $\|w_k - \tilde{w}\|$ y la **curvatura empírica** (Figura 3) para diagnosticar el origen del fallo. Se complementa con la **varianza del paso streaming** en función de iteraciones desde el snapshot (Figura 4), el **test error** comparando SGD, SVRG y SCSG en CIFAR-10 (LeNet, DenseNet-40-36, ResNet-110) e ImageNet (ResNet-18), y el **test error post-fine-tuning** activando SVRG desde diferentes épocas en ResNet-50 y DenseNet-169 (Figura 6).

### Conclusiones

Los hallazgos son consistentemente negativos para SVRG en deep learning real. La reducción de varianza resulta inexistente o efímera: en el modelo pequeño (LeNet) la varianza SVRG es $2\text{–}4\times$ menor que la de SGD durante la fase inicial, mientras que en los modelos grandes (DenseNet-40-36 y ResNet-110) la varianza SVRG es *mayor* que la de SGD durante la mayor parte de cada época hasta el primer LR drop en la época 150. En estos modelos grandes, incluso al 2 % de progreso intra-época —el punto más favorable, justo tras el snapshot—, la reducción de varianza es como mucho un factor de 2, insuficiente para amortizar el coste del snapshot. La distancia iterada se identifica como causa raíz: la curvatura empírica es similar entre LeNet y DenseNet, pero la distancia recorrida desde el snapshot crece mucho más rápido en DenseNet, descorrelacionando el control variate. Aumentar la frecuencia de snapshots no rescata el método, ya que multiplicaría el coste wall-clock en un orden de magnitud sin mejorar la convergencia por época. La variante streaming SCSG tampoco mejora: en la época 50 reduce la varianza $10\times$ solo en el primer paso post-snapshot y vuelve rápidamente a niveles SGD. Resulta la peor en todas las comparaciones del paper ("The SCSG variant performs the worst in each comparison", §8), sin la excepción de LeNet que versiones previas de estas notas afirmaban. El fine-tuning con SVRG no aporta nada: activarlo desde cualquier época (0, 20, 40, 60, 80) en ResNet-50/ImageNet no mejora el error final ($23.61\%$ baseline SGD frente a $23.65\text{–}28.60\%$ con SVRG fine-tuning). El patrón se replica en DenseNet-169 ($23.22\%$ baseline frente a $23.30\text{–}23.38\%$ desde las épocas 60 y 80). La falta de suavidad no es la causa: usar ELU en lugar de ReLU sólo mejora marginalmente la reducción de varianza en DenseNet. Los autores subrayan además que MNIST no es representativo, porque los resultados positivos previos de SVRG en MNIST con arquitecturas pequeñas no se extienden a problemas reales, lo que sugiere un problema fundamental con MNIST como benchmark de optimización. Finalmente, la reducción de varianza que SVRG sí logra (en LeNet) viene acompañada de fuerte correlación entre pasos consecutivos, lo que explica por qué la teoría de SVRG necesita demostraciones cuidadosas y por qué reducir varianza no se traduce directamente en mejora de convergencia. Esto contrasta con el efecto directo que tienen aumentar el batch size o reducir el learning rate. Los autores concluyen que no descartan el uso de SVR en deep learning, pero la versión naïve falla, y que las direcciones futuras requieren adaptación —SVR aplicado adaptativamente, sobre el learning rate, sobre matrices de escalado, o combinado con Adagrad/Adam.

## Medición y pipeline

*Rol en el pipeline, claves de logging, coste y auditoría: [[Métricas]].*

**Métrica concreta.** El paper no propone una métrica nueva, sino que adopta como instrumento de diagnóstico la **ratio de varianzas** entre el estimador SVRG y el estimador SGD,

$$
\mathrm{ratio} = \frac{\mathrm{Var}[\hat{g}_{\text{SVRG}}]}{\mathrm{Var}[\hat{g}_{\text{SGD}}]},
$$

evaluada bajo el mismo mini-batch sobre el dataset completo (o un subsample estratificado) en *snapshots* fijos del modelo. Como referencia operativa: valores en torno a $2$ señalan el régimen de fallo (control variate descorrelacionado, varianza inflada) y la rentabilidad real del método requiere ratios por debajo de $1/3$. El paper justifica el umbral solo como compensación de los costes computacionales adicionales del método; la descomposición en factor 2 por doble gradiente ($< 1/2$) más snapshots ($< 1/3$) es una reconstrucción interpretativa de estas notas. Como diagnósticos complementarios, los autores siguen la **distancia iterada** $\|w_k - \tilde{w}\|$ entre el iterado actual y el snapshot, y la **curvatura empírica** $\|\tfrac{1}{|S_i|}\sum_{j\in S_i}[f'_j(w_k) - f'_j(\tilde{w})]\| / \|w_k - \tilde{w}\|$, que separan los dos factores que componen la varianza del estimador.

**Entradas.** En cada paso $k$ se necesitan dos gradientes por ejemplo (o por mini-batch) evaluados sobre el *mismo* sample: el gradiente en el iterado actual $g_k = \nabla \ell_i(w_k)$ y el gradiente en el snapshot $g_{\tilde{w}} = \nabla \ell_i(\tilde{w})$. Con ambos se forma el estimador SVRG $\hat{g}_{\text{SVRG}} = g_k - g_{\tilde{w}} + \tilde{\mu}$, con $\tilde{\mu} = f'(\tilde{w})$ precomputado al instalar el snapshot. La varianza coordenada a coordenada se acumula sobre el dataset completo (o un subsample del orden de $1$–$5\,\%$ estratificado) en ventanas fijas de progreso intra-época.

**Cuándo computar.** La cadencia natural sigue dos escalas: por **snapshot** se reinstala $\tilde{w}$ y se recalcula $\tilde{\mu}$ una vez por época; la **ratio de varianzas** se mide a varios porcentajes de progreso intra-época (2 %, 11 %, 33 %, 100 %), porque medir solo entre épocas oculta el problema: la varianza SVRG se degrada conforme $w_k$ se aleja de $\tilde{w}$, y el grueso de los pasos del entrenamiento ocurre con esa varianza inflada. La **distancia iterada** $\|w_k - \tilde{w}\|$ se loggea cada paso (es $O(|W|)$) y la **curvatura empírica** una vez por ventana, junto con la ratio.

**Granularidad estructural.** Por defecto, agregación global sobre $w$. Opcionalmente, una versión por capa —ratio, distancia y curvatura restringidas al bloque de parámetros de cada layer— para localizar qué partes de la red descorrelacionan antes el control variate, en línea con la observación del paper de que la velocidad de movimiento del iterado, no la curvatura, es el factor causal.

**Coste.** Alto si se evalúa sobre el dataset completo: cada snapshot exige un pase full-batch ($O(n)$ forwards+backwards) y cada medición intra-época otro pase a través del subconjunto elegido. La memoria adicional para $\tilde{w}$ y $\tilde{\mu}$ es $2|W|$. Las dos vías habituales para amortiguar el coste son **subsamplear** el dataset (estratificado por clase, $1$–$5\,\%$) para la estimación de varianza, y usar **mini-batches grandes** en el cómputo de $\tilde{\mu}$ para reducir el ruido sin pagar full-batch.

**Trucos numéricos.** El estimador de varianza coordenada se calcula en *streaming* con el algoritmo de Welford, evitando almacenar todos los gradientes per-sample y manteniendo estabilidad numérica en parámetros con $|W|$ del orden de millones. La traza de la covarianza, $\mathrm{tr}(\Sigma) = \sum_j \mathrm{Var}[g_j]$, basta para reportar una varianza escalar agregada; reservar la covarianza por capa solo si la versión estructural está activa.

**Integración en el pipeline.** Pseudocódigo orientativo:

```python
# Una vez por época: instalar snapshot
snapshot_w = clone(model.parameters())
mu_tilde   = full_or_subsample_gradient(model, loader)        # ~g(w_tilde)

# Bucle intra-época
for k, batch in enumerate(loader):
    # Gradiente en el iterado actual
    g_k        = grad(model, batch)
    # Gradiente del snapshot sobre el MISMO mini-batch
    with swap_params(model, snapshot_w):
        g_tilde = grad(model, batch)

    g_svrg = g_k - g_tilde + mu_tilde
    step(model, g_svrg, lr)

    # Diagnóstico: distancia iterada cada step
    log("svrg_diag/iterate_distance", l2_norm(params(model) - snapshot_w))

    # Diagnóstico: ratio y curvatura en ventanas (2%, 11%, 33%, 100%)
    if at_window_boundary(k):
        var_svrg = streaming_variance(g_svrg over probe_subsample)
        var_sgd  = streaming_variance(g_k    over probe_subsample)
        log("svrg_diag/var_ratio", var_svrg / var_sgd)
        log("svrg_diag/curvature", l2_norm(g_k - g_tilde) / l2_norm(params(model) - snapshot_w))
```

**Logging.** Las claves principales son `svrg_diag/var_ratio` (escalar global, una entrada por ventana intra-época), `svrg_diag/iterate_distance` (escalar global, una por step) y `svrg_diag/curvature` (escalar global, una por ventana). Si se habilita la granularidad por capa, replicar cada clave con sufijo `/layer_<name>`.

**Avisos.** El fallo característico de SVRG en deep learning aparece justamente como ratio mayor que uno: cuando el snapshot está suficientemente descorrelacionado del iterado actual, el control variate inyecta varianza en lugar de reducirla. Tres causas habituales: data augmentation sin *transform locking*, batch normalization sin *BN reset* y dropout sin reutilización de semilla rompen la estructura finite-sum y degradan la ratio. Cualquier intento de usar SVRG como baseline debe aplicar estas tres correcciones antes de computar la ratio; en caso contrario el diagnóstico mide implementación, no algoritmo.

**Consideraciones.** En el contexto del TFG el rol del paper es de **refuerzo empírico del eje varianza**, no de métrica del registry. La ratio $\mathrm{Var}[\hat{g}_{\text{SVRG}}]/\mathrm{Var}[\hat{g}_{\text{SGD}}]$ no se loggea porque requiere ejecutar SVRG, fuera del scope de las corridas SGD/Adam estándar. Sí es trivialmente integrable la **distancia iterada** $\|w_k - w_{k-K}\|$ entre checkpoints consecutivos como proxy de movimiento del iterado, conceptualmente cercano a `weight_drift`, si en algún experimento se quisiera correlacionar la NGV o la GSNR con la velocidad de los parámetros. Junto con SVRG (predecesor teórico) y Faghri (NGV creciente), este paper cierra el triángulo lógico que justifica el TFG: SVRG promete reducir varianza bajo *strong growth*; Faghri muestra empíricamente que la varianza normalizada *crece* en CIFAR-10/ImageNet; Defazio y Bottou demuestran que, aplicado naïvemente, SVRG produce varianza *mayor* que SGD en modelos grandes. La conclusión transversal —la varianza es una señal diagnóstica, no un objetivo a minimizar directamente— es exactamente la postura que el TFG adopta.

## Notes

### Uso en el TFG

- **Rol: refuerzo empírico del eje varianza, NO métrica del registry.** Junto con SVRG (predecesor teórico) y Faghri (NGV creciente), forma el triángulo de papers que justifican por qué el TFG mide la varianza en lugar de reducirla. La cadena lógica es: SVRG promete reducir varianza $\Rightarrow$ asume *strong growth condition* $\Rightarrow$ Faghri muestra empíricamente que la varianza normalizada *crece* en CIFAR-10/ImageNet $\Rightarrow$ Defazio y Bottou muestran que, aplicado naïvemente, SVRG produce varianza *mayor* que SGD en modelos grandes. Conclusión: la varianza es una señal diagnóstica, no un objetivo a minimizar directamente.
- **Hallazgo clave a citar.** Para DenseNet-40-36 y ResNet-110 la ratio $\mathrm{Var}[\hat{g}_{\text{SVRG}}] / \mathrm{Var}[\hat{g}_{\text{SGD}}] > 1$ durante la mayor parte de cada época hasta el primer LR drop. Incluso al 2 % de progreso intra-época la reducción es a lo sumo $2\times$, insuficiente para amortizar el coste del snapshot. Es la evidencia empírica más fuerte disponible contra la aplicabilidad de SVR en deep learning moderno.
- **Causa raíz identificada.** La distancia iterada $\|w_k - \tilde{w}\|$ crece más rápido en modelos grandes; la curvatura empírica es similar entre LeNet y DenseNet. Esto sugiere que **la velocidad de movimiento del iterado** —no la geometría local— es lo que rompe SVRG. Relevante para el TFG porque conecta varianza con trayectoria: si en futuras métricas se quiere capturar movimiento del iterado, la norma $\|w_k - w_{k-K}\|$ entre checkpoints es un proxy directo y trivial de loggear.
- **Aviso metodológico sobre MNIST.** Los autores afirman explícitamente que el éxito de SVRG en MNIST no se extiende a problemas reales y que esto refleja "un problema fundamental con MNIST como baseline de optimización". Refuerza la decisión del TFG de **no usar MNIST** como benchmark principal y centrarse en CIFAR-10 e ImageNet/Tiny-ImageNet.
- **Streaming variants también fallan.** SCSG (Lei y Jordan, 2017) reduce la varianza $10\times$ solo en el primer paso post-mega-batch y vuelve rápidamente a niveles SGD. La conclusión es robusta a la elección de variante.

## Papers relacionados

- [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]] — SVRG, predecesor teórico directo cuya promesa de convergencia lineal este paper refuta empíricamente en deep learning real. Cierra el argumento: SVRG funciona en convexo, asume *strong growth*, falla en redes profundas.
- [[A Study of Gradient Variance in Deep Learning]] — complementario empírico: Faghri muestra que la NGV *crece* durante el entrenamiento en CIFAR-10/ImageNet (violando *strong growth*), y este paper muestra que SVRG, que asume lo contrario, falla en consecuencia. Ambos justifican el eje varianza del TFG.
- [[An Empirical Model of Large-Batch Training]] — la *gradient noise scale* $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma)/\|G\|^2$ es la métrica varianza-normalizada que sí se loggea; este paper aporta el contraste de que reducir varianza algorítmicamente (SVRG) no funciona, mientras que medirla (GNS, NGV) sí informa sobre la dificultad de optimización.
- [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]] — GSNR es el SNR por parámetro, conceptualmente el inverso de la varianza que SVRG fracasa en reducir. Mismo eje varianza, otra forma de agregación.

## Otros papers interesantes a revisar

- **SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient** (Nguyen et al., ICML 2017) — discutido en el paper como alternativa con mejor tasa teórica en no convexo; los autores no logran convergencia fiable por acumulación de error en el bucle interior. Relevante si se quiere matizar el panorama de variantes de SVR. arXiv:1703.00102
- **Less than a Single Pass: Stochastically Controlled Stochastic Gradient (SCSG)** (Lei y Jordan, AISTATS 2017) — el método streaming evaluado y desechado por los autores; útil si se quiere entender por qué los enfoques de *mega-batch* no rescatan SVRG. arXiv:1609.03261
- **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** (Goyal et al., 2017) — citado como evidencia de que reducir varianza vía aumento de batch size sí permite proporcionalmente aumentar el learning rate hasta $30\times$, contrastando con la ineficacia de SVR. Conexión directa con la motivación del eje varianza. arXiv:1706.02677
- **A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks** (Şimşekli et al., 2019) — el ruido del gradiente en deep learning no es gaussiano sino de cola pesada; explica por qué los supuestos clásicos de SVR fallan y refuerza la lectura empírica de Defazio y Bottou. arXiv:1901.06053
- **On the Insufficiency of Existing Momentum Schemes for Stochastic Optimization** (Kidambi et al., ICLR 2018) — citado por los autores en la observación de que SVRG induce correlación fuerte entre pasos consecutivos, similar a momentum; relevante para entender por qué reducir varianza no se traduce en mejora de convergencia. arXiv:1704.08227
