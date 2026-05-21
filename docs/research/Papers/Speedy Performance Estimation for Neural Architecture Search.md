---
authors:
  - Robin Ru
year: 2021
status: to-read
relevance: high
last_review: 2026-05-07
url: https://proceedings.neurips.cc/paper/2021/hash/2130eb640e0a272898a51da41363542d-Abstract.html
---

# Speedy Performance Estimation for Neural Architecture Search

## Summary

**Contextualización.** La búsqueda de arquitecturas neuronales (Neural Architecture Search, NAS) automatiza el diseño de redes profundas pero arrastra un cuello de botella central: estimar de forma fiable el rendimiento de generalización de cada arquitectura candidata. El protocolo tradicional (full training) entrena cada modelo durante el presupuesto completo de épocas y mide su precisión en validación, lo que requiere del orden de miles de GPU-días y resulta inasumible bajo presupuestos limitados. Las alternativas existentes presentan compromisos importantes. El early-stopping aproxima el rendimiento entrenando pocas épocas, pero el ranking relativo entre arquitecturas en validación temprana correlaciona pobremente con el ranking final tras entrenamiento completo. La extrapolación de curvas de aprendizaje (learning curve extrapolation) ajusta un modelo de regresión (procesos gaussianos, redes neuronales bayesianas, ν-SVR, etc.) sobre las primeras épocas, pero exige cientos de arquitecturas totalmente entrenadas como datos de entrenamiento del surrogate y la optimización de sus propios hiperparámetros. El weight sharing utilizado en NAS one-shot reduce el coste compartiendo pesos en una supernet, aunque infraestima sistemáticamente el rendimiento real y produce rankings poco fiables, sobre todo entre las mejores arquitecturas. Finalmente, los estimadores zero-cost (JacCov, SNIP, SynFlow) son baratos pero inconsistentes entre tareas y no escalan con presupuestos mayores.

**Aportación.** Los autores proponen el Training Speed Estimator (TSE), un estimador model-free, sin hiperparámetros que ajustar y sin necesidad de un surrogate, basado únicamente en sumar las pérdidas de entrenamiento (training losses) recogidas durante el optimizador SGD. La intuición es que las arquitecturas que aprenden rápidamente acumulan menos pérdida bajo la curva de entrenamiento y, por la conexión teórica entre velocidad de entrenamiento y generalización, tienden a generalizar mejor. El estimador se obtiene gratis durante la propia optimización (los mini-batches ya se evalúan) y proporciona un ranking fiable con un presupuesto de entrenamiento muy reducido.

**Metodología.** La motivación teórica se apoya en tres líneas. (1) Las cotas basadas en estabilidad de SGD (Hardt et al.) y las cotas tipo NTK (Arora et al., Cao y Gu) acotan la brecha de generalización en función del número de pasos de optimización, prediciendo que modelos que entrenan más rápido tienen menor error de generalización en el peor caso. (2) Las cotas de generalización information-theoretic (Negrea et al., Neu) dependen de la varianza de los gradientes durante el entrenamiento, una medida ligada a la velocidad de entrenamiento como aproximación de primer orden. (3) Lyle et al. demuestran que, en modelos lineales y redes infinitamente anchas con actualización bayesiana, la marginal likelihood —fundamentada como herramienta de selección de modelos— se descompone como una suma de log-verosimilitudes predictivas negativas que recuerdan a las pérdidas en datos nuevos durante un procedimiento de aprendizaje online; maximizarla equivale a minimizar una cota PAC-Bayes, por lo que TSE puede entenderse como una estimación de dicha cota.

Formalmente, sea $\ell$ una función de pérdida, $f_\theta$ la red con parámetros $\theta$ y $\theta_{t,i}$ los parámetros tras $t$ épocas e $i$ mini-batches. Tras entrenar $T$ épocas:

$$\text{TSE} = \sum_{t=1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(f_{\theta_{t,i}}(\mathbf{X}_i), \mathbf{y}_i)$$

donde $B$ es el número de mini-batches por época. Se proponen dos variantes para mitigar el ruido de las primeras épocas (inestables y poco informativas) y la saturación de las últimas (cercanas a pérdida cero). **TSE-E** trata las primeras épocas como burn-in y suma sólo las últimas $E$:

$$\text{TSE-E} = \sum_{t=T-E+1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$$

**TSE-EMA** aplica una media móvil exponencial con $\gamma = 0.9$, dando más peso a épocas tardías sin descartar las tempranas:

$$\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$$

Los autores justifican usar pérdidas de entrenamiento en lugar de validación: éstas capturan la generalización de los gradientes desde un mini-batch a los siguientes en los datos, mientras que la suma sobre validación se asemeja a una técnica de reducción de varianza alrededor de un mínimo local. La integración en NAS es directa: en métodos query-based (Random Search, Regularised Evolution, Bayesian Optimisation) se reemplaza la precisión de validación con TSE-EMA con $T=10$; en NAS one-shot (RandNAS, FairNAS, MultiPaths) se entrena la supernet, se heredan pesos a cada subred y se entrenan $B$ mini-batches adicionales para calcular TSE; en NAS diferenciable (DARTS, DrNAS) se sustituye el gradiente de la pérdida de validación por el de TSE calculado sobre 100 mini-batches.

**Datasets y modelos.** Se evalúan cuatro espacios de búsqueda: NAS-Bench-201 (NB201, 6.466 arquitecturas únicas en CIFAR-10, CIFAR-100 e ImageNet-16-120), DARTS (5.000 arquitecturas en CIFAR-10 con NAS-Bench-301), ResNet/ResNeXt (50.000 arquitecturas en CIFAR-10) y Randomly Wired Neural Networks (RWNN, 69×8 arquitecturas en Flower102). Para DARTS también se prueban tres set-ups distintos (learning rate inicial, scheduler, batch size) sobre arquitecturas grandes de 20 celdas.

**Métricas.** Calidad del ranking mediante correlación de Spearman entre el ranking predicho y el verdadero ranking en test (también se reporta correlación de Kendall en apéndices). Se mide además la eficiencia de búsqueda: error de test mínimo alcanzado en función del runtime, y precisión media del top-10 de arquitecturas seleccionadas en one-shot. Las baselines incluyen TSE base, TLmini (loss en un único mini-batch), SoVL (sum of validation losses), VAccES (validation accuracy at early epoch), LcSVR (learning curve extrapolation con ν-SVR) y los estimadores zero-cost JacCov, SNIP y SynFlow.

**Conclusiones.** TSE-E (con $E=1$) y TSE-EMA (con $\gamma=0.999$) superan consistentemente a todas las baselines bajo presupuestos $T \leq 0.5\,T_{end}$, manteniéndose competitivas incluso al rankear el top-1% de arquitecturas. TSE-EMA destaca cuando $T$ es muy pequeño, indicando que la dinámica temprana —pese a ser ruidosa— contiene información útil. En NAS query-based, TSE-EMA con $T=10$ acelera Regularised Evolution y Bayesian Optimisation hasta encontrar las mejores arquitecturas en mucho menos runtime que Val Acc con $T=200$ o $T=10$. En one-shot NAS, TSE incrementa la correlación de ranking entre 170% y 300% frente a la precisión de validación a través de RandNAS, FairNAS y MultiPaths. En NAS diferenciable, sustituir el gradiente de la pérdida de validación por el de TSE mitiga el problema de skip-connections que sufre DARTS sobre NB201 y mejora la precisión final. Los autores proponen además un procedimiento sencillo basado en muestrear $N_s$ arquitecturas y detectar la época mínima $T_o$ de overfitting para decidir el presupuesto efectivo (parar TSE en $0.9\,T_o$ y revertir a validación más allá). Como ventaja adicional, el método reduce significativamente el coste computacional y energético de NAS, facilitando su uso bajo presupuestos limitados.

## Medición y pipeline

**Métrica concreta.** El Training Speed Estimator (TSE) de Ru et al. se define como la suma de las training losses promediadas por mini-batch sobre las épocas consideradas. Sea $\ell$ la pérdida, $B$ el número de mini-batches por época y $\theta_{t,i}$ los parámetros tras $t$ épocas e $i$ mini-batches; tras $T$ épocas, la formulación base es $\text{TSE} = \sum_{t=1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(f_{\theta_{t,i}}(\mathbf{X}_i), \mathbf{y}_i)$. La variante **TSE-E** descarta las primeras épocas como burn-in y suma sólo las últimas $E$: $\text{TSE-E} = \sum_{t=T-E+1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$. La variante **TSE-EMA** aplica una media móvil exponencial con factor $\gamma$ (los autores recomiendan $\gamma=0.999$): $\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$.

**Entradas.** Únicamente la training loss por época (o por step, agregada después a nivel de época). No requiere acceso a gradientes, ni conjunto de validación, ni hiperparámetros adicionales más allá de $E$ o $\gamma$.

**Cuándo computar.** Tras cada época se actualiza el acumulador (suma corriente o EMA). Alcanzada la ventana $E$ definida como fracción del presupuesto, se congela el valor y se utiliza como predictor de la eficiencia final de la arquitectura o configuración.

**Coste.** Cero overhead adicional: las pérdidas de mini-batch ya se evalúan durante el paso forward del entrenamiento estándar, por lo que TSE se obtiene como subproducto del bucle de optimización.

**Integración en el pipeline.** Pseudocódigo de referencia:

```python
tse, tse_ema = 0.0, 0.0
gamma = 0.999
for t in range(1, T + 1):
    epoch_loss = train_one_epoch(model, loader)  # promedio por mini-batch
    tse += epoch_loss
    tse_ema = gamma * tse_ema + epoch_loss
    log_scalar("tse", tse, step=t)
    log_scalar("tse_ema", tse_ema, step=t)
```

**Consideraciones.** TSE constituye el baseline metodológico natural de la tesis: cualquier métrica de gradiente propuesta como proxy de eficiencia debe superarlo en correlación de Spearman frente a las métricas de eficiencia (épocas hasta umbral, AUC de test loss, mejor test loss); en caso contrario, el coste adicional de instrumentar gradientes no se justifica. Funciona, por tanto, como sanity check obligatorio. Se recomienda evaluarlo sobre las mismas ventanas que las métricas candidatas —5%, 10%, 25% y 50% del presupuesto total— para asegurar comparabilidad.

**Logging.** Registrar TSE y TSE-EMA acumulados por época como escalares; congelar y persistir su valor en los hitos del 5%, 10%, 25% y 50% del presupuesto para el sweep de correlaciones contra los proxies de eficiencia.

## Notes

### Uso en el TFG

- **Rol — BASELINE PREDICTOR obligatorio, NO métrica del registry**: TSE-EMA es el origen del predictor `tse_ema` y se taggea `metric_kind="baseline"` en `metrics_at_window.parquet` para separarlo de las 10 métricas de gradiente del `METRIC_REGISTRY` (7 alineación + 3 varianza). No es una métrica de gradiente: es un predictor de eficiencia derivado únicamente de la train loss.
- **Fórmula**: $\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t}\,\bar\ell_t$, con $\bar\ell_t$ la train loss media de la época $t$ y $\gamma = 0.999$ (suma ponderada exponencialmente de train losses, da más peso a épocas tardías sin descartar las tempranas). Se registran también `tse` (suma simple) y `tse_e` (E=1) por completitud, pero `tse_ema` es el baseline principal.
- **Coste CERO**: la train loss ya se computa en el forward de cada paso, así que TSE-EMA se obtiene como subproducto del bucle de optimización (acumulador EMA). Sin barrido extra, sin acceso a gradientes, sin val set.
- **Señal**: $\downarrow$ mejor — menor TSE-EMA $\Rightarrow$ la configuración acumula menos pérdida bajo la curva de entrenamiento (converge antes) $\Rightarrow$ mejor eficiencia/generalización esperada.
- **Por qué es CRÍTICO (el listón a batir)**: es el *sanity check* metodológico de la tesis. Cualquier métrica de gradiente propuesta debe **superar a TSE-EMA en correlación de Spearman** frente a los proxies de eficiencia (épocas-a-umbral, AUC de test loss, mejor test loss) sobre las **mismas ventanas** (5/10/25/50% de épocas); si no lo hace, el coste de instrumentar gradientes (barridos batch-grad, per-sample $\nabla L$, Jacobian NTK) **no se justifica**. Sin este baseline gratis la tesis no puede defender que la alineación/varianza de gradientes aporte valor sobre algo que ya se mide.
- **NOTA — paper comparable (delta a argumentar)**: es **uno de los 3 papers más comparables** al TFG (junto a Hölzl 2025 y Forouzesh & Thiran 2021) por atacar el mismo problema —predicción temprana barata de rendimiento. Ru et al. muestran que TSE-EMA supera validation accuracy, learning-curve extrapolation (LcSVR) y zero-cost proxies (JacCov/SNIP/SynFlow) en Spearman $\rho$ bajo presupuestos $T \le 0.5\,T_{end}$. Diferencia de scope: ellos rankean **arquitecturas** en NAS y *optimizan* la búsqueda; el TFG es **correlacional**, no optimiza, y usa TSE-EMA como referencia gratis dentro de un sweep controlado (SGD/Adam × FC/CNN-small/ResNet-18 × MNIST/CIFAR-10/(ImageNet)).

## Papers relacionados

- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — mismo problema (proxy train-time de generalización, barato); uno de los 3 comparables. GWA es métrica del registry y debe batir a este baseline.
- [[Disparity Between Batches as a Signal for Early Stopping]] — mismo problema (señal temprana barata de generalización sin val set); uno de los 3 comparables. `gradient_disparity` es métrica del registry medida contra este baseline.
- [[A Study of Gradient Variance in Deep Learning]] — la motivación teórica de TSE invoca cotas de generalización dependientes de la **varianza del gradiente**; conecta el baseline con la familia varianza (`normalized_variance`).
- [[An Empirical Model of Large-Batch Training]] — comparte el marco "predecir eficiencia de entrenamiento a partir de estadísticos baratos"; el GNS predice batch size crítico, TSE-EMA predice ranking de eficiencia.

## Otros papers interesantes a revisar

- **NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search** (Dong & Yang, 2020) — benchmark tabular (uno de los espacios donde Ru et al. evalúan TSE); referencia estándar para correlaciones ranking-vs-test en NAS. arXiv:2001.00326
- **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah et al., 2021) — proxies de coste cero (snip, grasp, synflow, etc.) que TSE-EMA supera en consistencia entre tareas; competidores directos del baseline. arXiv:2101.08134
- **Neural Architecture Search without Training** (NASWOT, Mellor et al., 2021) — estima rendimiento al inicializar usando la correlación de los mapas lineales (regiones ReLU) entre datos, sin entrenar; otro proxy barato comparable que TSE supera bajo presupuestos mayores. arXiv:2006.04647
- **Fantastic Generalization Measures and Where to Find Them** (Jiang et al., 2020) — estudio empírico a gran escala de medidas de generalización y su correlación con el gap real; marco metodológico para situar TSE-EMA y las métricas de gradiente como proxies competidores. arXiv:1912.02178

## Cited By
