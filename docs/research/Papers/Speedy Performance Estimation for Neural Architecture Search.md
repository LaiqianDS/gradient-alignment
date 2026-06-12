---
authors:
  - Binxin Ru # "Robin" es su alias de correspondencia; la bibliografía debe citar Binxin Ru
year: 2021
status: to-read
relevance: high
url: https://proceedings.neurips.cc/paper/2021/hash/2130eb640e0a272898a51da41363542d-Abstract.html
tfg_role:
  - method
tfg_note: "Origen del baseline `tse_ema` (suma exponencialmente ponderada de la train loss, coste cero). Miembro del nivel 0 junto a val-acc@f y val-loss@f — nunca validado para rankear LRs (caveat §4.2 del propio paper; ver Caveats verificados). Uno de los 3 papers más comparables."
---

# Speedy Performance Estimation for Neural Architecture Search

## Summary

### Contextualización

El trabajo se enmarca en la búsqueda automática de arquitecturas neuronales (Neural Architecture Search, NAS), una línea cuyo cuello de botella central no es la búsqueda en sí sino la estimación fiable del rendimiento de generalización de cada arquitectura candidata. El protocolo canónico (full training) requiere entrenar cada modelo durante el presupuesto completo de épocas y evaluar precisión en validación, lo que escala a miles de GPU-días y resulta inasumible bajo presupuestos limitados. Las alternativas existentes arrastran compromisos importantes. El early-stopping aproxima el rendimiento entrenando pocas épocas, pero el ranking relativo en validación temprana correlaciona pobremente con el ranking final. La extrapolación de curvas de aprendizaje ajusta un surrogate (procesos gaussianos, redes bayesianas, ν-SVR) sobre las primeras épocas, pero exige cientos de arquitecturas totalmente entrenadas como datos de ajuste y la optimización de sus propios hiperparámetros. El weight sharing del NAS one-shot reduce el coste compartiendo pesos en una supernet, aunque infraestima sistemáticamente el rendimiento real y produce rankings poco fiables, sobre todo entre las mejores arquitecturas. Por último, los estimadores zero-cost (JacCov, SNIP, SynFlow) resultan baratos pero inconsistentes entre tareas y no escalan con presupuestos mayores.

### Aportación

La contribución central es el **Training Speed Estimator (TSE)**, un estimador *model-free*, sin hiperparámetros que ajustar y sin necesidad de un surrogate, basado únicamente en sumar las pérdidas de entrenamiento (training losses) recogidas durante el optimizador SGD. La intuición operativa es directa: las arquitecturas que aprenden rápidamente acumulan menos pérdida bajo la curva de entrenamiento y, por la conexión teórica entre velocidad de entrenamiento y generalización, tienden a generalizar mejor. El estimador se obtiene gratis durante la propia optimización —los mini-batches ya se evalúan en el forward— y proporciona un ranking fiable con un presupuesto de entrenamiento muy reducido. Como ejemplo cuantitativo, el ranking por TSE correlaciona con el ranking de test final con Spearman elevado tras unas pocas épocas de entrenamiento, batiendo a precisión de validación, extrapolación de curvas y proxies zero-cost bajo presupuestos $T \leq 0.5\,T_{end}$.

### Metodología

La motivación teórica se apoya en tres líneas convergentes. Primero, las cotas basadas en estabilidad de SGD (Hardt et al.) y las cotas tipo NTK (Arora et al.; Cao y Gu) acotan la brecha de generalización en función del número de pasos de optimización, prediciendo que los modelos que entrenan más rápido tienen menor error de generalización en el peor caso. Segundo, las cotas information-theoretic (Negrea et al.; Neu) dependen de la varianza de los gradientes durante el entrenamiento, una magnitud ligada a la velocidad de aprendizaje como aproximación de primer orden. Tercero, Lyle et al. demuestran que, en modelos lineales y redes infinitamente anchas con actualización bayesiana, la marginal likelihood se descompone como una suma de log-verosimilitudes predictivas negativas que recuerdan a las pérdidas en datos nuevos durante un procedimiento de aprendizaje online; maximizarla equivale a minimizar una cota PAC-Bayes, por lo que TSE puede entenderse como una estimación de dicha cota.

Formalmente, sea $\ell$ una función de pérdida, $f_\theta$ la red con parámetros $\theta$ y $\theta_{t,i}$ los parámetros tras $t$ épocas e $i$ mini-batches. Tras entrenar $T$ épocas, la formulación base es

$$\text{TSE} = \sum_{t=1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(f_{\theta_{t,i}}(\mathbf{X}_i), \mathbf{y}_i),$$

donde $B$ es el número de mini-batches por época. Los autores proponen dos variantes para mitigar el ruido de las primeras épocas (inestables y poco informativas) y la saturación de las últimas (cercanas a pérdida cero). **TSE-E** trata las primeras épocas como burn-in y suma sólo las últimas $E$:

$$\text{TSE-E} = \sum_{t=T-E+1}^{T} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot).$$

**TSE-EMA** aplica una media móvil exponencial con factor $\gamma$ (los autores barren $\gamma \in \{0.9, 0.95, 0.99, 0.999\}$ — robusto en todo el rango, §4.1 — y fijan $\gamma=0.999$; la pareja $\{0.9, 0.999\}$ que loguea el pipeline es elección propia), dando más peso a épocas tardías sin descartar las tempranas:

$$\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t} \frac{1}{B}\sum_{i=1}^{B} \ell(\cdot).$$

La justificación de usar pérdidas de entrenamiento (no de validación) es que éstas capturan la generalización de los gradientes desde un mini-batch a los siguientes en los datos, mientras que la suma sobre validación se asemeja a una técnica de reducción de varianza alrededor de un mínimo local. La integración en NAS es directa: en métodos query-based (Random Search, Regularised Evolution, Bayesian Optimisation) se reemplaza la precisión de validación con TSE-EMA usando $T=10$ épocas; en NAS one-shot (RandNAS, FairNAS, MultiPaths) se entrena la supernet, se heredan pesos a cada subred y se entrenan $B$ mini-batches adicionales para calcular TSE; en NAS diferenciable (DARTS, DrNAS) se sustituye el gradiente de la pérdida de validación por el gradiente de TSE calculado sobre 100 mini-batches.

### Datasets y modelos

Se evalúan cuatro espacios de búsqueda. NAS-Bench-201 (NB201) contiene 6.466 arquitecturas únicas sobre CIFAR-10, CIFAR-100 e ImageNet-16-120. DARTS aporta 5.000 arquitecturas sobre CIFAR-10 mediante NAS-Bench-301. El espacio ResNet/ResNeXt cubre 50.000 arquitecturas en CIFAR-10. Por último, Randomly Wired Neural Networks (RWNN) recoge 69×8 arquitecturas sobre Flower102. Para DARTS se prueban además tres set-ups distintos (learning rate inicial, scheduler, batch size) sobre arquitecturas grandes de 20 celdas.

### Métricas

La calidad del ranking se mide mediante correlación de Spearman entre el ranking predicho y el verdadero ranking en test, complementada con Kendall en los apéndices. Se reporta también la eficiencia de búsqueda como error de test mínimo alcanzado en función del runtime, y la precisión media del top-10 de arquitecturas seleccionadas en one-shot. Las baselines incluyen TSE base, TLmini (loss en un único mini-batch), SoVL (sum of validation losses), VAccES (validation accuracy at early epoch), LcSVR (learning curve extrapolation con ν-SVR) y los estimadores zero-cost JacCov, SNIP y SynFlow.

### Conclusiones

TSE-E con $E=1$ y TSE-EMA con $\gamma=0.999$ superan consistentemente a todas las baselines bajo presupuestos $T \leq 0.5\,T_{end}$, manteniéndose competitivas incluso al rankear el top-1% de arquitecturas. TSE-EMA destaca cuando $T$ es muy pequeño, indicando que la dinámica temprana —pese a ser ruidosa— contiene información útil. En NAS query-based, TSE-EMA con $T=10$ acelera Regularised Evolution y Bayesian Optimisation hasta encontrar las mejores arquitecturas en mucho menos runtime que Val Acc con $T=200$ o $T=10$. En one-shot NAS, TSE incrementa la correlación de ranking entre 170% y 300% frente a la precisión de validación a través de RandNAS, FairNAS y MultiPaths. En NAS diferenciable, sustituir el gradiente de la pérdida de validación por el de TSE mitiga el problema de skip-connections que sufre DARTS sobre NB201 y mejora la precisión final. Los autores proponen además un procedimiento sencillo basado en muestrear $N_s$ arquitecturas y detectar la época mínima $T_o$ de overfitting para decidir el presupuesto efectivo (parar TSE en $0.9\,T_o$ y revertir a validación más allá). Como ventaja adicional, el método reduce significativamente el coste computacional y energético de NAS, facilitando su uso bajo presupuestos limitados.

## Medición y pipeline

**Métrica concreta.** Se adopta el Training Speed Estimator de Ru et al. en sus tres formulaciones complementarias sobre la pérdida de entrenamiento $\ell$ acumulada a lo largo de $T$ épocas. La base es $\text{TSE} = \sum_{t=1}^{T} \tfrac{1}{B}\sum_{i=1}^{B} \ell(f_{\theta_{t,i}}(\mathbf{X}_i), \mathbf{y}_i)$. La variante de burn-in **TSE-E** descarta las primeras épocas y suma sólo las últimas $E$: $\text{TSE-E} = \sum_{t=T-E+1}^{T} \tfrac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$. La variante de media móvil exponencial **TSE-EMA** pondera $\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t} \tfrac{1}{B}\sum_{i=1}^{B} \ell(\cdot)$, con dos elecciones típicas del factor de decaimiento, $\gamma=0.9$ (decaimiento rápido, dominan épocas finales) y $\gamma=0.999$ (decaimiento lento, integra trayectoria completa). Es una métrica escalar, no per-layer.

**Entradas.** Únicamente la training loss $\ell$ por mini-batch durante $T$ pasos. No se necesita acceso a los gradientes, ni a un conjunto de validación, ni hiperparámetros adicionales más allá de $E$ (TSE-E) o $\gamma$ (TSE-EMA).

**Granularidad temporal.** La acumulación se realiza por step (sumando $\ell_i$ del mini-batch corriente y dividiendo por $B$ al cierre de cada época); el valor se reporta por época o congelado al final del presupuesto. Para el sweep correlacional del TFG conviene persistir los valores en hitos del 5%, 10%, 25% y 50% del presupuesto total, alineados con las ventanas de las métricas candidatas.

**Cuándo computar.** Tras cada mini-batch se actualiza el acumulador. La versión cumulativa simplemente suma; la versión EMA aplica la recurrencia $\text{ema}_{t} \leftarrow \gamma\cdot \text{ema}_{t-1} + \bar\ell_t$ al cerrar la época. El valor se congela y persiste en los hitos del barrido para entrar en las correlaciones contra los proxies de eficiencia.

**Coste.** Cero overhead adicional: la training loss ya se computa en el forward de cada paso, por lo que TSE se obtiene como subproducto del bucle de optimización. No requiere barrido extra de gradientes, ni acceso a un val set, ni hiperparámetros a tunear.

**Integración en el pipeline.** Pseudocódigo de referencia que cubre las tres variantes y las claves de logging:

```python
T_END = num_epochs
E = 1                         # burn-in para TSE-E
GAMMA_FAST, GAMMA_SLOW = 0.9, 0.999

tse_cum = 0.0
tse_e_buffer = []             # últimas E pérdidas medias por época
tse_ema_09 = 0.0
tse_ema_0999 = 0.0

for t in range(1, T_END + 1):
    epoch_loss_sum = 0.0
    for x_i, y_i in train_loader:                  # B mini-batches
        loss = criterion(model(x_i), y_i)          # ya computada en forward
        loss.backward(); optimizer.step()
        epoch_loss_sum += loss.item()
    epoch_loss = epoch_loss_sum / B                # (1/B) sum_i ell_i

    tse_cum += epoch_loss
    tse_e_buffer.append(epoch_loss)
    if len(tse_e_buffer) > E:
        tse_e_buffer.pop(0)
    tse_e = sum(tse_e_buffer)
    tse_ema_09   = GAMMA_FAST  * tse_ema_09   + epoch_loss
    tse_ema_0999 = GAMMA_SLOW  * tse_ema_0999 + epoch_loss

    log_scalar("tse/cumulative", tse_cum,     step=t)
    log_scalar("tse/e_window",   tse_e,       step=t)
    log_scalar("tse/ema_0_9",    tse_ema_09,  step=t)
    log_scalar("tse/ema_0_999",  tse_ema_0999,step=t)
```

**Gotchas.** TSE depende directamente del learning rate schedule (un schedule más agresivo deprime artificialmente la pérdida acumulada) y de la elección de pérdida; comparar valores entre runs sólo es legítimo si comparten el mismo schedule, batch size y criterio. Las primeras épocas dominan la suma cumulativa, por lo que para presupuestos muy cortos conviene leer TSE-EMA con $\gamma=0.999$ (más estable) o TSE-E con $E=1$ (descarta el burn-in). En el régimen $T$ grande la métrica satura cerca de cero y pierde poder discriminativo: el barrido del TFG debe vivir en ventanas tempranas (5–50% de épocas), no al final.

**Logging.** Se registran cuatro escalares por época con las claves `tse/cumulative`, `tse/e_window`, `tse/ema_0_9` y `tse/ema_0_999`; el valor de cada uno se congela y persiste en los hitos del 5%, 10%, 25% y 50% del presupuesto para alimentar el barrido de correlaciones contra los proxies de eficiencia (épocas hasta umbral, AUC de test loss, mejor test loss).

## Notes

### Uso en el TFG

- **Rol — BASELINE PREDICTOR obligatorio, NO métrica del registry.** TSE-EMA es el origen del predictor `tse_ema` y se taggea `metric_kind="baseline"` en `metrics_at_window.parquet` para separarlo de las 10 métricas de gradiente del `METRIC_REGISTRY` (7 alineación + 3 varianza). No es una métrica de gradiente: es un predictor de eficiencia derivado únicamente de la train loss.
- **Fórmula.** $\text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t}\,\bar\ell_t$, con $\bar\ell_t$ la train loss media de la época $t$ y $\gamma = 0.999$ (suma ponderada exponencialmente que da más peso a épocas tardías sin descartar las tempranas). Se registran también `tse` (suma simple) y `tse_e` (E=1) por completitud, pero `tse_ema` es el baseline principal.
- **Coste CERO.** La train loss ya se computa en el forward de cada paso, así que TSE-EMA se obtiene como subproducto del bucle de optimización (acumulador EMA). Sin barrido extra, sin acceso a gradientes, sin val set.
- **Señal.** $\downarrow$ mejor — menor TSE-EMA $\Rightarrow$ la configuración acumula menos pérdida bajo la curva de entrenamiento (converge antes) $\Rightarrow$ mejor eficiencia/generalización esperada.
- **Por qué es CRÍTICO (el listón a batir).** Es el *sanity check* metodológico de la tesis. Cualquier métrica de gradiente propuesta debe **superar a TSE-EMA en correlación de Spearman** frente a los proxies de eficiencia (épocas-a-umbral, AUC de test loss, mejor test loss) sobre las **mismas ventanas** (5/10/25/50% de épocas); si no lo hace, el coste de instrumentar gradientes (barridos batch-grad, per-sample $\nabla L$, Jacobian NTK) **no se justifica**. Sin este baseline gratis la tesis no puede defender que la alineación/varianza de gradientes aporte valor sobre algo que ya se mide. **Revisión 2026-06-11:** deja de ser el listón único — pasa a miembro del conjunto condicionante de nivel 0 junto a `val-acc@f` y `val-loss@f` (titular: `val-acc@f`); ver "Caveats verificados" abajo y [[Plan de análisis congelado]].
- **Paper comparable (delta a argumentar).** Es **uno de los 3 papers más comparables** al TFG (junto a Hölzl 2025 y Forouzesh & Thiran 2021) por atacar el mismo problema: predicción temprana y barata del rendimiento. Ru et al. muestran que TSE-EMA supera validation accuracy, learning-curve extrapolation (LcSVR) y zero-cost proxies (JacCov/SNIP/SynFlow) en Spearman $\rho$ bajo presupuestos $T \leq 0.5\,T_{end}$. La diferencia de scope es clara: ellos rankean **arquitecturas** en NAS y *optimizan* la búsqueda; el TFG es **correlacional**, no optimiza, y usa TSE-EMA como referencia gratuita dentro de un sweep controlado (SGD/Adam × FC/CNN-small/ResNet-18 × MNIST/CIFAR-10/(ImageNet)).

### Caveats verificados (revisión bibliográfica, 2026-06-11)

Revisión hecha para decidir la composición del baseline de nivel 0 (registrada en [[Plan de análisis congelado]] §Predictores). Citas verificadas sobre los textos originales. **Aviso (auditoría 2026-06-12):** el PDF local (`SpeedyNAS_Ru2021.pdf`, 14 págs.) carece de apéndices, así que los claims de esta sección atribuidos a Ap. C/D/F (Kendall, $T_o$ = primera época con loss < 0,1, divergencia val-loss/val-acc, saturación) no son verificables con esa copia — conviene reemplazarla por la versión arXiv completa. **Corrección de integración (2026-06-12):** el pipeline alimentaba `compute_tse` con pérdidas per-step; TSE se define sobre **medias por época** (Eq. 1) y ahora se agrega vía `train.epoch_mean_losses` (la época parcial en curso aporta su media corrida) — con el feed per-step, TSE-E($E{=}1$) era exactamente TLmini (la baseline que el paper bate) y la constante de tiempo de las EMAs se encogía en un factor ≈B.

- **El régimen del TFG está explícitamente fuera del scope del paper.** Ru et al., §4.2, textual: *"if models are trained using different hyperparameters that influence the learning curve (e.g. learning rate), the prediction performance of our proposed estimators will be affected. […] Verifying the quality of various estimators for predicting the generalisation performance across different hyperparameters lies outside the scope of this paper but present an interesting direction for future work."* (cita corregida al texto exacto del PDF, p. 7) Todo su historial empírico rankea **arquitecturas con hiperparámetros fijos**; el experimento DARTS con "tres setups" (Fig. 2, Ap. B) rankea *dentro* de cada setup, nunca a través de LRs, y el caso más cercano a hiperparámetros (Ap. E, RandWiredNN) varía los del *generador de arquitecturas*, no los de entrenamiento. No existe trabajo publicado que use TSE/SoTL para rankear learning rates — la matriz del TFG (8 LR × 5 seeds a arquitectura fija) es de paso uno de los primeros tests controlados de TSE cross-LR, y la memoria puede venderlo como hueco que el propio paper deja abierto.
- **Por qué puede fallar cross-LR.** Dentro de una celda la train loss temprana es casi monótona en el LR, así que TSE colapsa hacia "rankear por LR" más ruido de seed; y el modo de fallo canónico de cualquier baseline de curva son las curvas que se cruzan por LR: el LR pequeño acumula menos loss temprana (mejor TSE) y acaba peor — *short-horizon bias* (Wu et al. 2018, arXiv:1803.02021) y efecto regularizador del LR grande (Li, Wei & Ma 2019, arXiv:1907.04595).
- **Degeneración en las ventanas del TFG.** Con f = 5–10% de presupuestos de 20–80 épocas la suma cubre 1–8 épocas: γ = 0,999 apenas decae (0,999⁸ ≈ 0,992), así que TSE-EMA ≈ suma simple, y TSE-E con E = 1 *es* la train loss media de la última época (`train-loss@f`). El Ap. D del paper encuentra E = 1 como mejor variante: la señal está en el **nivel reciente suavizado** de la curva, no en la historia integrada.
- **Saturación con loss baja.** TSE-E/TSE-EMA pierden poder discriminativo cuando las train losses se acercan a cero (Ap. C.4/F); el propio paper introduce por esto la heurística de parar el estimador en 0,9·T_o (T_o = primera época con loss < 0,1). Irrelevante en las ventanas tempranas del TFG; relevante si se leyera TSE en f = 1,0.
- **Divergencia val-loss / val-acc (afecta a las VD, no al baseline).** Ap. C.1–C.2: la val loss puede subir por sobreconfianza mientras la val accuracy sigue mejorando. Toca VD2 (AUC de val loss) y VD3 (best val loss) como objetivos en la cola del presupuesto; motivo adicional para que VD1 (accuracy) sea la primaria. Caveat declarado en el plan de análisis.
- **La teoría detrás de TSE está cuestionada.** La justificación vía marginal likelihood (Lyle et al. 2020, arXiv:2010.14499) fue contradicha por Lotfi et al. 2022 (ICML Outstanding Paper, arXiv:2202.11678): *"models that train faster do not necessarily have higher marginal likelihood, or better generalization"* — el entrenamiento rápido implica contracción rápida del posterior y puede pagar penalización de Occam. En la memoria, tratar TSE como baseline empírico sin apoyarse en esa teoría.
- **Sigue siendo el mejor de su familia (razón para mantenerlo).** White et al. 2021 (arXiv:2104.01177), benchmark de 31 predictores en NAS: SoTL-E es Pareto-óptimo y el mejor predictor basado en curva de aprendizaje en el régimen de query alto. Se mantiene en el logging (coste cero, comparabilidad con la literatura), pero como *miembro* del nivel 0, no como listón único: el titular pasa a `val-acc@f`, la señal estándar de la literatura HPO para rankear configuraciones que difieren en LR (Hyperband, arXiv:1603.06560; rankear por val tras una época ya es casi óptimo para top-k: Egele et al. 2024, arXiv:2404.04111).

## Papers relacionados

- [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]] — Mismo problema (proxy train-time de generalización, barato); uno de los 3 comparables. GWA es métrica del registry y debe batir a este baseline.
- [[Disparity Between Batches as a Signal for Early Stopping]] — Mismo problema (señal temprana barata de generalización sin val set); uno de los 3 comparables. `gradient_disparity` es métrica del registry medida contra este baseline.
- [[A Study of Gradient Variance in Deep Learning]] — La motivación teórica de TSE invoca cotas de generalización dependientes de la **varianza del gradiente**; conecta el baseline con la familia varianza (`normalized_variance`).
- [[An Empirical Model of Large-Batch Training]] — Comparte el marco "predecir eficiencia de entrenamiento a partir de estadísticos baratos"; el GNS predice batch size crítico, TSE-EMA predice ranking de eficiencia.

## Otros papers interesantes a revisar

- **NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search** (Dong & Yang, 2020) — Benchmark tabular (uno de los espacios donde Ru et al. evalúan TSE); referencia estándar para correlaciones ranking-vs-test en NAS. arXiv:2001.00326
- **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah et al., 2021) — Proxies de coste cero (snip, grasp, synflow, etc.) que TSE-EMA supera en consistencia entre tareas; competidores directos del baseline. arXiv:2101.08134
- **Neural Architecture Search without Training** (NASWOT, Mellor et al., 2021) — Estima rendimiento al inicializar usando la correlación de los mapas lineales (regiones ReLU) entre datos, sin entrenar; otro proxy barato comparable que TSE supera bajo presupuestos mayores. arXiv:2006.04647
- **Fantastic Generalization Measures and Where to Find Them** (Jiang et al., 2020) — Estudio empírico a gran escala de medidas de generalización y su correlación con el gap real; marco metodológico para situar TSE-EMA y las métricas de gradiente como proxies competidores. arXiv:1912.02178

## Cited By
