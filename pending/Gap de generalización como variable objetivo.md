# Gap de generalización como variable objetivo

**Estado: pendiente de confirmación expresa del tutor.** Propuesta del 2026-06-10, revisada en profundidad el 2026-06-11 contra los protocolos de Jiang et al. 2020 y Dziugaite et al. 2020 (ver §Historial de revisiones). Su dependencia técnica — la evaluación única de test al final del run — quedó desbloqueada el 2026-06-12 al confirmarse e implementarse el protocolo de evaluación ([[2 - Decisiones]]). La respuesta rápida del tutor no menciona el gap, así que no se implementa hasta tener su sí explícito.

## Resumen

Las variables objetivo actuales miden velocidad y rendimiento final, pero 5 de las 8 métricas de gradiente afirman predecir **generalización** en sus papers originales. Sin una variable de generalización, un resultado nulo para esas métricas sería ambiguo: ¿la métrica no funciona, o se midió contra el criterio equivocado? Se propone añadir el **gap de generalización** al final del run (`final_gap_loss`, con `final_gap_acc` como robustez), medido con una pasada extra de evaluación sobre un subconjunto fijo del train. A presupuesto fijo de épocas el gap crudo confunde sobreajuste con velocidad, así que sus contrastes llevan dos controles pre-registrados: un suelo de ajuste y una correlación parcial por el ajuste alcanzado.

## Qué se propone

### Definición

> **`final_gap_loss = final_test_loss − final_train_eval_loss`** — primaria; positivo = sobreajuste.
> **`final_gap_acc = final_train_eval_acc − final_test_acc`** — robustez; mismo sentido.

Ambos términos se calculan **al final del run, en modo eval, con los mismos pesos**:

- El término de **test** es la evaluación única final que el pipeline ya hace (protocolo de evaluación, 2026-06-12); solo falta añadir su loss al summary.
- El término de **train** es una pasada `evaluate()` extra sobre un **subconjunto fijo del train**: estratificado por clase, de tamaño comparable al test (p. ej. 10k), con semilla de muestreo fija e independiente de la semilla del run — la misma filosofía que el split de val y la probe. Estimar este término por muestreo tiene respaldo directo: Jiang et al. estiman la pérdida de train con minibatches por coste computacional (su §3).

El orden test − train es la convención estándar (Jiang et al. §1.2: riesgo poblacional menos riesgo empírico) y da "más gap = más sobreajuste". La **loss es la primaria** porque es el término en que están formuladas las garantías teóricas de GSNR y disparity; la accuracy satura y acota el gap, por eso acompaña como verificación — el mismo patrón espejo que accuracy + F1-macro.

Las cuatro claves (`final_gap_loss`, `final_gap_acc`, `final_train_eval_loss`, `final_train_eval_acc`) van a `summary.json`, con sus términos para auditoría; no se añade nada a la trayectoria por época. Coste: una llamada a `evaluate()` por run (segundos), una sola vez.

## Por qué hace falta

Las variables objetivo actuales (`efficiency_summary` en `src/train.py`) miden dos constructos: **velocidad** (`val_loss_auc`, `epochs_to_threshold`) y **rendimiento final** (`final_test_acc`). Ninguna mide cuánto sobreajusta el modelo. Y los papers originales de las métricas no reclaman todos lo mismo (ver `src/metrics/README.md` y `docs/research/metrics.md`):

| Reclaman generalización | Reclaman velocidad |
|---|---|
| GSNR — Liu et al. 2020 (su resultado central *es* sobre el gap: GSNR alto ⇒ train y test descienden juntos) | gradient confusion — Sankararaman et al. 2020 |
| gradient disparity — Forouzesh & Thiran 2021 (señal de sobreajuste para early stopping) | GNS — McCandlish et al. 2018 |
| stiffness — Fort et al. 2019 ("a new perspective on generalization") | normalized variance — Faghri et al. 2020 |
| m-coherence — Chatterjee & Zielinski 2020 ("the generalization mystery") | TSE (baseline) |
| GWA — Hölzl 2025 (proxy de generalización en train) | |

Con los objetivos actuales, al primer grupo solo se le puede preguntar algo que sus autores no afirmaron ("¿predices velocidad?"). Añadir el gap permite contrastar cada métrica **en los términos de su propio paper**, con precedente metodológico directo: Jiang et al. 2020 usan exactamente el gap como variable dependiente de su estudio correlacional a gran escala.

**Alcance del contraste (a declarar en la memoria):** varias de esas afirmaciones son sobre la *trayectoria* (disparity como señal de early stopping, GSNR sobre el descenso conjunto por paso); un escalar al final del run contrasta una versión gruesa. Es el contraste correcto al nivel del diseño, pero no debe sobrevenderse la equivalencia.

## Por qué no se puede derivar del `train_loss` ya logueado

El `train_loss` por época es la pérdida del **último minibatch**. Restárselo al test loss falla por tres razones independientes:

1. **Ruido de muestreo:** un único batch no estima el riesgo empírico.
2. **Confusión de modos:** se mide en modo train — con resnet18, BatchNorm usa estadísticas del batch, mientras el test se evalúa en modo eval con running stats. La resta mezclaría el gap real con un artefacto de BN.
3. **Pesos desfasados:** se midió a mitad de época; los pesos ya cambiaron cuando se evalúa el test.

La pasada extra en modo eval sobre el subconjunto fijo elimina las tres.

## El confound del presupuesto fijo y cómo se controla

A presupuesto fijo de épocas y sin early stopping, el gap crudo confunde "generaliza mal" con "ajustó el train". Un run de lr alto que diverge acaba con train ≈ test ≈ alto → gap ≈ 0 → "generaliza perfectamente" según la variable. El gap crudo premia el underfitting y mide en buena parte *cuánto consiguió ajustar el train ese run* — es decir, velocidad. El confound muerde también **dentro de celda** (el lr determina cuánto se ajustó al agotar el presupuesto), no solo entre celdas.

No es una objeción especulativa: es el motivo declarado por el que Jiang et al. no entrenan a épocas fijas (§3):

> "Putting the stopping criterion on *the training loss* rather than *the number of epochs* is crucial since otherwise one can simply use cross-entropy loss value to predict generalization."

Su apéndice A.2 describe exactamente un diseño como el nuestro (a épocas fijas, "the cross-entropy itself can be very predictive of generalization"). Su solución: entrenar cada modelo hasta train cross-entropy = 0.01 y **descartar** los que no llegan (<5% de su pool). Nuestro presupuesto fijo es innegociable — los objetivos de velocidad lo requieren —, así que el control pasa al análisis, con dos reglas pre-registradas en el plan congelado para la familia del gap:

1. **Suelo de ajuste:** los contrastes del gap (solo ellos; los runs siguen contando para velocidad y rendimiento) excluyen runs cuyo `final_train_eval_acc` no alcance un mínimo, calibrado en el pilot y fijado antes de lanzar la rejilla. Es el análogo del descarte de Jiang et al.
2. **Correlación parcial:** la asociación métrica↔gap se estima **controlando por `final_train_eval_loss`** (parcial de Spearman, coherente con el estadístico del plan). La pregunta pasa a ser "¿predice la métrica el gap *más allá* del ajuste alcanzado?" — la afirmación real de los papers de generalización.

El control estadístico es más débil que el experimental — la parcial supone monotonía y descuenta el acoplamiento en lugar de eliminarlo — y se declara como limitación en la memoria, citando el A.2 de Jiang et al.

## Encaje en el plan de análisis

- La batería queda en **tres constructos, uno por familia de afirmaciones**: velocidad (AUC, épocas-hasta-umbral), rendimiento final (acc/F1) y generalización (gap). No se prevén más: cada columna nueva agranda la familia FDR y diluye potencia. El criterio para añadir una variable objetivo es que contraste una hipótesis *distinta* — el gap lo cumple; precision/recall/AUC-ROC en datasets balanceados, no (colapsan en la accuracy).
- El [[Plan de análisis congelado]] debe pre-registrar el gap como familia propia, con su **predicción direccional**: las métricas del grupo "generalización" deberían asociarse más al gap que al AUC de velocidad, y las del grupo "velocidad" al revés. La predicción se formula **sobre la asociación parcial** (controlada por ajuste): con el gap crudo estaría contaminada por construcción del protocolo — a presupuesto fijo, entrenar rápido implica ajustar más el train e inflar el gap. Esa doble disociación, si aparece en las parciales, es un resultado más fuerte que cualquier correlación individual.
- **Lección de Dziugaite et al. 2020, para todas las familias:** promediar la calidad predictiva entre entornos no es un resumen adecuado (*"a satisfying theory cannot simply predict well on average"*) — una métrica puede correlacionar en el agregado y fallar sistemáticamente bajo una intervención concreta. Aplicación casi gratis con la estratificación por celda ya prevista: para cada métrica, **reportar la distribución de asociaciones por celda (mediana y peor caso)**, no solo el estadístico agregado. Una línea más del plan congelado.

## Alternativas consideradas y descartadas

- **Derivar el gap post-hoc de las columnas ya logueadas.** Gratis pero inválido: las tres razones de §Por qué no se puede derivar.
- **Gap a ajuste igualado (réplica del protocolo de Jiang dentro del presupuesto):** medir el gap en la primera época donde el train loss cruza un umbral fijo por dataset. Elimina el confound *por construcción* y es lo más fiel al precedente, pero en esta rejilla: (1) exige calibrar un umbral más — un grado de libertad del investigador extra, contra el espíritu del plan congelado; (2) los runs que nunca cruzan generan datos faltantes **correlacionados con los hiperparámetros bajo estudio** (lr altos, celdas difíciles — Tiny-ImageNet con budget 0.25 podría perder celdas enteras): un sesgo de selección peor que el confound que corrige; (3) como el test solo se evalúa al final, el gap igualado usaría val, debilitando la separación de roles; (4) duplica-triplica el coste de evaluación por época en toda la rejilla. Descartada a favor del control estadístico pre-registrado.
- **Evaluar el train completo cada época (trayectoria del gap).** Daría la curva entera de sobreajuste, pero multiplica el coste de evaluación de toda la rejilla y la variable dependiente solo necesita el valor final; la curva de val por época ya informa del inicio del sobreajuste (`best_val_acc` vs final). Exploratorio si algún resultado lo pide.
- **No añadir el gap.** Coste cero y familia FDR menor, pero los nulos de la mitad de las métricas seguirían siendo ambiguos — el desajuste de constructos que motiva esta propuesta.
- **Más métricas de clasificación (precision, recall, AUC-ROC, top-5).** Los datasets están balanceados: contrastan la misma hipótesis que la accuracy y solo agrandan la familia FDR. El F1-macro ya cubre la verificación de robustez.

## Implicaciones prácticas (al confirmar el tutor)

1. **Código:** `data.py` expone un subconjunto fijo de train para evaluación (análogo a `build_probe`, pero estratificado y en eval); `train.py` añade la pasada final y las cuatro claves al summary (más `final_test_loss`, que la evaluación final ya computa). Mismo cambio conceptual que el protocolo ya implementado: solo toca la evaluación final.
2. **Verificación:** run corto de MNIST → `final_gap_loss` parte de ≈ 0 y **crece** con épocas extra.
3. **Plan congelado:** añadir el suelo de ajuste, la parcial por `final_train_eval_loss`, la predicción direccional sobre parciales y el reporte por celda (mediana + peor caso) para todas las familias.
4. **Pilot:** sin impacto en presupuestos ni umbrales (coste añadido despreciable); calibra además el suelo de ajuste (distribución de `final_train_eval_acc` por celda).
5. Registrar la decisión en [[2 - Decisiones]] y añadir el tercer constructo a las variables dependientes de [[1 - Diseño]].

## Historial de revisiones

- **2026-06-10** — primera redacción, enviada al tutor junto al protocolo de evaluación. Definía la resta invertida (train − test), que daba valores negativos decrecientes con el sobreajuste y contradecía su propio paso de verificación.
- **2026-06-11** — revisión contra los papers: signo corregido a test − train (convención de Jiang §1.2, consistente con la verificación); añadido el control por ajuste alcanzado (suelo + parcial), sin el cual la variable no es interpretable a presupuesto fijo; añadido el reporte por celda (crítica de Dziugaite et al. al promediado).
- **2026-06-12** — el tutor confirma el protocolo de evaluación (implementado, [[2 - Decisiones]]): la dependencia técnica queda desbloqueada. El gap sigue pendiente de confirmación expresa.

## Referencias

- Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., Bengio, S. (2020). *Fantastic Generalization Measures and Where to Find Them.* ICLR 2020. [arXiv:1912.02178](https://arxiv.org/abs/1912.02178). — Protocolo de parada a cross-entropy fija (§3, apéndice A.2), definición del gap (§1.2), estimación del término de train por minibatches (§3), descarte de modelos que no convergen (<5%).
- Dziugaite, G. K., Drouin, A., Neal, B., Rajkumar, N., Caballero, E., Wang, L., Mitliagkas, I., Roy, D. M. (2020). *In Search of Robust Measures of Generalization.* NeurIPS 2020. [arXiv:2010.11924](https://arxiv.org/abs/2010.11924). — Crítica al promediado entre entornos; evaluación por entorno con énfasis en el peor caso; también entrenan a cross-entropy fija (su nota 4).
