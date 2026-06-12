# Gap de generalización como variable objetivo

**Estado: pendiente** — propuesta redactada en sesión de trabajo (2026-06-10) y **revisada en profundidad el 2026-06-11** tras lectura crítica de los protocolos de Jiang et al. 2020 y Dziugaite et al. 2020 (ver [Referencias](#referencias)): la revisión corrige el signo de la definición original y añade el control por ajuste alcanzado, sin el cual la variable no es interpretable a presupuesto fijo de épocas. Complementaria al [[Protocolo de evaluación y plan de análisis]] y enviada junto a él para consulta con el tutor. Depende de ese protocolo (usa su evaluación única de test al final del run); si aquel cambia, esta se adapta.

## Contexto: las variables objetivo no cubren lo que la mitad de las métricas afirma predecir

Las variables objetivo actuales (`efficiency_summary` en `src/train.py`) miden dos constructos: **velocidad** (`test_loss_auc`, `epochs_to_threshold`, `seconds_to_threshold`) y **rendimiento final** (`final_test_acc`, `best_test_acc`, más el `final_test_f1_macro` propuesto como robustez). Ninguna mide **generalización** — cuánto sobreajusta el modelo.

El desajuste importa porque los papers originales de las métricas de gradiente no reclaman todos lo mismo (ver `src/metrics/README.md` y `docs/research/metrics.md`):

- **Reclaman generalización:** GSNR (Liu et al. 2020 derivan que un GSNR alto implica que train y test descienden juntos — su resultado central *es* sobre el gap), gradient disparity (Forouzesh & Thiran 2021 la proponen como señal de sobreajuste para early stopping), stiffness (Fort et al. 2019, "a new perspective on generalization"), m-coherence (Chatterjee & Zielinski 2020, "the generalization mystery") y GWA (Hölzl 2025, proxy de generalización en train).
- **Reclaman velocidad:** gradient confusion (Sankararaman et al. 2020), GNS (McCandlish et al. 2018), normalized variance (Faghri et al. 2020) y el baseline TSE.

Con los objetivos actuales, a las métricas del primer grupo solo se les puede preguntar algo que sus autores no afirmaron ("¿predices velocidad?"), y un resultado nulo sería ambiguo: ¿la métrica no funciona, o se le midió contra el criterio equivocado? Añadir el gap permite contrastar cada métrica **en los términos de su propio paper**. Hay además precedente metodológico directo: Jiang et al. 2020 ("Fantastic Generalization Measures") usan exactamente el gap como variable dependiente en su estudio correlacional a gran escala — aunque, como se detalla abajo, su protocolo de parada difiere del nuestro y esa diferencia obliga a un control adicional. Una nota sobre el alcance del contraste: las afirmaciones originales de varias de estas métricas son sobre la *trayectoria* (disparity como señal de early stopping, GSNR sobre el descenso conjunto por paso); un escalar al final del run contrasta una versión gruesa de esas afirmaciones. Es el contraste correcto al nivel del diseño, pero la memoria no debe sobrevender la equivalencia.

## Propuesta

### Definición

- **`final_gap_loss = final_test_loss − final_train_eval_loss`** (primaria; **positivo = sobreajuste**), con ambos términos calculados **al final del run, en modo eval, con los mismos pesos**: el término de test es la evaluación única que ya introduce el protocolo de 3 vías; el de train es una pasada `evaluate()` extra sobre un **subconjunto fijo del train** (estratificado por clase, tamaño comparable al test —p. ej. 10k—, semilla de muestreo fija e independiente de la semilla del run, misma filosofía que el split de val y la probe). El orden test − train es la convención estándar: Jiang et al. definen el gap como riesgo poblacional menos riesgo empírico (su §1.2), y es el orden con el que "más gap = más sobreajuste". *(La redacción del 2026-06-10 tenía la resta invertida — train − test —, que da valores negativos decrecientes con el sobreajuste y contradecía su propio paso de verificación.)*
- **`final_gap_acc = final_train_eval_acc − final_test_acc`** como acompañante de robustez (la acc satura y acota el gap; la loss es el término en que están formuladas las garantías teóricas de GSNR y disparity). Mismo sentido: positivo = sobreajuste. El patrón espejo de acc + F1-macro: una primaria interpretable, una verificación.
- Ambos van al `summary.json` junto con sus términos (`final_train_eval_loss/acc`) para auditoría; no se añade nada a la trayectoria por época.
- Estimar el término de train con un subconjunto fijo en lugar del train completo tiene respaldo en el precedente: Jiang et al. estiman la pérdida de train con minibatches muestreados por coste computacional (su §3).

### Por qué no se puede derivar del `train_loss` ya registrado

El `train_loss` por época que hoy se loguea es la pérdida del **último minibatch**, y restarlo del test_loss no daría el gap por tres razones independientes:

1. **Ruido de muestreo:** es un único batch, no una estimación del riesgo empírico.
2. **Confusión de modos:** se calcula en modo train — con resnet18, BatchNorm usa estadísticas del batch, mientras el test se evalúa en modo eval con running stats. La diferencia mezclaría el gap real con la discrepancia train/eval de BN, que es un artefacto.
3. **Pesos distintos:** se midió a mitad de época; los pesos ya cambiaron cuando se evalúa test.

La pasada extra en modo eval sobre el subconjunto fijo elimina las tres. Coste: una llamada a `evaluate()` por run (segundos), una sola vez.

### El confound del presupuesto fijo y cómo se controla

A presupuesto fijo de épocas y sin early stopping, el gap crudo confunde "generaliza mal" con "ajustó el train". Dos runs de la misma celda que difieren solo en lr: el de lr alto que diverge acaba con train ≈ test ≈ alto → gap ≈ 0 → "generaliza perfectamente" según la variable. El gap crudo premia el underfitting y, a presupuesto fijo, mide en buena parte *cuánto consiguió ajustar el train ese run* — es decir, velocidad. El confound muerde **también dentro de celda** (el lr determina cuánto se ajustó el train al agotar el presupuesto), no solo entre celdas.

No es una objeción especulativa: es el motivo declarado por el que el precedente no entrena a épocas fijas. Jiang et al., §3:

> "Putting the stopping criterion on *the training loss* rather than *the number of epochs* is crucial since otherwise one can simply use cross-entropy loss value to predict generalization."

Y su apéndice A.2, describiendo exactamente un diseño como el nuestro:

> "if we pick the stopping criterion based on number of iterations or number of epochs, then since some models optimize faster than others, they end up fitting the training data more and in that case the cross-entropy itself can be very predictive of generalization."

Su solución: entrenar cada modelo hasta cross-entropy de train = 0.01 y **descartar** los que no llegan (<5% de su pool). Con el término de train clavado por construcción, las diferencias de gap reflejan solo generalización. Nuestro diseño de presupuesto fijo es innegociable (los objetivos de velocidad lo requieren), así que el control pasa al análisis. El plan congelado pre-registra dos reglas para la familia de contrastes del gap:

1. **Suelo de ajuste:** los contrastes del gap (solo ellos; los runs siguen contando para velocidad y rendimiento) excluyen runs cuyo `final_train_eval_acc` no alcanza un suelo mínimo, calibrado en el pilot y fijado en el plan antes de lanzar la rejilla principal. Es el análogo del descarte de Jiang et al.
2. **Correlación parcial:** la asociación métrica↔gap se estima **controlando por `final_train_eval_loss`** (parcial de Spearman, coherente con el estadístico del plan). La pregunta pasa a ser "¿predice la métrica el gap *más allá* del ajuste alcanzado?", que es la afirmación real de los papers de generalización.

El control estadístico es más débil que el experimental — la parcial supone monotonía y descuenta el acoplamiento en lugar de eliminarlo — y eso se declara en la memoria como limitación, con cita al A.2 de Jiang et al.

### Encaje en el plan de análisis

- Con esta adición la batería de objetivos queda en **tres constructos, uno por familia de afirmaciones**: velocidad (AUC, épocas-hasta-umbral), rendimiento final (acc/F1) y generalización (gap). No se prevén más objetivos: cada columna nueva agranda la familia de contrastes bajo FDR y diluye potencia, así que el criterio para añadir una variable objetivo es que contraste una hipótesis *distinta* — el gap lo cumple; precision/recall/AUC-ROC en datasets balanceados, no (colapsan en la accuracy).
- El **plan de análisis congelado** (pendiente de escribir en docs/) debe pre-registrar el gap como familia de hipótesis junto a las demás, incluyendo la predicción direccional que la teoría sugiere: las métricas del grupo "generalización" deberían asociarse más al gap que al AUC de velocidad, y las del grupo "velocidad" al revés. **La predicción se formula sobre la asociación parcial** (controlada por ajuste): con el gap crudo estaría mecánicamente contaminada — a presupuesto fijo, entrenar rápido implica ajustar más el train e inflar el gap, así que las métricas de velocidad correlacionarían con el gap crudo por construcción del protocolo, no por contenido informativo. Esa doble disociación, si aparece en las parciales, es un resultado más fuerte que cualquier correlación individual.
- **Lección de Dziugaite et al. 2020 para todas las familias, no solo el gap:** critican el protocolo de Jiang por promediar la calidad predictiva entre entornos — *"averaging across experimental settings is not an appropriate summarization […] a satisfying theory cannot simply predict well on average"*. Una métrica puede correlacionar en el agregado y fallar sistemáticamente bajo una intervención concreta. Aplicación casi gratis con la estratificación por celda ya prevista: para cada métrica, **reportar la distribución de asociaciones por celda (mediana y peor caso), no solo el estadístico agregado**. Entra como una línea más del plan congelado.

## Alternativas consideradas y descartadas

- **Derivar el gap post-hoc de las columnas ya logueadas.** Gratis pero inválido por las tres razones de arriba (ruido de minibatch, modo BN, pesos desfasados).
- **Gap a ajuste igualado (réplica del protocolo de Jiang dentro del presupuesto).** Evaluar el subconjunto de train cada época y medir el gap en la primera época donde el train loss cruza un umbral fijo por dataset. Elimina el confound *por construcción* y es lo más fiel al precedente, pero en esta rejilla concreta: (1) exige calibrar un umbral de train loss por dataset — un grado de libertad del investigador más, contra el espíritu del plan congelado; (2) los runs que nunca cruzan generan datos faltantes **correlacionados con los hiperparámetros bajo estudio** (lr altos, celdas difíciles — Tiny-ImageNet con budget 0.25 podría perder celdas enteras), un sesgo de selección peor que el confound que corrige; (3) como el test solo se evalúa al final (protocolo de 3 vías), el gap igualado usaría val, debilitando la separación de roles; (4) duplica-triplica el coste de evaluación por época en toda la rejilla. Descartada a favor del control estadístico pre-registrado.
- **Evaluar el train completo cada época (trayectoria del gap).** Daría la curva entera de sobreajuste, pero multiplica el coste de evaluación de toda la rejilla y la variable dependiente solo necesita el valor final; la curva de val por época ya informa del inicio del sobreajuste (`best_val_acc` vs final). Queda como exploratorio si algún resultado lo pide.
- **No añadir el gap.** Coste cero y familia FDR menor, pero deja sin resolver el desajuste de constructos que motiva esta propuesta: los nulos de la mitad de las métricas seguirían siendo ambiguos.
- **Más métricas de clasificación (precision, recall, AUC-ROC, top-5).** MNIST/CIFAR/Tiny-ImageNet están balanceados: todas estas variantes contrastan la misma hipótesis que la accuracy y solo agrandan la familia FDR. El F1-macro ya cubre el papel de verificación de robustez.

## Implicaciones prácticas

- **Cambios de código** (contenidos, y solo si el tutor confirma): `data.py` expone un subconjunto fijo de train para evaluación (análogo a `build_probe`, pero estratificado y en eval); `train.py` añade una pasada `evaluate()` final sobre él y cuatro claves al summary (`final_gap_loss`, `final_gap_acc`, `final_train_eval_loss`, `final_train_eval_acc`).
- **Plan de análisis congelado:** tres añadidos — suelo de ajuste pre-registrado para los contrastes del gap, parcial controlando por `final_train_eval_loss`, y reporte por celda (mediana + peor caso) para todas las familias.
- **Sin impacto en el pilot:** el coste añadido es despreciable y no toca presupuestos ni umbrales. El pilot calibra además el suelo de ajuste (distribución de `final_train_eval_acc` por celda).

## Siguientes pasos (al confirmar el tutor)

1. Implementar junto con el split de 3 vías (mismo PR conceptual: ambos tocan la evaluación final) → verificar con un run corto de MNIST que `final_gap_loss` parte de ≈ 0 y **crece** con épocas extra (con el signo corregido, la verificación y la definición son consistentes).
2. Registrar la decisión en [[2 - Decisiones]] y añadir el tercer constructo a la sección de variables dependientes de [[1 - Diseño]].
3. Escribir en el plan de análisis congelado de docs/ el gap con su predicción direccional por familias (sobre asociaciones parciales), el suelo de ajuste, y el reporte por celda.

## Referencias

- Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., Bengio, S. (2020). *Fantastic Generalization Measures and Where to Find Them.* ICLR 2020. [arXiv:1912.02178](https://arxiv.org/abs/1912.02178). — Protocolo de parada a cross-entropy fija (§3, apéndice A.2), definición del gap (§1.2), estimación del término de train por minibatches (§3), descarte de modelos que no convergen (<5%).
- Dziugaite, G. K., Drouin, A., Neal, B., Rajkumar, N., Caballero, E., Wang, L., Mitliagkas, I., Roy, D. M. (2020). *In Search of Robust Measures of Generalization.* NeurIPS 2020. [arXiv:2010.11924](https://arxiv.org/abs/2010.11924). — Crítica al promediado entre entornos; evaluación por entorno con énfasis en el peor caso; también entrenan a cross-entropy fija (su nota 4).
