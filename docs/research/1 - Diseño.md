## Enfoque

Estudio correlacional. Objetivo único: cuantificar en qué medida métricas de variabilidad y alineación de gradientes medidas en la fase inicial del entrenamiento predicen indicadores de eficiencia del entrenamiento completo. La componente de intervención (early stopping basado en la señal) queda explícitamente fuera del alcance del TFG por restricciones de tiempo. Puede mencionarse como trabajo futuro.

## Pregunta de investigación

¿Pueden las métricas de gradiente, medidas en la fase inicial del entrenamiento, predecir la eficiencia del entrenamiento completo?

La eficiencia se entiende como tres constructos: velocidad de convergencia (VD1-VD2), rendimiento final (VD3-VD4, desde el 2026-09-03) y generalización (VD5-VD6, el gap). La pregunta se evalúa contra los tres y se concreta en seis objetivos, formalizados como las hipótesis contrastables H1-H6 de §Hipótesis a contrastar. La intervención basada en la señal (early stopping, ajuste dinámico de LR) queda fuera de alcance (ver §Enfoque) y se recoge como trabajo futuro.

## Hipótesis operativa

La variabilidad y/o alineación de los gradientes, medida a través de distintas métricas en una fracción inicial del entrenamiento, se asocia con indicadores de eficiencia del entrenamiento completo, bajo variaciones de learning rate y optimizador, en arquitecturas de visión por computador.

Hipótesis falsada si las correlaciones son débiles o inestables entre configuraciones en la mayoría de condiciones. Con qué cuenta se decide está fijado desde el 2026-09-03 en §Método de análisis, posterior a los datos y anterior a cualquier coeficiente, porque el plan anterior se retiró el 2026-08-25. Diez puntos del método se revisaron la tarde del mismo día tras la revisión externa, después de la primera lectura de H1 y por motivos de estructura, nunca por un coeficiente (ver [[2 - Decisiones]]). Un resultado negativo con análisis robusto sigue siendo contribución válida.

## Hipótesis a contrastar

Formalización de la hipótesis operativa en seis afirmaciones falsables. Aquí se enuncia **qué** se quiere averiguar. El criterio con el que se decide cada una está en §Método de análisis, fijado el 2026-09-03.

- **H1 (existencia).** Al menos una métrica temprana de gradiente correlaciona con la eficiencia del entrenamiento completo. (Es la hipótesis operativa de arriba.)
- **H2 (valor incremental, la decisiva).** Al menos una métrica supera a la referencia nombrada de antemano, el *accuracy* de validación en la ventana para el rendimiento final y el *loss* de validación para el gap, y no es redundante con ella dentro de la celda. *Reformulada el 2026-09-03 (tarde):* antes prometía «tras controlar por lo que la curva de loss ya da gratis», y ninguna cuenta del método mide un descuento. Si H1 se cumple pero H2 falla, las métricas de gradiente son redundantes con una señal que no cuesta nada, y ese resultado negativo es una contribución válida y no trivial.
- **H3 (qué familia gana).** Una de las dos familias, alineación o variabilidad, es sistemáticamente más predictiva que la otra. Desde el 2026-09-03 se lee como un ranking de métricas individuales con la familia como etiqueta, porque tras la poda quedan tres por familia y comparar familias enteras no es lo que promete el objetivo. Las palabras clave de la portada apuestan por alineación; es una pregunta empírica abierta, ligada al riesgo de coherencia entre portada y contenido.
- **H4 (suficiencia temprana).** El poder predictivo satura pronto: medir en una fracción temprana del entrenamiento predice tan bien como medir más tarde. Beneficio práctico: permite decidir antes. En velocidad la comparación es entre *epochs* absolutas, la 1 frente a la 2 en MNIST y la 2 frente a la 4 en los demás conjuntos, y se declara así (2026-09-03, tarde).
- **H5 (invariancia cross-optimizador).** El signo de la correlación métrica↔eficiencia se preserva entre SGD y Adam. *Corregido el 2026-08-01:* la redacción anterior presentaba esto como "consecuencia comprobable de la decisión raw-grad", y no lo es. Computar la métrica sobre ∇L bruto y nunca sobre el update preacondicionado hace comparable la **métrica** entre optimizadores, que es la condición que hace la pregunta formulable; no implica en absoluto que su **correlación con la eficiencia** conserve el signo, porque Adam cambia la dinámica del entrenamiento. H5 es una afirmación empírica independiente, y de hecho más interesante así. Se lee como diferencia pareada, D con SGD menos D con Adam por par de celdas, con su intervalo; un no rechazo no prueba invariancia (2026-09-03, tarde).
- **H6 (mecanismo, con signo).** Cada métrica trae una predicción *con signo* de su paper: alta stiffness intra-clase, alta m-coherence y baja gradient confusion → convergencia más rápida; NGV/GNS altos → más lento o batch mayor; GWA alta → mejor generalización **en el artículo, que define el gradiente como $-\nabla\ell$; aquí se mide sobre $\nabla\ell$ bruto, así que la predicción heredada es la contraria, una GWA baja acompaña a mejor generalización** (la conversión está hecha en `fundamentos.tex:168`). Que el signo observado coincida con el predicho es prueba más exigente que la magnitud. Desde el 2026-09-03 (tarde) se cuenta a una cola, solo sobre las celdas cuyo intervalo excluye el cero y solo para las predicciones que el artículo enuncia; la m-coherencia y NGV están podadas y sus predicciones se leen sobre la escala de ruido (la de la m-coherencia con el signo cambiado).

## Diseño experimental

### Variables dependientes (eficiencia del entrenamiento)

Protocolo de evaluación (confirmado por el tutor y registrado el 2026-06-12, ver [[2 - Decisiones]]): el train optimiza, **val monitoriza** (curva por época y todos los indicadores de eficiencia) y el **test certifica una única vez al final** del run. VD1 y VD3 se leen sobre la curva de val suavizada con una mediana móvil centrada de 3 épocas, con el valor crudo en la primera y en la última época (corregido el 2026-09-03: antes se tomaba la media de dos en los bordes, que fabricaba cruces en la época 1; los campos suavizados de `summary.json` son obsoletos y la capa de análisis los recalcula). El motivo: los extremos de una curva ruidosa están sesgados por su volatilidad, y esa volatilidad depende del LR. La curva cruda queda en `trajectory.parquet` como análisis de sensibilidad.

1. **VD1 (primaria):** número de épocas hasta alcanzar un umbral de val-accuracy propio de cada dataset **y arquitectura** (decisión del 2026-09-01, ver [[2 - Decisiones]]), leído sobre la curva suavizada. Runs que no alcancen el umbral se tratan como censurados. El constructo es por tanto "épocas hasta llegar al nivel propio de esta arquitectura en este problema", y solo se lee dentro de una celda, nunca entre arquitecturas en términos absolutos.
2. **VD2:** área bajo la curva (cruda) de val loss dentro de un presupuesto fijo de épocas.
3. **VD3 (apoyo; constructo de rendimiento final desde el 2026-09-03):** mejor val loss alcanzada dentro de ese presupuesto, sobre la curva suavizada.
4. **VD4:** `final_test_acc`, la accuracy de test evaluada exactamente una vez al final del run (acompañada de `final_test_f1_macro` como verificación de robustez; en datasets balanceados F1-macro ≈ accuracy).
5. **VD5 (generalización, primaria):** `final_gap_loss = final_test_loss − final_train_eval_loss` (positivo = sobreajuste).
6. **VD6 (generalización, robustez):** `final_gap_acc = final_train_eval_acc − final_test_acc` (mismo sentido que VD5). El término de train de ambas se mide al final del run, en modo eval y con los mismos pesos, sobre un subconjunto fijo y estratificado del train (tamaño igual al test, `SPLIT_SEED`). VD5 y VD6 forman el tercer constructo de eficiencia, junto a velocidad (VD1-VD2) y rendimiento final (VD3-VD4): miden cuánto sobreajusta el modelo (decisión 2026-06-14, [[2 - Decisiones]]). La predicción direccional asociada es la doble disociación: las métricas que en sus papers reclaman generalización deberían asociarse más al gap que a la velocidad, y al revés. Cubren un hueco que el diseño ya tenía: H6 compromete una afirmación de generalización (GWA/GSNR) que hasta ahora no tenía diana contra la que contrastarse.

### Variables independientes (métricas tempranas)

El conjunto *computado* está implementado y fijado en código (`src/metrics/`, 8 métricas + baseline); la lista *reportada* se poda después por colinealidad con prueba (ver [[2 - Decisiones]]). Dos familias:

- **Alineación / coherencia direccional**: gradient confusion, stiffness, GWA. La m-coherence se poda por identidad con la escala de ruido (2026-09-03).
- **Variabilidad estocástica**: gradient noise scale, GSNR y gradient disparity, reclasificada el 2026-09-03 porque es √(2·tr Σ/51). La normalized gradient variance se poda porque es la escala de ruido con menos muestras (2026-09-03, tarde).

Una candidata temprana no aparece como métrica separada, porque la *cosine similarity entre gradientes de batches* ya está contenida en otras métricas (stiffness y gradient confusion se construyen sobre los cosenos por pares de gradientes per-ejemplo).

### Ventana temporal

Fracciones fijas del presupuesto total de entrenamiento. Barrido en 5%, 10%, 25%, 50%. El barrido en sí mismo es un resultado reportable (cuán temprano basta para predecir).

Cómo se mide en la práctica (`src/train.py`): las métricas se registran al final de *cada* época durante todo el entrenamiento. Los snapshots de 5/10/25/50/100% no se miden en el instante exacto. Se eligen *a posteriori* sobre la trayectoria completa, tomando para cada fracción la época cuyo progreso quede más cerca (`metrics_at_window.parquet`). Con los presupuestos calibrados (20/40/40/40 épocas, todos múltiplos de 20) la selección es exacta, porque cada fracción de `windows` cae justo en una frontera de época (0,05×20=1, 0,25×40=10, etc.), así que no hay desfase entre la fracción nominal y la época elegida.

### Setup de entrenamiento

- Datasets: MNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet. Núcleo decidido 2026-05-14; Tiny-ImageNet confirmado 2026-06-09 (ver [[2 - Decisiones]]).
- Particiones train/val/test (decisión 2026-06-12, justificación en [[2 - Decisiones]]): test oficial intacto; val del tamaño convencional de cada dataset, extraído del train con muestreo estratificado por clase y semilla fija independiente de la semilla del run (todos los runs ven la misma partición). Tamaños: MNIST 50k/10k/10k (convención clásica), CIFAR-10/100 45k/5k/10k (He et al. 2015) y Tiny-ImageNet 90k/10k/10k (su `val/` público hace de test, porque las etiquetas del test oficial no son públicas). La probe de métricas se muestrea del train recortado.
- Normalización: media/desviación por canal del *training set* de cada dataset, sin augmentation (estudio sensible al determinismo). Constantes verificadas por recálculo desde cero (2026-06-09): MNIST/CIFAR-10/CIFAR-100 coinciden a <5e-5. **Aviso de reproducibilidad:** Tiny-ImageNet coincide solo a ~6e-4 (media exacta, std algo menor en los tres canales). El desfase es consistente con la decodificación JPEG (versión de libjpeg/Pillow): la normalización exacta depende del entorno, así que hay que fijar versiones si se quiere reproducir bit a bit.
- Arquitecturas: FC, CNN simple, ResNet-18. Familia decidida 2026-05-14; variante ResNet-18 fijada 2026-06-09.
- Label noise: descartado en v1. Backlog si sobra tiempo (replicaría Forouzesh / Chatterjee&Zielinski).
- Learning rates: varios por condición (rejilla concreta en §Matriz de runs).
- Optimizadores: SGD y Adam.

### Matriz de runs (congelada 2026-06-09)

Rejilla completa: cuatro datasets × tres arquitecturas × dos optimizadores = **24 celdas** (celda = dataset × arquitectura × optimizador, la unidad del objetivo n ≥ 30 de §Riesgos abiertos #1). Decisión y justificación en [[2 - Decisiones]].

- **Profundidad.** 8 LR × 5 seeds = 40 runs por celda → **~960 runs**, por encima del suelo n ≥ 30. La dispersión del predictor la dan los LR, no las seeds; de ahí que se priorice el nº de LR. Seeds compartidas {0,1,2,3,4} en todas las celdas para comparación pareada entre SGD y Adam (H5).
- **Rejilla de LR (log-espaciada en medias décadas, por optimizador, no por modelo).** 8 puntos por optimizador, la misma rejilla para FC, CNN y ResNet-18 (decisión 2026-06-09 en [[2 - Decisiones]]). SGD (momentum 0,9): `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0}`. Adam: `{3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`, misma forma, una década más abajo porque su paso efectivo va preescalado por 1/√v. El rango ancho (3,5 décadas) cubre los óptimos de las tres arquitecturas; los extremos divergen o no alcanzan el umbral por diseño, y esos runs censurados aportan rango al eje de eficiencia (VD1). El centro se recalibra tras el pilot si el óptimo de alguna celda queda descentrado.
- **Hiperparámetros fijos (no se barren, para no añadir confusores).** `batch_size=128`, `weight_decay=0`, `momentum=0.9` (SGD) / betas por defecto (Adam), `probe_size=256`, `windows=[0.05, 0.10, 0.25, 0.50, 1.0]`. Las métricas leen ∇L de la pérdida (no el paso preacondicionado), así que el weight decay no entra en su valor; se fija a 0 solo para no introducir un eje de trayectoria extra. La justificación de cada valor, uno a uno, está en [[2 - Decisiones]].
- **Presupuesto por dataset** (calibrado en el pilot el 2026-06-17, evidencia registrada en [[2 - Decisiones]]; sin data augmentation → por debajo del SOTA): MNIST 20 épocas, CIFAR-10, CIFAR-100 y Tiny-ImageNet 40. El presupuesto no cambia con la arquitectura, porque es el eje temporal común que hace comparables las ventanas.
- **Umbral de val-accuracy por dataset y arquitectura** (fijado el 2026-09-01 sobre los 960 runs, ver [[2 - Decisiones]]). La norma: τ es el valor redondo más alto que alcanza al menos el **60 % de los entrenamientos que aprendieron**, entendiendo por aprender llegar a 1,25 veces el azar. Resultado, en el orden FC, CNN, ResNet-18: MNIST 0,975 / 0,98 / 0,99; CIFAR-10 0,50 / 0,60 / 0,75; CIFAR-100 0,20 / 0,30 / 0,40; Tiny-ImageNet 0,08 / 0,22 / 0,36. Se comparte entre optimizadores para que los doce pares de OE5 midan lo mismo en los dos brazos. Con estos valores VD1 existe en **las 24 celdas** y la definen 573 de los 960 runs (571 antes de corregir el suavizado de los bordes el 2026-09-03); con el umbral único anterior existía en 18 y la definían 494. La censura restante ya no es un techo de arquitectura, sino presupuesto agotado o entrenamientos que nunca aprendieron.
- **Métricas.** Se computa el conjunto completo de métricas implementadas en toda la rejilla. El barrido per-sample recorre **todos** los parámetros y se trocea en filas (`chunk_size`) para acotar la memoria; GWA es la única métrica que se calcula sobre la última capa. La lista *reportada* se poda luego por colinealidad con prueba (ver [[2 - Decisiones]]).

### Baselines

El baseline no es la métrica más simple ni la que diga el paper, sino *el mejor predictor obtenible sin instrumentar el gradiente*. Lo que da valor al estudio es el coste de medir el gradiente, así que el rival a batir es lo que se obtiene gratis de la curva de loss. Tres niveles:

- **Nivel 0, sin gradiente (suelo).** TSE (suma o EMA de las train loss tempranas; coste cero, estándar en NAS) y, sobre todo, **`early-val-accuracy@f`** (val accuracy/loss medida en la misma fracción $f$). Este último es un predictor muy fuerte y casi gratis: si las métricas de gradiente no lo superan, no hay aporte que defender.
- **Nivel 1, gradiente barato (benchmark interno).** Normalized variance (NGV) y gradient noise scale (GNS) en variabilidad; gradient disparity en alineación (Pearson 0,957 entre la disparidad train-train y la train-val sobre 220 configuraciones en Forouzesh & Thiran; la correlación con el error de test el paper solo la afirma en cualitativo). Es el rival a batir para justificar una métrica de gradiente *cara*. Tras la poda del 2026-09-03 el nivel 1 queda en la escala de ruido y la disparidad, esta última reclasificada como variabilidad.
- **Nivel 2, retadoras.** Las caras o novedosas (gradient confusion, m-coherence, stiffness, GSNR, GWA) deben superar a los niveles 0 y 1. GWA es barato y a la vez el de mayor correlación reportada en la literatura (Pearson 0,99 en Hölzl 2025): si una métrica barata domina a las caras, la conclusión sería que *no hace falta instrumentación cara*.

El resultado más valioso no es quién predice mejor a secas, sino quién predice mejor **dentro de su clase de coste** y frente al predictor de referencia, que no calcula ningún gradiente. Con coste asintótico el eje deja de ser continuo, así que no hay frente de Pareto que dibujar (decisión del 2026-08-27, ver [[2 - Decisiones]]).

## Método de análisis (fijado el 2026-09-03 por la mañana, revisado esa tarde)

Posterior a los datos y anterior a cualquier coeficiente en su forma de la mañana; diez puntos se revisaron por la tarde tras la revisión externa, después de la primera lectura de H1 y por motivos de estructura. Las elecciones, su base y su evidencia están en [[2 - Decisiones]] (2026-09-03, dos entradas), y la memoria lo escribe en `metodologia.tex` §Protocolo de análisis. Resumen del estado:

- **Población:** los 806 runs que aprendieron. Los 960 se calculan en `results/` como sensibilidad, pero no entran en la memoria como contraste, porque los colapsados aportan constantes. En el gap, además, solo los que alcanzan sobre el train el τ de su celda (693).
- **Unidad y estadístico:** un run es una observación dentro de su celda; el coeficiente es la D de Somers sobre pares comparables, que coincide con tau donde no hay empates ni censura. Cada D lleva su error típico por jackknife (quitar un run) y un intervalo normal al 95 %; una celda muestra asociación si el intervalo excluye el cero.
- **Predictores:** las columnas titulares de las seis métricas de gradiente que quedan tras la poda (escala de ruido, GSNR, gradient disparity; stiffness, gradient confusion, GWA), los tres componentes del predictor de referencia (val loss, val accuracy, TSE) y la posición en la rejilla (`log_lr`), techo de cualquier lectura monótona del learning rate.
- **Cuenta granulada:** la misma D sumando los pares dentro de cada learning rate con al menos tres runs (estratificada), con su número de pares; es la única cuenta que no es learning rate leído de otra manera (Jiang et al. 2020).
- **Velocidad:** la lectura primaria de VD1 es por hitos, la D solo entre los runs que aún no habían cruzado al cerrar la ventana, censurados incluidos, con el número en riesgo; la D sobre todos los runs es secundaria. Solo sirven las ventanas del 5 y el 10 %.
- **Agregación:** las 24 celdas a la vista; cuántas muestran asociación y con qué signo, con el desglose por conjunto de datos; el recuento de signos como descripción y la referencia binomial solo como vara; sin Benjamini-Hochberg.
- **Familia primaria:** ventana del 5 %, VD1, VD4 y VD5, los diez predictores de arriba; el resto exploratorio y etiquetado.
- **H2:** la métrica contra su referencia nombrada, val accuracy para VD4 y val loss para el gap, leída con el intervalo jackknife de la diferencia |D| métrica menos |D| referencia sobre los mismos runs y el recuento crudo al lado; `D_ref` como medida de redundancia dentro de la celda, redundante desde 0,8; y la lectura de selección para VD4, el accuracy de test perdido al elegir el learning rate con la métrica al 5 % frente a elegirlo con la validación al 5 %, por mediana y por celda, con el signo del artículo (el invertido, exploratorio). Decidida sobre VD4 y VD5, porque en velocidad la referencia es un prefijo de la variable. *Resultado del 2026-09-04: cae.*
- **H3:** ranking de métricas con la familia como etiqueta, por celdas con asociación del signo mayoritario y desempate por la mediana de |D|; la segunda mitad, «cambia de bando», se lee repitiendo la cuenta por conjunto y por arquitectura. *Resultado del 2026-09-04: cae; la alineación pone la primera métrica y también la última, y el ganador cambia con la arquitectura.* **H4:** |D| al 50 % menos |D| al 5 % por celda en las variables del final, con el jackknife pareado sobre los mismos runs, y en velocidad entre las epochs 1 y 2 (MNIST) o 2 y 4 (resto) por hitos con varianzas sumadas. *Resultado del 2026-09-04: se sostiene, las métricas no maduran.* **H5:** diferencia pareada SGD menos Adam con intervalo de varianzas sumadas; acuerdo e inversión solo entre D seguras. *Resultado del 2026-09-04: no se refuta ni queda demostrada; 0 a 2 inversiones de 12, y lo que cambia es el tamaño.* **H6:** signo predicho, a favor o en contra, sobre las celdas con intervalo fuera del cero, solo predicciones explícitas. *Resultado del 2026-09-04, revisado esa tarde: cae en parte; los signos se reproducen en el desenlace (GWA con test 15/1, GSNR con gap 11/2), no reciben apoyo en la velocidad, y se invierten en la stiffness con la velocidad (9 frente a 5) y en la disparidad con el accuracy de test (12 frente a 4), que entra como predicción del artículo.*
- **Forma de las relaciones:** leída lado a lado a lo largo del learning rate, sin mirar nubes; las celdas donde una métrica monótona no puede dar relación monótona se declaran antes de calcular.

## Procedimiento

```mermaid
flowchart LR
    A["Preparar datos\nMNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet"] --> B["Entrenar modelos\nFC, CNN simple, ResNet"]
    B --> C["Logging métricas alineación + variabilidad\npor época"]
    C --> D["Registrar eficiencia\n(épocas-a-umbral, AUC, best loss)"]
    D --> E["Barrido ventana temprana\n5% / 10% / 25% / 50%"]
    E --> F["Análisis de correlación\n(método fijado el 2026-09-03)"]
    F --> G["Análisis robustez\ncross arch × dataset × LR × optimizador"]
    G --> H["Resultados y conclusiones"]
```

### Ejecución y reanudación

`src/run_matrix.py` es la fuente única de verdad de la rejilla. `--init` genera los 24 YAML de celda en `experiments/` con el presupuesto por dataset y los hiperparámetros congelados; los ficheros existentes no se tocan, así sobreviven las ediciones a mano tras el pilot. LR y seed no van en los YAML, porque son los ejes de barrido y se inyectan por run, de modo que el nombre del run queda determinado por (modelo, dataset, optimizador, lr, seed). Un run cuenta como *hecho* si existe `reports/<run_name>/summary.json`: `train.py` lo escribe en último lugar, así que su presencia marca un run completo. El lanzador es idempotente. Relanzarlo ejecuta solo los pendientes, de modo que se reanuda tras una caída sin llevar contabilidad externa, y los flags `--dataset/--model/--optimizer` permiten ejecutar solo una parte de la rejilla.

Antes de la matriz va el pilot de calibración: `src/run_pilot.py` ejecuta un run por celda (LR centrado, seed 0, presupuesto doblado) escribiendo en `reports_pilot/` para no colisionar con la detección de reanudación de la matriz, y `--report` resume la evidencia para fijar presupuestos y umbrales definitivos. Protocolo y justificación en [[2 - Decisiones]].

## Convergencia de la literatura

Corpus de **16 papers** en el vault (`Papers/`). Los recuentos `/15` de abajo se basan en los 15 que proponen métrica o setup; queda fuera *On the Ineffectiveness of Variance Reduced Optimization for Deep Learning* (related-work, no aporta dataset/arquitectura/métrica al recuento). Esto es distinto del progreso de lectura, que vive en [[3 - Progreso]].

Extraído de [[Métricas]] y [[Corpus]]. Justifica el setup propuesto.

**Datasets** (núcleo común):
- CIFAR-10: 12/15 papers.
- MNIST: 10/15.
- CIFAR-100: 6/15.
- ImageNet: 4/15 (se sustituye por Tiny-ImageNet, ver [[Corpus]]).

**Arquitecturas** (familias dominantes):
- MLPs / Fully-Connected: 9/15.
- ResNets (con ResNet-18 recurrente en Faghri, Forouzesh, Chatterjee & Zielinski, Liu): 8/15.
- CNNs no-ResNet (típicamente 3 capas conv con filtros 3×3): 8/15.

**Métricas tempranas** (las dos familias de §Diseño experimental, confirmadas por la literatura):
- **Alineación / coherencia direccional** (6 papers): GWA (Hölzl), m-coherence (Chatterjee & Zielinski), stiffness (Fort et al.), gradient confusion η (Sankararaman et al.), gradient disparity $\|g_i - g_j\|_2$ (Forouzesh & Thiran), Coherent Gradients $f_t^p$ (Chatterjee).
- **Variabilidad estocástica** (3 papers): normalized variance $\mathbb{V}[g]/\mathbb{E}[g]^2$ (Faghri et al.), GSNR $\tilde{g}^2/\rho^2$ (Liu et al.), gradient noise scale $B_{\text{simple}} = \operatorname{tr}(\Sigma)/\|G\|^2$ (McCandlish et al.).

**Implicación para el TFG**:
- Setup fijado: MNIST + CIFAR-10 + CIFAR-100 + Tiny-ImageNet × {FC, CNN simple, ResNet-18} × ambas familias de métricas.
- Coincide con el setup propuesto en este documento.
- ImageNet (completa), transformers y dominios non-vision (Atari, Dota, MNLI) quedan explícitamente fuera del scope.

## Riesgos abiertos

1. **Número de runs por condición. Cerrado.** Correlaciones con n pequeña son inútiles. Objetivo mínimo n ≥ 30 por celda (arquitectura × dataset × optimizador). La matriz completa da 40 entrenamientos por celda y los 960 están hechos, así que el suelo se cumple con margen. Lo que queda por medir, en la fase A, es cuántos de esos 40 sobreviven en cada celda para cada variable dependiente.
2. **Coste computacional de métricas caras** (gradient confusion, m-coherence). **Cerrado el 2026-07-17:** se mantiene la medición completa de las ocho métricas, con un sobrecoste máximo medido de 2,048x sobre la matriz.
3. **Diferenciación frente a literatura existente** (McCandlish 2018, Faghri 2020). Aporte a defender: comparativa rigurosa entre múltiples familias de métricas, barrido en fracciones tempranas, análisis de robustez cross-architecture/cross-dataset. Debe ser explícito en la intro.
4. **Coherencia portada-contenido.** El título no nombra la alineación; son las palabras clave de la portada las que empiezan por «alineación de gradientes». Si el análisis acaba apoyándose más en métricas de variabilidad, revisar las palabras clave (el título de EBRON ya está comprometido).

## Confusores metodológicos

Amenazas a la validez del análisis correlacional que el diseño debe neutralizar (distintas de los riesgos de proyecto de arriba):

- **Dificultad del dataset al agregar.** Juntar MNIST + CIFAR-10 + CIFAR-100 puede hacer que "fácil vs difícil" domine la correlación y esta desaparezca dentro de cada condición (paradoja de Simpson). De ahí la necesidad de mirar primero dentro de cada condición y agregar después. Cómo se agrega está en §Método de análisis: cuántas de las 24 celdas muestran asociación con intervalo fuera del cero y con qué signo, con la consistencia por conjunto de datos a la vista.
- **Colinealidad entre predictores.** No son independientes: GNS ≈ B·NGV por el TLC; m-coherence, stiffness y gradient confusion son funciones del mismo Gram de gradientes per-ejemplo; GSNR es la versión por parámetro de NGV. La dimensionalidad efectiva de las métricas es menor que su número; una matriz de correlación (o PCA) entre *predictores* es en sí misma un resultado, y permite podar las redundantes (ver [[2 - Decisiones]]).
- **Censurado** en épocas-hasta-umbral: los runs que nunca alcanzan el umbral no son "infinito". La convención de registro está fijada, un valor censurado se anota como ausente y nunca como el presupuesto, y cómo entra en el contraste está en §Método de análisis: en pares comparables, con el censurado más lento que cualquier cruce, y en la lectura por hitos entre los que aún no habían cruzado al cerrar la ventana.
