# Plan de análisis estadístico (preregistro)

**Estado: borrador pendiente.** La dependencia del tutor se cerró el 2026-06-12 (protocolo de evaluación confirmado e implementado, [[2 - Decisiones]]); falta solo el **pilot de calibración** para fijar presupuestos y umbrales. Al cerrarse, este documento baja a `docs/research/`, la congelación se registra con fecha en [[2 - Decisiones]], y solo entonces se miran resultados de la matriz. Redactado el 2026-06-10, revisado en profundidad el 2026-06-11 (ver §Historial de revisiones).

## Resumen

Este documento fija, antes de mirar ningún resultado, cómo se analizan los ~960 runs: qué variables dependientes, qué estadístico y a qué nivel (Spearman por celda como descriptivo, Wilcoxon entre celdas como confirmatorio), qué reglas de censura y exclusión, qué corrección por comparaciones múltiples (Benjamini–Hochberg con cota Benjamini–Yekutieli, en familias preespecificadas) y qué criterio decide cada hipótesis H1–H6. La idea central es la **inferencia en dos etapas**: el ρ de cada celda es solo un estadístico-resumen (los 40 runs de una celda no son independientes — el LR los clusteriza), y la confirmación ocurre entre celdas, donde los runs son disjuntos. Todo lo que el análisis final haga y no esté aquí se reporta igualmente, etiquetado como **exploratorio** (jardín de senderos, Gelman & Loken).

## Qué congela este documento (y qué no)

**Congela:** estadístico primario y su nivel de inferencia, VD primaria, ventana primaria, reglas de censura y exclusión, familias de corrección múltiple, contraste por hipótesis y nivel de agregación.

**No congela:** los valores numéricos de presupuesto y umbral por dataset (los fija el pilot de calibración; el plan los referencia paramétricamente) ni la lista de métricas *reportada* en la memoria (poda con prueba a posteriori, decisión "Se mide todo, siempre"; la poda no altera qué se contrasta, solo qué se destaca).

**Por qué el pilot no contamina este plan** (tres salvaguardas):

1. Sus 24 runs calibran presupuesto/umbral y overhead — no se usan para contrastar hipótesis ni entran en el dataset de análisis.
2. La calibración ocurre antes de poder estimar ninguna correlación métrica↔VD: con un run por celda no hay dispersión intra-celda que correlacionar, así que no se puede elegir un umbral "que favorezca" a una métrica.
3. Los criterios de calibración están escritos desde antes de ejecutarlo (decisión del pilot, 2026-06-09, en [[2 - Decisiones]]): presupuesto = meseta de la curva de monitorización + margen, redondeado a múltiplo de 20; umbral cruzado al ~30–60% del presupuesto por CNN/ResNet a LR centrado. `run_pilot.py --report` imprime la evidencia; la decisión la aplica el investigador siguiendo esos criterios y se registra en [[2 - Decisiones]] con su evidencia antes de lanzar la matriz.

## Unidad de análisis y datos

- **Unidad de medida:** el run (~960). **Condición (celda):** dataset × arquitectura × optimizador — 24 celdas, 40 runs/celda (8 LR × 5 seeds). **Unidad de inferencia confirmatoria: la celda** (ver §Estadístico).
- **Fuente:** `metrics_at_window.parquet` (predictores por ventana) + `summary.json` / `trajectory.parquet` (VDs) de cada run en `reports/`.
- La dispersión del predictor la genera el barrido de LR dentro de cada celda; las seeds dan réplicas. Ambos ejes se analizan juntos dentro de la celda (no se promedia sobre seeds antes de correlacionar).
- **Pseudo-replicación reconocida:** los 40 runs de una celda no son intercambiables — forman 8 grupos de LR con 5 réplicas, y el n efectivo está entre 8 y 40. Por eso el ρ por celda es **descriptivo** (una observación del mapa) y la confirmación se hace entre celdas, donde los runs son disjuntos.
- **Qué afirma el contraste intra-celda (y qué no):** dentro de la celda, predictor y VD covarían en gran parte *a través del LR*; un ρ alto por celda no demuestra por sí solo que la métrica aporte información que el LR (conocido gratis a priori) no lleve. La afirmación de la tesis es la **transferibilidad**: la relación LR→eficiencia es específica de cada celda y desconocida a priori, mientras que un signo y una magnitud de ρ consistentes entre las 24 celdas (etapa 2, H5, tabla de signos) sí constituyen información que viaja entre configuraciones. El valor incremental dentro de la celda lo vigila H2 contra los baselines de loss.

## Variables

### Dependientes (eficiencia)

La curva de monitorización es la de **val**. VD1 y VD3 se leen sobre la curva **suavizada con mediana móvil centrada de 3 épocas** (lectura primaria); la curva cruda queda en `trajectory.parquet` como análisis de sensibilidad. VD2 integra la curva cruda — integrar ya es robusto. (Protocolo de evaluación, 2026-06-12, [[2 - Decisiones]].)

| VD | Definición | Curva | Notas |
|---|---|---|---|
| **VD1 (primaria)** | épocas-hasta-umbral de val-acc del dataset | suavizada | no cruza → **censurado** (§Censura) |
| **VD2** | AUC de la val loss dentro del presupuesto | cruda | caveat de sobreconfianza (abajo) |
| **VD3** | mejor val loss dentro del presupuesto | suavizada | caveat de sobreconfianza (abajo) |
| **VD4** | `final_test_acc`, evaluada una única vez al final del run | — | la variable objetivo certificada |

- **Excluida del análisis confirmatorio: `seconds_to_threshold`.** El wall-clock en cluster compartido está confundido por contención (decisión "Timing por run"); solo exploratorio, restando la columna acumulada `metric_seconds`, y nunca para afirmaciones cross-celda.
- **Caveat sobre VD2/VD3:** la val loss puede subir por sobreconfianza mientras la val accuracy sigue mejorando (Ru et al., arXiv:2006.04492, Ap. C.1–C.2), sobre todo en la cola del presupuesto; eso sesga el AUC y la best loss como proxies de eficiencia. VD1 (basada en accuracy) es primaria también por esto; VD2/VD3 se interpretan con este sesgo declarado.

### Predictores

Los 8 del registro completo + baselines, todos medidos en cada ventana `f` desde `metrics_at_window.parquet`:

- **Nivel 0 (baseline sin gradiente):** `val-acc@f` (titular), `val-loss@f` y TSE-EMA@f — curva de monitorización y train loss suavizada en la misma fracción. Rival a batir de H2. **Sin contraste confirmatorio propio:** sus rechazos son seguros "por construcción" (tabla de signos); se reportan como referencia descriptiva del mapa.
- **Nivel 1 (gradiente barato):** normalized gradient variance (NGV), gradient noise scale (GNS), gradient disparity.
- **Nivel 2 (retadoras):** gradient confusion, stiffness, m-coherence, GSNR, GWA.

Se reportan **todos** los predictores con sus estadísticos en todos los contrastes — significativos o no. Ningún filtrado previo.

#### Composición del nivel 0 — justificación y fuentes (revisión 2026-06-11, fuentes verificadas sobre los textos originales)

El nivel 0 debe ser *el mejor predictor obtenible sin instrumentar el gradiente* ([[1 - Diseño]] §Baselines):

- **`val-acc@f` es el titular.** Es la misma escala que define VD1 y la señal estándar de la literatura HPO para rankear configuraciones que difieren en LR — exactamente el eje de dispersión de cada celda: successive halving/Hyperband/ASHA promueven por métrica de validación temprana (Li et al. 2018, arXiv:1603.06560; Li et al. 2020, arXiv:1810.05934), y rankear por val tras una sola época ya es casi óptimo para selección top-k (Egele et al. 2024, arXiv:2404.04111). Se lee de la curva ya registrada (snap a posteriori); sin cambio de pipeline.
- **TSE-EMA se mantiene, sin titularidad.** Nunca fue validado para rankear hiperparámetros: Ru et al. (arXiv:2006.04492, §4.2) lo declaran fuera de scope ("Verifying the quality of various estimators for predicting the generalisation performance across different hyperparameters lies outside the scope of our work"). En este diseño la train loss temprana es casi monótona en LR, y el modo de fallo canónico de cualquier baseline de curva son las curvas cruzadas por LR — el LR pequeño acumula menos loss temprana y acaba peor (short-horizon bias: Wu et al. 2018, arXiv:1803.02021; Li, Wei & Ma 2019, arXiv:1907.04595). Se conserva porque es el mejor de su familia en su régimen nativo (benchmark de 31 predictores: White et al. 2021, arXiv:2104.01177), cuesta cero, y la matriz constituye de paso uno de los primeros tests controlados de TSE cross-LR. Caveats verificados con citas en [[Speedy Performance Estimation for Neural Architecture Search]].
- **En las ventanas de este estudio las variantes de TSE degeneran.** Con f = 5–10% de presupuestos de 20–80 épocas, la suma cubre 1–8 épocas: TSE-EMA con γ = 0,999 apenas decae (0,999⁸ ≈ 0,992) y colapsa en la suma simple, y TSE-E con E = 1 *es* `train-loss@f`. El propio paper halla E = 1 como mejor variante: la señal está en el nivel reciente suavizado de la curva, no en la historia integrada. El nivel 0 es de facto "el nivel suavizado de la curva en f".
- **Pendiente de la curva: fuera del nivel 0.** Ninguna fuente aísla valor de la pendiente sobre el nivel; tres líneas independientes apuntan a que el nivel suavizado domina (1-Epoch de Egele et al. 2024; E = 1 óptimo en Ru et al.; Baker et al. 2017, arXiv:1705.10823, mezcla niveles y derivadas sin ablación interna). Exploratorio si algún resultado lo pide.
- **El LR no entra como covariable del nivel 0.** Además de las razones de §Alternativas ("Condicionar por LR"), los valores de hiperparámetros por sí solos son predictores débiles frente a las features de la curva (Baker et al. 2017: R² 0,18 vs 0,95 en su espacio solo-hiperparámetros).

**Comparador emparejado de VD1 (descriptivo):** épocas-hasta-umbral *predichas* invirtiendo un ajuste power-law de 3 parámetros sobre la curva de val-acc dentro de la ventana f (la familia paramétrica con mejor respaldo para curvas de iteraciones: Kadra et al. 2023, arXiv:2302.00441; alternativa amortizada: LC-PFN, Adriaensen et al. 2023, arXiv:2310.20447). No existe baseline publicado para el objetivo épocas-hasta-umbral — lo más cercano es invertir curvas extrapoladas —, así que se construye aquí y se reporta como referencia descriptiva del mapa, igual que el resto del nivel 0: fuera de las familias confirmatorias y fuera del conjunto condicionante de H2 (no quemar grados de libertad con n = 40/celda).

**Contexto de exigencia:** ninguno de los tres papers comparables del corpus usó un baseline de curva de loss (Hölzl 2025, arXiv:2510.25480: validation splits 90/10 y 99/1, LabelWave y gradient disparity como rivales; Forouzesh & Thiran 2021, arXiv:2107.06665: k-fold CV; Jiang et al. 2020, arXiv:1912.02178: lo controlan *por diseño* entrenando todos los modelos hasta la misma loss final — "otherwise one can simply use cross-entropy loss value to predict generalization"). El protocolo de correlación parcial de este plan es más estricto que el estándar publicado, y se declara como tal en la memoria.

### Ventanas

- **Primaria: f = 0,10.** Coincide con `early_window_frac` y es la apuesta de H4 (la señal satura pronto).
- **Secundarias (barrido H4):** f ∈ {0,05, 0,25, 0,50}.
- **f = 1,0:** solo referencia de saturación; no es "temprana" y no entra en ninguna afirmación predictiva.

## Estadístico

Inferencia en **dos etapas** (enfoque *summary statistics*, estándar en análisis de grupo en neuroimagen — Holmes & Friston 1998; equivale a un meta-análisis donde cada celda es un estudio):

1. **Etapa 1 — descriptiva, por celda: Spearman ρ** entre cada predictor@f y cada VD, sobre los 40 runs de la celda. Rango-basado: absorbe no-linealidades, outliers y — clave — los runs censurados como peor rango sin descartarlos. El ρ̂ de celda es el estadístico-resumen que alimenta la etapa 2; sus p-valores se anotan en el mapa solo como descriptivos (anticonservadores: n efectivo < 40 por el clustering de LR).
2. **Etapa 2 — confirmatoria, entre celdas: Wilcoxon de rangos con signo**, por métrica, sobre los ρ de las celdas elegibles contra 0 (bilateral). Robusto a la dependencia intra-celda: bajo H₀ los ρ̂ se centran en 0 sea cual sea el clustering por LR, y las celdas son runs disjuntos. Tamaño de efecto reportado: mediana de ρ, IQR y fracción de celdas con signo consistente.

Complementos:

- **Secundario:** Pearson r por celda, solo sobre runs no censurados y con VD finita (sensibilidad a la elección de estadístico).
- **Sensibilidad a la pseudo-replicación:** repetir la etapa 1 sobre medianas por LR (n = 8 por celda) y comprobar que el mapa (medianas cross-celda, signos) no cambia.
- **Valor incremental (H2): correlación parcial de Spearman** ρ(métrica, VD | TSE-EMA@f, `val-loss@f`, `val-acc@f`) por celda como estadístico-resumen (k = 3 covariables → df = n − 2 − k = 35; coste marginal frente a k = 2), con el mismo Wilcoxon cross-celda como contraste confirmatorio — coherente con el estadístico primario y con la censura por rangos. **ΔR²** (regresión de la VD sobre baseline vs baseline+métrica, runs no censurados) solo como sensibilidad; si parcial y ΔR² discrepan (poblaciones distintas: el parcial incluye censurados, el ΔR² no), **manda el parcial**.
- Todos los tests **bilaterales** a α = 0,05 pre-corrección, con una excepción: H4 es unilateral por ser de no-inferioridad. H6 predice signo, pero el signo se evalúa por concordancia, no con tests unilaterales.

## Censura y exclusiones

- **Censura (VD1):** run que no cruza el umbral dentro del presupuesto → peor rango, empatado con los demás censurados. No es "infinito" ni se elimina: los extremos de la rejilla de LR divergen o no llegan *por diseño* y pueblan el eje de eficiencia.
- **Divergencia:** run con NaN/Inf en la loss de entrenamiento o monitorización en cualquier época → peor rango en VD1, VD2 y VD3 (empatado con los censurados); excluido de Pearson y ΔR². Se reporta el recuento por celda.
- **Celda degenerada (VD1):** celda con >80% de runs censurados, o con menos de 2 valores distintos de VD1, sale de los análisis de VD1 para *todas* las métricas — su ρ̂ no es informativo y solo mete ruido en la etapa 2; se sostiene en VD2/VD3 y se reporta cuáles son. [[1 - Diseño]] ya anticipa FC sobre CIFAR-100 y Tiny-ImageNet. La etapa 2 opera sobre las celdas elegibles restantes.
- **Exclusión total (solo fallo técnico):** runs sin `summary.json` no existen para el launcher (se relanzan, no se analizan). No hay más causas de exclusión total preespecificadas.
- **Missingness por métrica:** si una métrica falla en runtime (`measure` aísla fallos), ese run sale de los tests de *esa* métrica únicamente. Se reporta el recuento; >5% de fallos de una métrica en una celda se señala como descarte silencioso de facto y la celda se marca para esa métrica.

## Corrección por comparaciones múltiples

**Benjamini–Hochberg (BH) a q = 0,05**, no Bonferroni: los predictores están correlacionados entre sí, el objetivo es un mapa de qué métricas señalan algo (no una única afirmación), y Bonferroni — que controla FWER — sería brutalmente conservador sobre cientos de tests. BH controla la FDR de forma demostrada bajo independencia (Benjamini & Hochberg, 1995) y bajo dependencia positiva del tipo PRDS (Benjamini & Yekutieli, 2001).

**Punto delicado de validez.** Los tests son bilaterales y PRDS *no se cumple en general* para tests bilaterales de correlación: Yekutieli (2008) da contraejemplos explícitos en configuraciones gaussianas bilaterales. La garantía deja de ser automática justo cuando se juntan en una familia predictores correlacionados con signos esperados opuestos (p. ej. gradient confusion + frente a stiffness −). La dependencia se descompone así: **entre celdas** los runs son disjuntos → p-valores independientes; **dentro de una celda** los 11 predictores comparten los 40 runs → correlacionados. El problema vive *dentro* de la celda. A esto se suma que cada p-valor por celda es individualmente anticonservador (n efectivo < 40, §Unidad de análisis): segunda razón, independiente, por la que el nivel confirmatorio se sube al cross-celda.

**Familias preespecificadas** (la unidad de corrección coincide con la unidad de afirmación — H1–H2 afirman *por métrica al nivel cross-celda*; el mapa por celda es descriptivo):

1. **Familia confirmatoria primaria:** los **8 Wilcoxon cross-celda** (uno por métrica de gradiente) en (f = 0,10, VD1) → una pasada BH de 8 p-valores. Los 8 tests comparten celdas y runs (dependencia esperable positiva); como blindaje se reporta también la cota BY, barata a este nivel (c(8) ≈ 2,72). Los baselines quedan fuera (referencia descriptiva, §Predictores).
2. **Familia confirmatoria H2:** los 8 Wilcoxon análogos sobre las correlaciones parciales → pasada BH propia de 8, con su cota BY.
3. **Mapa por celda (descriptivo):** **una pasada BH por predictor** sobre sus 24 celdas en cada (ventana, VD) → familias de 24 p-valores. Las celdas usan runs disjuntos → los 24 p-valores son **independientes** y BH es válido por el resultado de 1995; el reagrupado por predictor elimina por construcción la objeción PRDS/bilateral, que solo surgiría agrupando los 11 predictores correlacionados *dentro* de una celda. Estas pasadas anotan el mapa; **ninguna afirmación se apoya en ellas** (sus p-valores individuales son anticonservadores).

**Robustez y cota conservadora:**

- **Cota de sensibilidad BY.** Se reporta además **Benjamini–Yekutieli**, válido bajo dependencia *arbitraria* (incluida negativa), como cota dura: lo que sobrevive a BY es robusto a cualquier estructura de dependencia. En el nivel confirmatorio es barata (c(8) ≈ 2,72); en el mapa por celda, el reagrupado por predictor la abarata frente a una pasada única (c(24) ≈ 3,78 frente a c(264) ≈ 6,15).
- **Validación por permutación (opcional, blindaje).** Permutar la VD dentro de cada celda y recalcular Spearman estima la nula *conjunta* empírica sin asumir PRDS. Variante que respeta el clustering: permutar grupos de LR completos (bloques de 5 seeds) en lugar de runs sueltos. Coherente con el bootstrap previsto para H4, y extiende a la etapa 2: recalcular los Wilcoxon sobre los ρ permutados da la nula conjunta de la familia confirmatoria.
- **Entre familias.** El control *entre* familias se gestiona por jerarquía de titulares (lo secundario no asciende). Si se exige garantía formal sobre el conjunto reportado, el marco de inferencia selectiva de Benjamini & Bogomolov (2014) formaliza esta estructura de dos niveles — la respuesta preparada si el tribunal aprieta.

Se reportan q-valores (BH y, como cota, BY) junto a los p crudos. Lo que solo sobreviva en familias secundarias (otras ventanas × VDs, con la misma estructura de tres niveles) no asciende a titular.

**Descartado como primario: Storey (q-valor adaptativo).** Gana potencia estimando π₀ en vez de asumir π₀ ≈ 1, pero es menos estable y exhibe rechazos paradójicos cuando muchas nulas son falsas (Reiss et al.). Para una tesis prima la estabilidad; queda, como mucho, para sensibilidad de potencia.

### Referencias (corrección múltiple)

- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B 57(1):289–300. — procedimiento BH original; control de FDR bajo independencia.
- Benjamini, Y. & Yekutieli, D. (2001). *The control of the FDR in multiple testing under dependency.* Annals of Statistics 29(4):1165–1188. — validez bajo PRDS y la corrección BY bajo dependencia arbitraria.
- Yekutieli, D. (2008). — contraejemplos de no-PRDS en tests gaussianos bilaterales; para el caso concreto de tests de correlación bilaterales ver [*Asymptotic control of FWER… application to correlation tests* (arXiv 2007.00909)](https://arxiv.org/pdf/2007.00909).
- Storey, J. D. (2002). [*A direct approach to false discovery rates* (q-valor adaptativo)](http://genomics.princeton.edu/storeylab/papers/directfdr.pdf) — más potencia, menos estable.
- [*Paradoxical results of adaptive false discovery rate procedures in neuroimaging studies*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3699340/) — por qué no se adopta Storey como primario.
- Benjamini, Y. & Bogomolov, M. (2014). [*Selective inference on multiple families of hypotheses* (JRSS-B 76:297–318)](https://academic.oup.com/jrsssb/article/76/1/297/7075946) — FDR jerárquica sobre familias.
- [*Conditional calibration for false discovery rate control under dependence* (Fithian & Lei)](https://www.stat.berkeley.edu/~wfithian/fdr-dependence.pdf) — panorama de control de FDR bajo dependencia.
- [*Multiple testing under negative dependence* (arXiv 2212.09706)](https://arxiv.org/pdf/2212.09706) — resultados recientes para dependencia negativa.

## Contrastes por hipótesis

| Hipótesis | Contraste | Criterio |
|---|---|---|
| **H1** — existencia | Wilcoxon cross-celda, familia confirmatoria primaria | ≥1 métrica de gradiente con \|mediana de ρ\| ≥ 0,3 sobre las celdas elegibles **y** q < 0,05 |
| **H2** — valor incremental (decisiva) | Wilcoxon cross-celda sobre las parciales, familia H2 | ≥1 métrica significativa. H1 ✓ y H2 ✗ = las métricas son redundantes con la curva de loss — negativo válido, se reporta como tal |
| **H3** — familia ganadora | descriptivo + Wilcoxon pareado por celda como apoyo | mediana de \|ρ\| por familia (alineación vs variabilidad) en la ventana primaria; sin promesa de significancia — pregunta abierta ligada al título |
| **H4** — suficiencia temprana | no-inferioridad: Wilcoxon unilateral sobre dᵢ − δ | "satura pronto" si se rechaza H₀: mediana(d) ≥ δ, con δ = 0,1; solo métricas que pasaron H2; puede salir "inconcluso" y se reporta como tal |
| **H5** — invariancia cross-optimizador | binomial exacto contra 0,5 sobre los 12 pares de celdas SGD↔Adam (seeds compartidas) | concordancia de signo de ρ por par; se reporta siempre junto a la mediana de \|ρ\| (el signo de un ρ ≈ 0 es una moneda al aire) |
| **H6** — mecanismo, con signo | concordancia con la tabla de signos (congelada abajo, antes de ver datos) | mediana de ρ cross-celda y fracción de celdas con el signo predicho |

Detalle de H4: dᵢ = \|ρᵢ@0,50\| − \|ρᵢ@0,10\| por celda elegible; margen preespecificado **δ = 0,1** (mínima diferencia de \|ρ\| con relevancia práctica, del orden de la precisión de la propia mediana cross-celda). Marco TOST/no-inferioridad (Lakens 2017): falsable en ambas direcciones. Descriptivo de apoyo por celda: IC de la diferencia de correlaciones dependientes — bootstrap BCa (10 000 remuestras de runs) o, como alternativa cerrada, los ICs de Zou (2007).

### Tabla de signos esperados (vs VD1, épocas-hasta-umbral: menor = más rápido)

| Métrica | Signo esperado | Base |
|---|---|---|
| gradient confusion | + | fuerte (Sankararaman et al.) |
| stiffness (intra-clase) | − | fuerte (Fort et al.) |
| m-coherence | − | fuerte (Chatterjee & Zielinski) |
| gradient disparity | + | extrapolada (Forouzesh predice test error, no velocidad) |
| NGV | + | fuerte (Faghri et al.) |
| GNS | + | fuerte (McCandlish: más ruido → más pasos a batch fijo) |
| GSNR | − | extrapolada (Liu predice generalización; vs VD4: + fuerte) |
| GWA | − | extrapolada (Hölzl predice generalización; vs VD4: + fuerte) |
| TSE / val-loss@f (baselines) | + | por construcción |
| val-acc@f (baseline) | − | por construcción |

GSNR y GWA llevan su predicción *fuerte* sobre VD4 (`final_test_acc`, signo +); su columna VD1 es extrapolación y se evalúa aparte. Revisar signos contra los papers al congelar (varios están en cola de lectura).

## Agregación entre celdas

- **Por celda primero, agregado después** — guardia contra Simpson: la dificultad del dataset no debe fabricar la correlación.
- **El resumen cross-celda es el contraste confirmatorio, no solo un resumen:** mediana e IQR de ρ sobre las celdas elegibles + fracción de celdas con signo consistente + Wilcoxon de etapa 2, por métrica y ventana. Es el formato del "mapa" que la memoria reporta.
- **Independencia entre celdas, aproximada:** las celdas no comparten runs, pero sí datasets, arquitecturas y seeds (misma inicialización para misma arquitectura + seed). Se declara como limitación; la dependencia residual esperable es débil y positiva, y la cota BY cubre el caso de que no lo sea.
- **Agregado pooled (secundario):** correlación sobre los ~960 runs con rangos calculados *dentro* de cada celda (equivale a estandarizar por condición). Modelos de efectos mixtos quedan como exploratorio si el resumen anterior se queda corto.
- **Figura de sanidad val↔test (descriptiva):** scatter `final_val_acc` vs `final_test_acc` sobre los ~960 runs, preespecificada en la decisión del protocolo de evaluación ([[2 - Decisiones]]); recupera el diagnóstico de concordancia que se pierde al no evaluar test por época. No alimenta ningún contraste.

## Nota de potencia

- **Etapa 2 (confirmatoria):** con ~24 celdas elegibles, si la mediana real de ρ es 0,3 y la dispersión cross-celda típica está en 0,2–0,3, el Wilcoxon contra 0 opera con un efecto estandarizado ≈ 1 y potencia > 95%. El criterio de H1 no depende de significancias por celda infrapotenciadas.
- **Etapa 1 (descriptiva, contexto del mapa):** con n = 40 por celda, el test de Spearman a α = 0,05 bilateral tiene ~80% de potencia para \|ρ\| ≥ 0,44 y solo ~47% para \|ρ\| = 0,3 (aprox. Fisher-z) — y sus p-valores son además anticonservadores porque el n efectivo está entre 8 y 40 (clustering por LR). Doble motivo por el que las anotaciones por celda son descriptivas y ninguna afirmación se apoya en una celda suelta.

## Alternativas consideradas y descartadas

- **Inferencia confirmatoria por celda (plan original).** Pseudo-replicación por LR (p-valores anticonservadores) y criterio de H1 incoherente con la potencia. El reagrupado de familias BH por predictor arreglaba la dependencia *entre tests*, pero no la calibración de cada p-valor individual; ambas piezas se conservan para el mapa descriptivo.
- **Inferencia intra-celda consciente del cluster:** bootstrap por clusters de LR (8 clusters es poco para remuestrear), modelos mixtos con LR como efecto (artillería difícil de justificar y de explicar en un TFG) o rmcorr (Bakdash & Marusich 2017 — responde otra pregunta: asociación a LR fijo, entre seeds, donde la varianza es minúscula). Sobrecomplicación frente a la etapa 2; quedan como exploratorio.
- **Condicionar por LR** para aislar el aporte de la métrica: eliminaría por diseño casi toda la varianza que el estudio genera (solo quedarían seeds) y la relación LR→eficiencia es no monótona (una parcial de rangos no la limpia). La respuesta correcta al confusor del LR es la transferibilidad cross-celda más H2 contra los baselines de loss, no una parcial por LR.
- **Conteo de celdas con \|ρ̂\| ≥ 0,3 sin exigir significancia por celda**, testado contra su expectativa nula (~6% por celda bajo ρ = 0): válido y fiel a la literalidad de [[1 - Diseño]], pero mezcla magnitud y conteo en un criterio menos estándar que el Wilcoxon agregado, y ese 6% es optimista con n efectivo < 40. Puede reportarse como secundario.
- **H4 solo descriptivo** (curva ρ vs ventana sin test): dejaba H4 sin contraste falsable; la curva se mantiene como figura, el contraste es la no-inferioridad.
- **Bonferroni y Storey:** ver §Corrección múltiple.

## Qué falta para congelar

1. ~~**Confirmación del tutor** al protocolo train/val/test~~ — **cerrada el 2026-06-12**: la monitorización de VD1–VD3 lee val (suavizada para VD1/VD3) y VD4 existe; protocolo implementado ([[2 - Decisiones]]).
2. **Pilot de calibración:** fija los valores definitivos de presupuesto y umbral por dataset que parametrizan VD1/VD2, aplicando los criterios preespecificados en §Qué congela. La estructura del plan no depende de ellos.

Si el gap de generalización se confirma ([[Gap de generalización como variable objetivo]]), se añaden aquí antes de congelar: su familia de contrastes con suelo de ajuste, la parcial por `final_train_eval_loss`, la predicción direccional por familias (sobre asociaciones parciales) y el reporte por celda (mediana + peor caso) para todas las familias.

Al cerrarse todo: mover este documento a `docs/research/`, registrar la congelación en [[2 - Decisiones]] con fecha, y solo entonces mirar resultados de la matriz.

## Historial de revisiones

- **2026-06-10** — primera redacción, antes de ejecutar pilot y matriz.
- **2026-06-11** — pasada crítica con cuatro correcciones. (1) Pseudo-replicación intra-celda reconocida (el LR clusteriza los 40 runs) → respuesta central: **inferencia en dos etapas**, el ρ por celda baja a descriptivo y la confirmación sube al nivel cross-celda. (2) Criterio de H1 retirado: exigía significancia por celda en ≥13/24, pero la propia nota de potencia estima que un ρ = 0,3 real solo sale significativo en ~11/24 celdas sin corregir — exigía de facto ρ ≳ 0,45 y desalineaba el plan con la barra de falsación de [[1 - Diseño]]. (3) Criterio de H4 retirado: "el IC de la diferencia contiene 0" afirmaba la nula y, con ICs anchos por n = 40 y correlaciones dependientes, se "confirmaba" casi gratis → sustituido por no-inferioridad. (4) Huecos en las reglas de pilot y celdas degeneradas completados; corrección sobre `--report`: imprime evidencia y la decisión es del investigador (la redacción anterior afirmaba que implementaba una regla mecánica). La misma fecha revisa la composición del nivel 0 tras revisión bibliográfica de TSE con fuentes verificadas: `val-acc@f` pasa a titular, TSE-EMA se mantiene sin titularidad.
- **2026-06-12** — dependencia del tutor cerrada: VD1–VD3 leen val y VD4 existe; lecturas suavizadas de VD1/VD3 incorporadas a §Variables; figura de sanidad val↔test añadida a §Agregación.

## Referencias generales del plan

*(Las de §Corrección múltiple están verificadas con enlace; las siguientes están citadas de memoria — comprobar título/año exactos antes de citarlas en la memoria.)*

- Gelman, A. & Loken, E. (2014). *The garden of forking paths.* — multiplicidad de análisis posibles; motiva el preregistro.
- Holmes, A. P. & Friston, K. J. (1998). *Generalisability, random effects and population inference.* NeuroImage. — enfoque *summary statistics* en dos etapas: estadístico por unidad, inferencia entre unidades.
- Lakens, D. (2017). *Equivalence tests: a practical primer for t-tests, correlations, and meta-analyses.* Social Psychological and Personality Science. — TOST y márgenes de equivalencia/no-inferioridad; base del criterio de H4.
- Zou, G. Y. (2007). *Toward using confidence intervals to compare correlations.* Psychological Methods. — ICs para diferencias de correlaciones dependientes (descriptivo de H4).
- Bakdash, J. Z. & Marusich, L. R. (2017). *Repeated measures correlation.* Frontiers in Psychology. — alternativa intra-celda descartada en §Alternativas.
