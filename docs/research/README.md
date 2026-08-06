# TFG: métricas de gradiente como predictores de eficiencia

**Resumen en cinco líneas.** Estudio correlacional que responde a una pregunta: ¿pueden las métricas de gradiente, medidas en la fase inicial del entrenamiento, predecir la eficiencia del entrenamiento completo? Para ello se entrena una rejilla congelada de 24 celdas ({MNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet} × {FC, CNN simple, ResNet-18} × {SGD, Adam}) con 8 learning rates × 5 seeds por celda (~960 runs), midiendo en todos los runs las 8 métricas de gradiente más el baseline TSE. Las ventanas tempranas (5 a 50% del presupuesto) se correlacionan con indicadores de eficiencia que cubren tres constructos: velocidad (épocas-hasta-umbral, AUC de val-loss, mejor val-loss), rendimiento final (accuracy de test) y generalización (el gap test−train). El protocolo de evaluación es: el train optimiza, el val monitoriza y el test certifica una única vez al final. Una métrica solo cuenta como aporte si supera al baseline de la curva de loss (hipótesis H2, la decisiva).

## Dónde vive cada cosa

| Quieres… | Mira en |
|---|---|
| El qué y el porqué (pregunta, hipótesis, diseño, matriz, baselines, análisis) | [[1 - Diseño]] |
| Por qué/cuándo se decidió algo (log de decisiones) | [[2 - Decisiones]] |
| Dónde estamos (fases, estado, pasos, cola de lectura) | [[3 - Progreso]] |
| Qué datasets y modelos usa cada paper + frecuencias del setup | [[Corpus]] |
| Cómo se implementa y loguea cada métrica + auditoría | [[Métricas]] |
| Definición de un concepto | [[Conceptos]] |
| Cómo se decide cada hipótesis (el preregistro) | [[4 - Análisis]] |
| Resumen y uso en el TFG de un paper concreto | `Papers/<paper>` |
| El trabajo contado de principio a fin | `thesis/` (la memoria) |

## Por dónde empezar

Antes de nada, dos avisos que ahorran tiempo. El primero es que aquí hay tres tipos de documento y conviene no confundirlos: los que explican **qué se hace y por qué**, los que registran **qué se decidió y cuándo**, y el que dice **dónde está el proyecto ahora**. El segundo es que ningún documento largo de este vault se lee de principio a fin; se consultan por la sección que hace falta.

Hay cuatro caminos de lectura según a qué vengas, y cada uno se sostiene solo.

**Para entender la pregunta (media hora).** Leer el resumen de cinco líneas de arriba y después [[1 - Diseño]] entero, que es el qué y el porqué completo: pregunta de investigación, las seis hipótesis, el diseño experimental, la matriz de runs y los baselines. Con eso ya se puede tener una conversación sobre el trabajo.

**Para entender cómo se decide.** Ir a la §Guía rápida de [[4 - Análisis]], que es la versión en lenguaje llano del preregistro y está pensada para leerse suelta, sin el resto del documento. Si además hace falta entender la maquinaria estadística, el último bloque de [[Conceptos]] la explica herramienta a herramienta, con un ejemplo numérico en cada entrada; se usa como diccionario, no se lee seguido.

**Para entender el código.** Empezar por `src/config.py`, que es la fuente de verdad: contiene a la vez los knobs de un run y los ejes congelados de la matriz, y todo lo demás los importa en vez de repetirlos. Después `src/train.py`, que es un run entero de principio a fin y se lee en una sentada. Luego `src/metrics/README.md` para saber qué mide cada métrica en lenguaje llano, `src/metrics/__init__.py` para ver por qué el registro y el baseline están separados, y `src/metrics/primitives.py` para el barrido compartido, que es la optimización que hace viable el estudio. Al final, una métrica cualquiera con su test al lado, para ver el patrón de `_core` puro más envoltorio.

**Para leer la memoria.** `thesis/` cuenta el trabajo entero y no necesita ningún otro documento: es el camino recomendado para el tutor, el tribunal o cualquiera que quiera el relato completo en vez del vault de trabajo. Sus capítulos se leen en orden, porque cada uno se apoya en el anterior: introducción (la pregunta y los objetivos), estado del arte (qué existe y qué falta), fundamentos (los objetos del gradiente y los indicadores de eficiencia, que son los dos lados de la correlación), metodología (las hipótesis, la matriz y el protocolo de análisis que las convierte en decisiones), implementación, resultados y conclusiones. Se compila con `latexmk -pdf -outdir=render main.tex` desde dentro de `thesis/`.

**Y para saber dónde está el proyecto**, [[3 - Progreso]], y de él solo el estado actual y los pasos inmediatos. Es el único sitio donde el estado vigente es fiable: este README y `CLAUDE.md` describen la estructura estable y van por detrás a propósito.

Lo que no conviene hacer: leer [[2 - Decisiones]] de principio a fin, porque es un log cronológico de decisiones con lo más reciente arriba y se busca en él la decisión concreta que se quiere entender; leer [[Conceptos]] entero, porque es un glosario; y leer [[4 - Análisis]] de una sentada, porque para eso está su guía rápida.

## Referencia (cambia poco)

- **[[Corpus]]**: datasets y modelos por paper (pares dataset → modelo), frecuencias que justifican el setup y decisiones de implementación (con la sustitución ImageNet → Tiny-ImageNet).
- **[[Métricas]]**: la métrica de cada paper y cómo se traslada al pipeline (estimador, claves de logging, coste, señal); incluye el plan de logging consolidado y la auditoría contra los PDFs.
- **[[Conceptos]]**: glosario: una entrada por concepto, agrupadas por tema (alineación · varianza · optimización · generalización · inferencia estadística del análisis), enlazadas a los papers que las fundamentan. El último bloque explica en lenguaje llano las herramientas estadísticas que usa el preregistro (inferencia en dos etapas, Spearman con censura, Wilcoxon, BH/BY, equivalencia, potencia).
- **[[4 - Análisis]]**: el preregistro estadístico, **congelado el 2026-08-01** y commiteado antes del primer resultado. Fija qué se contrasta y con qué criterio se decide cada hipótesis. Empezar por su §Guía rápida, que es la versión en lenguaje llano.
- **[[EBRON]]**: título, resumen y palabras clave registrados (no editar: es lo entregado).
- **[[Seminarios TFG - cosas a tener en cuenta]]**: guía de redacción, depósito y defensa (ETSINF-UPV), con lo que difiere para GCD.

## Papers

16 notas en `Papers/`, una por paper: resumen del trabajo (`## Summary`), cómo se mide la métrica en el pipeline (`## Medición y pipeline`, con puntero a [[Métricas]]), uso en el TFG y relaciones con otros papers. Los datasets/modelos de cada paper viven en [[Corpus]]; la **cola de lectura priorizada** (6/16 leídos), en [[3 - Progreso]].

## Código (raíz del repo)

- `src/train.py`: entrenamiento de un run con instrumentación de métricas; `src/run_pilot.py` y `src/run_matrix.py`: lanzadores del pilot de calibración y de la matriz (~960 runs), ambos reanudables.
- `src/metrics/`: registro completo de métricas (se mide todo, siempre); `src/data.py`: split train/val/test estratificado fijo.
- `tests/`: suite de verificación; `experiments/`: los 24 YAML de celda; `thesis/`: memoria LaTeX (plantilla ETSINF).

## Recordatorios

- Anexo ODS obligatorio (plantilla oficial; ver [[Seminarios TFG - cosas a tener en cuenta]] §6).
- El tutor evalúa parte de la nota: visto bueno antes del depósito.
- [Notas redacción TFG (UPV)](https://poliformat.upv.es/access/content/group/GRA_14056_2025/Seminario%20Redacción%20y%20Defensa%20del%20TFG/3_Trabajo%20Final%20de%20Grado.pdf)
