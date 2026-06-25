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
| Resumen y uso en el TFG de un paper concreto | `Papers/<paper>` |

## Documentos de trabajo (leer en orden)

1. **[[1 - Diseño]]**: el qué y el porqué: pregunta de investigación, hipótesis (H1 a H6), diseño experimental, matriz de runs, baselines, protocolo de análisis y confusores.
2. **[[2 - Decisiones]]**: lo que se va decidiendo: decisiones pendientes y log cronológico de las tomadas, con su justificación.
3. **[[3 - Progreso]]**: dónde estamos: plan por fases, estado actual, pasos inmediatos y cola de lectura priorizada. *El estado vigente del proyecto se consulta aquí.*

## Referencia (cambia poco)

- **[[Corpus]]**: datasets y modelos por paper (pares dataset → modelo), frecuencias que justifican el setup y decisiones de implementación (con la sustitución ImageNet → Tiny-ImageNet).
- **[[Métricas]]**: la métrica de cada paper y cómo se traslada al pipeline (estimador, claves de logging, coste, señal); incluye el plan de logging consolidado y la auditoría contra los PDFs.
- **[[Conceptos]]**: glosario: una entrada por concepto, agrupadas por tema (alineación · varianza · optimización · generalización), enlazadas a los papers que las fundamentan.
- **[[EBRON]]**: título, resumen y palabras clave registrados (no editar: es lo entregado).
- **[[Seminarios TFG - cosas a tener en cuenta]]**: guía de redacción, depósito y defensa (ETSINF-UPV), con lo que difiere para GCD.

## Papers

16 notas en `Papers/`, una por paper: resumen del trabajo (`## Summary`), cómo se mide la métrica en el pipeline (`## Medición y pipeline`, con puntero a [[Métricas]]), uso en el TFG y relaciones con otros papers. Los datasets/modelos de cada paper viven en [[Corpus]]; la **cola de lectura priorizada** (6/16 leídos), en [[3 - Progreso]].

## Pendiente de cerrar (carpeta `pending/` en la raíz del repo)

- **[[Plan de análisis congelado]]**: borrador del preregistro estadístico; se congela y se mueve a `docs/` tras el pilot de calibración.

## Código (raíz del repo)

- `src/train.py`: entrenamiento de un run con instrumentación de métricas; `src/run_pilot.py` y `src/run_matrix.py`: lanzadores del pilot de calibración y de la matriz (~960 runs), ambos reanudables.
- `src/metrics/`: registro completo de métricas (se mide todo, siempre); `src/data.py`: split train/val/test estratificado fijo.
- `tests/`: suite de verificación; `experiments/`: los 24 YAML de celda; `thesis/`: memoria LaTeX (plantilla ETSINF).

## Recordatorios

- Anexo ODS obligatorio (plantilla oficial; ver [[Seminarios TFG - cosas a tener en cuenta]] §6).
- El tutor evalúa parte de la nota: visto bueno antes del depósito.
- [Notas redacción TFG (UPV)](https://poliformat.upv.es/access/content/group/GRA_14056_2025/Seminario%20Redacción%20y%20Defensa%20del%20TFG/3_Trabajo%20Final%20de%20Grado.pdf)
