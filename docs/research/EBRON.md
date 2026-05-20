## Título
Métricas del gradiente como indicadores de la eficiencia del entrenamiento en redes neuronales
## Resumen
El entrenamiento de redes neuronales profundas es uno de los procesos más costosos en términos computacionales dentro del aprendizaje automático moderno. A medida que los modelos crecen en tamaño y complejidad, determinar de antemano si una configuración de entrenamiento va a ser eficiente se convierte en un problema con implicaciones prácticas y económicas reales. Hoy en día, la mayoría de decisiones sobre arquitectura, hiperparámetros y optimizadores se toman mediante experimentación empírica, lo que implica lanzar entrenamientos completos para comparar resultados, con el coste que eso conlleva.

Este trabajo parte de una hipótesis concreta: que el comportamiento de los gradientes durante las primeras etapas del entrenamiento contiene información suficiente para anticipar cómo va a evolucionar el proceso completo. Para explorarla, se estudian distintas familias de métricas que capturan cómo se comportan los gradientes a lo largo de los batches, incluyendo tanto su variabilidad estocástica como su grado de alineación o coherencia direccional.

El objetivo del trabajo es determinar en qué medida estas métricas, calculadas sobre distintas fracciones iniciales del entrenamiento, se relacionan con indicadores de eficiencia como el número de épocas necesarias para alcanzar una precisión objetivo o la mejor loss de test alcanzada bajo un número de épocas. Se analiza además si esas relaciones se mantienen estables al variar problemas, arquitecturas y algunos hiperparámetros.

El resultado que se persigue es caracterizar qué métricas ofrecen la mejor capacidad predictiva con un coste computacional razonable, sentando las bases para que trabajos posteriores exploren su uso como señal de diagnóstico en el proceso de optimización.

## Palabras clave
- gradient alignment
- gradient variance
- early stopping
- stochastic gradient descent
- Neural networks