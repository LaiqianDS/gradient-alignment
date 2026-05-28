
El gap de generalización es la diferencia $L_{\text{test}} - L_{\text{train}}$ entre la pérdida del modelo evaluada en el conjunto de entrenamiento y la evaluada en el conjunto de test (o equivalente en error de clasificación). Es la cantidad que toda la familia de métricas train-time intenta acotar o predecir sin tener que tocar el test set.

La paradoja motivadora es la de Zhang y colaboradores en 2017. La misma red, con la misma capacidad nominal y entrenada con el mismo procedimiento, puede ajustar tanto etiquetas reales (con gap pequeño y test accuracy aceptable) como etiquetas completamente aleatorias (con gap enorme: train accuracy del 100 por ciento, test accuracy de la random chance). Esto demuestra que la capacidad expresiva no explica por sí sola el gap: si la red puede memorizar cualquier asignación de etiquetas, ¿por qué con etiquetas reales prefiere aprender features generalizables en lugar de memorizar?

La pista está en la estructura del proceso de entrenamiento, no en la del modelo. Las redes que generalizan acumulan gradientes coherentes durante el ajuste: m-coherence alta, [[SNR del gradiente|GSNR]] alto por parámetro, [[Stiffness]] positiva intra-clase, KTA creciente, gradient disparity bajo entre minibatches. Las que memorizan tienen gradientes que se cancelan entre ejemplos: m-coherence cercana a 1, GSNR colapsado, stiffness cercana a cero o negativa. Es como si la red tuviera dos modos de operación distintos, uno cooperativo y otro individualista, y la calidad de las etiquetas decidiera implícitamente cuál de los dos toca.

Esto convierte el gap en una variable observable durante el entrenamiento, no solo en el post-mortem. Como ejemplo de aplicación, en NAS clásico hay que entrenar a fondo cada arquitectura candidata hasta convergencia para saber cuál es la mejor, lo que tiene un coste prohibitivo (miles de horas de GPU por búsqueda). Con un [[Proxy de generalización train-time]] como TSE-EMA o GWA basta con entrenar cada candidata pocas epochs y leer la métrica para predecir el ranking final, lo que reduce el coste por uno o dos órdenes de magnitud. El mismo argumento aplica a [[Early stopping]] sin validation set y a la detección de muestras ruidosas durante el entrenamiento.

## Enlaces

- Memorización vs aprendizaje y la paradoja original: [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization]]
- Predictores train-time del gap basados en alineación: [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]]
- GSNR como predictor del gap: [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]]
- TSE como predictor del gap en NAS: [[Speedy Performance Estimation for Neural Architecture Search]]
