---
authors:
  - Mahsa Forouzesh
year: 2021
status: read
relevance: high
last_review: 2026-05-07
url: https://link.springer.com/chapter/10.1007/978-3-030-86520-7_14
---

# Disparity Between Batches as a Signal for Early Stopping

## Summary

### Contextualización

El early stopping clásico es una de las técnicas más extendidas para evitar el sobreajuste (overfitting) en redes neuronales profundas entrenadas con métodos iterativos como el descenso por gradiente. La estrategia habitual consiste en separar un subconjunto del dataset disponible como conjunto de validación, etiquetado de manera fiable, y monitorizar la pérdida o el error de clasificación en ese conjunto durante el entrenamiento. Cuando el rendimiento sobre validación deja de mejorar mientras que el rendimiento sobre entrenamiento continúa aumentando, se interrumpe la optimización. Este protocolo presenta varias limitaciones prácticas. En primer lugar, requiere disponer de un conjunto de validación reservado, lo cual reduce la cantidad de datos disponibles para entrenamiento; este problema se agrava cuando los datos son escasos o costosos de recolectar (por ejemplo, en aplicaciones médicas como el dataset MRNet de resonancias de rodilla). En segundo lugar, cuando las etiquetas presentan ruido (label noise) provocado por anotadores no expertos, tareas complejas o ataques adversariales, el conjunto de validación puede contener una fracción elevada de muestras incorrectamente etiquetadas, comprometiendo su fiabilidad como estimador insesgado del error de test. La k-fold cross-validation mitiga parcialmente estos problemas, pero a costa de un sobrecoste computacional muy elevado y de entrenar el modelo únicamente sobre una fracción $(1 - 1/k)$ del dataset en cada ronda.

### Aportación

Forouzesh y Thiran proponen la *gradient disparity* (GD) como métrica de early stopping que no requiere conjunto de validación. La GD se define como la distancia $\ell_2$ entre los vectores gradiente calculados sobre dos mini-batches independientes muestreados del propio conjunto de entrenamiento. La motivación teórica parte de un upper bound probabilístico sobre la diferencia entre los errores de clasificación de un mini-batch cuando la red ha sido actualizada con ese mini-batch frente a cuando ha sido actualizada con otro mini-batch distinto del mismo dataset. Los autores demuestran empíricamente que la GD es una señal fiable de overfitting, particularmente cuando los datos son limitados (porque permite usar todas las muestras para entrenamiento) y cuando contienen etiquetas ruidosas (porque señala overfitting mejor que la pérdida de validación). Adicionalmente, observan una correlación positiva fuerte entre la GD y el error de generalización entre los conjuntos de entrenamiento y test, así como una sensibilidad informativa al nivel de label noise.

### Metodología

Sea $S_1$ y $S_2$ dos mini-batches de tamaños $m_1$ y $m_2$, con vectores gradiente $g_1$ y $g_2$ respectivamente. Tras una iteración SGD que actualiza el vector de parámetros $w$ a $w_1 = w - \gamma \nabla L_{S_1}(h_w)$, los autores definen la *generalization penalty* $\mathcal{R}_2 = L_{S_2}(h_{w_1}) - L_{S_2}(h_{w_2})$, que mide cuánto se penaliza la generalización al haber elegido $S_1$ en lugar de $S_2$. Adaptando el framework PAC-Bayesiano y modelando los predictores como gaussianos $\mathcal{N}(w_i, \sigma^2 I)$, derivan el Teorema 1: el bound sobre $\mathbb{E}[\mathcal{R}_1] + \mathbb{E}[\mathcal{R}_2]$ depende de divergencias KL que se reducen a $\mathrm{KL}(Q_1 \| Q_2) = \frac{1}{2} \frac{\gamma^2}{\sigma^2} \|g_1 - g_2\|_2^2$. La gradient disparity entre dos batches se define entonces como $\mathcal{D}_{i,j} = \|g_i - g_j\|_2$ (Eq. 6). Para mejorar la estimación se promedia sobre $s$ batches con $s \ll B$ (típicamente $s = 5$). El algoritmo de early stopping consiste en monitorizar la GD durante el entrenamiento y detener la optimización cuando esta haya aumentado durante $p$ epochs (patience), siendo $p = 5$ el valor empleado en los experimentos. Para corregir el desajuste de escala entre etapas tempranas (donde tanto $g_i$ como $L_{S_i}$ son grandes) y tardías del entrenamiento, los valores de loss se reescalan dentro de cada batch antes de calcular la GD media. Los autores extienden el análisis más allá de SGD vainilla a SGD con momento, Adagrad, Adadelta y Adam, observando que la GD aparece igualmente en el término KL multiplicado por factores que dependen de promedios decadentes de gradientes pasados.

### Datasets y modelos

Los experimentos cubren un rango amplio de configuraciones. Para datasets limitados utilizan MRNet (resonancias magnéticas de rodilla para detectar anormalidad, lesiones de ACL y meniscales) y subconjuntos pequeños de MNIST (256 muestras con AlexNet), CIFAR-10 (1.28k muestras con VGG-13) y CIFAR-100 (1.28k muestras con ResNet-18). Para evaluar el escenario con label noise, se corrompe una fracción de las etiquetas asignándolas aleatoriamente; el caso principal usa 50% de etiquetas aleatorias en MNIST con AlexNet y CIFAR-100 con ResNet-18. Las arquitecturas adicionales utilizadas en estudios de ancho y tamaño incluyen redes totalmente conectadas, ResNet y VGG.

### Métricas

Las métricas reportadas son: test loss, test accuracy, test AUC score (para MRNet), training accuracy, valores de gradient disparity, coeficiente de correlación de Pearson entre GD y error de test, y sensibilidad de cada método al threshold de early stopping (definida en Eq. 14 del paper extendido). Los resultados clave son: en MRNet, GD mejora más de 1% el AUC de test frente a 5-fold CV en las tres tareas (Tabla 1); en CIFAR-10/VGG-13 con datos limitados, GD alcanza 36.96% test accuracy frente a 35.98% de 5-fold CV; en MNIST/AlexNet limitado, GD obtiene 79.12% frente a 62.62% de 5-fold CV (Tabla 3); con 50% label noise en CIFAR-100/ResNet-18, GD alcanza 3.68% top-1 frente a 1.59% de 10-fold CV; en MNIST/AlexNet noisy, GD logra 97.32% frente a 97.28% de 10-fold CV (Tabla 4). La sensibilidad al threshold también favorece a GD: 0.916 (accuracy) y 0.886 (loss) frente a 1.613 y 1.019 de CV (Tabla 5). La correlación entre GD "train-train" y "train-val" alcanza $\rho = 0.957$ en 220 configuraciones experimentales.

### Conclusiones

La gradient disparity funciona mejor que el early stopping basado en validation set (incluyendo k-fold CV y la variante $k^+$-fold CV) cuando los datos son limitados o ruidosos. En el escenario con datos abundantes y limpios, ambas técnicas son comparables, pero GD evita el coste de reservar un conjunto de validación. GD refleja además fielmente el nivel de label noise incluso en etapas tempranas del entrenamiento, donde el gap de generalización todavía no lo hace, decrece con el tamaño del training set, aumenta con el batch size (correlacionado con el test error en este caso, contraintuitivamente), y disminuye con el ancho de la red. Una limitación del método es que pertenece a la familia de métricas basadas en similitud entre vectores gradiente y puede dejar de ser informativa cuando los gradientes son muy pequeños; sin embargo, los autores observan empíricamente que el momento en que GD detecta overfitting precede al colapso de la pérdida de entrenamiento, por lo que esta limitación no compromete su utilidad práctica como criterio de early stopping. Como dirección futura mencionan el escenario de epoch-wise double-descent.

## Medición y pipeline

- **Métrica concreta**: *gradient disparity* (GD). Para dos mini-batches $S_i, S_j$ independientes muestreados del training set, con gradientes $g_i = \nabla L_{S_i}(h_w)$ y $g_j = \nabla L_{S_j}(h_w)$ evaluados en los parámetros $w$ del paso actual, se define $\mathcal{D}_{i,j} = \|g_i - g_j\|_2$ (Eq. 6). El estimador reportado promedia las distancias pareadas sobre $s$ batches con $s \ll B$ (paper: $s = 5$): $\overline{\mathcal{D}} = \binom{s}{2}^{-1} \sum_{i<j} \mathcal{D}_{i,j}$.

- **Entradas requeridas**: $s$ vectores gradiente $g_i$ calculados sobre mini-batches *independientes* del propio training set (no se necesita validación) y el vector de parámetros $w$ congelado en el paso de medición. Se usa el gradiente del modelo completo concatenado; opcionalmente puede descomponerse por capa para diagnóstico, sin afectar al criterio.

- **Cuándo computar**: una vez por epoch es suficiente para reproducir el protocolo del paper (patience $p = 5$ epochs sobre la serie de GD). Frecuencias mayores (cada $N$ steps) son posibles pero multiplican el coste; conviene fijar la cadencia desde el inicio para que las series sean comparables entre runs.

- **Coste**: $s$ forward+backward extra por medición (≈ $5\times$ el coste de un step con $s=5$), sin update de pesos. Memoria adicional ≈ tamaño del vector de parámetros (para almacenar un gradiente; los demás se consumen en streaming al acumular distancias).

- **Integración en el training loop**: en el punto de medición, congelar el optimizador, muestrear $s$ batches frescos, computar $g_i$ sin llamar a `optimizer.step()`, calcular las $\binom{s}{2}$ distancias $\ell_2$ y loguear el escalar.

```python
def gradient_disparity(model, loss_fn, sampler, s=5):
    grads = []
    for _ in range(s):
        xb, yb = next(sampler)              # mini-batch independiente
        model.zero_grad()
        loss = loss_fn(model(xb), yb)       # rescalar loss por batch si procede
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        grads.append(g.detach().clone())
    dists = [torch.norm(grads[i] - grads[j]) for i in range(s) for j in range(i+1, s)]
    return torch.stack(dists).mean().item()
```

- **Consideraciones**: reescalar la loss dentro de cada batch antes de derivar (el paper lo hace explícitamente para corregir el desajuste de magnitud entre etapas tempranas y tardías); vigilar la contracción de $\|g\|$ al final del entrenamiento, que puede saturar la señal; para Adam, momentum, Adagrad o Adadelta el término KL recoge factores adicionales que dependen de promedios decadentes de gradientes pasados, por lo que la GD sigue siendo informativa pero su escala absoluta no es comparable directamente con la de SGD.

- **Logging**: un escalar `gradient_disparity` por paso de medición a CSV/TensorBoard/W&B, junto con `epoch`, `step`, `train_loss` y `test_loss` para la correlación posterior con eficiencia (epochs-to-threshold, AUC test loss, best test loss). Opcionalmente, GD por capa si el coste lo permite, útil para análisis comparativo entre FC/CNN/ResNet.

## Notes
## Cited By
